/**
 * Gate 3 — AI server connection (handshake only).
 *
 * Scope: POST /v1/sessions → WebSocket(ws_url) → hello → ready. Nothing else.
 * No stream is started here; `session.stream.*` and `session.ui.*` are never
 * touched by this module. Frame sending is a later Gate.
 *
 * Runtime is a bare JS engine (iOS JavaScriptCore / Android QuickJS):
 * runtime `fetch` / `WebSocket` only. No npm packages, no Node API, and
 * `new URL()` is never called — the URL global's presence is unconfirmed, so
 * `probeRuntime()` reports it but nothing depends on it.
 *
 * Wire contract verified against the running server (not assumed):
 *   server/app/api/sessions.py          POST /v1/sessions, POST /v1/sessions/{id}/stop
 *   server/app/api/session_websocket.py WS  /v1/sessions/{id}/ws, hello → ready
 *   server/app/schemas/websocket.py     required: type, schema_version, session_id
 *   server/app/constants.py             SCHEMA_VERSION === HELLO_SCHEMA
 *
 * Two server behaviours this module is built around:
 *  1. On receiving `stop`, the server calls stop_session() and closes the
 *     socket with code 1000 itself. Our own close(1000) may therefore lose a
 *     race — that is expected, not an error.
 *  2. stop_session() marks status="stopped" but KEEPS the record until TTL,
 *     so the follow-up POST /stop still returns 200 (not 404). It is
 *     idempotent and worth sending even after the socket is gone.
 */

import {AI_HTTP, CLIENT_NAME, HELLO_SCHEMA, STREAM_SCHEMA} from "../shared/config"

// ---------------------------------------------------------------------------
// Runtime capability probe
// ---------------------------------------------------------------------------

/**
 * register.d.ts documents the polyfill as installing "__dispatch / __deliver /
 * timers / fetch / etc." — `fetch` is named explicitly but WebSocket only
 * hides inside "etc.", and has never been measured on-device. Without it this
 * whole Gate is impossible, so it is checked before anything else runs.
 *
 * Returns false when WebSocket is missing; the caller must then abort.
 */
export function probeRuntime(): boolean {
  console.log("[Gate3] typeof fetch =", typeof fetch)
  console.log("[Gate3] typeof WebSocket =", typeof WebSocket)
  console.log("[Gate3] typeof URL =", typeof URL)
  console.log("[Gate3] typeof localStorage =", typeof localStorage)

  if (typeof WebSocket === "undefined") {
    console.error("[Gate3] WebSocket 이 런타임에 없다 — Gate 3 성립 불가. 중단한다.")
    console.error("[Gate3] 폴리필이 WebSocket 을 제공하지 않는다는 뜻이다. Mentra 문의 대상.")
    return false
  }
  if (typeof fetch === "undefined") {
    console.error("[Gate3] fetch 가 런타임에 없다 — 세션 생성 불가. 중단한다.")
    return false
  }
  return true
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

type UnknownRecord = Record<string, unknown>

function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null ? (value as UnknownRecord) : undefined
}

/**
 * `fetch` rejects with a TypeError on transport-level failure, and Errors
 * serialize to `{}` under JSON.stringify — so name/message/stack are pulled
 * out explicitly. This is the log that distinguishes:
 *   - ATS / cleartext blocked  → reject before any bytes leave the phone
 *   - server not running       → connection refused
 *   - wrong LAN IP             → timeout / no route to host
 * The exact wording differs per platform, so the raw message is what matters.
 */
function logFetchError(label: string, err: unknown): void {
  console.error(`[Gate3] ${label} fetch 실패 — err:`, err)
  console.error(`[Gate3] ${label} JSON:`, JSON.stringify(err))

  const e = asRecord(err)
  console.error(`[Gate3] ${label} name=`, String(e?.name))
  console.error(`[Gate3] ${label} message=`, String(e?.message))
  if (err instanceof Error && err.stack) {
    console.error(`[Gate3] ${label} stack=`, err.stack)
  }
  console.error(
    `[Gate3] ${label} 판별 힌트: 요청이 나가기도 전에 거부됐다면 iOS ATS / Android cleartext 차단,` +
      ` connection refused 면 서버 미기동, timeout 이면 AI_HTTP 의 IP 가 틀렸을 가능성.`,
  )
  console.error(`[Gate3] ${label} 현재 AI_HTTP=`, AI_HTTP)
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * A parsed `result` payload, handed to the caller via the `onResult` callback.
 *
 * Deliberately plain data with no UI types: this module owns the AI socket and
 * nothing else. index.ts decides what reaches the WebView.
 *
 * `windowIndex` comes from `result.sequence.window_index` — there is no
 * `result.sequence_index` on the wire. `-1` means the server didn't send one.
 */
export interface AiRecognitionResult {
  text: string
  confidence: number
  isFinal: boolean
  windowIndex: number
}

export type AiClientState =
  | "idle"
  | "creating_session"
  | "connecting_ws"
  | "handshaking"
  | "ai_ready"
  | "closing"
  | "error"

/** Backoff schedule for abnormal disconnects. */
const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_BASE_MS = 1000
const RECONNECT_CAP_MS = 30_000

/** How long to wait for `ready` after sending `hello` before treating it as a failure. */
const HANDSHAKE_TIMEOUT_MS = 10_000

export class AiClient {
  private state: AiClientState = "idle"
  private ws: WebSocket | undefined
  private sessionId: string | undefined
  private wsUrl: string | undefined
  private reconnectAttempt = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined
  private handshakeTimer: ReturnType<typeof setTimeout> | undefined
  /** Set before any intentional teardown so onclose does not schedule a reconnect. */
  private shuttingDown = false

  /** `model` block from the last `ready`. Re-logged at stream start — see getModel(). */
  private readyModel: unknown

  /**
   * Idempotency guards. A stream_start / stream_stop must never go out twice
   * for the same stream_id: cleanup runs from stop, error and disconnect paths
   * that can overlap, and duplicates would desync the server's stream state.
   */
  private readonly streamStartSent = new Set<string>()
  private readonly streamStopSent = new Set<string>()

  /** client_message_id → send timestamp, for measuring ack round-trip. */
  private readonly pendingAcks = new Map<string, {label: string; sentAt: number}>()

  /**
   * @param userId  session.userId, sent as `user_id` on POST /v1/sessions.
   * @param onReady Fired every time a `ready` arrives — including after a
   *                reconnect, so the caller can re-enter its ai_ready state.
   * @param onResult Fired once per `result` message with the parsed fields.
   *                Pure data out; this class holds no reference to the UI.
   */
  constructor(
    private readonly userId: string,
    private readonly onReady?: () => void,
    private readonly onResult?: (result: AiRecognitionResult) => void,
  ) {}

  getState(): AiClientState {
    return this.state
  }

  getSessionId(): string | undefined {
    return this.sessionId
  }

  /** True only when the socket is open AND the handshake completed. */
  isReady(): boolean {
    return this.state === "ai_ready" && this.ws !== undefined && this.ws.readyState === 1
  }

  /**
   * The `model` block from `ready` ({loaded, mode, version}). Kept so it can be
   * re-logged at stream start: when frames flow but no `result` ever comes
   * back, "was the model even loaded?" is the first question to answer.
   */
  getModel(): unknown {
    return this.readyModel
  }

  private setState(next: AiClientState): void {
    this.state = next
    console.log("[Gate3] state ->", next)
  }

  // -------------------------------------------------------------------------
  // Connect
  // -------------------------------------------------------------------------

  /**
   * Entry point. Deliberately returns void rather than a Promise: it is called
   * from `session.on("ready")`, a synchronous callback. Every failure path is
   * handled internally so nothing can produce an unhandled rejection.
   */
  connect(): void {
    if (this.state !== "idle" && this.state !== "error") {
      console.warn("[Gate3] connect 무시 — 이미 진행 중이다. state=", this.state)
      return
    }
    void this.runConnect().catch((err) => {
      this.setState("error")
      console.error("[Gate3] connect 최상위 예외 (여기까지 왔으면 버그다):", err)
    })
  }

  private async runConnect(): Promise<void> {
    this.setState("creating_session")

    const created = await this.createSession()
    if (created === undefined) {
      this.setState("error")
      this.scheduleReconnect("세션 생성 실패")
      return
    }

    this.openSocket(created)
  }

  /** POST /v1/sessions. Returns undefined on any failure (already logged). */
  private async createSession(): Promise<{sessionId: string; wsUrl: string} | undefined> {
    const url = `${AI_HTTP}/v1/sessions`
    const body = {client: CLIENT_NAME, user_id: this.userId}

    console.log("[Gate3] POST", url, JSON.stringify(body))

    let response: Response
    try {
      response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json", Accept: "application/json"},
        body: JSON.stringify(body),
      })
    } catch (err) {
      // Transport-level failure — the request never completed. This is where
      // ATS / cleartext blocking lands.
      logFetchError("POST /v1/sessions", err)
      return undefined
    }

    console.log("[Gate3] POST /v1/sessions status=", response.status, response.statusText)

    // Read as text first so a non-JSON body (proxy error page, HTML) is still
    // visible in the log instead of being swallowed by a parse error.
    let raw: string
    try {
      raw = await response.text()
    } catch (err) {
      logFetchError("POST /v1/sessions body 읽기", err)
      return undefined
    }

    if (!response.ok) {
      console.error("[Gate3] POST /v1/sessions 실패 body:", raw)
      return undefined
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch (err) {
      console.error("[Gate3] POST /v1/sessions JSON 파싱 실패. 원문:", raw)
      console.error("[Gate3] 파싱 에러:", err)
      return undefined
    }

    // Whole response, unfolded — undocumented fields must be visible here.
    console.log("[Gate3] 세션 생성 응답:", JSON.stringify(parsed, null, 2))

    const p = asRecord(parsed)
    const sessionId = p?.session_id
    // Separate line: dev reloads orphan sessions, and these ids are what gets
    // cross-referenced against the server log to find them.
    console.log("[Gate3] session_id =", sessionId)
    console.log("[Gate3] status =", p?.status)
    console.log("[Gate3] schema_version =", p?.schema_version)
    console.log("[Gate3] expires_at =", p?.expires_at)

    if (typeof sessionId !== "string" || sessionId.length === 0) {
      console.error("[Gate3] 응답에 session_id 가 없다 — 중단")
      return undefined
    }

    // Server-provided URL is authoritative: it is derived from the request's
    // own base_url, so it already carries the right host/port/scheme.
    const wsUrlRaw = p?.ws_url
    let wsUrl: string
    if (typeof wsUrlRaw === "string" && wsUrlRaw.length > 0) {
      wsUrl = wsUrlRaw
    } else {
      // Schema says `ws_url: str | None`. Assembling it ourselves is the
      // fallback only, because it re-guesses information the server already knew.
      wsUrl = `${AI_HTTP.replace(/^http/i, "ws")}/v1/sessions/${sessionId}/ws`
      console.warn("[Gate3] ws_url 이 null — AI_HTTP 기반으로 폴백 조립했다:", wsUrl)
    }
    console.log("[Gate3] ws_url =", wsUrl)

    return {sessionId, wsUrl}
  }

  // -------------------------------------------------------------------------
  // WebSocket
  // -------------------------------------------------------------------------

  private openSocket(created: {sessionId: string; wsUrl: string}): void {
    this.sessionId = created.sessionId
    this.wsUrl = created.wsUrl
    this.setState("connecting_ws")
    console.log("[Gate3] WebSocket 연결 시도:", created.wsUrl)

    let ws: WebSocket
    try {
      ws = new WebSocket(created.wsUrl)
    } catch (err) {
      // Constructor throwing (rather than firing onerror) usually means a
      // malformed URL or a scheme the runtime refuses outright.
      console.error("[Gate3] WebSocket 생성자 throw:", err)
      console.error("[Gate3] WebSocket 생성자 throw JSON:", JSON.stringify(err))
      this.setState("error")
      this.scheduleReconnect("WebSocket 생성 실패")
      return
    }
    this.ws = ws

    ws.onopen = () => {
      console.log("[Gate3] ws onopen")
      this.reconnectAttempt = 0
      this.sendHello()
    }

    ws.onmessage = (event: MessageEvent) => {
      try {
        this.handleMessage(event.data)
      } catch (err) {
        // A throw here would otherwise kill the socket's callback silently.
        console.error("[Gate3] onmessage 처리 중 예외:", err)
      }
    }

    ws.onerror = (event: Event) => {
      // The error event carries no useful detail in most engines — the real
      // information arrives in the following onclose.
      console.error("[Gate3] ws onerror. type=", (event as Event & {type?: string})?.type)
      console.error("[Gate3] ws onerror event JSON:", JSON.stringify(event))
      console.error("[Gate3] 자세한 원인은 다음 onclose 의 code/reason 을 봐라")
    }

    ws.onclose = (event: CloseEvent) => {
      const code = (event as CloseEvent & {code?: number})?.code
      const reason = (event as CloseEvent & {reason?: string})?.reason
      const wasClean = (event as CloseEvent & {wasClean?: boolean})?.wasClean
      console.log("[Gate3] ws onclose code=", code)
      console.log("[Gate3] ws onclose reason=", reason === "" ? "(빈 문자열)" : reason)
      console.log("[Gate3] ws onclose wasClean=", wasClean)
      // session_websocket.py calls close(1008) for an unknown/stopped/expired
      // session — but it does so BEFORE websocket.accept(), which makes it an
      // HTTP 403 handshake rejection, not a WebSocket close frame. Verified
      // against the running server: uvicorn logs `403 Forbidden` and the client
      // never receives 1008. Engines report the failed upgrade differently
      // (1002 protocol error under Bun, 1006 abnormal closure in most others),
      // so all three are treated as the same "session rejected" case.
      if (code === 1008 || code === 1002 || code === 1006) {
        console.error(`[Gate3] code=${String(code)} — 핸드셰이크 거부로 보인다 (HTTP 403)`)
        console.error("[Gate3] 원인 후보: session_id 미존재 / 이미 stopped / TTL 만료")
        console.error("[Gate3] 서버 로그에서 같은 시각의 '403 Forbidden' 줄을 대조해봐라")
      }

      this.clearHandshakeTimer()
      this.ws = undefined

      if (this.shuttingDown) {
        console.log("[Gate3] 의도된 종료였다 — 재연결하지 않는다")
        this.setState("idle")
        return
      }
      this.setState("error")
      this.scheduleReconnect(`비정상 종료 code=${String(code)}`)
    }
  }

  private sendHello(): void {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.error("[Gate3] sendHello 인데 session_id 가 없다 — 버그")
      return
    }
    this.setState("handshaking")

    const hello = {
      type: "hello",
      schema_version: HELLO_SCHEMA,
      session_id: sessionId,
      client_message_id: `hello-${Date.now()}`,
    }
    if (!this.sendJson(hello, "hello")) return

    // Guard against a server that accepts the socket but never answers.
    this.handshakeTimer = setTimeout(() => {
      if (this.state === "handshaking") {
        console.error(`[Gate3] hello 후 ${HANDSHAKE_TIMEOUT_MS}ms 안에 ready 가 오지 않았다`)
        this.setState("error")
        // Closing triggers onclose, which drives the reconnect path.
        this.closeSocket(4000, "handshake timeout")
      }
    }, HANDSHAKE_TIMEOUT_MS)
  }

  private clearHandshakeTimer(): void {
    if (this.handshakeTimer !== undefined) {
      clearTimeout(this.handshakeTimer)
      this.handshakeTimer = undefined
    }
  }

  /** Serialize + send. Returns false (and logs) when the socket can't take it. */
  private sendJson(payload: object, label: string): boolean {
    const ws = this.ws
    if (ws === undefined) {
      console.error(`[Gate3] ${label} 전송 불가 — ws 없음`)
      return false
    }
    if (ws.readyState !== 1 /* OPEN */) {
      console.error(`[Gate3] ${label} 전송 불가 — readyState=`, ws.readyState)
      return false
    }
    try {
      ws.send(JSON.stringify(payload))
      console.log(`[Gate3] ${label} 전송:`, JSON.stringify(payload))
      return true
    } catch (err) {
      console.error(`[Gate3] ${label} 전송 실패:`, err)
      console.error(`[Gate3] ${label} 전송 실패 JSON:`, JSON.stringify(err))
      return false
    }
  }

  // -------------------------------------------------------------------------
  // Inbound
  // -------------------------------------------------------------------------

  private handleMessage(data: unknown): void {
    if (typeof data !== "string") {
      // Binary frames aren't part of this contract; log rather than drop.
      console.warn("[Gate3] 문자열이 아닌 메시지 수신. typeof=", typeof data)
      return
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(data)
    } catch (err) {
      console.error("[Gate3] 수신 JSON 파싱 실패. 원문:", data)
      console.error("[Gate3] 파싱 에러:", err)
      return
    }

    const m = asRecord(parsed)
    const type = m?.type

    switch (type) {
      case "ready": {
        this.clearHandshakeTimer()
        // Only here does the session count as usable.
        this.setState("ai_ready")
        this.reconnectAttempt = 0
        console.log("[Gate3] ready 수신. 클라이언트 수신 시각=", new Date().toISOString())
        console.log("[Gate3] ready server_time=", m?.server_time)
        this.readyModel = m?.model
        console.log("[Gate3] ready model=", JSON.stringify(this.readyModel))
        console.log("[Gate3] ready 전문:", JSON.stringify(parsed, null, 2))
        // Reconnects re-fire this, letting the caller return to ai_ready.
        try {
          this.onReady?.()
        } catch (err) {
          console.error("[Gate3] onReady 콜백 예외:", err)
        }
        break
      }

      case "ack": {
        const cmid = m?.client_message_id
        console.log("[Gate3] ack status=", m?.status)
        console.log("[Gate3] ack stream_id=", m?.stream_id)
        console.log("[Gate3] ack client_message_id=", cmid)
        console.log("[Gate3] ack 수신 시각=", new Date().toISOString())

        // Round-trip for the stream_start / stream_stop we sent. This is the
        // number that says whether the server is keeping up.
        if (typeof cmid === "string") {
          const pending = this.pendingAcks.get(cmid)
          if (pending !== undefined) {
            this.pendingAcks.delete(cmid)
            console.log(`[Gate3] ack ${pending.label} 왕복 ${Date.now() - pending.sentAt}ms`)
          }
        }
        console.log("[Gate3] ack 전문:", JSON.stringify(parsed))
        break
      }

      case "result": {
        const r = asRecord(m?.result)
        console.log("[Gate3] result.text=", r?.text)
        console.log("[Gate3] result.confidence=", r?.confidence)
        console.log("[Gate3] result.is_final=", r?.is_final)
        // The index field is result.sequence.window_index — confirmed against
        // sequence_service.metadata(). There is no `sequence_index` on the wire.
        console.log("[Gate3] result.sequence.window_index=", asRecord(r?.sequence)?.window_index)
        console.log("[Gate3] result.sequence=", JSON.stringify(r?.sequence))

        // Hand the parsed values out. Every field is re-checked because these
        // came off the wire as `unknown`; the callback's signature promises
        // concrete types. A throwing consumer must not kill the socket.
        if (this.onResult !== undefined) {
          const windowIndex = asRecord(r?.sequence)?.window_index
          try {
            this.onResult({
              text: typeof r?.text === "string" ? r.text : "",
              confidence: typeof r?.confidence === "number" ? r.confidence : 0,
              isFinal: r?.is_final === true,
              windowIndex: typeof windowIndex === "number" ? windowIndex : -1,
            })
          } catch (err) {
            console.error("[Gate3] onResult 콜백 예외:", err)
          }
        }
        break
      }

      case "error": {
        console.error("[Gate3] error code=", m?.code)
        console.error("[Gate3] error message=", m?.message)
        console.error("[Gate3] error retryable=", m?.retryable)
        console.error("[Gate3] error 전문:", JSON.stringify(parsed))
        break
      }

      default: {
        // Unknown types must not be silently dropped — this is how a schema
        // drift between client and server gets noticed.
        console.warn("[Gate3] 미지의 메시지 type=", type)
        console.warn("[Gate3] 미지의 메시지 원문:", data)
        break
      }
    }
  }

  // -------------------------------------------------------------------------
  // Reconnect
  // -------------------------------------------------------------------------

  private scheduleReconnect(why: string): void {
    if (this.shuttingDown) {
      console.log("[Gate3] 종료 중이라 재연결하지 않는다")
      return
    }
    if (this.reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.error(`[Gate3] 재연결 ${MAX_RECONNECT_ATTEMPTS}회 모두 실패 — 포기한다. 마지막 사유: ${why}`)
      this.setState("error")
      return
    }

    this.reconnectAttempt += 1
    const delay = Math.min(RECONNECT_BASE_MS * Math.pow(2, this.reconnectAttempt - 1), RECONNECT_CAP_MS)
    console.warn(
      `[Gate3] 재연결 ${this.reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS} 회차 — ${delay}ms 후 시도. 사유: ${why}`,
    )

    if (this.reconnectTimer !== undefined) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined
      if (this.shuttingDown) return
      console.log(`[Gate3] 재연결 ${this.reconnectAttempt}회차 실행`)
      // A fresh POST /v1/sessions on purpose: the previous session_id may be
      // stopped or expired server-side, and reusing it would just earn a 1008.
      // Full hello → ready runs again against the new id.
      void this.runConnect().catch((err) => {
        this.setState("error")
        console.error("[Gate3] 재연결 중 예외:", err)
      })
    }, delay)
  }

  // -------------------------------------------------------------------------
  // Teardown
  // -------------------------------------------------------------------------

  private closeSocket(code: number, reason: string): void {
    const ws = this.ws
    if (ws === undefined) return
    try {
      ws.close(code, reason)
      console.log(`[Gate3] ws.close(${code}) 호출`)
    } catch (err) {
      console.error("[Gate3] ws.close 실패:", err)
    }
  }

  // -------------------------------------------------------------------------
  // Stream messages (Gate 4)
  // -------------------------------------------------------------------------

  /**
   * Tell the AI server to start pulling the Mentra WHEP stream.
   *
   * Guarded by `streamStartSent`: one start per stream_id, ever. Synchronous,
   * so it is safe from any callback. Returns false when nothing was sent.
   */
  sendStreamStart(streamId: string, webrtcUrl: string): boolean {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.error("[Gate4] stream_start 전송 불가 — AI session_id 없음")
      return false
    }
    if (this.streamStartSent.has(streamId)) {
      console.warn("[Gate4] stream_start 중복 방지 — 이미 보낸 stream_id:", streamId)
      return false
    }

    const clientMessageId = `stream-start-${Date.now()}`
    const sent = this.sendJson(
      {
        // stream_start/stream_stop ride the WebRTC schema, not HELLO_SCHEMA.
        type: "stream_start",
        schema_version: STREAM_SCHEMA,
        session_id: sessionId,
        stream_id: streamId,
        webrtc_url: webrtcUrl,
        client_message_id: clientMessageId,
      },
      "stream_start",
    )
    if (!sent) return false

    this.streamStartSent.add(streamId)
    this.pendingAcks.set(clientMessageId, {label: "stream_start", sentAt: Date.now()})
    console.log("[Gate4] stream_start 전송 시각=", new Date().toISOString())
    // Re-logged here on purpose: if frames flow but no result ever arrives,
    // this line answers "was the model loaded at the time?".
    console.log("[Gate4] 스트림 시작 시점 model=", JSON.stringify(this.readyModel))
    return sent
  }

  /**
   * Tell the AI server to stop pulling. Guarded by `streamStopSent` so the
   * shared cleanup path can run from stop / error / disconnect without
   * emitting duplicates. Synchronous — safe from beforeDisconnect.
   */
  sendStreamStop(streamId: string): boolean {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.error("[Gate4] stream_stop 전송 불가 — AI session_id 없음")
      return false
    }
    if (this.streamStopSent.has(streamId)) {
      console.warn("[Gate4] stream_stop 중복 방지 — 이미 보낸 stream_id:", streamId)
      return false
    }

    const clientMessageId = `stream-stop-${Date.now()}`
    const sent = this.sendJson(
      {
        type: "stream_stop",
        schema_version: STREAM_SCHEMA,
        session_id: sessionId,
        stream_id: streamId,
        client_message_id: clientMessageId,
      },
      "stream_stop",
    )
    if (!sent) return false

    this.streamStopSent.add(streamId)
    this.pendingAcks.set(clientMessageId, {label: "stream_stop", sentAt: Date.now()})
    console.log("[Gate4] stream_stop 전송 시각=", new Date().toISOString())
    return sent
  }

  /**
   * Synchronous half of teardown only — safe to call from
   * `session.on("beforeDisconnect")`, where async work cannot finish before
   * the host socket closes.
   */
  sendStopMessage(reason: string): boolean {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.warn("[Gate3] stop 전송 생략 — session_id 없음")
      return false
    }
    this.shuttingDown = true
    return this.sendJson(
      {
        type: "stop",
        schema_version: HELLO_SCHEMA,
        session_id: sessionId,
        client_message_id: `stop-${Date.now()}`,
        reason,
      },
      "stop",
    )
  }

  /** Synchronous close — safe from `session.on("disconnect")`. */
  closeNow(reason: string): void {
    this.shuttingDown = true
    this.clearHandshakeTimer()
    if (this.reconnectTimer !== undefined) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = undefined
    }
    this.closeSocket(1000, reason)
    this.ws = undefined
  }

  /**
   * POST /v1/sessions/{id}/stop, best effort.
   *
   * Idempotent server-side: stop_session() marks the record "stopped" but
   * keeps it until TTL, so this returns 200 even when the `stop` WS message
   * already ran it. Safe to call more than once.
   *
   * Called from the disconnect path, where completion is NOT guaranteed — the
   * host tears the JSContext down without waiting. A failure here is normal
   * and not an error: the server session expires on its own at `expires_at`
   * (creation + SESSION_TTL_SECONDS, 1 hour by default), so nothing leaks
   * permanently even when this never lands.
   */
  postStopBestEffort(): void {
    const sessionId = this.sessionId
    void this.postStop(sessionId).catch((err) => {
      console.warn("[Gate3] postStop 예외 (정상 취급 — 세션은 expires_at 에 만료된다):", err)
    })
  }

  private async postStop(sessionId: string | undefined): Promise<void> {
    if (sessionId === undefined) {
      console.warn("[Gate3] POST /stop 생략 — session_id 없음")
      return
    }
    const url = `${AI_HTTP}/v1/sessions/${sessionId}/stop`
    console.log("[Gate3] POST", url)

    let response: Response
    try {
      response = await fetch(url, {
        method: "POST",
        headers: {Accept: "application/json"},
      })
    } catch (err) {
      logFetchError("POST /stop", err)
      return
    }

    console.log("[Gate3] POST /stop status=", response.status, response.statusText)
    try {
      const raw = await response.text()
      console.log("[Gate3] POST /stop body:", raw)
    } catch (err) {
      console.error("[Gate3] POST /stop body 읽기 실패:", err)
    }

    if (response.status === 404) {
      console.warn("[Gate3] POST /stop 404 — 세션이 이미 서버에서 사라졌다 (TTL 만료 등)")
    }

    this.sessionId = undefined
    this.setState("idle")
  }
}
