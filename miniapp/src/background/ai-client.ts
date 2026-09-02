/**
 * AI 서버 연결 담당. POST /v1/sessions → WebSocket(ws_url) → hello → ready.
 *
 * 이 모듈은 AI 소켓만 소유한다. `session.stream.*` 도 `session.ui.*` 도 건드리지
 * 않는다. 인식 결과는 `onResult` 콜백으로 순수 데이터만 내보내고, 그것을 UI 로
 * 보낼지는 index.ts 가 정한다.
 *
 * 런타임은 bare JS 엔진(iOS JavaScriptCore / Android QuickJS)이다. npm 패키지도
 * Node API 도 없고, `typeof URL === "undefined"` 로 실측됐으므로 `new URL()` 은
 * 쓸 수 없다. URL 파싱이 필요하면 정규식뿐이다. fetch / WebSocket /
 * localStorage 는 사용 가능한 것으로 실측됐다.
 *
 * 와이어 규약은 실행 중인 서버 코드로 대조 확인한 것이다(추정 아님):
 *   server/app/api/sessions.py          POST /v1/sessions, POST /v1/sessions/{id}/stop
 *   server/app/api/session_websocket.py WS  /v1/sessions/{id}/ws, hello → ready
 *   server/app/schemas/websocket.py     필수 필드: type, schema_version, session_id
 *   server/app/constants.py             SCHEMA_VERSION === HELLO_SCHEMA
 *
 * 이 모듈이 전제하는 서버 동작 두 가지:
 *  1. `stop` 을 받으면 서버가 stop_session() 을 부르고 code 1000 으로 소켓을
 *     직접 닫는다. 우리 쪽 close(1000) 이 경쟁에서 질 수 있는데 정상이다.
 *  2. stop_session() 은 status="stopped" 로 표시할 뿐 TTL 까지 레코드를
 *     유지한다. 그래서 뒤이은 POST /stop 도 404 가 아니라 200 을 돌려준다.
 *     멱등이므로 소켓이 사라진 뒤에 보내도 무해하다.
 */

import {AI_HTTP, CLIENT_NAME, HELLO_SCHEMA, STREAM_SCHEMA} from "../shared/config"

// ---------------------------------------------------------------------------
// 런타임 전역 확인
// ---------------------------------------------------------------------------

/**
 * register.d.ts 는 폴리필이 설치하는 것을 "__dispatch / __deliver / timers /
 * fetch / etc." 라고만 적어 둔다. fetch 는 이름이 명시돼 있지만 WebSocket 은
 * "etc." 안에 숨어 있어 실측 전까지 보장이 없었다. WebSocket 이 없으면 AI 연결
 * 자체가 불가능하므로 다른 무엇보다 먼저 확인한다.
 *
 * WebSocket 이 없으면 false 를 돌려주고, 호출부는 거기서 중단해야 한다.
 */
export function probeRuntime(): boolean {
  console.log("[Runtime] typeof fetch =", typeof fetch)
  console.log("[Runtime] typeof WebSocket =", typeof WebSocket)
  console.log("[Runtime] typeof URL =", typeof URL)
  console.log("[Runtime] typeof localStorage =", typeof localStorage)

  if (typeof WebSocket === "undefined") {
    console.error("[Runtime] WebSocket 이 런타임에 없다 — AI 연결 성립 불가. 중단한다.")
    console.error("[Runtime] 폴리필이 WebSocket 을 제공하지 않는다는 뜻이다. Mentra 문의 대상.")
    return false
  }
  if (typeof fetch === "undefined") {
    console.error("[Runtime] fetch 가 런타임에 없다 — 세션 생성 불가. 중단한다.")
    return false
  }
  return true
}

// ---------------------------------------------------------------------------
// 로깅 헬퍼
// ---------------------------------------------------------------------------

type UnknownRecord = Record<string, unknown>

function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null ? (value as UnknownRecord) : undefined
}

/**
 * fetch 는 전송 계층 실패 시 TypeError 로 reject 하는데, Error 는
 * JSON.stringify 에서 `{}` 로 뭉개진다. 그래서 name/message/stack 을 따로 꺼낸다.
 * 이 로그가 구분해 주는 것:
 *   - ATS / cleartext 차단 → 바이트가 폰을 떠나기도 전에 거부
 *   - 서버 미기동          → connection refused
 *   - LAN IP 오류          → timeout / no route to host
 * 문구는 플랫폼마다 다르므로 결국 원본 message 가 판단 근거다.
 */
function logFetchError(tag: string, label: string, err: unknown): void {
  console.error(`${tag} ${label} fetch 실패 — err:`, err)
  console.error(`${tag} ${label} JSON:`, JSON.stringify(err))

  const e = asRecord(err)
  console.error(`${tag} ${label} name=`, String(e?.name))
  console.error(`${tag} ${label} message=`, String(e?.message))
  if (err instanceof Error && err.stack) {
    console.error(`${tag} ${label} stack=`, err.stack)
  }
  console.error(
    `${tag} ${label} 판별 힌트: 요청이 나가기도 전에 거부됐다면 iOS ATS / Android cleartext 차단,` +
      ` connection refused 면 서버 미기동, timeout 이면 AI_HTTP 의 IP 가 틀렸을 가능성.`,
  )
  console.error(`${tag} ${label} 현재 AI_HTTP=`, AI_HTTP)
}

// ---------------------------------------------------------------------------
// 클라이언트
// ---------------------------------------------------------------------------

/**
 * 파싱된 `result` 페이로드. `onResult` 콜백으로 호출부에 전달된다.
 *
 * UI 타입이 섞이지 않은 순수 데이터다. 이 모듈은 AI 소켓만 소유하고, WebView 로
 * 무엇을 보낼지는 index.ts 가 정한다.
 *
 * `windowIndex` 의 출처는 `result.sequence.window_index` 다.
 * 서버는 60프레임이 차기 전 구간에서 window_index 를 null 로 보내는데,
 * 그 경우 -1 을 넣는다(센티널).
 *
 * `sequenceIndex` 는 `result` 안이 아니라 메시지 최상위의 `sequence_index` 에서
 * 온다(whep_service.py 의 result 전송 블록). 서버가 실제로 처리한 누적 프레임
 * 수이며, 숫자가 아니면 null 이다. `result.sequence_index` 라는 중첩 필드는
 * 없으니 헷갈리지 말 것.
 */
export interface AiRecognitionResult {
  text: string
  confidence: number
  isFinal: boolean
  windowIndex: number
  /** 서버 누적 처리 프레임 수. 없거나 숫자가 아니면 null. */
  sequenceIndex: number | null
}

/**
 * 파싱된 서버 `error` 페이로드. `onError` 콜백으로 호출부에 전달된다.
 * `AiRecognitionResult` 와 같이 UI 타입이 섞이지 않은 순수 데이터다.
 */
export interface AiServerError {
  code: string
  message: string
  /** 서버가 boolean 으로 주지 않으면 null. */
  retryable: boolean | null
}

export type AiClientState =
  | "idle"
  | "creating_session"
  | "connecting_ws"
  | "handshaking"
  | "ai_ready"
  | "closing"
  | "error"

/** 비정상 종료 시 재연결 백오프. 1s → 2s → 4s → 8s → 16s, 5회로 종료. */
const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_BASE_MS = 1000
const RECONNECT_CAP_MS = 30_000

/** hello 를 보낸 뒤 ready 를 기다리는 한도. 넘기면 실패로 처리한다. */
const HANDSHAKE_TIMEOUT_MS = 10_000

/**
 * ping 주기.
 */
const PING_INTERVAL_MS = 20_000

/**
 * `result` 한 건의 전체 필드를 찍을지. 기본은 꺼짐이다.
 *
 * 켜면 건당 3줄이 된다. 실측으로 초당 114줄까지 나와 콘솔이 밀렸고, 그 바람에
 * ready / stream_start 처럼 드물지만 중요한 줄이 통째로 유실됐다. 이 모듈
 * 안에서만 쓰는 디버그 스위치라 shared/config.ts 가 아니라 여기에 둔다 —
 * 거기는 UI 번들과 공유하는 값들의 자리다.
 */
const RESULT_VERBOSE = false

export class AiClient {
  /**
   * 로그에서 인스턴스를 눈으로 구분하기 위한 짧은 id. 고아 인스턴스가 남아
   * 타이머를 발화시키는 정황을 추적하려면 어느 인스턴스가 찍은 줄인지 알아야
   * 한다. 사람이 읽는 용도라 충돌 가능성은 문제되지 않는다.
   */
  private readonly instanceId = Math.random().toString(16).slice(2, 6)
  /** 이 인스턴스가 찍는 모든 로그의 접두사. */
  private readonly tag = `[AI#${this.instanceId}]`
  /** 스트림 계열 로그용. 기존 `[Stream]` 과 같은 자리에 쓴다. */
  private readonly streamTag = `[Stream#${this.instanceId}]`

  private state: AiClientState = "idle"
  private ws: WebSocket | undefined
  private sessionId: string | undefined
  private wsUrl: string | undefined
  private reconnectAttempt = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined
  private handshakeTimer: ReturnType<typeof setTimeout> | undefined
  /** ready 이후에만 돈다. 유휴 상태로 끊기는 것을 막는다. */
  private pingTimer: ReturnType<typeof setTimeout> | undefined
  /** 의도된 종료 직전에 세운다. onclose 가 재연결을 걸지 않게 하는 유일한 근거. */
  private shuttingDown = false

  /** 마지막 `ready` 의 `model` 블록. 스트림 시작 시 다시 찍는다 — getModel() 참고. */
  private readyModel: unknown

  /**
   * 멱등성 가드. 같은 stream_id 로 stream_start / stream_stop 이 두 번 나가면
   * 안 된다. 정리 경로가 stop·error·disconnect 세 갈래라 겹칠 수 있고, 중복이
   * 나가면 서버의 스트림 상태와 어긋난다.
   *
   * 범위는 세션당 한 번이다. 막으려는 대상이 서버의 스트림 상태인데 그 상태는
   * 세션에 딸려 있으므로, 새 session_id 를 받으면 openSocket 이 둘 다 비운다.
   */
  private readonly streamStartSent = new Set<string>()
  private readonly streamStopSent = new Set<string>()

  /** client_message_id → 전송 시각. ack 왕복 시간 측정용. */
  private readonly pendingAcks = new Map<string, {label: string; sentAt: number}>()

  /**
   * @param userId  session.userId. POST /v1/sessions 의 `user_id` 로 나간다.
   * @param onReady `ready` 가 올 때마다 호출된다. 재연결 이후에도 다시 오므로
   *                호출부가 ai_ready 상태로 복귀할 수 있다.
   * @param onResult `result` 메시지 한 건마다 파싱된 필드로 호출된다.
   *                순수 데이터만 나가며 이 클래스는 UI 를 참조하지 않는다.
   * @param onError `error` 메시지 한 건마다 호출된다. stream_unavailable 처럼
   *                서버가 손을 뗀 상황이 호출부까지 닿는 유일한 통로다.
   */
  constructor(
    private readonly userId: string,
    private readonly onReady?: () => void,
    private readonly onResult?: (result: AiRecognitionResult) => void,
    private readonly onError?: (error: AiServerError) => void,
  ) {}

  /**
   * 로그 접두사 `[AI#xxx]` 에 쓰인 짧은 id. 호출부가 인스턴스의 생성과 종료를
   * 같은 이름으로 셀 수 있게 공개한다.
   */
  getId(): string {
    return this.instanceId
  }

  getState(): AiClientState {
    return this.state
  }

  getSessionId(): string | undefined {
    return this.sessionId
  }

  /** 소켓이 열려 있고 핸드셰이크까지 끝난 경우에만 true. */
  isReady(): boolean {
    return this.state === "ai_ready" && this.ws !== undefined && this.ws.readyState === 1
  }

  /**
   * `ready` 의 `model` 블록 ({loaded, mode, version}). 스트림 시작 시점에 다시
   * 찍으려고 들고 있다. 프레임은 흐르는데 `result` 가 하나도 안 올 때
   * "그 시점에 모델이 로드돼 있었나" 가 첫 확인 항목이기 때문이다.
   */
  getModel(): unknown {
    return this.readyModel
  }

  private setState(next: AiClientState): void {
    this.state = next
    console.log(`${this.tag} state ->`, next)
  }

  // -------------------------------------------------------------------------
  // 연결
  // -------------------------------------------------------------------------

  /**
   * 진입점. Promise 가 아니라 void 를 돌려주는 것은 의도다. 동기 콜백인
   * `session.on("ready")` 안에서 불리기 때문에, 모든 실패 경로를 내부에서
   * 처리해 unhandled rejection 이 생길 여지를 없앴다.
   */
  connect(): void {
    if (this.state !== "idle" && this.state !== "error") {
      console.warn(`${this.tag} connect 무시 — 이미 진행 중이다. state=`, this.state)
      return
    }
    void this.runConnect().catch((err) => {
      this.setState("error")
      console.error(`${this.tag} connect 최상위 예외 (여기까지 왔으면 버그다):`, err)
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

  /** POST /v1/sessions. 실패 시 undefined 를 돌려준다(로그는 이미 남긴 뒤다). */
  private async createSession(): Promise<{sessionId: string; wsUrl: string} | undefined> {
    const url = `${AI_HTTP}/v1/sessions`
    const body = {client: CLIENT_NAME, user_id: this.userId}

    console.log(`${this.tag} POST`, url, JSON.stringify(body))

    let response: Response
    try {
      response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json", Accept: "application/json"},
        body: JSON.stringify(body),
      })
    } catch (err) {
      // 전송 계층 실패. 요청 자체가 완료되지 않았다. ATS / cleartext 차단이
      // 여기로 떨어진다.
      logFetchError(this.tag, "POST /v1/sessions", err)
      return undefined
    }

    console.log(`${this.tag} POST /v1/sessions status=`, response.status, response.statusText)

    // 먼저 텍스트로 읽는다. JSON 이 아닌 본문(프록시 에러 페이지, HTML)이
    // 파싱 에러에 묻히지 않고 로그에 남는다.
    let raw: string
    try {
      raw = await response.text()
    } catch (err) {
      logFetchError(this.tag, "POST /v1/sessions body 읽기", err)
      return undefined
    }

    if (!response.ok) {
      console.error(`${this.tag} POST /v1/sessions 실패 body:`, raw)
      return undefined
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch (err) {
      console.error(`${this.tag} POST /v1/sessions JSON 파싱 실패. 원문:`, raw)
      console.error(`${this.tag} 파싱 에러:`, err)
      return undefined
    }

    // 응답 전문을 펼쳐서 찍는다. 문서에 없는 필드를 발견하는 통로다.
    console.log(`${this.tag} 세션 생성 응답:`, JSON.stringify(parsed, null, 2))

    const p = asRecord(parsed)
    const sessionId = p?.session_id
    // 줄을 따로 뺀 이유: dev 리로드가 세션을 고아로 남기는데, 서버 로그와
    // 대조해 그것을 찾아내는 열쇠가 이 id 다.
    console.log(`${this.tag} session_id =`, sessionId)
    console.log(`${this.tag} status =`, p?.status)
    console.log(`${this.tag} schema_version =`, p?.schema_version)
    console.log(`${this.tag} expires_at =`, p?.expires_at)

    if (typeof sessionId !== "string" || sessionId.length === 0) {
      console.error(`${this.tag} 응답에 session_id 가 없다 — 중단`)
      return undefined
    }

    // 서버가 준 URL 이 우선이다. 요청의 base_url 에서 파생된 값이라 host/port/
    // scheme 이 이미 맞게 들어 있다.
    const wsUrlRaw = p?.ws_url
    let wsUrl: string
    if (typeof wsUrlRaw === "string" && wsUrlRaw.length > 0) {
      wsUrl = wsUrlRaw
    } else {
      // 스키마상 `ws_url: str | None` 이다. 직접 조립하는 건 폴백일 뿐이다 —
      // 서버가 이미 알던 정보를 우리가 다시 추측하는 셈이라서.
      wsUrl = `${AI_HTTP.replace(/^http/i, "ws")}/v1/sessions/${sessionId}/ws`
      console.warn(`${this.tag} ws_url 이 null — AI_HTTP 기반으로 폴백 조립했다:`, wsUrl)
    }
    console.log(`${this.tag} ws_url =`, wsUrl)

    return {sessionId, wsUrl}
  }

  // -------------------------------------------------------------------------
  // WebSocket
  // -------------------------------------------------------------------------

  private openSocket(created: {sessionId: string; wsUrl: string}): void {
    this.sessionId = created.sessionId
    this.wsUrl = created.wsUrl
    // 새 세션은 이전 세션의 스트림을 모른다. 가드를 그대로 두면 재연결 이후
    // stream_start 가 영영 막힌다. session_id 가 바뀌는 이 지점이 유일한
    // 초기화 시점이다 — 같은 세션 안에서의 중복은 계속 막힌다.
    this.streamStartSent.clear()
    this.streamStopSent.clear()
    this.setState("connecting_ws")
    console.log(`${this.tag} WebSocket 연결 시도:`, created.wsUrl)

    let ws: WebSocket
    try {
      ws = new WebSocket(created.wsUrl)
    } catch (err) {
      // onerror 가 아니라 생성자가 throw 했다면 대개 URL 형식 오류이거나
      // 런타임이 아예 거부하는 scheme 이다.
      console.error(`${this.tag} WebSocket 생성자 throw:`, err)
      console.error(`${this.tag} WebSocket 생성자 throw JSON:`, JSON.stringify(err))
      this.setState("error")
      this.scheduleReconnect("WebSocket 생성 실패")
      return
    }
    this.ws = ws

    // 이 소켓이 열린 적이 있는지. onclose 진단이 "핸드셰이크 거부" 와 "열린 연결이 끊김" 을 가르는 근거.
    let opened = false

    ws.onopen = () => {
      opened = true
      console.log(`${this.tag} ws onopen`)
      this.sendHello()
    }

    ws.onmessage = (event: MessageEvent) => {
      try {
        this.handleMessage(event.data)
      } catch (err) {
        // 여기서 throw 가 새어 나가면 소켓 콜백이 조용히 죽는다.
        console.error(`${this.tag} onmessage 처리 중 예외:`, err)
      }
    }

    ws.onerror = (event: Event) => {
      // 대부분의 엔진에서 error 이벤트에는 쓸 만한 정보가 없다. 실제 원인은
      // 뒤따르는 onclose 에 담긴다.
      console.error(`${this.tag} ws onerror. type=`, (event as Event & {type?: string})?.type)
      console.error(`${this.tag} ws onerror event JSON:`, JSON.stringify(event))
      console.error(`${this.tag} 자세한 원인은 다음 onclose 의 code/reason 참고`)
    }

    ws.onclose = (event: CloseEvent) => {
      const code = (event as CloseEvent & {code?: number})?.code
      const reason = (event as CloseEvent & {reason?: string})?.reason
      const wasClean = (event as CloseEvent & {wasClean?: boolean})?.wasClean
      console.log(`${this.tag} ws onclose code=`, code)
      console.log(`${this.tag} ws onclose reason=`, reason === "" ? "(빈 문자열)" : reason)
      console.log(`${this.tag} ws onclose wasClean=`, wasClean)
      // 실측: wasClean 이 null 로 온다. 그래서 종료 원인은 code 로만 판단한다.
      //
      // session_websocket.py 는 미존재/stopped/만료 세션에 close(1008) 을
      // 부르지만, 그 호출이 websocket.accept() *이전* 이라 실제로는 WebSocket
      // close 프레임이 아니라 HTTP 403 핸드셰이크 거부가 된다. 실행 중인 서버로
      // 확인했다 — uvicorn 은 `403 Forbidden` 을 찍고 클라이언트는 1008 을 받지
      // 못한다. 실패한 업그레이드를 엔진마다 다르게 보고하므로(Bun 은 1002
      // protocol error, 대부분은 1006 abnormal closure) 세 값을 같은
      // "세션 거부" 로 묶어 처리한다.

      if (!opened && (code === 1008 || code === 1002 || code === 1006)) {
        console.error(`${this.tag} code=${String(code)} — 핸드셰이크 거부로 보인다 (HTTP 403)`)
        console.error(`${this.tag} 원인 후보: session_id 미존재 / 이미 stopped / TTL 만료`)
        console.error(`${this.tag} 서버 로그에서 같은 시각의 '403 Forbidden' 줄과 대조 필요`)
      } else if (opened && code !== 1000 && code !== 4000) {
        console.error(`${this.tag} code=${String(code)} — 연결 유지 실패. 열린 연결이 끊겼다`)
        console.error(`${this.tag} 원인 후보: 네트워크 단절 / 서버 종료 / 유휴 타임아웃`)
      }

      this.clearHandshakeTimer()
      this.clearPingTimer()
      this.ws = undefined

      if (this.shuttingDown) {
        console.log(`${this.tag} 의도된 종료였다 — 재연결하지 않는다`)
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
      console.error(`${this.tag} sendHello 인데 session_id 가 없다 — 버그`)
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

    // 소켓은 받아 놓고 답을 주지 않는 서버에 대한 방어.
    this.handshakeTimer = setTimeout(() => {
      if (this.state === "handshaking") {
        console.error(`${this.tag} hello 후 ${HANDSHAKE_TIMEOUT_MS}ms 안에 ready 가 오지 않았다`)
        this.setState("error")
        // 닫으면 onclose 가 뜨고, 그 경로가 재연결을 굴린다.
        this.closeSocket(4000, "handshake timeout")
      }
    }, HANDSHAKE_TIMEOUT_MS)
  }

  /**
   * ready 이후 PING_INTERVAL_MS 마다 ping 을 보낸다.
   *
   * setInterval 이 아니라 setTimeout 재귀인 이유: 이 런타임에서 실측된 것은
   * setTimeout 뿐이고 파일 전체가 그 패턴을 쓴다.
   */
  private startPingLoop(): void {
    this.clearPingTimer()
    this.pingTimer = setTimeout(() => {
      this.pingTimer = undefined
      if (this.shuttingDown) return
      const sessionId = this.sessionId
      if (sessionId === undefined) return
      const sent = this.sendJson(
        {
          type: "ping",
          schema_version: HELLO_SCHEMA,
          session_id: sessionId,
          client_message_id: `ping-${Date.now()}`,
        },
        "ping",
      )
      // 실패면 소켓이 이미 죽은 것이다. 다시 걸지 않는다 — onclose 가 재연결을
      // 굴리고, 새 소켓이 ready 를 받으면 거기서 다시 시작한다.
      if (sent) this.startPingLoop()
    }, PING_INTERVAL_MS)
  }

  private clearPingTimer(): void {
    if (this.pingTimer !== undefined) {
      clearTimeout(this.pingTimer)
      this.pingTimer = undefined
    }
  }

  private clearHandshakeTimer(): void {
    if (this.handshakeTimer !== undefined) {
      clearTimeout(this.handshakeTimer)
      this.handshakeTimer = undefined
    }
  }

  /** 직렬화 후 전송. 소켓이 받을 수 없는 상태면 로그를 남기고 false. */
  private sendJson(payload: object, label: string): boolean {
    const ws = this.ws
    if (ws === undefined) {
      console.error(`${this.tag} ${label} 전송 불가 — ws 없음`)
      return false
    }
    if (ws.readyState !== 1 /* OPEN */) {
      console.error(`${this.tag} ${label} 전송 불가 — readyState=`, ws.readyState)
      return false
    }
    try {
      ws.send(JSON.stringify(payload))
      console.log(`${this.tag} ${label} 전송:`, JSON.stringify(payload))
      return true
    } catch (err) {
      console.error(`${this.tag} ${label} 전송 실패:`, err)
      console.error(`${this.tag} ${label} 전송 실패 JSON:`, JSON.stringify(err))
      return false
    }
  }

  // -------------------------------------------------------------------------
  // 수신
  // -------------------------------------------------------------------------

  private handleMessage(data: unknown): void {
    if (typeof data !== "string") {
      // 바이너리 프레임은 이 규약에 없다. 버리지 말고 남겨서 드러나게 한다.
      console.warn(`${this.tag} 문자열이 아닌 메시지 수신. typeof=`, typeof data)
      return
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(data)
    } catch (err) {
      console.error(`${this.tag} 수신 JSON 파싱 실패. 원문:`, data)
      console.error(`${this.tag} 파싱 에러:`, err)
      return
    }

    const m = asRecord(parsed)
    const type = m?.type

    switch (type) {
      case "ready": {
        this.clearHandshakeTimer()
        // 여기 도달해야 비로소 세션이 쓸 수 있는 상태가 된다.
        this.setState("ai_ready")
        this.reconnectAttempt = 0
        console.log(`${this.tag} ready 수신. 클라이언트 수신 시각=`, new Date().toISOString())
        console.log(`${this.tag} ready server_time=`, m?.server_time)
        this.readyModel = m?.model
        console.log(`${this.tag} ready model=`, JSON.stringify(this.readyModel))
        console.log(`${this.tag} ready 전문:`, JSON.stringify(parsed, null, 2))
        // sendStopMessage 는 소켓을 연 채로 shuttingDown 만 세운다. 그 뒤에 온
        // ready 로 onReady 가 돌면 종료 중에 스트림을 다시 붙이게 된다. 가드가
        // 여기 있는 것은 의도다 — 위의 타이머 정리와 로그는 종료 중에도 필요하다.
        if (this.shuttingDown) {
          console.log(`${this.tag} 종료 중에 ready 수신 — onReady 를 부르지 않는다`)
          break
        }
        // 핸드셰이크가 끝난 여기서 시작한다.
        this.startPingLoop()
        // 재연결 때도 다시 불리므로 호출부가 ai_ready 로 복귀할 수 있다.
        try {
          this.onReady?.()
        } catch (err) {
          console.error(`${this.tag} onReady 콜백 예외:`, err)
        }
        break
      }

      case "ack": {
        const cmid = m?.client_message_id
        console.log(`${this.tag} ack status=`, m?.status)
        console.log(`${this.tag} ack stream_id=`, m?.stream_id)
        console.log(`${this.tag} ack client_message_id=`, cmid)
        console.log(`${this.tag} ack 수신 시각=`, new Date().toISOString())

        // 우리가 보낸 stream_start / stream_stop 의 왕복 시간. 서버가 따라오고
        // 있는지를 말해 주는 수치다.
        if (typeof cmid === "string") {
          const pending = this.pendingAcks.get(cmid)
          if (pending !== undefined) {
            this.pendingAcks.delete(cmid)
            console.log(`${this.tag} ack ${pending.label} 왕복 ${Date.now() - pending.sentAt}ms`)
          }
        }
        console.log(`${this.tag} ack 전문:`, JSON.stringify(parsed))
        break
      }

      case "result": {
        const r = asRecord(m?.result)
        // 인덱스 필드는 두 개고 층이 다르다. 창 번호는 result.sequence.
        // window_index (sequence_service.metadata()), 누적 처리 프레임 수는
        // result 안이 아니라 메시지 최상위의 sequence_index 다.
        console.log(
          `${this.tag} result text=${JSON.stringify(r?.text)}` +
            ` conf=${String(r?.confidence)} final=${String(r?.is_final)}` +
            ` window=${String(asRecord(r?.sequence)?.window_index)}` +
            ` seq=${String(m?.sequence_index)}`,
        )
        if (RESULT_VERBOSE) {
          console.log(`${this.tag} result.sequence=`, JSON.stringify(r?.sequence))
          console.log(`${this.tag} result 전문:`, JSON.stringify(parsed))
        }

        // 파싱한 값을 밖으로 넘긴다. 전부 와이어에서 `unknown` 으로 온 값인데
        // 콜백 시그니처는 구체 타입을 약속하므로 필드마다 다시 검사한다.
        //
        // windowIndex 가 -1 이면 서버가 값을 주지 않았다는 뜻이다(60프레임 미만
        // 구간에서 null 로 온다). 소비자가 던지는 예외로 소켓이 죽으면 안 된다.
        if (this.onResult !== undefined) {
          const windowIndex = asRecord(r?.sequence)?.window_index
          // r 이 아니라 m 에서 읽는다 — 최상위 형제 필드다.
          const sequenceIndex = m?.sequence_index
          try {
            this.onResult({
              text: typeof r?.text === "string" ? r.text : "",
              confidence: typeof r?.confidence === "number" ? r.confidence : 0,
              isFinal: r?.is_final === true,
              windowIndex: typeof windowIndex === "number" ? windowIndex : -1,
              sequenceIndex: typeof sequenceIndex === "number" ? sequenceIndex : null,
            })
          } catch (err) {
            console.error(`${this.tag} onResult 콜백 예외:`, err)
          }
        }
        break
      }

      case "pong": {
        // default 로 떨어뜨리면 20초마다 "미지의 메시지" 경고가 찍혀 스키마 불일치처럼 보인다.
        // client_message_id 는 서버가 echo 한 값이라 어떤 ping 의 응답인지 짚힌다.
        console.log(
          `${this.tag} pong client_message_id=${String(m?.client_message_id)}` +
            ` server_time=${String(m?.server_time)}`,
        )
        break
      }

      case "error": {
        console.error(`${this.tag} error code=`, m?.code)
        console.error(`${this.tag} error message=`, m?.message)
        console.error(`${this.tag} error retryable=`, m?.retryable)
        console.error(`${this.tag} error 전문:`, JSON.stringify(parsed))

        // 와이어에서 unknown 으로 온 값이라 필드마다 다시 검사한다.
        if (this.onError !== undefined) {
          const code = m?.code
          const message = m?.message
          const retryable = m?.retryable
          try {
            this.onError({
              code: typeof code === "string" ? code : "unknown",
              message: typeof message === "string" ? message : "",
              retryable: typeof retryable === "boolean" ? retryable : null,
            })
          } catch (err) {
            console.error(`${this.tag} onError 콜백 예외:`, err)
          }
        }
        break
      }

      default: {
        // 미지의 타입을 조용히 버리면 안 된다. 클라이언트와 서버의 스키마가
        // 어긋난 걸 알아채는 통로가 여기다.
        console.warn(`${this.tag} 미지의 메시지 type=`, type)
        console.warn(`${this.tag} 미지의 메시지 원문:`, data)
        break
      }
    }
  }

  // -------------------------------------------------------------------------
  // 재연결
  // -------------------------------------------------------------------------

  private scheduleReconnect(why: string): void {
    this.clearPingTimer()
    if (this.shuttingDown) {
      console.log(`${this.tag} 종료 중이라 재연결하지 않는다`)
      return
    }
    if (this.reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.error(
        `${this.tag} 재연결 ${MAX_RECONNECT_ATTEMPTS}회 모두 실패 — 포기한다. 마지막 사유: ${why}`,
      )
      this.setState("error")
      return
    }

    this.reconnectAttempt += 1
    const delay = Math.min(RECONNECT_BASE_MS * Math.pow(2, this.reconnectAttempt - 1), RECONNECT_CAP_MS)
    console.warn(
      `${this.tag} 재연결 ${this.reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS} 회차 — ${delay}ms 후 시도. 사유: ${why}`,
    )

    if (this.reconnectTimer !== undefined) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined
      if (this.shuttingDown) return
      console.log(`${this.tag} 재연결 ${this.reconnectAttempt}회차 실행`)
      // 재연결은 항상 새 세션을 발급받는다. 기존 session_id 를 이어 쓰지
      // 않는다 — 서버에서 이미 stopped 이거나 만료됐을 수 있고, 그러면 1008 만
      // 받는다. 새 id 로 hello → ready 를 처음부터 다시 밟는다.
      void this.runConnect().catch((err) => {
        this.setState("error")
        console.error(`${this.tag} 재연결 중 예외:`, err)
      })
    }, delay)
  }

  // -------------------------------------------------------------------------
  // 종료
  // -------------------------------------------------------------------------

  private closeSocket(code: number, reason: string): void {
    const ws = this.ws
    if (ws === undefined) return
    try {
      ws.close(code, reason)
      console.log(`${this.tag} ws.close(${code}) 호출`)
    } catch (err) {
      console.error(`${this.tag} ws.close 실패:`, err)
    }
  }

  // -------------------------------------------------------------------------
  // 스트림 메시지
  // -------------------------------------------------------------------------

  /**
   * Mentra WHEP 스트림을 당겨 가라고 AI 서버에 알린다.
   *
   * `streamStartSent` 가 막아 준다 — 한 세션 안에서 stream_id 하나당 한 번이다.
   * 재연결로 세션이 바뀌면 다시 보낼 수 있다.
   * 동기 함수라 어느 콜백에서 불러도 안전하다. 아무것도 안 보냈으면 false.
   */
  sendStreamStart(streamId: string, webrtcUrl: string): boolean {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.error(`${this.streamTag} stream_start 전송 불가 — AI session_id 없음`)
      return false
    }
    if (this.streamStartSent.has(streamId)) {
      console.warn(`${this.streamTag} stream_start 중복 방지 — 이미 보낸 stream_id:`, streamId)
      return false
    }

    const clientMessageId = `stream-start-${Date.now()}`
    const sent = this.sendJson(
      {
        // stream_start/stream_stop 은 HELLO_SCHEMA 가 아니라 WebRTC 스키마를 탄다.
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
    console.log(`${this.streamTag} stream_start 전송 시각=`, new Date().toISOString())
    // 여기서 다시 찍는 건 의도다. 프레임은 흐르는데 result 가 하나도 안 올 때
    // "그 시점에 모델이 로드돼 있었나" 에 답해 주는 줄이다.
    console.log(`${this.streamTag} 스트림 시작 시점 model=`, JSON.stringify(this.readyModel))
    return sent
  }

  /**
   * 그만 당기라고 AI 서버에 알린다. `streamStopSent` 가 막아 주므로 공용 정리
   * 경로가 stop / error / disconnect 어디에서 돌아도 중복이 나가지 않는다.
   * 동기 함수라 beforeDisconnect 에서도 안전하다.
   */
  sendStreamStop(streamId: string): boolean {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.error(`${this.streamTag} stream_stop 전송 불가 — AI session_id 없음`)
      return false
    }
    if (this.streamStopSent.has(streamId)) {
      console.warn(`${this.streamTag} stream_stop 중복 방지 — 이미 보낸 stream_id:`, streamId)
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
    console.log(`${this.streamTag} stream_stop 전송 시각=`, new Date().toISOString())
    return sent
  }

  /**
   * 종료 절차 중 동기 부분만 담당한다. `session.on("beforeDisconnect")` 에서는
   * 동기 코드만 완료가 보장되므로(호스트가 기다려 주지 않고 소켓을 닫는다)
   * 거기서 부를 수 있는 것은 이런 형태뿐이다.
   */
  sendStopMessage(reason: string): boolean {
    const sessionId = this.sessionId
    if (sessionId === undefined) {
      console.warn(`${this.tag} stop 전송 생략 — session_id 없음`)
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

  /** 동기 종료. `session.on("disconnect")` 에서 부를 수 있다. */
  closeNow(reason: string): void {
    this.shuttingDown = true
    this.clearHandshakeTimer()
    this.clearPingTimer()
    if (this.reconnectTimer !== undefined) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = undefined
    }
    this.closeSocket(1000, reason)
    this.ws = undefined
  }

  /**
   * POST /v1/sessions/{id}/stop. best effort.
   *
   * 서버 쪽이 멱등이다. stop_session() 은 레코드를 "stopped" 로 표시할 뿐 TTL
   * 까지 유지하므로, WS `stop` 메시지가 이미 같은 일을 했어도 200 이 온다.
   * 여러 번 불러도 무해하다.
   *
   * disconnect 경로에서 불리는데 거기서는 완료가 보장되지 않는다 — 호스트가
   * 기다리지 않고 JSContext 를 내린다. 실패해도 정상이다: 서버 세션은
   * `expires_at`(생성 시각 + SESSION_TTL_SECONDS, 기본 1시간)에 스스로
   * 만료되므로 이 호출이 끝내 닿지 않아도 영구히 새는 것은 없다.
   */
  postStopBestEffort(): void {
    const sessionId = this.sessionId
    void this.postStop(sessionId).catch((err) => {
      console.warn(`${this.tag} postStop 예외 (정상 취급 — 세션은 expires_at 에 만료된다):`, err)
    })
  }

  private async postStop(sessionId: string | undefined): Promise<void> {
    if (sessionId === undefined) {
      console.warn(`${this.tag} POST /stop 생략 — session_id 없음`)
      return
    }
    const url = `${AI_HTTP}/v1/sessions/${sessionId}/stop`
    console.log(`${this.tag} POST`, url)

    let response: Response
    try {
      response = await fetch(url, {
        method: "POST",
        headers: {Accept: "application/json"},
      })
    } catch (err) {
      logFetchError(this.tag, "POST /stop", err)
      return
    }

    console.log(`${this.tag} POST /stop status=`, response.status, response.statusText)
    try {
      const raw = await response.text()
      console.log(`${this.tag} POST /stop body:`, raw)
    } catch (err) {
      console.error(`${this.tag} POST /stop body 읽기 실패:`, err)
    }

    if (response.status === 404) {
      console.warn(`${this.tag} POST /stop 404 — 세션이 이미 서버에서 사라졌다 (TTL 만료 등)`)
    }

    this.sessionId = undefined
    this.setState("idle")
  }
}
