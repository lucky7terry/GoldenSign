/**
 * Background JSContext entry point. The MentraOS host loads this file
 * inside a per-miniapp JSContext (iOS-JSC / Android-QuickJS) and calls
 * the `registerMiniapp` handler once the init envelope arrives.
 *
 * Runtime is a bare JS engine: console / timers / fetch / WebSocket /
 * localStorage only. No window, no DOM, no Node API, no dynamic import.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ⚠️  HOT RELOAD WARNING — READ BEFORE SAVING THIS FILE WITH A LIVE STREAM
 * ─────────────────────────────────────────────────────────────────────────
 * Per background/register.js, a dev reload kills and respawns the JSContext:
 * the polyfill + bundle are re-evaluated and the handler runs again with a
 * brand-new session. Module state does NOT survive — `appState` and
 * `activeStreamId` below are reset to their initial values.
 *
 * So if you save this file while a stream is running, the new context has no
 * streamId and cannot call stop() for it. The camera stays held by the old
 * stream and every subsequent start fails with a busy error until the phone
 * reaps it. Recovery: long-press once — the no-arg `session.stream.stop()`
 * fallback targets "the active stream" and does not need the lost id. If that
 * fails, restart the miniapp from the phone.
 *
 * ALWAYS STOP THE STREAM BEFORE SAVING.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Flow: "ready" connects the AI session automatically → long-press toggles the
 * WHEP stream → "disconnect" tears the AI session down automatically. The
 * button has exactly one meaning; see the switch in the long-press handler.
 *
 * Deliberately absent —
 *   - session.ui.*       (Gate 5)
 *   - session.display.*  (hasDisplay=false measured on Mentra Live, so there is
 *     no way to show mode state — which is why the gesture stays single-purpose)
 *   - new URL()          (typeof URL === "undefined" measured on-device)
 *
 * connect(): NOT called here. registerMiniapp calls session.connect()
 * itself right after the handler returns (see
 * @mentra/miniapp/dist/background/register.js). The manual
 * `await session.connect()` shown in session.d.ts's lifecycle comment
 * describes direct `new MiniappSession()` use, not the register path.
 */

import {registerMiniapp, type GlassesCapabilities, type StreamModule, type WifiData} from "@mentra/miniapp/background"
import {AiClient, probeRuntime} from "./ai-client"

/**
 * The background entry point re-exports `StreamModule` but not its option /
 * result interfaces, so they're derived from the method signature rather than
 * deep-imported from `@mentra/miniapp/dist/modules/stream` (a private path
 * the package's `exports` map doesn't expose).
 */
type StartStreamOptions = NonNullable<Parameters<StreamModule["startStream"]>[0]>
type StreamResult = Awaited<ReturnType<StreamModule["startStream"]>>

type UnknownRecord = Record<string, unknown>

/**
 * Narrow an `unknown` capability field to something indexable. The SDK types
 * only declare `display`; every other field arrives through
 * `GlassesCapabilities`'s `[key: string]: unknown` index signature, so each
 * hop has to be guarded rather than optional-chained off a typed shape.
 */
function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null ? (value as UnknownRecord) : undefined
}

/**
 * One-line capability digest. Called from both "ready" and "capabilities"
 * so a mid-session device swap is directly comparable to the initial values.
 *
 * `hasDisplay` here is the device-reported boolean, which is a different
 * source from the `display != null` check below — logging both is
 * deliberate, so a disagreement between them is visible instead of silent.
 *
 * Every field is read defensively: absent (`undefined`) and explicitly
 * false are printed differently on purpose.
 */
function summarizeCapabilities(source: string, caps: GlassesCapabilities | null): void {
  const c = asRecord(caps)
  const buttons = asRecord(c?.button)?.buttons
  const firstButton = asRecord(Array.isArray(buttons) ? buttons[0] : undefined)

  console.log(
    `[Caps] ${source}` +
      ` modelName=${String(c?.modelName)}` +
      ` hasCamera=${String(c?.hasCamera)}` +
      ` hasDisplay=${String(c?.hasDisplay)}` +
      ` hasWifi=${String(c?.hasWifi)}` +
      ` hasLight=${String(c?.hasLight)}` +
      ` button[0].events=${JSON.stringify(firstButton?.events)}`,
  )
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

/**
 * `session.stream.*` goes through `session.sendRequest`, which rejects with a
 * PLAIN OBJECT `{code, message}` — verified in session.js (the REQUEST_RESULT
 * `ok:false` branch and the timeout branch both call `pending.reject(err)`
 * with an object literal, not an Error). So `err instanceof Error` is false
 * on the normal failure path and there is no `.stack`.
 *
 * The one exception is NotConnectedError, a real Error subclass thrown when
 * the session is disposed or pre-ACK. Both shapes are handled here.
 */
function logRequestError(label: string, err: unknown): void {
  // Raw value first, before any interpretation of it.
  console.error(`${label} err:`, err)

  const e = asRecord(err)
  console.error(`${label} err.code:`, String(e?.code))
  console.error(`${label} err.message:`, String(e?.message))

  // A plain {code,message} object stringifies to "[object Object]" in some
  // engines' console, so serialize it too. Errors don't survive
  // JSON.stringify, hence the instanceof branch.
  if (err instanceof Error) {
    console.error(`${label} err.name/stack:`, err.name, err.stack ?? "(no stack)")
  } else {
    console.error(`${label} err JSON:`, JSON.stringify(err))
  }
}

/**
 * Log a URL's parts. Delegates to parseUrlParts — measured on-device,
 * `typeof URL === "undefined"`, so regex is the only option. On a parse miss
 * the raw string is logged verbatim rather than dropped.
 */
function logUrlParts(label: string, url: unknown): void {
  if (typeof url !== "string" || url.length === 0) return

  const parts = parseUrlParts(url)
  if (parts === undefined) {
    console.warn(`[Gate4] ${label} URL 파싱 실패 — 원본:`, url)
    return
  }
  console.log(`[Gate4] ${label} protocol=${parts.protocol}`)
  console.log(`[Gate4] ${label} hostname=${parts.hostname}`)
  console.log(`[Gate4] ${label} port=${parts.port}`)
  console.log(`[Gate4] ${label} full=`, url)
}

/** Pull every field of interest out of a successful StreamResult. */
function logStreamResult(result: StreamResult): void {
  console.log("[Gate4] 성공 result 전체:", JSON.stringify(result, null, 2))

  console.log("[Gate4] streamId=", result?.streamId)
  console.log("[Gate4] status=", result?.status)
  console.log("[Gate4] mode=", result?.mode)
  console.log("[Gate4] liveInputId=", result?.liveInputId)
  console.log("[Gate4] webrtcUrl=", result?.webrtcUrl)
  console.log("[Gate4] hlsUrl=", result?.hlsUrl)
  console.log("[Gate4] dashUrl=", result?.dashUrl)

  const resolved = result?.resolvedConfig
  if (resolved === undefined) {
    // Loud on purpose: without resolvedConfig we cannot tell which transport
    // was actually negotiated, which is the core question of this Gate.
    console.warn("[Gate4] resolvedConfig 미제공 — Mentra 문의 대상")
  } else {
    console.log("[Gate4] resolvedConfig:", JSON.stringify(resolved, null, 2))
    // Which of rtmp/srt/whip actually won the negotiation.
    console.log("[Gate4] resolvedConfig.transport=", resolved?.transport)
    // Whether the requested fps:30 survived.
    console.log("[Gate4] resolvedConfig.video.fps=", resolved?.video?.fps)
    console.log(
      "[Gate4] resolvedConfig.video WxH=",
      `${String(resolved?.video?.width)}x${String(resolved?.video?.height)}`,
    )
  }

  logUrlParts("webrtcUrl", result?.webrtcUrl)
}

/** Requested encode config. Compared against resolvedConfig.video.fps at start. */
const VIDEO_CONFIG = {width: 1280, height: 720, fps: 30} as const

/** Requested fps, kept separate so the log can state request vs negotiated. */
const REQUESTED_FPS = VIDEO_CONFIG.fps

/** Low-resolution fallback for rung D. */
const VIDEO_CONFIG_LOW = {width: 640, height: 480, fps: 15} as const

/**
 * TEMPORARY diagnostic ladder. Ordered; the first success wins.
 *
 * `ingest:"whip"` is measured working on Mentra Live, so rung A is the normal
 * path and the rest only run when it starts failing. The point is to record
 * WHICH rung works when A stops working, so this trades startup latency for
 * information — each failed rung costs its own timeout plus INTER_ATTEMPT_MS.
 *
 * ⚠️  Only rungs A and D request WHIP. B and C negotiate SRT, which returns
 * hlsUrl/dashUrl and NO webrtcUrl — so even when they "succeed" the webrtcUrl
 * validation downstream rejects them and rolls the stream back. That is
 * correct (the AI server pulls WHEP; it cannot consume HLS), but it means a
 * B/C success is a diagnostic result, not a working stream. Expect the log to
 * read "시도 B 성공" followed by "롤백: webrtcUrl 없음".
 *
 * TODO: 진단이 끝나면 A 단일 호출로 되돌릴 것.
 */
const STREAM_ATTEMPTS: ReadonlyArray<{name: string; note: string; options: StartStreamOptions}> = [
  {
    name: "A",
    note: 'ingest:"whip" 1280x720@30 — 정상 경로. webrtcUrl 기대',
    options: {ingest: "whip", video: VIDEO_CONFIG, sound: false},
  },
  {
    name: "B",
    note: "ingest 생략(기본 srt) — 폰 기본값 관찰용. webrtcUrl 안 나온다",
    options: {video: VIDEO_CONFIG, sound: false},
  },
  {
    name: "C",
    note: 'ingest:"srt" 명시 — HLS/DASH 경로. webrtcUrl 안 나온다',
    options: {ingest: "srt", video: VIDEO_CONFIG, sound: false},
  },
  {
    name: "D",
    note: 'ingest:"whip" 640x480@15 — 저해상도. 인코더 부하가 원인인지 판별',
    options: {ingest: "whip", video: VIDEO_CONFIG_LOW, sound: false},
  },
]

/**
 * Pause between rungs, after a best-effort stop(). The phone needs a moment to
 * actually release the camera — retrying immediately just earns a busy error
 * and misattributes it to the next rung's options.
 */
const INTER_ATTEMPT_MS = 1000

/** Longest we wait for glasses Wi-Fi to come up before prompting the user. */
const WIFI_WAIT_MS = 5000
const WIFI_POLL_MS = 250

/**
 * Extract a WHEP URL's parts without `new URL()`.
 *
 * Measured on-device: `typeof URL === "undefined"` in this runtime, so the URL
 * global is not merely unconfirmed — it is absent. Regex is the only option.
 *
 * Returns undefined when the string doesn't parse, which the caller treats as
 * a hard failure: an unparseable webrtcUrl means the server can't pull it
 * either, so the stream must be rolled back rather than left running.
 */
function parseUrlParts(url: string): {protocol: string; hostname: string; port: string} | undefined {
  const m = /^([a-z]+):\/\/([^/:?#]+)(?::(\d+))?/i.exec(url)
  if (!m) return undefined
  return {protocol: m[1], hostname: m[2], port: m[3] ?? "(없음)"}
}

/**
 * Full app state. `starting_stream` in particular lasts ~5s (measured), which
 * is why the re-entrancy guard matters more than it looks.
 */
type AppState =
  | "idle"
  | "connecting_ai"
  | "ai_ready"
  | "waiting_wifi"
  | "starting_stream"
  | "streaming"
  | "stopping"
  | "error"

registerMiniapp((session) => {
  /**
   * Every subscription's unsubscribe fn lands here and is drained on
   * "disconnect". `session.on(...)` and `session.input.onButtonPress(...)`
   * both return an unsubscribe function.
   */
  const unsubscribers: Array<() => void> = []

  // --- mutable state ------------------------------------------------------
  // Per-session, NOT persisted. See the hot-reload warning at the top of the
  // file: a dev reload resets all of it.
  let appState: AppState = "idle"
  let activeStreamId: string | undefined
  /** Latest glasses Wi-Fi state. Undefined until the first onWifi delivery. */
  let latestWifi: WifiData | undefined
  /**
   * True while the start sequence (wifi wait → startStream) is running.
   * Separate from `appState` because the sequence spans two states and a
   * second long-press in either must not launch a parallel run.
   */
  let startInFlight = false

  // Created inside "ready" because session.userId is only populated by
  // CONNECT_ACK. Undefined when the runtime probe fails.
  let ai: AiClient | undefined

  function setState(next: AppState): void {
    if (appState === next) return
    console.log(`[State] ${appState} -> ${next}`)
    appState = next
  }

  // --- permissions (진단 로그 전용) ---------------------------------------

  // 여기서 등록하는 이유: 이 핸들러 본문은 전부 동기라 아직 첫 await 이전이고,
  // "permissions" 는 CONNECT_ACK 처리 중 applyPermissions 에서 방출되는데
  // "ready" 와의 방출 순서가 SDK 어디에도 명시돼 있지 않다. ready 구독보다
  // 먼저 걸어 두면 순서가 어느 쪽이든 놓치지 않는다.
  //
  // ⚠️ 여기서 말하는 permission 은 "매니페스트에 선언됐는가" 이지 "OS 가
  // 허가했는가" 가 아니다 (modules/permissions.d.ts 헤더 주석). 실제 게이트는
  // 폰 런타임에 있고 클라이언트 SDK 는 아무것도 막지 않는다 — 그래서 관찰이
  // 유일한 확인 수단이다.
  unsubscribers.push(
    session.on("permissions", (p) => {
      console.log("[PERM] update", JSON.stringify(p))
    }),
  )

  // PERMISSION_NOT_DECLARED 전용 채널. 관찰만 한다 — setState 를 부르지 않는다.
  // 미선언 권한은 해당 구독/요청 하나만 거부당할 뿐 세션은 살아있으므로, 우리
  // 스트림 경로와 무관한 권한 때문에 error 로 떨어뜨리면 롱프레스만 죽는다.
  unsubscribers.push(
    session.permissions.onPermissionError((err) => {
      console.log("[PERM] error code=", err.code)
      console.log("[PERM] error message=", err.message)
      console.log("[PERM] error permission=", String(err.permission))
      console.log("[PERM] error subscription=", String(err.subscription))
      console.log("[PERM] error operation=", String(err.operation))
    }),
  )

  // --- session lifecycle -------------------------------------------------

  // "ready" fires with no arguments — the CONNECT_ACK values are read off
  // the session object itself.
  unsubscribers.push(
    session.on("ready", () => {
      console.log("[Session] userId:", session.userId)
      console.log("[Session] packageName:", session.packageName)
      console.log("[Session] visibility:", session.visibility)
      summarizeCapabilities("ready", session.capabilities)
      // Dumped whole, unfolded, on purpose: the point of this Gate is to
      // discover fields the SDK types don't document. The summary above is a
      // convenience, not a replacement — read this dump for anything new.
      console.log("[Session] capabilities:", JSON.stringify(session.capabilities, null, 2))

      // Explicit display verdict. `false` on Mentra Live is correct, not a bug.
      const hasDisplay = session.capabilities?.display != null
      console.log("[Caps] hasDisplay =", hasDisplay)

      // 선언된 매니페스트 권한 스냅샷. _permissions 는 CONNECT_ACK 에서 채워지므로
      // 여기가 유효한 값이 나오는 첫 지점이다. session.permissions 는
      // PermissionsModule (session.d.ts:165) 이고 getAll() 이 PermissionRecord
      // ({location, microphone, camera, notifications, calendar}: boolean) 를 준다.
      console.log("[PERM] snapshot", JSON.stringify(session.permissions.getAll()))

      // --- AI connection ----------------------------------------------------
      // Runtime probe first: without WebSocket nothing downstream is possible.
      // All network work starts here rather than in the registerMiniapp handler
      // body, which must stay synchronous.
      if (!probeRuntime()) {
        setState("error")
        return
      }
      setState("connecting_ai")
      // onReady also fires after a reconnect, so this returns us to ai_ready
      // without the user having to do anything.
      ai = new AiClient(session.userId, () => {
        if (appState === "connecting_ai" || appState === "error") setState("ai_ready")
        else console.log("[Gate3] ready 수신했지만 state 유지:", appState)
      })
      ai.connect()
    }),
  )

  unsubscribers.push(session.on("error", (e) => console.error("[Session] error:", e)))

  unsubscribers.push(session.on("visibility", (v) => console.log("[Session] visibility:", v)))

  unsubscribers.push(
    session.on("capabilities", (c) => {
      summarizeCapabilities("changed", c)
      console.log("[Session] capabilities changed:", JSON.stringify(c))
    }),
  )

  // Synchronous only. Per SDK: async work started here will not complete
  // before the socket closes.
  unsubscribers.push(
    session.on("beforeDisconnect", (r) => {
      console.log("[Session] beforeDisconnect:", r)
      // ws.send() ONLY. Everything here is synchronous so it has a chance to
      // reach the wire before the host tears the socket down — per the SDK,
      // async work started in this handler will not complete.
      // Deliberately absent: session.stream.stop() and POST /stop, both async.
      if (activeStreamId !== undefined) {
        // stream_stop before stop, so the server unwinds the pull first.
        ai?.sendStreamStop(activeStreamId)
      }
      ai?.sendStopMessage(`beforeDisconnect: ${r}`)
    }),
  )

  unsubscribers.push(
    session.on("disconnect", (r) => {
      console.log("[Session] disconnect:", r)

      if (activeStreamId !== undefined || appState === "streaming") {
        console.warn("[Gate4] disconnect 시점에 스트림이 살아있었다. streamId=", activeStreamId)
      }

      ai?.closeNow(`disconnect: ${r}`)

      // Best effort from here on — the transport is already going away, so
      // neither of these is guaranteed to complete.
      const idAtDisconnect = activeStreamId
      void session.stream
        .stop(idAtDisconnect)
        .then(() => console.log("[Gate4] disconnect stream.stop 성공"))
        .catch((err) => {
          console.warn("[Gate4] disconnect stream.stop 실패 (정상 취급)")
          logRequestError("[Gate4] disconnect stream.stop", err)
        })

      // Attempted, but failure is expected and fine: the JSContext may be gone
      // before the request lands. The server session expires on its own at
      // expires_at (creation + SESSION_TTL_SECONDS, 1 hour by default), so
      // nothing leaks permanently even when this never completes.
      ai?.postStopBestEffort()

      activeStreamId = undefined
      setState("idle")

      for (const unsubscribe of unsubscribers) {
        unsubscribe()
      }
      unsubscribers.length = 0
    }),
  )

  // --- glasses Wi-Fi ------------------------------------------------------

  // Fires the current state on subscribe and on every change. Measured: the
  // first accurate value takes ~4s to arrive, which is exactly why the start
  // sequence waits rather than rejecting on the initial value.
  unsubscribers.push(
    session.glasses.onWifi((data) => {
      latestWifi = data
      console.log("[Gate4] WiFi:", JSON.stringify(data))
    }),
  )

  // --- preflight ----------------------------------------------------------

  /**
   * Hard capability gate — a device with no camera can never stream, so unlike
   * Wi-Fi this is a block rather than a wait.
   */
  function hasCameraOrLog(): boolean {
    const c = asRecord(session.capabilities)
    if (c?.hasCamera !== true) {
      console.error("[Gate4] hasCamera !== true — 스트림 불가. hasCamera=", c?.hasCamera)
      return false
    }
    console.log("[Gate4] modelName=", c?.modelName)
    console.log(
      "[Gate4] supportedStreamTypes=",
      JSON.stringify(asRecord(asRecord(c?.camera)?.video)?.supportedStreamTypes),
    )
    return true
  }

  /**
   * Wait (don't reject) for glasses Wi-Fi.
   *
   * onWifi takes ~4s to deliver an accurate value, so checking once right after
   * app open reports "not connected" for a device that is, in fact, fine. That
   * made the button look dead. Polling the latest value for up to 5s fixes it.
   */
  async function waitForWifi(): Promise<boolean> {
    const deadline = Date.now() + WIFI_WAIT_MS
    console.log("[Gate4] WiFi 대기 시작. 현재=", JSON.stringify(latestWifi))

    while (Date.now() < deadline) {
      if (latestWifi?.connected === true) {
        console.log("[Gate4] WiFi 확인됨:", JSON.stringify(latestWifi))
        return true
      }
      await new Promise((resolve) => setTimeout(resolve, WIFI_POLL_MS))
    }

    console.warn(`[Gate4] ${WIFI_WAIT_MS}ms 동안 WiFi 미연결. 마지막 값=`, JSON.stringify(latestWifi))
    return false
  }

  // --- shared cleanup -----------------------------------------------------

  /**
   * The single teardown path, reused by the stop sequence, the rollback, and
   * error recovery. Safe to call repeatedly:
   *   - `activeStreamId` is cleared up front, so a second call is a no-op
   *     against the same id;
   *   - stream_stop is additionally guarded by a Set inside AiClient.
   *
   * Order is deliberate: the AI server is told to stop pulling BEFORE the
   * Mentra stream goes away. Doing it the other way round leaves the server
   * pulling a dead WHEP endpoint.
   */
  async function cleanupStream(reason: string, opts: {notifyAi: boolean}): Promise<void> {
    const streamId = activeStreamId
    activeStreamId = undefined
    console.log(`[Gate4] cleanup 시작 (${reason}). streamId=`, streamId)

    // 1. AI WebSocket first.
    if (opts.notifyAi && streamId !== undefined) {
      ai?.sendStreamStop(streamId)
    }

    // 2. Then the Mentra stream. stop() resolves to void — only settled/failed
    // is observable. Falls back to the no-arg form, which the SDK documents as
    // "stop the active stream" and which is the escape hatch when the id is lost.
    try {
      await session.stream.stop(streamId)
      console.log("[Gate4] stream.stop 성공. streamId=", streamId)
    } catch (err) {
      console.error("[Gate4] stream.stop 실패 — 인자 없는 stop() 으로 폴백")
      logRequestError("[Gate4] stream.stop", err)
      try {
        await session.stream.stop()
        console.log("[Gate4] stream.stop() (인자 없음) 성공")
      } catch (err2) {
        console.error("[Gate4] stream.stop() (인자 없음)도 실패 — 스트림이 남아있을 수 있다")
        logRequestError("[Gate4] stream.stop()", err2)
        throw err2
      }
    }
  }

  // --- start ladder (temporary diagnostic) --------------------------------

  /**
   * Walk STREAM_ATTEMPTS until one succeeds. Returns the winning result plus
   * the fps that rung actually asked for, or undefined when all four fail.
   *
   * Between rungs: a best-effort no-arg stop() then a 1s pause. A rung can
   * fail *after* the phone has already provisioned something, and without the
   * stop the next rung inherits a held camera and fails for the wrong reason.
   * The stop is expected to fail when nothing is live — that is not an error.
   */
  async function runStreamLadder(): Promise<{result: StreamResult; requestedFps: number} | undefined> {
    for (let i = 0; i < STREAM_ATTEMPTS.length; i += 1) {
      const attempt = STREAM_ATTEMPTS[i]
      console.log(`[Gate4] 시도 ${attempt.name} 시작`, JSON.stringify(attempt.options), `(${attempt.note})`)
      const startedAt = Date.now()

      try {
        const result = await session.stream.startStream(attempt.options)
        console.log(`[Gate4] 시도 ${attempt.name} 성공 (${Date.now() - startedAt}ms)`)
        logStreamResult(result)
        return {result, requestedFps: attempt.options.video?.fps ?? REQUESTED_FPS}
      } catch (err) {
        console.error(`[Gate4] 시도 ${attempt.name} 실패 (${Date.now() - startedAt}ms)`)
        // logRequestError prints err.code and err.message on separate lines.
        // There is no camera-busy code in MiniappErrorCode — a held camera
        // arrives as INTERNAL with the detail in `message`, so read that line
        // rather than matching on code.
        logRequestError(`[Gate4] 시도 ${attempt.name}`, err)
      }

      if (i < STREAM_ATTEMPTS.length - 1) {
        try {
          await session.stream.stop()
          console.log(`[Gate4] 시도 ${attempt.name} 후 정리용 stop() 성공`)
        } catch {
          // Expected when nothing was provisioned. Not logged as an error.
          console.log(`[Gate4] 시도 ${attempt.name} 후 정리용 stop() — 정리할 스트림 없음`)
        }
        console.log(`[Gate4] ${INTER_ATTEMPT_MS}ms 대기 후 다음 시도`)
        await new Promise((resolve) => setTimeout(resolve, INTER_ATTEMPT_MS))
      }
    }
    return undefined
  }

  // --- start sequence -----------------------------------------------------

  async function runStartSequence(): Promise<void> {
    if (startInFlight) {
      console.warn("[Gate4] 시작 시퀀스가 이미 진행 중이다 — 무시. state=", appState)
      return
    }
    if (!hasCameraOrLog()) {
      setState("error")
      return
    }
    if (ai?.isReady() !== true) {
      console.error("[Gate4] AI 세션이 준비되지 않았다 — 스트림 시작 불가. aiState=", ai?.getState())
      return
    }

    startInFlight = true
    try {
      // 1. Wi-Fi.
      setState("waiting_wifi")
      if (!(await waitForWifi())) {
        // Stay in waiting_wifi so another long-press retries after the user
        // finishes the setup flow.
        void session.glasses
          .requestWifiSetup("수어 인식을 위해 안경을 WiFi에 연결해주세요")
          .then(() => console.log("[Gate4] requestWifiSetup 호출됨"))
          .catch((err) => logRequestError("[Gate4] requestWifiSetup", err))
        console.warn("[Gate4] WiFi 설정 유도 후 대기 — 연결 뒤 롱프레스로 재시도해라")
        return
      }

      // 2. Start the Mentra stream. Each rung is measured ~5s.
      setState("starting_stream")
      const attempt = await runStreamLadder()
      if (attempt === undefined) {
        setState("error")
        console.error("[Gate4] A/B/C/D 전부 실패 — error 상태")
        return
      }
      const {result, requestedFps} = attempt

      const streamId = result?.streamId
      const webrtcUrl = result?.webrtcUrl
      // Requested vs negotiated, side by side. requestedFps comes from the rung
      // that actually won, so D's 15 isn't reported as 30.
      console.log("[Gate4] fps 요청=", requestedFps, "/ resolvedConfig=", result?.resolvedConfig?.video?.fps)

      // Track it before validation so a rollback can always find the id.
      activeStreamId = streamId

      // 3. Validate the WHEP URL. Anything unusable here is a hard failure —
      // the server could not pull it either.
      if (typeof webrtcUrl !== "string" || webrtcUrl.length === 0) {
        console.error("[Gate4] webrtcUrl 이 없다 — 롤백한다. result.mode=", result?.mode)
        await rollback("webrtcUrl 없음")
        return
      }
      const parts = parseUrlParts(webrtcUrl)
      if (parts === undefined) {
        console.error("[Gate4] webrtcUrl 정규식 파싱 실패 — 롤백한다. 원본:", webrtcUrl)
        await rollback("webrtcUrl 파싱 실패")
        return
      }
      console.log(
        `[Gate4] webrtcUrl protocol=${parts.protocol} hostname=${parts.hostname} port=${parts.port}`,
      )
      if (typeof streamId !== "string" || streamId.length === 0) {
        console.error("[Gate4] streamId 가 없다 — stream_start 를 보낼 수 없다. 롤백한다")
        await rollback("streamId 없음")
        return
      }

      // 4. Hand the URL to the AI server.
      if (!ai.sendStreamStart(streamId, webrtcUrl)) {
        console.error("[Gate4] stream_start 전송 실패 — 롤백한다")
        await rollback("stream_start 전송 실패")
        return
      }

      // 5. Done.
      setState("streaming")
      console.log("[Gate4] streaming. streamId=", streamId)
    } finally {
      startInFlight = false
    }
  }

  /**
   * Undo a half-started stream.
   *
   * The previous build never called stop() on the failure paths, so a stream
   * that failed validation stayed alive holding the camera — every later start
   * then failed busy. Rollback is what prevents that.
   *
   * `notifyAi: false` because stream_start either was never sent or failed to
   * send; there is nothing for the server to stop pulling.
   *
   * Ends in ai_ready, not idle: the AI session is untouched by a stream failure.
   */
  async function rollback(why: string): Promise<void> {
    console.warn("[Gate4] 롤백:", why)
    try {
      await cleanupStream(`rollback: ${why}`, {notifyAi: false})
    } catch {
      // cleanupStream already logged both stop attempts.
      console.error("[Gate4] 롤백 중 stop 실패 — 카메라가 잡혀있을 수 있다")
    }
    setState(ai?.isReady() === true ? "ai_ready" : "error")
  }

  // --- stop sequence ------------------------------------------------------

  async function runStopSequence(reason: string): Promise<void> {
    setState("stopping")
    try {
      // stream_stop goes out over the still-open WebSocket first. The previous
      // build closed the socket before this, so stream_stop never shipped.
      await cleanupStream(reason, {notifyAi: true})
      setState(ai?.isReady() === true ? "ai_ready" : "error")
    } catch {
      setState("error")
      console.error("[Gate4] 정지 실패 — error 상태. 롱프레스로 재시도 가능")
    }
  }

  // --- input -------------------------------------------------------------

  // Physical button only. session.input.onTouch is intentionally NOT wired:
  // its "long_press" is a touchpad gesture, a different surface from the
  // physical button below.
  unsubscribers.push(
    session.input.onButtonPress((press) => {
      if (press.pressType === "long") {
        // Full payload logged so we can see what `buttonId` actually carries.
        console.log("[Input] LONG", JSON.stringify(press), "state=", appState)

        // ONE gesture, ONE meaning: toggle the stream. The AI session connects
        // on "ready" and tears down on "disconnect", both automatic — long-press
        // never touches it. No double_press either: with hasDisplay=false there
        // is no way to show the user which mode they are in, so a second
        // gesture would be unguessable.
        switch (appState) {
          case "ai_ready":
          case "waiting_wifi":
            // waiting_wifi is a retry entry point: the user connects Wi-Fi via
            // the setup flow, comes back, and presses again.
            void runStartSequence().catch((err) => {
              setState("error")
              logRequestError("[Gate4] 시작 시퀀스 예외", err)
            })
            return

          case "streaming":
            void runStopSequence(`long press (buttonId=${press.buttonId})`).catch((err) => {
              setState("error")
              logRequestError("[Gate4] 정지 시퀀스 예외", err)
            })
            return

          case "error":
            // Recovery: clear whatever is held, then fall back to ai_ready if
            // the AI socket survived. Also the hot-reload escape hatch.
            console.warn("[Gate4] error 상태 — 복구 정지 시도")
            void runStopSequence("error 복구").catch((err) => {
              setState("error")
              logRequestError("[Gate4] 복구 정지 예외", err)
            })
            return

          default:
            // idle / connecting_ai: AI not up yet.
            // starting_stream (~5s) / stopping: in flight, second press would race.
            console.warn("[Gate4] 지금은 롱프레스를 받지 않는다. state=", appState)
            return
        }
      } else {
        // Short press is inert by design. The old build triggered a photo
        // capture here, which is what caused camera_busy — not carried over.
        console.log("[Input] short (무시)")
      }
    }),
  )
})
