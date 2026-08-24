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

import {
  registerMiniapp,
  type GlassesCapabilities,
  type IsRpc,
  type LedColor,
  type StreamModule,
  type UIModule,
  type WifiData,
} from "@mentra/miniapp/background"
import type {Channels, Snapshot} from "../shared/channels"
import {AiClient, probeRuntime, type AiClientState} from "./ai-client"

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
 * Narrow the `model` block from the server's `ready` before it goes on a typed
 * channel. AiClient keeps it as `unknown` on purpose — it's a raw wire value —
 * so every field is checked here.
 *
 * Returns `null` (not undefined) on a shape mismatch: JSON.stringify deletes
 * undefined keys, and the UI must be able to see "we asked and got nothing".
 */
function readModelBlock(value: unknown): Channels["ai:state"]["model"] {
  const m = asRecord(value)
  if (m === undefined) return null
  return {
    loaded: m.loaded === true,
    mode: typeof m.mode === "string" ? m.mode : "",
    version: typeof m.version === "string" ? m.version : "",
  }
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

/**
 * Stream attempts, in order; the first success wins. One rung today.
 *
 * The B/C/D diagnostic rungs were removed — they could not do what they looked
 * like they were doing:
 *
 *   - B/C negotiate SRT, which returns hlsUrl/dashUrl and NO webrtcUrl. Our AI
 *     server PULLS WHEP and cannot consume HLS/DASH, so a B/C "success" was
 *     never a usable stream — runStartSequence's webrtcUrl check rejected it
 *     and rolled it back every time.
 *   - D was therefore unreachable. runStreamLadder returns on the first
 *     startStream success, and the rollback ends runStartSequence, so the walk
 *     stopped at B and never reached D.
 *   - The cost was paid in front of the audience: A fail (~5s) → cleanup stop
 *     + 1s → B "success" (~5s) → rollback → failure. Fifteen-plus seconds of
 *     nothing, ending in nothing.
 *
 * Reviving D (low-res WHIP, to test whether encoder load is the cause) needs
 * TWO changes, not just a new array entry: runStreamLadder must keep walking
 * when a rung succeeds but fails downstream validation, and the inter-rung
 * cleanup — a best-effort no-arg stop() plus a ~1s pause so the phone actually
 * releases the camera — has to come back with it. Without that pause the next
 * rung inherits a held camera and fails busy, misattributing the failure to
 * its own options. B and C are not worth reviving at all.
 */
const STREAM_ATTEMPTS: ReadonlyArray<{name: string; note: string; options: StartStreamOptions}> = [
  {
    name: "A",
    note: 'ingest:"whip" 1280x720@30 — 정상 경로. webrtcUrl 기대',
    options: {ingest: "whip", video: VIDEO_CONFIG, sound: false},
  },
]

/** Longest we wait for glasses Wi-Fi to come up before prompting the user. */
const WIFI_WAIT_MS = 5000
const WIFI_POLL_MS = 250

/**
 * Longest we wait for the AI socket before giving up on a long-press.
 *
 * Sized against AiClient's backoff: the first reconnect fires 1s after the
 * close, so 3s covers that attempt plus its handshake. Later attempts (2s, 4s,
 * 8s, 16s) fall outside on purpose — waiting 16s with no way to tell the
 * wearer anything is worse than letting them press again.
 */
const AI_WAIT_MS = 3000
const AI_POLL_MS = 100

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

// ---------------------------------------------------------------------------
// UI bridge
// ---------------------------------------------------------------------------

/**
 * Broadcast channels only — every key of `Channels` whose value is not
 * `Rpc<Req, Res>`. `IsRpc<T>` tests for that brand; mapping over the registry
 * and indexing back by its own keys drops the RPC entries to `never`.
 */
type BroadcastChannel = {
  [C in keyof Channels]: IsRpc<Channels[C]> extends true ? never : C
}[keyof Channels]

/** snapshot.results ring size. Oldest dropped first. */
const MAX_RESULTS = 20

/**
 * Minimum gap between NON-final recognition broadcasts.
 *
 * Two reasons, not one: the obvious render cost, and the SDK's per-channel
 * inbound buffer of 32 payloads (modules/ui.d.ts). A UI attaching mid-stream
 * would overflow that buffer and silently lose messages without a cap here.
 */
const RESULT_THROTTLE_MS = 200

/**
 * Collapse AiClient's seven-value state onto the four the UI cares about.
 * `AiClientState` itself is deliberately left alone — this is a projection,
 * not a new state in that machine.
 */
function aiPhase(state: AiClientState | undefined): Channels["ai:state"]["state"] {
  switch (state) {
    case "creating_session":
    case "connecting_ws":
    case "handshaking":
      return "connecting"
    case "ai_ready":
      return "ready"
    case "error":
      return "error"
    default:
      // idle / closing / undefined (client not constructed yet)
      return "disconnected"
  }
}

// ---------------------------------------------------------------------------
// LED feedback
// ---------------------------------------------------------------------------

/**
 * The wearer-facing LED is the ONLY status channel on this device —
 * hasDisplay is false on Mentra Live, so there is nowhere to draw text.
 *
 * Scope note: `LedModule` exposes no light-id parameter. Its header states the
 * phone maps color names onto per-device LED indices, so this API physically
 * cannot address the front-facing privacy light — the system owns that one and
 * we never touch it. Nothing here needs to guard against it.
 *
 * UNITS — verified, not assumed:
 *   - led.d.ts:16,18 annotate `ontime` / `offtime` as "LED on/off duration in ms".
 *   - `blink(color, ontime, offtime, count)` forwards its args straight into
 *     turnOn({ontime, offtime, count}) (led.js:31-33) → same ms units.
 *   - `solid(color, duration)` has NO unit annotation of its own, but led.js:36
 *     assigns `ontime: duration` — so duration inherits the documented ms.
 *   All three are therefore milliseconds by derivation from the SDK source.
 */

/**
 * How long one steady-state LED command is asked to hold.
 *
 * blink's `count` and solid's `duration` are both finite, but ai_ready and
 * streaming last minutes. Rather than adding a refresh interval (there is no
 * existing timer to piggyback on — we deliberately shipped no ping), each
 * transition arms the light for a generous window and we re-arm only when the
 * state actually changes. If the light is observed dying before the next
 * transition, that's the signal to revisit the refresh strategy.
 */
const LED_HOLD_MS = 30_000

/** Blink cadence. 400/400 reads as a clear pulse rather than a flicker. */
const LED_BLINK_ON_MS = 400
const LED_BLINK_OFF_MS = 400

/** Enough cycles to cover LED_HOLD_MS, so blink and solid hold equally long. */
const LED_BLINK_COUNT = Math.floor(LED_HOLD_MS / (LED_BLINK_ON_MS + LED_BLINK_OFF_MS))

type LedCommand = {kind: "off"} | {kind: "solid"; color: LedColor} | {kind: "blink"; color: LedColor}

/**
 * appState → LED. Exhaustive over AppState: adding a state without a mapping
 * is a compile error, not a silently dark light.
 *
 * There is no yellow in the five-color preset set, so waiting_wifi uses orange.
 */
function ledCommandFor(state: AppState): LedCommand {
  switch (state) {
    case "idle":
      return {kind: "off"}
    case "connecting_ai":
      return {kind: "blink", color: "blue"}
    case "ai_ready":
      return {kind: "solid", color: "blue"}
    case "waiting_wifi":
      return {kind: "blink", color: "orange"}
    case "starting_stream":
      return {kind: "blink", color: "green"}
    case "streaming":
      return {kind: "solid", color: "green"}
    case "stopping":
      return {kind: "off"}
    case "error":
      return {kind: "solid", color: "red"}
  }
}

/**
 * `hasLight` is not in GlassesCapabilities' declared surface — it arrives via
 * the `[key: string]: unknown` index signature, same as every field
 * summarizeCapabilities reads. Read live rather than cached so a mid-session
 * device swap ("capabilities" event) is respected.
 */
function deviceHasLight(caps: GlassesCapabilities | null): boolean {
  return asRecord(caps)?.hasLight === true
}

/**
 * One-line LED failure log.
 *
 * Not logRequestError: that prints four lines per failure, which is right for a
 * stream call that gates the whole app but far too loud for the light. An LED
 * failure changes nothing about what the app does, so it gets one warn line.
 *
 * `.message` is read structurally rather than off `Error`: session.led.* goes
 * through session.sendRequest, which rejects with a PLAIN {code, message}
 * object. JSON.stringify covers anything that isn't shaped that way.
 */
function logLedError(label: string, err: unknown): void {
  const message = asRecord(err)?.message
  console.warn(`[LED] ${label} 실패:`, typeof message === "string" ? message : JSON.stringify(err))
}

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

  // --- UI bridge ----------------------------------------------------------

  // session.d.ts:161 declares `readonly ui: UIModule` with NO generic argument,
  // so it defaults to Record<string, unknown>. That makes IsRpc<unknown> false,
  // which collapses handle()'s channel parameter to `never` and leaves send/on
  // payloads as `unknown`. Cast ONCE here; every call below goes through `ui`.
  //
  // Via `unknown` because a direct `as UIModule<Channels>` is rejected —
  // Record<string, unknown> and Channels don't overlap enough for TS to accept
  // a single-step assertion. The cast is sound: UIModule's generic only types
  // the channel/payload pairs, it doesn't change the object's runtime shape.
  const ui = session.ui as unknown as UIModule<Channels>

  // send()'s channel parameter is the conditional type
  // `IsRpc<Channels[C]> extends true ? never : C`, which TS refuses to evaluate
  // while C is still generic — so a generic wrapper can't call it directly.
  // BroadcastChannel already excludes every RPC channel, so narrow the
  // signature once here instead of casting at each call site.
  const sendBroadcast = ui.send as <C extends BroadcastChannel>(
    channel: C,
    payload: Channels[C],
  ) => void

  /**
   * What a late-mounting WebView gets from `getSnapshot`.
   *
   * Every unknown starts as `null`, never undefined — payloads cross the
   * bridge through JSON.stringify (envelope.js), which DELETES undefined keys.
   * With undefined the UI could not distinguish "not known yet" from "this
   * build doesn't have that field".
   */
  const snapshot: Snapshot = {
    ai: {state: "disconnected", sessionId: null, model: null, message: null},
    stream: {state: "idle", streamId: null, message: null},
    glasses: {wifiConnected: null, ssid: null, battery: null, charging: null},
    diagnostics: {
      requestedFps: null,
      resolvedFps: null,
      transport: null,
      webrtcHost: null,
      mode: null,
      status: null,
    },
    results: [],
  }

  // --- LED ------------------------------------------------------------------

  /** Last state actually pushed to the light. Guards against re-arming on every
   *  publishStreamState() call that only changed streamId. */
  let lastAppliedLedState: AppState | undefined

  /** Build (don't await) the LED promise for a command. */
  function sendLed(command: LedCommand) {
    switch (command.kind) {
      case "off":
        return session.led.turnOff()
      case "solid":
        return session.led.solid(command.color, LED_HOLD_MS)
      case "blink":
        return session.led.blink(command.color, LED_BLINK_ON_MS, LED_BLINK_OFF_MS, LED_BLINK_COUNT)
    }
  }

  /**
   * Drive the wearer LED from appState. Fire-and-forget: never awaited, so a
   * slow or absent ack can't delay a state transition, and a rejection can't
   * become an app error. The light is feedback, not a dependency.
   */
  function applyLed(state: AppState): void {
    if (!deviceHasLight(session.capabilities)) {
      // Logged on change only, so a lightless device doesn't spam every patch.
      if (lastAppliedLedState !== state) {
        lastAppliedLedState = state
        console.log("[LED] hasLight !== true — LED 표시 생략. state=", state)
      }
      return
    }
    if (lastAppliedLedState === state) return
    lastAppliedLedState = state

    const command = ledCommandFor(state)
    console.log("[LED]", state, "->", JSON.stringify(command))

    try {
      void sendLed(command).catch((err) => logLedError(state, err))
    } catch (err) {
      // Every LedModule method is `async`, so a synchronous throw shouldn't be
      // reachable. Caught anyway: an LED problem must never reach the caller.
      logLedError(`${state} 동기 예외`, err)
    }
  }

  /**
   * Unconditional off, bypassing the state map — for the teardown paths.
   *
   * Clears lastAppliedLedState so a later applyLed for the SAME state still
   * re-arms the light instead of being skipped as a duplicate.
   */
  function turnOffLed(reason: string): void {
    if (!deviceHasLight(session.capabilities)) return
    lastAppliedLedState = undefined
    console.log("[LED] turnOff:", reason)
    try {
      void session.led.turnOff().catch((err) => logLedError(`turnOff (${reason})`, err))
    } catch (err) {
      logLedError(`turnOff (${reason}) 동기 예외`, err)
    }
  }

  /**
   * The ONLY way state reaches the UI: update the snapshot slot, then
   * broadcast. Because both happen here, `getSnapshot` is current by
   * construction and no call site can send without recording.
   *
   * `ui.send` silently DROPS when no WebView is bound. That is the normal
   * case — background outlives the WebView — and is not an error.
   */
  function patch<C extends BroadcastChannel>(channel: C, next: Channels[C]): void {
    switch (channel) {
      case "ai:state":
        snapshot.ai = next as Channels["ai:state"]
        break
      case "stream:state": {
        const stream = next as Channels["stream:state"]
        snapshot.stream = stream
        // The one and only LED call site. Every appState transition funnels
        // through setState → publishStreamState → patch, so the 8-state mapping
        // lives here instead of being scattered across 16 setState call sites.
        // Calls that only changed streamId land here too and are skipped by
        // applyLed's lastAppliedLedState guard.
        applyLed(stream.state)
        break
      }
      case "glasses:state":
        snapshot.glasses = next as Channels["glasses:state"]
        break
      case "stream:diagnostics":
        snapshot.diagnostics = next as Channels["stream:diagnostics"]
        break
      case "recognition:result":
        snapshot.results.push(next as Channels["recognition:result"])
        if (snapshot.results.length > MAX_RESULTS) {
          snapshot.results.splice(0, snapshot.results.length - MAX_RESULTS)
        }
        break
      default:
        // "error" is an event, not a state — no snapshot slot. Broadcast only.
        break
    }
    sendBroadcast(channel, next)
  }

  /** Publish the appState + activeStreamId pair as it stands right now. */
  function publishStreamState(message: string | null = null): void {
    patch("stream:state", {state: appState, streamId: activeStreamId ?? null, message})
  }

  /** Publish the AI phase + identity as they stand right now. */
  function publishAiState(
    state: Channels["ai:state"]["state"],
    message: string | null = null,
  ): void {
    patch("ai:state", {
      state,
      sessionId: ai?.getSessionId() ?? null,
      model: readModelBlock(ai?.getModel()),
      message,
    })
  }

  // --- recognition throttle -------------------------------------------------

  let lastResultSentAt = 0
  let pendingResult: Channels["recognition:result"] | undefined
  let resultTimer: ReturnType<typeof setTimeout> | undefined

  function clearResultTimer(): void {
    if (resultTimer !== undefined) {
      clearTimeout(resultTimer)
      resultTimer = undefined
    }
  }

  /**
   * Throttle non-final results to one per RESULT_THROTTLE_MS; let finals
   * through immediately — those are what the wearer is actually waiting on.
   *
   * Trailing edge, newest-wins: a suppressed interim is held and emitted when
   * the window closes, and a final drops any held interim rather than letting
   * it arrive after the answer. Consequence worth knowing: snapshot.results
   * holds what was BROADCAST, not every result the server produced.
   */
  function publishResult(next: Channels["recognition:result"]): void {
    if (next.isFinal) {
      clearResultTimer()
      pendingResult = undefined
      lastResultSentAt = Date.now()
      patch("recognition:result", next)
      return
    }

    const elapsed = Date.now() - lastResultSentAt
    if (elapsed >= RESULT_THROTTLE_MS) {
      lastResultSentAt = Date.now()
      patch("recognition:result", next)
      return
    }

    pendingResult = next
    if (resultTimer === undefined) {
      resultTimer = setTimeout(() => {
        resultTimer = undefined
        const queued = pendingResult
        pendingResult = undefined
        if (queued !== undefined) {
          lastResultSentAt = Date.now()
          patch("recognition:result", queued)
        }
      }, RESULT_THROTTLE_MS - elapsed)
    }
  }

  // Registered exactly once. ui.handle throws SYNCHRONOUSLY on a second
  // registration for the same channel (one handler per channel), so this must
  // not be moved anywhere that can run twice. Returns the deregister fn.
  unsubscribers.push(ui.handle("getSnapshot", async () => snapshot))

  function setState(next: AppState): void {
    if (appState === next) return
    console.log(`[State] ${appState} -> ${next}`)
    appState = next
    publishStreamState()
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
        publishAiState("error", "런타임에 WebSocket 또는 fetch 가 없다")
        return
      }
      setState("connecting_ai")
      publishAiState("connecting")
      // onReady also fires after a reconnect, so this returns us to ai_ready
      // without the user having to do anything.
      ai = new AiClient(
        session.userId,
        () => {
          if (appState === "connecting_ai" || appState === "error") setState("ai_ready")
          else console.log("[Gate3] ready 수신했지만 state 유지:", appState)
          // Published unconditionally: the AI phase is its own axis, and a
          // reconnect that lands while we're already streaming still needs to
          // reach the UI even though appState doesn't move.
          publishAiState("ready")
        },
        // Results are forwarded as plain data. AiClient never sees `ui` —
        // the bridge lives entirely on this side.
        publishResult,
      )
      ai.connect()
    }),
  )

  unsubscribers.push(
    session.on("error", (e) => {
      console.error("[Session] error:", e)
      // `retryable` is null, not false: the session "error" event carries no
      // such field, and claiming false would be inventing an answer.
      const r = asRecord(e)
      patch("error", {
        code: typeof r?.code === "string" ? r.code : "session_error",
        message: e instanceof Error ? e.message : String(r?.message ?? e),
        retryable: null,
      })
    }),
  )

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
      //
      // The one exception: turnOffLed is fired and NOT waited on. Every
      // LedModule method is async, so this may well not complete before the
      // socket goes — that's accepted. Leaving the light on after teardown is
      // worse than a call that sometimes doesn't land, and there is no
      // synchronous way to darken it.
      turnOffLed(`beforeDisconnect: ${r}`)
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
      turnOffLed(`disconnect: ${r}`)
      publishAiState("disconnected", `disconnect: ${r}`)
      // A queued interim must not fire after the subscriptions are drained.
      clearResultTimer()
      pendingResult = undefined

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
      publishStreamState(`disconnect: ${r}`)
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
      // Spread the current slot: this event only knows about Wi-Fi, and
      // overwriting battery/charging with null would erase what onBattery said.
      patch("glasses:state", {
        ...snapshot.glasses,
        wifiConnected: data?.connected ?? null,
        ssid: data?.ssid ?? null,
      })
    }),
  )

  // Battery had no subscriber before — added for the glasses:state channel.
  // BatteryData is {level, charging}; `level` is the percentage.
  unsubscribers.push(
    session.glasses.onBattery((data) => {
      console.log("[Gate4] Battery:", JSON.stringify(data))
      patch("glasses:state", {
        ...snapshot.glasses,
        battery: data?.level ?? null,
        charging: data?.charging ?? null,
      })
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

  /**
   * Wait (don't reject) for the AI socket — same shape as waitForWifi.
   *
   * AiClient reconnects on its own after an abnormal close, but it has no
   * "reconnecting" state: the whole backoff window reads as `error`, so a
   * single isReady() check can't tell "dead" from "back in a second". Polling
   * for 3s answers that question without adding state to AiClient.
   */
  async function waitForAi(): Promise<boolean> {
    const deadline = Date.now() + AI_WAIT_MS
    console.log("[Gate4] AI 연결 대기 시작. aiState=", ai?.getState())
    // One of the two places index.ts reads ai.getState() — publish the
    // projection here rather than teaching AiClient about the UI.
    publishAiState(aiPhase(ai?.getState()))

    while (Date.now() < deadline) {
      if (ai?.isReady() === true) return true
      await new Promise((resolve) => setTimeout(resolve, AI_POLL_MS))
    }

    console.warn("[Gate4] AI 연결 대기 실패. aiState=", ai?.getState())
    publishAiState(aiPhase(ai?.getState()), `${AI_WAIT_MS}ms 안에 AI 세션이 준비되지 않았다`)
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
    // No LED call here on purpose: both callers (runStopSequence, rollback)
    // setState immediately afterwards, and that setState drives patch →
    // applyLed to the correct light. Darkening here would only add a second
    // blackout on the way through.
    publishStreamState(`cleanup: ${reason}`)
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

  // --- start ladder --------------------------------------------------------

  /**
   * Walk STREAM_ATTEMPTS until one succeeds. Returns the winning result plus
   * the fps that rung actually asked for, or undefined when every rung fails.
   *
   * The loop is kept even though STREAM_ATTEMPTS holds a single entry — see the
   * note on that constant for what reviving a rung would take.
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
    startInFlight = true
    try {
      // 0. AI 세션. 한 번 보고 포기하지 않고 잠깐 기다린다 — 소켓이 죽어도
      // AiClient 가 1s→2s→4s… 로 알아서 재연결하고, 그 사이 누른 롱프레스는
      // 원래대로면 조용히 리턴했다. hasDisplay=false 라 착용자에게 아무 피드백이
      // 없어서 버튼이 고장 난 것처럼 보였던 자리다.
      //
      // 실패해도 error 로 떨어뜨리지 않는다: 재연결은 계속 진행 중이므로
      // 상태를 보존한 채 다시 누르면 그때 성공한다. startInFlight 안쪽이라
      // 대기 중 두 번째 롱프레스가 병렬 진입하는 일은 없다.
      if (!(await waitForAi())) {
        console.error("[Gate4] AI 세션이 준비되지 않았다 — 스트림 시작 불가. 다시 눌러라")
        return
      }

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

      // 2. Start the Mentra stream. Measured ~5s.
      setState("starting_stream")
      const attempt = await runStreamLadder()
      if (attempt === undefined) {
        setState("error")
        console.error("[Gate4] 스트림 시작 실패 (rung A) — error 상태")
        return
      }
      const {result, requestedFps} = attempt

      const streamId = result?.streamId
      const webrtcUrl = result?.webrtcUrl
      // Requested vs negotiated, side by side. requestedFps comes from the rung
      // that actually won rather than a module constant, so a future rung that
      // asks for something other than 30 still reports honestly.
      console.log("[Gate4] fps 요청=", requestedFps, "/ resolvedConfig=", result?.resolvedConfig?.video?.fps)

      // Diagnostics before validation: a rung that "succeeded" but produced no
      // webrtcUrl is exactly the case worth showing, and the rollback below
      // would otherwise return before anything reached the UI.
      //
      // `requestedFps` is the winning rung's own request, not a module constant.
      // resolvedConfig is OPTIONAL: every hop is optional-chained.
      // Host via parseUrlParts() — `new URL()` does not exist in this runtime.
      patch("stream:diagnostics", {
        requestedFps,
        resolvedFps: result?.resolvedConfig?.video?.fps ?? null,
        transport: result?.resolvedConfig?.transport ?? null,
        webrtcHost:
          typeof webrtcUrl === "string" && webrtcUrl.length > 0
            ? (parseUrlParts(webrtcUrl)?.hostname ?? null)
            : null,
        mode: result?.mode ?? null,
        status: result?.status ?? null,
      })

      // Track it before validation so a rollback can always find the id.
      activeStreamId = streamId
      publishStreamState()

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
      // `ai?.` rather than `ai.`: the old `ai?.isReady() !== true` guard narrowed
      // `ai` for the rest of the function, but waitForAi returns a plain boolean
      // so TS can't carry that through. Undefined can't actually reach here
      // (waitForAi only returns true via `ai?.isReady() === true`), and if it
      // somehow did, `!undefined` rolls back — which is the right answer anyway.
      if (!ai?.sendStreamStart(streamId, webrtcUrl)) {
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
          case "connecting_ai":
          case "waiting_wifi":
            // connecting_ai is accepted because waitForAi handles it: the AI
            // socket may be mid-(re)connect and land well inside 3s. idle is
            // deliberately NOT here — there `ai` hasn't been constructed yet,
            // so there is nothing to wait for.
            //
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
            // idle: AI client not constructed yet ("ready" hasn't fired).
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
