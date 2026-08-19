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
 * brand-new session. Module state does NOT survive — `streamState` and
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
 * Gate scope: input, capability logging, and stream start/stop probing.
 * Deliberately absent —
 *   - fetch / WebSocket  (Gate 3 — no AI server wiring in this Gate)
 *   - session.ui.*       (Gate 5)
 *   - session.display.*  (Mentra Live has no display; SDK marks
 *     GlassesCapabilities.display "null/absent on displayless devices")
 *
 * connect(): NOT called here. registerMiniapp calls session.connect()
 * itself right after the handler returns (see
 * @mentra/miniapp/dist/background/register.js). The manual
 * `await session.connect()` shown in session.d.ts's lifecycle comment
 * describes direct `new MiniappSession()` use, not the register path.
 */

import {registerMiniapp, type GlassesCapabilities, type StreamModule, type WifiData} from "@mentra/miniapp/background"

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
// Gate 2 — stream probe
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
 * Split a URL without `new URL()`. The URL global's presence in JSC/QuickJS
 * is unconfirmed, so this never touches it. On a parse miss the raw string is
 * logged verbatim rather than dropped.
 */
function logUrlParts(label: string, url: unknown): void {
  if (typeof url !== "string" || url.length === 0) return

  const m = /^([a-z]+):\/\/([^/:?#]+)(?::(\d+))?/i.exec(url)
  if (!m) {
    console.warn(`[Gate2] ${label} URL 파싱 실패 — 원본:`, url)
    return
  }
  console.log(`[Gate2] ${label} protocol=${m[1]}`)
  console.log(`[Gate2] ${label} hostname=${m[2]}`)
  console.log(`[Gate2] ${label} port=${m[3] ?? "(없음)"}`)
  console.log(`[Gate2] ${label} full=`, url)
}

/** Pull every field of interest out of a successful StreamResult. */
function logStreamResult(result: StreamResult): void {
  console.log("[Gate2] 성공 result 전체:", JSON.stringify(result, null, 2))

  console.log("[Gate2] streamId=", result?.streamId)
  console.log("[Gate2] status=", result?.status)
  console.log("[Gate2] mode=", result?.mode)
  console.log("[Gate2] liveInputId=", result?.liveInputId)
  console.log("[Gate2] webrtcUrl=", result?.webrtcUrl)
  console.log("[Gate2] hlsUrl=", result?.hlsUrl)
  console.log("[Gate2] dashUrl=", result?.dashUrl)

  const resolved = result?.resolvedConfig
  if (resolved === undefined) {
    // Loud on purpose: without resolvedConfig we cannot tell which transport
    // was actually negotiated, which is the core question of this Gate.
    console.warn("[Gate2] resolvedConfig 미제공 — Mentra 문의 대상")
  } else {
    console.log("[Gate2] resolvedConfig:", JSON.stringify(resolved, null, 2))
    // Which of rtmp/srt/whip actually won the negotiation.
    console.log("[Gate2] resolvedConfig.transport=", resolved?.transport)
    // Whether the requested fps:30 survived.
    console.log("[Gate2] resolvedConfig.video.fps=", resolved?.video?.fps)
    console.log(
      "[Gate2] resolvedConfig.video WxH=",
      `${String(resolved?.video?.width)}x${String(resolved?.video?.height)}`,
    )
  }

  logUrlParts("webrtcUrl", result?.webrtcUrl)
}

/** Shared encode request across all three attempts, so results are comparable. */
const VIDEO_CONFIG = {width: 1280, height: 720, fps: 30} as const

/**
 * The fallback ladder. Ordered; the first success wins.
 *
 * Rationale: Mentra Live reports
 * `capabilities.camera.video.supportedStreamTypes === ["rtmp"]` — neither
 * "whip" nor "srt" is listed. Attempt A may therefore fail outright. That is
 * a legitimate outcome of this Gate, not a bug to route around: the goal is
 * to record exactly HOW it fails.
 *
 * Note the two are not necessarily the same axis — `supportedStreamTypes`
 * describes what the glasses can publish, while `ingest` selects the managed
 * relay's intake protocol (glasses → relay → playback). They may well be
 * negotiated independently. `resolvedConfig.transport` is what settles it,
 * which is why its absence is escalated above.
 */
const STREAM_ATTEMPTS: ReadonlyArray<{name: string; note: string; options: StartStreamOptions}> = [
  {
    name: "A",
    note: 'ingest:"whip" 명시 — sub-second WebRTC, webrtcUrl(WHEP) 기대',
    options: {ingest: "whip", video: VIDEO_CONFIG, sound: false},
  },
  {
    name: "B",
    note: "ingest 생략 — 폰이 기본값을 고르게 두고 mode 가 뭐로 오는지 관찰",
    options: {video: VIDEO_CONFIG, sound: false},
  },
  {
    name: "C",
    note: 'ingest:"srt" — SDK 기본값. HLS/DASH 경로',
    options: {ingest: "srt", video: VIDEO_CONFIG, sound: false},
  },
]

type StreamState = "idle" | "starting_stream" | "streaming" | "stopping" | "error"

registerMiniapp((session) => {
  /**
   * Every subscription's unsubscribe fn lands here and is drained on
   * "disconnect". `session.on(...)` and `session.input.onButtonPress(...)`
   * both return an unsubscribe function.
   */
  const unsubscribers: Array<() => void> = []

  // --- Gate 2 mutable state ----------------------------------------------
  // Per-session, NOT persisted. See the hot-reload warning at the top of the
  // file: a dev reload resets all three.
  let streamState: StreamState = "idle"
  let activeStreamId: string | undefined
  /** Latest glasses Wi-Fi state. Undefined until the first onWifi delivery. */
  let latestWifi: WifiData | undefined

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
  unsubscribers.push(session.on("beforeDisconnect", (r) => console.log("[Session] beforeDisconnect:", r)))

  unsubscribers.push(
    session.on("disconnect", (r) => {
      console.log("[Session] disconnect:", r)
      if (activeStreamId !== undefined || streamState === "streaming") {
        // Can't stop it from here — the transport is already going away and
        // any request would just fail. Logged so the id survives in the
        // console if the camera turns out to still be held afterward.
        console.warn("[Gate2] disconnect 시점에 스트림이 살아있었다. streamId=", activeStreamId)
      }
      for (const unsubscribe of unsubscribers) {
        unsubscribe()
      }
      unsubscribers.length = 0
    }),
  )

  // --- glasses Wi-Fi ------------------------------------------------------

  // Fires the current state on subscribe and on every change (per SDK), so
  // `latestWifi` is populated without an explicit query. Streaming requires
  // Wi-Fi, so this gates the ladder below.
  unsubscribers.push(
    session.glasses.onWifi((data) => {
      latestWifi = data
      console.log("[Gate2] WiFi:", JSON.stringify(data))
    }),
  )

  // --- Gate 2 preflight ---------------------------------------------------

  /**
   * Checks that must pass before any startStream call. Returns false and logs
   * the reason when the device isn't a real streaming-capable set of glasses.
   */
  function preflight(): boolean {
    const c = asRecord(session.capabilities)

    // Guard against running the ladder on a simulator / displayless stub with
    // no camera at all.
    if (c?.hasCamera !== true) {
      console.error("[Gate2] hasCamera !== true — 스트림 중단. hasCamera=", c?.hasCamera)
      return false
    }

    console.log("[Gate2] modelName=", c?.modelName)

    const supported = asRecord(asRecord(c?.camera)?.video)?.supportedStreamTypes
    console.log("[Gate2] supportedStreamTypes=", JSON.stringify(supported))

    if (latestWifi === undefined) {
      // Not the same as "disconnected" — onWifi simply hasn't delivered yet.
      console.warn("[Gate2] WiFi 상태 미수신 — 스트림 시작하지 않음")
      return false
    }
    if (latestWifi.connected !== true) {
      console.warn("[Gate2] 안경 WiFi 미연결 — 스트림 시작하지 않음. wifi=", JSON.stringify(latestWifi))
      // TODO(Gate 3): session.glasses.requestWifiSetup(reason) 으로 설정 유도.
      return false
    }

    console.log("[Gate2] preflight 통과. wifi=", JSON.stringify(latestWifi))
    return true
  }

  // --- Gate 2 ladder ------------------------------------------------------

  async function runStreamLadder(): Promise<void> {
    streamState = "starting_stream"
    console.log("[Gate2] state -> starting_stream")

    for (const attempt of STREAM_ATTEMPTS) {
      console.log(`[Gate2] 시도 ${attempt.name} 시작`, JSON.stringify(attempt.options), `(${attempt.note})`)

      try {
        const result = await session.stream.startStream(attempt.options)
        console.log(`[Gate2] 시도 ${attempt.name} 성공`)
        logStreamResult(result)

        activeStreamId = result?.streamId
        streamState = "streaming"
        console.log("[Gate2] state -> streaming. activeStreamId=", activeStreamId)
        if (activeStreamId === undefined) {
          // stop(streamId) is then impossible; only the no-arg fallback remains.
          console.warn("[Gate2] 성공했지만 streamId 가 없다 — stop 은 인자 없는 호출에 의존해야 한다")
        }
        return
      } catch (err) {
        console.error(`[Gate2] 시도 ${attempt.name} 실패`)
        logRequestError(`[Gate2] 시도 ${attempt.name}`, err)
        // Fall through to the next rung. Note there is no dedicated
        // camera-busy code in MiniappErrorCode — a held camera surfaces as
        // INTERNAL (or a phone-side code) with the detail in `message`, so
        // read the message line above rather than matching on code.
      }
    }

    streamState = "error"
    console.error("[Gate2] state -> error. A/B/C 전부 실패")
  }

  async function stopStream(): Promise<void> {
    streamState = "stopping"
    console.log("[Gate2] state -> stopping. activeStreamId=", activeStreamId)

    // stop() resolves to void, so there is no result payload to log — only
    // whether it settled.
    if (activeStreamId !== undefined) {
      try {
        await session.stream.stop(activeStreamId)
        console.log("[Gate2] stop(streamId) 성공. streamId=", activeStreamId)
        activeStreamId = undefined
        streamState = "idle"
        console.log("[Gate2] state -> idle")
        return
      } catch (err) {
        console.error("[Gate2] stop(streamId) 실패 — 인자 없는 stop() 으로 폴백")
        logRequestError("[Gate2] stop(streamId)", err)
      }
    } else {
      console.warn("[Gate2] activeStreamId 없음 — 인자 없는 stop() 만 시도한다")
    }

    // streamId is optional in the SDK signature and documented as "stop the
    // active stream". This is the recovery path after a hot reload lost the id.
    try {
      await session.stream.stop()
      console.log("[Gate2] stop() (인자 없음) 성공")
      activeStreamId = undefined
      streamState = "idle"
      console.log("[Gate2] state -> idle")
    } catch (err) {
      console.error("[Gate2] stop() (인자 없음) 실패")
      logRequestError("[Gate2] stop()", err)
      streamState = "error"
      console.error("[Gate2] state -> error. 스트림이 남아있을 수 있다")
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
        console.log("[Input] LONG", JSON.stringify(press))

        // Re-entrancy guard. starting_stream / stopping are in-flight async
        // states; a second long-press during either would race the first.
        if (streamState === "starting_stream" || streamState === "stopping") {
          console.warn("[Gate2] 진행 중이라 무시. state=", streamState)
          return
        }

        if (streamState === "idle") {
          if (!preflight()) return
          // Fire-and-forget: onButtonPress handlers are sync. Errors are
          // handled inside the ladder; this catch is the last-resort net for
          // a throw outside the per-attempt try.
          void runStreamLadder().catch((err) => {
            streamState = "error"
            logRequestError("[Gate2] ladder 예외", err)
          })
          return
        }

        if (streamState === "streaming") {
          void stopStream().catch((err) => {
            streamState = "error"
            logRequestError("[Gate2] stop 예외", err)
          })
          return
        }

        // state === "error": stuck. One long-press attempts recovery via the
        // no-arg stop(), which is also the hot-reload escape hatch.
        console.warn("[Gate2] error 상태 — 복구용 stop() 시도")
        void stopStream().catch((err) => {
          streamState = "error"
          logRequestError("[Gate2] 복구 stop 예외", err)
        })
      } else {
        // Short press is inert by design. The old build triggered a photo
        // capture here, which is what caused camera_busy — not carried over.
        console.log("[Input] short (무시)")
      }
    }),
  )
})
