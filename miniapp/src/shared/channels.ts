/**
 * Typed channel registry — single source of truth for the names + payload
 * shapes that flow between this miniapp's background JSContext and its UI
 * WebView. Both halves import this file at build time; the bundler inlines
 * the declarations so there's no runtime resolution.
 *
 * Add a key per channel. The TypeScript generic on `mentra.send` /
 * `mentra.on` / `session.ui.send` / `session.ui.on` enforces names + payload
 * shapes at compile time so the two halves can't drift.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * BROADCAST vs RPC
 * ─────────────────────────────────────────────────────────────────────────
 * A channel is RPC iff its value is `Rpc<Req, Res>` — the branded type from
 * @mentra/miniapp (dist/modules/ui.d.ts). `IsRpc<T>` tests for exactly that
 * brand, and `RpcReq` / `RpcRes` destructure it. Everything else is a
 * broadcast channel. Using the wrong API for a channel is a compile error:
 *
 *   broadcast → session.ui.send / session.ui.on   +  mentra.send / mentra.on
 *   RPC       → session.ui.handle                 +  mentra.request
 *
 * ─────────────────────────────────────────────────────────────────────────
 * `?: T | null` — why both
 * ─────────────────────────────────────────────────────────────────────────
 * Payloads cross the bridge through JSON.stringify (envelope.js:10), which
 * DELETES undefined-valued keys. So "we don't know yet" must travel as an
 * explicit `null`, otherwise the UI can't tell it apart from "this field
 * doesn't exist in this build". The `?` stays so producers may omit a key
 * they aren't responsible for; `| null` is what they write when the answer
 * is genuinely "unknown".
 *
 * ─────────────────────────────────────────────────────────────────────────
 * Ownership
 * ─────────────────────────────────────────────────────────────────────────
 * Network requests, the AI WebSocket and stream control are background's
 * alone. The UI subscribes and renders — it never initiates hardware or
 * network work. `getSnapshot` exists because `session.ui.send` silently
 * DROPS when no WebView is bound, so a late-mounting UI has no backlog to
 * replay and must ask for current state once on mount.
 */

import type {Rpc} from "@mentra/miniapp/ui"

/**
 * Everything a freshly-mounted UI needs to render without waiting for the
 * next broadcast. Kept in lockstep with the broadcast channels by a single
 * `patch()` in background — every field here is the last value sent on the
 * matching channel.
 */
export interface Snapshot {
  ai: Channels["ai:state"]
  stream: Channels["stream:state"]
  glasses: Channels["glasses:state"]
  diagnostics: Channels["stream:diagnostics"]
  /** Most recent results, oldest dropped first. Capped in background. */
  results: Channels["recognition:result"][]
}

export interface Channels {
  // ---------------------------------------------------------------------
  // background → WebView (broadcast)
  // ---------------------------------------------------------------------

  /**
   * AI server connection phase. Collapsed from AiClient's 7-value
   * `AiClientState` — the UI doesn't need to distinguish creating_session
   * from connecting_ws, and AiClientState is deliberately left untouched.
   */
  "ai:state": {
    state: "disconnected" | "connecting" | "ready" | "error"
    sessionId?: string | null
    /** `model` block from the server's `ready`. */
    model?: {loaded: boolean; mode: string; version: string} | null
    message?: string | null
  }

  /** Mirrors background's `AppState` verbatim — same eight names. */
  "stream:state": {
    state:
      | "idle"
      | "connecting_ai"
      | "ai_ready"
      | "waiting_wifi"
      | "starting_stream"
      | "streaming"
      | "stopping"
      | "error"
    streamId?: string | null
    message?: string | null
  }

  /**
   * One recognition result.
   *
   * `windowIndex` comes from the server's `result.sequence.window_index`
   * (sequence_service.metadata()). There is NO `result.sequence_index` on
   * the wire — don't reintroduce it.
   */
  "recognition:result": {
    text: string
    confidence: number
    isFinal: boolean
    windowIndex: number
  }

  "glasses:state": {
    wifiConnected?: boolean | null
    ssid?: string | null
    /** Battery percentage (BatteryData.level). */
    battery?: number | null
    charging?: boolean | null
  }

  "error": {code: string; message: string; retryable?: boolean | null}

  /** Requested-vs-negotiated stream config. Populated after startStream wins. */
  "stream:diagnostics": {
    requestedFps?: number | null
    resolvedFps?: number | null
    transport?: string | null
    /** Host only — parsed by parseUrlParts(), never `new URL()`. */
    webrtcHost?: string | null
    mode?: string | null
    status?: string | null
  }

  // ---------------------------------------------------------------------
  // WebView → background (RPC)
  // ---------------------------------------------------------------------

  /**
   * Pull current state once on UI mount. Takes no arguments; `{}` is the
   * request payload type, so callers write `mentra.request("getSnapshot", {})`.
   */
  "getSnapshot": Rpc<{}, Snapshot>
}

declare global {
  // Augment the `mentra` global from @mentra/miniapp/ui with this miniapp's
  // typed channel registry so authors get compile-time enforcement on every
  // mentra.send / mentra.on / mentra.request call without re-declaring it.
  // eslint-disable-next-line no-var
  var mentra: import("@mentra/miniapp/ui").MentraTyped<Channels>
}
