/**
 * WebView UI — subscription and display only.
 *
 * Owns no network, no AI socket, no stream control, no storage. Every value on
 * screen arrived over a typed channel or from the single `getSnapshot` call on
 * mount (see src/shared/channels.ts). There is deliberately no "start" button:
 * streaming is triggered by the glasses temple button and nothing else.
 *
 * Why the snapshot is mandatory rather than nice-to-have: `session.ui.send`
 * silently DROPS when no WebView is bound, so background keeps no backlog for
 * us. A WebView that mounts mid-session has missed every broadcast so far and
 * can only catch up by asking.
 */

import {useEffect, useState} from "react"
import {MiniappHeader, useRpc, useSafeArea} from "@mentra/miniapp/ui"
import type {Channels, Snapshot} from "../shared/channels"

/**
 * `mentra.request` has NO default timeout (ui/index.d.ts). Without this the
 * promise hangs forever whenever background isn't attached.
 */
const SNAPSHOT_TIMEOUT_MS = 3000

/** Mirrors background's own cap so the two buffers can't disagree. */
const MAX_RESULTS = 20

type Tone = "idle" | "connecting" | "waiting" | "streaming" | "error"

type Phase = {kind: "loading"} | {kind: "error"; message: string} | {kind: "ready"}

const EMPTY_SNAPSHOT: Snapshot = {
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

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/** null / undefined both render as "—". A raw `undefined` must never reach the DOM. */
function fmt(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "boolean") return value ? "예" : "아니오"
  return String(value)
}

/**
 * `windowIndex` is a required `number` on the channel, so background writes -1
 * when the server omitted `result.sequence.window_index`. That's a sentinel,
 * not an index — rendering "w-1" would read as real data.
 */
function fmtWindowIndex(value: number | undefined): string {
  if (value === undefined || value === -1) return "—"
  return String(value)
}

/**
 * Fold the snapshot's history in UNDER the results that already arrived live.
 *
 * history-then-live because history is older by construction. Where the same
 * windowIndex appears in both, the live copy wins: it came from a later
 * broadcast and may carry an upgraded isFinal / confidence for that window.
 *
 * windowIndex -1 is EXEMPT from the dedupe. Background writes -1 when the
 * server omitted `result.sequence.window_index`, so it is a sentinel, not an
 * identity — two unrelated results both carrying -1 must both survive, and
 * collapsing them would silently delete recognition history.
 */
function mergeResults(
  history: Channels["recognition:result"][],
  live: Channels["recognition:result"][],
): Channels["recognition:result"][] {
  const liveIndexes = new Set<number>()
  for (const r of live) {
    if (r.windowIndex !== -1) liveIndexes.add(r.windowIndex)
  }

  const kept = history.filter((r) => r.windowIndex === -1 || !liveIndexes.has(r.windowIndex))
  // Dedupe BEFORE the cap: slicing first would leave the buffer short by
  // however many duplicates the merge dropped.
  return [...kept, ...live].slice(-MAX_RESULTS)
}

// ---------------------------------------------------------------------------
// State → Korean copy
//
// The raw state strings (idle / ai_ready / …) are internal vocabulary. They
// appear ONLY in the diagnostics panel; everything above it is wearer-facing.
// ---------------------------------------------------------------------------

interface StateView {
  label: string
  detail: string
  tone: Tone
}

/**
 * Error copy names the cause AND what is still working, so the wearer knows
 * whether to retry or to fix something first.
 */
function describeError(snap: Snapshot): string {
  const alive: string[] = []
  if (snap.ai.state === "ready") alive.push("인식 서버")
  if (snap.glasses.wifiConnected === true) alive.push("안경 WiFi")

  let cause: string
  if (snap.ai.state === "error") {
    cause = snap.ai.message ?? "인식 서버에 연결하지 못했습니다"
  } else if (snap.glasses.wifiConnected === false) {
    cause = "안경이 WiFi에 연결되어 있지 않습니다"
  } else {
    cause = snap.stream.message ?? "카메라 스트림을 시작하지 못했습니다"
  }

  const alivePart = alive.length > 0 ? `${alive.join(" · ")}는 정상입니다.` : "정상 동작 중인 항목이 없습니다."
  return `${cause}. ${alivePart} 버튼을 길게 눌러 다시 시도할 수 있습니다.`
}

function describeStream(snap: Snapshot): StateView {
  switch (snap.stream.state) {
    case "idle":
    case "ai_ready":
      return {label: "대기 중", detail: "언제든 시작할 수 있습니다", tone: "idle"}
    case "connecting_ai":
      return {label: "연결 중", detail: "인식 서버에 연결하고 있습니다", tone: "connecting"}
    case "waiting_wifi":
      return {label: "안경 WiFi 확인 중", detail: "영상을 보내려면 안경이 WiFi에 연결되어야 합니다", tone: "waiting"}
    case "starting_stream":
      return {label: "카메라 준비 중", detail: "안경 카메라를 여는 중입니다", tone: "connecting"}
    case "streaming":
      return {label: "인식 중", detail: "수어를 읽고 있습니다", tone: "streaming"}
    case "stopping":
      return {label: "정리 중", detail: "스트림을 정리하고 있습니다", tone: "connecting"}
    case "error":
      return {label: "오류", detail: describeError(snap), tone: "error"}
  }
}

function describeAi(snap: Snapshot): StateView {
  switch (snap.ai.state) {
    case "ready":
      return {label: "서버 연결됨", detail: "", tone: "streaming"}
    case "connecting":
      return {label: "서버 연결 중", detail: "", tone: "connecting"}
    case "error":
      return {label: "서버 오류", detail: "", tone: "error"}
    case "disconnected":
      return {label: "서버 끊김", detail: "", tone: "idle"}
  }
}

function describeGlasses(snap: Snapshot): StateView {
  const battery = snap.glasses.battery
  const suffix = battery === null || battery === undefined ? "" : ` · ${battery}%`

  if (snap.glasses.wifiConnected === true) {
    return {label: `WiFi 연결됨${suffix}`, detail: "", tone: "streaming"}
  }
  if (snap.glasses.wifiConnected === false) {
    return {label: `WiFi 없음${suffix}`, detail: "", tone: "waiting"}
  }
  return {label: `WiFi 확인 중${suffix}`, detail: "", tone: "idle"}
}

function describeRpcFailure(name: string | undefined, err: unknown): string {
  if (name === "MentraRpcTimeoutError") {
    return `백그라운드가 ${SNAPSHOT_TIMEOUT_MS / 1000}초 안에 응답하지 않았습니다.`
  }
  if (name === "MentraRpcError") {
    const message = (err as {message?: string} | null)?.message
    return message !== undefined && message.length > 0
      ? `백그라운드 오류: ${message}`
      : "백그라운드에서 오류가 발생했습니다."
  }
  return "상태를 불러오지 못했습니다."
}

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

function StatusBadge({name, view}: {name: string; view: StateView}) {
  return (
    <div className="gs-badge">
      <span className={`gs-dot gs-tone-${view.tone}`} aria-hidden="true" />
      <span className="gs-badge-name">{name}</span>
      <span className="gs-badge-value">{view.label}</span>
    </div>
  )
}

/**
 * The visual channel, isolated on purpose.
 *
 * Today it renders a pulsing indicator and nothing else. There is NO live video
 * here: the WebView has no camera access (background owns all hardware) and
 * echoing the glasses feed back to the phone would double the bandwidth for no
 * diagnostic gain.
 *
 * When the server eventually ships keypoints, this component is the only place
 * that changes. Nothing has been pre-built for that — no channel, no parser —
 * because the server-side keypoint schema does not exist yet.
 */
function VisualFeedback({tone, animated}: {tone: Tone; animated: boolean}) {
  return (
    <div className={`gs-visual gs-tone-${tone}`} aria-hidden="true">
      <span className={`gs-pulse-ring${animated ? " gs-pulse-on" : ""}`} />
      <span className="gs-pulse-core" />
    </div>
  )
}

/**
 * Interim vs final is signalled by border style, weight, opacity AND a trailing
 * ellipsis — never by color alone. Stage lighting and re-encoded demo video
 * flatten hue differences; shape survives both.
 */
function ResultPanel({results}: {results: Channels["recognition:result"][]}) {
  const finals = results.filter((r) => r.isFinal)
  const sentence = finals.length > 0 ? finals[finals.length - 1].text : null

  return (
    <section className="gs-card">
      <h2 className="gs-card-title">인식 결과</h2>

      {results.length === 0 ? (
        <p className="gs-empty">아직 인식된 내용이 없습니다.</p>
      ) : (
        <div className="gs-chips">
          {results.map((r, i) => (
            <span
              // Results carry no unique id; index + windowIndex is stable enough
              // for an append-only, capped list.
              key={`${i}-${r.windowIndex}`}
              className={`gs-chip ${r.isFinal ? "gs-chip-final" : "gs-chip-interim"}`}
            >
              {r.text}
              {r.isFinal ? "" : "…"}
            </span>
          ))}
        </div>
      )}

      <p className="gs-sentence" aria-live="polite">
        {sentence ?? "—"}
      </p>
    </section>
  )
}

function DiagnosticsPanel({snap}: {snap: Snapshot}) {
  const latest = snap.results[snap.results.length - 1]
  const d = snap.diagnostics

  return (
    <details className="gs-diag">
      <summary className="gs-diag-summary">진단</summary>

      <dl className="gs-diag-list">
        <div className="gs-diag-row">
          <dt>fps</dt>
          <dd className="gs-mono">
            {/*
              Three stages, but the app only knows two. The third — how many
              frames the AI server actually processed — never reaches the
              miniapp: no channel carries it and the server's `result` payload
              has no such field. Left as "—" rather than guessed.
            */}
            {fmt(d.requestedFps)} → {fmt(d.resolvedFps)} → —
          </dd>
        </div>
        <div className="gs-diag-row">
          <dt>transport</dt>
          <dd className="gs-mono">{fmt(d.transport)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>webrtc host</dt>
          <dd className="gs-mono">{fmt(d.webrtcHost)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>mode</dt>
          <dd className="gs-mono">{fmt(d.mode)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>status</dt>
          <dd className="gs-mono">{fmt(d.status)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>window index</dt>
          <dd className="gs-mono">{fmtWindowIndex(latest?.windowIndex)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>confidence</dt>
          <dd className="gs-mono">{fmt(latest?.confidence)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>stream.state</dt>
          <dd className="gs-mono">{snap.stream.state}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>stream id</dt>
          <dd className="gs-mono">{fmt(snap.stream.streamId)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>ai.state</dt>
          <dd className="gs-mono">{snap.ai.state}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>ai session</dt>
          <dd className="gs-mono">{fmt(snap.ai.sessionId)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>model</dt>
          <dd className="gs-mono">
            {snap.ai.model === null || snap.ai.model === undefined
              ? "—"
              : `${snap.ai.model.mode} ${snap.ai.model.version}${snap.ai.model.loaded ? "" : " (미로드)"}`}
          </dd>
        </div>
        <div className="gs-diag-row">
          <dt>ssid</dt>
          <dd className="gs-mono">{fmt(snap.glasses.ssid)}</dd>
        </div>
        <div className="gs-diag-row">
          <dt>충전</dt>
          <dd className="gs-mono">{fmt(snap.glasses.charging)}</dd>
        </div>
      </dl>
    </details>
  )
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

export function App() {
  const [snap, setSnap] = useState<Snapshot>(EMPTY_SNAPSHOT)
  const [phase, setPhase] = useState<Phase>({kind: "loading"})

  // insets and capsuleMenu are read ONCE at mount (useState initialiser inside
  // useSafeArea) and never change at runtime — the host would have to force a
  // reload. So there is no rotation handling here: a resize listener would fire
  // but the values behind it would be identical.
  const {insets, capsuleMenu} = useSafeArea()

  // Both generics must be written out; neither is inferred from the argument.
  const getSnapshot = useRpc<Channels, "getSnapshot">("getSnapshot")

  useEffect(() => {
    // Subscribe FIRST, then ready(), then ask. Background flushes on ready, so
    // a broadcast arriving before the handlers exist would need the SDK's
    // 32-deep inbound buffer to save us — ordering it correctly costs nothing.
    // Tracked PER SLOT, not as one boolean. Opening the UI mid-stream means
    // recognition:result fires several times per second, so a single flag would
    // always be set before the RPC returned — and every other slot, including
    // the 20-deep result history, would be thrown away with it. That would
    // defeat the entire reason getSnapshot exists.
    const seen = new Set<string>()

    const offs = [
      mentra.on("ai:state", (ai) => {
        seen.add("ai")
        setSnap((s) => ({...s, ai}))
      }),
      mentra.on("stream:state", (stream) => {
        seen.add("stream")
        setSnap((s) => ({...s, stream}))
      }),
      mentra.on("glasses:state", (glasses) => {
        seen.add("glasses")
        setSnap((s) => ({...s, glasses}))
      }),
      mentra.on("stream:diagnostics", (diagnostics) => {
        seen.add("diagnostics")
        setSnap((s) => ({...s, diagnostics}))
      }),
      mentra.on("recognition:result", (r) => {
        seen.add("results")
        setSnap((s) => ({...s, results: [...s.results, r].slice(-MAX_RESULTS)}))
      }),
      mentra.on("error", (e) => {
        console.error("[UI] error", e.code, e.message)
      }),
    ]

    // MUST run once on bootstrap. Idempotent per the SDK. Called here rather
    // than in main.tsx so it lands after the handlers above are armed.
    mentra.ready()

    getSnapshot({}, {timeout: SNAPSHOT_TIMEOUT_MS})
      .then((snapshot) => {
        // Per slot: a broadcast is newer than the snapshot, so a slot that
        // already got one keeps its live value. Untouched slots take the
        // snapshot's. Results are the exception — history and live are merged
        // rather than one discarding the other.
        setSnap((s) => ({
          ai: seen.has("ai") ? s.ai : snapshot.ai,
          stream: seen.has("stream") ? s.stream : snapshot.stream,
          glasses: seen.has("glasses") ? s.glasses : snapshot.glasses,
          diagnostics: seen.has("diagnostics") ? s.diagnostics : snapshot.diagnostics,
          results: seen.has("results") ? mergeResults(snapshot.results, s.results) : snapshot.results,
        }))
        setPhase({kind: "ready"})
      })
      .catch((err: unknown) => {
        // These errors are constructed in the WebView's bare runtime scope, so
        // `instanceof` is unreliable. Match on err.name — the SDK says so.
        const name = (err as {name?: string} | null)?.name
        // Unmount: useRpc aborts every in-flight call for us. Nothing to clean.
        if (name === "AbortError") return
        setPhase({kind: "error", message: describeRpcFailure(name, err)})
      })

    return () => {
      for (const off of offs) off()
    }
  }, [getSnapshot])

  const stream = describeStream(snap)
  const animated = snap.stream.state !== "idle" && snap.stream.state !== "error"

  return (
    <div
      className="gs-root"
      // The top inset is handled by .gs-headerbar below, not here — see there.
      style={{
        paddingLeft: insets.left,
        paddingRight: insets.right,
        paddingBottom: insets.bottom,
      }}
    >
      {/*
        MiniappHeader does NOT reserve the status bar. useCapsuleHeaderStyle
        computes `marginTop = capsuleMenu.top - insets.top` — it SUBTRACTS the
        top inset, which only lands correctly if the container has already
        padded by that much. We never did, so the whole header sat insets.top
        (~47-59px on iPhone) too high and the title printed over the clock.

        Padding here rather than inside the header keeps the SDK's own maths
        untouched. With insets.top === 0 this is a no-op.

        minHeight is a floor, not a layout driver: it guarantees that whatever
        follows this bar starts at or below the capsule menu's bottom edge, in
        the same viewport-absolute coordinates capsuleMenu.top uses. With the
        padding above in place the natural height already exceeds it, so it
        normally does nothing — it only bites if the SDK's alignment maths
        changes under us. Null capsuleMenu (older hosts) → no constraint.

        flex-column matters: it stops the header's own marginTop from collapsing
        out through this wrapper when insets.top happens to be 0.
      */}
      <div
        className="gs-headerbar"
        style={{
          paddingTop: insets.top,
          minHeight: capsuleMenu === null ? undefined : capsuleMenu.top + capsuleMenu.height,
        }}
      >
        {/* No onBack: this is the root screen, there is nowhere to return to. */}
        <MiniappHeader title="Golden Sign" className="gs-header" />
      </div>

      <main className="gs-main">
        {phase.kind === "loading" ? (
          <p className="gs-notice">상태를 불러오는 중입니다…</p>
        ) : null}

        {phase.kind === "error" ? (
          <p className="gs-notice gs-notice-error">
            {phase.message} 안경 쪽 동작은 계속 진행됩니다.
          </p>
        ) : null}

        <div className="gs-badges">
          <StatusBadge name="AI" view={describeAi(snap)} />
          <StatusBadge name="안경" view={describeGlasses(snap)} />
          <StatusBadge name="스트림" view={stream} />
        </div>

        <section className={`gs-card gs-status gs-tone-${stream.tone}`}>
          <VisualFeedback tone={stream.tone} animated={animated} />
          <p className="gs-status-label">{stream.label}</p>
          <p className="gs-status-detail">{stream.detail}</p>
        </section>

        <p className="gs-hint">안경 관자놀이 버튼을 길게 눌러 시작하세요</p>

        <ResultPanel results={snap.results} />

        <DiagnosticsPanel snap={snap} />
      </main>
    </div>
  )
}
