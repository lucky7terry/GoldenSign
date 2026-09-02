/**
 * WebView UI. 구독하고 표시하는 일만 한다.
 *
 * 네트워크도 AI 소켓도 스트림 제어도 저장소도 소유하지 않는다. 화면의 모든 값은
 * 타입 지정 채널로 왔거나 마운트 시 한 번의 `getSnapshot` 으로 온 것이다
 * (src/shared/channels.ts 참고). "변환 시작" 버튼을 두지 않은 것도 의도다.
 * 스트리밍은 안경 관자놀이 버튼으로만 시작한다.
 *
 * 스냅샷이 선택이 아니라 필수인 이유: `session.ui.send` 는 바인딩된 WebView 가
 * 없으면 조용히 드롭하고 background 가 밀린 것을 모아 두지 않는다. 세션 도중
 * 마운트된 WebView 는 그때까지의 모든 방송을 놓친 상태라, 물어보는 것 말고는
 * 따라잡을 방법이 없다.
 */

import {useEffect, useState} from "react"
import {MiniappHeader, useRpc, useSafeArea} from "@mentra/miniapp/ui"
import type {Channels, Snapshot} from "../shared/channels"

/**
 * `mentra.request` 에는 기본 timeout 이 없다(ui/index.d.ts). 이 값을 주지 않으면
 * background 가 붙어 있지 않을 때 Promise 가 영원히 매달린다.
 */
const SNAPSHOT_TIMEOUT_MS = 3000

/** background 쪽 상한과 같은 값. 두 버퍼가 어긋나지 않게 한다. */
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
  error: null,
}

// ---------------------------------------------------------------------------
// 표시 포맷
// ---------------------------------------------------------------------------

/** null 과 undefined 모두 "—" 로 표시한다. 날 `undefined` 가 DOM 에 닿으면 안 된다. */
function fmt(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "boolean") return value ? "예" : "아니오"
  return String(value)
}

/**
 * 채널에서 `windowIndex` 는 필수 `number` 라, 서버가
 * `result.sequence.window_index` 를 주지 않으면 background 가 -1 을 쓴다
 * (60프레임이 차기 전 구간에서 서버가 null 로 보낸다). 인덱스가 아니라
 * 센티널이므로 "w-1" 로 그리면 실제 데이터처럼 읽힌다.
 */
function fmtWindowIndex(value: number | undefined): string {
  if (value === undefined || value === -1) return "—"
  return String(value)
}

/**
 * 스냅샷의 히스토리를 이미 도착한 live 결과 *아래* 로 접어 넣는다.
 *
 * 히스토리가 먼저이고 live 가 뒤인 이유는 히스토리가 구조적으로 더 오래됐기
 * 때문이다. 같은 windowIndex 가 양쪽에 있으면 live 가 이긴다 — 더 나중 방송이라
 * 그 창의 isFinal / confidence 가 갱신돼 있을 수 있다.
 *
 * windowIndex 가 -1 인 항목은 중복 판정에서 제외한다. -1 은 서버가 값을 주지
 * 않았을 때 background 가 넣는 센티널이지 식별자가 아니다. 서로 무관한 두 결과가
 * 모두 -1 을 달고 있을 수 있으므로 둘 다 살아남아야 하고, 합쳐 버리면 인식
 * 기록이 조용히 지워진다.
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
  // 상한을 걸기 전에 중복을 제거한다. 먼저 자르면 병합에서 지운 중복 수만큼
  // 버퍼가 모자라게 된다.
  return [...kept, ...live].slice(-MAX_RESULTS)
}

// ---------------------------------------------------------------------------
// 상태 → 화면 문구
//
// 원본 상태 문자열(idle / ai_ready / …)은 내부 용어다. 진단 패널에만 노출되고,
// 그 위의 모든 것은 착용자를 향한 문장이다.
// ---------------------------------------------------------------------------

interface StateView {
  label: string
  detail: string
  tone: Tone
}

/**
 * 오류 문구는 원인과 함께 아직 살아 있는 항목도 말한다. 그래야 착용자가 다시
 * 시도하면 되는지, 먼저 고칠 게 있는지 판단할 수 있다.
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
// 조각들
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
 * 시각 피드백. 일부러 이 컴포넌트 하나로 격리해 두었다.
 *
 * 지금은 맥동 인디케이터만 그린다. 라이브 영상은 없다 — WebView 에는 카메라
 * 접근 권한이 없고(하드웨어는 전부 background 소유), 안경 영상을 폰으로 되돌려
 * 보내면 진단상 이득 없이 대역폭만 두 배가 된다.
 *
 * 나중에 서버가 keypoints 를 싣기 시작하면 바뀌는 곳은 여기뿐이다. 그것을 위한
 * 채널도 파서도 미리 만들지 않았다. 서버 쪽 keypoint 스키마가 아직 없기 때문이다.
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
 * 중간 결과와 확정 결과는 테두리 스타일·굵기·투명도, 그리고 뒤에 붙는 말줄임표로
 * 구분한다. 색만으로 구분하지 않는다. 시연 조명과 재인코딩된 영상은 색차를
 * 뭉개지만 형태는 살아남는다.
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
              // 결과에는 고유 id 가 없다. 뒤에만 붙고 상한이 있는 목록이라
              // 인덱스 + windowIndex 조합이면 충분히 안정적이다.
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
              요청 → 협상 → 서버가 실제로 처리 중인 값. 세 번째는 background 가
              연속한 두 result 의 최상위 `sequence_index` 차분으로 계산한다.
              첫 result 한 건만으로는 차분이 안 나오므로 그동안은 "—" 다.
            */}
            {fmt(d.requestedFps)} → {fmt(d.resolvedFps)} → {fmt(d.processedFps)}
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
// 루트
// ---------------------------------------------------------------------------

export function App() {
  const [snap, setSnap] = useState<Snapshot>(EMPTY_SNAPSHOT)
  const [phase, setPhase] = useState<Phase>({kind: "loading"})

  // insets 와 capsuleMenu 는 마운트 시 한 번만 읽힌다(useSafeArea 내부의
  // useState 이니셜라이저). 런타임에 갱신되지 않으며 바뀌려면 호스트가 리로드를
  // 강제해야 한다. 그래서 회전 대응이 없다 — resize 리스너를 달아도 값이 그대로다.
  const {insets, capsuleMenu} = useSafeArea()

  // 제네릭 두 개를 직접 써야 한다. 인자에서 추론되지 않는다.
  const getSnapshot = useRpc<Channels, "getSnapshot">("getSnapshot")

  useEffect(() => {
    // 순서는 구독 → ready() → 요청이다. background 가 ready 시점에 밀린 것을
    // 흘려보내므로, 핸들러가 없는 사이 도착한 방송은 SDK 의 32개 인바운드 버퍼에
    // 기대야 한다. 순서를 맞추는 데 드는 비용이 0 이라 맞춰 둔다.
    //
    // 슬롯별로 추적한다. 불리언 하나로 두면 안 된다 — 스트리밍 중 UI 를 열면
    // recognition:result 가 초당 여러 번 오므로 RPC 왕복 전에 반드시 참이 되고,
    // 결과 히스토리 20개를 포함한 나머지 슬롯이 통째로 버려진다. getSnapshot 이
    // 존재하는 이유 자체가 무효가 된다.
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

    // 부트스트랩에서 반드시 한 번 불러야 한다. SDK 가 멱등이라고 명시한다.
    // main.tsx 가 아니라 여기서 부르는 이유는 위 핸들러가 걸린 뒤에 도달하게
    // 하려는 것이다.
    mentra.ready()

    getSnapshot({}, {timeout: SNAPSHOT_TIMEOUT_MS})
      .then((snapshot) => {
        // 슬롯 단위로 판단한다. 방송은 스냅샷보다 새로우므로 이미 방송을 받은
        // 슬롯은 live 값을 지키고, 받지 않은 슬롯만 스냅샷 값을 취한다. 결과는
        // 예외다 — 한쪽이 다른 쪽을 버리지 않고 히스토리와 live 를 병합한다.
        setSnap((s) => ({
          ai: seen.has("ai") ? s.ai : snapshot.ai,
          stream: seen.has("stream") ? s.stream : snapshot.stream,
          glasses: seen.has("glasses") ? s.glasses : snapshot.glasses,
          diagnostics: seen.has("diagnostics") ? s.diagnostics : snapshot.diagnostics,
          results: seen.has("results") ? mergeResults(snapshot.results, s.results) : snapshot.results,
          // 아직 "error" 채널을 구독하지 않으므로 seen 에 들어오지 않는다.
          // 통로만 열어 둔 상태이고 UI 표시는 붙이지 않았다.
          error: seen.has("error") ? s.error : snapshot.error,
        }))
        setPhase({kind: "ready"})
      })
      .catch((err: unknown) => {
        // 이 에러들은 WebView 의 bare runtime 스코프에서 생성되므로
        // `instanceof` 가 통하지 않는다. SDK 안내대로 err.name 으로 판별한다.
        const name = (err as {name?: string} | null)?.name
        // 언마운트. useRpc 가 진행 중인 호출을 알아서 abort 한다. 정리할 것 없음.
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
      // 상단 inset 은 여기가 아니라 아래 .gs-headerbar 가 처리한다 — 그쪽 주석 참고.
      style={{
        paddingLeft: insets.left,
        paddingRight: insets.right,
        paddingBottom: insets.bottom,
      }}
    >
      {/*
        MiniappHeader 는 폰 상태 표시줄을 확보해 주지 않는다. useCapsuleHeaderStyle
        은 `marginTop = capsuleMenu.top - insets.top` 을 계산한다 — 상단 inset 을
        *빼는* 식이라, 컨테이너가 그만큼 이미 패딩을 준 경우에만 결과가 맞는다.
        우리가 주지 않아서 헤더 전체가 insets.top(아이폰 기준 약 47~59px)만큼
        위로 올라갔고 제목이 시계 위에 겹쳐 찍혔다.

        헤더 내부가 아니라 여기서 패딩을 주면 SDK 의 계산을 건드리지 않는다.
        insets.top 이 0 이면 아무 일도 일어나지 않는다.

        minHeight 는 레이아웃을 주도하는 값이 아니라 바닥값이다. 이 막대 다음에
        오는 것이 캡슐 메뉴 하단 이하에서 시작함을 보장한다(capsuleMenu.top 과
        같은 뷰포트 절대 좌표 기준). 위 패딩이 들어간 상태에서는 자연 높이가 이미
        이 값을 넘으므로 평소에는 아무 역할도 하지 않는다 — SDK 의 정렬 계산이
        바뀔 때만 걸린다. capsuleMenu 가 null 이면(구버전 호스트) 제약 없음.

        flex-column 은 필요해서 넣었다. insets.top 이 0 일 때 헤더 자신의
        marginTop 이 이 래퍼 밖으로 마진 붕괴하는 것을 막는다.
      */}
      <div
        className="gs-headerbar"
        style={{
          paddingTop: insets.top,
          minHeight: capsuleMenu === null ? undefined : capsuleMenu.top + capsuleMenu.height,
        }}
      >
        {/* onBack 을 주지 않는다. 루트 화면이라 돌아갈 곳이 없다. */}
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

        {/*
          박스 하나에 서로 배타적인 두 문구. 동시에 보이는 일은 없다. 시작 안내는
          대기 상태의 것이고, 자세 안내는 실제로 프레임이 측정되는 동안에만
          의미가 있다.

          실측: 상대의 얼굴만 응시하면 얼굴 70/70, 양손 0/21 이다 — 손이 안경
          시야 밖으로 완전히 벗어나 인식할 대상이 없다. 얼굴과 손이 함께 보이도록
          잡으면 70/70, 21/21 이 된다. 스트리밍 중 이 줄이 화면을 차지할 값어치가
          있는 이유다.
        */}
        <p className="gs-hint">
          {snap.stream.state === "streaming"
            ? "상대방의 얼굴과 손이 함께 보이도록 바라보세요"
            : "안경 관자놀이 버튼을 길게 눌러 시작하세요"}
        </p>

        <ResultPanel results={snap.results} />

        <DiagnosticsPanel snap={snap} />
      </main>
    </div>
  )
}
