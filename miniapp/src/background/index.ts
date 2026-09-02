/**
 * background JSContext 진입점. MentraOS 호스트가 미니앱마다 하나씩 띄우는
 * JSContext(iOS-JSC / Android-QuickJS) 안에서 이 파일을 로드하고, init 봉투가
 * 도착하면 `registerMiniapp` 핸들러를 호출한다.
 *
 * 런타임은 bare JS 엔진이다. console / timers / fetch / WebSocket /
 * localStorage 만 있다. window 도 DOM 도 Node API 도 동적 import 도 없다.
 * `typeof URL === "undefined"` 로 실측됐으므로 `new URL()` 은 쓸 수 없고,
 * URL 파싱은 정규식(parseUrlParts)뿐이다.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ⚠️  핫 리로드 경고 — 스트림이 살아 있는 상태로 이 파일을 저장하기 전에 읽을 것
 * ─────────────────────────────────────────────────────────────────────────
 * background/register.js 기준으로, dev 리로드는 JSContext 를 죽이고 다시
 * 띄운다. 폴리필과 번들이 재평가되고 핸들러가 완전히 새 세션으로 다시 돈다.
 * 모듈 상태는 살아남지 못한다 — 아래 `appState` 와 `activeStream` 은 초기값
 * 으로 돌아간다.
 *
 * 그래서 스트림이 도는 중에 저장하면 새 컨텍스트에는 streamId 가 없어 stop() 을
 * 걸 수 없다. 카메라는 옛 스트림이 쥔 채로 남고, 폰이 회수할 때까지 이후의 모든
 * start 가 busy 로 실패한다. 복구: 롱프레스 한 번. 인자 없는
 * `session.stream.stop()` 폴백이 "활성 스트림" 을 대상으로 삼으므로 잃어버린
 * id 가 필요 없다. 그래도 안 되면 폰에서 미니앱을 재시작한다.
 *
 * 저장 전에는 항상 스트림을 먼저 정지할 것.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * 흐름: "ready" 가 AI 세션을 자동 연결 → 롱프레스가 WHEP 스트림을 토글 →
 * "disconnect" 가 AI 세션을 자동 정리. 버튼의 의미는 하나뿐이며, 롱프레스
 * 핸들러의 switch 가 그 전부다.
 *
 * 일부러 쓰지 않는 것 —
 *   - session.display.*  (Mentra Live 는 hasDisplay=false 로 실측됐다. 모드
 *     상태를 보여줄 방법이 없고, 그래서 제스처가 단일 목적으로 남아 있다)
 *   - new URL()          (런타임에 URL 전역이 없다)
 *
 * connect(): 여기서 부르지 않는다. 핸들러가 반환된 직후 registerMiniapp 이
 * session.connect() 를 스스로 호출한다(@mentra/miniapp/dist/background/
 * register.js). session.d.ts 의 라이프사이클 주석에 나오는 수동
 * `await session.connect()` 는 `new MiniappSession()` 을 직접 쓰는 경우의
 * 이야기이며 register 경로와 무관하다.
 *
 * 또한 이 핸들러를 async 로 만들면 안 된다. 구독이 첫 await 이전에 전부
 * 등록돼 있어야 한다.
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
 * background 엔트리는 `StreamModule` 은 re-export 하지만 옵션/결과 인터페이스는
 * 내보내지 않는다. `@mentra/miniapp/dist/modules/stream` 은 패키지 `exports` 에
 * 없는 비공개 경로라 deep import 할 수 없으므로 메서드 시그니처에서 파생시킨다.
 */
type StartStreamOptions = NonNullable<Parameters<StreamModule["startStream"]>[0]>
type StreamResult = Awaited<ReturnType<StreamModule["startStream"]>>

type UnknownRecord = Record<string, unknown>

/**
 * `unknown` 인 capability 필드를 인덱싱 가능한 형태로 좁힌다. SDK 타입이 선언한
 * 것은 `display` 뿐이고 나머지는 전부 `GlassesCapabilities` 의
 * `[key: string]: unknown` 인덱스 시그니처로 들어온다. 그래서 타입 있는 모양에
 * 옵셔널 체이닝을 거는 대신 단계마다 가드가 필요하다.
 */
function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null ? (value as UnknownRecord) : undefined
}

/**
 * 서버 `ready` 의 `model` 블록을 타입 지정 채널에 싣기 전에 좁힌다. AiClient 가
 * 이걸 `unknown` 으로 들고 있는 건 의도다(와이어 원본 값). 그래서 필드 검사는
 * 여기서 한다.
 *
 * 모양이 안 맞으면 undefined 가 아니라 `null` 을 돌려준다. JSON.stringify 가
 * undefined 키를 지워버리므로, UI 가 "물어봤는데 없더라" 를 볼 수 있어야 한다.
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
 * capability 한 줄 요약. "ready" 와 "capabilities" 양쪽에서 부르므로 세션 도중
 * 기기가 바뀌어도 초기값과 바로 비교된다.
 *
 * 여기 `hasDisplay` 는 기기가 보고한 불리언이고, 아래의 `display != null` 검사와
 * 출처가 다르다. 둘을 함께 찍는 건 의도다 — 어긋나면 조용히 넘어가지 않고
 * 눈에 띈다.
 *
 * 모든 필드를 방어적으로 읽는다. 부재(`undefined`)와 명시적 false 를 다르게
 * 찍는 것도 의도다.
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
// 로깅 헬퍼
// ---------------------------------------------------------------------------

/**
 * `session.stream.*` 은 `session.sendRequest` 를 타는데, 이쪽은 Error 가 아니라
 * 평범한 객체 `{code, message}` 로 reject 한다. session.js 에서 확인했다
 * (REQUEST_RESULT 의 `ok:false` 분기와 타임아웃 분기 모두 객체 리터럴로
 * `pending.reject(err)` 를 부른다). 그래서 정상적인 실패 경로에서
 * `err instanceof Error` 는 false 이고 `.stack` 도 없다.
 *
 * 예외는 NotConnectedError 하나다. 세션이 dispose 됐거나 ACK 이전일 때 던져지는
 * 진짜 Error 서브클래스다. 두 모양을 모두 처리한다.
 */
function logRequestError(label: string, err: unknown): void {
  // 해석하기 전에 원본 값을 먼저 찍는다.
  console.error(`${label} err:`, err)

  const e = asRecord(err)
  console.error(`${label} err.code:`, String(e?.code))
  console.error(`${label} err.message:`, String(e?.message))

  // 평범한 {code,message} 객체는 일부 엔진 콘솔에서 "[object Object]" 로
  // 찍히므로 직렬화도 함께 남긴다. 반대로 Error 는 JSON.stringify 에서
  // 살아남지 못해 instanceof 분기가 필요하다.
  if (err instanceof Error) {
    console.error(`${label} err.name/stack:`, err.name, err.stack ?? "(no stack)")
  } else {
    console.error(`${label} err JSON:`, JSON.stringify(err))
  }
}

/**
 * URL 을 조각내 찍는다. parseUrlParts 에 위임한다 — 실측상
 * `typeof URL === "undefined"` 라 정규식 외에 방법이 없다. 파싱에 실패하면
 * 버리지 않고 원문을 그대로 남긴다.
 */
function logUrlParts(label: string, url: unknown): void {
  if (typeof url !== "string" || url.length === 0) return

  const parts = parseUrlParts(url)
  if (parts === undefined) {
    console.warn(`[Stream] ${label} URL 파싱 실패 — 원본:`, url)
    return
  }
  console.log(`[Stream] ${label} protocol=${parts.protocol}`)
  console.log(`[Stream] ${label} hostname=${parts.hostname}`)
  console.log(`[Stream] ${label} port=${parts.port}`)
  console.log(`[Stream] ${label} full=`, url)
}

/** 성공한 StreamResult 에서 관심 필드를 전부 꺼낸다. */
function logStreamResult(result: StreamResult): void {
  console.log("[Stream] 성공 result 전체:", JSON.stringify(result, null, 2))

  console.log("[Stream] streamId=", result?.streamId)
  console.log("[Stream] status=", result?.status)
  console.log("[Stream] mode=", result?.mode)
  console.log("[Stream] liveInputId=", result?.liveInputId)
  console.log("[Stream] webrtcUrl=", result?.webrtcUrl)
  console.log("[Stream] hlsUrl=", result?.hlsUrl)
  console.log("[Stream] dashUrl=", result?.dashUrl)

  const resolved = result?.resolvedConfig
  if (resolved === undefined) {
    // 일부러 시끄럽게 남긴다. resolvedConfig 가 없으면 어떤 transport 로
    // 협상됐는지 알 방법이 없다.
    console.warn("[Stream] resolvedConfig 미제공 — Mentra 문의 대상")
  } else {
    console.log("[Stream] resolvedConfig:", JSON.stringify(resolved, null, 2))
    // rtmp/srt/whip 중 실제로 협상에서 이긴 값.
    console.log("[Stream] resolvedConfig.transport=", resolved?.transport)
    // 요청한 fps 가 그대로 살아남았는지.
    console.log("[Stream] resolvedConfig.video.fps=", resolved?.video?.fps)
    console.log(
      "[Stream] resolvedConfig.video WxH=",
      `${String(resolved?.video?.width)}x${String(resolved?.video?.height)}`,
    )
  }

  logUrlParts("webrtcUrl", result?.webrtcUrl)
}

/** 요청 인코딩 설정. 시작 시 resolvedConfig.video.fps 와 대조한다. */
const VIDEO_CONFIG = {width: 1280, height: 720, fps: 30} as const

/** 요청 fps. 요청값과 협상값을 나란히 찍기 위해 따로 둔다. */
const REQUESTED_FPS = VIDEO_CONFIG.fps

/**
 * 스트림 시도 목록. 순서대로 시도하고 첫 성공이 이긴다. 현재는 한 단뿐이다.
 *
 * 진단용이던 B/C/D 단을 제거했다. 겉보기와 달리 제 역할을 할 수 없는 것들이다.
 *
 *   - B/C 는 SRT 로 협상되어 hlsUrl/dashUrl 만 주고 webrtcUrl 을 주지 않는다.
 *     우리 AI 서버는 WHEP 를 pull 하며 HLS/DASH 를 소비하지 못하므로, B/C 의
 *     "성공" 은 한 번도 쓸 수 있는 스트림이 아니었다. runStartSequence 의
 *     webrtcUrl 검증이 매번 걸러 롤백했다.
 *   - 따라서 D 는 도달 불가능했다. runStreamLadder 는 첫 startStream 성공에서
 *     반환하고 롤백이 runStartSequence 를 끝내므로, 순회는 B 에서 멈췄다.
 *   - 그 대가는 시연 중에 치렀다. A 실패(~5초) → 정리용 stop + 1초 →
 *     B "성공"(~5초) → 롤백 → 실패. 15초 넘게 아무 일도 없다가 실패한다.
 *
 * D(저해상도 WHIP, 인코더 부하가 원인인지 판별)를 되살리려면 배열에 항목 하나를
 * 더하는 것으로는 부족하고 두 가지가 함께 필요하다. 첫째, 어떤 단이 성공했지만
 * 이후 검증에서 걸렸을 때 runStreamLadder 가 다음 단으로 계속 진행해야 한다.
 * 둘째, 단 사이의 정리 — best effort 인자 없는 stop() 과 폰이 실제로 카메라를
 * 놓도록 두는 ~1초 대기 — 를 되살려야 한다. 그 대기가 없으면 다음 단이 잡힌
 * 카메라를 물려받아 busy 로 실패하고, 그 실패가 자기 옵션 탓으로 오인된다.
 * B 와 C 는 되살릴 가치가 없다.
 */
const STREAM_ATTEMPTS: ReadonlyArray<{name: string; note: string; options: StartStreamOptions}> = [
  {
    name: "A",
    note: 'ingest:"whip" 1280x720@30 — 정상 경로. webrtcUrl 기대',
    options: {ingest: "whip", video: VIDEO_CONFIG, sound: false},
  },
]

/** 사용자에게 WiFi 설정을 유도하기 전에 기다리는 한도. */
const WIFI_WAIT_MS = 5000
const WIFI_POLL_MS = 250

/**
 * 롱프레스를 포기하기 전에 AI 소켓을 기다리는 한도.
 *
 * AiClient 의 백오프에 맞춰 잡은 값이다. 첫 재연결이 종료 1초 뒤에 뜨므로 3초면
 * 그 시도와 핸드셰이크까지 덮는다. 이후 시도(2s, 4s, 8s, 16s)는 일부러 뺐다 —
 * 착용자에게 아무것도 알릴 수 없는 채로 16초를 기다리는 것보다 다시 누르게
 * 하는 편이 낫다.
 */
const AI_WAIT_MS = 3000
const AI_POLL_MS = 100

/**
 * `new URL()` 없이 WHEP URL 을 조각낸다.
 *
 * 실측 결과 이 런타임에서 `typeof URL === "undefined"` 다. 확인이 안 된 게
 * 아니라 아예 없다. 정규식 외에 방법이 없다.
 *
 * 파싱에 실패하면 undefined 를 돌려주고, 호출부는 이를 치명적 실패로 다룬다.
 * 파싱되지 않는 webrtcUrl 은 서버도 당길 수 없으므로, 스트림을 살려두지 말고
 * 롤백해야 한다.
 */
function parseUrlParts(url: string): {protocol: string; hostname: string; port: string} | undefined {
  const m = /^([a-z]+):\/\/([^/:?#]+)(?::(\d+))?/i.exec(url)
  if (!m) return undefined
  return {protocol: m[1], hostname: m[2], port: m[3] ?? "(없음)"}
}

/**
 * 앱 전체 상태. 특히 `starting_stream` 은 실측 약 6~7초 걸린다. 재진입 가드가
 * 보기보다 중요한 이유다.
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
// UI 브리지
// ---------------------------------------------------------------------------

/**
 * 브로드캐스트 채널만. `Channels` 의 키 중 값이 `Rpc<Req, Res>` 가 아닌 것들이다.
 * `IsRpc<T>` 가 그 브랜드를 검사하고, 레지스트리를 매핑한 뒤 자기 키로 다시
 * 인덱싱하면 RPC 항목이 `never` 로 떨어져 사라진다.
 */
type BroadcastChannel = {
  [C in keyof Channels]: IsRpc<Channels[C]> extends true ? never : C
}[keyof Channels]

/** snapshot.results 링 크기. 오래된 것부터 버린다. */
const MAX_RESULTS = 20

/**
 * final 이 아닌 인식 결과 브로드캐스트 사이의 최소 간격.
 *
 * 이유가 둘이다. 하나는 렌더 비용이고, 다른 하나는 SDK 의 채널당 인바운드 버퍼가
 * 32개라는 점이다(modules/ui.d.ts). 여기서 막지 않으면 스트리밍 도중 붙는 UI 가
 * 그 버퍼를 넘겨 메시지를 조용히 잃는다.
 */
const RESULT_THROTTLE_MS = 200

/**
 * AiClient 의 7개짜리 상태를 UI 가 신경 쓰는 4개로 접는다. `AiClientState` 자체는
 * 일부러 두었다 — 이건 투영일 뿐 그 상태머신에 새 상태를 만드는 게 아니다.
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
      // idle / closing / undefined (아직 클라이언트를 만들지 않은 상태)
      return "disconnected"
  }
}

// ---------------------------------------------------------------------------
// LED 피드백
// ---------------------------------------------------------------------------

/**
 * 착용자 방향 LED 가 이 기기의 유일한 상태 전달 수단이다. Mentra Live 는
 * hasDisplay=false 라 글자를 그릴 곳이 없다.
 *
 * 색은 프리셋 5색뿐이다 — red / green / blue / orange / white. hex 도 rgb 객체도
 * 없고 노랑도 없다.
 *
 * 범위: `LedModule` 에는 light-id 파라미터가 없다. 헤더가 "the phone maps them
 * to per-device LED indices" 라고 밝히듯 색 이름을 폰이 기기별 LED 인덱스로
 * 매핑한다. 즉 이 API 로는 전면 privacy LED 를 지목하는 것이 애초에 불가능하다.
 * 그쪽은 시스템 소유이며 우리가 건드리지 않는다. 별도 방어 코드도 필요 없다.
 *
 * 단위 — 가정이 아니라 SDK 소스에서 확인한 것:
 *   - led.d.ts:16,18 이 `ontime` / `offtime` 을 "LED on/off duration in ms" 라고
 *     주석으로 명시한다.
 *   - `blink(color, ontime, offtime, count)` 는 인자를 그대로
 *     turnOn({ontime, offtime, count}) 로 넘긴다(led.js:31-33) → 같은 ms.
 *   - `solid(color, duration)` 에는 자체 단위 주석이 없지만 led.js:36 이
 *     `ontime: duration` 으로 대입하므로 위의 ms 를 물려받는다.
 *   따라서 셋 다 밀리초다.
 *
 * 실기 확인: ack 는 유실되는 경우가 있으나 명령 자체는 안경에 적용된다(육안).
 */

/**
 * 지속 상태용 LED 명령 하나가 유지되도록 요청하는 시간.
 *
 * blink 의 `count` 도 solid 의 `duration` 도 유한한데 ai_ready 와 streaming 은
 * 몇 분씩 간다. 갱신용 인터벌을 새로 만드는 대신(얹을 기존 타이머가 없다 —
 * ping 을 넣지 않기로 했다) 전이마다 넉넉한 시간으로 걸어 두고, 상태가 실제로
 * 바뀔 때만 다시 건다. 다음 전이 전에 불이 꺼지는 게 관측되면 그때 갱신 방법을
 * 다시 논의한다.
 */
const LED_HOLD_MS = 30_000

/** 깜빡임 주기. 400/400 이면 깜박거림이 아니라 또렷한 맥동으로 읽힌다. */
const LED_BLINK_ON_MS = 400
const LED_BLINK_OFF_MS = 400

/** LED_HOLD_MS 를 덮을 만큼의 주기 수. blink 와 solid 의 유지 시간을 맞춘다. */
const LED_BLINK_COUNT = Math.floor(LED_HOLD_MS / (LED_BLINK_ON_MS + LED_BLINK_OFF_MS))

type LedCommand = {kind: "off"} | {kind: "solid"; color: LedColor} | {kind: "blink"; color: LedColor}

/**
 * appState → LED. AppState 에 대해 exhaustive 하다. 매핑 없이 상태를 추가하면
 * 불이 조용히 꺼지는 대신 컴파일 에러가 난다.
 *
 * 프리셋 5색에 노랑이 없으므로 waiting_wifi 는 orange 를 쓴다.
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
 * `hasLight` 는 GlassesCapabilities 의 선언된 표면에 없다. summarizeCapabilities
 * 가 읽는 다른 필드들과 마찬가지로 `[key: string]: unknown` 인덱스 시그니처를
 * 타고 들어온다. 캐시하지 않고 매번 읽는 이유는 세션 도중 기기가 바뀌는 경우
 * ("capabilities" 이벤트)를 반영하기 위해서다.
 */
function deviceHasLight(caps: GlassesCapabilities | null): boolean {
  return asRecord(caps)?.hasLight === true
}

/**
 * LED 실패 로그 한 줄.
 *
 * logRequestError 를 쓰지 않는 이유: 그쪽은 실패마다 네 줄을 찍는다. 앱 전체를
 * 좌우하는 스트림 호출에는 맞지만 LED 에는 과하다. LED 실패는 앱 동작을 전혀
 * 바꾸지 않으므로 warn 한 줄이면 된다.
 *
 * `.message` 를 `Error` 가 아니라 구조적으로 읽는 이유: session.led.* 는
 * session.sendRequest 를 타고, 이쪽은 평범한 {code, message} 객체로 reject
 * 한다. 그 모양이 아닌 값은 JSON.stringify 가 받아 준다.
 */
function logLedError(label: string, err: unknown): void {
  const message = asRecord(err)?.message
  console.warn(`[LED] ${label} 실패:`, typeof message === "string" ? message : JSON.stringify(err))
}

registerMiniapp((session) => {
  /**
   * 모든 구독의 해제 함수가 여기 모이고 "disconnect" 에서 한 번에 비운다.
   * `session.on(...)` 과 `session.input.onButtonPress(...)` 둘 다 해제 함수를
   * 돌려준다.
   */
  const unsubscribers: Array<() => void> = []

  // --- 가변 상태 ------------------------------------------------------------
  // 세션 단위이며 저장되지 않는다. 파일 상단의 핫 리로드 경고 참고 — dev
  // 리로드가 전부 초기화한다.
  let appState: AppState = "idle"
  /**
   * 활성 스트림. 재연결 뒤 stream_start 를 다시 보내려면 id 만으로는 부족해서
   * webrtcUrl 을 같이 들고 있는다. 둘이 따로 놀지 않도록 한 객체로 묶었다.
   *
   * `webrtcUrl` 이 undefined 인 구간이 있는 것은 의도다 — 롤백이 id 를 찾을 수
   * 있도록 URL 검증 *전에* 먼저 기록하기 때문이다. 재전송은 URL 이 채워진
   * 뒤에만 가능하다.
   */
  let activeStream: {streamId: string; webrtcUrl: string | undefined} | undefined
  /** 안경 WiFi 최신 상태. 첫 onWifi 전달 전까지 undefined 다. */
  let latestWifi: WifiData | undefined
  /**
   * 시작 시퀀스(WiFi 대기 → startStream)가 도는 동안 true.
   * `appState` 와 따로 두는 이유는 시퀀스가 두 상태에 걸쳐 있고, 그 사이의 두
   * 번째 롱프레스가 병렬 실행을 시작해서는 안 되기 때문이다.
   */
  let startInFlight = false

  // "ready" 안에서 만든다. session.userId 는 CONNECT_ACK 이후에야 채워지기
  // 때문이다. 런타임 확인에 실패하면 undefined 로 남는다.
  let ai: AiClient | undefined

  // --- UI 브리지 -------------------------------------------------------------

  // session.d.ts:161 이 `readonly ui: UIModule` 로 제네릭 인자 없이 선언돼 있어
  // 기본값 Record<string, unknown> 이 된다. 그러면 IsRpc<unknown> 이 false 라
  // handle() 의 채널 파라미터가 `never` 로 좁혀지고 send/on 의 페이로드도
  // `unknown` 이 된다. 여기서 한 번만 캐스트하고 이후는 전부 `ui` 를 쓴다.
  //
  // `unknown` 을 거치는 이유: `as UIModule<Channels>` 직접 캐스트는 거부된다.
  // Record<string, unknown> 과 Channels 가 겹치지 않아 1단계 단언이 안 된다.
  // 캐스트 자체는 안전하다 — UIModule 의 제네릭은 채널/페이로드 짝에만 타입을
  // 입힐 뿐 객체의 런타임 모양을 바꾸지 않는다.
  const ui = session.ui as unknown as UIModule<Channels>

  // send() 의 채널 파라미터는 조건부 타입
  // `IsRpc<Channels[C]> extends true ? never : C` 인데, C 가 아직 제네릭인
  // 동안에는 TS 가 평가를 미뤄서 제네릭 래퍼가 직접 호출하지 못한다.
  // BroadcastChannel 이 이미 RPC 채널을 전부 배제하므로, 호출부마다 캐스트하는
  // 대신 여기서 시그니처를 한 번 좁힌다.
  const sendBroadcast = ui.send as <C extends BroadcastChannel>(
    channel: C,
    payload: Channels[C],
  ) => void

  /**
   * 늦게 마운트된 WebView 가 `getSnapshot` 으로 받아 가는 값.
   *
   * 모르는 값은 undefined 가 아니라 `null` 로 시작한다. 페이로드는
   * JSON.stringify 로 브리지를 건너는데(envelope.js) 이때 undefined 키가
   * 삭제되기 때문이다. undefined 로 두면 UI 가 "아직 모름" 과 "이 빌드에 없는
   * 필드" 를 구분하지 못한다.
   */
  const snapshot: Snapshot = {
    ai: {state: "disconnected", sessionId: null, model: null, message: null},
    stream: {state: "idle", streamId: null, message: null},
    glasses: {wifiConnected: null, ssid: null, battery: null, charging: null},
    diagnostics: {
      requestedFps: null,
      resolvedFps: null,
      processedFps: null,
      transport: null,
      webrtcHost: null,
      mode: null,
      status: null,
    },
    results: [],
  }

  // --- LED ------------------------------------------------------------------

  /** 실제로 불에 반영한 마지막 상태. streamId 만 바뀐 publishStreamState()
   *  호출마다 다시 거는 것을 막는다. */
  let lastAppliedLedState: AppState | undefined

  /** 명령에 해당하는 LED Promise 를 만들기만 한다(await 하지 않는다). */
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
   * appState 로 착용자 LED 를 구동한다. fire-and-forget 이다 — await 하지 않으므로
   * ack 가 늦거나 오지 않아도 상태 전이가 지연되지 않고, reject 가 앱 에러로
   * 번지지도 않는다. 불은 피드백이지 의존 대상이 아니다.
   */
  function applyLed(state: AppState): void {
    if (!deviceHasLight(session.capabilities)) {
      // 상태가 바뀔 때만 찍는다. 불 없는 기기에서 patch 마다 도배되지 않도록.
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
      // LedModule 의 메서드는 전부 `async` 라 동기 throw 는 도달할 수 없다.
      // 그래도 잡는다 — LED 문제가 호출부까지 올라가면 안 되기 때문이다.
      logLedError(`${state} 동기 예외`, err)
    }
  }

  /**
   * 상태 맵을 우회하는 무조건 소등. 종료 경로 전용이다.
   *
   * lastAppliedLedState 를 지우므로, 나중에 같은 상태로 applyLed 가 불려도
   * 중복으로 건너뛰지 않고 다시 점등한다.
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
   * 상태가 UI 로 가는 유일한 통로. 스냅샷 슬롯을 갱신한 뒤 방송한다. 두 가지가
   * 여기서 함께 일어나므로 `getSnapshot` 은 구조적으로 항상 최신이고, 기록 없이
   * 보내는 호출부가 생길 수 없다.
   *
   * `ui.send` 는 바인딩된 WebView 가 없으면 조용히 드롭한다. background 가
   * WebView 보다 오래 사는 이상 정상적인 상황이며 에러가 아니다.
   */
  function patch<C extends BroadcastChannel>(channel: C, next: Channels[C]): void {
    switch (channel) {
      case "ai:state":
        snapshot.ai = next as Channels["ai:state"]
        break
      case "stream:state": {
        const stream = next as Channels["stream:state"]
        snapshot.stream = stream
        // LED 호출 지점은 여기 하나뿐이다. 모든 appState 전이가
        // setState → publishStreamState → patch 를 통과하므로, 8개 상태 매핑을
        // 16곳의 setState 호출부에 흩지 않고 여기 모아 둔다. streamId 만 바뀐
        // 호출도 여기 오지만 applyLed 의 lastAppliedLedState 가드가 걸러 낸다.
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
        // "error" 는 상태가 아니라 이벤트다. 스냅샷 슬롯이 없고 방송만 한다.
        break
    }
    sendBroadcast(channel, next)
  }

  /** 현재 시점의 appState 와 activeStream 짝을 그대로 방송한다. */
  function publishStreamState(message: string | null = null): void {
    patch("stream:state", {state: appState, streamId: activeStream?.streamId ?? null, message})
  }

  /** 현재 시점의 AI 단계와 세션 식별 정보를 방송한다. */
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

  // --- 인식 결과 스로틀 -------------------------------------------------------

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
   * final 이 아닌 결과는 RESULT_THROTTLE_MS 당 한 건으로 줄이고, final 은 즉시
   * 통과시킨다. 착용자가 실제로 기다리는 건 final 이다.
   *
   * trailing edge, 최신 우선: 눌린 중간 결과는 들고 있다가 창이 닫힐 때 내보내고,
   * final 이 오면 들고 있던 중간 결과를 버린다(답 뒤에 도착하면 안 되므로).
   * 알아 둘 결과: snapshot.results 에는 서버가 만든 모든 결과가 아니라 실제로
   * 방송된 것만 남는다.
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

  // --- 처리 fps ---------------------------------------------------------------

  /**
   * 서버 최상위 `sequence_index`(= 누적 처리 프레임 수) 의 직전 값과 그 수신 시각.
   * 두 지점의 차분으로 처리 fps 를 낸다. 절대값이 아니라 차분인 이유는
   * sequence_index 가 스트림 시작 이후의 누적치라, 나눌 기준 시각이 따로 없기
   * 때문이다.
   */
  let lastSequenceIndex: number | undefined
  let lastSequenceAt = 0

  /** 새 스트림에서 서버 카운터가 1부터 다시 시작하므로 기준점도 함께 버린다. */
  function resetProcessedFps(): void {
    lastSequenceIndex = undefined
    lastSequenceAt = 0
  }

  /**
   * 스로틀 이전, 서버가 보낸 모든 result 에서 부른다. 방송된 것만 세면 우리가
   * 건 스로틀이 서버의 처리 속도로 둔갑한다.
   */
  function trackProcessedFps(sequenceIndex: number | null): void {
    if (sequenceIndex === null) return

    const now = Date.now()
    const prevIndex = lastSequenceIndex
    const prevAt = lastSequenceAt
    lastSequenceIndex = sequenceIndex
    lastSequenceAt = now

    // 첫 표본은 기준점만 남기고 끝낸다 — 차분을 낼 짝이 아직 없다.
    if (prevIndex === undefined) return

    const frames = sequenceIndex - prevIndex
    const elapsedSec = (now - prevAt) / 1000
    // 인덱스가 되감겼거나(서버 재시작·새 스트림) 같은 밀리초에 두 건이 온 경우.
    // 기준점은 위에서 이미 갱신했으니 다음 result 부터 정상 복귀한다.
    if (frames <= 0 || elapsedSec <= 0) return

    patch("stream:diagnostics", {
      ...snapshot.diagnostics,
      // patch 는 진단 슬롯을 통째로 갈아 끼우므로 나머지 필드를 함께 실어야 한다.
      processedFps: Math.round((frames / elapsedSec) * 10) / 10,
    })
  }

  /** AiClient 가 result 를 받을 때마다 부르는 곳. fps 를 먼저 세고 방송한다. */
  function handleResult(next: Channels["recognition:result"]): void {
    trackProcessedFps(next.sequenceIndex)
    publishResult(next)
  }

  // 정확히 한 번만 등록한다. 같은 채널에 두 번째로 등록하면 ui.handle 이 동기적
  // 으로 throw 한다(채널당 핸들러 하나). 두 번 실행될 수 있는 자리로 옮기면
  // 안 된다. 반환값은 해제 함수다.
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

  // --- 세션 라이프사이클 -------------------------------------------------------

  // "ready" 는 인자 없이 발생한다. CONNECT_ACK 값들은 session 객체에서 직접
  // 읽는다.
  unsubscribers.push(
    session.on("ready", () => {
      console.log("[Session] userId:", session.userId)
      console.log("[Session] packageName:", session.packageName)
      console.log("[Session] visibility:", session.visibility)
      summarizeCapabilities("ready", session.capabilities)
      // 전문을 펼쳐서 찍는 건 의도다. SDK 타입이 문서화하지 않은 필드를 찾는
      // 통로다. 위의 요약은 편의일 뿐 대체물이 아니다.
      console.log("[Session] capabilities:", JSON.stringify(session.capabilities, null, 2))

      // display 판정을 명시적으로 남긴다. Mentra Live 에서 false 가 정상이며
      // 버그가 아니다. 따라서 session.display 는 이 기기에서 쓸 수 없다.
      const hasDisplay = session.capabilities?.display != null
      console.log("[Caps] hasDisplay =", hasDisplay)

      // 선언된 매니페스트 권한 스냅샷. _permissions 는 CONNECT_ACK 에서 채워지므로
      // 여기가 유효한 값이 나오는 첫 지점이다. session.permissions 는
      // PermissionsModule (session.d.ts:165) 이고 getAll() 이 PermissionRecord
      // ({location, microphone, camera, notifications, calendar}: boolean) 를 준다.
      console.log("[PERM] snapshot", JSON.stringify(session.permissions.getAll()))

      // --- AI 연결 -------------------------------------------------------------
      // 런타임 확인이 먼저다. WebSocket 이 없으면 이후 전부가 불가능하다.
      // 네트워크 작업을 registerMiniapp 핸들러 본문이 아니라 여기서 시작하는
      // 이유는 그 본문이 동기로 유지돼야 하기 때문이다.
      if (!probeRuntime()) {
        setState("error")
        publishAiState("error", "런타임에 WebSocket 또는 fetch 가 없다")
        return
      }
      setState("connecting_ai")
      publishAiState("connecting")
      // onReady 는 재연결 이후에도 발생하므로, 사용자가 아무것도 하지 않아도
      // ai_ready 로 복귀한다.
      // 이 경로를 두 번 타면 이전 인스턴스가 참조만 잃은 채 재연결을 계속한다.
      ai?.closeNow("새 AiClient 로 교체")
      ai = new AiClient(
        session.userId,
        () => {
          if (appState === "connecting_ai" || appState === "error") setState("ai_ready")
          else console.log("[AI] ready 수신했지만 state 유지:", appState)
          // 조건 없이 방송한다. AI 단계는 별개의 축이라, 이미 스트리밍 중일 때
          // 재연결이 성공하면 appState 는 그대로여도 UI 에는 알려야 한다.
          publishAiState("ready")

          // 재연결이면 세션이 새로 발급됐고 서버는 이 스트림을 모른다. 살아있는
          // 스트림을 다시 붙여 준다 — 이게 없으면 사용자가 앱을 재시작할 때까지
          // 프레임이 영영 흐르지 않는다. 첫 연결에서는 activeStream 이 undefined
          // 라 아무것도 하지 않고, 시작 시퀀스가 제 자리에서 보낸다.
          const live = activeStream
          if (live?.webrtcUrl === undefined) return
          console.log("[Stream] ready 이후 stream_start 재전송. streamId=", live.streamId)
          if (!ai?.sendStreamStart(live.streamId, live.webrtcUrl)) {
            // 롤백도 error 방송도 하지 않는다. ready 직후라 소켓은 열려 있으므로
            // 여기서 false 가 나오는 현실적인 경우는 같은 세션에서 이미 보냈다는
            // 중복 가드뿐이고, 그건 정상이다. 스트림도 살아 있다.
            console.warn("[Stream] stream_start 재전송 실패 streamId=", live.streamId)
          }
        },
        // 결과는 순수 데이터로 넘어온다. AiClient 는 `ui` 를 보지 못한다 —
        // 브리지는 전적으로 이쪽에 있다.
        handleResult,
      )
      ai.connect()
    }),
  )

  unsubscribers.push(
    session.on("error", (e) => {
      console.error("[Session] error:", e)
      // `retryable` 은 false 가 아니라 null 이다. 세션 "error" 이벤트에는 그런
      // 필드가 없고, false 라고 쓰면 없는 답을 지어내는 셈이다.
      const r = asRecord(e)
      patch("error", {
        code: typeof r?.code === "string" ? r.code : "session_error",
        message: e instanceof Error ? e.message : String(r?.message ?? e),
        retryable: null,
      })
    }),
  )

  // 실측 세션 수명: 미니앱 화면을 벗어나면 세션이 종료된다. 반면 폰에서 다른 앱으로
  // 전환하는 것은 세션을 유지한다. 그래서 "화면을 나갔다" 와 "앱을 바꿨다" 는
  // 서로 다른 결과가 되고, 전자만 disconnect 로 이어진다.
  unsubscribers.push(session.on("visibility", (v) => console.log("[Session] visibility:", v)))

  unsubscribers.push(
    session.on("capabilities", (c) => {
      summarizeCapabilities("changed", c)
      console.log("[Session] capabilities changed:", JSON.stringify(c))
    }),
  )

  // 동기 코드만. SDK 기준으로 여기서 시작한 비동기 작업은 소켓이 닫히기 전에
  // 끝나지 않는다.
  unsubscribers.push(
    session.on("beforeDisconnect", (r) => {
      console.log("[Session] beforeDisconnect:", r)
      // ws.send() 만 쓴다. 여기 있는 것은 전부 동기라 호스트가 소켓을 내리기
      // 전에 와이어에 닿을 가능성이 있다. SDK 기준으로 이 핸들러에서 시작한
      // 비동기 작업은 완료되지 않는다.
      // 일부러 뺀 것: session.stream.stop() 과 POST /stop. 둘 다 비동기다.
      //
      // 예외 하나 — turnOffLed 는 호출만 하고 기다리지 않는다. LedModule 메서드는
      // 전부 async 라 소켓이 사라지기 전에 완료되지 않을 수 있고, 그건 감수한다.
      // 종료 후에도 불이 켜져 있는 쪽이 가끔 닿지 않는 호출보다 나쁘고, 불을
      // 끄는 동기 수단은 존재하지 않는다.
      turnOffLed(`beforeDisconnect: ${r}`)
      if (activeStream !== undefined) {
        // stop 보다 stream_stop 을 먼저 보내 서버가 pull 을 먼저 풀게 한다.
        ai?.sendStreamStop(activeStream.streamId)
      }
      ai?.sendStopMessage(`beforeDisconnect: ${r}`)
    }),
  )

  unsubscribers.push(
    session.on("disconnect", (r) => {
      console.log("[Session] disconnect:", r)

      if (activeStream !== undefined || appState === "streaming") {
        console.warn("[Stream] disconnect 시점에 스트림이 살아있었다. streamId=", activeStream?.streamId)
      }

      ai?.closeNow(`disconnect: ${r}`)
      turnOffLed(`disconnect: ${r}`)
      publishAiState("disconnected", `disconnect: ${r}`)
      // 대기 중인 중간 결과가 구독 해제 이후에 발사되면 안 된다.
      clearResultTimer()
      pendingResult = undefined

      // 여기부터는 best effort 다. 전송 계층이 이미 사라지는 중이라 둘 다 완료를
      // 보장할 수 없다.
      const idAtDisconnect = activeStream?.streamId
      void session.stream
        .stop(idAtDisconnect)
        .then(() => console.log("[Stream] disconnect stream.stop 성공"))
        .catch((err) => {
          console.warn("[Stream] disconnect stream.stop 실패 (정상 취급)")
          logRequestError("[Stream] disconnect stream.stop", err)
        })

      // 시도는 하지만 실패가 정상이다. 요청이 닿기 전에 JSContext 가 사라질 수
      // 있다. 서버 세션은 expires_at(생성 시각 + SESSION_TTL_SECONDS, 기본
      // 1시간)에 스스로 만료되므로 이게 끝내 완료되지 않아도 영구히 새지 않는다.
      ai?.postStopBestEffort()

      activeStream = undefined
      publishStreamState(`disconnect: ${r}`)
      setState("idle")

      for (const unsubscribe of unsubscribers) {
        unsubscribe()
      }
      unsubscribers.length = 0
    }),
  )

  // --- 안경 WiFi ------------------------------------------------------------

  // 구독 시점의 현재 상태와 이후 모든 변경에서 발생한다. 실측상 정확한 값이
  // 오기까지 약 2초 걸린다. 시작 시퀀스가 초기값으로 거절하지 않고 기다리는
  // 이유가 이것이다.
  unsubscribers.push(
    session.glasses.onWifi((data) => {
      latestWifi = data
      console.log("[Glasses] WiFi:", JSON.stringify(data))
      // 현재 슬롯을 펼쳐서 넘긴다. 이 이벤트는 WiFi 만 알고 있어서,
      // battery/charging 을 null 로 덮으면 onBattery 가 채운 값이 지워진다.
      patch("glasses:state", {
        ...snapshot.glasses,
        wifiConnected: data?.connected ?? null,
        ssid: data?.ssid ?? null,
      })
    }),
  )

  // 배터리는 구독자가 없었고 glasses:state 채널을 위해 추가했다.
  // BatteryData 는 {level, charging} 이고 `level` 이 퍼센트다.
  unsubscribers.push(
    session.glasses.onBattery((data) => {
      console.log("[Glasses] Battery:", JSON.stringify(data))
      patch("glasses:state", {
        ...snapshot.glasses,
        battery: data?.level ?? null,
        charging: data?.charging ?? null,
      })
    }),
  )

  // --- 사전 점검 ---------------------------------------------------------------

  /**
   * 하드 게이트. 카메라 없는 기기는 영영 스트리밍할 수 없으므로 WiFi 와 달리
   * 기다리지 않고 막는다.
   */
  function hasCameraOrLog(): boolean {
    const c = asRecord(session.capabilities)
    if (c?.hasCamera !== true) {
      console.error("[Stream] hasCamera !== true — 스트림 불가. hasCamera=", c?.hasCamera)
      return false
    }
    console.log("[Stream] modelName=", c?.modelName)
    console.log(
      "[Stream] supportedStreamTypes=",
      JSON.stringify(asRecord(asRecord(c?.camera)?.video)?.supportedStreamTypes),
    )
    return true
  }

  /**
   * 안경 WiFi 를 기다린다(거절하지 않는다).
   *
   * onWifi 가 정확한 값을 주기까지 약 2초 걸리므로, 앱을 연 직후 한 번만 보면
   * 멀쩡한 기기를 "연결 안 됨" 으로 보고한다. 그 탓에 버튼이 죽은 것처럼 보였다.
   * 최신 값을 최대 5초 동안 폴링해서 해결한다.
   */
  async function waitForWifi(): Promise<boolean> {
    const deadline = Date.now() + WIFI_WAIT_MS
    console.log("[Stream] WiFi 대기 시작. 현재=", JSON.stringify(latestWifi))

    while (Date.now() < deadline) {
      if (latestWifi?.connected === true) {
        console.log("[Stream] WiFi 확인됨:", JSON.stringify(latestWifi))
        return true
      }
      await new Promise((resolve) => setTimeout(resolve, WIFI_POLL_MS))
    }

    console.warn(`[Stream] ${WIFI_WAIT_MS}ms 동안 WiFi 미연결. 마지막 값=`, JSON.stringify(latestWifi))
    return false
  }

  /**
   * AI 소켓을 기다린다(거절하지 않는다). waitForWifi 와 같은 모양이다.
   *
   * AiClient 는 비정상 종료 후 스스로 재연결하지만 "재연결 중" 상태가 없다.
   * 백오프 구간 전체가 `error` 로 보이므로 isReady() 한 번으로는 "죽었다" 와
   * "1초 뒤 돌아온다" 를 구분할 수 없다. 3초 폴링이 AiClient 에 상태를 추가하지
   * 않고 그 질문에 답한다.
   */
  async function waitForAi(): Promise<boolean> {
    const deadline = Date.now() + AI_WAIT_MS
    console.log("[Stream] AI 연결 대기 시작. aiState=", ai?.getState())
    // index.ts 가 ai.getState() 를 읽는 두 지점 중 하나다. AiClient 에 UI 를
    // 가르치는 대신 여기서 투영값을 방송한다.
    publishAiState(aiPhase(ai?.getState()))

    while (Date.now() < deadline) {
      if (ai?.isReady() === true) return true
      await new Promise((resolve) => setTimeout(resolve, AI_POLL_MS))
    }

    console.warn("[Stream] AI 연결 대기 실패. aiState=", ai?.getState())
    publishAiState(aiPhase(ai?.getState()), `${AI_WAIT_MS}ms 안에 AI 세션이 준비되지 않았다`)
    return false
  }

  // --- 공용 정리 ---------------------------------------------------------------

  /**
   * 단일 종료 경로. 정지 시퀀스·롤백·에러 복구가 모두 재사용한다. 반복 호출해도
   * 안전하다.
   *   - `activeStream` 을 먼저 비우므로 같은 id 에 대한 두 번째 호출은 무동작.
   *   - stream_stop 은 AiClient 내부의 Set 이 한 번 더 막는다.
   *
   * 순서는 의도적이다. Mentra 스트림이 사라지기 *전에* AI 서버에 pull 중지를
   * 알린다. 반대로 하면 서버가 죽은 WHEP 엔드포인트를 계속 당긴다.
   */
  async function cleanupStream(reason: string, opts: {notifyAi: boolean}): Promise<void> {
    const streamId = activeStream?.streamId
    activeStream = undefined
    // 여기서 LED 를 건드리지 않는 건 의도다. 호출부 둘(runStopSequence, rollback)
    // 모두 직후에 setState 를 부르고, 그 setState 가 patch → applyLed 로 불을
    // 정확히 다시 잡는다. 여기서 끄면 지나가는 길에 암전만 하나 더 생긴다.
    publishStreamState(`cleanup: ${reason}`)
    console.log(`[Stream] cleanup 시작 (${reason}). streamId=`, streamId)

    // 1. AI WebSocket 이 먼저.
    if (opts.notifyAi && streamId !== undefined) {
      ai?.sendStreamStop(streamId)
    }

    // 2. 그다음 Mentra 스트림. stop() 은 void 로 resolve 하므로 성공/실패 여부만
    // 관측된다. 실패 시 인자 없는 형태로 폴백한다. SDK 가 "stop the active
    // stream" 이라고 문서화한 그것이며, id 를 잃었을 때의 탈출구다.
    try {
      await session.stream.stop(streamId)
      console.log("[Stream] stream.stop 성공. streamId=", streamId)
    } catch (err) {
      console.error("[Stream] stream.stop 실패 — 인자 없는 stop() 으로 폴백")
      logRequestError("[Stream] stream.stop", err)
      try {
        await session.stream.stop()
        console.log("[Stream] stream.stop() (인자 없음) 성공")
      } catch (err2) {
        console.error("[Stream] stream.stop() (인자 없음)도 실패 — 스트림이 남아있을 수 있다")
        logRequestError("[Stream] stream.stop()", err2)
        throw err2
      }
    }
  }

  // --- 시작 사다리 -------------------------------------------------------------

  /**
   * STREAM_ATTEMPTS 를 하나가 성공할 때까지 훑는다. 이긴 결과와 그 단이 실제로
   * 요청한 fps 를 돌려주고, 전부 실패하면 undefined 를 돌려준다.
   *
   * STREAM_ATTEMPTS 가 한 항목뿐이어도 루프를 남겨 둔다. 단을 되살릴 때 무엇이
   * 필요한지는 그 상수의 주석 참고.
   */
  async function runStreamLadder(): Promise<{result: StreamResult; requestedFps: number} | undefined> {
    for (let i = 0; i < STREAM_ATTEMPTS.length; i += 1) {
      const attempt = STREAM_ATTEMPTS[i]
      console.log(`[Stream] 시도 ${attempt.name} 시작`, JSON.stringify(attempt.options), `(${attempt.note})`)
      const startedAt = Date.now()

      try {
        const result = await session.stream.startStream(attempt.options)
        console.log(`[Stream] 시도 ${attempt.name} 성공 (${Date.now() - startedAt}ms)`)
        logStreamResult(result)
        return {result, requestedFps: attempt.options.video?.fps ?? REQUESTED_FPS}
      } catch (err) {
        console.error(`[Stream] 시도 ${attempt.name} 실패 (${Date.now() - startedAt}ms)`)
        // logRequestError 가 err.code 와 err.message 를 각각 다른 줄에 찍는다.
        // MiniappErrorCode 에는 camera_busy 코드가 없다. 잡혀 있는 카메라는
        // INTERNAL 로 오고 상세는 `message` 에 담기므로, code 로 매칭하지 말고
        // 그 줄을 읽어야 한다.
        logRequestError(`[Stream] 시도 ${attempt.name}`, err)
      }
    }
    return undefined
  }

  // --- 시작 시퀀스 -------------------------------------------------------------

  async function runStartSequence(): Promise<void> {
    if (startInFlight) {
      console.warn("[Stream] 시작 시퀀스가 이미 진행 중이다 — 무시. state=", appState)
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
        console.error("[Stream] AI 세션이 준비되지 않았다 — 스트림 시작 불가. 다시 누르면 재시도 가능")
        return
      }

      // 1. Wi-Fi.
      setState("waiting_wifi")
      if (!(await waitForWifi())) {
        // waiting_wifi 에 머문다. 사용자가 설정 흐름을 마친 뒤 롱프레스를 다시
        // 누르면 그때 재시도된다.
        void session.glasses
          .requestWifiSetup("수어 인식을 위해 안경을 WiFi에 연결해주세요")
          .then(() => console.log("[Stream] requestWifiSetup 호출됨"))
          .catch((err) => logRequestError("[Stream] requestWifiSetup", err))
        console.warn("[Stream] WiFi 설정 유도 후 대기 — 연결 뒤 롱프레스로 재시도 가능")
        return
      }

      // 2. Mentra 스트림 시작. 실측 약 6~7초.
      setState("starting_stream")
      const attempt = await runStreamLadder()
      if (attempt === undefined) {
        setState("error")
        console.error("[Stream] 스트림 시작 실패 (rung A) — error 상태")
        return
      }
      const {result, requestedFps} = attempt

      const streamId = result?.streamId
      const webrtcUrl = result?.webrtcUrl
      // 요청값과 협상값을 나란히 찍는다. requestedFps 는 모듈 상수가 아니라
      // 실제로 이긴 단의 요청값이라, 나중에 30 이외를 요청하는 단이 생겨도
      // 정직하게 보고된다.
      console.log("[Stream] fps 요청=", requestedFps, "/ resolvedConfig=", result?.resolvedConfig?.video?.fps)

      // 검증보다 진단을 먼저 보내는 이유: "성공" 했지만 webrtcUrl 이 없는 단이야
      // 말로 보여 줄 가치가 있는 경우인데, 아래 롤백이 먼저 반환해 버리면 UI 에
      // 아무것도 닿지 않는다.
      //
      // resolvedConfig 는 Optional 이다. 모든 단계를 옵셔널 체이닝으로 탄다.
      // 호스트명은 parseUrlParts() 로 뽑는다 — 이 런타임에 `new URL()` 은 없다.
      // 새 스트림의 서버 카운터는 1부터 다시 시작한다. 이전 스트림의 처리 fps 를
      // 그대로 들고 있으면 안 되므로 기준점과 표시값을 함께 비운다.
      resetProcessedFps()
      patch("stream:diagnostics", {
        requestedFps,
        resolvedFps: result?.resolvedConfig?.video?.fps ?? null,
        processedFps: null,
        transport: result?.resolvedConfig?.transport ?? null,
        webrtcHost:
          typeof webrtcUrl === "string" && webrtcUrl.length > 0
            ? (parseUrlParts(webrtcUrl)?.hostname ?? null)
            : null,
        mode: result?.mode ?? null,
        status: result?.status ?? null,
      })

      // 검증 전에 기록해 둔다. 롤백이 언제든 id 를 찾을 수 있어야 한다.
      // webrtcUrl 은 아래 검증을 통과한 뒤에 채운다.
      activeStream = typeof streamId === "string" ? {streamId, webrtcUrl: undefined} : undefined
      publishStreamState()

      // 3. WHEP URL 검증. 여기서 쓸 수 없는 값은 치명적 실패다 — 서버도 그것을
      // 당길 수 없다.
      if (typeof webrtcUrl !== "string" || webrtcUrl.length === 0) {
        console.error("[Stream] webrtcUrl 이 없다 — 롤백한다. result.mode=", result?.mode)
        await rollback("webrtcUrl 없음")
        return
      }
      const parts = parseUrlParts(webrtcUrl)
      if (parts === undefined) {
        console.error("[Stream] webrtcUrl 정규식 파싱 실패 — 롤백한다. 원본:", webrtcUrl)
        await rollback("webrtcUrl 파싱 실패")
        return
      }
      console.log(
        `[Stream] webrtcUrl protocol=${parts.protocol} hostname=${parts.hostname} port=${parts.port}`,
      )
      if (typeof streamId !== "string" || streamId.length === 0) {
        console.error("[Stream] streamId 가 없다 — stream_start 를 보낼 수 없다. 롤백한다")
        await rollback("streamId 없음")
        return
      }

      // 검증된 URL 을 보관한다. 재연결 뒤 onReady 가 이걸로 재전송한다.
      activeStream = {streamId, webrtcUrl}

      // 4. URL 을 AI 서버에 넘긴다.
      // `ai.` 가 아니라 `ai?.` 인 이유: 예전의 `ai?.isReady() !== true` 가드는
      // 함수 끝까지 `ai` 를 좁혀 주었지만, waitForAi 는 평범한 boolean 을
      // 반환해서 TS 가 그 정보를 이어받지 못한다. 실제로 undefined 가 여기 올 수는
      // 없고(waitForAi 는 `ai?.isReady() === true` 로만 true 를 준다), 설령
      // 온다 해도 `!undefined` 가 롤백으로 보내므로 그것이 옳은 답이다.
      if (!ai?.sendStreamStart(streamId, webrtcUrl)) {
        console.error("[Stream] stream_start 전송 실패 — 롤백한다")
        await rollback("stream_start 전송 실패")
        return
      }

      // 5. 완료.
      setState("streaming")
      console.log("[Stream] streaming. streamId=", streamId)
    } finally {
      startInFlight = false
    }
  }

  /**
   * 반쯤 시작된 스트림을 되돌린다.
   *
   * 이전 빌드는 실패 경로에서 stop() 을 부르지 않아, 검증에 걸린 스트림이 카메라를
   * 쥔 채 살아남았고 이후의 모든 start 가 busy 로 실패했다. 롤백이 그걸 막는다.
   *
   * `notifyAi: false` 인 이유: stream_start 를 아예 보내지 않았거나 전송에
   * 실패했으므로 서버가 중지할 pull 이 없다.
   *
   * idle 이 아니라 ai_ready 로 끝난다. 스트림 실패는 AI 세션과 무관하다.
   */
  async function rollback(why: string): Promise<void> {
    console.warn("[Stream] 롤백:", why)
    try {
      await cleanupStream(`rollback: ${why}`, {notifyAi: false})
    } catch {
      // cleanupStream 이 두 번의 stop 시도를 이미 로그로 남겼다.
      console.error("[Stream] 롤백 중 stop 실패 — 카메라가 잡혀있을 수 있다")
    }
    setState(ai?.isReady() === true ? "ai_ready" : "error")
  }

  // --- 정지 시퀀스 -------------------------------------------------------------

  async function runStopSequence(reason: string): Promise<void> {
    setState("stopping")
    try {
      // stream_stop 이 아직 열려 있는 WebSocket 으로 먼저 나간다. 이전 빌드는
      // 이보다 먼저 소켓을 닫아 stream_stop 이 끝내 나가지 못했다.
      await cleanupStream(reason, {notifyAi: true})
      setState(ai?.isReady() === true ? "ai_ready" : "error")
    } catch {
      setState("error")
      console.error("[Stream] 정지 실패 — error 상태. 롱프레스로 재시도 가능")
    }
  }

  // --- 입력 -------------------------------------------------------------------

  // 물리 버튼만 쓴다. session.input.onTouch 는 일부러 연결하지 않았다. 그쪽의
  // "long_press" 는 터치패드 제스처라 아래 물리 버튼과 다른 표면이다.
  unsubscribers.push(
    session.input.onButtonPress((press) => {
      if (press.pressType === "long") {
        // `buttonId` 에 실제로 무엇이 담기는지 보려고 페이로드 전문을 찍는다.
        console.log("[Input] LONG", JSON.stringify(press), "state=", appState)

        // 제스처 하나에 의미 하나 — 스트림 토글. AI 세션은 "ready" 에서 연결되고
        // "disconnect" 에서 정리되며 둘 다 자동이다. 롱프레스는 거기에 관여하지
        // 않는다. double_press 도 두지 않았다. hasDisplay=false 라 지금 어떤
        // 모드인지 보여 줄 방법이 없어서, 두 번째 제스처는 짐작할 수 없게 된다.
        switch (appState) {
          case "ai_ready":
          case "connecting_ai":
          case "waiting_wifi":
            // connecting_ai 를 받는 이유는 waitForAi 가 처리해 주기 때문이다.
            // AI 소켓이 (재)연결 중이더라도 3초 안에 붙는 경우가 많다. idle 은
            // 일부러 넣지 않았다 — 거기서는 `ai` 를 아직 만들지도 않아 기다릴
            // 대상이 없다.
            //
            // waiting_wifi 는 재시도 진입점이다. 사용자가 설정 흐름으로 WiFi 를
            // 연결하고 돌아와 다시 누르는 자리다.
            void runStartSequence().catch((err) => {
              setState("error")
              logRequestError("[Stream] 시작 시퀀스 예외", err)
            })
            return

          case "streaming":
            void runStopSequence(`long press (buttonId=${press.buttonId})`).catch((err) => {
              setState("error")
              logRequestError("[Stream] 정지 시퀀스 예외", err)
            })
            return

          case "error":
            // 복구: 잡혀 있는 것을 정리하고, AI 소켓이 살아 있으면 ai_ready 로
            // 돌아간다. 핫 리로드 탈출구이기도 하다.
            console.warn("[Stream] error 상태 — 복구 정지 시도")
            void runStopSequence("error 복구").catch((err) => {
              setState("error")
              logRequestError("[Stream] 복구 정지 예외", err)
            })
            return

          default:
            // idle: 아직 AI 클라이언트를 만들지 않았다("ready" 미발생).
            // starting_stream(약 6~7초) / stopping: 진행 중이라 두 번째 누름이
            // 경쟁 상태를 만든다.
            console.warn("[Stream] 지금은 롱프레스를 받지 않는다. state=", appState)
            return
        }
      } else {
        // 짧게 누르는 것은 의도적으로 무동작이다. 예전 빌드는 여기서 사진 촬영을
        // 걸었고 그것이 camera_busy 의 원인이었다. 이어받지 않았다.
        console.log("[Input] short (무시)")
      }
    }),
  )
})
