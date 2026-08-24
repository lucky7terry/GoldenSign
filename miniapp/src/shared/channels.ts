/**
 * 타입 지정 채널 레지스트리. background JSContext 와 UI WebView 사이를 오가는
 * 채널 이름과 페이로드 모양의 단일 출처다. 양쪽이 이 파일을 빌드 타임에
 * 임포트하고 번들러가 선언을 인라인하므로 런타임 해석은 없다.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * 브로드캐스트 vs RPC
 * ─────────────────────────────────────────────────────────────────────────
 * 값이 `Rpc<Req, Res>` 인 채널만 RPC 다. @mentra/miniapp 의 브랜드 타입이고
 * (dist/modules/ui.d.ts), `IsRpc<T>` 가 그 브랜드를 검사한다. 나머지는 전부
 * 브로드캐스트다. 채널에 맞지 않는 API 를 쓰면 컴파일 에러가 난다.
 *
 *   브로드캐스트 → session.ui.send / session.ui.on   +  mentra.send / mentra.on
 *   RPC          → session.ui.handle                 +  mentra.request
 *
 * ─────────────────────────────────────────────────────────────────────────
 * `?: T | null` — 둘 다 붙이는 이유
 * ─────────────────────────────────────────────────────────────────────────
 * 페이로드는 JSON.stringify 를 거쳐 브리지를 건너는데(envelope.js:10), 이때
 * undefined 값을 가진 키가 삭제된다. 그래서 "아직 모른다" 는 명시적 `null` 로
 * 실어야 한다. undefined 면 UI 가 "이 빌드에 없는 필드" 와 구분하지 못한다.
 * `?` 는 담당이 아닌 생산자가 키를 생략할 수 있게 남겨 둔 것이고, 답이 정말로
 * "모름" 일 때 쓰는 값이 `| null` 이다.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * 소유권
 * ─────────────────────────────────────────────────────────────────────────
 * 네트워크 요청, AI WebSocket, 스트림 제어는 전부 background 전담이다. UI 는
 * 구독하고 그릴 뿐 하드웨어나 네트워크 작업을 시작하지 않는다.
 *
 * `getSnapshot` 이 필요한 이유: `session.ui.send` 는 바인딩된 WebView 가 없으면
 * 조용히 드롭한다. 밀린 메시지를 나중에 재생해 주지 않으므로, 늦게 마운트된
 * UI 는 마운트 시 현재 상태를 한 번 물어보는 수밖에 없다.
 */

import type {Rpc} from "@mentra/miniapp/ui"

/**
 * 갓 마운트된 UI 가 다음 브로드캐스트를 기다리지 않고 바로 그릴 수 있는 만큼의
 * 상태. background 의 단일 `patch()` 가 브로드캐스트와 같은 값을 여기에도
 * 기록하므로, 모든 필드는 해당 채널로 마지막에 보낸 값과 일치한다.
 */
export interface Snapshot {
  ai: Channels["ai:state"]
  stream: Channels["stream:state"]
  glasses: Channels["glasses:state"]
  diagnostics: Channels["stream:diagnostics"]
  /** 최근 결과. 오래된 것부터 버린다. 상한은 background 가 건다. */
  results: Channels["recognition:result"][]
}

export interface Channels {
  // ---------------------------------------------------------------------
  // background → WebView (브로드캐스트)
  // ---------------------------------------------------------------------

  /**
   * AI 서버 연결 단계. AiClient 의 7개짜리 `AiClientState` 를 4개로 접었다.
   * UI 는 creating_session 과 connecting_ws 를 구분할 이유가 없고,
   * AiClientState 자체는 일부러 손대지 않았다.
   */
  "ai:state": {
    state: "disconnected" | "connecting" | "ready" | "error"
    sessionId?: string | null
    /** 서버 `ready` 메시지의 `model` 블록. */
    model?: {loaded: boolean; mode: string; version: string} | null
    message?: string | null
  }

  /** background 의 `AppState` 를 그대로 옮긴 것. 여덟 개 이름이 동일하다. */
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
   * 인식 결과 한 건.
   *
   * `windowIndex` 는 서버의 `result.sequence.window_index` 에서 온다
   * (sequence_service.metadata()). `result.sequence_index` 라는 필드는
   * 서버에 존재하지 않는다.
   *
   * 서버는 60프레임이 차기 전 구간에서 window_index 를 null 로 보낸다.
   * 그 경우 background 가 -1 센티널을 넣는다.
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
    /** 배터리 잔량 퍼센트 (BatteryData.level). */
    battery?: number | null
    charging?: boolean | null
  }

  "error": {code: string; message: string; retryable?: boolean | null}

  /** 요청값 대 협상값. startStream 성공 이후 채워진다. */
  "stream:diagnostics": {
    requestedFps?: number | null
    resolvedFps?: number | null
    transport?: string | null
    /** 호스트명만. parseUrlParts() 로 뽑는다. `new URL()` 은 런타임에 없다. */
    webrtcHost?: string | null
    mode?: string | null
    status?: string | null
  }

  // ---------------------------------------------------------------------
  // WebView → background (RPC)
  // ---------------------------------------------------------------------

  /**
   * UI 마운트 시 현재 상태를 한 번 가져온다. 인자가 없으므로 요청 페이로드
   * 타입이 `{}` 이고, 호출부는 `mentra.request("getSnapshot", {})` 로 쓴다.
   */
  "getSnapshot": Rpc<{}, Snapshot>
}

declare global {
  // @mentra/miniapp/ui 의 `mentra` 전역에 이 미니앱의 채널 레지스트리를 입힌다.
  // 이렇게 해두면 mentra.send / mentra.on / mentra.request 호출마다 채널 이름과
  // 페이로드가 컴파일 타임에 검사된다.
  // eslint-disable-next-line no-var
  var mentra: import("@mentra/miniapp/ui").MentraTyped<Channels>
}
