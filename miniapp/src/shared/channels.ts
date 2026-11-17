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

/** 대기 화면에서 고르는 진행 방식. 마지막 선택은 background 가 메모리로만 유지한다. */
export type Mode = "questionnaire" | "free"

/** 스트림 상태와 별개의 화면 상태. 전이 규칙은 background 가 소유한다. */
export type Screen = "idle" | "active" | "summary"

/**
 * 답으로 모인 단어 한 개. `id` 는 background 가 붙이는 증가 번호다 —
 * windowIndex 는 -1 센티널이 있어 식별자로 쓸 수 없다.
 */
export interface AnswerWord {
  id: number
  text: string
  confidence: number
  windowIndex: number
}

/** 질문 하나의 답. `adopted` 는 문장 후보에서 채택한 문장이다(없으면 null). */
export interface SessionQuestion {
  words: AnswerWord[]
  adopted?: string | null
}

/**
 * 기록 세션 한 건. 경계는 스트림이 아니라 "모드 시작~종료"다 — 요약에서 자유
 * 모드로 이어 가면 스트림은 그대로여도 세션은 새로 시작하고, 재연결로 서버
 * session_id 가 바뀌어도 세션은 유지된다. 자유 모드는 questions[0] 하나만 쓴다.
 */
export interface SessionRecord {
  mode: Mode
  startedAt: number
  endedAt?: number | null
  questions: SessionQuestion[]
}

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
  /** 마지막으로 받은 에러. 아직 없으면 null. */
  error: Channels["error"] | null
  /** 화면·기록 상태. 한 덩어리로 다뤄 스냅샷 병합이 슬롯 하나로 끝난다. */
  session: Channels["session:state"]
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
   * (sequence_service.metadata()).
   *
   * 서버는 60프레임이 차기 전 구간에서 window_index 를 null 로 보낸다.
   * 그 경우 background 가 -1 센티널을 넣는다.
   *
   * `sequenceIndex` 는 `result` 안이 아니라 서버 메시지 **최상위** 의
   * `sequence_index` 에서 온다. #46 이후 서버가 이 필드를 보내지 않아 대개
   * null 이고, 처리 fps 는 background 가 `word_progress` 의 frame_count
   * 차분으로 따로 계산한다.
   */
  "recognition:result": {
    /** 모델이 판정한 단어. 판정이 없으면 null 이다. */
    text: string | null
    confidence: number
    isFinal: boolean
    windowIndex: number
    /** 서버 누적 처리 프레임 수. 서버가 숫자로 주지 않으면 null. */
    sequenceIndex: number | null
    /**
     * 이 결과가 단어 구간 결과인지. background 가 서버의 `word` 블록 유무로
     * 정한다 — 서버 필드가 아니라 앱이 만든 판단이라 화면이 기대도 된다.
     */
    isWordResult: boolean
    // 아래는 단어 구간 결과에만 실려 온다. 프레임 스트림 결과에는 없다.
    /** "client" | "timeout" — 구간이 왜 닫혔는지. */
    closeReason?: string
    /** 구간이 모은 원본 프레임 수(리샘플 전). */
    wordFrameCount?: number
    /** 구간의 실제 길이(ms). */
    spanMs?: number
    /** 결과 생성 시점의 모델 로드 여부. false 면 confidence 는 의미가 없다. */
    modelLoaded?: boolean
    /** 서버의 채택 임계값. 확신도 막대의 기준선이다. 안 실려 오면 null. */
    confidenceThreshold?: number | null
    /** 서버가 1위로 본 후보. 확신 부족 카드에 보여 준다. */
    candidate?: string | null
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
    /**
     * 서버가 실제로 처리하고 있는 fps. 연속한 두 result 의 최상위
     * `sequence_index` 차이를 경과 시간으로 나눈 값이라, 첫 result 하나만으로는
     * 계산되지 않는다(그동안은 null).
     */
    processedFps?: number | null
    transport?: string | null
    /** 호스트명만. parseUrlParts() 로 뽑는다. `new URL()` 은 런타임에 없다. */
    webrtcHost?: string | null
    mode?: string | null
    status?: string | null
  }

  /**
   * 화면·기록 상태 한 덩어리. 모드 선택, 화면 전이, 단어 추가·삭제, 구간
   * 열림/닫힘이 전부 이 채널 하나로 방송된다.
   */
  "session:state": {
    mode: Mode
    screen: Screen
    /** 문진 모드의 현재 질문(0~3). 자유 모드에서는 늘 0 이다. */
    questionIndex: number
    /** 단어 구간이 열려 있는지. UI 가 진행 중 점선 칩을 그리는 근거다. */
    segmentOpen: boolean
    /** 진행 중인 기록 세션. 없으면 null. */
    session: SessionRecord | null
    /** 끝난 세션 목록. 최신순 최대 10건이며 영구 저장은 하지 않는다. */
    pastSessions: SessionRecord[]
  }

  // ---------------------------------------------------------------------
  // WebView → background (RPC)
  // ---------------------------------------------------------------------

  /**
   * UI 마운트 시 현재 상태를 한 번 가져온다. 인자가 없으므로 요청 페이로드
   * 타입이 `{}` 이고, 호출부는 `mentra.request("getSnapshot", {})` 로 쓴다.
   */
  "getSnapshot": Rpc<{}, Snapshot>

  /** 대기 화면의 진행 방식 선택. idle 화면에서만 의미가 있다. */
  "setMode": Rpc<{mode: Mode}, {}>

  /** 다음 질문으로. 마지막 질문에서 부르면 summary 로 넘어간다(스트림 유지). */
  "nextQuestion": Rpc<{}, {}>

  /** 이전 질문으로. 기존 답은 그대로 두고 이어서 추가한다. */
  "prevQuestion": Rpc<{}, {}>

  /** 답 단어 삭제. id 는 AnswerWord.id 다. */
  "removeWord": Rpc<{wordId: number}, {}>

  /** 문장 후보 채택. 후보는 background 가 buildCandidates 로 다시 만들어 찾는다. */
  "adoptCandidate": Rpc<{questionIndex: number; candidateIndex: number}, {}>

  /** 요약에서 새 자유 세션으로 이어 간다. 스트림이 죽어 있으면 다시 시작한다. */
  "continueFree": Rpc<{}, {}>

  /** 요약에서 대기 화면으로. 스트림이 살아 있으면 종료한다. */
  "goHome": Rpc<{}, {}>

  /** 문장 후보의 듣기 버튼. 넘긴 문장을 안경 스피커로 읽는다. */
  "speakText": Rpc<{text: string}, {}>

  /** 지난 기록 목록을 전부 비운다. */
  "clearHistory": Rpc<{}, {}>
}

declare global {
  // @mentra/miniapp/ui 의 `mentra` 전역에 이 미니앱의 채널 레지스트리를 입힌다.
  // 이렇게 해두면 mentra.send / mentra.on / mentra.request 호출마다 채널 이름과
  // 페이로드가 컴파일 타임에 검사된다.
  // eslint-disable-next-line no-var
  var mentra: import("@mentra/miniapp/ui").MentraTyped<Channels>
}
