// 음성 출력 헬퍼. 문구에 기호를 쓰지 않는다 — 폰 TTS 가 %, /, + 를 영어 단어로 읽는다.

import type {SpeakerModule} from "@mentra/miniapp/background"

// 기동 시 한 번 출력. 인식 결과는 서버 text 를 그대로 읽으므로 상수가 없다.
export const START_ANNOUNCEMENT =
  "안녕하세요, 골든사인입니다. 환자분의 수어를 안경 카메라로 읽고, 그 뜻을 바로 음성으로 알려 드릴게요. 버튼을 길게 누르면 인식이 시작되고, 수어 단어 하나를 시작할 때와 끝날 때 짧게 눌러 주세요."

// 설정을 넘기면 서버 기본값이 통째로 교체되므로 속도만 바꿀 때도 네 값을 모두 넘긴다.
const VOICE_SETTINGS = {speed: 1.0, stability: 0.68, similarity_boost: 0.75, style: 0}

type UnknownRecord = Record<string, unknown>

function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null ? (value as UnknownRecord) : undefined
}

// speak 실패는 Error 가 아니라 {code, message} 객체로 올 수 있어 구조적으로 읽는다.
function logSpeakError(tag: string, err: unknown): void {
  const e = asRecord(err)
  const code = typeof e?.code === "string" ? e.code : "(없음)"
  const message = typeof e?.message === "string" ? e.message : JSON.stringify(err)
  console.warn(`[TTS] ${tag} 실패 code=${code} message=${message}`)
}

// 재생이 끝나야 풀리므로 기다리지 않는다. 어떤 경우에도 throw 하지 않는다.
export function speakSafe(speaker: SpeakerModule, text: string, tag: string): void {
  if (text.trim() === "") {
    console.warn(`[TTS] ${tag} 문구가 비어 있다 — speak 생략`)
    return
  }

  // voice_id 금지 — "ko" 를 넣으면 클라우드가 실패하고 오프라인으로 조용히 넘어간다.
  try {
    void speaker
      .speak(text, {voice_settings: VOICE_SETTINGS})
      .then((result) => {
        // completed=false 는 실패가 아니라 중단
        console.log(`[TTS] ${tag} 완료 completed=${result.completed}`)
      })
      .catch((err: unknown) => {
        logSpeakError(tag, err)
      })
  } catch (err) {
    logSpeakError(`${tag} 동기 예외`, err)
  }
}