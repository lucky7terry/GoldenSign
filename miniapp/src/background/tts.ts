// 음성 출력 헬퍼. 서버가 합성한 한국어 음성을 먼저 재생하고 실패하면 폰 TTS 로 폴백한다.
// 문구에 기호를 쓰지 않는다 — 폴백의 폰 TTS 가 %, /, + 를 영어 단어로 읽는다.

import type {SpeakerModule} from "@mentra/miniapp/background"
import {AI_HTTP} from "../shared/config"

// 기동 시 한 번 출력. 인식 결과는 서버 text 를 그대로 읽으므로 상수가 없다.
export const START_ANNOUNCEMENT =
  "골든사인입니다. 버튼을 길게 누르면 인식이 시작됩니다. 수어 단어의 시작과 끝에 버튼을 짧게 눌러 주세요."

// 구간 인식이 실패했을 때 LED 대신 소리로 알린다.
export const RETRY_ANNOUNCEMENT = "인식에 실패했습니다. 재시도해주세요"

// 서버 app/api/tts.py 의 FRAMING_ANNOUNCEMENT 와 글자까지 같아야 미리 만든 음성 캐시가 맞는다.
export const FRAMING_ANNOUNCEMENT = "상대방의 얼굴과 손이 함께 보이도록 바라보세요"

// 폴백용 폰 TTS 설정. 넘기면 호스트 기본값이 통째로 교체되므로 속도만 바꿀 때도 네 값을 모두 넘긴다.
const VOICE_SETTINGS = {speed: 1.0, stability: 0.68, similarity_boost: 0.75, style: 0}

// 서버 TTS 엔드포인트. shared/config.ts 가 아니라 여기 두는 것은 이 파일만 쓰기 때문이다.
const TTS_PATH = "/v1/tts"

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

// 이 런타임에는 URL 전역이 없어 문자열로 잇는다.
function ttsAudioUrl(text: string): string {
  return `${AI_HTTP}${TTS_PATH}?text=${encodeURIComponent(text)}`
}

/**
 * 서버 재생이 실패한 자리. 폰 TTS 는 클라우드가 안 되면 오프라인 음성으로라도 읽으므로,
 * 서버가 죽었거나 키가 없어도 착용자는 무언가를 듣는다.
 */
function speakOnPhone(speaker: SpeakerModule, text: string, tag: string, cause: unknown): void {
  logSpeakError(`${tag} 서버 재생`, cause)
  console.warn(`[TTS] ${tag} 폰 TTS 로 폴백`)

  // voice_id 금지 — "ko" 를 넣으면 클라우드가 실패하고 오프라인으로 조용히 넘어간다.
  try {
    void speaker
      .speak(text, {voice_settings: VOICE_SETTINGS, stopOtherAudio: true})
      .then((result) => {
        // completed=false 는 실패가 아니라 중단
        console.log(`[TTS] ${tag} 폴백 완료 completed=${result.completed}`)
      })
      .catch((err: unknown) => {
        logSpeakError(`${tag} 폴백`, err)
      })
  } catch (err) {
    logSpeakError(`${tag} 폴백 동기 예외`, err)
  }
}

/**
 * 재생이 끝나야 풀리므로 기다리지 않는다. 어떤 경우에도 throw 하지 않는다.
 *
 * play 는 오디오 바이트가 아니라 URL 을 받으므로 앱은 mp3 를 다룰 일이 없다.
 * 서버가 404 나 502 를 주면 reject 되고 그때 폰 TTS 가 받는다 — 서버가 빈
 * 200 을 주면 폴백할 줄 모르니, /v1/tts 는 실패를 status 로 내야 한다.
 */
export function speakSafe(speaker: SpeakerModule, text: string, tag: string): void {
  if (text.trim() === "") {
    console.warn(`[TTS] ${tag} 문구가 비어 있다 — 재생 생략`)
    return
  }

  try {
    void speaker
      .play({audioUrl: ttsAudioUrl(text), stopOtherAudio: true})
      .then(() => {
        console.log(`[TTS] ${tag} 서버 재생 완료`)
      })
      .catch((err: unknown) => {
        speakOnPhone(speaker, text, tag, err)
      })
  } catch (err) {
    speakOnPhone(speaker, text, tag, err)
  }
}
