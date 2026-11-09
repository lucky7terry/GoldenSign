// 음성 출력 헬퍼. 서버가 합성한 한국어 음성을 먼저 재생하고 실패하면 폰 TTS 로 폴백한다.
// 문구에 기호를 쓰지 않는다 — 폴백의 폰 TTS 가 %, /, + 를 영어 단어로 읽는다.

import type {SpeakerModule, SpeakerStateEvent} from "@mentra/miniapp/background"
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

/**
 * 재생 결과를 서버 로그로 보고하는 함수. 이 모듈은 AI 소켓을 모르므로
 * index.ts 가 ai-client 로 가는 함수를 넘긴다.
 *
 * 착용자가 아무 소리도 못 듣는 경우가 앱 안에서만 보이는 사건이다 — 서버는
 * /v1/tts 요청이 왔는지만 알고, 폰 TTS 폴백은 전혀 알지 못한다.
 */
export type TtsReporter = (
  level: "info" | "warn" | "error",
  fields: Record<string, string | number | boolean>,
) => void

// speak 실패는 Error 가 아니라 {code, message} 객체로 올 수 있어 구조적으로 읽는다.
function logSpeakError(tag: string, err: unknown): void {
  const e = asRecord(err)
  const code = typeof e?.code === "string" ? e.code : "(없음)"
  const message = typeof e?.message === "string" ? e.message : JSON.stringify(err)
  console.warn(`[TTS] ${tag} 실패 code=${code} message=${message}`)
}

// 서버 로그에 실을 짧은 식별자. code 가 없으면 Error 의 name 이라도 남긴다.
function errorCodeOf(err: unknown): string {
  const code = asRecord(err)?.code
  if (typeof code === "string" && code !== "") return code
  if (err instanceof Error && err.name !== "") return err.name
  return "unknown"
}

// 이 런타임에는 URL 전역이 없어 문자열로 잇는다.
function ttsAudioUrl(text: string): string {
  return `${AI_HTTP}${TTS_PATH}?text=${encodeURIComponent(text)}`
}

// ---------------------------------------------------------------------------
// 재생 시작·끝 측정
// ---------------------------------------------------------------------------

/** 음성이 서버 합성에서 났는지 폰 TTS 폴백에서 났는지. 로그에만 쓴다. */
type SpeechSource = "server" | "fallback"

/**
 * 마지막 음성 요청. 재생 시작·끝까지 걸린 시간을 재는 기준점이다.
 *
 * 하나로 충분하다 — `stopOtherAudio` 때문에 새 요청이 이전 재생을 끊으므로
 * 동시에 두 개가 나지 않고, 끊긴 이전 요청은 덮어쓰면서 자연히 버려진다.
 * background 런타임에는 performance 가 없어 시각은 Date.now() 를 쓴다.
 */
let pendingSpeech:
  | {requestedAt: number; tag: string; source: SpeechSource; startMs?: number; errorCode?: string}
  | undefined

function beginSpeech(tag: string, source: SpeechSource): void {
  pendingSpeech = {requestedAt: Date.now(), tag, source}
}

/**
 * `speaker.onStateChange` 에 걸어 재생 시작·끝을 로그로 남긴다. 측정 전용이며
 * 화면도 동작도 바꾸지 않는다.
 *
 * `play()` 의 promise 는 재생이 *끝날* 때 풀리므로(modules/speaker.d.ts) 소리가
 * 나기 시작한 시각은 이 이벤트로만 알 수 있다. `loading` 과 `idle` 은 남길 것이
 * 없어 무시한다. `error` 는 과도 상태라 곧 `stopped` 가 따라오고, 요청 상태는
 * 거기서 비운다.
 */
export function handleSpeakerState(event: SpeakerStateEvent, report?: TtsReporter): void {
  const pending = pendingSpeech
  // 우리가 요청하지 않은 재생이거나, 이미 끝낸 요청의 뒤늦은 이벤트다.
  if (pending === undefined) return

  const elapsedMs = Date.now() - pending.requestedAt
  const head = `[Latency] tts tag=${pending.tag} source=${pending.source}`

  if (event.state === "playing") {
    // 재생 중 상태 변화가 여러 번 와도 시작은 한 번만 남긴다.
    if (pending.startMs !== undefined) return
    pending.startMs = elapsedMs
    console.log(`${head} start_ms=${elapsedMs}`)
    return
  }
  if (event.state === "stopped") {
    console.log(`${head} end_ms=${elapsedMs} duration_ms=${String(event.durationMs)}`)
    // 재생 한 건의 측정을 stopped 에서 한 번에 보낸다. 못 잰 숫자 자리에는 사유 문자열이 들어간다.
    const fields: Record<string, string | number | boolean> = {
      tag: pending.tag,
      source: pending.source,
      start_ms: pending.startMs ?? "not_started",
      end_ms: elapsedMs,
      duration_ms: typeof event.durationMs === "number" ? event.durationMs : "unknown",
    }
    if (pending.errorCode !== undefined) fields.error_code = pending.errorCode
    report?.("info", fields)
    pendingSpeech = undefined
    return
  }
  if (event.state === "error") {
    console.log(
      `${head} error_ms=${elapsedMs} error_code=${String(event.errorCode)}` +
        ` error_message=${String(event.errorMessage)}`,
    )
    // error 는 과도 상태라 보고는 뒤따르는 stopped 에서 하고 코드만 담아 둔다.
    pending.errorCode = String(event.errorCode)
  }
}

/**
 * 서버 재생이 실패한 자리. 폰 TTS 는 클라우드가 안 되면 오프라인 음성으로라도 읽으므로,
 * 서버가 죽었거나 키가 없어도 착용자는 무언가를 듣는다.
 */
function speakOnPhone(
  speaker: SpeakerModule,
  text: string,
  tag: string,
  cause: unknown,
  report?: TtsReporter,
): void {
  logSpeakError(`${tag} 서버 재생`, cause)
  console.warn(`[TTS] ${tag} 폰 TTS 로 폴백`)

  // 서버가 왜 실패했는지는 폴백이 성공해도 남겨야 한다. 들리긴 했지만 음색이
  // 다르고, 고쳐야 할 것은 서버 쪽이다.
  const serverErrorCode = errorCodeOf(cause)

  // voice_id 금지 — "ko" 를 넣으면 클라우드가 실패하고 오프라인으로 조용히 넘어간다.
  try {
    // 폴백은 새 요청이다. 측정 기준을 서버 재생 시도 시각에 두면 실패에 쓴
    // 시간이 재생 시작까지의 시간에 섞인다.
    beginSpeech(tag, "fallback")
    void speaker
      .speak(text, {voice_settings: VOICE_SETTINGS, stopOtherAudio: true})
      .then((result) => {
        // completed=false 는 실패가 아니라 중단
        console.log(`[TTS] ${tag} 폴백 완료 completed=${result.completed}`)
        report?.("warn", {
          tag,
          source: "fallback",
          result: "ok",
          server_error_code: serverErrorCode,
        })
      })
      .catch((err: unknown) => {
        logSpeakError(`${tag} 폴백`, err)
        // 착용자가 아무 소리도 듣지 못한 경우다.
        report?.("error", {
          tag,
          source: "fallback",
          result: "fail",
          error_code: errorCodeOf(err),
          server_error_code: serverErrorCode,
        })
      })
  } catch (err) {
    logSpeakError(`${tag} 폴백 동기 예외`, err)
    report?.("error", {
      tag,
      source: "fallback",
      result: "fail",
      error_code: errorCodeOf(err),
      server_error_code: serverErrorCode,
    })
  }
}

/**
 * 재생이 끝나야 풀리므로 기다리지 않는다. 어떤 경우에도 throw 하지 않는다.
 *
 * play 는 오디오 바이트가 아니라 URL 을 받으므로 앱은 mp3 를 다룰 일이 없다.
 * 서버가 404 나 502 를 주면 reject 되고 그때 폰 TTS 가 받는다 — 서버가 빈
 * 200 을 주면 폴백할 줄 모르니, /v1/tts 는 실패를 status 로 내야 한다.
 */
export function speakSafe(
  speaker: SpeakerModule,
  text: string,
  tag: string,
  report?: TtsReporter,
): void {
  if (text.trim() === "") {
    console.warn(`[TTS] ${tag} 문구가 비어 있다 — 재생 생략`)
    return
  }

  try {
    beginSpeech(tag, "server")
    void speaker
      .play({audioUrl: ttsAudioUrl(text), stopOtherAudio: true})
      .then(() => {
        console.log(`[TTS] ${tag} 서버 재생 완료`)
        report?.("info", {tag, source: "server", result: "ok"})
      })
      .catch((err: unknown) => {
        speakOnPhone(speaker, text, tag, err, report)
      })
  } catch (err) {
    speakOnPhone(speaker, text, tag, err, report)
  }
}
