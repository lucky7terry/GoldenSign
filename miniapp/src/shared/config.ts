/**
 * 미니앱 공용 빌드 타임 설정.
 *
 * background JSContext 가 임포트한다. 값은 번들러가 인라인하므로 런타임 설정
 * 조회도, 환경변수 접근도 없다 — bare JS 엔진에는 `process` 가 없다.
 */

/**
 * AI 서버 base URL. 배포 서버를 가리킨다.
 *
 * ai-client.ts 가 이 값의 scheme 만 바꿔 WebSocket 주소를 만들므로
 * (https -> wss) https 로 둔다. tts.ts 도 이 값으로 /v1/tts URL 을 만든다.
 *
 * 로컬 서버로 돌릴 때는 Mac 의 LAN IP(http://192.168.0.24:8000 꼴)로 바꾼다 —
 * background 번들은 *폰* 에서 도는 코드라 localhost/127.0.0.1 은 폰 자신을
 * 가리키고, 사설 IP 로의 평문 http:// 는 iOS ATS 와 Android cleartext 정책이
 * 차단할 수 있다.
 */
export const AI_HTTP = "https://goldensign.duckdns.org"

/**
 * 세션·핸드셰이크 메시지 계열(hello / ready / frame / result / ack / stop)의
 * 스키마 버전. server/app/constants.py 의 SCHEMA_VERSION 과 일치해야 한다.
 */
export const HELLO_SCHEMA = "dev-0.2"

/**
 * WebRTC 스트림 메시지 계열(stream_start / stream_stop)의 스키마 버전.
 * server/app/constants.py 의 WEBRTC_SCHEMA_VERSION 과 일치해야 한다.
 *
 * 세션 계열과 버전이 다르다. 서버가 stream_start/stop 에 대해서만
 * dev-0.3 을 요구하고 불일치 시 unsupported_schema_version 으로 거절하므로,
 * 두 상수를 하나로 합칠 수 없다.
 */
export const STREAM_SCHEMA = "dev-0.3"

/**
 * 단어 구간 메시지 계열(word_start / word_end)의 스키마 버전.
 * server/app/constants.py 의 WORD_SCHEMA_VERSION 과 일치해야 한다.
 */
export const WORD_SCHEMA = "dev-0.4"

/** POST /v1/sessions 의 `client` 필드. 서버 로그에서 세션 출처를 식별한다. */
export const CLIENT_NAME = "mentra-local-miniapp"
