/**
 * 미니앱 공용 빌드 타임 설정.
 *
 * background JSContext 가 임포트한다. 값은 번들러가 인라인하므로 런타임 설정
 * 조회도, 환경변수 접근도 없다 — bare JS 엔진에는 `process` 가 없다.
 */

/**
 * AI 서버 base URL.
 *
 * 반드시 Mac 의 LAN IP 여야 한다. background 번들은 *폰* 에서 도는 코드라
 * localhost/127.0.0.1 은 폰 자신을 가리킨다.
 *
 * Mac 의 네트워크가 바뀌면(카페 WiFi, 핫스팟, VPN) 재확인 필요:
 *   ipconfig getifaddr en0
 *
 * 사설 IP 로의 평문 http:// 는 iOS ATS 와 Android cleartext 정책이 각각
 * 차단할 수 있다. ai-client.ts 의 로그가 그 경우와 "서버 미기동" 을 구분한다.
 */
export const AI_HTTP = "http://192.168.35.161:8000"

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

/** POST /v1/sessions 의 `client` 필드. 서버 로그에서 세션 출처를 식별한다. */
export const CLIENT_NAME = "mentra-local-miniapp"
