# Week 2 WebSocket Contract

이 문서는 Mentra MiniApp과 FastAPI AI 서버가 Week 2에서 맞춰야 하는
실시간 프레임 전송 규격이다.

## 목표

- MiniApp은 세션 생성 후 WebSocket으로 카메라 프레임을 서버에 보낸다.
- AI 서버는 프레임 수신 여부와 추론 결과를 같은 WebSocket으로 돌려준다.
- 첫 구현은 디버깅이 쉬운 JSON 메시지로 고정한다.
- 바이너리 프레임 전송은 성능 최적화가 필요해질 때 `schema_version`을 올려 도입한다.

## 연결 흐름

1. MiniApp이 `POST /v1/sessions`를 호출한다.
2. 서버는 `session_id`와 `ws_url`을 응답한다.
3. MiniApp은 `ws_url`로 WebSocket 연결을 연다.
4. MiniApp은 연결 직후 `hello` 메시지를 보낸다.
5. 서버는 `ready` 메시지를 응답한다.
6. MiniApp은 `frame` 메시지를 반복 전송한다.
7. 서버는 각 프레임에 대해 `ack`, `result`, 또는 `error`를 응답한다.
8. 종료 시 MiniApp은 `stop` 메시지를 보내고 기존 `POST /v1/sessions/{id}/stop`도 호출한다.

## Endpoint

```text
WS /v1/sessions/{session_id}/ws
```

`POST /v1/sessions` 응답의 `ws_url`은 아래 형태를 권장한다.

```json
{
  "session_id": "abc-123",
  "status": "created",
  "schema_version": "dev-0.2",
  "ws_url": "ws://127.0.0.1:8000/v1/sessions/abc-123/ws"
}
```

운영 환경에서 HTTPS를 쓰면 `ws_url`은 `wss://`가 된다.

## 공통 메시지 규칙

- 모든 WebSocket 메시지는 JSON text frame이다.
- 모든 메시지는 `type`, `schema_version`, `session_id`를 가진다.
- 시간 값은 ISO 8601 문자열을 사용한다.
- 이미지 데이터는 첫 버전에서 base64 문자열로 보낸다.
- 서버는 알 수 없는 `type`을 받으면 연결을 끊지 않고 `error` 메시지를 보낸다.

공통 필드:

```ts
interface BaseMessage {
  type: string;
  schema_version: "dev-0.2";
  session_id: string;
  client_message_id?: string;
  sent_at?: string;
}
```

## Client -> Server

### hello

연결 직후 한 번 보낸다.

```json
{
  "type": "hello",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "msg-001",
  "client": "mentra-local-miniapp",
  "user_id": "mentra-user-id",
  "capabilities": {
    "frame_encoding": ["jpeg_base64"],
    "max_frame_bytes": 262144
  }
}
```

### frame

카메라 프레임 또는 사진 한 장을 보낸다.

```json
{
  "type": "frame",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "frame-000001",
  "request_id": "req-000001",
  "frame_index": 1,
  "captured_at": "2026-07-15T12:00:00.000Z",
  "image": {
    "encoding": "jpeg_base64",
    "mime_type": "image/jpeg",
    "width": 640,
    "height": 480,
    "data": "/9j/4AAQSkZJRgABAQ..."
  }
}
```

권장 제한:

- `image.data` 디코딩 후 최대 크기: 256 KB
- 권장 전송 주기: 초기 구현은 1초에 1장 이하
- 서버가 느릴 경우 MiniApp은 이전 프레임 결과를 기다리지 않고 새 프레임을 무한히 쌓지 않는다.

### ping

연결 유지 및 지연 시간 확인용이다.

```json
{
  "type": "ping",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "ping-001",
  "sent_at": "2026-07-15T12:00:01.000Z"
}
```

### stop

WebSocket 스트림 종료 의도를 알린다.

```json
{
  "type": "stop",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "stop-001",
  "reason": "app_stopped"
}
```

## Server -> Client

### ready

`hello`를 정상 처리한 뒤 응답한다.

```json
{
  "type": "ready",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "server_time": "2026-07-15T12:00:00.100Z",
  "model": {
    "loaded": true,
    "mode": "mock",
    "version": "mock-0.1"
  }
}
```

### ack

프레임을 수신했고 처리 큐에 넣었음을 알린다.

```json
{
  "type": "ack",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "frame-000001",
  "request_id": "req-000001",
  "received_at": "2026-07-15T12:00:00.250Z"
}
```

### result

모델 추론 결과를 돌려준다. 아직 실제 모델이 없으면 `mode: "mock"`으로 응답한다.

```json
{
  "type": "result",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "frame-000001",
  "request_id": "req-000001",
  "frame_index": 1,
  "captured_at": "2026-07-15T12:00:00.000Z",
  "result": {
    "text": "안녕하세요",
    "confidence": 0.92,
    "is_final": false
  },
  "model": {
    "mode": "mock",
    "version": "mock-0.1"
  },
  "processed_at": "2026-07-15T12:00:00.500Z"
}
```

### pong

`ping`에 대한 응답이다.

```json
{
  "type": "pong",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "ping-001",
  "server_time": "2026-07-15T12:00:01.100Z"
}
```

### error

검증 실패나 처리 실패를 전달한다.

```json
{
  "type": "error",
  "schema_version": "dev-0.2",
  "session_id": "abc-123",
  "client_message_id": "frame-000001",
  "code": "frame_too_large",
  "message": "Decoded image exceeds 262144 bytes.",
  "retryable": false
}
```

권장 에러 코드:

| code | 의미 |
| --- | --- |
| `invalid_json` | JSON 파싱 실패 |
| `invalid_schema` | 필수 필드 누락 또는 타입 오류 |
| `unsupported_schema_version` | 지원하지 않는 `schema_version` |
| `unknown_message_type` | 알 수 없는 `type` |
| `session_not_found` | 존재하지 않는 세션 |
| `frame_too_large` | 허용 크기 초과 |
| `model_unavailable` | 모델 처리 불가 |
| `internal_error` | 서버 내부 오류 |

## Week 2 구현 범위

서버:

- `POST /v1/sessions` 응답의 `schema_version`을 `dev-0.2`로 올린다.
- `ws_url`을 `/v1/sessions/{session_id}/ws`로 채운다.
- FastAPI WebSocket endpoint를 추가한다.
- `hello`, `frame`, `ping`, `stop`을 처리한다.
- 첫 `frame` 결과는 mock inference로 돌려준다.

MiniApp:

- `createSession()` 응답의 `ws_url`이 있으면 WebSocket 연결을 연다.
- 연결 직후 `hello`를 보낸다.
- 기존 사진 버퍼를 `jpeg_base64`로 인코딩해 `frame`으로 보낸다.
- `ready`, `ack`, `result`, `error`를 로그와 HUD 상태에 반영한다.
- `onStop`에서 `stop` 메시지와 REST stop 요청을 정리한다.

## 합의가 필요한 항목

- 태리 MiniApp이 실제로 보낼 이미지 크기와 주기
- 사진 단위 전송인지 연속 프레임 스트림인지
- 모델 결과가 한글 문장 하나인지, 토큰/글자 단위 누적인지
- 인증 토큰이 필요한지 여부

---

## WebRTC/WHEP Control Messages (dev-0.3)

This section extends the existing WebSocket channel for WebRTC/WHEP input. The WebSocket connection is still created from `POST /v1/sessions`, but video frames are no longer pushed as Base64 JSON messages. Instead, the client sends a managed stream URL and the AI server pulls frames from WHEP in a follow-up implementation.

### Client -> Server: stream_start

```json
{
  "type": "stream_start",
  "schema_version": "dev-0.3",
  "session_id": "abc-123",
  "client_message_id": "stream-start-001",
  "webrtc_url": "https://example.cloudflarestream.com/webRTC/play",
  "stream_id": "cf-stream-123"
}
```

Required fields:

| field | meaning |
| --- | --- |
| `type` | Must be `stream_start`. |
| `schema_version` | Must be `dev-0.3`. |
| `session_id` | Existing recognition session ID. |
| `webrtc_url` | WHEP playback URL returned by the managed stream provider. |
| `stream_id` | Managed stream identifier returned with the WebRTC URL. |

Current #22 behavior: the server validates the message and returns an ack. Actual aiortc/WHEP connection is implemented in a follow-up issue.

```json
{
  "type": "ack",
  "schema_version": "dev-0.3",
  "session_id": "abc-123",
  "client_message_id": "stream-start-001",
  "stream_id": "cf-stream-123",
  "status": "stream_start_accepted",
  "received_at": "2026-08-07T12:00:00Z"
}
```

### Client -> Server: stream_stop

```json
{
  "type": "stream_stop",
  "schema_version": "dev-0.3",
  "session_id": "abc-123",
  "client_message_id": "stream-stop-001",
  "stream_id": "cf-stream-123"
}
```

Current #22 behavior: the server validates the message and returns an ack. Actual WHEP stream shutdown is implemented in a follow-up issue.

```json
{
  "type": "ack",
  "schema_version": "dev-0.3",
  "session_id": "abc-123",
  "client_message_id": "stream-stop-001",
  "stream_id": "cf-stream-123",
  "status": "stream_stop_accepted",
  "received_at": "2026-08-07T12:00:05Z"
}
```

### Server -> Client: WebRTC result

WebRTC results are stream-based. A result can be produced from many pulled frames, so dev-0.3 WebRTC result messages do not include `request_id` or `frame_index`.

```json
{
  "type": "result",
  "schema_version": "dev-0.3",
  "session_id": "abc-123",
  "stream_id": "cf-stream-123",
  "sequence_index": 12,
  "result": {
    "text": "안녕하세요",
    "confidence": 0.9,
    "is_final": false
  },
  "processed_at": "2026-08-07T12:00:00Z"
}
```

- `stream_id` is required for stream-level tracking.
- `sequence_index` is an optional server-generated result sequence number.
- `is_final: false` means an interim recognition result.
- `is_final: true` means a finalized sentence or signing segment.

### Server -> Client: stream_unavailable

The server owns WHEP connection retry. The app should send `stream_start` after `onManagedStreamStatus` reports ready. If the server still cannot connect after bounded retry, it returns this error:

```json
{
  "type": "error",
  "schema_version": "dev-0.3",
  "session_id": "abc-123",
  "client_message_id": "stream-start-001",
  "code": "stream_unavailable",
  "message": "WebRTC stream is not available yet.",
  "retryable": true
}
```
