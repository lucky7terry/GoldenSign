# API Contract — Golden Sign AI Server

Mentra MiniApp(`mentra/`)과 FastAPI AI 서버(`server/`) 사이의 통신 계약이다.
서버가 아직 구현되지 않은 상태에서도 이 문서 기준으로 양쪽이 독립적으로 개발을 진행할 수 있도록 만들었다.

## 현재 상태 (Week 1)

- FastAPI 서버(`server/`)는 아직 구현 전이다.
- MiniApp은 `MOCK_AI_SERVER=true`일 때 실제 HTTP 호출 없이 아래 계약과 동일한 형태의 mock 응답을 반환하며 개발·테스트한다.
- 실제 서버가 준비되면 `MOCK_AI_SERVER=false`로 바꾸고 `.env`의 `AI_SERVER_URL`만 실제 서버 주소로 교체하면 연결된다. MiniApp 코드 변경은 필요 없다.
- **Week 1에 MiniApp이 실제로 호출하는 API는 `GET /health`와 `POST /v1/sessions` 두 개뿐이다.** 아래 3, 4번은 계약만 정의해두고 MiniApp 쪽 호출부는 함수 틀만 준비한다.

| Endpoint | Week 1에서 MiniApp이 호출? |
|---|---|
| `GET /health` | 호출함 |
| `POST /v1/sessions` | 호출함 |
| `GET /v1/sessions/{id}` | 호출 안 함 (계약만) |
| `POST /v1/sessions/{id}/stop` | 호출 안 함 (계약만, 함수 틀만 준비) |

---

## 1. GET /health

**용도**: 서버가 떠 있는지, 모델이 로드됐는지 확인.

**응답 예시**: [`fixtures/health-response.example.json`](../fixtures/health-response.example.json)

```json
{
  "status": "ok",
  "api": "ready",
  "model": {
    "loaded": true,
    "mode": "mock",
    "version": "mock-0.1"
  },
  "time": "2026-07-06T00:00:00Z"
}
```

- `model.mode`는 실서버 연결 시 `mock`이 아닌 실제 모드 값(예: `production`)으로 채워질 예정.
- `model.version`도 실제 모델 버전 문자열로 대체될 예정 — MiniApp은 이 값을 그대로 화면에 표시만 하고 파싱/분기하지 않는다.

## 2. POST /v1/sessions

**용도**: 새 인식 세션 생성.

**요청 예시**:

```json
{
  "client": "mentra",
  "user_id": "mentra-user-id"
}
```

**응답 예시**: [`fixtures/session-response.example.json`](../fixtures/session-response.example.json)

```json
{
  "session_id": "abc-123",
  "status": "created",
  "schema_version": "dev-0.1",
  "ws_url": null
}
```

- `ws_url`은 향후 실시간 프레임 전송(WebSocket)을 위한 자리로 예약된 필드. Week 1에서는 `null`로 고정, MiniApp도 이 값을 사용하지 않는다.

## 3. GET /v1/sessions/{id}

**용도**: 세션 상태 조회 (세션이 살아있는지, 현재 상태가 무엇인지).

**상태**: 계약만 정의. Week 1 MiniApp은 호출하지 않는다. 응답 형태는 `POST /v1/sessions` 응답과 동일한 스키마를 따를 예정.

## 4. POST /v1/sessions/{id}/stop

**용도**: 세션 종료.

**상태**: 계약만 정의. Week 1 MiniApp은 호출하지 않지만, `AIServerClient.stopSession()` 함수 시그니처는 미리 준비해둔다.

---

## 서버 구현 담당자에게

- 위 4개 엔드포인트를 이 문서 형태 그대로 구현하면 MiniApp과 바로 연결된다.
- `schema_version`, `ws_url` 같은 필드는 아직 미확정 상태를 표현하는 placeholder다. 실제 구현하면서 값을 채우거나 형태를 바꿔야 하면 이 문서도 같이 업데이트해서 PR로 올려주면 된다.
- MediaPipe 좌표 추출, 문장 시작/종료 판단, 모델 추론 등 내부 로직은 이 계약과 무관하게 자유롭게 구현하면 된다 — 위 요청/응답 형태만 지키면 됨.

---

## 5. WebRTC/WHEP Stream Contract (dev-0.3)

**Purpose**: Define the app-server contract for moving smart-glass video input from Base64 WebSocket frame push to Cloudflare Stream WebRTC/WHEP pull.

Only the input path changes:

```text
Before: MiniApp sends Base64 frame messages over WebSocket.
After: MiniApp sends a WHEP URL, and the AI server pulls frames from that stream.
```

The session creation HTTP API and the existing WebSocket result channel are reused.

### Flow

```text
Smart glasses
-> MentraOS / Cloudflare Stream
-> MiniApp sends webrtc_url + stream_id to AI server
-> AI server pulls frames from the WHEP URL
-> MediaPipe / OpenPose / model pipeline
-> AI server sends result messages over the existing WebSocket
```

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

### Server -> Client: WebRTC result

WebRTC results are stream-based, not one result per client frame request. Therefore `request_id` and `frame_index` are not part of the dev-0.3 WebRTC result contract.

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

- `stream_id`: identifies the Cloudflare/Mentra managed stream.
- `sequence_index`: optional server-generated result sequence number.
- `is_final: false`: interim recognition result.
- `is_final: true`: finalized sentence or signing segment.
- `text: null`: 아직 인식된 단어가 없다는 뜻이다. 소비자는 이 경우 아무것도
  표시하지 않아야 한다. **인식 모델이 연결되기 전까지 서버는 항상 `null` 을
  보낸다** — 자리표시자 문자열을 넣으면 그게 그대로 안경 화면에 뜬다.
  이때 `model.loaded` 도 `false` 이므로 상태를 구분할 수 있다.

### Server -> Client: stream_unavailable

The server owns WHEP connection retry. The MiniApp sends `stream_start` after `onManagedStreamStatus` reports ready. If the server cannot connect after bounded retry, it returns:

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

Actual aiortc/WHEP frame pulling is implemented in a follow-up issue.
