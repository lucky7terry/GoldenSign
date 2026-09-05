# API Contract — Golden Sign AI Server

Mentra MiniApp(`miniapp/`)과 FastAPI AI 서버(`server/`) 사이의 통신 계약이다.
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
  "client": "mentra-local-miniapp",
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

---

## 6. Word Segment Contract (dev-0.4)

**Purpose**: 사용자가 한 단어의 시작과 끝을 직접 표시하고, 서버는 그 구간만
모아 한 번 판정한다.

지금까지는 프레임이 60장 쌓일 때마다 판정했다(sliding window). 학습은
"한 단어 = 한 구간"을 전제로 했고 증강도 가변 길이 구간을 60프레임으로
늘리는 방식(`crop_resample`)이었으므로, 구간 단위가 학습과 같은 구조다.
서버가 단어의 끝을 추측할 필요도 없어진다.

입력 경로(dev-0.3 WHEP)는 그대로다. 바뀌는 것은 **언제 판정하는가**뿐이다.

### Flow

```text
버튼 누름   -> word_start        -> 서버가 프레임을 모으기 시작
(수어)                              word_progress 를 초당 1회 보냄
버튼 누름   -> word_end          -> 서버가 구간을 모델 입력으로 만들어 판정
                                    result(is_final: true) 1회

WORD_MAX_SECONDS(기본 8초)가 지나면 서버가 알아서 닫고 결과를 보낸다.
```

### Client -> Server: word_start

```json
{
  "type": "word_start",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "client_message_id": "word-start-001"
}
```

응답:

```json
{
  "type": "ack",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "client_message_id": "word-start-001",
  "status": "word_start_accepted",
  "max_seconds": 8.0,
  "received_at": "2026-09-05T01:00:00Z"
}
```

`max_seconds`는 서버 설정값이다. 앱이 자체 타이머를 둘 거라면 이 값을 쓴다.

### Client -> Server: word_end

```json
{
  "type": "word_end",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "client_message_id": "word-end-001"
}
```

### Server -> Client: word_progress

구간이 열려 있는 동안 **초당 1회**만 나간다. 구간 밖에서는 나가지 않는다.

```json
{
  "type": "word_progress",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "stream_id": "cf-stream-123",
  "frame_count": 27,
  "processed_at": "2026-09-05T01:00:02Z"
}
```

> **dev-0.3 에서 바뀐 점**: WHEP 경로는 더 이상 프레임마다 `result` 를
> 보내지 않는다. 판정이 `word_end` 한 번뿐이라 그 전에 나가는 `result` 는
> `text` 가 항상 `null` 이어서 앱이 쓸 것이 없었다. 초당 약 10건이던 것이
> 초당 1건의 진행 상황으로 줄어든다.

### Server -> Client: word result

`word_end` 직후, 또는 8초 자동 종료 시 **한 번** 나간다.

```json
{
  "type": "result",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "client_message_id": "word-end-001",
  "result": {
    "text": null,
    "confidence": 0.0,
    "is_final": true
  },
  "word": {
    "word_index": 1,
    "frame_count": 27,
    "uniform_frame_count": 64,
    "target_frames": 60,
    "feature_dim": 420,
    "span_ms": 2140.0,
    "resampled_on_time": true,
    "dropped_after_cap": 0,
    "close_reason": "client"
  },
  "model": { "loaded": false, "mode": "keypoints_only" },
  "processed_at": "2026-09-05T01:00:02Z"
}
```

- `word_index`: 이 연결에서 몇 번째 단어인지. 1부터.
- `frame_count`: 실제로 **도착한** 프레임 수. MediaPipe 가 12.7fps 로 도는
  동안 30fps 입력의 일부만 살아남는다.
- `uniform_frame_count`: 그것을 30fps 로 되돌린 뒤의 프레임 수. 아래
  "왜 세 단계인가" 참고.
- `span_ms`: 구간의 실제 길이. 시각을 못 얻었으면 `null`.
- `resampled_on_time`: 시간축으로 맞췄으면 `true`, 촬영 시각이 없거나
  믿을 수 없어 도착 순서로 떨어졌으면 `false`. **`false` 가 계속 나오면
  입력 경로에 타임스탬프가 안 붙고 있다는 뜻이니 알려달라.**
- `close_reason`: `"client"`(word_end) 또는 `"timeout"`(8초 자동 종료).
- `text: null`: **인식 모델 연결 전까지 서버는 항상 `null` 을 보낸다.**
  다음 PR 에서 실제 단어가 들어온다. `model.loaded` 로 구분할 수 있다.

### Server -> Client: 오류

| code | 언제 | retryable |
|---|---|---|
| `word_already_started` | 이미 열린 구간에 `word_start` 가 또 왔다 | false |
| `word_not_started` | 열린 구간이 없는데 `word_end` 가 왔다 | false |
| `word_too_short` | 구간 프레임이 `WORD_MIN_FRAMES`(기본 8) 미만 | true |

`word_too_short` 로 거절해도 **구간은 닫힌다.** 앱은 다시 `word_start` 부터
보내면 된다.

### 8초 자동 종료와 word_end 가 겹칠 때

서버가 먼저 닫은 뒤에 `word_end` 가 도착하면, 사용자는 잘못한 것이 없으므로
오류가 아니라 ack 로 답한다. 결과는 이미 나간 뒤다.

```json
{
  "type": "ack",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "client_message_id": "word-end-001",
  "status": "word_already_closed",
  "received_at": "2026-09-05T01:00:10Z"
}
```

### 구간이 버려지는 경우

`stream_stop`, `stop`, 연결 종료가 열린 구간 중에 오면 그 구간은 **결과 없이
버린다.** 영상이 끊겼거나 사용자가 세션을 끝낸 것이므로 판정할 근거가 없다.

### 서버 설정

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `WORD_MAX_SECONDS` | `8.0` | 이 시간이 지나면 서버가 구간을 닫는다 |
| `WORD_MIN_FRAMES` | `8` | 이보다 적으면 `word_too_short` |
| `WORD_TARGET_FRAMES` | `60` | 모델 입력 길이. 학습이 60이다 |
| `WORD_SOURCE_FPS` | `30.0` | 학습 영상의 프레임레이트 |

---

### 왜 세 단계인가

학습 노트북은 이 순서로 돌았다.

```text
30fps 영상 (T, 411)
  -> build_features                       (T, 420)   전체 길이에 대해
  -> crop_resample (50~100% 구간 -> 60)   (60, 420)
```

`robust_scale` 이 영상 전체의 중앙값을 쓰고 `interp_missing` 도 시간축
전체를 보간하므로, `build_features` 가 먼저 전체 길이에 도는 것이 확정이다.

서버는 두 가지가 다르다. **프레임 간격이 불규칙**하고, 그래서 **실효
프레임레이트가 30이 아니다**(측정 12.7fps, #44). 두 번째가 조용히 아프다 —
속도 특징 208개는 `np.diff(pos)` 로 만드는데 이것은 *한 프레임 간격당
변위*이고, 학습에서 그 간격은 1/30초였다. 12.7fps 프레임을 그대로 넣으면
같은 동작인데도 속도가 2.4배로 잡힌다.

그래서 서버는 앞에 한 단계를 더 둔다.

```text
도착한 프레임 (T, 411) + 촬영 시각
  -> [1] 시간축 30fps 등간격 리샘플      (T30, 411)
  -> [2] build_features                  (T30, 420)
  -> [3] 인덱스 half-pixel 리샘플 -> 60  (60, 420)
```

측정값(손목·손가락이 움직이는 2초 구간, 속도 블록 RMS):

| | RMS | 30fps 기준 대비 |
|---|---|---|
| 30fps 원본 | 0.07269 | 1.000 |
| 12.7fps -> 30fps 복원 | 0.07248 | **0.997** |
| 복원 없이 그대로 | 0.16507 | 2.271 |

입력이 이미 30fps 로 균일하면 [1] 은 항등이 되고, 전체가 학습 순서와
**비트 단위로 같아진다**(테스트로 고정).

**[3] 의 좌표 규칙**은 학습의 `tf.image.resize(bilinear)` 와 같은
half-pixel centers 다. 출력 i번째가 보는 원본 위치는
`(i + 0.5) * T / target - 0.5`. `np.linspace(0, T-1, target)` 로 하면 양 끝을
맞추는 다른 규칙이 되어 T=143 에서 0.69프레임, T=600 에서 4.5프레임 어긋난다.

**[1] 에서 신뢰도는 선형 보간하지 않는다.** 411 은 `(x, y, 신뢰도)` 137쌍인데,
검출된 프레임과 미검출 프레임`(0,0,0)` 사이를 선형 보간하면 원점 쪽으로 끌린
가짜 좌표가 생기고 신뢰도까지 섞여서 `CONF_THRESHOLD`(0.05)를 넘는다. 그러면
`feature_service._interpolate_missing` 이 그것을 진짜 검출로 받아들여
보정하지 않는다. 그래서 신뢰도 열만 **양옆 표본의 최솟값**을 쓴다 — 한쪽이라도
미검출이면 그 프레임은 미검출로 남고, 좌표는 `build_features` 가 보간한다.

타임스탬프가 함의하는 프레임레이트가 1~240 범위를 벗어나면(pts 32비트
랩어라운드, 기기 시계 점프, 두 입력 경로의 시각이 섞인 경우) 시각을 버리고
도착 순서를 쓴다. 그때 `resampled_on_time` 이 `false` 로 나간다.
