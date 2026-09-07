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

### 버전 정책

`schema_version` 은 메시지마다 붙는 값이고, 서버는 **세 버전을 동시에
받는다**. `SUPPORTED_SCHEMA_VERSIONS = {dev-0.2, dev-0.3, dev-0.4}` 이고
`hello` 에서 이 집합에 없는 값이면 `unsupported_schema_version` 으로
거절한다.

| 버전 | 담당 | 상태 |
|---|---|---|
| `dev-0.2` | `hello` / `frame` / `ping` / `stop` | 유지 |
| `dev-0.3` | `stream_start` / `stream_stop` / WHEP result | 유지 |
| `dev-0.4` | `word_start` / `word_end` / `word_progress` | 이번에 추가 |

dev-0.4 는 **더한 것이지 바꾼 것이 아니다.** 기존 메시지의 필드는 하나도
건드리지 않았으므로 dev-0.3 만 아는 미니앱은 고칠 것 없이 그대로 돈다.
단어 모드를 쓰려는 클라이언트만 `word_*` 메시지에 `dev-0.4` 를 실으면
된다. 한 연결 안에서 메시지마다 버전이 섞여도 된다 - 실제로 미니앱은
`stream_start` 를 dev-0.3 으로, `word_start` 를 dev-0.4 로 보낸다.

서버가 내보내는 응답은 그 요청의 버전을 따라간다. `word_progress` 와
단어 구간 `result` 는 `dev-0.4`, WHEP 스트림 관련 응답은 `dev-0.3` 이다.

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
  "client_message_id": "stream-start-001",
  "stream_id": "cf-stream-123",
  "frame_count": 27,
  "processed_frame_count": 412,
  "processed_fps": 11.8,
  "model": { "loaded": false, "mode": "keypoints_only", "version": "..." },
  "processed_at": "2026-09-05T01:00:02Z"
}
```

- `client_message_id`: 이 스트림을 연 **`stream_start` 의 값**이다.
  `word_start` 의 것이 아니다 — 이 메시지는 WHEP 스트림이 보내는 것이라
  구간과 수명이 다르다. `stream_start` 에 값을 안 보냈다면 `null`
- `model`: `result` 에 실리는 것과 같은 모델 상태
- `frame_count`: **이 구간에** 모인 프레임 수. 단어마다 0부터 다시 센다
- `processed_frame_count`: 스트림 시작 이후 처리한 총 프레임 수. 단조 증가하며
  구간 경계에서 초기화되지 않는다
- `processed_fps`: 서버가 잰 최근 1초의 처리 속도. 첫 메시지에서는 `null`

> **dev-0.3 에서 바뀐 점**: WHEP 경로는 더 이상 프레임마다 `result` 를
> 보내지 않는다. 판정이 `word_end` 한 번뿐이라 그 전에 나가는 `result` 는
> `text` 가 항상 `null` 이어서 앱이 쓸 것이 없었다. 초당 약 10건이던 것이
> 초당 1건의 진행 상황으로 줄어든다.
>
> 그 `result` 에 있던 `sequence_index` 도 함께 사라졌다. 미니앱이 그 값의
> 차이로 처리 속도를 표시했으므로(`background/index.ts` 의
> `trackProcessedFps`), 대신 `processed_frame_count` 를 쓰면 된다 — 같은
> 의미의 단조 증가 카운터다. 계산이 필요 없으면 `processed_fps` 를 그대로
> 써도 된다.
>
> 다만 `word_progress` 는 **구간이 열려 있는 동안에만** 나간다. 단어와 단어
> 사이에는 아무것도 오지 않으므로, 표시를 0 으로 떨어뜨리지 말고 마지막 값을
> 유지하거나 "대기"로 두는 편이 낫다.

### Server -> Client: word result

`word_end` 직후, 또는 8초 자동 종료 시 **한 번** 나간다.

```json
{
  "type": "result",
  "schema_version": "dev-0.4",
  "session_id": "abc-123",
  "client_message_id": "word-end-001",
  "result": {
    "text": "배",
    "confidence": 0.748,
    "is_final": true
  },
  "recognition": {
    "candidate": "배",
    "class_index": 0,
    "confidence": 0.748,
    "margin": 0.592,
    "accepted": true,
    "confidence_threshold": 0.5,
    "margin_threshold": 0.15
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
  "model": { "loaded": true, "mode": "recognition" },
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
- `text`: 임계값을 넘었을 때만 채운다. 못 넘으면 `null` 이다 — 서버가 확신하지
  못한 것이므로 앱은 아무것도 표시하지 않아야 한다. 인식 모델이 아직 안 올라온
  서버도 `null` 을 보내며, 그 경우는 `model.loaded` 가 `false` 다.
- `recognition`: 판정 근거. **거절당했을 때도 무엇이 1위였는지 남는다.**
  임계값을 조정하거나 왜 안 나왔는지 볼 때 쓴다. 모델이 없으면 이 블록이
  통째로 빠진다.

### 임계값

확신도만 보면 안 된다. softmax 는 합이 1 이라 모르는 동작을 넣어도 어딘가에
확률이 몰린다. 실제로 전부 0 인 입력을 넣으면 `"허리" 0.323` 이 나온다.
그래서 **2위와의 격차**를 같이 본다 — 그 경우 격차가 0.081 이라 거절된다.

    확신도 >= 0.5  그리고  2위와의 격차 >= 0.15

근거는 영상 5개(WORD0001, 5시점)를 **서버 코드 그대로** 통과시킨 결과다
(`scripts/verify_word_pipeline.py`). 프레임을 버리는 방식을 세 가지로 나눠
잰다 — MediaPipe 가 30fps 를 못 따라가서 버리는 간격이 실제로는 균일하지
않기 때문이다.

| 드롭 방식 | 간격 | 1위가 정답 | 임계값 통과 | 확신도 평균 | 최저 |
|---|---|---|---|---|---|
| uniform | 100ms 고정 | 5/5 | 5/5 | 0.708 | **0.610** |
| random | 33~633ms | 5/5 | 5/5 | 0.776 | 0.730 |
| stall | 100~600ms | 5/5 | 5/5 | 0.759 | 0.691 |

15가지 경우 전부 1위가 정답이었고 전부 임계값을 통과했다. 최저 확신도
0.610 은 기준 0.5 보다 22% 높다.

불규칙한 쪽(random, stall)이 균일한 쪽보다 오히려 높다. `[1]` 단계의 30fps
복원이 불규칙을 제대로 펴준다는 뜻이다 — 복원 없이 재면 random 에서
4/5 로 떨어진다(아래 "왜 세 단계인가" 참고).

격차는 15가지 중 최저가 0.352 로 기준 0.15 의 두 배가 넘는다. 위에서 계산한
대로 **격차 기준은 이 데이터에서 한 번도 작동하지 않았다.**

시점별로는 이렇다(uniform 기준).

반대로 슬라이딩 윈도우 방식의 애매한 판정(확신도 0.366, 격차 0.083)은
확신도 기준에서 걸러진다.

### Server -> Client: 오류

| code | 언제 | retryable |
|---|---|---|
| `word_already_started` | 이미 열린 구간에 `word_start` 가 또 왔다 | false |
| `word_not_started` | 열린 구간이 없는데 `word_end` 가 왔다 | false |
| `word_too_short` | 구간 프레임이 `WORD_MIN_FRAMES`(기본 8) 미만 | true |

`word_too_short` 로 거절해도 **구간은 닫힌다.** 앱은 다시 `word_start` 부터
보내면 된다.

### 8초 자동 종료와 word_end 가 겹칠 때

자동 종료로 나가는 `result` 의 `client_message_id` 는 그 구간을 연
**`word_start` 의 것**이다. 그 시점에는 `word_end` 요청이 없으므로
`word-end-001` 같은 값이 올 수 없다. `word_start` 에 `client_message_id` 를
안 보냈다면 `null` 이다.

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
| `WORD_SOURCE_FPS` | `30.0` | 되돌릴 격자의 프레임레이트. 원본 영상과 같게 둔다 |
| `RECOGNITION_CONFIDENCE_THRESHOLD` | `0.5` | 이 아래면 단어를 주장하지 않는다 |
| `RECOGNITION_MARGIN_THRESHOLD` | `0.15` | 2위와의 격차가 이 아래면 주장하지 않는다 |
| `RECOGNITION_MODEL_FILENAME` | `model_fold0.keras` | `server/models/` 안의 파일명 |

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

서버가 다른 점은 하나다. **프레임이 불규칙한 간격으로 도착한다.** MediaPipe 가
5.8~13.7fps 로 도는데(#44, 발열에 따라 변한다) 스트림은 30fps 라 처리하지
못한 프레임은 버려지고, 얼마나 버려지는지가 순간마다 다르다. 그대로 넣으면
궤적이 시간축으로 일그러진다 — 촘촘하게 살아남은 구간은 느리게, 성기게
살아남은 구간은 빠르게 움직인 것처럼 보인다.

그래서 앞에 한 단계를 둔다.

```text
도착한 프레임 (T, 411) + 촬영 시각
  → [1] 시간축 30fps 등간격 리샘플       (T30, 411)
  → [2] build_features                  (T30, 420)
  → [3] 인덱스 half-pixel 리샘플 → 60   (60, 420)
```

영상 5개(WORD0001, 5시점)를 여러 프레임레이트로 떨어뜨려 실측했다.
불규칙 드롭(random) 기준 평균 확신도다.

| 구간 프레임 수 | `[1]` 없음 | `[1]` 30fps (채택) | `[1]` 관측 간격 |
|---|---|---|---|
| 41~47 | 0.637 (4/5) | 0.776 | 0.782 |
| 25~32 | 0.675 | 0.689 | 0.727 |
| 16~18 | 0.705 | 0.761 | 0.769 |
| 8~12 | 0.839 | 0.809 | **0.479** |

세 가지를 알 수 있다.

1. **`[1]` 은 필요하다.** 41~47프레임 구간에서 `[1]` 이 없으면 한 번 틀렸다
   (R 시점, "허리" 0.298, 2위와의 격차 0.019 → 임계값 0.15 에 걸려 인식
   실패로 처리됐을 것).
2. **격자를 "관측 간격"으로 잡으면 안 된다.** 프레임 수는 보존되지만 간격이
   들쭉날쭉할 때(100~1567ms) 각 프레임이 원래 시각에서 크게 밀려난다.
   8프레임 구간에서 확신도가 임계값 0.5 아래로 무너졌고, 하필 그 8이
   `WORD_MIN_FRAMES` 의 경계다.
3. **30fps 고정은 12가지 조건**(4개 프레임 수 × 3개 드롭 방식) **전부에서
   0.689~0.809** 였다. 균일한 구간에서 0.006 손해를 보지만 무너지지 않는다.

덤으로 `WORD_MIN_FRAMES = 8` 의 근거가 나왔다. 8프레임에서도 5/5 정답이고
확신도 0.765~0.809 다. MediaPipe 가 5.8fps 까지 떨어져도 정확도 자체는
문제가 없다.

**`[3]` 의 좌표 규칙**은 학습의 `tf.image.resize(bilinear)` 와 같은
half-pixel centers 다. 출력 i번째가 보는 원본 위치는
`(i + 0.5) * T / target - 0.5`. `np.linspace(0, T-1, target)` 로 하면 양 끝을
맞추는 다른 규칙이 되어 T=143 에서 0.69프레임, T=600 에서 4.5프레임 어긋난다.

**`[1]` 에서 신뢰도는 선형 보간하지 않는다.** 411 은 `(x, y, 신뢰도)` 137쌍인데,
검출된 프레임과 미검출 프레임`(0,0,0)` 사이를 선형 보간하면 원점 쪽으로 끌린
가짜 좌표가 생기고 신뢰도까지 섞여서 `CONF_THRESHOLD`(0.05)를 넘는다. 그러면
`feature_service._interpolate_missing` 이 그것을 진짜 검출로 받아들여
보정하지 않는다. 그래서 신뢰도 열만 **양옆 표본의 최솟값**을 쓴다 — 한쪽이라도
미검출이면 그 프레임은 미검출로 남고, 좌표는 `build_features` 가 보간한다.

타임스탬프가 함의하는 프레임레이트가 1~240 범위를 벗어나면(pts 32비트
랩어라운드, 기기 시계 점프, 두 입력 경로의 시각이 섞인 경우) 시각을 버리고
도착 순서를 쓴다. 그때 `resampled_on_time` 이 `false` 로 나간다.
