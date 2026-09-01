# Golden Sign AI Server

## Runtime

- Python 3.12
- FastAPI
- Uvicorn
- MediaPipe

## Setup

```bash
pip install -r requirements.txt
python scripts/download_mediapipe_models.py
```

The model download script prepares:

- `server/models/hand_landmarker.task`
- `server/models/pose_landmarker_lite.task`
- `server/models/face_landmarker.task`

Run this script in every local, CI, and deployment environment before starting
the server. The MediaPipe model files are not committed to the repository.

세 파일이 모두 있어야 한다. 하나라도 없으면 서버는 기동 로그에
`MediaPipe landmarkers unavailable` 을 남기고, `/health` 가 503 과
`status: degraded` 를 돌려주며, 프레임 요청에는 `model_unavailable`
(`retryable: false`) 로 응답한다. 파일을 채운 뒤 서버를 재시작해야 한다.

## Run

```bash
python -m uvicorn app.main:app --reload
```

## Session Store

The server uses the in-memory session store by default.

```bash
SESSION_STORE_BACKEND=memory
```

To store sessions in Redis:

```bash
SESSION_STORE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=3600
```

Frame recognition runs with small global and per-connection limits by default:

```bash
MAX_CONCURRENT_RECOGNITIONS=2
FRAME_QUEUE_MAX_SIZE=30
```

Set a public WebSocket base URL when the server runs behind HTTPS or a reverse
proxy:

```bash
PUBLIC_WS_BASE_URL=wss://api.example.com
```

## Swagger

http://127.0.0.1:8000/docs

## Implemented APIs

- `GET /health`
- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/stop`
- `WS /v1/sessions/{session_id}/ws`
