from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.session_websocket import router as session_websocket_router
from app.api.sessions import router as session_router
from app.config import LOG_LEVEL
from app.logging_config import configure_logging
from app.services.mediapipe_service import preload_mediapipe_service
from app.services.recognition_model import preload_recognition_model

configure_logging(LOG_LEVEL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 프레임이 아니라 기동 시점에 모델을 올린다. 여기서 실패를 알리지
    # 않으면, 모델 파일이 없을 때 프레임마다 로딩을 재시도하며 조용히
    # 아무것도 인식하지 못하는 상태가 된다.
    preload_mediapipe_service()
    # 인식 모델은 3~8초가 걸린다. 첫 단어에서 올리면 그 사용자만 그 시간을
    # 통째로 기다린다. 실패해도 기동은 계속한다 - 좌표 추출은 되고,
    # /health 의 loaded 가 false 로 나가 상태를 구분할 수 있다.
    preload_recognition_model()
    yield


app = FastAPI(
    title="Golden Sign AI Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(session_router)
app.include_router(session_websocket_router)
