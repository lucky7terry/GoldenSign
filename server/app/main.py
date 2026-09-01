from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.session_websocket import router as session_websocket_router
from app.api.sessions import router as session_router
from app.config import LOG_LEVEL
from app.logging_config import configure_logging
from app.services.mediapipe_service import preload_mediapipe_service

configure_logging(LOG_LEVEL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 프레임이 아니라 기동 시점에 모델을 올린다. 여기서 실패를 알리지
    # 않으면, 모델 파일이 없을 때 프레임마다 로딩을 재시도하며 조용히
    # 아무것도 인식하지 못하는 상태가 된다.
    preload_mediapipe_service()
    yield


app = FastAPI(
    title="Golden Sign AI Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(session_router)
app.include_router(session_websocket_router)
