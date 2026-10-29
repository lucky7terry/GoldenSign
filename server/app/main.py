import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.session_websocket import router as session_websocket_router
from app.api.sessions import router as session_router
from app.api.tts import prewarm_phrases, router as tts_router
from app.config import LOG_LEVEL
from app.logging_config import configure_logging
from app.services.mediapipe_service import preload_mediapipe_service
from app.services.recognition_model import preload_recognition_model
from app.services.tts_service import prewarm

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
    # 음성은 뒤에서 미리 만든다. 단어 50개와 안내 문구를 합성하는 데
    # 네트워크 왕복이 쉰 번 넘게 필요한데, 그동안 세션을 받지 못하면
    # 미니앱은 서버가 죽은 것과 구분하지 못한다. 모델 로딩 뒤에 띄우는
    # 이유는 그쪽이 첫 단어를 실제로 막는 쪽이기 때문이다.
    prewarm_task = asyncio.create_task(prewarm(prewarm_phrases()))
    try:
        yield
    finally:
        prewarm_task.cancel()
        # 종료 중에 올라오는 CancelledError 는 삼킨다.
        await asyncio.gather(prewarm_task, return_exceptions=True)


app = FastAPI(
    title="Golden Sign AI Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(session_router)
app.include_router(session_websocket_router)
app.include_router(tts_router)
