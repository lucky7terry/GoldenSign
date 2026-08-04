from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.session_websocket import router as session_websocket_router
from app.api.sessions import router as session_router

app = FastAPI(
    title="Golden Sign AI Server",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(session_router)
app.include_router(session_websocket_router)
