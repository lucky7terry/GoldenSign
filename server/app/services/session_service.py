from datetime import datetime, timezone
from uuid import uuid4

from app.config import PUBLIC_WS_BASE_URL
from app.constants import SCHEMA_VERSION
from app.schemas.session import SessionCreateRequest, SessionCreateResponse
from app.services.session_store import (
    RecognitionSession,
    activate_session,
    create_session,
    get_session,
)


def _websocket_url(base_url: str, session_id: str):
    if PUBLIC_WS_BASE_URL:
        return f"{PUBLIC_WS_BASE_URL.rstrip('/')}/v1/sessions/{session_id}/ws"

    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.startswith("https://"):
        ws_base_url = "wss://" + normalized_base_url.removeprefix("https://")
    else:
        ws_base_url = "ws://" + normalized_base_url.removeprefix("http://")
    return f"{ws_base_url}/v1/sessions/{session_id}/ws"


def create_recognition_session(
    base_url: str,
    session_create_request: SessionCreateRequest,
):
    session_id = str(uuid4())
    session = create_session(
        session_id=session_id,
        client=session_create_request.client,
        user_id=session_create_request.user_id,
    )

    return SessionCreateResponse(
        session_id=session.session_id,
        status=session.status,
        schema_version=SCHEMA_VERSION,
        ws_url=_websocket_url(base_url, session.session_id),
        expires_at=session.expires_at,
    )


def validate_recognition_session(session_id: str) -> RecognitionSession | None:
    session = get_session(session_id)
    if session is None or session.status == "stopped":
        return None
    if session.expires_at <= datetime.now(timezone.utc):
        return None
    return session


def activate_recognition_session(session_id: str) -> RecognitionSession | None:
    session = validate_recognition_session(session_id)
    if session is None:
        return None
    return activate_session(session_id)
