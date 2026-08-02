from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStopResponse,
)
from app.services.session_service import (
    create_recognition_session as create_session_response,
)
from app.services.session_store import stop_session

router = APIRouter()


@router.post(
    "/v1/sessions",
    response_model=SessionCreateResponse,
)
def create_recognition_session(
    request: Request,
    session_create_request: SessionCreateRequest,
):
    return create_session_response(
        base_url=str(request.base_url),
        session_create_request=session_create_request,
    )


@router.post(
    "/v1/sessions/{session_id}/stop",
    response_model=SessionStopResponse,
)
def stop_recognition_session(session_id: str):
    session = stop_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"error": "session_not_found"},
        )

    return SessionStopResponse(
        session_id=session.session_id,
        status=session.status,
    )
