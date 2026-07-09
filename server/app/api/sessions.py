from uuid import uuid4

from fastapi import APIRouter

from app.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
)

router = APIRouter()


@router.post(
    "/v1/sessions",
    response_model=SessionCreateResponse,
)
def create_recognition_session(
    session_create_request: SessionCreateRequest,
):
    return SessionCreateResponse(
        session_id=str(uuid4()),
        status="created",
        schema_version="dev-0.1",
        ws_url=None,
    )