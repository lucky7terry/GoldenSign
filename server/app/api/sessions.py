import asyncio
import base64
import binascii
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStopResponse,
)
from app.services.model_service import get_model_health_status, recognize_frame
from app.services.session_store import create_session, get_session, stop_session

router = APIRouter()
SCHEMA_VERSION = "dev-0.2"
MAX_FRAME_BYTES = 262_144


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _websocket_url(request: Request, session_id: str):
    base_url = str(request.base_url).rstrip("/")
    if base_url.startswith("https://"):
        ws_base_url = "wss://" + base_url.removeprefix("https://")
    else:
        ws_base_url = "ws://" + base_url.removeprefix("http://")
    return f"{ws_base_url}/v1/sessions/{session_id}/ws"


def _error_message(
    session_id: str,
    code: str,
    message: str,
    client_message_id: str | None = None,
    retryable: bool = False,
):
    payload = {
        "type": "error",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if client_message_id is not None:
        payload["client_message_id"] = client_message_id
    return payload


def _validate_common_message(message: dict, session_id: str):
    if message.get("schema_version") != SCHEMA_VERSION:
        return _error_message(
            session_id,
            "unsupported_schema_version",
            f"Only {SCHEMA_VERSION} is supported.",
            message.get("client_message_id"),
        )
    if message.get("session_id") != session_id:
        return _error_message(
            session_id,
            "invalid_schema",
            "Message session_id does not match the WebSocket path.",
            message.get("client_message_id"),
        )
    if not isinstance(message.get("type"), str):
        return _error_message(
            session_id,
            "invalid_schema",
            "Message type is required.",
            message.get("client_message_id"),
        )
    return None


async def _send_json(websocket: WebSocket, send_lock: asyncio.Lock, payload: dict):
    async with send_lock:
        await websocket.send_json(payload)


async def _run_frame_recognition(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    frame_message: dict,
):
    client_message_id = frame_message.get("client_message_id")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, recognize_frame, frame_message)
        payload = {
            "type": "result",
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "client_message_id": client_message_id,
            "request_id": frame_message.get("request_id"),
            "frame_index": frame_message.get("frame_index"),
            "result": result,
            "model": get_model_health_status(),
            "processed_at": _utc_now_iso(),
        }
    except Exception:
        payload = _error_message(
            session_id,
            "model_unavailable",
            "Frame recognition failed.",
            client_message_id,
            retryable=True,
        )

    try:
        await _send_json(websocket, send_lock, payload)
    except RuntimeError:
        return


@router.post(
    "/v1/sessions",
    response_model=SessionCreateResponse,
)
def create_recognition_session(
    request: Request,
    session_create_request: SessionCreateRequest,
):
    session_id = str(uuid4())
    create_session(
        session_id=session_id,
        client=session_create_request.client,
        user_id=session_create_request.user_id,
    )

    return SessionCreateResponse(
        session_id=session_id,
        status="created",
        schema_version=SCHEMA_VERSION,
        ws_url=_websocket_url(request, session_id),
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


@router.websocket("/v1/sessions/{session_id}/ws")
async def stream_recognition_frames(websocket: WebSocket, session_id: str):
    session = get_session(session_id)
    if session is None or session.status == "stopped":
        await websocket.close(code=1008)
        return

    await websocket.accept()
    send_lock = asyncio.Lock()
    recognition_tasks: set[asyncio.Task] = set()

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except ValueError:
                await _send_json(
                    websocket,
                    send_lock,
                    _error_message(
                        session_id,
                        "invalid_json",
                        "Message must be valid JSON.",
                    )
                )
                continue

            if not isinstance(message, dict):
                await _send_json(
                    websocket,
                    send_lock,
                    _error_message(
                        session_id,
                        "invalid_schema",
                        "Message must be a JSON object.",
                    )
                )
                continue

            validation_error = _validate_common_message(message, session_id)
            if validation_error is not None:
                await _send_json(websocket, send_lock, validation_error)
                continue

            message_type = message["type"]
            client_message_id = message.get("client_message_id")

            if message_type == "hello":
                session.status = "active"
                await _send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "ready",
                        "schema_version": SCHEMA_VERSION,
                        "session_id": session_id,
                        "server_time": _utc_now_iso(),
                        "model": get_model_health_status(),
                    }
                )
            elif message_type == "frame":
                image = message.get("image")
                image_data = image.get("data") if isinstance(image, dict) else None
                if not isinstance(image_data, str):
                    await _send_json(
                        websocket,
                        send_lock,
                        _error_message(
                            session_id,
                            "invalid_schema",
                            "Frame image.data is required.",
                            client_message_id,
                        )
                    )
                    continue
                try:
                    decoded_image = base64.b64decode(image_data, validate=True)
                except (binascii.Error, ValueError):
                    await _send_json(
                        websocket,
                        send_lock,
                        _error_message(
                            session_id,
                            "invalid_schema",
                            "Frame image.data must be valid base64.",
                            client_message_id,
                        )
                    )
                    continue
                if len(decoded_image) > MAX_FRAME_BYTES:
                    await _send_json(
                        websocket,
                        send_lock,
                        _error_message(
                            session_id,
                            "frame_too_large",
                            f"Decoded image exceeds {MAX_FRAME_BYTES} bytes.",
                            client_message_id,
                        )
                    )
                    continue

                await _send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "ack",
                        "schema_version": SCHEMA_VERSION,
                        "session_id": session_id,
                        "client_message_id": client_message_id,
                        "request_id": message.get("request_id"),
                        "received_at": _utc_now_iso(),
                    }
                )
                task = asyncio.create_task(
                    _run_frame_recognition(
                        websocket,
                        send_lock,
                        session_id,
                        message,
                    )
                )
                recognition_tasks.add(task)
                task.add_done_callback(recognition_tasks.discard)
            elif message_type == "ping":
                await _send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "pong",
                        "schema_version": SCHEMA_VERSION,
                        "session_id": session_id,
                        "client_message_id": client_message_id,
                        "server_time": _utc_now_iso(),
                    }
                )
            elif message_type == "stop":
                stop_session(session_id)
                await websocket.close(code=1000)
                return
            else:
                await _send_json(
                    websocket,
                    send_lock,
                    _error_message(
                        session_id,
                        "unknown_message_type",
                        f"Unknown message type: {message_type}",
                        client_message_id,
                    )
                )
    except WebSocketDisconnect:
        return
    finally:
        tasks = list(recognition_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
