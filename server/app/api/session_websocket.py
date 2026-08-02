import asyncio
import base64
import binascii
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.constants import SCHEMA_VERSION
from app.error import error_message
from app.schemas.websocket import FrameMessage, WebSocketMessage
from app.services.model_service import (
    FrameValidationError,
    get_model_health_status,
    recognize_frame_from_image_bytes,
)
from app.services.session_service import (
    activate_recognition_session,
    validate_recognition_session,
)
from app.services.session_store import stop_session

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_FRAME_BYTES = 262_144


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _validate_common_message(message: WebSocketMessage, session_id: str):
    if message.schema_version != SCHEMA_VERSION:
        return error_message(
            SCHEMA_VERSION,
            session_id,
            "unsupported_schema_version",
            f"Only {SCHEMA_VERSION} is supported.",
            message.client_message_id,
        )
    if message.session_id != session_id:
        return error_message(
            SCHEMA_VERSION,
            session_id,
            "invalid_schema",
            "Message session_id does not match the WebSocket path.",
            message.client_message_id,
        )
    return None


async def _send_json(websocket: WebSocket, send_lock: asyncio.Lock, payload: dict):
    async with send_lock:
        await websocket.send_json(payload)


async def _run_frame_recognition(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    frame_message: FrameMessage,
    image_bytes: bytes,
):
    client_message_id = frame_message.client_message_id
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            recognize_frame_from_image_bytes,
            image_bytes,
        )
        payload = {
            "type": "result",
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "client_message_id": client_message_id,
            "request_id": frame_message.request_id,
            "frame_index": frame_message.frame_index,
            "captured_at": frame_message.captured_at,
            "result": result,
            "model": get_model_health_status(),
            "processed_at": _utc_now_iso(),
        }
    except FrameValidationError as exc:
        payload = error_message(
            SCHEMA_VERSION,
            session_id,
            "invalid_schema",
            str(exc),
            client_message_id,
            retryable=False,
        )
    except Exception:
        logger.exception(
            "Frame recognition failed",
            extra={
                "session_id": session_id,
                "client_message_id": client_message_id,
            },
        )
        payload = error_message(
            SCHEMA_VERSION,
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


def _decode_frame_image_data(image_data: str) -> bytes:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    return base64.b64decode(image_data, validate=True)


@router.websocket("/v1/sessions/{session_id}/ws")
async def stream_recognition_frames(websocket: WebSocket, session_id: str):
    session = validate_recognition_session(session_id)
    if session is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    send_lock = asyncio.Lock()
    recognition_tasks: set[asyncio.Task] = set()

    try:
        while True:
            try:
                raw_message = await websocket.receive_json()
            except ValueError:
                await _send_json(
                    websocket,
                    send_lock,
                    error_message(
                        SCHEMA_VERSION,
                        session_id,
                        "invalid_json",
                        "Message must be valid JSON.",
                    )
                )
                continue

            if not isinstance(raw_message, dict):
                await _send_json(
                    websocket,
                    send_lock,
                    error_message(
                        SCHEMA_VERSION,
                        session_id,
                        "invalid_schema",
                        "Message must be a JSON object.",
                    )
                )
                continue

            try:
                message = WebSocketMessage.model_validate(raw_message)
            except ValidationError:
                await _send_json(
                    websocket,
                    send_lock,
                    error_message(
                        SCHEMA_VERSION,
                        session_id,
                        "invalid_schema",
                        "Message schema_version, session_id, and type are required.",
                    )
                )
                continue

            validation_error = _validate_common_message(message, session_id)
            if validation_error is not None:
                await _send_json(websocket, send_lock, validation_error)
                continue

            message_type = message.type
            client_message_id = message.client_message_id

            if message_type == "hello":
                session = activate_recognition_session(session_id)
                if session is None:
                    await websocket.close(code=1008)
                    return
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
                try:
                    frame_message = FrameMessage.model_validate(raw_message)
                except ValidationError:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            SCHEMA_VERSION,
                            session_id,
                            "invalid_schema",
                            "Frame image.data is required.",
                            client_message_id,
                        )
                    )
                    continue
                try:
                    decoded_image = _decode_frame_image_data(frame_message.image.data)
                except (binascii.Error, ValueError):
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            SCHEMA_VERSION,
                            session_id,
                            "invalid_schema",
                            "Frame image.data must be valid base64.",
                            client_message_id,
                        )
                    )
                    continue
                if not decoded_image:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            SCHEMA_VERSION,
                            session_id,
                            "invalid_schema",
                            "Frame image.data is required.",
                            client_message_id,
                        )
                    )
                    continue
                if len(decoded_image) > MAX_FRAME_BYTES:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            SCHEMA_VERSION,
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
                        "request_id": frame_message.request_id,
                        "received_at": _utc_now_iso(),
                    }
                )
                task = asyncio.create_task(
                    _run_frame_recognition(
                        websocket,
                        send_lock,
                        session_id,
                        frame_message,
                        decoded_image,
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
                    error_message(
                        SCHEMA_VERSION,
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