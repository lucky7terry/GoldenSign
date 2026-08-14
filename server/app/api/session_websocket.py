import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.config import FRAME_QUEUE_MAX_SIZE, MAX_CONCURRENT_RECOGNITIONS
from app.constants import (
    MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    WEBRTC_SCHEMA_VERSION,
)
from app.error import error_message
from app.schemas.websocket import (
    FrameMessage,
    StreamStartMessage,
    StreamStopMessage,
    WebSocketMessage,
)
from app.services.mediapipe_service import (
    MediaPipeProcessingError,
    decode_base64_image_data,
)
from app.services.model_service import (
    FrameValidationError,
    get_model_health_status,
    recognize_frame_from_image_bytes,
)
from app.services.sequence_service import SequenceSessionClosed, sequence_store
from app.services.session_service import (
    activate_recognition_session,
    validate_recognition_session,
)
from app.services.session_store import stop_session
from app.services.whep_service import whep_pull_service

router = APIRouter()
logger = logging.getLogger(__name__)
_recognition_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RECOGNITIONS)


@dataclass
class FrameRecognitionJob:
    message: FrameMessage
    image_bytes: bytes


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _validate_common_message(message: WebSocketMessage, session_id: str):
    if message.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        return error_message(
            SCHEMA_VERSION,
            session_id,
            "unsupported_schema_version",
            f"Supported schema versions: {', '.join(sorted(SUPPORTED_SCHEMA_VERSIONS))}.",
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
    sequence_generation: int,
    job: FrameRecognitionJob,
):
    frame_message = job.message
    client_message_id = frame_message.client_message_id
    try:
        loop = asyncio.get_running_loop()
        async with _recognition_semaphore:
            result = await loop.run_in_executor(
                None,
                recognize_frame_from_image_bytes,
                job.image_bytes,
                session_id,
                sequence_generation,
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
    except SequenceSessionClosed:
        return
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


async def _recognition_worker(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    sequence_generation: int,
    frame_queue: asyncio.Queue[FrameRecognitionJob],
):
    while True:
        job = await frame_queue.get()
        try:
            await _run_frame_recognition(
                websocket,
                send_lock,
                session_id,
                sequence_generation,
                job,
            )
        finally:
            frame_queue.task_done()


def _decode_frame_image_data(image_data: str) -> bytes:
    return decode_base64_image_data(image_data)


@router.websocket("/v1/sessions/{session_id}/ws")
async def stream_recognition_frames(websocket: WebSocket, session_id: str):
    session = validate_recognition_session(session_id)
    if session is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    send_lock = asyncio.Lock()
    sequence_generation = sequence_store.start_session(session_id)
    frame_queue: asyncio.Queue[FrameRecognitionJob] = asyncio.Queue(
        maxsize=FRAME_QUEUE_MAX_SIZE,
    )
    recognition_worker = asyncio.create_task(
        _recognition_worker(
            websocket,
            send_lock,
            session_id,
            sequence_generation,
            frame_queue,
        )
    )

    async def send_stream_payload(payload: dict):
        await _send_json(websocket, send_lock, payload)

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
                session = validate_recognition_session(session_id)
                if session is None:
                    await websocket.close(code=1008)
                    return

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
                if frame_queue.full():
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            SCHEMA_VERSION,
                            session_id,
                            "frame_queue_full",
                            "Frame queue is full. Please slow down.",
                            client_message_id,
                            retryable=True,
                        )
                    )
                    continue
                try:
                    decoded_image = _decode_frame_image_data(frame_message.image.data)
                except (MediaPipeProcessingError, ValueError):
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

                try:
                    frame_queue.put_nowait(
                        FrameRecognitionJob(
                            message=frame_message,
                            image_bytes=decoded_image,
                        )
                    )
                except asyncio.QueueFull:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            SCHEMA_VERSION,
                            session_id,
                            "frame_queue_full",
                            "Frame queue is full. Please slow down.",
                            client_message_id,
                            retryable=True,
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
            elif message_type == "stream_start":
                try:
                    stream_message = StreamStartMessage.model_validate(raw_message)
                except ValidationError:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            session_id,
                            "invalid_schema",
                            "Stream webrtc_url and stream_id are required.",
                            client_message_id,
                        )
                    )
                    continue
                if stream_message.schema_version != WEBRTC_SCHEMA_VERSION:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            session_id,
                            "unsupported_schema_version",
                            f"stream_start requires {WEBRTC_SCHEMA_VERSION}.",
                            client_message_id,
                        )
                    )
                    continue

                try:
                    await whep_pull_service.start_stream(
                        session_id=session_id,
                        stream_id=stream_message.stream_id,
                        webrtc_url=stream_message.webrtc_url,
                        client_message_id=client_message_id,
                        send_json=send_stream_payload,
                    )
                except ValueError as exc:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            session_id,
                            "invalid_schema",
                            str(exc),
                            client_message_id,
                        )
                    )
                    continue
                except Exception:
                    logger.exception(
                        "Failed to start WHEP stream",
                        extra={
                            "session_id": session_id,
                            "stream_id": stream_message.stream_id,
                        },
                    )
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            session_id,
                            "stream_unavailable",
                            "WHEP stream is unavailable. Please try again.",
                            client_message_id,
                            retryable=True,
                        )
                    )
                    continue
                await _send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "ack",
                        "schema_version": WEBRTC_SCHEMA_VERSION,
                        "session_id": session_id,
                        "client_message_id": client_message_id,
                        "stream_id": stream_message.stream_id,
                        "status": "stream_start_accepted",
                        "received_at": _utc_now_iso(),
                    }
                )
            elif message_type == "stream_stop":
                try:
                    stream_message = StreamStopMessage.model_validate(raw_message)
                except ValidationError:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            session_id,
                            "invalid_schema",
                            "Stream stream_id is required.",
                            client_message_id,
                        )
                    )
                    continue
                if stream_message.schema_version != WEBRTC_SCHEMA_VERSION:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            session_id,
                            "unsupported_schema_version",
                            f"stream_stop requires {WEBRTC_SCHEMA_VERSION}.",
                            client_message_id,
                        )
                    )
                    continue

                await whep_pull_service.stop_stream(
                    session_id,
                    stream_message.stream_id,
                )
                await _send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "ack",
                        "schema_version": WEBRTC_SCHEMA_VERSION,
                        "session_id": session_id,
                        "client_message_id": client_message_id,
                        "stream_id": stream_message.stream_id,
                        "status": "stream_stop_accepted",
                        "received_at": _utc_now_iso(),
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
        await whep_pull_service.stop_session(session_id)
        recognition_worker.cancel()
        await asyncio.gather(recognition_worker, return_exceptions=True)
        sequence_store.clear_session(session_id)
