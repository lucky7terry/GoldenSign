import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.config import (
    FRAME_QUEUE_MAX_SIZE,
    MAX_CONCURRENT_RECOGNITIONS,
    WORD_MAX_SECONDS,
    WS_IDLE_TIMEOUT_SECONDS,
)
from app.constants import (
    MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    WEBRTC_SCHEMA_VERSION,
    WORD_SCHEMA_VERSION,
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
    MediaPipeUnavailableError,
    decode_base64_image_data,
)
from app.services.model_service import (
    FrameValidationError,
    get_model_health_status,
    public_result,
    recognize_frame_from_image_bytes,
)
from app.services.recognition_model import RecognitionModelUnavailableError
from app.services.recognition_service import (
    WordRecognition,
    recognize_word_segment,
)
from app.services.word_segment_service import (
    WordAlreadyStarted,
    WordNotStarted,
    WordSessionClosed,
    WordTooShort,
    word_store,
)
from app.services.session_service import (
    activate_recognition_session,
    validate_recognition_session,
)
from app.services.session_store import stop_session
from app.services.whep_service import whep_pull_service

router = APIRouter()
logger = logging.getLogger(__name__)
_recognition_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RECOGNITIONS)

# 유휴 여부를 얼마나 자주 확인할지. 타임아웃 자체보다 짧아야 한다.
_IDLE_POLL_SECONDS = 5.0


class ConnectionActivity:
    """이 연결에 마지막으로 무언가 오간 시각.

    WHEP 스트리밍 중에는 클라이언트가 WebSocket 으로 아무것도 보내지 않는다.
    영상은 서버가 Cloudflare 에서 직접 당겨오고, 서버는 결과만 내보낸다.
    그래서 유휴 판정을 수신만으로 하면 정상 동작 중인 스트림을 끊게 된다.
    송신도 활동으로 친다.

    글래스 쪽 연결이 실제로 죽으면 WHEP 프레임이 먼저 끊기고
    (WHEP_FRAME_IDLE_TIMEOUT_SECONDS), 그러면 보낼 결과도 없어지므로
    이 타이머가 다시 의미를 갖는다.
    """

    def __init__(self) -> None:
        self._last = monotonic()

    def touch(self) -> None:
        self._last = monotonic()

    def idle_seconds(self) -> float:
        return monotonic() - self._last


@dataclass
class FrameRecognitionJob:
    message: FrameMessage
    image_bytes: bytes
    captured_at_ms: float | None


def _word_result(recognition: WordRecognition | None) -> dict:
    """클라이언트로 나갈 result. 모델이 없으면 단어를 주장하지 않는다."""
    if recognition is None:
        return {"text": None, "confidence": 0.0, "is_final": True}
    return recognition.public()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _parse_captured_at_ms(captured_at: str | None) -> float | None:
    """클라이언트가 준 촬영 시각을 ms 로. 못 읽으면 None.

    단어 구간 안의 상대 간격만 쓰므로 클라이언트 시계가 서버와 어긋나
    있어도 상관없다. 한 구간 안에서 같은 시계면 된다.
    """
    # WebSocketMessage 가 extra="allow" 라 스키마를 통과한 값이 항상
    # 문자열이라는 보장이 없다. 파서에서 AttributeError 가 나면 그 프레임
    # 전체가 model_unavailable 로 떨어진다.
    if not isinstance(captured_at, str) or not captured_at:
        return None
    try:
        parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    try:
        return parsed.timestamp() * 1000.0
    except (OSError, OverflowError, ValueError):
        return None


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


async def _send_json(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    payload: dict,
    activity: ConnectionActivity | None = None,
):
    async with send_lock:
        await websocket.send_json(payload)
    if activity is not None:
        activity.touch()


async def _run_frame_recognition(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    word_generation: int,
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
                word_generation,
                job.captured_at_ms,
            )
        payload = {
            "type": "result",
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "client_message_id": client_message_id,
            "request_id": frame_message.request_id,
            "frame_index": frame_message.frame_index,
            "captured_at": frame_message.captured_at,
            "result": public_result(result),
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
    except WordSessionClosed:
        return
    except MediaPipeUnavailableError as exc:
        # 모델 파일이 없는 상태다. 다시 보내도 결과가 같으므로 재시도시키지 않는다.
        logger.error(
            "Keypoint extraction unavailable",
            extra={"session_id": session_id, "error": str(exc)},
        )
        payload = error_message(
            SCHEMA_VERSION,
            session_id,
            "model_unavailable",
            "Keypoint extraction is unavailable on this server.",
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


async def _recognition_worker(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    session_id: str,
    word_generation: int,
    frame_queue: asyncio.Queue[FrameRecognitionJob],
):
    while True:
        job = await frame_queue.get()
        try:
            await _run_frame_recognition(
                websocket,
                send_lock,
                session_id,
                word_generation,
                job,
            )
        finally:
            frame_queue.task_done()


def _close_word_and_recognize(session_id: str, word_generation: int):
    """구간을 닫고 단어까지 판정한다. executor 에서 돈다.

    end_word 안의 특징 변환(30fps 복원 -> build_features -> 60프레임)과
    모델 추론을 합치면 단어당 20ms 쯤이다. 이벤트 루프에서 하면 그동안
    다른 연결의 프레임 처리와 ping 응답이 전부 멈춘다.

    인식 모델이 없으면 recognition 이 None 이다 - 구간은 정상적으로
    닫히고 좌표 수집도 계속된다. 단어만 주장하지 않는다.
    """
    segment = word_store.end_word(session_id, word_generation)
    try:
        recognition = recognize_word_segment(segment.sequence)
    except RecognitionModelUnavailableError as exc:
        logger.warning(
            "Recognition model unavailable; returning no word",
            extra={"session_id": session_id, "error": str(exc)},
        )
        recognition = None
    return segment, recognition


def _decode_frame_image_data(image_data: str) -> bytes:
    return decode_base64_image_data(image_data)


@router.websocket("/v1/sessions/{session_id}/ws")
async def stream_recognition_frames(websocket: WebSocket, session_id: str):
    session = validate_recognition_session(session_id)
    if session is None:
        logger.info(
            "WebSocket rejected",
            extra={"session_id": session_id, "reason": "session_invalid"},
        )
        await websocket.close(code=1008)
        return

    await websocket.accept()
    connected_at = monotonic()
    activity = ConnectionActivity()
    close_reason = "unknown"
    logger.info(
        "WebSocket connected",
        extra={
            "session_id": session_id,
            "client": session.client,
            "user_id": session.user_id,
        },
    )
    send_lock = asyncio.Lock()
    word_generation = word_store.start_session(session_id)
    # 이 연결이 연 스트림. 종료할 때 이것만 닫는다 - session_id 만으로
    # 닫으면 재접속한 다른 연결의 스트림을 끊는다.
    started_stream_id: str | None = None
    frame_queue: asyncio.Queue[FrameRecognitionJob] = asyncio.Queue(
        maxsize=FRAME_QUEUE_MAX_SIZE,
    )
    recognition_worker = asyncio.create_task(
        _recognition_worker(
            websocket,
            send_lock,
            session_id,
            word_generation,
            frame_queue,
        )
    )

    async def send_stream_payload(payload: dict):
        await _send_json(websocket, send_lock, payload, activity)

    def touch_activity() -> None:
        """WHEP 프레임을 하나 처리했다.

        단어 모드로 바뀌면서 프레임마다 나가던 result 가 없어졌다. 그래서
        스트리밍 중에도 서버 송신이 거의 없고, 유휴 판정이 클라이언트 ping
        에만 의존하게 됐다. 프레임을 처리하고 있다는 사실 자체를 활동으로
        쳐서 서버가 스스로 판단할 수 있게 한다.
        """
        activity.touch()

    # 열린 단어 구간을 WORD_MAX_SECONDS 뒤에 서버가 알아서 닫는 타이머.
    # 사용자가 끝 표시를 잊어도 결과는 나온다.
    word_timer: asyncio.Task | None = None
    # 자동 종료 타이머가 이미 결과를 만들어 보내는 중인지. 이 상태에서
    # 취소하면 결과가 중간에 잘려서 사용자는 아무것도 못 받는다.
    word_timer_finalizing = False
    # 자동 종료로 이미 닫힌 뒤에 word_end 가 도착했는지. 사용자는 잘못한
    # 것이 없으므로 오류가 아니라 ack 로 답한다.
    auto_closed_pending = False

    def cancel_word_timer():
        """아직 대기 중인 자동 종료 타이머만 취소한다.

        이미 결과를 만들어 보내는 중이면 건드리지 않는다. 그 시점의 취소는
        구간을 닫아놓고 결과는 안 보낸 상태로 끝나서, 사용자가 단어를
        통째로 잃는다. 끝난 뒤 정리는 바깥 finally 가 한다.
        """
        nonlocal word_timer
        if word_timer is not None and not word_timer_finalizing:
            word_timer.cancel()
            word_timer = None

    async def finalize_word(
        finalize_client_message_id: str | None,
        close_reason: str,
    ) -> str:
        """구간을 닫고 결과를 보낸다. "closed" | "too_short" | "not_open"."""
        loop = asyncio.get_running_loop()
        try:
            # 프레임 경로와 같은 한도를 쓴다. 구간 마감은 특징 변환과 추론이
            # 붙어 있어 프레임 한 장보다 무겁고, 기본 executor 를 MediaPipe 와
            # 나눠 쓴다. 여기만 무제한이면 그 한도가 의미를 잃는다.
            async with _recognition_semaphore:
                segment, recognition = await loop.run_in_executor(
                    None,
                    _close_word_and_recognize,
                    session_id,
                    word_generation,
                )
        except (WordNotStarted, WordSessionClosed):
            return "not_open"
        except WordTooShort as exc:
            await _send_json(
                websocket,
                send_lock,
                error_message(
                    WORD_SCHEMA_VERSION,
                    session_id,
                    "word_too_short",
                    str(exc),
                    finalize_client_message_id,
                    retryable=True,
                ),
                activity,
            )
            return "too_short"
        except Exception:
            # 여기서 새면 예외가 수신 루프를 뚫고 나가 연결이 끊긴다.
            # 클라이언트는 result 도 error 도 못 받고, 로그에는 session_id
            # 없는 ASGI 트레이스백만 남는다. 라벨 파일 문제, 텐서플로 런타임
            # 오류, 입력 shape 불일치가 전부 이 경로다.
            logger.exception(
                "Word finalization failed",
                extra={"session_id": session_id},
            )
            await _send_json(
                websocket,
                send_lock,
                error_message(
                    WORD_SCHEMA_VERSION,
                    session_id,
                    "model_unavailable",
                    "Word recognition failed.",
                    finalize_client_message_id,
                    retryable=True,
                ),
                activity,
            )
            return "closed"

        metadata = segment.metadata()
        metadata["close_reason"] = close_reason


        payload = {
            "type": "result",
            "schema_version": WORD_SCHEMA_VERSION,
            "session_id": session_id,
            "client_message_id": finalize_client_message_id,
            "result": _word_result(recognition),
            "word": metadata,
            "model": get_model_health_status(),
            "processed_at": _utc_now_iso(),
        }
        if recognition is not None:
            # 거절당했을 때 무엇이 1위였는지 남긴다. 임계값을 조정할 때
            # 이 값이 없으면 왜 안 나왔는지 알 수 없다.
            payload["recognition"] = recognition.detail()

        logger.info(
            "Word recognized" if recognition is not None else "Word closed",
            extra={
                "session_id": session_id,
                "word_index": segment.word_index,
                "close_reason": close_reason,
                **(recognition.detail() if recognition is not None else {}),
            },
        )
        await _send_json(websocket, send_lock, payload, activity)
        return "closed"

    async def auto_close_word(timer_client_message_id: str | None):
        nonlocal auto_closed_pending, word_timer, word_timer_finalizing
        try:
            await asyncio.sleep(WORD_MAX_SECONDS)
        except asyncio.CancelledError:
            # 삼키지 않는다. 바깥 gather 가 취소로 인지해야 한다.
            raise
        # word_timer 를 여기서 None 으로 만들면 안 된다. 아래 send 가
        # await 라서, 그 사이에 연결이 끊기면 finally 가 이 태스크를
        # 놓치고 고아로 남는다(websocket 과 send_lock 을 붙잡은 채로).
        # 끝난 태스크를 cancel 하는 것은 무해하므로 참조를 그대로 둔다.
        # 결과를 보내기 "전에" 세운다. 여기서부터 finalize_word 안의 send 는
        # await 라서, 바로 뒤에 도착한 word_end 의 cancel 이 중간을 자를 수
        # 있다. 그때 이 값이 아직 False 면 사용자는 잘못한 게 없는데
        # word_not_started 오류를 받는다.
        auto_closed_pending = True
        word_timer_finalizing = True
        logger.info(
            "Word segment auto-closed",
            extra={
                "session_id": session_id,
                "max_seconds": WORD_MAX_SECONDS,
            },
        )
        try:
            await finalize_word(timer_client_message_id, "timeout")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 아래에서 남기고 삼킨다
            # 소켓이 이미 죽었을 때 올라오는 예외는 ASGI 서버 구현마다
            # 다르다(RuntimeError / ConnectionClosed / ...). 좁게 잡으면
            # "Task exception was never retrieved" 로 세션 정보 없이
            # 튀어나온다. 넓게 잡고 여기서 남긴다.
            logger.info(
                "Could not deliver auto-closed word result",
                extra={"session_id": session_id},
                exc_info=True,
            )
        finally:
            # 반드시 내려야 한다. 이 값이 True 로 남으면 cancel_word_timer 가
            # 영구히 무력화되고, 그 뒤 모든 word_start 가 이전 타이머를
            # 살려둔 채 새 타이머를 만든다. 살아남은 타이머는 8초 뒤에
            # "그때 열려 있던 다른 단어"를 잘라버린다.
            word_timer_finalizing = False

    try:
        while True:
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=_IDLE_POLL_SECONDS,
                )
            except asyncio.TimeoutError:
                # 수신이 없어도 결과를 내보내고 있으면 살아있는 연결이다.
                if activity.idle_seconds() < WS_IDLE_TIMEOUT_SECONDS:
                    continue
                close_reason = "idle_timeout"
                logger.warning(
                    "WebSocket idle timeout; closing",
                    extra={
                        "session_id": session_id,
                        "idle_seconds": round(activity.idle_seconds(), 1),
                        "idle_timeout_seconds": WS_IDLE_TIMEOUT_SECONDS,
                    },
                )
                await websocket.close(code=1001)
                return
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

            activity.touch()

            validation_error = _validate_common_message(message, session_id)
            if validation_error is not None:
                await _send_json(websocket, send_lock, validation_error)
                continue

            message_type = message.type
            client_message_id = message.client_message_id

            if message_type == "hello":
                session = activate_recognition_session(session_id)
                if session is None:
                    close_reason = "session_invalid"
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
                    close_reason = "session_invalid"
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
                            captured_at_ms=_parse_captured_at_ms(
                                frame_message.captured_at
                            ),
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
                session = validate_recognition_session(session_id)
                if session is None:
                    close_reason = "session_invalid"
                    await websocket.close(code=1008)
                    return
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
                        on_frame=touch_activity,
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
                started_stream_id = stream_message.stream_id
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

                stopped = await whep_pull_service.stop_stream(
                    session_id,
                    stream_message.stream_id,
                )
                if stopped:
                    # 영상이 끊겼으므로 모으던 구간은 의미가 없다. 결과를
                    # 내지 않고 버린다.
                    cancel_word_timer()
                    if word_store.abort_word(session_id, word_generation):
                        logger.info(
                            "Discarded open word segment on stream_stop",
                            extra={
                                "session_id": session_id,
                                "stream_id": stream_message.stream_id,
                            },
                        )
                else:
                    # 지금 도는 스트림이 아니다. stop_stream 이 소유권 검사에
                    # 걸러낸 경우(늦게 도착한 이전 스트림의 stop)이거나 애초에
                    # 스트림이 없는 경우다. 여기서 구간을 버리면, 같은 연결에서
                    # stream_start(A) -> stream_start(B) -> 늦은 stop(A) 순서일
                    # 때 멀쩡히 도는 B 의 단어가 사라진다.
                    logger.info(
                        "Ignored stream_stop for a stream that is not running",
                        extra={
                            "session_id": session_id,
                            "stream_id": stream_message.stream_id,
                        },
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
            elif message_type == "word_start":
                if message.schema_version != WORD_SCHEMA_VERSION:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WORD_SCHEMA_VERSION,
                            session_id,
                            "unsupported_schema_version",
                            f"word_start requires {WORD_SCHEMA_VERSION}.",
                            client_message_id,
                        )
                    )
                    continue
                session = validate_recognition_session(session_id)
                if session is None:
                    close_reason = "session_invalid"
                    await websocket.close(code=1008)
                    return

                try:
                    word_store.start_word(session_id, word_generation)
                except WordAlreadyStarted:
                    # 이미 열린 구간을 말없이 버리면 앞부분 프레임이
                    # 사라진다. 앱이 상태를 정리하도록 알린다.
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WORD_SCHEMA_VERSION,
                            session_id,
                            "word_already_started",
                            "A word segment is already open. Send word_end first.",
                            client_message_id,
                            retryable=False,
                        )
                    )
                    continue
                except WordSessionClosed:
                    close_reason = "session_invalid"
                    await websocket.close(code=1008)
                    return

                auto_closed_pending = False
                cancel_word_timer()
                word_timer = asyncio.create_task(
                    auto_close_word(client_message_id)
                )
                await _send_json(
                    websocket,
                    send_lock,
                    {
                        "type": "ack",
                        "schema_version": WORD_SCHEMA_VERSION,
                        "session_id": session_id,
                        "client_message_id": client_message_id,
                        "status": "word_start_accepted",
                        "max_seconds": WORD_MAX_SECONDS,
                        "received_at": _utc_now_iso(),
                    }
                )
            elif message_type == "word_end":
                if message.schema_version != WORD_SCHEMA_VERSION:
                    await _send_json(
                        websocket,
                        send_lock,
                        error_message(
                            WORD_SCHEMA_VERSION,
                            session_id,
                            "unsupported_schema_version",
                            f"word_end requires {WORD_SCHEMA_VERSION}.",
                            client_message_id,
                        )
                    )
                    continue

                cancel_word_timer()
                outcome = await finalize_word(client_message_id, "client")
                if outcome == "not_open":
                    if auto_closed_pending:
                        # 8초가 먼저 지나서 서버가 이미 닫았다. 결과는
                        # 이미 나갔으므로 오류가 아니다.
                        auto_closed_pending = False
                        await _send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "ack",
                                "schema_version": WORD_SCHEMA_VERSION,
                                "session_id": session_id,
                                "client_message_id": client_message_id,
                                "status": "word_already_closed",
                                "received_at": _utc_now_iso(),
                            }
                        )
                    else:
                        await _send_json(
                            websocket,
                            send_lock,
                            error_message(
                                WORD_SCHEMA_VERSION,
                                session_id,
                                "word_not_started",
                                "No word segment is open. Send word_start first.",
                                client_message_id,
                                retryable=False,
                            )
                        )
            elif message_type == "stop":
                close_reason = "client_stop"
                cancel_word_timer()
                word_store.abort_word(session_id, word_generation)
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
        close_reason = "client_disconnect"
        return
    finally:
        # 취소될 수 없는 정리를 먼저 한다. 아래 await 에서 CancelledError 를
        # 맞으면(서버 종료 등) 그 뒤 줄이 안 돌기 때문이다.
        recognition_worker.cancel()
        pending_word_timer = word_timer
        cancel_word_timer()
        # 같은 session_id 로 이미 다른 연결이 이어받았다면 그쪽 상태를
        # 건드리면 안 된다. 늦게 죽는 연결이 살아있는 연결을 지우는 것이
        # 재접속 직후 세션이 원인 없이 죽는 흔한 경로다.
        word_store.clear_session(session_id, word_generation)
        if started_stream_id is not None:
            await whep_pull_service.stop_stream(session_id, started_stream_id)
        await asyncio.gather(recognition_worker, return_exceptions=True)
        if pending_word_timer is not None:
            await asyncio.gather(pending_word_timer, return_exceptions=True)
        logger.info(
            "WebSocket closed",
            extra={
                "session_id": session_id,
                "reason": close_reason,
                "duration_seconds": round(monotonic() - connected_at, 1),
            },
        )
