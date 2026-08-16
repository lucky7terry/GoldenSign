import asyncio
import inspect
import ipaddress
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from urllib.parse import urljoin, urlparse

from app.config import (
    WHEP_ALLOWED_HOST_SUFFIXES,
    WHEP_CONNECT_TIMEOUT_SECONDS,
    WHEP_FIRST_FRAME_TIMEOUT_SECONDS,
    WHEP_FRAME_IDLE_TIMEOUT_SECONDS,
    WHEP_KEYFRAME_REQUEST_INTERVAL_SECONDS,
    WHEP_MAX_RETRIES,
    WHEP_RETRY_DELAY_SECONDS,
    WHEP_STUN_SERVER_URLS,
)
from app.constants import WEBRTC_SCHEMA_VERSION
from app.error import error_message

logger = logging.getLogger(__name__)

SendJson = Callable[[dict], Awaitable[None]]


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _recognize_whep_frame(session_id: str, sequence_generation: int, image):
    from app.services.model_service import (
        get_model_health_status,
        recognize_frame_from_image,
    )

    return (
        recognize_frame_from_image(image, session_id, sequence_generation),
        get_model_health_status(),
    )


def _current_sequence_generation(session_id: str) -> int | None:
    from app.services.sequence_service import sequence_store

    return sequence_store.current_generation(session_id)


def _is_sequence_session_closed(error: Exception) -> bool:
    from app.services.sequence_service import SequenceSessionClosed

    return isinstance(error, SequenceSessionClosed)


def _keypoint_summary(keypoints: dict) -> str:
    person = keypoints.get("people", {}) or {}

    def detected(points: list[float]) -> int:
        return sum(1 for index in range(2, len(points), 3) if points[index] > 0.0)

    pose = person.get("pose_keypoints_2d", [])
    left_hand = person.get("hand_left_keypoints_2d", [])
    right_hand = person.get("hand_right_keypoints_2d", [])
    return (
        f"pose {detected(pose)}/{len(pose) // 3}, "
        f"L-hand {detected(left_hand)}/{len(left_hand) // 3}, "
        f"R-hand {detected(right_hand)}/{len(right_hand) // 3}"
    )


@dataclass
class WhepStreamHandle:
    session_id: str
    stream_id: str
    webrtc_url: str
    client_message_id: str | None
    send_json: SendJson
    stop_event: asyncio.Event
    task: asyncio.Task | None = None


class WhepPullService:
    def __init__(self):
        self._streams: dict[str, WhepStreamHandle] = {}
        self._lock = asyncio.Lock()

    async def start_stream(
        self,
        *,
        session_id: str,
        stream_id: str,
        webrtc_url: str,
        client_message_id: str | None,
        send_json: SendJson,
    ) -> None:
        self._validate_whep_url(webrtc_url)

        stop_event = asyncio.Event()
        handle = WhepStreamHandle(
            session_id=session_id,
            stream_id=stream_id,
            webrtc_url=webrtc_url,
            client_message_id=client_message_id,
            send_json=send_json,
            stop_event=stop_event,
        )

        async with self._lock:
            existing = self._streams.pop(session_id, None)
            if existing is not None:
                existing.stop_event.set()
            self._streams[session_id] = handle
            handle.task = asyncio.create_task(self._run_stream(handle))

        if existing is not None:
            await self._cancel_stream(existing)

    async def stop_stream(self, session_id: str, stream_id: str | None = None) -> bool:
        async with self._lock:
            handle = self._streams.get(session_id)
            if handle is None:
                return False
            if stream_id is not None and handle.stream_id != stream_id:
                logger.info(
                    "Ignoring WHEP stop for non-current stream",
                    extra={
                        "session_id": session_id,
                        "stream_id": stream_id,
                        "current_stream_id": handle.stream_id,
                    },
                )
                return False
            self._streams.pop(session_id, None)

        await self._cancel_stream(handle)
        logger.info(
            "Stopped WHEP stream",
            extra={"session_id": session_id, "stream_id": handle.stream_id},
        )
        return True

    async def stop_session(self, session_id: str) -> None:
        await self.stop_stream(session_id)

    async def _get_stream(self, session_id: str) -> WhepStreamHandle | None:
        async with self._lock:
            return self._streams.get(session_id)

    async def _run_stream(self, handle: WhepStreamHandle) -> None:
        try:
            last_error: Exception | None = None
            for attempt in range(1, WHEP_MAX_RETRIES + 1):
                if handle.stop_event.is_set():
                    return
                try:
                    await self._connect_and_receive(handle)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "WHEP connection attempt failed",
                        extra={
                            "session_id": handle.session_id,
                            "stream_id": handle.stream_id,
                            "attempt": attempt,
                            "max_retries": WHEP_MAX_RETRIES,
                        },
                        exc_info=True,
                    )
                    if attempt >= WHEP_MAX_RETRIES:
                        break
                    try:
                        await asyncio.wait_for(
                            handle.stop_event.wait(),
                            timeout=WHEP_RETRY_DELAY_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        pass

            if not handle.stop_event.is_set():
                try:
                    await handle.send_json(
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            handle.session_id,
                            "stream_unavailable",
                            "WHEP stream is unavailable. Please try again.",
                            handle.client_message_id,
                            retryable=True,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to send stream_unavailable error",
                        extra={
                            "session_id": handle.session_id,
                            "stream_id": handle.stream_id,
                        },
                    )
                logger.error(
                    "WHEP stream unavailable after retry exhaustion",
                    extra={
                        "session_id": handle.session_id,
                        "stream_id": handle.stream_id,
                        "error": str(last_error) if last_error else "",
                    },
                )
        finally:
            await self._remove_if_current(handle)

    async def _connect_and_receive(self, handle: WhepStreamHandle) -> None:
        import aiohttp
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )

        peer_connection = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=list(WHEP_STUN_SERVER_URLS)),
                ]
            )
        )
        session: aiohttp.ClientSession | None = None
        resource_url: str | None = None

        try:
            video_track_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
            connection_closed = asyncio.Event()

            @peer_connection.on("track")
            def on_track(track):
                if track.kind == "video" and video_track_queue.empty():
                    video_track_queue.put_nowait(track)

            @peer_connection.on("connectionstatechange")
            def on_connection_state_change():
                if peer_connection.connectionState in {"failed", "closed"}:
                    connection_closed.set()

            peer_connection.addTransceiver("video", direction="recvonly")
            peer_connection.addTransceiver("audio", direction="recvonly")
            offer = await peer_connection.createOffer()
            await peer_connection.setLocalDescription(offer)
            await self._wait_for_ice_gathering(peer_connection)

            session = aiohttp.ClientSession()
            async with session.post(
                handle.webrtc_url,
                data=peer_connection.localDescription.sdp,
                headers={
                    "Content-Type": "application/sdp",
                    "Accept": "application/sdp",
                },
                timeout=aiohttp.ClientTimeout(total=WHEP_CONNECT_TIMEOUT_SECONDS),
                allow_redirects=False,
            ) as response:
                if response.status not in {200, 201}:
                    raise RuntimeError(
                        f"WHEP POST failed with status {response.status}"
                    )
                answer_sdp = await response.text()
                location = response.headers.get("Location")
                if location:
                    candidate_resource_url = urljoin(handle.webrtc_url, location)
                    self._validate_whep_url(candidate_resource_url)
                    resource_url = candidate_resource_url

            await peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type="answer")
            )
            logger.info(
                "WHEP connection established",
                extra={
                    "session_id": handle.session_id,
                    "stream_id": handle.stream_id,
                },
            )

            video_track = await asyncio.wait_for(
                video_track_queue.get(),
                timeout=WHEP_CONNECT_TIMEOUT_SECONDS,
            )
            await self._request_video_keyframe(video_track)
            await self._receive_video_frames(handle, video_track, connection_closed)
        finally:
            await self._cleanup_connection(session, peer_connection, resource_url)

    async def _receive_video_frames(
        self,
        handle: WhepStreamHandle,
        video_track,
        connection_closed: asyncio.Event,
    ) -> None:
        from aiortc.mediastreams import MediaStreamError

        frame_count = 0
        last_log_at = monotonic()
        last_frame_at = monotonic()
        first_frame_received = False
        last_keyframe_request_at = 0.0
        latest_frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        processing_task = asyncio.create_task(
            self._process_latest_video_frames(handle, latest_frame_queue)
        )
        recv_task = asyncio.create_task(video_track.recv())

        try:
            while not handle.stop_event.is_set():
                self._raise_if_processing_task_done(processing_task)
                done, _ = await asyncio.wait({recv_task}, timeout=1.0)
                if not done:
                    self._raise_if_processing_task_done(processing_task)
                    now = monotonic()
                    if connection_closed.is_set():
                        raise RuntimeError(
                            "WHEP connection closed while waiting for frames."
                        )
                    frame_timeout = (
                        WHEP_FRAME_IDLE_TIMEOUT_SECONDS
                        if first_frame_received
                        else WHEP_FIRST_FRAME_TIMEOUT_SECONDS
                    )
                    if not first_frame_received and (
                        now - last_keyframe_request_at
                        >= WHEP_KEYFRAME_REQUEST_INTERVAL_SECONDS
                    ):
                        await self._request_video_keyframe(video_track)
                        last_keyframe_request_at = now
                    if now - last_frame_at >= frame_timeout:
                        raise RuntimeError("WHEP video frames stopped.")
                    continue

                self._raise_if_processing_task_done(processing_task)
                try:
                    video_frame = recv_task.result()
                except MediaStreamError as exc:
                    if handle.stop_event.is_set():
                        logger.info(
                            "WHEP video track ended",
                            extra={
                                "session_id": handle.session_id,
                                "stream_id": handle.stream_id,
                                "frame_count": frame_count,
                            },
                        )
                        return
                    logger.info(
                        "WHEP video track ended unexpectedly",
                        extra={
                            "session_id": handle.session_id,
                            "stream_id": handle.stream_id,
                            "frame_count": frame_count,
                        },
                    )
                    raise RuntimeError("WHEP video track ended unexpectedly.") from exc
                finally:
                    recv_task = asyncio.create_task(video_track.recv())

                frame_count += 1
                now = monotonic()
                last_frame_at = now
                first_frame_received = True
                self._replace_latest_video_frame(
                    latest_frame_queue,
                    frame_count,
                    video_frame,
                )

                if frame_count == 1 or now - last_log_at >= 5.0:
                    logger.info(
                        "Received WHEP video frames",
                        extra={
                            "session_id": handle.session_id,
                            "stream_id": handle.stream_id,
                            "frame_count": frame_count,
                        },
                    )
                    last_log_at = now
        finally:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)
            processing_task.cancel()
            await asyncio.gather(processing_task, return_exceptions=True)

    @staticmethod
    def _raise_if_processing_task_done(processing_task: asyncio.Task) -> None:
        if not processing_task.done():
            return
        if processing_task.cancelled():
            raise RuntimeError("WHEP frame processing was cancelled.")

        error = processing_task.exception()
        if error is None:
            raise RuntimeError("WHEP frame processing stopped unexpectedly.")
        raise RuntimeError("WHEP frame processing failed.") from error

    @staticmethod
    def _replace_latest_video_frame(
        latest_frame_queue: asyncio.Queue,
        frame_count: int,
        video_frame,
    ) -> None:
        if latest_frame_queue.full():
            try:
                latest_frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        latest_frame_queue.put_nowait((frame_count, video_frame))

    async def _process_latest_video_frames(
        self,
        handle: WhepStreamHandle,
        latest_frame_queue: asyncio.Queue,
    ) -> None:
        processed_frame_count = 0
        last_processed_log_at = monotonic()
        last_processed_frame_count = 0
        last_error_sent_at = 0.0

        while not handle.stop_event.is_set():
            try:
                frame_count, video_frame = await asyncio.wait_for(
                    latest_frame_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            try:
                image = video_frame.to_ndarray(format="bgr24")
                sequence_generation = _current_sequence_generation(handle.session_id)
                if sequence_generation is None:
                    continue
                loop = asyncio.get_running_loop()
                result, model_status = await loop.run_in_executor(
                    None,
                    _recognize_whep_frame,
                    handle.session_id,
                    sequence_generation,
                    image,
                )
            except ValueError:
                logger.exception(
                    "WHEP frame validation failed",
                    extra={
                        "session_id": handle.session_id,
                        "stream_id": handle.stream_id,
                        "frame_count": frame_count,
                    },
                )
                continue
            except Exception as exc:
                if _is_sequence_session_closed(exc):
                    return
                logger.exception(
                    "WHEP frame processing failed",
                    extra={
                        "session_id": handle.session_id,
                        "stream_id": handle.stream_id,
                        "frame_count": frame_count,
                    },
                )
                now = monotonic()
                if now - last_error_sent_at >= 5.0:
                    last_error_sent_at = now
                    await handle.send_json(
                        error_message(
                            WEBRTC_SCHEMA_VERSION,
                            handle.session_id,
                            "model_unavailable",
                            "WHEP frame processing failed.",
                            handle.client_message_id,
                            retryable=True,
                        )
                    )
                continue

            processed_frame_count += 1
            await handle.send_json(
                {
                    "type": "result",
                    "schema_version": WEBRTC_SCHEMA_VERSION,
                    "session_id": handle.session_id,
                    "client_message_id": handle.client_message_id,
                    "stream_id": handle.stream_id,
                    "sequence_index": processed_frame_count,
                    "result": result,
                    "model": model_status,
                    "processed_at": _utc_now_iso(),
                }
            )

            now = monotonic()
            if processed_frame_count == 1 or now - last_processed_log_at >= 5.0:
                elapsed = max(now - last_processed_log_at, 0.001)
                interval_processed_frames = (
                    processed_frame_count - last_processed_frame_count
                )
                logger.info(
                    "Processed WHEP video frames",
                    extra={
                        "session_id": handle.session_id,
                        "stream_id": handle.stream_id,
                        "frame_count": frame_count,
                        "processed_frame_count": processed_frame_count,
                        "fps": round(interval_processed_frames / elapsed, 2),
                        "keypoints": _keypoint_summary(result.get("keypoints", {})),
                    },
                )
                last_processed_log_at = now
                last_processed_frame_count = processed_frame_count

    async def _request_video_keyframe(self, video_track) -> None:
        receiver = getattr(video_track, "_receiver", None)
        if receiver is None:
            return

        ssrc = (
            getattr(receiver, "_ssrc", None)
            or getattr(video_track, "_ssrc", None)
            or getattr(video_track, "ssrc", None)
        )
        if ssrc is None:
            return

        for method_name in ("_send_rtcp_pli", "send_rtcp_pli"):
            method = getattr(receiver, method_name, None)
            if method is None:
                continue

            try:
                result = method(ssrc)
                if inspect.isawaitable(result):
                    await result
                logger.info("Requested WHEP video keyframe")
                return
            except Exception:
                logger.debug("Failed to request WHEP video keyframe", exc_info=True)

    async def _wait_for_ice_gathering(self, peer_connection) -> None:
        if peer_connection.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @peer_connection.on("icegatheringstatechange")
        def on_ice_gathering_state_change():
            if peer_connection.iceGatheringState == "complete":
                done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=WHEP_CONNECT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out; continuing with local description")

    async def _delete_whep_resource(
        self,
        session,
        resource_url: str | None,
    ) -> None:
        if session is None or resource_url is None:
            return

        try:
            async with session.delete(
                resource_url,
                timeout=WHEP_CONNECT_TIMEOUT_SECONDS,
                allow_redirects=False,
            ) as response:
                if response.status >= 400:
                    logger.warning(
                        "WHEP DELETE returned an error status",
                        extra={
                            "status": response.status,
                            "resource_url": resource_url,
                        },
                    )
        except Exception:
            logger.exception(
                "Failed to delete WHEP resource",
                extra={"resource_url": resource_url},
            )

    async def _cleanup_connection(
        self,
        session,
        peer_connection,
        resource_url: str | None,
    ) -> None:
        cleanup_task = asyncio.create_task(
            self._cleanup_connection_unshielded(session, peer_connection, resource_url)
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise

    async def _cleanup_connection_unshielded(
        self,
        session,
        peer_connection,
        resource_url: str | None,
    ) -> None:
        await self._delete_whep_resource(session, resource_url)
        if session is not None:
            await session.close()
        await peer_connection.close()

    async def _remove_if_current(self, handle: WhepStreamHandle) -> None:
        async with self._lock:
            if self._streams.get(handle.session_id) is handle:
                self._streams.pop(handle.session_id, None)

    async def _cancel_stream(self, handle: WhepStreamHandle) -> None:
        handle.stop_event.set()
        if handle.task is not None:
            handle.task.cancel()
            await asyncio.gather(handle.task, return_exceptions=True)

    def _validate_whep_url(self, whep_url: str) -> None:
        parsed = urlparse(whep_url)
        if parsed.scheme != "https":
            raise ValueError("WHEP URL must use https.")
        if parsed.username or parsed.password:
            raise ValueError("WHEP URL must not include credentials.")
        if parsed.port not in (None, 443):
            raise ValueError("WHEP URL must use the default HTTPS port.")
        if not parsed.hostname:
            raise ValueError("WHEP URL host is required.")

        hostname = parsed.hostname.lower()
        try:
            ip_address = ipaddress.ip_address(hostname)
        except ValueError:
            ip_address = None

        if ip_address is not None:
            raise ValueError("WHEP URL host is not allowed.")

        if not any(
            hostname == suffix or hostname.endswith(f".{suffix}")
            for suffix in WHEP_ALLOWED_HOST_SUFFIXES
        ):
            raise ValueError("WHEP URL host is not allowed.")


whep_pull_service = WhepPullService()
