import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import SEQUENCE_STRIDE, SEQUENCE_WINDOW_SIZE
from app.schemas.openpose import OpenPoseResult

logger = logging.getLogger(__name__)

POSE_2D_FEATURE_COUNT = 25 * 3
HAND_2D_FEATURE_COUNT = 21 * 3
OPENPOSE_FEATURE_DIM = POSE_2D_FEATURE_COUNT + HAND_2D_FEATURE_COUNT * 2


@dataclass(frozen=True)
class SequenceAppendResult:
    frame_count: int
    feature_dim: int
    window_size: int
    stride: int
    ready: bool
    window_index: int | None = None
    sequence: list[list[float]] | None = None

    def metadata(self) -> dict[str, int | bool | None]:
        return {
            "ready": self.ready,
            "frame_count": self.frame_count,
            "window_size": self.window_size,
            "stride": self.stride,
            "feature_dim": self.feature_dim,
            "window_index": self.window_index,
        }


@dataclass
class _SessionSequenceState:
    frames: list[list[float]] = field(default_factory=list)
    frame_count: int = 0
    window_count: int = 0


def _coerce_openpose_result(openpose_result: OpenPoseResult | dict[str, Any]):
    if isinstance(openpose_result, OpenPoseResult):
        return openpose_result
    return OpenPoseResult.model_validate(openpose_result)


def build_openpose_feature_vector(
    openpose_result: OpenPoseResult | dict[str, Any],
) -> list[float]:
    result = _coerce_openpose_result(openpose_result)
    person = result.people
    feature_vector = (
        person.pose_keypoints_2d
        + person.hand_left_keypoints_2d
        + person.hand_right_keypoints_2d
    )

    if len(feature_vector) != OPENPOSE_FEATURE_DIM:
        raise ValueError(
            f"OpenPose feature vector must have {OPENPOSE_FEATURE_DIM} values."
        )

    return [float(value) for value in feature_vector]


class SlidingWindowSequenceStore:
    def __init__(
        self,
        *,
        window_size: int = SEQUENCE_WINDOW_SIZE,
        stride: int = SEQUENCE_STRIDE,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")
        if stride <= 0:
            raise ValueError("stride must be greater than 0.")

        self.window_size = window_size
        self.stride = stride
        self._states: dict[str, _SessionSequenceState] = {}
        self._lock = threading.Lock()

    def append_openpose_result(
        self,
        session_id: str,
        openpose_result: OpenPoseResult | dict[str, Any],
    ) -> SequenceAppendResult:
        feature_vector = build_openpose_feature_vector(openpose_result)

        with self._lock:
            state = self._states.setdefault(session_id, _SessionSequenceState())
            state.frames.append(feature_vector)
            state.frame_count += 1

            if len(state.frames) > self.window_size:
                state.frames = state.frames[-self.window_size :]

            ready = (
                len(state.frames) == self.window_size
                and (state.frame_count - self.window_size) % self.stride == 0
            )
            if not ready:
                return SequenceAppendResult(
                    frame_count=state.frame_count,
                    feature_dim=OPENPOSE_FEATURE_DIM,
                    window_size=self.window_size,
                    stride=self.stride,
                    ready=False,
                )

            state.window_count += 1
            sequence = [frame.copy() for frame in state.frames]
            logger.info(
                "Created sliding window sequence",
                extra={
                    "session_id": session_id,
                    "frame_count": state.frame_count,
                    "window_index": state.window_count,
                    "window_size": self.window_size,
                    "stride": self.stride,
                    "feature_dim": OPENPOSE_FEATURE_DIM,
                },
            )
            return SequenceAppendResult(
                frame_count=state.frame_count,
                feature_dim=OPENPOSE_FEATURE_DIM,
                window_size=self.window_size,
                stride=self.stride,
                ready=True,
                window_index=state.window_count,
                sequence=sequence,
            )

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)


sequence_store = SlidingWindowSequenceStore()
