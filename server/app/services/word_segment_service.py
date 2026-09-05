"""단어 구간 버퍼와 모델 입력 생성.

슬라이딩 윈도우(프레임이 60장 쌓일 때마다 추론)를 대체한다. 사용자가
`word_start` / `word_end` 로 구간을 표시하면 그 구간의 프레임만 모아
한 번 추론한다.

## 학습 재현이 이 모듈의 전부다

학습 노트북은 이 순서로 돌았다.

    30fps 영상 (T, 411)
      -> build_features                      (T, 420)   전체 길이에 대해
      -> crop_resample (50~100% 구간 -> 60)  (60, 420)

`robust_scale` 이 영상 전체 중앙값을 쓰고 `interp_missing` 도 시간축 전체를
보간하므로, build_features 가 먼저 전체 길이에 도는 것이 확정이다.

서버가 다른 점은 하나다. **프레임이 불규칙한 간격으로 도착한다.** MediaPipe 가
12.7fps 로 도는데 스트림은 30fps 라 처리하지 못한 프레임은 버려지고, 얼마나
버려지는지가 순간마다 다르다. 그대로 build_features 에 넣으면 궤적이 시간축으로
일그러진다 - 촘촘하게 살아남은 구간은 느리게, 성기게 살아남은 구간은 빠르게
움직인 것처럼 보인다.

그래서 앞에 한 단계를 둔다.

    도착한 프레임 (T, 411) + 촬영 시각
      -> [1] 시간축 등간격 리샘플            (T', 411)
      -> [2] build_features                  (T', 420)
      -> [3] 인덱스 half-pixel 리샘플 -> 60  (60, 420)

[1] 의 격자는 원본 영상과 같은 30fps 다(WORD_SOURCE_FPS). 격자가 촘촘해야
도착한 프레임이 제 시각 근처에 떨어진다.

영상 5개(WORD0001, 5시점)를 여러 프레임레이트로 떨어뜨려 실측했다.
아래는 불규칙 드롭(random) 기준 평균 확신도다.

    구간 프레임 수    [1] 없음      [1] 30fps      [1] 관측 간격
    41~47            0.637(4/5)    0.776          0.782
    25~32            0.675         0.689          0.727
    16~18            0.705         0.761          0.769
     8~12            0.839         0.809          0.479  <- 붕괴

세 가지를 알 수 있다.

1. [1] 은 필요하다. 41~47프레임 구간에서 [1] 이 없으면 한 번 틀렸다
   (R 시점, "허리" 0.298, 2위와의 격차 0.019).
2. 격자를 "관측 간격"으로 잡으면 안 된다. 프레임 수는 보존되지만 간격이
   들쭉날쭉할 때(100~1567ms) 각 프레임이 원래 시각에서 크게 밀려난다.
   8프레임 구간에서 확신도가 임계값 0.5 아래로 떨어졌다. 하필 그 8이
   WORD_MIN_FRAMES 의 경계다.
3. 30fps 고정은 12가지 조건(4개 프레임 수 x 3개 드롭 방식) 전부에서
   0.689~0.809 였다. 균일한 구간에서 0.006 손해를 보지만 무너지지 않는다.

8프레임에서도 5/5 정답이므로 WORD_MIN_FRAMES=8 은 안전하다. MediaPipe 가
5.8fps 까지 떨어져도 정확도 자체는 문제가 없다.

## [1] 에서 신뢰도를 선형 보간하면 안 된다

411 은 (x, y, 신뢰도) 137쌍이다. 검출된 프레임과 미검출 프레임(0,0,0) 사이를
선형 보간하면 원점 쪽으로 끌린 가짜 좌표가 생기는데, 신뢰도까지 같이 섞여서
CONF_THRESHOLD(0.05) 를 넘는다. 그러면 feature_service._interpolate_missing 이
그것을 진짜 검출로 받아들여 보정하지 않는다.

그래서 신뢰도 열만 양옆 표본의 **최솟값**을 쓴다. 한쪽이라도 미검출이면
그 프레임은 미검출로 남고, 좌표는 build_features 가 제대로 보간한다.
"""

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import (
    WORD_MAX_SECONDS,
    WORD_MIN_FRAMES,
    WORD_SOURCE_FPS,
    WORD_TARGET_FRAMES,
)
from app.schemas.openpose import OpenPoseResult
from app.services.feature_service import build_features

logger = logging.getLogger(__name__)

POSE_2D_FEATURE_COUNT = 25 * 3
HAND_2D_FEATURE_COUNT = 21 * 3
FACE_2D_FEATURE_COUNT = 70 * 3
OPENPOSE_FEATURE_DIM = (
    POSE_2D_FEATURE_COUNT
    + HAND_2D_FEATURE_COUNT * 2
    + FACE_2D_FEATURE_COUNT
)

# 411 = (x, y, 신뢰도) x 137. 신뢰도만 다른 규칙으로 리샘플한다.
_CONFIDENCE_COLUMNS = np.arange(2, OPENPOSE_FEATURE_DIM, 3)

# 타임스탬프가 이 범위 밖의 프레임레이트를 함의하면 믿지 않는다.
# pts 32비트 랩어라운드, 기기 시계 점프, WebSocket/WHEP 시각이 한 구간에
# 섞이는 경우가 전부 여기 걸린다.
_MIN_PLAUSIBLE_FPS = 1.0
_MAX_PLAUSIBLE_FPS = 240.0


class WordSessionClosed(RuntimeError):
    """정리됐거나 다른 연결이 이어받은 세션에 접근했을 때."""


class WordAlreadyStarted(RuntimeError):
    """이미 열린 구간에 word_start 가 또 왔을 때."""


class WordNotStarted(RuntimeError):
    """열린 구간이 없는데 word_end 가 왔을 때."""


class WordTooShort(RuntimeError):
    """구간이 너무 짧아 60프레임으로 늘리는 것이 무의미할 때."""


def build_openpose_feature_vector(
    openpose_result: OpenPoseResult | dict[str, Any],
) -> list[float]:
    """OpenPose 결과를 411개 실수로 편다. 순서는 pose, 왼손, 오른손, 얼굴."""
    result = (
        openpose_result
        if isinstance(openpose_result, OpenPoseResult)
        else OpenPoseResult.model_validate(openpose_result)
    )
    person = result.people
    feature_vector = (
        person.pose_keypoints_2d
        + person.hand_left_keypoints_2d
        + person.hand_right_keypoints_2d
        + person.face_keypoints_2d
    )

    if len(feature_vector) != OPENPOSE_FEATURE_DIM:
        raise ValueError(
            f"OpenPose feature vector must have {OPENPOSE_FEATURE_DIM} values."
        )

    return [float(value) for value in feature_vector]


# ---------------------------------------------------------------------------
# 리샘플
# ---------------------------------------------------------------------------


def _impute_timestamps(
    timestamps_ms: list[float | None],
) -> np.ndarray | None:
    """빠진 시각을 인덱스 기준으로 채운다. 둘 미만이면 None.

    프레임을 버리지 않는 것이 핵심이다. 시각이 붙은 것만 남기면 그것들이
    구간 앞쪽에 몰려 있을 때 뒤쪽 절반이 출력에서 통째로 사라진다.
    """
    values = np.array(
        [
            np.nan if t is None else float(t)
            for t in timestamps_ms
        ],
        dtype=np.float64,
    )
    values[~np.isfinite(values)] = np.nan
    known = ~np.isnan(values)
    if known.sum() < 2:
        return None

    index = np.arange(values.shape[0], dtype=np.float64)
    # np.interp 는 양끝을 마지막 known 값으로 평평하게 채우므로, 그러면
    # 끝쪽 프레임들이 같은 시각을 갖는다. 평균 간격으로 이어 붙인다.
    filled = np.interp(index, index[known], values[known])
    first, last = index[known][0], index[known][-1]
    spacing = (values[known][-1] - values[known][0]) / max(last - first, 1.0)
    if spacing <= 0.0:
        return None
    head = index < first
    tail = index > last
    filled[head] = values[known][0] - (first - index[head]) * spacing
    filled[tail] = values[known][-1] + (index[tail] - last) * spacing
    return filled


def _bracketing_min(
    source_x: np.ndarray,
    column: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """양옆 표본의 최솟값. 신뢰도 열 전용.

    한쪽이라도 미검출(0)이면 결과도 0이 되어, 그 프레임은 미검출로 남는다.
    """
    count = source_x.shape[0]
    upper = np.searchsorted(source_x, target_x, side="left")
    at_sample = (upper < count) & (
        source_x[np.clip(upper, 0, count - 1)] == target_x
    )
    high = np.clip(upper, 1, count - 1)
    low = high - 1
    out = np.minimum(column[low], column[high])
    exact = np.clip(upper, 0, count - 1)
    out[at_sample] = column[exact][at_sample]
    return out


def _interpolate(
    source_x: np.ndarray,
    values: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """열마다 선형 보간하되 신뢰도 열은 양옆 최솟값."""
    out = np.empty((target_x.shape[0], values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        out[:, column] = np.interp(target_x, source_x, values[:, column])
    if values.shape[1] == OPENPOSE_FEATURE_DIM:
        for column in _CONFIDENCE_COLUMNS:
            out[:, column] = _bracketing_min(
                source_x,
                values[:, column],
                target_x,
            )
    return out


def resample_to_uniform_fps(
    frames: list[list[float]],
    timestamps_ms: list[float | None],
    fps: float = WORD_SOURCE_FPS,
) -> tuple[np.ndarray, bool]:
    """[1단계] 불규칙하게 도착한 411 프레임을 fps 등간격으로 되돌린다.

    격자가 촘촘할수록 도착한 프레임이 제 시각 근처에 떨어진다. 원본 영상과
    같은 30fps 가 기본이다 - 그보다 성기게 잡으면 프레임이 밀려난다.

    시각을 못 믿으면 원본을 그대로 돌려주고 False 를 반환한다.
    """
    if not frames:
        raise ValueError("Cannot resample an empty word segment.")
    if len(frames) != len(timestamps_ms):
        raise ValueError("frames and timestamps_ms must be the same length.")

    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Word segment frames must be 2-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Word segment frames contain NaN or Inf.")

    times = _impute_timestamps(timestamps_ms)
    if times is None:
        return values, False

    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]

    keep = np.concatenate(([True], np.diff(times) > 0.0))
    times = times[keep]
    values = values[keep]
    if times.shape[0] < 2:
        return np.asarray(frames, dtype=np.float64), False

    span_seconds = (times[-1] - times[0]) / 1000.0
    if span_seconds <= 0.0:
        return np.asarray(frames, dtype=np.float64), False

    implied_fps = (times.shape[0] - 1) / span_seconds
    if not (_MIN_PLAUSIBLE_FPS <= implied_fps <= _MAX_PLAUSIBLE_FPS):
        # 시계 점프나 pts 랩어라운드. 시각을 버리고 도착 순서를 믿는다.
        logger.warning(
            "Ignoring word segment timestamps; implied frame rate is impossible",
            extra={"implied_fps": round(implied_fps, 2)},
        )
        return np.asarray(frames, dtype=np.float64), False

    target_count = max(2, int(round(span_seconds * fps)) + 1)
    target_times = np.linspace(times[0], times[-1], target_count)
    return _interpolate(times, values, target_times), True


def resample_index_half_pixel(
    values: np.ndarray,
    target: int = WORD_TARGET_FRAMES,
) -> np.ndarray:
    """[3단계] 학습의 tf.image.resize(bilinear) 와 같은 좌표 규칙.

    출력 i 번째가 보는 원본 위치는 (i + 0.5) * T / target - 0.5 다.
    np.linspace(0, T-1, target) 로 하면 양 끝을 맞추는 다른 규칙이 되어
    값이 어긋난다 (T=143 에서 0.69프레임, T=600 에서 4.5프레임).
    """
    if target <= 1:
        raise ValueError("target must be greater than 1.")
    count = values.shape[0]
    if count == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if count == 1:
        return np.repeat(values, target, axis=0)

    positions = (np.arange(target) + 0.5) * (count / target) - 0.5
    positions = np.clip(positions, 0.0, count - 1.0)
    source = np.arange(count, dtype=np.float64)
    return np.stack(
        [
            np.interp(positions, source, values[:, dim])
            for dim in range(values.shape[1])
        ],
        axis=1,
    )


def build_model_input(
    frames: list[list[float]],
    timestamps_ms: list[float | None],
) -> tuple[np.ndarray, bool, int]:
    """도착한 411 프레임 -> 모델 입력 (60, 420).

    돌려주는 값: (특징, 시간축을 실제로 썼는지, 등간격으로 편 뒤의 프레임 수)
    """
    uniform, on_time = resample_to_uniform_fps(frames, timestamps_ms)
    features = build_features(uniform)
    sequence = resample_index_half_pixel(features, WORD_TARGET_FRAMES)
    return sequence.astype(np.float32), on_time, int(uniform.shape[0])


# ---------------------------------------------------------------------------
# 구간 저장소
# ---------------------------------------------------------------------------


@dataclass
class _WordState:
    frames: list[list[float]] = field(default_factory=list)
    timestamps_ms: list[float | None] = field(default_factory=list)
    dropped_after_cap: int = 0


@dataclass
class _SessionState:
    generation: int
    word: _WordState | None = None
    word_count: int = 0


@dataclass(frozen=True)
class WordAppendResult:
    buffered: bool
    frame_count: int
    span_ms: float | None
    at_cap: bool


@dataclass(frozen=True)
class WordSegment:
    """닫힌 구간 하나. sequence 는 (WORD_TARGET_FRAMES, 420) — 모델 입력."""

    word_index: int
    sequence: np.ndarray
    frame_count: int
    uniform_frame_count: int
    span_ms: float | None
    resampled_on_time: bool
    dropped_after_cap: int

    def metadata(self) -> dict:
        return {
            "word_index": self.word_index,
            "frame_count": self.frame_count,
            "uniform_frame_count": self.uniform_frame_count,
            "target_frames": int(self.sequence.shape[0]),
            "feature_dim": int(self.sequence.shape[1]),
            "span_ms": (
                None if self.span_ms is None else round(self.span_ms, 1)
            ),
            "resampled_on_time": self.resampled_on_time,
            "dropped_after_cap": self.dropped_after_cap,
        }


class WordSegmentStore:
    def __init__(
        self,
        *,
        target_frames: int = WORD_TARGET_FRAMES,
        min_frames: int = WORD_MIN_FRAMES,
        max_seconds: float = WORD_MAX_SECONDS,
    ) -> None:
        if target_frames <= 1:
            raise ValueError("target_frames must be greater than 1.")
        if min_frames <= 1:
            raise ValueError("min_frames must be greater than 1.")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be greater than 0.")

        self.target_frames = target_frames
        self.min_frames = min_frames
        self.max_seconds = max_seconds
        # 시간 상한(WORD_MAX_SECONDS)은 WebSocket 핸들러의 타이머가 건다.
        # 여기 상한은 그 타이머가 못 도는 상황에 대한 최후 방어라, 도달
        # 가능한 최대 프레임레이트를 넉넉히 잡아 유도한다.
        self.frame_cap = max(
            target_frames,
            math.ceil(max_seconds * _MAX_PLAUSIBLE_FPS / 4.0),
        )
        self._states: dict[str, _SessionState] = {}
        self._lock = threading.Lock()

    # 세션 수명 -------------------------------------------------------

    def start_session(self, session_id: str) -> int:
        with self._lock:
            previous = self._states.get(session_id)
            generation = (previous.generation if previous else 0) + 1
            self._states[session_id] = _SessionState(generation=generation)
            return generation

    def current_generation(self, session_id: str) -> int | None:
        with self._lock:
            state = self._states.get(session_id)
            return None if state is None else state.generation

    def clear_session(
        self,
        session_id: str,
        generation: int | None = None,
    ) -> bool:
        """세션 상태를 지운다. generation 을 주면 그 세대일 때만 지운다.

        세대를 확인하지 않으면, 늦게 죽는 연결의 정리 코드가 같은
        session_id 로 이미 재접속한 살아있는 연결을 지워버린다.
        """
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return False
            if generation is not None and state.generation != generation:
                return False
            self._states.pop(session_id, None)
            return True

    # 단어 구간 -------------------------------------------------------

    def is_word_open(self, session_id: str) -> bool:
        with self._lock:
            state = self._states.get(session_id)
            return state is not None and state.word is not None

    def start_word(self, session_id: str, generation: int | None = None) -> None:
        with self._lock:
            state = self._require_state(session_id, generation)
            if state.word is not None:
                raise WordAlreadyStarted("A word segment is already open.")
            state.word = _WordState()

    def abort_word(
        self,
        session_id: str,
        generation: int | None = None,
    ) -> bool:
        """결과를 내지 않고 구간을 버린다. generation 이 다르면 아무것도 안 한다."""
        with self._lock:
            state = self._states.get(session_id)
            if state is None or state.word is None:
                return False
            if generation is not None and state.generation != generation:
                return False
            state.word = None
            return True

    def append(
        self,
        session_id: str,
        feature_vector: list[float],
        captured_at_ms: float | None = None,
        generation: int | None = None,
    ) -> WordAppendResult:
        if len(feature_vector) != OPENPOSE_FEATURE_DIM:
            raise ValueError(
                f"OpenPose feature vector must have {OPENPOSE_FEATURE_DIM} values."
            )
        row = [float(value) for value in feature_vector]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("OpenPose feature vector contains NaN or Inf.")
        if captured_at_ms is not None and not math.isfinite(captured_at_ms):
            captured_at_ms = None

        with self._lock:
            state = self._require_state(session_id, generation)
            word = state.word
            if word is None:
                return WordAppendResult(
                    buffered=False,
                    frame_count=0,
                    span_ms=None,
                    at_cap=False,
                )

            if len(word.frames) >= self.frame_cap:
                word.dropped_after_cap += 1
                return WordAppendResult(
                    buffered=False,
                    frame_count=len(word.frames),
                    span_ms=_span_ms(word.timestamps_ms),
                    at_cap=True,
                )

            word.frames.append(row)
            word.timestamps_ms.append(captured_at_ms)
            return WordAppendResult(
                buffered=True,
                frame_count=len(word.frames),
                span_ms=_span_ms(word.timestamps_ms),
                at_cap=False,
            )

    def end_word(
        self,
        session_id: str,
        generation: int | None = None,
    ) -> WordSegment:
        """구간을 닫고 모델 입력 (target_frames, 420) 으로 만들어 돌려준다."""
        with self._lock:
            state = self._require_state(session_id, generation)
            word = state.word
            if word is None:
                raise WordNotStarted("No word segment is open.")

            frame_count = len(word.frames)
            if frame_count < self.min_frames:
                # 거절해도 구간은 닫는다. 안 닫으면 다음 word_start 가
                # WordAlreadyStarted 로 막혀 세션을 못 쓰게 된다.
                state.word = None
                raise WordTooShort(
                    f"Word segment has {frame_count} frames; "
                    f"at least {self.min_frames} are required."
                )

            state.word = None
            state.word_count += 1
            word_index = state.word_count
            frames = word.frames
            timestamps = word.timestamps_ms
            dropped = word.dropped_after_cap

        # 변환은 락 밖에서 한다. 다른 세션의 append 를 막을 이유가 없다.
        sequence, on_time, uniform_count = build_model_input(frames, timestamps)
        segment = WordSegment(
            word_index=word_index,
            sequence=sequence,
            frame_count=frame_count,
            uniform_frame_count=uniform_count,
            span_ms=_span_ms(timestamps),
            resampled_on_time=on_time,
            dropped_after_cap=dropped,
        )
        logger.info(
            "Closed word segment",
            extra={
                "session_id": session_id,
                "word_index": word_index,
                "frame_count": frame_count,
                "uniform_frame_count": uniform_count,
                "span_ms": segment.metadata()["span_ms"],
                "resampled_on_time": on_time,
                "effective_fps": _effective_fps(frame_count, segment.span_ms),
                "dropped_after_cap": dropped,
            },
        )
        return segment

    # 내부 -----------------------------------------------------------

    def _require_state(
        self,
        session_id: str,
        generation: int | None,
    ) -> _SessionState:
        """락을 잡은 채로 호출할 것."""
        state = self._states.get(session_id)
        current = state.generation if state is not None else None
        if generation is not None and generation != current:
            raise WordSessionClosed(
                "Cannot use a word segment from a closed recognition session."
            )
        if state is None:
            raise WordSessionClosed("Recognition session is not open.")
        return state


def _span_ms(timestamps_ms: list[float | None]) -> float | None:
    known = [t for t in timestamps_ms if t is not None and math.isfinite(t)]
    if len(known) < 2:
        return None
    return float(max(known) - min(known))


def _effective_fps(frame_count: int, span_ms: float | None) -> float | None:
    """n 개 프레임은 n-1 개 간격을 만든다. n 으로 나누면 fps 가 과대평가된다."""
    if span_ms is None or span_ms <= 0.0 or frame_count < 2:
        return None
    return round((frame_count - 1) / (span_ms / 1000.0), 2)


word_store = WordSegmentStore()
