"""프레임 처리 지연(ms) 계측용 링버퍼와 주기 보고 타이머.

기록은 O(1)이고, 백분위 계산은 snapshot 호출 시점으로 미룬다.
"""

import math
import threading
import time
from collections import deque


class TimingStats:
    """최근 capacity개의 측정값만 유지하는 고정 크기 통계 버퍼."""

    def __init__(self, capacity: int = 512):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive: {capacity}")
        self._capacity = capacity
        self._values: deque[float] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def record(self, value_ms: float) -> None:
        # 매 프레임 호출되는 경로. 락 구간은 append 하나로 유지한다.
        with self._lock:
            self._values.append(float(value_ms))

    def snapshot(self) -> dict:
        with self._lock:
            values = list(self._values)

        if not values:
            return {"count": 0, "avg_ms": None, "p95_ms": None, "max_ms": None}

        ordered = sorted(values)
        # nearest-rank: 하위 95% 지점의 실제 관측값을 고른다.
        index = min(math.ceil(0.95 * len(ordered)) - 1, len(ordered) - 1)
        return {
            "count": len(ordered),
            "avg_ms": sum(ordered) / len(ordered),
            "p95_ms": ordered[index],
            "max_ms": ordered[-1],
        }

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class TimingReporter:
    """interval_seconds 마다 한 번만 True를 돌려주는 보고 게이트."""

    def __init__(self, interval_seconds: float):
        if interval_seconds < 0:
            raise ValueError(f"interval must be non-negative: {interval_seconds}")
        self._interval = float(interval_seconds)
        self._last_report = time.monotonic()

    def should_report(self) -> bool:
        now = time.monotonic()
        if now - self._last_report < self._interval:
            return False
        self._last_report = now
        return True