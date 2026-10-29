"""프로세스 메모리 안의 슬라이딩 윈도 요청 카운터.

/v1/tts 가 인증 없이 외부 유료 API 를 부르는 경로라서 둔다. Redis 를 쓰지
않는 것은 인스턴스가 하나이고, 공유 저장소에 의존하면 그쪽이 죽을 때 음성이
같이 죽기 때문이다. 인스턴스를 늘리면 한도도 인스턴스 수만큼 늘어난다.

시계를 주입받는다. 윈도가 지나면 다시 통과한다는 것을 60초 기다리지 않고
확인할 방법이 그것뿐이다.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


@dataclass(frozen=True)
class RateLimitVerdict:
    allowed: bool
    # 거절했을 때 어느 한도였는지. "per_ip" | "global".
    scope: str | None
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    """최근 window_seconds 안의 요청을 클라이언트별로, 그리고 전체로 센다."""

    def __init__(
        self,
        per_ip: int,
        total: int,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_ip <= 0:
            raise ValueError("per_ip must be greater than 0.")
        if total <= 0:
            raise ValueError("total must be greater than 0.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0.")

        self._per_ip_limit = per_ip
        self._total_limit = total
        self._window = window_seconds
        self._clock = clock
        self._per_ip: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()
        # 비동기 라우트 하나만 쓰므로 경합은 없지만, 동기 핸들러에서 불려도 안전해야 한다.
        self._lock = threading.Lock()

    def acquire(self, client: str) -> RateLimitVerdict:
        """요청 하나를 센다. 거절하면 아무것도 세지 않는다."""
        with self._lock:
            now = self._clock()
            self._expire(now)

            hits = self._per_ip.get(client)
            if hits is not None and len(hits) >= self._per_ip_limit:
                return self._denied("per_ip", hits[0], now)
            if len(self._all) >= self._total_limit:
                return self._denied("global", self._all[0], now)

            if hits is None:
                hits = deque()
                self._per_ip[client] = hits
            hits.append(now)
            self._all.append(now)
            return RateLimitVerdict(allowed=True, scope=None, retry_after_seconds=0)

    def _expire(self, now: float) -> None:
        """윈도를 벗어난 기록과 빈 클라이언트 키를 버린다."""
        cutoff = now - self._window
        while self._all and self._all[0] <= cutoff:
            self._all.popleft()

        # 키를 지우지 않으면 임의의 IP 가 들어오는 경로에서 메모리가 계속 늘어난다.
        for client in list(self._per_ip):
            hits = self._per_ip[client]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if not hits:
                del self._per_ip[client]

    def _denied(self, scope: str, oldest: float, now: float) -> RateLimitVerdict:
        remaining = self._window - (now - oldest)
        return RateLimitVerdict(
            allowed=False,
            scope=scope,
            # 0 을 주면 클라이언트가 곧바로 다시 요청한다.
            retry_after_seconds=max(1, math.ceil(remaining)),
        )
