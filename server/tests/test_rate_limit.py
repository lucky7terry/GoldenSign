"""슬라이딩 윈도 요청 카운터.

시계를 주입해서 "윈도가 지나면 다시 통과한다"를 60초 기다리지 않고 센다.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rate_limit import SlidingWindowRateLimiter


class _FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SlidingWindowRateLimiterTest(unittest.TestCase):
    def setUp(self):
        self.clock = _FakeClock()

    def _limiter(self, per_ip=2, total=10, window_seconds=60.0):
        return SlidingWindowRateLimiter(
            per_ip=per_ip,
            total=total,
            window_seconds=window_seconds,
            clock=self.clock,
        )

    def test_requests_under_the_limit_pass(self):
        limiter = self._limiter(per_ip=2)

        self.assertTrue(limiter.acquire("a").allowed)
        self.assertTrue(limiter.acquire("a").allowed)

    def test_the_per_ip_limit_rejects_the_next_request(self):
        limiter = self._limiter(per_ip=2)
        limiter.acquire("a")
        limiter.acquire("a")

        verdict = limiter.acquire("a")

        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.scope, "per_ip")

    def test_another_client_is_not_affected(self):
        limiter = self._limiter(per_ip=1)
        limiter.acquire("a")

        self.assertFalse(limiter.acquire("a").allowed)
        self.assertTrue(limiter.acquire("b").allowed)

    def test_the_global_limit_applies_across_clients(self):
        limiter = self._limiter(per_ip=5, total=2)
        limiter.acquire("a")
        limiter.acquire("b")

        verdict = limiter.acquire("c")

        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.scope, "global")

    def test_the_window_slides(self):
        limiter = self._limiter(per_ip=1)
        limiter.acquire("a")

        self.clock.advance(59.0)
        self.assertFalse(limiter.acquire("a").allowed)

        self.clock.advance(1.5)
        self.assertTrue(limiter.acquire("a").allowed)

    def test_a_rejected_request_is_not_counted(self):
        """거절을 세면 한 번 막힌 클라이언트가 윈도 내내 못 들어온다."""
        limiter = self._limiter(per_ip=1)
        limiter.acquire("a")

        self.clock.advance(30.0)
        for _ in range(5):
            limiter.acquire("a")

        # 처음 통과한 요청만 세어졌으므로 그 기준으로 윈도가 풀린다.
        self.clock.advance(30.5)
        self.assertTrue(limiter.acquire("a").allowed)

    def test_retry_after_counts_down_with_the_window(self):
        limiter = self._limiter(per_ip=1)
        limiter.acquire("a")

        self.assertEqual(limiter.acquire("a").retry_after_seconds, 60)
        self.clock.advance(45.0)
        self.assertEqual(limiter.acquire("a").retry_after_seconds, 15)

    def test_retry_after_is_never_zero(self):
        """0 을 주면 클라이언트가 곧바로 다시 요청한다."""
        limiter = self._limiter(per_ip=1)
        limiter.acquire("a")
        self.clock.advance(59.999)

        self.assertGreaterEqual(limiter.acquire("a").retry_after_seconds, 1)

    def test_expired_clients_are_forgotten(self):
        """임의의 IP 가 들어오는 경로라 키를 지우지 않으면 메모리가 늘어난다."""
        limiter = self._limiter(per_ip=1, total=1000)
        for index in range(50):
            limiter.acquire(f"ip-{index}")

        self.clock.advance(61.0)
        limiter.acquire("ip-last")

        self.assertEqual(list(limiter._per_ip), ["ip-last"])

    def test_a_passing_request_is_allowed_with_no_scope(self):
        verdict = self._limiter().acquire("a")

        self.assertIsNone(verdict.scope)
        self.assertEqual(verdict.retry_after_seconds, 0)

    def test_bad_limits_are_rejected(self):
        for kwargs in (
            {"per_ip": 0, "total": 10},
            {"per_ip": 10, "total": 0},
            {"per_ip": 10, "total": 10, "window_seconds": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SlidingWindowRateLimiter(clock=self.clock, **kwargs)


if __name__ == "__main__":
    unittest.main()
