import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.timing_stats import TimingReporter, TimingStats


class TimingStatsTest(unittest.TestCase):
    def test_empty_snapshot_has_no_values(self):
        snapshot = TimingStats().snapshot()
        self.assertEqual(snapshot["count"], 0)
        self.assertIsNone(snapshot["avg_ms"])
        self.assertIsNone(snapshot["p95_ms"])
        self.assertIsNone(snapshot["max_ms"])

    def test_single_value_fills_every_field(self):
        stats = TimingStats()
        stats.record(12.5)
        self.assertEqual(
            stats.snapshot(),
            {"count": 1, "avg_ms": 12.5, "p95_ms": 12.5, "max_ms": 12.5},
        )

    def test_avg_and_p95_over_many_values(self):
        stats = TimingStats()
        for value in range(1, 101):
            stats.record(float(value))

        snapshot = stats.snapshot()
        self.assertEqual(snapshot["count"], 100)
        self.assertAlmostEqual(snapshot["avg_ms"], 50.5)
        self.assertAlmostEqual(snapshot["p95_ms"], 95.0)
        self.assertAlmostEqual(snapshot["max_ms"], 100.0)

    def test_p95_ignores_insertion_order(self):
        stats = TimingStats()
        for value in reversed(range(1, 21)):
            stats.record(float(value))
        self.assertAlmostEqual(stats.snapshot()["p95_ms"], 19.0)

    def test_capacity_drops_oldest_values(self):
        stats = TimingStats(capacity=3)
        for value in (1.0, 2.0, 3.0, 4.0):
            stats.record(value)

        snapshot = stats.snapshot()
        self.assertEqual(snapshot["count"], 3)
        self.assertAlmostEqual(snapshot["avg_ms"], 3.0)  # 1.0 이 밀려남
        self.assertAlmostEqual(snapshot["max_ms"], 4.0)

    def test_reset_clears_buffer(self):
        stats = TimingStats()
        stats.record(5.0)
        stats.reset()
        self.assertEqual(stats.snapshot()["count"], 0)

    def test_non_positive_capacity_is_rejected(self):
        for capacity in (0, -1):
            with self.assertRaises(ValueError):
                TimingStats(capacity=capacity)


class TimingReporterTest(unittest.TestCase):
    def test_reports_nothing_before_the_interval(self):
        reporter = TimingReporter(interval_seconds=60.0)
        self.assertFalse(reporter.should_report())
        self.assertFalse(reporter.should_report())

    def test_reports_once_the_interval_has_passed(self):
        reporter = TimingReporter(interval_seconds=60.0)
        reporter._last_report -= 61.0  # 시계를 앞당겨 sleep 없이 확인
        self.assertTrue(reporter.should_report())
        self.assertFalse(reporter.should_report())  # 보고 후 시각이 갱신됨

    def test_check_and_update_happen_while_holding_the_lock(self):
        reporter = TimingReporter(interval_seconds=60.0)
        reporter._last_report -= 61.0
        finished = threading.Event()

        def call_should_report():
            reporter.should_report()
            finished.set()

        # 락을 잡은 채로 부르면, 확인부터 갱신까지 전부 기다려야 한다.
        with reporter._lock:
            threading.Thread(target=call_should_report, daemon=True).start()
            self.assertFalse(finished.wait(timeout=0.1))

        self.assertTrue(finished.wait(timeout=2.0))

    def test_zero_interval_always_reports(self):
        reporter = TimingReporter(interval_seconds=0.0)
        self.assertTrue(reporter.should_report())
        self.assertTrue(reporter.should_report())


if __name__ == "__main__":
    unittest.main()