"""단어 구간 리샘플이 쓰는 촬영 시각 추출기.

두 함수 다 mediapipe / aiortc 를 끌어오는 모듈에 있으므로, 기존
테스트들과 같은 방식으로 함수 정의만 떼어 실행한다.
"""

import ast
import sys
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(__file__).resolve().parents[1]


def _extract_function(relative_path: str, name: str, namespace: dict):
    source = (_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module([node], []), f"<{name}>", "exec"), namespace)
    return namespace[name]


_parse_captured_at_ms = _extract_function(
    "app/api/session_websocket.py",
    "_parse_captured_at_ms",
    {"datetime": datetime},
)
_frame_timestamp_ms = _extract_function(
    "app/services/whep_service.py",
    "_frame_timestamp_ms",
    {},
)


class _Frame:
    def __init__(self, pts, time_base):
        self.pts = pts
        self.time_base = time_base


class ParseCapturedAtTest(unittest.TestCase):
    def test_iso_with_offset(self):
        value = _parse_captured_at_ms("2026-09-03T01:02:03.500+00:00")

        expected = datetime(
            2026, 9, 3, 1, 2, 3, 500000, tzinfo=timezone.utc
        ).timestamp() * 1000.0
        self.assertAlmostEqual(value, expected, places=3)

    def test_trailing_z_is_accepted(self):
        with_z = _parse_captured_at_ms("2026-09-03T01:02:03.500Z")
        with_offset = _parse_captured_at_ms("2026-09-03T01:02:03.500+00:00")

        self.assertEqual(with_z, with_offset)

    def test_spacing_is_preserved(self):
        """구간 안의 상대 간격만 쓰므로 그 간격이 정확하면 된다."""
        first = _parse_captured_at_ms("2026-09-03T01:02:03.000Z")
        second = _parse_captured_at_ms("2026-09-03T01:02:03.040Z")

        self.assertAlmostEqual(second - first, 40.0, places=3)

    def test_missing_and_garbage_return_none(self):
        for value in (None, "", "어제", "2026-13-45T99:99:99Z", 12345):
            with self.subTest(value=value):
                self.assertIsNone(_parse_captured_at_ms(value))


class FrameTimestampTest(unittest.TestCase):
    def test_pts_and_time_base_become_milliseconds(self):
        # 90kHz 클럭에서 pts=9000 이면 0.1초.
        frame = _Frame(pts=9000, time_base=Fraction(1, 90000))

        self.assertAlmostEqual(_frame_timestamp_ms(frame), 100.0)

    def test_consecutive_frames_are_spaced_by_the_frame_interval(self):
        base = Fraction(1, 90000)
        first = _frame_timestamp_ms(_Frame(0, base))
        second = _frame_timestamp_ms(_Frame(3000, base))

        # 30fps -> 90000/30 = 3000 tick = 33.33ms
        self.assertAlmostEqual(second - first, 1000.0 / 30.0, places=4)

    def test_missing_fields_return_none(self):
        self.assertIsNone(_frame_timestamp_ms(_Frame(None, Fraction(1, 90000))))
        self.assertIsNone(_frame_timestamp_ms(_Frame(9000, None)))
        self.assertIsNone(_frame_timestamp_ms(object()))

    def test_unusable_time_base_returns_none(self):
        self.assertIsNone(_frame_timestamp_ms(_Frame(9000, "1/90000")))


if __name__ == "__main__":
    unittest.main()
