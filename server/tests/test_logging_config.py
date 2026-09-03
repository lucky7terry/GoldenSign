import logging
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logging_config import ExtraFieldFormatter


def _format(**extra):
    formatter = ExtraFieldFormatter(fmt="%(levelname)s | %(message)s")
    record = logging.LogRecord(
        "app.api.session_websocket", logging.INFO, "", 0,
        "WebSocket connected", (), None,
    )
    record.__dict__.update(extra)
    return formatter.format(record)


class ExtraFieldFormatterTest(unittest.TestCase):
    def test_extra_fields_are_rendered(self):
        line = _format(session_id="s-1", fps=11.9)

        self.assertIn("session_id=s-1", line)
        self.assertIn("fps=11.9", line)

    def test_record_without_extra_is_unchanged(self):
        self.assertEqual(_format(), "INFO | WebSocket connected")

    def test_newline_cannot_split_the_log_line(self):
        # 클라이언트가 보낸 값으로 가짜 로그 이벤트를 끼워 넣으려는 시도.
        evil = "abc\nINFO | WebSocket connected | session_id=victim"

        line = _format(client_message_id=evil)

        self.assertEqual(len(line.splitlines()), 1)
        self.assertNotIn("\n", line)
        self.assertIn("\\n", line)

    def test_carriage_return_and_escape_are_escaped(self):
        line = _format(stream_id="cf\r\x1b[2Kfake")

        self.assertNotIn("\r", line)
        self.assertNotIn("\x1b", line)

    def test_quotes_and_backslashes_survive_round_trip(self):
        line = _format(user_id='he said "hi"\\done')

        self.assertEqual(len(line.splitlines()), 1)
        self.assertNotIn('"hi"', line.replace('\\"', ""))

    def test_hangul_is_not_escaped_into_unicode_sequences(self):
        line = _format(keypoints="얼굴 70/70")

        self.assertIn("얼굴", line)
        self.assertNotIn("\\u", line)

    def test_value_without_special_characters_is_not_quoted(self):
        self.assertIn("reason=idle_timeout", _format(reason="idle_timeout"))


if __name__ == "__main__":
    unittest.main()
