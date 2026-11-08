"""앱이 보낸 client_log 를 서버 로그로 옮기는 경로.

session_websocket 은 FastAPI 와 mediapipe 를 끌어오므로 분기 본문이 부르는
app.client_log 를 직접 센다. 시계를 주입해서 "1초가 지나면 다시 통과한다"를
실제로 기다리지 않고 확인한다.
"""

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.client_log import log_client_event, new_client_log_limiter, normalize_fields
from app.config import (
    CLIENT_LOG_MAX_FIELDS,
    CLIENT_LOG_MAX_PER_SECOND,
    CLIENT_LOG_MAX_VALUE_CHARS,
)
from app.constants import SCHEMA_VERSION

_LOGGER = logging.getLogger("tests.client_log")


class _FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _message(**overrides) -> dict:
    message = {
        "type": "client_log",
        "schema_version": SCHEMA_VERSION,
        "session_id": "s-1",
        "client_message_id": "log-1",
        "event": "tts",
        "level": "info",
        "at": "2026-11-08T12:00:00.000Z",
        "fields": {"fallback": True, "tag": "result"},
    }
    message.update(overrides)
    return message


class ClientLogTest(unittest.TestCase):
    def setUp(self):
        self.clock = _FakeClock()
        self.limiter = new_client_log_limiter(clock=self.clock)

    def _log(self, raw: dict) -> list[logging.LogRecord]:
        with self.assertLogs(_LOGGER, level="DEBUG") as captured:
            self.logged = log_client_event(_LOGGER, "s-1", raw, self.limiter)
        return captured.records

    def test_a_valid_message_becomes_one_log_line(self):
        records = self._log(_message())

        self.assertTrue(self.logged)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.getMessage(), "Client event")
        self.assertEqual(record.levelno, logging.INFO)
        self.assertEqual(record.session_id, "s-1")
        self.assertEqual(record.event, "tts")
        self.assertEqual(record.at, "2026-11-08T12:00:00.000Z")
        self.assertEqual(record.fallback, True)
        self.assertEqual(record.tag, "result")

    def test_level_chooses_the_log_level(self):
        self.assertEqual(self._log(_message(level="warn"))[0].levelno, logging.WARNING)
        self.assertEqual(self._log(_message(level="error"))[0].levelno, logging.ERROR)

    def test_at_is_kept_as_the_app_sent_it(self):
        """끊긴 동안 모아 둔 이벤트는 수신 시각과 발생 시각이 다르다."""
        records = self._log(_message(at="2026-11-08T11:59:00.000Z"))

        self.assertEqual(records[0].at, "2026-11-08T11:59:00.000Z")

    def test_fields_may_be_omitted(self):
        message = _message()
        del message["fields"]

        self.assertEqual(len(self._log(message)), 1)
        self.assertTrue(self.logged)

    def test_a_malformed_message_only_logs_a_warning(self):
        """오류를 돌려주지도, 연결을 끊지도 않는다. 로그가 목적인 메시지다."""
        records = self._log(_message(level="verbose"))

        self.assertFalse(self.logged)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].levelno, logging.WARNING)
        self.assertEqual(records[0].getMessage(), "Ignored a malformed client_log")

    def test_a_missing_event_is_malformed(self):
        message = _message()
        del message["event"]

        self.assertEqual(self._log(message)[0].levelno, logging.WARNING)
        self.assertFalse(self.logged)

    def test_a_missing_at_is_malformed(self):
        message = _message()
        del message["at"]

        self.assertEqual(self._log(message)[0].levelno, logging.WARNING)
        self.assertFalse(self.logged)

    def test_fields_of_the_wrong_shape_is_malformed(self):
        self.assertFalse(log_client_event(_LOGGER, "s-1", _message(fields="tts"), self.limiter))

    def test_the_overflow_is_dropped_silently(self):
        for _ in range(CLIENT_LOG_MAX_PER_SECOND):
            self.assertTrue(log_client_event(_LOGGER, "s-1", _message(), self.limiter))

        with self.assertNoLogs(_LOGGER, level="DEBUG"):
            self.assertFalse(log_client_event(_LOGGER, "s-1", _message(), self.limiter))

    def test_the_next_second_passes_again(self):
        for _ in range(CLIENT_LOG_MAX_PER_SECOND + 1):
            log_client_event(_LOGGER, "s-1", _message(), self.limiter)
        self.clock.advance(1.0)

        self.assertTrue(log_client_event(_LOGGER, "s-1", _message(), self.limiter))

    def test_malformed_messages_are_counted_too(self):
        """모양이 틀린 쪽이 쏟아질 때도 막혀야 한다."""
        for _ in range(CLIENT_LOG_MAX_PER_SECOND):
            log_client_event(_LOGGER, "s-1", _message(level="verbose"), self.limiter)

        with self.assertNoLogs(_LOGGER, level="DEBUG"):
            self.assertFalse(log_client_event(_LOGGER, "s-1", _message(), self.limiter))


class NormalizeFieldsTest(unittest.TestCase):
    def test_long_strings_are_truncated(self):
        kept = normalize_fields({"error_message": "x" * 500})

        self.assertEqual(len(kept["error_message"]), CLIENT_LOG_MAX_VALUE_CHARS)

    def test_the_key_count_is_capped(self):
        kept = normalize_fields({f"k{i}": i for i in range(CLIENT_LOG_MAX_FIELDS + 10)})

        self.assertEqual(len(kept), CLIENT_LOG_MAX_FIELDS)

    def test_scalars_keep_their_type(self):
        kept = normalize_fields({"fps": 14.2, "frames": 8, "ok": False, "mode": "whip"})

        self.assertEqual(kept, {"fps": 14.2, "frames": 8, "ok": False, "mode": "whip"})

    def test_values_that_are_not_scalars_are_dropped(self):
        kept = normalize_fields({"nested": {"a": 1}, "list": [1, 2], "none": None, "ok": True})

        self.assertEqual(kept, {"ok": True})

    def test_reserved_record_keys_are_dropped(self):
        """extra 로 'message' 를 넣으면 logging 이 KeyError 를 던진다."""
        kept = normalize_fields(
            {"message": "x", "name": "x", "session_id": "spoof", "event": "spoof", "fps": 1}
        )

        self.assertEqual(kept, {"fps": 1})

    def test_keys_that_could_split_the_log_line_are_dropped(self):
        kept = normalize_fields({"a b": 1, "c\nd=e": 2, 'f"g': 3, "ok": 4})

        self.assertEqual(kept, {"ok": 4})

    def test_a_reserved_key_does_not_break_logging(self):
        """위의 거르기가 없으면 이 호출이 KeyError 로 죽는다."""
        limiter = new_client_log_limiter(clock=_FakeClock())

        with self.assertLogs(_LOGGER, level="DEBUG"):
            logged = log_client_event(
                _LOGGER,
                "s-1",
                _message(fields={"message": "spoof", "args": "spoof"}),
                limiter,
            )

        self.assertTrue(logged)


if __name__ == "__main__":
    unittest.main()