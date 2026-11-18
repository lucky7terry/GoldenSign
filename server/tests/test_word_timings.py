"""단어 결과에 실리는 서버 처리 시간(word.timings).

앱은 (결과 수신 - word_end 송신) 에서 total_ms 를 빼 통신 왕복을 구한다.
그래서 키 이름과 "total_ms 는 송신 직전까지" 라는 정의가 바뀌면 앱 계산이
조용히 틀어진다. 이 두 가지를 고정한다.

session_websocket 은 FastAPI 를 끌어오므로 필요한 함수만 소스에서 떼어
실행한다(test_word_error_codes 와 같은 방식).
"""

import ast
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

_SERVER = Path(__file__).resolve().parents[1]
_SOURCE = (_SERVER / "app" / "api" / "session_websocket.py").read_text(encoding="utf-8")
_CONTRACT = (_SERVER.parent / "docs" / "api-contract.md").read_text(encoding="utf-8")

_TIMING_KEYS = ["drain_ms", "wait_ms", "features_ms", "inference_ms", "total_ms"]


class _ModelMissing(RuntimeError):
    pass


def _load_helpers(end_word, recognize):
    tree = ast.parse(_SOURCE)
    wanted = {"_close_word_and_recognize", "_elapsed_ms"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(nodes) == len(wanted), "함수 이름이 바뀌었다"
    namespace = {
        "perf_counter": time.perf_counter,
        "word_store": SimpleNamespace(end_word=end_word),
        "recognize_word_segment": recognize,
        "RecognitionModelUnavailableError": _ModelMissing,
        "logger": logging.getLogger("test"),
    }
    exec(compile(ast.Module(nodes, []), "<helpers>", "exec"), namespace)
    return namespace


def _finalize_word_source() -> str:
    start = _SOURCE.index("    async def finalize_word(")
    end = _SOURCE.index("    async def auto_close_word(")
    return _SOURCE[start:end]


class CloseWordTimingTest(unittest.TestCase):
    def test_features_and_inference_are_measured_separately(self):
        def end_word(session_id, generation):
            time.sleep(0.02)
            return SimpleNamespace(sequence="seq")

        def recognize(sequence):
            time.sleep(0.01)
            return "word"

        helpers = _load_helpers(end_word, recognize)
        before = time.perf_counter()
        segment, recognition, timings, started = helpers["_close_word_and_recognize"]("s", 1)

        self.assertEqual(recognition, "word")
        self.assertEqual(set(timings), {"features_ms", "inference_ms"})
        self.assertGreaterEqual(timings["features_ms"], 15.0)
        self.assertGreaterEqual(timings["inference_ms"], 5.0)
        # started 는 executor 안에서 찍은 시각이라 호출 직전 이후여야 한다.
        self.assertGreaterEqual(started, before)

    def test_a_missing_model_still_reports_timings(self):
        def recognize(sequence):
            raise _ModelMissing("no model")

        helpers = _load_helpers(lambda s, g: SimpleNamespace(sequence="seq"), recognize)
        _, recognition, timings, _ = helpers["_close_word_and_recognize"]("s", 1)

        self.assertIsNone(recognition)
        self.assertIn("inference_ms", timings)

    def test_elapsed_ms_rounds_to_a_tenth(self):
        helpers = _load_helpers(None, None)
        self.assertEqual(helpers["_elapsed_ms"](1.0, 1.01234), 12.3)


class WordTimingsPayloadTest(unittest.TestCase):
    def test_the_payload_carries_every_timing_key(self):
        body = _finalize_word_source()
        block = body[body.index('metadata["timings"] = {'):body.index("payload = {")]
        for key in _TIMING_KEYS:
            # features_ms / inference_ms 는 helper 의 timings 를 펼쳐 넣는다.
            if key in {"features_ms", "inference_ms"}:
                self.assertIn("**timings", block)
            else:
                self.assertIn(f'"{key}"', block)

    def test_total_is_taken_before_the_result_is_sent(self):
        """송신 뒤에 재면 앱이 빼는 값에 소켓 쓰기까지 들어가 통신이 줄어 보인다."""
        body = _finalize_word_source()
        self.assertLess(
            body.index('"total_ms": _elapsed_ms(finalize_started'),
            body.index("await _send_json(websocket, send_lock, payload, activity)"),
        )

    def test_the_contract_documents_every_timing_key(self):
        section = _CONTRACT.split("### Server -> Client: word result", 1)[1]
        section = section.split("\n### ", 1)[0]
        for key in _TIMING_KEYS:
            self.assertIn(key, section, f"api-contract.md word result 에 {key} 설명이 없다")


if __name__ == "__main__":
    unittest.main()
