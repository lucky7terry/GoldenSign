"""단어 구간(dev-0.4) 오류 code 가 계약 문서 표와 어긋나지 않는다.

앱은 code 로 분기한다. 서버가 새 code 를 내면서 표에 안 적으면 앱은
message 문자열을 비교하는 수밖에 없고, 프레임 경로와 같은 code 를 쓰면
"프레임 한 장 실패(계속 보내면 됨)"와 "구간 판정 실패(word_start 부터
다시)"를 구분하지 못한다. 둘 다 리뷰에서 실제로 나온 문제다.

session_websocket 은 FastAPI 를 끌어오므로 소스만 ast 로 읽는다.
"""

import ast
import re
import unittest
from pathlib import Path

_SERVER = Path(__file__).resolve().parents[1]
_SOURCE = (_SERVER / "app" / "api" / "session_websocket.py").read_text(encoding="utf-8")
_CONTRACT = (_SERVER.parent / "docs" / "api-contract.md").read_text(encoding="utf-8")

# 프레임 경로(dev-0.2 / dev-0.3) 전용 code. 단어 경로가 쓰면 안 된다.
_FRAME_ONLY_CODES = {"model_unavailable", "frame_too_large", "frame_queue_full"}

# 모든 스키마에 공통이라 dev-0.4 표에 따로 적지 않는 code.
_COMMON_CODES = {"unsupported_schema_version", "invalid_schema", "invalid_json"}


def _word_error_codes() -> set[str]:
    """WORD_SCHEMA_VERSION 으로 나가는 error_message 의 code 전부."""
    codes = set()
    for node in ast.walk(ast.parse(_SOURCE)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "error_message"):
            continue
        schema = getattr(node.args[0], "id", None)
        if schema == "WORD_SCHEMA_VERSION":
            codes.add(node.args[2].value)
    assert codes, "error_message 호출을 하나도 못 찾았다 - 함수 이름이 바뀌었나"
    return codes


def _documented_word_codes() -> set[str]:
    """api-contract.md 의 dev-0.4 'Server -> Client: 오류' 표에 적힌 code."""
    section = _CONTRACT.split("### Server -> Client: 오류", 1)[1]
    section = section.split("\n### ", 1)[0]
    return set(re.findall(r"^\| `([a-z_]+)` \|", section, re.M))


class WordErrorCodeContractTest(unittest.TestCase):

    def test_every_word_error_code_is_in_the_contract_table(self):
        missing = _word_error_codes() - _COMMON_CODES - _documented_word_codes()
        self.assertFalse(missing, f"api-contract.md 오류 표에 없는 code: {sorted(missing)}")

    def test_word_path_does_not_reuse_frame_path_codes(self):
        reused = _word_error_codes() & _FRAME_ONLY_CODES
        self.assertFalse(reused, f"프레임 경로 code 를 단어 경로가 쓴다: {sorted(reused)}")

    def test_finalization_failure_has_its_own_code(self):
        self.assertIn("word_recognition_failed", _word_error_codes())
        self.assertIn("word_recognition_failed", _documented_word_codes())


if __name__ == "__main__":
    unittest.main()
