"""클라이언트로 나가는 result 에 불필요한 좌표가 실리지 않는지 고정한다."""

import ast
import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SOURCE = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "model_service.py").read_text(encoding="utf-8")


def _public_result(include_keypoints: bool):
    """model_service 는 mediapipe 를 끌어오므로 함수 정의만 떼어 실행한다."""
    tree = ast.parse(_SOURCE)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "public_result")
    namespace = {"INCLUDE_KEYPOINTS_IN_RESULT": include_keypoints}
    exec(compile(ast.Module([node], []), "<model_service>", "exec"), namespace)
    return namespace["public_result"]


def _result():
    person = {
        "face_keypoints_2d": [0.0] * 210,
        "pose_keypoints_2d": [0.0] * 75,
        "hand_left_keypoints_2d": [0.0] * 63,
        "hand_right_keypoints_2d": [0.0] * 63,
        "face_keypoints_3d": [0.0] * 280,
        "pose_keypoints_3d": [0.0] * 100,
        "hand_left_keypoints_3d": [0.0] * 84,
        "hand_right_keypoints_3d": [0.0] * 84,
    }
    return {
        "text": None,
        "confidence": 0.0,
        "is_final": False,
        "keypoints": {"version": 1.3, "people": person},
        "sequence": {"ready": False, "frame_count": 12, "window_index": None},
    }


class PublicResultTest(unittest.TestCase):
    def test_keypoints_are_stripped_by_default(self):
        public = _public_result(False)(_result())

        self.assertNotIn("keypoints", public)

    def test_fields_the_client_reads_survive(self):
        # miniapp/src/background/ai-client.ts 가 읽는 것들.
        public = _public_result(False)(_result())

        for key in ("text", "confidence", "is_final", "sequence"):
            self.assertIn(key, public)
        self.assertIsNone(public["sequence"]["window_index"])

    def test_flag_restores_keypoints_for_debugging(self):
        public = _public_result(True)(_result())

        self.assertIn("keypoints", public)

    def test_original_is_not_mutated_so_server_logs_still_work(self):
        # whep_service 의 검출률 요약이 원본 result 를 그대로 쓴다.
        original = _result()

        _public_result(False)(original)

        self.assertIn("keypoints", original)

    def test_stripping_removes_most_of_the_message(self):
        original = _result()
        full = len(json.dumps(original))
        slim = len(json.dumps(_public_result(False)(original)))

        self.assertLess(slim, full * 0.2)


if __name__ == "__main__":
    unittest.main()
