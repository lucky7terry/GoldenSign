import ast
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SOURCE = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "model_service.py").read_text(encoding="utf-8")


def _function(name):
    """model_service 는 mediapipe 를 끌어오므로 함수 정의만 떼어 실행한다."""
    tree = ast.parse(_SOURCE)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module([node], []), "<model_service>", "exec"), namespace)
    return namespace[name]


class ModelStatusTest(unittest.TestCase):
    """인식 모델이 없는 동안 서버가 인식을 주장하지 않는지 고정한다."""

    def test_status_reports_recognition_model_as_not_loaded(self):
        status = _function("get_model_health_status")()

        self.assertFalse(status["loaded"])
        self.assertIn("mode", status)
        self.assertIn("version", status)

    def test_status_keeps_the_three_fields_the_client_types(self):
        # miniapp/src/shared/channels.ts 가 {loaded, mode, version} 로 타입을 잡는다.
        self.assertEqual(
            set(_function("get_model_health_status")()),
            {"loaded", "mode", "version"},
        )

    def test_result_does_not_claim_a_recognized_word(self):
        # 자리표시자 문자열을 넣으면 그게 그대로 안경 화면에 뜬다.
        class _Keypoints:
            def model_dump(self, by_alias=False):
                return {"people": {}}

        payload = _function("_recognition_result")(_Keypoints())

        self.assertIsNone(payload["text"])
        self.assertEqual(payload["confidence"], 0.0)
        self.assertFalse(payload["is_final"])

    def test_result_still_carries_keypoints(self):
        class _Keypoints:
            def model_dump(self, by_alias=False):
                return {"people": {"pose_keypoints_2d": [1.0, 2.0, 0.9]}}

        payload = _function("_recognition_result")(_Keypoints())

        self.assertIn("keypoints", payload)
        self.assertEqual(payload["keypoints"]["people"]["pose_keypoints_2d"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
