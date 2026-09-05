import ast
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SOURCE = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "model_service.py").read_text(encoding="utf-8")


def _function(name, **injected):
    """model_service 는 mediapipe 를 끌어오므로 함수 정의만 떼어 실행한다."""
    tree = ast.parse(_SOURCE)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {
        "keypoint_extraction_available": lambda: True,
        "keypoint_extraction_error": lambda: None,
        "recognition_model_available": lambda: True,
        "recognition_model_error": lambda: None,
        "model_path": lambda: Path("models/model_fold0.keras"),
    }
    namespace.update(injected)
    exec(compile(ast.Module([node], []), "<model_service>", "exec"), namespace)
    return namespace[name]


class ModelStatusTest(unittest.TestCase):
    """loaded 는 "수어 단어를 인식할 수 있는가" 여야 한다.

    좌표만 뽑는 상태에서 True 를 반환하면 /health 가 정상을 보고하고
    미니앱이 인식되는 것처럼 표시한다.
    """

    def test_status_is_loaded_only_when_the_recognition_model_is_up(self):
        status = _function("get_model_health_status")()

        self.assertTrue(status["loaded"])
        self.assertEqual(status["mode"], "recognition")
        self.assertIn("model_fold0", status["version"])

    def test_status_is_not_loaded_while_only_keypoints_work(self):
        """MediaPipe 는 되는데 인식 모델만 없는 상태."""
        status = _function(
            "get_model_health_status",
            recognition_model_available=lambda: False,
            recognition_model_error=lambda: "Recognition model not found: models/model_fold0.keras",
        )()

        self.assertFalse(status["loaded"])
        self.assertEqual(status["mode"], "keypoints_only")
        self.assertIn("model_fold0", status["version"])
        self.assertEqual(set(status), {"loaded", "mode", "version"})

    def test_status_keeps_the_three_fields_the_client_types(self):
        # miniapp/src/shared/channels.ts 가 {loaded, mode, version} 로 타입을 잡는다.
        self.assertEqual(
            set(_function("get_model_health_status")()),
            {"loaded", "mode", "version"},
        )

    def test_status_says_unavailable_when_landmarkers_failed_to_load(self):
        """좌표 추출이 죽으면 인식 모델이 올라와 있어도 unavailable 이다."""
        status = _function(
            "get_model_health_status",
            keypoint_extraction_available=lambda: False,
            keypoint_extraction_error=lambda: "Face model not found: models/face_landmarker.task",
        )()

        self.assertFalse(status["loaded"])
        self.assertEqual(status["mode"], "unavailable")
        self.assertIn("face_landmarker", status["version"])
        # 필드 모양은 추출 가능 여부와 무관하게 같아야 한다.
        self.assertEqual(set(status), {"loaded", "mode", "version"})

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
