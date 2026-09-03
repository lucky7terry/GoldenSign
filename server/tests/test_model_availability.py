"""모델을 못 올렸을 때 서버가 조용히 반쯤 살아나지 않는지 고정한다."""

import ast
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SERVICE = Path(__file__).resolve().parents[1] / "app" / "services" / "mediapipe_service.py"


def _load_service_singleton():
    """mediapipe/cv2 없이 싱글턴 로직만 떼어 실행한다.

    MediaPipeService 자체는 스텁으로 갈아끼워, 로딩 실패를 기억하는지만 본다.
    """
    tree = ast.parse(_SERVICE.read_text(encoding="utf-8"))
    wanted = {
        "MediaPipeUnavailableError",
        "get_mediapipe_service",
        "preload_mediapipe_service",
        "keypoint_extraction_available",
        "keypoint_extraction_error",
    }
    state_names = {"_mediapipe_service", "_initialization_error",
                   "_mediapipe_service_lock"}

    def _is_state(node):
        if isinstance(node, ast.AnnAssign):
            return getattr(node.target, "id", "") in state_names
        if isinstance(node, ast.Assign):
            return any(getattr(t, "id", "") in state_names for t in node.targets)
        return False

    nodes = [n for n in tree.body
             if getattr(n, "name", None) in wanted or _is_state(n)]

    module = types.ModuleType("service_under_test")
    module.__dict__.update({
        "threading": __import__("threading"),
        "logger": __import__("logging").getLogger("test"),
        # 타입 주석이 런타임에 평가되므로 자리만 채워둔다. 각 테스트가 교체한다.
        "MediaPipeService": object,
    })
    exec(compile(ast.Module(nodes, []), "<service>", "exec"), module.__dict__)
    return module


class ModelAvailabilityTest(unittest.TestCase):
    def setUp(self):
        self.service = _load_service_singleton()

    def _fail_construction(self, message="Face model not found"):
        def _boom():
            raise FileNotFoundError(message)
        self.service.MediaPipeService = _boom

    def _succeed_construction(self):
        self.service.MediaPipeService = lambda: object()

    def test_loading_failure_is_remembered_not_retried(self):
        attempts = []

        def _boom():
            attempts.append(1)
            raise FileNotFoundError("Face model not found")

        self.service.MediaPipeService = _boom

        for _ in range(5):
            with self.assertRaises(self.service.MediaPipeUnavailableError):
                self.service.get_mediapipe_service()

        # 프레임마다 재시도하면 여기가 5가 된다.
        self.assertEqual(len(attempts), 1)

    def test_preload_reports_failure_without_raising(self):
        self._fail_construction()

        self.assertFalse(self.service.preload_mediapipe_service())
        self.assertFalse(self.service.keypoint_extraction_available())
        self.assertIn("Face model not found", self.service.keypoint_extraction_error())

    def test_preload_succeeds_when_models_load(self):
        self._succeed_construction()

        self.assertTrue(self.service.preload_mediapipe_service())
        self.assertTrue(self.service.keypoint_extraction_available())
        self.assertIsNone(self.service.keypoint_extraction_error())

    def test_service_is_created_once(self):
        created = []
        self.service.MediaPipeService = lambda: created.append(1) or object()

        first = self.service.get_mediapipe_service()
        second = self.service.get_mediapipe_service()

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
