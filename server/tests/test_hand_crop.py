"""손 검출용 정사각 크롭.

얼굴(test_face_crop.py)과 같은 문제다. 손 검출기도 입력을 정사각형으로
리사이즈해서, 16:9 원본에서는 화면 안에 있는 손을 놓친다.

실측 (느린 수어 안경 영상 50개, 1920x1080):

    원본         왼손 0.22 / 오른손 0.72
    정사각 크롭   왼손 0.94 / 오른손 0.98

학습 영상(정면 50개)에서도 같은 크롭으로 정확도가 유지되는 것을 확인했다.
"""

import ast
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SOURCE = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "mediapipe_service.py").read_text(encoding="utf-8")


def _load_helpers():
    """mediapipe_service 는 mediapipe 를 끌어오므로 메서드만 떼어 실행한다."""
    tree = ast.parse(_SOURCE)
    klass = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "MediaPipeService")
    wanted = {"_hand_crop_box", "_restore_face_to_frame"}
    body = [n for n in klass.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.startswith("_POSE_"))]
    assert {n.name for n in body if isinstance(n, ast.FunctionDef)} == wanted
    shell = ast.ClassDef(name="Service", bases=[], keywords=[],
                         body=body, decorator_list=[], type_params=[])
    module = ast.Module([shell], [])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "Any": object}
    exec(compile(module, "<mediapipe_service>", "exec"), namespace)
    return namespace["Service"]


Service = _load_helpers()


class _Landmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _PoseResult:
    def __init__(self, landmarks):
        self.pose_landmarks = [landmarks] if landmarks is not None else []


def _pose_with_shoulders(left_x, right_x, y=0.4):
    points = [_Landmark(0.5, 0.5) for _ in range(33)]
    points[11] = _Landmark(left_x, y)
    points[12] = _Landmark(right_x, y)
    return _PoseResult(points)


class HandCropBoxTest(unittest.TestCase):

    def test_box_is_a_full_height_square_on_16_9(self):
        left, top, right, bottom = Service._hand_crop_box(
            _pose_with_shoulders(0.45, 0.55), 1920, 1080)
        self.assertEqual(right - left, 1080)
        self.assertEqual(bottom - top, 1080)
        self.assertEqual(top, 0)

    def test_box_is_centered_on_the_shoulders(self):
        left, _, right, _ = Service._hand_crop_box(
            _pose_with_shoulders(0.55, 0.65), 1920, 1080)
        self.assertAlmostEqual((left + right) / 2, 0.60 * 1920, delta=1)

    def test_box_is_pushed_back_inside_the_frame(self):
        """사람이 화면 가장자리에 있어도 정사각형은 유지하고 화면 안으로 민다."""
        left, _, right, _ = Service._hand_crop_box(
            _pose_with_shoulders(0.02, 0.08), 1920, 1080)
        self.assertEqual(left, 0)
        self.assertEqual(right, 1080)
        left, _, right, _ = Service._hand_crop_box(
            _pose_with_shoulders(0.95, 0.99), 1920, 1080)
        self.assertEqual(right, 1920)
        self.assertEqual(right - left, 1080)

    def test_without_pose_it_falls_back_to_the_center_square(self):
        self.assertEqual(
            Service._hand_crop_box(_PoseResult(None), 1920, 1080),
            (420, 0, 1500, 1080),
        )

    def test_portrait_frame_uses_the_width_as_the_side(self):
        left, top, right, bottom = Service._hand_crop_box(
            _pose_with_shoulders(0.4, 0.6, y=0.3), 1080, 1920)
        self.assertEqual((left, right), (0, 1080))
        self.assertEqual(bottom - top, 1080)

    def test_non_finite_shoulders_fall_back_to_center(self):
        box = Service._hand_crop_box(
            _pose_with_shoulders(float("nan"), 0.5), 1920, 1080)
        self.assertEqual(box, (420, 0, 1500, 1080))


class HandRestoreTest(unittest.TestCase):

    def test_crop_coordinates_go_back_to_frame_coordinates(self):
        """빼먹으면 손이 크롭 기준 좌표로 나가 좌우 배정까지 틀어진다."""
        box = (420, 0, 1500, 1080)
        hand = [{"x": 0.5, "y": 0.5, "z": 0.1, "visibility": None},
                {"x": 0.0, "y": 1.0, "z": None, "visibility": None}]
        restored = Service._restore_face_to_frame(hand, box, 1920, 1080)
        self.assertAlmostEqual(restored[0]["x"], (420 + 540) / 1920)
        self.assertAlmostEqual(restored[0]["y"], 0.5)
        self.assertAlmostEqual(restored[0]["z"], 0.1 * 1080 / 1920)
        self.assertAlmostEqual(restored[1]["x"], 420 / 1920)
        self.assertAlmostEqual(restored[1]["y"], 1.0)
        self.assertIsNone(restored[1]["z"])


class CallOrderTest(unittest.TestCase):

    def test_pose_runs_before_hands(self):
        """손 크롭이 pose 어깨를 쓰므로 pose 가 먼저 돌아야 한다."""
        body = _SOURCE.split("def extract_keypoints_from_image", 1)[1]
        self.assertLess(body.index("self._pose_landmarker.detect("),
                        body.index("self._hand_landmarker.detect("))

    def test_hands_are_restored_before_assignment(self):
        body = _SOURCE.split("def extract_keypoints_from_image", 1)[1]
        self.assertLess(body.index("hand_box,"), body.index("assign_hands("))


if __name__ == "__main__":
    unittest.main()
