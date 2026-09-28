"""얼굴 검출용 크롭 상자와 좌표 복원.

원본 프레임을 그대로 넣으면 얼굴 랜드마커가 아무것도 못 찾는다.
검출기가 입력을 정사각형으로 리사이즈하는데 16:9 를 넣으면 위아래가
레터박스로 채워져 얼굴이 더 작아지기 때문이다.

실측 (WORD0001_REAL01_F, 1920x1080, 귀 사이 109px):

    원본 1920x1080  임계 0.5 / 0.3 / 0.1   -> 전부 실패
    축소 960x540                           -> 실패
    축소 633x356                           -> 실패
    중앙 크롭 960x1080                      -> 478점 검출

축소가 안 듣는 것이 핵심이다. 얼굴도 같이 작아져 화면에서 차지하는
비율이 그대로이기 때문이다. 그래서 잘라야 한다.
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
    wanted = {"_face_crop_box", "_restore_face_to_frame"}
    body = [n for n in klass.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.startswith(("_POSE_", "_FACE_CROP_")))]
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
        self.pose_landmarks = [landmarks] if landmarks else []


def _head(center_x, center_y, ear_span, width, height):
    """코/눈/귀 5점을 만든다. 좌표는 정규화 값이다."""
    half = ear_span / 2.0
    points = {
        0: (center_x, center_y),                       # 코
        2: (center_x - half * 0.55, center_y - half * 0.22),   # 왼눈
        5: (center_x + half * 0.55, center_y - half * 0.22),   # 오른눈
        7: (center_x - half, center_y),                # 왼귀
        8: (center_x + half, center_y),                # 오른귀
    }
    landmarks = [_Landmark(0.0, 0.0) for _ in range(33)]
    for index, (x, y) in points.items():
        landmarks[index] = _Landmark(x / width, y / height)
    return _PoseResult(landmarks)


class CropBoxTest(unittest.TestCase):

    WIDTH, HEIGHT = 1920, 1080

    def _box(self, center_x=960.0, center_y=290.0, ear_span=109.0):
        return Service._face_crop_box(
            _head(center_x, center_y, ear_span, self.WIDTH, self.HEIGHT),
            self.WIDTH, self.HEIGHT,
        )

    def test_the_crop_is_square(self):
        """검출기가 정사각형으로 리샘플하므로 직사각형을 주면 또 레터박스가 생긴다."""
        left, top, right, bottom = self._box()

        self.assertEqual(right - left, bottom - top)

    def test_the_face_fills_much_more_of_the_crop_than_of_the_frame(self):
        """이 비율이 올라가는 것이 고치려는 전부다."""
        left, top, right, bottom = self._box()

        before = 109.0 / self.WIDTH
        after = 109.0 / (right - left)

        self.assertLess(before, 0.06)
        self.assertGreater(after, 0.40)

    def test_the_crop_contains_the_whole_head(self):
        """턱선과 이마가 잘리면 70점 중 윤곽점이 통째로 틀어진다."""
        left, top, right, bottom = self._box()

        self.assertLess(left, 960.0 - 109.0 / 2)
        self.assertGreater(right, 960.0 + 109.0 / 2)
        self.assertLess(top, 290.0 - 109.0 / 2)
        self.assertGreater(bottom, 290.0 + 109.0 / 2)

    def test_a_face_at_the_edge_stays_inside_the_frame(self):
        left, top, right, bottom = self._box(center_x=30.0, center_y=20.0)

        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, self.WIDTH)
        self.assertLessEqual(bottom, self.HEIGHT)
        self.assertGreater(right - left, 1)
        self.assertGreater(bottom - top, 1)

    def test_a_profile_view_still_gets_a_usable_crop(self):
        """옆모습이면 두 귀가 겹쳐 폭이 0 에 가까워진다. 눈·코 퍼짐으로 받친다."""
        left, top, right, bottom = self._box(ear_span=2.0)

        self.assertGreaterEqual(right - left, Service._FACE_CROP_MIN_SIDE)

    def test_no_pose_means_no_crop(self):
        """호출자가 원본 프레임으로 떨어지도록 None 이어야 한다."""
        self.assertIsNone(
            Service._face_crop_box(_PoseResult(None), self.WIDTH, self.HEIGHT)
        )

    def test_a_short_landmark_list_is_rejected(self):
        self.assertIsNone(
            Service._face_crop_box(
                _PoseResult([_Landmark(0.5, 0.5)]), self.WIDTH, self.HEIGHT
            )
        )

    def test_nan_landmarks_are_rejected(self):
        landmarks = [_Landmark(0.5, 0.5) for _ in range(33)]
        landmarks[8] = _Landmark(float("nan"), 0.5)

        self.assertIsNone(
            Service._face_crop_box(
                _PoseResult(landmarks), self.WIDTH, self.HEIGHT
            )
        )


class RestoreTest(unittest.TestCase):
    """복원을 빼먹으면 검출은 성공했는데 좌표만 틀린다 - 눈에 안 띄는 실패다."""

    WIDTH, HEIGHT = 1920, 1080

    def test_the_crop_center_maps_to_the_crop_center_in_the_frame(self):
        box = (840, 170, 1080, 410)          # 240x240, 중심 (960, 290)
        face = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": None}]

        restored = Service._restore_face_to_frame(
            face, box, self.WIDTH, self.HEIGHT
        )

        self.assertAlmostEqual(restored[0]["x"] * self.WIDTH, 960.0, places=6)
        self.assertAlmostEqual(restored[0]["y"] * self.HEIGHT, 290.0, places=6)

    def test_the_crop_corners_map_to_the_crop_corners(self):
        box = (840, 170, 1080, 410)
        face = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": None},
            {"x": 1.0, "y": 1.0, "z": 0.0, "visibility": None},
        ]

        restored = Service._restore_face_to_frame(
            face, box, self.WIDTH, self.HEIGHT
        )

        self.assertAlmostEqual(restored[0]["x"] * self.WIDTH, 840.0, places=6)
        self.assertAlmostEqual(restored[0]["y"] * self.HEIGHT, 170.0, places=6)
        self.assertAlmostEqual(restored[1]["x"] * self.WIDTH, 1080.0, places=6)
        self.assertAlmostEqual(restored[1]["y"] * self.HEIGHT, 410.0, places=6)

    def test_restored_coordinates_stay_normalized(self):
        box = (840, 170, 1080, 410)
        face = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": None}]

        restored = Service._restore_face_to_frame(
            face, box, self.WIDTH, self.HEIGHT
        )

        self.assertTrue(0.0 <= restored[0]["x"] <= 1.0)
        self.assertTrue(0.0 <= restored[0]["y"] <= 1.0)

    def test_z_shrinks_with_the_crop(self):
        """z 는 x 와 같은 단위라 같은 비율로 줄여야 한다."""
        box = (840, 170, 1080, 410)
        face = [{"x": 0.5, "y": 0.5, "z": 0.1, "visibility": None}]

        restored = Service._restore_face_to_frame(
            face, box, self.WIDTH, self.HEIGHT
        )

        self.assertAlmostEqual(restored[0]["z"], 0.1 * 240 / 1920, places=9)


class AssignmentOrderTest(unittest.TestCase):
    """쓰기 전에 만들어졌는지 정적으로 본다.

    실제로 냈던 실수다. 얼굴 크롭이 image_width 를 검출 단계에서 쓰는데
    그 변수는 검출이 끝난 뒤에 대입되고 있었다. compileall 도 통과하고
    테스트 158개도 통과했다 - mediapipe 를 CI 에서 import 할 수 없어
    이 함수를 아무도 실행하지 않기 때문이다. 첫 프레임에서 터졌다.
    """

    # 검출 결과들은 try 안에서만 대입되고 try 뒤에서 쓰인다. 여기에 넣어
    # 두어야 대입 전에 쓰는 순서 실수가 잡힌다.
    NAMES = (
        "image_width",
        "image_height",
        "hand_result",
        "pose_result",
        "face_box",
        "face_result",
    )

    @staticmethod
    def _function():
        tree = ast.parse(_SOURCE)
        klass = next(n for n in tree.body
                     if isinstance(n, ast.ClassDef)
                     and n.name == "MediaPipeService")
        return next(n for n in ast.walk(klass)
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "extract_keypoints_from_image")

    def test_frame_size_is_assigned_before_it_is_used(self):
        func = self._function()

        for name in self.NAMES:
            stores = [n.lineno for n in ast.walk(func)
                      if isinstance(n, ast.Name)
                      and isinstance(n.ctx, ast.Store)
                      and n.id == name]
            loads = [n.lineno for n in ast.walk(func)
                     if isinstance(n, ast.Name)
                     and isinstance(n.ctx, ast.Load)
                     and n.id == name]
            self.assertTrue(stores, f"{name} 을 대입하는 곳이 없다")
            if not loads:
                continue
            self.assertLess(
                min(stores), min(loads),
                f"{name} 을 만들기 전에 쓴다 "
                f"(대입 {min(stores)}행, 사용 {min(loads)}행)",
            )

    def test_detection_failure_never_falls_through(self):
        """검출 결과는 try 안에서만 대입된다. except 가 삼키면 try 뒤에서
        대입 안 된 face_box / face_result 를 읽어 NameError 가 난다.
        except 가 전부 다시 던지는지 본다."""
        func = self._function()
        tries = [n for n in ast.walk(func) if isinstance(n, ast.Try)]
        self.assertTrue(tries, "검출을 감싸는 try 가 없다")
        for node in tries:
            for handler in node.handlers:
                self.assertIsInstance(
                    handler.body[-1], ast.Raise,
                    f"{handler.lineno}행 except 가 예외를 삼킨다",
                )

    def test_face_is_detected_once_per_frame(self):
        """크롭이 빗나가도 원본으로 다시 보지 않는다.

        원본은 검출률 0.00 이 확인된 조건이라 비용만 두 배가 되고,
        face_detect_stats 평균에 한 번/두 번 돈 프레임이 섞인다. 크롭과
        원본이 if/else 로 갈라져 있어야 한 프레임에 한 번만 돈다.
        """
        func = self._function()

        def face_detect_calls(nodes):
            return [
                n for node in nodes for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "detect"
                and isinstance(n.func.value, ast.Attribute)
                and n.func.value.attr == "_face_landmarker"
            ]

        calls = face_detect_calls([func])
        self.assertEqual(len(calls), 2, "크롭 한 번, 원본 한 번이어야 한다")
        branches = [
            n for n in ast.walk(func)
            if isinstance(n, ast.If)
            and len(face_detect_calls(n.body)) == 1
            and len(face_detect_calls(n.orelse)) == 1
        ]
        self.assertEqual(
            len(branches), 1,
            "크롭 검출과 원본 검출이 같은 if/else 의 양쪽에 있어야 한다",
        )


if __name__ == "__main__":
    unittest.main()
