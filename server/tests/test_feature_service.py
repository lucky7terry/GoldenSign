import json
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.feature_service import (
    FEATURE_DIM,
    RAW_DIM,
    CLIP_POSITION,
    CLIP_VELOCITY,
    FeatureShapeError,
    build_features,
)

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "build_features_golden.json"


def deterministic_frames(count=3):
    """골든 픽스처를 만들 때 쓴 것과 동일한 입력 (난수 없음)."""
    frames = np.zeros((count, RAW_DIM), np.float32)
    for t in range(count):
        for joint in range(25):
            frames[t, joint * 3 + 0] = 400.0 + 7.0 * joint + 3.0 * t
            frames[t, joint * 3 + 1] = 200.0 + 5.0 * joint - 2.0 * t
            frames[t, joint * 3 + 2] = 0.9
        for offset in (75, 138):
            for k in range(21):
                frames[t, offset + k * 3 + 0] = 500.0 + 4.0 * k + 2.0 * t
                frames[t, offset + k * 3 + 1] = 300.0 + 3.0 * k + 1.0 * t
                frames[t, offset + k * 3 + 2] = 1.0
        for k in range(70):
            frames[t, 201 + k * 3 + 0] = 620.0 + 1.5 * k + 0.5 * t
            frames[t, 201 + k * 3 + 1] = 190.0 + 1.2 * k - 0.5 * t
            frames[t, 201 + k * 3 + 2] = 1.0
    return frames


def moving_frames(count=60):
    times = np.arange(count)
    frames = np.zeros((count, RAW_DIM), np.float32)
    pose = {0: (640, 180), 1: (640, 260), 2: (560, 265), 3: (520, 350),
            4: (500, 430), 5: (720, 265), 6: (760, 350), 7: (780, 430),
            8: (640, 470), 9: (600, 470), 12: (680, 470), 15: (620, 170),
            16: (660, 170), 17: (590, 185), 18: (690, 185)}
    for joint, (x, y) in pose.items():
        frames[:, joint * 3 + 0] = x + 6 * np.sin(times / 7)
        frames[:, joint * 3 + 1] = y + 4 * np.cos(times / 9)
        frames[:, joint * 3 + 2] = 0.9
    for offset, wrist in ((75, 4), (138, 7)):
        wx, wy = pose[wrist]
        for k in range(21):
            frames[:, offset + k * 3 + 0] = wx + 30 * np.sin(times / 5 + k)
            frames[:, offset + k * 3 + 1] = wy + 30 * np.cos(times / 5 + k)
            frames[:, offset + k * 3 + 2] = 1.0
    for k in range(70):
        frames[:, 201 + k * 3 + 0] = 640 + 40 * np.sin(k)
        frames[:, 201 + k * 3 + 1] = 180 + 30 * np.cos(k)
        frames[:, 201 + k * 3 + 2] = 1.0
    return frames


class FeatureContractTest(unittest.TestCase):
    """모델 입력 규격이 학습 때와 어긋나지 않는지 고정한다."""

    def test_matches_training_notebook_golden_output(self):
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        expected = np.array(golden["expected"], dtype=np.float32)

        actual = build_features(deterministic_frames(golden["frame_count"]))

        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_output_is_sixty_by_four_hundred_twenty(self):
        features = build_features(moving_frames(60))

        self.assertEqual(features.shape, (60, FEATURE_DIM))
        self.assertEqual(features.dtype, np.float32)

    def test_output_never_contains_nan_or_inf(self):
        frames = moving_frames(60)
        frames[:, 75:138] = 0.0          # 왼손 전체 미검출
        frames[20:40, 138:201] = 0.0     # 오른손 중간 끊김
        frames[:, 201:411] = 0.0         # 얼굴 전체 미검출

        features = build_features(frames)

        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

    def test_values_stay_within_clipping_bounds(self):
        features = build_features(moving_frames(60))

        self.assertLessEqual(np.abs(features[:, :208]).max(), CLIP_POSITION)
        self.assertLessEqual(np.abs(features[:, 208:416]).max(), CLIP_VELOCITY)

    def test_first_frame_velocity_is_zero(self):
        features = build_features(moving_frames(10))

        np.testing.assert_array_equal(features[0, 208:416], np.zeros(208, np.float32))

    def test_detection_flags_reflect_missing_parts(self):
        frames = moving_frames(10)
        frames[:, 75:138] = 0.0          # 왼손 없음
        frames[:, 201:411] = 0.0         # 얼굴 없음

        flags = build_features(frames)[0, 416:]

        self.assertAlmostEqual(float(flags[1]), 0.0)   # left hand
        self.assertAlmostEqual(float(flags[2]), 1.0)   # right hand
        self.assertAlmostEqual(float(flags[3]), 0.0)   # face

    def test_translating_the_whole_body_does_not_change_features(self):
        """신체 기준 정규화라 카메라 위치가 바뀌어도 같은 특징이 나와야 한다."""
        frames = moving_frames(30)
        shifted = frames.copy()
        for start in (0, 75, 138, 201):
            end = start + (75 if start == 0 else (63 if start in (75, 138) else 210))
            shifted[:, start:end:3] += 120.0        # x 이동
            shifted[:, start + 1:end:3] += 80.0     # y 이동

        np.testing.assert_allclose(
            build_features(frames), build_features(shifted), atol=1e-4
        )

    def test_rejects_wrong_dimension(self):
        with self.assertRaises(FeatureShapeError):
            build_features(np.zeros((10, 400), np.float32))

    def test_rejects_empty_input(self):
        with self.assertRaises(FeatureShapeError):
            build_features(np.zeros((0, RAW_DIM), np.float32))


if __name__ == "__main__":
    unittest.main()
