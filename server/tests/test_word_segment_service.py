"""단어 구간 버퍼와 모델 입력 생성.

이 파일이 지키는 것은 하나다 - 서버가 만드는 (60, 420) 이 학습 노트북이
만든 것과 같은 규칙으로 나오는가. 학습 순서는

    (T, 411) 30fps -> build_features -> (T, 420) -> crop_resample -> (60, 420)

이고, 서버는 프레임이 불규칙하게 도착하므로 앞에 "시간축 30fps 복원"이
한 단계 더 붙는다.
"""

import math
import unittest

import numpy as np

from app.services.feature_service import build_features
from app.services.word_segment_service import (
    OPENPOSE_FEATURE_DIM,
    WordAlreadyStarted,
    WordNotStarted,
    WordSegmentStore,
    WordSessionClosed,
    WordTooShort,
    _effective_fps,
    build_model_input,
    resample_index_half_pixel,
    resample_to_uniform_fps,
)


def _frame(value: float, confidence: float = 0.9) -> list[float]:
    """137개 점을 전부 같은 좌표에 두고 신뢰도를 지정한 411 벡터."""
    row = [0.0] * OPENPOSE_FEATURE_DIM
    for point in range(137):
        row[3 * point] = 500.0 + value
        row[3 * point + 1] = 300.0 + value
        row[3 * point + 2] = confidence
    return row


def _motion_frame(seconds: float, frequency: float = 0.5) -> list[float]:
    """수어를 하는 것처럼 손목과 손가락이 움직이는 프레임.

    특징은 신체 기준으로 정규화된다 - pose 는 목 기준, 손은 **손목** 기준,
    얼굴은 코끝 기준이다. 그래서 몸 전체나 손 전체가 그대로 평행이동하면
    정규화 뒤에는 아무 변화가 없어 속도 특징이 0 이 된다. 실제로 움직이는
    것은 (a) 목 대비 손목, (b) 손목 대비 손가락 둘이다.

    411 배치: pose 25 / 왼손 21 / 오른손 21 / 얼굴 70.
    """
    row = [0.0] * OPENPOSE_FEATURE_DIM
    swing = math.sin(2 * math.pi * frequency * seconds)
    spread = math.cos(2 * math.pi * frequency * seconds)

    for point in range(137):
        x = 500.0 + point * 3.0
        y = 300.0 + point * 2.0
        if point in (4, 7):                     # BODY_25 손목 - 목 대비 이동
            x += 90.0 * swing
            y += 70.0 * spread
        elif 25 <= point < 67:                  # 손가락 - 손목 대비 이동
            finger = (point - 25) % 21
            x += finger * 4.0 * spread
            y += finger * 3.0 * swing
        row[3 * point] = x
        row[3 * point + 1] = y
        row[3 * point + 2] = 0.9
    return row


class HalfPixelConventionTest(unittest.TestCase):
    """학습의 tf.image.resize(bilinear) 좌표 규칙을 고정한다.

    np.linspace(0, T-1, target) 은 양 끝을 맞추는 다른 규칙이다. 업샘플링
    에서는 끝점이 같아 눈에 안 띄지만 다운샘플링에서 갈라진다.
    """

    @staticmethod
    def _expected_positions(count: int, target: int) -> np.ndarray:
        return np.clip(
            (np.arange(target) + 0.5) * (count / target) - 0.5,
            0.0,
            count - 1.0,
        )

    def test_matches_the_training_coordinate_rule(self):
        for count in (25, 30, 60, 143, 600):
            with self.subTest(count=count):
                ramp = np.arange(count, dtype=np.float64).reshape(-1, 1)

                out = resample_index_half_pixel(ramp, 60)

                expected = self._expected_positions(count, 60)
                np.testing.assert_allclose(out[:, 0], expected, atol=1e-12)

    def test_downsampling_excludes_the_outer_half_bin(self):
        """600 -> 60 에서 두 규칙이 갈라진다. linspace 면 0.0 이 나온다."""
        ramp = np.arange(600, dtype=np.float64).reshape(-1, 1)

        out = resample_index_half_pixel(ramp, 60)

        self.assertAlmostEqual(out[0, 0], 4.5)
        self.assertAlmostEqual(out[-1, 0], 594.5)

    def test_identity_when_lengths_match(self):
        ramp = np.arange(60, dtype=np.float64).reshape(-1, 1)

        out = resample_index_half_pixel(ramp, 60)

        np.testing.assert_allclose(out[:, 0], ramp[:, 0], atol=1e-12)

    def test_single_frame_is_repeated(self):
        out = resample_index_half_pixel(np.array([[7.0]]), 60)

        self.assertEqual(out.shape, (60, 1))
        self.assertTrue(np.all(out == 7.0))


class PipelineOrderTest(unittest.TestCase):
    """학습과 같은 순서인지. build_features 가 리샘플보다 먼저다."""

    def test_uniform_30fps_input_matches_the_reference_order(self):
        """이미 30fps 로 균일하면 서버 경로가 학습 순서와 같아야 한다.

        학습:   build_features(전체) -> half-pixel 리샘플 -> (60, 420)
        서버:   30fps 복원(항등) -> build_features -> half-pixel -> (60, 420)
        """
        rng = np.random.default_rng(0)
        count = 143
        raw = np.abs(rng.normal(400.0, 150.0, (count, OPENPOSE_FEATURE_DIM)))
        raw[:, 2::3] = 0.9

        reference = resample_index_half_pixel(build_features(raw), 60)
        actual, on_time, uniform_count = build_model_input(
            [row.tolist() for row in raw],
            [i * 1000.0 / 30.0 for i in range(count)],
        )

        self.assertTrue(on_time)
        self.assertEqual(uniform_count, count)
        self.assertEqual(actual.shape, (60, 420))
        np.testing.assert_allclose(actual, reference, atol=1e-5)

    def test_output_is_float32_and_finite(self):
        frames = [_motion_frame(i / 12.7) for i in range(30)]
        times = [i * 1000.0 / 12.7 for i in range(30)]

        out, _, _ = build_model_input(frames, times)

        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(out)))


class IrregularSpacingTest(unittest.TestCase):
    """[1] 단계가 실제로 하는 일 - 불규칙한 간격을 펴는 것.

    프레임이 불규칙하게 버려지면 궤적이 시간축으로 일그러진다. 촘촘하게
    살아남은 구간은 느리게, 성기게 살아남은 구간은 빠르게 움직인 것처럼
    보인다. 영상 5개 실측에서 이것이 확신도를 0.782 -> 0.637 로 떨어뜨렸고
    한 번은 오답을 냈다.
    """

    @staticmethod
    def _reference(times: list[float]) -> np.ndarray:
        frames = np.asarray(
            [_motion_frame(t / 1000.0) for t in times], dtype=np.float64
        )
        return resample_index_half_pixel(build_features(frames), 60)

    def test_restoring_the_grid_recovers_the_uniform_trajectory(self):
        duration_ms = 2000.0
        uniform_times = [i * 1000.0 / 30.0 for i in range(60)]

        # 불규칙하게 살아남은 프레임: 앞은 촘촘, 뒤는 성기게.
        kept = [i * 1000.0 / 30.0 for i in range(0, 24, 1)]
        kept += [800.0 + i * 200.0 for i in range(7)]
        kept = [t for t in kept if t <= duration_ms]
        frames = [_motion_frame(t / 1000.0) for t in kept]

        truth = self._reference(uniform_times)
        restored, on_time, _ = build_model_input(frames, kept)
        raw_only = resample_index_half_pixel(
            build_features(np.asarray(frames, dtype=np.float64)), 60
        )

        self.assertTrue(on_time)
        restored_error = float(np.abs(restored - truth).mean())
        raw_error = float(np.abs(raw_only - truth).mean())
        self.assertLess(restored_error, raw_error)

    def test_observed_rate_is_identity_on_uniform_input(self):
        """간격이 이미 균일하면 [1] 은 아무것도 하지 않아야 한다.

        도착하지 않은 프레임을 지어내면 실측에서 오히려 손해였다
        (uniform 확신도 0.708 대 0.714).
        """
        count = 40
        frames = [_motion_frame(i / 12.7) for i in range(count)]
        times = [i * 1000.0 / 12.7 for i in range(count)]

        out, on_time = resample_to_uniform_fps(frames, times, fps=None)

        self.assertTrue(on_time)
        self.assertEqual(out.shape[0], count)
        np.testing.assert_allclose(
            out, np.asarray(frames, dtype=np.float64), atol=1e-9
        )

    def test_explicit_fps_still_rescales(self):
        """WORD_SOURCE_FPS 를 주면 그 간격으로 되돌린다(선택 기능)."""
        count = 26
        frames = [_motion_frame(i / 12.7) for i in range(count)]
        times = [i * 1000.0 / 12.7 for i in range(count)]

        out, on_time = resample_to_uniform_fps(frames, times, fps=30.0)

        self.assertTrue(on_time)
        # 2초 구간을 30fps 로 채우면 60프레임 근처가 된다.
        self.assertGreater(out.shape[0], count)


class ConfidenceResampleTest(unittest.TestCase):
    """신뢰도 열은 선형 보간하면 안 된다.

    검출과 미검출 사이를 선형 보간하면 원점 쪽으로 끌린 가짜 좌표에
    CONF_THRESHOLD(0.05) 를 넘는 신뢰도가 붙어서, feature_service 가
    그것을 진짜 검출로 받아들이고 보정하지 않는다.
    """

    def test_a_gap_stays_a_gap(self):
        frames = [
            _frame(0.0, confidence=0.9),
            _frame(0.0, confidence=0.0),
            _frame(20.0, confidence=0.9),
        ]

        out, on_time = resample_to_uniform_fps(
            frames, [0.0, 80.0, 160.0], fps=30.0
        )

        self.assertTrue(on_time)
        confidences = out[:, 2::3]
        # 양옆 중 하나라도 미검출이면 그 프레임은 미검출로 남는다.
        blended = confidences[(confidences > 0.0) & (confidences < 0.9)]
        self.assertEqual(blended.size, 0)

    def test_a_fully_detected_run_keeps_its_confidence(self):
        frames = [_frame(float(i), confidence=0.9) for i in range(5)]

        out, _ = resample_to_uniform_fps(
            frames, [i * 80.0 for i in range(5)], fps=30.0
        )

        np.testing.assert_allclose(out[:, 2::3], 0.9, atol=1e-9)


class TimestampRobustnessTest(unittest.TestCase):
    def test_partially_timestamped_frames_are_not_discarded(self):
        """시각이 앞쪽에만 몰려도 뒤쪽 프레임이 사라지면 안 된다.

        빠진 시각을 버리면 linspace(times[0], times[-1]) 이 앞 절반만
        덮어서 단어의 뒷부분이 출력에 존재하지 않게 된다.
        """
        count = 20
        frames = [_frame(float(i)) for i in range(count)]
        times = [i * 50.0 for i in range(10)] + [None] * 10

        out, on_time = resample_to_uniform_fps(frames, times, fps=30.0)

        self.assertTrue(on_time)
        # 마지막 입력 프레임(값 19)이 출력 끝에 남아 있어야 한다.
        self.assertAlmostEqual(out[-1, 0], 500.0 + 19.0, delta=0.5)

    def test_an_impossible_timestamp_falls_back_to_arrival_order(self):
        """시계 점프나 pts 랩어라운드. 한 프레임이 한 시간 이르면
        정렬이 구간을 통째로 망가뜨린다. 시각을 버리는 쪽이 맞다."""
        count = 20
        frames = [_frame(float(i)) for i in range(count)]
        times = [i * 100.0 for i in range(count)]
        times[7] = -3_600_000.0

        out, on_time = resample_to_uniform_fps(frames, times, fps=30.0)

        self.assertFalse(on_time)
        self.assertEqual(out.shape[0], count)
        np.testing.assert_allclose(out[:, 0], [500.0 + i for i in range(count)])

    def test_identical_timestamps_fall_back(self):
        frames = [_frame(float(i)) for i in range(10)]

        out, on_time = resample_to_uniform_fps(frames, [5.0] * 10, fps=30.0)

        self.assertFalse(on_time)
        self.assertEqual(out.shape[0], 10)

    def test_no_timestamps_fall_back(self):
        frames = [_frame(float(i)) for i in range(10)]

        out, on_time = resample_to_uniform_fps(frames, [None] * 10, fps=30.0)

        self.assertFalse(on_time)
        self.assertEqual(out.shape[0], 10)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            resample_to_uniform_fps([_frame(0.0)] * 5, [0.0, 1.0], fps=30.0)

    def test_non_finite_values_raise(self):
        bad = _frame(0.0)
        bad[0] = float("nan")

        with self.assertRaises(ValueError):
            resample_to_uniform_fps([bad, _frame(1.0)], [0.0, 40.0], fps=30.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            resample_to_uniform_fps([], [], fps=30.0)


class EffectiveFpsTest(unittest.TestCase):
    def test_n_frames_span_n_minus_one_intervals(self):
        # 30프레임을 40ms 간격으로 = 25fps. span 은 29 x 40 = 1160ms.
        self.assertAlmostEqual(_effective_fps(30, 1160.0), 25.0)

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(_effective_fps(1, 100.0))
        self.assertIsNone(_effective_fps(10, 0.0))
        self.assertIsNone(_effective_fps(10, None))


class WordSegmentStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = WordSegmentStore(target_frames=60, min_frames=8)
        self.generation = self.store.start_session("s1")

    def _fill(self, count: int, step_ms: float = 1000.0 / 12.7):
        for i in range(count):
            self.store.append(
                "s1", _motion_frame(i / 12.7), i * step_ms, self.generation
            )

    def test_frames_outside_a_word_are_dropped(self):
        result = self.store.append("s1", _frame(1.0), 0.0, self.generation)

        self.assertFalse(result.buffered)

    def test_start_append_end(self):
        self.store.start_word("s1", self.generation)
        self._fill(25)

        segment = self.store.end_word("s1", self.generation)

        self.assertEqual(segment.sequence.shape, (60, 420))
        self.assertEqual(segment.sequence.dtype, np.float32)
        self.assertEqual(segment.frame_count, 25)
        self.assertEqual(segment.word_index, 1)
        self.assertTrue(segment.resampled_on_time)
        self.assertTrue(np.all(np.isfinite(segment.sequence)))
        # 일정한 간격으로 도착했으므로 [1] 은 항등 - 프레임 수가 그대로다.
        self.assertEqual(segment.uniform_frame_count, 25)

    def test_metadata_reports_the_model_input_shape(self):
        self.store.start_word("s1", self.generation)
        self._fill(25)

        metadata = self.store.end_word("s1", self.generation).metadata()

        self.assertEqual(metadata["target_frames"], 60)
        self.assertEqual(metadata["feature_dim"], 420)
        self.assertTrue(metadata["resampled_on_time"])

    def test_min_frames_boundary(self):
        self.store.start_word("s1", self.generation)
        self._fill(7)
        with self.assertRaises(WordTooShort):
            self.store.end_word("s1", self.generation)
        # 거절해도 구간은 닫혀야 다음 단어를 시작할 수 있다.
        self.assertFalse(self.store.is_word_open("s1"))

        self.store.start_word("s1", self.generation)
        self._fill(8)
        self.assertEqual(
            self.store.end_word("s1", self.generation).frame_count, 8
        )

    def test_double_start_is_rejected(self):
        self.store.start_word("s1", self.generation)

        with self.assertRaises(WordAlreadyStarted):
            self.store.start_word("s1", self.generation)

    def test_end_without_start_is_rejected(self):
        with self.assertRaises(WordNotStarted):
            self.store.end_word("s1", self.generation)

    def test_non_finite_frame_is_rejected(self):
        self.store.start_word("s1", self.generation)
        bad = _frame(0.0)
        bad[3] = float("inf")

        with self.assertRaises(ValueError):
            self.store.append("s1", bad, 0.0, self.generation)

    def test_wrong_feature_dim_is_rejected(self):
        self.store.start_word("s1", self.generation)

        with self.assertRaises(ValueError):
            self.store.append("s1", [0.0] * 10, 0.0, self.generation)

    def test_frame_cap_stops_growth_and_keeps_the_head(self):
        self.store.start_word("s1", self.generation)
        overflow = 25
        self._fill(self.store.frame_cap + overflow)

        segment = self.store.end_word("s1", self.generation)

        self.assertEqual(segment.frame_count, self.store.frame_cap)
        self.assertEqual(segment.dropped_after_cap, overflow)


class GenerationFencingTest(unittest.TestCase):
    """같은 session_id 로 재접속했을 때, 늦게 죽는 연결이 살아있는
    연결의 상태를 건드리면 안 된다."""

    def setUp(self):
        self.store = WordSegmentStore(target_frames=60, min_frames=8)
        self.stale = self.store.start_session("s1")
        self.live = self.store.start_session("s1")

    def _open_live_word(self):
        self.store.start_word("s1", self.live)
        for i in range(20):
            self.store.append("s1", _motion_frame(i / 12.7), i * 78.7, self.live)

    def test_stale_append_is_rejected(self):
        with self.assertRaises(WordSessionClosed):
            self.store.append("s1", _frame(1.0), 0.0, self.stale)

    def test_stale_abort_does_not_touch_the_live_word(self):
        self._open_live_word()

        self.assertFalse(self.store.abort_word("s1", self.stale))
        self.assertTrue(self.store.is_word_open("s1"))

    def test_stale_clear_does_not_drop_the_live_session(self):
        self._open_live_word()

        self.assertFalse(self.store.clear_session("s1", self.stale))
        self.assertTrue(self.store.is_word_open("s1"))
        self.assertEqual(
            self.store.end_word("s1", self.live).frame_count, 20
        )

    def test_live_clear_removes_everything(self):
        self._open_live_word()

        self.assertTrue(self.store.clear_session("s1", self.live))
        self.assertEqual(self.store._states, {})
        with self.assertRaises(WordSessionClosed):
            self.store.append("s1", _frame(1.0), 0.0, self.live)

    def test_live_abort_discards_the_word(self):
        self._open_live_word()

        self.assertTrue(self.store.abort_word("s1", self.live))
        self.assertFalse(self.store.is_word_open("s1"))


if __name__ == "__main__":
    unittest.main()
