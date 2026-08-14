import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.openpose import OpenPosePerson, OpenPoseResult
from app.services.sequence_service import (
    OPENPOSE_FEATURE_DIM,
    SequenceSessionClosed,
    SlidingWindowSequenceStore,
    build_openpose_feature_vector,
)


def _openpose_result(value: float = 1.0) -> OpenPoseResult:
    return OpenPoseResult(
        people=OpenPosePerson(
            pose_keypoints_2d=[value] * (25 * 3),
            hand_left_keypoints_2d=[value + 1] * (21 * 3),
            hand_right_keypoints_2d=[value + 2] * (21 * 3),
        )
    )


class SequenceServiceTest(unittest.TestCase):
    def test_build_openpose_feature_vector_uses_201_upper_body_values(self):
        feature_vector = build_openpose_feature_vector(_openpose_result())

        self.assertEqual(len(feature_vector), OPENPOSE_FEATURE_DIM)
        self.assertEqual(OPENPOSE_FEATURE_DIM, 201)
        self.assertEqual(feature_vector[:3], [1.0, 1.0, 1.0])
        self.assertEqual(feature_vector[75:78], [2.0, 2.0, 2.0])
        self.assertEqual(feature_vector[-3:], [3.0, 3.0, 3.0])

    def test_build_openpose_feature_vector_accepts_contract_dict(self):
        payload = _openpose_result().model_dump(by_alias=True)

        feature_vector = build_openpose_feature_vector(payload)

        self.assertEqual(len(feature_vector), 201)
        self.assertEqual(feature_vector[:3], [1.0, 1.0, 1.0])
        self.assertEqual(feature_vector[75:78], [2.0, 2.0, 2.0])
        self.assertEqual(feature_vector[-3:], [3.0, 3.0, 3.0])

    def test_default_window_waits_for_sixty_frames(self):
        store = SlidingWindowSequenceStore()

        for index in range(59):
            result = store.append_openpose_result(
                "session-1",
                _openpose_result(float(index)),
            )
            self.assertFalse(result.ready)
            self.assertEqual(result.window_size, 60)

        result = store.append_openpose_result("session-1", _openpose_result(60.0))

        self.assertTrue(result.ready)
        self.assertEqual(result.window_size, 60)
        self.assertEqual(result.frame_count, 60)
        self.assertEqual(result.window_index, 1)
        self.assertEqual(len(result.sequence), 60)

    def test_sliding_window_returns_sequence_when_window_is_full(self):
        store = SlidingWindowSequenceStore(window_size=3, stride=2)

        first = store.append_openpose_result("session-1", _openpose_result(1.0))
        second = store.append_openpose_result("session-1", _openpose_result(2.0))
        third = store.append_openpose_result("session-1", _openpose_result(3.0))
        fourth = store.append_openpose_result("session-1", _openpose_result(4.0))
        fifth = store.append_openpose_result("session-1", _openpose_result(5.0))

        self.assertFalse(first.ready)
        self.assertFalse(second.ready)
        self.assertTrue(third.ready)
        self.assertEqual(third.window_index, 1)
        self.assertEqual(len(third.sequence), 3)
        self.assertFalse(fourth.ready)
        self.assertTrue(fifth.ready)
        self.assertEqual(fifth.window_index, 2)
        self.assertEqual(fifth.sequence[0][0], 3.0)
        self.assertEqual(fifth.sequence[-1][0], 5.0)

    def test_sessions_keep_independent_sequence_buffers(self):
        store = SlidingWindowSequenceStore(window_size=2, stride=1)

        first_a = store.append_openpose_result("session-1", _openpose_result(1.0))
        first_b = store.append_openpose_result("session-2", _openpose_result(10.0))
        second_a = store.append_openpose_result("session-1", _openpose_result(2.0))
        second_b = store.append_openpose_result("session-2", _openpose_result(20.0))

        self.assertFalse(first_a.ready)
        self.assertFalse(first_b.ready)
        self.assertTrue(second_a.ready)
        self.assertTrue(second_b.ready)
        self.assertEqual(second_a.frame_count, 2)
        self.assertEqual(second_b.frame_count, 2)
        self.assertEqual(second_a.window_index, 1)
        self.assertEqual(second_b.window_index, 1)
        self.assertEqual(second_a.sequence[0][0], 1.0)
        self.assertEqual(second_b.sequence[0][0], 10.0)

    def test_clear_session_resets_sequence_state(self):
        store = SlidingWindowSequenceStore(window_size=2, stride=1)

        self.assertFalse(
            store.append_openpose_result("session-1", _openpose_result()).ready
        )
        store.clear_session("session-1")
        result = store.append_openpose_result("session-1", _openpose_result())

        self.assertFalse(result.ready)
        self.assertEqual(result.frame_count, 1)

    def test_clear_session_resets_window_index(self):
        store = SlidingWindowSequenceStore(window_size=2, stride=1)

        first_window = store.append_openpose_result("session-1", _openpose_result())
        self.assertFalse(first_window.ready)
        first_window = store.append_openpose_result("session-1", _openpose_result())
        self.assertTrue(first_window.ready)
        self.assertEqual(first_window.window_index, 1)

        store.clear_session("session-1")
        store.append_openpose_result("session-1", _openpose_result())
        second_window = store.append_openpose_result("session-1", _openpose_result())

        self.assertTrue(second_window.ready)
        self.assertEqual(second_window.window_index, 1)

    def test_stale_generation_cannot_append_after_clear_session(self):
        store = SlidingWindowSequenceStore(window_size=2, stride=1)
        generation = store.start_session("session-1")

        store.clear_session("session-1")

        with self.assertRaises(SequenceSessionClosed):
            store.append_openpose_result(
                "session-1",
                _openpose_result(),
                generation,
            )


if __name__ == "__main__":
    unittest.main()
