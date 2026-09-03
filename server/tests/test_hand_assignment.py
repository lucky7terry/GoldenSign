import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.hand_assignment import (
    POSE_LEFT_WRIST_INDEX,
    POSE_RIGHT_WRIST_INDEX,
    assign_by_pose_wrists,
    assign_hands,
)


def _hand(index: int, x: float, y: float, label: str = "", score=None):
    """손목(0번)만 의미 있는 최소 손 후보."""
    return {
        "index": index,
        "label": label,
        "score": score,
        "landmarks": [{"x": x, "y": y, "z": 0.0, "visibility": None}]
        + [{"x": x, "y": y, "z": 0.0, "visibility": None}] * 20,
    }


def _pose(left_x=0.3, right_x=0.7, visibility=0.9, y=0.5):
    pose = [
        {"x": 0.5, "y": 0.1, "z": 0.0, "visibility": visibility}
        for _ in range(33)
    ]
    pose[POSE_LEFT_WRIST_INDEX] = {
        "x": left_x, "y": y, "z": 0.0, "visibility": visibility,
    }
    pose[POSE_RIGHT_WRIST_INDEX] = {
        "x": right_x, "y": y, "z": 0.0, "visibility": visibility,
    }
    return pose


class HandAssignmentTest(unittest.TestCase):
    def test_pose_wrists_override_flipped_handedness_labels(self):
        # 검출기는 좌우를 뒤집어 라벨링했지만(안경 카메라 상황),
        # Pose 손목 좌표는 올바른 쪽을 가리킨다.
        near_left = _hand(0, 0.31, 0.5, label="right", score=0.99)
        near_right = _hand(1, 0.69, 0.5, label="left", score=0.99)

        assignment = assign_hands([near_left, near_right], _pose())

        self.assertEqual(assignment["left"]["index"], 0)
        self.assertEqual(assignment["right"]["index"], 1)

    def test_swapped_input_order_still_resolves_correctly(self):
        near_right = _hand(0, 0.72, 0.5)
        near_left = _hand(1, 0.28, 0.5)

        assignment = assign_hands([near_right, near_left], _pose())

        self.assertEqual(assignment["left"]["index"], 1)
        self.assertEqual(assignment["right"]["index"], 0)

    def test_single_hand_goes_to_nearest_wrist(self):
        assignment = assign_hands([_hand(0, 0.72, 0.5, label="left")], _pose())

        self.assertIsNone(assignment["left"])
        self.assertEqual(assignment["right"]["index"], 0)

    def test_two_hands_never_collapse_onto_one_side(self):
        # 두 손이 모두 오른손목 쪽에 몰려 있어도 한쪽으로 뭉치지 않는다.
        assignment = assign_hands(
            [_hand(0, 0.68, 0.5), _hand(1, 0.74, 0.5)],
            _pose(),
        )

        self.assertIsNotNone(assignment["left"])
        self.assertIsNotNone(assignment["right"])
        self.assertNotEqual(
            assignment["left"]["index"],
            assignment["right"]["index"],
        )

    def test_low_visibility_wrists_fall_back_to_handedness(self):
        invisible_pose = _pose(visibility=0.1)

        self.assertIsNone(
            assign_by_pose_wrists([_hand(0, 0.3, 0.5)], invisible_pose)
        )

        assignment = assign_hands(
            [_hand(0, 0.3, 0.5, label="left", score=0.9)],
            invisible_pose,
        )
        self.assertEqual(assignment["left"]["index"], 0)

    def test_missing_pose_falls_back_to_handedness(self):
        assignment = assign_hands(
            [_hand(0, 0.3, 0.5, label="right", score=0.9)],
            [],
        )

        self.assertEqual(assignment["right"]["index"], 0)
        self.assertIsNone(assignment["left"])

    def test_one_visible_wrist_still_discriminates(self):
        pose = _pose()
        pose[POSE_RIGHT_WRIST_INDEX]["visibility"] = 0.1

        assignment = assign_hands(
            [_hand(0, 0.71, 0.5), _hand(1, 0.29, 0.5)],
            pose,
        )

        self.assertEqual(assignment["left"]["index"], 1)
        self.assertEqual(assignment["right"]["index"], 0)

    def test_no_hands_returns_empty_assignment(self):
        assignment = assign_hands([], _pose())

        self.assertIsNone(assignment["left"])
        self.assertIsNone(assignment["right"])


if __name__ == "__main__":
    unittest.main()
