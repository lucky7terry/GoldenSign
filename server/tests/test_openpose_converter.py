import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.keypoint import ExtractedKeypoints, Landmark
from app.services.openpose_converter import convert_to_openpose


class OpenPoseConverterTest(unittest.TestCase):
    def test_convert_to_openpose_preserves_contract_lengths(self):
        keypoints = ExtractedKeypoints(
            pose=[
                Landmark(x=index / 100, y=index / 200, z=index / 300, visibility=0.9)
                for index in range(33)
            ],
            left_hand=[
                Landmark(x=index / 100, y=index / 200, z=index / 300)
                for index in range(21)
            ],
            right_hand=[
                Landmark(x=index / 100, y=index / 200, z=index / 300)
                for index in range(21)
            ],
            face=[
                Landmark(x=index / 1000, y=index / 2000, z=index / 3000)
                for index in range(478)
            ],
            image_width=640,
            image_height=480,
        )

        result = convert_to_openpose(keypoints)
        person = result.people

        self.assertEqual(result.version, 1.3)
        self.assertEqual(len(person.pose_keypoints_2d), 25 * 3)
        self.assertEqual(len(person.pose_keypoints_3d), 25 * 4)
        self.assertEqual(len(person.face_keypoints_2d), 70 * 3)
        self.assertEqual(len(person.face_keypoints_3d), 70 * 4)
        self.assertEqual(len(person.hand_left_keypoints_2d), 21 * 3)
        self.assertEqual(len(person.hand_left_keypoints_3d), 21 * 4)
        self.assertEqual(len(person.hand_right_keypoints_2d), 21 * 3)
        self.assertEqual(len(person.hand_right_keypoints_3d), 21 * 4)
        self.assertEqual(
            person.face_keypoints_2d[:3],
            [234 / 1000 * 640, 234 / 2000 * 480, 1.0],
        )
        self.assertEqual(
            person.face_keypoints_3d[:4],
            [234 / 1000, 234 / 2000, 234 / 3000, 1.0],
        )

    def test_convert_to_openpose_zero_pads_missing_face(self):
        result = convert_to_openpose(ExtractedKeypoints())
        person = result.people

        self.assertEqual(person.face_keypoints_2d, [0.0] * (70 * 3))
        self.assertEqual(person.face_keypoints_3d, [0.0] * (70 * 4))

    def test_convert_to_openpose_serializes_camera_aliases(self):
        result = convert_to_openpose(ExtractedKeypoints())

        payload = result.model_dump(by_alias=True)

        self.assertIn("version", payload)
        self.assertIn("people", payload)
        self.assertIn("camparam", payload)
        self.assertIsInstance(payload["people"], dict)
        self.assertEqual(
            payload["camparam"],
            {
                "Intrinsics": {"data": ""},
                "CameraMatrix": {"data": ""},
                "Distortion": {"rows": "0", "data": ""},
            },
        )


if __name__ == "__main__":
    unittest.main()
