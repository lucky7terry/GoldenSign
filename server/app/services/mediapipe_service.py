import base64
import binascii
import threading
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeProcessingError(Exception):
    """이미지 디코딩 또는 MediaPipe 처리 실패 예외."""


class MediaPipeService:
    def __init__(self) -> None:
        # 현재 파일 위치:
        # server/app/services/mediapipe_service.py
        #
        # parents[2]:
        # server/
        server_directory = Path(__file__).resolve().parents[2]

        hand_model_path = (
            server_directory
            / "models"
            / "hand_landmarker.task"
        )

        pose_model_path = (
            server_directory
            / "models"
            / "pose_landmarker_lite.task"
        )

        if not hand_model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found: {hand_model_path}"
            )

        if not pose_model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {pose_model_path}"
            )

        # 여러 WebSocket 프레임이 동시에 처리될 때
        # MediaPipe 객체가 동시에 실행되지 않도록 보호한다.
        self._lock = threading.Lock()

        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(hand_model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(pose_model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        self._hand_landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(
                hand_options
            )
        )

        self._pose_landmarker = (
            mp.tasks.vision.PoseLandmarker.create_from_options(
                pose_options
            )
        )

    def decode_base64_image(
        self,
        encoded_image: str,
        max_bytes: int = 262_144,
    ) -> np.ndarray:
        """
        Base64 문자열을 OpenCV BGR 이미지로 변환한다.
        """
        if not encoded_image:
            raise MediaPipeProcessingError(
                "Image data is empty."
            )

        # data:image/jpeg;base64,... 형태도 허용
        if "," in encoded_image:
            encoded_image = encoded_image.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(
                encoded_image,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise MediaPipeProcessingError(
                "Invalid base64 image data."
            ) from exc

        if len(image_bytes) > max_bytes:
            raise MediaPipeProcessingError(
                f"Decoded image exceeds {max_bytes} bytes."
            )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise MediaPipeProcessingError(
                "Failed to decode image."
            )

        return image

    @staticmethod
    def _serialize_hand_landmarks(
        landmarks: Any,
    ) -> list[dict[str, float | None]]:
        return [
            {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": None,
            }
            for landmark in landmarks
        ]

    @staticmethod
    def _serialize_pose_landmarks(
        landmarks: Any,
    ) -> list[dict[str, float | None]]:
        return [
            {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": float(
                    getattr(landmark, "visibility", 0.0)
                ),
            }
            for landmark in landmarks
        ]

    def extract_keypoints_from_image(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        """
        이미지에서 왼손, 오른손, Pose Keypoint를 추출한다.
        """
        if image is None or image.size == 0:
            raise MediaPipeProcessingError(
                "Image is empty."
            )

        try:
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image,
            )

            with self._lock:
                hand_result = self._hand_landmarker.detect(
                    mediapipe_image
                )

                pose_result = self._pose_landmarker.detect(
                    mediapipe_image
                )

        except Exception as exc:
            raise MediaPipeProcessingError(
                f"MediaPipe processing failed: {exc}"
            ) from exc

        left_hand: list[dict[str, float | None]] = []
        right_hand: list[dict[str, float | None]] = []
        pose: list[dict[str, float | None]] = []

        # 손 결과 분류
        for index, landmarks in enumerate(
            hand_result.hand_landmarks
        ):
            serialized = self._serialize_hand_landmarks(
                landmarks
            )

            handedness_name = ""

            if index < len(hand_result.handedness):
                categories = hand_result.handedness[index]

                if categories:
                    handedness_name = (
                        categories[0].category_name.lower()
                    )

            if handedness_name == "left":
                left_hand = serialized
            elif handedness_name == "right":
                right_hand = serialized

        # Pose 결과
        if pose_result.pose_landmarks:
            pose = self._serialize_pose_landmarks(
                pose_result.pose_landmarks[0]
            )

        return {
            "pose": pose,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "pose_detected": bool(pose),
            "left_hand_detected": bool(left_hand),
            "right_hand_detected": bool(right_hand),
        }

    def extract_keypoints_from_base64(
        self,
        encoded_image: str,
    ) -> dict[str, Any]:
        image = self.decode_base64_image(encoded_image)

        return self.extract_keypoints_from_image(image)

    def close(self) -> None:
        self._hand_landmarker.close()
        self._pose_landmarker.close()


mediapipe_service = MediaPipeService()