import base64
import binascii
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.constants import MAX_FRAME_BYTES

logger = logging.getLogger(__name__)


class MediaPipeProcessingError(Exception):
    """이미지 디코딩 또는 MediaPipe 처리 실패 예외."""


def decode_base64_image_data(encoded_image: str) -> bytes:
    if "," in encoded_image:
        encoded_image = encoded_image.split(",", 1)[1]

    try:
        return base64.b64decode(
            encoded_image,
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise MediaPipeProcessingError(
            "Invalid base64 image data."
        ) from exc


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

        face_model_path = (
            server_directory
            / "models"
            / "face_landmarker.task"
        )

        if not hand_model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found: {hand_model_path}"
            )

        if not pose_model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {pose_model_path}"
            )

        if not face_model_path.exists():
            raise FileNotFoundError(
                f"Face model not found: {face_model_path}"
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

        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(face_model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
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

        self._face_landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                face_options
            )
        )

    def decode_base64_image(
        self,
        encoded_image: str,
        max_bytes: int = MAX_FRAME_BYTES,
    ) -> np.ndarray:
        """
        Base64 문자열을 OpenCV BGR 이미지로 변환한다.
        """
        if not encoded_image:
            raise MediaPipeProcessingError(
                "Image data is empty."
            )

        # data:image/jpeg;base64,... 형태도 허용
        image_bytes = decode_base64_image_data(encoded_image)

        if len(image_bytes) > max_bytes:
            raise MediaPipeProcessingError(
                f"Decoded image exceeds {max_bytes} bytes."
            )

        return self.decode_image_bytes(image_bytes)

    @staticmethod
    def decode_image_bytes(
        image_bytes: bytes,
        max_bytes: int = MAX_FRAME_BYTES,
    ) -> np.ndarray:
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

    @staticmethod
    def _serialize_face_landmarks(
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
    def _handedness_label_and_score(
        handedness: Any,
    ) -> tuple[str, float | None]:
        if not handedness:
            return "", None

        category = handedness[0]
        return (
            getattr(category, "category_name", "").lower(),
            getattr(category, "score", None),
        )

    @staticmethod
    def _score_or_default(score: float | None) -> float:
        if score is None:
            return -1.0
        return score

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

                face_result = self._face_landmarker.detect(
                    mediapipe_image
                )

        except Exception as exc:
            raise MediaPipeProcessingError(
                f"MediaPipe processing failed: {exc}"
            ) from exc

        left_hand: list[dict[str, float | None]] = []
        right_hand: list[dict[str, float | None]] = []
        face: list[dict[str, float | None]] = []
        pose: list[dict[str, float | None]] = []
        image_height, image_width = image.shape[:2]

        # 손 결과 분류
        hand_candidates: list[dict[str, Any]] = []
        for index, landmarks in enumerate(
            hand_result.hand_landmarks
        ):
            serialized = self._serialize_hand_landmarks(
                landmarks
            )

            handedness_name = ""
            handedness_score = None

            if index < len(hand_result.handedness):
                handedness_name, handedness_score = (
                    self._handedness_label_and_score(
                        hand_result.handedness[index]
                    )
                )

            hand_candidates.append(
                {
                    "index": index,
                    "label": handedness_name,
                    "score": handedness_score,
                    "landmarks": serialized,
                }
            )

        selected_hand_indexes: set[int] = set()

        for label in ("left", "right"):
            labeled_candidates = [
                candidate
                for candidate in hand_candidates
                if candidate["label"] == label
            ]
            if not labeled_candidates:
                continue
            if len(labeled_candidates) > 1:
                logger.warning(
                    "Duplicate %s hand detected; preserving higher score",
                    label,
                )
            selected_candidate = max(
                labeled_candidates,
                key=lambda candidate: self._score_or_default(candidate["score"]),
            )
            selected_hand_indexes.add(selected_candidate["index"])
            if label == "left":
                left_hand = selected_candidate["landmarks"]
            else:
                right_hand = selected_candidate["landmarks"]

        for label in ("left", "right"):
            if label == "left" and left_hand:
                continue
            if label == "right" and right_hand:
                continue

            remaining_candidates = [
                candidate
                for candidate in hand_candidates
                if candidate["index"] not in selected_hand_indexes
            ]
            if not remaining_candidates:
                continue
            selected_candidate = max(
                remaining_candidates,
                key=lambda candidate: self._score_or_default(candidate["score"]),
            )
            selected_hand_indexes.add(selected_candidate["index"])
            logger.warning(
                "Reassigned duplicate or unlabeled hand to missing %s hand",
                label,
            )
            if label == "left":
                left_hand = selected_candidate["landmarks"]
            else:
                right_hand = selected_candidate["landmarks"]

        # Pose 결과
        if pose_result.pose_landmarks:
            pose = self._serialize_pose_landmarks(
                pose_result.pose_landmarks[0]
            )

        # Face 결과
        if face_result.face_landmarks:
            face = self._serialize_face_landmarks(
                face_result.face_landmarks[0]
            )

        return {
            "face": face,
            "pose": pose,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "face_detected": bool(face),
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

    def extract_keypoints_from_bytes(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        image = self.decode_image_bytes(image_bytes)

        return self.extract_keypoints_from_image(image)

    def close(self) -> None:
        self._hand_landmarker.close()
        self._pose_landmarker.close()
        self._face_landmarker.close()


_mediapipe_service: MediaPipeService | None = None
_mediapipe_service_lock = threading.Lock()


def get_mediapipe_service() -> MediaPipeService:
    global _mediapipe_service
    if _mediapipe_service is None:
        with _mediapipe_service_lock:
            if _mediapipe_service is None:
                _mediapipe_service = MediaPipeService()
    return _mediapipe_service
