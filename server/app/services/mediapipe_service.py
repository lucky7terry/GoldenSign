import base64
import binascii
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.constants import MAX_FRAME_BYTES
from app.services.hand_assignment import assign_hands
from app.services.timing_stats import TimingStats

logger = logging.getLogger(__name__)


class MediaPipeProcessingError(Exception):
    """이미지 디코딩 또는 MediaPipe 처리 실패 예외."""


class MediaPipeUnavailableError(RuntimeError):
    """랜드마커 모델을 로드하지 못해 키포인트 추출이 불가능한 상태.

    모델 파일이 없는 경우가 대부분이다. 프레임마다 다시 시도해도 결과가
    같으므로 재시도 대상이 아니다. 클라이언트에는 retryable=false 로 나가야 한다.
    """


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
                f"Hand model not found: {hand_model_path}. "
                "Run `python scripts/download_mediapipe_models.py` "
                "from the server directory."
            )

        if not pose_model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {pose_model_path}. "
                "Run `python scripts/download_mediapipe_models.py` "
                "from the server directory."
            )

        if not face_model_path.exists():
            raise FileNotFoundError(
                f"Face model not found: {face_model_path}. "
                "Run `python scripts/download_mediapipe_models.py` "
                "from the server directory."
            )

        # 여러 WebSocket 프레임이 동시에 처리될 때
        # MediaPipe 객체가 동시에 실행되지 않도록 보호한다.
        self._lock = threading.Lock()

        # 프레임당 지연을 나눠 보기 위한 계측 버퍼 (수집만; 출력은 별도).
        self.lock_wait_stats = TimingStats()
        self.hand_detect_stats = TimingStats()
        self.pose_detect_stats = TimingStats()
        self.face_detect_stats = TimingStats()

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

            # 락을 기다린 시간과 실제 추론 시간을 분리해서 본다.
            lock_requested_at = time.perf_counter()

            with self._lock:
                lock_acquired_at = time.perf_counter()
                self.lock_wait_stats.record(
                    (lock_acquired_at - lock_requested_at) * 1000.0
                )

                # 세 검출기 각각의 소요 시간 — 어느 쪽이 지배적인지 보려고 나눈다.
                hand_started_at = time.perf_counter()
                hand_result = self._hand_landmarker.detect(
                    mediapipe_image
                )
                pose_started_at = time.perf_counter()
                self.hand_detect_stats.record(
                    (pose_started_at - hand_started_at) * 1000.0
                )

                pose_result = self._pose_landmarker.detect(
                    mediapipe_image
                )
                face_started_at = time.perf_counter()
                self.pose_detect_stats.record(
                    (face_started_at - pose_started_at) * 1000.0
                )

                face_result = self._face_landmarker.detect(
                    mediapipe_image
                )
                self.face_detect_stats.record(
                    (time.perf_counter() - face_started_at) * 1000.0
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

        # Pose 결과 — 손 좌우 배정에 손목 좌표를 쓰므로 먼저 계산한다.
        if pose_result.pose_landmarks:
            pose = self._serialize_pose_landmarks(
                pose_result.pose_landmarks[0]
            )

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

        # 안경 카메라에서는 검출기 handedness의 좌우가 뒤집히므로
        # Pose 손목 좌표를 기준으로 배정한다(app/services/hand_assignment.py).
        assignment = assign_hands(hand_candidates, pose)
        if assignment["left"] is not None:
            left_hand = assignment["left"]["landmarks"]
        if assignment["right"] is not None:
            right_hand = assignment["right"]["landmarks"]

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
_initialization_error: MediaPipeUnavailableError | None = None
_mediapipe_service_lock = threading.Lock()


def get_mediapipe_service() -> MediaPipeService:
    """랜드마커 서비스를 돌려준다. 로드에 실패했으면 매번 같은 예외를 던진다.

    실패를 기억하지 않으면 프레임마다 모델 로딩을 다시 시도하게 된다.
    모델 파일이 없는 상태에서 초당 수십 번 같은 실패를 반복하는 셈이라,
    한 번 실패하면 그 사실을 붙잡고 있는다. 파일을 채운 뒤에는 서버를
    재시작해야 한다 — 조용히 반쯤 살아나는 것보다 낫다.
    """
    global _mediapipe_service, _initialization_error

    if _mediapipe_service is not None:
        return _mediapipe_service

    with _mediapipe_service_lock:
        if _mediapipe_service is not None:
            return _mediapipe_service
        if _initialization_error is not None:
            raise _initialization_error
        try:
            _mediapipe_service = MediaPipeService()
        except Exception as exc:
            _initialization_error = MediaPipeUnavailableError(str(exc))
            raise _initialization_error from exc

    return _mediapipe_service


def preload_mediapipe_service() -> bool:
    """기동 시 모델을 미리 로드한다. 실패해도 예외를 밖으로 내지 않는다.

    첫 프레임이 들어올 때 로딩하면 그 프레임이 타임아웃되고, 모델이 없으면
    무엇이 잘못됐는지 로그에도 안 남는다. 기동 시점에 크게 한 번 알린다.
    """
    try:
        get_mediapipe_service()
    except MediaPipeUnavailableError as exc:
        logger.error(
            "MediaPipe landmarkers unavailable; keypoint extraction is disabled",
            extra={"error": str(exc)},
        )
        return False
    logger.info("MediaPipe landmarkers loaded")
    return True


def keypoint_extraction_available() -> bool:
    return _mediapipe_service is not None


def keypoint_extraction_error() -> str | None:
    if _initialization_error is None:
        return None
    return str(_initialization_error)
