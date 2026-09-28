import base64
import binascii
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.config import KEYPOINT_TIMING_INTERVAL_SECONDS
from app.constants import MAX_FRAME_BYTES
from app.services.hand_assignment import assign_hands
from app.services.timing_stats import TimingReporter, TimingStats

logger = logging.getLogger(__name__)


def _round_ms(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


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

        # 주기가 0 이하면 요약 로그를 끈다.
        self._timing_reporter = (
            TimingReporter(KEYPOINT_TIMING_INTERVAL_SECONDS)
            if KEYPOINT_TIMING_INTERVAL_SECONDS > 0
            else None
        )

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

    # MediaPipe Pose 의 얼굴 관련 랜드마크 번호.
    _POSE_NOSE = 0
    _POSE_LEFT_EYE = 2
    _POSE_RIGHT_EYE = 5
    _POSE_LEFT_EAR = 7
    _POSE_RIGHT_EAR = 8

    # 얼굴 폭의 몇 배를 자를지. 2.2 면 실측 기준 얼굴이 크롭의 45% 를
    # 차지한다 - 검출기가 편하게 보는 비율이다. 너무 좁게 자르면 턱선과
    # 이마가 잘려 70점 중 윤곽점이 틀어진다.
    _FACE_CROP_RATIO = 2.2
    # 크롭 한 변의 최소 픽셀. 얼굴이 아주 작게 잡힌 프레임에서 몇 픽셀짜리
    # 이미지를 넣으면 검출기가 아무것도 못 한다.
    _FACE_CROP_MIN_SIDE = 96

    @classmethod
    def _face_crop_box(
        cls,
        pose_result: Any,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        """pose 가 찾은 얼굴 둘레를 정사각형으로 돌려준다.

        정사각형인 이유는 검출기가 어차피 정사각형으로 리사이즈하기
        때문이다. 직사각형을 넣으면 그 안에서 또 레터박스가 생겨 지금
        고치려는 문제가 그대로 남는다.

        pose 가 없거나 얼굴 점을 못 찾았으면 None - 호출자가 원본으로
        떨어진다.
        """
        if not pose_result.pose_landmarks:
            return None
        landmarks = pose_result.pose_landmarks[0]
        needed = (
            cls._POSE_NOSE,
            cls._POSE_LEFT_EYE,
            cls._POSE_RIGHT_EYE,
            cls._POSE_LEFT_EAR,
            cls._POSE_RIGHT_EAR,
        )
        if len(landmarks) <= max(needed):
            return None

        points = [
            (
                landmarks[index].x * image_width,
                landmarks[index].y * image_height,
            )
            for index in needed
        ]
        if not all(
            math.isfinite(x) and math.isfinite(y) for x, y in points
        ):
            return None

        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        center_x = 0.5 * (min(xs) + max(xs))
        center_y = 0.5 * (min(ys) + max(ys))

        # 귀 사이 거리를 얼굴 폭으로 본다. 옆모습이면 두 귀가 겹쳐 0 에
        # 가까워지므로, 눈·코까지 포함한 점들의 퍼짐을 같이 본다.
        ear_span = math.dist(points[3], points[4])
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        side = cls._FACE_CROP_RATIO * max(ear_span, spread)
        side = max(side, float(cls._FACE_CROP_MIN_SIDE))
        if not math.isfinite(side) or side <= 0:
            return None

        half = side / 2.0
        left = int(round(center_x - half))
        top = int(round(center_y - half))
        right = int(round(center_x + half))
        bottom = int(round(center_y + half))

        # 화면 밖으로 나가면 잘라낸다. 이때 정사각형이 깨지지만, 얼굴이
        # 화면 가장자리에 붙은 경우라 어차피 일부가 안 보인다.
        left = max(0, min(left, image_width - 1))
        top = max(0, min(top, image_height - 1))
        right = max(left + 1, min(right, image_width))
        bottom = max(top + 1, min(bottom, image_height))

        if right - left < 2 or bottom - top < 2:
            return None
        return left, top, right, bottom

    @staticmethod
    def _restore_face_to_frame(
        face: list[dict[str, float | None]],
        face_box: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> list[dict[str, float | None]]:
        """크롭 기준 정규화 좌표를 원본 프레임 기준으로 되돌린다.

        이 변환을 빼먹으면 얼굴 좌표가 화면 좌상단 근처에 몰린 채로
        나가고, 그게 그대로 모델 입력이 된다. 검출은 성공했는데 값은
        틀린 - 눈에 잘 안 띄는 실패다.
        """
        left, top, right, bottom = face_box
        crop_width = float(right - left)
        crop_height = float(bottom - top)
        for landmark in face:
            landmark["x"] = (
                left + float(landmark["x"]) * crop_width
            ) / image_width
            landmark["y"] = (
                top + float(landmark["y"]) * crop_height
            ) / image_height
            # z 는 x 와 같은 단위(폭 기준)라 같은 비율로 줄인다.
            if landmark.get("z") is not None:
                landmark["z"] = (
                    float(landmark["z"]) * crop_width / image_width
                )
        return face

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

    def _report_timing_if_due(self) -> None:
        """주기가 되면 모아둔 계측을 한 줄로 남기고 버퍼를 비운다."""
        if self._timing_reporter is None:
            return

        if not self._timing_reporter.should_report():
            return

        lock_wait = self.lock_wait_stats.snapshot()
        hand = self.hand_detect_stats.snapshot()
        pose = self.pose_detect_stats.snapshot()
        face = self.face_detect_stats.snapshot()

        # 다음 구간은 새로 센다.
        self.lock_wait_stats.reset()
        self.hand_detect_stats.reset()
        self.pose_detect_stats.reset()
        self.face_detect_stats.reset()

        if lock_wait["count"] == 0:
            return

        logger.info(
            "MediaPipe timing",
            extra={
                "frames": lock_wait["count"],
                "lock_wait_avg_ms": _round_ms(lock_wait["avg_ms"]),
                "lock_wait_p95_ms": _round_ms(lock_wait["p95_ms"]),
                "hand_avg_ms": _round_ms(hand["avg_ms"]),
                "hand_p95_ms": _round_ms(hand["p95_ms"]),
                "pose_avg_ms": _round_ms(pose["avg_ms"]),
                "pose_p95_ms": _round_ms(pose["p95_ms"]),
                "face_avg_ms": _round_ms(face["avg_ms"]),
                "face_p95_ms": _round_ms(face["p95_ms"]),
            },
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

        # 얼굴 크롭이 검출 단계에서 픽셀 좌표를 쓰므로 미리 구한다.
        image_height, image_width = image.shape[:2]

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

                # 얼굴은 원본 프레임 그대로 넣으면 못 찾는다. 검출기가
                # 입력을 정사각형으로 리사이즈하는데 16:9 를 넣으면 위아래가
                # 레터박스로 채워져 얼굴이 더 작아진다. 실측: 1920x1080
                # 원본에서 임계값을 0.1 까지 낮춰도 실패, 축소(960x540,
                # 633x356)도 실패, 중앙 크롭(960x1080)에서만 478점 검출.
                # 축소가 안 듣는 이유는 얼굴도 같이 작아져 비율이 그대로라서다.
                #
                # 그래서 pose 가 찾아 둔 얼굴 위치를 써서 잘라 넣는다.
                # 부수 효과로 검출기가 보는 이미지가 작아져 더 빠르다.
                face_box = self._face_crop_box(
                    pose_result, image_width, image_height
                )
                #
                # 크롭에서 못 찾았을 때 원본으로 다시 보지 않는다. 원본은
                # 위에서 실패가 확인된 바로 그 조건이라(학습 영상 250개 검출률
                # 0.00) 비용만 두 배가 된다. 원본은 pose 가 없어 크롭을 못
                # 만들 때만 쓴다. 이렇게 하면 프레임당 검출이 항상 한 번이라
                # face_detect_stats 평균에 한 번/두 번 돈 프레임이 섞이지 않는다.
                if face_box is not None:
                    left, top, right, bottom = face_box
                    face_result = self._face_landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(
                                rgb_image[top:bottom, left:right]
                            ),
                        )
                    )
                else:
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

        # Face 결과 — 크롭해서 넣었으면 좌표가 크롭 기준이므로 되돌린다.
        if face_result.face_landmarks:
            face = self._serialize_face_landmarks(
                face_result.face_landmarks[0]
            )
            if face_box is not None:
                face = self._restore_face_to_frame(
                    face, face_box, image_width, image_height
                )

        self._report_timing_if_due()

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
