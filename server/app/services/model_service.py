from app.config import INCLUDE_KEYPOINTS_IN_RESULT
from app.services.mediapipe_service import (
    MediaPipeProcessingError,
    get_mediapipe_service,
    keypoint_extraction_available,
    keypoint_extraction_error,
)
from app.services.openpose_converter import convert_to_openpose
from app.services.word_segment_service import (
    build_openpose_feature_vector,
    word_store,
)


class FrameValidationError(ValueError):
    """Raised when a client frame message is invalid."""


def _raise_frame_validation_error_if_client_error(
    error: MediaPipeProcessingError,
) -> None:
    message = str(error)
    if message.startswith(
        (
            "Image data",
            "Invalid base64",
            "Decoded image",
            "Failed to decode",
        )
    ):
        raise FrameValidationError(message) from error
    raise error


def get_model_health_status():
    """모델 상태.

    loaded 는 "수어 단어를 인식할 수 있는가"를 뜻한다. MediaPipe 좌표 추출은
    동작하지만 단어를 판정하는 인식 모델은 아직 연결 전이므로 False 다.

    여기서 True 를 반환하면 /health 가 항상 정상을 보고하고, 미니앱은
    인식이 되는 것처럼 표시한다. 실제로는 좌표만 뽑고 있다.

    인식 모델을 붙일 때 이 함수가 실제 로딩 상태를 읽도록 바꿔야 한다.
    (단어 구간 파이프라인 다음 PR에서 한다.)
    """
    if not keypoint_extraction_available():
        return {
            "loaded": False,
            "mode": "unavailable",
            "version": keypoint_extraction_error() or "landmarkers not loaded",
        }

    return {
        "loaded": False,
        "mode": "keypoints_only",
        "version": "mediapipe tasks-0.10.35",
    }


def public_result(result: dict) -> dict:
    """클라이언트로 내보낼 형태.

    keypoints 는 좌표 959개라 result 메시지의 94% 를 차지하는데
    (8,338 -> 479 바이트) 미니앱은 text / confidence / is_final 만 읽는다.
    안경이 폰을 거쳐 받는 구조라 초당 13개면 110KB/s 가 그냥 버려진다.

    서버 로그의 검출률 요약은 원본 result 를 쓰므로 영향받지 않는다.
    """
    if INCLUDE_KEYPOINTS_IN_RESULT:
        return result
    return {key: value for key, value in result.items() if key != "keypoints"}


def _recognition_result(
    keypoints,
    session_id: str | None = None,
    word_generation: int | None = None,
    captured_at_ms: float | None = None,
):
    """한 프레임의 좌표 추출 결과.

    단어 구간이 열려 있으면 이 프레임을 구간 버퍼에 담는다. 열려 있지
    않으면 담지 않는다 - 사용자가 단어를 표시하지 않는 동안의 영상은
    쓸 데가 없다. 단어 판정은 word_end 시점에 구간 전체로 한 번 한다.

    인식 모델이 없으므로 여기서 단어를 주장하지 않는다. text 에 자리표시자
    문자열을 넣으면 그게 그대로 안경 화면에 뜬다("keypoints_extracted").
    인식된 단어가 없음은 null 로 표현하고, 판단은 소비자에게 맡긴다.
    """
    payload = {
        "text": None,
        "confidence": 0.0,
        "is_final": False,
        "keypoints": keypoints.model_dump(by_alias=True),
    }

    if session_id is not None:
        append_result = word_store.append(
            session_id,
            build_openpose_feature_vector(keypoints),
            captured_at_ms,
            word_generation,
        )
        payload["word"] = {
            "buffered": append_result.buffered,
            "frame_count": append_result.frame_count,
            "at_cap": append_result.at_cap,
        }

    return payload


def recognize_frame(frame_message: dict):
    image = frame_message.get("image") or {}
    image_data = image.get("data")

    if not isinstance(image_data, str) or not image_data:
        raise FrameValidationError(
            "Frame image.data is required."
        )

    try:
        extracted_keypoints = (
            get_mediapipe_service().extract_keypoints_from_base64(
                image_data
            )
        )
    except MediaPipeProcessingError as exc:
        _raise_frame_validation_error_if_client_error(exc)

    keypoints = convert_to_openpose(extracted_keypoints)

    return _recognition_result(keypoints)


def recognize_frame_from_image_bytes(
    image_bytes: bytes,
    session_id: str | None = None,
    word_generation: int | None = None,
    captured_at_ms: float | None = None,
):
    if not image_bytes:
        raise FrameValidationError(
            "Frame image data is required."
        )

    try:
        extracted_keypoints = get_mediapipe_service().extract_keypoints_from_bytes(
            image_bytes
        )
    except MediaPipeProcessingError as exc:
        _raise_frame_validation_error_if_client_error(exc)

    keypoints = convert_to_openpose(extracted_keypoints)

    return _recognition_result(
        keypoints,
        session_id,
        word_generation,
        captured_at_ms,
    )


def recognize_frame_from_image(
    image,
    session_id: str | None = None,
    word_generation: int | None = None,
    captured_at_ms: float | None = None,
):
    if image is None or image.size == 0:
        raise FrameValidationError(
            "Frame image data is required."
        )

    try:
        extracted_keypoints = get_mediapipe_service().extract_keypoints_from_image(
            image
        )
    except MediaPipeProcessingError as exc:
        _raise_frame_validation_error_if_client_error(exc)

    keypoints = convert_to_openpose(extracted_keypoints)

    return _recognition_result(
        keypoints,
        session_id,
        word_generation,
        captured_at_ms,
    )
