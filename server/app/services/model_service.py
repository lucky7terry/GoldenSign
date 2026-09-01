from app.services.mediapipe_service import (
    MediaPipeProcessingError,
    get_mediapipe_service,
)
from app.services.openpose_converter import convert_to_openpose
from app.services.sequence_service import sequence_store


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
    """
    return {
        "loaded": False,
        "mode": "keypoints_only",
        "version": "mediapipe tasks-0.10.35",
    }


def _recognition_result(
    keypoints,
    session_id: str | None = None,
    sequence_generation: int | None = None,
):
    # 인식 모델이 없으므로 단어를 주장하지 않는다. text 에 자리표시자
    # 문자열을 넣으면 그게 그대로 안경 화면에 뜬다("keypoints_extracted").
    # 인식된 단어가 없음은 null 로 표현하고, 판단은 소비자에게 맡긴다.
    payload = {
        "text": None,
        "confidence": 0.0,
        "is_final": False,
        "keypoints": keypoints.model_dump(by_alias=True),
    }

    if session_id is not None:
        sequence_result = sequence_store.append_openpose_result(
            session_id,
            keypoints,
            sequence_generation,
        )
        payload["sequence"] = sequence_result.metadata()

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
    sequence_generation: int | None = None,
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

    return _recognition_result(keypoints, session_id, sequence_generation)


def recognize_frame_from_image(
    image,
    session_id: str | None = None,
    sequence_generation: int | None = None,
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

    return _recognition_result(keypoints, session_id, sequence_generation)
