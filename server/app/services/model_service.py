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
    return {
        "loaded": True,
        "mode": "mediapipe",
        "version": "tasks-0.10.35",
    }


def _recognition_result(
    keypoints,
    session_id: str | None = None,
    sequence_generation: int | None = None,
):
    payload = {
        "text": "keypoints_extracted",
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
