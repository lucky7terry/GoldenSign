from app.services.mediapipe_service import (
    MediaPipeProcessingError,
    get_mediapipe_service,
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
    return {
        "loaded": True,
        "mode": "mediapipe",
        "version": "tasks-0.10.35",
    }


def recognize_frame(frame_message: dict):
    image = frame_message.get("image") or {}
    image_data = image.get("data")

    if not isinstance(image_data, str) or not image_data:
        raise FrameValidationError(
            "Frame image.data is required."
        )

    try:
        keypoints = (
            get_mediapipe_service().extract_keypoints_from_base64(
                image_data
            )
        )
    except MediaPipeProcessingError as exc:
        _raise_frame_validation_error_if_client_error(exc)

    return {
        "text": "keypoints_extracted",
        "confidence": 0.0,
        "is_final": False,
        "keypoints": keypoints,
    }


def recognize_frame_from_image_bytes(image_bytes: bytes):
    if not image_bytes:
        raise FrameValidationError(
            "Frame image data is required."
        )

    try:
        keypoints = get_mediapipe_service().extract_keypoints_from_bytes(
            image_bytes
        )
    except MediaPipeProcessingError as exc:
        _raise_frame_validation_error_if_client_error(exc)

    return {
        "text": "keypoints_extracted",
        "confidence": 0.0,
        "is_final": False,
        "keypoints": keypoints,
    }
