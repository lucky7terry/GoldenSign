from app.services.mediapipe_service import mediapipe_service


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
        raise ValueError(
            "Frame image.data is required."
        )

    keypoints = (
        mediapipe_service.extract_keypoints_from_base64(
            image_data
        )
    )

    return {
        "text": "keypoints_extracted",
        "confidence": 0.0,
        "is_final": False,
        "keypoints": keypoints,
    }