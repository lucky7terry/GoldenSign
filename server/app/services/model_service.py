def get_model_health_status():
    return {
        "loaded": True,
        "mode": "mock",
        "version": "mock-0.1",
    }


def recognize_frame(frame_message: dict):
    return {
        "text": "안녕하세요",
        "confidence": 0.92,
        "is_final": False,
    }
