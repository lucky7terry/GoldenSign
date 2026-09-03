import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Response

from app.schemas.health import HealthResponse
from app.services.mediapipe_service import keypoint_extraction_available
from app.services.model_service import get_model_health_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def get_health_status(response: Response) -> HealthResponse:
    """서버 상태.

    키포인트 추출이 불가능하면 503 을 돌려준다. 이 상태에서 200 을 주면
    로드밸런서나 오케스트레이터가 이 인스턴스를 정상으로 보고 트래픽을
    계속 보낸다. 프레임을 받아도 아무것도 못 하는 서버다.
    """
    try:
        model_health_status = get_model_health_status()
    except Exception:
        logger.exception("Failed to retrieve model health status")
        model_health_status = {
            "loaded": False,
            "mode": "unavailable",
            "version": "status lookup failed",
        }

    if keypoint_extraction_available():
        status = "ok"
    else:
        status = "degraded"
        response.status_code = 503

    return HealthResponse(
        status=status,
        api="ready",
        model=model_health_status,
        time=datetime.now(timezone.utc),
    )
