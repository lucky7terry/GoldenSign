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

    인식 모델만 없는 경우는 200 이되 status 가 degraded 다. 좌표 수집은
    계속되므로 죽일 이유가 없고, 모델 파일 문제라면 재시작해도 낫지 않아
    503 은 재시작 반복만 만든다.
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

    if not keypoint_extraction_available():
        # 프레임을 받아도 아무것도 못 하는 서버다. 트래픽을 받으면 안 된다.
        status = "degraded"
        response.status_code = 503
    elif not model_health_status["loaded"]:
        # 좌표는 뽑지만 단어는 못 만든다. 503 을 주면 오케스트레이터가
        # 재시작을 반복하는데, 모델 파일이 없어서 그런 거라면 재시작해도
        # 낫지 않는다. 상태만 정직하게 알린다.
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        api="ready",
        model=model_health_status,
        time=datetime.now(timezone.utc),
    )
