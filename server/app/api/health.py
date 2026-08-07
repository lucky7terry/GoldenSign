import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas.health import HealthResponse
from app.services.model_service import get_model_health_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def get_health_status() -> HealthResponse:
    try:
        model_health_status = get_model_health_status()
    except Exception:
        logger.exception("Failed to retrieve model health status")
        model_health_status = {
            "loaded": False,
            "status": "unavailable",
        }

    return HealthResponse(
        status="ok",
        api="ready",
        model=model_health_status,
        time=datetime.now(timezone.utc),
    )
