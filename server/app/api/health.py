from datetime import datetime, timezone

from fastapi import APIRouter

from app.services.model_service import get_model_health_status

router = APIRouter()


@router.get("/health")
def get_health_status():
    model_health_status = get_model_health_status()

    return {
        "status": "ok",
        "api": "ready",
        "model": model_health_status,
        "time": datetime.now(timezone.utc).isoformat(),
    }