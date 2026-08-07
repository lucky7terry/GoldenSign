from datetime import datetime
from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    api: str
    model: dict[str, Any]
    time: datetime
