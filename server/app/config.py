import math
import os


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def _env_float(
    name: str,
    default: float,
    allow_disable: bool = False,
) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    # allow_disable=True면 0 이하는 "끔"을 뜻하므로 그대로 통과시킨다.
    if not allow_disable and value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, default)
    return tuple(
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    )


SESSION_STORE_BACKEND = os.getenv("SESSION_STORE_BACKEND", "memory").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", 3600)
MAX_CONCURRENT_RECOGNITIONS = _env_int("MAX_CONCURRENT_RECOGNITIONS", 2)
FRAME_QUEUE_MAX_SIZE = _env_int("FRAME_QUEUE_MAX_SIZE", 30)
SEQUENCE_WINDOW_SIZE = _env_int("SEQUENCE_WINDOW_SIZE", 60)
SEQUENCE_STRIDE = _env_int("SEQUENCE_STRIDE", 1)
PUBLIC_WS_BASE_URL = os.getenv("PUBLIC_WS_BASE_URL")
WS_IDLE_TIMEOUT_SECONDS = _env_float("WS_IDLE_TIMEOUT_SECONDS", 60.0)

# result 메시지에 좌표를 실을지. 좌표는 실수 959개로 메시지의 94% 를 차지하는데
# (8,338 -> 479 바이트) 미니앱은 읽지 않는다. 기본은 빼고, 서버 좌표를 눈으로
# 확인해야 할 때만 켠다. 켜면 초당 13개 결과 기준 110KB/s 가 나간다.
INCLUDE_KEYPOINTS_IN_RESULT = _env_bool("INCLUDE_KEYPOINTS_IN_RESULT", False)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 키포인트 추출 지연 요약 로그 주기. 0 이하면 요약 로그를 끈다.
KEYPOINT_TIMING_INTERVAL_SECONDS = _env_float(
    "KEYPOINT_TIMING_INTERVAL_SECONDS",
    5.0,
    allow_disable=True,
)
WHEP_MAX_RETRIES = _env_int("WHEP_MAX_RETRIES", 5)
WHEP_RETRY_DELAY_SECONDS = _env_float("WHEP_RETRY_DELAY_SECONDS", 1.0)
WHEP_CONNECT_TIMEOUT_SECONDS = _env_float("WHEP_CONNECT_TIMEOUT_SECONDS", 10.0)
WHEP_FIRST_FRAME_TIMEOUT_SECONDS = _env_float(
    "WHEP_FIRST_FRAME_TIMEOUT_SECONDS",
    15.0,
)
WHEP_FRAME_IDLE_TIMEOUT_SECONDS = _env_float("WHEP_FRAME_IDLE_TIMEOUT_SECONDS", 5.0)
WHEP_KEYFRAME_REQUEST_INTERVAL_SECONDS = _env_float(
    "WHEP_KEYFRAME_REQUEST_INTERVAL_SECONDS",
    2.0,
)
WHEP_STUN_SERVER_URLS = _env_csv(
    "WHEP_STUN_SERVER_URLS",
    "stun:stun.cloudflare.com:3478",
)
WHEP_ALLOWED_HOST_SUFFIXES = _env_csv(
    "WHEP_ALLOWED_HOST_SUFFIXES",
    "cloudflarestream.com,videodelivery.net",
)
