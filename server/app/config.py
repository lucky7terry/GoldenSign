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


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or value <= 0:
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

# 단어 구간 모드. 사용자가 word_start / word_end 로 한 단어의 시작과 끝을
# 표시하면 그 구간만 모아 한 번 추론한다.
#
# WORD_MAX_SECONDS: 이 시간이 지나면 서버가 알아서 구간을 닫고 결과를 낸다.
#   사용자가 끝 표시를 잊어도 결과는 나온다.
# WORD_MIN_FRAMES: 이보다 적으면 거절한다. 3장을 60장으로 늘려봐야
#   같은 자세가 20번 반복될 뿐이다. MediaPipe 가 최악 5fps 이므로
#   8장이면 대략 1.6초 - 가장 짧은 단어도 이보다는 길다.
# WORD_TARGET_FRAMES: 모델 입력 길이. 학습이 60이다.
WORD_MAX_SECONDS = _env_float("WORD_MAX_SECONDS", 8.0)
WORD_MIN_FRAMES = _env_int("WORD_MIN_FRAMES", 8)
WORD_TARGET_FRAMES = _env_int("WORD_TARGET_FRAMES", 60)
# WORD_SOURCE_FPS: 단어 구간을 되돌릴 격자의 프레임레이트. 원본 영상과
#   같은 30 이 기본이다.
#
#   이 격자가 촘촘해야 도착한 프레임이 제 시각 근처에 떨어진다. "구간이
#   실제로 도착한 평균 간격"을 쓰면 프레임 수는 보존되지만, 간격이
#   들쭉날쭉할 때 각 프레임이 원래 시각에서 크게 밀려난다. 실측에서
#   8프레임 구간(간격 100~1567ms)의 확신도가 0.479 로 무너졌다 -
#   임계값 0.5 아래다. 30 고정은 12가지 조건 전부에서 0.689~0.809 였다.
WORD_SOURCE_FPS = _env_float("WORD_SOURCE_FPS", 30.0)


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
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
