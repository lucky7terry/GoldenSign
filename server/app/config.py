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
PUBLIC_WS_BASE_URL = os.getenv("PUBLIC_WS_BASE_URL")
WS_IDLE_TIMEOUT_SECONDS = _env_float("WS_IDLE_TIMEOUT_SECONDS", 60.0)

# 단어 구간 모드. 사용자가 word_start / word_end 로 한 단어의 시작과 끝을
# 표시하면 그 구간만 모아 한 번 추론한다.
#
# WORD_MAX_SECONDS: 이 시간이 지나면 서버가 알아서 구간을 닫고 결과를 낸다.
#   사용자가 끝 표시를 잊어도 결과는 나온다.
# WORD_MIN_FRAMES: 이보다 적으면 거절한다. 3장을 60장으로 늘려봐야
#   같은 자세가 20번 반복될 뿐이다.
#
#   이슈 #42 에는 10 으로 적었는데 실측 후 8 로 낮췄다. 영상 5개를
#   8~12프레임까지 떨어뜨려도 5/5 정답이었고 확신도 0.765~0.809 였다.
#   낮출수록 짧은 단어를 덜 거절하므로, 정확도가 버티는 선까지 내린다.
#   MediaPipe 가 5.8fps 까지 떨어지는 것을 감안하면 8장은 약 1.4초다.
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

# 단어 판정 임계값. 이 아래면 단어를 주장하지 않고 text 를 null 로 보낸다.
#
# 근거는 영상 5개(WORD0001, 5시점) 검증이다. 단어 단위 결과가 최저 확신도
# 0.615, 2위와의 격차 최저 0.377 이었으므로 전 시점이 통과한다. 반대로
# 슬라이딩 윈도우의 애매한 판정(0.366, 격차 0.083)은 걸러진다.
#
# 확신도만 보면 안 된다. 50개 클래스가 고르게 나뉜 경우와 두 개가 팽팽한
# 경우를 구분하지 못한다. 2위와의 격차를 같이 본다.
#
# 두 값 모두 확률이므로 (0, 1] 을 벗어날 수 없다. 1 을 넘게 적으면 모든
# 구간이 조용히 거절되고, 서버는 아무 불평 없이 단어를 하나도 못 낸다.
# 오타 하나로 그렇게 되는 것보다 기동 때 세우는 편이 낫다.
def _env_probability(name: str, default: float) -> float:
    """확률 임계값. (0, 1] 을 벗어나면 기동 때 세운다."""
    value = _env_float(name, default)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]; got {value}.")
    return value


RECOGNITION_CONFIDENCE_THRESHOLD = _env_probability(
    "RECOGNITION_CONFIDENCE_THRESHOLD",
    0.5,
)
RECOGNITION_MARGIN_THRESHOLD = _env_probability(
    "RECOGNITION_MARGIN_THRESHOLD",
    0.15,
)

# 단어 구간을 닫기 전에 큐에 남은 프레임을 기다리는 최대 시간. 0 이면
# 기다리지 않는다.
WORD_DRAIN_TIMEOUT_SECONDS = _env_float("WORD_DRAIN_TIMEOUT_SECONDS", 2.0)

# result 메시지에 좌표를 실을지. 좌표는 실수 959개로 메시지의 94% 를 차지하는데
# (8,338 -> 479 바이트) 미니앱은 읽지 않는다. 기본은 빼고, 서버 좌표를 눈으로
# 확인해야 할 때만 켠다.
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
