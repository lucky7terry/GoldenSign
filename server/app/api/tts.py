"""GET /v1/tts - 한국어 문장을 mp3 로 돌려준다.

미니앱은 오디오 바이트를 받지 않고 이 URL 을 재생기에 그대로 넘긴다. 그래서
실패는 HTTP status 로만 전해지고, 앱은 어떤 실패든 폰 TTS 로 폴백한다.
서버가 조용히 빈 응답을 주면 앱은 폴백할 줄 모르므로 status 를 정직하게 낸다.

기동 때 미리 합성할 문구도 여기 둔다. 서비스는 어떤 문장이든 받는 일반
함수이고, "무엇을 읽는가"는 앱과 맞춰야 하는 이쪽의 관심사다.
"""

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.config import TTS_RATE_LIMIT_GLOBAL, TTS_RATE_LIMIT_PER_IP
from app.rate_limit import RateLimitVerdict, SlidingWindowRateLimiter
from app.services.label_service import LabelError, all_words
from app.services.tts_service import (
    TtsInvalidText,
    TtsUnavailable,
    TtsUpstreamError,
    is_cached,
    synthesize,
)

router = APIRouter()
logger = logging.getLogger(__name__)

AUDIO_MEDIA_TYPE = "audio/mpeg"
# 문구는 고정이고 캐시 키에 음색과 속도가 들어가므로 하루는 안전하다.
CACHE_CONTROL = "public, max-age=86400"

# 캐시 미스만 센다. 프로세스 안에 있으므로 재기동하면 비워진다.
_rate_limiter = SlidingWindowRateLimiter(
    per_ip=TTS_RATE_LIMIT_PER_IP,
    total=TTS_RATE_LIMIT_GLOBAL,
)

# 아래 세 문구는 miniapp/src/background/tts.ts 의 문구와 같아야 한다.
START_ANNOUNCEMENT = (
    "골든사인입니다. 버튼을 길게 누르면 인식이 시작됩니다. "
    "수어 단어의 시작과 끝에 버튼을 짧게 눌러 주세요."
)
RETRY_ANNOUNCEMENT = "인식에 실패했습니다. 재시도해주세요"
FRAMING_ANNOUNCEMENT = "상대방의 얼굴과 손이 함께 보이도록 바라보세요"

ANNOUNCEMENTS = (
    START_ANNOUNCEMENT,
    RETRY_ANNOUNCEMENT,
    FRAMING_ANNOUNCEMENT,
)


def prewarm_phrases() -> tuple[str, ...]:
    """기동 때 미리 합성할 문구. 안내 멘트와 모델이 낼 수 있는 단어 전부다."""
    try:
        words = all_words()
    except LabelError:
        # 라벨 파일 문제는 인식 경로가 따로 알린다. 안내 문구만이라도 미리 만든다.
        logger.warning("Could not read word labels; prewarming announcements only")
        words = ()
    return ANNOUNCEMENTS + words


def _error(
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
        headers=headers,
    )


def _client_host(request: Request) -> str:
    """요청을 보낸 IP. Caddy 뒤에서는 uvicorn 의 --proxy-headers 가 채워 준다."""
    return request.client.host if request.client is not None else "unknown"


@router.get("/v1/tts")
async def get_tts_audio(request: Request, text: str = ""):
    """문장을 mp3 로. 같은 문장은 서버 캐시에서 바로 나온다.

    text 기본값이 빈 문자열인 것은 의도다. 필수 파라미터로 두면 빠뜨렸을 때
    FastAPI 의 422 가 나가는데, 앱이 분기하는 모양이 아니다. 빠뜨린 것과
    비워 보낸 것을 같은 400 으로 돌려준다.

    인증은 두지 않는다. 토큰을 두어도 앱 번들에 그대로 들어가므로 가져다
    쓰는 것을 막지 못한다. 대신 Google 을 새로 부르는 요청만 센다 - 앱이
    실제로 읽는 문구는 기동 때 미리 합성돼 있어 제한에 걸리지 않는다.
    """
    try:
        cached = await is_cached(text)
    except TtsInvalidText as exc:
        # 제한을 세기 전에 거른다. 빈 text 로 남의 몫을 깎을 수 있으면 안 된다.
        return _error(400, "tts_invalid_text", str(exc))

    if not cached:
        client = _client_host(request)
        verdict = _rate_limiter.acquire(client)
        if not verdict.allowed:
            logger.warning(
                "Rate limited a TTS cache miss",
                extra={
                    "client": client,
                    "scope": verdict.scope,
                    "retry_after_seconds": verdict.retry_after_seconds,
                },
            )
            return _rate_limited(verdict)

    try:
        audio = await synthesize(text)
    except TtsInvalidText as exc:
        return _error(400, "tts_invalid_text", str(exc))
    except TtsUnavailable as exc:
        logger.warning("TTS is not configured on this server")
        return _error(503, "tts_unavailable", str(exc))
    except TtsUpstreamError as exc:
        # 업스트림 실패는 tts_service 가 status 와 본문까지 남겼다.
        return _error(502, "tts_upstream_error", str(exc))

    return Response(
        content=audio,
        media_type=AUDIO_MEDIA_TYPE,
        headers={"Cache-Control": CACHE_CONTROL},
    )


def _rate_limited(verdict: RateLimitVerdict) -> JSONResponse:
    retry_after = str(verdict.retry_after_seconds)
    return _error(
        429,
        "tts_rate_limited",
        f"Too many new TTS phrases; retry in {retry_after} seconds.",
        {"Retry-After": retry_after},
    )
