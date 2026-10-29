"""GET /v1/tts - 한국어 문장을 mp3 로 돌려준다.

미니앱은 오디오 바이트를 받지 않고 이 URL 을 재생기에 그대로 넘긴다. 그래서
실패는 HTTP status 로만 전해지고, 앱은 어떤 실패든 폰 TTS 로 폴백한다.
서버가 조용히 빈 응답을 주면 앱은 폴백할 줄 모르므로 status 를 정직하게 낸다.

기동 때 미리 합성할 문구도 여기 둔다. 서비스는 어떤 문장이든 받는 일반
함수이고, "무엇을 읽는가"는 앱과 맞춰야 하는 이쪽의 관심사다.
"""

import logging

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from app.services.label_service import LabelError, all_words
from app.services.tts_service import (
    TtsInvalidText,
    TtsUnavailable,
    TtsUpstreamError,
    synthesize,
)

router = APIRouter()
logger = logging.getLogger(__name__)

AUDIO_MEDIA_TYPE = "audio/mpeg"
# 문구는 고정이고 캐시 키에 음색과 속도가 들어가므로 하루는 안전하다.
CACHE_CONTROL = "public, max-age=86400"

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


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


@router.get("/v1/tts")
async def get_tts_audio(text: str = ""):
    """문장을 mp3 로. 같은 문장은 서버 캐시에서 바로 나온다.

    text 기본값이 빈 문자열인 것은 의도다. 필수 파라미터로 두면 빠뜨렸을 때
    FastAPI 의 422 가 나가는데, 앱이 분기하는 모양이 아니다. 빠뜨린 것과
    비워 보낸 것을 같은 400 으로 돌려준다.
    """
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
