"""앱이 보낸 client_log 를 서버 로그로 옮긴다.

스트리밍 중에는 폰이 안경 핫스팟으로 넘어가 앱 콘솔 로그가 늦게 오거나
잘린다. 그 통로가 막혀도 앱만 아는 사건(스트림 협상 결과, 시작 실패 사유,
TTS 폴백, 앱 쪽 타이머 발화)은 남아야 해서 AI WebSocket 으로 받는다. 서버가
이미 아는 값(단어 결과, 확신도, 얼굴·손 비율)은 받지 않는다.

받은 값을 그대로 로그에 넣지는 않는다. 앱이 보낸 키가 LogRecord 의 예약
이름과 겹치면 logging 이 KeyError 를 던지고, 키에 공백이나 제어 문자가 섞이면
로그 한 줄이 쪼개져 가짜 이벤트를 끼워 넣을 수 있다.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.config import (
    CLIENT_LOG_MAX_FIELDS,
    CLIENT_LOG_MAX_PER_SECOND,
    CLIENT_LOG_MAX_VALUE_CHARS,
)
from app.logging_config import RESERVED_RECORD_KEYS
from app.rate_limit import SlidingWindowRateLimiter
from app.schemas.websocket import ClientLogMessage

_LOG_LEVELS = {
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}

# 예약 이름에 우리가 직접 넣는 키를 더한다. 앱 필드가 이것들을 덮으면 어느
# 세션의 어떤 이벤트인지가 로그에서 사라진다.
_RESERVED_FIELD_KEYS = RESERVED_RECORD_KEYS | {"session_id", "event", "at"}

# 키는 로그 한 줄에 그대로 들어가므로 식별자 모양만 받는다.
_FIELD_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,40}$")


def new_client_log_limiter(
    clock: Callable[[], float] = time.monotonic,
) -> SlidingWindowRateLimiter:
    """연결 하나가 쓰는 초당 한도.

    키가 그 연결의 session_id 하나뿐이라 per_ip 와 total 이 같은 뜻이다.
    """
    return SlidingWindowRateLimiter(
        per_ip=CLIENT_LOG_MAX_PER_SECOND,
        total=CLIENT_LOG_MAX_PER_SECOND,
        window_seconds=1.0,
        clock=clock,
    )


def _truncate(text: str) -> str:
    return text[:CLIENT_LOG_MAX_VALUE_CHARS]


def normalize_fields(raw: dict[str, Any] | None) -> dict[str, str | int | float | bool]:
    """로그에 실어도 되는 값만 남긴다.

    버릴 때 알리지 않는다. 값 하나가 모양이 틀렸다고 이벤트 전체를 버리면
    정작 보려던 사건이 사라진다.
    """
    if not raw:
        return {}

    kept: dict[str, str | int | float | bool] = {}
    for key, value in raw.items():
        if len(kept) >= CLIENT_LOG_MAX_FIELDS:
            break
        if key in _RESERVED_FIELD_KEYS or not _FIELD_KEY_PATTERN.match(key):
            continue
        # bool 을 int 보다 먼저 본다 - isinstance(True, int) 가 True 다.
        if isinstance(value, bool):
            kept[key] = value
        elif isinstance(value, (int, float)):
            kept[key] = value
        elif isinstance(value, str):
            kept[key] = _truncate(value)
    return kept


def log_client_event(
    logger: logging.Logger,
    session_id: str,
    raw_message: dict,
    limiter: SlidingWindowRateLimiter,
) -> bool:
    """client_log 한 건을 서버 로그로 옮긴다. 남겼으면 True.

    예외를 올리지 않고 클라이언트에 아무것도 돌려주지 않는다. 이 메시지는
    로그가 목적이라, 모양이 틀렸다고 오류를 보내거나 연결을 끊을 이유가 없다.
    """
    # 한도를 파싱보다 먼저 본다. 쏟아지는 쪽이 모양까지 멀쩡할 이유가 없다.
    # 버린 것을 로그로 남기지 않는 것도 의도다 - 그 경고가 같은 속도로 쏟아져
    # 막으려던 폭주를 그대로 재현한다.
    if not limiter.acquire(session_id).allowed:
        return False

    try:
        message = ClientLogMessage.model_validate(raw_message)
    except ValidationError:
        # 전문을 찍지 않는다. 모양을 모르는 값이라 그 자체가 로그를 더럽힌다.
        logger.warning(
            "Ignored a malformed client_log",
            extra={"session_id": session_id},
        )
        return False

    logger.log(
        _LOG_LEVELS[message.level],
        "Client event",
        extra={
            "session_id": session_id,
            "event": _truncate(message.event),
            "at": _truncate(message.at),
            **normalize_fields(message.fields),
        },
    )
    return True