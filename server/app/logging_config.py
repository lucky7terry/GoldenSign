"""로깅 설정.

logger.info(..., extra={...})로 넘긴 필드는 기본 포매터가 렌더링하지 않는다.
이 모듈의 포매터는 표준 LogRecord 속성이 아닌 값을 전부 메시지 뒤에 붙여
session_id / stream_id / frame_count / fps / keypoints 등이 로그에 남도록 한다.
"""

import json
import logging
import re

_RESERVED_RECORD_KEYS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}

# 공백, 따옴표, 역슬래시, 제어 문자(CR/LF/TAB/ESC/DEL 포함).
# 제어 문자를 그대로 흘리면 client_message_id 같은 클라이언트 입력으로
# 로그 한 줄을 쪼개 가짜 이벤트를 끼워 넣을 수 있다(로그 인젝션).
_NEEDS_ESCAPING = re.compile(r"[\s\"\\\x00-\x1f\x7f]")


def _render_value(value: object) -> str:
    text = str(value)
    if _NEEDS_ESCAPING.search(text):
        # json.dumps가 따옴표 감싸기와 제어 문자 이스케이프를 함께 처리한다.
        # ensure_ascii=False로 한글은 그대로 읽히게 둔다.
        return json.dumps(text, ensure_ascii=False)
    return text


class ExtraFieldFormatter(logging.Formatter):
    """extra로 전달된 필드를 `key=value` 형태로 덧붙이는 포매터."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_KEYS
        }
        if not extra_fields:
            return formatted

        rendered = " ".join(
            f"{key}={_render_value(value)}"
            for key, value in sorted(extra_fields.items())
        )
        return f"{formatted} | {rendered}"


def configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        ExtraFieldFormatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # uvicorn은 자기 로거에만 핸들러를 붙이고 root는 건드리지 않는다.
    # root 핸들러만 교체하므로 uvicorn access 로그는 그대로 유지된다.
    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
