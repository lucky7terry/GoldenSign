"""로그의 extra 필드를 한 줄에 함께 출력하는 포매터.

서버 코드는 logger.info(..., extra={...}) 로 fps·keypoints·frame_count 같은
진단 값을 남기는데, uvicorn 기본 포매터는 이를 버린다. 이 포매터를 써야
그 값들이 보인다.

실행 (반드시 server/ 안에서 — devlog.yaml 이 이 모듈을 import 한다):
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-config devlog.yaml
"""
import logging

_STD = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class ExtraFormatter(logging.Formatter):
    def format(self, record):
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STD and not k.startswith("_")
        }
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base
