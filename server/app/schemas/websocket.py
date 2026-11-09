from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WebSocketMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    schema_version: str
    session_id: str
    client_message_id: str | None = None
    request_id: str | None = None


class ImagePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    data: str


class HelloMessage(WebSocketMessage):
    type: Literal["hello"]


class FrameMessage(WebSocketMessage):
    type: Literal["frame"]
    image: ImagePayload
    frame_index: int | None = None
    captured_at: str | None = None


class PingMessage(WebSocketMessage):
    type: Literal["ping"]


class StopMessage(WebSocketMessage):
    type: Literal["stop"]


class StreamStartMessage(WebSocketMessage):
    type: Literal["stream_start"]
    webrtc_url: str = Field(min_length=1)
    stream_id: str = Field(min_length=1)


class StreamStopMessage(WebSocketMessage):
    type: Literal["stream_stop"]
    stream_id: str = Field(min_length=1)


class WordStartMessage(WebSocketMessage):
    """한 단어의 시작 표시. 이 시점부터 word_end 까지의 프레임만 모은다."""

    type: Literal["word_start"]


class WordEndMessage(WebSocketMessage):
    """한 단어의 끝 표시. 서버는 모은 구간으로 결과를 낸다."""

    type: Literal["word_end"]


class ClientLogMessage(WebSocketMessage):
    """앱만 아는 사건 하나. 서버 로그에 남기는 것이 목적이라 응답하지 않는다.

    at 은 앱이 그 사건을 기록한 시각이다. 스트리밍 중에는 폰이 안경 핫스팟에
    있어 연결이 끊겼다 붙으므로, 모아 둔 이벤트가 한참 뒤에 도착할 수 있다.
    그래서 서버 수신 시각으로 대신할 수 없다.
    """

    type: Literal["client_log"]
    event: str = Field(min_length=1)
    level: Literal["info", "warn", "error"]
    at: str = Field(min_length=1)
    # 값 검사는 app.client_log 가 한다 - 여기서 좁히면 값 하나 때문에 이벤트
    # 전체가 버려진다.
    fields: dict[str, Any] | None = None
