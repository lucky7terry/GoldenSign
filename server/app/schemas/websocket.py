from typing import Literal

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
