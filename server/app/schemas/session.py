from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    client: str
    user_id: str


class SessionCreateResponse(BaseModel):
    session_id: str
    status: str
    schema_version: str
    ws_url: str | None


class SessionStopResponse(BaseModel):
    session_id: str
    status: str
