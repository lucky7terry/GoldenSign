from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RecognitionSession:
    session_id: str
    client: str
    user_id: str
    status: str
    created_at: datetime
    stopped_at: datetime | None = None


_sessions: dict[str, RecognitionSession] = {}


def create_session(session_id: str, client: str, user_id: str) -> RecognitionSession:
    session = RecognitionSession(
        session_id=session_id,
        client=client,
        user_id=user_id,
        status="created",
        created_at=datetime.now(timezone.utc),
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> RecognitionSession | None:
    return _sessions.get(session_id)


def stop_session(session_id: str) -> RecognitionSession | None:
    session = _sessions.get(session_id)
    if session is None:
        return None

    session.status = "stopped"
    session.stopped_at = datetime.now(timezone.utc)
    return session
