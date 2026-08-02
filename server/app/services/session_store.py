import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import REDIS_URL, SESSION_STORE_BACKEND, SESSION_TTL_SECONDS


@dataclass
class RecognitionSession:
    session_id: str
    client: str
    user_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    stopped_at: datetime | None = None


_sessions: dict[str, RecognitionSession] = {}
_redis_client = None


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _serialize_session(session: RecognitionSession) -> str:
    return json.dumps(
        {
            "session_id": session.session_id,
            "client": session.client,
            "user_id": session.user_id,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "stopped_at": (
                session.stopped_at.isoformat()
                if session.stopped_at is not None
                else None
            ),
        }
    )


def _deserialize_session(value: str) -> RecognitionSession:
    payload = json.loads(value)
    stopped_at = payload.get("stopped_at")
    return RecognitionSession(
        session_id=payload["session_id"],
        client=payload["client"],
        user_id=payload["user_id"],
        status=payload["status"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        expires_at=datetime.fromisoformat(payload["expires_at"]),
        stopped_at=datetime.fromisoformat(stopped_at) if stopped_at else None,
    )


def _get_redis_client():
    global _redis_client
    if _redis_client is None:
        from redis import Redis

        _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _use_redis() -> bool:
    return SESSION_STORE_BACKEND == "redis"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _remaining_ttl_seconds(session: RecognitionSession) -> int:
    return int((session.expires_at - _utc_now()).total_seconds())


def _persist_session(session: RecognitionSession) -> RecognitionSession | None:
    ttl = _remaining_ttl_seconds(session)

    if _use_redis():
        redis_client = _get_redis_client()
        key = _session_key(session.session_id)
        if ttl <= 0:
            redis_client.delete(key)
            return None
        redis_client.setex(key, ttl, _serialize_session(session))
        return session

    if ttl <= 0:
        _sessions.pop(session.session_id, None)
        return None

    _sessions[session.session_id] = session
    return session


def create_session(session_id: str, client: str, user_id: str) -> RecognitionSession:
    created_at = _utc_now()
    session = RecognitionSession(
        session_id=session_id,
        client=client,
        user_id=user_id,
        status="created",
        created_at=created_at,
        expires_at=created_at + timedelta(seconds=SESSION_TTL_SECONDS),
    )

    if _use_redis():
        redis_client = _get_redis_client()
        redis_client.setex(
            _session_key(session_id),
            SESSION_TTL_SECONDS,
            _serialize_session(session),
        )
        return session

    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> RecognitionSession | None:
    if _use_redis():
        value = _get_redis_client().get(_session_key(session_id))
        if value is None:
            return None
        session = _deserialize_session(value)
        if _remaining_ttl_seconds(session) <= 0:
            _get_redis_client().delete(_session_key(session_id))
            return None
        return session

    session = _sessions.get(session_id)
    if session is None:
        return None
    if _remaining_ttl_seconds(session) <= 0:
        _sessions.pop(session_id, None)
        return None
    return session


def stop_session(session_id: str) -> RecognitionSession | None:
    session = get_session(session_id)
    if session is None:
        return None

    session.status = "stopped"
    session.stopped_at = _utc_now()

    return _persist_session(session)


def activate_session(session_id: str) -> RecognitionSession | None:
    session = get_session(session_id)
    if session is None:
        return None

    session.status = "active"

    return _persist_session(session)
