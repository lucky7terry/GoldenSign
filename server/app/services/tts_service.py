"""한국어 TTS 합성. Google Cloud Text-to-Speech REST 를 직접 부른다.

google-cloud-texttospeech 를 쓰지 않는다. 필요한 것은 엔드포인트 하나이고
인증도 API 키 헤더 한 줄이라, 이미 의존성에 있는 aiohttp 로 충분하다.

합성 결과는 디스크에 쌓는다. 앱이 읽는 문구는 시작 안내와 단어 50개, 그리고
몇 가지 오류 안내로 사실상 고정이라 한 번 합성하면 그 뒤로는 같은 바이트를
돌려주면 된다. 캐시 키에 음색과 속도를 함께 넣으므로 설정을 바꾸면 자연히
새로 합성한다.

aiohttp 는 requirements-test.txt 에 없다. 그래서 HTTP 호출은 함수 안에서
import 하고, synthesize 는 호출부를 주입받을 수 있게 열어 둔다.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app.config import (
    GOOGLE_TTS_API_KEY,
    TTS_CACHE_DIR,
    TTS_CACHE_MAX_FILES,
    TTS_MAX_CHARS,
    TTS_REQUEST_TIMEOUT_SECONDS,
    TTS_SPEAKING_RATE,
    TTS_VOICE_NAME,
)

logger = logging.getLogger(__name__)

SYNTHESIZE_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
LANGUAGE_CODE = "ko-KR"
AUDIO_ENCODING = "MP3"

# 오류 로그에 남길 응답 본문 길이.
_ERROR_BODY_CHARS = 200

# (url, payload, timeout_seconds) -> (status, body)
HttpPost = Callable[[str, dict, float], Awaitable[tuple[int, bytes]]]


class TtsError(RuntimeError):
    """TTS 합성이 실패했다."""


class TtsInvalidText(TtsError):
    """합성할 텍스트가 비었거나 TTS_MAX_CHARS 를 넘었다."""


class TtsUnavailable(TtsError):
    """API 키가 없어 합성을 시도할 수 없다."""


class TtsUpstreamError(TtsError):
    """Google TTS 가 2xx 를 주지 않았거나 제때 답하지 않았다."""


@dataclass
class _KeyLock:
    lock: asyncio.Lock
    waiters: int = 0


_key_locks: dict[str, _KeyLock] = {}


@asynccontextmanager
async def _locked(key: str):
    """같은 키의 동시 요청이 API 를 한 번만 부르게 한다."""
    # 대기자가 없으면 지운다 - 임의의 텍스트를 받는 경로라 락이 쌓이면 샌다.
    entry = _key_locks.get(key)
    if entry is None:
        entry = _KeyLock(lock=asyncio.Lock())
        _key_locks[key] = entry
    entry.waiters += 1
    try:
        async with entry.lock:
            yield
    finally:
        entry.waiters -= 1
        if entry.waiters == 0 and _key_locks.get(key) is entry:
            del _key_locks[key]


def _cache_key(text: str) -> str:
    """음색과 속도까지 넣는다. 설정을 바꾸면 옛 음성을 돌려주지 않는다."""
    raw = f"{TTS_VOICE_NAME}|{TTS_SPEAKING_RATE}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(TTS_CACHE_DIR, f"{key}.mp3")


def _read_cache(path: str) -> bytes | None:
    try:
        with open(path, "rb") as cached:
            audio = cached.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not read the TTS cache", extra={"error": str(exc)})
        return None

    if not audio:
        return None
    _touch(path)
    return audio


def _touch(path: str) -> None:
    """최근 사용 시각을 남긴다. mtime 은 정리 순서에만 쓰이므로 실패는 넘어간다."""
    try:
        os.utime(path)
    except OSError:
        pass


def _write_cache(path: str, audio: bytes) -> None:
    """같은 디렉터리의 임시 파일에 쓰고 os.replace 로 갈아 끼운다."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as pending:
            pending.write(audio)
        os.replace(temporary, path)
    except OSError:
        _remove_quietly(temporary)
        raise


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _prune_cache(directory: str, max_files: int) -> int:
    """파일 수가 상한을 넘으면 mtime 이 오래된 것부터 지운다.

    여기서 실패해도 요청은 성공시킨다. 지우지 못한 캐시는 디스크를 조금 더
    쓸 뿐이고, 그것 때문에 방금 합성한 음성을 버릴 이유가 없다.
    """
    try:
        with os.scandir(directory) as entries:
            files = [
                (entry.stat().st_mtime, entry.path)
                for entry in entries
                if entry.name.endswith(".mp3") and entry.is_file()
            ]
    except OSError as exc:
        logger.warning("Could not scan the TTS cache", extra={"error": str(exc)})
        return 0

    excess = len(files) - max_files
    if excess <= 0:
        return 0

    files.sort()
    removed = 0
    for _, path in files[:excess]:
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning(
                "Could not evict a TTS cache file",
                extra={"error": str(exc)},
            )
            continue
        removed += 1

    logger.info(
        "Evicted the least recently used TTS audio",
        extra={"removed": removed, "max_files": max_files},
    )
    return removed


def _request_payload(text: str) -> dict:
    return {
        "input": {"text": text},
        "voice": {"languageCode": LANGUAGE_CODE, "name": TTS_VOICE_NAME},
        "audioConfig": {
            "audioEncoding": AUDIO_ENCODING,
            "speakingRate": TTS_SPEAKING_RATE,
        },
    }


async def _post_json(
    url: str,
    payload: dict,
    timeout_seconds: float,
) -> tuple[int, bytes]:
    """기본 HTTP 호출부. aiohttp 가 테스트 의존성에 없어 여기서 import 한다."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers={"X-Goog-Api-Key": GOOGLE_TTS_API_KEY},
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as response:
            return response.status, await response.read()


def _body_excerpt(body: bytes) -> str:
    """오류 로그용 응답 앞부분. API 키는 헤더에만 있어 여기 섞이지 않는다."""
    return body.decode("utf-8", errors="replace")[:_ERROR_BODY_CHARS]


def _decode_audio(body: bytes) -> bytes:
    try:
        audio = base64.b64decode(
            json.loads(body.decode("utf-8"))["audioContent"],
            validate=True,
        )
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise TtsUpstreamError("Google TTS returned an unreadable response.") from exc
    if not audio:
        raise TtsUpstreamError("Google TTS returned empty audio.")
    return audio


async def _synthesize_upstream(text: str, http_post: HttpPost) -> bytes:
    try:
        status, body = await http_post(
            SYNTHESIZE_URL,
            _request_payload(text),
            TTS_REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        logger.warning(
            "Google TTS did not answer in time",
            extra={"timeout_seconds": TTS_REQUEST_TIMEOUT_SECONDS},
        )
        raise TtsUpstreamError("Google TTS did not answer in time.") from exc
    except TtsError:
        raise
    except Exception as exc:
        # 예외 문자열은 남기지 않는다 - 요청 헤더를 담아 오면 키가 새어 나간다.
        logger.warning(
            "Google TTS request failed",
            extra={"error": type(exc).__name__},
        )
        raise TtsUpstreamError("Google TTS request failed.") from exc

    if not 200 <= status < 300:
        logger.warning(
            "Google TTS rejected the request",
            extra={"status": status, "body": _body_excerpt(body)},
        )
        raise TtsUpstreamError(f"Google TTS responded with status {status}.")

    return _decode_audio(body)


def _clean_text(text: str) -> str:
    """합성할 수 있는 텍스트인지 보고 다듬는다. 검증은 여기 한 곳이다."""
    cleaned = text.strip()
    if not cleaned:
        raise TtsInvalidText("TTS text must not be empty.")
    if len(cleaned) > TTS_MAX_CHARS:
        raise TtsInvalidText(
            f"TTS text must be at most {TTS_MAX_CHARS} characters; got {len(cleaned)}."
        )
    return cleaned


async def is_cached(text: str) -> bool:
    """이 문장이 이미 디스크에 있는지. 없으면 synthesize 가 Google 을 부른다.

    호출부가 요청 제한을 걸 자리를 고르는 데 쓴다. 캐시 hit 은 외부 호출이
    아니므로 세지 않아야 하고, 그 판단에 필요한 캐시 키 규칙은 이 모듈만 안다.
    """
    cleaned = _clean_text(text)
    return await asyncio.to_thread(
        os.path.exists,
        _cache_path(_cache_key(cleaned)),
    )


async def synthesize(text: str, http_post: HttpPost | None = None) -> bytes:
    """한국어 문장을 mp3 바이트로. 캐시에 있으면 API 를 부르지 않는다."""
    cleaned = _clean_text(text)
    if not GOOGLE_TTS_API_KEY:
        raise TtsUnavailable("GOOGLE_TTS_API_KEY is not set.")

    path = _cache_path(_cache_key(cleaned))
    cached = await asyncio.to_thread(_read_cache, path)
    if cached is not None:
        return cached

    async with _locked(path):
        # 락을 기다리는 동안 먼저 들어온 요청이 채웠을 수 있다.
        cached = await asyncio.to_thread(_read_cache, path)
        if cached is not None:
            return cached

        audio = await _synthesize_upstream(cleaned, http_post or _post_json)
        try:
            await asyncio.to_thread(_write_cache, path, audio)
        except OSError as exc:
            # 캐시에 못 써도 이번 음성은 돌려준다. 다음 요청이 다시 합성한다.
            logger.warning(
                "Could not cache the synthesized audio",
                extra={"error": str(exc)},
            )
        else:
            await asyncio.to_thread(
                _prune_cache,
                TTS_CACHE_DIR,
                TTS_CACHE_MAX_FILES,
            )
        logger.info(
            "Synthesized Korean speech",
            extra={
                "voice": TTS_VOICE_NAME,
                "chars": len(cleaned),
                "audio_bytes": len(audio),
            },
        )
        return audio


async def prewarm(texts: Iterable[str]) -> None:
    """고정 문구를 미리 합성해 캐시에 넣는다. 실패해도 다음 문구로 넘어간다."""
    if not GOOGLE_TTS_API_KEY:
        logger.info("Skipping TTS prewarm because GOOGLE_TTS_API_KEY is not set")
        return

    for text in texts:
        try:
            await synthesize(text)
        except Exception as exc:
            # TtsError 의 문자열은 우리가 쓴 것이고, 그 밖의 예외는 키를 담아 올 수 있다.
            detail = str(exc) if isinstance(exc, TtsError) else type(exc).__name__
            logger.warning(
                "Could not prewarm a TTS phrase",
                extra={"error": detail, "chars": len(text)},
            )