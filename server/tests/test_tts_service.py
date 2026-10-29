"""TTS 합성의 캐시, 동시성, 실패 경로.

HTTP 호출부를 주입해서 aiohttp 없이 돈다 - CI 는 requirements-test.txt 만
설치한다.

API 키가 로그에 남지 않는지도 여기서 지킨다. 키는 요청 헤더 하나에만
있어야 하고, 실패 로그에 섞이면 서버 로그를 보는 모든 사람에게 새어 나간다.
그래서 서버가 실제로 쓰는 포매터(ExtraFieldFormatter)로 렌더링한 전문을
검사한다 - assertLogs 의 output 은 extra 필드를 펼치지 않아 통과만 한다.
"""

import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logging_config import ExtraFieldFormatter
from app.services import tts_service

_API_KEY = "test-api-key-must-never-be-logged"
_LOGGER_NAME = "app.services.tts_service"
_AUDIO = b"fake-mp3-bytes"


def _audio_response(audio: bytes = _AUDIO) -> tuple[int, bytes]:
    body = json.dumps({"audioContent": base64.b64encode(audio).decode("ascii")})
    return 200, body.encode("utf-8")


def _rendered(records) -> str:
    """서버 포매터와 같은 조건으로 extra 까지 펼친 로그 전문."""
    formatter = ExtraFieldFormatter(fmt="%(levelname)s %(message)s")
    return "\n".join(formatter.format(record) for record in records)


class _FakePost:
    """호출을 기록하는 HTTP 호출부. per_text 로 특정 문구만 실패시킨다."""

    def __init__(self, default=None, per_text=None, delay=0.0):
        self.default = _audio_response() if default is None else default
        self.per_text = per_text or {}
        self.delay = delay
        self.calls = []

    async def __call__(self, url, payload, timeout_seconds):
        text = payload["input"]["text"]
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
                "text": text,
            }
        )
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        outcome = self.per_text.get(text, self.default)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class TtsServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache_dir = Path(cache.name)
        settings = mock.patch.multiple(
            tts_service,
            GOOGLE_TTS_API_KEY=_API_KEY,
            TTS_CACHE_DIR=str(self.cache_dir),
            TTS_VOICE_NAME="ko-KR-Chirp3-HD-Algieba",
            TTS_SPEAKING_RATE=1.0,
            TTS_MAX_CHARS=200,
        )
        settings.start()
        self.addCleanup(settings.stop)

    def cached_files(self):
        return sorted(path.name for path in self.cache_dir.iterdir())


class SynthesizeRequestTest(TtsServiceTestCase):
    async def test_request_matches_the_rest_api(self):
        post = _FakePost()

        audio = await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(audio, _AUDIO)
        call = post.calls[0]
        self.assertEqual(call["url"], tts_service.SYNTHESIZE_URL)
        self.assertEqual(call["timeout_seconds"], tts_service.TTS_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(
            call["payload"],
            {
                "input": {"text": "안녕하세요"},
                "voice": {
                    "languageCode": "ko-KR",
                    "name": "ko-KR-Chirp3-HD-Algieba",
                },
                "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.0},
            },
        )

    async def test_text_is_stripped_before_synthesis(self):
        post = _FakePost()

        await tts_service.synthesize("  안녕하세요  ", post)

        self.assertEqual(post.calls[0]["payload"]["input"]["text"], "안녕하세요")


class SynthesizeCacheTest(TtsServiceTestCase):
    async def test_cache_hit_does_not_call_the_api(self):
        post = _FakePost()

        first = await tts_service.synthesize("안녕하세요", post)
        second = await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(first, _AUDIO)
        self.assertEqual(second, _AUDIO)
        self.assertEqual(len(post.calls), 1)

    async def test_synthesized_audio_lands_in_the_cache_directory(self):
        await tts_service.synthesize("안녕하세요", _FakePost())

        files = self.cached_files()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".mp3"))
        self.assertEqual((self.cache_dir / files[0]).read_bytes(), _AUDIO)

    async def test_a_different_voice_does_not_reuse_the_cache(self):
        post = _FakePost()
        await tts_service.synthesize("안녕하세요", post)

        with mock.patch.object(tts_service, "TTS_VOICE_NAME", "ko-KR-Chirp3-HD-Aoede"):
            await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(len(post.calls), 2)
        self.assertEqual(len(self.cached_files()), 2)

    async def test_concurrent_requests_for_one_text_call_the_api_once(self):
        # 첫 호출이 끝나기 전에 둘째가 락에 걸려야 한다. 지연이 그것을 만든다.
        post = _FakePost(delay=0.05)

        results = await asyncio.gather(
            tts_service.synthesize("같은 문장", post),
            tts_service.synthesize("같은 문장", post),
        )

        self.assertEqual(results, [_AUDIO, _AUDIO])
        self.assertEqual(len(post.calls), 1)

    async def test_locks_do_not_pile_up(self):
        post = _FakePost()

        await asyncio.gather(
            tts_service.synthesize("첫 문장", post),
            tts_service.synthesize("둘째 문장", post),
        )

        self.assertEqual(tts_service._key_locks, {})


class SynthesizeRejectionTest(TtsServiceTestCase):
    async def test_empty_text_is_rejected(self):
        post = _FakePost()

        for text in ("", "   ", "\n\t"):
            with self.assertRaises(tts_service.TtsInvalidText):
                await tts_service.synthesize(text, post)

        self.assertEqual(post.calls, [])

    async def test_text_over_the_limit_is_rejected(self):
        post = _FakePost()

        with mock.patch.object(tts_service, "TTS_MAX_CHARS", 5):
            with self.assertRaises(tts_service.TtsInvalidText):
                await tts_service.synthesize("여섯 글자가 넘는다", post)

        self.assertEqual(post.calls, [])

    async def test_text_at_the_limit_is_accepted(self):
        post = _FakePost()

        with mock.patch.object(tts_service, "TTS_MAX_CHARS", 5):
            await tts_service.synthesize("다섯글자", post)

        self.assertEqual(len(post.calls), 1)

    async def test_a_missing_api_key_is_unavailable(self):
        post = _FakePost()

        with mock.patch.object(tts_service, "GOOGLE_TTS_API_KEY", ""):
            with self.assertRaises(tts_service.TtsUnavailable):
                await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(post.calls, [])


class SynthesizeUpstreamFailureTest(TtsServiceTestCase):
    async def test_a_non_2xx_status_is_an_upstream_error(self):
        post = _FakePost(default=(403, b'{"error": {"message": "forbidden"}}'))

        with self.assertRaises(tts_service.TtsUpstreamError):
            await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(self.cached_files(), [])

    async def test_a_timeout_is_an_upstream_error(self):
        post = _FakePost(default=TimeoutError())

        with self.assertRaises(tts_service.TtsUpstreamError):
            await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(self.cached_files(), [])

    async def test_a_transport_failure_is_an_upstream_error(self):
        post = _FakePost(default=OSError("connection reset"))

        with self.assertRaises(tts_service.TtsUpstreamError):
            await tts_service.synthesize("안녕하세요", post)

    async def test_an_unreadable_body_is_an_upstream_error(self):
        post = _FakePost(default=(200, b"not json"))

        with self.assertRaises(tts_service.TtsUpstreamError):
            await tts_service.synthesize("안녕하세요", post)

        self.assertEqual(self.cached_files(), [])

    async def test_empty_audio_is_an_upstream_error(self):
        post = _FakePost(default=_audio_response(b""))

        with self.assertRaises(tts_service.TtsUpstreamError):
            await tts_service.synthesize("안녕하세요", post)

    async def test_a_failure_does_not_poison_the_next_attempt(self):
        post = _FakePost(default=(500, b"boom"))
        with self.assertRaises(tts_service.TtsUpstreamError):
            await tts_service.synthesize("안녕하세요", post)

        post.default = _audio_response()
        self.assertEqual(await tts_service.synthesize("안녕하세요", post), _AUDIO)


class ApiKeySecrecyTest(TtsServiceTestCase):
    async def test_a_rejected_request_does_not_log_the_api_key(self):
        body = json.dumps({"error": {"message": "API key not valid"}}).encode("utf-8")
        post = _FakePost(default=(403, body))

        with self.assertLogs(_LOGGER_NAME, level="WARNING") as logs:
            with self.assertRaises(tts_service.TtsUpstreamError):
                await tts_service.synthesize("안녕하세요", post)

        rendered = _rendered(logs.records)
        self.assertIn("403", rendered)
        self.assertIn("API key not valid", rendered)
        self.assertNotIn(_API_KEY, rendered)

    async def test_a_transport_failure_does_not_log_the_api_key(self):
        post = _FakePost(default=OSError(f"POST failed with key={_API_KEY}"))

        with self.assertLogs(_LOGGER_NAME, level="WARNING") as logs:
            with self.assertRaises(tts_service.TtsUpstreamError):
                await tts_service.synthesize("안녕하세요", post)

        self.assertNotIn(_API_KEY, _rendered(logs.records))

    async def test_prewarm_does_not_log_the_api_key(self):
        post = _FakePost(default=OSError(f"POST failed with key={_API_KEY}"))

        with mock.patch.object(tts_service, "_post_json", post):
            with self.assertLogs(_LOGGER_NAME, level="WARNING") as logs:
                await tts_service.prewarm(["안녕하세요"])

        self.assertNotIn(_API_KEY, _rendered(logs.records))

    async def test_a_successful_synthesis_does_not_log_the_api_key(self):
        with self.assertLogs(_LOGGER_NAME, level="INFO") as logs:
            await tts_service.synthesize("안녕하세요", _FakePost())

        self.assertNotIn(_API_KEY, _rendered(logs.records))


class PrewarmTest(TtsServiceTestCase):
    async def test_prewarm_synthesizes_every_phrase(self):
        post = _FakePost()

        with mock.patch.object(tts_service, "_post_json", post):
            await tts_service.prewarm(["첫 문장", "둘째 문장"])

        self.assertEqual([call["text"] for call in post.calls], ["첫 문장", "둘째 문장"])

    async def test_prewarm_keeps_going_after_a_failure(self):
        post = _FakePost(per_text={"실패할 문장": (500, b"boom")})

        with mock.patch.object(tts_service, "_post_json", post):
            with self.assertLogs(_LOGGER_NAME, level="WARNING"):
                await tts_service.prewarm(["실패할 문장", "성공할 문장"])

        self.assertEqual(
            [call["text"] for call in post.calls],
            ["실패할 문장", "성공할 문장"],
        )
        self.assertEqual(len(self.cached_files()), 1)

    async def test_prewarm_without_an_api_key_does_nothing(self):
        post = _FakePost()

        with mock.patch.object(tts_service, "GOOGLE_TTS_API_KEY", ""):
            with mock.patch.object(tts_service, "_post_json", post):
                with self.assertLogs(_LOGGER_NAME, level="INFO") as logs:
                    await tts_service.prewarm(["첫 문장"])

        self.assertEqual(post.calls, [])
        self.assertEqual(len(logs.records), 1)


if __name__ == "__main__":
    unittest.main()