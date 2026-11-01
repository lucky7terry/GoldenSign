"""GET /v1/tts 의 status 매핑과 기동 때 미리 합성할 문구.

app.api.tts 는 FastAPI 를 끌어오고 CI 는 requirements-test.txt 만 설치하므로
(pydantic, numpy) import 할 수 없다. test_connection_activity 와 같은 방식으로
필요한 정의만 소스에서 떼어 실행하고, Response / JSONResponse / synthesize 를
가짜로 넣는다. httpx 가 없어 TestClient 도 쓸 수 없다.
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rate_limit import RateLimitVerdict, SlidingWindowRateLimiter
from app.services.label_service import LabelError, NUM_CLASSES, all_words
from app.services.tts_service import (
    TtsInvalidText,
    TtsUnavailable,
    TtsUpstreamError,
)

_SERVER = Path(__file__).resolve().parents[1]
_SOURCE = (_SERVER / "app" / "api" / "tts.py").read_text(encoding="utf-8")
_MAIN_SOURCE = (_SERVER / "app" / "main.py").read_text(encoding="utf-8")
_MINIAPP_TTS = (
    _SERVER.parent / "miniapp" / "src" / "background" / "tts.ts"
).read_text(encoding="utf-8")

_AUDIO = b"fake-mp3-bytes"
_WANTED_CONSTANTS = {
    "AUDIO_MEDIA_TYPE",
    "CACHE_CONTROL",
    "_rate_limiter",
    "START_ANNOUNCEMENT",
    "RETRY_ANNOUNCEMENT",
    "FRAMING_ANNOUNCEMENT",
    "ANNOUNCEMENTS",
}
_WANTED_FUNCTIONS = {
    "prewarm_phrases",
    "_error",
    "_client_host",
    "get_tts_audio",
    "_rate_limited",
}


class _FakeResponse:
    """fastapi.Response 대역. 생성 인자를 그대로 들고 있는다."""

    def __init__(self, content=None, media_type=None, headers=None, status_code=200):
        self.content = content
        self.media_type = media_type
        self.headers = headers or {}
        self.status_code = status_code


class _FakeJSONResponse:
    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class _FakeRequest:
    """fastapi.Request 대역. 라우터가 읽는 것은 client.host 뿐이다."""

    def __init__(self, host="203.0.113.1"):
        self.client = None if host is None else SimpleNamespace(host=host)


class _FakeIsCached:
    """캐시 조회 대역. 캐시 hit 이면 라우터가 요청 제한을 세지 않는다."""

    def __init__(self, cached=True, error=None):
        self.cached = cached
        self.error = error
        self.calls = []

    async def __call__(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.cached


def _load(**overrides) -> dict:
    """라우터의 상수와 함수만 떼어 실행한 이름 공간."""
    tree = ast.parse(_SOURCE)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in _WANTED_CONSTANTS
            for target in node.targets
        ):
            nodes.append(node)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in _WANTED_FUNCTIONS
        ):
            # @router.get 은 FastAPI 를 부르므로 떼어 낸다.
            node.decorator_list = []
            nodes.append(node)

    assert len(nodes) == len(_WANTED_CONSTANTS) + len(_WANTED_FUNCTIONS), (
        "라우터의 상수나 함수 이름이 바뀌었다"
    )

    namespace = {
        "logger": logging.getLogger("app.api.tts"),
        "Response": _FakeResponse,
        "JSONResponse": _FakeJSONResponse,
        # 라우터 시그니처의 애너테이션은 def 시점에 평가된다.
        "Request": _FakeRequest,
        "SlidingWindowRateLimiter": SlidingWindowRateLimiter,
        "RateLimitVerdict": RateLimitVerdict,
        # 실제 기본값(20 / 60)으로는 한도 테스트가 느려진다.
        "TTS_RATE_LIMIT_PER_IP": 2,
        "TTS_RATE_LIMIT_GLOBAL": 10,
        "all_words": all_words,
        "LabelError": LabelError,
        "TtsInvalidText": TtsInvalidText,
        "TtsUnavailable": TtsUnavailable,
        "TtsUpstreamError": TtsUpstreamError,
        "is_cached": _FakeIsCached(cached=True),
    }
    namespace.update(overrides)
    exec(compile(ast.Module(nodes, []), "<tts_api>", "exec"), namespace)
    return namespace


class _RecordingSynthesize:
    def __init__(self, result=_AUDIO, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def __call__(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.result


class TtsEndpointSuccessTest(unittest.IsolatedAsyncioTestCase):
    async def test_audio_is_returned_as_mpeg(self):
        synthesize = _RecordingSynthesize()
        api = _load(synthesize=synthesize)

        response = await api["get_tts_audio"](_FakeRequest(), "안녕하세요")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, _AUDIO)
        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertEqual(synthesize.calls, ["안녕하세요"])

    async def test_the_response_can_be_cached_for_a_day(self):
        api = _load(synthesize=_RecordingSynthesize())

        response = await api["get_tts_audio"](_FakeRequest(), "안녕하세요")

        self.assertEqual(response.headers["Cache-Control"], "public, max-age=86400")

    async def test_text_reaches_the_service_unchanged(self):
        synthesize = _RecordingSynthesize()
        api = _load(synthesize=synthesize)

        await api["get_tts_audio"](_FakeRequest(), "  배  ")

        self.assertEqual(synthesize.calls, ["  배  "])


class TtsEndpointErrorTest(unittest.IsolatedAsyncioTestCase):
    async def _failing(self, error):
        api = _load(synthesize=_RecordingSynthesize(error=error))
        return await api["get_tts_audio"](_FakeRequest(), "안녕하세요")

    async def test_invalid_text_is_400(self):
        response = await self._failing(TtsInvalidText("empty"))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content["code"], "tts_invalid_text")

    async def test_a_missing_api_key_is_503(self):
        response = await self._failing(TtsUnavailable("no key"))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.content["code"], "tts_unavailable")

    async def test_an_upstream_failure_is_502(self):
        response = await self._failing(TtsUpstreamError("status 403"))

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.content["code"], "tts_upstream_error")

    async def test_the_error_body_carries_a_message(self):
        response = await self._failing(TtsInvalidText("text must not be empty"))

        self.assertEqual(
            response.content,
            {"code": "tts_invalid_text", "message": "text must not be empty"},
        )

    async def test_a_missing_text_parameter_is_400_not_422(self):
        """빠뜨린 것과 비워 보낸 것을 같은 code 로 돌려준다."""
        synthesize = _RecordingSynthesize(error=TtsInvalidText("empty"))
        api = _load(synthesize=synthesize)

        response = await api["get_tts_audio"](_FakeRequest())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(synthesize.calls, [""])


class TtsRateLimitTest(unittest.IsolatedAsyncioTestCase):
    """제한은 Google 을 새로 부르는 요청에만 걸린다.

    인증은 두지 않는다 - 토큰을 두어도 앱 번들에 그대로 들어가므로 가져다
    쓰는 것을 막지 못한다. 막아야 하는 것은 "임의의 문장으로 유료 API 를
    반복 호출하는 것"이고, 캐시 hit 은 그 호출을 하지 않는다. 앱이 실제로
    읽는 문구는 기동 때 미리 합성되므로 제한에 닿지 않는다.
    """

    def _api(self, cached: bool, **overrides):
        self.synthesize = _RecordingSynthesize()
        self.is_cached = _FakeIsCached(cached=cached)
        return _load(
            synthesize=self.synthesize,
            is_cached=self.is_cached,
            **overrides,
        )

    async def test_a_cache_hit_is_never_rate_limited(self):
        api = self._api(cached=True)

        for _ in range(5):
            response = await api["get_tts_audio"](_FakeRequest(), "안녕하세요")
            self.assertEqual(response.status_code, 200)

        # 한도는 2 다. 캐시 hit 을 셌다면 세 번째에서 429 가 났다.
        self.assertEqual(len(self.synthesize.calls), 5)

    async def test_cache_misses_over_the_ip_limit_are_429(self):
        api = self._api(cached=False)
        request = _FakeRequest("198.51.100.7")

        for _ in range(2):
            self.assertEqual(
                (await api["get_tts_audio"](request, "안녕하세요")).status_code,
                200,
            )
        with self.assertLogs("app.api.tts", level="WARNING"):
            response = await api["get_tts_audio"](request, "안녕하세요")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.content["code"], "tts_rate_limited")
        # 거절한 요청은 합성까지 가지 않는다.
        self.assertEqual(len(self.synthesize.calls), 2)

    async def test_a_rejected_request_carries_retry_after(self):
        api = self._api(cached=False)
        request = _FakeRequest("198.51.100.7")
        for _ in range(2):
            await api["get_tts_audio"](request, "안녕하세요")

        with self.assertLogs("app.api.tts", level="WARNING"):
            response = await api["get_tts_audio"](request, "안녕하세요")

        self.assertGreaterEqual(int(response.headers["Retry-After"]), 1)

    async def test_another_ip_is_not_affected(self):
        api = self._api(cached=False)
        exhausted = _FakeRequest("198.51.100.7")
        for _ in range(3):
            await api["get_tts_audio"](exhausted, "안녕하세요")

        response = await api["get_tts_audio"](_FakeRequest("203.0.113.9"), "안녕하세요")

        self.assertEqual(response.status_code, 200)

    async def test_the_global_limit_applies_across_ips(self):
        api = self._api(cached=False, TTS_RATE_LIMIT_PER_IP=5, TTS_RATE_LIMIT_GLOBAL=2)

        for index in range(2):
            response = await api["get_tts_audio"](
                _FakeRequest(f"203.0.113.{index}"), "안녕하세요"
            )
            self.assertEqual(response.status_code, 200)

        with self.assertLogs("app.api.tts", level="WARNING"):
            response = await api["get_tts_audio"](_FakeRequest("203.0.113.99"), "안녕")

        self.assertEqual(response.status_code, 429)

    async def test_invalid_text_does_not_consume_the_limit(self):
        """빈 text 를 반복해서 남의 몫을 깎을 수 있으면 안 된다."""
        is_cached = _FakeIsCached(error=TtsInvalidText("empty"))
        synthesize = _RecordingSynthesize()
        api = _load(synthesize=synthesize, is_cached=is_cached)
        request = _FakeRequest("198.51.100.7")

        for _ in range(5):
            response = await api["get_tts_audio"](request, "")
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.content["code"], "tts_invalid_text")

        self.assertEqual(synthesize.calls, [])
        # 한도가 깎이지 않았으므로 멀쩡한 요청은 통과한다.
        api["is_cached"] = _FakeIsCached(cached=False)
        self.assertEqual(
            (await api["get_tts_audio"](request, "안녕하세요")).status_code,
            200,
        )

    async def test_a_request_without_a_client_is_still_counted(self):
        """client 가 None 인 전송(ASGI 테스트 클라이언트 등)도 한 키로 센다."""
        api = self._api(cached=False)

        self.assertEqual(api["_client_host"](_FakeRequest(None)), "unknown")
        for _ in range(2):
            await api["get_tts_audio"](_FakeRequest(None), "안녕하세요")
        with self.assertLogs("app.api.tts", level="WARNING"):
            response = await api["get_tts_audio"](_FakeRequest(None), "안녕하세요")

        self.assertEqual(response.status_code, 429)


class PrewarmPhrasesTest(unittest.TestCase):
    def test_announcements_come_with_every_word(self):
        api = _load()

        phrases = api["prewarm_phrases"]()

        self.assertEqual(len(phrases), len(api["ANNOUNCEMENTS"]) + NUM_CLASSES)
        self.assertEqual(phrases[: len(api["ANNOUNCEMENTS"])], api["ANNOUNCEMENTS"])
        for word in all_words():
            self.assertIn(word, phrases)

    def test_every_phrase_is_speakable(self):
        for phrase in _load()["prewarm_phrases"]():
            self.assertTrue(phrase.strip())

    def test_missing_labels_still_prewarm_the_announcements(self):
        def broken_all_words():
            raise LabelError("label file is missing")

        api = _load(all_words=broken_all_words)

        with self.assertLogs("app.api.tts", level="WARNING"):
            phrases = api["prewarm_phrases"]()

        self.assertEqual(phrases, api["ANNOUNCEMENTS"])


class AnnouncementContractTest(unittest.TestCase):
    """서버가 미리 합성하는 문구는 앱이 읽는 문구와 글자까지 같아야 한다.

    다르면 미리 만든 음성이 캐시에서 한 번도 맞지 않는다 - 캐시 키가 텍스트의
    sha256 이라, 마침표 하나만 달라도 착용자는 매번 합성을 기다린다.
    """

    def test_the_start_announcement_matches_the_miniapp(self):
        self.assertIn(_load()["START_ANNOUNCEMENT"], _MINIAPP_TTS)

    def test_the_retry_announcement_matches_the_miniapp(self):
        self.assertIn(_load()["RETRY_ANNOUNCEMENT"], _MINIAPP_TTS)

    def test_the_framing_announcement_matches_the_miniapp(self):
        self.assertIn(_load()["FRAMING_ANNOUNCEMENT"], _MINIAPP_TTS)

    def test_every_announcement_is_read_by_the_miniapp(self):
        """앱이 읽지 않는 문구를 미리 합성하면 그 왕복은 버려진다."""
        for announcement in _load()["ANNOUNCEMENTS"]:
            with self.subTest(announcement=announcement):
                self.assertIn(announcement, _MINIAPP_TTS)


class LifespanPrewarmTest(unittest.TestCase):
    """prewarm 은 preload 두 개가 끝난 뒤 뒤에서 돌고, 종료 때 정리된다.

    main 도 FastAPI 를 끌어오므로 소스 순서로 확인한다.
    """

    def test_prewarm_starts_after_both_preloads(self):
        mediapipe = _MAIN_SOURCE.index("preload_mediapipe_service()")
        recognition = _MAIN_SOURCE.index("preload_recognition_model()")
        prewarm = _MAIN_SOURCE.index("asyncio.create_task(prewarm(")

        self.assertLess(mediapipe, prewarm)
        self.assertLess(recognition, prewarm)

    def test_prewarm_does_not_block_startup(self):
        """await 로 기다리면 기동이 쉰 번의 네트워크 왕복만큼 늦어진다."""
        self.assertNotIn("await prewarm(", _MAIN_SOURCE)

    def test_the_prewarm_task_is_cancelled_on_shutdown(self):
        self.assertIn("prewarm_task.cancel()", _MAIN_SOURCE)
        self.assertIn("return_exceptions=True", _MAIN_SOURCE)

    def test_the_router_is_registered(self):
        self.assertIn("app.include_router(tts_router)", _MAIN_SOURCE)


if __name__ == "__main__":
    unittest.main()
