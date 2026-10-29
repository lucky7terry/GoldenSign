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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    "START_ANNOUNCEMENT",
    "RETRY_ANNOUNCEMENT",
    "FRAMING_ANNOUNCEMENT",
    "ANNOUNCEMENTS",
}
_WANTED_FUNCTIONS = {"prewarm_phrases", "_error", "get_tts_audio"}


class _FakeResponse:
    """fastapi.Response 대역. 생성 인자를 그대로 들고 있는다."""

    def __init__(self, content=None, media_type=None, headers=None, status_code=200):
        self.content = content
        self.media_type = media_type
        self.headers = headers or {}
        self.status_code = status_code


class _FakeJSONResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content


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
        "all_words": all_words,
        "LabelError": LabelError,
        "TtsInvalidText": TtsInvalidText,
        "TtsUnavailable": TtsUnavailable,
        "TtsUpstreamError": TtsUpstreamError,
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

        response = await api["get_tts_audio"]("안녕하세요")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, _AUDIO)
        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertEqual(synthesize.calls, ["안녕하세요"])

    async def test_the_response_can_be_cached_for_a_day(self):
        api = _load(synthesize=_RecordingSynthesize())

        response = await api["get_tts_audio"]("안녕하세요")

        self.assertEqual(response.headers["Cache-Control"], "public, max-age=86400")

    async def test_text_reaches_the_service_unchanged(self):
        synthesize = _RecordingSynthesize()
        api = _load(synthesize=synthesize)

        await api["get_tts_audio"]("  배  ")

        self.assertEqual(synthesize.calls, ["  배  "])


class TtsEndpointErrorTest(unittest.IsolatedAsyncioTestCase):
    async def _failing(self, error):
        api = _load(synthesize=_RecordingSynthesize(error=error))
        return await api["get_tts_audio"]("안녕하세요")

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

        response = await api["get_tts_audio"]()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(synthesize.calls, [""])


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

    FRAMING_ANNOUNCEMENT 는 앱이 아직 읽지 않으므로 앱 쪽 커밋에서 같이 고정한다.
    """

    def test_the_start_announcement_matches_the_miniapp(self):
        self.assertIn(_load()["START_ANNOUNCEMENT"], _MINIAPP_TTS)

    def test_the_retry_announcement_matches_the_miniapp(self):
        self.assertIn(_load()["RETRY_ANNOUNCEMENT"], _MINIAPP_TTS)


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
