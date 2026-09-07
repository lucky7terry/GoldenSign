"""워밍업에 실패한 모델을 쓰지 못하게 막는다.

모델 객체가 만들어졌다고 인식이 되는 것은 아니다. 라벨 파일의 단어 수가
모델의 클래스 수와 다르거나 WORD_TARGET_FRAMES 가 학습 길이와 어긋나면,
모델은 멀쩡히 로딩되지만 그 출력을 라벨로 옮기는 순간 엉뚱한 단어가 나온다.

이 검사는 preload 의 _warm_up 에서 하는데, 그때는 이미 _model 이 채워진
뒤다. 그래서 _model 만 보고 판단하면 실패한 모델을 그대로 쓰게 되고
/health 는 loaded: true 를 보고한다. 조용히 틀리는 것이 가장 나쁘다.
"""

import ast
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SOURCE = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "recognition_model.py").read_text(encoding="utf-8")

_WANTED = (
    "get_recognition_model",
    "get_recognition_predictor",
    "recognition_model_available",
    "recognition_model_error",
    "_remember_failure",
)


class _Unavailable(Exception):
    pass


def _module(*, model=None, predictor=None, error=None, loader=None):
    """recognition_model 은 텐서플로를 끌어오므로 함수 정의만 떼어 실행한다."""
    tree = ast.parse(_SOURCE)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _WANTED]
    assert len(nodes) == len(_WANTED), "함수 이름이 바뀌었다"

    namespace = {
        "_model": model,
        "_predictor": predictor,
        "_initialization_error": error,
        "_model_lock": threading.Lock(),
        "_predictor_lock": threading.Lock(),
        "RecognitionModelUnavailableError": _Unavailable,
        "load_recognition_model": loader or (lambda: object()),
        "make_predictor": lambda m: ("predictor", m),
    }
    exec(compile(ast.Module(nodes, []), "<recognition_model>", "exec"), namespace)
    return namespace


class WarmUpFailureTest(unittest.TestCase):

    def test_a_loaded_model_that_failed_warm_up_is_not_available(self):
        ns = _module(model=object())
        self.assertTrue(ns["recognition_model_available"]())

        ns["_remember_failure"](RuntimeError("Label file has 49 words but the model has 50 classes."))

        self.assertFalse(ns["recognition_model_available"]())

    def test_the_model_getter_raises_instead_of_returning_the_cached_model(self):
        ns = _module(model=object())
        ns["_remember_failure"](RuntimeError("sequence length mismatch"))

        with self.assertRaises(_Unavailable):
            ns["get_recognition_model"]()

    def test_the_predictor_getter_raises_instead_of_returning_the_cached_one(self):
        """캐시된 predictor 가 있어도 막아야 한다.

        워밍업은 predictor 를 만든 뒤 한 번 돌려 보는 순서라, 실패한
        시점에 predictor 가 이미 캐시에 남아 있을 수 있다.
        """
        ns = _module(model=object(), predictor=("predictor", "stale"))
        ns["_remember_failure"](RuntimeError("graph execution failed"))

        with self.assertRaises(_Unavailable):
            ns["get_recognition_predictor"]()

    def test_the_failure_reason_survives_for_health(self):
        ns = _module(model=object())
        ns["_remember_failure"](RuntimeError("Label file has 49 words"))

        self.assertIn("49 words", ns["recognition_model_error"]())


class HealthyPathTest(unittest.TestCase):

    def test_a_model_loads_once_and_is_reused(self):
        calls = []

        def loader():
            calls.append(1)
            return "model"

        ns = _module(loader=loader)

        self.assertEqual(ns["get_recognition_model"](), "model")
        self.assertEqual(ns["get_recognition_model"](), "model")
        self.assertEqual(len(calls), 1, "두 번 부르면 3~8초를 다시 낸다")

    def test_a_loading_failure_is_remembered_instead_of_retried(self):
        calls = []

        def loader():
            calls.append(1)
            raise _Unavailable("model file not found")

        ns = _module(loader=loader)

        with self.assertRaises(_Unavailable):
            ns["get_recognition_model"]()
        with self.assertRaises(_Unavailable):
            ns["get_recognition_model"]()

        self.assertEqual(len(calls), 1, "단어마다 로딩을 재시도하면 안 된다")
        self.assertFalse(ns["recognition_model_available"]())


if __name__ == "__main__":
    unittest.main()
