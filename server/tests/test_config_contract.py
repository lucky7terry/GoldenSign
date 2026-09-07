"""app.config 에서 가져다 쓰는 이름이 실제로 있는지 정적으로 확인한다.

서버 모듈 대부분은 mediapipe / tensorflow / aiortc 를 끌어와서 CI 에서
import 할 수 없다. 그래서 `from app.config import X` 의 X 가 사라져도
테스트가 전부 통과하고, 서버를 띄우는 순간 ImportError 로 죽는다.

실제로 그런 일이 있었다. config 를 편집하다 INCLUDE_KEYPOINTS_IN_RESULT 를
지웠는데 테스트 95개가 전부 통과했다 - model_service 를 import 하는 테스트가
하나도 없었기 때문이다.

app.config 는 os 와 math 만 쓰므로 CI 에서 안전하게 import 할 수 있다.
나머지 모듈은 import 하지 않고 소스만 파싱한다.
"""

import ast
import math
import sys
import os
from unittest import mock
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.config as config  # noqa: E402

_APP_ROOT = Path(__file__).resolve().parents[1] / "app"
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _config_imports() -> list[tuple[Path, int, str]]:
    """`from app.config import ...` 로 가져다 쓰는 이름을 전부 모은다."""
    found = []
    for root in (_APP_ROOT, _SCRIPTS_ROOT):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "app.config":
                    continue
                for alias in node.names:
                    found.append((path, node.lineno, alias.name))
    return found


class ConfigContractTest(unittest.TestCase):
    def test_every_imported_name_exists(self):
        missing = [
            f"{path.name}:{line} -> {name}"
            for path, line, name in _config_imports()
            if not hasattr(config, name)
        ]

        self.assertEqual(
            missing,
            [],
            "app.config 에 없는 이름을 import 하고 있다. 서버가 뜨지 않는다:\n"
            + "\n".join(missing),
        )

    def test_something_is_actually_imported(self):
        """위 테스트가 빈 목록을 훑고 통과하는 것을 막는다."""
        self.assertGreater(len(_config_imports()), 10)


class ProbabilityBoundsTest(unittest.TestCase):
    """임계값이 (0, 1] 을 벗어나면 기동 때 세운다.

    _env_float 는 1.1 도 그냥 받는다. 확률은 1 을 넘을 수 없으므로 그런
    값이 들어오면 모든 구간이 조용히 거절되고, 서버는 아무 불평 없이
    단어를 하나도 못 낸다. 오타 하나로 그렇게 되는 것보다 못 뜨는 편이 낫다.
    """

    @staticmethod
    def _parse(raw: str):
        source = (Path(__file__).resolve().parents[1]
                  / "app" / "config.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted = ("_env_float", "_env_probability")
        nodes = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in wanted]
        assert len(nodes) == len(wanted), "헬퍼 이름이 바뀌었다"
        namespace = {"os": os, "math": math}
        exec(compile(ast.Module(nodes, []), "<config>", "exec"), namespace)
        with mock.patch.dict(os.environ, {"T": raw}):
            return namespace["_env_probability"]("T", 0.5)

    def test_a_valid_threshold_passes(self):
        self.assertAlmostEqual(self._parse("0.62"), 0.62)

    def test_one_is_allowed(self):
        self.assertAlmostEqual(self._parse("1.0"), 1.0)

    def test_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            self._parse("1.1")

    def test_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            self._parse("0")

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            self._parse("-0.2")


if __name__ == "__main__":
    unittest.main()
