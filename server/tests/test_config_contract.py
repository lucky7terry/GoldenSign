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
import sys
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


if __name__ == "__main__":
    unittest.main()
