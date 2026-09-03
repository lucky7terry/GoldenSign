import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _activity_class():
    """session_websocket 은 mediapipe 를 끌어오므로 클래스 정의만 떼어 실행한다."""
    import ast

    source = (Path(__file__).resolve().parents[1]
              / "app" / "api" / "session_websocket.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "ConnectionActivity")
    namespace = {"monotonic": time.monotonic}
    exec(compile(ast.Module([node], []), "<activity>", "exec"), namespace)
    return namespace["ConnectionActivity"]


ConnectionActivity = _activity_class()


class ConnectionActivityTest(unittest.TestCase):
    """WHEP 스트리밍 중 유휴 오판으로 연결이 끊기지 않는지 고정한다."""

    def test_new_connection_starts_idle_at_zero(self):
        self.assertLess(ConnectionActivity().idle_seconds(), 0.1)

    def test_idle_grows_without_activity(self):
        activity = ConnectionActivity()
        time.sleep(0.05)

        self.assertGreaterEqual(activity.idle_seconds(), 0.05)

    def test_touch_resets_idle(self):
        activity = ConnectionActivity()
        time.sleep(0.05)
        activity.touch()

        self.assertLess(activity.idle_seconds(), 0.05)

    def test_outbound_only_traffic_keeps_connection_alive(self):
        # WHEP 경로: 클라이언트는 아무것도 안 보내고 서버만 결과를 내보낸다.
        # 그래도 유휴로 판정되면 안 된다.
        activity = ConnectionActivity()
        for _ in range(4):
            time.sleep(0.02)
            activity.touch()          # 결과 송신 1회

        self.assertLess(activity.idle_seconds(), 0.02)


if __name__ == "__main__":
    unittest.main()
