"""키포인트 추출을 여러 워커로 나눠 돌리는 부분.

MediaPipe 한 세트는 프레임을 하나씩만 처리해서 코어 1개만 쓴다. 세트를
여러 개 두고(MediaPipeServicePool), WHEP 프레임을 동시에 최대 N개까지
처리해서 코어 수만큼 처리량을 늘린다. 실제 MediaPipe 없이 가짜 서비스로
"동시에 몇 개까지 도는지"와 "카운터가 맞는지"를 고정한다.
"""

import asyncio
import ast
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SERVICE = Path(__file__).resolve().parents[1] / "app" / "services" / "mediapipe_service.py"


def _load_pool_class():
    """mediapipe/cv2 없이 MediaPipeServicePool 만 떼어 실행한다."""
    tree = ast.parse(_SERVICE.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MediaPipeServicePool"
    )
    namespace = {
        "queue": __import__("queue"),
        "contextmanager": __import__("contextlib").contextmanager,
        "Any": object,
        "np": types.SimpleNamespace(ndarray=object),
    }
    exec(compile(ast.Module([node], []), "<pool>", "exec"), namespace)
    return namespace["MediaPipeServicePool"]


MediaPipeServicePool = _load_pool_class()


class _SlowService:
    """한 번에 하나만 처리해야 하는 서비스. 겹쳐 들어오면 실패로 남긴다."""

    def __init__(self, tracker, delay=0.1):
        self._busy = threading.Lock()
        self._tracker = tracker
        self._delay = delay
        self.closed = False

    def extract_keypoints_from_image(self, image):
        if not self._busy.acquire(blocking=False):
            self._tracker["overlap_on_same_service"] += 1
            return None
        try:
            with self._tracker["lock"]:
                self._tracker["running"] += 1
                self._tracker["peak"] = max(
                    self._tracker["peak"], self._tracker["running"]
                )
            time.sleep(self._delay)
            with self._tracker["lock"]:
                self._tracker["running"] -= 1
            return {"image": image}
        finally:
            self._busy.release()

    def close(self):
        self.closed = True


def _tracker():
    return {"lock": threading.Lock(), "running": 0, "peak": 0,
            "overlap_on_same_service": 0}


class ServicePoolTest(unittest.TestCase):
    def _run_threads(self, pool, n_calls):
        results = []
        threads = [
            threading.Thread(
                target=lambda i=i: results.append(
                    pool.extract_keypoints_from_image(i)
                )
            )
            for i in range(n_calls)
        ]
        started = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, time.perf_counter() - started

    def test_two_services_process_two_frames_at_once(self):
        tracker = _tracker()
        pool = MediaPipeServicePool([_SlowService(tracker), _SlowService(tracker)])

        results, elapsed = self._run_threads(pool, 4)

        self.assertEqual(tracker["peak"], 2)
        # 4장 x 0.1초를 2개가 나눠 하면 0.2초 근처. 직렬이면 0.4초.
        self.assertLess(elapsed, 0.35)
        self.assertEqual(sorted(r["image"] for r in results), [0, 1, 2, 3])

    def test_a_service_is_never_used_by_two_threads_at_once(self):
        tracker = _tracker()
        pool = MediaPipeServicePool([_SlowService(tracker, 0.02) for _ in range(2)])

        self._run_threads(pool, 20)

        self.assertEqual(tracker["overlap_on_same_service"], 0)
        self.assertLessEqual(tracker["peak"], 2)

    def test_service_is_returned_even_when_extraction_fails(self):
        class _Failing:
            def extract_keypoints_from_image(self, image):
                raise RuntimeError("boom")

        pool = MediaPipeServicePool([_Failing()])
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                pool.extract_keypoints_from_image(None)
        # 빌린 서비스를 돌려주지 않았다면 두 번째 호출에서 영원히 멈춘다.

    def test_close_closes_every_service(self):
        services = [_SlowService(_tracker()) for _ in range(3)]
        MediaPipeServicePool(services).close()
        self.assertTrue(all(s.closed for s in services))

    def test_empty_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            MediaPipeServicePool([])


class _Frame:
    pts = None
    time_base = None


class WhepConcurrentProcessingTest(unittest.TestCase):
    """_process_latest_video_frames 가 프레임을 동시에 몇 개까지 넘기는지."""

    def setUp(self):
        from app.services import whep_service

        self.whep = whep_service
        self.service = object.__new__(whep_service.WhepPullService)
        self.running = 0
        self.peak = 0
        self.processed = 0
        self.lock = threading.Lock()

    def _fake_recognize(self, session_id, generation, video_frame, captured_at_ms):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        # 프레임은 30fps 로 들어오는데 처리는 0.1초 걸린다 - 워커가
        # 1개면 초당 10장, 2개면 20장. 입력(30장)이 병목이 되지 않게 잡았다.
        time.sleep(0.1)
        with self.lock:
            self.running -= 1
            self.processed += 1
        return {"word": {}, "keypoints": {}}, {"loaded": True}

    def _handle(self):
        async def _send(_payload):
            return None

        return types.SimpleNamespace(
            session_id="s",
            stream_id="st",
            client_message_id=None,
            send_json=_send,
            stop_event=asyncio.Event(),
            on_frame=None,
        )

    async def _drive(self, workers, seconds=0.6):
        handle = self._handle()
        queue = asyncio.Queue(maxsize=1)
        with mock.patch.object(self.whep, "KEYPOINT_WORKERS", workers), \
             mock.patch.object(self.whep, "_recognize_whep_frame", self._fake_recognize), \
             mock.patch.object(self.whep, "_current_word_generation", lambda _s: 1), \
             mock.patch.object(self.whep, "_keypoint_summary", lambda _k: ""):
            task = asyncio.create_task(
                self.service._process_latest_video_frames(handle, queue)
            )
            # 30fps 로 프레임이 들어온다. 큐에는 항상 최신 1장만 남는다.
            deadline = time.monotonic() + seconds
            n = 0
            while time.monotonic() < deadline:
                n += 1
                self.whep.WhepPullService._replace_latest_video_frame(queue, n, _Frame())
                await asyncio.sleep(1 / 30)
            handle.stop_event.set()
            await asyncio.wait_for(task, timeout=3)

    def test_single_worker_keeps_the_old_one_at_a_time_behavior(self):
        asyncio.run(self._drive(workers=1))
        self.assertEqual(self.peak, 1)

    def test_two_workers_run_two_frames_at_once_and_double_throughput(self):
        asyncio.run(self._drive(workers=1))
        serial = self.processed
        self.processed = 0
        self.peak = 0

        asyncio.run(self._drive(workers=2))

        self.assertEqual(self.peak, 2)
        self.assertGreater(self.processed, serial * 1.5)

    def test_closed_session_stops_processing(self):
        from app.services.word_segment_service import WordSessionClosed

        def _closed(*_args):
            raise WordSessionClosed("closed")

        async def _run():
            handle = self._handle()
            queue = asyncio.Queue(maxsize=1)
            with mock.patch.object(self.whep, "KEYPOINT_WORKERS", 2), \
                 mock.patch.object(self.whep, "_recognize_whep_frame", _closed), \
                 mock.patch.object(self.whep, "_current_word_generation", lambda _s: 1):
                task = asyncio.create_task(
                    self.service._process_latest_video_frames(handle, queue)
                )
                self.whep.WhepPullService._replace_latest_video_frame(queue, 1, _Frame())
                with self.assertRaises(self.whep.WhepFrameProcessingStopped):
                    await asyncio.wait_for(task, timeout=3)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
