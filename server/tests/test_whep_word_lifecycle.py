"""WHEP frame ownership, bounded draining, readiness and retry regressions."""
import asyncio
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from app.services import whep_service as whep
from app.services.word_segment_service import WordSegmentStore, WordTooShort, OPENPOSE_FEATURE_DIM


class WordOwnershipTest(unittest.TestCase):
    def test_late_frame_cannot_enter_next_word(self):
        store = WordSegmentStore()
        generation = store.start_session("s")
        store.start_word("s", generation)
        token = store.frame_token("s", generation)
        self.assertIs(store.begin_close("s", generation), token)
        self.assertIs(store.frame_token("s", generation), False)
        row = [0.0] * OPENPOSE_FEATURE_DIM
        self.assertTrue(store.append("s", row, generation=generation, frame_token=token).buffered)
        with self.assertRaises(WordTooShort):
            store.end_word("s", generation)
        store.start_word("s", generation)
        self.assertFalse(store.append("s", row, generation=generation, frame_token=token).buffered)
        self.assertFalse(store.append("s", row, generation=generation, frame_token=False).buffered)
        self.assertTrue(store.append("s", row, generation=generation,
                                     frame_token=store.frame_token("s", generation)).buffered)


class WhepLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = whep.WhepPullService()
        self.messages = []
        async def send(payload):
            self.messages.append(payload)
        self.handle = whep.WhepStreamHandle("s", "stream", "https://example.com", None,
                                            send, asyncio.Event())
        self.service._streams["s"] = self.handle
        self.store = WordSegmentStore()
        self.generation = self.store.start_session("s")
        self.store.start_word("s", self.generation)
        self.store_patch = patch("app.services.word_segment_service.word_store", self.store)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)

    def admit(self):
        return self.service._admit_frame(self.handle, SimpleNamespace(pts=None, time_base=None))

    async def test_drain_waits_for_received_frame_and_logs_counts(self):
        item = self.admit()
        token = self.store.begin_close("s", self.generation)
        with self.assertLogs(whep.logger, level="INFO") as logs:
            drain = asyncio.create_task(self.service.drain_word("s", token, 1))
            await asyncio.sleep(0)
            self.assertFalse(drain.done())
            item.counts["processed"] += 1
            item.finish("buffered")
            await drain
        record = logs.records[0]
        self.assertEqual((record.received, record.processed, record.buffered), (1, 1, 1))
        self.assertEqual(record.pending_after_drain, 0)

    async def test_drain_timeout_does_not_cancel_worker_completion(self):
        item = self.admit()
        token = self.store.begin_close("s", self.generation)
        with self.assertLogs(whep.logger, level="INFO") as logs:
            await self.service.drain_word("s", token, 0.001)
        self.assertFalse(item.done.done())
        self.assertEqual(logs.records[0].pending_after_drain, 1)
        item.finish("not_buffered")

    async def test_queue_replacement_releases_old_frame(self):
        queue = asyncio.Queue(maxsize=1)
        old, new = self.admit(), self.admit()
        self.service._replace_latest_video_frame(queue, 1, old)
        self.service._replace_latest_video_frame(queue, 2, new)
        self.assertTrue(old.done.done())
        self.assertEqual(old.counts["queue_replaced"], 1)
        self.assertFalse(new.done.done())
        new.finish()

    async def test_processing_passes_receipt_token_and_records_failures(self):
        item = self.admit()
        def recognize(*args):
            self.assertIs(args[-1], item.token)
            raise ValueError("invalid frame")
        with patch.object(whep, "_recognize_whep_frame", recognize):
            with self.assertLogs(whep.logger, level="ERROR"):
                await self.service._process_one_video_frame(self.handle, whep._FrameProcessingState(), 1, item)
        self.assertEqual(item.counts["validation_failed"], 1)
        self.assertTrue(item.done.done())

    async def test_slow_extraction_is_drained_before_word_closes(self):
        import threading
        item = self.admit()
        entered, release = threading.Event(), threading.Event()
        row = [0.0] * OPENPOSE_FEATURE_DIM
        def recognize(session_id, generation, frame, timestamp, token):
            entered.set()
            release.wait(2)
            result = self.store.append(session_id, row, generation=generation, frame_token=token)
            return {"word": {"buffered": result.buffered}}, {}
        with patch.object(whep, "_recognize_whep_frame", recognize):
            worker = asyncio.create_task(self.service._process_one_video_frame(
                self.handle, whep._FrameProcessingState(), 1, item))
            try:
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                token = self.store.begin_close("s", self.generation)
                drain = asyncio.create_task(self.service.drain_word("s", token, 1))
                await asyncio.sleep(0)
                self.assertFalse(drain.done())
                release.set()
                await drain
                await worker
                self.assertEqual(item.counts["buffered"], 1)
                with self.assertRaisesRegex(WordTooShort, "has 1 frames"):
                    self.store.end_word("s", self.generation)
            finally:
                release.set()
                await worker

    async def test_old_finalizer_cannot_close_replacement_word(self):
        from app.services.word_segment_service import WordNotStarted
        token = self.store.begin_close("s", self.generation)
        self.store.abort_word("s", self.generation)
        self.store.start_word("s", self.generation)
        replacement = self.store.frame_token("s", self.generation)
        with self.assertRaises(WordNotStarted):
            self.store.end_word("s", self.generation, frame_token=token)
        self.assertIs(self.store.frame_token("s", self.generation), replacement)
        self.assertTrue(self.store.is_word_open("s"))

    async def test_cancelled_async_task_keeps_real_worker_pending(self):
        import threading
        item = self.admit()
        entered, release = threading.Event(), threading.Event()
        row = [0.0] * OPENPOSE_FEATURE_DIM
        def recognize(session_id, generation, frame, timestamp, token):
            entered.set()
            release.wait(5)
            result = self.store.append(session_id, row, generation=generation, frame_token=token)
            return {"word": {"buffered": result.buffered}}, {}
        with patch.object(whep, "_recognize_whep_frame", recognize):
            worker = asyncio.create_task(self.service._process_one_video_frame(
                self.handle, whep._FrameProcessingState(), 1, item))
            try:
                async def wait_entered():
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                await asyncio.wait_for(wait_entered(), 2)
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                self.assertFalse(item.done.done())
                token = self.store.begin_close("s", self.generation)
                with self.assertLogs(whep.logger, level="INFO") as logs:
                    await self.service.drain_word("s", token, 0.001)
                self.assertEqual(logs.records[0].pending_after_drain, 1)
                self.assertEqual(logs.records[0].cancellation_requested, 1)
                self.store.abort_word("s", self.generation)
                self.store.start_word("s", self.generation)
                replacement = self.store.frame_token("s", self.generation)
                release.set()
                await asyncio.wait_for(asyncio.shield(item.done), 2)
                self.assertEqual(item.counts["processed"], 1)
                self.assertEqual(item.counts["not_buffered"], 1)
                self.assertEqual(item.counts["cancelled"], 0)
                self.assertIs(self.store.frame_token("s", self.generation), replacement)
            finally:
                release.set()
                await asyncio.gather(worker, return_exceptions=True)

    async def test_ready_requires_recent_frame_and_live_stream(self):
        self.assertFalse(await self.service.is_ready("s"))
        self.handle.ready = True
        self.handle.last_received_at = whep.monotonic()
        self.assertTrue(await self.service.is_ready("s"))
        self.handle.last_received_at -= whep.WHEP_FRAME_IDLE_TIMEOUT_SECONDS + 1
        self.assertFalse(await self.service.is_ready("s"))
        self.service._streams.clear()
        self.assertFalse(await self.service.is_ready("s"))

    async def test_stable_connection_resets_consecutive_failures(self):
        attempts = 0
        async def connect(handle):
            nonlocal attempts
            attempts += 1
            handle.stable = attempts == 2
            raise RuntimeError("lost")
        with patch.object(self.service, "_connect_and_receive", connect), \
             patch.object(whep, "WHEP_MAX_RETRIES", 3), \
             patch.object(whep, "WHEP_RETRY_DELAY_SECONDS", 0.001):
            with self.assertLogs(whep.logger, level="WARNING"):
                await self.service._run_stream(self.handle)
        self.assertEqual(attempts, 4)
        self.assertEqual(self.messages[-1]["code"], "stream_unavailable")

    async def test_unstable_connections_exhaust_budget(self):
        attempts = 0
        async def connect(handle):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("lost")
        with patch.object(self.service, "_connect_and_receive", connect), \
             patch.object(whep, "WHEP_MAX_RETRIES", 3), \
             patch.object(whep, "WHEP_RETRY_DELAY_SECONDS", 0.001):
            with self.assertLogs(whep.logger, level="WARNING"):
                await self.service._run_stream(self.handle)
        self.assertEqual(attempts, 3)
