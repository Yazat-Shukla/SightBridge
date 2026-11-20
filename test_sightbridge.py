import io
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

import app


class MockSpeechBackend:
    def __init__(self, output, lock, fail_once=False, block_start=None, block_release=None,
                 timeout_text=None, delay=0.0):
        self.output = output
        self.lock = lock
        self.fail_once = fail_once
        self.block_start = block_start
        self.block_release = block_release
        self.timeout_text = timeout_text
        self.delay = delay
        self.voice_count = 2
        self.closed = False
        self.terminated = False
        self.close_calls = 0
        self.terminate_calls = 0

    def speak(self, text, timeout, on_accepted=None):
        if on_accepted is not None:
            on_accepted()
        if text == "Block" and self.block_start is not None:
            self.block_start.set()
            self.block_release.wait(5.0)
        if text == self.timeout_text or self.delay > timeout:
            raise TimeoutError("mock backend timeout")
        if self.delay:
            time.sleep(self.delay)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("mock utterance failure")
        with self.lock:
            self.output.append(text)
        return {"status": "completed"}

    def close(self, timeout=2.0):
        self.close_calls += 1
        self.closed = True

    def terminate(self):
        self.terminate_calls += 1
        self.terminated = True
        self.closed = True
        if self.block_release is not None:
            self.block_release.set()


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class SightBridgeUnitTests(unittest.TestCase):
    def setUp(self):
        self.output = []
        self.output_lock = threading.Lock()
        self.backends = []
        self.worker = None

    def make_worker(self, fail_first=False, timeout_first=False):
        def factory():
            backend = MockSpeechBackend(
                self.output, self.output_lock,
                fail_once=fail_first and not self.backends,
                timeout_text="Timeout" if timeout_first and not self.backends else None,
            )
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0)
        self.assertTrue(self.worker.ready)
        return self.worker

    def tearDown(self):
        if self.worker is not None:
            self.worker.stop(timeout=1.0)

    def test_speech_worker_processes_consecutive_announcements(self):
        worker = self.make_worker()
        self.assertTrue(worker.say("Person ahead", key="track:1"))
        self.assertTrue(worker.say("Chair on your left", key="track:2"))
        self.assertTrue(worker.wait_idle())
        self.assertEqual(self.output, ["Person ahead", "Chair on your left"])

    def test_speech_worker_survives_utterance_exception(self):
        worker = self.make_worker(fail_first=True)
        self.assertTrue(worker.say("First", key="track:1"))
        self.assertTrue(wait_until(lambda: worker.failure_generation > 0))
        self.assertTrue(wait_until(lambda: worker.ready))
        self.assertTrue(worker.say("Second", key="track:2"))
        self.assertTrue(worker.wait_idle())
        self.assertIn("Second", self.output)
        self.assertTrue(worker.thread.is_alive())

    def test_backend_timeout_is_reported_and_worker_recovers(self):
        worker = self.make_worker(timeout_first=True)
        self.assertTrue(worker.say("Timeout", key="track:1"))
        self.assertTrue(wait_until(lambda: worker.failure_generation > 0))
        self.assertTrue(wait_until(lambda: worker.ready))
        self.assertTrue(worker.say("After timeout", key="track:2"))
        self.assertTrue(worker.wait_idle())
        self.assertIn("After timeout", self.output)
        self.assertNotIn("Timeout", self.output)

    def test_shutdown_closes_backend_and_joins_worker(self):
        worker = self.make_worker()
        backend = self.backends[0]
        worker.stop(timeout=1.0)
        worker.stop(timeout=1.0)
        self.assertFalse(worker.thread.is_alive())
        self.assertTrue(backend.closed)
        self.assertFalse(backend.terminated)
        self.assertEqual(backend.close_calls, 1)

    def test_shutdown_waits_for_active_speech_and_discards_pending_requests(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(
                self.output, self.output_lock,
                block_start=block_start, block_release=block_release,
            )
            self.backends.append(backend)
            return backend

        worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0)
        self.worker = worker
        self.assertTrue(worker.say("Block", key="active"))
        self.assertTrue(block_start.wait(1.0))
        self.assertTrue(worker.say("Pending one", key="track:1"))
        self.assertTrue(worker.say("Pending two", key="track:2"))
        backend = self.backends[0]

        stopper = threading.Thread(target=worker.stop, kwargs={"timeout": 0.05})
        stopper.start()
        self.assertTrue(wait_until(lambda: worker._stopping))
        self.assertTrue(worker.thread.is_alive())
        self.assertFalse(backend.terminated)
        self.assertFalse(backend.closed)

        block_release.set()
        stopper.join(timeout=2.0)
        self.assertFalse(stopper.is_alive())
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(self.output, ["Block"])
        self.assertTrue(backend.closed)
        self.assertFalse(backend.terminated)
        self.assertEqual(backend.close_calls, 1)
        self.assertEqual(worker.consume_dropped_keys(), {"track:1", "track:2"})

    def test_shutdown_timeout_uses_forced_termination_fallback(self):
        started = threading.Event()
        release = threading.Event()

        class UnresponsiveBackend(MockSpeechBackend):
            def speak(self, text, timeout, on_accepted=None):
                started.set()
                release.wait(10.0)
                return {"status": "completed"}

        def factory():
            backend = UnresponsiveBackend(self.output, self.output_lock, block_release=release)
            self.backends.append(backend)
            return backend

        worker = app.SpeechWorker(
            backend_factory=factory, init_timeout=1.0, request_timeout=0.01,
        )
        self.worker = worker
        self.assertTrue(worker.say("Hanging request"))
        self.assertTrue(started.wait(1.0))
        worker.stop(timeout=0.01)
        self.assertFalse(worker.thread.is_alive())
        self.assertTrue(self.backends[0].terminated)
        self.assertEqual(self.backends[0].terminate_calls, 1)

    def test_speech_diagnostic_runs_six_sequential_backend_requests(self):
        backends = []

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock)
            backends.append(backend)
            return backend

        self.assertEqual(app.run_speech_diagnostic(factory, request_timeout=1.0), 0)
        self.assertEqual(len(self.output), 6)
        self.assertTrue(backends[0].closed)

    def test_voice_disable_and_reenable(self):
        worker = self.make_worker()
        worker.set_enabled(False)
        self.assertFalse(worker.say("Suppressed"))
        worker.set_enabled(True)
        self.assertTrue(worker.say("Enabled again"))
        self.assertTrue(worker.wait_idle())
        self.assertEqual(self.output, ["Enabled again"])

    def test_already_expired_speech_job_is_rejected(self):
        worker = self.make_worker()
        self.assertFalse(worker.say("Stale distance", expires_at=time.monotonic() - 1.0))
        self.assertTrue(worker.wait_idle())
        self.assertEqual(self.output, [])

    def test_distance_announcement_expiring_while_queued_is_skipped(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0)
        self.assertTrue(self.worker.say("Block", key="system"))
        self.assertTrue(block_start.wait(1.0))
        deadline = time.monotonic() + 0.20
        self.assertTrue(self.worker.say("Queued stale distance", expires_at=deadline, key="track:4"))
        time.sleep(0.25)
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertNotIn("Queued stale distance", self.output)
        self.assertIn("track:4", self.worker.consume_dropped_keys())

    def test_newer_same_track_announcement_replaces_outdated_queued_message(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0)
        self.assertTrue(self.worker.say("Block", key="system"))
        self.assertTrue(block_start.wait(1.0))
        self.assertTrue(self.worker.say("Person ahead, 2 metres", key="track:8"))
        self.assertTrue(self.worker.say("Person on your right, 1.5 metres", key="track:8"))
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertEqual(self.output, ["Block", "Person on your right, 1.5 metres"])

    def test_urgent_close_obstacle_announcement_takes_priority_over_ordinary_pending_job(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(
            backend_factory=factory, init_timeout=1.0, max_pending=1,
        )
        self.assertTrue(self.worker.say("Block", key="system"))
        self.assertTrue(block_start.wait(1.0))
        self.assertTrue(self.worker.say("Person ahead", priority=1, key="track:1"))
        self.assertTrue(self.worker.say("Obstacle close on your left", priority=3, key="track:2"))
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertEqual(self.output, ["Block", "Obstacle close on your left"])

    def test_lower_priority_same_track_update_cannot_replace_pending_close_warning(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0, max_pending=2)
        self.assertTrue(self.worker.say("Block", key="active"))
        self.assertTrue(block_start.wait(1.0))
        self.assertTrue(self.worker.say("Close person", priority=3, key="track:5"))
        self.assertFalse(self.worker.say("Person ahead", priority=1, key="track:5"))
        with self.worker._condition:
            pending = [job for job in self.worker._jobs if job.key == "track:5"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].text, "Close person")
        self.assertEqual(pending[0].priority, 3)
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertEqual(self.output, ["Block", "Close person"])

    def test_expired_close_warning_can_be_replaced_by_current_track_update(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0)
        self.assertTrue(self.worker.say("Block", key="active"))
        self.assertTrue(block_start.wait(1.0))
        self.assertTrue(self.worker.say(
            "Expiring close warning", priority=3, key="track:6",
            expires_at=time.monotonic() + 0.05,
        ))
        time.sleep(0.07)
        self.assertTrue(self.worker.say("Current ordinary update", priority=1, key="track:6"))
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertEqual(self.output, ["Block", "Current ordinary update"])

    def test_speech_queue_capacity_remains_bounded(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0, max_pending=2)
        self.assertTrue(self.worker.say("Block", key="active"))
        self.assertTrue(block_start.wait(1.0))
        for index in range(5):
            self.assertTrue(self.worker.say(f"Pending {index}", priority=1, key=f"track:{index}"))
            with self.worker._condition:
                self.assertLessEqual(len(self.worker._jobs), 2)
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertEqual(len(self.output), 3)

    def test_speech_worker_revalidates_object_state_before_backend_playback(self):
        block_start = threading.Event()
        block_release = threading.Event()

        def factory():
            backend = MockSpeechBackend(self.output, self.output_lock, block_start=block_start,
                                        block_release=block_release)
            self.backends.append(backend)
            return backend

        self.worker = app.SpeechWorker(backend_factory=factory, init_timeout=1.0)
        self.worker.set_job_validator(lambda _job: "track_disappeared")
        self.assertTrue(self.worker.say("Block", key="system"))
        self.assertTrue(block_start.wait(1.0))
        token = app.SpeechValidationToken(4, "person", "ahead", 1.5, "unknown")
        self.assertTrue(self.worker.say("Person ahead, 1.5 metres", key="track:4", validation_token=token))
        block_release.set()
        self.assertTrue(self.worker.wait_idle())
        self.assertNotIn("Person ahead, 1.5 metres", self.output)
        self.assertIn("track:4", self.worker.consume_dropped_keys())

    def test_pending_speech_validation_tracks_distance_and_observation_changes(self):
        token = app.SpeechValidationToken(4, "person", "ahead", 1.56, "unknown")
        live = app.SpeechLiveState("person", "ahead", 10.0, 1.56, 9.8, "unknown")
        self.assertIsNone(app.validate_pending_announcement(token, live, 10.1, 0.6, 0.6, 20.0))
        self.assertEqual(
            app.validate_pending_announcement(token, None, 10.1, 0.6, 0.6, 20.0),
            "track_disappeared",
        )
        changed = app.SpeechLiveState("person", "right", 10.0, 1.56, 9.8, "unknown")
        self.assertEqual(
            app.validate_pending_announcement(token, changed, 10.1, 0.6, 0.6, 20.0),
            "track_description_changed",
        )
        no_longer_fresh = app.SpeechLiveState("person", "ahead", 10.0, 1.56, 9.0, "unknown")
        self.assertEqual(
            app.validate_pending_announcement(token, no_longer_fresh, 10.1, 0.6, 0.6, 20.0),
            "distance_expired",
        )
        distance_available = app.SpeechLiveState("person", "ahead", 10.0, 1.56, 10.0, "unknown")
        unavailable_token = app.SpeechValidationToken(4, "person", "ahead", None, "unknown")
        self.assertEqual(
            app.validate_pending_announcement(unavailable_token, distance_available, 10.1, 0.6, 0.6, 20.0),
            "distance_became_available",
        )
        ambiguous = app.SpeechLiveState("person", "ahead", 10.0, 1.56, 10.0, "unknown", True)
        self.assertEqual(
            app.validate_pending_announcement(token, ambiguous, 10.1, 0.6, 0.6, 20.0),
            "track_identity_uncertain",
        )

    def test_depth_worker_tracks_pending_replacement_and_unconsumed_result(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingEstimator:
            device = "cpu"

            def predict(self, _frame):
                if not started.is_set():
                    started.set()
                    release.wait(1.0)
                return np.full((4, 4), 2.0, dtype=np.float32)

        worker = app.DepthInferenceWorker(BlockingEstimator(), None)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(1, time.perf_counter(), frame, ()))
            self.assertTrue(started.wait(1.0))
            self.assertTrue(worker.submit(2, time.perf_counter(), frame, ()))
            self.assertTrue(worker.submit(3, time.perf_counter(), frame, ()))
            release.set()
            self.assertTrue(wait_until(lambda: worker.latest_result() is not None
                                       and worker.latest_result().frame_id == 3))
            status = worker.status_snapshot()
            self.assertEqual(status["submitted"], 3)
            self.assertEqual(status["pending_replaced"], 1)
            self.assertEqual(status["completed"], 2)
            self.assertEqual(status["unconsumed_result_replaced"], 1)
            worker.mark_consumed(3)
            self.assertEqual(worker.status_snapshot()["last_consumed_frame_id"], 3)
        finally:
            release.set()
            worker.stop(timeout=1.0)

    def test_depth_worker_failure_is_reported_without_losing_worker_state(self):
        class FailingEstimator:
            device = "cpu"

            def predict(self, _frame):
                raise RuntimeError("synthetic depth failure")

        worker = app.DepthInferenceWorker(FailingEstimator(), None)
        try:
            self.assertTrue(worker.submit(12, time.perf_counter(), np.zeros((4, 4, 3), dtype=np.uint8), ()))
            self.assertTrue(wait_until(lambda: worker.status_snapshot()["failures"] == 1))
            status = worker.status_snapshot()
            self.assertEqual(status["last_error"], "synthetic depth failure")
            self.assertIn("DEPTH_WORKER_FAILURE", app.depth_worker_unavailable_reason(12, status))
        finally:
            worker.stop(timeout=1.0)

    def test_depth_unavailable_diagnostics_distinguish_wait_states(self):
        pending = {"pending_frame_id": 24, "processing_frame_id": None, "failures": 0,
                   "last_submitted_frame_id": 24, "latest_completed_frame_id": 18,
                   "submitted": 5, "pending_replaced": 2, "completed": 3,
                   "unconsumed_result_replaced": 0}
        self.assertIn("DEPTH_JOB_PENDING(frame=24)", app.depth_worker_unavailable_reason(24, pending))
        processing = dict(pending, pending_frame_id=None, processing_frame_id=24)
        self.assertIn("DEPTH_WORKER_PROCESSING(frame=24)", app.depth_worker_unavailable_reason(24, processing))
        no_job = dict(pending, pending_frame_id=None, processing_frame_id=None, last_submitted_frame_id=18)
        self.assertIn("NO_DEPTH_JOB_SUBMITTED_FOR_FRAME", app.depth_worker_unavailable_reason(24, no_job))

    def test_announcement_new_available_changed_and_cooldown(self):
        should, priority, reason = app.announcement_decision(None, "ahead", None, "unknown", 1.0)
        self.assertTrue(should)
        self.assertEqual(reason, "new_object")
        previous = app.AnnouncementState("ahead", 1.0, "unknown", None)
        should, _priority, reason = app.announcement_decision(previous, "ahead", 1.56, "unknown", 1.2)
        self.assertTrue(should)
        self.assertEqual(reason, "distance_availability_changed")
        previous = app.AnnouncementState("ahead", 1.0, "unknown", 2.0)
        should, _priority, reason = app.announcement_decision(previous, "ahead", 2.1, "unknown", 2.0)
        self.assertFalse(should)
        self.assertEqual(reason, "unchanged_or_cooldown")
        should, _priority, reason = app.announcement_decision(previous, "ahead", 2.7, "unknown", 5.0)
        self.assertTrue(should)
        self.assertEqual(reason, "distance_changed")
        should, _priority, reason = app.announcement_decision(previous, "left", 2.0, "unknown", 2.0)
        self.assertFalse(should)
        self.assertEqual(reason, "unchanged_or_cooldown")
        should, _priority, reason = app.announcement_decision(previous, "left", 2.0, "unknown", 5.0)
        self.assertTrue(should)
        self.assertEqual(reason, "direction_changed")

    def test_unchanged_close_obstacle_is_not_reannounced(self):
        previous = app.AnnouncementState("ahead", 1.0, "stationary", 1.0, "person")
        should, _priority, reason = app.announcement_decision(
            previous, "ahead", 1.0, "stationary", 1.1,
            label="person", close_confirmed=True,
        )
        self.assertFalse(should)
        self.assertEqual(reason, "unchanged_or_cooldown")

    def test_global_ordinary_cooldown_and_confirmed_close_bypass(self):
        cooldown = app.ORDINARY_ANNOUNCEMENT_GLOBAL_COOLDOWN_SECONDS
        self.assertEqual(
            app.ordinary_announcement_suppression_reason(10.0, 10.1, close_confirmed=False),
            "global_cooldown",
        )
        self.assertIsNone(app.ordinary_announcement_suppression_reason(
            10.0, 10.0 + cooldown, close_confirmed=False,
        ))
        self.assertIsNone(app.ordinary_announcement_suppression_reason(
            10.0, 10.1, close_confirmed=True,
        ))

    def test_crowded_scene_orders_nearby_and_fairly_rotates_tracks(self):
        recent = app.DetectionSnapshot(1, True, "person", 0.9, "ahead", (0, 0, 10, 10))
        waiting = app.DetectionSnapshot(2, True, "person", 0.8, "left", (0, 0, 10, 10))
        distant = app.DetectionSnapshot(3, True, "person", 0.99, "right", (0, 0, 10, 10))
        ordered = sorted([
            (recent, 1.2, 20.0, True),
            (waiting, 1.1, 5.0, True),
            (distant, 3.0, None, True),
        ], key=lambda entry: app.announcement_candidate_key(*entry))
        self.assertEqual([entry[0].track_id for entry in ordered], [2, 1, 3])
        # After track 2 receives a turn, the still-near track 1 gets the next one.
        next_order = sorted([
            (recent, 1.2, 20.0, True),
            (waiting, 1.1, 22.0, True),
        ], key=lambda entry: app.announcement_candidate_key(*entry))
        self.assertEqual(next_order[0][0].track_id, 1)

    def test_close_obstacle_bypasses_ordinary_cooldown(self):
        previous = app.AnnouncementState("ahead", 5.0, "stationary", 2.0)
        should, priority, reason = app.announcement_decision(previous, "ahead", 1.0, "stationary", 5.1)
        self.assertTrue(should)
        self.assertEqual(priority, 3)
        self.assertEqual(reason, "close_obstacle")

    def test_distance_expiry_does_not_announce_repeated_unavailable(self):
        previous = app.AnnouncementState("ahead", 1.0, "unknown", 1.56)
        should, _priority, reason = app.announcement_decision(previous, "ahead", None, "unknown", 1.1)
        self.assertFalse(should)
        self.assertEqual(reason, "unchanged_or_cooldown")
        self.assertEqual(app.readable_distance(None, 20.0), "distance unavailable")

    def test_duplicate_same_class_box_is_suppressed_but_separate_person_is_kept(self):
        first = app.DetectionSnapshot(1, True, "person", 0.90, "ahead", (10, 10, 100, 150))
        duplicate = app.DetectionSnapshot(2, True, "person", 0.70, "ahead", (11, 11, 101, 151))
        separate = app.DetectionSnapshot(3, True, "person", 0.80, "right", (90, 10, 180, 150))
        duplicate_track_row = app.DetectionSnapshot(1, True, "person", 0.60, "ahead", (12, 12, 102, 152))
        kept, removed = app.suppress_duplicate_detections([first, duplicate, separate, duplicate_track_row])
        self.assertEqual({item.track_id for item in kept}, {1, 3})
        self.assertEqual(len(removed), 2)

    def test_duplicate_cross_class_geometry_is_suppressed(self):
        laptop = app.DetectionSnapshot(15, True, "laptop", 0.82, "ahead", (110, 246, 509, 690))
        book = app.DetectionSnapshot(22, True, "book", 0.75, "ahead", (112, 248, 512, 691))
        kept, removed = app.suppress_duplicate_detections([laptop, book])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].label, "laptop")
        self.assertEqual(len(removed), 1)

    def test_near_full_height_duplicate_person_boxes_keep_older_track_id(self):
        # Recorded frame 609: the broad, lower-confidence person box almost fully
        # contains the stable person's box but has IoU below the ordinary cutoff.
        stable_person = app.DetectionSnapshot(
            3, True, "person", 0.753, "ahead", (525, 226, 1015, 712)
        )
        broad_duplicate = app.DetectionSnapshot(
            9, True, "person", 0.504, "ahead", (68, 228, 1016, 714)
        )
        self.assertLess(app.bbox_iou(stable_person.bbox, broad_duplicate.bbox), 0.82)
        kept, removed = app.suppress_duplicate_detections([stable_person, broad_duplicate])
        self.assertEqual([item.track_id for item in kept], [3])
        self.assertEqual([(winner.track_id, dropped.track_id) for winner, dropped, _ in removed], [(3, 9)])

    def test_track_state_purge_clears_only_that_ids_depth_and_motion_history(self):
        person_state = app.VisionTrackState(stable_label="person")
        book_state = app.VisionTrackState(stable_label="book")
        last_seen = {3: 10.0, 9: 10.1}
        direction = {3: "ahead", 9: "left"}
        vision = {3: person_state, 9: book_state}
        distance = {3: app.DistanceState(1.8, 10.0), 9: app.DistanceState(0.9, 10.1)}
        histories = {3: deque([(10.0, 1.8)]), 9: deque([(10.1, 0.9)])}
        raw_histories = {3: deque([(10.0, 1.8)]), 9: deque([(10.1, 0.9)])}
        pending_depth = {3: (10.0, 1.7), 9: (10.1, 0.8)}
        depth_diagnostics = {3: {"reason": "ok"}, 9: {"reason": "ok"}}
        motion = {3: "stationary", 9: "approaching"}
        announcements = {
            3: app.AnnouncementState("ahead", 10.0, "stationary", 1.8, "person"),
            9: app.AnnouncementState("left", 10.1, "approaching", 0.9, "book"),
        }
        app.purge_application_track_state(
            9,
            last_seen_by_track=last_seen,
            direction_by_track=direction,
            vision_state_by_track=vision,
            distance_state_by_track=distance,
            histories=histories,
            raw_histories=raw_histories,
            pending_depth_by_track=pending_depth,
            depth_diagnostic_by_track=depth_diagnostics,
            motion_by_track=motion,
            last_announced=announcements,
        )
        for mapping in (
            last_seen, direction, vision, distance, histories, raw_histories,
            pending_depth, depth_diagnostics, motion, announcements,
        ):
            self.assertNotIn(9, mapping)
            self.assertIn(3, mapping)
        self.assertIs(vision[3], person_state)
        self.assertEqual(distance[3], app.DistanceState(1.8, 10.0))
        self.assertEqual(list(histories[3]), [(10.0, 1.8)])
        self.assertEqual(motion[3], "stationary")
        self.assertIsNone(app.fresh_track_distance(distance.get(9), 10.2, 0.6, 20.0))
        self.assertEqual(app.fresh_track_distance(distance[3], 10.2, 0.6, 20.0), 1.8)

    def test_reacquired_track_keeps_its_class_history_separate_from_new_id(self):
        person_state = app.VisionTrackState()
        person = app.DetectionSnapshot(3, True, "person", 0.90, "ahead", (500, 200, 900, 700))
        app.stabilize_detection(person, person_state, 1280, observed_at=1.0)
        original_history = tuple(person_state.class_history)

        # Same ByteTrack ID returns after a brief miss; low-confidence class noise
        # must not erase its prior class vote.
        low_confidence_flip = app.DetectionSnapshot(
            3, True, "book", 0.18, "ahead", (501, 201, 901, 700)
        )
        reacquired = app.stabilize_detection(
            low_confidence_flip, person_state, 1280, observed_at=1.3
        )
        self.assertEqual(reacquired.track_id, 3)
        self.assertEqual(reacquired.label, "person")
        self.assertEqual(tuple(person_state.class_history), original_history)

        # A different ID gets independent history; there is no state transfer.
        book_state = app.VisionTrackState()
        book = app.DetectionSnapshot(9, True, "book", 0.90, "ahead", (100, 200, 450, 650))
        app.stabilize_detection(book, book_state, 1280, observed_at=1.4)
        self.assertEqual(book_state.stable_label, "book")
        self.assertEqual(len(book_state.class_history), 1)
        self.assertEqual(tuple(person_state.class_history), original_history)
        self.assertIsNot(person_state, book_state)

    def test_long_reacquisition_resets_old_track_history_but_short_gap_preserves_it(self):
        state = app.VisionTrackState()
        person = app.DetectionSnapshot(3, True, "person", 0.90, "ahead", (500, 200, 900, 700))
        app.stabilize_detection(person, state, 1280, observed_at=1.0)
        original_history = tuple(state.class_history)
        vision = {3: state, 9: app.VisionTrackState(stable_label="book")}
        last_seen = {3: 1.0, 9: 1.0}
        direction = {3: "ahead", 9: "left"}
        distance = {3: app.DistanceState(1.8, 1.0), 9: app.DistanceState(0.9, 1.0)}
        histories = {3: deque([(1.0, 1.8)]), 9: deque([(1.0, 0.9)])}
        raw_histories = {3: deque([(1.0, 1.8)]), 9: deque([(1.0, 0.9)])}
        pending = {3: (1.0, 1.7), 9: (1.0, 0.8)}
        diagnostics = {3: {}, 9: {}}
        motion = {3: "stationary", 9: "approaching"}
        announcements = {3: object(), 9: object()}

        self.assertFalse(app.reacquisition_needs_state_reset(state, 1.0, 1.5))
        self.assertEqual(tuple(state.class_history), original_history)
        self.assertTrue(app.reacquisition_needs_state_reset(state, 1.0, 3.3))
        reset = app.reset_reacquired_track_state(
            3,
            last_seen_by_track=last_seen,
            direction_by_track=direction,
            vision_state_by_track=vision,
            distance_state_by_track=distance,
            histories=histories,
            raw_histories=raw_histories,
            pending_depth_by_track=pending,
            depth_diagnostic_by_track=diagnostics,
            motion_by_track=motion,
            last_announced=announcements,
        )
        self.assertIs(vision[3], reset)
        self.assertIsNone(reset.stable_label)
        self.assertEqual(len(reset.class_history), 0)
        self.assertFalse(reset.announcement_eligible)
        self.assertEqual(last_seen[3], 1.0)  # retained only to report the reacquisition gap
        for mapping in (direction, distance, histories, raw_histories, pending, diagnostics, motion, announcements):
            self.assertNotIn(3, mapping)
            self.assertIn(9, mapping)

    def test_two_nearby_separate_objects_are_not_suppressed(self):
        left = app.DetectionSnapshot(1, True, "person", 0.90, "left", (100, 100, 240, 360))
        right = app.DetectionSnapshot(2, True, "person", 0.88, "right", (220, 100, 360, 360))
        kept, removed = app.suppress_duplicate_detections([left, right])
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, [])

    def test_overlapping_nested_people_are_not_merged_by_containment(self):
        taller_person = app.DetectionSnapshot(2, True, "person", 0.91, "ahead", (490, 185, 1024, 715))
        lower_person = app.DetectionSnapshot(4, True, "person", 0.56, "ahead", (507, 485, 1012, 714))
        kept, removed = app.suppress_duplicate_detections([taller_person, lower_person])
        self.assertEqual({item.track_id for item in kept}, {2, 4})
        self.assertEqual(removed, [])

    def test_partial_person_with_stable_face_occluder_is_uncertain_not_merged(self):
        parent_state = app.VisionTrackState(
            stable_label="person", smoothed_bbox=(490.0, 185.0, 1024.0, 715.0),
            announcement_eligible=True,
        )
        book_state = app.VisionTrackState(
            stable_label="book",
            class_history=deque([(4.8, "book", 0.57), (4.9, "book", 0.39)], maxlen=11),
        )
        partial_state = app.VisionTrackState(
            stable_label="person", smoothed_bbox=(507.0, 485.0, 1012.0, 714.0),
            announcement_eligible=True,
        )
        states = {2: parent_state, 3: book_state, 4: partial_state}
        last_seen = {2: 4.95}
        parent_depth = app.DistanceState(1.8, 4.9)
        child_depth = app.DistanceState(1.2, 4.95)
        distance_by_track = {2: parent_depth, 4: child_depth}
        parent_announcement = app.AnnouncementState("ahead", 4.8, "stationary", 1.8, "person")
        announcements = {2: parent_announcement}
        child = app.DetectionSnapshot(4, True, "person", 0.49, "ahead", (507, 485, 1012, 714))
        book = app.DetectionSnapshot(3, True, "book", 0.18, "ahead", (576, 241, 834, 584))

        candidate = app.find_person_occlusion_ambiguity(
            child, [child, book], states, last_seen, now=5.0
        )
        uncertain_child = app.replace(
            child,
            identity_uncertain=candidate is not None,
            possible_parent_track_id=candidate.possible_parent_track_id if candidate else None,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.possible_parent_track_id, 2)
        self.assertEqual(candidate.occluder_track_id, 3)
        self.assertFalse(app.is_detection_announcement_eligible(uncertain_child, partial_state))
        self.assertTrue(app.is_detection_announcement_eligible(
            app.DetectionSnapshot(2, True, "person", 0.93, "ahead", (490, 185, 1024, 715)),
            parent_state,
        ))
        self.assertIs(states[2], parent_state)
        self.assertIs(states[4], partial_state)
        self.assertIs(distance_by_track[2], parent_depth)
        self.assertIs(distance_by_track[4], child_depth)
        self.assertIs(announcements[2], parent_announcement)

    def test_nested_person_without_reliable_face_occluder_is_not_flagged(self):
        parent_state = app.VisionTrackState(
            stable_label="person", smoothed_bbox=(490.0, 185.0, 1024.0, 715.0)
        )
        weak_object_state = app.VisionTrackState(stable_label="book")
        child = app.DetectionSnapshot(4, True, "person", 0.56, "ahead", (507, 485, 1012, 714))
        unrelated_object = app.DetectionSnapshot(3, True, "book", 0.18, "ahead", (20, 20, 80, 100))
        self.assertIsNone(app.find_person_occlusion_ambiguity(
            child, [child, unrelated_object], {2: parent_state, 3: weak_object_state},
            {2: 4.95}, now=5.0,
        ))

    def test_person_uncertainty_clears_when_face_occluder_moves_away(self):
        parent_state = app.VisionTrackState(
            stable_label="person", smoothed_bbox=(490.0, 185.0, 1024.0, 715.0)
        )
        book_state = app.VisionTrackState(
            stable_label="book",
            class_history=deque([(4.8, "book", 0.57), (4.9, "book", 0.39)], maxlen=11),
        )
        partial_state = app.VisionTrackState(
            stable_label="person", strong_evidence=deque([(4.8, 0.9), (4.9, 0.9)], maxlen=5),
            announcement_eligible=True,
        )
        states = {2: parent_state, 3: book_state, 4: partial_state}
        child = app.DetectionSnapshot(4, True, "person", 0.70, "ahead", (507, 485, 1012, 714))
        book_covering_face = app.DetectionSnapshot(3, True, "book", 0.57, "ahead", (576, 241, 834, 584))
        self.assertIsNotNone(app.find_person_occlusion_ambiguity(
            child, [child, book_covering_face], states, {2: 4.95}, now=5.0,
        ))

        # The child remains detected, but the book has moved away from the face region.
        moved_book = app.replace(book_covering_face, bbox=(900, 500, 980, 620))
        self.assertIsNone(app.find_person_occlusion_ambiguity(
            child, [child, moved_book], states, {2: 4.95}, now=5.05,
        ))
        current, entered, cleared = app.reconcile_person_uncertainty(
            {4}, set(), states,
        )
        self.assertEqual(current, set())
        self.assertEqual(entered, set())
        self.assertEqual(cleared, {4})
        self.assertFalse(partial_state.announcement_eligible)
        self.assertEqual(len(partial_state.strong_evidence), 0)

    def test_person_uncertainty_clears_on_detection_loss_before_tracker_removal(self):
        state = app.VisionTrackState(
            stable_label="person", announcement_eligible=True,
            strong_evidence=deque([(1.0, 0.9)], maxlen=5),
        )
        current, _entered, cleared = app.reconcile_person_uncertainty({4}, set(), {4: state})
        self.assertEqual(current, set())
        self.assertEqual(cleared, {4})
        self.assertFalse(state.announcement_eligible)
        self.assertFalse(app.application_tracks_to_purge(
            {4}, {4: 1.0}, 2.0, removed_tracker_ids=set(),
            tracker_lifecycle_available=True,
        ))
        self.assertEqual(app.application_tracks_to_purge(
            {4}, {4: 1.0}, 2.0, removed_tracker_ids={4},
            tracker_lifecycle_available=True,
        ), {4})

    def test_uncertain_person_requires_fresh_evidence_before_announcement_recovery(self):
        state = app.VisionTrackState(
            stable_label="person", announcement_eligible=True,
            strong_evidence=deque([(0.0, 0.95), (0.1, 0.95), (0.2, 0.95)], maxlen=5),
        )
        # Ongoing uncertainty resets accumulated evidence without changing class or geometry state.
        current, _entered, cleared = app.reconcile_person_uncertainty({4}, {4}, {4: state})
        self.assertEqual(current, {4})
        self.assertEqual(cleared, set())
        self.assertFalse(state.announcement_eligible)
        self.assertEqual(len(state.strong_evidence), 0)

        # Clearing the condition resets once more; existing policy then requires three fresh hits.
        app.reconcile_person_uncertainty({4}, set(), {4: state})
        person = app.DetectionSnapshot(4, True, "person", 0.90, "ahead", (100, 100, 220, 360))
        for stamp in (1.0, 1.07):
            app.stabilize_detection(person, state, 640, observed_at=stamp)
            self.assertFalse(app.is_announcement_eligible(state, person.confidence))
        app.stabilize_detection(person, state, 640, observed_at=1.14)
        self.assertTrue(app.is_announcement_eligible(state, person.confidence))

    def test_class_stabilization_rejects_single_flip_and_accepts_sustained_change(self):
        state = app.VisionTrackState()
        detection = app.DetectionSnapshot(4, True, "book", 0.80, "ahead", (100, 100, 200, 240))
        current = app.stabilize_detection(detection, state, 640)
        self.assertEqual(current.label, "book")
        laptop = app.DetectionSnapshot(4, True, "laptop", 0.95, "ahead", (100, 100, 200, 240))
        self.assertEqual(app.stabilize_detection(laptop, state, 640).label, "book")
        for _ in range(3):
            current = app.stabilize_detection(laptop, state, 640)
        self.assertEqual(current.label, "laptop")

    def test_empty_class_history_uses_frame_label_without_mutating_state(self):
        state = app.VisionTrackState(stable_label="person", pending_label="chair", pending_label_count=2)
        detection = app.DetectionSnapshot(4, True, "chair", 0.10, "ahead", (100, 100, 200, 240))
        result = app.stabilize_detection(detection, state, 640, observed_at=5.0)
        self.assertEqual(result.label, "chair")
        self.assertEqual(state.stable_label, "person")
        self.assertEqual(state.pending_label, "chair")
        self.assertEqual(state.pending_label_count, 2)
        self.assertEqual(len(state.class_history), 0)

    def test_expired_class_history_does_not_break_weighted_vote(self):
        state = app.VisionTrackState(
            stable_label="person",
            class_history=deque([(1.0, "chair", 0.95)], maxlen=11),
        )
        detection = app.DetectionSnapshot(4, True, "person", 0.10, "ahead", (100, 100, 200, 240))
        result = app.stabilize_detection(detection, state, 640, observed_at=5.0)
        self.assertEqual(result.label, "person")
        self.assertEqual(len(state.class_history), 0)

    def test_invalid_or_filtered_class_entries_are_removed_safely(self):
        state = app.VisionTrackState(
            stable_label="person",
            class_history=deque([
                (4.9, "", 0.9),
                (4.9, "chair", float("nan")),
                (4.9, "remote", 0.10),
            ], maxlen=11),
        )
        detection = app.DetectionSnapshot(4, True, "person", 0.10, "ahead", (100, 100, 200, 240))
        result = app.stabilize_detection(detection, state, 640, observed_at=5.0)
        self.assertEqual(result.label, "person")
        self.assertEqual(len(state.class_history), 0)

    def test_valid_weighted_votes_keep_existing_class_stabilization(self):
        state = app.VisionTrackState(stable_label="person")
        state.class_history.extend([
            (9.8, "person", 0.8),
            (9.9, "chair", 0.95),
        ])
        detection = app.DetectionSnapshot(4, True, "chair", 0.95, "ahead", (100, 100, 200, 240))
        result = app.stabilize_detection(detection, state, 640, observed_at=10.0)
        self.assertEqual(result.label, "person")
        self.assertEqual(state.pending_label, "chair")
        self.assertEqual(state.pending_label_count, 1)

    def test_person_identity_state_survives_temporary_missed_frames(self):
        state = app.VisionTrackState()
        person = app.DetectionSnapshot(7, True, "person", 0.92, "ahead", (200, 100, 330, 390))
        initial = app.stabilize_detection(person, state, 640)
        # No observations arrive for two frames. The main loop retains this state while
        # ByteTrack keeps lost tracks for up to track_buffer frames.
        same_identity = app.stabilize_detection(person, state, 640)
        self.assertEqual(initial.track_id, same_identity.track_id)
        self.assertEqual(state.stable_label, "person")
        tracker_yaml = app.TRACKER_CONFIG.read_text(encoding="utf-8")
        self.assertIn("track_buffer: 45", tracker_yaml)

    def test_app_track_state_waits_for_byte_track_removal_before_purge(self):
        tracked_ids = {2, 4}
        last_seen = {2: 1.0, 4: 1.0}
        # Wall-clock age alone must not erase state while ByteTrack can still reactivate an ID.
        self.assertEqual(app.application_tracks_to_purge(
            tracked_ids, last_seen, 20.0, removed_tracker_ids=set(),
            tracker_lifecycle_available=True,
        ), set())
        self.assertEqual(app.application_tracks_to_purge(
            tracked_ids, last_seen, 20.0, removed_tracker_ids={2},
            tracker_lifecycle_available=True,
        ), {2})
        # Older tracker versions retain the bounded time-based cleanup fallback.
        self.assertEqual(app.application_tracks_to_purge(
            tracked_ids, last_seen, 7.0, tracker_lifecycle_available=False,
        ), tracked_ids)

    def test_byte_track_keeps_low_score_recovery_but_raises_new_track_threshold(self):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML

        tracker_yaml = app.TRACKER_CONFIG.read_text(encoding="utf-8")
        self.assertIn("track_low_thresh: 0.10", tracker_yaml)
        self.assertIn("track_high_thresh: 0.25", tracker_yaml)
        self.assertIn("new_track_thresh: 0.45", tracker_yaml)
        self.assertGreater(app.BYTE_TRACK_NEW_TRACK_CONFIDENCE, app.BYTE_TRACK_HIGH_CONFIDENCE)
        self.assertEqual(app.confidence_stage(0.18), "low_stage_recovery_candidate")

        tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(str(app.TRACKER_CONFIG))))
        high = Boxes(app.torch.tensor([[100, 100, 160, 160, 0.80, 65]], dtype=app.torch.float32), (480, 640))
        first = tracker.update(high)
        self.assertEqual(len(first), 1)
        track_id = int(first[0, 4])
        low = Boxes(app.torch.tensor([[102, 100, 162, 160, 0.18, 65]], dtype=app.torch.float32), (480, 640))
        recovered = tracker.update(low)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(int(recovered[0, 4]), track_id)
        missed = Boxes(app.torch.empty((0, 6), dtype=app.torch.float32), (480, 640))
        self.assertEqual(len(tracker.update(missed)), 0)
        reacquired = tracker.update(high)
        self.assertEqual(len(reacquired), 1)
        self.assertEqual(int(reacquired[0, 4]), track_id)

        new_tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(str(app.TRACKER_CONFIG))))
        medium = Boxes(app.torch.tensor([[100, 100, 160, 160, 0.35, 65]], dtype=app.torch.float32), (480, 640))
        self.assertEqual(len(new_tracker.update(medium)), 0)
        stronger = Boxes(app.torch.tensor([[100, 100, 160, 160, 0.50, 65]], dtype=app.torch.float32), (480, 640))
        self.assertEqual(len(new_tracker.update(stronger)), 0)
        self.assertEqual(len(new_tracker.update(stronger)), 1)

    def test_vision_candidate_diagnostic_identifies_confidence_and_agnostic_nms_rejections(self):
        book = {"label": "book", "confidence": 0.08, "bbox": (100, 100, 200, 220)}
        self.assertEqual(
            app.candidate_pretracker_status(book, [], 0.10, 0.70),
            "rejected_below_detector_confidence_threshold",
        )

        book["confidence"] = 0.62
        person = {"label": "person", "confidence": 0.91, "bbox": (101, 101, 201, 221)}
        self.assertEqual(
            app.candidate_pretracker_status(book, [person], 0.10, 0.70),
            "likely_agnostic_nms_suppression_by_person",
        )

        surviving_book = {"label": "book", "confidence": 0.60, "bbox": (100, 100, 200, 220)}
        self.assertEqual(
            app.candidate_pretracker_status(book, [surviving_book], 0.10, 0.70),
            "passed_confidence_and_nms",
        )

    def test_cross_class_candidate_geometry_is_not_reported_as_track_assignment(self):
        candidate = {
            "label": "refrigerator", "confidence": 0.13,
            "bbox": (40, 2, 1270, 715),
        }
        person_track = {
            "label": "person", "confidence": 0.91,
            "bbox": (300, 210, 1040, 715), "track_id": 1,
        }
        class_match, geometry_only = app.diagnostic_candidate_tracker_match(
            candidate, [person_track],
        )
        self.assertIsNone(class_match)
        self.assertIs(person_track, geometry_only)

        # Only the tracker output is passed to application stabilization. A broad
        # refrigerator candidate that overlaps the person cannot enter this history.
        state = app.VisionTrackState(stable_label="person")
        tracked_person = app.DetectionSnapshot(
            1, True, "person", 0.91, "ahead", (300, 210, 1040, 715),
        )
        stabilized = app.stabilize_detection(tracked_person, state, 1280, observed_at=1.0)
        self.assertEqual(stabilized.label, "person")
        self.assertEqual([label for _stamp, label, _conf in state.class_history], ["person"])

    def test_partial_person_detection_can_create_parallel_track_then_reactivate(self):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML

        tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(str(app.TRACKER_CONFIG))))

        def update(rows):
            tensor = app.torch.tensor(rows, dtype=app.torch.float32) if rows else app.torch.empty((0, 6))
            return tracker.update(Boxes(tensor, (720, 1280)))

        full = [480, 196, 1036, 714, 0.93, 0]
        partial = [507, 485, 1012, 714, 0.56, 0]
        first = update([full])
        parent_id = int(first[0, 4])
        update([full, partial])  # The second high-score region is initially unconfirmed.
        simultaneous = update([full, partial])
        ids = {int(row[4]) for row in simultaneous}
        self.assertEqual(len(ids), 2)
        partial_track_id = next(track_id for track_id in ids if track_id != parent_id)

        update([full])
        self.assertIn(partial_track_id, {track.track_id for track in tracker.lost_stracks})
        reacquired = update([full, partial])
        self.assertEqual({int(row[4]) for row in reacquired}, ids)

    def test_person_depth_sampling_masks_book_pixels_in_torso_roi(self):
        depth = np.full((40, 30), 1.8, dtype=np.float32)
        person_box = (4, 2, 26, 38)
        book_box = (10, 8, 20, 25)
        depth[8:25, 10:20] = 0.9
        sample = app.median_object_depth(depth, person_box, 20.0, "person", (book_box,))
        self.assertAlmostEqual(sample.median_m, 1.8, places=6)
        self.assertGreater(sample.overlap_masked_pixels, 0)

    def test_weak_observations_do_not_make_a_new_track_announceable(self):
        state = app.VisionTrackState()
        weak = app.DetectionSnapshot(19, True, "remote", 0.46, "ahead", (10, 10, 40, 40))
        for stamp in (0.0, 0.1):
            app.stabilize_detection(weak, state, 640, observed_at=stamp)
            self.assertFalse(app.is_announcement_eligible(state, weak.confidence))
        app.stabilize_detection(weak, state, 640, observed_at=0.2)
        self.assertTrue(app.is_announcement_eligible(state, weak.confidence))
        self.assertFalse(app.is_announcement_eligible(state, 0.20))

    def test_low_confidence_recovery_preserves_class_history_and_identity(self):
        state = app.VisionTrackState()
        strong = app.DetectionSnapshot(21, True, "person", 0.90, "ahead", (100, 100, 220, 360))
        app.stabilize_detection(strong, state, 640, observed_at=1.0)
        history_before = tuple(state.class_history)
        recovered = app.DetectionSnapshot(21, True, "remote", 0.18, "ahead", (101, 100, 221, 360))
        result = app.stabilize_detection(recovered, state, 640, observed_at=1.2)
        self.assertEqual(result.track_id, strong.track_id)
        self.assertEqual(result.label, "person")
        self.assertEqual(tuple(state.class_history), history_before)
        self.assertFalse(app.is_announcement_eligible(state, recovered.confidence))

    def test_repeated_one_frame_false_tracks_never_become_announceable(self):
        for track_id in range(30, 34):
            state = app.VisionTrackState()
            one_frame = app.DetectionSnapshot(track_id, True, "cat", 0.48, "ahead", (20, 20, 60, 60))
            app.stabilize_detection(one_frame, state, 640, observed_at=float(track_id))
            self.assertFalse(app.is_announcement_eligible(state, one_frame.confidence))

    def test_fragmentation_diagnostic_finds_recent_same_class_overlap(self):
        old_state = app.VisionTrackState(
            smoothed_bbox=(100.0, 100.0, 160.0, 160.0), stable_label="remote"
        )
        new = app.DetectionSnapshot(56, True, "remote", 0.50, "ahead", (102, 101, 162, 161))
        candidate = app.possible_fragmented_track(new, 2.0, {21: old_state}, {21: 1.8})
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate[0], 21)

    def test_distance_change_requires_a_confirming_sample(self):
        history = deque([(0.0, 1.8), (0.3, 1.9)], maxlen=3)
        accepted, pending, reason = app.confirm_depth_change(history, 0.6, 0.6, None)
        self.assertFalse(accepted)
        self.assertEqual(reason, "depth_change_pending_confirmation")
        accepted, next_pending, reason = app.confirm_depth_change(history, 0.62, 0.9, pending)
        self.assertTrue(accepted)
        self.assertIsNone(next_pending)
        self.assertEqual(reason, "sustained_change_confirmed")

    def test_close_obstacle_needs_two_fresh_depth_samples(self):
        history = deque([(1.0, 1.0)], maxlen=3)
        self.assertFalse(app.close_measurement_confirmed(history, 1.2))
        history.append((1.3, 1.1))
        self.assertTrue(app.close_measurement_confirmed(history, 1.4))
        inconsistent = deque([(1.0, 0.6), (1.3, 1.0)], maxlen=3)
        self.assertFalse(app.close_measurement_confirmed(inconsistent, 1.4))

    def test_bbox_smoothing_reduces_jitter_but_new_track_initializes_immediately(self):
        state = app.VisionTrackState()
        first = app.DetectionSnapshot(2, True, "chair", 0.8, "ahead", (100, 100, 200, 240))
        initial = app.stabilize_detection(first, state, 640)
        self.assertEqual(initial.bbox, first.bbox)
        jitter = app.DetectionSnapshot(2, True, "chair", 0.8, "ahead", (110, 100, 210, 240))
        smoothed = app.stabilize_detection(jitter, state, 640)
        self.assertGreater(smoothed.bbox[0], first.bbox[0])
        self.assertLess(smoothed.bbox[0], jitter.bbox[0])

    def test_direction_change_requires_sustained_stabilized_box_evidence(self):
        state = app.VisionTrackState()
        ahead = app.DetectionSnapshot(8, True, "person", 0.9, "ahead", (450, 100, 550, 300))
        self.assertEqual(app.stabilize_detection(ahead, state, 1000).direction, "ahead")
        right = app.DetectionSnapshot(8, True, "person", 0.9, "ahead", (650, 100, 750, 300))
        self.assertEqual(app.stabilize_detection(right, state, 1000).direction, "ahead")
        self.assertEqual(app.stabilize_detection(right, state, 1000).direction, "ahead")
        self.assertEqual(app.stabilize_detection(right, state, 1000).direction, "right")

    def test_unchanged_object_state_does_not_repeat_speech(self):
        previous = app.AnnouncementState("ahead", 10.0, "stationary", 2.0)
        should, _priority, reason = app.announcement_decision(
            previous, "ahead", 2.0, "stationary", 10.5
        )
        self.assertFalse(should)
        self.assertEqual(reason, "unchanged_or_cooldown")

    def test_debug_log_sessions_truncate_same_file_and_capture_both_streams(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "logs" / "sightbridge_debug.txt"
            path.parent.mkdir(parents=True)
            path.write_text("old-session-content\n", encoding="utf-8")
            console_out, console_err = io.StringIO(), io.StringIO()
            with patch.object(sys, "stdout", console_out), patch.object(sys, "stderr", console_err):
                original_stdout, original_stderr = sys.stdout, sys.stderr
                first = app.start_debug_log_if_enabled(True, path)
                self.assertIsNotNone(first)
                print("first-run-startup")
                print("first-run-warning", file=sys.stderr)
                first.close()
                first.close()  # cleanup is safe to call repeatedly
                first_contents = path.read_text(encoding="utf-8")
                self.assertIn("first-run-startup", first_contents)
                self.assertNotIn("old-session-content", first_contents)

                second = app.start_debug_log_if_enabled(True, path)
                print("second-run-shutdown")
                second.close()
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
            final_contents = path.read_text(encoding="utf-8")
            self.assertNotIn("first-run-startup", final_contents)
            self.assertNotIn("first-run-warning", final_contents)
            self.assertIn("second-run-shutdown", final_contents)
            self.assertEqual(final_contents.count("SightBridge session started:"), 1)
            self.assertIn("second-run-shutdown", console_out.getvalue())
            self.assertTrue(second._file.closed)

    def test_debug_logging_disabled_leaves_existing_file_untouched(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "logs" / "sightbridge_debug.txt"
            path.parent.mkdir(parents=True)
            path.write_text("keep-this-session\n", encoding="utf-8")
            self.assertIsNone(app.start_debug_log_if_enabled(False, path))
            self.assertEqual(path.read_text(encoding="utf-8"), "keep-this-session\n")

            missing_path = Path(temporary_directory) / "missing" / "sightbridge_debug.txt"
            self.assertIsNone(app.start_debug_log_if_enabled(False, missing_path))
            self.assertFalse(missing_path.exists())

    def test_debug_log_captures_complete_unhandled_traceback_and_closes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sightbridge_debug.txt"
            console_out, console_err = io.StringIO(), io.StringIO()
            with patch.object(sys, "stdout", console_out), patch.object(sys, "stderr", console_err):
                session = app.DebugLogSession(path)
                original_stdout, original_stderr = sys.stdout, sys.stderr
                session.start()
                try:
                    raise RuntimeError("debug traceback regression")
                except RuntimeError:
                    sys.excepthook(*sys.exc_info())
                session.flush()
                log_contents = path.read_text(encoding="utf-8")
                self.assertIn("Traceback (most recent call last)", log_contents)
                self.assertIn("RuntimeError: debug traceback regression", log_contents)
                self.assertIn("RuntimeError: debug traceback regression", console_err.getvalue())
                session.close()
                self.assertTrue(session._file.closed)
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)

    def test_depth_sampler_rejects_invalid_pixels_and_small_boxes(self):
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        depth[8, 8] = np.nan
        depth[8, 9] = np.inf
        depth[9, 8] = 0.0
        sample = app.median_object_depth(depth, (4, 4, 16, 16), 20.0)
        self.assertIsNotNone(sample.median_m)
        self.assertGreaterEqual(sample.valid_pixels, 4)
        out_of_range = app.median_object_depth(np.full((20, 20), 21.0, dtype=np.float32), (4, 4, 16, 16), 20.0)
        self.assertIsNone(out_of_range.median_m)
        self.assertEqual(out_of_range.valid_pixels, 0)
        small = app.median_object_depth(depth, (5, 5, 7, 7), 20.0)
        self.assertIsNone(small.median_m)
        self.assertEqual(small.rejection_reason, "fewer_than_4_valid_roi_pixels")
        self.assertEqual(app.depth_diagnostic_reason(
            small, depth_result_present=True, source_frame_match=True,
            result_fresh=True,
        ), "ROI_TOO_SMALL")

    def test_enclosing_person_detection_does_not_mask_phone_roi(self):
        depth = np.full((720, 1280), 1.7, dtype=np.float32)
        phone = (835, 306, 943, 493)
        enclosing_person = (532, 205, 1058, 717)
        sample = app.median_object_depth(
            depth, phone, 20.0, "cell phone", (enclosing_person,)
        )
        self.assertAlmostEqual(sample.median_m, 1.7, places=6)
        self.assertGreaterEqual(sample.valid_pixels, 4)
        self.assertEqual(sample.overlap_masked_pixels, 0)
        self.assertEqual((sample.roi_width, sample.roi_height), (54, 93))

    def test_all_invalid_and_out_of_range_pixels_have_distinct_reasons(self):
        box = (2, 2, 18, 18)
        invalid_depth = np.full((20, 20), np.nan, dtype=np.float32)
        invalid = app.median_object_depth(invalid_depth, box, 20.0)
        self.assertEqual(app.depth_diagnostic_reason(
            invalid, depth_result_present=True, source_frame_match=True,
            result_fresh=True,
        ), "INVALID_DEPTH_VALUES")
        out_of_range = app.median_object_depth(np.full((20, 20), 25.0, dtype=np.float32), box, 20.0)
        self.assertEqual(app.depth_diagnostic_reason(
            out_of_range, depth_result_present=True, source_frame_match=True,
            result_fresh=True,
        ), "OUT_OF_RANGE")

    def test_overlapping_masks_report_when_they_remove_the_sampling_roi(self):
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        box = (2, 2, 18, 18)  # inner ROI is [6:14, 6:14]
        split_masks = ((6, 6, 10, 14), (10, 6, 14, 14))
        sample = app.median_object_depth(depth, box, 20.0, other_boxes=split_masks)
        self.assertEqual(sample.valid_pixels, 0)
        self.assertEqual(sample.overlap_masked_pixels, sample.sampled_pixels)
        self.assertEqual(app.depth_diagnostic_reason(
            sample, depth_result_present=True, source_frame_match=True,
            result_fresh=True,
        ), "OVERLAPPING_DETECTION_MASK")

    def test_stale_and_source_frame_mismatch_have_exact_diagnostic_reasons(self):
        sample = app.median_object_depth(np.full((20, 20), 2.0, dtype=np.float32), (2, 2, 18, 18), 20.0)
        self.assertEqual(app.depth_diagnostic_reason(
            sample, depth_result_present=True, source_frame_match=True,
            result_fresh=False,
        ), "STALE_DEPTH_RESULT")
        self.assertFalse(app.depth_source_frames_match(20, 21))
        self.assertEqual(app.depth_diagnostic_reason(
            None, depth_result_present=True, source_frame_match=False,
            result_fresh=True,
        ), "SOURCE_FRAME_MISMATCH")
        self.assertEqual(app.depth_diagnostic_reason(
            None, depth_result_present=False, source_frame_match=None,
            result_fresh=False,
        ), "NO_DEPTH_RESULT")

    def test_person_sampling_dimensions_and_distance_remain_unchanged(self):
        depth = np.full((40, 30), 2.0, dtype=np.float32)
        depth[8:16, 8:22] = 9.0
        sample = app.median_object_depth(depth, (4, 4, 26, 36), 20.0, "person")
        self.assertEqual(sample.median_m, 2.0)
        self.assertEqual((sample.roi_width, sample.roi_height), (10, 15))

    def test_depth_diagnostic_formatting_does_not_change_sample(self):
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        detection = app.DetectionSnapshot(3, True, "person", 0.9, "ahead", (2, 2, 18, 18), 45)
        sample = app.median_object_depth(depth, detection.bbox, 20.0, "person")
        before = sample
        text = app.format_depth_sampling_diagnostic(
            detection, sample, depth_frame_id=45, depth_age_seconds=0.1,
            source_frame_match=app.depth_source_frames_match(detection.source_frame_id, 45),
            reason="DISTANCE_AVAILABLE",
        )
        self.assertIn("same_source_frame=yes", text)
        self.assertIn("valid=", text)
        self.assertEqual(sample, before)
        self.assertEqual(sample.median_m, 2.0)

    def test_depth_roi_clips_at_image_edges_and_masks_overlaps(self):
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        depth[7:13, 7:13] = 9.0
        clipped = app.median_object_depth(depth, (-5, -5, 12, 12), 20.0)
        self.assertEqual(clipped.median_m, 2.0)
        overlap = app.median_object_depth(
            depth, (2, 2, 18, 18), 20.0, other_boxes=((7, 7, 13, 13),)
        )
        self.assertEqual(overlap.median_m, 2.0)
        self.assertLess(overlap.valid_pixels, clipped.valid_pixels)

    def test_person_sampler_uses_torso_region(self):
        depth = np.full((40, 30), 2.0, dtype=np.float32)
        depth[8:16, 8:22] = 9.0  # Head/upper edge should not dominate torso sample.
        sample = app.median_object_depth(depth, (4, 4, 26, 36), 20.0, "person")
        self.assertEqual(sample.median_m, 2.0)

    def test_smoothing_rejects_single_spike_and_recovers_to_sustained_change(self):
        history = deque([(0.0, 2.0), (0.3, 2.1)], maxlen=3)
        previous = app.DistanceState(2.05, 0.3)
        filtered, reason = app.smooth_depth_measurement(history, 8.0, 0.6, previous, 20.0)
        self.assertLess(filtered, 3.0)
        self.assertEqual(reason, "isolated_spike_median_filtered")
        filtered, _reason = app.smooth_depth_measurement(history, 1.0, 0.9, app.DistanceState(filtered, 0.6), 20.0)
        filtered, _reason = app.smooth_depth_measurement(history, 1.0, 1.2, app.DistanceState(filtered, 0.9), 20.0)
        self.assertLess(filtered, 2.0)

    def test_motion_estimator_requires_consistent_timed_evidence(self):
        noisy = deque([(0.0, 3.0), (0.3, 3.2), (1.4, 2.9), (1.7, 3.1), (2.0, 3.0), (2.3, 3.2)])
        self.assertEqual(app.motion_from_history(noisy), "stationary")
        approach = deque([(0.0, 3.2), (0.3, 3.1), (1.4, 2.0), (1.7, 1.9), (2.0, 1.8), (2.3, 1.7)])
        self.assertEqual(app.motion_from_history(approach), "approaching")
        receding = deque([(0.0, 1.8), (0.3, 1.9), (1.4, 3.0), (1.7, 3.1), (2.0, 3.2), (2.3, 3.3)])
        self.assertEqual(app.motion_from_history(receding), "receding")
        self.assertEqual(app.motion_from_history(deque([(0.0, 3.0), (0.3, 2.8)])), "unknown")

    def test_stale_distance_is_unavailable(self):
        app.assert_distance_output_cases()
        state = app.DistanceState(2.0, 1.0)
        self.assertIsNone(app.fresh_track_distance(state, 2.0, 0.6, 20.0))

    def test_direction_zones_and_hysteresis(self):
        self.assertEqual(app.direction_for_box(100, 100, 1000), "far left")
        self.assertEqual(app.direction_for_box(200, 200, 1000), "left")
        self.assertEqual(app.direction_for_box(205, 205, 1000, previous="far left"), "far left")
        self.assertEqual(app.direction_for_box(245, 245, 1000, previous="far left"), "left")
        self.assertEqual(app.direction_for_box(500, 500, 1000), "ahead")
        self.assertEqual(app.direction_for_box(700, 700, 1000), "right")
        self.assertEqual(app.direction_for_box(900, 900, 1000), "far right")


if __name__ == "__main__":
    unittest.main()
