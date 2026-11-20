"""SightBridge Spatial Awareness Demonstrator.

Research/demo software only. It is not a safe mobility aid.
"""
from __future__ import annotations

import argparse
import atexit
import base64
import csv
import json
import math
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

YOLO_MODEL = "yolo11n.pt"
TRACKER_CONFIG = Path(__file__).with_name("bytetrack-sightbridge.yaml")
DEPTH_MODELS = {
    "indoor": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", 20.0),
    "outdoor": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", 80.0),
}
# FP16 depth results arrive about every 0.33s; allow normal scheduling variance,
# while still expiring measurements if inference stalls for roughly 0.6s.
MAX_DEPTH_RESULT_AGE_SECONDS = 0.60
# Speech/vision thresholds are named here so they can be adjusted after camera testing.
SPEECH_COOLDOWN_SECONDS = 3.0
ORDINARY_ANNOUNCEMENT_GLOBAL_COOLDOWN_SECONDS = 3.0
SPEECH_SHUTDOWN_CLOSE_GRACE_SECONDS = 2.5
SPEECH_SHUTDOWN_MAX_WAIT_SECONDS = 30.0
CLOSE_OBSTACLE_DISTANCE_M = 1.2
DISTANCE_CHANGE_MIN_M = 0.5
DISTANCE_CHANGE_RATIO = 0.20
DIRECTION_HYSTERESIS = 0.04
TRACK_STATE_RESET_SECONDS = MAX_DEPTH_RESULT_AGE_SECONDS
TRACK_STATE_PURGE_SECONDS = 5.0
TRACK_REACQUIRE_RESET_SECONDS = 2.25
DEPTH_EMA_ALPHA = 0.45
DEPTH_SPIKE_MIN_M = 1.0
DEPTH_SPIKE_RATIO = 0.45
APPROACH_MIN_CHANGE_M = 0.75
APPROACH_CHANGE_RATIO = 0.10
BYTE_TRACK_LOW_CONFIDENCE = 0.10
BYTE_TRACK_HIGH_CONFIDENCE = 0.25
BYTE_TRACK_NEW_TRACK_CONFIDENCE = 0.45
CLASS_HISTORY_MIN_CONFIDENCE = 0.25
ANNOUNCEMENT_MIN_CONFIDENCE = 0.45
ANNOUNCEMENT_STRONG_HITS = 3
ANNOUNCEMENT_EVIDENCE_WINDOW_SECONDS = 0.60
ANNOUNCEMENT_EVIDENCE_MIN_SPAN_SECONDS = 0.12
DEPTH_CHANGE_CONFIRM_WINDOW_SECONDS = 1.0


class _TeeBinaryStream:
    def __init__(self, text_stream: "_TeeTextStream") -> None:
        self._text_stream = text_stream

    def write(self, value: bytes) -> int:
        text = value.decode(self._text_stream.encoding, errors="replace")
        self._text_stream.write(text)
        return len(value)

    def flush(self) -> None:
        self._text_stream.flush()


class _TeeTextStream:
    def __init__(self, console: Any, log_file: Any, lock: threading.RLock) -> None:
        self._console = console
        self._log_file = log_file
        self._lock = lock
        self.encoding = getattr(console, "encoding", None) or "utf-8"
        self.errors = getattr(console, "errors", "replace")
        self.buffer = _TeeBinaryStream(self)

    def write(self, text: str) -> int:
        with self._lock:
            written = self._console.write(text)
            self._log_file.write(text)
            if "\n" in text:
                self._console.flush()
                self._log_file.flush()
            return len(text) if written is None else written

    def flush(self) -> None:
        with self._lock:
            self._console.flush()
            self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._console.fileno()

    def writable(self) -> bool:
        return True


class DebugLogSession:
    """Capture one application's stdout/stderr to a fresh log and the terminal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: Any = None
        self._stdout: Any = None
        self._stderr: Any = None
        self._excepthook: Any = None
        self._lock = threading.RLock()
        self._stdout_tee: Optional[_TeeTextStream] = None
        self._stderr_tee: Optional[_TeeTextStream] = None
        self._closed = False

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", errors="replace", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._excepthook = sys.excepthook
        self._stdout_tee = _TeeTextStream(self._stdout, self._file, self._lock)
        self._stderr_tee = _TeeTextStream(self._stderr, self._file, self._lock)
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

        def log_unhandled_exception(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
            try:
                self._excepthook(exc_type, exc_value, exc_traceback)
            finally:
                self.flush()

        sys.excepthook = log_unhandled_exception
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        self._file.write(f"\n{'=' * 24} SightBridge session started: {timestamp} {'=' * 24}\n")
        self._file.flush()
        atexit.register(self.close)

    def flush(self) -> None:
        if self._file is not None and not self._file.closed:
            with self._lock:
                self._file.flush()
                try:
                    os.fsync(self._file.fileno())
                except OSError:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
        if sys.stdout is self._stdout_tee:
            sys.stdout = self._stdout
        if sys.stderr is self._stderr_tee:
            sys.stderr = self._stderr
        if self._excepthook is not None and sys.excepthook is not self._excepthook:
            sys.excepthook = self._excepthook
        if self._file is not None and not self._file.closed:
            self._file.close()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass


def start_debug_log_if_enabled(enabled: bool, path: Path) -> Optional[DebugLogSession]:
    """Start per-run logging only when explicitly requested."""
    if not enabled:
        return None
    session = DebugLogSession(path)
    session.start()
    return session


@dataclass(frozen=True)
class SpeechJob:
    text: str
    expires_at: Optional[float]
    priority: int
    key: Optional[str]
    sequence: int
    validation_token: Optional["SpeechValidationToken"] = None


@dataclass(frozen=True)
class SpeechValidationToken:
    track_id: int
    label: str
    direction: str
    distance_m: Optional[float]
    motion: str


@dataclass(frozen=True)
class SpeechLiveState:
    label: str
    direction: str
    seen_at: float
    distance_m: Optional[float]
    distance_captured_at: Optional[float]
    motion: str
    identity_uncertain: bool = False


def validate_pending_announcement(
    token: SpeechValidationToken,
    current: Optional[SpeechLiveState],
    now: float,
    max_track_age: float,
    max_depth_age: float,
    max_depth_m: float,
) -> Optional[str]:
    """Reject queued object speech when the underlying live observation changed."""
    if current is None:
        return "track_disappeared"
    if now < current.seen_at or now - current.seen_at > max_track_age:
        return "track_observation_stale"
    if current.identity_uncertain:
        return "track_identity_uncertain"
    if current.label != token.label or current.direction != token.direction:
        return "track_description_changed"

    distance_is_fresh = (
        current.distance_m is not None
        and current.distance_captured_at is not None
        and 0.0 <= now - current.distance_captured_at <= max_depth_age
        and is_valid_distance(current.distance_m, max_depth_m)
    )
    if token.distance_m is None:
        if distance_is_fresh:
            return "distance_became_available"
    else:
        if not distance_is_fresh:
            return "distance_expired"
        if readable_distance(token.distance_m, max_depth_m) != readable_distance(current.distance_m, max_depth_m):
            return "distance_changed"
    if current.motion != token.motion:
        return "motion_description_changed"
    return None


POWERSHELL_SPEECH_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
function Write-Protocol($value) {
    [Console]::Out.WriteLine((ConvertTo-Json -InputObject $value -Compress))
    [Console]::Out.Flush()
}
$synth = $null
try {
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SetOutputToDefaultAudioDevice()
    $synth.Volume = 100
    $voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled })
    if ($voices.Count -eq 0) { throw 'No enabled System.Speech voices are installed.' }
    Write-Protocol @{ kind = 'ready'; ok = $true; voices = $voices.Count }
} catch {
    Write-Protocol @{ kind = 'ready'; ok = $false; error = $_.Exception.Message }
    exit 2
}
try {
    while ($null -ne ($line = [Console]::In.ReadLine())) {
        try {
            $request = ConvertFrom-Json -InputObject $line
            if ($request.op -eq 'shutdown') {
                Write-Protocol @{ id = $request.id; status = 'stopped' }
                break
            }
            if ($request.op -ne 'speak') { throw "Unknown speech operation: $($request.op)" }
            Write-Protocol @{ id = $request.id; status = 'accepted' }
            try {
                $synth.Speak([string]$request.text)
                Write-Protocol @{ id = $request.id; status = 'completed' }
            } catch {
                Write-Protocol @{ id = $request.id; status = 'failed'; error = $_.Exception.Message }
            }
        } catch {
            Write-Protocol @{ id = ''; status = 'failed'; error = $_.Exception.Message }
        }
    }
} finally {
    if ($null -ne $synth) { $synth.Dispose() }
}
"""


class PowerShellSpeechBackend:
    """Persistent Windows PowerShell process with one System.Speech synthesizer."""

    def __init__(self, startup_timeout: float = 8.0) -> None:
        if os.name != "nt":
            raise OSError("The System.Speech backend is available only on Windows.")
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if executable is None:
            raise FileNotFoundError("Windows PowerShell (powershell.exe) was not found.")
        encoded_script = base64.b64encode(POWERSHELL_SPEECH_SCRIPT.encode("utf-16le")).decode("ascii")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-EncodedCommand", encoded_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self.responses: queue.Queue[Optional[str]] = queue.Queue()
        self.stderr_tail: Deque[str] = deque(maxlen=12)
        self._write_lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self._stdout_thread = threading.Thread(target=self._read_stdout, name="SightBridgeSpeechPipe", daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="SightBridgeSpeechErrors", daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready = self._read_response(startup_timeout)
            if ready.get("kind") != "ready" or not ready.get("ok"):
                error = ready.get("error", f"Unexpected startup response: {ready}")
                raise RuntimeError(f"System.Speech initialization failed: {error}")
        except Exception:
            self.terminate()
            raise
        self.voice_count = int(ready.get("voices", 0))

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.responses.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            self.responses.put(None)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            cleaned = line.rstrip()
            if cleaned:
                self.stderr_tail.append(cleaned)

    def _read_response(self, timeout: float) -> dict[str, Any]:
        try:
            line = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"No response from speech PowerShell process within {timeout:.1f}s.") from exc
        if line is None:
            detail = " | ".join(self.stderr_tail)
            raise RuntimeError(f"Speech PowerShell process exited unexpectedly. {detail}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid response from speech PowerShell process: {line!r}") from exc

    def speak(
        self,
        text: str,
        timeout: float,
        on_accepted: Optional[Callable[[], None]] = None,
    ) -> dict[str, Any]:
        if self._closed or self.process.poll() is not None:
            raise RuntimeError("Speech PowerShell process is not running.")
        if self.process.stdin is None:
            raise RuntimeError("Speech PowerShell input pipe is unavailable.")
        with self._write_lock:
            self._request_id += 1
            request_id = str(self._request_id)
            payload = json.dumps(
                {"id": request_id, "op": "speak", "text": text}, ensure_ascii=True
            )
            try:
                self.process.stdin.write(payload + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"Could not send speech request: {exc}") from exc

            deadline = time.monotonic() + timeout
            accepted = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.terminate()
                    raise TimeoutError(f"Speech request {request_id} timed out after {timeout:.1f}s.")
                response = self._read_response(remaining)
                if response.get("id") != request_id:
                    raise RuntimeError(f"Speech protocol response ID mismatch: {response}")
                status = response.get("status")
                if status == "accepted" and not accepted:
                    accepted = True
                    if on_accepted is not None:
                        on_accepted()
                    continue
                if status == "completed" and accepted:
                    return response
                if status == "failed":
                    raise RuntimeError(response.get("error", "System.Speech reported playback failure."))
                raise RuntimeError(f"Unexpected speech protocol response: {response}")

    def close(self, timeout: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                with self._write_lock:
                    self._request_id += 1
                    request_id = str(self._request_id)
                    payload = json.dumps({"id": request_id, "op": "shutdown"})
                    self.process.stdin.write(payload + "\n")
                    self.process.stdin.flush()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=1.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.terminate()
        self._close_pipes()

    def terminate(self) -> None:
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        self._close_pipes()

    def _close_pipes(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def run_speech_diagnostic(
    backend_factory: Optional[Callable[[], Any]] = None,
    request_timeout: float = 20.0,
) -> int:
    """Exercise repeated Windows speech without initializing camera or vision models."""
    factory = backend_factory or PowerShellSpeechBackend
    sentences = [
        "SightBridge speech test starting.",
        "This is repeated speech test one.",
        "This is repeated speech test two.",
        "This is repeated speech test three.",
        "This is repeated speech test four.",
        "This is repeated speech test five.",
    ]
    backend = None
    failures = 0
    try:
        print("[SpeechTest] Initializing the Windows System.Speech backend.")
        backend = factory()
        print(f"[SpeechTest] Backend ready; enabled voices={getattr(backend, 'voice_count', 'system default')}.")
        for index, sentence in enumerate(sentences, start=1):
            print(f"[SpeechTest] Sent {index}/{len(sentences)}: {sentence}")

            def accepted() -> None:
                print(f"[SpeechTest] Backend accepted {index}/{len(sentences)}.")

            try:
                response = backend.speak(sentence, request_timeout, accepted)
                if response.get("status") != "completed":
                    raise RuntimeError(f"Backend did not confirm completion: {response}")
                print(f"[SpeechTest] Synthesis completed {index}/{len(sentences)} (backend confirmation).")
            except Exception as exc:
                failures += 1
                print(f"[SpeechTest] Playback failure or timeout {index}/{len(sentences)}: {exc}")
        print(f"[SpeechTest] Finished: {len(sentences) - failures}/{len(sentences)} requests completed.")
    except Exception as exc:
        failures += 1
        print(f"[SpeechTest] Backend initialization failed: {exc}")
    finally:
        if backend is not None:
            try:
                backend.close(timeout=2.0)
            except Exception as exc:
                failures += 1
                print(f"[SpeechTest] Backend shutdown failed: {exc}")
    return 0 if failures == 0 else 1


class SpeechWorker:
    """Bounded announcement queue; one thread owns the persistent speech backend."""

    def __init__(
        self,
        enabled: bool = True,
        debug: bool = False,
        backend_factory: Optional[Callable[[], Any]] = None,
        max_pending: int = 4,
        init_timeout: float = 8.0,
        request_timeout: float = 15.0,
    ) -> None:
        self._enabled = enabled
        self.debug = debug
        self.max_pending = max_pending
        self.request_timeout = request_timeout
        self._backend_factory = backend_factory or PowerShellSpeechBackend
        self._job_validator: Optional[Callable[[SpeechJob], Optional[str]]] = None
        self._condition = threading.Condition()
        self._stop_lock = threading.Lock()
        self._jobs: list[SpeechJob] = []
        self._dropped_keys: set[str] = set()
        self._sequence = 0
        self._stopping = False
        self._inflight = False
        self._backend_ready = False
        self._status = "starting"
        self.failure_generation = 0
        self.completed_count = 0
        self._ready_event = threading.Event()
        self._last_rejection_log: Dict[str, float] = {}
        self._backend: Optional[Any] = None
        self.thread = threading.Thread(target=self._run, name="SightBridgeSpeech", daemon=True)
        self.thread.start()
        if not self._ready_event.wait(init_timeout):
            print("[Audio] Speech backend initialization is still pending.")
        print("[Audio] Speech worker started.")

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    @property
    def ready(self) -> bool:
        with self._condition:
            return self._backend_ready

    @property
    def status(self) -> str:
        with self._condition:
            return self._status

    def _log_rejection(self, reason: str, text: str) -> None:
        now = time.monotonic()
        if self.debug and now - self._last_rejection_log.get(reason, 0.0) >= 2.0:
            print(f"[Audio] Announcement rejected ({reason}): {text}")
            self._last_rejection_log[reason] = now

    def set_enabled(self, enabled: bool) -> None:
        with self._condition:
            changed = self._enabled != enabled
            self._enabled = enabled
            if not enabled:
                self._dropped_keys.update(
                    job.key for job in self._jobs if job.key is not None
                )
                self._jobs.clear()
            self._condition.notify_all()
        if changed:
            print(f"[Audio] Voice {'enabled' if enabled else 'disabled'}.")

    def set_job_validator(self, validator: Callable[[SpeechJob], Optional[str]]) -> None:
        """Install an application-state check run immediately before backend playback."""
        with self._condition:
            self._job_validator = validator

    def say(
        self,
        text: str,
        force: bool = False,
        expires_at: Optional[float] = None,
        priority: int = 0,
        key: Optional[str] = None,
        validation_token: Optional[SpeechValidationToken] = None,
    ) -> bool:
        del force  # Cooldowns are tracked per object in main().
        with self._condition:
            if not self._enabled:
                self._log_rejection("voice disabled", text)
                return False
            if not self._backend_ready:
                self._log_rejection("speech backend unavailable", text)
                return False
            if self._stopping or not self.thread.is_alive():
                self._log_rejection("worker stopped", text)
                return False
            if expires_at is not None and time.monotonic() > expires_at:
                self._log_rejection("announcement expired", text)
                return False

            if key is not None:
                replaced_jobs = [job for job in self._jobs if job.key == key]
                protected_close_job = next((
                    job for job in replaced_jobs
                    if job.priority >= 3
                    and (job.expires_at is None or time.monotonic() <= job.expires_at)
                    and priority < job.priority
                ), None)
                if protected_close_job is not None:
                    self._log_rejection(
                        "lower-priority update cannot replace pending close warning", text
                    )
                    return False
                self._jobs = [job for job in self._jobs if job.key != key]
                if self.debug:
                    for replaced_job in replaced_jobs:
                        print(
                            f"[Audio] Replaced queued announcement for {key}: "
                            f"{replaced_job.text} -> {text}"
                        )
            if len(self._jobs) >= self.max_pending:
                lowest = min(range(len(self._jobs)), key=lambda i: (self._jobs[i].priority, -self._jobs[i].sequence))
                if priority < self._jobs[lowest].priority:
                    self._log_rejection("queue full; higher priority pending", text)
                    return False
                dropped = self._jobs.pop(lowest)
                if dropped.key is not None:
                    self._dropped_keys.add(dropped.key)
                if self.debug:
                    print(f"[Audio] Replaced outdated queued announcement: {dropped.text}")

            self._sequence += 1
            self._jobs.append(SpeechJob(text, expires_at, priority, key, self._sequence, validation_token))
            self._condition.notify()
        print(f"[Audio] Announcement queued (priority={priority}): {text}")
        return True

    def wait_idle(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._jobs or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def consume_dropped_keys(self) -> set[str]:
        with self._condition:
            dropped = self._dropped_keys
            self._dropped_keys = set()
            return dropped

    def stop(self, timeout: float = 3.0) -> None:
        # Serialize concurrent/repeated stop calls so only one caller manages
        # graceful waiting and the forced-termination fallback.
        with self._stop_lock:
            with self._condition:
                self._stopping = True
                pending_count = len(self._jobs)
                self._dropped_keys.update(
                    job.key for job in self._jobs if job.key is not None
                )
                self._jobs.clear()
                self._condition.notify_all()
            # The active backend request has its own request_timeout. Give it
            # that full bounded interval plus time for PowerShell's shutdown.
            graceful_wait = max(
                timeout,
                min(
                    self.request_timeout + SPEECH_SHUTDOWN_CLOSE_GRACE_SECONDS,
                    SPEECH_SHUTDOWN_MAX_WAIT_SECONDS,
                ),
            )
            if pending_count and self.debug:
                print(f"[Audio] Discarded {pending_count} queued announcement(s) during shutdown.")
            self.thread.join(timeout=graceful_wait)
            if self.thread.is_alive():
                print("[Audio] Speech backend exceeded its bounded graceful-shutdown deadline; terminating it.")
                # Re-read after waiting: the worker may have changed its backend
                # reference while an in-flight request was resolving.
                with self._condition:
                    backend = self._backend
                if backend is not None:
                    backend.terminate()
                self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                print("[Audio] Speech worker did not stop within the bounded timeout.")

    def _set_backend_status(self, ready: bool, status: str) -> None:
        with self._condition:
            self._backend_ready = ready
            self._status = status
            self._condition.notify_all()

    def _run(self) -> None:
        backend: Optional[Any] = None
        next_retry = 0.0
        try:
            try:
                backend = self._backend_factory()
                with self._condition:
                    self._backend = backend
                self._set_backend_status(True, "ready")
                voices = getattr(backend, "voice_count", "system default")
                print(f"[Audio] System.Speech backend ready; enabled voices={voices}.")
            except Exception as exc:
                self.failure_generation += 1
                self._set_backend_status(False, "unavailable")
                next_retry = time.monotonic() + 1.0
                print(f"[Audio] System.Speech initialization failed: {exc}")
            finally:
                self._ready_event.set()

            while True:
                with self._condition:
                    while not self._jobs and not self._stopping:
                        if backend is None:
                            retry_in = next_retry - time.monotonic()
                            if retry_in <= 0:
                                break
                            self._condition.wait(min(retry_in, 0.5))
                        else:
                            self._condition.wait()
                    if self._stopping:
                        break
                    if self._jobs:
                        index = max(range(len(self._jobs)), key=lambda i: (self._jobs[i].priority, -self._jobs[i].sequence))
                        job: Optional[SpeechJob] = self._jobs.pop(index)
                    else:
                        job = None
                    self._inflight = True

                try:
                    if job is None:
                        try:
                            backend = self._backend_factory()
                            with self._condition:
                                self._backend = backend
                            self._set_backend_status(True, "ready")
                            print("[Audio] System.Speech backend recovered.")
                            next_retry = 0.0
                        except Exception as exc:
                            self.failure_generation += 1
                            self._set_backend_status(False, "unavailable")
                            next_retry = time.monotonic() + 3.0
                            print(f"[Audio] System.Speech retry failed: {exc}")
                        continue
                    if not self.enabled:
                        continue
                    if job.expires_at is not None and time.monotonic() > job.expires_at:
                        if job.key is not None:
                            with self._condition:
                                self._dropped_keys.add(job.key)
                        self._log_rejection("announcement expired in queue", job.text)
                        continue
                    validator = self._job_validator
                    if validator is not None and job.validation_token is not None:
                        try:
                            invalid_reason = validator(job)
                        except Exception as validation_error:
                            invalid_reason = f"announcement_validation_failed: {validation_error}"
                        if invalid_reason is not None:
                            if job.key is not None:
                                with self._condition:
                                    self._dropped_keys.add(job.key)
                            self._log_rejection(invalid_reason, job.text)
                            continue
                    if backend is None:
                        if time.monotonic() < next_retry:
                            if job.key is not None:
                                with self._condition:
                                    self._dropped_keys.add(job.key)
                            self._log_rejection("backend retry delay", job.text)
                            continue
                        backend = self._backend_factory()
                        with self._condition:
                            self._backend = backend
                        self._set_backend_status(True, "ready")
                        print("[Audio] System.Speech backend recovered.")

                    print(f"[Audio] Sent to speech backend: {job.text}")

                    def log_accepted() -> None:
                        print(f"[Audio] Backend accepted request: {job.text}")

                    result = backend.speak(job.text, self.request_timeout, log_accepted)
                    if result.get("status") != "completed":
                        raise RuntimeError(f"Speech backend did not confirm completion: {result}")
                    self.completed_count += 1
                    print(f"[Audio] Speech synthesis completed (backend confirmation): {job.text}")
                except Exception as exc:
                    self.failure_generation += 1
                    if job is not None and job.key is not None:
                        with self._condition:
                            self._dropped_keys.add(job.key)
                    print(f"[Audio] Playback failure or timeout: {exc}")
                    self._set_backend_status(False, "recovering")
                    try:
                        if backend is not None:
                            backend.close(timeout=0.5)
                    except Exception as close_exc:
                        print(f"[Audio] Backend cleanup failed: {close_exc}")
                    backend = None
                    with self._condition:
                        self._backend = None
                    next_retry = time.monotonic() + 0.25
                finally:
                    with self._condition:
                        self._inflight = False
                        self._condition.notify_all()
        except Exception as exc:
            self.failure_generation += 1
            self._set_backend_status(False, "worker error")
            print(f"[Audio] Speech worker failed unexpectedly: {exc}")
        finally:
            self._ready_event.set()
            try:
                if backend is not None:
                    backend.close(timeout=2.0)
            except Exception as exc:
                print(f"[Audio] Speech backend shutdown failed: {exc}")
            with self._condition:
                self._backend = None
            self._set_backend_status(False, "stopped")
            print("[Audio] Speech worker stopped.")


@dataclass
class ObjectObservation:
    track_id: int
    label: str
    confidence: float
    direction: str
    distance_m: Optional[float]
    approaching: bool
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class DistanceState:
    distance_m: float
    captured_at: float


def is_valid_distance(distance_m: Optional[float], max_depth_m: float) -> bool:
    return (
        distance_m is not None
        and math.isfinite(distance_m)
        and 0.0 < distance_m <= max_depth_m
    )


def fresh_track_distance(
    state: Optional[DistanceState],
    now: float,
    max_age_seconds: float,
    max_depth_m: float,
) -> Optional[float]:
    if state is None or not is_valid_distance(state.distance_m, max_depth_m):
        return None
    age = now - state.captured_at
    if not 0.0 <= age <= max_age_seconds:
        return None
    return state.distance_m


def format_distance(distance_m: Optional[float], max_depth_m: float) -> str:
    if not is_valid_distance(distance_m, max_depth_m):
        return "Distance unavailable"
    return f"~{distance_m:.2f} m est."


def direction_for_box(
    x1: int,
    x2: int,
    frame_width: int,
    previous: Optional[str] = None,
    hysteresis: float = DIRECTION_HYSTERESIS,
) -> str:
    """Camera-relative zones with a small hold margin at zone boundaries."""
    cx = (x1 + x2) / 2.0
    frac = cx / max(1.0, float(frame_width))
    boundaries = (0.20, 0.38, 0.62, 0.80)
    zones = ("far left", "left", "ahead", "right", "far right")
    index = sum(frac >= boundary for boundary in boundaries)
    if previous in zones:
        previous_index = zones.index(previous)
        if index > previous_index and frac - boundaries[previous_index] < hysteresis:
            index = previous_index
        elif index < previous_index and boundaries[previous_index - 1] - frac < hysteresis:
            index = previous_index
    return zones[index]


def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def suppress_duplicate_detections(
    detections: list["DetectionSnapshot"],
    duplicate_iou: float = 0.82,
) -> Tuple[list["DetectionSnapshot"], list[Tuple["DetectionSnapshot", "DetectionSnapshot", float]]]:
    """Suppress duplicate geometry while retaining spatially distinct detections."""
    kept: list[DetectionSnapshot] = []
    removed: list[Tuple[DetectionSnapshot, DetectionSnapshot, float]] = []
    for detection in sorted(
        detections,
        key=lambda item: (not item.has_track_id, item.track_id if item.has_track_id else -item.confidence),
    ):
        duplicate = None
        for existing in kept:
            overlap = bbox_iou(detection.bbox, existing.bbox)
            ax1, ay1, ax2, ay2 = detection.bbox
            bx1, by1, bx2, by2 = existing.bbox
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            smaller_area = min(area_a, area_b)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            containment = intersection / smaller_area if smaller_area else 0.0
            center_a = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
            center_b = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            max_diagonal = max(
                math.hypot(max(0, ax2 - ax1), max(0, ay2 - ay1)),
                math.hypot(max(0, bx2 - bx1), max(0, by2 - by1)),
                1.0,
            )
            center_separation = math.dist(center_a, center_b) / max_diagonal
            same_class = detection.label == existing.label
            height_a, height_b = max(0, ay2 - ay1), max(0, by2 - by1)
            max_height = max(height_a, height_b, 1)
            person_height_similarity = min(height_a, height_b) / max_height
            person_vertical_edge_delta = max(abs(ay1 - by1), abs(ay2 - by2)) / max_height
            same_track = (
                detection.has_track_id
                and existing.has_track_id
                and detection.track_id == existing.track_id
                and (overlap >= 0.50 or containment >= 0.90)
            )
            nested_person_duplicate = (
                same_class
                and detection.label.lower() == "person"
                and containment >= 0.90
                and center_separation <= 0.25
                and person_height_similarity >= 0.85
                and person_vertical_edge_delta <= 0.10
            )
            same_class_duplicate = same_class and (
                overlap >= duplicate_iou
                or nested_person_duplicate
                or (detection.label.lower() != "person"
                    and containment >= 0.92 and center_separation <= 0.20)
            )
            cross_class_duplicate = not same_class and (
                overlap >= 0.88 and center_separation <= 0.10
            )
            if same_track or same_class_duplicate or cross_class_duplicate:
                duplicate = existing, overlap
                break
        if duplicate is None:
            kept.append(detection)
        else:
            existing, overlap = duplicate
            # Track IDs are allocated monotonically by ByteTrack. Keep the older ID for
            # repeatable identity when two track rows describe the same image region.
            if (detection.has_track_id and existing.has_track_id
                    and detection.track_id < existing.track_id):
                kept.remove(existing)
                kept.append(detection)
                removed.append((detection, existing, overlap))
            else:
                removed.append((existing, detection, overlap))
    return kept, removed


@dataclass(frozen=True)
class DepthSample:
    median_m: Optional[float]
    raw_median_m: Optional[float]
    valid_pixels: int
    rejection_reason: Optional[str]
    roi_width: int = 0
    roi_height: int = 0
    sampled_pixels: int = 0
    invalid_depth_pixels: int = 0
    out_of_range_pixels: int = 0
    overlap_masked_pixels: int = 0


def object_sampling_roi(
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, ...],
    label: str = "",
) -> Optional[Tuple[int, int, int, int]]:
    """Return the same conservative inner ROI used by the depth sampler."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    margin_x = max(1, int(math.ceil(bw * 0.25)))
    ix1, ix2 = x1 + margin_x, x2 - margin_x
    if label.lower() == "person":
        iy1 = y1 + max(1, int(math.ceil(bh * 0.25)))
        iy2 = y1 + max(2, int(math.floor(bh * 0.72)))
    else:
        margin_y = max(1, int(math.ceil(bh * 0.25)))
        iy1, iy2 = y1 + margin_y, y2 - margin_y
    iy1 = max(y1, min(y2 - 1, iy1))
    iy2 = max(y1, min(y2, iy2))
    if ix2 <= ix1:
        center_x = (x1 + x2 - 1) // 2
        ix1, ix2 = center_x, center_x + 1
    if iy2 <= iy1:
        center_y = (y1 + y2 - 1) // 2
        iy1, iy2 = center_y, center_y + 1
    return ix1, iy1, ix2, iy2


def median_object_depth(
    depth_map: np.ndarray,
    box: Tuple[int, int, int, int],
    max_depth_m: float,
    label: str = "",
    other_boxes: Tuple[Tuple[int, int, int, int], ...] = (),
) -> DepthSample:
    """Sample an inner object ROI, excluding frame edges and overlapping detections."""
    roi_bounds = object_sampling_roi(box, depth_map.shape, label)
    if roi_bounds is None:
        return DepthSample(None, None, 0, "box_outside_depth_map")
    ix1, iy1, ix2, iy2 = roi_bounds
    roi = depth_map[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return DepthSample(None, None, 0, "empty_sampling_roi", ix2 - ix1, iy2 - iy1)
    roi_valid = np.ones(roi.shape, dtype=bool)
    for other_box in other_boxes:
        ox1, oy1, ox2, oy2 = other_box
        ox1, ox2 = max(ix1, ox1), min(ix2, ox2)
        oy1, oy2 = max(iy1, oy1), min(iy2, oy2)
        if ox2 > ox1 and oy2 > oy1:
            # A larger enclosing detection (commonly a person around a held phone)
            # does not identify competing pixels inside this object-specific ROI.
            if other_box[0] <= ix1 and other_box[1] <= iy1 and other_box[2] >= ix2 and other_box[3] >= iy2:
                continue
            roi_valid[oy1 - iy1:oy2 - iy1, ox1 - ix1:ox2 - ix1] = False
    unmasked = roi[roi_valid]
    finite = np.isfinite(unmasked)
    invalid_depth_pixels = int(np.count_nonzero(~finite | (finite & (unmasked <= 0.15))))
    out_of_range_pixels = int(np.count_nonzero(finite & (unmasked > max_depth_m)))
    vals = unmasked[finite & (unmasked > 0.15) & (unmasked <= max_depth_m)]
    overlap_masked_pixels = int(roi.size - unmasked.size)
    common = dict(
        roi_width=ix2 - ix1, roi_height=iy2 - iy1, sampled_pixels=int(roi.size),
        invalid_depth_pixels=invalid_depth_pixels, out_of_range_pixels=out_of_range_pixels,
        overlap_masked_pixels=overlap_masked_pixels,
    )
    raw_median = float(np.median(vals)) if vals.size else None
    if vals.size < 4:
        return DepthSample(None, raw_median, int(vals.size), "fewer_than_4_valid_roi_pixels", **common)
    d = raw_median
    if not math.isfinite(d) or d <= 0 or d > max_depth_m:
        return DepthSample(None, raw_median, int(vals.size), "median_out_of_range", **common)
    return DepthSample(d, raw_median, int(vals.size), None, **common)


def depth_source_frames_match(detection_frame_id: Optional[int], depth_frame_id: int) -> bool:
    return detection_frame_id is not None and detection_frame_id == depth_frame_id


def depth_diagnostic_reason(
    sample: Optional[DepthSample],
    *,
    depth_result_present: bool,
    source_frame_match: Optional[bool],
    result_fresh: bool,
    final_reason: Optional[str] = None,
) -> str:
    if not depth_result_present:
        return "NO_DEPTH_RESULT"
    if source_frame_match is False:
        return "SOURCE_FRAME_MISMATCH"
    if not result_fresh:
        return "STALE_DEPTH_RESULT"
    if final_reason in ("depth_change_pending_confirmation", "smoothed_distance_invalid"):
        return final_reason.upper()
    if sample is None:
        return final_reason.upper() if final_reason else "INSUFFICIENT_VALID_PIXELS"
    if sample.valid_pixels >= 4 and sample.median_m is not None:
        return final_reason.upper() if final_reason else "DISTANCE_AVAILABLE"
    roi_pixels = sample.roi_width * sample.roi_height
    if roi_pixels < 4:
        return "ROI_TOO_SMALL"
    if sample.valid_pixels == 0 and sample.overlap_masked_pixels >= roi_pixels:
        return "OVERLAPPING_DETECTION_MASK"
    if sample.valid_pixels == 0 and sample.out_of_range_pixels:
        return "OUT_OF_RANGE"
    if sample.valid_pixels == 0 and sample.invalid_depth_pixels:
        return "INVALID_DEPTH_VALUES"
    return "INSUFFICIENT_VALID_PIXELS"


def depth_worker_unavailable_reason(frame_id: int, status: dict[str, Any]) -> str:
    """Describe a missing same-frame result without implying every wait is a model failure."""
    processing = status.get("processing_frame_id")
    pending = status.get("pending_frame_id")
    if processing is not None:
        state = f"DEPTH_WORKER_PROCESSING(frame={processing})"
    elif pending is not None:
        state = f"DEPTH_JOB_PENDING(frame={pending})"
    elif status.get("failures", 0) and status.get("last_error"):
        state = f"DEPTH_WORKER_FAILURE({status['last_error']})"
    elif status.get("last_submitted_frame_id") is None:
        state = "NO_DEPTH_JOB_SUBMITTED"
    elif status.get("latest_completed_frame_id") is None:
        state = "NO_COMPLETED_DEPTH_RESULT"
    elif status.get("last_submitted_frame_id", -1) < frame_id:
        state = "NO_DEPTH_JOB_SUBMITTED_FOR_FRAME"
    else:
        state = f"NO_COMPLETED_RESULT_FOR_SOURCE_FRAME(latest={status['latest_completed_frame_id']})"
    return (
        f"{state}; submitted={status.get('submitted', 0)}"
        f" pending_replaced={status.get('pending_replaced', 0)}"
        f" completed={status.get('completed', 0)}"
        f" unconsumed_replaced={status.get('unconsumed_result_replaced', 0)}"
        f" failures={status.get('failures', 0)}"
    )


def format_depth_sampling_diagnostic(
    detection: DetectionSnapshot,
    sample: Optional[DepthSample],
    *,
    depth_frame_id: Optional[int],
    depth_age_seconds: Optional[float],
    source_frame_match: Optional[bool],
    reason: str,
) -> str:
    x1, y1, x2, y2 = detection.bbox
    box_width, box_height = max(0, x2 - x1), max(0, y2 - y1)
    if sample is None:
        roi_width = roi_height = sampled = valid = invalid = out_of_range = masked = 0
        raw_median = "n/a"
    else:
        roi_width, roi_height = sample.roi_width, sample.roi_height
        sampled, valid = sample.sampled_pixels, sample.valid_pixels
        invalid = sample.invalid_depth_pixels
        out_of_range = sample.out_of_range_pixels
        masked = sample.overlap_masked_pixels
        raw_median = f"{sample.raw_median_m:.3f}m" if sample.raw_median_m is not None else "n/a"
    valid_pct = 100.0 * valid / sampled if sampled else 0.0
    age_ms = f"{depth_age_seconds * 1000.0:.1f}" if depth_age_seconds is not None else "n/a"
    same_frame = "yes" if source_frame_match is True else "no" if source_frame_match is False else "n/a"
    median = f"{sample.median_m:.3f}m" if sample is not None and sample.median_m is not None else "n/a"
    return (
        f"track={detection.track_id} class={detection.label} conf={detection.confidence:.2f} "
        f"bbox_px={box_width}x{box_height} roi_px={roi_width}x{roi_height} sampled={sampled} "
        f"valid={valid} ({valid_pct:.1f}%) invalid={invalid} out_of_range={out_of_range} "
        f"overlap_masked={masked} raw_depth={raw_median} median={median} "
        f"depth_frame={depth_frame_id if depth_frame_id is not None else 'n/a'} "
        f"detection_frame={detection.source_frame_id if detection.source_frame_id is not None else 'n/a'} "
        f"same_source_frame={same_frame} age_ms={age_ms} reason={reason}"
    )


class MetricDepthEstimator:
    def __init__(self, device: str, use_fp16: bool, environment: str) -> None:
        self.environment = environment
        self.model_id, self.max_depth_m = DEPTH_MODELS[environment]
        print(f"[Depth] Environment: {environment}; maximum range: {self.max_depth_m:.0f} m")
        print(f"[Depth] Loading metric {environment} depth model: {self.model_id}")
        cache_dir = Path(__file__).resolve().parent / "models" / "huggingface"
        print(f"[Depth] Resolving local {environment} checkpoint from: {cache_dir}")
        snapshot_path = snapshot_download(
            self.model_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        print(f"[Depth] Local snapshot: {snapshot_path}")
        self.use_fp16 = use_fp16 and device == "cuda"
        print(f"[Depth] Precision: {'FP16 autocast' if self.use_fp16 else 'FP32'}")
        self.processor = AutoImageProcessor.from_pretrained(snapshot_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(snapshot_path, local_files_only=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(**inputs)
            # Keep interpolation and depth post-processing in stable FP32.
            if getattr(outputs, "predicted_depth", None) is not None:
                outputs.predicted_depth = outputs.predicted_depth.float()
        else:
            outputs = self.model(**inputs)
        post = self.processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
        depth = post[0]["predicted_depth"].detach().float().cpu().numpy()
        depth = np.squeeze(depth)
        if depth.shape != (h, w):
            depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        return depth.astype(np.float32)


@dataclass(frozen=True)
class DetectionSnapshot:
    track_id: int
    has_track_id: bool
    label: str
    confidence: float
    direction: str
    bbox: Tuple[int, int, int, int]
    source_frame_id: Optional[int] = None
    identity_uncertain: bool = False
    possible_parent_track_id: Optional[int] = None


@dataclass
class VisionTrackState:
    """Short-lived temporal state for one genuine ByteTrack identity."""

    smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
    stable_label: Optional[str] = None
    class_history: Deque[Tuple[float, str, float]] = field(default_factory=lambda: deque(maxlen=11))
    pending_label: Optional[str] = None
    pending_label_count: int = 0
    stable_direction: Optional[str] = None
    pending_direction: Optional[str] = None
    pending_direction_count: int = 0
    first_seen_at: Optional[float] = None
    strong_evidence: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=5))
    announcement_eligible: bool = False


@dataclass(frozen=True)
class OcclusionIdentityCandidate:
    possible_parent_track_id: int
    occluder_track_id: int
    containment: float
    parent_age_seconds: float


def find_person_occlusion_ambiguity(
    detection: DetectionSnapshot,
    detections: list[DetectionSnapshot],
    states: Dict[int, VisionTrackState],
    last_seen: Dict[int, float],
    now: float,
    max_parent_age_seconds: float = TRACK_REACQUIRE_RESET_SECONDS,
) -> Optional[OcclusionIdentityCandidate]:
    """Flag a lower-body person box nested in a recent person box while a stable object covers its upper region.

    This is deliberately a warning state, not an ID merge: tracker IDs and their
    depth, motion, and announcement histories remain independent.
    """
    if not detection.has_track_id or detection.label.lower() != "person":
        return None
    x1, y1, x2, y2 = detection.bbox
    child_width, child_height = max(0, x2 - x1), max(0, y2 - y1)
    child_area = child_width * child_height
    if child_area <= 0:
        return None

    matches: list[OcclusionIdentityCandidate] = []
    for parent_id, parent_state in states.items():
        seen_at = last_seen.get(parent_id)
        if (parent_id == detection.track_id or parent_state.stable_label != "person"
                or parent_state.smoothed_bbox is None or seen_at is None):
            continue
        age = now - seen_at
        if not 0.0 <= age <= max_parent_age_seconds:
            continue
        px1, py1, px2, py2 = (int(round(value)) for value in parent_state.smoothed_bbox)
        parent_width, parent_height = max(0, px2 - px1), max(0, py2 - py1)
        parent_area = parent_width * parent_height
        if parent_area <= 0 or child_area / parent_area > 0.75:
            continue
        intersection = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
        containment = intersection / child_area
        if containment < 0.92:
            continue
        if (y1 < py1 + 0.35 * parent_height
                or abs(y2 - py2) > 0.10 * parent_height
                or abs((x1 + x2) / 2.0 - (px1 + px2) / 2.0) > 0.15 * parent_width):
            continue

        face_region = (px1, py1, px2, py1 + int(0.40 * parent_height))
        for occluder in detections:
            if not occluder.has_track_id or occluder.label.lower() == "person":
                continue
            occluder_state = states.get(occluder.track_id)
            if occluder_state is None or occluder_state.stable_label != occluder.label:
                continue
            recent_class_evidence = [
                stamp for stamp, label, confidence in occluder_state.class_history
                if label == occluder.label and confidence >= CLASS_HISTORY_MIN_CONFIDENCE
                and 0.0 <= now - stamp <= 2.0
            ]
            if len(recent_class_evidence) < 2:
                continue
            ox1, oy1, ox2, oy2 = occluder.bbox
            occluder_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            face_overlap = max(0, min(face_region[2], ox2) - max(face_region[0], ox1)) * max(
                0, min(face_region[3], oy2) - max(face_region[1], oy1)
            )
            if occluder_area and face_overlap / occluder_area >= 0.25:
                matches.append(OcclusionIdentityCandidate(parent_id, occluder.track_id, containment, age))

    if not matches:
        return None
    return min(matches, key=lambda item: (item.parent_age_seconds, -item.containment))


def is_detection_announcement_eligible(
    detection: DetectionSnapshot,
    state: Optional[VisionTrackState],
) -> bool:
    """Keep uncertain partial-person detections visible but out of spoken identity claims."""
    return (
        detection.has_track_id
        and not detection.identity_uncertain
        and state is not None
        and is_announcement_eligible(state, detection.confidence)
    )


def reconcile_person_uncertainty(
    previous_uncertain_ids: set[int],
    current_uncertain_ids: set[int],
    states: Dict[int, VisionTrackState],
) -> Tuple[set[int], set[int], set[int]]:
    """Reset speech evidence during ambiguity and on its clearing transition.

    Returns current IDs, newly uncertain IDs, and IDs whose uncertainty cleared.
    A track must accumulate the existing strong-evidence window again before speech.
    """
    current = set(current_uncertain_ids)
    entered = current - previous_uncertain_ids
    cleared = previous_uncertain_ids - current
    for track_id in previous_uncertain_ids | current:
        state = states.get(track_id)
        if state is not None:
            state.strong_evidence.clear()
            state.announcement_eligible = False
    return current, entered, cleared


def application_tracks_to_purge(
    tracked_state_ids: set[int],
    last_seen: Dict[int, float],
    now: float,
    *,
    removed_tracker_ids: Optional[set[int]] = None,
    tracker_lifecycle_available: bool,
) -> set[int]:
    """Keep app state for tracker IDs through ByteTrack's frame-based reactivation window."""
    if tracker_lifecycle_available:
        return tracked_state_ids.intersection(removed_tracker_ids or set())
    return {
        track_id for track_id in tracked_state_ids
        if track_id in last_seen and now - last_seen[track_id] > TRACK_STATE_PURGE_SECONDS
    }


def purge_application_track_state(
    track_id: int,
    *,
    last_seen_by_track: Dict[int, Any],
    direction_by_track: Dict[int, Any],
    vision_state_by_track: Dict[int, Any],
    distance_state_by_track: Dict[int, Any],
    histories: Dict[int, Any],
    raw_histories: Dict[int, Any],
    pending_depth_by_track: Dict[int, Any],
    depth_diagnostic_by_track: Dict[int, Any],
    motion_by_track: Dict[int, Any],
    last_announced: Dict[int, Any],
    preserve_last_seen: bool = False,
) -> None:
    """Remove every app-owned state entry for one expired or suppressed tracker ID."""
    state_maps = (
        direction_by_track,
        vision_state_by_track,
        distance_state_by_track,
        histories,
        raw_histories,
        pending_depth_by_track,
        depth_diagnostic_by_track,
        motion_by_track,
        last_announced,
    )
    if not preserve_last_seen:
        state_maps = (last_seen_by_track, *state_maps)
    for state_by_track in state_maps:
        state_by_track.pop(track_id, None)


def reacquisition_needs_state_reset(
    state: Optional[VisionTrackState],
    previous_seen_at: Optional[float],
    now: float,
) -> bool:
    """Keep state through brief ByteTrack gaps; reset it after a long reactivation gap."""
    return (
        state is not None
        and state.first_seen_at is not None
        and previous_seen_at is not None
        and now - previous_seen_at > TRACK_REACQUIRE_RESET_SECONDS
    )


def reset_reacquired_track_state(
    track_id: int,
    *,
    last_seen_by_track: Dict[int, Any],
    direction_by_track: Dict[int, Any],
    vision_state_by_track: Dict[int, Any],
    distance_state_by_track: Dict[int, Any],
    histories: Dict[int, Any],
    raw_histories: Dict[int, Any],
    pending_depth_by_track: Dict[int, Any],
    depth_diagnostic_by_track: Dict[int, Any],
    motion_by_track: Dict[int, Any],
    last_announced: Dict[int, Any],
) -> VisionTrackState:
    """Drop temporal state before reusing an ID after a prolonged tracker gap."""
    purge_application_track_state(
        track_id,
        last_seen_by_track=last_seen_by_track,
        direction_by_track=direction_by_track,
        vision_state_by_track=vision_state_by_track,
        distance_state_by_track=distance_state_by_track,
        histories=histories,
        raw_histories=raw_histories,
        pending_depth_by_track=pending_depth_by_track,
        depth_diagnostic_by_track=depth_diagnostic_by_track,
        motion_by_track=motion_by_track,
        last_announced=last_announced,
        preserve_last_seen=True,
    )
    state = VisionTrackState()
    vision_state_by_track[track_id] = state
    return state


def possible_fragmented_track(
    detection: DetectionSnapshot,
    now: float,
    states: Dict[int, VisionTrackState],
    last_seen: Dict[int, float],
    max_age_seconds: float = 3.0,
) -> Optional[Tuple[int, float, float]]:
    """Return a geometry-based ID-fragmentation candidate for diagnostics only."""
    best: Optional[Tuple[int, float, float]] = None
    for other_id, state in states.items():
        seen_at = last_seen.get(other_id)
        if (other_id == detection.track_id or seen_at is None
                or not 0.0 <= now - seen_at <= max_age_seconds
                or state.stable_label != detection.label or state.smoothed_bbox is None):
            continue
        previous_box = tuple(int(round(value)) for value in state.smoothed_bbox)
        overlap = bbox_iou(detection.bbox, previous_box)
        if overlap >= 0.25 and (best is None or overlap > best[1]):
            best = (other_id, overlap, now - seen_at)
    return best


def stabilize_detection(
    detection: DetectionSnapshot,
    state: VisionTrackState,
    frame_width: int,
    observed_at: Optional[float] = None,
    class_switch_frames: int = 4,
    direction_switch_frames: int = 3,
) -> DetectionSnapshot:
    """Apply confidence voting, responsive box smoothing, and sustained direction changes."""
    x1, y1, x2, y2 = detection.bbox
    observed_at = time.perf_counter() if observed_at is None else observed_at
    if state.first_seen_at is None:
        state.first_seen_at = observed_at
    state.strong_evidence.append((observed_at, detection.confidence))
    recent_strong = [
        (stamp, confidence) for stamp, confidence in state.strong_evidence
        if 0.0 <= observed_at - stamp <= ANNOUNCEMENT_EVIDENCE_WINDOW_SECONDS
        and confidence >= ANNOUNCEMENT_MIN_CONFIDENCE
    ]
    if (len(recent_strong) >= ANNOUNCEMENT_STRONG_HITS
            and recent_strong[-1][0] - recent_strong[0][0] >= ANNOUNCEMENT_EVIDENCE_MIN_SPAN_SECONDS):
        state.announcement_eligible = True

    raw_box = tuple(float(value) for value in detection.bbox)
    if state.smoothed_bbox is None:
        state.smoothed_bbox = raw_box
    else:
        old = state.smoothed_bbox
        old_cx, old_cy = (old[0] + old[2]) / 2.0, (old[1] + old[3]) / 2.0
        new_cx, new_cy = (raw_box[0] + raw_box[2]) / 2.0, (raw_box[1] + raw_box[3]) / 2.0
        old_diagonal = max(1.0, math.hypot(old[2] - old[0], old[3] - old[1]))
        displacement = math.hypot(new_cx - old_cx, new_cy - old_cy) / old_diagonal
        # Favor the new box on rapid motion to limit lag; smooth ordinary detector jitter.
        new_weight = 0.82 if displacement >= 0.18 else 0.62
        state.smoothed_bbox = tuple(
            previous * (1.0 - new_weight) + current * new_weight
            for previous, current in zip(old, raw_box)
        )
    box = tuple(int(round(value)) for value in state.smoothed_bbox)
    sx1, sy1, sx2, sy2 = box
    if sx2 <= sx1:
        sx2 = sx1 + 1
    if sy2 <= sy1:
        sy2 = sy1 + 1
    stable_box = (sx1, sy1, sx2, sy2)

    if (isinstance(detection.label, str) and detection.label.strip()
            and math.isfinite(detection.confidence)
            and detection.confidence >= CLASS_HISTORY_MIN_CONFIDENCE):
        state.class_history.append((observed_at, detection.label, detection.confidence))
    state.class_history = deque(
        ((stamp, label, confidence)
         for stamp, label, confidence in state.class_history
         if (0.0 <= observed_at - stamp <= 2.0
             and isinstance(label, str) and bool(label.strip())
             and math.isfinite(confidence) and confidence >= CLASS_HISTORY_MIN_CONFIDENCE)),
        maxlen=11,
    )
    output_label = detection.label
    if state.stable_label is None:
        if detection.label:
            state.stable_label = detection.label
            output_label = state.stable_label
    else:
        weighted_votes: Dict[str, float] = defaultdict(float)
        valid_history = list(state.class_history)
        for age, (_stamp, label, confidence) in enumerate(reversed(valid_history)):
            weighted_votes[label] += confidence * (0.78 ** age)
        if not weighted_votes:
            # No reliable votes remain (for example after expiry or invalidation).
            # Keep stabilization state untouched and use this frame's label safely.
            output_label = detection.label
        else:
            winner = max(weighted_votes, key=weighted_votes.get)
            if winner == state.stable_label:
                state.pending_label = None
                state.pending_label_count = 0
            elif winner == state.pending_label:
                state.pending_label_count += 1
            else:
                state.pending_label = winner
                state.pending_label_count = 1
            incumbent_score = weighted_votes.get(state.stable_label, 0.0)
            winner_score = weighted_votes.get(winner, 0.0)
            if (state.pending_label_count >= class_switch_frames
                    and winner_score >= incumbent_score * 1.10):
                state.stable_label = winner
                state.pending_label = None
                state.pending_label_count = 0
            output_label = state.stable_label or detection.label

    candidate_direction = direction_for_box(
        sx1, sx2, frame_width, state.stable_direction
    )
    if state.stable_direction is None:
        state.stable_direction = candidate_direction
    elif candidate_direction == state.stable_direction:
        state.pending_direction = None
        state.pending_direction_count = 0
    elif candidate_direction == state.pending_direction:
        state.pending_direction_count += 1
    else:
        state.pending_direction = candidate_direction
        state.pending_direction_count = 1
    if state.pending_direction_count >= direction_switch_frames:
        state.stable_direction = candidate_direction
        state.pending_direction = None
        state.pending_direction_count = 0

    return replace(
        detection,
        label=output_label,
        direction=state.stable_direction or candidate_direction,
        bbox=stable_box,
    )


def is_announcement_eligible(state: VisionTrackState, current_confidence: float) -> bool:
    """Only speak for tracks with repeated strong evidence and a strong current observation."""
    return (
        state.announcement_eligible
        and current_confidence >= ANNOUNCEMENT_MIN_CONFIDENCE
    )


def confidence_stage(confidence: float) -> str:
    if confidence >= BYTE_TRACK_NEW_TRACK_CONFIDENCE:
        return "new_track_score_band"
    if confidence >= BYTE_TRACK_HIGH_CONFIDENCE:
        return "high_stage_match_band"
    if confidence >= BYTE_TRACK_LOW_CONFIDENCE:
        return "low_stage_recovery_candidate"
    return "below_tracker_floor"


def yolo_result_rows(results: list[Any]) -> list[dict[str, Any]]:
    """Copy relevant Ultralytics result boxes into CPU-side diagnostic rows."""
    rows: list[dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        ids = (
            boxes.id.detach().cpu().numpy().astype(int)
            if boxes.id is not None else [None] * len(xyxy)
        )
        names = getattr(result, "names", {})
        for box, class_id, confidence, track_id in zip(xyxy, classes, confidences, ids):
            rows.append({
                "label": names.get(int(class_id), str(class_id)),
                "confidence": float(confidence),
                "bbox": tuple(int(round(value)) for value in box),
                "track_id": int(track_id) if track_id is not None else None,
            })
    return rows


def install_vision_debug_candidate_capture(predictor: Any, candidate_floor: float = 0.001) -> None:
    """Capture lower-threshold classwise YOLO candidates before ByteTrack, debug mode only."""
    if getattr(predictor, "_sightbridge_candidate_capture_installed", False):
        return
    from ultralytics.utils import nms

    original_postprocess = predictor.postprocess

    def postprocess_with_candidate_capture(predictions: Any, image: Any, original_images: Any, *args: Any, **kwargs: Any):
        snapshot: dict[str, Any] = {"candidate_floor": candidate_floor}
        try:
            if isinstance(predictions, torch.Tensor):
                diagnostic_predictions = predictions.clone()
            elif isinstance(predictions, tuple) and predictions and isinstance(predictions[0], torch.Tensor):
                diagnostic_predictions = (predictions[0].clone(), *predictions[1:])
            elif isinstance(predictions, list) and predictions and isinstance(predictions[0], torch.Tensor):
                diagnostic_predictions = [predictions[0].clone(), *predictions[1:]]
            else:
                diagnostic_predictions = predictions
            classes_count = 0 if predictor.args.task == "detect" else len(predictor.model.names)
            candidates = nms.non_max_suppression(
                diagnostic_predictions,
                conf_thres=candidate_floor,
                iou_thres=kwargs.get("iou", predictor.args.iou),
                classes=predictor.args.classes,
                agnostic=False,
                max_det=predictor.args.max_det,
                nc=classes_count,
                end2end=getattr(predictor.model, "end2end", False),
            )
            candidate_results = predictor.construct_results(candidates, image, original_images, **kwargs)
            snapshot["candidates"] = yolo_result_rows(candidate_results)
        except Exception as exc:
            snapshot["capture_error"] = f"{type(exc).__name__}: {exc}"

        # These are the normal confidence-filtered/NMS results supplied to ByteTrack.
        standard_results = original_postprocess(predictions, image, original_images, *args, **kwargs)
        snapshot["pretracker"] = yolo_result_rows(standard_results)
        predictor._sightbridge_candidate_snapshot = snapshot
        return standard_results

    predictor.postprocess = postprocess_with_candidate_capture
    predictor._sightbridge_candidate_capture_installed = True


def candidate_pretracker_status(
    candidate: dict[str, Any],
    standard_candidates: list[dict[str, Any]],
    confidence_threshold: float,
    iou_threshold: float,
) -> str:
    """Explain why a low-threshold classwise candidate is absent before tracking."""
    label = str(candidate["label"])
    bbox = candidate["bbox"]
    same_class_match = any(
        row["label"] == label and bbox_iou(bbox, row["bbox"]) >= 0.90
        for row in standard_candidates
    )
    if same_class_match:
        return "passed_confidence_and_nms"
    confidence = float(candidate["confidence"])
    if confidence < confidence_threshold:
        return "rejected_below_detector_confidence_threshold"
    competitors = [
        row for row in standard_candidates
        if row["label"] != label
        and float(row["confidence"]) >= confidence
        and bbox_iou(bbox, row["bbox"]) >= iou_threshold
    ]
    if competitors:
        competitor = max(competitors, key=lambda row: float(row["confidence"]))
        return f"likely_agnostic_nms_suppression_by_{competitor['label']}"
    return "not_in_standard_yolo_output_nms_or_max_det"


def best_candidate_match(
    candidate: dict[str, Any], rows: list[dict[str, Any]], min_iou: float = 0.25,
    *, same_class: bool = False,
) -> Optional[dict[str, Any]]:
    matches = [
        (bbox_iou(candidate["bbox"], row["bbox"]), row)
        for row in rows
        if (not same_class or row["label"] == candidate["label"])
    ]
    if not matches:
        return None
    overlap, row = max(matches, key=lambda item: item[0])
    return row if overlap >= min_iou else None


def diagnostic_candidate_tracker_match(
    candidate: dict[str, Any], tracked_rows: list[dict[str, Any]],
) -> Tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Match debug candidates by both class and geometry; expose geometry-only overlaps separately."""
    class_match = best_candidate_match(candidate, tracked_rows, same_class=True)
    if class_match is not None:
        return class_match, None
    geometry_only_match = best_candidate_match(candidate, tracked_rows)
    return None, geometry_only_match


def snapshot_detection_rows(detections: list[DetectionSnapshot]) -> list[dict[str, Any]]:
    return [
        {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox": detection.bbox,
            "track_id": detection.track_id if detection.has_track_id else None,
            "identity_uncertain": detection.identity_uncertain,
        }
        for detection in detections
    ]


@dataclass(frozen=True)
class DepthResult:
    frame_id: int
    captured_at: float
    depth_map: np.ndarray
    detections: Tuple[DetectionSnapshot, ...]
    latency_ms: float


class DepthInferenceWorker:
    """Run depth inference on a bounded latest-frame slot in a daemon thread."""

    def __init__(
        self, estimator: MetricDepthEstimator, benchmark: Optional["BenchmarkRecorder"], debug: bool = False
    ) -> None:
        self.estimator = estimator
        self.benchmark = benchmark
        self.debug = debug
        self.device = estimator.device
        self._cuda_stream = torch.cuda.Stream(device=self.device) if self.device == "cuda" else None
        if self._cuda_stream is not None:
            model_ready = torch.cuda.Event()
            model_ready.record(torch.cuda.current_stream(device=self.device))
            self._cuda_stream.wait_event(model_ready)
        self._condition = threading.Condition()
        self._pending: Optional[Tuple[int, float, np.ndarray, Tuple[DetectionSnapshot, ...]]] = None
        self._latest_result: Optional[DepthResult] = None
        self._stopping = False
        self._last_error: Optional[str] = None
        self._last_error_time = 0.0
        self._processing_frame_id: Optional[int] = None
        self._last_submitted_frame_id: Optional[int] = None
        self._last_consumed_frame_id = -1
        self._submitted_count = 0
        self._pending_replaced_count = 0
        self._completed_count = 0
        self._unconsumed_result_replaced_count = 0
        self._failure_count = 0
        self._thread = threading.Thread(target=self._run, name="SightBridgeDepth", daemon=True)
        self._thread.start()

    def submit(
        self,
        frame_id: int,
        captured_at: float,
        frame: np.ndarray,
        detections: Tuple[DetectionSnapshot, ...],
    ) -> bool:
        # Own this immutable snapshot; the main thread can continue processing its frame.
        frame_copy = frame.copy()
        with self._condition:
            if self._stopping:
                return False
            replaced_frame_id = self._pending[0] if self._pending is not None else None
            self._submitted_count += 1
            self._last_submitted_frame_id = frame_id
            if replaced_frame_id is not None:
                self._pending_replaced_count += 1
            self._pending = (frame_id, captured_at, frame_copy, detections)
            self._condition.notify()
        if replaced_frame_id is not None and self.debug:
            print(
                f"[DepthLifecycle] pending_job_replaced old_frame={replaced_frame_id} "
                f"new_frame={frame_id} total_replaced={self._pending_replaced_count}"
            )
        return True

    def latest_result(self) -> Optional[DepthResult]:
        with self._condition:
            return self._latest_result

    def mark_consumed(self, frame_id: int) -> None:
        with self._condition:
            self._last_consumed_frame_id = max(self._last_consumed_frame_id, frame_id)

    def status_snapshot(self) -> dict[str, Any]:
        """Return worker lifecycle state without waiting for inference or taking the GPU lock."""
        with self._condition:
            return {
                "pending_frame_id": self._pending[0] if self._pending is not None else None,
                "processing_frame_id": self._processing_frame_id,
                "latest_completed_frame_id": self._latest_result.frame_id if self._latest_result else None,
                "last_submitted_frame_id": self._last_submitted_frame_id,
                "last_consumed_frame_id": self._last_consumed_frame_id,
                "submitted": self._submitted_count,
                "pending_replaced": self._pending_replaced_count,
                "completed": self._completed_count,
                "unconsumed_result_replaced": self._unconsumed_result_replaced_count,
                "failures": self._failure_count,
                "last_error": self._last_error,
                "stopping": self._stopping,
            }

    def stop(self, timeout: float = 3.0) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("[Depth] Worker is still finishing inference; it will exit with the process.")

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job = self._pending
                self._pending = None
                if job is not None:
                    self._processing_frame_id = job[0]

            if job is None:
                continue
            frame_id, captured_at, frame, detections = job
            measure = self.benchmark is not None and self.benchmark.is_measuring(time.perf_counter())
            try:
                if measure and self._cuda_stream is not None:
                    self._cuda_stream.synchronize()
                started_perf = time.perf_counter()
                measure = measure and self.benchmark is not None and self.benchmark.is_measuring(started_perf)
                if self._cuda_stream is not None:
                    with torch.cuda.stream(self._cuda_stream):
                        depth_map = self.estimator.predict(frame)
                else:
                    depth_map = self.estimator.predict(frame)
                if measure and self._cuda_stream is not None:
                    self._cuda_stream.synchronize()
                completed_perf = time.perf_counter()
                latency_ms = (completed_perf - started_perf) * 1000.0
                if depth_map is None:
                    raise RuntimeError("Empty depth output")
                depth_map.setflags(write=False)
                if self.benchmark is not None:
                    self.benchmark.record_depth_success(latency_ms, started_perf, completed_perf)
                result = DepthResult(frame_id, captured_at, depth_map, detections, latency_ms)
                with self._condition:
                    if not self._stopping and (self._latest_result is None or frame_id > self._latest_result.frame_id):
                        if (self._latest_result is not None
                                and self._latest_result.frame_id > self._last_consumed_frame_id):
                            self._unconsumed_result_replaced_count += 1
                        self._latest_result = result
                        self._completed_count += 1
                    self._last_error = None
            except Exception as exc:
                message = str(exc)
                now = time.monotonic()
                with self._condition:
                    self._failure_count += 1
                    should_log = message != self._last_error or now - self._last_error_time >= 5.0
                    self._last_error = message
                    self._last_error_time = now
                if should_log:
                    print(f"[Depth] Background inference failed: {exc}")
            finally:
                with self._condition:
                    self._processing_frame_id = None


def smooth_depth_measurement(
    raw_history: Deque[Tuple[float, float]],
    raw_distance: float,
    captured_at: float,
    previous_state: Optional[DistanceState],
    max_depth_m: float,
    reset_after_seconds: float = TRACK_STATE_RESET_SECONDS,
) -> Tuple[float, str]:
    """Use a short median window before EMA so one raw spike cannot dominate."""
    if previous_state is not None and captured_at - previous_state.captured_at > reset_after_seconds:
        raw_history.clear()
        previous_state = None
    median_window_seconds = max(1.0, reset_after_seconds)
    recent = [distance for stamp, distance in raw_history if 0.0 <= captured_at - stamp <= median_window_seconds]
    previous_center = float(np.median(recent[-2:])) if recent else None
    candidates = (recent[-2:] + [raw_distance])[-3:]
    robust_sample = float(np.median(candidates))
    if previous_state is None or not is_valid_distance(previous_state.distance_m, max_depth_m):
        filtered = robust_sample
    else:
        filtered = (1.0 - DEPTH_EMA_ALPHA) * previous_state.distance_m + DEPTH_EMA_ALPHA * robust_sample
    reason = "accepted"
    if previous_center is not None and len(recent) >= 2:
        spike_threshold = max(DEPTH_SPIKE_MIN_M, previous_center * DEPTH_SPIKE_RATIO)
        if abs(raw_distance - previous_center) > spike_threshold and abs(robust_sample - previous_center) < spike_threshold:
            reason = "isolated_spike_median_filtered"
    raw_history.append((captured_at, raw_distance))
    return filtered, reason


def confirm_depth_change(
    raw_history: Deque[Tuple[float, float]],
    raw_distance: float,
    captured_at: float,
    pending_candidate: Optional[Tuple[float, float]],
    window_seconds: float = DEPTH_CHANGE_CONFIRM_WINDOW_SECONDS,
) -> Tuple[bool, Optional[Tuple[float, float]], str]:
    """Require a second nearby sample before accepting a large one-frame depth jump."""
    recent = [
        distance for stamp, distance in raw_history
        if 0.0 <= captured_at - stamp <= max(1.5, window_seconds)
    ]
    if len(recent) < 2:
        return True, None, "accepted_no_baseline"
    baseline = float(np.median(recent[-2:]))
    jump_limit = max(0.55, baseline * 0.40)
    if abs(raw_distance - baseline) <= jump_limit:
        return True, None, "accepted"
    if pending_candidate is not None:
        pending_at, pending_distance = pending_candidate
        candidate_tolerance = max(0.35, raw_distance * 0.25)
        if (0.0 <= captured_at - pending_at <= window_seconds
                and abs(raw_distance - pending_distance) <= candidate_tolerance):
            return True, None, "sustained_change_confirmed"
    return False, (captured_at, raw_distance), "depth_change_pending_confirmation"


def close_measurement_confirmed(
    raw_history: Deque[Tuple[float, float]],
    now: float,
    threshold_m: float = CLOSE_OBSTACLE_DISTANCE_M,
    window_seconds: float = 1.5,
) -> bool:
    """Close status needs two distinct, recent depth samples inside the threshold."""
    recent_close = [
        (stamp, distance) for stamp, distance in raw_history
        if 0.0 <= now - stamp <= window_seconds and 0.0 < distance <= threshold_m
    ]
    if len(recent_close) < 2 or recent_close[-1][0] - recent_close[0][0] < 0.15:
        return False
    pair = [distance for _stamp, distance in recent_close[-2:]]
    center = float(np.median(pair))
    return max(pair) - min(pair) <= max(0.30, center * 0.25)


def motion_from_history(history: Deque[Tuple[float, float]]) -> str:
    """Estimate camera-relative depth trend only when both time windows have support."""
    if len(history) < 5:
        return "unknown"
    now = history[-1][0]
    old = [(t, d) for t, d in history if 1.1 <= now - t <= 3.5]
    recent = [(t, d) for t, d in history if 0.0 <= now - t <= 0.8]
    if len(old) < 2 or len(recent) < 2:
        return "unknown"
    if recent[-1][0] - old[0][0] < 1.2:
        return "unknown"
    old_distance = float(np.median([distance for _, distance in old]))
    recent_distance = float(np.median([distance for _, distance in recent]))
    change = old_distance - recent_distance
    threshold = max(APPROACH_MIN_CHANGE_M, old_distance * APPROACH_CHANGE_RATIO)
    if change >= threshold:
        return "approaching"
    if change <= -threshold:
        return "receding"
    return "stationary"


@dataclass(frozen=True)
class AnnouncementState:
    direction: str
    announced_at: float
    motion: str
    distance_m: Optional[float]
    label: str = ""


def ordinary_announcement_suppression_reason(
    last_ordinary_at: Optional[float], now: float, *, close_confirmed: bool,
    cooldown_seconds: float = ORDINARY_ANNOUNCEMENT_GLOBAL_COOLDOWN_SECONDS,
) -> Optional[str]:
    """Apply scene-wide pacing while allowing confirmed close warnings through."""
    if close_confirmed:
        return None
    if last_ordinary_at is not None and now - last_ordinary_at < cooldown_seconds:
        return "global_cooldown"
    return None


def announcement_candidate_key(
    detection: DetectionSnapshot,
    distance_m: Optional[float],
    last_announced_at: Optional[float],
    eligible: bool,
) -> Tuple[bool, bool, float, float, float]:
    """Order by eligibility, half-metre proximity bands, fair turn, then confidence."""
    return (
        not eligible,
        distance_m is None,
        int(distance_m / DISTANCE_CHANGE_MIN_M) if distance_m is not None else 10**9,
        last_announced_at if last_announced_at is not None else -1.0,
        -detection.confidence,
    )


def announcement_decision(
    previous: Optional[AnnouncementState],
    direction: str,
    distance_m: Optional[float],
    motion: str,
    now: float,
    cooldown_seconds: float = SPEECH_COOLDOWN_SECONDS,
    close_threshold_m: float = CLOSE_OBSTACLE_DISTANCE_M,
    label: Optional[str] = None,
    close_confirmed: bool = True,
) -> Tuple[bool, int, str]:
    """Return should-speak, priority, and reason for a tracked object's state."""
    if previous is None:
        reason = "new_object"
    else:
        # Fresh distance becoming available is useful; a stale-distance transition to
        # unavailable is intentionally silent to prevent repeated warnings.
        availability_changed = previous.distance_m is None and distance_m is not None
        close_transition = (
            close_confirmed and distance_m is not None and distance_m <= close_threshold_m
            and (previous.distance_m is None or previous.distance_m > close_threshold_m)
        )
        distance_changed = (
            previous.distance_m is not None and distance_m is not None
            and abs(distance_m - previous.distance_m) >= max(
                DISTANCE_CHANGE_MIN_M, previous.distance_m * DISTANCE_CHANGE_RATIO
            )
        )
        direction_changed = direction != previous.direction
        label_changed = bool(label and previous.label and label != previous.label)
        motion_changed_to_approaching = motion == "approaching" and previous.motion != "approaching"
        motion_changed = motion in ("approaching", "receding", "stationary") and motion != previous.motion
        critical = close_transition or (
            close_confirmed and distance_m is not None
            and distance_m <= close_threshold_m and motion_changed_to_approaching
        )
        if availability_changed:
            reason = "distance_availability_changed"
        elif close_transition:
            reason = "close_obstacle"
        elif motion_changed_to_approaching:
            reason = "approaching"
        elif now - previous.announced_at >= cooldown_seconds and motion_changed:
            reason = f"motion_{motion}"
        elif now - previous.announced_at >= cooldown_seconds and label_changed:
            reason = "class_changed"
        elif now - previous.announced_at >= cooldown_seconds and direction_changed:
            reason = "direction_changed"
        elif now - previous.announced_at >= cooldown_seconds and distance_changed:
            reason = "distance_changed"
        else:
            return False, 0, "unchanged_or_cooldown"
        if critical:
            return True, 3, reason
        if reason == "approaching":
            return True, 2, reason
    if close_confirmed and distance_m is not None and distance_m <= close_threshold_m:
        return True, 3, reason
    if motion == "approaching":
        return True, 2, reason
    return True, 1, reason


def readable_distance(distance_m: Optional[float], max_depth_m: float) -> str:
    if not is_valid_distance(distance_m, max_depth_m):
        return "distance unavailable"
    if distance_m < 10:
        return f"approximately {distance_m:.1f} metres"
    return f"approximately {int(round(distance_m / 5.0) * 5)} metres"


def assert_distance_output_cases() -> None:
    """Guard shared overlay/speech handling for missing, invalid, fresh, and stale data."""
    now = 10.0
    unavailable = "Distance unavailable"
    assert fresh_track_distance(None, now, MAX_DEPTH_RESULT_AGE_SECONDS, 20.0) is None
    assert format_distance(None, 20.0) == unavailable
    assert readable_distance(None, 20.0) == "distance unavailable"

    negative = DistanceState(-1.0, now)
    assert fresh_track_distance(negative, now, MAX_DEPTH_RESULT_AGE_SECONDS, 20.0) is None
    assert format_distance(-1.0, 20.0) == unavailable
    assert readable_distance(-1.0, 20.0) == "distance unavailable"

    fresh = DistanceState(1.56, now - 0.1)
    resolved = fresh_track_distance(fresh, now, MAX_DEPTH_RESULT_AGE_SECONDS, 20.0)
    assert resolved == 1.56
    assert "1.56 m" in format_distance(resolved, 20.0)
    assert readable_distance(resolved, 20.0) == "approximately 1.6 metres"

    stale = DistanceState(1.56, now - MAX_DEPTH_RESULT_AGE_SECONDS - 0.01)
    stale_distance = fresh_track_distance(stale, now, MAX_DEPTH_RESULT_AGE_SECONDS, 20.0)
    assert stale_distance is None
    assert format_distance(stale_distance, 20.0) == unavailable
    assert readable_distance(stale_distance, 20.0) == "distance unavailable"


class BenchmarkRecorder:
    """Collect a warm-up and fixed-duration performance sample for one run."""

    WARMUP_SECONDS = 10.0
    MEASUREMENT_SECONDS = 60.0

    def __init__(self, device: str, use_fp16: bool, depth_environment: str, depth_model: str) -> None:
        self.device = device
        self.precision = "FP16" if use_fp16 else "FP32"
        self.depth_environment = depth_environment
        self.depth_model = depth_model
        self.run_started_perf = time.perf_counter()
        self.run_started_at = datetime.now().astimezone()
        self.warmup_end_perf = self.run_started_perf + self.WARMUP_SECONDS
        self.measurement_start_perf: Optional[float] = None
        self.measurement_end_perf: Optional[float] = None
        self.measurement_started_at: Optional[datetime] = None
        self.frame_count = 0
        self.fps_samples: list[float] = []
        self.yolo_latencies: list[float] = []
        self.depth_latencies: list[float] = []
        self.successful_depth_inferences = 0
        self.depth_jobs_submitted = 0
        self.peak_gpu_memory_mb: Optional[float] = None
        self.memory_sampled = False
        self.window_complete_reported = False
        self.lock = threading.Lock()
        self.finalized = False

    def _start_measurement(self, now_perf: float) -> None:
        self.measurement_start_perf = now_perf
        self.measurement_end_perf = now_perf + self.MEASUREMENT_SECONDS
        self.measurement_started_at = self.run_started_at + timedelta(seconds=now_perf - self.run_started_perf)
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                print(f"[Benchmark] GPU memory tracking unavailable: {exc}")

    def _sample_peak_gpu_memory(self) -> None:
        if self.memory_sampled:
            return
        self.memory_sampled = True
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception as exc:
                print(f"[Benchmark] GPU memory tracking unavailable: {exc}")

    def record_frame(self, now_perf: float, previous_frame_perf: float) -> None:
        with self.lock:
            if self.finalized:
                return
            if self.measurement_start_perf is None and now_perf >= self.warmup_end_perf:
                self._start_measurement(now_perf)
                print("[Benchmark] Warm-up complete; collecting for 60 seconds.")
            if self.measurement_start_perf is None or self.measurement_end_perf is None:
                return
            if now_perf >= self.measurement_end_perf:
                self._sample_peak_gpu_memory()
                if not self.window_complete_reported:
                    print("[Benchmark] Measurement window complete; results will be finalized when the application exits.")
                    self.window_complete_reported = True
                return
            self.frame_count += 1
            if previous_frame_perf >= self.measurement_start_perf:
                elapsed = now_perf - previous_frame_perf
                if elapsed > 0:
                    self.fps_samples.append(1.0 / elapsed)

    def is_measuring(self, now_perf: float) -> bool:
        with self.lock:
            return self._is_measuring_unlocked(now_perf)

    def _is_measuring_unlocked(self, now_perf: float) -> bool:
        return self.measurement_start_perf is not None and self.measurement_end_perf is not None and self.measurement_start_perf <= now_perf < self.measurement_end_perf

    def record_depth_submission(self, submitted_perf: float) -> None:
        with self.lock:
            if not self.finalized and self._is_measuring_unlocked(submitted_perf):
                self.depth_jobs_submitted += 1

    def record_yolo_latency(self, latency_ms: float, completed_perf: float) -> None:
        with self.lock:
            if not self.finalized and self._is_measuring_unlocked(completed_perf):
                self.yolo_latencies.append(latency_ms)

    def record_depth_success(self, latency_ms: float, started_perf: float, completed_perf: float) -> None:
        with self.lock:
            if not self.finalized and self._is_measuring_unlocked(started_perf) and self._is_measuring_unlocked(completed_perf):
                self.depth_latencies.append(latency_ms)
                self.successful_depth_inferences += 1

    @staticmethod
    def _average(values: list[float]) -> Optional[float]:
        return statistics.fmean(values) if values else None

    @staticmethod
    def _median(values: list[float]) -> Optional[float]:
        return statistics.median(values) if values else None

    def finish(self, now_perf: float) -> dict[str, object]:
        with self.lock:
            self.finalized = True
            complete = self.measurement_end_perf is not None and now_perf >= self.measurement_end_perf
            measurement_seconds = 0.0
            if self.measurement_start_perf is not None:
                end_perf = min(now_perf, self.measurement_end_perf or now_perf)
                measurement_seconds = max(0.0, end_perf - self.measurement_start_perf)
            ended_at = self.run_started_at + timedelta(seconds=now_perf - self.run_started_perf)
            if complete and self.measurement_end_perf is not None:
                ended_at = self.run_started_at + timedelta(seconds=self.measurement_end_perf - self.run_started_perf)
            if self.measurement_start_perf is not None:
                self._sample_peak_gpu_memory()

            fps_avg = self._average(self.fps_samples)
            yolo_avg = self._average(self.yolo_latencies)
            depth_avg = self._average(self.depth_latencies)
            return {
                "benchmark_started_at": self.run_started_at.isoformat(timespec="seconds"),
                "measurement_started_at": self.measurement_started_at.isoformat(timespec="seconds") if self.measurement_started_at else "",
                "measurement_ended_at": ended_at.isoformat(timespec="seconds"),
                "precision_mode": self.precision,
                "depth_environment": self.depth_environment,
                "depth_model": self.depth_model,
                "status": "COMPLETE" if complete else "INCOMPLETE",
                "measurement_seconds": round(measurement_seconds, 3),
                "frames_processed": self.frame_count,
                "average_fps": round(fps_avg, 3) if fps_avg is not None else "",
                "minimum_fps": round(min(self.fps_samples), 3) if self.fps_samples else "",
                "maximum_fps": round(max(self.fps_samples), 3) if self.fps_samples else "",
                "yolo_average_latency_ms": round(yolo_avg, 3) if yolo_avg is not None else "",
                "yolo_median_latency_ms": round(self._median(self.yolo_latencies), 3) if self.yolo_latencies else "",
                "depth_average_latency_ms": round(depth_avg, 3) if depth_avg is not None else "",
                "depth_median_latency_ms": round(self._median(self.depth_latencies), 3) if self.depth_latencies else "",
                "depth_jobs_submitted": self.depth_jobs_submitted,
                "successful_depth_inferences": self.successful_depth_inferences,
                "successful_depth_inferences_per_second": round(self.successful_depth_inferences / measurement_seconds, 3) if measurement_seconds > 0 else "",
                "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 1) if self.peak_gpu_memory_mb is not None else "",
            }

    @staticmethod
    def save_and_print(results: dict[str, object]) -> None:
        output_path = Path(__file__).resolve().with_name("benchmark_results.csv")
        try:
            current_fields = list(results)
            if not output_path.exists() or output_path.stat().st_size == 0:
                with output_path.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=current_fields)
                    writer.writeheader()
                    writer.writerow(results)
            else:
                with output_path.open("r", newline="", encoding="utf-8") as input_file:
                    reader = csv.DictReader(input_file)
                    existing_fields = reader.fieldnames or []
                    existing_rows = list(reader)
                fieldnames = existing_fields + [field for field in current_fields if field not in existing_fields]
                if fieldnames != existing_fields:
                    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                    with temp_path.open("w", newline="", encoding="utf-8") as output_file:
                        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(existing_rows)
                        writer.writerow(results)
                    temp_path.replace(output_path)
                else:
                    with output_path.open("a", newline="", encoding="utf-8") as output_file:
                        csv.DictWriter(output_file, fieldnames=existing_fields).writerow(results)
            print(f"[Benchmark] Results appended to {output_path}")
        except OSError as exc:
            print(f"[Benchmark] Could not save CSV results: {exc}")

        print("\n[Benchmark] Final summary")
        for key, value in results.items():
            print(f"  {key}: {value if value != '' else 'n/a'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SightBridge spatial-awareness proof of concept")
    parser.add_argument("--source", default="0", help="Camera index (default 0) or a video file path")
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Detector floor; low-confidence boxes are passed to ByteTrack for recovery"
    )
    parser.add_argument("--depth-every", type=int, default=6, help="Run the depth model every N frames; raise if too slow")
    parser.add_argument("--depth-environment", choices=("indoor", "outdoor"), default="indoor",
                        help="Select the metric depth checkpoint for indoor or outdoor scenes")
    parser.add_argument("--depth-diagnostics", action="store_true",
                        help="Log throttled per-object depth sampling diagnostics")
    parser.add_argument("--speech-debug", action="store_true",
                        help="Log throttled speech decisions and speech-worker activity")
    parser.add_argument("--speech-test", action="store_true",
                        help="Run six sequential Windows speech checks without camera or model initialization")
    parser.add_argument("--vision-debug", action="store_true",
                        help="Log YOLO candidates and their path through tracking, filtering, and overlay")
    parser.add_argument("--no-audio", action="store_true", help="Disable spoken announcements")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 inference and use FP32 for comparison")
    parser.add_argument("--benchmark", action="store_true", help="Measure performance after a 10-second warm-up")
    args = parser.parse_args()
    debug_log = start_debug_log_if_enabled(
        args.vision_debug,
        Path(__file__).resolve().parent / "logs" / "sightbridge_debug.txt",
    )
    if debug_log is not None:
        print(f"[DebugLog] Writing this session's output to {debug_log.path}")
    if args.speech_test:
        try:
            return run_speech_diagnostic()
        finally:
            if debug_log is not None:
                debug_log.close()
    assert_distance_output_cases()

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda" and not args.fp32
    print(f"[System] Torch device: {device}")
    print(f"[YOLO] Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(
        f"[Tracking] Detector floor={args.conf:.2f}; ByteTrack recovery={BYTE_TRACK_LOW_CONFIDENCE:.2f}, "
        f"high-stage={BYTE_TRACK_HIGH_CONFIDENCE:.2f}, new-track={BYTE_TRACK_NEW_TRACK_CONFIDENCE:.2f}; "
        f"config={TRACKER_CONFIG.name}"
    )
    if device == "cpu":
        print("[Performance] CPU mode may be slow because object detection and depth estimation both use deep learning.")

    selected_depth_model, selected_max_depth = DEPTH_MODELS[args.depth_environment]
    print(
        f"[Depth] Selected environment: {args.depth_environment}; model: {selected_depth_model}; "
        f"maximum range: {selected_max_depth:.0f} m"
    )
    print(f"[YOLO] Loading object detector: {YOLO_MODEL}")
    detector = YOLO(YOLO_MODEL)  # downloads official small weights on first run
    if args.vision_debug:
        detector.add_callback(
            "on_predict_start",
            lambda predictor: install_vision_debug_candidate_capture(predictor),
        )
    depth_estimator = MetricDepthEstimator(device, use_fp16, args.depth_environment)
    speech = SpeechWorker(enabled=not args.no_audio, debug=args.speech_debug)
    if speech.enabled and speech.ready:
        speech.say("SightBridge voice assistance is enabled.", priority=1, key="startup")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Camera] Could not open source {source!r}. Try --source 0 or check the camera/video file.")
        speech.stop(timeout=3.0)
        if debug_log is not None:
            debug_log.close()
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    histories: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=24))
    raw_histories: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=3))
    pending_depth_by_track: Dict[int, Tuple[float, float]] = {}
    depth_diagnostic_by_track: Dict[int, dict[str, Any]] = {}
    last_unavailable_diag_by_track: Dict[int, Tuple[str, float]] = {}
    distance_state_by_track: Dict[int, DistanceState] = {}
    motion_by_track: Dict[int, str] = {}
    direction_by_track: Dict[int, str] = {}
    vision_state_by_track: Dict[int, VisionTrackState] = {}
    last_seen_by_track: Dict[int, float] = {}
    last_announced: Dict[int, AnnouncementState] = {}
    speech_live_lock = threading.Lock()
    speech_live_by_track: Dict[int, SpeechLiveState] = {}

    def validate_speech_job(job: SpeechJob) -> Optional[str]:
        token = job.validation_token
        if token is None:
            return None
        with speech_live_lock:
            live_state = speech_live_by_track.get(token.track_id)
        return validate_pending_announcement(
            token, live_state, time.perf_counter(), MAX_DEPTH_RESULT_AGE_SECONDS,
            MAX_DEPTH_RESULT_AGE_SECONDS, depth_estimator.max_depth_m,
        )

    speech.set_job_validator(validate_speech_job)
    previous_tracker_lost_ids: set[int] = set()
    previous_tracker_removed_ids: set[int] = set()
    previous_ambiguous_person_ids: set[int] = set()
    frame_no = 0
    last_consumed_depth_frame_id = -1
    depth_latency_ms = 0.0
    detect_latency_ms = 0.0
    depth_diagnostics_pending: list[str] = []
    last_depth_diagnostics_print = time.monotonic()
    last_depth_lifecycle_print = time.monotonic()
    last_speech_debug_print = time.monotonic()
    last_vision_debug_print = time.monotonic()
    last_ordinary_announcement_at: Optional[float] = None
    last_announcement_suppression_log: Dict[Tuple[int, str], float] = {}
    pending_vision_debug_records: Deque[str] = deque(maxlen=10000)
    last_no_book_candidate_print = 0.0
    speech_ready_last = speech.ready
    fps_ema = 0.0
    last_frame_time = time.perf_counter()
    paused = False
    running = True
    banner = "MODEL STARTUP: loading weights; first load can take several minutes"
    benchmark = BenchmarkRecorder(device, use_fp16, depth_estimator.environment, depth_estimator.model_id) if args.benchmark else None
    depth_worker = DepthInferenceWorker(depth_estimator, benchmark, debug=args.vision_debug)

    print("\n[Controls] Q = quit | V = toggle voice | P = pause/resume camera analysis")
    print("[Safety] Prototype only. Do not use it for real-world navigation or collision avoidance.\n")

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                print("[Camera] No more frames or camera read failed.")
                break
            frame_no += 1
            previous_frame_perf = last_frame_time
            now_perf = time.perf_counter()
            instant_fps = 1.0 / max(0.001, now_perf - last_frame_time)
            fps_ema = instant_fps if fps_ema == 0 else (0.85 * fps_ema + 0.15 * instant_fps)
            last_frame_time = now_perf
            if benchmark is not None:
                benchmark.record_frame(now_perf, previous_frame_perf)
            h, w = frame.shape[:2]
            speech_ready_now = speech.ready
            if speech_ready_now and not speech_ready_last and speech.enabled:
                last_announced.clear()
                speech.say("SightBridge voice assistance is ready again.", priority=2, key="startup")
            speech_ready_last = speech_ready_now
            for dropped_key in speech.consume_dropped_keys():
                if dropped_key.startswith("track:"):
                    try:
                        last_announced.pop(int(dropped_key.split(":", 1)[1]), None)
                    except ValueError:
                        pass
            if paused:
                cv2.putText(frame, "ANALYSIS PAUSED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 190, 255), 2, cv2.LINE_AA)
                cv2.imshow("SightBridge | Spatial Awareness Demo", frame)
                key = cv2.waitKey(20) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('p'):
                    paused = False
                if key == ord('v'):
                    speech.set_enabled(not speech.enabled)
                    if speech.enabled:
                        last_announced.clear()
                        speech.say("Voice feedback enabled.", force=True, priority=1, key="startup")
                continue

            measure_yolo = benchmark is not None and benchmark.is_measuring(time.perf_counter())
            if measure_yolo and device == "cuda":
                torch.cuda.current_stream(device=device).synchronize()
            start_detect = time.perf_counter()
            measure_yolo = measure_yolo and benchmark is not None and benchmark.is_measuring(start_detect)
            result = detector.track(
                frame, persist=True, tracker=str(TRACKER_CONFIG), imgsz=640,
                conf=args.conf, iou=0.70, agnostic_nms=True,
                device=device, quantize=16 if use_fp16 else 32, verbose=False,
            )[0]
            candidate_snapshot = (
                getattr(detector.predictor, "_sightbridge_candidate_snapshot", {})
                if args.vision_debug and detector.predictor is not None else {}
            )
            tracked_yolo_rows = yolo_result_rows([result]) if args.vision_debug else []
            if measure_yolo and device == "cuda":
                torch.cuda.current_stream(device=device).synchronize()
            detect_latency_ms = (time.perf_counter() - start_detect) * 1000.0
            if measure_yolo and benchmark is not None:
                benchmark.record_yolo_latency(detect_latency_ms, time.perf_counter())

            current_detections: list[DetectionSnapshot] = []
            vision_debug_records: list[str] = []
            lifecycle_records_by_track: Dict[int, str] = {}
            trackers = getattr(getattr(detector, "predictor", None), "trackers", None) or []
            tracker_state = trackers[0] if trackers else None
            current_lost = (
                {int(track.track_id) for track in getattr(tracker_state, "lost_stracks", ())}
                if tracker_state is not None else set()
            )
            removed_this_frame = (
                getattr(tracker_state, "removed_stracks_frame", None)
                if tracker_state is not None else None
            )
            tracker_lifecycle_available = removed_this_frame is not None
            current_removed = (
                {int(track.track_id) for track in removed_this_frame}
                if tracker_lifecycle_available else set()
            )
            if args.vision_debug and tracker_state is not None:
                for lost_id in sorted(current_lost - previous_tracker_lost_ids):
                    vision_debug_records.append(
                        f"frame={frame_no} bytetrack_lost id={lost_id} "
                        f"buffer_frames={getattr(tracker_state, 'max_frames_lost', 'unknown')}"
                    )
                for removed_id in sorted(current_removed - previous_tracker_removed_ids):
                    vision_debug_records.append(
                        f"frame={frame_no} bytetrack_expired_or_removed id={removed_id} "
                        f"buffer_frames={getattr(tracker_state, 'max_frames_lost', 'unknown')}"
                    )
                previous_tracker_lost_ids = current_lost
                previous_tracker_removed_ids = current_removed
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
                cls_values = boxes.cls.detach().cpu().numpy().astype(int)
                conf_values = boxes.conf.detach().cpu().numpy()
                has_track_ids = boxes.id is not None
                ids = boxes.id.detach().cpu().numpy().astype(int) if has_track_ids else np.arange(len(xyxy)) + 10000
                for box_arr, class_id, conf, track_id in zip(xyxy, cls_values, conf_values, ids):
                    x1, y1, x2, y2 = [int(v) for v in box_arr]
                    x1, x2 = max(0, x1), min(w - 1, x2)
                    y1, y2 = max(0, y1), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    label = detector.names.get(int(class_id), str(class_id))
                    bbox = (x1, y1, x2, y2)
                    current_detections.append(
                        DetectionSnapshot(
                            int(track_id), has_track_ids, label, float(conf), "ahead", bbox,
                            source_frame_id=frame_no,
                        )
                    )

            converted_detections = list(current_detections)
            stabilized_detections: list[DetectionSnapshot] = []
            for detection in current_detections:
                if detection.has_track_id:
                    state = vision_state_by_track.setdefault(detection.track_id, VisionTrackState())
                    previous_seen = last_seen_by_track.get(detection.track_id)
                    reacquire_gap = now_perf - previous_seen if previous_seen is not None else None
                    long_reacquisition = reacquisition_needs_state_reset(
                        state, previous_seen, now_perf,
                    )
                    if long_reacquisition:
                        state = reset_reacquired_track_state(
                            detection.track_id,
                            last_seen_by_track=last_seen_by_track,
                            direction_by_track=direction_by_track,
                            vision_state_by_track=vision_state_by_track,
                            distance_state_by_track=distance_state_by_track,
                            histories=histories,
                            raw_histories=raw_histories,
                            pending_depth_by_track=pending_depth_by_track,
                            depth_diagnostic_by_track=depth_diagnostic_by_track,
                            motion_by_track=motion_by_track,
                            last_announced=last_announced,
                        )
                    if state.first_seen_at is None:
                        if previous_seen is None:
                            lifecycle_records_by_track[detection.track_id] = (
                                f"app_track_state_initialized id={detection.track_id} "
                                f"reported_conf={detection.confidence:.2f} "
                                f"reported_score_stage={confidence_stage(detection.confidence)} "
                                "tracker_birth_confidence=not_exposed_by_ultralytics "
                                "association_score=not_exposed_by_ultralytics"
                            )
                            possible_old = possible_fragmented_track(
                                detection, now_perf, vision_state_by_track, last_seen_by_track
                            )
                            if possible_old is not None:
                                old_id, overlap, age = possible_old
                                lifecycle_records_by_track[detection.track_id] += (
                                    f" possible_id_change_from={old_id} overlap={overlap:.3f} "
                                    f"previous_age_ms={age * 1000.0:.0f}"
                                )
                        elif reacquire_gap is not None and reacquire_gap > 0.20:
                            lifecycle_records_by_track[detection.track_id] = (
                                f"track_reacquired id={detection.track_id} "
                                f"gap_ms={reacquire_gap * 1000.0:.0f} "
                                f"conf={detection.confidence:.2f} "
                                f"application_state_reset={'yes' if long_reacquisition else 'no'}"
                            )
                        state.first_seen_at = now_perf
                    elif (reacquire_gap is not None and reacquire_gap > 0.20):
                        lifecycle_records_by_track[detection.track_id] = (
                            f"track_reacquired id={detection.track_id} "
                            f"gap_ms={reacquire_gap * 1000.0:.0f} "
                            f"conf={detection.confidence:.2f} application_state_reset=no"
                        )
                    detection = stabilize_detection(detection, state, w, observed_at=now_perf)
                else:
                    x1, _y1, x2, _y2 = detection.bbox
                    detection = replace(detection, direction=direction_for_box(x1, x2, w))
                stabilized_detections.append(detection)
            current_detections = stabilized_detections
            stabilized_detection_rows = snapshot_detection_rows(current_detections)

            current_detections, duplicate_detections = suppress_duplicate_detections(current_detections)
            current_ambiguous_person_ids: set[int] = set()
            for index, detection in enumerate(current_detections):
                ambiguity = find_person_occlusion_ambiguity(
                    detection, current_detections, vision_state_by_track,
                    last_seen_by_track, now_perf,
                )
                if ambiguity is not None:
                    current_ambiguous_person_ids.add(detection.track_id)
                    current_detections[index] = replace(
                        detection,
                        identity_uncertain=True,
                        possible_parent_track_id=ambiguity.possible_parent_track_id,
                    )
                    if detection.track_id not in previous_ambiguous_person_ids:
                        if args.vision_debug:
                            print(
                                f"[VisionDiag] frame={frame_no} track={detection.track_id} "
                                f"person_identity_uncertain "
                                f"possible_parent={ambiguity.possible_parent_track_id} "
                                f"occluder={ambiguity.occluder_track_id} "
                                f"containment={ambiguity.containment:.3f} "
                                f"parent_age_ms={ambiguity.parent_age_seconds * 1000.0:.0f}; "
                                "IDs_and_track_state_not_merged; speech_withheld"
                            )
            (
                current_ambiguous_person_ids,
                _newly_uncertain_ids,
                cleared_uncertainty_ids,
            ) = reconcile_person_uncertainty(
                previous_ambiguous_person_ids,
                current_ambiguous_person_ids,
                vision_state_by_track,
            )
            for resolved_id in sorted(cleared_uncertainty_ids):
                if args.vision_debug:
                    print(
                        f"[VisionDiag] frame={frame_no} track={resolved_id} "
                        "person_identity_uncertainty_cleared; announcement_evidence_reset"
                    )
            previous_ambiguous_person_ids = current_ambiguous_person_ids
            for detection in current_detections:
                if detection.has_track_id:
                    direction_by_track[detection.track_id] = detection.direction
                    last_seen_by_track[detection.track_id] = now_perf
                    lifecycle = lifecycle_records_by_track.get(detection.track_id)
                    if args.vision_debug and lifecycle:
                        vision_debug_records.append(f"frame={frame_no} {lifecycle}")
                if args.vision_debug:
                    vision_debug_records.append(
                        f"frame={frame_no} track={detection.track_id if detection.has_track_id else 'untracked'} "
                        f"class={detection.label} conf={detection.confidence:.2f} "
                        f"score_stage={confidence_stage(detection.confidence)} bbox={detection.bbox} "
                        f"direction={detection.direction} "
                        f"identity={'uncertain' if detection.identity_uncertain else 'tracked'} "
                        f"possible_parent={detection.possible_parent_track_id if detection.possible_parent_track_id is not None else 'n/a'}"
                    )

            if args.vision_debug:
                raw_candidates = candidate_snapshot.get("candidates", [])
                standard_candidates = candidate_snapshot.get("pretracker", [])
                relevant_candidates = [
                    row for row in raw_candidates if str(row["label"]).lower() in {"person", "book"}
                ]
                relevant_people = [
                    row for row in relevant_candidates if str(row["label"]).lower() == "person"
                ]
                for row in raw_candidates:
                    if str(row["label"]).lower() in {"person", "book"} or row["confidence"] < 0.05:
                        continue
                    cx1, cy1, cx2, cy2 = row["bbox"]
                    candidate_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
                    if candidate_area == 0:
                        continue
                    for person in relevant_people:
                        px1, py1, px2, py2 = person["bbox"]
                        overlap = max(0, min(cx2, px2) - max(cx1, px1)) * max(
                            0, min(cy2, py2) - max(cy1, py1)
                        )
                        if overlap / candidate_area >= 0.50:
                            relevant_candidates.append(row)
                            break
                book_candidates = [
                    row for row in relevant_candidates if str(row["label"]).lower() == "book"
                ]
                if candidate_snapshot.get("capture_error"):
                    vision_debug_records.append(
                        f"frame={frame_no} yolo_candidate_capture_error="
                        f"{candidate_snapshot['capture_error']}"
                    )
                if not book_candidates and time.monotonic() - last_no_book_candidate_print >= 1.0:
                    person_summary = [
                        f"conf={row['confidence']:.3f} bbox={row['bbox']}"
                        for row in relevant_people
                    ]
                    other_candidate_summary = [
                        f"{row['label']}:{row['confidence']:.3f}@{row['bbox']}"
                        for row in relevant_candidates
                        if str(row["label"]).lower() not in {"person", "book"}
                    ]
                    vision_debug_records.append(
                        f"frame={frame_no} raw_yolo_book_candidates=none "
                        f"scan=classwise_NMS_pretracker_floor={candidate_snapshot.get('candidate_floor', 'n/a')} "
                        f"person_candidates={person_summary or 'none'} "
                        f"other_overlapping_candidates={other_candidate_summary or 'none'} "
                        f"normal_detector_conf={args.conf:.3f}"
                    )
                    last_no_book_candidate_print = time.monotonic()

                converted_rows = snapshot_detection_rows(converted_detections)
                final_rows = snapshot_detection_rows(current_detections)
                for candidate in relevant_candidates:
                    yolo_status = candidate_pretracker_status(
                        candidate, standard_candidates, args.conf, 0.70,
                    )
                    standard_match = best_candidate_match(
                        candidate, standard_candidates, same_class=True,
                    )
                    tracker_match, geometry_only_tracker_match = (
                        diagnostic_candidate_tracker_match(candidate, tracked_yolo_rows)
                        if standard_match is not None else (None, None)
                    )
                    converted_match = next((
                        row for row in converted_rows
                        if tracker_match is not None and (
                            (tracker_match.get("track_id") is not None
                             and row["track_id"] == tracker_match["track_id"])
                            or (tracker_match.get("track_id") is None
                                and row["label"] == tracker_match["label"]
                                and bbox_iou(candidate["bbox"], row["bbox"]) >= 0.25)
                        )
                    ), None)
                    candidate_track_id = (
                        tracker_match.get("track_id") if tracker_match else None
                    )
                    app_detection_id = (
                        candidate_track_id if candidate_track_id is not None
                        else converted_match.get("track_id") if converted_match is not None
                        else None
                    )
                    stabilized_match = next((
                        row for row in stabilized_detection_rows
                        if app_detection_id is not None and row["track_id"] == app_detection_id
                    ), None)
                    duplicate_pair = next((
                        (kept, removed) for kept, removed, _overlap in duplicate_detections
                        if app_detection_id is not None and removed.track_id == app_detection_id
                    ), None)
                    final_match = next((
                        row for row in final_rows
                        if app_detection_id is not None and row["track_id"] == app_detection_id
                    ), None)
                    if standard_match is None:
                        tracker_status = "not_reached"
                        conversion_status = "not_reached"
                        class_status = "not_reached"
                        duplicate_status = "not_reached"
                        overlay_status = "not_visible"
                        announcement_status = "not_eligible_no_overlay_track"
                    elif tracker_match is None:
                        if geometry_only_tracker_match is not None:
                            geometry_track_id = geometry_only_tracker_match.get("track_id")
                            geometry_track = (
                                f"{geometry_only_tracker_match['label']}#{geometry_track_id}"
                                if geometry_track_id is not None
                                else str(geometry_only_tracker_match["label"])
                            )
                            tracker_status = (
                                "no_same_class_ByteTrack_output; "
                                f"geometry_only_overlap={geometry_track}"
                            )
                        else:
                            tracker_status = "no_same_class_ByteTrack_output"
                        conversion_status = "not_reached"
                        class_status = "not_reached"
                        duplicate_status = "not_reached"
                        overlay_status = "not_visible"
                        announcement_status = "not_eligible_no_overlay_track"
                    elif converted_match is None:
                        tracker_status = (
                            f"track_{candidate_track_id}" if candidate_track_id is not None
                            else "untracked_result"
                        )
                        conversion_status = "rejected_invalid_box_or_conversion"
                        class_status = "not_reached"
                        duplicate_status = "not_reached"
                        overlay_status = "not_visible"
                        announcement_status = "not_eligible_no_overlay_track"
                    else:
                        tracker_status = (
                            f"track_{candidate_track_id}" if candidate_track_id is not None
                            else "untracked_result"
                        )
                        conversion_status = "survived"
                        tracker_label = tracker_match["label"]
                        stable_label = stabilized_match["label"] if stabilized_match else "n/a"
                        class_status = (
                            f"yolo={candidate['label']}; tracker={tracker_label}; "
                            f"stable={stable_label}"
                        )
                        if duplicate_pair is not None:
                            duplicate_status = f"suppressed_as_duplicate_of_track_{duplicate_pair[0].track_id}"
                        else:
                            duplicate_status = "survived"
                        overlay_status = (
                            f"visible_as_{final_match['label']}#{candidate_track_id}"
                            if final_match is not None else "not_visible_after_duplicate_suppression"
                        )
                        final_detection = next((
                            detection for detection in current_detections
                            if detection.track_id == app_detection_id
                        ), None)
                        eligible = is_detection_announcement_eligible(
                            final_detection,
                            vision_state_by_track.get(candidate_track_id),
                        ) if final_detection is not None else False
                        announcement_status = "eligible" if eligible else "withheld_by_existing_policy"
                    vision_debug_records.append(
                        f"frame={frame_no} yolo_candidate class={candidate['label']} "
                        f"conf={candidate['confidence']:.3f} bbox={candidate['bbox']} "
                        f"normal_yolo_filter={yolo_status} bytetrack={tracker_status} "
                        f"conversion={conversion_status} class_history={class_status} "
                        f"duplicate_suppression={duplicate_status} overlay={overlay_status} "
                        f"announcement={announcement_status}"
                    )

            for kept, removed, overlap in duplicate_detections:
                if removed.has_track_id and removed.track_id != kept.track_id:
                    duplicate_track_id = removed.track_id
                    purge_application_track_state(
                        duplicate_track_id,
                        last_seen_by_track=last_seen_by_track,
                        direction_by_track=direction_by_track,
                        vision_state_by_track=vision_state_by_track,
                        distance_state_by_track=distance_state_by_track,
                        histories=histories,
                        raw_histories=raw_histories,
                        pending_depth_by_track=pending_depth_by_track,
                        depth_diagnostic_by_track=depth_diagnostic_by_track,
                        motion_by_track=motion_by_track,
                        last_announced=last_announced,
                    )
                if args.vision_debug:
                    vision_debug_records.append(
                        f"frame={frame_no} duplicate_suppressed track={removed.track_id} class={removed.label} "
                        f"conf={removed.confidence:.2f} bbox={removed.bbox} kept_track={kept.track_id} "
                        f"reason={'cross_class_geometry' if removed.label != kept.label else 'same_class_geometry'} "
                        f"iou={overlap:.3f}"
                    )
            all_app_track_ids = (
                set(last_seen_by_track) | set(vision_state_by_track) | set(distance_state_by_track)
            )
            expired_tracks = application_tracks_to_purge(
                all_app_track_ids, last_seen_by_track, now_perf,
                removed_tracker_ids=current_removed,
                tracker_lifecycle_available=tracker_lifecycle_available,
            )
            for track_id in expired_tracks:
                if args.vision_debug:
                    vision_debug_records.append(f"frame={frame_no} track_expired id={track_id}")
                purge_application_track_state(
                    track_id,
                    last_seen_by_track=last_seen_by_track,
                    direction_by_track=direction_by_track,
                    vision_state_by_track=vision_state_by_track,
                    distance_state_by_track=distance_state_by_track,
                    histories=histories,
                    raw_histories=raw_histories,
                    pending_depth_by_track=pending_depth_by_track,
                    depth_diagnostic_by_track=depth_diagnostic_by_track,
                    motion_by_track=motion_by_track,
                    last_announced=last_announced,
                )
            if frame_no == 1 or frame_no % max(1, args.depth_every) == 0:
                submitted_perf = time.perf_counter()
                if depth_worker.submit(frame_no, now_perf, frame, tuple(current_detections)) and benchmark is not None:
                    benchmark.record_depth_submission(submitted_perf)

            processing_perf = time.perf_counter()
            depth_result = depth_worker.latest_result()
            latest_depth: Optional[np.ndarray] = None
            if depth_result is not None:
                depth_latency_ms = depth_result.latency_ms
                depth_age = processing_perf - depth_result.captured_at
                if 0.0 <= depth_age <= MAX_DEPTH_RESULT_AGE_SECONDS:
                    latest_depth = depth_result.depth_map
                if depth_result.frame_id > last_consumed_depth_frame_id:
                    last_consumed_depth_frame_id = depth_result.frame_id
                    depth_worker.mark_consumed(depth_result.frame_id)
                    for detection in depth_result.detections:
                        if not detection.has_track_id:
                            continue
                        source_frame_match = depth_source_frames_match(
                            detection.source_frame_id, depth_result.frame_id
                        )
                        sibling_boxes = tuple(
                            other.bbox for other in depth_result.detections
                            if other.has_track_id and other.track_id != detection.track_id
                        )
                        if source_frame_match:
                            sample = median_object_depth(
                                depth_result.depth_map, detection.bbox, depth_estimator.max_depth_m,
                                detection.label, sibling_boxes,
                            )
                        else:
                            roi_bounds = object_sampling_roi(
                                detection.bbox, depth_result.depth_map.shape, detection.label
                            )
                            roi_width = roi_bounds[2] - roi_bounds[0] if roi_bounds else 0
                            roi_height = roi_bounds[3] - roi_bounds[1] if roi_bounds else 0
                            sample = DepthSample(
                                None, None, 0, "source_frame_mismatch", roi_width, roi_height
                            )
                        track_id = detection.track_id
                        previous_state = distance_state_by_track.get(track_id)
                        smooth = previous_state.distance_m if previous_state is not None else None
                        filter_reason = "not_sampled"
                        rejection_reason = sample.rejection_reason
                        if latest_depth is None:
                            rejection_reason = "depth_result_too_old"
                            if sample.rejection_reason:
                                rejection_reason += f"; sample={sample.rejection_reason}"
                        elif sample.median_m is not None:
                            if previous_state is not None and depth_result.captured_at - previous_state.captured_at > TRACK_STATE_RESET_SECONDS:
                                raw_histories[track_id].clear()
                                histories[track_id].clear()
                                pending_depth_by_track.pop(track_id, None)
                                distance_state_by_track.pop(track_id, None)
                                last_announced.pop(track_id, None)
                                motion_by_track[track_id] = "unknown"
                                previous_state = None
                            candidate_before = pending_depth_by_track.get(track_id)
                            change_accepted, pending_candidate, filter_reason = confirm_depth_change(
                                raw_histories[track_id], sample.median_m, depth_result.captured_at,
                                candidate_before,
                            )
                            if not change_accepted:
                                pending_depth_by_track[track_id] = pending_candidate  # type: ignore[assignment]
                                rejection_reason = filter_reason
                            else:
                                pending_depth_by_track.pop(track_id, None)
                                if filter_reason == "sustained_change_confirmed" and candidate_before is not None:
                                    raw_histories[track_id].append(candidate_before)
                                smooth, smooth_reason = smooth_depth_measurement(
                                    raw_histories[track_id], sample.median_m, depth_result.captured_at,
                                    previous_state, depth_estimator.max_depth_m,
                                )
                                if filter_reason == "accepted_no_baseline":
                                    filter_reason = smooth_reason
                                elif smooth_reason != "accepted":
                                    filter_reason = f"{filter_reason}; {smooth_reason}"
                                if is_valid_distance(smooth, depth_estimator.max_depth_m):
                                    distance_state_by_track[track_id] = DistanceState(
                                        smooth, depth_result.captured_at
                                    )
                                    histories[track_id].append((depth_result.captured_at, smooth))
                                    motion_by_track[track_id] = motion_from_history(histories[track_id])
                                    rejection_reason = None
                                else:
                                    smooth = None
                                    distance_state_by_track.pop(track_id, None)
                                    rejection_reason = "smoothed_distance_invalid"

                        diag_reason = depth_diagnostic_reason(
                            sample,
                            depth_result_present=True,
                            source_frame_match=source_frame_match,
                            result_fresh=latest_depth is not None,
                            final_reason=rejection_reason,
                        )
                        depth_diagnostic_by_track[track_id] = {
                            "sample": sample,
                            "depth_frame_id": depth_result.frame_id,
                            "age_seconds": depth_age,
                            "source_frame_match": source_frame_match,
                            "reason": diag_reason,
                        }

                        if args.depth_diagnostics:
                            raw_text = f"{sample.raw_median_m:.2f}m" if sample.raw_median_m is not None else "n/a"
                            smooth_text = f"{smooth:.2f}m" if smooth is not None else "n/a"
                            rejected_text = rejection_reason or "no"
                            depth_diagnostics_pending.append(
                                f"frame={depth_result.frame_id} track={track_id} label={detection.label} "
                                f"conf={detection.confidence:.2f} bbox={detection.bbox} direction={detection.direction} "
                                f"raw_median={raw_text} smoothed={smooth_text} "
                                f"valid_pixels={sample.valid_pixels} age_ms={depth_age * 1000.0:.1f} "
                                f"motion={motion_by_track.get(track_id, 'unknown')} filter={filter_reason} "
                                f"rejected={rejected_text}"
                            )

            if args.depth_diagnostics and depth_diagnostics_pending and time.monotonic() - last_depth_diagnostics_print >= 1.0:
                print(
                    f"[DepthDiag] environment={depth_estimator.environment} model={depth_estimator.model_id} "
                    f"max_range_m={depth_estimator.max_depth_m:.0f}"
                )
                for diagnostic in depth_diagnostics_pending:
                    print(f"[DepthDiag] {diagnostic}")
                depth_diagnostics_pending.clear()
                last_depth_diagnostics_print = time.monotonic()

            next_speech_live: Dict[int, SpeechLiveState] = {}
            for detection in current_detections:
                if not detection.has_track_id:
                    continue
                current_state = distance_state_by_track.get(detection.track_id)
                current_distance = fresh_track_distance(
                    current_state, processing_perf, MAX_DEPTH_RESULT_AGE_SECONDS,
                    depth_estimator.max_depth_m,
                )
                next_speech_live[detection.track_id] = SpeechLiveState(
                    detection.label,
                    detection.direction,
                    last_seen_by_track.get(detection.track_id, processing_perf),
                    current_distance,
                    current_state.captured_at if current_distance is not None and current_state is not None else None,
                    motion_by_track.get(detection.track_id, "unknown") if current_distance is not None else "unknown",
                    detection.identity_uncertain,
                )
            with speech_live_lock:
                speech_live_by_track.clear()
                speech_live_by_track.update(next_speech_live)

            display = frame.copy()
            current: list[ObjectObservation] = []
            speech_debug_records: list[str] = []
            final_overlay_rows: list[str] = []
            # Process eligible nearby detections first. Within similar relevance,
            # prefer tracks not announced recently so one person cannot monopolize
            # the scene-wide ordinary-announcement opportunity.
            ordered_detections = sorted(
                current_detections,
                key=lambda item: announcement_candidate_key(
                    item,
                    fresh_track_distance(
                        distance_state_by_track.get(item.track_id) if item.has_track_id else None,
                        processing_perf, MAX_DEPTH_RESULT_AGE_SECONDS,
                        depth_estimator.max_depth_m,
                    ),
                    last_announced[item.track_id].announced_at
                    if item.has_track_id and item.track_id in last_announced else None,
                    is_detection_announcement_eligible(
                        item, vision_state_by_track.get(item.track_id)
                        if item.has_track_id else None,
                    ),
                ),
            )
            for detection in ordered_detections:
                track_id = detection.track_id
                x1, y1, x2, y2 = detection.bbox
                distance_state = distance_state_by_track.get(track_id) if detection.has_track_id else None
                distance = fresh_track_distance(
                    distance_state, processing_perf, MAX_DEPTH_RESULT_AGE_SECONDS,
                    depth_estimator.max_depth_m,
                )
                if args.vision_debug and distance is None and detection.has_track_id:
                    diag = depth_diagnostic_by_track.get(track_id)
                    current_result_fresh = (
                        depth_result is not None
                        and 0.0 <= processing_perf - depth_result.captured_at <= MAX_DEPTH_RESULT_AGE_SECONDS
                    )
                    if distance_state is not None and fresh_track_distance(
                        distance_state, processing_perf, MAX_DEPTH_RESULT_AGE_SECONDS,
                        depth_estimator.max_depth_m,
                    ) is None:
                        reason = "STALE_DEPTH_RESULT"
                    elif diag is not None:
                        reason = diag["reason"]
                    elif depth_result is not None and not current_result_fresh:
                        reason = "STALE_DEPTH_RESULT"
                    else:
                        reason = depth_worker_unavailable_reason(
                            detection.source_frame_id if detection.source_frame_id is not None else frame_no,
                            depth_worker.status_snapshot(),
                        )
                    last_reason, last_logged = last_unavailable_diag_by_track.get(track_id, ("", 0.0))
                    if reason != last_reason or time.monotonic() - last_logged >= 1.0:
                        diagnostic_sample = diag["sample"] if diag is not None else None
                        if diagnostic_sample is None:
                            roi_bounds = object_sampling_roi(detection.bbox, frame.shape, detection.label)
                            roi_width = roi_bounds[2] - roi_bounds[0] if roi_bounds else 0
                            roi_height = roi_bounds[3] - roi_bounds[1] if roi_bounds else 0
                            diagnostic_sample = DepthSample(
                                None, None, 0, "no_depth_result", roi_width, roi_height
                            )
                        vision_debug_records.append(format_depth_sampling_diagnostic(
                            detection,
                            diagnostic_sample,
                            depth_frame_id=diag["depth_frame_id"] if diag is not None else (
                                depth_result.frame_id if depth_result is not None else None
                            ),
                            depth_age_seconds=diag["age_seconds"] if diag is not None else (
                                processing_perf - depth_result.captured_at if depth_result is not None else None
                            ),
                            source_frame_match=diag["source_frame_match"] if diag is not None else None,
                            reason=reason,
                        ))
                        last_unavailable_diag_by_track[track_id] = (reason, time.monotonic())
                has_fresh_distance = distance is not None
                motion = motion_by_track.get(track_id, "unknown") if has_fresh_distance else "unknown"
                approaching = motion == "approaching"
                obs = ObjectObservation(
                    track_id, detection.label, detection.confidence, detection.direction,
                    distance, approaching, detection.bbox,
                )
                current.append(obs)

                # Visual overlay: different color if depth trend suggests approach.
                color = (40, 60, 255) if approaching else (0, 230, 190)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                range_txt = format_distance(distance, depth_estimator.max_depth_m)
                motion_text = {
                    "approaching": " | DEPTH CLOSING",
                    "receding": " | DEPTH RECEDING",
                    "stationary": " | DEPTH STABLE",
                }.get(motion, "")
                approach_txt = motion_text
                identity_text = (
                    f" | ID uncertain (possibly #{detection.possible_parent_track_id})"
                    if detection.identity_uncertain else ""
                )
                txt = f"{detection.label} #{track_id} {detection.confidence*100:.0f}% | {detection.direction} | {range_txt}{approach_txt}{identity_text}"
                font_scale = 0.48 if w < 1000 else 0.55
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                ty = max(th + 7, y1 - 8)
                cv2.rectangle(display, (x1, ty - th - 6), (min(w - 1, x1 + tw + 8), ty + 3), (8, 17, 35), -1)
                cv2.putText(display, txt, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

                prev_state = last_announced.get(track_id) if detection.has_track_id else None
                vision_state = vision_state_by_track.get(track_id) if detection.has_track_id else None
                track_eligible = is_detection_announcement_eligible(detection, vision_state)
                final_overlay_rows.append(
                    f"{detection.label}#{track_id if detection.has_track_id else 'untracked'} "
                    f"conf={detection.confidence:.2f} bbox={detection.bbox} "
                    f"identity={'uncertain' if detection.identity_uncertain else 'tracked'} "
                    f"announcement_eligible={track_eligible}"
                )
                close_confirmed = (
                    distance is not None
                    and distance <= CLOSE_OBSTACLE_DISTANCE_M
                    and track_eligible
                    and close_measurement_confirmed(raw_histories[track_id], processing_perf)
                )
                should_announce, priority, decision_reason = announcement_decision(
                    prev_state, detection.direction, distance, motion, processing_perf,
                    label=detection.label,
                    close_confirmed=close_confirmed,
                ) if detection.has_track_id else (False, 0, "no_stable_track_id")
                if detection.identity_uncertain:
                    should_announce = False
                    decision_reason = "occlusion_identity_uncertain"
                elif should_announce and not track_eligible:
                    decision_reason = "insufficient_repeated_detection_evidence"
                elif should_announce and distance is not None and distance <= CLOSE_OBSTACLE_DISTANCE_M and not close_confirmed:
                    decision_reason = "awaiting_confirmed_close_depth"
                elif should_announce and speech.enabled:
                    direction_text = {
                        "ahead": "ahead",
                        "left": "on your left",
                        "far left": "far to your left",
                        "right": "on your right",
                        "far right": "far to your right",
                    }.get(detection.direction, detection.direction)
                    phrase = f"{detection.label.title()} {direction_text}"
                    if distance is not None:
                        phrase += f", {readable_distance(distance, depth_estimator.max_depth_m)} away"
                    elif prev_state is None:
                        phrase += ", distance unavailable"
                    if close_confirmed:
                        phrase = f"Obstacle close {direction_text}: {phrase}"
                    if motion == "approaching":
                        phrase += ". Distance trend suggests it is approaching"
                    elif motion == "receding":
                        phrase += ". Distance trend suggests it is moving away"
                    elif motion == "stationary":
                        phrase += ". Distance trend is roughly stable"
                    phrase += "."
                    expires_at = (
                        distance_state.captured_at + MAX_DEPTH_RESULT_AGE_SECONDS
                        if distance is not None and distance_state is not None
                        else processing_perf + 1.5
                    )
                    global_suppression = ordinary_announcement_suppression_reason(
                        last_ordinary_announcement_at,
                        processing_perf,
                        close_confirmed=close_confirmed,
                    )
                    if global_suppression is not None:
                        decision_reason = global_suppression
                        log_key = (track_id, global_suppression)
                        last_logged = last_announcement_suppression_log.get(log_key, -float("inf"))
                        if args.vision_debug and processing_perf - last_logged >= 2.0:
                            vision_debug_records.append(
                                f"frame={frame_no} announcement_suppressed track={track_id} "
                                f"class={detection.label} category=ordinary priority={priority} "
                                f"reason={global_suppression} cooldown_s="
                                f"{ORDINARY_ANNOUNCEMENT_GLOBAL_COOLDOWN_SECONDS:.1f}"
                            )
                            last_announcement_suppression_log[log_key] = processing_perf
                    elif speech.say(
                        phrase, expires_at=expires_at, priority=priority,
                        key=f"track:{track_id}",
                        validation_token=SpeechValidationToken(
                            track_id, detection.label, detection.direction, distance, motion,
                        ),
                    ):
                        last_announced[track_id] = AnnouncementState(
                            detection.direction, processing_perf, motion, distance, detection.label
                        )
                        if not close_confirmed:
                            last_ordinary_announcement_at = processing_perf
                        decision_reason = f"{decision_reason}_queued"
                    else:
                        decision_reason = "speech_worker_rejected"
                elif should_announce:
                    decision_reason = "voice_disabled"
                if args.speech_debug:
                    distance_age_ms = (
                        (processing_perf - distance_state.captured_at) * 1000.0
                        if distance_state is not None else None
                    )
                    speech_debug_records.append(
                        f"frame={frame_no} track={track_id if detection.has_track_id else 'untracked'} "
                        f"class={detection.label} conf={detection.confidence:.2f} bbox={detection.bbox} "
                        f"direction={detection.direction} distance={distance if distance is not None else 'unavailable'} "
                        f"announce_eligible={track_eligible} close_confirmed={close_confirmed} "
                        f"depth_age_ms={distance_age_ms if distance_age_ms is not None else 'n/a'} "
                        f"motion={motion} decision={'queued' if decision_reason.endswith('_queued') else 'suppressed'} "
                        f"reason={decision_reason} speech_status={speech.status}"
                    )

            if args.vision_debug:
                if final_overlay_rows:
                    vision_debug_records.append(
                        f"frame={frame_no} final_overlay_detection_list={final_overlay_rows}"
                    )
                else:
                    vision_debug_records.append(f"frame={frame_no} final_overlay_detection_list=empty")
                pending_vision_debug_records.extend(vision_debug_records)
            if (args.vision_debug and pending_vision_debug_records
                    and time.monotonic() - last_vision_debug_print >= 1.0):
                while pending_vision_debug_records:
                    record = pending_vision_debug_records.popleft()
                    print(f"[VisionDiag] {record}")
                last_vision_debug_print = time.monotonic()

            if args.vision_debug and time.monotonic() - last_depth_lifecycle_print >= 1.0:
                status = depth_worker.status_snapshot()
                print(
                    "[DepthLifecycle] "
                    f"processing={status['processing_frame_id'] if status['processing_frame_id'] is not None else 'none'} "
                    f"pending={status['pending_frame_id'] if status['pending_frame_id'] is not None else 'none'} "
                    f"last_submitted={status['last_submitted_frame_id']} "
                    f"latest_completed={status['latest_completed_frame_id']} "
                    f"last_consumed={status['last_consumed_frame_id']} "
                    f"submitted={status['submitted']} pending_replaced={status['pending_replaced']} "
                    f"completed={status['completed']} "
                    f"unconsumed_result_replaced={status['unconsumed_result_replaced']} "
                    f"failures={status['failures']} last_error={status['last_error'] or 'none'}"
                )
                last_depth_lifecycle_print = time.monotonic()

            if args.speech_debug and speech_debug_records and time.monotonic() - last_speech_debug_print >= 1.0:
                for record in speech_debug_records:
                    print(f"[SpeechDiag] {record}")
                last_speech_debug_print = time.monotonic()

            # Compact side HUD below the frame status, no invented values.
            cv2.rectangle(display, (0, 0), (w, 98), (5, 12, 28), -1)
            cv2.putText(display, "SIGHTBRIDGE | SPATIAL AWARENESS PROOF OF CONCEPT", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 230, 190), 2, cv2.LINE_AA)
            cv2.putText(display, f"Objects: {len(current)} | YOLO: {detect_latency_ms:.0f} ms | Depth: {depth_latency_ms:.0f} ms | FPS: {fps_ema:.1f} | Device: {device.upper()}", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 239, 245), 1, cv2.LINE_AA)
            cv2.putText(display, "Q quit | V voice on/off | P pause | Direction is camera-relative; distance and approach are estimates.", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 205), 1, cv2.LINE_AA)

            if latest_depth is None:
                cv2.putText(display, "DEPTH UNAVAILABLE - NOT REPORTING DISTANCES", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow("SightBridge | Spatial Awareness Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('v'):
                speech.set_enabled(not speech.enabled)
                if speech.enabled:
                    last_announced.clear()
                    speech.say("Voice feedback enabled.", force=True, priority=1, key="startup")
            elif key == ord('p'):
                paused = True

    except KeyboardInterrupt:
        pass
    finally:
        if args.vision_debug:
            while pending_vision_debug_records:
                print(f"[VisionDiag] {pending_vision_debug_records.popleft()}")
        depth_worker.stop(timeout=3.0)
        if args.depth_diagnostics and depth_diagnostics_pending:
            print(
                f"[DepthDiag] environment={depth_estimator.environment} model={depth_estimator.model_id} "
                f"max_range_m={depth_estimator.max_depth_m:.0f}"
            )
            for diagnostic in depth_diagnostics_pending:
                print(f"[DepthDiag] {diagnostic}")
            depth_diagnostics_pending.clear()
        benchmark_results = benchmark.finish(time.perf_counter()) if benchmark is not None else None
        cap.release()
        cv2.destroyAllWindows()
        speech.stop(timeout=3.0)
        if benchmark is not None and benchmark_results is not None:
            benchmark.save_and_print(benchmark_results)
    if debug_log is not None:
        debug_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
