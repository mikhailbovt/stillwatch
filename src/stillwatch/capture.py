from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .buffer import FrameBuffer
from .config import Settings
from .types import Frame


@dataclass(frozen=True)
class Sample:
    frame: Frame
    image: np.ndarray


class FileClock:
    """Use native presentation time once observed; a repeated PTS adds no time."""

    def __init__(self, fps: float):
        self.fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 25
        self.native = False
        self.frames = 0
        self.nominal_frames = 0
        self.held_frames = 0
        self.invalid_timestamps = 0
        self.regressed_timestamps = 0
        self.source_ts: float | None = None
        self.last_raw_ts: float | None = None
        self.last_progress_mono: float | None = None
        self.holding = False

    def advance(self, raw_ts: float) -> float:
        valid = math.isfinite(raw_ts) and raw_ts >= 0
        self.last_raw_ts = raw_ts if valid else None
        self.invalid_timestamps += not valid
        self.native = self.native or (valid and raw_ts > 0)
        previous = self.source_ts
        if self.native:
            ts = max(previous or 0.0, raw_ts if valid else previous or 0.0)
            self.holding = previous is not None and ts == previous
            if self.holding:
                self.held_frames += 1
                self.regressed_timestamps += valid and raw_ts < previous
        else:
            ts = self.frames / self.fps
            self.nominal_frames += 1
        if previous is None or ts > previous:
            self.last_progress_mono = time.monotonic()
        self.frames += 1
        self.source_ts = ts
        return ts

    def stalled(self, maximum_age: float) -> bool:
        return bool(
            self.native
            and self.holding
            and self.last_progress_mono is not None
            and time.monotonic() - self.last_progress_mono > maximum_age
        )

    def snapshot(self) -> dict:
        return {
            "mode": "native_pts" if self.native else "nominal_fps",
            "nominal_fps": self.fps,
            "frames_decoded": self.frames,
            "nominal_frames": self.nominal_frames,
            "held_frames": self.held_frames,
            "holding": self.holding,
            "invalid_timestamps": self.invalid_timestamps,
            "regressed_timestamps": self.regressed_timestamps,
            "last_raw_ts": self.last_raw_ts,
            "source_ts": self.source_ts,
        }


def sanitize(image: np.ndarray, settings: Settings) -> np.ndarray:
    image = image.copy()
    h, w = image.shape[:2]
    for x1, y1, x2, y2 in settings.privacy_masks:
        image[int(y1 * h) : int(np.ceil(y2 * h)), int(x1 * w) : int(np.ceil(x2 * w))] = 0
    scale = min(1.0, settings.jpeg_long_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
    return image


def make_sample(image: np.ndarray, frame_id: int, ts: float, epoch: int, settings: Settings) -> Sample:
    image = sanitize(image, settings)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quality = []
    if gray.mean() < 18:
        quality.append("dark")
    if np.mean(gray > 248) > 0.6:
        quality.append("overexposed")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 25:
        quality.append("blurred")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    h, w = image.shape[:2]
    return Sample(
        Frame(frame_id, settings.camera_id, ts, w, h, encoded.tobytes(), epoch, quality=tuple(quality)), image
    )


def open_capture(source: str | int | None):
    # Do not let FFmpeg echo RTSP credentials from a failed URL.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    if isinstance(source, str) and source.startswith(("rtsp://", "rtsps://")):
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                3000,
                cv2.CAP_PROP_N_THREADS,
                2,
            ],
        )
    if isinstance(source, str):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    return cv2.VideoCapture(source)


class VideoCapture:
    """A single replaceable sample: a slow consumer never accumulates decoded frames."""

    def __init__(self, settings: Settings, buffer: FrameBuffer):
        self.settings = settings
        self.buffer = buffer
        self.latest: Sample | None = None
        self.state = "not_configured" if settings.source is None else "starting"
        self.epoch = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.dropped_decode = 0
        self.file_clock: FileClock | None = None

    def start(self):
        if self.settings.source is not None and not (self._thread and self._thread.is_alive()):
            self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=8)
        return not (self._thread and self._thread.is_alive())

    def _open(self):
        return open_capture(self.settings.source)

    def _run(self):
        try:
            self._capture()
        except Exception:
            self.state = "capture_error"
        finally:
            if self._cap is not None:
                self._cap.release()

    def _capture(self):
        frame_id = 0
        is_file = isinstance(self.settings.source, str) and "://" not in self.settings.source
        while not self._stop.is_set():
            cap = self._cap = self._open()
            if not cap.isOpened():
                cap.release()
                self.state = "unavailable"
                if is_file:
                    return
                self._stop.wait(2)
                continue
            self.epoch += 1
            self.buffer.clear()
            self.latest = None
            self.state = "streaming"
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.file_clock = FileClock(fps) if is_file else None
            started = time.monotonic()
            shape = None
            try:
                while not self._stop.is_set():
                    ok, image = cap.read()
                    if self._stop.is_set():
                        break
                    if not ok:
                        self.state = "ended" if is_file else "disconnected"
                        break
                    raw_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.monotonic() - started
                    stalled_clock = self.file_clock and self.file_clock.stalled(
                        self.settings.frame_stale_seconds
                    )
                    previous_ts = self.file_clock.source_ts if self.file_clock else None
                    ts = self.file_clock.advance(raw_ts) if self.file_clock else raw_ts
                    if stalled_clock and previous_ts is not None and ts > previous_ts:
                        self.epoch += 1
                        self.buffer.clear()
                        self.latest = None
                    if is_file and self.settings.realtime:
                        self._stop.wait(max(0, started + ts - time.monotonic()))
                    if self._stop.is_set():
                        break
                    if shape is not None and image.shape != shape:
                        self.epoch += 1
                        self.buffer.clear()
                    shape = image.shape
                    sample = make_sample(image, frame_id, ts, self.epoch, self.settings)
                    self.buffer.add(sample.frame)
                    self.latest = sample
                    frame_id += 1
            finally:
                cap.release()
            if is_file and not self.settings.loop:
                return
            if not is_file:
                self._stop.wait(2)
        self.state = "stopped"
