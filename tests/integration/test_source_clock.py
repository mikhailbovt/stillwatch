import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stillwatch.buffer import FrameBuffer
from stillwatch.capture import VideoCapture
from stillwatch.runtime import Analysis
from stillwatch.telemetry import Telemetry
from stillwatch.types import Box, Observation


class TimestampCapture:
    def __init__(self, times, fps=25):
        self.times, self.fps, self.index = times, fps, 0

    def isOpened(self):
        return True

    def read(self):
        if self.index == len(self.times):
            return False, None
        self.index += 1
        return True, np.full((240, 320, 3), 50 + self.index % 150, np.uint8)

    def get(self, key):
        return self.fps if key == cv2.CAP_PROP_FPS else self.times[self.index - 1] * 1000

    def release(self):
        pass


def test_file_capture_does_not_accumulate_time_on_repeated_native_timestamps(settings, monkeypatch):
    settings.source, settings.realtime = "clock.mpg", False
    buffer = FrameBuffer(30, 1024**2, 10)
    capture = VideoCapture(settings, buffer)
    times = [0, 0.2] + [0.2] * 400 + [0.24, 0.28]
    monkeypatch.setattr(capture, "_open", lambda: TimestampCapture(times))
    capture._capture()
    assert capture.state == "ended"
    assert capture.latest.frame.frame_id == len(times) - 1
    assert capture.latest.frame.capture_ts == pytest.approx(0.28)
    assert all(frame.capture_ts <= 0.28 for frame in buffer.frames())


def test_resuming_after_a_timestamp_plateau_breaks_source_continuity(settings, monkeypatch):
    import time

    settings.source, settings.realtime = "clock.mpg", False
    buffer = FrameBuffer(30, 1024**2, 10)
    capture = VideoCapture(settings, buffer)

    class Stalled(TimestampCapture):
        def read(self):
            ok, image = super().read()
            if ok and self.index == 3:
                capture.file_clock.last_progress_mono = time.monotonic() - settings.frame_stale_seconds - 1
            return ok, image

    monkeypatch.setattr(capture, "_open", lambda: Stalled([0, 0.2, 0.2, 0.24, 0.4]))
    capture._capture()
    assert capture.epoch == 2
    assert capture.latest.frame.epoch == 2
    assert all(frame.epoch == 2 and frame.capture_ts >= 0.24 for frame in buffer.frames())


async def test_repeated_native_time_cannot_complete_a_medical_observation(settings, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tools.replay import replay

    class Perception:
        states = {"pose": "ready", "fire": "disabled"}
        errors = {}

        def reset(self):
            pass

        def people(self, image, ts):
            box = Box(x1=100, y1=20, x2=160, y2=220) if ts == 0 else Box(x1=100, y1=140, x2=260, y2=220)
            return [Observation(capture_ts=ts, box=box, label="person", score=0.9)]

    source = tmp_path / "repeated.mpg"
    source.write_bytes(b"decoder supplied by the test")
    times = [0, 0.2] + [0.2] * 400
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: TimestampCapture(times))
    output = tmp_path / "replay"
    result = await replay(source, settings, output, analysis=Analysis(settings, Telemetry(), Perception()))
    medical = [event for event in result["events"] if event["category"] == "medical"]
    assert medical
    assert all(event["full_assessment_source_ts"] is None for event in medical)
    assert all(event["request_count"] == 0 for event in medical)
    assert result["duration_seconds"] == pytest.approx(0.2)
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert max(row["source_ts"] for row in rows) == pytest.approx(0.2)
