import threading
import time

import cv2
import numpy as np

from stillwatch.buffer import FrameBuffer
from stillwatch.capture import VideoCapture


def write_video(path, seconds=1.2, fps=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(round(seconds * fps)):
        image = np.full((120, 160, 3), i * 5, np.uint8)
        cv2.rectangle(image, (i * 3 % 100, 30), (i * 3 % 100 + 20, 70), (240, 240, 240), -1)
        writer.write(image)
    writer.release()


def test_slow_consumer_uses_latest_frame_and_capture_stops(settings, tmp_path):
    path = tmp_path / "test.mp4"
    write_video(path)
    settings.source = str(path)
    buffer = FrameBuffer(12, 1024**2, 10)
    capture = VideoCapture(settings, buffer)
    capture.start()
    try:
        deadline = time.monotonic() + 3
        observed = []
        while time.monotonic() < deadline and capture.state != "ended":
            if capture.latest:
                observed.append(capture.latest.frame.frame_id)
                assert time.monotonic() - capture.latest.frame.monotonic_ts < 0.5
            time.sleep(0.25)  # Simulated slow inference.
        assert capture.state == "ended"
        assert len(observed) < 24 and capture.latest.frame.frame_id == 23
        assert max(observed, default=0) - min(observed, default=0) > 1
        assert buffer.stats()["bytes"] <= 1024**2
    finally:
        capture.stop()
    assert not capture._thread.is_alive()


def test_live_disconnect_reconnect_changes_epoch_and_clears_history(settings, monkeypatch):
    settings.source = 0
    settings.realtime = False
    buffer = FrameBuffer(12, 1024**2, 10)
    capture = VideoCapture(settings, buffer)
    calls = []

    class Fake:
        def __init__(self, index):
            self.index = index
            self.count = 0

        def isOpened(self):
            return True

        def get(self, key):
            return 10

        def read(self):
            self.count += 1
            if self.count > 2:
                return False, None
            return True, np.full((120, 160, 3), self.index * 50, np.uint8)

        def release(self):
            pass

    def opened():
        calls.append(len(calls) + 1)
        return Fake(len(calls))

    monkeypatch.setattr(capture, "_open", opened)
    capture.start()
    deadline = time.monotonic() + 4
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    capture.stop()
    assert len(calls) >= 2
    assert capture.epoch >= 2
    assert all(f.epoch == capture.epoch for f in buffer.frames())
    assert not any(t.name == "capture" and t.is_alive() for t in threading.enumerate())
