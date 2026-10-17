from __future__ import annotations

import threading
from collections import deque

from .types import Frame


class FrameBuffer:
    def __init__(self, seconds: float, max_bytes: int, fps: float):
        self.seconds = seconds
        self.max_bytes = max_bytes
        self.fps = fps
        self._frames: deque[Frame] = deque()
        self._bytes = 0
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._frames.clear()
            self._bytes = 0

    def add(self, frame: Frame):
        with self._lock:
            if self._frames and (
                frame.epoch != self._frames[-1].epoch or frame.capture_ts < self._frames[-1].capture_ts
            ):
                self.clear()
            if self._frames and frame.capture_ts - self._frames[-1].capture_ts < 1 / self.fps - 1e-6:
                return
            if len(frame.jpeg) > self.max_bytes:
                return
            self._frames.append(frame)
            self._bytes += len(frame.jpeg)
            while self._frames and (
                self._bytes > self.max_bytes or frame.capture_ts - self._frames[0].capture_ts > self.seconds
            ):
                self._bytes -= len(self._frames.popleft().jpeg)

    def frames(self, since: float | None = None) -> list[Frame]:
        with self._lock:
            return [f for f in self._frames if since is None or f.capture_ts >= since]

    def stats(self) -> dict:
        with self._lock:
            return {
                "frames": len(self._frames),
                "bytes": self._bytes,
                "depth_seconds": self._frames[-1].capture_ts - self._frames[0].capture_ts
                if len(self._frames) > 1
                else 0,
            }


def evenly_spaced(frames: list[Frame], count: int) -> list[Frame]:
    if len(frames) <= count:
        return frames
    # Select by source time, preserving irregular frame spacing.
    start, end = frames[0].capture_ts, frames[-1].capture_ts
    targets = [start + (end - start) * i / (count - 1) for i in range(count)] if count > 1 else [end]
    selected = [min(frames, key=lambda f: abs(f.capture_ts - t)) for t in targets]
    return list({f.frame_id: f for f in selected}.values())
