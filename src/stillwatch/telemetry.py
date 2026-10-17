from __future__ import annotations

import time
from collections import defaultdict, deque

import psutil


class Telemetry:
    def __init__(self):
        self.started_mono = time.monotonic()
        self.ticks: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=600))
        self.inference_ms: dict[str, float] = {}
        self.process = psutil.Process()
        self.error: str | None = None

    def record(self, route: str, elapsed: float):
        self.ticks[route].append(time.monotonic())
        self.inference_ms[route] = elapsed * 1000

    def snapshot(self) -> dict:
        now = time.monotonic()
        fps = {}
        for route in ("pose", "fire"):
            ticks = [t for t in self.ticks[route] if now - t <= 10]
            fps[route] = (len(ticks) - 1) / (ticks[-1] - ticks[0]) if len(ticks) > 1 else 0
        return {
            "uptime_seconds": now - self.started_mono,
            "fps": fps,
            "inference_ms": self.inference_ms.copy(),
            "rss_bytes": self.process.memory_info().rss,
            "cpu_percent": self.process.cpu_percent(),
            "error": self.error,
        }
