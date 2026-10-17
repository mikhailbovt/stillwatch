from __future__ import annotations

import asyncio
import threading
import time

from . import __version__
from .buffer import FrameBuffer
from .capture import Sample, VideoCapture
from .config import Settings, implementation_fingerprint
from .decisions import DecisionProvider, MockDecisionProvider, OpenAIDecisionProvider
from .evidence import EvidenceBuilder
from .incidents import IncidentManager
from .interfaces import PerceptionAdapter
from .perception import Perception
from .scheduler import Scheduler
from .sources import SourceDescriptor, SourceRequest, SourceStore, describe_source
from .storage import Storage
from .telemetry import Telemetry
from .triggers import FireRules, MedicalRules
from .types import Candidate, Observation


class Analysis:
    """Sequential local inference, shared by live processing and deterministic replay."""

    def __init__(self, settings: Settings, telemetry: Telemetry, perception: PerceptionAdapter | None = None):
        self.settings, self.telemetry = settings, telemetry
        self.perception = perception if perception is not None else Perception(settings)
        self.medical, self.fire = MedicalRules(settings), FireRules(settings)
        self.epoch = -1
        self.last_pose = self.last_fire = -float("inf")
        self.observations: dict[str, list[Observation]] = {"pose": [], "fire": []}
        self.detections: dict[str, list[Observation]] = {"pose": [], "fire": []}

    def process(self, sample: Sample) -> list[Candidate]:
        frame = sample.frame
        if frame.epoch != self.epoch:
            self.perception.reset()
            self.medical.reset()
            self.fire.reset()
            self.epoch = frame.epoch
            self.last_pose = self.last_fire = -float("inf")
            self.observations = {"pose": [], "fire": []}
            self.detections = {"pose": [], "fire": []}
        candidates = []
        for route, cadence in (("pose", self.settings.pose_fps), ("fire", self.settings.fire_fps)):
            previous = self.last_pose if route == "pose" else self.last_fire
            if frame.capture_ts - previous < 1 / cadence - 1e-6 or self.perception.states[route] != "ready":
                continue
            started = time.monotonic()
            try:
                observations = (
                    self.perception.people(sample.image, frame.capture_ts)
                    if route == "pose"
                    else self.perception.hazards(sample.image, frame.capture_ts)
                )
                rules = self.medical if route == "pose" else self.fire
                self.detections[route] = observations
                candidates.extend(rules.update(observations, frame.capture_ts, frame.epoch))
                self.observations[route] = self.medical.observations if route == "pose" else observations
                if route == "pose":
                    self.last_pose = frame.capture_ts
                else:
                    self.last_fire = frame.capture_ts
                self.telemetry.record(route, time.monotonic() - started)
            except Exception as error:
                self.perception.states[route] = "runtime_error"
                self.perception.errors[route] = type(error).__name__
        return candidates


class Runtime:
    def __init__(
        self,
        settings: Settings,
        *,
        perception: PerceptionAdapter | None = None,
        provider: DecisionProvider | None = None,
    ):
        self.settings = settings
        self.implementation_hash = implementation_fingerprint()
        self.buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
        self.storage = Storage(settings)
        self.evidence = EvidenceBuilder(settings, self.buffer)
        self.manager = IncidentManager(settings, self.storage, self.evidence)
        provider = (
            provider
            if provider is not None
            else (
                MockDecisionProvider()
                if settings.provider == "mock"
                else OpenAIDecisionProvider(settings.request_timeout)
            )
        )
        self.capture = VideoCapture(settings, self.buffer)
        self.sources = SourceStore(settings)
        self.source_kind, self.source_label = describe_source(settings.source)
        self.source_file_id: str | None = None
        self._source_lock = asyncio.Lock()
        self._switching = False
        self.scheduler = Scheduler(settings, self.manager, provider, self._fresh_source)
        self.capture.epoch = max((i.epoch for i in self.manager.items.values()), default=0)
        self.telemetry = Telemetry()
        self.analysis = Analysis(settings, self.telemetry, perception)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tasks: list[asyncio.Task] = []
        self.revision = 0
        self.overlays: list[dict] = []
        self.last_analysis_mono: float | None = None
        self.last_analysis_source: float | None = None
        self._last_cleanup = time.monotonic()

    def _fresh_source(self) -> bool:
        latest = self.capture.latest
        return bool(
            not self._switching
            and not self._source_clock_stalled()
            and latest
            and self.capture.state == "streaming"
            and time.monotonic() - latest.frame.monotonic_ts
            <= min(self.settings.frame_stale_seconds, self.settings.packet_stale_seconds)
        )

    def _source_clock_stalled(self) -> bool:
        clock = getattr(self.capture, "file_clock", None)
        return bool(
            self.capture.state == "streaming" and clock and clock.stalled(self.settings.frame_stale_seconds)
        )

    async def start(self):
        loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._analyze, args=(loop,), name="inference", daemon=True)
        self._thread.start()
        self._tasks = [asyncio.create_task(self.scheduler.run()), asyncio.create_task(self._tick())]

    def _analyze(self, loop: asyncio.AbstractEventLoop):
        try:
            self.analysis.perception.load()
            if not self._stop.is_set():
                self.capture.start()
            previous = None
            while not self._stop.is_set():
                sample = self.capture.latest
                if sample is None or (sample.frame.epoch, sample.frame.frame_id) == previous:
                    self._stop.wait(0.01)
                    continue
                previous = (sample.frame.epoch, sample.frame.frame_id)
                candidates = self.analysis.process(sample)
                observations = [
                    o.model_dump(mode="json")
                    for values in self.analysis.observations.values()
                    for o in values
                ]
                motion = self.analysis.medical.motion.snapshot()
                loop.call_soon_threadsafe(self._receive, sample, candidates, observations, motion)
        except Exception as error:
            self.telemetry.error = type(error).__name__

    def _receive(self, sample: Sample, candidates: list[Candidate], observations: list[dict], motion=None):
        if self._stop.is_set():
            return
        if sample.frame.epoch != self.capture.epoch:
            return
        self.last_analysis_mono = time.monotonic()
        self.last_analysis_source = sample.frame.capture_ts
        self.overlays = observations
        self.evidence.observe(observations, sample.frame.capture_ts, sample.frame.epoch)
        self.manager.observe_motion(motion or [], sample.frame.capture_ts)
        self.scheduler.advance(sample.frame.capture_ts, sample.frame.epoch)
        for candidate in candidates:
            item, created = self.manager.ingest(candidate)
            if created:
                self.scheduler.enqueue(item)
        self.revision += 1

    async def _tick(self):
        while not self._stop.is_set():
            # The overview clock follows source time, independently of route output or availability.
            latest = self.capture.latest
            if latest and self.capture.state == "streaming":
                self.scheduler.advance(latest.frame.capture_ts, latest.frame.epoch)
            if time.monotonic() - self._last_cleanup >= 60:
                self.storage.cleanup()
                self._last_cleanup = time.monotonic()
                persisted = {i.incident_id for i in self.storage.incidents()}
                for key in list(self.manager.items):
                    if key not in persisted and not self.storage.error:
                        self.manager.items.pop(key)
                        self.evidence.release(key)
            await asyncio.sleep(0.2)

    def source_descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            kind=self.source_kind,
            label=self.source_label,
            state="switching" if self._switching else self.capture.state,
            epoch=self.capture.epoch,
            loop=self.capture.settings.loop,
            file_id=self.source_file_id,
        )

    async def switch_source(self, selection: SourceRequest) -> SourceDescriptor:
        source, label = self.sources.resolve(selection)
        async with self._source_lock:
            self._switching = True
            old = self.capture
            try:
                stopped = await asyncio.to_thread(old.stop)
                if not stopped:
                    old.state = "stop_timeout"
                    raise RuntimeError("Previous capture has not stopped; retry after it releases the device")
                # Each capture owns its buffer. A late old worker cannot contaminate the new source.
                buffer = FrameBuffer(
                    self.settings.buffer_seconds, self.settings.buffer_max_bytes, self.settings.buffer_fps
                )
                config = self.settings.model_copy(update={"source": source, "loop": selection.loop})
                capture = VideoCapture(config, buffer)
                capture.epoch = max(old.epoch, self.scheduler.epoch) + 1
                self.buffer = buffer
                self.evidence.buffer = buffer
                self.capture = capture
                self.source_kind, self.source_label = selection.kind, label
                self.source_file_id = selection.file_id
                self.overlays = []
                self.manager.latest_motion.clear()
                self.last_analysis_mono = self.last_analysis_source = None
                self.scheduler.change_source(capture.epoch)
                capture.start()
                self.revision += 1
                self._switching = False
                return self.source_descriptor()
            finally:
                self._switching = False

    async def stop(self):
        self._stop.set()
        await asyncio.to_thread(self.capture.stop)
        if self._thread:
            await asyncio.to_thread(self._thread.join, 10)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.scheduler.stop()
        self.storage.close()

    def snapshot(self) -> dict:
        latest = self.capture.latest
        age = time.monotonic() - latest.frame.monotonic_ts if latest else None
        routes = self.analysis.perception.states.copy()
        reasons = []
        clock = getattr(self.capture, "file_clock", None)
        clock_stalled = self._source_clock_stalled()
        if clock_stalled:
            reasons.append("source_time_stalled")
        if clock and not clock.native:
            reasons.append("nominal_source_time")
        if self.capture.state != "streaming":
            reasons.append(f"video_{self.capture.state}")
        if age is not None and age > self.settings.frame_stale_seconds:
            reasons.append("stale_video")
        for route, state in routes.items():
            if state != "ready":
                reasons.append(f"{route}_{state}")
        analysis_age = time.monotonic() - self.last_analysis_mono if self.last_analysis_mono else None
        if latest and (analysis_age is None or analysis_age > self.settings.frame_stale_seconds):
            reasons.append("analysis_stale")
        if latest:
            reasons.extend(latest.frame.quality)
        if self.telemetry.error:
            reasons.append("analysis_failed")
        scheduler = self.scheduler.stats()
        if scheduler["api_state"] not in ("ready", "mock"):
            reasons.append("cloud_unavailable")
        if self.settings.provider == "mock":
            reasons.append("mock_has_no_semantic_evaluation")
        storage_state = self.storage.stats()
        if storage_state["error"]:
            reasons.append(storage_state["error"])
        if any(o.get("limited") for o in self.overlays):
            reasons.append("small_or_limited_person")
        active = [
            i.model_dump(mode="json")
            for i in sorted(self.manager.items.values(), key=lambda i: i.created_at, reverse=True)
        ]
        return {
            "name": "Stillwatch",
            "version": __version__,
            "mode": self.settings.provider,
            "camera_id": self.settings.camera_id,
            "source": self.source_descriptor().model_dump(mode="json"),
            "capture_state": self.capture.state,
            "frame_age_seconds": age,
            "source_ts": latest.frame.capture_ts if latest else None,
            "source_clock": clock.snapshot() | {"stalled": clock_stalled} if clock else None,
            "frame": {
                "id": latest.frame.frame_id,
                "width": latest.frame.width,
                "height": latest.frame.height,
                "epoch": latest.frame.epoch,
            }
            if latest
            else None,
            "analysis_age_seconds": analysis_age,
            "routes": routes,
            "limitations": list(dict.fromkeys(reasons)),
            "buffer": self.buffer.stats(),
            "scheduler": scheduler,
            "storage": storage_state,
            "performance": self.telemetry.snapshot(),
            "config_hash": self.settings.fingerprint,
            "implementation_hash": self.implementation_hash,
            "events": active[:200],
            "overlays": self.overlays,
            "motion": [m.model_dump(mode="json") for m in self.manager.latest_motion.values()],
            "revision": self.revision,
        }
