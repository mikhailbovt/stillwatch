from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import Settings
from .decisions import DecisionProvider
from .incidents import IncidentManager
from .storage import Storage
from .types import Category, DecisionResult, Incident, Lifecycle, Status


class Budget:
    def __init__(self, settings: Settings, storage: Storage, provider: str):
        self.settings, self.storage, self.provider = settings, storage, provider

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        day = datetime.fromtimestamp(now, UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        recent = self.storage.calls(self.provider, min(day, now - 60))
        calls = [c for c in recent if c["started"] >= day]
        return {
            "day_utc": datetime.fromtimestamp(day, UTC).isoformat(),
            "requests": len(calls),
            "minute_requests": sum(c["started"] > now - 60 for c in recent),
            "input_tokens": sum(c["tokens"] if c["tokens"] is not None else c["reserved"] for c in calls),
            "unknown_usage_requests": sum(c["tokens"] is None for c in calls),
            "provider": self.provider,
            "usage_kind": "measured" if self.provider == "openai" else "simulated",
            "reservation_is_upper_bound": False,
        }

    def reserve(self, now: float | None = None) -> tuple[int | None, str | None]:
        now = time.time() if now is None else now
        state = self.snapshot(now)
        if state["minute_requests"] >= self.settings.requests_per_minute:
            return None, "minute_request_limit"
        if state["requests"] >= self.settings.requests_per_day:
            return None, "daily_request_limit"
        if state["input_tokens"] + self.settings.token_reservation > self.settings.input_tokens_per_day:
            return None, "daily_token_limit"
        identifier = self.storage.reserve_call(self.provider, self.settings.token_reservation, now)
        return (identifier, None) if identifier is not None else (None, "budget_persistence_failed")


@dataclass
class Job:
    incident_id: str
    enqueued_mono: float
    due_ts: float
    confirmation_frame_ids: set[int] = field(default_factory=set)
    observation_until_ts: float | None = None
    last_observed_frame_id: int = -1
    resolution: bool = False
    observed_resolution: bool = False


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        manager: IncidentManager,
        provider: DecisionProvider,
        can_send: Callable[[], bool] | None = None,
    ):
        self.settings, self.manager, self.provider = settings, manager, provider
        self.can_send = can_send
        self.budget = Budget(settings, manager.storage, provider.name)
        self.jobs: dict[str, Job] = {}
        self.next_due: dict[str, float] = {}
        self.transport_retries: dict[str, int] = {}
        self.inflight: str | None = None
        self.source_ts = 0.0
        self.elapsed_source = 0.0
        self.epoch = 0
        self.last_overview_scheduled: float | None = None
        self.last_overview_completed_mono: float | None = None
        self.last_category: Category | None = None
        self.failure_count = 0
        self.pause_until = 0.0
        self.api_state = "mock" if provider.name == "mock" else "ready"
        self.over_capacity = 0
        self._stopped = False

    def _resolution_observed(self, item: Incident) -> bool:
        motion = item.motion
        return bool(
            item.category == Category.MEDICAL
            and item.epoch == self.epoch
            and item.request_count > 0
            and item.full_assessment_source_ts is not None
            and item.analysis_status in (Status.UNKNOWN, Status.REVIEW, Status.UNAVAILABLE)
            and item.last_assessed_motion_phase not in ("recovered_activity", "ordinary_activity")
            and motion
            and motion.clear_eligible
            and motion.phase in ("recovered_activity", "ordinary_activity")
            and not (item.identity_uncertain or motion.identity_uncertain)
            and motion.epoch == self.epoch
            and 0 <= self.source_ts - motion.last_seen_ts <= self.settings.lost_track_seconds + 1e-6
            and 0 <= self.source_ts - motion.observed_ts <= self.settings.lost_track_seconds + 1e-6
        )

    def _resolution_ready(self, item: Incident) -> bool:
        return (
            self._resolution_observed(item)
            and item.request_count >= self.settings.max_episode_requests
            and item.resolution_checks == 0
        )

    def enqueue(self, item: Incident, due_ts: float | None = None) -> bool:
        if item.incident_id == self.inflight or item.lifecycle == Lifecycle.CLOSED or item.superseded_by:
            return False
        resolution = self._resolution_ready(item)
        observed_resolution = self._resolution_observed(item)
        if (
            item.request_count >= self.settings.max_episode_requests and not resolution
        ) or item.analysis_status == Status.URGENT:
            return False
        if item.incident_id in self.jobs:
            job = self.jobs[item.incident_id]
            if observed_resolution:
                job.due_ts = min(job.due_ts, self.source_ts)
            elif job.observed_resolution and due_ts is not None:
                job.due_ts = due_ts
            job.resolution = resolution
            job.observed_resolution = observed_resolution
            self._confirmation_ready(job)
            return True
        if len(self.jobs) >= self.settings.queue_capacity:
            if item.category != Category.OVERVIEW:
                overview = next(
                    (key for key in self.jobs if self.manager.items[key].category == Category.OVERVIEW), None
                )
                if overview:
                    self.jobs.pop(overview)
                    old = self.manager.items[overview]
                    old.queue_reason = "overview_deferred"
                    self.manager.storage.save_incident(old)
                else:
                    item.queue_reason = "queue_capacity_wait"
                    self.manager.storage.save_incident(item)
                    self.over_capacity += 1
                    return False
            else:
                item.queue_reason = "overview_deferred"
                return False
        self.jobs[item.incident_id] = Job(
            item.incident_id,
            time.monotonic(),
            self.source_ts if observed_resolution or due_ts is None else due_ts,
            resolution=resolution,
            observed_resolution=observed_resolution,
        )
        item.queue_reason = "queued"
        self._confirmation_ready(self.jobs[item.incident_id])
        self.manager.storage.save_incident(item)
        return True

    def _confirmation_ready(self, job: Job) -> bool:
        item = self.manager.items[job.incident_id]
        if item.category != Category.MEDICAL or item.full_assessment_source_ts is not None:
            return True
        required = self.settings.medical_confirmation_frames
        before = (
            item.confirmation_frames_seen,
            item.confirmation_frames_required,
            item.observation_seconds_seen,
            item.observation_seconds_required,
            item.queue_reason,
        )
        # Count distinct accepted buffer frames, not decoded-frame IDs or elapsed time.
        # Keep bounded progress when older frames leave the ring while a job waits.
        for frame in self.manager.evidence.buffer.frames(since=item.detected_ts):
            if (
                frame.epoch == item.epoch
                and item.detected_ts < frame.capture_ts <= self.source_ts
                and frame.frame_id > job.last_observed_frame_id
            ):
                if len(job.confirmation_frame_ids) < required:
                    job.confirmation_frame_ids.add(frame.frame_id)
                job.observation_until_ts = max(job.observation_until_ts or item.detected_ts, frame.capture_ts)
                job.last_observed_frame_id = frame.frame_id
        item.confirmation_frames_seen = min(required, len(job.confirmation_frame_ids))
        item.confirmation_frames_required = required
        item.observation_seconds_required = self.settings.medical_observation_seconds
        observation_start = (
            item.detected_ts
            if item.observation_started_ts is None
            else min(item.detected_ts, item.observation_started_ts)
        )
        item.observation_seconds_seen = min(
            item.observation_seconds_required,
            max(0, (job.observation_until_ts or item.detected_ts) - observation_start),
        )
        ready = (
            item.confirmation_frames_seen >= required
            and item.observation_seconds_seen + 1e-6 >= item.observation_seconds_required
        )
        early = (
            not item.early_screen_complete
            and self.settings.max_episode_requests > 1
            and item.reason == "rapid_posture_change"
            and 0 < self.settings.medical_early_screen_seconds < self.settings.medical_observation_seconds
            and item.confirmation_frames_seen >= required
            and item.observation_seconds_seen >= self.settings.medical_early_screen_seconds
        )
        if not ready:
            item.queue_reason = "awaiting_event_development"
        elif item.queue_reason == "awaiting_event_development":
            item.queue_reason = "queued"
        if before != (
            item.confirmation_frames_seen,
            item.confirmation_frames_required,
            item.observation_seconds_seen,
            item.observation_seconds_required,
            item.queue_reason,
        ):
            self.manager.storage.save_incident(item)
        return ready or early

    def change_source(self, epoch: int):
        """Invalidate the old source immediately; assess the new one after its first frame."""
        self.advance(0, epoch, schedule_overview=False)
        self.last_overview_scheduled = None
        self.last_overview_completed_mono = None
        if self.api_state == "no_evidence":
            self.api_state = "mock" if self.provider.name == "mock" else "ready"

    def advance(self, source_ts: float, epoch: int, *, schedule_overview: bool = True):
        if epoch == self.epoch and source_ts < self.source_ts:
            return
        self.elapsed_source += (
            max(0, source_ts - self.source_ts) if epoch == self.epoch else max(0, source_ts)
        )
        if epoch != self.epoch:
            self.jobs.clear()
            self.next_due.clear()
            for item in self.manager.items.values():
                if item.lifecycle != Lifecycle.CLOSED and item.epoch != epoch and not item.assessment_stale:
                    item.assessment_stale = True
                    item.queue_reason = "source_epoch_changed"
                    self.manager.storage.save_incident(item)
            self.epoch = epoch
        self.source_ts = source_ts
        pending_overview = any(self.manager.items[k].category == Category.OVERVIEW for k in self.jobs)
        if (
            schedule_overview
            and not pending_overview
            and (
                self.last_overview_scheduled is None
                or self.elapsed_source - self.last_overview_scheduled >= self.settings.overview_seconds
            )
        ):
            item = self.manager.overview(source_ts, epoch)
            self.enqueue(item)
            self.last_overview_scheduled = self.elapsed_source
        for item in sorted(self.manager.items.values(), key=lambda i: i.created_at):
            if item.superseded_by:
                self.jobs.pop(item.incident_id, None)
                self.next_due.pop(item.incident_id, None)
                self.transport_retries.pop(item.incident_id, None)
                continue
            if (
                item.lifecycle == Lifecycle.CLOSED
                or item.epoch != epoch
                or item.category == Category.OVERVIEW
            ):
                continue
            due = self.next_due.get(item.incident_id)
            if (
                item.request_count == 0
                or (due is not None and source_ts >= due)
                or self._resolution_observed(item)
            ):
                self.enqueue(item, due)
        self.manager.archive_clear(source_ts, epoch)

    def _choose(self) -> Job | None:
        now = time.monotonic()
        for key in list(self.jobs):
            item = self.manager.items[key]
            if item.lifecycle == Lifecycle.CLOSED or item.epoch != self.epoch or item.superseded_by:
                self.jobs.pop(key)
                self.next_due.pop(key, None)
            elif (self.jobs[key].observed_resolution and not self._resolution_observed(item)) or (
                self.jobs[key].resolution and not self._resolution_ready(item)
            ):
                self.jobs.pop(key)
        eligible = [
            j for j in self.jobs.values() if j.due_ts <= self.source_ts and self._confirmation_ready(j)
        ]
        if not eligible:
            return None

        def score(job: Job):
            item = self.manager.items[job.incident_id]
            fairness = 2 if item.category != self.last_category else 0
            return item.priority * 3 + min(120, now - job.enqueued_mono) + fairness - item.request_count

        return max(eligible, key=score)

    async def step(self) -> bool:
        if (
            self.inflight
            or self._stopped
            or time.monotonic() < self.pause_until
            or self.api_state == "access_denied"
        ):
            return False
        job = self._choose()
        if job is None:
            return False
        item = self.manager.items[job.incident_id]
        if item.lifecycle == Lifecycle.CLOSED or item.epoch != self.epoch:
            self.jobs.pop(item.incident_id, None)
            return False
        if self.can_send is not None and not self.can_send():
            if item.queue_reason != "source_not_fresh":
                item.queue_reason = "source_not_fresh"
                self.manager.storage.save_incident(item)
            return False
        # Source replacement/opening can race with capture's next epoch. Do not
        # consume a request or report a provider failure before evidence exists.
        packet = self.manager.evidence.build(
            item.model_copy(update={"packet_version": item.packet_version + 1}), self.source_ts
        )
        if not packet.images:
            if item.queue_reason != "awaiting_evidence":
                item.queue_reason = "awaiting_evidence"
                self.manager.storage.save_incident(item)
            return False
        call_id, reason = self.budget.reserve()
        if reason:
            item.queue_reason = reason
            self.api_state = reason
            self.manager.storage.save_incident(item)
            return False
        self.jobs.pop(item.incident_id, None)
        self.inflight = item.incident_id
        early_screen = (
            item.category == Category.MEDICAL
            and item.observation_seconds_seen + 1e-6 < self.settings.medical_observation_seconds
        )
        if early_screen:
            item.early_screen_complete = True
        elif item.full_assessment_source_ts is None:
            item.full_assessment_source_ts = self.source_ts
        if item.first_assessment_source_ts is None:
            item.first_assessment_source_ts = self.source_ts
        item.packet_version += 1
        item.request_count += 1
        item.resolution_checks += int(job.resolution)
        item.last_assessed_motion_phase = item.motion.phase if item.motion else None
        item.queue_reason = "analyzing"
        self.manager.storage.save_incident(item)
        # Always rebuild from the current buffer, retaining separately pinned onset frames.
        packet.temporal_context["early_danger_screen"] = early_screen
        packet.temporal_context["observed_resolution_check"] = job.observed_resolution
        item.evidence_missing = not self.manager.storage.save_packet(packet)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(self.provider.evaluate(packet), self.settings.request_timeout)
            if (
                result.incident_id != item.incident_id
                or result.packet_id != packet.packet_id
                or result.version != packet.version
            ):
                result = DecisionResult(
                    incident_id=item.incident_id,
                    packet_id=packet.packet_id,
                    version=packet.version,
                    provider=self.provider.name,
                    model="unknown",
                    error="response_binding_mismatch",
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result = DecisionResult(
                incident_id=item.incident_id,
                packet_id=packet.packet_id,
                version=packet.version,
                provider=self.provider.name,
                model="unknown",
                medical_status=Status.UNAVAILABLE,
                fire_status=Status.UNAVAILABLE,
                error=type(error).__name__,
                retryable=isinstance(error, TimeoutError),
            )
        finally:
            self.inflight = None
        result.latency_ms = (time.monotonic() - started) * 1000
        if item.superseded_by:
            result.stale = True
        elif item.epoch != self.epoch or (self.can_send is not None and not self.can_send()):
            result.stale = True
            item.assessment_stale = True
            item.queue_reason = "response_source_not_fresh"
            self.manager.storage.save_incident(item)
        else:
            if (
                early_screen
                and not result.error
                and not (
                    result.medical_status == Status.URGENT
                    and result.medical_observation == "immediate_danger"
                )
            ):
                # A preliminary answer cannot shorten the recovery observation or clear concern.
                result.medical_status = Status.PENDING
                result.policy_reason = "early_screen_observation_pending"
            self.manager.apply(result, self.source_ts)
        self.manager.storage.save_result(packet, result)
        self.manager.storage.finish_call(call_id, result.input_tokens)
        self.last_category = item.category
        self.next_due.pop(item.incident_id, None)
        if early_screen and item.analysis_status != Status.URGENT:
            self.next_due[item.incident_id] = item.detected_ts + self.settings.medical_observation_seconds
        if result.stale and item.epoch == self.epoch:
            self.next_due[item.incident_id] = self.source_ts
        if item.category == Category.OVERVIEW and not result.stale:
            self.last_overview_completed_mono = time.monotonic()
        if result.error:
            self.failure_count += 1
            self.api_state = result.error
            if result.error == "access_denied":
                self.pause_until = float("inf")
            elif self.failure_count >= 3:
                self.pause_until = time.monotonic() + 30
            used = self.transport_retries.get(item.incident_id, 0)
            if result.retryable and used < self.settings.retries:
                self.transport_retries[item.incident_id] = used + 1
                self.next_due[item.incident_id] = self.source_ts + 1
        elif not result.stale:
            self.failure_count = 0
            self.api_state = "mock" if self.provider.name == "mock" else "ready"
            if early_screen:
                anchor = (
                    item.detected_ts
                    if item.observation_started_ts is None
                    else min(item.detected_ts, item.observation_started_ts)
                )
                self.next_due[item.incident_id] = anchor + self.settings.medical_observation_seconds
            contradictory = (
                item.analysis_status == Status.NO_ALERT
                and self.source_ts - item.last_ts < 1
                and not item.ordinary_activity_confirmed
            )
            if (
                not early_screen
                and not job.resolution
                and (item.analysis_status in (Status.UNKNOWN, Status.REVIEW) or contradictory)
                and item.category != Category.OVERVIEW
            ):
                index = item.request_count - 1 - int(item.early_screen_complete)
                if index < len(self.settings.recheck_seconds):
                    anchor = (
                        item.full_assessment_source_ts
                        if item.category == Category.MEDICAL and self.settings.medical_observation_seconds
                        else item.detected_ts
                    )
                    self.next_due[item.incident_id] = anchor + self.settings.recheck_seconds[index]
        if item.superseded_by:
            self.next_due.pop(item.incident_id, None)
            self.transport_retries.pop(item.incident_id, None)
        return True

    async def run(self):
        while not self._stopped:
            await self.step()
            await asyncio.sleep(0.05)

    async def stop(self):
        self._stopped = True
        await self.provider.close()

    def stats(self) -> dict:
        return {
            "queue_size": len(self.jobs),
            "capacity": self.settings.queue_capacity,
            "inflight": self.inflight,
            "api_state": self.api_state,
            "waiting_events": sum(
                i.queue_reason == "queue_capacity_wait" for i in self.manager.items.values()
            ),
            "oldest_queue_seconds": max(
                (time.monotonic() - j.enqueued_mono for j in self.jobs.values()), default=0
            ),
            "overview_age_seconds": time.monotonic() - self.last_overview_completed_mono
            if self.last_overview_completed_mono is not None
            else None,
            "budget": self.budget.snapshot(),
        }
