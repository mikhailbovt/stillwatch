import asyncio

import pytest
from test_episode_memory import event, evidence

from stillwatch.incidents import IncidentManager
from stillwatch.scheduler import Scheduler
from stillwatch.types import Candidate, Category, DecisionResult, Status


def previous_event(manager):
    return event(manager, evidence(observed_ts=2, last_seen_ts=2))


def next_descent(manager, previous, **changes):
    motion = evidence(
        episode_id="next-episode",
        previous_episode_id=previous.motion.episode_id,
        started_ts=5.2,
        observed_ts=5.4,
        last_seen_ts=5.4,
    ).model_copy(update=changes)
    return manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=5.2,
            last_ts=5.4,
            epoch=motion.epoch,
            track_id=motion.track_id,
            box=previous.box,
            motion=motion,
        )
    )[0]


@pytest.mark.asyncio
async def test_repeat_descent_replaces_queued_work_without_losing_local_evidence(system):
    settings, _, storage, builder, manager, provider, scheduler = system
    previous = previous_event(manager)
    assert scheduler.enqueue(previous)
    current = next_descent(manager, previous)
    scheduler.advance(6, 1, schedule_overview=False)
    assert previous.incident_id not in scheduler.jobs
    assert current.incident_id in scheduler.jobs
    assert previous.superseded_by == current.incident_id
    assert previous.assessment_stale and previous.analysis_status == Status.UNKNOWN
    assert previous.request_count == 0 and previous.feedback is None
    assert previous.incident_id not in builder.onset
    packet = storage.detail(previous.incident_id)["packets"][0]
    assert packet["temporal_context"]["archived_without_assessment"]
    assert packet["images"] and all(i["capture_ts"] <= 5.4 for i in packet["images"])
    assert not storage.detail(previous.incident_id)["assessments"]
    assert scheduler.budget.snapshot()["requests"] == 0
    assert await scheduler.step()
    assert len(provider.calls) == 1 and current.request_count == 1
    restored = IncidentManager(settings, storage, builder)
    after_restart = Scheduler(settings, restored, provider)
    after_restart.advance(6, 1, schedule_overview=False)
    assert previous.incident_id not in after_restart.jobs
    assert restored.items[previous.incident_id].superseded_by == current.incident_id


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [Status.NO_ALERT, Status.REVIEW, Status.URGENT])
async def test_response_after_a_later_episode_stays_in_history_and_is_not_retried(system, status):
    _, _, storage, _, manager, provider, scheduler = system
    previous = previous_event(manager)
    started, finish = asyncio.Event(), asyncio.Event()

    async def delayed(packet):
        started.set()
        await finish.wait()
        return DecisionResult(
            incident_id=packet.incident_id,
            packet_id=packet.packet_id,
            version=packet.version,
            provider="mock",
            model="fixture",
            medical_status=status,
            fire_status=Status.NO_ALERT,
            medical_observation="immediate_danger" if status == Status.URGENT else "no_concern",
            input_tokens=100,
        )

    provider.evaluate = delayed
    assert scheduler.enqueue(previous)
    request = asyncio.create_task(scheduler.step())
    await started.wait()
    current = next_descent(manager, previous)
    finish.set()
    assert await request
    assert previous.analysis_status == Status.UNKNOWN
    assert current.analysis_status == Status.PENDING and current.request_count == 0
    assert previous.incident_id not in scheduler.next_due
    assert not scheduler.enqueue(previous)
    result = storage.detail(previous.incident_id)["assessments"][0]
    assert result["stale"] and result["medical_status"] == status
    assert scheduler.budget.snapshot()["input_tokens"] == 100


@pytest.mark.parametrize(
    "changes",
    [
        {"previous_episode_id": None},
        {"track_id": 2},
        {"epoch": 2},
        {"identity_uncertain": True},
        {"started_ts": 0},
    ],
)
def test_unproven_identity_or_missing_activity_boundary_cannot_retire_an_episode(system, changes):
    manager = system[4]
    previous = previous_event(manager)
    next_descent(manager, previous, **changes)
    assert previous.superseded_by is None and not previous.assessment_stale


def test_existing_urgent_alert_survives_handoff_until_operator_action(system):
    _, _, storage, _, manager, _, scheduler = system
    previous = previous_event(manager)
    previous.analysis_status = previous.medical_status = Status.URGENT
    previous.request_count = 1
    current = next_descent(manager, previous)
    assert previous.superseded_by == current.incident_id
    assert previous.analysis_status == Status.URGENT and not previous.assessment_stale
    assert previous.lifecycle == "OPEN" and previous.feedback is None
    assert not scheduler.enqueue(previous)
    manager.annotate(previous.incident_id, "close")
    assert next(i for i in storage.incidents() if i.incident_id == previous.incident_id).lifecycle == "CLOSED"


def test_repeated_observations_do_not_rearchive_or_repin_an_earlier_episode(system):
    _, _, storage, builder, manager, _, scheduler = system
    previous = previous_event(manager)
    current = next_descent(manager, previous)
    for _ in range(10):
        assert next_descent(manager, previous).incident_id == current.incident_id
        builder.pin(previous)
        scheduler.advance(6, 1, schedule_overview=False)
    assert len(storage.detail(previous.incident_id)["packets"]) == 1
    assert previous.incident_id not in builder.onset
    assert previous.incident_id not in scheduler.jobs


def test_snapshot_failure_is_visible_and_does_not_prevent_current_event(system, monkeypatch):
    _, _, _, builder, manager, _, scheduler = system
    previous = previous_event(manager)
    monkeypatch.setattr(manager.storage, "save_packet", lambda packet: False)
    current = next_descent(manager, previous)
    assert previous.evidence_missing
    assert previous.incident_id not in builder.onset
    assert scheduler.enqueue(current)


@pytest.mark.parametrize("scope", ["incident", "motion"])
def test_uncertain_predecessor_cannot_be_retired_by_a_confident_current_pose(system, scope):
    manager = system[4]
    previous = previous_event(manager)
    if scope == "incident":
        previous.identity_uncertain = True
    else:
        previous.motion.identity_uncertain = True
    next_descent(manager, previous)
    assert previous.superseded_by is None


@pytest.mark.asyncio
async def test_local_snapshot_does_not_bypass_new_episode_observation(system):
    settings, _, _, _, manager, _, scheduler = system
    settings.medical_observation_seconds = 15
    settings.medical_confirmation_frames = 12
    previous = previous_event(manager)
    current = next_descent(manager, previous)
    scheduler.advance(6, 1, schedule_overview=False)
    assert not await scheduler.step()
    assert current.full_assessment_source_ts is previous.full_assessment_source_ts is None
    assert current.request_count == previous.request_count == 0
    assert current.queue_reason == "awaiting_event_development"
    assert scheduler.budget.snapshot()["requests"] == 0


@pytest.mark.asyncio
async def test_failed_inflight_request_cannot_resurrect_superseded_work(system):
    _, _, _, _, manager, provider, scheduler = system
    previous = previous_event(manager)

    async def failure(packet):
        next_descent(manager, previous)
        return DecisionResult(
            incident_id=packet.incident_id,
            packet_id=packet.packet_id,
            version=packet.version,
            provider="mock",
            model="fixture",
            medical_status=Status.UNAVAILABLE,
            error="timeout",
            retryable=True,
        )

    provider.evaluate = failure
    assert scheduler.enqueue(previous) and await scheduler.step()
    scheduler.advance(7, 1, schedule_overview=False)
    assert previous.incident_id not in scheduler.next_due
    assert previous.incident_id not in scheduler.jobs
    assert previous.incident_id not in scheduler.transport_retries


def test_handoff_reclaims_pinned_memory_before_preserving_the_new_onset(system):
    _, _, _, builder, manager, _, _ = system
    previous = previous_event(manager)
    builder.onset_max_bytes = sum(len(f.jpeg) for f in builder.onset[previous.incident_id])
    current = next_descent(manager, previous)
    assert previous.incident_id not in builder.onset
    assert current.incident_id in builder.onset and not current.evidence_missing
