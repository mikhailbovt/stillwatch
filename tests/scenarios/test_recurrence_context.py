from dataclasses import replace

import pytest
from test_episode_handoff import next_descent, previous_event

from stillwatch.incidents import IncidentManager
from stillwatch.scheduler import Scheduler
from stillwatch.types import Candidate, Category, Lifecycle, Status


def repeat(manager, previous, ts):
    return manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=ts - 0.2,
            last_ts=ts,
            epoch=1,
            track_id=1,
            box=previous.box,
            motion=previous.motion.model_copy(
                update={
                    "episode_id": f"episode-{ts}",
                    "previous_episode_id": previous.motion.episode_id,
                    "started_ts": ts - 0.2,
                    "observed_ts": ts,
                    "last_seen_ts": ts,
                }
            ),
        )
    )[0]


def frames(system, start, stop):
    buffer, scheduler = system[1], system[6]
    template = buffer.frames()[-1]
    for n in range(round((stop - start) * 5) + 1):
        ts = round(start + n / 5, 6)
        buffer.add(replace(template, frame_id=1000 + round(ts * 5), capture_ts=ts))
        scheduler.advance(ts, 1, schedule_overview=False)


@pytest.mark.asyncio
async def test_repeated_descents_keep_the_first_unresolved_clock_and_their_own_frames(system):
    settings, _, storage, builder, manager, _, scheduler = system
    settings.medical_observation_seconds = 15
    settings.medical_confirmation_frames = 12
    settings.medical_early_screen_seconds = 0
    first = previous_event(manager)
    second = next_descent(manager, first)
    frames(system, 6.2, 12.2)
    third = repeat(manager, second, 12.4)
    assert third.observation_started_ts == first.detected_ts == 2
    assert third.recurrence_count == 2
    scheduler.advance(12.4, 1, schedule_overview=False)
    assert third.confirmation_frames_seen == 0
    assert not await scheduler.step()
    frames(system, 12.6, 16.8)
    assert third.confirmation_frames_seen == 12
    assert not await scheduler.step()
    frames(system, 17, 17)
    assert await scheduler.step()
    assert third.full_assessment_source_ts == 17
    assert third.request_count == 1 and first.request_count == second.request_count == 0
    packet = storage.detail(third.incident_id)["packets"][-1]
    context = packet["temporal_context"]
    assert context["observation_started_source_ts"] == 2 and context["recurrence_count"] == 2
    assert any(i["capture_ts"] <= first.detected_ts + 1.2 for i in packet["images"])
    assert all(i["capture_ts"] <= 17 for i in packet["images"])
    assert len([i for i in packet["images"] if i["kind"] == "overview"]) <= 12
    assert first.incident_id not in builder.onset and second.incident_id not in builder.onset


@pytest.mark.asyncio
async def test_recurrence_clock_cannot_advance_without_new_accepted_frames(system):
    settings, _, _, _, manager, _, scheduler = system
    settings.medical_observation_seconds = 15
    settings.medical_confirmation_frames = 12
    settings.medical_early_screen_seconds = 0
    current = next_descent(manager, previous_event(manager))
    scheduler.advance(100, 1, schedule_overview=False)
    assert not await scheduler.step()
    assert current.observation_seconds_seen < 15 and current.request_count == 0


@pytest.mark.parametrize("boundary", ["closed", "identity", "source"])
def test_an_independent_or_unproven_event_cannot_inherit_a_clock(system, boundary):
    manager = system[4]
    previous = previous_event(manager)
    changes = {}
    if boundary == "closed":
        previous.lifecycle = Lifecycle.CLOSED
    elif boundary == "identity":
        changes["identity_uncertain"] = True
    else:
        changes["epoch"] = 2
    current = next_descent(manager, previous, **changes)
    assert current.observation_started_ts is None and current.recurrence_count == 0


def test_recurrence_clock_survives_restart_without_inheriting_request_counts(system):
    settings, _, storage, builder, manager, provider, _ = system
    previous = previous_event(manager)
    previous.request_count = 3
    previous.analysis_status = previous.medical_status = Status.UNKNOWN
    current = next_descent(manager, previous)
    restored = IncidentManager(settings, storage, builder)
    item = restored.items[current.incident_id]
    scheduler = Scheduler(settings, restored, provider)
    scheduler.advance(6, 1, schedule_overview=False)
    assert item.observation_started_ts == 2 and item.recurrence_count == 1
    assert item.request_count == 0 and scheduler.enqueue(item)


def test_inherited_onset_obeys_the_shared_memory_cap(system):
    _, _, _, builder, manager, _, _ = system
    previous = previous_event(manager)
    builder.onset_max_bytes = sum(len(f.jpeg) for f in builder.onset[previous.incident_id])
    current = next_descent(manager, previous)
    assert sum(len(f.jpeg) for frames in builder.onset.values() for f in frames) <= builder.onset_max_bytes
    assert len(builder.onset[current.incident_id]) <= 24


@pytest.mark.asyncio
async def test_early_screen_schedules_full_assessment_without_another_candidate(system):
    settings, _, _, _, manager, _, scheduler = system
    settings.medical_observation_seconds = 15
    settings.medical_confirmation_frames = 12
    current = previous_event(manager)
    scheduler.advance(6, 1, schedule_overview=False)
    assert await scheduler.step()
    assert current.early_screen_complete and current.full_assessment_source_ts is None
    assert scheduler.next_due[current.incident_id] == 17
    frames(system, 6.2, 16.8)
    assert not await scheduler.step()
    frames(system, 17, 17)
    assert await scheduler.step()
    assert current.full_assessment_source_ts == 17 and current.request_count == 2
