from dataclasses import replace

import pytest

from stillwatch.config import Settings
from stillwatch.types import Box, Candidate, Category, Status


@pytest.fixture
def confirmation(system):
    settings, _, _, _, _, _, scheduler = system
    settings.medical_confirmation_frames = 12
    scheduler.last_overview_scheduled = 0
    return system


def candidate(manager, category=Category.MEDICAL, ts=6, epoch=1):
    return manager.ingest(
        Candidate(
            category=category,
            reason="rapid_posture_change",
            first_ts=ts - 0.5,
            last_ts=ts,
            box=Box(x1=10, y1=10, x2=100, y2=100),
            epoch=epoch,
            priority=3,
        )
    )[0]


def append_frame(buffer, scheduler, index, epoch=1, start=6):
    frame = replace(buffer.frames()[-1], frame_id=1000 + index, capture_ts=start + index / 10, epoch=epoch)
    buffer.add(frame)
    scheduler.advance(frame.capture_ts, epoch)


def test_default_confirmation_is_twelve_frames():
    assert Settings().medical_confirmation_frames == 12
    assert Settings().medical_observation_seconds == 15


@pytest.mark.parametrize(
    "status,observation,expected",
    [
        (Status.NO_ALERT, "no_concern", Status.PENDING),
        (Status.REVIEW, "ongoing_concern", Status.PENDING),
        (Status.URGENT, "ongoing_concern", Status.PENDING),
        (Status.URGENT, "immediate_danger", Status.URGENT),
    ],
)
async def test_early_screen_cannot_clear_or_shortcut_observation(confirmation, status, observation, expected):
    settings, buffer, _, _, manager, provider, scheduler = confirmation
    settings.medical_observation_seconds = 15
    original = provider.evaluate

    async def response(packet):
        result = await original(packet)
        result.medical_status, result.medical_observation = status, observation
        result.fire_status = Status.NO_ALERT
        return result

    provider.evaluate = response
    item = candidate(manager)
    scheduler.enqueue(item)
    for index in range(1, 20):
        append_frame(buffer, scheduler, index)
        assert not await scheduler.step()
    append_frame(buffer, scheduler, 20)
    assert await scheduler.step()
    assert item.early_screen_complete and item.analysis_status == expected
    assert item.full_assessment_source_ts is None
    if expected == Status.URGENT:
        return
    for index in range(21, 150):
        append_frame(buffer, scheduler, index)
        assert not await scheduler.step()
    append_frame(buffer, scheduler, 150)
    assert await scheduler.step()
    assert item.full_assessment_source_ts == pytest.approx(21)
    assert item.request_count == 2


async def test_first_medical_request_waits_for_twelve_new_buffer_frames(confirmation):
    _, buffer, storage, _, manager, provider, scheduler = confirmation
    item = candidate(manager)
    scheduler.enqueue(item)
    assert not await scheduler.step()
    for index in range(1, 12):
        append_frame(buffer, scheduler, index)
        assert not await scheduler.step()
    assert item.confirmation_frames_seen == 11
    assert item.analysis_status == Status.PENDING
    assert item.queue_reason == "awaiting_event_development"
    assert not provider.calls and scheduler.budget.snapshot()["requests"] == 0
    assert (
        next(i for i in storage.incidents() if i.incident_id == item.incident_id).confirmation_frames_seen
        == 11
    )

    append_frame(buffer, scheduler, 12)
    assert await scheduler.step()
    assert item.request_count == 1 and len(provider.calls) == 1
    detail = storage.detail(item.incident_id)
    images = detail["packets"][0]["images"]
    assert len(images) == 8  # Waiting for 12 frames does not enlarge the packet.
    assert max(image["capture_ts"] for image in images) == pytest.approx(7.2)
    assert min(image["capture_ts"] for image in images) < item.detected_ts


async def test_elapsed_time_and_repeated_frames_do_not_satisfy_confirmation(confirmation):
    _, buffer, _, _, manager, provider, scheduler = confirmation
    item = candidate(manager)
    scheduler.enqueue(item)
    initial = buffer.frames()[-1]
    # Repeated IDs, even with later timestamps, cannot stand in for new frames.
    for index in range(1, 13):
        buffer.add(replace(initial, frame_id=1000, capture_ts=6 + index / 10))
        scheduler.advance(6 + index / 10, 1)
        assert not await scheduler.step()
    assert item.confirmation_frames_seen == 1
    scheduler.advance(30, 1)
    for _ in range(12):
        assert not await scheduler.step()
    assert item.confirmation_frames_seen == 1 and not provider.calls


async def test_future_and_rejected_cadence_frames_do_not_count(confirmation):
    _, buffer, _, _, manager, _, scheduler = confirmation
    item = candidate(manager)
    scheduler.enqueue(item)
    initial = buffer.frames()[-1]
    buffer.add(replace(initial, frame_id=999, capture_ts=6.01))
    buffer.add(replace(initial, frame_id=1000, capture_ts=6.1))
    assert not await scheduler.step()  # Scheduler has only advanced to 6.0.
    assert item.confirmation_frames_seen == 0
    scheduler.advance(6.1, 1)
    assert item.confirmation_frames_seen == 1


async def test_pending_medical_does_not_block_fire_or_overview(confirmation):
    _, _, _, _, manager, provider, scheduler = confirmation
    medical = candidate(manager)
    scheduler.enqueue(medical)
    fire = candidate(manager, Category.FIRE)
    scheduler.enqueue(fire)
    overview = manager.overview(6, 1)
    scheduler.enqueue(overview)
    assert await scheduler.step()
    assert fire.request_count == 1
    assert await scheduler.step()
    assert overview.request_count == 1
    assert medical.request_count == 0 and len(provider.calls) == 2


async def test_confirmation_progress_survives_buffer_eviction(confirmation):
    _, buffer, _, _, manager, _, scheduler = confirmation
    buffer.seconds = 0.11
    item = candidate(manager)
    scheduler.enqueue(item)
    for index in range(1, 13):
        append_frame(buffer, scheduler, index)
    assert buffer.stats()["frames"] <= 2
    assert item.confirmation_frames_seen == 12
    assert len(scheduler.jobs[item.incident_id].confirmation_frame_ids) == 12
    assert await scheduler.step()


async def test_eof_does_not_turn_incomplete_confirmation_into_an_assessment(confirmation):
    _, buffer, storage, _, manager, _, scheduler = confirmation
    item = candidate(manager)
    scheduler.enqueue(item)
    for index in range(1, 12):
        append_frame(buffer, scheduler, index)
    scheduler.can_send = lambda: False
    for _ in range(12):
        assert not await scheduler.step()
    assert item.analysis_status == Status.PENDING
    assert item.confirmation_frames_seen == 11
    assert not storage.detail(item.incident_id)["assessments"]
    assert scheduler.budget.snapshot()["requests"] == 0


async def test_new_epoch_cannot_complete_old_confirmation(confirmation):
    _, buffer, _, _, manager, _, scheduler = confirmation
    old = candidate(manager)
    scheduler.enqueue(old)
    for index in range(1, 12):
        append_frame(buffer, scheduler, index)
    append_frame(buffer, scheduler, 12, epoch=2)
    assert old.assessment_stale and old.queue_reason == "source_epoch_changed"
    assert old.incident_id not in scheduler.jobs
    new = candidate(manager, ts=7.2, epoch=2)
    scheduler.enqueue(new)
    assert not await scheduler.step()
    assert new.confirmation_frames_seen == 0 and old.request_count == 0


async def test_rechecks_keep_existing_schedule_without_another_confirmation(confirmation):
    _, buffer, _, _, manager, _, scheduler = confirmation
    item = candidate(manager)
    scheduler.enqueue(item)
    for index in range(1, 13):
        append_frame(buffer, scheduler, index)
    assert await scheduler.step()
    assert scheduler.next_due[item.incident_id] == 9
    scheduler.advance(8.9, 1)
    assert not await scheduler.step()
    scheduler.advance(9, 1)
    assert await scheduler.step()
    assert item.request_count == 2


async def test_closing_a_pending_event_releases_its_queue_slot(confirmation):
    _, _, _, _, manager, _, scheduler = confirmation
    item = candidate(manager)
    scheduler.enqueue(item)
    manager.annotate(item.incident_id, "close")
    assert not await scheduler.step()
    assert item.incident_id not in scheduler.jobs


async def test_timed_observation_waits_for_real_frames_and_keeps_first_suspicion(confirmation):
    settings, buffer, storage, _, manager, provider, scheduler = confirmation
    settings.medical_observation_seconds = 15
    settings.medical_early_screen_seconds = 0
    item = candidate(manager)
    scheduler.enqueue(item)
    for index in range(1, 150):
        append_frame(buffer, scheduler, index)
        assert not await scheduler.step()
    assert item.observation_seconds_seen == pytest.approx(14.9)
    assert item.request_count == 0 and not provider.calls
    # Repeated candidates cannot reset the original deadline, even after the dedup interval.
    same = candidate(manager, ts=20)
    assert same is item and item.detected_ts == 6
    append_frame(buffer, scheduler, 150)
    assert await scheduler.step()
    packet = storage.detail(item.incident_id)["packets"][0]
    assert packet["source_ts"] == 21
    assert all(i["capture_ts"] <= 21 and i["kind"] == "overview" for i in packet["images"])
    assert len([i for i in packet["images"] if i["capture_ts"] >= 18]) >= 3
    assert packet["temporal_context"]["elapsed_source_seconds"] == 15
    assert scheduler.next_due[item.incident_id] == 24
    scheduler.advance(21.1, 1)
    assert not await scheduler.step()  # No burst of overdue 3/10-second rechecks.


async def test_frozen_buffer_cannot_complete_the_observation_timer(confirmation):
    settings, buffer, _, _, manager, _, scheduler = confirmation
    settings.medical_observation_seconds = 15
    item = candidate(manager)
    scheduler.enqueue(item)
    for index in range(1, 13):
        append_frame(buffer, scheduler, index)
    scheduler.advance(40, 1)
    assert not await scheduler.step()
    assert item.observation_seconds_seen == pytest.approx(1.2)
    assert item.request_count == 0


async def test_timed_medical_wait_does_not_block_fire_or_overview(confirmation):
    settings, _, _, _, manager, _, scheduler = confirmation
    settings.medical_observation_seconds = 15
    medical = candidate(manager)
    scheduler.enqueue(medical)
    fire = candidate(manager, Category.FIRE)
    scheduler.enqueue(fire)
    overview = manager.overview(6, 1)
    scheduler.enqueue(overview)
    assert await scheduler.step() and fire.request_count == 1
    assert await scheduler.step() and overview.request_count == 1
    assert medical.request_count == 0


def test_temporal_packet_is_unchanged_by_unobserved_future_frames(confirmation):
    settings, buffer, _, evidence, manager, _, scheduler = confirmation
    settings.medical_observation_seconds = 15
    item = candidate(manager)
    for index in range(1, 151):
        append_frame(buffer, scheduler, index)
    before = evidence.build(item, 21)
    latest = buffer.frames()[-1]
    # A preloaded file suffix must never influence the packet at the same clock time.
    buffer.add(replace(latest, frame_id=9999, capture_ts=21.1, jpeg=b"future scene"))
    after = evidence.build(item, 21)
    assert before.images == after.images
    assert before.temporal_context == after.temporal_context
    assert all(image.capture_ts <= 21 for image in after.images)
