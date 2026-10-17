import pytest
from test_episode_memory import answer, event, evidence

from stillwatch.types import Candidate, Category, Lifecycle, Status


def recovered():
    return evidence(
        phase="recovered_activity",
        posture="upright",
        clear_eligible=True,
        observed_ts=6,
        last_seen_ts=6,
        recovery_seconds=3,
    )


def waiting(manager):
    item = event(manager, evidence(observed_ts=6, last_seen_ts=6))
    item.analysis_status = item.medical_status = Status.UNKNOWN
    item.request_count = 2
    item.full_assessment_source_ts = 3
    item.last_assessed_motion_phase = "following_concern"
    return item


@pytest.mark.asyncio
async def test_new_recovery_uses_remaining_allowance_before_a_future_timer(system):
    manager, scheduler = system[4], system[6]
    item = waiting(manager)
    scheduler.next_due[item.incident_id] = 9
    item.motion = recovered()
    scheduler.advance(6, 1, schedule_overview=False)
    job = scheduler.jobs[item.incident_id]
    assert job.due_ts == 6 and job.observed_resolution and not job.resolution
    assert await scheduler.step()
    assert item.request_count == 3 and item.resolution_checks == 0
    packet = manager.storage.detail(item.incident_id)["packets"][-1]
    assert packet["temporal_context"]["observed_resolution_check"]
    scheduler.advance(6.2, 1, schedule_overview=False)
    assert not await scheduler.step()
    assert item.request_count == 3


def test_recovery_moves_an_existing_delayed_job_to_current_source_time(system):
    manager, scheduler = system[4], system[6]
    item = waiting(manager)
    assert scheduler.enqueue(item, 12)
    assert scheduler.jobs[item.incident_id].due_ts == 12
    scheduler.next_due[item.incident_id] = 12
    item.motion = recovered()
    scheduler.advance(6, 1, schedule_overview=False)
    assert scheduler.jobs[item.incident_id].due_ts == 6
    assert scheduler.next_due[item.incident_id] == 12


@pytest.mark.asyncio
async def test_deterioration_cancels_an_advanced_check_and_keeps_the_original_timer(system):
    manager, scheduler = system[4], system[6]
    item = waiting(manager)
    item.motion = recovered()
    scheduler.next_due[item.incident_id] = 12
    scheduler.advance(6, 1, schedule_overview=False)
    item.motion = evidence(observed_ts=6, last_seen_ts=6)
    assert not await scheduler.step()
    assert item.incident_id not in scheduler.jobs
    assert scheduler.next_due[item.incident_id] == 12
    assert item.request_count == 2 and scheduler.budget.snapshot()["requests"] == 0


def test_recovery_cannot_advance_the_first_full_observation_window(system):
    settings, _, _, _, manager, _, scheduler = system
    settings.medical_observation_seconds = 15
    item = waiting(manager)
    item.request_count = 1
    item.full_assessment_source_ts = None
    item.early_screen_complete = True
    item.motion = recovered()
    scheduler.next_due[item.incident_id] = 20
    scheduler.advance(6, 1, schedule_overview=False)
    assert item.incident_id not in scheduler.jobs


@pytest.mark.parametrize(
    ("basis", "posture"),
    [("upright_activity", "upright"), ("prior_supported_posture", "seated_on_furniture")],
)
def test_confirmed_recovery_closes_before_visibility_loss_without_operator_feedback(system, basis, posture):
    _, _, storage, builder, manager, _, _ = system
    item = waiting(manager)
    item.motion = recovered()
    item.motion.recovery_basis = basis
    item.motion.posture = "seated" if posture == "seated_on_furniture" else "upright"
    result = answer(item, "recovered")
    result.latest_posture = posture
    manager.apply(result, 6)
    assert item.lifecycle == Lifecycle.CLOSED and item.closed_source_ts == 6
    assert item.incident_id not in builder.onset
    manager.observe_motion(
        [
            item.motion.model_copy(
                update={"phase": "visibility_lost", "clear_eligible": False, "observed_ts": 7}
            )
        ],
        7,
    )
    assert item.analysis_status == Status.NO_ALERT
    original_motion = item.motion.model_copy(deep=True)
    duplicate, created = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="view_lost_after_suspicion",
            first_ts=2,
            last_ts=7,
            track_id=1,
            epoch=1,
            motion=item.motion.model_copy(
                update={"phase": "visibility_lost", "clear_eligible": False, "observed_ts": 7}
            ),
        )
    )
    assert not created and duplicate.incident_id == item.incident_id
    assert item.motion == original_motion and item.last_ts == 2
    assert storage.db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_new_descent_during_recovery_request_cannot_close_the_episode(system):
    manager, provider, scheduler = system[4], system[5], system[6]
    item = waiting(manager)
    item.motion = recovered()

    async def delayed_recovery(packet):
        manager.observe_motion([evidence(observed_ts=6.2, last_seen_ts=6.2)], 6.2)
        scheduler.advance(6.2, 1, schedule_overview=False)
        result = answer(item, "recovered")
        result.packet_id, result.version = packet.packet_id, packet.version
        result.latest_posture = "upright"
        return result

    provider.evaluate = delayed_recovery
    assert scheduler.enqueue(item)
    assert await scheduler.step()
    assert item.analysis_status == Status.UNKNOWN and item.lifecycle == Lifecycle.OPEN
    assert item.policy_reason == "local_episode_unresolved"
    assert item.request_count == 3 and item.resolution_checks == 0


def test_new_fall_after_confirmed_recovery_keeps_a_separate_fresh_episode(system):
    manager = system[4]
    previous = waiting(manager)
    previous.motion = recovered()
    result = answer(previous, "recovered")
    result.latest_posture = "upright"
    manager.apply(result, 6)
    motion = evidence(
        episode_id="later-fall",
        previous_episode_id=previous.motion.episode_id,
        started_ts=6.2,
        observed_ts=6.4,
        last_seen_ts=6.4,
    )
    current, created = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=6.2,
            last_ts=6.4,
            track_id=1,
            epoch=1,
            motion=motion,
        )
    )
    assert created and current.incident_id != previous.incident_id
    assert current.request_count == 0 and current.analysis_status == Status.PENDING
    assert previous.lifecycle == Lifecycle.CLOSED and previous.analysis_status == Status.NO_ALERT
    assert not previous.assessment_stale and previous.superseded_by is None


@pytest.mark.parametrize("change", [{"epoch": 2}, {"observed_ts": 7}, {"last_seen_ts": 7}])
def test_recovery_from_another_source_or_future_time_cannot_close(system, change):
    manager = system[4]
    item = waiting(manager)
    item.motion = recovered().model_copy(update=change)
    result = answer(item, "recovered")
    result.latest_posture = "upright"
    manager.apply(result, 6)
    assert item.analysis_status == Status.UNKNOWN and item.lifecycle == Lifecycle.OPEN
    assert item.policy_reason == "local_motion_source_mismatch"


@pytest.mark.parametrize("posture", ["lying_floor", "unseen", "seated_on_furniture"])
def test_cloud_posture_must_agree_with_the_observed_recovery_basis(system, posture):
    manager = system[4]
    item = waiting(manager)
    item.motion = recovered()
    result = answer(item, "recovered")
    result.latest_posture = posture
    manager.apply(result, 6)
    assert item.analysis_status == Status.UNKNOWN and item.lifecycle == Lifecycle.OPEN


def test_automatic_archival_does_not_record_a_user_action(system):
    manager, storage = system[4], system[2]
    item = manager.overview(0, 1)
    item.analysis_status = Status.NO_ALERT
    manager.archive_clear(6, 1)
    assert item.lifecycle == Lifecycle.CLOSED
    assert storage.db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
    manual = manager.overview(7, 1)
    manager.annotate(manual.incident_id, "close")
    assert storage.db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
