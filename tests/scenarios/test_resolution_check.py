import pytest
from test_episode_memory import event, evidence

from stillwatch.incidents import IncidentManager
from stillwatch.types import DecisionResult, Status


def exhausted(manager):
    item = event(manager, evidence(observed_ts=6, last_seen_ts=6))
    item.analysis_status = item.medical_status = Status.UNKNOWN
    item.request_count = 3
    item.full_assessment_source_ts = 3
    item.last_assessed_motion_phase = "following_concern"
    return item


@pytest.mark.asyncio
async def test_late_observed_recovery_gets_one_bounded_fresh_check(system):
    settings, _, storage, builder, manager, provider, scheduler = system
    item = exhausted(manager)
    item.motion = evidence(
        phase="recovered_activity",
        posture="upright",
        clear_eligible=True,
        observed_ts=6,
        last_seen_ts=6,
    )

    async def recovered(packet):
        assert packet.temporal_context["observed_resolution_check"] is True
        return DecisionResult(
            incident_id=packet.incident_id,
            packet_id=packet.packet_id,
            version=packet.version,
            provider="mock",
            model="fixture",
            medical_status=Status.NO_ALERT,
            fire_status=Status.NO_ALERT,
            medical_observation="recovered",
            latest_posture="upright",
            input_tokens=100,
        )

    provider.evaluate = recovered
    assert scheduler.enqueue(item)
    assert await scheduler.step()
    assert item.request_count == 4 and item.resolution_checks == 1
    assert item.analysis_status == Status.NO_ALERT
    item.analysis_status = Status.UNKNOWN
    assert not scheduler.enqueue(item)
    storage.save_incident(item)
    restored = IncidentManager(settings, storage, builder).items[item.incident_id]
    assert restored.resolution_checks == 1 and restored.request_count == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"identity_uncertain": True},
        {"last_seen_ts": 4},
        {"epoch": 2},
        {"observed_ts": 7},
        {"phase": "following_concern", "clear_eligible": False},
    ],
)
def test_resolution_requires_fresh_unambiguous_motion(system, changes):
    manager, scheduler = system[4], system[6]
    item = exhausted(manager)
    item.motion = evidence(
        phase="recovered_activity",
        clear_eligible=True,
        last_seen_ts=6,
        observed_ts=6,
    ).model_copy(update=changes)
    assert not scheduler.enqueue(item)


@pytest.mark.asyncio
async def test_new_descent_cancels_queued_resolution_without_spending(system):
    manager, scheduler = system[4], system[6]
    item = exhausted(manager)
    item.motion = evidence(phase="recovered_activity", clear_eligible=True, last_seen_ts=6, observed_ts=6)
    assert scheduler.enqueue(item)
    item.motion.clear_eligible = False
    item.motion.phase = "following_concern"
    assert not await scheduler.step()
    assert item.request_count == 3 and item.resolution_checks == 0
    assert scheduler.budget.snapshot()["requests"] == 0


def test_resolution_does_not_override_urgent_status_or_repeatedly_query_same_outcome(system):
    manager, scheduler = system[4], system[6]
    item = exhausted(manager)
    item.motion = evidence(phase="recovered_activity", clear_eligible=True, last_seen_ts=6, observed_ts=6)
    item.analysis_status = Status.URGENT
    assert not scheduler.enqueue(item)
    item.analysis_status = Status.UNKNOWN
    item.last_assessed_motion_phase = "recovered_activity"
    assert not scheduler.enqueue(item)
