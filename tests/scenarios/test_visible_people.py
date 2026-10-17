import pytest

from stillwatch.types import Box, Candidate, Category, MotionEvidence, Status


def motion(track, ts, *, epoch=1, **changes):
    return MotionEvidence(
        episode_id=f"{epoch}-{track}",
        track_id=track,
        epoch=epoch,
        started_ts=0,
        observed_ts=ts,
        last_seen_ts=ts,
    ).model_copy(update=changes)


def lost(track, ts, *, epoch=1, with_motion=True):
    return Candidate(
        category=Category.MEDICAL,
        reason="view_lost_after_suspicion",
        first_ts=ts - 0.6,
        last_ts=ts,
        epoch=epoch,
        track_id=track,
        box=Box(x1=20, y1=20, x2=80, y2=90),
        motion=motion(track, ts, epoch=epoch, phase="visibility_lost", last_seen_ts=ts - 0.6)
        if with_motion
        else None,
    )


@pytest.mark.parametrize("with_motion", [False, True])
def test_two_people_lost_behind_same_object_keep_separate_incidents(system, with_motion):
    manager = system[4]
    manager.observe_motion([motion(1, 10), motion(2, 10)], 10)
    first, _ = manager.ingest(lost(1, 10.6, with_motion=with_motion))
    first.request_count = 2
    first.analysis_status = Status.REVIEW
    second, created = manager.ingest(lost(2, 11, with_motion=with_motion))
    assert created and second.incident_id != first.incident_id
    assert first.track_ids == [1] and second.track_ids == [2]
    assert first.request_count == 2 and first.analysis_status == Status.REVIEW
    assert second.request_count == 0 and second.detected_ts == 11


def test_cached_missing_person_does_not_count_as_simultaneously_observed(system):
    manager = system[4]
    manager.observe_motion([motion(1, 10)], 10)
    manager.observe_motion([motion(1, 10), motion(2, 10.6)], 10.6)
    first, _ = manager.ingest(lost(1, 10.6))
    same, created = manager.ingest(lost(2, 11))
    assert not created and same.incident_id == first.incident_id
    assert same.identity_uncertain


def test_previous_source_cannot_constrain_reused_track_numbers(system):
    manager = system[4]
    manager.observe_motion([motion(1, 10), motion(2, 10)], 10)
    first, _ = manager.ingest(lost(1, 10.6, epoch=2))
    same, created = manager.ingest(lost(2, 11, epoch=2))
    assert not created and same.incident_id == first.incident_id


def test_visibility_history_survives_the_observation_period_and_expires(system):
    manager = system[4]
    manager.settings.medical_observation_seconds = 15
    manager.observe_motion([motion(1, 0), motion(2, 0)], 0)
    manager.observe_motion([], manager.settings.medical_observation_seconds)
    first, _ = manager.ingest(lost(1, 15.6))
    second, created = manager.ingest(lost(2, 16))
    assert created and second.incident_id != first.incident_id
    manager.observe_motion([], 100)
    assert not manager.visible_pairs


def test_visibility_history_is_bounded_during_continuous_turnover(system):
    manager = system[4]
    for n in range(40):
        ts = n / 5
        manager.observe_motion([motion(track, ts) for track in range(n * 128, (n + 1) * 128)], ts)
    assert len(manager.visible_pairs) <= manager.max_visible_pairs
    first, _ = manager.ingest(lost(39 * 128, 8))
    second, created = manager.ingest(lost(39 * 128 + 1, 8.4))
    assert created and second.incident_id != first.incident_id
