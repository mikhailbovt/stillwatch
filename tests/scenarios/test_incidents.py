import pytest

from stillwatch.incidents import IncidentManager
from stillwatch.types import Box, Candidate, Category, Lifecycle, Status


def candidate(ts=6, track=1, category=Category.MEDICAL, x=20):
    return Candidate(
        category=category,
        reason="persistent_low_posture",
        first_ts=4,
        last_ts=ts,
        box=Box(x1=x, y1=20, x2=x + 80, y2=90),
        track_id=track,
        epoch=1,
    )


def test_same_event_deduplicates_but_two_people_and_fire_do_not(system):
    _, _, _, _, manager, _, _ = system
    first, created = manager.ingest(candidate())
    assert created
    assert manager.ingest(candidate(ts=7))[0].incident_id == first.incident_id
    second, created = manager.ingest(candidate(track=2, x=90))
    assert created and second.incident_id != first.incident_id
    fire, created = manager.ingest(candidate(category=Category.FIRE, track=None))
    assert created and fire.category == Category.FIRE


def test_closed_event_does_not_suppress_future_events(system):
    _, _, _, _, manager, _, _ = system
    old, _ = manager.ingest(candidate())
    manager.annotate(old.incident_id, "close")
    new, created = manager.ingest(candidate(ts=8))
    assert created and new.incident_id != old.incident_id


def test_operator_feedback_is_separate_and_restart_marks_results_stale(system):
    settings, _, storage, evidence, manager, _, _ = system
    item, _ = manager.ingest(candidate())
    item.analysis_status = Status.NO_ALERT
    manager.annotate(item.incident_id, "confirmed")
    assert item.analysis_status == Status.NO_ALERT and item.feedback == "confirmed"
    assert item.lifecycle == Lifecycle.ACKNOWLEDGED
    restored = IncidentManager(settings, storage, evidence).items[item.incident_id]
    assert restored.assessment_stale and restored.feedback == "confirmed"


def test_track_reassignment_after_occlusion_can_match(system):
    _, _, _, _, manager, _, _ = system
    old, _ = manager.ingest(candidate(ts=6))
    new = Candidate(
        category=Category.MEDICAL,
        reason="persistent_low_posture",
        first_ts=8,
        last_ts=10,
        box=old.box,
        track_id=7,
        epoch=1,
    )
    same, created = manager.ingest(new)
    assert not created and same.incident_id == old.incident_id


def test_evidence_is_chronological_and_preserves_onset(system):
    _, buffer, _, evidence, manager, _, _ = system
    item, _ = manager.ingest(candidate())
    packet = evidence.build(item, 6)
    assert len(packet.images) <= 8
    assert packet.images[0].capture_ts < item.first_ts
    assert [i.capture_ts for i in packet.images] == sorted(i.capture_ts for i in packet.images)
    assert any(i.kind == "overview" for i in packet.images)
    buffer.clear()
    assert evidence.build(item, 30).images


@pytest.mark.parametrize("status", [Status.NO_ALERT, Status.REVIEW, Status.UNKNOWN, Status.URGENT])
def test_new_transition_does_not_inherit_previous_assessment_or_request_cap(system, status):
    _, _, _, _, manager, _, scheduler = system
    old, _ = manager.ingest(candidate(ts=28))
    old.request_count = 3
    old.analysis_status = old.medical_status = status
    transition = candidate(ts=30).model_copy(update={"reason": "rapid_posture_change", "first_ts": 29.5})
    new, created = manager.ingest(transition)
    assert created and new.incident_id != old.incident_id
    assert new.request_count == 0 and new.analysis_status == Status.PENDING
    assert new.detected_ts == 30 and scheduler.enqueue(new)
    assert old.analysis_status == status and old.request_count == 3
    repeated, created = manager.ingest(transition.model_copy(update={"last_ts": 30.2}))
    assert not created and repeated.incident_id == new.incident_id
    continued, created = manager.ingest(candidate(ts=32).model_copy(update={"first_ts": 29.8}))
    assert not created and continued.incident_id == new.incident_id


def test_overlapping_transition_stays_in_assessed_episode(system):
    _, _, _, _, manager, _, _ = system
    old, _ = manager.ingest(candidate(ts=28))
    old.request_count = 3
    same, created = manager.ingest(
        candidate(ts=28.4).model_copy(update={"reason": "rapid_posture_change", "first_ts": 27.9})
    )
    assert not created and same.incident_id == old.incident_id


def test_repeated_candidates_do_not_split_the_initial_observation_window(system):
    _, _, _, _, manager, _, _ = system
    old, _ = manager.ingest(candidate(ts=28))
    same, created = manager.ingest(
        candidate(ts=30).model_copy(update={"reason": "rapid_posture_change", "first_ts": 29.5})
    )
    assert not created and same.incident_id == old.incident_id and same.detected_ts == 28
