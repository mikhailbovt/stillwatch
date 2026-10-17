import pytest
from test_episode_memory import pose

from stillwatch.motion import MotionClassifier
from stillwatch.types import Candidate, Category, Lifecycle, Status


def fall(ts, first=None):
    return Candidate(
        category=Category.MEDICAL,
        reason="rapid_posture_change",
        first_ts=ts if first is None else first,
        last_ts=ts,
        track_id=1,
        epoch=1,
        box=pose(ts, low=True).box,
    )


def unresolved(classifier):
    first = classifier.update([pose(0, low=True)], 0, 1, [fall(0)])[0]
    classifier.update([], 0.8, 1, [])
    classifier.update([pose(1, low=True)], 1, 1, [])
    assert classifier.snapshot()[0].identity_uncertain
    return first


@pytest.mark.parametrize("seated", [False, True])
def test_fresh_observed_activity_allows_new_fall_without_clearing_old_identity(settings, seated):
    classifier = MotionClassifier(settings)
    previous = unresolved(classifier)
    for n in range(6, 27):
        assert not classifier.update([pose(n / 5, seated=seated)], n / 5, 1, [])
        state = classifier.snapshot()[0]
        assert state.episode_id == previous.motion.episode_id
        assert state.identity_uncertain and not state.clear_eligible
    current = classifier.update([pose(5.4, low=True)], 5.4, 1, [fall(5.4, 5.2)])[0]
    assert current.motion.episode_id != previous.motion.episode_id
    assert current.motion.previous_episode_id is None
    assert current.motion.started_ts == 5.2
    assert not current.motion.identity_uncertain and not current.motion.clear_eligible
    again = classifier.update([pose(5.6, low=True)], 5.6, 1, [fall(5.6, 5.2)])[0]
    assert again.motion.episode_id == current.motion.episode_id


@pytest.mark.parametrize(
    "interruption", ["short_activity", "ambiguous", "gap", "partial", "still_low", "ambiguous_onset"]
)
def test_incomplete_new_context_cannot_split_an_unresolved_episode(settings, interruption):
    classifier = MotionClassifier(settings)
    previous = unresolved(classifier)
    end = 15 if interruption == "short_activity" else 27
    for n in range(6, end):
        if interruption == "gap" and n in (21, 22, 23, 24):
            classifier.update([], n / 5, 1, [])
            continue
        current = pose(n / 5, low=interruption == "still_low")
        if interruption == "ambiguous" and n == 23:
            current.association_uncertain = True
        if interruption == "partial":
            current.keypoints = [(0, 0, 0)] * 17
        classifier.update([current], n / 5, 1, [])
    ts = end / 5
    at_onset = pose(ts, low=True)
    at_onset.association_uncertain = interruption == "ambiguous_onset"
    current = classifier.update([at_onset], ts, 1, [fall(ts, ts - 0.2)])[0]
    assert current.motion.episode_id == previous.motion.episode_id
    assert current.motion.identity_uncertain and not current.motion.clear_eligible


def test_one_weak_pose_does_not_erase_fresh_observed_context(settings):
    classifier = MotionClassifier(settings)
    previous = unresolved(classifier)
    for n in range(6, 26):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    weak = pose(5.2).model_copy(update={"keypoints": [(0, 0, 0)] * 17})
    classifier.update([weak], 5.2, 1, [])
    current = classifier.update([pose(5.4, low=True)], 5.4, 1, [fall(5.4, 5.2)])[0]
    assert current.motion.episode_id != previous.motion.episode_id
    assert current.motion.previous_episode_id is None and not current.motion.clear_eligible


@pytest.mark.asyncio
@pytest.mark.parametrize("old_status", [Status.UNKNOWN, Status.URGENT])
async def test_new_identity_episode_gets_its_own_budget_and_preserves_old_incident(system, old_status):
    settings, _, storage, _, manager, provider, scheduler = system
    classifier = MotionClassifier(settings)
    previous, _ = manager.ingest(unresolved(classifier))
    previous.request_count = settings.max_episode_requests
    previous.analysis_status = previous.medical_status = old_status
    previous.lifecycle = Lifecycle.OPEN
    for n in range(6, 27):
        classifier.update([pose(n / 5)], n / 5, 1, [])
        manager.observe_motion(classifier.snapshot(), n / 5)
    storage.save_incident(previous)
    current_motion = classifier.update([pose(5.4, low=True)], 5.4, 1, [fall(5.4, 5.2)])[0]
    current, created = manager.ingest(current_motion)
    manager.observe_motion(classifier.snapshot(), 5.4)
    assert created and current.incident_id != previous.incident_id
    assert current.request_count == 0 and current.analysis_status == Status.PENDING
    assert current.detected_ts == 5.4
    assert previous.request_count == settings.max_episode_requests
    assert previous.superseded_by is None and previous.lifecycle == Lifecycle.OPEN
    assert previous.analysis_status == old_status and previous.motion.identity_uncertain
    assert not previous.motion.clear_eligible
    assert scheduler.enqueue(current)
    assert await scheduler.step()
    assert len(provider.calls) == 1 and current.request_count == 1
    assert previous.request_count == settings.max_episode_requests and previous.analysis_status == old_status
