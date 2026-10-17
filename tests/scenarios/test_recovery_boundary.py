from test_episode_memory import answer, event, pose

from stillwatch.motion import MotionClassifier
from stillwatch.types import Candidate, Category, Lifecycle, Status


def candidate(first, last, reason="rapid_posture_change"):
    return Candidate(
        category=Category.MEDICAL,
        reason=reason,
        first_ts=first,
        last_ts=last,
        track_id=1,
        epoch=1,
    )


def recovery(settings):
    classifier = MotionClassifier(settings)
    classifier.update([pose(0, low=True)], 0, 1, [candidate(0, 0)])
    for n in range(1, 23):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    assert classifier.tracks[1].evidence.phase == "recovered_activity"
    assert classifier.tracks[1].recovery_observed
    return classifier


def unseen_low(classifier):
    classifier.update([], 5.2, 1, [])
    for n in range(27, 37):
        classifier.update([pose(n / 5, low=True)], n / 5, 1, [])
    return classifier.update([pose(7.4, low=True)], 7.4, 1, [candidate(5.4, 7.4, "persistent_low_posture")])[
        0
    ]


def test_new_low_posture_after_recovery_is_independent_when_the_descent_was_unseen(settings):
    classifier = recovery(settings)
    previous = classifier.snapshot()[0].episode_id
    current = unseen_low(classifier)
    assert current.motion.episode_id != previous
    assert current.motion.previous_episode_id is None
    assert current.motion.started_ts == 5.4
    assert current.motion.transition == "unknown_origin" and not current.motion.onset_observed
    assert not current.motion.clear_eligible
    assert classifier.tracks[1].baseline is None


def test_a_closed_recovered_episode_cannot_swallow_a_new_unseen_descent(system):
    settings, manager = system[0], system[4]
    classifier = recovery(settings)
    previous = event(manager, classifier.snapshot()[0])
    result = answer(previous, "recovered")
    result.latest_posture = "upright"
    manager.apply(result, 4.4)
    assert previous.lifecycle == Lifecycle.CLOSED
    current, created = manager.ingest(unseen_low(classifier))
    assert created and current.incident_id != previous.incident_id
    assert current.analysis_status == Status.PENDING and current.request_count == 0
    assert previous.analysis_status == Status.NO_ALERT and previous.lifecycle == Lifecycle.CLOSED


def test_new_rapid_concern_after_recovery_and_a_gap_keeps_no_identity_link(settings):
    classifier = recovery(settings)
    previous = classifier.snapshot()[0].episode_id
    classifier.update([], 5.2, 1, [])
    current = classifier.update([pose(5.4, low=True)], 5.4, 1, [candidate(4.2, 5.4)])[0]
    assert current.motion.episode_id != previous and current.motion.previous_episode_id is None
    assert current.motion.started_ts == 5.4
    assert current.motion.transition == "unknown_origin" and not current.motion.onset_observed
    assert current.limited and not current.motion.clear_eligible


def test_continuing_supported_floor_activity_keeps_its_episode(settings):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    for n in range(10, 31):
        classifier.update([pose(n / 5, low=True, shift=-80)], n / 5, 1, [])
    previous = classifier.snapshot()[0]
    assert previous.phase == "ordinary_activity"
    current = classifier.update(
        [pose(6.2, low=True, shift=-80)], 6.2, 1, [candidate(2, 6.2, "persistent_low_posture")]
    )[0]
    assert current.motion.episode_id == previous.episode_id
    assert current.motion.phase == "ordinary_activity" and current.motion.clear_eligible


def test_duplicate_old_low_observation_does_not_create_a_new_episode(settings):
    classifier = recovery(settings)
    previous = classifier.snapshot()[0].episode_id
    classifier.update([pose(4.6)], 4.6, 1, [candidate(0, 4.6, "persistent_low_posture")])
    assert classifier.snapshot()[0].episode_id == previous


def test_recovery_boundary_resets_with_the_source(settings):
    classifier = recovery(settings)
    previous = classifier.snapshot()[0].episode_id
    classifier.update([pose(0, low=True)], 0, 2, [])
    current = classifier.snapshot()[0]
    assert current.epoch == 2 and current.episode_id != previous
    assert not classifier.tracks[1].recovery_observed and current.previous_episode_id is None
