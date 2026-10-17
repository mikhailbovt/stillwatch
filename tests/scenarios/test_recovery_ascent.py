import pytest
from test_episode_memory import answer, event, pose
from test_recovery_boundary import candidate

from stillwatch.motion import MotionClassifier
from stillwatch.types import Box, Status


def compact(ts, *, track=1):
    """A rising body with one folded leg and one apparently extended leg."""
    observation = pose(ts, track=track)
    points = list(observation.keypoints)
    points[5], points[6] = (44, 90, 0.95), (60, 90, 0.95)
    points[11], points[12] = (44, 110, 0.95), (60, 110, 0.95)
    points[13], points[14] = (50, 136, 0.95), (60, 130, 0.95)
    points[15], points[16] = (46, 115, 0.95), (60, 149, 0.95)
    return observation.model_copy(update={"box": Box(x1=30, y1=80, x2=90, y2=150), "keypoints": points})


def start(settings):
    classifier = MotionClassifier(settings)
    classifier.update([pose(0, low=True)], 0, 1, [candidate(0, 0)])
    return classifier


def ascent(classifier, until=16):
    for n in range(1, until + 1):
        ts = n / 5
        classifier.update([compact(ts) if n <= 5 else pose(ts)], ts, 1, [])
    return classifier.snapshot()[0]


def test_a_compact_projected_upright_pose_does_not_prove_completed_ascent(settings):
    classifier = start(settings)
    measured = compact(1)
    projected = classifier.pose(measured)
    assert projected.posture == "upright" and projected.reliable
    ascent(classifier, until=5)
    before = pose(1.2).model_dump()
    classifier.update([pose(1.2)], 1.2, 1, [])
    current = classifier.snapshot()[0]
    assert current.recovery_seconds == 0
    assert current.recovery_basis is None and not current.clear_eligible
    assert classifier.tracks[1].history[-1].observation.model_dump() == before


def test_recovery_requires_settled_activity_after_the_visible_rise(settings):
    classifier = start(settings)
    early = ascent(classifier)
    assert not early.clear_eligible and early.phase == "following_concern"
    for n in range(17, 27):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    settled = classifier.snapshot()[0]
    assert settled.clear_eligible and settled.phase == "recovered_activity"
    assert settled.episode_id == early.episode_id
    assert settled.recovery_seconds >= settings.recovery_seconds


def test_cloud_recovered_answer_cannot_use_credit_from_an_unfinished_rise(system):
    settings, manager = system[0], system[4]
    classifier = start(settings)
    motion = ascent(classifier)
    item = event(manager, motion)
    result = answer(item, "recovered")
    result.latest_posture = "upright"
    manager.apply(result, 3.2)
    assert item.analysis_status == Status.UNKNOWN
    assert item.policy_reason == "local_episode_unresolved"


def test_a_repeat_fall_before_settled_recovery_keeps_the_unresolved_episode(settings):
    classifier = start(settings)
    previous = classifier.snapshot()[0].episode_id
    ascent(classifier)
    current = classifier.update([pose(3.4, low=True)], 3.4, 1, [candidate(3.2, 3.4)])[0]
    assert current.motion.episode_id == previous
    assert current.motion.started_ts == 0
    assert not current.motion.clear_eligible


@pytest.mark.parametrize("movement", ["walking", "approaching", "gait"])
def test_ordinary_upright_motion_does_not_delay_recovery(settings, movement):
    def scaled(x, y, scale):
        return 50 + (x - 50) * scale, 440 + (y - 440) * scale

    classifier = MotionClassifier(settings)
    classifier.update([pose(0, shift=300)], 0, 1, [candidate(0, 0)])
    for n in range(1, 17):
        ts = n / 5
        observation = pose(ts, shift=300 - 3 * n if movement == "walking" else 300)
        if movement == "approaching":
            scale = 1 + n * 0.08
            points = [
                (*scaled(x, y, scale), score) if score else (x, y, score)
                for x, y, score in observation.keypoints
            ]
            box = observation.box
            x1, y1 = scaled(box.x1, box.y1, scale)
            x2, y2 = scaled(box.x2, box.y2, scale)
            observation = observation.model_copy(
                update={"keypoints": points, "box": Box(x1=x1, y1=y1, x2=x2, y2=y2)}
            )
        elif movement == "gait":
            points = list(observation.keypoints)
            points[13 if n % 2 else 14] = (75, 395, 0.95)
            observation = observation.model_copy(update={"keypoints": points})
        classifier.update([observation], ts, 1, [])
    current = classifier.snapshot()[0]
    assert current.clear_eligible and current.phase == "recovered_activity"
    assert current.recovery_seconds == pytest.approx(3)


def test_return_to_prior_supported_seating_still_establishes_recovery(settings):
    classifier = MotionClassifier(settings)
    for n in range(7):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    classifier.update([pose(1.4, low=True)], 1.4, 1, [candidate(1.2, 1.4)])
    for n in range(8, 25):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    current = classifier.snapshot()[0]
    assert current.clear_eligible and current.phase == "recovered_activity"
    assert current.recovery_basis == "prior_supported_posture"


@pytest.mark.parametrize("interruption", ["missing", "ambiguous"])
def test_visible_ascent_cannot_override_uncertain_identity(settings, interruption):
    classifier = start(settings)
    ascent(classifier, until=5)
    if interruption == "missing":
        classifier.update([], 2, 1, [])
        begin = 11
    else:
        classifier.update([pose(1.2).model_copy(update={"association_uncertain": True})], 1.2, 1, [])
        begin = 7
    for n in range(begin, 37):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    current = classifier.snapshot()[0]
    assert current.identity_uncertain and not current.clear_eligible
    assert current.recovery_seconds == 0


def shifted(observation, dx):
    box = observation.box
    return observation.model_copy(
        update={
            "box": Box(x1=box.x1 + dx, y1=box.y1, x2=box.x2 + dx, y2=box.y2),
            "keypoints": [
                (x + dx, y, score) if score else (x, y, score) for x, y, score in observation.keypoints
            ],
        }
    )


def test_one_person_rising_does_not_reset_another_person_recovering(settings):
    classifier = MotionClassifier(settings)
    classifier.update(
        [pose(0, low=True), shifted(pose(0, track=2), 300)],
        0,
        1,
        [candidate(0, 0), candidate(0, 0).model_copy(update={"track_id": 2})],
    )
    for n in range(1, 17):
        ts = n / 5
        first = compact(ts) if n <= 5 else pose(ts)
        classifier.update([first, shifted(pose(ts, track=2), 300)], ts, 1, [])
    assert not classifier.tracks[1].evidence.clear_eligible
    assert classifier.tracks[2].evidence.clear_eligible
