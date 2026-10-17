import pytest
from test_episode_memory import pose

from stillwatch.motion import MotionClassifier
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box


def bend(ts, *, cropped=False):
    observation = pose(ts)
    observation.keypoints[5], observation.keypoints[6] = (43, 52, 0.95), (57, 52, 0.95)
    observation.box = Box(x1=30, y1=48, x2=70, y2=150)
    observation.truncated = cropped
    return observation


def test_upper_body_foreshortening_above_a_planted_leg_does_not_start_a_fall(settings):
    rules = MedicalRules(settings)
    for observation in [pose(0), pose(0.2), bend(0.4), bend(0.6), bend(0.8), pose(1)]:
        assert not rules.update([observation], observation.capture_ts, 1)
    assert not rules.motion.tracks[1].concern
    assert not rules.motion.tracks[1].evidence.clear_eligible


def test_cropped_raised_arms_do_not_turn_a_bend_into_a_partial_body_fall(settings):
    rules = MedicalRules(settings)
    initial = pose(0)
    initial.box = Box(x1=30, y1=-70, x2=70, y2=150)
    initial.truncated = True
    assert not rules.update([initial], 0, 1)
    assert not rules.update([bend(0.4, cropped=True)], 0.4, 1)


def test_actual_descent_after_a_bend_still_starts_an_episode(settings):
    rules = MedicalRules(settings)
    for observation in [pose(0), pose(0.2), bend(0.4)]:
        assert not rules.update([observation], observation.capture_ts, 1)
    candidates = rules.update([pose(0.6, low=True)], 0.6, 1)
    assert candidates
    assert candidates[0].motion.transition == "fall_like"
    assert not candidates[0].motion.clear_eligible


def test_stable_leg_support_does_not_clear_an_existing_concern(settings):
    rules = MedicalRules(settings)
    rules.update([pose(0)], 0, 1)
    incident = rules.update([pose(0.4, low=True)], 0.4, 1)[0]
    episode_id = incident.motion.episode_id
    rules.update([pose(0.6)], 0.6, 1)
    rules.update([bend(0.8)], 0.8, 1)
    state = rules.motion.tracks[1]
    assert state.concern and state.evidence.episode_id == episode_id
    assert not state.evidence.clear_eligible


def test_a_planted_foot_does_not_veto_pelvis_descent(settings):
    classifier = MotionClassifier(settings)
    falling = bend(0.4)
    for k in (11, 12):
        x, y, confidence = falling.keypoints[k]
        falling.keypoints[k] = x, y + 30, confidence
    assert not classifier.standing_bend(pose(0), falling)


def test_a_seated_baseline_does_not_veto_a_slump(settings):
    assert not MotionClassifier(settings).standing_bend(pose(0, seated=True), bend(0.4))


@pytest.mark.parametrize("missing", [(11, 12), (13, 14), (15, 16)])
def test_missing_leg_joints_cannot_establish_support(settings, missing):
    current = bend(0.4)
    for k in missing:
        x, y, _ = current.keypoints[k]
        current.keypoints[k] = x, y, 0.2
    assert not MotionClassifier(settings).standing_bend(pose(0), current)


def test_uncertain_identity_cannot_establish_support(settings):
    current = bend(0.4).model_copy(update={"association_uncertain": True})
    assert not MotionClassifier(settings).standing_bend(pose(0), current)


@pytest.mark.parametrize("time", [-0.2, 0, 5.1])
def test_future_stale_or_expired_support_is_rejected(settings, time):
    assert not MotionClassifier(settings).standing_bend(pose(0), bend(time))
