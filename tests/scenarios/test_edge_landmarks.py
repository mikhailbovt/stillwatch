import pytest
from test_episode_memory import pose

from stillwatch.motion import MotionClassifier
from stillwatch.triggers import MedicalRules
from stillwatch.types import Candidate, Category


def exit_pose(n, *, head=0.95, change=None):
    extra = 10 if change == "acceleration" else 0
    observation = pose(n / 5, shift=n * 5 + extra)
    observation.keypoints[0] = (50 + (45 if change == "head_jump" else 0), n * 5 + 5 + extra, head)
    if change == "deformation":
        x, y, confidence = observation.keypoints[5]
        observation.keypoints[5] = (x, y + 20, confidence)
    observation.keypoints = [
        (x, min(y, 200), confidence if y < 200 else 0.1) for x, y, confidence in observation.keypoints
    ]
    observation.box.y2 = min(observation.box.y2, 200)
    observation.truncated = observation.box.y2 >= 197
    observation.limited = observation.box.height < 40
    return observation


@pytest.mark.parametrize("confidence", [0.4, 0.55, 0.649])
def test_transient_weak_head_does_not_turn_observed_exit_into_a_fall(settings, confidence):
    rules = MedicalRules(settings)
    candidates = []
    for n in range(36):
        observation = exit_pose(n, head=confidence if n == 30 else 0.95)
        candidates.extend(rules.update([observation], n / 5, 1))
    candidates.extend(rules.update([], 8, 1))
    assert not candidates


@pytest.mark.parametrize(
    "missing",
    ["strong_history", "continuous_motion", "before_identity", "after_identity", "epoch", "head", "shoulder"],
)
def test_weak_head_exit_needs_measured_unambiguous_context(settings, missing):
    classifier = MotionClassifier(settings)
    for n in range(30):
        if missing == "continuous_motion" and n in (26, 27, 28):
            continue
        before = exit_pose(n, head=0.55 if missing == "strong_history" else 0.95)
        if missing == "before_identity" and n == 29:
            before.association_uncertain = True
        classifier.update([before], n / 5, 1, [])
    after = exit_pose(30, head=0.2 if missing == "head" else 0.55)
    if missing == "after_identity":
        after.association_uncertain = True
    if missing == "shoulder":
        x, y, _ = after.keypoints[6]
        after.keypoints[6] = (x, y, 0.6)
    if missing == "epoch":
        classifier.update([], 6, 2, [])
    assert not classifier.cropped_translation(before, after)


@pytest.mark.parametrize("change", ["acceleration", "deformation", "head_jump"])
def test_weak_head_cannot_explain_a_new_drop_or_geometry_change(settings, change):
    rules = MedicalRules(settings)
    for n in range(30):
        assert not rules.update([exit_pose(n)], n / 5, 1)
    after = exit_pose(30, head=0.55, change=change)
    candidates = rules.update([after], 6, 1)
    assert candidates
    assert all(not candidate.motion.clear_eligible for candidate in candidates)


def test_weak_head_exit_cannot_cancel_an_established_concern(settings):
    classifier = MotionClassifier(settings)
    for n in range(30):
        before = exit_pose(n)
        candidate = Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=5.6,
            last_ts=5.8,
            track_id=1,
            epoch=1,
        )
        classifier.update([before], n / 5, 1, [candidate] if n == 29 else [])
    after = exit_pose(30, head=0.55)
    assert not classifier.cropped_translation(before, after)
    classifier.update([after], 6, 1, [])
    assert not classifier.snapshot()[0].clear_eligible
