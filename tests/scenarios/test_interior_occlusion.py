"""A shrinking visible box during lateral walking does not establish a fall."""

import pytest

from stillwatch.motion import MotionClassifier
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Candidate, Category, Observation


def walking(n, *, drop=False, weak_torso=False, body_context=True):
    x, y = 500 - n * 12, n * 5 + (45 if drop and n >= 10 else 0)
    points = [(0.0, 0.0, 0.0)] * 17
    for k, dx, dy in (
        (0, 25, 15),
        (5, 10, 40),
        (6, 40, 40),
        (11, 15, 95),
        (12, 35, 95),
        (13, 15, 130),
        (14, 35, 130),
        (15, 15, 170),
        (16, 35, 170),
    ):
        confidence = 0.95 if n < 5 or y + dy <= 190 else 0.1
        if (weak_torso and n >= 8 and k in (11, 12)) or (not body_context and k >= 13):
            confidence = 0.1
        points[k] = (x + dx, y + dy, confidence)
    return Observation(
        capture_ts=n / 5,
        box=Box(x1=x, y1=y, x2=x + 50, y2=min(180 + y, 190) if n >= 5 else 180 + y),
        score=0.9,
        label="person",
        keypoints=points,
        truncated=False,
    )


@pytest.mark.parametrize("drop", [False, True])
def test_lateral_walk_behind_obstacle_preserves_a_new_descent(settings, drop):
    rules = MedicalRules(settings)
    events = []
    for n in range(14):
        events.extend(rules.update([walking(n, drop=drop)], n / 5, 1))
    events.extend(rules.update([], 3.4, 1))
    assert bool(events) is drop
    if drop:
        assert events[0].reason == "rapid_posture_change"
        assert events[0].last_ts == 2
        assert all(not c.motion.clear_eligible for c in events)


def test_weak_torso_does_not_explain_away_occluded_descent(settings):
    rules = MedicalRules(settings)
    events = []
    for n in range(14):
        events.extend(rules.update([walking(n, weak_torso=True)], n / 5, 1))
    events.extend(rules.update([], 3.4, 1))
    assert any(c.reason == "view_lost_after_suspicion" for c in events)
    assert all(not c.motion.clear_eligible for c in events)


@pytest.mark.parametrize("missing", ["body_context", "identity", "continuous_frames", "source_epoch"])
def test_interior_translation_requires_observed_continuity(settings, missing):
    classifier = MotionClassifier(settings)
    before = None
    for n in range(13):
        if missing == "continuous_frames" and n in (9, 10, 11):
            continue
        before = walking(n, body_context=missing != "body_context")
        before.track_id = 1
        if missing == "identity" and n == 12:
            before.association_uncertain = True
        classifier.update([before], before.capture_ts, 1, [])
    after = walking(13, body_context=missing != "body_context")
    after.track_id = 1
    if missing == "source_epoch":
        classifier.update([], after.capture_ts, 2, [])
    assert not classifier.cropped_translation(before, after)


def test_interior_translation_cannot_cancel_an_existing_concern(settings):
    classifier = MotionClassifier(settings)
    for n in range(13):
        before = walking(n)
        before.track_id = 1
        candidates = []
        if n == 10:
            candidates.append(
                Candidate(
                    category=Category.MEDICAL,
                    reason="rapid_posture_change",
                    first_ts=1.8,
                    last_ts=2,
                    track_id=1,
                    epoch=1,
                )
            )
        classifier.update([before], before.capture_ts, 1, candidates)
    after = walking(13)
    after.track_id = 1
    assert classifier.tracks[1].concern
    assert not classifier.cropped_translation(before, after)
    classifier.update([after], after.capture_ts, 1, [])
    assert not classifier.snapshot()[0].clear_eligible
