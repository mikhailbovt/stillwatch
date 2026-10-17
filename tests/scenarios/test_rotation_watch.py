import pytest

from stillwatch.config import Settings
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Observation


def body(ts, *, turned=False, descending=False, weak_torso=False):
    points = [(0, 0, 0)] * 17
    if turned:
        points[0] = (108, 100, 0.9)
        points[5], points[6] = (110, 105, 0.9), (120, 105, 0.9)
        points[11], points[12] = (145, 105, 0.9), (155, 105, 0.9)
        points[13], points[14] = (155, 116, 0.9), (160, 119, 0.9)
        points[15], points[16] = (162, 125, 0.9), (166, 128, 0.9)
        box = Box(x1=100, y1=90, x2=170, y2=160)
    else:
        points[0] = (120, 112, 0.9)
        points[5], points[6] = (110, 125, 0.9), (130, 125, 0.9)
        points[11], points[12] = (112, 160, 0.9), (128, 160, 0.9)
        points[13], points[14] = (110, 185, 0.9), (130, 185, 0.9)
        points[15], points[16] = (112, 215, 0.9), (128, 215, 0.9)
        box = Box(x1=100, y1=100, x2=140, y2=220)
    if descending:
        points = [(x, y + 70, confidence) for x, y, confidence in points]
        box = Box(x1=100, y1=160, x2=170, y2=230)
    if weak_torso:
        points[11] = (*points[11][:2], 0.1)
        points[12] = (*points[12][:2], 0.1)
    return Observation(capture_ts=ts, label="person", score=0.9, box=box, keypoints=points)


def lost_after_turn(rules, history):
    for observation in history:
        rules.update([observation], observation.capture_ts, 1)
    turn_ts = history[-1].capture_ts + 0.2
    assert not rules.update([body(turn_ts, turned=True)], turn_ts, 1)
    return rules.update([], turn_ts + 0.8, 1)


@pytest.mark.parametrize("samples", [1, 2, 3])
def test_short_track_rotation_without_descent_does_not_turn_a_gap_into_a_medical_event(samples):
    rules = MedicalRules(Settings())
    assert not lost_after_turn(rules, [body(n / 5) for n in range(samples)])


def test_established_rotation_watch_does_not_require_downward_image_translation():
    rules = MedicalRules(Settings())
    events = lost_after_turn(rules, [body(n / 5) for n in range(4)])
    assert len(events) == 1 and events[0].reason == "view_lost_after_suspicion"
    assert events[0].limited


def test_measured_descent_can_start_a_concern_before_rotation_history_is_established():
    rules = MedicalRules(Settings())
    rules.update([body(0)], 0, 1)
    events = rules.update([body(0.2, turned=True, descending=True)], 0.2, 1)
    assert any(event.reason == "rapid_posture_change" for event in events)


def test_missing_torso_evidence_cannot_establish_a_rotation_only_watch():
    rules = MedicalRules(Settings())
    history = [body(n / 5, weak_torso=n == 1) for n in range(4)]
    assert not lost_after_turn(rules, history)
