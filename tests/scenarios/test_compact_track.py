import json
from pathlib import Path

import pytest

from stillwatch.tracking import PoseTracker
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Observation


def source_samples():
    path = Path(__file__).parents[1] / "fixtures" / "compact_track.json"
    return {
        row["source_ts"]: Observation.model_validate(row["observation"])
        for row in json.loads(path.read_text())["samples"]
    }


def test_crossing_runner_cannot_take_the_grounded_person_identity():
    samples = source_samples()
    tracker = PoseTracker()
    previous = []
    for ts in (13.2, 13.4, 13.6, 15.0, 15.2):
        previous = tracker.update([samples[ts]], ts, 1)
    grounded_id = previous[0].track_id
    for ts in (16.0, 16.2):
        runner = tracker.update([samples[ts]], ts, 1)
        assert all(o.track_id != grounded_id for o in runner)


def test_grounded_person_keeps_identity_when_seen_after_the_runner():
    samples = source_samples()
    tracker = PoseTracker()
    for ts in (13.2, 13.4, 13.6, 15.0, 15.2):
        previous = tracker.update([samples[ts]], ts, 1)
    grounded_id = previous[0].track_id
    for ts in (16.0, 16.2, 18.2, 18.4):
        current = tracker.update([samples[ts]], ts, 1)
    assert len(current) == 1 and current[0].track_id == grounded_id


def test_unrelated_runner_does_not_create_a_fall_in_the_grounded_person_history(settings):
    rules = MedicalRules(settings)
    candidates = []
    for ts, obs in source_samples().items():
        candidates.extend(rules.update([obs], ts, 1))
    assert not any(c.reason == "rapid_posture_change" and c.first_ts == 16.0 for c in candidates)
    assert not any(m.clear_eligible for m in rules.motion.snapshot())


@pytest.mark.parametrize("limitation", ["before_crop", "after_crop", "weak_torso", "nonfinite_score"])
def test_incomplete_body_evidence_does_not_impose_the_expansion_guard(limitation):
    samples = source_samples()
    before = samples[15.2].model_copy(update={"score": 0.9})
    after = samples[16.0]
    if limitation == "before_crop":
        before = before.model_copy(update={"truncated": True})
    elif limitation == "after_crop":
        after = after.model_copy(update={"truncated": True})
    else:
        points = after.keypoints.copy()
        points[5] = (*points[5][:2], float("nan") if limitation == "nonfinite_score" else 0.64)
        after = after.model_copy(update={"keypoints": points})
    tracker = PoseTracker()
    identity = tracker.update([before], before.capture_ts, 1)[0].track_id
    current = tracker.update([after], after.capture_ts, 1)
    assert len(current) == 1 and current[0].track_id == identity


def test_a_measured_support_point_preserves_a_supported_rise():
    points = [(0, 0, 0)] * 17
    points[5], points[6] = (110, 110, 0.95), (110, 120, 0.95)
    points[11], points[12] = (130, 110, 0.95), (130, 120, 0.95)
    points[15], points[16] = (150, 128, 0.95), (140, 128, 0.95)
    before = Observation(
        capture_ts=0, box=Box(x1=100, y1=100, x2=160, y2=130), score=0.9, label="person", keypoints=points
    )
    risen = points.copy()
    risen[5], risen[6] = (125, 70, 0.95), (135, 70, 0.95)
    risen[11], risen[12] = (125, 90, 0.95), (135, 90, 0.95)
    after = before.model_copy(
        update={"capture_ts": 0.2, "box": Box(x1=90, y1=30, x2=180, y2=135), "keypoints": risen}
    )
    tracker = PoseTracker()
    identity = tracker.update([before], 0, 1)[0].track_id
    current = tracker.update([after], 0.2, 1)
    assert len(current) == 1 and current[0].track_id == identity


def test_compact_same_size_translation_keeps_identity():
    before = source_samples()[15.2].model_copy(update={"score": 0.9})
    after = before.model_copy(
        update={
            "capture_ts": 15.4,
            "box": Box(x1=before.box.x1 + 10, y1=before.box.y1, x2=before.box.x2 + 10, y2=before.box.y2),
            "keypoints": [(x + 10, y, c) for x, y, c in before.keypoints],
        }
    )
    tracker = PoseTracker()
    identity = tracker.update([before], before.capture_ts, 1)[0].track_id
    current = tracker.update([after], after.capture_ts, 1)
    assert len(current) == 1 and current[0].track_id == identity


def test_both_people_remain_distinct_when_the_grounded_person_is_also_detected():
    samples = source_samples()
    before = samples[15.2].model_copy(update={"score": 0.9})
    tracker = PoseTracker()
    identity = tracker.update([before], before.capture_ts, 1)[0].track_id
    for ts in (16.0, 16.2, 16.4):
        grounded = before.model_copy(update={"capture_ts": ts})
        runner = samples[16.0].model_copy(update={"capture_ts": ts})
        current = tracker.update([grounded, runner], ts, 1)
    assert len(current) == 2 and len({o.track_id for o in current}) == 2
    assert next(o for o in current if o.box == grounded.box).track_id == identity


def test_box_only_adapter_retains_its_matching_contract():
    samples = source_samples()
    before = samples[15.2].model_copy(update={"score": 0.9, "keypoints": []})
    after = samples[16.0].model_copy(update={"keypoints": []})
    tracker = PoseTracker()
    identity = tracker.update([before], before.capture_ts, 1)[0].track_id
    current = tracker.update([after], after.capture_ts, 1)
    assert len(current) == 1 and current[0].track_id == identity
