import pytest

from stillwatch.tracking import PoseTracker
from stillwatch.types import Box, Observation


def detections(ts, scale=1):
    points = [(0, 0, 0)] * 17
    points[4] = (196, 192, 0.6)
    points[5], points[6] = (185, 189, 0.9), (179, 188, 0.9)
    points[11], points[12] = (140, 191, 0.9), (136, 188, 0.9)
    points[13], points[14] = (105, 194, 0.9), (102, 190, 0.9)
    points[15], points[16] = (70, 190, 0.9), (67, 185, 0.9)
    full = Observation(
        capture_ts=ts,
        label="person",
        score=0.9,
        box=Box(x1=55, y1=170, x2=211, y2=215),
        keypoints=points,
    )
    partial = full.model_copy(deep=True)
    partial.score = 0.8
    partial.box = Box(x1=129, y1=173, x2=210, y2=215)
    for k in (11, 12, 13, 14, 15, 16):
        partial.keypoints[k] = (150 - k, 200, 0.9)
    if scale != 1:
        for obs in (full, partial):
            obs.box = Box(**{k: v * scale for k, v in obs.box.model_dump().items()})
            obs.keypoints = [(x * scale, y * scale, confidence) for x, y, confidence in obs.keypoints]
    return full, partial


@pytest.mark.parametrize("scale", [1, 4])
def test_nested_partial_body_cannot_create_a_competing_person(scale):
    tracker = PoseTracker()
    for n in range(5):
        observations = tracker.update(list(detections(n / 5, scale)), n / 5, 1)
        assert len(observations) == 1
        assert observations[0].track_id == 1 and not observations[0].association_uncertain


@pytest.mark.parametrize("difference", ["head", "missing_head", "shoulder", "weak_shoulder", "region"])
def test_a_shared_region_without_the_same_visible_upper_body_keeps_two_people(difference):
    full, partial = detections(0)
    if difference == "head":
        partial.keypoints[4] = (204, 192, 0.9)
    elif difference == "missing_head":
        partial.keypoints[4] = (196, 192, 0.3)
    elif difference == "shoulder":
        partial.keypoints[6] = (187, 188, 0.9)
    elif difference == "weak_shoulder":
        partial.keypoints[6] = (179, 188, 0.6)
    else:
        partial.box = Box(x1=100, y1=140, x2=210, y2=200)
    assert len(PoseTracker().update([full, partial], 0, 1)) == 2


def test_alternating_full_and_partial_views_keep_one_observed_identity():
    tracker = PoseTracker()
    for n in range(12):
        full, partial = detections(n / 5)
        observations = [full] if n < 2 else [full, partial] if n < 7 else [partial] if n < 9 else [full]
        output = tracker.update(observations, n / 5, 1)
        assert len(output) == 1 and output[0].track_id == 1
        assert not output[0].association_uncertain
