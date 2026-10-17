from types import SimpleNamespace

import numpy as np
import pytest

from stillwatch.perception import Perception, torso_histogram
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Observation


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def observed(perception, points, box=(30, 30, 220, 210), ts=0):
    result = SimpleNamespace(
        boxes=[SimpleNamespace(xyxy=Tensor([box]), conf=[0.9], id=None)],
        keypoints=SimpleNamespace(data=Tensor([points])),
    )
    perception.pose = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    return perception.people(np.zeros((240, 320, 3), dtype=np.uint8), ts)[0]


def test_clipped_and_nonfinite_joints_cannot_claim_visibility(settings):
    points = [
        (0, 20, 0.99),
        (-1, 20, 0.99),
        (320, 20, 0.99),
        (20, 0, 0.99),
        (20, 240, 0.99),
        (float("nan"), 20, 0.99),
        (20, float("inf"), 0.99),
        (20, 20, float("nan")),
        (0.25, 0.25, 0.9),
        (319.75, 239.75, 0.8),
    ]
    obs = observed(Perception(settings), points)
    assert all(confidence == 0 for _, _, confidence in obs.keypoints[:8])
    assert np.isfinite(obs.keypoints).all()
    assert obs.keypoints[8:] == points[8:]


@pytest.mark.parametrize(
    ("box", "truncated"),
    [
        ((0, 30, 220, 210), True),
        ((30, 30, 320, 210), True),
        ((30, 0, 220, 210), True),
        ((30, 30, 220, 240), True),
        ((30, 30, 220, 210), False),
    ],
)
def test_cropping_is_reported_on_all_four_edges(settings, box, truncated):
    obs = observed(Perception(settings), [(0, 0, 0)] * 17, box)
    assert obs.truncated is truncated


@pytest.mark.parametrize("second_shoulder_x", [300, 320])
def test_partial_descent_requires_two_actually_visible_shoulders(settings, second_shoulder_x):
    perception = Perception(settings)
    rules = MedicalRules(settings)
    before = [(0, 0, 0)] * 17
    before[5], before[6] = (260, 130, 0.95), (300, 130, 0.95)
    for ts in [0, 0.2, 0.4]:
        obs = observed(perception, before, (200, 80, 320, 240), ts)
        assert not rules.update([obs], ts, 1)
    after = before.copy()
    after[5], after[6] = (260, 190, 0.95), (second_shoulder_x, 190, 0.95)
    obs = observed(perception, after, (200, 170, 320, 240), 0.6)
    candidates = rules.update([obs], 0.6, 1)
    assert bool(candidates) is (second_shoulder_x == 300)
    if candidates:
        assert candidates[0].reason == "partial_body_descent"


def torso_points():
    points = [(0, 0, 0)] * 17
    points[5], points[6] = (100, 60, 0.9), (140, 60, 0.9)
    points[11], points[12] = (100, 140, 0.9), (140, 140, 0.9)
    return points


def test_clothing_descriptor_ignores_background_and_moderate_exposure():
    image = np.full((240, 320, 3), (0, 0, 240), np.uint8)
    image[60:141, 100:141] = (0, 160, 0)
    expected = torso_histogram(np.full_like(image, (0, 80, 0)), torso_points())
    actual = torso_histogram(image, torso_points())
    assert len(actual) == 16 and sum(actual) == pytest.approx(1)
    assert actual == expected


@pytest.mark.parametrize("missing", ["weak", "boundary", "nonfinite", "tiny"])
def test_unreliable_torso_has_no_clothing_descriptor(missing):
    points = torso_points()
    if missing == "weak":
        points[11] = (100, 140, 0.2)
    elif missing == "boundary":
        points[11] = (0, 140, 0.9)
    elif missing == "nonfinite":
        points[11] = (float("nan"), 140, 0.9)
    else:
        points = [(x / 20, y / 20, c) for x, y, c in points]
    assert torso_histogram(np.zeros((240, 320, 3), np.uint8), points) == ()


def test_torso_mask_stays_inside_image_at_the_right_edge():
    points = torso_points()
    points[5], points[6] = (313.5, 60, 0.9), (319.9, 60, 0.9)
    points[11], points[12] = (313.5, 140, 0.9), (319.9, 140, 0.9)
    descriptor = torso_histogram(np.zeros((240, 320, 3), np.uint8), points)
    assert len(descriptor) == 16 and sum(descriptor) == pytest.approx(1)


@pytest.mark.parametrize("invalid", [[1], [0.1] * 16, [float("nan")] * 16, [-1, 2] + [0] * 14])
def test_invalid_clothing_input_is_rejected(invalid):
    with pytest.raises(ValueError, match="histogram"):
        Observation(
            capture_ts=0,
            box=Box(x1=0, y1=0, x2=50, y2=100),
            score=1,
            label="person",
            torso_histogram=invalid,
        )


def test_clothing_descriptor_is_internal_to_perception(settings):
    obs = observed(Perception(settings), torso_points())
    assert len(obs.torso_histogram) == 16
    assert "torso_histogram" not in obs.model_dump()
    assert "torso_histogram" not in obs.model_dump_json()
