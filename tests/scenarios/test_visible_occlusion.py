import pytest
from test_episode_memory import pose

from stillwatch.tracking import PoseTracker
from stillwatch.triggers import MedicalRules


def fragment(ts, *, low=False):
    obs = pose(ts, low=low, track=None)
    obs.keypoints = [(0, 0, 0)] * 16 + [obs.keypoints[16]]
    return obs


def test_visible_fragment_breaks_continuity_through_reacquisition():
    tracker = PoseTracker()
    for n in range(3):
        found = tracker.update([pose(n / 5), pose(n / 5, shift=90)], n / 5, 1)
        assert len(found) == 2 and not any(o.association_uncertain for o in found)
    first_id = found[0].track_id
    weak = fragment(0.6)
    found = tracker.update([weak, pose(0.6, shift=90)], 0.6, 1)
    current = next(o for o in found if o.track_id == first_id)
    assert current.association_uncertain
    assert current.box == weak.box and current.keypoints == weak.keypoints
    assert not weak.association_uncertain  # The caller's detection is unchanged.
    assert tracker.update([pose(0.8)], 0.8, 1)[0].association_uncertain
    resumed = tracker.update([pose(1)], 1, 1)[0]
    assert resumed.track_id == first_id and not resumed.association_uncertain


@pytest.mark.parametrize("context", ["alone", "separate_person", "visible_torso", "box_adapter"])
def test_missing_landmarks_alone_do_not_imply_another_identity(context):
    tracker = PoseTracker()
    for n in range(3):
        tracker.update([pose(n / 5)], n / 5, 1)
    obs = fragment(0.6)
    if context == "visible_torso":
        complete = pose(0.6)
        for k in (5, 11):
            obs.keypoints[k] = complete.keypoints[k]
    if context == "box_adapter":
        obs.keypoints = []
    companions = [] if context == "alone" else [pose(0.6, shift=220 if context == "separate_person" else 90)]
    found = tracker.update([obs, *companions], 0.6, 1)
    assert not next(o for o in found if o.track_id == 1).association_uncertain


def test_occlusion_retains_an_existing_fall_concern(settings):
    rules = MedicalRules(settings)
    for n in range(8):
        rules.update([pose(n / 5)], n / 5, 1)
    candidates = rules.update([pose(1.6, low=True)], 1.6, 1)
    assert candidates
    first = rules.motion.tracks[1].evidence
    rules.update([fragment(1.8, low=True), pose(1.8, shift=90)], 1.8, 1)
    current = rules.motion.tracks[1].evidence
    assert current.episode_id == first.episode_id
    assert current.identity_uncertain and not current.clear_eligible
    assert rules.motion.tracks[1].concern
