import pytest
from test_episode_memory import pose
from test_reacquisition_context import RecordedAssociation

from stillwatch.config import Settings
from stillwatch.motion import MotionClassifier
from stillwatch.triggers import MedicalRules


def test_identical_geometry_keeps_each_observation_time_and_identity():
    motion = MotionClassifier(Settings(fire_enabled=False))
    first = pose(1)
    second = pose(2).model_copy(update={"track_id": 2})
    with motion.frame_poses():
        a, b = motion.pose(first), motion.pose(second)
        assert a.ts == 1 and b.ts == 2
        assert a.observation.track_id == 1 and b.observation.track_id == 2
    assert motion._pose_cache is None


def test_reused_input_is_measured_again_in_a_later_frame():
    motion = MotionClassifier(Settings(fire_enabled=False))
    observation = pose(0)
    with motion.frame_poses():
        before = motion.pose(observation)
    observation.box = pose(1, low=True).box
    observation.keypoints = pose(1, low=True).keypoints
    observation.capture_ts = 1
    with motion.frame_poses():
        after = motion.pose(observation)
    assert not before.low and after.low
    assert after.observation is observation
    assert motion._pose_cache is None


def test_direct_pose_calls_use_current_settings_and_observation_flags():
    settings = Settings(fire_enabled=False)
    motion = MotionClassifier(settings)
    observation = pose(0)
    assert motion.pose(observation).reliable
    observation.limited = True
    assert not motion.pose(observation).reliable
    observation.limited = False
    settings.keypoint_confidence = 1
    assert not motion.pose(observation).torso_visible


def test_rules_release_frame_measurements_after_an_exception(monkeypatch):
    rules = MedicalRules(Settings(fire_enabled=False))

    def fail(observations, ts, epoch):
        rules.motion.pose(observations[0])
        raise ValueError("interrupted update")

    monkeypatch.setattr(rules, "_update", fail)
    with pytest.raises(ValueError, match="interrupted update"):
        rules.update([pose(0)], 0, 1)
    assert rules.motion._pose_cache is None
    observation = pose(1).model_copy(update={"limited": True})
    assert not rules.motion.pose(observation).reliable


def test_source_replacement_does_not_reuse_another_sources_measurement():
    rules = MedicalRules(Settings(fire_enabled=False))
    rules.tracker = RecordedAssociation()
    observation = pose(0)
    rules.update([observation], 0, 1)
    assert rules.motion.pose(observation).reliable
    observation.limited = True
    rules.update([observation], 0, 2)
    assert not rules.motion.pose(observation).reliable
    assert rules.motion._pose_cache is None
