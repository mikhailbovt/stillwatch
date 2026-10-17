import pytest
from test_episode_memory import pose

from stillwatch.config import Settings
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Observation


class RecordedAssociation:
    """Hold the already observed identities fixed; exercise motion references."""

    def reset(self):
        pass

    def update(self, observations, ts, epoch):
        return observations


@pytest.fixture
def rules():
    value = MedicalRules(Settings(fire_enabled=False))
    value.tracker = RecordedAssociation()
    return value


def measured(ts, *, compressed=False, uncertain=False, track=1, shift=0):
    top, bottom = (450, 540) if compressed else (400, 550)
    shoulder, hip, knee, ankle = (470, 490, 515, 538) if compressed else (432, 472, 510, 548)
    points = [(0, 0, 0)] * 17
    for k, x, y in (
        (5, 24, shoulder),
        (6, 44, shoulder),
        (11, 26, hip),
        (12, 42, hip),
        (13, 28, knee),
        (14, 40, knee),
        (15, 28, ankle),
        (16, 40, ankle),
    ):
        points[k] = (x + shift, y, 0.95)
    return Observation(
        capture_ts=ts,
        box=Box(x1=10 + shift, y1=top, x2=60 + shift, y2=bottom),
        score=0.95,
        label="person",
        track_id=track,
        keypoints=points,
        association_uncertain=uncertain,
    )


def test_one_clean_pose_after_ambiguity_cannot_establish_a_collapse(rules):
    # A mixed-body estimate briefly extends the torso, then returns to the
    # measured smaller body. Identity uncertainty preceded that single reference.
    rules.update([measured(0, compressed=True, uncertain=True)], 0, 1)
    rules.update([measured(0.2)], 0.2, 1)
    assert not rules.update([measured(0.4, compressed=True)], 0.4, 1)


def test_startup_retains_a_strong_observed_descent(rules):
    rules.update([measured(0)], 0, 1)
    events = rules.update([measured(0.2, compressed=True)], 0.2, 1)
    assert any(c.reason == "rapid_posture_change" for c in events)


def test_descent_uses_a_reference_established_after_reacquisition(rules):
    rules.update([measured(0, compressed=True, uncertain=True)], 0, 1)
    for ts in (0.2, 0.4, 0.6):
        rules.update([measured(ts)], ts, 1)
    events = rules.update([measured(0.8, compressed=True)], 0.8, 1)
    descent = next(c for c in events if c.reason == "rapid_posture_change")
    assert descent.first_ts == 0.6
    assert descent.last_ts == 0.8


def test_ambiguity_for_one_person_does_not_delay_another(rules):
    rules.update([measured(0, uncertain=True), measured(0, track=2, shift=200)], 0, 1)
    events = rules.update([measured(0.2), measured(0.2, compressed=True, track=2, shift=200)], 0.2, 1)
    assert any(c.track_id == 2 and c.reason == "rapid_posture_change" for c in events)


def test_reacquisition_does_not_clear_or_restart_an_existing_concern(rules):
    for n in range(8):
        rules.update([pose(n / 5)], n / 5, 1)
    assert rules.update([pose(1.6, low=True)], 1.6, 1)
    episode = rules.motion.tracks[1].evidence.episode_id
    onset = rules.motion.tracks[1].evidence.started_ts
    rules.update([pose(1.8, low=True).model_copy(update={"association_uncertain": True})], 1.8, 1)
    for ts in (2, 2.2, 2.4):
        rules.update([pose(ts, low=True)], ts, 1)
        state = rules.motion.tracks[1]
        assert state.concern and not state.evidence.clear_eligible
        assert state.evidence.episode_id == episode and state.evidence.started_ts == onset


def test_source_switch_does_not_keep_another_sources_ambiguity(rules):
    rules.update([measured(0, uncertain=True)], 0, 1)
    rules.update([measured(0.2)], 0.2, 1)
    rules.update([measured(0)], 0, 2)
    assert not rules.descent_boundary
    events = rules.update([measured(0.2, compressed=True)], 0.2, 2)
    assert any(c.reason == "rapid_posture_change" for c in events)


@pytest.mark.parametrize("rate,window,last", [(5, 5, 21), (30, 2, 41)])
def test_quarantined_reference_stays_excluded_after_history_evicts_ambiguity(rules, rate, window, last):
    rules.settings.pose_fps = rate
    rules.settings.drop_window_seconds = window
    rules.update([measured(0, compressed=True, uncertain=True)], 0, 1)
    rules.update([measured(1 / rate)], 1 / rate, 1)
    # Either age pruning or the 40-item bound removes the ambiguity marker
    # while the first mixed-body pose is still a possible descent reference.
    for n in range(2, last + 1):
        ts = n / rate
        events = rules.update([measured(ts, compressed=True)], ts, 1)
        assert not any(c.reason == "rapid_posture_change" for c in events)


def test_track_expiry_releases_reference_boundary(rules):
    rules.update([measured(0, uncertain=True)], 0, 1)
    assert len(rules.descent_boundary) == 1
    rules.update([], rules.settings.dedup_seconds + 1, 1)
    assert not rules.descent_boundary
