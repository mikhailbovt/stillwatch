from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Observation


def person(ts, box, track_id=1):
    return Observation(
        capture_ts=ts,
        box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        label="person",
        score=0.4,
        track_id=track_id,
    )


def test_partial_drop_then_pose_loss_is_independent_of_primary_trigger(settings):
    rules = MedicalRules(settings)
    assert not rules.update([person(0, (10, 0, 50, 100))], 0, 1)
    # Below the primary drop threshold, but height shrinks during downward motion.
    assert not rules.update([person(0.4, (10, 35, 60, 105))], 0.4, 1)
    found = rules.update([], 1.0, 1)
    assert found and found[0].reason == "view_lost_after_suspicion"
    assert found[0].limited


def test_concern_first_emitted_during_absence_keeps_identity_uncertain_on_return(settings):
    rules = MedicalRules(settings)
    rules.update([person(0, (10, 0, 50, 100))], 0, 1)
    assert not rules.update([person(0.4, (10, 35, 60, 105))], 0.4, 1)
    candidate = rules.update([], 1.0, 1)[0]
    assert candidate.motion.phase == "visibility_lost"
    assert candidate.motion.transition == "unknown_origin"
    assert not candidate.motion.onset_observed
    # Retaining the geometric ID does not identify the person after the gap.
    # The first concern arose without an observation, so it must survive the
    # normal observed-track update path when the region becomes visible again.
    for n in range(6, 30):
        rules.update([person(n / 5, (10, 0, 50, 100))], n / 5, 1)
        state = rules.motion.snapshot()[0]
        assert state.episode_id == candidate.motion.episode_id
        assert state.identity_uncertain and not state.clear_eligible
        assert state.transition == "unknown_origin"


def test_low_confidence_untracked_lying_region_can_still_trigger(settings):
    rules = MedicalRules(settings)
    found = []
    # A weak new region needs three observations before starting the low-pose timer.
    for i in range(14):
        found.extend(rules.update([person(i * 0.2, (10, 40, 130, 80), track_id=None)], i * 0.2, 1))
    assert found and found[0].reason == "persistent_low_posture"
    assert found[0].track_id == 1 and found[0].limited
    assert len({m.track_id for m in rules.motion.snapshot()}) == 1


def test_recovered_visible_posture_expires_weak_motion_watch(settings):
    rules = MedicalRules(settings)
    rules.update([person(0, (10, 0, 50, 100))], 0, 1)
    rules.update([person(0.4, (10, 35, 60, 105))], 0.4, 1)
    for i in range(4, 12):
        rules.update([person(i * 0.2, (10, 0, 50, 100))], i * 0.2, 1)
    assert not rules.update([], 3, 1)
