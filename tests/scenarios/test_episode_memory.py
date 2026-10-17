import pytest

from stillwatch.decisions import parse_response
from stillwatch.motion import MotionClassifier
from stillwatch.triggers import MedicalRules
from stillwatch.types import (
    Box,
    Candidate,
    Category,
    DecisionResult,
    EvidencePacket,
    MotionEvidence,
    Observation,
    Status,
)


def pose(ts, *, low=False, seated=False, track=1, shift=0):
    points = [(0, 0, 0)] * 17
    points[5], points[6] = (40, 20 + shift, 0.95), (60, 20 + shift, 0.95)
    points[11], points[12] = (43, 60 + shift, 0.95), (57, 60 + shift, 0.95)
    points[13], points[14] = (
        (80 if seated else 43, 100 + shift, 0.95),
        (94 if seated else 57, 100 + shift, 0.95),
    )
    points[15], points[16] = (43, 140 + shift, 0.95), (57, 140 + shift, 0.95)
    box = Box(x1=30, y1=shift, x2=100 if seated else 70, y2=150 + shift)
    if low:
        points[5], points[6] = (20, 120 + shift, 0.95), (30, 120 + shift, 0.95)
        points[11], points[12] = (70, 125 + shift, 0.95), (80, 125 + shift, 0.95)
        box = Box(x1=10, y1=100 + shift, x2=150, y2=150 + shift)
    return Observation(capture_ts=ts, box=box, score=0.95, label="person", track_id=track, keypoints=points)


def evidence(**changes):
    values = dict(
        episode_id="episode",
        track_id=1,
        epoch=1,
        started_ts=1,
        observed_ts=20,
        last_seen_ts=20,
        phase="following_concern",
        transition="fall_like",
        posture="low",
        onset_observed=True,
    )
    return MotionEvidence(**(values | changes))


def event(manager, motion):
    return manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=1,
            last_ts=2,
            epoch=1,
            box=Box(x1=0, y1=0, x2=100, y2=60),
            track_id=1,
            motion=motion,
        )
    )[0]


def answer(item, observation="no_concern"):
    return DecisionResult(
        incident_id=item.incident_id,
        packet_id="packet",
        version=item.packet_version,
        medical_status=Status.NO_ALERT,
        fire_status=Status.NO_ALERT,
        medical_observation=observation,
        latest_posture="low",
        provider="test",
        model="fixture",
        raw_answer={"answers": ["NO_ALERT", observation]},
    )


@pytest.mark.parametrize("transition", ["fall_like", "slow_descent", "unknown_origin"])
def test_model_cannot_clear_unresolved_local_episode(system, transition):
    manager = system[4]
    item = event(manager, evidence(transition=transition))
    result = answer(item)
    raw = result.raw_answer.copy()
    manager.apply(result, 20)
    assert item.analysis_status == Status.UNKNOWN
    assert item.policy_reason == "local_episode_unresolved"
    assert result.raw_answer == raw and result.fire_status == Status.NO_ALERT


def test_ordinary_activity_requires_local_and_cloud_agreement_and_retains_episode(system):
    manager = system[4]
    motion = evidence(transition="controlled_pattern", phase="ordinary_activity", clear_eligible=True)
    item = event(manager, motion)
    manager.apply(answer(item), 20)
    assert item.ordinary_activity_confirmed and item.analysis_status == Status.NO_ALERT
    manager.archive_clear(50, 1)
    assert item.lifecycle == "OPEN"
    repeated, created = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="persistent_low_posture",
            first_ts=1,
            last_ts=50,
            epoch=1,
            track_id=1,
            motion=motion.model_copy(update={"observed_ts": 50, "last_seen_ts": 50}),
        )
    )
    assert not created and repeated.incident_id == item.incident_id


def test_loss_invalidates_ordinary_activity_without_waiting_for_cloud(system):
    manager = system[4]
    item = event(
        manager, evidence(transition="controlled_pattern", phase="ordinary_activity", clear_eligible=True)
    )
    manager.apply(answer(item), 20)
    manager.observe_motion(
        [
            item.motion.model_copy(
                update={"phase": "visibility_lost", "clear_eligible": False, "observed_ts": 21}
            )
        ],
        21,
    )
    assert not item.ordinary_activity_confirmed
    assert item.analysis_status == Status.UNKNOWN


def test_new_episode_during_cloud_request_cannot_clear_new_concern(system):
    manager = system[4]
    item = event(manager, evidence(phase="recovered_activity", posture="upright", clear_eligible=True))
    manager.observe_motion([evidence(episode_id="new", started_ts=19)], 20)
    manager.apply(answer(item, "recovered"), 20)
    assert item.policy_reason == "new_transition_after_assessment"
    assert item.analysis_status == Status.UNKNOWN


def test_short_tracking_fragment_keeps_deadline_and_retains_uncertainty(system):
    manager = system[4]
    item = event(manager, evidence(phase="visibility_lost", last_seen_ts=2, observed_ts=2.6))
    item.reason = "view_lost_after_suspicion"
    item.last_ts = 2.6
    resumed, created = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="persistent_low_posture",
            first_ts=2.8,
            last_ts=4.8,
            epoch=1,
            box=item.box,
            track_id=2,
            motion=evidence(episode_id="fragment", track_id=2, observed_ts=4.8, last_seen_ts=4.8),
        )
    )
    assert not created and resumed.incident_id == item.incident_id
    assert resumed.detected_ts == 2 and resumed.identity_uncertain
    assert resumed.track_ids == [1, 2]
    manager.apply(answer(resumed), 4.8)
    assert resumed.analysis_status == Status.UNKNOWN


def test_operator_closure_does_not_suppress_a_later_transition(system):
    manager = system[4]
    item = event(manager, evidence())
    manager.annotate(item.incident_id, "close")
    continuation = Candidate(
        category=Category.MEDICAL,
        reason="persistent_low_posture",
        first_ts=1,
        last_ts=3,
        epoch=1,
        track_id=1,
        box=item.box,
        motion=evidence(),
    )
    assert manager.ingest(continuation)[0].incident_id == item.incident_id
    next_fall = continuation.model_copy(
        update={"reason": "rapid_posture_change", "first_ts": 4, "last_ts": 4.8}
    )
    new, created = manager.ingest(next_fall)
    assert created and new.incident_id != item.incident_id
    continued, created = manager.ingest(continuation.model_copy(update={"last_ts": 5}))
    assert not created and continued.incident_id == new.incident_id


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"identity_uncertain": True}, "local_identity_uncertain"),
        ({"last_seen_ts": 18}, "local_visibility_lost"),
    ],
)
def test_recovery_cannot_override_identity_or_visibility_loss(system, changes, reason):
    manager = system[4]
    item = event(manager, evidence(phase="recovered_activity", clear_eligible=True, **changes))
    manager.apply(answer(item, "recovered"), 20)
    assert item.policy_reason == reason and item.analysis_status == Status.UNKNOWN


def test_only_sustained_same_person_recovery_supports_clearance(settings):
    classifier = MotionClassifier(settings)
    for n in range(8):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    fall = Candidate(
        category=Category.MEDICAL,
        reason="rapid_posture_change",
        first_ts=1.4,
        last_ts=1.6,
        track_id=1,
        epoch=1,
    )
    classifier.update([pose(1.6, low=True)], 1.6, 1, [fall])
    classifier.update([pose(1.8, low=True), pose(1.8, track=2, shift=-300)], 1.8, 1, [])
    assert not classifier.tracks[1].evidence.clear_eligible
    # The first upright sample still unfolds from the observed low pose.
    for n in range(10, 29):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    assert not classifier.tracks[1].evidence.clear_eligible
    classifier.update([pose(5.8)], 5.8, 1, [])
    assert classifier.tracks[1].evidence.phase == "recovered_activity"


def test_repeated_fall_opens_episode_without_inheriting_recovery(settings):
    rules = MedicalRules(settings)
    for n in range(8):
        rules.update([pose(n / 5)], n / 5, 1)
    first = rules.update([pose(1.6, low=True)], 1.6, 1)[0].motion.episode_id
    for n in range(9, 34):
        rules.update([pose(n / 5)], n / 5, 1)
    next_fall = rules.update([pose(6.8, low=True)], 6.8, 1)[0]
    assert next_fall.motion.episode_id != first
    assert not next_fall.motion.clear_eligible


def test_partial_recovery_and_missing_origin_do_not_clear(settings):
    rules = MedicalRules(settings)
    for n in range(20):
        rules.update([pose(n / 5, low=True)], n / 5, 1)
    state = rules.motion.tracks[1].evidence
    assert state.transition == "unknown_origin" and not state.clear_eligible
    for n in range(20, 40):
        rules.update([pose(n / 5, seated=True)], n / 5, 1)
    assert not rules.motion.tracks[1].evidence.clear_eligible


def test_prior_seated_baseline_can_support_recovery(settings):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    fall = Candidate(
        category=Category.MEDICAL, reason="rapid_posture_change", first_ts=1.8, last_ts=2, track_id=1, epoch=1
    )
    classifier.update([pose(2, low=True)], 2, 1, [fall])
    for n in range(11, 28):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    state = classifier.tracks[1].evidence
    assert state.recovery_basis == "prior_supported_posture" and state.clear_eligible
    packet = EvidencePacket(
        "p",
        "i",
        1,
        Category.MEDICAL,
        (),
        "test",
        "test",
        6,
        temporal_context={"local_motion": state.model_dump()},
    )
    result = parse_response(
        packet,
        {
            "model": "gpt-6-luna",
            "answers": [
                {"name": name, "type": "choice", "choice": value}
                for name, value in {
                    "medical_status": "NO_ALERT",
                    "fire_status": "NO_ALERT",
                    "medical_observation": "recovered",
                    "latest_posture": "seated_on_furniture",
                }.items()
            ],
        },
    )
    assert result.medical_status == Status.NO_ALERT


def test_motion_memory_is_bounded_and_resets_with_source(settings):
    classifier = MotionClassifier(settings)
    for n in range(200):
        classifier.update([pose(n / 10, track=n, shift=n * 200)], n / 10, 1, [])
    assert len(classifier.tracks) <= classifier.max_tracks
    assert all(len(t.history) <= 120 for t in classifier.tracks.values())
    classifier.update([pose(0, low=True, track=1)], 0, 2, [])
    assert len(classifier.tracks) == 1 and not classifier.tracks[1].evidence.clear_eligible


def test_supported_recline_is_remembered_until_a_new_descent(settings):
    classifier = MotionClassifier(settings)
    # Stable flexed posture, then recline with almost unchanged torso height.
    for n in range(10):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    for n in range(10, 31):
        classifier.update([pose(n / 5, low=True, shift=-80)], n / 5, 1, [])
    state = classifier.tracks[1].evidence
    assert state.transition == "controlled_pattern" and state.clear_eligible
    assert state.phase == "ordinary_activity"
    episode = state.episode_id
    classifier.update([pose(6.2, low=True, shift=-35)], 6.2, 1, [])
    changed = classifier.tracks[1].evidence
    assert changed.episode_id != episode and not changed.clear_eligible


@pytest.mark.parametrize("upright", [True, False])
def test_upward_movement_invalidates_support_without_opening_a_descent(settings, upright):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5, seated=True, shift=80)], n / 5, 1, [])
    for n in range(10, 31):
        classifier.update([pose(n / 5, low=True)], n / 5, 1, [])
    state = classifier.tracks[1].evidence
    assert state.phase == "ordinary_activity" and state.clear_eligible
    episode = state.episode_id

    rising = pose(6.2, shift=40) if upright else pose(6.2, low=True, shift=-40)
    assert not classifier.update([rising], 6.2, 1, [])
    state = classifier.tracks[1].evidence
    assert state.episode_id == episode and not state.clear_eligible
    assert classifier.tracks[1].support is None

    if upright:
        for n in range(32, 48):
            assert not classifier.update([pose(n / 5, shift=40)], n / 5, 1, [])
        state = classifier.tracks[1].evidence
        assert state.episode_id == episode and state.phase == "recovered_activity"
    else:
        # A partial rise cannot clear the episode. A subsequent actual drop
        # still starts a fresh event after the previously observed activity.
        for ts in (6.4, 6.6):
            assert not classifier.update([pose(ts, low=True, shift=-40)], ts, 1, [])
        emitted = classifier.update([pose(6.8, low=True, shift=40)], 6.8, 1, [])
        assert any(c.reason == "slow_posture_descent" for c in emitted)
        state = classifier.tracks[1].evidence
        assert state.episode_id != episode and not state.clear_eligible


def test_slow_slide_is_a_concern_even_without_a_rapid_trigger(settings):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5, seated=True)], n / 5, 1, [])
    emitted = []
    for n in range(10, 31):
        # A slow loss of torso height and support; no raw rapid-change candidate.
        emitted.extend(classifier.update([pose(n / 5, low=True, shift=-80 + (n - 10) * 3)], n / 5, 1, []))
    assert any(c.reason in ("slow_posture_descent", "support_pattern_lost") for c in emitted)
    assert not classifier.tracks[1].evidence.clear_eligible
