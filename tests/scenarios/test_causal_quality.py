import itertools
import random

import pytest
from test_episode_memory import pose

from stillwatch.motion import MotionClassifier
from stillwatch.tracking import PoseTracker, minimum_assignment
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Candidate, Category, Observation


def test_assignment_matches_exhaustive_optimum():
    rng = random.Random(21)
    for rows, columns in ((1, 3), (2, 3), (3, 5), (4, 6)):
        for _ in range(12):
            costs = [[rng.random() for _ in range(columns)] for _ in range(rows)]
            result = minimum_assignment(costs)
            optimum = min(
                sum(costs[i][j] for i, j in enumerate(order))
                for order in itertools.permutations(range(columns), rows)
            )
            assert len(set(result)) == rows
            assert sum(costs[i][j] for i, j in enumerate(result)) == pytest.approx(optimum)


def test_assignment_does_not_steal_the_only_match_of_another_person():
    assert minimum_assignment([[0.05, 0.15, 0.6, 0.6], [0.1, 1e6, 0.6, 0.6]]) == [1, 0]
    assert minimum_assignment([]) == []


def test_source_epoch_cannot_reuse_previous_person_geometry(settings):
    rules = MedicalRules(settings)
    rules.update([pose(0)], 0, 1)
    assert not rules.update([pose(0.4, low=True)], 0.4, 2)
    assert all(len(history) == 1 for history in rules.history.values())
    assert all(m.epoch == 2 and m.transition == "none" for m in rules.motion.snapshot())


def test_ambiguity_is_not_cleared_by_matching_order():
    tracker = PoseTracker()
    # Two distinct nearby people, both plausible matches after an occlusion.
    initial = [pose(0, track=None), pose(0, shift=40, track=None)]
    assert len(tracker.update(initial, 0, 1)) == 2
    current = tracker.update([pose(0.4, shift=15, track=None), pose(0.4, shift=25, track=None)], 0.4, 1)
    assert len(current) == 2
    assert all(o.association_uncertain for o in current)


def test_tentative_fragment_does_not_replace_an_established_person():
    tracker = PoseTracker()
    main = tracker.update([pose(0, track=None)], 0, 1)[0]
    fragment = pose(0.2, shift=30, track=None).model_copy(update={"score": 0.3})
    assert len(tracker.update([pose(0.2, track=None), fragment], 0.2, 1)) == 1
    observed = tracker.update([pose(0.4, shift=30, track=None)], 0.4, 1)
    assert len(observed) == 1 and observed[0].track_id == main.track_id


def test_weak_fragment_cannot_displace_a_confident_person(monkeypatch):
    tracker = PoseTracker()
    initial = tracker.update([pose(0, track=None)], 0, 1)[0]
    monkeypatch.setattr(tracker, "_cost", lambda track, obs: 0.2 if obs.score >= 0.5 else 0.01)
    weak = pose(0.2, shift=70, track=None).model_copy(update={"score": 0.3})
    observed = tracker.update([weak, pose(0.2, shift=30, track=None)], 0.2, 1)
    assert len(observed) == 1
    assert observed[0].track_id == initial.track_id and observed[0].score == 0.95


def test_observed_direction_preserves_ids_during_a_crossing():
    tracker = PoseTracker()

    def at(ts, x):
        obs = pose(ts, track=None)
        obs.box.x1 += x
        obs.box.x2 += x
        obs.keypoints = [(px + x, py, confidence) for px, py, confidence in obs.keypoints]
        return obs

    initial = tracker.update([at(0, 0), at(0, 120)], 0, 1)
    tracker.update([at(0.2, 40), at(0.2, 80)], 0.2, 1)
    after = tracker.update([at(0.4, 80), at(0.4, 40)], 0.4, 1)
    assert [o.track_id for o in after] == [o.track_id for o in initial]


@pytest.mark.parametrize("scale", [2, 3])
def test_close_foreground_person_cannot_take_a_background_persons_id(scale):
    tracker = PoseTracker()
    initial = tracker.update([pose(0, track=None)], 0, 1)[0]
    foreground = pose(0.2, track=None)
    foreground.keypoints = [(x * scale, y * scale, c) for x, y, c in foreground.keypoints]
    foreground.box = Box(x1=30 * scale, y1=0, x2=70 * scale, y2=150 * scale)
    current = tracker.update([foreground], 0.2, 1)[0]
    assert current.track_id != initial.track_id
    # Keep the hidden person's observations available for a later match.
    assert tracker.tracks[initial.track_id].current.capture_ts == 0


def test_gradual_approach_preserves_identity():
    tracker = PoseTracker()
    identities = set()
    for n in range(12):
        scale = 1 + n * 0.1
        current = pose(n / 5, track=None)
        current.keypoints = [(x * scale, y * scale, c) for x, y, c in current.keypoints]
        current.box = Box(x1=30 * scale, y1=0, x2=70 * scale, y2=150 * scale)
        observed = tracker.update([current], n / 5, 1)[0]
        identities.add(observed.track_id)
        assert not observed.association_uncertain
    assert len(identities) == 1


def test_horizontal_fall_keeps_identity_despite_box_width_growth():
    tracker = PoseTracker()
    initial = tracker.update([pose(0, track=None)], 0, 1)[0]
    fallen = tracker.update([pose(0.2, low=True, track=None)], 0.2, 1)[0]
    assert fallen.track_id == initial.track_id and not fallen.association_uncertain


def colored_pose(ts, bin_index, **kwargs):
    return pose(ts, **kwargs).model_copy(
        update={"torso_histogram": tuple(float(i == bin_index) for i in range(16))}
    )


@pytest.mark.parametrize("same_clothing", [True, False])
def test_clothing_conflict_prevents_a_new_descent_from_mixed_identity(settings, same_clothing):
    rules = MedicalRules(settings)
    rules.update([colored_pose(0, 0, track=None)], 0, 1)
    current = colored_pose(0.2, 0 if same_clothing else 6, shift=50, track=None)
    current.box.y2 = 150
    current.keypoints[13:] = [(0, 0, 0)] * 4
    candidate = rules.update([current], 0.2, 1)
    assert bool(candidate) is same_clothing
    assert rules.observations[0].association_uncertain is not same_clothing
    assert not any(m.clear_eligible for m in rules.motion.snapshot())


def test_rotating_fall_keeps_its_candidate_despite_changed_visible_clothing(settings):
    rules = MedicalRules(settings)
    rules.update([colored_pose(0, 0, track=None)], 0, 1)
    candidates = rules.update([colored_pose(0.2, 6, low=True, track=None)], 0.2, 1)
    assert candidates and candidates[0].reason == "rapid_posture_change"
    assert not rules.observations[0].association_uncertain


@pytest.mark.parametrize("gap", [0.2, 0.4])
def test_clothing_conflict_requires_close_source_times(gap):
    tracker = PoseTracker()
    initial = tracker.update([colored_pose(0, 0, track=None)], 0, 1)[0]
    current = tracker.update([colored_pose(gap, 6, track=None)], gap, 1)[0]
    assert current.track_id == initial.track_id
    assert current.association_uncertain is (gap == 0.2)
    assert not tracker.update([colored_pose(gap + 0.2, 0, track=None)], gap + 0.2, 2)[0].association_uncertain


def test_clothing_agreement_cannot_resolve_geometric_ambiguity():
    tracker = PoseTracker()
    tracker.update([colored_pose(0, 0, track=None), colored_pose(0, 0, shift=40, track=None)], 0, 1)
    current = tracker.update(
        [colored_pose(0.2, 0, shift=15, track=None), colored_pose(0.2, 0, shift=25, track=None)], 0.2, 1
    )
    assert len(current) == 2 and all(o.association_uncertain for o in current)


@pytest.mark.parametrize("visible_pair", [True, False])
def test_large_box_growth_with_insufficient_scale_evidence_is_uncertain(visible_pair):
    tracker = PoseTracker()
    before = pose(0, track=None)
    before.keypoints = [(0, 0, 0)] * 17
    before.keypoints[0], before.keypoints[6] = (50, 10, 0.95), (60, 20, 0.95)
    first = tracker.update([before], 0, 1)[0]
    after = before.model_copy(deep=True)
    after.capture_ts = 0.2
    after.box = Box(x1=40, y1=10, x2=140, y2=300)
    after.keypoints[0] = (95, 60, 0.95)
    after.keypoints[6] = (75, 140, 0.95 if visible_pair else 0.1)
    current = tracker.update([after], 0.2, 1)[0]
    assert current.track_id == first.track_id
    assert current.association_uncertain


def test_larger_visible_box_with_unchanged_body_geometry_preserves_continuity():
    tracker = PoseTracker()
    before = pose(0, track=None)
    before.keypoints = [(0, 0, 0)] * 17
    before.keypoints[0], before.keypoints[6] = (50, 10, 0.95), (60, 20, 0.95)
    first = tracker.update([before], 0, 1)[0]
    after = before.model_copy(deep=True)
    after.capture_ts = 0.2
    after.box = Box(x1=30, y1=0, x2=70, y2=300)
    current = tracker.update([after], 0.2, 1)[0]
    assert current.track_id == first.track_id and not current.association_uncertain


@pytest.mark.parametrize("weak_ts", [0.2, 1.0])
def test_recent_visible_body_cannot_be_replaced_through_a_weak_pose(weak_ts):
    tracker = PoseTracker()
    first = tracker.update([pose(0, track=None)], 0, 1)[0]
    weak = pose(weak_ts, track=None)
    weak.box = Box(x1=60, y1=0, x2=140, y2=300)
    weak.keypoints = [(0, 0, 0)] * 17
    intermediate = tracker.update([weak], weak_ts, 1)[0]
    assert intermediate.track_id == first.track_id
    foreground = pose(weak_ts + 0.2, track=None)
    foreground.box = weak.box
    foreground.keypoints = [(x * 2, y * 2, c) for x, y, c in foreground.keypoints]
    current = tracker.update([foreground], weak_ts + 0.2, 1)[0]
    # A short missing-pose interval must not erase known scale. A reference
    # older than the association window cannot veto a current geometric match.
    assert (current.track_id != first.track_id) is (weak_ts == 0.2)


@pytest.mark.parametrize("background_count", [0, 6])
@pytest.mark.parametrize("reverse", [False, True])
def test_entering_person_cannot_rotate_ids_of_people_still_visible(background_count, reverse):
    tracker = PoseTracker()

    def at(ts, x):
        obs = pose(ts, track=None)
        obs.box.x1 += x
        obs.box.x2 += x
        obs.keypoints = [(px + x, py, confidence) for px, py, confidence in obs.keypoints]
        return obs

    background = [500 + 200 * n for n in range(background_count)]
    first = tracker.update([at(0, x) for x in [0, 55, 110, *background]], 0, 1)
    before = {o.box.x1: o.track_id for o in first}
    positions = [55, 110, 165, *background]
    observed = tracker.update([at(0.2, x) for x in positions[:: -1 if reverse else 1]], 0.2, 1)
    after = {o.box.x1: o.track_id for o in observed}
    assert after[85] == before[85] and after[140] == before[140]
    assert after[195] not in before.values()
    assert len(set(after.values())) == len(positions)
    for x in background:
        assert after[30 + x] == before[30 + x]


@pytest.mark.parametrize("joint", [5, 6])
def test_isolated_torso_joint_after_occlusion_cannot_create_a_descent(settings, joint):
    rules = MedicalRules(settings)
    rules.update([pose(0)], 0, 1)
    current = pose(0.4)
    current.box = Box(x1=30, y1=80, x2=70, y2=150)
    current.keypoints = [(0, 0, 0)] * 17
    current.keypoints[joint] = (40 if joint == 5 else 60, 105, 0.99)
    assert not rules.update([current], 0.4, 1)
    assert not rules.update([], 1.2, 1)


@pytest.mark.parametrize("joints", [(5, 6), (5, 11)])
def test_two_visible_torso_joints_retain_a_partial_descent(settings, joints):
    rules = MedicalRules(settings)
    previous = pose(0)
    rules.update([previous], 0, 1)
    current = pose(0.4)
    current.box = Box(x1=30, y1=80, x2=70, y2=150)
    current.keypoints = [(0, 0, 0)] * 17
    for joint in joints:
        x, y, confidence = previous.keypoints[joint]
        current.keypoints[joint] = (x, y + 80, confidence)
    assert any(c.reason == "rapid_posture_change" for c in rules.update([current], 0.4, 1))


def partial_pose(ts, top, shoulder, *, confidence=0.95):
    obs = pose(ts, track=None)
    obs.box = Box(x1=30, y1=top, x2=130, y2=240)
    obs.truncated = True
    obs.keypoints = [(0, 0, 0)] * 17
    obs.keypoints[5], obs.keypoints[6] = (60, shoulder, confidence), (90, shoulder, confidence)
    return obs


def test_cropped_descent_retains_an_unresolved_episode(settings):
    rules = MedicalRules(settings)
    for n in range(10):
        assert not rules.update([partial_pose(n / 5, 80, 150)], n / 5, 1)
    rules.update([partial_pose(2, 115, 165)], 2, 1)
    candidates = rules.update([partial_pose(2.2, 170, 190)], 2.2, 1)
    assert any(c.reason == "partial_body_descent" and c.limited for c in candidates)
    state = rules.motion.snapshot()[0]
    assert state.onset_observed and not state.clear_eligible
    rules.update([], 3, 1)
    assert rules.motion.snapshot()[0].phase == "visibility_lost"


def test_end_on_compression_requires_torso_evidence_and_visible_descent(settings):
    before = partial_pose(10.6, 80, 150)
    after = partial_pose(11.4, 170, 156)
    after.keypoints[5], after.keypoints[6] = (68, 156, 0.95), (82, 156, 0.95)
    after.keypoints[11], after.keypoints[12] = (70, 210, 0.95), (80, 210, 0.95)
    rules = MedicalRules(settings)
    rules.update([before], 10.6, 1)
    assert any(c.reason == "partial_body_descent" for c in rules.update([after], 11.4, 1))
    after.keypoints[12] = (80, 210, 0.2)
    assert not rules._partial_descent(before, after)
    after.keypoints[12] = (80, 210, 0.95)
    after.keypoints[5], after.keypoints[6] = (68, 150, 0.95), (82, 150, 0.95)
    assert not rules._partial_descent(before, after)


@pytest.mark.parametrize("kind", ["stable", "box_resize", "weak_shoulders", "moving_up"])
def test_partial_view_alone_is_not_a_descent(settings, kind):
    rules = MedicalRules(settings)
    for n in range(10):
        rules.update([partial_pose(n / 5, 80, 150)], n / 5, 1)
    obs = partial_pose(2, 170 if kind != "stable" else 80, 190 if kind == "weak_shoulders" else 150)
    if kind == "weak_shoulders":
        obs.keypoints[6] = (90, 190, 0.2)
    if kind == "moving_up":
        obs.keypoints[5], obs.keypoints[6] = (60, 130, 0.95), (90, 130, 0.95)
    assert not rules.update([obs], 2, 1)


def test_descent_towards_camera_followed_by_loss_keeps_suspicion(settings):
    rules = MedicalRules(settings)
    for n in range(10):
        before = pose(n / 5)
        before.keypoints[0] = (50, 5, 0.95)
        rules.update([before], n / 5, 1)
    # Perspective can make the box grow while a person starts descending.
    incoming = pose(2, shift=50)
    incoming.keypoints[0] = (50, 55, 0.95)
    incoming.truncated = True
    incoming.box = Box(x1=10, y1=10, x2=110, y2=200)
    rules.update([incoming], 2, 1)
    candidates = rules.update([], 2.8, 1)
    assert any(c.reason == "view_lost_after_suspicion" for c in candidates)


def test_foreshortened_floor_torso_does_not_establish_upright_recovery(settings):
    obs = pose(0)
    obs.keypoints[5], obs.keypoints[6] = (40, 80, 0.95), (60, 80, 0.95)
    obs.keypoints[11], obs.keypoints[12] = (43, 100, 0.95), (57, 100, 0.95)
    obs.keypoints[15], obs.keypoints[16] = (43, 160, 0.95), (57, 160, 0.95)
    obs.box = Box(x1=30, y1=60, x2=70, y2=165)
    assert MotionClassifier(settings).pose(obs).posture != "upright"


def test_straight_legs_on_floor_do_not_establish_upright_recovery(settings):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    fall = Candidate(
        category=Category.MEDICAL,
        reason="rapid_posture_change",
        first_ts=1.8,
        last_ts=2,
        track_id=1,
        epoch=1,
    )
    classifier.update([pose(2, low=True)], 2, 1, [fall])
    for n in range(11, 60):
        seated = pose(n / 5)
        seated.box = Box(x1=30, y1=0, x2=160, y2=80)
        seated.keypoints[13], seated.keypoints[14] = (90, 60, 0.95), (104, 60, 0.95)
        seated.keypoints[15], seated.keypoints[16] = (140, 60, 0.95), (154, 60, 0.95)
        assert classifier.pose(seated).posture == "seated"
        classifier.update([seated], n / 5, 1, [])
        assert not classifier.snapshot()[0].clear_eligible


def test_changing_joint_visibility_does_not_create_body_motion(settings):
    before = pose(0)
    before.keypoints[11] = (43, 50, 0.95)
    before.keypoints[12] = (57, 90, 0.95)
    after = before.model_copy(deep=True, update={"capture_ts": 0.2})
    after.keypoints[11] = (43, 50, 0.1)
    assert MotionClassifier(settings).displacement(before, after) == (0, 0)


def test_expired_reference_does_not_turn_stationary_seating_into_a_new_descent(settings):
    classifier = MotionClassifier(settings)
    first = pose(0, shift=100)
    first.box = Box(x1=30, y1=0, x2=70, y2=300)
    classifier.update([first], 0, 1, [])
    classifier.update([pose(0.2)], 0.2, 1, [])
    for n in range(5, 33):
        assert not classifier.update([pose(n / 5, shift=60)], n / 5, 1, [])


def test_return_after_occlusion_by_another_person_is_ambiguous():
    tracker = PoseTracker()
    initial = tracker.update([pose(0, track=None), pose(0, shift=50, track=None)], 0, 1)
    first_id = initial[0].track_id
    tracker.update([pose(0.2, shift=60, track=None)], 0.2, 1)
    current = tracker.update([pose(0.4, track=None), pose(0.4, shift=70, track=None)], 0.4, 1)
    assert next(o for o in current if o.track_id == first_id).association_uncertain


def test_walking_through_bottom_edge_is_not_body_compression(settings):
    rules = MedicalRules(settings)
    candidates = []
    for n in range(36):
        obs = pose(n / 5, shift=n * 5, track=None)
        obs.keypoints[0] = (50, n * 5 + 5, 0.95)
        obs.keypoints = [
            (x, min(y, 200), confidence if y < 200 else 0.1) for x, y, confidence in obs.keypoints
        ]
        obs.box.y2 = min(obs.box.y2, 200)
        obs.truncated = obs.box.y2 >= 197 and n not in (24, 25)
        obs.limited = obs.box.height < settings.minimum_person_height
        candidates.extend(rules.update([obs], n / 5, 1))
    candidates.extend(rules.update([], 8, 1))
    assert not candidates


def diagonal_exit_pose(n, movement="steady"):
    """Walk towards the lower-left image edge; keep only actually visible joints."""
    shift = n * 8
    extra = 35 if n == 12 and movement == "accelerated_descent" else 0
    obs = pose(n / 5, shift=shift + extra, track=None)
    points = [(x + 100 - n * 10, y, c) for x, y, c in obs.keypoints]
    if n == 12 and movement == "posture_change":
        points = [(x, y + (30 if k in (5, 6) else 0), c) for k, (x, y, c) in enumerate(points)]
    obs.keypoints = [(x, y, c if 0 < x < 200 and 0 < y < 200 else 0) for x, y, c in points]
    obs.box = Box(
        x1=max(0, 130 - n * 10),
        y1=shift + extra + (30 if n == 12 and movement == "posture_change" else 0),
        x2=170 - n * 10,
        y2=min(200, 150 + shift + extra),
    )
    obs.truncated = obs.box.x1 <= 3 or obs.box.y2 >= 197
    return obs


@pytest.mark.parametrize("movement", ["steady", "accelerated_descent", "posture_change"])
def test_diagonal_exit_with_one_visible_body_side_preserves_real_descent(settings, movement):
    rules = MedicalRules(settings)
    found = []
    for n in range(16 if movement == "steady" else 13):
        found.extend(rules.update([diagonal_exit_pose(n, movement)], n / 5, 1))
    found.extend(rules.update([], 4, 1))
    assert bool(found) is (movement != "steady")
    if found:
        assert any(c.reason in ("rapid_posture_change", "partial_body_descent") for c in found)
        assert all(not c.motion.clear_eligible for c in found)


@pytest.mark.parametrize(
    "missing", ["upright_context", "continuous_observations", "identity", "source_epoch"]
)
def test_partial_exit_needs_continuous_observed_activity(settings, missing):
    classifier = MotionClassifier(settings)
    previous = None
    for n in range(12):
        if missing == "continuous_observations" and n in (8, 9, 10):
            continue
        obs = diagonal_exit_pose(n)
        obs.track_id = 1
        if missing == "upright_context":
            obs.keypoints = [(x, y, 0 if k >= 13 else c) for k, (x, y, c) in enumerate(obs.keypoints)]
        if missing == "identity" and n == 10:
            obs.association_uncertain = True
        classifier.update([obs], obs.capture_ts, 1, [])
        previous = obs
    after = diagonal_exit_pose(12)
    after.track_id = 1
    if missing == "source_epoch":
        classifier.update([], after.capture_ts, 2, [])
    assert not classifier.cropped_translation(previous, after)


@pytest.mark.parametrize("movement", ["steady_exit", "accelerated_descent", "posture_change"])
@pytest.mark.parametrize("head_anchor", ["nose", "ear"])
def test_edge_translation_tolerates_recent_torso_measurement_noise(settings, movement, head_anchor):
    # Rounded observed geometry from MOT17-04, 10.6-11.4 s. The pelvis leaves
    # the image while the head and shoulders continue their walking trajectory.
    geometry = [
        (
            (175.29, 446.72, 223.86, 539.34),
            [(199.58, 459.9), (213.19, 470.33), (186.64, 471.04), (211.01, 509.07), (194.26, 509.9)],
        ),
        (
            (181.6, 455.35, 229.97, 540),
            [(210.13, 471.98), (218.64, 479.69), (193.22, 483.19), (217.42, 516.58), (200.94, 518.57)],
        ),
        (
            (186.67, 464.32, 236.11, 536.22),
            [(213.92, 477.81), (224.75, 487.64), (198.42, 489.81), (222.58, 521.17), (205.16, 522.42)],
        ),
        (
            (193.12, 468.89, 241.94, 535.84),
            [(224.74, 485.55), (230.67, 495.16), (204.42, 496.9), (228.45, 530.25), (210.6, 531.76)],
        ),
        ((202.43, 480.5, 251.34, 539.85), [(232, 496.17), (240.16, 506.75), (213.92, 508.27)]),
    ]
    rules = MedicalRules(settings)
    candidates = []
    for n, (coordinates, joints) in enumerate(geometry):
        points = [(0, 0, 0)] * 17
        for k, (x, y) in zip((0, 5, 6, 11, 12), joints, strict=False):
            if n == 4 and movement == "accelerated_descent":
                y += 10
            elif n == 4 and movement == "posture_change" and k == 5:
                y += 20
            points[k] = (x, y, 0.95)
        if head_anchor == "ear":
            x, y, confidence = points[0]
            points[3] = (x + 6, y - 1, confidence)
            if n == 3:
                points[0] = (x, y, 0.64)
        x1, y1, x2, y2 = coordinates
        obs = Observation(
            capture_ts=n / 5,
            box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
            score=0.9,
            label="person",
            keypoints=points,
            truncated=y2 >= 537,
        )
        candidates.extend(rules.update([obs], n / 5, 1))
    candidates.extend(rules.update([], 1.6, 1))
    assert bool(candidates) is (movement != "steady_exit")


def test_exit_geometry_cannot_cancel_an_existing_fall(settings):
    classifier = MotionClassifier(settings)
    fall = Candidate(
        category=Category.MEDICAL, reason="rapid_posture_change", first_ts=0, last_ts=0, track_id=1, epoch=1
    )
    classifier.update([pose(0, low=True)], 0, 1, [fall])
    for n in range(1, 20):
        obs = partial_pose(n / 5, 100, 130)
        obs.track_id = 1
        classifier.update([obs], n / 5, 1, [])
        assert not classifier.snapshot()[0].clear_eligible


def test_head_landmark_fallback_cannot_veto_a_descent_from_rest(settings):
    classifier = MotionClassifier(settings)
    for n in range(4):
        before = pose(n / 5)
        before.truncated = True
        before.keypoints[3] = (50, 5, 0.95)
        classifier.update([before], n / 5, 1, [])
    after = before.model_copy(deep=True, update={"capture_ts": 1.0})
    after.keypoints = [
        (x, y + 25, confidence if k in (3, 5, 6) else 0.1)
        for k, (x, y, confidence) in enumerate(after.keypoints)
    ]
    after.box = Box(x1=30, y1=25, x2=70, y2=150)
    assert not classifier.cropped_translation(before, after)


def test_walking_with_legs_occluded_does_not_become_a_slow_collapse(settings):
    rules = MedicalRules(settings)
    candidates = []
    for n in range(30):
        obs = pose(n / 5, shift=n * 7, track=None)
        obs.keypoints = [(x + n * 10, y, confidence) for x, y, confidence in obs.keypoints]
        obs.box.x1 += n * 10
        obs.box.x2 += n * 10
        if n >= 10:
            # The visible torso keeps its shape while the legs pass behind an
            # occluder. The smaller detector box is not a shorter human body.
            obs.keypoints = [(x, y, 0.1 if k >= 13 else c) for k, (x, y, c) in enumerate(obs.keypoints)]
            obs.box.y2 = n * 7 + 80
        candidates.extend(rules.update([obs], n / 5, 1))
    assert not candidates
    assert not any(m.onset_observed for m in rules.motion.snapshot())


@pytest.mark.parametrize("partial", [False, True])
def test_slow_descent_remains_detectable_with_visible_body_change(settings, partial):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5, seated=partial)], n / 5, 1, [])
    candidates = []
    for n in range(10, 23):
        shift = (n - 10) * (5 if partial else 4)
        obs = pose(n / 5, shift=shift)
        obs.box = Box(x1=30, y1=shift, x2=120, y2=150)
        if partial:
            # A slumping torso still changes visibly even when the knees and
            # ankles cannot be measured. Leg visibility is not mandatory.
            obs.keypoints[11] = (43, 60 + shift * 0.6, 0.95)
            obs.keypoints[12] = (57, 60 + shift * 0.6, 0.95)
            obs.keypoints = [(x, y, 0.1 if k >= 13 else c) for k, (x, y, c) in enumerate(obs.keypoints)]
        else:
            obs.keypoints[13], obs.keypoints[14] = (90, 110, 0.95), (104, 110, 0.95)
            obs.keypoints[15], obs.keypoints[16] = (43, 140, 0.95), (57, 140, 0.95)
        candidates.extend(classifier.update([obs], n / 5, 1, []))
    assert any(c.reason == "slow_posture_descent" for c in candidates)
    assert not classifier.snapshot()[0].clear_eligible


@pytest.mark.parametrize("report_missing", [False, True])
def test_reused_track_after_a_gap_cannot_clear_an_unresolved_fall(settings, report_missing):
    classifier = MotionClassifier(settings)
    for n in range(10):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    fall = Candidate(
        category=Category.MEDICAL, reason="rapid_posture_change", first_ts=1.8, last_ts=2, track_id=1, epoch=1
    )
    classifier.update([pose(2, low=True)], 2, 1, [fall])
    if report_missing:
        classifier.update([], 2.8, 1, [])
    # The box matcher can reuse an ID after a one-second gap. Neither an empty
    # detector result nor a skipped analysis frame proves who returned.
    classifier.update([pose(3, low=True)], 3, 1, [])
    for n in range(16, 40):
        classifier.update([pose(n / 5)], n / 5, 1, [])
        state = classifier.snapshot()[0]
        assert state.identity_uncertain and not state.clear_eligible


def test_gap_during_ordinary_walking_does_not_create_a_concern(settings):
    classifier = MotionClassifier(settings)
    classifier.update([pose(0)], 0, 1, [])
    for n in range(5, 30):
        assert not classifier.update([pose(n / 5)], n / 5, 1, [])
    state = classifier.snapshot()[0]
    assert state.phase == "observing" and not state.identity_uncertain


def test_one_missing_sample_does_not_block_continuous_recovery(settings):
    classifier = MotionClassifier(settings)
    fall = Candidate(
        category=Category.MEDICAL, reason="rapid_posture_change", first_ts=0, last_ts=0, track_id=1, epoch=1
    )
    classifier.update([pose(0, low=True)], 0, 1, [fall])
    for n in range(2, 30):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    state = classifier.snapshot()[0]
    assert state.phase == "recovered_activity" and not state.identity_uncertain


def test_weak_pose_before_a_repeat_fall_does_not_reuse_the_previous_episode(settings):
    classifier = MotionClassifier(settings)
    fall = Candidate(
        category=Category.MEDICAL, reason="rapid_posture_change", first_ts=0, last_ts=0, track_id=1, epoch=1
    )
    classifier.update([pose(0, low=True)], 0, 1, [fall])
    for n in range(1, 26):
        classifier.update([pose(n / 5)], n / 5, 1, [])
    previous = classifier.snapshot()[0]
    assert previous.phase == "recovered_activity"
    weak = pose(5.2).model_copy(update={"keypoints": [(0, 0, 0)] * 17})
    classifier.update([weak], 5.2, 1, [])
    assert not classifier.snapshot()[0].clear_eligible
    repeat = fall.model_copy(update={"first_ts": 5.2, "last_ts": 5.4})
    result = classifier.update([pose(5.4, low=True)], 5.4, 1, [repeat])[0]
    assert result.motion.episode_id != previous.episode_id
    assert result.motion.previous_episode_id == previous.episode_id
    assert result.motion.started_ts == 5.2 and not result.motion.clear_eligible
    # Repeated measurements of this same new descent keep one episode.
    again = repeat.model_copy(update={"last_ts": 5.6})
    state = classifier.update([pose(5.6, low=True)], 5.6, 1, [again])[0].motion
    assert state.episode_id == result.motion.episode_id
