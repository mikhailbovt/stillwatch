from test_episode_memory import answer, event, evidence, pose

from stillwatch.motion import MotionClassifier
from stillwatch.tracking import PoseTracker
from stillwatch.triggers import MedicalRules
from stillwatch.types import Box, Candidate, Category, Observation, Status


def test_cropped_upper_body_does_not_create_a_low_posture_event(settings):
    rules = MedicalRules(settings)
    for n in range(150):
        points = [(0, 0, 0)] * 17
        points[5], points[6] = (180, 300, 0.9), (400, 300, 0.9)
        points[11], points[12] = (220, 470, 0.2), (380, 470, 0.2)
        obs = Observation(
            capture_ts=n / 5,
            label="person",
            score=0.9,
            box=Box(x1=20, y1=160 + n % 10, x2=620, y2=480),
            keypoints=points,
            truncated=True,
        )
        assert not rules.update([obs], n / 5, 1)


def test_detector_id_changes_do_not_reset_stable_geometry():
    tracker = PoseTracker()
    ids = []
    for n in range(15):
        result = tracker.update([pose(n / 5, track=100 + n)], n / 5, 1)
        ids.append(result[0].track_id)
        assert not result[0].association_uncertain
    assert set(ids) == {1}


def test_short_missing_detection_preserves_id_without_inventing_observations():
    tracker = PoseTracker()
    first = tracker.update([pose(0)], 0, 1)[0]
    assert tracker.update([], 0.8, 1) == []
    resumed = tracker.update([pose(1)], 1, 1)[0]
    assert resumed.track_id == first.track_id and not resumed.association_uncertain
    assert tracker.update([pose(5)], 5, 1)[0].track_id != first.track_id


def test_weak_duplicate_pose_does_not_create_another_person():
    tracker = PoseTracker()
    main = pose(0)
    duplicate = main.model_copy(update={"score": 0.2, "track_id": 12})
    assert len(tracker.update([main, duplicate], 0, 1)) == 1


def test_distinct_people_keep_separate_ids():
    tracker = PoseTracker()
    for n in range(10):
        observations = tracker.update([pose(n / 5), pose(n / 5, shift=200, track=2)], n / 5, 1)
        assert len({o.track_id for o in observations}) == 2
        assert not any(o.association_uncertain for o in observations)


def test_recovery_and_current_cloud_posture_can_agree_without_wording_recovered(system):
    manager = system[4]
    item = event(manager, evidence(phase="recovered_activity", posture="upright", clear_eligible=True))
    result = answer(item, "no_concern")
    result.latest_posture = "upright"
    manager.apply(result, 20)
    assert item.analysis_status == Status.NO_ALERT


def test_recovery_clearance_still_rejects_a_low_cloud_posture(system):
    manager = system[4]
    item = event(manager, evidence(phase="recovered_activity", posture="upright", clear_eligible=True))
    manager.apply(answer(item, "no_concern"), 20)
    assert item.analysis_status == Status.UNKNOWN


def test_inverted_floor_pose_is_not_upright(settings):
    obs = pose(0)
    obs.keypoints = [(x, 150 - y, confidence) for x, y, confidence in obs.keypoints]
    classified = MotionClassifier(settings).pose(obs)
    assert classified.low and classified.posture != "upright"


def test_weak_transient_detection_does_not_poison_an_established_track(settings):
    rules = MedicalRules(settings)
    for n in range(15):
        observations = [pose(n / 5, track=None)]
        if n == 7:
            observations.append(pose(n / 5, shift=25, track=None).model_copy(update={"score": 0.3}))
        rules.update(observations, n / 5, 1)
        assert len(rules.observations) == 1
        assert not rules.motion.tracks[1].evidence.identity_uncertain


def test_persistent_weak_second_person_is_not_hidden(settings):
    rules = MedicalRules(settings)
    for n in range(5):
        rules.update(
            [
                pose(n / 5, track=None),
                pose(n / 5, shift=25, track=None).model_copy(update={"score": 0.3}),
            ],
            n / 5,
            1,
        )
    assert len(rules.observations) == 2
    assert rules.motion.tracks[1].evidence.identity_uncertain


def test_track_reuse_after_visibility_loss_preserves_identity_uncertainty(settings):
    rules = MedicalRules(settings)
    for n in range(8):
        rules.update([pose(n / 5)], n / 5, 1)
    rules.update([pose(1.6, low=True)], 1.6, 1)
    episode = rules.motion.tracks[1].evidence.episode_id
    rules.update([], 2.4, 1)
    assert rules.motion.tracks[1].evidence.phase == "visibility_lost"
    for n in range(15, 30):
        rules.update([pose(n / 5)], n / 5, 1)
        assert not rules.motion.tracks[1].evidence.clear_eligible
    rules.update([pose(6)], 6, 1)
    state = rules.motion.tracks[1].evidence
    # A geometric ID reused after 1.4 seconds cannot establish who returned.
    # Fresh upright poses alone must not clear the disappeared person's event.
    assert state.episode_id == episode and state.phase == "following_concern"
    assert state.identity_uncertain and not state.clear_eligible


def test_one_weak_pose_cannot_clear_but_does_not_erase_recovery_history(settings):
    rules = MedicalRules(settings)
    for n in range(8):
        rules.update([pose(n / 5)], n / 5, 1)
    rules.update([pose(1.6, low=True)], 1.6, 1)
    # Allow three settled seconds after the visible ascent, including the weak sample.
    for n in range(9, 30):
        obs = pose(n / 5)
        if n == 20:
            obs.keypoints = [(0, 0, 0)] * 17
        rules.update([obs], n / 5, 1)
        if n == 20:
            assert not rules.motion.tracks[1].evidence.clear_eligible
    assert rules.motion.tracks[1].evidence.phase == "recovered_activity"


def test_sitting_briefly_after_a_fall_cannot_relabel_it_as_ordinary_rest(settings):
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
    episode = classifier.tracks[1].evidence.episode_id
    # A partial sit-up followed by a stable recline resembles ordinary preparation,
    # but cannot erase the observed landing without sustained recovery.
    for n in range(9, 19):
        classifier.update([pose(n / 5, seated=True, shift=80)], n / 5, 1, [])
    for n in range(19, 40):
        classifier.update([pose(n / 5, low=True)], n / 5, 1, [])
    state = classifier.tracks[1].evidence
    assert state.episode_id == episode and state.transition == "fall_like"
    assert not state.clear_eligible
