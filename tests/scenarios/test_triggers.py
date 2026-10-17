from stillwatch.triggers import FireRules, MedicalRules
from stillwatch.types import Box, Observation


def person(ts, box, track=1, limited=False):
    return Observation(
        capture_ts=ts,
        box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        score=0.9,
        label="person",
        track_id=track,
        limited=limited,
    )


def test_fast_fall_does_not_wait_for_stillness(settings):
    rules = MedicalRules(settings)
    assert not rules.update([person(0, (30, 0, 70, 100))], 0, 1)
    candidates = rules.update([person(0.6, (30, 95, 120, 135))], 0.6, 1)
    assert candidates[0].reason == "rapid_posture_change"
    assert candidates[0].last_ts == 0.6


def test_already_lying_is_independent_of_fall_history(settings):
    rules = MedicalRules(settings)
    candidates = []
    for i in range(12):
        candidates.extend(rules.update([person(i / 5, (20, 70, 150, 110))], i / 5, 1))
    assert candidates[0].reason == "persistent_low_posture"
    assert candidates[0].first_ts == 0


def test_standing_and_ordinary_exit_do_not_trigger(settings):
    rules = MedicalRules(settings)
    for i in range(40):
        assert not rules.update([person(i / 5, (20, 0, 60, 100))], i / 5, 1)
    assert not rules.update([], 10, 1)


def test_loss_after_suspicion_remains_reviewable(settings):
    rules = MedicalRules(settings)
    rules.update([person(0, (30, 0, 70, 100))], 0, 1)
    rules.update([person(0.5, (30, 95, 120, 135))], 0.5, 1)
    candidates = rules.update([], 2, 1)
    assert candidates[0].reason == "view_lost_after_suspicion"
    assert candidates[0].limited


def test_tiny_people_do_not_produce_confident_geometry(settings):
    rules = MedicalRules(settings)
    for i in range(20):
        assert not rules.update([person(i / 5, (20, 70, 80, 90), limited=True)], i / 5, 1)


def test_fire_without_people_and_smoke_independently(settings):
    for label in ("fire", "smoke"):
        rules = FireRules(settings)
        obs = Observation(capture_ts=0, box=Box(x1=1, y1=1, x2=40, y2=40), score=0.3, label=label)
        assert not rules.update([obs], 0, 1)
        assert (
            rules.update([obs.model_copy(update={"capture_ts": 0.5})], 0.5, 1)[0].reason == f"visible_{label}"
        )


def test_high_confidence_fire_and_separate_regions(settings):
    rules = FireRules(settings)
    observations = [
        Observation(capture_ts=0, box=Box(x1=x, y1=0, x2=x + 40, y2=40), score=0.8, label="fire")
        for x in (0, 100)
    ]
    assert len(rules.update(observations, 0, 1)) == 2


def test_nonconsecutive_fire_observations(settings):
    rules = FireRules(settings)
    obs = Observation(capture_ts=0, box=Box(x1=0, y1=0, x2=40, y2=40), score=0.3, label="smoke")
    rules.update([obs], 0, 1)
    rules.update([], 0.5, 1)
    assert rules.update([obs], 1, 1)
    rules.reset()
    assert not rules.update([obs], 2, 2)
