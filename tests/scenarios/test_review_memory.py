import pytest
from test_episode_memory import answer, event, evidence

from stillwatch.types import Candidate, Category, Lifecycle, Status


def reviewed(manager, category=Category.MEDICAL):
    item = event(manager, evidence())
    item.category = category
    result = answer(item, "persistent_concern")
    result.medical_status = Status.REVIEW if category == Category.MEDICAL else Status.NO_ALERT
    result.fire_status = Status.REVIEW if category == Category.FIRE else Status.NO_ALERT
    manager.apply(result, 20)
    assert item.analysis_status == Status.REVIEW
    return item


@pytest.mark.parametrize("category", [Category.MEDICAL, Category.FIRE])
@pytest.mark.parametrize("status", [Status.UNKNOWN, Status.UNAVAILABLE, Status.PENDING])
def test_uncertain_recheck_retains_operator_review_without_changing_the_latest_answer(
    system, category, status
):
    manager = system[4]
    item = reviewed(manager, category)
    result = answer(item, "persistent_concern")
    if category == Category.MEDICAL:
        result.medical_status = status
    else:
        result.fire_status = status
    result.raw_answer = {"category": category, "choice": status}
    raw = result.raw_answer.copy()
    manager.apply(result, 20)
    assert item.analysis_status == Status.REVIEW
    assert item.medical_status == result.medical_status
    assert item.fire_status == result.fire_status
    assert item.policy_reason == "review_pending_resolution"
    assert result.raw_answer == raw
    assert getattr(result, f"{category.value}_status") == status
    manager.apply(result.model_copy(), 20)
    assert item.analysis_status == Status.REVIEW


@pytest.mark.parametrize(
    ("basis", "posture"),
    [("upright_activity", "upright"), ("prior_supported_posture", "seated_on_furniture")],
)
def test_joint_recovery_can_resolve_a_retained_review(system, basis, posture):
    settings, _, _, _, manager, _, _ = system
    settings.medical_observation_seconds = 15
    item = reviewed(manager)
    uncertain = answer(item, "uncertain")
    uncertain.medical_status = Status.UNKNOWN
    manager.apply(uncertain, 20)
    item.motion = evidence(
        phase="recovered_activity",
        posture="upright" if posture == "upright" else "seated",
        clear_eligible=True,
        recovery_seconds=3,
        recovery_basis=basis,
    )
    item.full_assessment_source_ts = 17
    result = answer(item, "recovered")
    result.latest_posture = posture
    manager.apply(result, 20)
    assert item.analysis_status == Status.NO_ALERT
    assert item.lifecycle == Lifecycle.CLOSED
    assert item.policy_reason is None


def test_unsupported_clearance_does_not_remove_review(system):
    manager = system[4]
    item = reviewed(manager)
    result = answer(item, "no_concern")
    manager.apply(result, 20)
    assert result.medical_status == Status.UNKNOWN
    assert result.policy_reason == "local_episode_unresolved"
    assert item.medical_status == Status.UNKNOWN and item.analysis_status == Status.REVIEW


def test_supported_ordinary_activity_can_resolve_review_and_retain_its_context(system):
    manager = system[4]
    item = reviewed(manager)
    item.motion = evidence(phase="ordinary_activity", transition="controlled_pattern", clear_eligible=True)
    manager.apply(answer(item, "no_concern"), 20)
    assert item.analysis_status == Status.NO_ALERT
    assert item.ordinary_activity_confirmed and item.lifecycle == Lifecycle.OPEN
    assert item.policy_reason is None


def test_unknown_without_prior_review_does_not_gain_attention(system):
    manager = system[4]
    item = event(manager, evidence())
    result = answer(item, "uncertain")
    result.medical_status = Status.UNKNOWN
    manager.apply(result, 20)
    assert item.analysis_status == Status.UNKNOWN


def test_fresh_independent_episode_does_not_inherit_review(system):
    manager = system[4]
    old = reviewed(manager)
    new, created = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=29.5,
            last_ts=30,
            track_id=1,
            epoch=1,
            box=old.box,
        )
    )
    assert created and new.incident_id != old.incident_id
    result = answer(new, "uncertain")
    result.medical_status = Status.UNKNOWN
    manager.apply(result, 30)
    assert new.analysis_status == Status.UNKNOWN
    assert old.analysis_status == Status.REVIEW


def test_late_answer_cannot_reopen_an_operator_closed_review(system):
    manager = system[4]
    item = reviewed(manager)
    manager.annotate(item.incident_id, "close")
    result = answer(item, "uncertain")
    result.medical_status = Status.UNKNOWN
    manager.apply(result, 20)
    assert result.stale and item.lifecycle == Lifecycle.CLOSED


def test_fresh_fire_no_alert_can_resolve_review(system):
    manager = system[4]
    item = reviewed(manager, Category.FIRE)
    manager.apply(answer(item), 20)
    assert item.fire_status == item.analysis_status == Status.NO_ALERT
