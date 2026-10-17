from stillwatch.types import Box, Candidate, Category, DecisionResult, Status


def outcome(item, packet, medical=Status.REVIEW, fire=Status.NO_ALERT):
    return DecisionResult(
        incident_id=item.incident_id,
        packet_id=packet.packet_id,
        version=packet.version,
        medical_status=medical,
        fire_status=fire,
        provider="mock",
        model="scenario",
    )


def test_overview_discovery_links_the_exact_original_evidence(system):
    _, _, storage, evidence, manager, _, _ = system
    overview = manager.overview(6, 1)
    overview.packet_version = 1
    packet = evidence.build(overview, 6)
    storage.save_packet(packet)
    result = outcome(overview, packet, medical=Status.REVIEW, fire=Status.URGENT)
    manager.apply(result, 6)
    storage.save_result(packet, result)
    children = [i for i in manager.items.values() if i.origin_incident_id == overview.incident_id]
    assert len(children) == 2
    for item in children:
        detail = manager.detail(item.incident_id)
        assert detail["packets"][0]["packet_id"] == packet.packet_id
        assert detail["assessments"][0]["request_id"] == result.request_id
        assert detail["evidence_origin"] == overview.incident_id


def test_other_direction_is_not_hidden_in_an_event_check(system):
    _, _, storage, evidence, manager, _, _ = system
    item, _ = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="persistent_low_posture",
            first_ts=2,
            last_ts=6,
            box=Box(x1=0, y1=0, x2=100, y2=50),
            epoch=1,
        )
    )
    packet = evidence.build(item, 6)
    result = outcome(item, packet, medical=Status.NO_ALERT, fire=Status.URGENT)
    manager.apply(result, 6)
    manager.apply(result, 6.1)
    children = [i for i in manager.items.values() if i.category == Category.FIRE]
    assert len(children) == 1 and children[0].analysis_status == Status.URGENT


def test_urgent_assessment_requires_operator_closure(system):
    _, _, _, evidence, manager, _, _ = system
    item, _ = manager.ingest(
        Candidate(category=Category.MEDICAL, reason="rapid_posture_change", first_ts=2, last_ts=6, epoch=1)
    )
    packet = evidence.build(item, 6)
    manager.apply(outcome(item, packet, medical=Status.URGENT), 6)
    manager.apply(outcome(item, packet, medical=Status.NO_ALERT), 7)
    manager.archive_clear(30, 1)
    assert item.analysis_status == Status.URGENT and item.lifecycle == "OPEN"


def test_overview_concern_waits_for_recovery_but_immediate_danger_does_not(system):
    settings, _, storage, evidence, manager, _, scheduler = system
    settings.medical_observation_seconds = 15
    settings.medical_confirmation_frames = 12
    overview = manager.overview(6, 1)
    packet = evidence.build(overview, 6)
    result = outcome(overview, packet, medical=Status.REVIEW)
    result.medical_observation = "persistent_concern"
    manager.apply(result, 6)
    child = next(i for i in manager.items.values() if i.origin_incident_id == overview.incident_id)
    assert child.analysis_status == Status.PENDING and child.request_count == 0
    assert overview.lifecycle == "CLOSED" and overview.analysis_status == Status.PENDING
    assert overview.medical_status == Status.REVIEW  # Original preliminary answer is retained.
    scheduler.enqueue(child)
    assert child.queue_reason == "awaiting_event_development"
    # A later overview must not reset the first-suspicion timestamp.
    manager.apply(result, 9)
    assert child.detected_ts == 6

    other = manager.overview(10, 1)
    immediate = outcome(other, evidence.build(other, 10), medical=Status.URGENT)
    immediate.medical_observation = "immediate_danger"
    manager.apply(immediate, 10)
    urgent = next(i for i in manager.items.values() if i.origin_incident_id == other.incident_id)
    assert urgent.analysis_status == Status.URGENT


def test_observing_medical_concern_never_hides_an_overview_fire_alert(system):
    settings, _, _, evidence, manager, _, _ = system
    settings.medical_observation_seconds = 15
    overview = manager.overview(6, 1)
    result = outcome(overview, evidence.build(overview, 6), medical=Status.REVIEW, fire=Status.URGENT)
    result.medical_observation = "persistent_concern"
    manager.apply(result, 6)
    assert overview.lifecycle == "OPEN" and overview.analysis_status == Status.URGENT
    medical = next(i for i in manager.items.values() if i.category == Category.MEDICAL)
    fire = next(i for i in manager.items.values() if i.category == Category.FIRE)
    assert medical.analysis_status == Status.PENDING and fire.analysis_status == Status.URGENT
