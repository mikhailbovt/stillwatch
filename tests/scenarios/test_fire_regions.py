from stillwatch.types import Box, Candidate, Category


def hazard(box, ts=6):
    return Candidate(
        category=Category.FIRE,
        reason="visible_smoke",
        first_ts=ts,
        last_ts=ts,
        box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        epoch=1,
    )


def test_changing_fire_region_scale_does_not_create_new_incident(system):
    _, _, _, _, manager, _, _ = system
    first, _ = manager.ingest(hazard((0, 0, 200, 200)))
    second, created = manager.ingest(hazard((20, 20, 40, 40), 7))
    assert not created and second.incident_id == first.incident_id
    separate, created = manager.ingest(hazard((300, 300, 400, 400), 7))
    assert created and separate.incident_id != first.incident_id
