import os
import time
from pathlib import Path

from stillwatch.types import Box, Candidate, Category


def event(manager):
    return manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="persistent_low_posture",
            first_ts=2,
            last_ts=6,
            box=Box(x1=10, y1=20, x2=80, y2=60),
            epoch=1,
        )
    )[0]


def test_disk_full_preserves_metadata(system, monkeypatch):
    _, _, storage, evidence, manager, _, _ = system
    item = event(manager)
    packet = evidence.build(item, 6)

    def full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "write_bytes", full)
    assert not storage.save_packet(packet)
    assert storage.incidents()[0].incident_id == item.incident_id
    assert storage.detail(item.incident_id)["packets"][0]["files"] == []


def test_retention_and_quota_keep_events_without_hiding_missing_frames(system):
    settings, _, storage, evidence, manager, _, _ = system
    item = event(manager)
    packet = evidence.build(item, 6)
    assert storage.save_packet(packet)
    file = storage.evidence_path(item.incident_id, packet.packet_id, 0)
    assert file
    old = time.time() - 25 * 3600
    for path in storage.evidence.rglob("*.jpg"):
        os.utime(path, (old, old))
    storage.cleanup()
    assert storage.evidence_path(item.incident_id, packet.packet_id, 0) is None
    assert storage.incidents()
    assert not storage.detail(item.incident_id)["packets"][0]["files"][0]["available"]
    settings.storage_quota_bytes = 1
    new_packet = evidence.build(item, 6)
    assert not storage.save_packet(new_packet)


def test_controlled_endpoint_cannot_escape_storage(system):
    _, _, storage, evidence, manager, _, _ = system
    item = event(manager)
    packet = evidence.build(item, 6)
    storage.save_packet(packet)
    assert storage.evidence_path("..", packet.packet_id, 0) is None
    assert storage.evidence_path(item.incident_id, packet.packet_id, -1) is None
    assert storage.evidence_path(item.incident_id, packet.packet_id, 999) is None


def test_metadata_expiry_removes_related_rows_and_files(system):
    settings, _, storage, evidence, manager, _, _ = system
    item = event(manager)
    packet = evidence.build(item, 6)
    storage.save_packet(packet)
    storage.feedback(item.incident_id, "operator", "confirmed")
    item.updated_at = time.time() - 8 * 86400
    storage.save_incident(item)
    storage.cleanup()
    assert not storage.incidents()
    assert not storage.detail(item.incident_id)["packets"]
    assert not storage.detail(item.incident_id)["feedback_history"]
