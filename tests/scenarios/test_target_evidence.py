import json

import cv2
import numpy as np
import pytest

from stillwatch.buffer import FrameBuffer
from stillwatch.capture import make_sample
from stillwatch.decisions import request_payload
from stillwatch.evidence import EvidenceBuilder
from stillwatch.types import Box, Category, Incident, MotionEvidence, Observation


def observation(ts, x, *, uncertain=False):
    return Observation(
        capture_ts=ts,
        box=Box(x1=x, y1=20, x2=x + 20, y2=60),
        label="person",
        score=0.9,
        track_id=1,
        association_uncertain=uncertain,
    )


def item():
    return Incident(
        camera_id="test",
        category=Category.MEDICAL,
        reason="rapid_posture_change",
        first_ts=0,
        last_ts=1,
        detected_ts=1,
        epoch=1,
        track_ids=[1],
        box=Box(x1=10, y1=20, x2=30, y2=60),
    )


def test_target_crops_follow_the_observed_person_and_preserve_onset(settings):
    buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
    builder = EvidenceBuilder(settings, buffer)
    incident = item()
    for n in range(206):
        ts = n / 5
        x = min(70, 10 + n)
        image = np.zeros((120, 160, 3), np.uint8)
        image[20:60, x : x + 20] = (0, 0, 255)
        image[20:60, 125:145] = (255, 0, 0)
        buffer.add(make_sample(image, n, ts, 1, settings).frame)
        builder.observe([observation(ts, x)], ts, 1)
        if ts >= 1:
            builder.pin(incident)
    packet = builder.build(incident, 41)
    crops = [i for i in packet.images if i.kind == "crop"]
    assert 1 <= len(crops) <= 4 and len(packet.images) <= 16
    assert any(i.capture_ts < 2 for i in crops)  # Onset survived ring eviction.
    assert any(i.capture_ts > 38 for i in crops)
    for crop in crops:
        decoded = cv2.imdecode(np.frombuffer(crop.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded[:, :, 2].mean() > decoded[:, :, 0].mean() + 40
    for region in packet.temporal_context["target_regions"]:
        assert region["observed_source_ts"] <= region["frame_source_ts"] <= packet.source_ts
        assert region["track_id"] == 1
    payload = request_payload(packet)
    assert any("Target person: track 1" in c.get("text", "") for c in payload["input"][0]["content"])


def test_future_missing_or_ambiguous_regions_do_not_create_target_crops(settings):
    buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
    builder = EvidenceBuilder(settings, buffer)
    incident = item()
    image = np.zeros((120, 160, 3), np.uint8)
    for n in range(31):
        buffer.add(make_sample(image, n, n / 5, 1, settings).frame)
    builder.observe([observation(2, 10)], 1, 1)  # Future poses are rejected.
    assert not builder.regions
    builder.observe([observation(3, 10, uncertain=True)], 3, 1)
    packet = builder.build(incident, 6)
    assert not any(i.kind == "crop" for i in packet.images)
    assert all(r["association_uncertain"] for r in packet.temporal_context["target_regions"])
    assert not any(r["frame_source_ts"] > 3.3 + 1e-6 for r in packet.temporal_context["target_regions"])
    builder.observe([observation(0, 100)], 0, 2)
    assert not builder.build(incident, 6).temporal_context["target_regions"]


def test_position_memory_is_bounded_and_release_removes_pinned_regions(settings):
    buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
    builder = EvidenceBuilder(settings, buffer)
    for n in range(1500):
        builder.observe([observation(n / 100, 10)], n / 100, 1)
    assert len(builder.regions[1]) <= 600
    for tid in range(200):
        obs = observation(16, 10).model_copy(update={"track_id": tid})
        builder.observe([obs], 16, 1)
    assert len(builder.regions) <= 128
    builder.onset_regions["closed"] = {1: {"track_id": 1}}
    builder.release("closed")
    assert "closed" not in builder.onset_regions


@pytest.mark.parametrize(
    "group_uncertain,motion_uncertain,expected",
    [
        (False, True, True),
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (True, None, True),
        (False, None, False),
    ],
)
def test_request_identity_context_includes_local_motion_uncertainty(
    settings, group_uncertain, motion_uncertain, expected
):
    buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
    builder = EvidenceBuilder(settings, buffer)
    buffer.add(make_sample(np.zeros((120, 160, 3), np.uint8), 1, 2, 1, settings).frame)
    incident = item()
    incident.identity_uncertain = group_uncertain
    if motion_uncertain is not None:
        incident.motion = MotionEvidence(
            episode_id="test-episode",
            track_id=1,
            epoch=1,
            started_ts=0,
            observed_ts=6,
            last_seen_ts=2,
            identity_uncertain=motion_uncertain,
        )
    before = incident.model_dump(mode="json")
    packet = builder.build(incident, 6)
    payload = request_payload(packet)
    text = payload["input"][0]["content"][0]["text"]
    context = json.loads(text.split("\nObserved timeline: ", 1)[1].split(". Elapsed source time", 1)[0])
    assert packet.temporal_context["identity_uncertain"] is expected
    assert context["identity_uncertain"] is expected
    assert context["elapsed_source_seconds"] == 5
    assert all(image.capture_ts <= packet.source_ts for image in packet.images)
    # Packet aggregation must not latch or alter the incident's distinct
    # group-identity state, motion history or observation deadline.
    assert incident.model_dump(mode="json") == before
