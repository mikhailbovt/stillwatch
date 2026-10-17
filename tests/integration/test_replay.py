import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stillwatch.capture import make_sample
from stillwatch.runtime import Analysis
from stillwatch.telemetry import Telemetry
from stillwatch.types import Box, Observation


async def test_annotated_replay_preserves_24hz_source_duration(settings, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tools.replay import replay

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120))
    for index in range(48):
        writer.write(np.full((120, 160, 3), index * 4, np.uint8))
    writer.release()
    settings.pose_enabled = settings.fire_enabled = False
    output = tmp_path / "replay"
    result = await replay(source, settings, output, render=True)
    trace = output / "observations.jsonl"
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows and all((row["width"], row["height"]) == (160, 120) for row in rows)
    video = cv2.VideoCapture(str(output / "annotated.mp4"))
    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
    video.release()
    assert abs(duration - 2) <= 0.15


@pytest.mark.parametrize(
    ("source_fps", "pose_fps", "fire_fps", "buffer_fps"),
    [(14, 5, 2, 10), (25, 5, 2, 10), (30, 5, 2, 10), (25, 4, 7, 3)],
)
async def test_replay_routes_match_processing_every_source_frame(
    settings, tmp_path, monkeypatch, source_fps, pose_fps, fire_fps, buffer_fps
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tools.replay import replay

    class Recorder:
        states = {"pose": "ready", "fire": "ready"}
        errors = {}

        def __init__(self):
            self.calls = {"pose": [], "fire": []}

        def reset(self):
            pass

        def people(self, image, capture_ts):
            self.calls["pose"].append(capture_ts)
            return []

        def hazards(self, image, capture_ts):
            self.calls["fire"].append(capture_ts)
            return []

    settings.pose_fps, settings.fire_fps, settings.buffer_fps = pose_fps, fire_fps, buffer_fps
    source = tmp_path / "cadence.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (160, 120))
    assert writer.isOpened()
    for index in range(source_fps * 4):
        writer.write(np.full((120, 160, 3), index % 255, np.uint8))
    writer.release()

    reference = Recorder()
    direct = Analysis(settings, Telemetry(), reference)
    capture = cv2.VideoCapture(str(source))
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        ts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        index += 1
        direct.process(make_sample(image, index, ts, 1, settings))
    capture.release()

    recorded = Recorder()
    output = tmp_path / "replay"
    result = await replay(source, settings, output, analysis=Analysis(settings, Telemetry(), recorded))
    assert recorded.calls == reference.calls
    assert result["frames_decoded"] == index
    assert result["inference_calls"] == {route: len(calls) for route, calls in reference.calls.items()}
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    for route, calls in reference.calls.items():
        assert [row["source_ts"] for row in rows if route in row["inferred_routes"]] == calls


async def test_replay_retains_internal_clothing_input_for_reproducible_association(
    settings, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tools.replay import replay

    histogram = (1.0,) + (0.0,) * 15

    class Perception:
        states = {"pose": "ready", "fire": "disabled"}
        errors = {}

        def reset(self):
            pass

        def people(self, image, capture_ts):
            return [
                Observation(
                    capture_ts=capture_ts,
                    box=Box(x1=20, y1=20, x2=60, y2=110),
                    label="person",
                    score=0.9,
                    torso_histogram=histogram,
                )
            ]

    source = tmp_path / "appearance.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    output = tmp_path / "replay"
    await replay(source, settings, output, analysis=Analysis(settings, Telemetry(), Perception()))
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        assert Observation.model_validate(row["detections"]["pose"][0]).torso_histogram == histogram
        assert Observation.model_validate(row["observations"]["pose"][0]).torso_histogram == histogram
