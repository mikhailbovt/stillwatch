"""Replay a local video on its source clock without dropping analytics frames."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import cv2
import psutil

from stillwatch.buffer import FrameBuffer
from stillwatch.capture import FileClock, make_sample, open_capture
from stillwatch.config import Settings, implementation_fingerprint, load_settings
from stillwatch.decisions import MockDecisionProvider, OpenAIDecisionProvider
from stillwatch.evidence import EvidenceBuilder
from stillwatch.incidents import IncidentManager
from stillwatch.runtime import Analysis
from stillwatch.scheduler import Scheduler
from stillwatch.storage import Storage
from stillwatch.telemetry import Telemetry
from stillwatch.types import Category, Observation


def trace_observation(observation: Observation) -> dict:
    """Keep local association inputs in diagnostic traces, outside public snapshots."""
    return observation.model_dump(mode="json") | (
        {"torso_histogram": observation.torso_histogram} if observation.torso_histogram else {}
    )


async def replay(
    source: Path,
    settings: Settings,
    output: Path,
    analysis: Analysis | None = None,
    max_seconds: float | None = None,
    cloud: bool = False,
    render: bool = False,
) -> dict:
    if (output / "state" / "stillwatch.sqlite3").exists():
        raise ValueError("Output already contains a replay. Choose a new output directory.")
    with source.open("rb") as handle:
        source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    settings = settings.model_copy(update={"data_dir": output / "state"})
    telemetry = analysis.telemetry if analysis else Telemetry()
    analysis = analysis or Analysis(settings, telemetry)
    if not any(state == "ready" for state in analysis.perception.states.values()):
        analysis.perception.load()
    analysis.epoch = -1
    buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
    storage = Storage(settings)
    evidence = EvidenceBuilder(settings, buffer)
    manager = IncidentManager(settings, storage, evidence)
    provider = OpenAIDecisionProvider(settings.request_timeout) if cloud else MockDecisionProvider(delay=0)
    scheduler = Scheduler(settings, manager, provider)
    epoch = max((i.epoch for i in manager.items.values()), default=0) + 1
    cap = open_capture(str(source))
    if not cap.isOpened():
        storage.close()
        raise ValueError("Video cannot be opened")
    clock = FileClock(cap.get(cv2.CAP_PROP_FPS))
    index, previous_ts, last_sample_ts = 0, -1.0, -float("inf")
    writer = None
    rendered_frames = 0
    render_fps = max(settings.buffer_fps, settings.pose_fps, settings.fire_fps)
    context_fps = max(settings.buffer_fps, render_fps if render else 0)
    started = time.monotonic()
    candidate_count = 0
    inference_calls = {"pose": 0, "fire": 0}
    maximum_buffer_bytes = 0
    pending_task: asyncio.Task | None = None
    try:
        with (output / "observations.jsonl").open("w", encoding="utf-8") as trace:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                raw_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                ts = clock.advance(raw_ts)
                previous_ts = ts
                index += 1
                if max_seconds is not None and ts > max_seconds:
                    break
                context_due = ts - last_sample_ts >= 1 / context_fps - 1e-6
                analytics_due = any(
                    analysis.perception.states[route] == "ready" and ts - previous >= 1 / cadence - 1e-6
                    for route, previous, cadence in (
                        ("pose", analysis.last_pose, settings.pose_fps),
                        ("fire", analysis.last_fire, settings.fire_fps),
                    )
                )
                if not (context_due or analytics_due):
                    continue
                # The JPEG cadence must not discard a frame due for either model.
                # Keep each source clock independent, as in live Analysis.process.
                if context_due:
                    last_sample_ts = ts
                sample = make_sample(image, index, ts, epoch, settings)
                buffer.add(sample.frame)
                candidates = analysis.process(sample)
                inferred_routes = [
                    route
                    for route, evaluated_ts in (("pose", analysis.last_pose), ("fire", analysis.last_fire))
                    if evaluated_ts == ts
                ]
                for route in inferred_routes:
                    inference_calls[route] += 1
                evidence.observe(analysis.observations["pose"], ts, epoch)
                manager.observe_motion(analysis.medical.motion.snapshot(), ts)
                scheduler.advance(ts, epoch)
                for candidate in candidates:
                    candidate_count += 1
                    item, created = manager.ingest(candidate)
                    if created:
                        scheduler.enqueue(item)
                trace.write(
                    json.dumps(
                        {
                            "frame_id": index,
                            "source_ts": ts,
                            "width": sample.frame.width,
                            "height": sample.frame.height,
                            "inferred_routes": inferred_routes,
                            "observations": {
                                k: [trace_observation(o) for o in v] for k, v in analysis.observations.items()
                            },
                            "detections": {
                                k: [trace_observation(o) for o in v] for k, v in analysis.detections.items()
                            },
                            "candidates": [c.model_dump(mode="json") for c in candidates],
                            "motion": [m.model_dump(mode="json") for m in analysis.medical.motion.snapshot()],
                        }
                    )
                    + "\n"
                )
                maximum_buffer_bytes = max(maximum_buffer_bytes, buffer.stats()["bytes"])
                if cloud:
                    # Real network time must not stop source decoding; one task is allowed in flight.
                    if pending_task is None or pending_task.done():
                        if pending_task:
                            await pending_task
                        pending_task = asyncio.create_task(scheduler.step())
                    await asyncio.sleep(0)
                else:
                    await scheduler.step()
                if render:
                    view = sample.image.copy()
                    for observations in analysis.observations.values():
                        for obs in observations:
                            b = obs.box
                            color = (139, 202, 175) if obs.label == "person" else (118, 183, 239)
                            cv2.rectangle(view, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
                            cv2.putText(
                                view,
                                f"{obs.label} {obs.track_id or ''}",
                                (int(b.x1), max(15, int(b.y1) - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                    cv2.putText(
                        view,
                        f"Stillwatch | {ts:.2f}s | {candidate_count} candidate observations",
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (240, 240, 240),
                        1,
                        cv2.LINE_AA,
                    )
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            render_fps,
                            (view.shape[1], view.shape[0]),
                        )
                    # Sampling a 24 Hz input at a 10 Hz ceiling yields about 8 Hz.
                    # Fill output slots by source time so annotated playback does not speed up.
                    while rendered_frames <= round(ts * render_fps):
                        writer.write(view)
                        rendered_frames += 1
            if pending_task:
                await pending_task
            # Flush existing ready jobs at EOF, without inventing future source time.
            for _ in range(settings.queue_capacity + 1):
                if not await scheduler.step():
                    break
        events = [i.model_dump(mode="json") for i in manager.items.values()]
        with (output / "observations.jsonl").open("rb") as handle:
            trace_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        result = {
            "source": source.name,
            "config_hash": settings.fingerprint,
            "implementation_hash": implementation_fingerprint(),
            "replay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sampling_policy": "independent_context_and_model_cadences",
            "inference_calls": inference_calls,
            "duration_seconds": max(0, previous_ts),
            "source_clock": clock.snapshot(),
            "source_sha256": source_sha256,
            "trace_sha256": trace_sha256,
            "wall_seconds": time.monotonic() - started,
            "frames_decoded": index,
            "routes": analysis.perception.states.copy(),
            "route_errors": analysis.perception.errors.copy(),
            "candidate_observations": candidate_count,
            "events": events,
            "local_event_count": sum(i.category != Category.OVERVIEW for i in manager.items.values()),
            "budget": scheduler.budget.snapshot(),
            "max_buffer_bytes": maximum_buffer_bytes,
            "rss_bytes": psutil.Process().memory_info().rss,
            "cloud_enabled": cloud,
            "semantic_evaluation": cloud,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        cap.release()
        if writer:
            writer.release()
        await scheduler.stop()
        storage.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/runs/replay"))
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--cloud", action="store_true", help="Spend configured API budget; requires cloud consent"
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.cloud and not settings.cloud_spending_enabled:
        parser.error("Cloud replay requires explicit spending consent in configuration")
    result = asyncio.run(
        replay(
            args.source,
            settings,
            args.output,
            max_seconds=args.max_seconds,
            cloud=args.cloud,
            render=args.render,
        )
    )
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
