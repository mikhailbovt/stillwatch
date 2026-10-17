"""Inspect local video timestamps without loading models or using the cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import cv2

from stillwatch.capture import FileClock
from stillwatch.config import implementation_fingerprint


def audit(source: Path) -> dict:
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("A local video file is required")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Video cannot be opened")
    clock = FileClock(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    declared_frames = (
        int(declared_frames) if math.isfinite(declared_frames) and declared_frames >= 0 else None
    )
    backend = capture.getBackendName()
    first_ts = None
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            ts = clock.advance(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            if first_ts is None:
                first_ts = ts
    finally:
        capture.release()
    return {
        "source": source.name,
        "source_sha256": digest,
        "decoder_backend": backend,
        "opencv_version": cv2.__version__,
        "declared_frames": declared_frames,
        "first_source_ts": first_ts,
        "observed_source_span_seconds": clock.source_ts - first_ts if first_ts is not None else None,
        "source_clock": clock.snapshot(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new path")
    report = {
        "schema": "stillwatch-clock-audit-v1",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "implementation_hash": implementation_fingerprint(),
        "records": [audit(source) for source in args.sources],
        "scope": "Decoder timestamp diagnostics only. Held or nominal times do not reconstruct missing physical time or validate model quality.",
        "api_requests": 0,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
