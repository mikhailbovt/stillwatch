"""Check one real-time file session through the production runtime with bounded cloud use."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from stillwatch.config import Settings, load_settings
from stillwatch.runtime import Runtime
from stillwatch.scheduler import Budget


class SessionBudget(Budget):
    def reserve(self, now: float | None = None) -> tuple[int | None, str | None]:
        # A bounded test must not gain another allowance when UTC midnight passes.
        if len(self.storage.calls(self.provider, 0)) >= self.settings.requests_per_day:
            return None, "session_request_limit"
        return super().reserve(now)


async def check(source: Path, settings: Settings, output: Path, max_seconds: float) -> dict:
    source = source.resolve(strict=True)
    if not source.is_file() or max_seconds <= 0:
        raise ValueError("A local file and positive wall-clock limit are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings.model_validate(
        {
            **settings.model_dump(),
            "source": str(source),
            "loop": False,
            "realtime": True,
            "data_dir": output / "state",
        }
    )
    runtime = Runtime(settings)
    runtime.scheduler.budget = SessionBudget(settings, runtime.storage, settings.provider)
    started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    termination = "wall_limit"
    seen_assessments = set()
    assessments = []
    await runtime.start()
    try:
        with (output / "samples.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
            while time.monotonic() - started < max_seconds:
                snapshot = runtime.snapshot()
                snapshot["wall_elapsed_seconds"] = time.monotonic() - started
                trace.write(json.dumps(snapshot) + "\n")
                for incident in runtime.storage.incidents():
                    for result in runtime.storage.detail(incident.incident_id)["assessments"]:
                        if result["packet_id"] in seen_assessments:
                            continue
                        seen_assessments.add(result["packet_id"])
                        entry = {
                            "category": incident.category,
                            "detected_source_ts": incident.detected_ts,
                            "candidate_to_result_seconds": result["received_at"] - incident.created_at,
                            "result": result,
                        }
                        assessments.append(entry)
                        print(
                            json.dumps(
                                {
                                    "source": source.name,
                                    "category": incident.category,
                                    "medical": result["medical_status"],
                                    "fire": result["fire_status"],
                                    "tokens": result["input_tokens"],
                                    "stale": result["stale"],
                                    "error": result["error"],
                                }
                            ),
                            flush=True,
                        )
                if snapshot["capture_state"] in ("ended", "disconnected", "capture_error", "unavailable"):
                    # Let an outstanding response be recorded as stale after EOF.
                    # Runtime freshness checks prevent new requests from historical frames.
                    if runtime.scheduler.inflight is None:
                        termination = snapshot["capture_state"]
                        break
                if snapshot["scheduler"]["api_state"] == "access_denied":
                    termination = "access_denied"
                    break
                if snapshot["performance"]["error"] or snapshot["storage"]["error"]:
                    termination = "runtime_error"
                    break
                await asyncio.sleep(0.1)
        snapshot = runtime.snapshot()
        with source.open("rb") as source_file:
            source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
        summary = {
            "test": "real_time_file_runtime",
            "started_utc": started_utc,
            "elapsed_seconds": time.monotonic() - started,
            "termination": termination,
            "source": source.name,
            "source_sha256": source_hash,
            "provider": settings.provider,
            "implementation_hash": runtime.implementation_hash,
            "config_hash": settings.fingerprint,
            "budget": runtime.scheduler.budget.snapshot(),
            "final_snapshot": snapshot,
            "assessments": assessments,
            "limitations": [
                "Candidate-to-result time starts at the local candidate, not the visible event onset.",
                "Post-EOF responses remain in history and do not change the current assessment.",
                "This short session does not establish quality, tail latency, or daily cost.",
            ],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        await runtime.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0.1)
    parser.add_argument("--input-price-per-million", type=float, default=0.1)
    parser.add_argument("--max-requests", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=90)
    args = parser.parse_args()
    if args.cloud and not args.confirm_spend:
        parser.error("Cloud checks require --confirm-spend")
    if min(args.budget_usd, args.input_price_per_million, args.max_requests, args.max_seconds) <= 0:
        parser.error("Limits must be positive")
    base = load_settings(args.config)
    settings = Settings.model_validate(
        {
            **base.model_dump(),
            "provider": "openai" if args.cloud else "mock",
            "cloud_spending_enabled": args.cloud,
            "requests_per_minute": min(base.requests_per_minute, args.max_requests),
            "requests_per_day": args.max_requests,
            "input_tokens_per_day": math.floor(args.budget_usd / args.input_price_per_million * 1e6),
            "token_reservation": 64000,
        }
    )
    summary = asyncio.run(check(args.source, settings, args.output, args.max_seconds))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("assessments", "final_snapshot")}))


if __name__ == "__main__":
    main()
