from pathlib import Path

import cv2
import numpy as np
import pytest


async def test_live_check_stops_at_eof_and_preserves_previous_evidence(settings, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from tools.check_live import check

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for index in range(10):
        writer.write(np.full((120, 160, 3), index * 20, np.uint8))
    writer.release()
    settings.pose_enabled = settings.fire_enabled = False
    output = tmp_path / "live"
    result = await check(source, settings, output, max_seconds=10)
    assert result["termination"] == "ended"
    assert result["elapsed_seconds"] < 5
    assert result["provider"] == result["budget"]["provider"] == "mock"
    assert result["budget"]["usage_kind"] == "simulated"
    assert result["final_snapshot"]["performance"]["error"] is None
    original = (output / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="new output directory"):
        await check(source, settings, output, max_seconds=10)
    assert (output / "summary.json").read_bytes() == original


def test_live_request_cap_survives_utc_midnight(settings, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from stillwatch.config import Settings
    from stillwatch.storage import Storage
    from tools.check_live import SessionBudget

    settings = Settings.model_validate(
        {**settings.model_dump(), "requests_per_minute": 1, "requests_per_day": 1}
    )
    storage = Storage(settings)
    try:
        budget = SessionBudget(settings, storage, "mock")
        identifier, reason = budget.reserve(now=86399)
        assert identifier is not None and reason is None
        storage.finish_call(identifier, 10)
        assert budget.reserve(now=86401) == (None, "session_request_limit")
    finally:
        storage.close()
