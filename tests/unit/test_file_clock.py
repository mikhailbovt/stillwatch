import math
import time

import pytest

from stillwatch.capture import FileClock, make_sample
from stillwatch.runtime import Runtime


@pytest.mark.parametrize(
    ("times", "expected", "mode"),
    [
        (
            [0, 0.08, 0.3, 0.3, 0.28, math.nan, math.inf, -1, 0.8],
            [0, 0.08, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8],
            "native_pts",
        ),
        ([0] * 5, [0, 0.04, 0.08, 0.12, 0.16], "nominal_fps"),
        ([math.nan, 0, 0.04, 0.08, 0.2], [0, 0.04, 0.04, 0.08, 0.2], "native_pts"),
        ([12, 12, 11, 12.2], [12, 12, 12, 12.2], "native_pts"),
    ],
)
def test_observed_time_never_advances_past_native_time_after_it_becomes_available(times, expected, mode):
    clock = FileClock(25)
    assert [clock.advance(ts) for ts in times] == pytest.approx(expected)
    snapshot = clock.snapshot()
    assert snapshot["mode"] == mode
    assert snapshot["frames_decoded"] == len(times)
    assert snapshot["source_ts"] == pytest.approx(expected[-1])


@pytest.mark.parametrize("fps", [0, -1, math.nan, math.inf, 241])
def test_missing_native_time_uses_a_finite_declared_nominal_rate(fps):
    clock = FileClock(fps)
    assert [clock.advance(0) for _ in range(4)] == pytest.approx([0, 0.04, 0.08, 0.12])
    assert clock.snapshot()["mode"] == "nominal_fps"
    assert clock.snapshot()["nominal_fps"] == 25


async def test_stalled_source_time_prevents_dispatch_despite_fresh_decoded_images(settings):
    import numpy as np

    runtime = Runtime(settings)
    try:
        capture = runtime.capture
        capture.state = "streaming"
        capture.file_clock = FileClock(25)
        capture.file_clock.advance(1)
        capture.file_clock.advance(1)
        capture.latest = make_sample(np.zeros((80, 100, 3), np.uint8), 0, 1, 1, settings)
        capture.file_clock.last_progress_mono = time.monotonic() - settings.frame_stale_seconds - 1
        assert not runtime._fresh_source()
        assert "source_time_stalled" in runtime.snapshot()["limitations"]
        assert runtime.snapshot()["source_clock"]["stalled"]
        capture.file_clock.advance(1.04)
        capture.latest = make_sample(np.zeros((80, 100, 3), np.uint8), 1, 1.04, 1, settings)
        assert runtime._fresh_source()
        assert "source_time_stalled" not in runtime.snapshot()["limitations"]
    finally:
        await runtime.stop()


def test_waiting_between_valid_native_times_is_not_a_timestamp_plateau():
    clock = FileClock(25)
    clock.advance(1)
    clock.last_progress_mono = time.monotonic() - 10
    assert not clock.stalled(3)
    clock.advance(1)
    assert clock.stalled(3)
