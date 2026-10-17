from stillwatch.buffer import FrameBuffer, evenly_spaced
from stillwatch.types import Frame


def frame(i, ts=None, size=10, epoch=0):
    return Frame(i, "test", i / 10 if ts is None else ts, 16, 16, bytes(size), epoch)


def test_time_and_byte_limits():
    buffer = FrameBuffer(12, 64, 10)
    for i in range(200):
        buffer.add(frame(i, size=10 if i % 2 else 20))
        assert buffer.stats()["bytes"] <= 64
        assert buffer.stats()["depth_seconds"] <= 12
    assert buffer.frames()[-1].frame_id == 199


def test_vfr_repeated_frame_and_reset():
    buffer = FrameBuffer(12, 10000, 10)
    for i, ts in enumerate([0, 0, 0.01, 0.11, 0.39, 0.41, 1.3]):
        buffer.add(frame(i, ts))
    assert [f.capture_ts for f in buffer.frames()] == [0, 0.11, 0.39, 1.3]
    buffer.add(frame(7, 0.05, epoch=1))
    assert [f.frame_id for f in buffer.frames()] == [7]
    buffer.add(frame(8, 0, epoch=1))
    assert [f.frame_id for f in buffer.frames()] == [8]


def test_selection_keeps_pretrigger_context():
    frames = [frame(i) for i in range(120)]
    selected = evenly_spaced(frames, 6)
    assert selected[0].capture_ts == 0
    assert selected[-1].capture_ts == 11.9
    assert len({f.frame_id for f in selected}) == 6


def test_oversized_frame_is_not_retained():
    buffer = FrameBuffer(12, 64, 10)
    buffer.add(frame(1, size=100))
    assert buffer.stats()["bytes"] == 0
