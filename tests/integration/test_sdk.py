from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from stillwatch.app import create_app
from stillwatch.types import Candidate, Category


@pytest.fixture(autouse=True)
def sdk_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "sdk/python/src"))


def test_sync_sdk_uses_versioned_contract_and_operator_actions(settings, tmp_path):
    from stillwatch_sdk import SourceSelection, Stillwatch, StillwatchError

    app = create_app(settings)
    with TestClient(app) as server:
        item = app.state.runtime.manager.ingest(
            Candidate(category=Category.MEDICAL, reason="test", first_ts=1, last_ts=2)
        )[0]

        def handle(request):
            assert request.url.path.startswith("/api/v1/")
            result = server.request(
                request.method, str(request.url), headers=dict(request.headers), content=request.read()
            )
            return httpx.Response(result.status_code, json=result.json())

        with Stillwatch("http://testserver", transport=httpx.MockTransport(handle)) as client:
            assert client.health()["service"] == "running"
            assert client.source().kind == "none"
            page = client.events(limit=1)
            assert page.total == 1 and page.events[0].incident_id == item.incident_id
            assert client.event(item.incident_id).event.analysis_status == "PENDING"
            assert client.feedback(item.incident_id, "false_alarm").feedback == "false_alarm"
            assert client.close_event(item.incident_id).lifecycle == "CLOSED"
            with pytest.raises(StillwatchError) as error:
                client.event("missing")
            assert error.value.status_code == 404
            video = tmp_path / "small.mp4"
            video.write_bytes(b"streaming-upload-fixture")
            uploaded = client.upload_video(video)
            assert uploaded.bytes == video.stat().st_size
            assert client.videos()[0].file_id == uploaded.file_id
            assert client.select_source(SourceSelection(kind="none")).kind == "none"
            assert client.stop_source().kind == "none"


async def test_async_sdk_matches_server_contract(settings, tmp_path):
    from stillwatch_sdk import AsyncStillwatch

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncStillwatch("http://testserver", transport=httpx.ASGITransport(app=app)) as client:
            assert (await client.health())["service"] == "running"
            assert (await client.events()).events == []
            assert (await client.source()).kind == "none"
            assert (await client.stop_source()).kind == "none"
            path = tmp_path / "upload.mp4"
            path.write_bytes(b"async-upload-fixture")
            assert (await client.upload_video(path)).bytes == path.stat().st_size
            assert len(await client.videos()) == 1


def test_sdk_preserves_optional_episode_links_in_server_responses(system):
    from stillwatch_sdk.models import Event

    from stillwatch.types import Candidate, Category, MotionEvidence

    manager = system[4]
    previous, _ = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=1,
            last_ts=2,
            track_id=1,
            epoch=1,
            motion=MotionEvidence(
                episode_id="before", track_id=1, epoch=1, started_ts=1, observed_ts=2, last_seen_ts=2
            ),
        )
    )
    current, _ = manager.ingest(
        Candidate(
            category=Category.MEDICAL,
            reason="rapid_posture_change",
            first_ts=5,
            last_ts=6,
            track_id=1,
            epoch=1,
            motion=MotionEvidence(
                episode_id="after",
                previous_episode_id="before",
                track_id=1,
                epoch=1,
                started_ts=5,
                observed_ts=6,
                last_seen_ts=6,
            ),
        )
    )
    assert Event.model_validate(previous.model_dump()).superseded_by == current.incident_id
    linked = Event.model_validate(current.model_dump())
    assert linked.motion.previous_episode_id == "before"
    assert linked.observation_started_ts == 2 and linked.recurrence_count == 1
    old_payload = previous.model_dump(exclude={"superseded_by", "observation_started_ts", "recurrence_count"})
    old_payload["motion"].pop("previous_episode_id")
    restored = Event.model_validate(old_payload)
    assert restored.superseded_by is None and restored.motion.previous_episode_id is None
    assert restored.observation_started_ts is None and restored.recurrence_count == 0


def test_sdk_streams_snapshots_without_accumulating_history():
    from stillwatch_sdk import Stillwatch

    def response(request):
        assert request.headers["Authorization"] == "Bearer client-token"
        return httpx.Response(
            200, content=b'event: snapshot\ndata: {"revision":1}\n\nevent: snapshot\ndata: {"revision":2}\n\n'
        )

    with Stillwatch(
        "http://testserver", token="client-token", transport=httpx.MockTransport(response)
    ) as client:
        assert list(client.updates()) == [{"revision": 1}, {"revision": 2}]


async def test_async_sdk_stream_and_bounded_decoder():
    from stillwatch_sdk import AsyncStillwatch
    from stillwatch_sdk.client import SnapshotDecoder

    async def response(request):
        return httpx.Response(200, content=b'data: {"revision":3}\n\n')

    async with AsyncStillwatch("http://testserver", transport=httpx.MockTransport(response)) as client:
        assert [s async for s in client.updates()] == [{"revision": 3}]
    decoder = SnapshotDecoder()
    with pytest.raises(ValueError, match="limit"):
        decoder.feed("data:" + "x" * (8 * 1024**2))


def test_sdk_never_retries_a_mutation_and_hides_stream_credentials():
    from stillwatch_sdk import SourceSelection, Stillwatch, StillwatchError

    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": "Temporarily unavailable"})

    selection = SourceSelection(kind="rtsp", url="rtsp://user:PRIVATE_PASSWORD@camera/live")
    assert "PRIVATE_PASSWORD" not in repr(selection)
    with Stillwatch("http://testserver", transport=httpx.MockTransport(response)) as client:
        with pytest.raises(StillwatchError):
            client.select_source(selection)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        Stillwatch("http://user:password@example.test")
