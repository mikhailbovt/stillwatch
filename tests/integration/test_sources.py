import asyncio
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stillwatch.app import create_app
from stillwatch.runtime import Runtime
from stillwatch.sources import SourceRequest, SourceStore

HEADERS = {"X-Stillwatch-Action": "operator"}


def test_versioned_openapi_documents_auth_and_binary_upload(settings):
    schema = create_app(settings).openapi()
    assert "/source" not in schema["paths"]
    assert schema["paths"]["/api/v1/source"]["put"]["security"] == [{"BearerAuth": []}]
    assert "security" not in schema["paths"]["/api/v1/health"]["get"]
    upload = schema["paths"]["/api/v1/videos"]["post"]
    assert upload["requestBody"]["content"]["application/octet-stream"]["schema"]["format"] == "binary"


def test_injected_perception_avoids_builtin_model_loading(settings):
    class LocalAdapter:
        states = {"pose": "ready", "fire": "disabled"}
        errors = {}
        loaded = False

        def load(self):
            self.loaded = True

        def reset(self):
            pass

        def people(self, image, capture_ts):
            return []

        def hazards(self, image, capture_ts):
            return []

    adapter = LocalAdapter()
    app = create_app(settings, runtime_factory=lambda config: Runtime(config, perception=adapter))
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["routes"]["pose"] == "ready"
        assert adapter.loaded and app.state.runtime.analysis.perception is adapter


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for n in range(30):
        writer.write(np.full((120, 160, 3), 50 + n, np.uint8))
    writer.release()
    return path


def test_upload_select_stop_and_restart_preserve_service(settings, video):
    app = create_app(settings)
    with TestClient(app) as client:
        denied = client.post("/api/v1/videos?name=input.mp4", content=video.read_bytes())
        assert denied.status_code == 403
        uploaded = client.post("/api/v1/videos?name=input.mp4", content=video.read_bytes(), headers=HEADERS)
        assert uploaded.status_code == 201
        file_id = uploaded.json()["file_id"]
        assert client.get("/api/v1/videos").json()[0]["file_id"] == file_id
        first = client.put("/api/v1/source", json={"kind": "file", "file_id": file_id}, headers=HEADERS)
        assert first.status_code == 200
        deadline = time.monotonic() + 3
        while app.state.runtime.capture.latest is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert app.state.runtime.capture.latest is not None
        old_capture = app.state.runtime.capture
        old_epoch = old_capture.epoch
        stopped = client.put("/api/v1/source", json={"kind": "none"}, headers=HEADERS).json()
        assert stopped["epoch"] > old_epoch
        assert app.state.runtime.capture.latest is None
        assert not old_capture._thread.is_alive()
        assert client.get("/health").status_code == 200
        client.put("/api/v1/source", json={"kind": "file", "file_id": file_id}, headers=HEADERS)
        assert app.state.runtime.capture.epoch > stopped["epoch"]


def test_invalid_source_preserves_active_selection_and_redacts_input(settings):
    with TestClient(create_app(settings)) as client:
        before = client.get("/api/v1/source").json()
        invalid = client.put(
            "/api/v1/source",
            json={"kind": "rtsp", "url": "https://user:PRIVATE_PASSWORD@example.test"},
            headers=HEADERS,
        )
        assert invalid.status_code == 400 and "PRIVATE_PASSWORD" not in invalid.text
        bad_schema = client.put(
            "/api/v1/source", json={"kind": "rtsp", "camera_index": "PRIVATE_PASSWORD"}, headers=HEADERS
        )
        assert bad_schema.status_code == 422 and "PRIVATE_PASSWORD" not in bad_schema.text
        assert client.get("/api/v1/source").json() == before


def test_upload_limits_cleanup_partial_data(settings):
    settings.source_upload_max_bytes = 8
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/api/v1/videos?name=video.mp4", content=b"x" * 10, headers=HEADERS).status_code
            == 413
        )
        assert client.post("/api/v1/videos?name=code.py", content=b"x", headers=HEADERS).status_code == 400
        assert client.post("/api/v1/videos?name=empty.mp4", content=b"", headers=HEADERS).status_code == 400
        assert not list((settings.data_dir / "sources").iterdir())


def test_paths_require_explicit_allowlist_and_rtsp_descriptors_exclude_password(settings, video):
    store = SourceStore(settings)
    with pytest.raises(ValueError, match="source_roots"):
        store.resolve(SourceRequest(kind="file", path=str(video)))
    settings.source_roots = [video.parent]
    assert store.resolve(SourceRequest(kind="file", path=str(video)))[0] == str(video)
    source, label = store.resolve(
        SourceRequest(kind="rtsp", url="rtsp://user:PRIVATE_PASSWORD@127.0.0.1/live")
    )
    assert "PRIVATE_PASSWORD" in source and label == "RTSP · 127.0.0.1"
    with pytest.raises(ValueError):
        store.file("../../sensitive")


def test_network_auth_and_operator_session(settings, monkeypatch):
    settings.host = "0.0.0.0"
    monkeypatch.delenv(settings.api_token_env, raising=False)
    with pytest.raises(ValueError, match="API token"):
        create_app(settings)
    token = "test-token-01234567890123456789"
    monkeypatch.setenv(settings.api_token_env, token)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/events").status_code == 401
        assert client.get("/events").status_code == 401
        assert client.get("/api/v1/events", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        login = client.post("/api/v1/session", headers={"Authorization": f"Bearer {token}"})
        assert (
            login.status_code == 200 and token not in login.text and token not in login.headers["set-cookie"]
        )
        assert "HttpOnly" in login.headers["set-cookie"]
        assert client.get("/api/v1/events").status_code == 200


async def test_concurrent_source_changes_have_distinct_epochs(settings, video):
    settings.source_roots = [video.parent]
    runtime = Runtime(settings)
    try:
        a, b = await asyncio.gather(
            runtime.switch_source(SourceRequest(kind="file", path=str(video))),
            runtime.switch_source(SourceRequest(kind="none")),
        )
        assert b.epoch > a.epoch
        assert runtime.source_descriptor().kind == "none"
    finally:
        await runtime.stop()
