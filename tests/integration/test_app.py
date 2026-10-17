import socket

import numpy as np
from fastapi.testclient import TestClient

from stillwatch.app import create_app
from stillwatch.capture import sanitize
from stillwatch.types import Box, Candidate, Category


def test_offline_service_never_connects_to_provider(settings, monkeypatch):
    original = socket.socket.connect

    def checked(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("Offline service attempted a network connection")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", checked)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["mode"] == "mock" and health["service"] == "running"
        assert "mock_has_no_semantic_evaluation" in health["limitations"]
        assert client.get("/").status_code == 200
        assert client.get("/events").json()["events"] == []
        assert client.get("/telemetry").json()["mode"] == "mock"


def test_operator_workflow_and_controlled_evidence_endpoint(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        runtime = app.state.runtime
        # Inject a local candidate, never a fabricated cloud evaluation.
        candidate = Candidate(
            category=Category.MEDICAL,
            reason="persistent_low_posture",
            first_ts=0,
            last_ts=2,
            box=Box(x1=0, y1=0, x2=100, y2=50),
            epoch=1,
        )
        item, _ = runtime.manager.ingest(candidate)
        path = f"/events/{item.incident_id}"
        assert client.get(path).status_code == 200
        assert client.post(path + "/feedback", json={"action": "confirmed"}).status_code == 403
        marked = client.post(
            path + "/feedback", json={"action": "confirmed"}, headers={"X-Stillwatch-Action": "operator"}
        ).json()
        assert marked["feedback"] == "confirmed" and marked["analysis_status"] == "PENDING"
        closed = client.post(path + "/close", headers={"X-Stillwatch-Action": "operator"}).json()
        assert closed["lifecycle"] == "CLOSED"
        assert client.get(f"/evidence/{item.incident_id}/{'f' * 32}/0").status_code == 404
        assert client.get("/evidence/invalid/invalid/0").status_code == 404
        assert len(client.get(path).json()["feedback_history"]) == 2


def test_privacy_mask_precedes_detection_buffer_and_evidence(settings):
    settings.privacy_masks = [(0, 0, 0.5, 1)]
    image = np.full((100, 100, 3), 200, np.uint8)
    result = sanitize(image, settings)
    assert result[:, :50].sum() == 0
    assert result[:, 50:].min() == 200
    assert image.min() == 200
