"""Verify the running CI container and perform one CPU call per locked route."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request


def main():
    request = urllib.request.Request(
        "http://127.0.0.1:8765/api/v1/telemetry",
        headers={"Authorization": f"Bearer {os.environ['STILLWATCH_API_TOKEN']}"},
    )
    deadline = time.monotonic() + 90
    while True:
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                snapshot = json.load(response)
            if snapshot["routes"] == {"pose": "ready", "fire": "ready"}:
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Container did not make both model routes ready")
        time.sleep(1)
    assert snapshot["version"] == "0.5.0" and snapshot["mode"] == "mock"
    assert not snapshot["performance"]["error"] and not snapshot["storage"]["error"]
    assert snapshot["scheduler"]["budget"]["requests"] == 0
    code = """
import json
import cv2
import numpy as np
import torch
from stillwatch.config import Settings
from stillwatch.perception import Perception
p = Perception(Settings(device='cpu'))
p.load()
assert p.states == {'pose': 'ready', 'fire': 'ready'} and not p.errors
image = np.zeros((480, 640, 3), dtype=np.uint8)
p.people(image, 0)
p.hazards(image, 0)
assert torch.get_num_threads() == 4 and cv2.getNumThreads() == 1
print(json.dumps({'routes': p.states, 'device': 'cpu', 'torch_workers': 4, 'opencv_workers': 1, 'api_requests': 0}))
"""
    subprocess.run(["docker", "exec", "stillwatch-smoke", "/app/.venv/bin/python", "-c", code], check=True)
    print(json.dumps({"version": snapshot["version"], "routes": snapshot["routes"], "api_requests": 0}))


if __name__ == "__main__":
    main()
