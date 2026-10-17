# Stillwatch SDK

Typed synchronous and asynchronous clients for Stillwatch HTTP API v1. Python 3.10+. No camera, OpenAI, PyTorch or OpenCV dependencies. The SDK talks to a running Stillwatch server; it does not classify video itself.

```python
import os
from stillwatch_sdk import Stillwatch, SourceSelection

with Stillwatch("http://127.0.0.1:8765", token=os.getenv("STILLWATCH_API_TOKEN")) as client:
    video = client.upload_video("recording.mp4")
    client.select_source(SourceSelection(kind="file", file_id=video.file_id))
    for snapshot in client.updates():
        for event in snapshot.get("events", []):
            print(event["incident_id"], event["analysis_status"])
```

```python
from stillwatch_sdk import AsyncStillwatch


async def inspect():
    async with AsyncStillwatch("http://127.0.0.1:8765") as client:
        page = await client.events(limit=50)
        for event in page.events:
            print(event.incident_id, event.analysis_status, event.motion)
```

Both clients expose `health`, `telemetry`, `source`, `select_source`, `stop_source`, `videos`, `upload_video`, `events`, `event`, `feedback`, `close_event`, `evidence`, and `updates`. Pagination is explicit. Uploads stream bytes. `updates` yields full SSE snapshots; consumers own reconnection and deduplication. It is not a durable event bus.

Use `SourceSelection(kind="camera", camera_index=0)` for a camera on the server or `SourceSelection(kind="rtsp", url="rtsp://camera/stream")` for a stream. Server file paths require an explicit `source_roots` allowlist. Source replacement starts a new epoch and preserves old incidents.

Network access requires a server access token. Send it as `token=...`; never put it in a URL. Use HTTPS outside a trusted local connection. `StillwatchError` contains the HTTP status and a redacted server error; transport errors are standard `httpx` exceptions. Mutating requests are never retried automatically.

`NO_ALERT` is an assessment of the supplied evidence, not a guarantee of safety. Keep `UNKNOWN`, `UNAVAILABLE`, stale observations and human feedback distinct. API v1 may add fields; models preserve unknown fields. Breaking contract changes require a new API version.

Original SDK code: Apache-2.0. Model and video licenses are independent of this client.
