# Integration

The server owns one source, inference, episode state, evidence and the request ledger. The SDK is a separate HTTP client. Deploy one server process per source; aggregate events in your application. Run exactly one ASGI worker per data directory.

## Install the release

Download the wheel and `SHA256SUMS.txt` from the GitHub release and verify the checksum. Neither package has been published to PyPI.

```sh
pip install ./stillwatch_sdk-0.5.0-py3-none-any.whl
```

The SDK supports Python 3.10+ and depends only on HTTPX and Pydantic. The server wheel requires Python 3.12. Install the server with `[inference,cloud]` extras when those routes are needed. Model artifacts are separate and must match `models.lock.json`.

```python
import os
from stillwatch_sdk import Stillwatch, SourceSelection

with Stillwatch("http://127.0.0.1:8765", token=os.getenv("STILLWATCH_API_TOKEN")) as client:
    uploaded = client.upload_video("recording.mp4")
    client.select_source(SourceSelection(kind="file", file_id=uploaded.file_id))
    page = client.events(limit=100, offset=0)
    for event in page.events:
        if not event.assessment_stale:
            print(event.incident_id, event.analysis_status)
```

`AsyncStillwatch` exposes the same asynchronous methods. See the [SDK reference](../sdk/python/README.md) and [complete observation example](../examples/observe.py).

## HTTP contract

The versioned API is `/api/v1`. [OpenAPI](../api/openapi.json) is exported from the application and checked during release builds. Legacy root paths remain aliases for existing clients. API v1 can add fields; breaking field meanings or removals require a new version.

| Operation | Endpoint |
| --- | --- |
| Read health and live diagnostics | `GET /health`, `GET /telemetry` |
| List events and assessment history | `GET /events`, `GET /events/{id}` |
| Record feedback or close | `POST /events/{id}/feedback`, `POST /events/{id}/close` |
| Read/change/stop source | `GET /source`, `PUT /source` |
| Upload/list local videos | `POST /videos?name=recording.mp4`, `GET /videos` |
| Read registered JPEG evidence | `GET /evidence/{incident_id}/{packet_id}/{index}` |
| Observe full snapshots / MJPEG | `GET /updates`, `GET /stream` |

Uploads use a raw `application/octet-stream` body. Mutations require `X-Stillwatch-Action: operator`; the SDK supplies it. With authentication configured, pass `Authorization: Bearer ...`. The browser exchanges that token for a one-hour HttpOnly, SameSite Strict cookie. Use HTTPS for network access.

Source kinds are `file` (uploaded ID or an allowlisted server path), `camera` (server device index), `rtsp` and `none`. Selection is serialized, stops the previous capture and creates a fresh epoch and buffer. Models stay loaded; the browser's stream reconnects to the new capture without navigation. Invalid selection leaves the old source in place. A valid but unreachable camera changes to an explicit unavailable state. A `409` means the old capture has not yet stopped. Selection is session state; restart uses the configured source.

File telemetry includes `source_clock`: `mode`, decoded/nominal/held frame counts, timestamp errors, the last raw time and the monotonic source time. `source_time_stalled` blocks cloud work while native timestamps remain held; resumption starts a new epoch. `nominal_source_time` indicates FPS fallback without established native timing. Other source types return `source_clock: null`. Clock diagnostics do not establish physical timing or scene coverage.

## Event semantics

Keep lifecycle, assessment and operator feedback separate. `PENDING` means observation or assessment is unfinished. `UNKNOWN` includes unresolved local motion even when the cloud proposes clearance. `UNAVAILABLE` is an operational failure. `REVIEW` and `URGENT` request human attention. `NO_ALERT` describes observed evidence and cannot establish health.

Use `analysis_status` for event attention. After an established `REVIEW`, an uncertain or failed recheck keeps this field at `REVIEW` with `policy_reason: review_pending_resolution`. The route status fields retain the latest interpreted answers for that recheck; stored raw answers remain unchanged. A valid negative assessment can resolve review under the existing route rules. Medical resolution still requires compatible local evidence. Fresh independent episodes start with their own status and budget.

An `episode_id` follows continuous activity. New descents, ambiguous identity or lost visibility invalidate ordinary-activity clearance. Track IDs are short-lived geometric associations, not persistent person identity. Source epochs prevent old frames and answers from affecting a new source.

`motion.previous_episode_id` records a new descent after observed ordinary activity or recovery with continuous identity. An open predecessor records its successor in `superseded_by`. Superseded work is removed from the cloud queue; late responses remain stale in history. An already closed predecessor retains its outcome and snapshot. An established `URGENT` stays visible until handled. Local archive packets have `temporal_context.archived_without_assessment: true`; they consume no API request and are not assessments.

For an unresolved, identity-linked recurrence, `observation_started_ts` retains the first concern's detection time and `recurrence_count` counts handoffs. The successor has its own request allowance and needs twelve new frames, while the fifteen-second observation clock continues. Original onset images transfer within existing memory limits. The packet reports `observation_started_source_ts`, `recurrence_count`, total `elapsed_source_seconds` and `current_episode_elapsed_seconds`. Older payloads omit the optional start and default to zero recurrences. Closed predecessors, lost identity and different sources do not transfer a clock.

Joint local/cloud confirmation of recovery closes an open, unacknowledged medical episode immediately. Automatic closure creates no operator feedback entry. Continuing supported floor activity retains its episode; later concerns have independent assessment allowances. `temporal_context.observed_resolution_check` identifies a check triggered by newly observed recovery or ordinary activity, including checks within the initial allowance.

SSE is a live snapshot stream, not a durable event bus: it has no replay cursor or exactly-once delivery. Snapshots include at most the newest 200 events. On reconnect, paginate `/events` and deduplicate on incident ID, packet version and update time. Keep a separate durable consumer ledger before sending any external notification. The SDK does not retry writes or automatically notify anyone.

## Replace local models or the decision provider

`PerceptionAdapter` in `stillwatch.interfaces` describes `load`, `reset`, `people`, `hazards`, `states` and `errors`. Return source-timestamped `Observation` values in image pixel coordinates. Pose observations use 17 COCO keypoints; unreliable points must have low confidence. Detector IDs are optional: the server associates observations into bounded local tracks. Mark cropped observations with `truncated` and missing/inadequate observations with `limited`. Clearance requires observed continuity and reliable current posture.

Inject an adapter and/or `DecisionProvider` into `Runtime(settings, perception=..., provider=...)`. Inject that runtime through `create_app(settings, runtime_factory=...)`. Local inference runs on one dedicated thread; decisions run on the asyncio loop. Return answers bound to the supplied packet/incident/version. Preserve raw responses, usage and errors. Adapters run as trusted server code, not uploaded plugins.

The built-in motion classifier is a bounded causal geometric baseline, not a trained or clinically validated action model. Integrations can substitute licensed perception while retaining scheduling and operator workflows. They must independently validate the resulting motion quality; the adapter interface does not transfer the supplied models' validation or licenses.
