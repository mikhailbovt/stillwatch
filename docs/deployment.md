# Deployment

Stillwatch 0.5.0 provides integration artifacts for a preview human-review workflow. See [known limitations](limitations.md) before deployment.

## Process and access

Use one process and one ASGI worker per camera/data directory. Do not share a SQLite file between servers. Keep the default localhost binding for a local desk. Any other bind address requires `STILLWATCH_API_TOKEN` in the process environment, with at least 24 random characters. This token is distinct from the OpenAI API key. Put a TLS reverse proxy in front of remote access, limit access to intended operators, and disable proxy buffering for SSE/MJPEG. The default health route is public and exposes operational status only.

CPU mode uses four PyTorch inference threads and one OpenCV worker. File/RTSP decoding requests at most two FFmpeg threads; USB keeps its native backend. Passive OpenMP waiting defaults reduce idle CPU use. Existing `OMP_WAIT_POLICY` and `KMP_BLOCKTIME` values are preserved. When embedding the service, set these before importing PyTorch. Worker counts are not a total process CPU quota.

The server reads `OPENAI_API_KEY` from ignored `.env.local` or the environment. Cloud operation also requires `provider: openai` and `cloud_spending_enabled: true`. Only selected masked images are sent to OpenAI. The SDK never needs that key. Apply account-level spending limits in addition to local request limits; local token reservations are estimates, not dollar guarantees.

## Container

```powershell
docker build -t stillwatch:0.5.0 .
# Set STILLWATCH_API_TOKEN securely in the calling environment first.
docker run --rm --name stillwatch -p 127.0.0.1:8765:8765 `
  -e STILLWATCH_API_TOKEN `
  --mount "type=bind,source=$PWD/models,target=/app/models,readonly" `
  --mount "type=bind,source=$PWD/data,target=/app/data" `
  stillwatch:0.5.0
```

The image runs as UID 10001. Its internal `0.0.0.0` bind requires the access token even when the published port is loopback. On Linux, provision writable data storage for that UID. Mount sources read-only and allowlist their container paths in `source_roots`, or upload through the desk. USB passthrough and RTSP connectivity depend on the host and remain hardware acceptance tasks.

## Storage and recovery

The JPEG buffer is bounded to 30 seconds / 64 MiB; pinned onset evidence has a shared 16 MiB cap. Selected evidence defaults to 500 MiB, 24-hour frames and seven-day metadata. SQLite, logs and source uploads are outside the JPEG quota. Uploads have a separate 512 MiB per-file and 2 GiB total quota. They persist until an administrator removes unused uploads from the stopped service's `data/sources` directory. Keep its JSON metadata with each video; continuous capture is not recorded there.

Restart marks old assessments stale and requires fresh source evidence. Source switching invalidates queued and in-flight old-source results. If capture cannot stop, switching fails visibly instead of opening a second device reader. Authentication failure pauses cloud requests. Missing models, transport failure, storage failure and stale frames remain visible; none mean `NO_ALERT`.

Back up the data directory with the service stopped, or use a SQLite-aware backup procedure including registered evidence files. Test restore before relying on retained evidence. Use process supervision and read `/api/v1/health` plus `/api/v1/telemetry`; HTTP availability alone does not establish model availability or fresh frames.

## Release contents and licenses

Release assets include server and SDK wheels/source archives, OpenAPI, integration examples and SHA-256 checksums. No credentials, model weights, source videos, run databases or evidence frames are included. The integration ZIP contains deployment notes and a default offline configuration. Rebuild with `uv run python tools/build_release.py --output dist/new-build` after checking the frozen lock and tests.

Original application and SDK code are Apache-2.0. The supplied inference stack and candidate weights have separate AGPL terms; see [third-party notices](../THIRD_PARTY_NOTICES.md). The HTTP SDK has no model dependency. Validate suitable model/runtime licensing for your deployment or provide a separately licensed adapter. Research footage is not licensed by the application's Apache license and is excluded from distributions.
