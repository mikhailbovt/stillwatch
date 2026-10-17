<img src="src/stillwatch/web/mark.svg" width="44" height="44" alt="Stillwatch">

# Stillwatch

Video event triage for human review. Local detection, retained event context, cloud assessment and a Python SDK.

**0.5.0 · Preview**

https://github.com/user-attachments/assets/0d268116-8ec5-4f05-9032-de4e49ac49ea

53 seconds · Staged fall observation and fire detection · Original speed. [Download MP4](https://github.com/mikhailbovt/stillwatch/releases/download/v0.5.0/stillwatch-demo-0.5.0.mp4) · [Footage credits and outcomes](docs/demo.md).

## Run

Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker for initial model conversion.

```sh
git clone https://github.com/mikhailbovt/stillwatch.git
cd stillwatch
uv sync --frozen --extra inference --extra cloud --group dev
docker build -t stillwatch:0.5.0 .
uv run python tools/fetch_models.py --convert
uv run stillwatch
```

Open **http://127.0.0.1:8765**. Select a video, connected camera or RTSP stream. Sources change without reloading the page or models. Run one process per source and data directory.

Startup uses mock assessments (`UNKNOWN`) and makes no API requests. To enable cloud assessment, create ignored `.env.local` with `OPENAI_API_KEY`, then set `provider: openai` and `cloud_spending_enabled: true` in your configuration. The account needs access to `gpt-6-luna` and the Decisions API. Selected masked frames are sent to OpenAI. [Setup and deployment](docs/deployment.md).

## Processing

| Component | Default |
| --- | --- |
| People and posture | YOLO26s Pose · CPU · 5 Hz target |
| Smoke and flame | YOLOv8n Fire/Smoke · 2 Hz target |
| Event context | 30 seconds · 64 MiB buffer |
| Full medical assessment | 15 observed source seconds and 12 new frames |
| Recovery evidence | Sustained activity or return to an observed seated baseline |
| Cloud queue | One request in flight · eight waiting |

The service retains the beginning and development of an episode. Repeated suspicion does not restart its observation clock. A bounded early screen handles visible immediate danger. Local motion and cloud evidence must agree before medical clearance; lost or ambiguous identity remains uncertain.

## Integrate

[Releases](https://github.com/mikhailbovt/stillwatch/releases) include server and SDK wheels, source archives, OpenAPI, examples and SHA-256 checksums. The SDK supports Python 3.10+ and has no inference dependency.

```sh
pip install ./stillwatch_sdk-0.5.0-py3-none-any.whl
```

```python
from stillwatch_sdk import Stillwatch

with Stillwatch("http://127.0.0.1:8765") as client:
    for event in client.events(limit=20).events:
        print(event.incident_id, event.analysis_status)
```

[API and embedding](docs/integration.md) · [SDK reference](sdk/python/README.md) · [Model setup](docs/models.md)

## Status

Preview for supervised evaluation. Misses and false signals remain possible, especially with distant people, occlusion and crowds. Injury, actual assistance need and deployment alert accuracy are not established. Physical camera and RTSP acceptance depend on the installation. [Known limitations](docs/limitations.md).

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## License

Original service and SDK: [Apache-2.0](LICENSE). The supplied inference runtime and model candidates have separate AGPL terms. Weights, raw datasets, credentials and event databases are excluded. Demonstration footage retains its own license. [Third-party notices](THIRD_PARTY_NOTICES.md).
