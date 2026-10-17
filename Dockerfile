FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    YOLO_AUTOINSTALL=false YOLO_OFFLINE=true OPENCV_LOG_LEVEL=SILENT
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md models.lock.json ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra inference --extra cloud
COPY configs ./configs
RUN useradd --create-home --uid 10001 stillwatch && mkdir -p /app/data /app/models \
    && chown -R stillwatch:stillwatch /app/data /app/models
USER stillwatch
EXPOSE 8765
CMD ["uv", "run", "--frozen", "--no-sync", "stillwatch", "--config", "configs/default.yaml", "--host", "0.0.0.0"]
