from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import load_settings


def main():
    parser = argparse.ArgumentParser(description="Stillwatch local video triage")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", help="Override source with a file, USB index, or RTSP URL")
    parser.add_argument(
        "--host", help="Override bind address; use a protected network for non-loopback access"
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.source is not None:
        if args.source.isdecimal():
            settings.source = int(args.source)
        elif args.source.startswith(("rtsp://", "rtsps://")):
            settings.source = args.source
        else:
            path = Path(args.source).resolve()
            if not path.is_file():
                parser.error("Video source does not exist")
            settings.source = str(path)
    if args.host:
        settings.host = args.host
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        access_log=False,
        log_level="info",
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
