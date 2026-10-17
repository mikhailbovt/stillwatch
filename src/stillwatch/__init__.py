"""Stillwatch: bounded, local-first video triage."""

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

__version__ = "0.5.0"
