from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


@lru_cache(maxsize=1)
def implementation_fingerprint() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    camera_id: str = "local-01"
    source: str | int | None = None
    realtime: bool = True
    loop: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    provider: Literal["mock", "openai"] = "mock"
    cloud_spending_enabled: bool = False
    data_dir: Path = Path("data")
    pose_weights: Path = Path("models/pose-balanced.safetensors")
    fire_weights: Path = Path("models/fire.safetensors")
    pose_enabled: bool = True
    fire_enabled: bool = True
    device: str = "cpu"
    inference_size: int = Field(default=640, ge=320, le=1280)
    pose_fps: float = Field(default=5, gt=0, le=30)
    fire_fps: float = Field(default=2, gt=0, le=30)
    buffer_seconds: float = Field(default=30, gt=0, le=120)
    buffer_fps: float = Field(default=10, gt=0, le=30)
    buffer_max_bytes: int = Field(default=64 * 1024**2, gt=0)
    jpeg_long_side: int = Field(default=1280, ge=320, le=1920)
    jpeg_quality: int = Field(default=80, ge=20, le=95)
    overview_seconds: float = Field(default=120, gt=0)
    evidence_seconds: float = Field(default=6, gt=0)
    queue_capacity: int = Field(default=8, ge=1, le=64)
    requests_per_minute: int = Field(default=12, gt=0)
    requests_per_day: int = Field(default=2000, gt=0)
    input_tokens_per_day: int = Field(default=20_000_000, gt=0)
    token_reservation: int = Field(default=32_000, gt=0)
    request_timeout: float = Field(default=10, gt=0, le=120)
    retries: int = Field(default=1, ge=0, le=1)
    medical_confirmation_frames: int = Field(default=12, ge=0, le=120)
    medical_observation_seconds: float = Field(default=15, ge=0, le=60)
    medical_early_screen_seconds: float = Field(default=2, ge=0, le=10)
    recheck_seconds: list[float] = Field(default_factory=lambda: [3, 10])
    max_episode_requests: int = Field(default=3, ge=1, le=3)
    packet_stale_seconds: float = Field(default=5, gt=0)
    frame_stale_seconds: float = Field(default=3, gt=0)
    drop_ratio: float = Field(default=0.28, gt=0, le=1)
    pre_drop_ratio: float = Field(default=0.10, gt=0, le=1)
    drop_window_seconds: float = Field(default=0.8, gt=0)
    low_pose_seconds: float = Field(default=2, gt=0)
    low_aspect_ratio: float = Field(default=1.2, gt=0)
    keypoint_confidence: float = Field(default=0.4, ge=0, le=1)
    minimum_person_height: int = Field(default=40, gt=0)
    lost_track_seconds: float = Field(default=0.6, gt=0)
    dedup_seconds: float = Field(default=12, gt=0)
    dedup_iou: float = Field(default=0.25, gt=0, le=1)
    fire_confidence: float = Field(default=0.2, ge=0, le=1)
    fire_immediate_confidence: float = Field(default=0.65, ge=0, le=1)
    frame_retention_hours: float = Field(default=24, gt=0)
    metadata_retention_days: float = Field(default=7, gt=0)
    storage_quota_bytes: int = Field(default=500 * 1024**2, gt=0)
    privacy_masks: list[tuple[float, float, float, float]] = Field(default_factory=list)
    operator_id: str = "local-operator"
    source_roots: list[Path] = Field(default_factory=list)
    source_upload_max_bytes: int = Field(default=512 * 1024**2, gt=0)
    source_storage_quota_bytes: int = Field(default=2 * 1024**3, gt=0)
    api_token_env: str = "STILLWATCH_API_TOKEN"
    recovery_seconds: float = Field(default=3, ge=1, le=15)
    ordinary_activity_seconds: float = Field(default=2, ge=1, le=15)

    @model_validator(mode="after")
    def coherent(self):
        if self.evidence_seconds > self.buffer_seconds:
            raise ValueError("Evidence window must fit the buffer")
        if self.medical_observation_seconds + self.evidence_seconds > self.buffer_seconds:
            raise ValueError("Medical observation and pre-event context must fit the buffer")
        if self.requests_per_day < self.requests_per_minute:
            raise ValueError("Daily request limit must cover a minute")
        if self.token_reservation > self.input_tokens_per_day:
            raise ValueError("Token reservation exceeds daily budget")
        if any(s <= 0 for s in self.recheck_seconds) or self.recheck_seconds != sorted(self.recheck_seconds):
            raise ValueError("Recheck offsets must be positive and sorted")
        if self.fire_immediate_confidence < self.fire_confidence:
            raise ValueError("Immediate confidence must exceed candidate confidence")
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("Source cannot be empty")
        if isinstance(self.source, int) and self.source < 0:
            raise ValueError("Camera index must be nonnegative")
        if self.provider == "openai" and not self.cloud_spending_enabled:
            raise ValueError("Cloud mode requires explicit spending consent")
        for x1, y1, x2, y2 in self.privacy_masks:
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Privacy masks must be normalized rectangles")
        return self

    @property
    def fingerprint(self) -> str:
        values = self.model_dump(mode="json", exclude={"data_dir", "source", "host", "port"})
        values["pose_weights"] = self.pose_weights.name
        values["fire_weights"] = self.fire_weights.name
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


def load_settings(path: Path) -> Settings:
    path = path.resolve()
    settings = Settings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    root = path.parent.parent
    for name in ("data_dir", "pose_weights", "fire_weights"):
        p = getattr(settings, name)
        if not p.is_absolute():
            setattr(settings, name, root / p)
    settings.source_roots = [(root / p).resolve() for p in settings.source_roots]
    if isinstance(settings.source, str) and "://" not in settings.source:
        settings.source = str((root / settings.source).resolve())
        if not Path(settings.source).is_file():
            raise ValueError("Configured video file does not exist")
    if isinstance(settings.source, str) and "://" in settings.source:
        if not settings.source.startswith(("rtsp://", "rtsps://")):
            raise ValueError("Only local files, USB indices, and RTSP sources are supported")
    return settings
