from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

INSTRUCTION_VERSION = "visible-evidence-v7"


def uid() -> str:
    return uuid4().hex


class Category(StrEnum):
    MEDICAL = "medical"
    FIRE = "fire"
    OVERVIEW = "overview"


class Status(StrEnum):
    PENDING = "PENDING"
    NO_ALERT = "NO_ALERT"
    REVIEW = "REVIEW"
    URGENT = "URGENT"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"


class Lifecycle(StrEnum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CLOSED = "CLOSED"


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: Box) -> float:
        inter = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0.0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )
        return inter / max(1.0, self.area + other.area - inter)

    def contained_overlap(self, other: Box) -> float:
        inter = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0.0, min(self.y2, other.y2) - max(self.y1, other.y1)
        )
        return inter / max(1.0, min(self.area, other.area))

    def union(self, other: Box) -> Box:
        return Box(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )


@dataclass(frozen=True)
class Frame:
    frame_id: int
    camera_id: str
    capture_ts: float
    width: int
    height: int
    jpeg: bytes
    epoch: int = 0
    monotonic_ts: float = field(default_factory=time.monotonic)
    quality: tuple[str, ...] = ()


class Observation(BaseModel):
    capture_ts: float
    box: Box
    score: float = Field(ge=0, le=1)
    label: Literal["person", "smoke", "fire"]
    track_id: int | None = None
    keypoints: list[tuple[float, float, float]] = Field(default_factory=list)
    limited: bool = False
    truncated: bool = False
    association_uncertain: bool = False
    torso_histogram: tuple[float, ...] = Field(default=(), exclude=True, repr=False)

    @field_validator("torso_histogram")
    @classmethod
    def valid_histogram(cls, values):
        if values and (
            len(values) != 16
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or not math.isclose(sum(values), 1, abs_tol=1e-6)
        ):
            raise ValueError("Torso histogram must contain 16 normalized finite bins")
        return values


class MotionEvidence(BaseModel):
    """Causal geometric observations; never a probability of injury or a claim of intent."""

    classifier_version: str = "causal-motion-v4"
    episode_id: str
    previous_episode_id: str | None = None
    track_id: int
    epoch: int
    started_ts: float
    observed_ts: float
    last_seen_ts: float
    phase: Literal[
        "observing", "following_concern", "ordinary_activity", "recovered_activity", "visibility_lost"
    ] = "observing"
    transition: Literal["none", "fall_like", "slow_descent", "controlled_pattern", "unknown_origin"] = "none"
    posture: Literal["upright", "seated", "low", "uncertain"] = "uncertain"
    onset_observed: bool = False
    identity_uncertain: bool = False
    recovery_seconds: float = 0
    recovery_basis: Literal["upright_activity", "prior_supported_posture"] | None = None
    clear_eligible: bool = False
    features: dict[str, float] = Field(default_factory=dict)


class Candidate(BaseModel):
    candidate_id: str = Field(default_factory=uid)
    category: Category
    reason: str
    first_ts: float
    last_ts: float
    box: Box | None = None
    track_id: int | None = None
    priority: int = 1
    limited: bool = False
    epoch: int = 0
    motion: MotionEvidence | None = None


@dataclass(frozen=True)
class EvidenceImage:
    frame_id: int
    capture_ts: float
    kind: Literal["overview", "crop"]
    jpeg: bytes
    width: int
    height: int


@dataclass(frozen=True)
class EvidencePacket:
    packet_id: str
    incident_id: str
    version: int
    category: Category
    images: tuple[EvidenceImage, ...]
    reason: str
    config_hash: str
    source_ts: float
    created_mono: float = field(default_factory=time.monotonic)
    instruction_version: str = INSTRUCTION_VERSION
    implementation_hash: str = ""
    temporal_context: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "incident_id": self.incident_id,
            "version": self.version,
            "category": self.category,
            "reason": self.reason,
            "config_hash": self.config_hash,
            "source_ts": self.source_ts,
            "instruction_version": self.instruction_version,
            "implementation_hash": self.implementation_hash,
            "temporal_context": self.temporal_context,
            "images": [
                {
                    "frame_id": i.frame_id,
                    "capture_ts": i.capture_ts,
                    "kind": i.kind,
                    "width": i.width,
                    "height": i.height,
                    "bytes": len(i.jpeg),
                }
                for i in self.images
            ],
        }


class DecisionResult(BaseModel):
    incident_id: str
    packet_id: str
    version: int
    medical_status: Status = Status.UNKNOWN
    fire_status: Status = Status.UNKNOWN
    medical_observation: str | None = None
    latest_posture: str | None = None
    policy_reason: str | None = None
    provider: str
    model: str
    request_id: str | None = None
    raw_answer: dict = Field(default_factory=dict)
    input_tokens: int | None = None
    latency_ms: float = 0
    received_at: float = Field(default_factory=time.time)
    error: str | None = None
    retryable: bool = False
    http_status: int | None = None
    provider_error_code: str | None = None
    stale: bool = False


class Incident(BaseModel):
    incident_id: str = Field(default_factory=uid)
    camera_id: str
    category: Category
    reason: str
    first_ts: float
    last_ts: float
    detected_ts: float = 0
    observation_started_ts: float | None = None
    recurrence_count: int = 0
    epoch: int
    box: Box | None = None
    track_ids: list[int] = Field(default_factory=list)
    candidate_refs: list[dict] = Field(default_factory=list)
    identity_uncertain: bool = False
    origin_incident_id: str | None = None
    origin_packet_id: str | None = None
    superseded_by: str | None = None
    priority: int = 1
    limited: bool = False
    lifecycle: Lifecycle = Lifecycle.OPEN
    analysis_status: Status = Status.PENDING
    medical_status: Status = Status.PENDING
    fire_status: Status = Status.PENDING
    medical_observation: str | None = None
    latest_posture: str | None = None
    policy_reason: str | None = None
    feedback: Literal["confirmed", "false_alarm"] | None = None
    feedback_operator: str | None = None
    feedback_at: float | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    assessment_stale: bool = False
    packet_version: int = 0
    request_count: int = 0
    resolution_checks: int = 0
    last_assessed_motion_phase: str | None = None
    confirmation_frames_seen: int = 0
    confirmation_frames_required: int = 0
    observation_seconds_seen: float = 0
    observation_seconds_required: float = 0
    first_assessment_source_ts: float | None = None
    full_assessment_source_ts: float | None = None
    early_screen_complete: bool = False
    queue_reason: str | None = None
    evidence_missing: bool = False
    motion: MotionEvidence | None = None
    ordinary_activity_confirmed: bool = False
    closed_source_ts: float | None = None
