from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["PENDING", "NO_ALERT", "REVIEW", "URGENT", "UNKNOWN", "UNAVAILABLE"]


class Record(BaseModel):
    model_config = ConfigDict(extra="allow")


class Motion(Record):
    classifier_version: str
    episode_id: str
    previous_episode_id: str | None = None
    track_id: int
    epoch: int
    phase: str
    transition: str
    posture: str
    started_ts: float
    observed_ts: float
    last_seen_ts: float
    onset_observed: bool = False
    identity_uncertain: bool = False
    clear_eligible: bool = False
    recovery_seconds: float = 0
    recovery_basis: str | None = None


class Event(Record):
    incident_id: str
    camera_id: str
    category: Literal["medical", "fire", "overview"]
    reason: str
    first_ts: float
    last_ts: float
    detected_ts: float
    observation_started_ts: float | None = None
    recurrence_count: int = 0
    epoch: int
    lifecycle: Literal["OPEN", "ACKNOWLEDGED", "CLOSED"]
    analysis_status: Status
    medical_status: Status
    fire_status: Status
    assessment_stale: bool = False
    superseded_by: str | None = None
    medical_observation: str | None = None
    policy_reason: str | None = None
    feedback: str | None = None
    motion: Motion | None = None
    ordinary_activity_confirmed: bool = False


class EventPage(Record):
    events: list[Event]
    total: int


class EventDetail(Record):
    event: Event
    packets: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    feedback_history: list[dict[str, Any]] = Field(default_factory=list)


class SourceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["none", "file", "camera", "rtsp"]
    file_id: str | None = None
    path: str | None = None
    camera_index: int = Field(default=0, ge=0, le=31)
    url: str | None = Field(default=None, repr=False)
    loop: bool = False


class Source(Record):
    kind: Literal["none", "file", "camera", "rtsp"]
    label: str
    state: str
    epoch: int
    loop: bool = False
    file_id: str | None = None


class Video(Record):
    file_id: str
    name: str
    bytes: int
