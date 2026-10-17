from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .buffer import FrameBuffer, evenly_spaced
from .config import Settings, implementation_fingerprint
from .types import Box, Category, EvidenceImage, EvidencePacket, Frame, Incident, Observation, uid


class EvidenceBuilder:
    def __init__(self, settings: Settings, buffer: FrameBuffer):
        self.settings = settings
        self.buffer = buffer
        self.onset: dict[str, list[Frame]] = {}
        self.onset_max_bytes = 16 * 1024**2
        self.regions: dict[int, deque] = {}
        self.onset_regions: dict[str, dict[int, dict]] = {}
        self.region_epoch = -1

    def observe(self, observations: list[Observation | dict], ts: float, epoch: int):
        """Retain only observed per-person locations, with bounded source-time history."""
        if epoch != self.region_epoch:
            self.regions.clear()
            self.region_epoch = epoch
        for value in observations:
            obs = value if isinstance(value, Observation) else Observation.model_validate(value)
            if obs.label != "person" or obs.track_id is None or obs.capture_ts > ts:
                continue
            history = self.regions.setdefault(obs.track_id, deque(maxlen=600))
            if history and obs.capture_ts <= history[-1][0]:
                continue
            history.append((obs.capture_ts, obs.box.model_copy(), obs.association_uncertain, obs.limited))
        for tid, history in list(self.regions.items()):
            while history and ts - history[0][0] > self.settings.buffer_seconds:
                history.popleft()
            if not history:
                self.regions.pop(tid)
        while len(self.regions) > 128:
            self.regions.pop(min(self.regions, key=lambda tid: self.regions[tid][-1][0]))

    def _region(self, incident: Incident, frame: Frame) -> dict | None:
        pinned = self.onset_regions.get(incident.incident_id, {}).get(frame.frame_id)
        if pinned:
            return pinned
        if frame.epoch != self.region_epoch or len(incident.track_ids) != 1 or incident.identity_uncertain:
            return None
        tid = incident.track_ids[0]
        # Do not interpolate across disappearance or use a later pose to label an
        # earlier image. A region is an observed location, not identity proof.
        max_age = min(self.settings.lost_track_seconds, 1.5 / self.settings.pose_fps)
        for ts, box, uncertain, limited in reversed(self.regions.get(tid, [])):
            if ts > frame.capture_ts:
                continue
            if frame.capture_ts - ts > max_age + 1e-6:
                return None
            return {
                "frame_id": frame.frame_id,
                "frame_source_ts": frame.capture_ts,
                "observed_source_ts": ts,
                "track_id": tid,
                "box": box.model_dump(),
                "association_uncertain": uncertain,
                "limited": limited,
            }
        return None

    def pin(self, incident: Incident):
        if incident.lifecycle == "CLOSED" or incident.superseded_by:
            return
        available = [
            f
            for f in self.buffer.frames(incident.first_ts - 1.2)
            if f.epoch == incident.epoch and f.capture_ts <= incident.detected_ts + 1.2
        ]
        previous = self.onset.get(incident.incident_id, [])
        available = sorted({f.frame_id: f for f in previous + available}.values(), key=lambda f: f.capture_ts)
        if incident.recurrence_count:
            boundary = incident.first_ts - 1.2
            selected = evenly_spaced([f for f in available if f.capture_ts < boundary], 2) + evenly_spaced(
                [f for f in available if f.capture_ts >= boundary], 22
            )
        else:
            selected = evenly_spaced(available, 24)
        used = sum(
            len(f.jpeg) for key, frames in self.onset.items() if key != incident.incident_id for f in frames
        )
        if used + sum(len(f.jpeg) for f in selected) <= self.onset_max_bytes:
            regions = {f.frame_id: region for f in selected if (region := self._region(incident, f))}
            self.onset[incident.incident_id] = selected
            self.onset_regions[incident.incident_id] = regions
        else:
            incident.evidence_missing = True

    def release(self, incident_id: str):
        self.onset.pop(incident_id, None)
        self.onset_regions.pop(incident_id, None)

    def handoff(self, previous: Incident, current: Incident):
        """Transfer bounded visual context after a proven same-person recurrence."""
        anchor = current.observation_started_ts
        prior = evenly_spaced(
            [
                f
                for f in self.onset.get(previous.incident_id, [])
                if anchor is not None and f.capture_ts <= anchor + 1.2
            ],
            2,
        )
        regions = self.onset_regions.get(previous.incident_id, {}).copy()
        self.release(previous.incident_id)
        self.pin(current)
        own = self.onset.get(current.incident_id, [])
        selected = sorted(
            {f.frame_id: f for f in prior + evenly_spaced(own, 22)}.values(), key=lambda f: f.capture_ts
        )
        used = sum(
            len(f.jpeg) for key, frames in self.onset.items() if key != current.incident_id for f in frames
        )
        if used + sum(len(f.jpeg) for f in selected) <= self.onset_max_bytes:
            self.onset[current.incident_id] = selected
            self.onset_regions.setdefault(current.incident_id, {}).update(
                {f.frame_id: regions[f.frame_id] for f in prior if f.frame_id in regions}
            )
        else:
            current.evidence_missing = True

    def build(self, incident: Incident, source_ts: float) -> EvidencePacket:
        temporal_medical = (
            incident.category == Category.MEDICAL and self.settings.medical_observation_seconds > 0
        )
        since = (
            incident.detected_ts - self.settings.evidence_seconds
            if temporal_medical
            else source_ts - self.settings.evidence_seconds
        )
        frames = self.onset.get(incident.incident_id, []) + self.buffer.frames(since)
        frames = sorted(
            {
                f.frame_id: f for f in frames if f.epoch == incident.epoch and f.capture_ts <= source_ts
            }.values(),
            key=lambda f: f.capture_ts,
        )
        full_count = (
            4
            if incident.category == Category.OVERVIEW
            else (2 if incident.category == Category.MEDICAL else 6)
        )
        crop_count = 6 if incident.category == Category.MEDICAL else 2
        if incident.limited or incident.box is None:
            full_count = 4 if incident.category == Category.OVERVIEW else 8
            crop_count = 0
        if temporal_medical:
            # Recovery can move the person outside the original fall region.
            # Keep onset, event development and several recent wide views.
            preceding = (
                evenly_spaced([f for f in frames if f.capture_ts < incident.first_ts - 1.2], 2)
                if incident.recurrence_count
                else []
            )
            onset = evenly_spaced(
                [f for f in frames if incident.first_ts - 1.2 <= f.capture_ts <= incident.detected_ts + 1.2],
                4 if preceding else 6,
            )
            development = evenly_spaced([f for f in frames if f.capture_ts > incident.detected_ts + 1.2], 2)
            recent = evenly_spaced([f for f in frames if f.capture_ts >= source_ts - 3], 4)
            selected = {f.frame_id: f for f in preceding + onset + development + recent}
            for frame in evenly_spaced(frames, 12):
                if len(selected) >= 12:
                    break
                selected.setdefault(frame.frame_id, frame)
            frames = sorted(selected.values(), key=lambda f: f.capture_ts)
            full_count, crop_count = 12, 0
        images = [
            EvidenceImage(f.frame_id, f.capture_ts, "overview", f.jpeg, f.width, f.height)
            for f in evenly_spaced(frames, full_count)
        ]
        target_regions = [region for f in frames if (region := self._region(incident, f))]
        if temporal_medical:
            by_id = {r["frame_id"]: r for r in target_regions}
            # Give distant people their own view without removing scene context.
            # Use two onset and two recent locations, not a union of the person's
            # whole trajectory (which can include an unrelated upright person).
            eligible = [
                f
                for f in frames
                if (region := by_id.get(f.frame_id))
                and not region["association_uncertain"]
                and Box.model_validate(region["box"]).area < f.width * f.height * 0.2
            ]
            closeups = (
                evenly_spaced([f for f in eligible if f.capture_ts <= incident.detected_ts + 1.2], 2)
                + eligible[-2:]
            )
            for frame in {f.frame_id: f for f in closeups}.values():
                box = Box.model_validate(by_id[frame.frame_id]["box"])
                image = self._crop(frame, box)
                if image is not None:
                    images.append(image)
        box = incident.box
        if box is not None and crop_count and incident.category != Category.OVERVIEW:
            for frame in evenly_spaced(frames, crop_count):
                image = self._crop(frame, box)
                if image is not None:
                    images.append(image)
        images.sort(key=lambda i: (i.capture_ts, i.kind == "crop"))
        return EvidencePacket(
            uid(),
            incident.incident_id,
            incident.packet_version,
            incident.category,
            tuple(images),
            incident.reason,
            self.settings.fingerprint,
            source_ts,
            implementation_hash=implementation_fingerprint(),
            temporal_context={
                "detected_source_ts": incident.detected_ts,
                "observation_started_source_ts": (
                    incident.detected_ts
                    if incident.observation_started_ts is None
                    else incident.observation_started_ts
                ),
                "recurrence_count": incident.recurrence_count,
                "decision_source_ts": source_ts,
                "elapsed_source_seconds": max(
                    0,
                    source_ts
                    - (
                        incident.detected_ts
                        if incident.observation_started_ts is None
                        else incident.observation_started_ts
                    ),
                ),
                "current_episode_elapsed_seconds": max(0, source_ts - incident.detected_ts),
                "required_observation_seconds": self.settings.medical_observation_seconds,
                "local_view_limited": incident.limited,
                # Incident grouping and the current track can lose identity
                # independently. Present their combined uncertainty to the model.
                "identity_uncertain": incident.identity_uncertain
                or (incident.motion is not None and incident.motion.identity_uncertain),
                "local_motion": incident.motion.model_dump(mode="json") if incident.motion else None,
                "target_regions": target_regions,
            }
            if temporal_medical
            else {},
        )

    def _crop(self, frame: Frame, box: Box) -> EvidenceImage | None:
        pad_x, pad_y = box.width * 0.3, box.height * 0.3
        x1, y1 = max(0, int(box.x1 - pad_x)), max(0, int(box.y1 - pad_y))
        x2, y2 = min(frame.width, int(box.x2 + pad_x)), min(frame.height, int(box.y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None
        image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        ok, encoded = cv2.imencode(
            ".jpg", image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
        )
        if not ok:
            return None
        return EvidenceImage(frame.frame_id, frame.capture_ts, "crop", encoded.tobytes(), x2 - x1, y2 - y1)
