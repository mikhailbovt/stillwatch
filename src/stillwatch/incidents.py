from __future__ import annotations

import time
from collections import OrderedDict
from itertools import combinations

from .config import Settings
from .evidence import EvidenceBuilder
from .storage import Storage
from .types import Candidate, Category, DecisionResult, Incident, Lifecycle, MotionEvidence, Status


class IncidentManager:
    max_visible_pairs = 8192

    def __init__(self, settings: Settings, storage: Storage, evidence: EvidenceBuilder):
        self.settings = settings
        self.storage = storage
        self.evidence = evidence
        self.items = {i.incident_id: i for i in storage.incidents()}
        self.latest_motion: dict[int, MotionEvidence] = {}
        self.visible_pairs: OrderedDict[tuple[int, int, int], float] = OrderedDict()
        for item in self.items.values():
            if item.superseded_by:
                continue
            if item.lifecycle != Lifecycle.CLOSED:
                item.assessment_stale = True
                item.queue_reason = "restart_requires_fresh_evidence"
                self.storage.save_incident(item)

    def ingest(self, candidate: Candidate) -> tuple[Incident, bool]:
        matches = []
        for item in sorted(self.items.values(), key=lambda i: i.created_at, reverse=True):
            if (
                item.category == candidate.category == Category.MEDICAL
                and item.epoch == candidate.epoch
                and candidate.track_id is not None
                and candidate.track_id not in item.track_ids
                and any(
                    (candidate.epoch, *sorted((candidate.track_id, track_id))) in self.visible_pairs
                    for track_id in item.track_ids
                )
            ):
                # People observed together cannot be fragments of one track.
                # Losing both behind the same object does not join their episodes.
                continue
            if (
                candidate.motion
                and item.motion
                and candidate.motion.episode_id == item.motion.episode_id
                and item.epoch == candidate.epoch
            ):
                if (
                    item.lifecycle == Lifecycle.CLOSED
                    and candidate.reason in ("rapid_posture_change", "partial_body_descent")
                    and candidate.first_ts
                    > (item.closed_source_ts if item.closed_source_ts is not None else item.last_ts)
                ):
                    continue
                if item.lifecycle == Lifecycle.CLOSED:
                    return item, False
                # One uninterrupted activity keeps its episode, including an explicit operator closure.
                item.motion = candidate.motion.model_copy(deep=True)
                item.last_ts = candidate.last_ts
                item.priority = max(item.priority, candidate.priority)
                item.limited = item.limited or candidate.limited
                if item.box and candidate.box:
                    item.box = item.box.union(candidate.box)
                if candidate.reason == "view_lost_after_suspicion":
                    item.reason = candidate.reason
                item.updated_at = time.time()
                self.evidence.pin(item)
                self.storage.save_incident(item)
                return item, False
            if (
                item.lifecycle == Lifecycle.CLOSED
                or item.category != candidate.category
                or item.epoch != candidate.epoch
            ):
                continue
            if candidate.motion and item.motion and candidate.motion.episode_id != item.motion.episode_id:
                possible_fragment = (
                    candidate.track_id is not None
                    and candidate.track_id not in item.track_ids
                    and item.motion.phase == "visibility_lost"
                    and 0 <= candidate.last_ts - item.motion.last_seen_ts <= 3
                    and item.box is not None
                    and candidate.box is not None
                    and item.box.iou(candidate.box) >= self.settings.dedup_iou
                )
                if not possible_fragment:
                    continue
            observing_medical = (
                item.category == Category.MEDICAL
                and item.request_count == 0
                and candidate.last_ts - item.detected_ts <= self.settings.medical_observation_seconds
            )
            if candidate.last_ts - item.last_ts > self.settings.dedup_seconds and not observing_medical:
                continue
            if (
                item.category == Category.MEDICAL
                and item.request_count > 0
                and candidate.reason in ("rapid_posture_change", "partial_body_descent")
                and candidate.first_ts > item.last_ts
            ):
                # A disjoint new transition must not inherit an older assessment
                # or its exhausted request cap, even for the same tracked person.
                continue
            overlap = item.box.iou(candidate.box) if item.box and candidate.box else 0
            same_track = candidate.track_id is not None and candidate.track_id in item.track_ids
            # Two simultaneously visible, distinct tracks must retain separate incidents.
            ambiguous_id_change = (
                candidate.track_id is not None
                and item.track_ids
                and not same_track
                and candidate.first_ts <= item.last_ts + self.settings.lost_track_seconds
            )
            possible_reappearance = (
                item.category == Category.MEDICAL
                and item.reason == "view_lost_after_suspicion"
                and candidate.first_ts >= item.last_ts - self.settings.lost_track_seconds
                and overlap >= self.settings.dedup_iou * 0.8
            )
            same_fire_region = (
                item.category == Category.FIRE
                and item.box is not None
                and candidate.box is not None
                and item.box.contained_overlap(candidate.box) >= 0.5
            )
            if (
                (same_track and overlap > 0.05)
                or (overlap >= self.settings.dedup_iou and not ambiguous_id_change)
                or possible_reappearance
                or same_fire_region
            ):
                matches.append((overlap + int(same_track), item))
        if matches:
            same_person = [
                pair
                for pair in matches
                if candidate.category == Category.MEDICAL
                and candidate.track_id is not None
                and candidate.track_id in pair[1].track_ids
            ]
            # Once a new transition opens an episode, its continued low posture
            # belongs to that episode rather than an older overlapping region.
            item = (
                max(same_person, key=lambda m: (m[1].detected_ts, m[0]))[1]
                if same_person
                else max(matches, key=lambda m: m[0])[1]
            )
            item.last_ts = candidate.last_ts
            item.priority = max(item.priority, candidate.priority)
            item.limited = item.limited or candidate.limited
            item.reason = candidate.reason if candidate.reason == "view_lost_after_suspicion" else item.reason
            if item.box and candidate.box:
                item.box = item.box.union(candidate.box)
            if candidate.motion:
                item.motion = candidate.motion.model_copy(deep=True)
            if candidate.track_id is not None and candidate.track_id not in item.track_ids:
                item.track_ids.append(candidate.track_id)
                item.identity_uncertain = True
                item.limited = True
                item.candidate_refs.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "track_id": candidate.track_id,
                        "reason": candidate.reason,
                        "first_ts": candidate.first_ts,
                    }
                )
            item.updated_at = time.time()
            self.storage.save_incident(item)
            return item, False
        item = Incident(
            camera_id=self.settings.camera_id,
            category=candidate.category,
            reason=candidate.reason,
            first_ts=candidate.first_ts,
            last_ts=candidate.last_ts,
            detected_ts=candidate.last_ts,
            box=candidate.box,
            track_ids=[candidate.track_id] if candidate.track_id is not None else [],
            candidate_refs=[
                {
                    "candidate_id": candidate.candidate_id,
                    "track_id": candidate.track_id,
                    "reason": candidate.reason,
                    "first_ts": candidate.first_ts,
                }
            ],
            priority=candidate.priority,
            limited=candidate.limited,
            epoch=candidate.epoch,
            motion=candidate.motion,
        )
        self.items[item.incident_id] = item
        self._supersede_previous(item)
        self.evidence.pin(item)
        self.storage.save_incident(item)
        return item, True

    def _supersede_previous(self, current: Incident):
        motion = current.motion
        if (
            current.category != Category.MEDICAL
            or motion is None
            or not motion.previous_episode_id
            or current.identity_uncertain
            or motion.identity_uncertain
        ):
            return
        previous = next(
            (
                item
                for item in self.items.values()
                if item.incident_id != current.incident_id
                and item.category == Category.MEDICAL
                and item.epoch == current.epoch
                and item.motion
                and item.motion.episode_id == motion.previous_episode_id
                and motion.started_ts > item.motion.started_ts
                and current.detected_ts > item.detected_ts
                and item.track_ids == current.track_ids == [motion.track_id]
                and not (item.identity_uncertain or item.motion.identity_uncertain)
                and item.superseded_by is None
            ),
            None,
        )
        if previous is None or previous.lifecycle == Lifecycle.CLOSED:
            return
        # A locally observed rise is not a confirmed resolution. Keep the first
        # unresolved concern's clock through subsequent, identity-linked descents.
        current.observation_started_ts = (
            previous.detected_ts
            if previous.observation_started_ts is None
            else min(previous.detected_ts, previous.observation_started_ts)
        )
        current.recurrence_count = previous.recurrence_count + 1
        # Persist the successor before storing a link to it. Release the old
        # pinned frames before admitting the new onset into the bounded pool.
        self.storage.save_incident(current)
        # Preserve locally observed context even if no cloud request was sent.
        # This is an archive snapshot, not a decision or an observation-window bypass.
        previous.packet_version += 1
        packet = self.evidence.build(previous, current.detected_ts)
        packet.temporal_context["archived_without_assessment"] = True
        packet.temporal_context["superseded_by"] = current.incident_id
        previous.evidence_missing = not packet.images or not self.storage.save_packet(packet)
        previous.superseded_by = current.incident_id
        previous.queue_reason = "superseded_by_new_episode"
        # Established urgent alerts remain visible until an operator handles them.
        # Other earlier outcomes remain in history and are never relabelled NO_ALERT.
        if previous.analysis_status != Status.URGENT:
            previous.assessment_stale = True
            if previous.analysis_status == Status.PENDING:
                previous.medical_status = previous.analysis_status = Status.UNKNOWN
        previous.updated_at = time.time()
        self.storage.save_incident(previous)
        self.evidence.handoff(previous, current)

    def observe_motion(self, observations: list[MotionEvidence], source_ts: float):
        retention = max(self.settings.dedup_seconds, self.settings.medical_observation_seconds) + 3
        self.visible_pairs = OrderedDict(
            (pair, ts) for pair, ts in self.visible_pairs.items() if 0 <= source_ts - ts <= retention
        )
        visible = [
            m
            for m in observations
            if m.track_id >= 0
            and m.phase != "visibility_lost"
            and abs(m.last_seen_ts - source_ts) < 1e-6
            and abs(m.observed_ts - source_ts) < 1e-6
        ]
        for first, second in combinations(visible, 2):
            if first.epoch != second.epoch or first.track_id == second.track_id:
                continue
            pair = (first.epoch, *sorted((first.track_id, second.track_id)))
            self.visible_pairs[pair] = source_ts
            self.visible_pairs.move_to_end(pair)
        while len(self.visible_pairs) > self.max_visible_pairs:
            self.visible_pairs.popitem(last=False)
        self.latest_motion = {m.track_id: m for m in observations}
        by_episode = {m.episode_id: m for m in observations}
        for item in self.items.values():
            if item.motion is None or item.lifecycle == Lifecycle.CLOSED or item.superseded_by:
                continue
            motion = by_episode.get(item.motion.episode_id)
            if motion is None or motion.epoch != item.epoch:
                continue
            changed = item.motion.phase != motion.phase or item.motion.clear_eligible != motion.clear_eligible
            item.motion = motion.model_copy(deep=True)
            if motion.phase == "ordinary_activity" and item.ordinary_activity_confirmed:
                item.last_ts = max(item.last_ts, motion.last_seen_ts)
            if not motion.clear_eligible and item.ordinary_activity_confirmed:
                item.ordinary_activity_confirmed = False
                item.medical_status = item.analysis_status = Status.UNKNOWN
                item.policy_reason = "ordinary_activity_context_lost"
                changed = True
            if motion.phase == "visibility_lost" and item.analysis_status == Status.NO_ALERT:
                item.medical_status = item.analysis_status = Status.UNKNOWN
                item.policy_reason = "local_visibility_lost"
                changed = True
            self.evidence.pin(item)
            if changed or source_ts - item.motion.started_ts < 2:
                self.storage.save_incident(item)

    def _clearance_guard(self, item: Incident, result: DecisionResult, source_ts: float):
        if item.category != Category.MEDICAL or result.medical_status != Status.NO_ALERT:
            return
        motion = item.motion
        reason = None
        if item.identity_uncertain or (motion and motion.identity_uncertain):
            reason = "local_identity_uncertain"
        elif motion is None:
            reason = "local_motion_unavailable"
        elif motion.epoch != item.epoch or max(motion.last_seen_ts, motion.observed_ts) > source_ts + 1e-6:
            reason = "local_motion_source_mismatch"
        elif source_ts - motion.last_seen_ts > self.settings.lost_track_seconds + 1e-6:
            reason = "local_visibility_lost"
        elif any(
            current.epoch == item.epoch
            and current.track_id in item.track_ids
            and current.episode_id != motion.episode_id
            and not current.clear_eligible
            for current in self.latest_motion.values()
        ):
            reason = "new_transition_after_assessment"
        elif not motion.clear_eligible:
            reason = "local_episode_unresolved"
        elif (
            motion.phase == "recovered_activity"
            and result.latest_posture != "upright"
            and not (
                result.latest_posture == "seated_on_furniture"
                and motion.recovery_basis == "prior_supported_posture"
            )
        ):
            reason = "recovery_requires_current_posture_agreement"
        elif result.medical_observation == "recovered" and motion.phase != "recovered_activity":
            reason = "recovery_not_supported_by_motion"
        if reason:
            result.medical_status = Status.UNKNOWN
            result.policy_reason = reason

    def overview(self, source_ts: float, epoch: int) -> Incident:
        item = Incident(
            camera_id=self.settings.camera_id,
            category=Category.OVERVIEW,
            reason="scheduled_scene_overview",
            first_ts=source_ts,
            last_ts=source_ts,
            detected_ts=source_ts,
            epoch=epoch,
            priority=0,
        )
        self.items[item.incident_id] = item
        self.storage.save_incident(item)
        return item

    def apply(self, result: DecisionResult, source_ts: float):
        item = self.items.get(result.incident_id)
        if item is None:
            return
        result.stale = (
            result.stale
            or item.lifecycle == Lifecycle.CLOSED
            or result.version != item.packet_version
            or item.superseded_by is not None
        )
        if result.stale:
            return
        reviewed = item.analysis_status == Status.REVIEW
        self._clearance_guard(item, result, source_ts)
        if item.medical_status != Status.URGENT:
            item.medical_status = result.medical_status
        if item.fire_status != Status.URGENT:
            item.fire_status = result.fire_status
        item.medical_observation = result.medical_observation
        item.latest_posture = result.latest_posture
        item.policy_reason = result.policy_reason
        item.ordinary_activity_confirmed = bool(
            item.category == Category.MEDICAL
            and item.motion
            and item.motion.phase == "ordinary_activity"
            and result.medical_status == Status.NO_ALERT
            and result.medical_observation == "no_concern"
        )
        if item.category == Category.MEDICAL:
            item.analysis_status = item.medical_status
        elif item.category == Category.FIRE:
            item.analysis_status = item.fire_status
        else:
            rank = {
                Status.NO_ALERT: 0,
                Status.PENDING: 1,
                Status.UNKNOWN: 2,
                Status.UNAVAILABLE: 3,
                Status.REVIEW: 4,
                Status.URGENT: 5,
            }
            item.analysis_status = max((item.medical_status, item.fire_status), key=rank.__getitem__)
        if (
            reviewed
            and item.category != Category.OVERVIEW
            and item.analysis_status in (Status.PENDING, Status.UNKNOWN, Status.UNAVAILABLE)
        ):
            # Preserve operator attention, while retaining the latest answer separately.
            item.analysis_status = Status.REVIEW
            item.policy_reason = "review_pending_resolution"
        item.assessment_stale = False
        item.queue_reason = result.error
        item.updated_at = time.time()
        self.storage.save_incident(item)
        if (
            item.category == Category.MEDICAL
            and item.lifecycle == Lifecycle.OPEN
            and item.analysis_status == Status.NO_ALERT
            and item.feedback is None
            and item.motion
            and item.motion.phase == "recovered_activity"
            and item.motion.clear_eligible
            and not (item.identity_uncertain or item.motion.identity_uncertain)
            and item.medical_observation in ("recovered", "no_concern")
            and (item.full_assessment_source_ts is not None or self.settings.medical_observation_seconds == 0)
        ):
            # A completed, jointly confirmed recovery ends this episode now.
            # Leaving view afterwards cannot reopen it; a new concern retains
            # its own episode and assessment allowance.
            self._close(item, source_ts)
        # Positive evidence can identify the other direction as well. It never
        # clears unseen parts of the scene with a negative answer.
        for category, status in (
            (Category.MEDICAL, result.medical_status),
            (Category.FIRE, result.fire_status),
        ):
            if category != item.category and status in (Status.REVIEW, Status.URGENT):
                existing = next(
                    (
                        i
                        for i in self.items.values()
                        if i.origin_incident_id == item.incident_id
                        and i.category == category
                        and i.lifecycle != Lifecycle.CLOSED
                    ),
                    None,
                )
                candidate = Candidate(
                    category=category,
                    reason="overview_detected_concern"
                    if item.category == Category.OVERVIEW
                    else "cross_route_evidence_concern",
                    first_ts=source_ts,
                    last_ts=source_ts,
                    limited=True,
                    epoch=item.epoch,
                    priority=2,
                )
                found = existing or self.ingest(candidate)[0]
                if (
                    category == Category.MEDICAL
                    and self.settings.medical_observation_seconds > 0
                    and result.medical_observation != "immediate_danger"
                    and found.request_count == 0
                ):
                    # Overview/cross-route concerns enter the same observation gate.
                    # Keep their original answer in history without bypassing the delay.
                    found.origin_incident_id = item.incident_id
                    found.origin_packet_id = result.packet_id
                    self.storage.save_incident(found)
                    if item.category == Category.OVERVIEW and item.fire_status == Status.NO_ALERT:
                        # The overview has handed off an unconfirmed medical concern.
                        # Keep the raw answer in history without a duplicate early alert.
                        item.analysis_status = Status.PENDING
                        item.queue_reason = "medical_observation_started"
                        item.lifecycle = Lifecycle.CLOSED
                        self.storage.save_incident(item)
                    continue
                found.analysis_status = status
                found.medical_status = result.medical_status
                found.fire_status = result.fire_status
                found.medical_observation = result.medical_observation
                found.latest_posture = result.latest_posture
                found.policy_reason = result.policy_reason
                found.request_count = 1
                found.evidence_missing = item.evidence_missing
                found.origin_incident_id = item.incident_id
                found.origin_packet_id = result.packet_id
                self.storage.save_incident(found)

    def detail(self, incident_id: str) -> dict:
        item = self.items[incident_id]
        detail = self.storage.detail(incident_id)
        if item.origin_incident_id and not detail["packets"]:
            origin = self.storage.detail(item.origin_incident_id)
            detail["packets"] = [p for p in origin["packets"] if p["packet_id"] == item.origin_packet_id]
            detail["assessments"] = [
                r for r in origin["assessments"] if r["packet_id"] == item.origin_packet_id
            ]
            detail["evidence_origin"] = item.origin_incident_id
        return detail

    def _close(self, item: Incident, source_ts: float):
        item.lifecycle = Lifecycle.CLOSED
        item.closed_source_ts = max(item.last_ts, source_ts)
        self.evidence.release(item.incident_id)
        item.updated_at = time.time()
        self.storage.save_incident(item)

    def annotate(self, incident_id: str, action: str):
        item = self.items[incident_id]
        if action == "close":
            self._close(
                item,
                max(
                    [item.last_ts]
                    + [
                        m.observed_ts
                        for m in self.latest_motion.values()
                        if m.epoch == item.epoch and m.track_id in item.track_ids
                    ]
                ),
            )
        elif action in ("confirmed", "false_alarm"):
            item.feedback = action
            item.feedback_operator = self.settings.operator_id
            item.feedback_at = time.time()
            if item.lifecycle == Lifecycle.OPEN:
                item.lifecycle = Lifecycle.ACKNOWLEDGED
        else:
            raise ValueError("Unsupported operator action")
        item.updated_at = time.time()
        self.storage.feedback(incident_id, self.settings.operator_id, action)
        self.storage.save_incident(item)

    def archive_clear(self, source_ts: float, epoch: int):
        for item in self.items.values():
            if (
                item.epoch == epoch
                and item.lifecycle == Lifecycle.OPEN
                and item.analysis_status == Status.NO_ALERT
                and not item.assessment_stale
                and item.feedback is None
                and not item.ordinary_activity_confirmed
                and source_ts - item.last_ts >= 5
            ):
                self._close(item, source_ts)
