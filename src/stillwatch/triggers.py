from __future__ import annotations

import math
from collections import defaultdict, deque

from .config import Settings
from .motion import MotionClassifier
from .tracking import PoseTracker
from .types import Box, Candidate, Category, Observation


class MedicalRules:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.epoch = -1
        self.history: dict[int, deque[Observation]] = defaultdict(lambda: deque(maxlen=40))
        self.descent_boundary: dict[int, tuple[float, int]] = {}
        self.low_since: dict[int, float] = {}
        self.suspected: dict[int, Candidate] = {}
        self.motion_watch: dict[int, Candidate] = {}
        self.lost_reported: set[int] = set()
        self.motion = MotionClassifier(settings)
        self.tracker = PoseTracker()
        self.observations: list[Observation] = []

    def reset(self):
        self.epoch = -1
        self.history.clear()
        self.descent_boundary.clear()
        self.low_since.clear()
        self.suspected.clear()
        self.motion_watch.clear()
        self.lost_reported.clear()
        self.motion.reset()
        self.tracker.reset()
        self.observations = []

    def _torso(self, obs: Observation) -> tuple[float, float] | None:
        points = obs.keypoints
        if len(points) < 13:
            return None
        shoulders = [points[k] for k in (5, 6) if points[k][2] >= self.settings.keypoint_confidence]
        hips = [points[k] for k in (11, 12) if points[k][2] >= self.settings.keypoint_confidence]
        if not shoulders or not hips:
            return None
        sx, sy = (sum(p[j] for p in shoulders) / len(shoulders) for j in (0, 1))
        hx, hy = (sum(p[j] for p in hips) / len(hips) for j in (0, 1))
        return ((sy + hy) / 2, abs(sx - hx) / max(1, abs(sy - hy)))

    @staticmethod
    def _partial_descent(previous: Observation, current: Observation) -> bool:
        """A visible descent can occur towards the optical axis with cropped legs.

        Require simultaneous box compression and downward movement of BOTH
        confidently observed shoulders. A wide crop or a resized box alone is
        insufficient. This opens a limited episode, never establishes a fall.
        """
        if not (previous.truncated and current.truncated):
            return False
        height = max(1, previous.box.height)
        if current.box.height > height * 0.65 or current.box.y1 - previous.box.y1 < height * 0.3:
            return False
        if abs(current.box.y2 - previous.box.y2) > height * 0.15:
            return False
        if min(len(previous.keypoints), len(current.keypoints)) < 7:
            return False
        shoulder_descent = []
        for k in (5, 6):
            old, new = previous.keypoints[k], current.keypoints[k]
            if min(old[2], new[2]) < 0.65:
                return False
            shoulder_descent.append((new[1] - old[1]) / height)
        # Large consistent movement, or severe foreshortening with a smaller
        # visible shoulder descent. The latter is common in end-on views.
        shoulder_span = math.dist(previous.keypoints[5][:2], previous.keypoints[6][:2])
        span_compression = (
            shoulder_span >= 10
            and math.dist(current.keypoints[5][:2], current.keypoints[6][:2]) <= shoulder_span * 0.6
            and len(current.keypoints) >= 13
            and min(current.keypoints[11][2], current.keypoints[12][2]) >= 0.65
        )
        return min(shoulder_descent) >= 0.2 or (
            current.box.height <= height * 0.55
            and (min(shoulder_descent) >= 0.05 or (min(shoulder_descent) >= 0.02 and span_compression))
        )

    def _rotation_context(self, history) -> bool:
        """Weak rotation-only hints need observed same-person torso context."""
        recent = list(history)[-4:]
        if len(recent) < 4 or recent[-1].capture_ts - recent[0].capture_ts < 0.6 - 1e-6:
            return False
        if any(
            not 0 < b.capture_ts - a.capture_ts <= self.settings.lost_track_seconds + 1e-6
            for a, b in zip(recent, recent[1:], strict=False)
        ):
            return False
        return all(
            not o.association_uncertain
            and self.motion.pose(o).torso_visible
            and self.motion.pose(o).torso_length >= 8
            for o in recent
        )

    def _descent_references(self, tid: int, history: deque[Observation]) -> list[Observation]:
        """Reestablish a motion reference after an ambiguous identity match.

        A single clean-looking pose can still contain another person's joints.
        Keep startup observations available; after ambiguity, require three
        clean updates before a pose can become a relative-descent reference.
        Keep the boundary until track expiry, even if bounded history evicts
        the ambiguous observation. Existing concern and recovery state stays.
        """
        excluded_through = self.descent_boundary.get(tid, (-math.inf, 0))[0]
        return [o for o in history if o.capture_ts > excluded_through]

    def update(self, observations: list[Observation], ts: float, epoch: int) -> list[Candidate]:
        with self.motion.frame_poses():
            return self._update(observations, ts, epoch)

    def _update(self, observations: list[Observation], ts: float, epoch: int) -> list[Candidate]:
        if epoch != self.epoch:
            self.reset()
            self.epoch = epoch
        observations = self.tracker.update(observations, ts, epoch)
        self.observations = observations
        candidates = []
        seen = set()
        normalized = []
        for obs in observations:
            tid = obs.track_id
            normalized.append(obs)
            seen.add(tid)
            history = self.history[tid]
            if obs.association_uncertain:
                history.clear()  # Do not compare two possible identities as one descent.
                self.descent_boundary[tid] = (obs.capture_ts, 0)
            elif tid in self.descent_boundary:
                _, clean_count = self.descent_boundary[tid]
                if clean_count < 2:
                    self.descent_boundary[tid] = (obs.capture_ts, clean_count + 1)
            while history and ts - history[0].capture_ts > 4:
                history.popleft()
            torso = self._torso(obs)
            pose = self.motion.pose(obs)
            low = pose.low
            body_context = pose.body_visible or any(self.motion.pose(h).body_visible for h in history)
            previous_motion = self.motion.tracks.get(tid)
            rotation_context = self._rotation_context(history)
            boundary_translation = bool(history) and self.motion.cropped_translation(history[-1], obs)
            if boundary_translation and not (previous_motion and previous_motion.concern):
                self.motion_watch.pop(tid, None)
            if previous_motion and previous_motion.concern and pose.legs_visible:
                body_context = True
            if not obs.keypoints and not obs.truncated:
                body_context = True  # Box-only perception adapters retain an explicitly limited path.
            if low and not obs.association_uncertain:
                self.low_since.setdefault(tid, ts)
            else:
                self.low_since.pop(tid, None)
            reason = None
            priority = 1
            first_ts = ts
            region = obs.box
            if not obs.limited and (body_context or (low and not obs.truncated) or obs.truncated):
                # Compare source-time observations. Missing keypoints never imply stillness.
                for previous in self._descent_references(tid, history):
                    elapsed = ts - previous.capture_ts
                    if not 0.15 <= elapsed <= self.settings.drop_window_seconds + 1e-6:
                        continue
                    if obs.association_uncertain or previous.association_uncertain:
                        continue
                    if boundary_translation:
                        continue
                    if self.motion.standing_bend(previous, obs):
                        continue
                    if self._partial_descent(previous, obs):
                        reason, priority, first_ts = "partial_body_descent", 2, previous.capture_ts
                        region = previous.box.union(obs.box)
                        break
                    old_torso = self._torso(previous)
                    # Cropped torsos and box resizing do not establish a body descent.
                    if not body_context and (obs.truncated or torso is None or old_torso is None):
                        continue
                    drop, shoulder_drop = self.motion.displacement(previous, obs)
                    if drop is None:
                        continue
                    old_pose = self.motion.pose(previous)
                    forward_collapse = bool(
                        ((previous_motion and previous_motion.concern) or old_pose.body_visible)
                        and pose.shoulder_y is not None
                        and old_pose.shoulder_y is not None
                        and pose.torso_visible
                        and old_pose.torso_visible
                        and old_pose.torso_length >= previous.box.height * 0.2
                        and pose.torso_length < old_pose.torso_length * 0.6
                        and (pose.shoulder_y - old_pose.shoulder_y) / max(1, previous.box.height) > 0.1
                        and obs.box.height < previous.box.height * 0.8
                    )
                    changed = obs.box.height < previous.box.height * 0.75 or low
                    watch_change = obs.box.height < previous.box.height * 0.9 or low
                    leaning_transition = (
                        torso is not None
                        and old_torso is not None
                        and old_torso[1] < 0.7
                        and torso[1] > 1.4
                        and obs.box.height < previous.box.height * 0.85
                    )
                    descending_visible_body = bool(
                        body_context
                        and torso is not None
                        and old_torso is not None
                        and drop >= self.settings.drop_ratio
                        and shoulder_drop is not None
                        and shoulder_drop >= 0.2
                    )
                    if (
                        (
                            drop >= self.settings.pre_drop_ratio
                            and watch_change
                            and (pose.posture != "upright" or not pose.reliable)
                            and (shoulder_drop is None or shoulder_drop >= self.settings.pre_drop_ratio)
                        )
                        or (leaning_transition and (drop >= self.settings.pre_drop_ratio or rotation_context))
                        or descending_visible_body
                    ):
                        self.motion_watch[tid] = Candidate(
                            category=Category.MEDICAL,
                            reason="view_lost_after_suspicion",
                            first_ts=previous.capture_ts,
                            last_ts=ts,
                            box=previous.box.union(obs.box),
                            track_id=tid if tid >= 0 else None,
                            priority=2,
                            limited=True,
                            epoch=epoch,
                        )
                    if (
                        (
                            drop >= self.settings.drop_ratio
                            and changed
                            and (shoulder_drop is None or shoulder_drop >= self.settings.pre_drop_ratio)
                        )
                        or forward_collapse
                        or (leaning_transition and low and drop >= self.settings.pre_drop_ratio)
                    ):
                        reason, priority, first_ts = "rapid_posture_change", 3, previous.capture_ts
                        region = previous.box.union(obs.box)
                        break
                if (
                    reason is None
                    and low
                    and tid in self.low_since
                    and ts - self.low_since[tid] >= self.settings.low_pose_seconds
                ):
                    reason, first_ts = "persistent_low_posture", self.low_since[tid]
                watch = self.motion_watch.get(tid)
                if (
                    reason is None
                    and watch
                    and ts - watch.last_ts <= 1.5
                    and not pose.torso_visible
                    and any(self.motion.pose(h).torso_visible for h in history)
                ):
                    reason, first_ts = "view_lost_after_suspicion", watch.first_ts
            if reason:
                candidate = Candidate(
                    category=Category.MEDICAL,
                    reason=reason,
                    first_ts=first_ts,
                    last_ts=ts,
                    box=region,
                    track_id=tid if tid >= 0 else None,
                    priority=priority,
                    limited=torso is None or tid < 0 or reason == "partial_body_descent",
                    epoch=epoch,
                )
                self.suspected[tid] = candidate
                self.lost_reported.discard(tid)
                candidates.append(candidate)
            history.append(obs)
            watch = self.motion_watch.get(tid)
            if (
                watch
                and not low
                and (
                    ts - watch.last_ts > 1
                    or (pose.reliable and pose.posture == "upright" and ts - watch.last_ts >= 0.2 - 1e-6)
                )
            ):
                self.motion_watch.pop(tid, None)
        for tid, history in list(self.history.items()):
            age = ts - history[-1].capture_ts
            if (
                tid not in seen
                and (tid in self.suspected or tid in self.motion_watch)
                and tid not in self.lost_reported
                and age + 1e-6 >= self.settings.lost_track_seconds
            ):
                old = self.suspected.get(tid) or self.motion_watch[tid]
                candidates.append(
                    old.model_copy(
                        update={
                            "reason": "view_lost_after_suspicion",
                            "last_ts": ts,
                            "limited": True,
                            "box": history[-1].box,
                        }
                    )
                )
                self.lost_reported.add(tid)
            if age > self.settings.dedup_seconds:
                del self.history[tid]
                self.descent_boundary.pop(tid, None)
                self.low_since.pop(tid, None)
                self.suspected.pop(tid, None)
                self.motion_watch.pop(tid, None)
                self.lost_reported.discard(tid)
        return self.motion.update(normalized, ts, epoch, candidates)


class FireRules:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.history: deque[list[Observation]] = deque(maxlen=3)

    def reset(self):
        self.history.clear()

    def update(self, observations: list[Observation], ts: float, epoch: int) -> list[Candidate]:
        candidates = []
        previous = list(self.history)[-2:]
        for obs in observations:
            matched = sum(
                any(p.label == obs.label and p.box.iou(obs.box) >= 0.1 for p in frame) for frame in previous
            )
            if obs.score >= self.settings.fire_immediate_confidence or (
                obs.score >= self.settings.fire_confidence and matched >= 1
            ):
                candidates.append(
                    Candidate(
                        category=Category.FIRE,
                        reason=f"visible_{obs.label}",
                        first_ts=ts,
                        last_ts=ts,
                        box=obs.box,
                        priority=3 if obs.score >= 0.65 else 2,
                        epoch=epoch,
                    )
                )
        self.history.append(observations)
        return candidates


def union_boxes(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result
