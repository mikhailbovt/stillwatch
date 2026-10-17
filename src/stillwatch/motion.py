"""Bounded causal motion classifier, independent of cloud answers.

Geometric labels are deliberately conservative. Slow movement alone never
establishes an ordinary descent. No biometric identity or clinical inference.
"""

from __future__ import annotations

import math
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import median

from .config import Settings
from .types import Candidate, Category, MotionEvidence, Observation, uid


@dataclass
class Pose:
    observation: Observation
    center_y: float
    lean: float | None
    low: bool
    posture: str
    knees_bent: bool
    reliable: bool
    torso_visible: bool
    legs_visible: bool
    body_visible: bool
    shoulder_y: float | None
    torso_length: float

    @property
    def ts(self):
        return self.observation.capture_ts

    @property
    def height(self):
        return max(1, self.observation.box.height)


@dataclass
class Track:
    evidence: MotionEvidence
    history: deque[Pose] = field(default_factory=lambda: deque(maxlen=120))
    baseline: Pose | None = None
    baseline_since: float | None = None
    recovery_since: float | None = None
    last_recovery_ts: float | None = None
    low_since: float | None = None
    concern: bool = False
    emitted: bool = False
    transition_end: float | None = None
    support: Pose | None = None
    last_low_ts: float | None = None
    unresolved_landing: bool = False
    activity_resumed: bool = False
    recovery_observed: bool = False
    reobserved_since: float | None = None
    reobserved_pose: Pose | None = None


class MotionClassifier:
    max_tracks = 128

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tracks: dict[int, Track] = {}
        self.epoch = -1
        self._pose_cache: dict[int, Pose] | None = None

    @contextmanager
    def frame_poses(self):
        """Reuse geometry of read-only observations during one rules update."""
        previous = self._pose_cache
        self._pose_cache = {}
        try:
            yield
        finally:
            self._pose_cache = previous

    def reset(self):
        self.tracks.clear()
        self.epoch = -1

    def displacement(self, before: Observation, after: Observation) -> tuple[float | None, float | None]:
        """Compare the same visible joints; changing visibility must not move a centroid."""
        offsets = []
        shared_count = 0
        for group in ((5, 6), (11, 12)):
            shared = [
                after.keypoints[k][1] - before.keypoints[k][1]
                for k in group
                if min(len(before.keypoints), len(after.keypoints)) > k
                and min(before.keypoints[k][2], after.keypoints[k][2]) >= self.settings.keypoint_confidence
            ]
            shared_count += len(shared)
            offsets.append(sum(shared) / len(shared) / max(1, before.box.height) if shared else None)
        shoulder, hip = offsets
        # An isolated confident point can jump to a different body part during
        # occlusion. It cannot establish descent without another observed joint.
        if shoulder is not None and shared_count >= 2:
            return ((shoulder + hip) / 2 if hip is not None else shoulder), shoulder
        if not before.keypoints and not after.keypoints:
            delta = (after.box.y1 + after.box.y2 - before.box.y1 - before.box.y2) / 2
            return delta / max(1, before.box.height), None
        return None, None

    def cropped_translation(self, before: Observation, after: Observation) -> bool:
        return self._cropped_shape_translation(before, after) or self._continued_translation(before, after)

    def standing_bend(self, before: Observation, after: Observation) -> bool:
        """Recognize retained leg support during upper-body compression.

        A previously upright person can bend while the pelvis stays above an
        extended leg and a planted foot. This withholds a new descent from that
        pair of observations; it never clears a concern or establishes intent.
        """
        if (
            before.association_uncertain
            or after.association_uncertain
            or not 0 < after.capture_ts - before.capture_ts <= 5
        ):
            return False
        old, new = self.pose(before), self.pose(after)
        if old.posture != "upright" or not old.reliable or not new.torso_visible:
            return False
        height = max(1, before.box.height)
        for chain in ((11, 13, 15), (12, 14, 16)):
            if any(len(o.keypoints) <= k or o.keypoints[k][2] < 0.65 for o in (before, after) for k in chain):
                continue
            # Bending moves the pelvis backward. Its vertical support and the
            # foot must remain observed; a rigid whole leg is not required.
            if any(abs(before.keypoints[k][1] - after.keypoints[k][1]) > height * 0.08 for k in chain):
                continue
            if math.dist(before.keypoints[chain[-1]][:2], after.keypoints[chain[-1]][:2]) > height * 0.08:
                continue
            extended = True
            for observation in (before, after):
                a, b, c = (observation.keypoints[k][:2] for k in chain)
                u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
                lengths = math.hypot(*u), math.hypot(*v)
                if min(lengths) < height * 0.1:
                    extended = False
                    break
                cosine = sum(x * y for x, y in zip(u, v, strict=True)) / (lengths[0] * lengths[1])
                if cosine > math.cos(math.radians(150)) or c[1] - a[1] < max(
                    height * 0.35, sum(lengths) * 0.85
                ):
                    extended = False
                    break
            if extended:
                return True
        return False

    def _cropped_shape_translation(self, before: Observation, after: Observation) -> bool:
        """Recognize unchanged visible body geometry moving through a frame edge.

        A clipped detection box shrinks when a walking person leaves the image.
        That is not observed body compression. This only vetoes a new box-based
        onset; it cannot clear an existing concern or establish recovery.
        """
        track = self.tracks.get(before.track_id)
        recently_cropped = track and any(
            0 <= before.capture_ts - p.ts <= 0.6 + 1e-6 and p.observation.truncated for p in track.history
        )
        if not (before.truncated or after.truncated or recently_cropped):
            return False
        elapsed = after.capture_ts - before.capture_ts
        if not 0 < elapsed <= self.settings.lost_track_seconds + 1e-6:
            return False
        old, new = self.pose(before), self.pose(after)
        if not old.torso_visible:
            old = (
                max(
                    (
                        p
                        for p in track.history
                        if before.capture_ts - p.ts <= 1
                        and p.torso_visible
                        and not p.observation.association_uncertain
                    ),
                    key=lambda p: p.torso_length,
                    default=old,
                )
                if track
                else old
            )
        if (
            not old.torso_visible
            or old.lean is None
            or old.lean >= 0.7
            or old.torso_length < 10
            or new.low
            or new.body_visible
            or (new.lean is not None and new.lean >= 0.7)
        ):
            return False
        # A head turn can hide the nose while an eye or ear remains visible.
        # Require the same confident head landmark and both shoulders, rather
        # than turning one weak nose estimate into apparent body compression.
        head = [
            k
            for k in range(5)
            if min(len(before.keypoints), len(after.keypoints)) > k
            and min(before.keypoints[k][2], after.keypoints[k][2]) >= 0.65
        ]
        weak_head = not head
        if weak_head:
            # A brief confidence dip must retain measured geometry, not create
            # a new drop from box shrinkage. Require prior strong observations
            # and continued shoulder motion below; never invent a head point.
            if not track or track.concern or before.association_uncertain or after.association_uncertain:
                return False
            head = [
                k
                for k in range(5)
                if min(len(before.keypoints), len(after.keypoints)) > k
                and min(before.keypoints[k][2], after.keypoints[k][2]) >= self.settings.keypoint_confidence
            ]
        joints = (5, 6)
        if not head or any(
            min(len(before.keypoints), len(after.keypoints)) <= k
            or min(before.keypoints[k][2], after.keypoints[k][2]) < 0.65
            for k in joints
        ):
            return False
        head = head[:1]  # Keep the nose when visible; otherwise use one shared landmark.
        shared = (
            head
            + list(joints)
            + [
                k
                for k in (11, 12)
                if min(len(before.keypoints), len(after.keypoints)) > k
                and min(before.keypoints[k][2], after.keypoints[k][2]) >= 0.65
            ]
        )
        offsets = [
            tuple(after.keypoints[k][axis] - before.keypoints[k][axis] for axis in (0, 1)) for k in shared
        ]
        center = tuple(sum(p[axis] for p in offsets) / len(offsets) for axis in (0, 1))
        scales = (
            [
                p.torso_length
                for p in track.history
                if 0 <= before.capture_ts - p.ts <= 0.6 + 1e-6
                and p.torso_visible
                and p.lean is not None
                and p.lean < 0.7
                and not p.observation.association_uncertain
            ]
            if track
            else []
        )
        # The last visible pelvis measurement can shorten as it reaches the
        # image edge. Normalize with recent observed geometry, never clipped
        # joints, so that noise alone does not turn steady walking into a drop.
        scale = max(old.torso_length, median(scales)) if scales else old.torso_length
        if weak_head or head[0] != 0:
            # A weak or fallback landmark must continue observed translation.
            # Without that history, a rigid downward movement may be the start
            # of a collapse. One plausible ear/eye cannot veto that onset.
            history = (
                [p for p in track.history if 0 <= before.capture_ts - p.ts <= 0.8 + 1e-6] if track else []
            )
            if len(history) < 3 or history[-1].ts - history[0].ts < 0.4 - 1e-6:
                return False
            if any(
                p.observation.association_uncertain
                or p.low
                or len(p.observation.keypoints) < 7
                or min(p.observation.keypoints[k][2] for k in (5, 6)) < 0.65
                for p in history
            ):
                return False
            if (
                weak_head
                and sum(
                    len(p.observation.keypoints) > head[0] and p.observation.keypoints[head[0]][2] >= 0.65
                    for p in history
                )
                < 3
            ):
                return False
            velocities = []
            for a, b in zip(history, history[1:], strict=False):
                if not 0 < b.ts - a.ts <= 0.4 + 1e-6:
                    return False
                velocities.append(
                    tuple(
                        sum(
                            b.observation.keypoints[k][axis] - a.observation.keypoints[k][axis]
                            for k in (5, 6)
                        )
                        / 2
                        / (b.ts - a.ts)
                        for axis in (0, 1)
                    )
                )
            prior_velocity = tuple(median(v[axis] for v in velocities) for axis in (0, 1))
            velocity = tuple(
                sum(after.keypoints[k][axis] - before.keypoints[k][axis] for k in (5, 6)) / 2 / elapsed
                for axis in (0, 1)
            )
            prior_speed, speed = math.hypot(*prior_velocity), math.hypot(*velocity)
            if (
                prior_speed < scale * 0.2
                or speed > prior_speed * 1.75 + scale * 0.2
                or sum(a * b for a, b in zip(prior_velocity, velocity, strict=True))
                < prior_speed * speed * 0.8
            ):
                return False
        return (
            math.hypot(*center) / elapsed <= scale * 2
            and max(math.dist(p, center) for p in offsets) <= scale * 0.25
        )

    def _continued_translation(self, before: Observation, after: Observation) -> bool:
        """Recognize observed torso translation as the visible body becomes smaller.

        A lateral exit can hide the head and one shoulder. The remaining
        same-side shoulder/hip pair must retain its geometry and continue
        established motion from a recently visible upright body. A new drop,
        a gap or an uncertain identity cannot establish this explanation. Away
        from an image edge, require stronger joints, stable torso shape and
        predominantly lateral motion without new vertical acceleration.
        """
        track = self.tracks.get(before.track_id)
        if not track or track.concern or after.association_uncertain:
            return False
        interior = not after.truncated
        if interior and before.association_uncertain:
            return False
        if not 0 < after.capture_ts - before.capture_ts <= 0.4 + 1e-6:
            return False
        old, new = self.pose(before), self.pose(after)
        if new.low or new.body_visible or not old.torso_visible or old.torso_length < 12:
            return False
        if old.lean is None or old.lean >= 0.7 or new.lean is None or new.lean >= 0.7:
            return False
        history = [p for p in track.history if 0 <= before.capture_ts - p.ts <= 0.8 + 1e-6]
        if len(history) < 3 or history[-1].ts - history[0].ts < 0.4 - 1e-6:
            return False
        if not any(
            p.body_visible and p.posture == "upright"
            for p in track.history
            if 0 <= before.capture_ts - p.ts <= 2
        ):
            return False
        if any(p.low or p.observation.association_uncertain for p in history):
            return False
        if any(not 0 < b.ts - a.ts <= 0.4 + 1e-6 for a, b in zip(history, history[1:], strict=False)):
            return False
        for shoulder, hip in ((5, 11), (6, 12)):
            observations = [p.observation for p in history] + [after]
            if any(
                len(o.keypoints) <= hip
                or o.keypoints[shoulder][2] < 0.65
                or o.keypoints[hip][2] < (0.65 if interior else 0.4)
                for o in observations
            ):
                continue
            axes = [
                (
                    o.keypoints[hip][0] - o.keypoints[shoulder][0],
                    o.keypoints[hip][1] - o.keypoints[shoulder][1],
                )
                for o in observations
            ]
            lengths = [math.hypot(*a) for a in axes]
            scale = median(lengths[:-1])
            if scale < 12 or any(
                a[1] / max(1, n) < math.sqrt(0.5) for a, n in zip(axes, lengths, strict=True)
            ):
                continue
            if not 0.75 <= lengths[-1] / scale <= 1.33 or math.dist(axes[-1], axes[-2]) > scale * 0.25:
                continue
            if interior and (
                not 0.8 <= lengths[-1] / scale <= 1.25 or math.dist(axes[-1], axes[-2]) > scale * 0.2
            ):
                continue
            centers = [
                tuple((o.keypoints[shoulder][axis] + o.keypoints[hip][axis]) / 2 for axis in (0, 1))
                for o in observations
            ]
            velocities = [
                tuple((b[axis] - a[axis]) / (ob.capture_ts - oa.capture_ts) for axis in (0, 1))
                for a, b, oa, ob in zip(centers, centers[1:], observations, observations[1:], strict=False)
            ]
            prior = tuple(median(v[axis] for v in velocities[:-1]) for axis in (0, 1))
            current = velocities[-1]
            prior_speed, speed = math.hypot(*prior), math.hypot(*current)
            if interior and (
                abs(prior[0]) < max(abs(prior[1]), scale * 0.3)
                or abs(current[0]) < abs(current[1])
                or speed > prior_speed * 1.5 + scale * 0.2
                or abs(current[1] - prior[1]) > scale * 0.3
                or sum(a * b for a, b in zip(prior, current, strict=True)) < prior_speed * speed * 0.9
            ):
                continue
            if (
                prior_speed >= scale * 0.2
                and speed <= prior_speed * 1.75 + scale * 0.2
                and sum(a * b for a, b in zip(prior, current, strict=True)) >= prior_speed * speed * 0.8
            ):
                return True
        return False

    def pose(self, observation: Observation) -> Pose:
        if self._pose_cache is None:
            return self._measure_pose(observation)
        key = id(observation)
        if key not in self._pose_cache:
            self._pose_cache[key] = self._measure_pose(observation)
        return self._pose_cache[key]

    def _measure_pose(self, observation: Observation) -> Pose:
        points = observation.keypoints
        threshold = self.settings.keypoint_confidence

        def point(index):
            return points[index] if len(points) > index and points[index][2] >= threshold else None

        def midpoint(a, b):
            present = [p for p in (point(a), point(b)) if p is not None]
            return tuple(sum(p[j] for p in present) / len(present) for j in (0, 1)) if present else None

        shoulder, hip = midpoint(5, 6), midpoint(11, 12)
        lean = abs(shoulder[0] - hip[0]) / max(1, abs(shoulder[1] - hip[1])) if shoulder and hip else None
        center = (
            (shoulder[1] + hip[1]) / 2 if shoulder and hip else (observation.box.y1 + observation.box.y2) / 2
        )
        aspect = observation.box.width / max(1, observation.box.height)
        legs_visible = any(point(k) is not None for k in (13, 14, 15, 16))
        torso_length = math.dist(shoulder, hip) if shoulder and hip else 0
        ankles = [point(k) for k in (15, 16) if point(k) is not None]
        extended_below = bool(hip and ankles and min(p[1] for p in ankles) - hip[1] > torso_length * 0.8)
        # Wide head-and-shoulders crops are not lying people. A horizontal torso
        # above extended legs is usually bending; it is not a floor landing.
        horizontal = lean is not None and lean > 1.4 and torso_length >= 10
        inverted = bool(shoulder and hip and shoulder[1] - hip[1] > max(8, torso_length * 0.25))
        low = bool(
            (
                horizontal
                and (aspect >= 0.85 or not extended_below)
                and (legs_visible or not observation.truncated)
            )
            or (aspect >= self.settings.low_aspect_ratio and legs_visible and (lean is None or lean >= 0.7))
            or (inverted and legs_visible and not extended_below)
            or (
                aspect >= self.settings.low_aspect_ratio
                and not observation.truncated
                and not shoulder
                and not hip
            )
        )
        bent = []
        standing_legs = []
        for a, b, c in ((11, 13, 15), (12, 14, 16)):
            pa, pb, pc = point(a), point(b), point(c)
            if pa and pb and pc:
                u, v = (pa[0] - pb[0], pa[1] - pb[1]), (pc[0] - pb[0], pc[1] - pb[1])
                norm = math.hypot(*u) * math.hypot(*v)
                if norm > 1:
                    angle = math.degrees(math.acos(max(-1, min(1, (u[0] * v[0] + u[1] * v[1]) / norm))))
                    bent.append(angle < 125)
                    leg_length = math.hypot(*u) + math.hypot(*v)
                    standing_legs.append(
                        angle >= 125 and pc[1] - pa[1] >= max(torso_length * 0.8, leg_length * 0.6)
                    )
        reliable = (
            not observation.limited
            and observation.track_id is not None
            and observation.track_id >= 0
            and shoulder is not None
            and hip is not None
            and len(bent) >= 1
            and torso_length >= max(8, observation.box.height * 0.12)
        )
        posture = "low" if low else "uncertain"
        # A person lying towards the camera can have vertically aligned joints.
        # Severe torso foreshortening is insufficient evidence of standing.
        leg_extent = max((p[1] - hip[1] for p in ankles), default=0) if hip else 0
        if (
            reliable
            and lean is not None
            and lean < 0.7
            and not low
            and hip[1] - shoulder[1] > max(6, torso_length * 0.6)
            and leg_extent <= torso_length * 2.6
        ):
            # Straight knees also occur while sitting on the floor. Standing
            # requires a visible extended leg below the pelvis, not just an
            # upright torso or a straight hip-knee-ankle chain.
            posture = "upright" if any(standing_legs) else "seated"
        torso_visible = not observation.limited and shoulder is not None and hip is not None
        body_visible = bool(
            torso_visible
            and any(
                all(point(k) is not None and point(k)[2] >= 0.65 for k in chain)
                and math.dist(point(chain[0])[:2], point(chain[-1])[:2]) > observation.box.height * 0.55
                for chain in ((5, 11, 13, 15), (6, 12, 14, 16))
            )
        )
        return Pose(
            observation,
            center,
            lean,
            low,
            posture,
            bool(bent) and all(bent),
            reliable,
            torso_visible,
            legs_visible,
            body_visible,
            shoulder[1] if shoulder else None,
            torso_length,
        )

    def _rising_after_concern(self, track: Track, pose: Pose) -> bool:
        """Visible ascent and leg straightening cannot establish settled recovery.

        Compare only current/past strong observations of the same joints. This
        withholds recovery credit while a compact body is visibly unfolding;
        it never creates a concern, infers intent or changes an observed ID.
        Uniform scale change or a moving box alone is insufficient.
        """
        if (
            not track.concern
            or track.activity_resumed
            or track.recovery_observed
            or track.evidence.identity_uncertain
            or not pose.reliable
            or pose.posture != "upright"
        ):
            return False
        current = pose.observation
        for before in track.history:
            old = before.observation
            if (
                not 0 < pose.ts - before.ts <= self.settings.drop_window_seconds + 1e-6
                or old.limited
                or old.association_uncertain
                or current.association_uncertain
                or pose.height < before.height * 1.2
                or min(len(old.keypoints), len(current.keypoints)) < 17
            ):
                continue
            if any(min(old.keypoints[k][2], current.keypoints[k][2]) < 0.65 for k in (5, 6, 11, 12)):
                continue
            if any(old.keypoints[k][1] - current.keypoints[k][1] < before.height * 0.1 for k in (5, 6)):
                continue
            if any(old.keypoints[k][1] - current.keypoints[k][1] < before.height * 0.05 for k in (11, 12)):
                continue
            for chain in ((11, 13, 15), (12, 14, 16)):
                if any(min(old.keypoints[k][2], current.keypoints[k][2]) < 0.65 for k in chain):
                    continue
                cosines = []
                for obs in (old, current):
                    a, b, c = (obs.keypoints[k][:2] for k in chain)
                    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
                    lengths = math.hypot(*u), math.hypot(*v)
                    if min(lengths) < max(3, before.height * 0.08):
                        break
                    cosines.append(sum(x * y for x, y in zip(u, v, strict=True)) / (lengths[0] * lengths[1]))
                if (
                    len(cosines) == 2
                    and cosines[0] >= math.cos(math.radians(125))
                    and cosines[1] <= math.cos(math.radians(150))
                ):
                    return True
        return False

    def _controlled(self, track: Track, pose: Pose) -> Pose | None:
        """Find a stable flexed intermediate pose followed by a supported recline.

        Speed alone is insufficient: require a visible pause, then little torso
        descent during the recline. This cannot establish intent or actual support
        from furniture. Cloud agreement is still required for ordinary activity.
        """
        history = [p for p in track.history if pose.ts - p.ts <= 5] + [pose]
        if not pose.low or len(history) < 8 or not all(p.torso_visible for p in history):
            return None
        for end, reference in enumerate(history[:-1]):
            preparation = [p for p in history[: end + 1] if reference.ts - p.ts <= 1.2]
            if len(preparation) < 5 or preparation[-1].ts - preparation[0].ts < 0.8 - 1e-6:
                continue
            if not all(p.reliable and p.posture == "seated" for p in preparation):
                continue
            if (
                max(p.center_y for p in preparation) - min(p.center_y for p in preparation)
                > reference.height * 0.08
            ):
                continue
            sequence = history[end:]
            if pose.ts - reference.ts < 0.8:
                continue
            if max(abs(p.center_y - reference.center_y) for p in sequence) > reference.height * 0.18:
                continue
            if any(
                b.ts - a.ts > self.settings.lost_track_seconds + 1e-6
                or abs(b.center_y - a.center_y) / reference.height / (b.ts - a.ts) > 0.55
                for a, b in zip(sequence, sequence[1:], strict=False)
            ):
                continue
            return reference
        return None

    def _new_episode(self, track: Track, start: float, *, after_activity: bool = False):
        old = track.evidence
        track.evidence = MotionEvidence(
            episode_id=uid(),
            previous_episode_id=old.episode_id if after_activity else None,
            track_id=old.track_id,
            epoch=old.epoch,
            started_ts=start,
            observed_ts=start,
            last_seen_ts=start,
        )
        track.concern = track.emitted = False
        track.recovery_since = track.low_since = track.transition_end = None
        track.last_recovery_ts = None
        track.support = None
        track.unresolved_landing = False
        track.activity_resumed = False
        track.recovery_observed = False
        track.reobserved_since = None
        track.reobserved_pose = None

    def update(
        self, observations: list[Observation], ts: float, epoch: int, candidates: list[Candidate]
    ) -> list[Candidate]:
        if epoch != self.epoch:
            self.reset()
            self.epoch = epoch
        seen = set()
        extra = []
        for obs in observations:
            tid = obs.track_id
            if tid is None:
                continue
            seen.add(tid)
            pose = self.pose(obs)
            track = self.tracks.get(tid)
            if track is None:
                if len(self.tracks) >= self.max_tracks:
                    oldest = min(self.tracks, key=lambda key: self.tracks[key].evidence.last_seen_ts)
                    self.tracks.pop(oldest)
                track = self.tracks[tid] = Track(
                    MotionEvidence(
                        episode_id=uid(),
                        track_id=tid,
                        epoch=epoch,
                        started_ts=ts,
                        observed_ts=ts,
                        last_seen_ts=ts,
                    )
                )
            elif ts <= track.evidence.last_seen_ts:
                continue
            evidence = track.evidence
            gap = bool(track.history) and ts - evidence.last_seen_ts > self.settings.lost_track_seconds + 1e-6
            ambiguous = obs.association_uncertain or any(
                other.track_id != tid and other.box.iou(obs.box) > 0.5 for other in observations
            )
            if gap or ambiguous:
                track.reobserved_since = None
                track.reobserved_pose = None
            if gap:
                track.support = None
                track.activity_resumed = False
                track.baseline_since = track.recovery_since = track.low_since = None
                evidence.clear_eligible = False
                if track.concern:
                    # Reusing a geometric track ID after missing observations
                    # does not prove that the same person returned. Keep an
                    # unresolved episode uncertain even without a visible occluder.
                    evidence.identity_uncertain = True
                if evidence.transition == "controlled_pattern":
                    self._new_episode(track, ts)
                    evidence = track.evidence
            if ambiguous:
                evidence.identity_uncertain = True
                track.activity_resumed = False
                track.baseline = None
                track.history.clear()
            elif not track.concern:
                # Identity before an event does not permanently taint a new,
                # continuously visible baseline. Active episodes stay unresolved.
                evidence.identity_uncertain = False
            history = [p for p in track.history if ts - p.ts <= 5]
            local = [
                c
                for c in candidates
                if c.track_id == tid
                or (tid < 0 and c.track_id is None and c.box and c.box.iou(obs.box) > 0.25)
            ]
            rapid = next(
                (c for c in local if c.reason in ("rapid_posture_change", "partial_body_descent")), None
            )
            reference = max(
                (p for p in history if not p.observation.association_uncertain),
                key=lambda p: p.height,
                default=None,
            )
            descent, shoulder_descent = (
                self.displacement(reference.observation, obs) if reference else (None, None)
            )
            drop = descent if descent is not None else 0
            slow = bool(
                reference
                and 1 <= ts - reference.ts <= 5
                and drop >= 0.25
                and shoulder_descent is not None
                and shoulder_descent >= self.settings.pre_drop_ratio
                and not obs.association_uncertain
                and (pose.height < reference.height * 0.78 or pose.low)
                # Losing the legs behind an occluder shrinks a walking person's
                # box. Require a visible posture change, not that box change alone.
                and (
                    pose.low
                    or (pose.reliable and pose.body_visible)
                    or (
                        pose.torso_visible
                        and reference.torso_visible
                        and pose.torso_length < reference.torso_length * 0.75
                    )
                )
                and pose.torso_visible
                and (not obs.truncated or pose.low)
                and (pose.body_visible or any(p.body_visible for p in history))
                and not self.cropped_translation(history[-1].observation, obs)
                and not self.standing_bend(reference.observation, obs)
            )
            controlled = self._controlled(track, pose) if not track.unresolved_landing else None
            if controlled is not None:
                rapid = None  # Stable preparation is more specific than box shrinkage.
            recent_descent = self.displacement(history[-1].observation, obs)[0] if history else None
            recent_drop = bool(
                recent_descent is not None and recent_descent > 0.035 and not obs.association_uncertain
            )
            material = rapid is not None or (slow and recent_drop)
            onset = rapid.first_ts if rapid else reference.ts if slow else ts
            if (
                material
                and track.concern
                and evidence.identity_uncertain
                and not ambiguous
                and track.reobserved_since is not None
                and track.reobserved_pose is not None
                and onset - track.reobserved_since >= self.settings.recovery_seconds - 1e-6
                and ts - track.reobserved_pose.ts <= 0.4 + 1e-6
            ):
                # Fresh activity establishes context for a later observed event,
                # not the identity or recovery of the missing person. The old
                # incident stays unresolved; this episode has no predecessor link.
                baseline = track.reobserved_pose
                self._new_episode(track, onset)
                track.baseline = baseline
                track.baseline_since = baseline.ts
                evidence = track.evidence
            new_low = (
                track.recovery_observed
                and pose.low
                and any(
                    c.reason == "persistent_low_posture"
                    and track.last_recovery_ts is not None
                    and c.first_ts > track.last_recovery_ts
                    for c in local
                )
            )
            if (material and (track.activity_resumed or (track.recovery_observed and local))) or new_low:
                # A weak pose between observed recovery and another descent
                # must not merge the new event into an exhausted old episode.
                continued = track.activity_resumed and not evidence.identity_uncertain
                start = (
                    min(c.first_ts for c in local if c.reason == "persistent_low_posture")
                    if new_low
                    else onset
                    if continued
                    else ts
                )
                self._new_episode(track, start, after_activity=continued)
                if not continued:
                    track.baseline = None
                    track.history.clear()
                    history = []
                    drop = 0
                if new_low or not continued:
                    rapid = controlled = None
                    slow = False
                    track.unresolved_landing = any(c.reason == "partial_body_descent" for c in local)
                evidence = track.evidence
            if rapid:
                evidence.transition = "fall_like"
                evidence.onset_observed = True
                evidence.started_ts = min(evidence.started_ts, rapid.first_ts)
                track.concern = True
                track.transition_end = ts
                # A cropped landing cannot be relabelled as harmless rest from
                # a later recline. Its missing legs are not recovery evidence.
                if rapid.reason == "partial_body_descent":
                    track.unresolved_landing = True
            elif controlled is not None:
                evidence.transition = "controlled_pattern"
                evidence.onset_observed = True
                track.concern = True
                track.transition_end = ts
                track.support = controlled
            elif (
                slow
                and recent_drop
                and evidence.transition != "fall_like"
                and not (evidence.transition == "controlled_pattern" and track.support is not None)
            ):
                evidence.transition = "slow_descent"
                evidence.onset_observed = True
                track.concern = True
                track.transition_end = ts
                if not local and not track.emitted:
                    extra.append(
                        Candidate(
                            category=Category.MEDICAL,
                            reason="slow_posture_descent",
                            first_ts=reference.ts,
                            last_ts=ts,
                            box=reference.observation.box.union(obs.box),
                            track_id=tid if tid >= 0 else None,
                            priority=2,
                            limited=not pose.reliable,
                            epoch=epoch,
                        )
                    )
            support_shift = (
                (pose.center_y - track.support.center_y) / track.support.height if track.support else 0
            )
            if evidence.transition == "controlled_pattern" and support_shift < -0.18:
                # Rising invalidates the old support pattern, but is not a new
                # descent. Keep this episode; recovery still needs observation.
                track.support = None
            if evidence.transition == "controlled_pattern" and support_shift > 0.18:
                self._new_episode(
                    track, history[-1].ts if history else ts, after_activity=track.activity_resumed
                )
                evidence = track.evidence
                evidence.transition = "slow_descent"
                evidence.onset_observed = True
                track.concern = True
                extra.append(
                    Candidate(
                        category=Category.MEDICAL,
                        reason="support_pattern_lost",
                        first_ts=evidence.started_ts,
                        last_ts=ts,
                        box=obs.box,
                        track_id=tid if tid >= 0 else None,
                        priority=2,
                        limited=not pose.torso_visible,
                        epoch=epoch,
                    )
                )
            if pose.low:
                if rapid is not None:
                    track.unresolved_landing = True
                track.low_since = ts if track.low_since is None else track.low_since
                track.last_low_ts = ts
                track.recovery_since = None
            elif (
                pose.posture in ("upright", "seated")
                or track.last_low_ts is None
                or ts - track.last_low_ts > 0.6
            ):
                track.low_since = None
            if local or (
                track.low_since is not None
                and ts - track.low_since >= self.settings.low_pose_seconds
                and not obs.limited
                and not obs.truncated
            ):
                track.concern = True
                if evidence.transition == "none":
                    evidence.transition = "unknown_origin"
            if track.concern:
                evidence.phase = "following_concern"
                # A previous supported seated baseline supports non-ambulatory recovery.
                baseline_return = bool(
                    track.baseline
                    and track.baseline.posture == "seated"
                    and pose.posture == "seated"
                    and pose.height >= track.baseline.height * 0.85
                    and abs(pose.center_y - track.baseline.center_y) < track.baseline.height * 0.18
                )
                rising = self._rising_after_concern(track, pose)
                recovering = (
                    pose.reliable
                    and not rising
                    and not evidence.identity_uncertain
                    and (pose.posture == "upright" or baseline_return)
                )
                if recovering and not material:
                    track.recovery_since = ts if track.recovery_since is None else track.recovery_since
                    track.last_recovery_ts = ts
                elif (
                    not material
                    and not rising
                    and not pose.low
                    and not evidence.identity_uncertain
                    and track.last_recovery_ts is not None
                    and ts - track.last_recovery_ts <= 0.4 + 1e-6
                ):
                    pass  # One weak pose may interrupt measurement, never establish recovery.
                else:
                    track.recovery_since = None
                evidence.recovery_seconds = (
                    max(0, ts - track.recovery_since) if track.recovery_since is not None else 0
                )
                evidence.recovery_basis = (
                    ("prior_supported_posture" if baseline_return else "upright_activity")
                    if recovering
                    else None
                )
                ordinary = (
                    evidence.transition == "controlled_pattern"
                    and track.support is not None
                    and track.low_since is not None
                    and ts - track.low_since >= self.settings.ordinary_activity_seconds
                    and pose.torso_visible
                    and not evidence.identity_uncertain
                )
                if ordinary:
                    evidence.phase = "ordinary_activity"
                elif recovering and evidence.recovery_seconds >= self.settings.recovery_seconds:
                    evidence.phase = "recovered_activity"
                evidence.clear_eligible = evidence.phase in ("ordinary_activity", "recovered_activity")
                if evidence.clear_eligible:
                    track.activity_resumed = True
                    track.recovery_observed |= evidence.phase == "recovered_activity"
            elif pose.reliable and pose.posture in ("upright", "seated") and abs(drop) < 0.08:
                evidence.phase = "observing"
                if history and history[-1].posture != pose.posture:
                    track.baseline_since = None
                track.baseline_since = ts if track.baseline_since is None else track.baseline_since
                if ts - track.baseline_since >= 1:
                    track.baseline = pose
            else:
                track.baseline_since = None
                if not track.concern:
                    evidence.phase = "observing"
            if track.concern and evidence.identity_uncertain and not ambiguous and not material:
                if pose.reliable and pose.posture in ("upright", "seated"):
                    if (
                        track.reobserved_pose is None
                        or pose.posture != track.reobserved_pose.posture
                        or ts - track.reobserved_pose.ts > 0.4 + 1e-6
                    ):
                        track.reobserved_since = ts
                    track.reobserved_pose = pose
                elif not (
                    not pose.low
                    and track.reobserved_pose is not None
                    and ts - track.reobserved_pose.ts <= 0.4 + 1e-6
                ):
                    track.reobserved_since = None
                    track.reobserved_pose = None
            else:
                track.reobserved_since = None
                track.reobserved_pose = None
            evidence.observed_ts = evidence.last_seen_ts = ts
            evidence.posture = pose.posture
            evidence.features = {
                "normalized_descent": round(drop, 4),
                "visible_samples": len(history) + 1,
                "reliable_pose": float(pose.reliable),
            }
            track.history.append(pose)

        for tid, track in list(self.tracks.items()):
            evidence = track.evidence
            if tid not in seen and ts - evidence.last_seen_ts + 1e-6 >= self.settings.lost_track_seconds:
                evidence.clear_eligible = False
                evidence.phase = "visibility_lost"
                evidence.observed_ts = ts
                track.recovery_since = track.baseline_since = None
                track.reobserved_since = None
                track.reobserved_pose = None
            if ts - evidence.last_seen_ts > self.settings.dedup_seconds:
                self.tracks.pop(tid)
        output = []
        for candidate in candidates + extra:
            matching = [
                t
                for tid, t in self.tracks.items()
                if candidate.track_id == tid
                or (
                    candidate.track_id is None
                    and tid < 0
                    and t.history
                    and candidate.box
                    and candidate.box.iou(t.history[-1].observation.box) > 0.25
                )
            ]
            if matching:
                track = max(matching, key=lambda t: t.evidence.last_seen_ts)
                # A first concern may be emitted only after the person vanishes,
                # outside the observed-track loop. Retain it so a reused ID
                # cannot silently resume ordinary observation after that gap.
                track.concern = True
                if track.evidence.transition == "none":
                    track.evidence.transition = "unknown_origin"
                track.emitted = True
                candidate = candidate.model_copy(update={"motion": track.evidence.model_copy(deep=True)})
                if track.evidence.transition == "unknown_origin" and candidate.reason in (
                    "rapid_posture_change",
                    "partial_body_descent",
                ):
                    candidate.limited = True
                if (
                    track.evidence.transition == "controlled_pattern"
                    and candidate.reason == "rapid_posture_change"
                ):
                    candidate.reason = "posture_transition"
            output.append(candidate)
        return output

    def snapshot(self) -> list[MotionEvidence]:
        return [track.evidence.model_copy(deep=True) for track in self.tracks.values()]
