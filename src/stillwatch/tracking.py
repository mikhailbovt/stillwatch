"""Short, source-time pose association. No biometric identity or predicted observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .types import Observation


def minimum_assignment(costs: list[list[float]]) -> list[int]:
    """Minimum-cost row assignment to distinct columns, including caller dummies.

    Rectangular Hungarian algorithm. Callers provide at least one column per
    row and finite costs. A whole frame is matched together, not greedily.
    """
    if not costs:
        return []
    rows, columns = len(costs), len(costs[0])
    if columns < rows or any(len(row) != columns for row in costs):
        raise ValueError("Assignment requires a rectangular matrix with columns >= rows")
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    owner, previous = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        owner[0] = row
        column = 0
        distance, used = [math.inf] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column] = True
            active_row = owner[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = costs[active_row - 1][candidate - 1] - u[active_row] - v[candidate]
                if reduced < distance[candidate]:
                    distance[candidate], previous[candidate] = reduced, column
                if distance[candidate] < delta:
                    delta, next_column = distance[candidate], candidate
            if not math.isfinite(delta):
                raise ValueError("Assignment costs must permit a finite matching")
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            predecessor = previous[column]
            owner[column] = owner[predecessor]
            column = predecessor
    result = [-1] * rows
    for column in range(1, columns + 1):
        if owner[column]:
            result[owner[column] - 1] = column - 1
    return result


@dataclass
class Track:
    current: Observation
    previous: Observation | None = None
    hits: int = 1
    confirmed: bool = False
    occluded: bool = False


class PoseTracker:
    """Associate all detector outputs, including detections a box tracker would discard.

    Matching uses observed torso joints, box overlap and bounded motion prediction.
    A competing match marks identity uncertain; an absent person is never synthesized.
    """

    def __init__(self, retention_seconds: float = 3):
        self.retention_seconds = retention_seconds
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.epoch = -1

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.epoch = -1

    @staticmethod
    def _center(o: Observation):
        return ((o.box.x1 + o.box.x2) / 2, (o.box.y1 + o.box.y2) / 2)

    @staticmethod
    def _scale_ratios(before: Observation, after: Observation) -> list[float] | None:
        """None means no box growth; an empty list means insufficient shared joints."""
        if after.box.height <= before.box.height * 1.4:
            return None
        ratios = []
        for a, b in ((0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12)):
            if min(len(before.keypoints), len(after.keypoints)) <= max(a, b):
                continue
            if (
                min(before.keypoints[k][2] for k in (a, b)) < 0.4
                or min(after.keypoints[k][2] for k in (a, b)) < 0.4
            ):
                continue
            old_length = math.dist(before.keypoints[a][:2], before.keypoints[b][:2])
            if old_length >= 10:
                ratios.append(math.dist(after.keypoints[a][:2], after.keypoints[b][:2]) / old_length)
        return ratios

    @classmethod
    def _scale_discontinuity(cls, before: Observation, after: Observation) -> bool:
        """A closer occluder must not inherit a smaller background person's ID."""
        ratios = cls._scale_ratios(before, after)
        # Box growth alone also occurs during recovery. Require a simultaneous
        # change in several visible body segments. Horizontal falls can widen
        # the box without increasing body scale; gradual approach remains valid.
        return ratios is not None and len(ratios) >= 2 and median(ratios) > 1.8

    @staticmethod
    def _unanchored_expansion(before: Observation, after: Observation) -> bool:
        """A larger crossing person must not take a compact body's identity.

        Box overlap and similar bone lengths can fit a different nearby body.
        A supported rise can retain a measured joint near its prior position.
        Missing, cropped or weak torso evidence cannot establish this veto.
        """
        if before.truncated or after.truncated or min(len(before.keypoints), len(after.keypoints)) < 13:
            return False
        if not (
            before.box.width >= before.box.height * 1.2
            and after.box.height > before.box.height * 1.8
            and after.box.width > before.box.width * 1.4
        ):
            return False
        shared = [
            k
            for k in range(min(len(before.keypoints), len(after.keypoints)))
            if all(math.isfinite(v) for v in (*before.keypoints[k], *after.keypoints[k]))
            and min(before.keypoints[k][2], after.keypoints[k][2]) >= 0.65
        ]
        if not all(k in shared for k in (5, 6, 11, 12)):
            return False
        tolerance = max(4, max(before.box.width, before.box.height) * 0.2)
        return all(math.dist(before.keypoints[k][:2], after.keypoints[k][:2]) > tolerance for k in shared)

    @staticmethod
    def _appearance_conflict(before: Observation, after: Observation) -> bool:
        """A color conflict weakens continuity; color agreement never proves identity.

        Compare only close observations with downward, sufficiently long torso
        axes. Foreshortening, inversion and a detection gap can expose another
        part of the same person's clothing during a fall or recovery.
        """
        if not before.torso_histogram or not after.torso_histogram:
            return False
        if not 0 < after.capture_ts - before.capture_ts <= 0.3 + 1e-6:
            return False
        for obs in (before, after):
            if len(obs.keypoints) < 13 or min(obs.keypoints[k][2] for k in (5, 6, 11, 12)) < 0.4:
                return False
            dx, dy = (
                sum(obs.keypoints[k][axis] for k in (11, 12)) / 2
                - sum(obs.keypoints[k][axis] for k in (5, 6)) / 2
                for axis in (0, 1)
            )
            length = math.hypot(dx, dy)
            if length < 12 or dy / length < math.sqrt(0.5):
                return False
        similarity = sum(
            math.sqrt(a * b) for a, b in zip(before.torso_histogram, after.torso_histogram, strict=True)
        )
        return math.sqrt(max(0, 1 - similarity)) > 0.75

    def _cost(self, track: Track, obs: Observation) -> float:
        old = track.current
        dt = obs.capture_ts - old.capture_ts
        if dt <= 0 or dt > self.retention_seconds:
            return math.inf
        if self._scale_discontinuity(old, obs) or self._unanchored_expansion(old, obs):
            return math.inf
        if (
            track.previous
            and not track.previous.association_uncertain
            and obs.capture_ts - track.previous.capture_ts <= 0.8 + 1e-6
            and self._scale_discontinuity(track.previous, obs)
        ):
            return math.inf  # One weak pose cannot erase recently observed body scale.
        scale = max(40, old.box.height, old.box.width, obs.box.height, obs.box.width)
        ox, oy = self._center(old)
        nx, ny = self._center(obs)
        dx = dy = 0.0
        if (
            track.previous
            and old.capture_ts > track.previous.capture_ts
            and min(old.box.height, track.previous.box.height)
            >= max(old.box.height, track.previous.box.height) * 0.8
        ):
            # A collapse changes the box centre through deformation. It must
            # not become a walking velocity extrapolated into recovery.
            px, py = self._center(track.previous)
            period = old.capture_ts - track.previous.capture_ts
            horizon = min(dt, 0.4)
            dx = max(-scale * 0.4, min(scale * 0.4, (ox - px) / period * horizon))
            dy = max(-scale * 0.4, min(scale * 0.4, (oy - py) / period * horizon))
        distance = math.hypot(nx - ox - dx, ny - oy - dy) / scale
        overlap = old.box.contained_overlap(obs.box)
        downward_transition = (
            dt <= 0.8
            and abs(nx - ox) < scale * 0.35
            and 0 < ny - oy < scale * 1.25
            and obs.box.height < old.box.height * 0.85
        )
        if not downward_transition and (distance > 0.7 or (overlap < 0.05 and distance > 0.4)):
            return math.inf
        joints = []
        for k in (0, 5, 6, 11, 12):
            if len(old.keypoints) > k and len(obs.keypoints) > k:
                a, b = old.keypoints[k], obs.keypoints[k]
                if min(a[2], b[2]) >= 0.4:
                    joints.append(math.hypot(a[0] + dx - b[0], a[1] + dy - b[1]) / scale)
        joint_distance = sorted(joints)[len(joints) // 2] if joints else distance
        if joint_distance > (1.25 if downward_transition else 0.8):
            return math.inf
        cost = 0.45 * distance + 0.35 * min(joint_distance, 1) + 0.2 * (1 - overlap)
        return min(cost, 0.5) if downward_transition else cost

    # A partial skeleton can disagree below the torso while sharing the same
    # visible head and both shoulders. It must not create a competing person.
    @staticmethod
    def _same_upper_body(a, b):
        if min(len(a.keypoints), len(b.keypoints)) < 7:
            return False
        if a.box.contained_overlap(b.box) < 0.9 or min(a.box.area, b.box.area) > 0.7 * max(
            a.box.area, b.box.area
        ):
            return False
        epsilon = max(2, min(a.box.width, a.box.height, b.box.width, b.box.height) * 0.08)
        for k in (5, 6):
            if (
                min(a.keypoints[k][2], b.keypoints[k][2]) < 0.7
                or math.dist(a.keypoints[k][:2], b.keypoints[k][:2]) > epsilon
            ):
                return False
        shared = [k for k in range(5) if min(a.keypoints[k][2], b.keypoints[k][2]) >= 0.4]
        if not shared or any(math.dist(a.keypoints[k][:2], b.keypoints[k][:2]) > epsilon for k in shared):
            return False
        for obs in (a, b):
            shoulders = tuple(sum(obs.keypoints[k][j] for k in (5, 6)) / 2 for j in range(2))
            if max(math.dist(obs.keypoints[k][:2], shoulders) for k in shared) < epsilon * 2:
                return False
        return True

    def update(self, observations: list[Observation], ts: float, epoch: int) -> list[Observation]:
        if epoch != self.epoch:
            self.reset()
            self.epoch = epoch
        self.tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if ts - t.current.capture_ts <= (self.retention_seconds if t.confirmed else 0.6 + 1e-6)
        }
        unique = []
        for obs in sorted(observations, key=lambda o: o.score, reverse=True)[:128]:
            duplicate = False
            for other in unique:
                scale = max(40, other.box.width, other.box.height, obs.box.width, obs.box.height)
                distances = [
                    math.hypot(a[0] - b[0], a[1] - b[1]) / scale
                    for k in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
                    if len(obs.keypoints) > k
                    and len(other.keypoints) > k
                    and min((a := obs.keypoints[k])[2], (b := other.keypoints[k])[2]) >= 0.4
                ]
                joint_distance = sorted(distances)[len(distances) // 2] if distances else 1
                head_distances = [
                    math.hypot(a[0] - b[0], a[1] - b[1]) / scale
                    for k in (0, 1, 2, 3, 4)
                    if len(obs.keypoints) > k
                    and len(other.keypoints) > k
                    and min((a := obs.keypoints[k])[2], (b := other.keypoints[k])[2]) >= 0.4
                ]
                # Nearby people can share a few projected joints. Suppress a
                # duplicate only when several joints agree, including the tail
                # of the distances rather than just the median.
                duplicate_pose = len(distances) >= 4 and (
                    (joint_distance < 0.06 and sorted(distances)[int((len(distances) - 1) * 0.9)] < 0.12)
                    or (len(head_distances) >= 2 and max(head_distances) < 0.04 and joint_distance < 0.12)
                )
                weak_nested = (
                    other.score >= 0.35
                    and obs.score < other.score * 0.5
                    and other.box.contained_overlap(obs.box) > 0.9
                    and (joint_distance < 0.3 or len(distances) < 2)
                )
                if (
                    (other.box.contained_overlap(obs.box) > 0.6 and duplicate_pose)
                    or weak_nested
                    or self._same_upper_body(obs, other)
                ):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(obs)
        observations = unique
        costs = {
            (i, tid): self._cost(t, o) for i, o in enumerate(observations) for tid, t in self.tracks.items()
        }
        # Excluding one matched index requires at most the two lowest costs.
        # NaNs never satisfy the original ambiguity threshold comparison.
        row_best = {
            i: sorted(
                (costs[i, tid], tid)
                for tid, track in self.tracks.items()
                if track.confirmed and not math.isnan(costs[i, tid])
            )[:2]
            for i in range(len(observations))
        }
        column_best = {
            tid: sorted(
                (costs[i, tid], i)
                for i, obs in enumerate(observations)
                if obs.score >= 0.5 and not obs.limited and not math.isnan(costs[i, tid])
            )[:2]
            for tid in self.tracks
        }
        # Match confident detections first, then recover remaining tracks from
        # weaker poses. Weak fragments must not displace a confidently seen
        # person. Confirmed tracks precede tentative fragments. A new ID has a
        # fixed cost: preserving a stale track must not rotate several visible
        # people between IDs as the crowd grows.
        matches = {}
        assigned_tracks = set()
        stages = (
            [i for i, o in enumerate(observations) if o.score >= 0.5],
            [i for i, o in enumerate(observations) if o.score < 0.5],
        )
        for stage in stages:
            for confirmed in (True, False):
                indices = [i for i in stage if i not in matches]
                track_ids = [
                    tid
                    for tid, track in self.tracks.items()
                    if tid not in assigned_tracks and track.confirmed == confirmed
                ]
                if not indices or not track_ids:
                    continue
                matrix = [
                    [costs[i, tid] if costs[i, tid] <= 0.55 else 1e6 for tid in track_ids]
                    + [0.6] * len(indices)
                    for i in indices
                ]
                for index, column in enumerate(minimum_assignment(matrix)):
                    if column >= len(track_ids):
                        continue
                    i, tid = indices[index], track_ids[column]
                    cost = costs[i, tid]
                    uncertain = any(
                        value <= cost + 0.08
                        for choices, excluded in ((row_best[i], tid), (column_best[tid], i))
                        for value, identity in choices
                        if identity != excluded
                    )
                    matches[i] = (tid, uncertain)
                    assigned_tracks.add(tid)
        output = []
        for i, obs in enumerate(observations):
            if i in matches:
                tid, uncertain = matches[i]
                old = self.tracks[tid]
                previous = old.current
                uncertain = uncertain or old.occluded
                uncertain = uncertain or self._appearance_conflict(previous, obs)
                growth = self._scale_ratios(previous, obs)
                if (
                    previous.keypoints
                    and obs.keypoints
                    and growth is not None
                    and (not growth or (len(growth) == 1 and growth[0] > 1.8))
                ):
                    # Keep a tentative geometric match, but do not bridge motion
                    # histories across growth with no corroborated body scale.
                    uncertain = True
                hits = old.hits + 1 if ts - previous.capture_ts <= 0.6 + 1e-6 else 1
                confirmed = old.confirmed or hits >= 3 or obs.score >= 0.5 or obs.track_id is not None
            else:
                tid = self.next_id
                self.next_id += 1
                previous, uncertain = None, False
                hits, confirmed = 1, obs.score >= 0.5 or obs.track_id is not None
            associated = obs.model_copy(update={"track_id": tid, "association_uncertain": uncertain})
            self.tracks[tid] = Track(associated, previous, hits, confirmed)
            if confirmed:
                output.append(associated)
        visible_ids = {o.track_id for o in output}
        for obs in output:
            body_joints = sum(
                len(obs.keypoints) > k and obs.keypoints[k][2] >= 0.4 for k in (0, 5, 6, 11, 12)
            )
            if (
                obs.keypoints
                and body_joints < 2
                and any(
                    other.track_id != obs.track_id and obs.box.contained_overlap(other.box) > 0.3
                    for other in output
                )
            ):
                # An emitted fragment can still be occluded. A box or ankle
                # alone cannot bridge the motion history to a nearby person.
                # Keep the observation and tentative ID, but mark both this
                # sample and its next match uncertain, as for an absent track.
                obs.association_uncertain = True
                self.tracks[obs.track_id].occluded = True
        for tid, track in self.tracks.items():
            if (
                track.confirmed
                and tid not in visible_ids
                and any(
                    other.track_id != tid and track.current.box.contained_overlap(other.box) > 0.3
                    for other in output
                )
            ):
                # A unique geometric match after another person covered the
                # track does not prove identity continuity.
                track.occluded = True
        while len(self.tracks) > 128:
            self.tracks.pop(min(self.tracks, key=lambda k: self.tracks[k].current.capture_ts))
        return output
