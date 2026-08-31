"""Whole-episode derived temporal metrics, computed once and reused across frames.

Built on top of ``semantics.py`` (never on raw Parquet fields directly), so the
formulas here operate on physically meaningful shapes (21-point hand tracks,
camera position) rather than opaque flat vectors. Pure functions, no I/O, no
Flask import -- intended to be computed once per episode and cached by the
caller (``webapp.py``), not recomputed per frame or per navigation event.

Metrics implemented (see docs/phase-4-visualizer.md for the rationale on which
candidates were included vs. demoted to diagnostics):

- Hand centroid speed (left/right): |centroid_i - centroid_{i-1}| / (t_i - t_{i-1}).
  ``None`` at frame 0 (no prior frame) -- never fabricated as 0.
- Hand span (left/right): max pairwise Euclidean distance among the 21 landmarks
  in a frame. Deliberately avoids any landmark-connectivity assumption.
- Camera speed: |position_i - position_{i-1}| / (t_i - t_{i-1}). Same frame-0 rule.
- Action-to-next-track residual (left/right): ||action_i - track_{i+1}|| over the
  flattened 21x3 vectors. ``None`` where the action row is the dataset's
  documented terminal all-zero convention row (no target defined).
- Sync diagnostics: frame count, median stored-timestamp interval, and both the
  counts *and* the specific frame indices of duplicate/large-gap/negative
  (regressed) timestamp intervals and frame-index discontinuities, computed
  once at episode load. The frame indices (not just counts) exist so
  ``analysis.py`` can turn each occurrence into a located, evidence-linkable
  finding rather than an unlocalized number.
- Hand pose articulation rate (left/right): ``||hand_pose_i - hand_pose_{i-1}|| /
  (t_i - t_{i-1})``, where ``hand_pose`` is the 45-value MANO joint-pose segment
  (``semantics.parse_mano_hand``). ``None`` at frame 0, same convention as the
  other speed series. Deliberately independent of hand centroid speed/span above:
  a hand can translate quickly with rigid fingers (high speed, low articulation)
  or stay in place while grasping (near-zero speed, high articulation) -- a
  distinction centroid-based metrics cannot make.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from . import semantics

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class SyncDiagnostics:
    frame_count: int
    median_interval_seconds: float | None
    large_interval_threshold_seconds: float | None
    large_interval_count: int
    large_interval_frame_indices: tuple[int, ...]
    duplicate_timestamp_count: int
    duplicate_timestamp_frame_indices: tuple[int, ...]
    negative_interval_count: int
    negative_interval_frame_indices: tuple[int, ...]
    frame_index_discontinuity_count: int
    frame_index_discontinuity_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class EpisodeMetrics:
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    left_hand_speed: tuple[float | None, ...]
    right_hand_speed: tuple[float | None, ...]
    left_hand_span: tuple[float, ...]
    right_hand_span: tuple[float, ...]
    camera_speed: tuple[float | None, ...]
    left_action_residual: tuple[float | None, ...]
    right_action_residual: tuple[float | None, ...]
    left_hand_articulation_rate: tuple[float | None, ...]
    right_hand_articulation_rate: tuple[float | None, ...]
    sync_diagnostics: SyncDiagnostics


def _centroid(points: Sequence[Point3]) -> Point3:
    n = len(points)
    return (
        sum(point[0] for point in points) / n,
        sum(point[1] for point in points) / n,
        sum(point[2] for point in points) / n,
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    # Dimension-agnostic on purpose: reused by _speed_series for both 3D points
    # (hand centroid, camera position) and the 45-dim MANO hand-pose vector
    # (articulation rate) -- same "Euclidean distance between two same-shaped
    # readings" formula either way.
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _max_pairwise_distance(points: Sequence[Point3]) -> float:
    best = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = _distance(points[i], points[j])
            if distance > best:
                best = distance
    return best


def _l2_distance_point_sets(a: Sequence[Point3], b: Sequence[Point3]) -> float:
    return math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for i in range(len(a)) for k in range(3)))


def _speed_series(positions: Sequence[Sequence[float]], timestamps: Sequence[float]) -> tuple[float | None, ...]:
    series: list[float | None] = [None]
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i - 1]
        series.append(_distance(positions[i], positions[i - 1]) / dt if dt > 0 else None)
    return tuple(series)


def _sync_diagnostics(frame_indices: Sequence[int], timestamps: Sequence[float]) -> SyncDiagnostics:
    """Flag both large positive gaps and (previously unchecked) negative
    intervals -- a large-gap check alone would let a timestamp regression
    slip through undetected, since a negative delta is smaller than any
    positive threshold, not larger."""
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    median_interval = statistics.median(intervals) if intervals else None
    threshold = median_interval * 1.5 if median_interval else None

    large_interval_frames = tuple(
        frame_indices[i + 1] for i, delta in enumerate(intervals) if threshold is not None and delta > threshold
    )
    negative_interval_frames = tuple(frame_indices[i + 1] for i, delta in enumerate(intervals) if delta < 0)

    seen: set[float] = set()
    duplicate_frames: list[int] = []
    for frame_index, timestamp in zip(frame_indices, timestamps):
        if timestamp in seen:
            duplicate_frames.append(frame_index)
        seen.add(timestamp)

    discontinuity_frames = tuple(
        frame_indices[i + 1]
        for i in range(len(frame_indices) - 1)
        if frame_indices[i + 1] != frame_indices[i] + 1
    )

    return SyncDiagnostics(
        frame_count=len(timestamps),
        median_interval_seconds=median_interval,
        large_interval_threshold_seconds=threshold,
        large_interval_count=len(large_interval_frames),
        large_interval_frame_indices=large_interval_frames,
        duplicate_timestamp_count=len(duplicate_frames),
        duplicate_timestamp_frame_indices=tuple(duplicate_frames),
        negative_interval_count=len(negative_interval_frames),
        negative_interval_frame_indices=negative_interval_frames,
        frame_index_discontinuity_count=len(discontinuity_frames),
        frame_index_discontinuity_frame_indices=discontinuity_frames,
    )


def compute_episode_metrics(
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    left_hand_tracks: Sequence[Sequence[float]],
    right_hand_tracks: Sequence[Sequence[float]],
    left_action_tracks: Sequence[Sequence[float]],
    right_action_tracks: Sequence[Sequence[float]],
    camera_positions: Sequence[Sequence[float]],
    left_hand_mano: Sequence[Sequence[float]],
    right_hand_mano: Sequence[Sequence[float]],
) -> EpisodeMetrics:
    """Compute all whole-episode derived series from raw stored field columns.

    Each ``*_tracks``/``camera_positions``/``*_hand_mano`` argument is the raw
    stored column for one episode (e.g. from
    ``TimeSeriesData.values["left_hand/tracks"]``) -- still in its original
    stored shape (with leading ``ts_ns`` where present). This function shapes
    them via ``semantics.py`` internally.
    """
    n = len(frame_indices)
    left_points = [semantics.parse_hand_tracks(row).points for row in left_hand_tracks]
    right_points = [semantics.parse_hand_tracks(row).points for row in right_hand_tracks]
    camera_points = [(row[1], row[2], row[3]) for row in camera_positions]

    left_centroids = [_centroid(points) for points in left_points]
    right_centroids = [_centroid(points) for points in right_points]

    left_hand_speed = _speed_series(left_centroids, timestamps)
    right_hand_speed = _speed_series(right_centroids, timestamps)
    camera_speed = _speed_series(camera_points, timestamps)

    left_hand_span = tuple(_max_pairwise_distance(points) for points in left_points)
    right_hand_span = tuple(_max_pairwise_distance(points) for points in right_points)

    left_action_residual: list[float | None] = []
    right_action_residual: list[float | None] = []
    for i in range(n):
        left_target = semantics.parse_action_targets(left_action_tracks[i])
        right_target = semantics.parse_action_targets(right_action_tracks[i])
        if left_target.points is not None and i + 1 < n:
            left_action_residual.append(_l2_distance_point_sets(left_target.points, left_points[i + 1]))
        else:
            left_action_residual.append(None)
        if right_target.points is not None and i + 1 < n:
            right_action_residual.append(_l2_distance_point_sets(right_target.points, right_points[i + 1]))
        else:
            right_action_residual.append(None)

    left_hand_pose = [semantics.parse_mano_hand(row).hand_pose for row in left_hand_mano]
    right_hand_pose = [semantics.parse_mano_hand(row).hand_pose for row in right_hand_mano]
    left_hand_articulation_rate = _speed_series(left_hand_pose, timestamps)
    right_hand_articulation_rate = _speed_series(right_hand_pose, timestamps)

    return EpisodeMetrics(
        frame_indices=tuple(frame_indices),
        timestamps=tuple(timestamps),
        left_hand_speed=left_hand_speed,
        right_hand_speed=right_hand_speed,
        left_hand_span=left_hand_span,
        right_hand_span=right_hand_span,
        camera_speed=camera_speed,
        left_action_residual=tuple(left_action_residual),
        right_action_residual=tuple(right_action_residual),
        left_hand_articulation_rate=left_hand_articulation_rate,
        right_hand_articulation_rate=right_hand_articulation_rate,
        sync_diagnostics=_sync_diagnostics(frame_indices, timestamps),
    )
