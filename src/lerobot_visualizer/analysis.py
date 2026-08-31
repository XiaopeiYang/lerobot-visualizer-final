"""Phase 5/6: dataset-wide statistical analysis and cross-modal QA.

Pure computational helpers (``summarize_distribution``, ``quaternion_norms``,
``compute_lag_residuals``, ``pearson_correlation``, ``detect_motion_events``) take
already-fetched values and do no I/O, mirroring ``metrics.py``'s style so they stay
independently unit-testable. ``compute_episode_overview``, ``audit_schema_consistency``,
``compute_sync_validation``, and ``run_full_analysis`` are the orchestration layer:
they take a ``LeRobotDataset`` and loop over ``list_episode_indices()``, so the same
code path works whether the local dataset root has one episode or many.

Every numeric finding here is a measurement, not a claim about data quality by itself
-- see docs/analysis-report.md for the VERIFIED FACT / METADATA DECLARATION /
INFERENCE / HYPOTHESIS labeling applied to these numbers.

Three-tier evidence system for extreme/unusual frames (deliberately distinct
concepts, not one flat "anomaly" bucket -- calling a statistically extreme
but physically ordinary fast motion an "anomaly" overclaims what was actually
found):

- **Motion event** (``detect_motion_events``): ``metric > statistical
  threshold`` (robust z-score, MAD-based). Real signal -- evidence of
  unusual motion -- but on its own says nothing about data quality. Fast
  hand motion, a quick turn, a grasp are all expected to produce these.
- **Suspicious event** (``promote_suspicious_events``): a motion event
  corroborated by a *second, independent* signal at the exact same frame
  (another metric also extreme, or a verified issue at that frame). Worth a
  human look, still not a confirmed defect.
- **Verified issue** (``detect_*_issues`` functions below): a hard,
  physically/format-impossible invariant violation -- NaN/Inf, schema
  mismatch, timestamp regression/duplicate/large-gap, frame-index
  discontinuity, video/timestamp desync past a fixed tolerance, quaternion
  norm violation, or the dataset's documented terminal-sentinel convention
  appearing on a non-terminal frame. These are the only things that belong
  in a "data quality issues" list.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import av

from . import metrics as metrics_module
from . import semantics
from .dataset import LeRobotDataset

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    finite_count: int
    nan_count: int
    inf_count: int
    min: float | None
    max: float | None
    mean: float | None
    stddev: float | None
    p01: float | None
    p05: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True)
class EpisodeOverview:
    episode_index: int
    frame_count: int
    duration_seconds: float
    tasks: tuple[str, ...]
    declared_fps: float
    timestamp_derived_fps: float | None
    video_header_frame_count: int | None
    video_header_average_fps: float | None


@dataclass(frozen=True)
class SchemaFieldDiff:
    field: str
    episode_index: int
    baseline_dtype: str
    dtype: str
    baseline_lengths: tuple[int, ...] | None
    lengths: tuple[int, ...] | None


@dataclass(frozen=True)
class MotionEvent:
    """A frame where one pooled metric exceeds the statistical threshold --
    evidence of unusual motion, not by itself evidence of a data defect. See
    the module docstring's three-tier evidence system.

    ``tier`` starts as ``"motion"`` and is only ever promoted to
    ``"suspicious"`` by ``promote_suspicious_events`` (never demoted), when a
    second, independent signal corroborates this exact frame.
    """

    episode_index: int
    frame_index: int
    metric: str
    value: float
    robust_z: float
    tier: str = "motion"
    corroboration: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedIssue:
    """A hard invariant violation -- physically or format-impossible, not a
    statistical judgment call. ``frame_index`` is ``None`` for episode-level
    issues (e.g. a schema mismatch) that aren't located at one frame."""

    kind: str
    episode_index: int
    frame_index: int | None
    detail: str


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_distribution(values: Sequence[float]) -> DistributionSummary:
    """Count/mean/stddev/quantile profile for one continuous series.

    NaN/Inf are counted but excluded from the finite-only statistics below, same
    convention as ``scripts/inspect_phase2.py``'s ``numeric_summary``.
    """
    numeric = [float(v) for v in values]
    finite = sorted(v for v in numeric if math.isfinite(v))
    nan_count = sum(math.isnan(v) for v in numeric)
    inf_count = sum(math.isinf(v) for v in numeric)
    if not finite:
        return DistributionSummary(
            count=len(numeric),
            finite_count=0,
            nan_count=nan_count,
            inf_count=inf_count,
            min=None,
            max=None,
            mean=None,
            stddev=None,
            p01=None,
            p05=None,
            p25=None,
            median=None,
            p75=None,
            p95=None,
            p99=None,
        )
    return DistributionSummary(
        count=len(numeric),
        finite_count=len(finite),
        nan_count=nan_count,
        inf_count=inf_count,
        min=finite[0],
        max=finite[-1],
        mean=statistics.fmean(finite),
        stddev=statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        p01=_percentile(finite, 0.01),
        p05=_percentile(finite, 0.05),
        p25=_percentile(finite, 0.25),
        median=_percentile(finite, 0.5),
        p75=_percentile(finite, 0.75),
        p95=_percentile(finite, 0.95),
        p99=_percentile(finite, 0.99),
    )


def compute_histogram(values: Sequence[float], bin_count: int = 20) -> dict[str, list[float]]:
    """Fixed-width histogram over the finite values in ``values``.

    Precomputed server-side (rather than shipping every raw per-frame value to the
    browser) so the visualizer's distribution charts show a genuine binned histogram,
    not a shape guessed from quantiles alone.
    """
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return {"bin_edges": [], "counts": []}
    low, high = min(finite), max(finite)
    if low == high:
        return {"bin_edges": [low, high], "counts": [len(finite)]}
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for value in finite:
        index = min(int((value - low) / width), bin_count - 1)
        counts[index] += 1
    bin_edges = [low + i * width for i in range(bin_count + 1)]
    return {"bin_edges": bin_edges, "counts": counts}


def quaternion_norms(quaternion_rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """``||q||`` per frame for stored ``[ts_ns, x, y, z, w]`` quaternion rows.

    A well-formed unit quaternion has norm 1; this is a numeric QA check, not a
    rendering claim, so it operates on raw rows directly rather than through
    ``semantics.parse_camera_pose`` (which also requires a position row).
    """
    norms = []
    for row in quaternion_rows:
        x, y, z, w = row[1], row[2], row[3], row[4]
        norms.append(math.sqrt(x * x + y * y + z * z + w * w))
    return tuple(norms)


# Tolerance for flagging a quaternion (camera or per-keypoint) as invalid --
# same order of magnitude as the floating-point deviation this dataset
# actually shows (~1e-7), generous enough not to flag ordinary float32
# rounding as a real defect. Shared by both quaternion fields so the two
# checks are symmetric, not two different bars for the same physical property.
QUATERNION_NORM_TOLERANCE = 1e-3


def keypoint_quaternion_max_deviation(quaternion_rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Per-frame max ``|norm - 1|`` across a row's 21 stacked (x,y,z,w) keypoint
    quaternions (``observation.*_keypoints_quaternion``, no leading ``ts_ns``).

    Same numeric-QA spirit as ``quaternion_norms`` above (which checks the single
    camera quaternion), extended to the 21-per-frame keypoint-orientation field --
    see ``semantics.py``'s module docstring for the empirical unit-norm
    verification this check is built on.
    """
    deviations = []
    for row in quaternion_rows:
        quaternions = semantics.parse_keypoint_quaternions(row)
        max_deviation = max(abs(math.sqrt(sum(c * c for c in q)) - 1.0) for q in quaternions)
        deviations.append(max_deviation)
    return tuple(deviations)


def mano_betas_nonzero_frame_count(mano_rows: Sequence[Sequence[float]]) -> int:
    """Count of frames where MANO shape parameters (``betas``) are not all exactly
    zero (``observation.*_hand_mano``).

    A data-completeness fact, not a correctness check: 0 across this dataset means
    the shape/personalization channel was never populated -- every frame assumes a
    fixed/mean hand shape -- not that anything is broken. See ``semantics.py``'s
    module docstring for the empirical basis.
    """
    return sum(1 for row in mano_rows if any(value != 0.0 for value in semantics.parse_mano_hand(row).betas))


def _l2_distance_point_sets(a: Sequence[Point3], b: Sequence[Point3]) -> float:
    return math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for i in range(len(a)) for k in range(3)))


def compute_lag_residuals(
    hands: Sequence[tuple[Sequence[tuple[Point3, ...] | None], Sequence[tuple[Point3, ...]]]],
    lags: Sequence[int] = (-2, -1, 0, 1, 2),
) -> dict[int, DistributionSummary]:
    """residual(action_i, track_{i+k}) for each lag k, pooled across all given hands.

    Each element of ``hands`` is ``(action_points_per_frame, track_points_per_frame)``
    for one hand -- already parsed via ``semantics.parse_action_targets``/
    ``parse_hand_tracks`` (``action_points_per_frame[i]`` is ``None`` at the dataset's
    documented terminal all-zero convention row). Comparing k=+1 against the other
    lags is what turns "action looks like next-frame track" from an assumption into
    a measured, falsifiable claim (see docs/analysis-report.md).
    """
    pooled: dict[int, list[float]] = {lag: [] for lag in lags}
    for action_points, track_points in hands:
        n = len(track_points)
        for lag in lags:
            for i in range(n):
                target = action_points[i]
                if target is None:
                    continue
                j = i + lag
                if 0 <= j < n:
                    pooled[lag].append(_l2_distance_point_sets(target, track_points[j]))
    return {lag: summarize_distribution(values) for lag, values in pooled.items()}


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation over paired samples; ``None`` if fewer than 2 pairs or no variance.

    Reported strictly as co-variation between two derived speed series -- never as a
    causal claim (see docs/analysis-report.md's cross-modal section).
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def compute_camera_hand_correlation(computed: metrics_module.EpisodeMetrics) -> dict[str, float | None]:
    def _paired(a: Sequence[float | None], b: Sequence[float | None]) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for x, y in zip(a, b):
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        return xs, ys

    left_x, left_y = _paired(computed.camera_speed, computed.left_hand_speed)
    right_x, right_y = _paired(computed.camera_speed, computed.right_hand_speed)
    return {
        "left_hand": pearson_correlation(left_x, left_y),
        "right_hand": pearson_correlation(right_x, right_y),
    }


def robust_z_scores(values: Sequence[float]) -> tuple[float | None, ...]:
    """``(x - median) / (1.4826 * MAD)``; ``None`` where MAD is 0 (score undefined)."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return tuple(None for _ in values)
    median = statistics.median(finite)
    mad = statistics.median(abs(v - median) for v in finite)
    if mad == 0:
        return tuple(None for _ in values)
    scale = 1.4826 * mad
    return tuple((v - median) / scale if math.isfinite(v) else None for v in values)


def detect_motion_events(
    named_series: dict[str, Sequence[tuple[int, int, float]]],
    threshold: float = 6.0,
) -> list[MotionEvent]:
    """Robust-z motion events per metric, pooled across whatever episodes are given.

    ``named_series`` maps a metric name to a list of ``(episode_index, frame_index,
    value)`` triples already pooled across episodes. A flagged point is evidence of
    *statistically unusual motion* at that frame (e.g. via the visualizer's
    episode/frame deep link) -- never by itself an assertion that the frame is
    corrupted; the threshold is a display heuristic, not a physical-unit fact. See
    ``promote_suspicious_events`` for the second tier that adds corroborating
    evidence before a frame is worth flagging as more than "fast motion happened."
    """
    events: list[MotionEvent] = []
    for metric, triples in named_series.items():
        values = [value for _, _, value in triples]
        z_scores = robust_z_scores(values)
        for (episode_index, frame_index, value), z in zip(triples, z_scores):
            if z is not None and abs(z) > threshold:
                events.append(
                    MotionEvent(
                        episode_index=episode_index,
                        frame_index=frame_index,
                        metric=metric,
                        value=value,
                        robust_z=z,
                    )
                )
    events.sort(key=lambda e: abs(e.robust_z), reverse=True)
    return events


def promote_suspicious_events(
    events: Sequence[MotionEvent], verified_issues: Sequence[VerifiedIssue]
) -> list[MotionEvent]:
    """Promote a motion event to ``tier="suspicious"`` when a second, independent
    signal corroborates the exact same frame -- either another motion event
    (a different metric, or the same metric from the other hand: two hands
    "teleporting" on the same frame falls out of this one rule for free) or a
    verified issue located at that frame. Never invents new evidence; only
    re-labels events using evidence this module already computed elsewhere.
    """
    hits_per_frame = Counter((event.episode_index, event.frame_index) for event in events)
    issue_frames = {
        (issue.episode_index, issue.frame_index) for issue in verified_issues if issue.frame_index is not None
    }

    promoted: list[MotionEvent] = []
    for event in events:
        key = (event.episode_index, event.frame_index)
        reasons: list[str] = []
        if hits_per_frame[key] >= 2:
            reasons.append(f"{hits_per_frame[key]} motion-event signals fired at this frame")
        if key in issue_frames:
            reasons.append("coincides with a verified data-quality issue at this frame")
        if reasons:
            promoted.append(replace(event, tier="suspicious", corroboration=tuple(reasons)))
        else:
            promoted.append(event)
    return promoted


def _video_header_probe(path: Path) -> tuple[int | None, float | None]:
    """(header_frame_count, average_fps) from container/stream metadata only -- no decode."""
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if not streams:
            return None, None
        stream = streams[0]
        average_rate: Fraction | None = stream.average_rate
        return stream.frames, (float(average_rate) if average_rate is not None else None)


def compute_episode_overview(dataset: LeRobotDataset, episode_index: int) -> EpisodeOverview:
    timing = dataset.get_timing(episode_index)
    episode_meta = dataset.get_episode_metadata(episode_index)
    duration = timing.timestamp_end - timing.timestamp_start
    timestamp_derived_fps = (timing.frame_count - 1) / duration if timing.frame_count > 1 and duration > 0 else None

    video_header_frame_count: int | None = None
    video_header_average_fps: float | None = None
    if episode_meta.video_keys:
        try:
            resource = dataset.get_video_resource(episode_index, episode_meta.video_keys[0])
            video_path = dataset.resolve_video_path(resource)
            video_header_frame_count, video_header_average_fps = _video_header_probe(video_path)
        except Exception:
            pass  # video header probe is best-effort metadata, not required for the overview

    return EpisodeOverview(
        episode_index=episode_index,
        frame_count=timing.frame_count,
        duration_seconds=duration,
        tasks=episode_meta.tasks,
        declared_fps=dataset.metadata.fps,
        timestamp_derived_fps=timestamp_derived_fps,
        video_header_frame_count=video_header_frame_count,
        video_header_average_fps=video_header_average_fps,
    )


def audit_schema_consistency(dataset: LeRobotDataset) -> dict[str, Any]:
    """Compare each episode's observed field dtype/shape against episode 0's (the baseline).

    With only one local episode today this necessarily reports zero differences by
    construction (there is nothing to compare against) -- see the Known Limitations
    note in docs/analysis-report.md.
    """
    episode_indices = dataset.list_episode_indices()
    if not episode_indices:
        return {"baseline_episode": None, "differences": []}
    baseline_index = episode_indices[0]
    baseline_schemas = dataset.get_feature_schemas(baseline_index)
    differences: list[SchemaFieldDiff] = []
    for episode_index in episode_indices[1:]:
        schemas = dataset.get_feature_schemas(episode_index)
        all_fields = set(baseline_schemas) | set(schemas)
        for field in sorted(all_fields):
            baseline_schema = baseline_schemas.get(field)
            schema = schemas.get(field)
            baseline_dtype = baseline_schema.physical_arrow_dtype if baseline_schema else "<missing>"
            dtype = schema.physical_arrow_dtype if schema else "<missing>"
            baseline_lengths = baseline_schema.observed_lengths if baseline_schema else None
            lengths = schema.observed_lengths if schema else None
            if dtype != baseline_dtype or lengths != baseline_lengths:
                differences.append(
                    SchemaFieldDiff(
                        field=field,
                        episode_index=episode_index,
                        baseline_dtype=baseline_dtype,
                        dtype=dtype,
                        baseline_lengths=baseline_lengths,
                        lengths=lengths,
                    )
                )
    return {
        "baseline_episode": baseline_index,
        "differences": [vars(d) for d in differences],
    }


def select_target_frames(frame_indices: tuple[int, ...], seed: int, random_count: int = 10) -> list[tuple[str, int]]:
    """Return (label, frame_index) pairs: structural positions plus seeded-random ones."""
    n = len(frame_indices)
    structural_positions = {
        "first": 0,
        "p25": round(0.25 * (n - 1)),
        "p50": round(0.50 * (n - 1)),
        "p75": round(0.75 * (n - 1)),
        "last": n - 1,
    }
    targets: list[tuple[str, int]] = [
        (label, frame_indices[position]) for label, position in structural_positions.items()
    ]
    used_positions = set(structural_positions.values())
    rng = random.Random(seed)
    remaining = [position for position in range(n) if position not in used_positions]
    random_positions = rng.sample(remaining, min(random_count, len(remaining)))
    for index, position in enumerate(sorted(random_positions)):
        targets.append((f"random_{index}", frame_indices[position]))
    return targets


def find_nearest_video_frame(
    container: av.container.InputContainer, stream: av.video.stream.VideoStream, target_seconds: float
) -> float | None:
    """Seek near target_seconds and decode forward to the frame with closest PTS."""
    time_base: Fraction = stream.time_base
    target_pts = int(round(target_seconds / time_base))
    seek_pts = max(0, target_pts - int(round(1.0 / time_base)))
    container.seek(seek_pts, backward=True, any_frame=False, stream=stream)

    best_time: float | None = None
    best_diff: float | None = None
    safety_margin_seconds = 0.5
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * time_base)
        diff = abs(frame_time - target_seconds)
        if best_diff is None or diff < best_diff:
            best_time = frame_time
            best_diff = diff
        if frame_time > target_seconds + safety_margin_seconds:
            break
    return best_time


def compute_sync_validation(
    dataset: LeRobotDataset, episode_index: int, seed: int, random_count: int
) -> dict[str, Any]:
    """Stored ``timestamp`` vs. decoded video presentation time, for a sampled set of frames.

    Moved here from ``scripts/validate_sync.py`` (Phase 4.1) so it can run once per
    episode inside a dataset-wide batch instead of being copy-pasted; the script now
    just calls this and adds its own CLI-level wrapper fields.
    """
    timing = dataset.get_timing(episode_index)
    intervals = [right - left for left, right in zip(timing.timestamps, timing.timestamps[1:])]
    median_interval = statistics.median(intervals) if intervals else None

    video_resource = dataset.get_video_resource(episode_index)
    video_path = dataset.resolve_video_path(video_resource)

    targets = select_target_frames(timing.frame_indices, seed=seed, random_count=random_count)

    results = []
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        for label, frame_index in targets:
            frame = dataset.get_frame(episode_index, frame_index)
            video_media_time = find_nearest_video_frame(container, stream, frame.timestamp)
            sync_error = video_media_time - frame.timestamp if video_media_time is not None else None
            results.append(
                {
                    "label": label,
                    "frame_index": frame_index,
                    "stored_timestamp_seconds": frame.timestamp,
                    "stored_timestamp_ns": frame.timestamp_ns,
                    "video_media_time_seconds": video_media_time,
                    "sync_error_seconds": sync_error,
                    "abs_sync_error_seconds": abs(sync_error) if sync_error is not None else None,
                }
            )

    abs_errors = [row["abs_sync_error_seconds"] for row in results if row["abs_sync_error_seconds"] is not None]
    summary = {
        "target_count": len(results),
        "resolved_count": len(abs_errors),
        "max_abs_sync_error_seconds": max(abs_errors) if abs_errors else None,
        "mean_abs_sync_error_seconds": statistics.fmean(abs_errors) if abs_errors else None,
        "median_frame_interval_seconds": median_interval,
    }
    return {
        "episode_index": episode_index,
        "video_path": video_path.as_posix(),
        "frame_count": timing.frame_count,
        "results": results,
        "summary": summary,
    }


def detect_nan_or_inf_issues(named_series: dict[str, Sequence[tuple[int, int, float]]]) -> list[VerifiedIssue]:
    """Every metric value that failed to be a finite number -- a hard invariant
    violation (a measurement is a real number or it isn't), independent of
    whether it would also clear the motion-event statistical threshold.
    ``robust_z_scores`` silently maps a non-finite value to ``None`` (excluded
    from motion-event detection), so without this check a NaN/Inf reading
    would vanish from every report without a trace.
    """
    issues: list[VerifiedIssue] = []
    for metric, triples in named_series.items():
        for episode_index, frame_index, value in triples:
            if not math.isfinite(value):
                issues.append(
                    VerifiedIssue(
                        kind="nan_or_inf_value",
                        episode_index=episode_index,
                        frame_index=frame_index,
                        detail=f"{metric} = {value!r} (not finite)",
                    )
                )
    return issues


def detect_schema_mismatch_issues(schema_audit: dict[str, Any]) -> list[VerifiedIssue]:
    """Re-surface ``audit_schema_consistency``'s findings as verified issues --
    a dtype/length mismatch against the baseline episode is a hard structural
    fact, not a statistical judgment call.
    """
    return [
        VerifiedIssue(
            kind="schema_mismatch",
            episode_index=diff["episode_index"],
            frame_index=None,
            detail=(
                f"{diff['field']}: baseline dtype {diff['baseline_dtype']!r} "
                f"(lengths {diff['baseline_lengths']}) vs. {diff['dtype']!r} "
                f"(lengths {diff['lengths']})"
            ),
        )
        for diff in schema_audit["differences"]
    ]


def detect_timestamp_issues(
    episode_index: int, sync_diagnostics: metrics_module.SyncDiagnostics
) -> list[VerifiedIssue]:
    """Timestamp regressions, duplicates, large gaps, and frame-index
    discontinuities -- all hard sequence-integrity facts about the stored
    ``timestamp``/``frame_index`` columns, computed once per episode by
    ``metrics._sync_diagnostics`` and located here per offending frame.
    """
    issues: list[VerifiedIssue] = []
    for frame_index in sync_diagnostics.negative_interval_frame_indices:
        issues.append(
            VerifiedIssue(
                kind="timestamp_regression",
                episode_index=episode_index,
                frame_index=frame_index,
                detail="timestamp is earlier than the previous frame's",
            )
        )
    for frame_index in sync_diagnostics.duplicate_timestamp_frame_indices:
        issues.append(
            VerifiedIssue(
                kind="timestamp_duplicate",
                episode_index=episode_index,
                frame_index=frame_index,
                detail="timestamp repeats an earlier frame's exactly",
            )
        )
    for frame_index in sync_diagnostics.large_interval_frame_indices:
        issues.append(
            VerifiedIssue(
                kind="timestamp_large_gap",
                episode_index=episode_index,
                frame_index=frame_index,
                detail=(
                    "interval before this frame exceeds 1.5x the median "
                    f"({sync_diagnostics.median_interval_seconds:.6f}s)"
                ),
            )
        )
    for frame_index in sync_diagnostics.frame_index_discontinuity_frame_indices:
        issues.append(
            VerifiedIssue(
                kind="frame_index_discontinuity",
                episode_index=episode_index,
                frame_index=frame_index,
                detail="frame_index does not follow the previous row's by exactly 1",
            )
        )
    return issues


# A sampled sync check that's off by more than half a frame interval is
# unambiguously mis-synced, not just imprecise -- half a frame is already
# more slack than the ~microsecond errors this dataset actually shows.
VIDEO_SYNC_ERROR_FRACTION_OF_INTERVAL = 0.5


def detect_video_sync_issues(episode_index: int, sync_validation: dict[str, Any]) -> list[VerifiedIssue]:
    """Sampled video/timestamp pairs whose error exceeds a fixed fraction of one
    frame interval -- a hard tolerance, not a statistical outlier judgment.
    Only covers the frames ``compute_sync_validation`` actually sampled (5
    structural + N random per episode), not every frame in the episode.
    """
    median_interval = sync_validation["summary"]["median_frame_interval_seconds"]
    if not median_interval:
        return []
    tolerance = median_interval * VIDEO_SYNC_ERROR_FRACTION_OF_INTERVAL
    issues: list[VerifiedIssue] = []
    for row in sync_validation["results"]:
        error = row["abs_sync_error_seconds"]
        if error is not None and error > tolerance:
            issues.append(
                VerifiedIssue(
                    kind="video_timestamp_desync",
                    episode_index=episode_index,
                    frame_index=row["frame_index"],
                    detail=(
                        f"sampled sync error {error:.6f}s exceeds {tolerance:.6f}s "
                        "(half the median frame interval)"
                    ),
                )
            )
    return issues


def detect_quaternion_violation_issues(
    episode_index: int, field_label: str, deviations: Sequence[float], frame_indices: Sequence[int]
) -> list[VerifiedIssue]:
    """Frames where a quaternion's norm deviates from 1.0 past
    ``QUATERNION_NORM_TOLERANCE`` -- shared by the single camera quaternion and
    the 21-per-frame keypoint quaternions, since both are the same physical
    validity check applied to a different field.
    """
    return [
        VerifiedIssue(
            kind="quaternion_norm_violation",
            episode_index=episode_index,
            frame_index=frame_index,
            detail=f"{field_label} |norm - 1| = {deviation:.6g} exceeds tolerance {QUATERNION_NORM_TOLERANCE}",
        )
        for frame_index, deviation in zip(frame_indices, deviations)
        if deviation > QUATERNION_NORM_TOLERANCE
    ]


def detect_terminal_sentinel_issues(
    episode_index: int,
    field_label: str,
    action_points: Sequence[tuple[Point3, ...] | None],
    frame_indices: Sequence[int],
) -> list[VerifiedIssue]:
    """The dataset's documented convention (``semantics.parse_action_targets``)
    is that an all-zero action-target row -- parsed as ``points=None`` -- marks
    only the episode's terminal frame. An all-zero row anywhere else violates
    that documented convention: structurally identical to "no target", but in
    a position the convention doesn't license, regardless of how a consumer
    downstream chooses to interpret it.
    """
    if not frame_indices:
        return []
    last_frame_index = frame_indices[-1]
    return [
        VerifiedIssue(
            kind="terminal_sentinel_misplaced",
            episode_index=episode_index,
            frame_index=frame_index,
            detail=f"{field_label}: all-zero terminal-convention row on a non-terminal frame",
        )
        for frame_index, points in zip(frame_indices, action_points)
        if points is None and frame_index != last_frame_index
    ]


_ANALYSIS_FIELDS = (
    "left_hand/tracks",
    "right_hand/tracks",
    "action.left_hand_tracks",
    "action.right_hand_tracks",
    "base_0_camera/position",
    "base_0_camera/quaternion_xyzw",
    "observation.left_hand_mano",
    "observation.right_hand_mano",
    "observation.left_keypoints_quaternion",
    "observation.right_keypoints_quaternion",
)


def run_full_analysis(
    dataset_root: str | Path,
    *,
    seed: int = 20260830,
    random_count: int = 10,
    motion_event_threshold: float = 6.0,
) -> dict[str, Any]:
    """Run the full six-layer analysis over every episode in ``dataset_root``.

    Returns one JSON-serializable structure: per-episode overview/statistics/QA,
    pooled global distributions, motion events, and verified data-quality issues
    (see the module docstring's three-tier evidence system). This is what
    ``scripts/analyze_dataset.py`` writes to ``artifacts/analysis/`` and what
    ``GET /api/analysis`` in the visualizer serves.
    """
    dataset = LeRobotDataset(dataset_root)
    episode_indices = dataset.list_episode_indices()

    episodes_payload: list[dict[str, Any]] = []
    pooled_hand_speed: list[tuple[int, int, float]] = []
    pooled_hand_span: list[tuple[int, int, float]] = []
    pooled_camera_speed: list[tuple[int, int, float]] = []
    pooled_quaternion_norm: list[tuple[int, int, float]] = []
    pooled_action_residual: list[tuple[int, int, float]] = []
    pooled_articulation_rate: list[tuple[int, int, float]] = []
    pooled_keypoint_quaternion_deviation: list[tuple[int, int, float]] = []
    pooled_camera_quaternion_deviation: list[tuple[int, int, float]] = []
    total_betas_nonzero_frame_count = 0
    lag_hands_accumulator: list[tuple[Sequence[tuple[Point3, ...] | None], Sequence[tuple[Point3, ...]]]] = []
    verified_issues: list[VerifiedIssue] = []

    for episode_index in episode_indices:
        overview = compute_episode_overview(dataset, episode_index)
        series = dataset.get_series(episode_index, fields=_ANALYSIS_FIELDS)
        computed = metrics_module.compute_episode_metrics(
            series.frame_indices,
            series.timestamps,
            series.values["left_hand/tracks"],
            series.values["right_hand/tracks"],
            series.values["action.left_hand_tracks"],
            series.values["action.right_hand_tracks"],
            series.values["base_0_camera/position"],
            series.values["observation.left_hand_mano"],
            series.values["observation.right_hand_mano"],
        )
        norms = quaternion_norms(series.values["base_0_camera/quaternion_xyzw"])
        camera_quaternion_deviation = tuple(abs(norm - 1.0) for norm in norms)
        correlation = compute_camera_hand_correlation(computed)
        sync_validation = compute_sync_validation(dataset, episode_index, seed, random_count)

        left_keypoint_deviation = keypoint_quaternion_max_deviation(series.values["observation.left_keypoints_quaternion"])
        right_keypoint_deviation = keypoint_quaternion_max_deviation(series.values["observation.right_keypoints_quaternion"])
        left_betas_nonzero = mano_betas_nonzero_frame_count(series.values["observation.left_hand_mano"])
        right_betas_nonzero = mano_betas_nonzero_frame_count(series.values["observation.right_hand_mano"])
        total_betas_nonzero_frame_count += left_betas_nonzero + right_betas_nonzero

        verified_issues.extend(detect_timestamp_issues(episode_index, computed.sync_diagnostics))
        verified_issues.extend(detect_video_sync_issues(episode_index, sync_validation))
        verified_issues.extend(
            detect_quaternion_violation_issues(
                episode_index, "base_0_camera/quaternion_xyzw", camera_quaternion_deviation, series.frame_indices
            )
        )
        verified_issues.extend(
            detect_quaternion_violation_issues(
                episode_index,
                "observation.left_keypoints_quaternion",
                left_keypoint_deviation,
                series.frame_indices,
            )
        )
        verified_issues.extend(
            detect_quaternion_violation_issues(
                episode_index,
                "observation.right_keypoints_quaternion",
                right_keypoint_deviation,
                series.frame_indices,
            )
        )

        for frame_index, value in zip(series.frame_indices, computed.left_hand_speed):
            if value is not None:
                pooled_hand_speed.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.right_hand_speed):
            if value is not None:
                pooled_hand_speed.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.left_hand_span):
            pooled_hand_span.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.right_hand_span):
            pooled_hand_span.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.camera_speed):
            if value is not None:
                pooled_camera_speed.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, norms):
            pooled_quaternion_norm.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.left_action_residual):
            if value is not None:
                pooled_action_residual.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.right_action_residual):
            if value is not None:
                pooled_action_residual.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.left_hand_articulation_rate):
            if value is not None:
                pooled_articulation_rate.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, computed.right_hand_articulation_rate):
            if value is not None:
                pooled_articulation_rate.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, left_keypoint_deviation):
            pooled_keypoint_quaternion_deviation.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, right_keypoint_deviation):
            pooled_keypoint_quaternion_deviation.append((episode_index, frame_index, value))
        for frame_index, value in zip(series.frame_indices, camera_quaternion_deviation):
            pooled_camera_quaternion_deviation.append((episode_index, frame_index, value))

        left_action_points = [semantics.parse_action_targets(row).points for row in series.values["action.left_hand_tracks"]]
        left_track_points = [semantics.parse_hand_tracks(row).points for row in series.values["left_hand/tracks"]]
        right_action_points = [semantics.parse_action_targets(row).points for row in series.values["action.right_hand_tracks"]]
        right_track_points = [semantics.parse_hand_tracks(row).points for row in series.values["right_hand/tracks"]]
        lag_hands_accumulator.append((left_action_points, left_track_points))
        lag_hands_accumulator.append((right_action_points, right_track_points))
        verified_issues.extend(
            detect_terminal_sentinel_issues(
                episode_index, "action.left_hand_tracks", left_action_points, series.frame_indices
            )
        )
        verified_issues.extend(
            detect_terminal_sentinel_issues(
                episode_index, "action.right_hand_tracks", right_action_points, series.frame_indices
            )
        )
        episode_lag_residuals = compute_lag_residuals(
            [(left_action_points, left_track_points), (right_action_points, right_track_points)]
        )

        episodes_payload.append(
            {
                "overview": vars(overview),
                "sync_diagnostics": vars(computed.sync_diagnostics),
                "sync_validation": sync_validation,
                "hand_speed": {
                    "left": vars(summarize_distribution([v for v in computed.left_hand_speed if v is not None])),
                    "right": vars(summarize_distribution([v for v in computed.right_hand_speed if v is not None])),
                },
                "hand_span": {
                    "left": vars(summarize_distribution(computed.left_hand_span)),
                    "right": vars(summarize_distribution(computed.right_hand_span)),
                },
                "camera_speed": vars(summarize_distribution([v for v in computed.camera_speed if v is not None])),
                "quaternion_norm": vars(summarize_distribution(norms)),
                "articulation_rate": {
                    "left": vars(summarize_distribution([v for v in computed.left_hand_articulation_rate if v is not None])),
                    "right": vars(summarize_distribution([v for v in computed.right_hand_articulation_rate if v is not None])),
                },
                "mano_quality": {
                    "left_betas_nonzero_frame_count": left_betas_nonzero,
                    "right_betas_nonzero_frame_count": right_betas_nonzero,
                    "left_keypoint_quaternion_max_deviation": vars(summarize_distribution(left_keypoint_deviation)),
                    "right_keypoint_quaternion_max_deviation": vars(summarize_distribution(right_keypoint_deviation)),
                },
                "camera_hand_correlation": correlation,
                "lag_residuals": {str(lag): vars(summary) for lag, summary in episode_lag_residuals.items()},
            }
        )

    global_lag_residuals = compute_lag_residuals(lag_hands_accumulator)
    named_series = {
        "hand_speed": pooled_hand_speed,
        "hand_span": pooled_hand_span,
        "camera_speed": pooled_camera_speed,
        "quaternion_norm": pooled_quaternion_norm,
        "action_residual_k1": pooled_action_residual,
        "articulation_rate": pooled_articulation_rate,
    }

    schema_audit = audit_schema_consistency(dataset)
    verified_issues.extend(detect_schema_mismatch_issues(schema_audit))
    verified_issues.extend(detect_nan_or_inf_issues(named_series))
    verified_issues.sort(key=lambda issue: (issue.episode_index, issue.frame_index if issue.frame_index is not None else -1, issue.kind))

    motion_events = promote_suspicious_events(
        detect_motion_events(named_series, threshold=motion_event_threshold), verified_issues
    )

    keypoint_quaternion_invalid_frame_count = sum(
        1 for _, _, deviation in pooled_keypoint_quaternion_deviation if deviation > QUATERNION_NORM_TOLERANCE
    )
    camera_quaternion_invalid_frame_count = sum(
        1 for _, _, deviation in pooled_camera_quaternion_deviation if deviation > QUATERNION_NORM_TOLERANCE
    )

    return {
        "dataset_root": Path(dataset_root).resolve().as_posix(),
        "tool_versions": {"av": av.__version__},
        "seed": seed,
        "motion_event_threshold": motion_event_threshold,
        "episode_count": len(episode_indices),
        "episodes": episodes_payload,
        "schema_audit": schema_audit,
        "global": {
            "hand_speed": vars(summarize_distribution([v for _, _, v in pooled_hand_speed])),
            "hand_span": vars(summarize_distribution([v for _, _, v in pooled_hand_span])),
            "camera_speed": vars(summarize_distribution([v for _, _, v in pooled_camera_speed])),
            "quaternion_norm": vars(summarize_distribution([v for _, _, v in pooled_quaternion_norm])),
            "action_residual_k1": vars(summarize_distribution([v for _, _, v in pooled_action_residual])),
            "articulation_rate": vars(summarize_distribution([v for _, _, v in pooled_articulation_rate])),
            "lag_residuals": {str(lag): vars(summary) for lag, summary in global_lag_residuals.items()},
        },
        "histograms": {
            "hand_speed": compute_histogram([v for _, _, v in pooled_hand_speed]),
            "camera_speed": compute_histogram([v for _, _, v in pooled_camera_speed]),
            "action_residual_k1": compute_histogram([v for _, _, v in pooled_action_residual]),
            "articulation_rate": compute_histogram([v for _, _, v in pooled_articulation_rate]),
        },
        # Data-quality/schema facts for the two observation.* fields that are
        # parsed (semantics.parse_keypoint_quaternions/parse_mano_hand) and fed
        # into articulation_rate above but otherwise never surfaced -- see
        # semantics.py's module docstring for what these facts mean.
        "observation_field_quality": {
            "keypoint_quaternion_max_deviation": vars(
                summarize_distribution([v for _, _, v in pooled_keypoint_quaternion_deviation])
            ),
            "keypoint_quaternion_invalid_frame_count": keypoint_quaternion_invalid_frame_count,
            "camera_quaternion_max_deviation": vars(
                summarize_distribution([v for _, _, v in pooled_camera_quaternion_deviation])
            ),
            "camera_quaternion_invalid_frame_count": camera_quaternion_invalid_frame_count,
            "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE,
            "mano_betas_nonzero_frame_count": total_betas_nonzero_frame_count,
            "mano_betas_all_zero": total_betas_nonzero_frame_count == 0,
        },
        "motion_events": [vars(event) for event in motion_events],
        "verified_issues": [vars(issue) for issue in verified_issues],
        "limitations": {
            "episode_count": len(episode_indices),
            "single_episode": len(episode_indices) == 1,
            "note": (
                "Only one local episode is available; cross-episode comparisons "
                "(task balance, per-episode outlier ranking, schema drift) are "
                "structurally computed but statistically vacant with N=1."
                if len(episode_indices) == 1
                else None
            ),
        },
    }
