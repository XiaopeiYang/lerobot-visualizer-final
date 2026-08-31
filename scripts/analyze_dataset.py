"""Phase 5/6: dataset-wide statistical analysis, cross-modal QA, and a 3-tier
motion-event / verified-issue scan (see ``analysis.py``'s module docstring for
why "anomaly" was replaced with this tiering).

Runs ``lerobot_visualizer.analysis.run_full_analysis`` over every episode in the
local dataset root and writes:

- ``artifacts/analysis/summary.json`` -- the full nested result.
- ``artifacts/analysis/episode_statistics.csv`` -- one row per episode.
- ``artifacts/analysis/field_statistics.csv`` -- one row per (episode-or-global, metric).
- ``artifacts/analysis/motion_events.csv`` -- one row per statistically extreme frame (tier 1/2).
- ``artifacts/analysis/verified_issues.csv`` -- one row per hard invariant violation (tier 3).

Read-only, local-only, deterministic (fixed default seed), no network access --
consistent with the rest of this project. See docs/analysis-report.md for the
written report built from this output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_visualizer.analysis import run_full_analysis  # noqa: E402


def _write_episode_statistics_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "episode_index",
        "frame_count",
        "duration_seconds",
        "tasks",
        "declared_fps",
        "timestamp_derived_fps",
        "video_header_frame_count",
        "video_header_average_fps",
        "large_interval_count",
        "duplicate_timestamp_count",
        "sync_validation_max_abs_error_seconds",
        "camera_hand_correlation_left",
        "camera_hand_correlation_right",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in report["episodes"]:
            overview = episode["overview"]
            writer.writerow(
                {
                    "episode_index": overview["episode_index"],
                    "frame_count": overview["frame_count"],
                    "duration_seconds": overview["duration_seconds"],
                    "tasks": ";".join(overview["tasks"]),
                    "declared_fps": overview["declared_fps"],
                    "timestamp_derived_fps": overview["timestamp_derived_fps"],
                    "video_header_frame_count": overview["video_header_frame_count"],
                    "video_header_average_fps": overview["video_header_average_fps"],
                    "large_interval_count": episode["sync_diagnostics"]["large_interval_count"],
                    "duplicate_timestamp_count": episode["sync_diagnostics"]["duplicate_timestamp_count"],
                    "sync_validation_max_abs_error_seconds": episode["sync_validation"]["summary"][
                        "max_abs_sync_error_seconds"
                    ],
                    "camera_hand_correlation_left": episode["camera_hand_correlation"]["left_hand"],
                    "camera_hand_correlation_right": episode["camera_hand_correlation"]["right_hand"],
                }
            )


def _distribution_row(scope: str, metric: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {"scope": scope, "metric": metric, **summary}


def _write_field_statistics_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "scope",
        "metric",
        "count",
        "finite_count",
        "nan_count",
        "inf_count",
        "min",
        "max",
        "mean",
        "stddev",
        "p01",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
    ]
    rows: list[dict[str, Any]] = []
    for episode in report["episodes"]:
        scope = f"episode_{episode['overview']['episode_index']}"
        rows.append(_distribution_row(scope, "hand_speed_left", episode["hand_speed"]["left"]))
        rows.append(_distribution_row(scope, "hand_speed_right", episode["hand_speed"]["right"]))
        rows.append(_distribution_row(scope, "hand_span_left", episode["hand_span"]["left"]))
        rows.append(_distribution_row(scope, "hand_span_right", episode["hand_span"]["right"]))
        rows.append(_distribution_row(scope, "camera_speed", episode["camera_speed"]))
        rows.append(_distribution_row(scope, "quaternion_norm", episode["quaternion_norm"]))
        for lag, summary in episode["lag_residuals"].items():
            rows.append(_distribution_row(scope, f"action_track_residual_lag{lag}", summary))
    for metric in ("hand_speed", "hand_span", "camera_speed", "quaternion_norm", "action_residual_k1"):
        rows.append(_distribution_row("global", metric, report["global"][metric]))
    for lag, summary in report["global"]["lag_residuals"].items():
        rows.append(_distribution_row("global", f"action_track_residual_lag{lag}", summary))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_motion_events_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = ["episode_index", "frame_index", "metric", "value", "robust_z", "tier", "corroboration"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in report["motion_events"]:
            row = dict(event)
            row["corroboration"] = "; ".join(row["corroboration"])
            writer.writerow(row)


def _write_verified_issues_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = ["kind", "episode_index", "frame_index", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["verified_issues"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument("--motion-event-threshold", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_full_analysis(
        args.dataset_root,
        seed=args.seed,
        random_count=args.random_count,
        motion_event_threshold=args.motion_event_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_episode_statistics_csv(args.output_dir / "episode_statistics.csv", report)
    _write_field_statistics_csv(args.output_dir / "field_statistics.csv", report)
    _write_motion_events_csv(args.output_dir / "motion_events.csv", report)
    _write_verified_issues_csv(args.output_dir / "verified_issues.csv", report)

    print(f"Wrote {summary_path}")
    print(f"Wrote {args.output_dir / 'episode_statistics.csv'}")
    print(f"Wrote {args.output_dir / 'field_statistics.csv'}")
    print(f"Wrote {args.output_dir / 'motion_events.csv'}")
    print(f"Wrote {args.output_dir / 'verified_issues.csv'}")
    print()
    print(
        json.dumps(
            {
                "episode_count": report["episode_count"],
                "motion_event_count": len(report["motion_events"]),
                "suspicious_event_count": sum(1 for e in report["motion_events"] if e["tier"] == "suspicious"),
                "verified_issue_count": len(report["verified_issues"]),
                "schema_differences": len(report["schema_audit"]["differences"]),
                "limitations": report["limitations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
