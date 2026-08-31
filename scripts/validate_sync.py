"""Phase 4.1 synchronization validation: stored timestamp vs. decoded video time.

For a deterministic set of target frames (first, 25%, 50%, 75%, last, plus 10
seeded-random frames), this script compares each frame's stored ``timestamp``
(from the Phase 3 access layer) against the presentation time of the nearest
decoded video frame (via ``av``), and reports the signed/absolute difference.

This does not modify the dataset and does not decode the whole video; it seeks
near each target time and decodes forward only far enough to find the closest
frame. Read-only, local-only, consistent with the rest of this project.

The core comparison (``select_target_frames``, ``find_nearest_video_frame``, and
the body of what used to be this script's own ``validate()``) now lives in
``lerobot_visualizer.analysis.compute_sync_validation`` so the same logic can run
once per episode inside ``scripts/analyze_dataset.py``'s dataset-wide batch. This
script is a thin CLI wrapper around it; its output format is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import av
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_visualizer import LeRobotDataset  # noqa: E402
from lerobot_visualizer.analysis import compute_sync_validation  # noqa: E402


def validate(dataset_root: Path, episode_index: int, seed: int, random_count: int) -> dict[str, Any]:
    dataset = LeRobotDataset(dataset_root)
    result = compute_sync_validation(dataset, episode_index, seed, random_count)
    return {
        "tool_versions": {"pyarrow": pa.__version__, "av": av.__version__},
        "dataset_root": Path(dataset_root).resolve().as_posix(),
        "seed": seed,
        **result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260830, help="Deterministic seed for the 10 random frames.")
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase-4-sync-validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args.dataset_root, args.episode_index, args.seed, args.random_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")

    header = f"{'label':<10} {'frame':>6} {'stored_ts':>12} {'video_time':>12} {'error_ms':>10}"
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        error_ms = row["sync_error_seconds"] * 1000 if row["sync_error_seconds"] is not None else float("nan")
        video_time = row["video_media_time_seconds"] if row["video_media_time_seconds"] is not None else float("nan")
        print(
            f"{row['label']:<10} {row['frame_index']:>6} {row['stored_timestamp_seconds']:>12.6f} "
            f"{video_time:>12.6f} {error_ms:>10.3f}"
        )
    print()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
