"""Flask app exposing the local access layer as a read-only JSON + video API.

This module is the only place in the project that touches Flask. It never
reinterprets raw Parquet fields itself -- frame/metrics payloads are shaped by
``semantics.py``/``metrics.py`` before being serialized to JSON, so the frontend
never sees raw field names like ``left_hand/tracks`` verbatim.

Binds to 127.0.0.1 by default (see ``scripts/run_visualizer.py``) since the
dataset is confidential; the built-in Werkzeug dev server used here is
single-user/local/dev-only, not for concurrent or production use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory

from . import metrics as metrics_module
from . import semantics
from .dataset import LeRobotDataset
from .errors import (
    DatasetAccessError,
    EpisodeNotFoundError,
    FieldNotFoundError,
    FrameNotFoundError,
    InvalidDatasetRootError,
    TimestampLookupError,
    TimestampOutOfRangeError,
    VideoNotFoundError,
    VideoReferenceError,
)

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
_ANALYSIS_SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "analysis" / "summary.json"

_ERROR_STATUS_CODES: dict[type[DatasetAccessError], int] = {
    EpisodeNotFoundError: 404,
    FrameNotFoundError: 404,
    VideoNotFoundError: 404,
    FieldNotFoundError: 400,
    TimestampOutOfRangeError: 400,
    TimestampLookupError: 400,
    VideoReferenceError: 404,
    InvalidDatasetRootError: 500,
}


def _status_for(error: DatasetAccessError) -> int:
    for error_type, status in _ERROR_STATUS_CODES.items():
        if isinstance(error, error_type):
            return status
    return 500


def _point_list(points: tuple[tuple[float, float, float], ...] | None) -> list[list[float]] | None:
    if points is None:
        return None
    return [list(point) for point in points]


def _frame_payload(dataset: LeRobotDataset, episode_index: int, frame_index: int) -> dict[str, Any]:
    frame = dataset.get_frame(episode_index, frame_index)
    semantic = semantics.parse_frame_semantics(frame.values)
    task_index = frame.values.get("task_index")
    task_label = None
    if task_index is not None:
        task_label = dataset.get_task_metadata(int(task_index)).task
    return {
        "episode_index": frame.episode_index,
        "frame_index": frame.frame_index,
        "timestamp": frame.timestamp,
        "timestamp_ns": frame.timestamp_ns,
        "task": task_label,
        "left_hand": {"points": _point_list(semantic.left_hand.points)},
        "right_hand": {"points": _point_list(semantic.right_hand.points)},
        "left_action_target": {"points": _point_list(semantic.left_action_target.points)},
        "right_action_target": {"points": _point_list(semantic.right_action_target.points)},
        "camera_pose": {
            "position": list(semantic.camera_pose.position),
            "quaternion_xyzw": list(semantic.camera_pose.quaternion_xyzw),
        },
        # Full raw stored values (everything get_frame() fetched, minus the
        # identifiers already surfaced above) -- a debugging escape hatch for the
        # Frame Inspector's "Raw Frame Data" drill-down. Deliberately nested under
        # its own key, separate from the semantic fields above, so this never
        # becomes the primary way the frontend reads frame data.
        "raw": dict(frame.values),
    }


def _sync_diagnostics_payload(diagnostics: metrics_module.SyncDiagnostics) -> dict[str, Any]:
    return {
        "frame_count": diagnostics.frame_count,
        "median_interval_seconds": diagnostics.median_interval_seconds,
        "large_interval_threshold_seconds": diagnostics.large_interval_threshold_seconds,
        "large_interval_count": diagnostics.large_interval_count,
        "duplicate_timestamp_count": diagnostics.duplicate_timestamp_count,
    }


def _metrics_payload(computed: metrics_module.EpisodeMetrics) -> dict[str, Any]:
    return {
        "frame_indices": list(computed.frame_indices),
        "timestamps": list(computed.timestamps),
        "left_hand_speed": list(computed.left_hand_speed),
        "right_hand_speed": list(computed.right_hand_speed),
        "left_hand_span": list(computed.left_hand_span),
        "right_hand_span": list(computed.right_hand_span),
        "camera_speed": list(computed.camera_speed),
        "left_action_residual": list(computed.left_action_residual),
        "right_action_residual": list(computed.right_action_residual),
        "left_hand_articulation_rate": list(computed.left_hand_articulation_rate),
        "right_hand_articulation_rate": list(computed.right_hand_articulation_rate),
        "sync_diagnostics": _sync_diagnostics_payload(computed.sync_diagnostics),
    }


def create_app(dataset_root: str | Path, *, analysis_summary_path: str | Path | None = None) -> Flask:
    dataset = LeRobotDataset(dataset_root)
    summary_path = Path(analysis_summary_path) if analysis_summary_path is not None else _ANALYSIS_SUMMARY_PATH
    app = Flask(__name__, static_folder=None)
    metrics_cache: dict[int, metrics_module.EpisodeMetrics] = {}

    def _episode_metrics(episode_index: int) -> metrics_module.EpisodeMetrics:
        if episode_index not in metrics_cache:
            series = dataset.get_series(
                episode_index,
                fields=(
                    "left_hand/tracks",
                    "right_hand/tracks",
                    "action.left_hand_tracks",
                    "action.right_hand_tracks",
                    "base_0_camera/position",
                    "observation.left_hand_mano",
                    "observation.right_hand_mano",
                ),
            )
            metrics_cache.clear()
            metrics_cache[episode_index] = metrics_module.compute_episode_metrics(
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
        return metrics_cache[episode_index]

    @app.errorhandler(DatasetAccessError)
    def _handle_dataset_error(error: DatasetAccessError):
        response = jsonify({"error": {"type": type(error).__name__, "message": str(error)}})
        response.status_code = _status_for(error)
        return response

    @app.get("/")
    def index():
        return send_file(_STATIC_DIR / "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        # send_from_directory (not a manually joined path) rejects traversal
        # attempts (e.g. "../pyproject.toml") with a 404 via Werkzeug's safe_join.
        return send_from_directory(_STATIC_DIR, filename)

    @app.get("/api/dataset")
    def dataset_info():
        return jsonify(
            {
                "robot_type": dataset.metadata.robot_type,
                "fps": dataset.metadata.fps,
                "codebase_version": dataset.metadata.codebase_version,
            }
        )

    @app.get("/api/episodes")
    def list_episodes():
        episodes = []
        for episode_index in dataset.list_episode_indices():
            episode_meta = dataset.get_episode_metadata(episode_index)
            timing = dataset.get_timing(episode_index)
            episodes.append(
                {
                    "episode_index": episode_index,
                    "frame_count": timing.frame_count,
                    "declared_length": episode_meta.declared_length,
                    "tasks": list(episode_meta.tasks),
                    "video_keys": list(episode_meta.video_keys),
                    "duration_seconds": timing.timestamp_end - timing.timestamp_start,
                }
            )
        return jsonify({"episodes": episodes})

    @app.get("/api/episodes/<int:episode_index>")
    def episode_summary(episode_index: int):
        episode_meta = dataset.get_episode_metadata(episode_index)
        timing = dataset.get_timing(episode_index)
        computed = _episode_metrics(episode_index)
        return jsonify(
            {
                "episode_index": episode_index,
                "frame_count": timing.frame_count,
                "declared_length": episode_meta.declared_length,
                "tasks": list(episode_meta.tasks),
                "video_keys": list(episode_meta.video_keys),
                "timestamp_start": timing.timestamp_start,
                "timestamp_end": timing.timestamp_end,
                "metadata_declared_fps": timing.metadata_declared_fps,
                "sync_diagnostics": _sync_diagnostics_payload(computed.sync_diagnostics),
            }
        )

    @app.get("/api/episodes/<int:episode_index>/frames/<int:frame_index>")
    def frame(episode_index: int, frame_index: int):
        return jsonify(_frame_payload(dataset, episode_index, frame_index))

    @app.get("/api/episodes/<int:episode_index>/nearest")
    def nearest(episode_index: int):
        timestamp = request.args.get("timestamp", type=float)
        if timestamp is None:
            return jsonify({"error": {"type": "TimestampLookupError", "message": "timestamp query parameter is required"}}), 400
        match = dataset.get_nearest_frame(episode_index, timestamp, out_of_range="clamp")
        return jsonify(
            {
                "frame_index": match.frame.frame_index,
                "timestamp": match.matched_timestamp,
                "requested_timestamp": match.requested_timestamp,
                "delta_seconds": match.delta_seconds,
                "exact": match.exact,
            }
        )

    @app.get("/api/episodes/<int:episode_index>/metrics")
    def episode_metrics_route(episode_index: int):
        return jsonify(_metrics_payload(_episode_metrics(episode_index)))

    @app.get("/api/episodes/<int:episode_index>/video")
    def video(episode_index: int):
        video_key = request.args.get("video_key")
        resource = dataset.get_video_resource(episode_index, video_key)
        path = dataset.resolve_video_path(resource)
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/analysis")
    def analysis_summary():
        # Precomputed by `scripts/analyze_dataset.py`, not recomputed per-request --
        # the batch analysis includes `av` decode work (per-episode sync validation),
        # the same cost concern that already keeps validate_sync.py a separate manual
        # step (see README.md "Run analysis").
        if not summary_path.is_file():
            return (
                jsonify(
                    {
                        "error": {
                            "type": "AnalysisNotComputed",
                            "message": (
                                "No analysis artifact found. Run "
                                "`python scripts/analyze_dataset.py --dataset-root <root>` first."
                            ),
                        }
                    }
                ),
                404,
            )
        return jsonify(json.loads(summary_path.read_text(encoding="utf-8")))

    @app.get("/analysis.html")
    def analysis_page():
        return send_file(_STATIC_DIR / "analysis.html")

    return app
