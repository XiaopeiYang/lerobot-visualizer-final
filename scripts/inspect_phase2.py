"""Read-only Phase 2 schema and temporal inspection for a local LeRobot dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import av
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def finite_or_label(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(key): finite_or_label(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_label(item) for item in value]
    return value


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    finite = sorted(value for value in numeric if math.isfinite(value))
    result: dict[str, Any] = {
        "count": len(numeric),
        "finite_count": len(finite),
        "nan_count": sum(math.isnan(value) for value in numeric),
        "positive_infinity_count": sum(value == math.inf for value in numeric),
        "negative_infinity_count": sum(value == -math.inf for value in numeric),
    }
    if finite:
        result.update(
            {
                "min": finite[0],
                "max": finite[-1],
                "mean": statistics.fmean(finite),
                "median": statistics.median(finite),
                "stddev_population": statistics.pstdev(finite),
                "p01": percentile(finite, 0.01),
                "p05": percentile(finite, 0.05),
                "p95": percentile(finite, 0.95),
                "p99": percentile(finite, 0.99),
            }
        )
    return result


def representative_values(values: list[Any]) -> dict[str, Any]:
    if not values:
        return {}
    indices = sorted({0, len(values) // 2, len(values) - 1})
    return {str(index): values[index] for index in indices}


def list_profile(values: list[Any]) -> dict[str, Any]:
    non_null = [value for value in values if value is not None]
    lengths = [len(value) for value in non_null]
    flattened = [item for value in non_null for item in value]
    numeric_flattened = [item for item in flattened if isinstance(item, (int, float))]
    nested_null_count = sum(item is None for item in flattened)
    result: dict[str, Any] = {
        "observed_lengths": sorted(set(lengths)),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "nested_value_count": len(flattened),
        "nested_null_count": nested_null_count,
        "all_zero_row_count": sum(
            all(item == 0 for item in value if item is not None) for value in non_null
        ),
        "first_row_all_zero": bool(non_null and all(item == 0 for item in non_null[0] if item is not None)),
        "last_row_all_zero": bool(non_null and all(item == 0 for item in non_null[-1] if item is not None)),
    }
    if numeric_flattened:
        result["flattened_numeric_range"] = numeric_summary(numeric_flattened)
        if lengths and len(set(lengths)) == 1 and lengths[0] > 0:
            width = lengths[0]
            component_ranges = []
            for component in range(width):
                component_values = [
                    value[component]
                    for value in non_null
                    if value[component] is not None and isinstance(value[component], (int, float))
                ]
                summary = numeric_summary(component_values)
                component_ranges.append(
                    {
                        "component": component,
                        "min": summary.get("min"),
                        "max": summary.get("max"),
                        "finite_count": summary["finite_count"],
                        "nan_count": summary["nan_count"],
                        "positive_infinity_count": summary["positive_infinity_count"],
                        "negative_infinity_count": summary["negative_infinity_count"],
                    }
                )
            result["component_ranges"] = component_ranges
    return result


def column_profile(field: pa.Field, column: pa.ChunkedArray) -> dict[str, Any]:
    values = column.to_pylist()
    profile: dict[str, Any] = {
        "dtype": str(field.type),
        "nullable": field.nullable,
        "row_count": len(column),
        "null_count": column.null_count,
        "representative_values": representative_values(values),
    }
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type) or pa.types.is_fixed_size_list(field.type):
        profile["array"] = list_profile(values)
    else:
        non_null = [value for value in values if value is not None]
        if non_null and all(isinstance(value, (int, float)) for value in non_null):
            profile["numeric"] = numeric_summary(non_null)
        try:
            profile["distinct_count"] = len(pc.unique(column))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            profile["distinct_count"] = None
    return profile


def parquet_profile(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read()
    metadata = parquet_file.metadata
    profile = {
        "path": path.as_posix(),
        "file_size_bytes": path.stat().st_size,
        "row_count": table.num_rows,
        "column_count": table.num_columns,
        "row_group_count": metadata.num_row_groups,
        "created_by": metadata.created_by,
        "schema": str(table.schema),
        "columns": {
            field.name: column_profile(field, table.column(field.name)) for field in table.schema
        },
    }
    return table, profile


def json_structure(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "field_count": len(value),
            "fields": {key: json_structure(item) for key, item in value.items()},
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "item_types": sorted({json_structure(item)["type"] for item in value}),
        }
    if value is None:
        kind = "null"
    elif isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, int):
        kind = "integer"
    elif isinstance(value, float):
        kind = "number"
    else:
        kind = "string"
    return {"type": kind, "value": value}


def csv_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0]) if rows else []
    inferred_table = pacsv.read_csv(path)
    return {
        "path": path.as_posix(),
        "file_size_bytes": path.stat().st_size,
        "row_count": len(rows),
        "column_count": len(columns),
        "inferred_arrow_schema": str(inferred_table.schema),
        "columns": {
            column: {
                "lexical_dtype": "string",
                "inferred": column_profile(
                    inferred_table.schema.field(column), inferred_table.column(column)
                ),
                "null_or_empty_count": sum(row[column] == "" for row in rows),
                "distinct_count": len({row[column] for row in rows}),
                "representative_values": representative_values([row[column] for row in rows]),
            }
            for column in columns
        },
    }


def intervals(values: list[int | float]) -> list[float]:
    return [float(right) - float(left) for left, right in zip(values, values[1:])]


def temporal_profile(frame_indices: list[int], timestamps: list[float], timestamps_ns: list[int]) -> dict[str, Any]:
    timestamp_intervals = intervals(timestamps)
    timestamp_ns_intervals = intervals(timestamps_ns)
    median_interval = statistics.median(timestamp_intervals) if timestamp_intervals else None
    frame_differences = intervals(frame_indices)
    timestamp_anomalies = []
    for offset, delta in enumerate(timestamp_intervals):
        if delta <= 0 or (median_interval and delta > median_interval * 1.5):
            timestamp_anomalies.append(
                {
                    "left_row": offset,
                    "right_row": offset + 1,
                    "left_frame_index": frame_indices[offset],
                    "right_frame_index": frame_indices[offset + 1],
                    "left_timestamp": timestamps[offset],
                    "right_timestamp": timestamps[offset + 1],
                    "interval_seconds": delta,
                }
            )
    timestamp_ns_seconds = [value / 1_000_000_000 for value in timestamps_ns]
    timestamp_agreement = [abs(left - right) for left, right in zip(timestamps, timestamp_ns_seconds)]
    return {
        "row_count": len(frame_indices),
        "frame_index": {
            "min": min(frame_indices),
            "max": max(frame_indices),
            "unique_count": len(set(frame_indices)),
            "nondecreasing": all(delta >= 0 for delta in frame_differences),
            "strictly_increasing": all(delta > 0 for delta in frame_differences),
            "duplicate_count": len(frame_indices) - len(set(frame_indices)),
            "non_unit_step_count": sum(delta != 1 for delta in frame_differences),
        },
        "timestamp": {
            "min": min(timestamps),
            "max": max(timestamps),
            "span_seconds": max(timestamps) - min(timestamps),
            "unique_count": len(set(timestamps)),
            "nondecreasing": all(delta >= 0 for delta in timestamp_intervals),
            "strictly_increasing": all(delta > 0 for delta in timestamp_intervals),
            "duplicate_count": len(timestamps) - len(set(timestamps)),
            "interval_seconds": numeric_summary(timestamp_intervals),
            "zero_interval_count": sum(delta == 0 for delta in timestamp_intervals),
            "negative_interval_count": sum(delta < 0 for delta in timestamp_intervals),
            "large_interval_threshold_seconds": median_interval * 1.5 if median_interval else None,
            "large_interval_count": sum(delta > median_interval * 1.5 for delta in timestamp_intervals)
            if median_interval
            else 0,
            "effective_rate_hz_from_median_interval": 1 / median_interval if median_interval else None,
            "whole_span_rate_hz": (len(timestamps) - 1) / (max(timestamps) - min(timestamps))
            if len(timestamps) > 1 and max(timestamps) > min(timestamps)
            else None,
        },
        "timestamp_ns": {
            "min": min(timestamps_ns),
            "max": max(timestamps_ns),
            "unique_count": len(set(timestamps_ns)),
            "nondecreasing": all(delta >= 0 for delta in timestamp_ns_intervals),
            "strictly_increasing": all(delta > 0 for delta in timestamp_ns_intervals),
            "duplicate_count": len(timestamps_ns) - len(set(timestamps_ns)),
            "unique_interval_nanoseconds": sorted(set(int(delta) for delta in timestamp_ns_intervals)),
            "interval_nanoseconds": numeric_summary(timestamp_ns_intervals),
        },
        "timestamp_vs_timestamp_ns": {
            "absolute_difference_seconds": numeric_summary(timestamp_agreement),
        },
        "structural_interval_anomalies": timestamp_anomalies,
    }


def ratio_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def video_profile(path: Path) -> dict[str, Any]:
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        stream_profiles = []
        for stream in streams:
            codec = stream.codec_context.codec
            stream_profiles.append(
                {
                    "index": stream.index,
                    "codec_name": stream.codec_context.name,
                    "codec_long_name": codec.long_name if codec else None,
                    "width": stream.codec_context.width,
                    "height": stream.codec_context.height,
                    "pixel_format": stream.codec_context.format.name if stream.codec_context.format else None,
                    "average_rate": ratio_to_float(stream.average_rate),
                    "base_rate": ratio_to_float(stream.base_rate),
                    "guessed_rate": ratio_to_float(stream.guessed_rate),
                    "time_base": str(stream.time_base) if stream.time_base else None,
                    "start_time": stream.start_time,
                    "start_seconds": float(stream.start_time * stream.time_base)
                    if stream.start_time is not None and stream.time_base
                    else None,
                    "duration": stream.duration,
                    "duration_seconds": float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base
                    else None,
                    "header_frame_count": stream.frames,
                }
            )
        profile = {
            "path": path.as_posix(),
            "file_size_bytes": path.stat().st_size,
            "container_format": container.format.name,
            "container_format_long_name": container.format.long_name,
            "container_duration_seconds": float(container.duration / av.time_base)
            if container.duration is not None
            else None,
            "container_start_seconds": float(container.start_time / av.time_base)
            if container.start_time is not None
            else None,
            "container_bit_rate": container.bit_rate,
            "video_stream_count": len(streams),
            "streams": stream_profiles,
        }

    packet_count = 0
    packet_pts_seconds: list[float] = []
    packet_end_seconds: list[float] = []
    keyframe_packet_count = 0
    with av.open(str(path), mode="r") as container:
        video_streams = list(container.streams.video)
        if video_streams:
            target = video_streams[0]
            for packet in container.demux(target):
                if packet.size <= 0:
                    continue
                packet_count += 1
                keyframe_packet_count += int(packet.is_keyframe)
                if packet.pts is not None and packet.time_base is not None:
                    pts = float(packet.pts * packet.time_base)
                    packet_pts_seconds.append(pts)
                    duration = float(packet.duration * packet.time_base) if packet.duration else 0.0
                    packet_end_seconds.append(pts + duration)
    profile["packet_scan"] = {
        "packet_count_not_frame_count": packet_count,
        "keyframe_packet_count": keyframe_packet_count,
        "pts_min_seconds": min(packet_pts_seconds) if packet_pts_seconds else None,
        "pts_max_seconds": max(packet_pts_seconds) if packet_pts_seconds else None,
        "pts_end_max_seconds": max(packet_end_seconds) if packet_end_seconds else None,
        "pts_nondecreasing_in_demux_order": all(
            right >= left for left, right in zip(packet_pts_seconds, packet_pts_seconds[1:])
        ),
    }
    return profile


def table_column(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def physical_numeric_dtype(field_type: pa.DataType) -> str | None:
    value_type = field_type.value_type if pa.types.is_list(field_type) else field_type
    if pa.types.is_float32(value_type):
        return "float32"
    if pa.types.is_float64(value_type):
        return "float64"
    if pa.types.is_int64(value_type):
        return "int64"
    return str(value_type)


def inspect(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.parquet"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    custom_path = dataset_root / "meta" / "custom_metadata.csv"
    episode_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    video_path = (
        dataset_root
        / "videos"
        / "base_0_camera"
        / "rgb"
        / "image"
        / "chunk-000"
        / "episode_000000.mp4"
    )
    required = [info_path, episodes_path, tasks_path, custom_path, episode_path, video_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset files: {missing}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    with custom_path.open("r", encoding="utf-8", newline="") as handle:
        custom_rows = list(csv.DictReader(handle))
    episodes_table, episodes = parquet_profile(episodes_path)
    tasks_table, tasks = parquet_profile(tasks_path)
    episode_table, episode = parquet_profile(episode_path)
    video = video_profile(video_path)

    frame_indices = table_column(episode_table, "frame_index")
    timestamps = table_column(episode_table, "timestamp")
    timestamps_ns = table_column(episode_table, "timestamp_ns")
    temporal = temporal_profile(frame_indices, timestamps, timestamps_ns)

    metadata_shapes = {
        name: declaration.get("shape")
        for name, declaration in info["features"].items()
        if isinstance(declaration, dict) and "shape" in declaration
    }
    observed_shapes = {
        name: profile.get("array", {}).get("observed_lengths")
        for name, profile in episode["columns"].items()
        if "array" in profile
    }
    shape_comparison = {}
    dtype_comparison = {}
    for field in episode_table.schema:
        name = field.name
        declared_shape = metadata_shapes.get(name)
        observed_lengths = observed_shapes.get(name)
        physical_is_list = pa.types.is_list(field.type)
        shape_comparison[name] = {
            "declared_shape": declared_shape,
            "physical_representation": "list" if physical_is_list else "scalar",
            "observed_list_lengths": observed_lengths,
            "compatible": declared_shape == observed_lengths
            if physical_is_list
            else declared_shape == [1],
        }
        declaration = info["features"].get(name, {})
        declared_dtype = declaration.get("dtype") if isinstance(declaration, dict) else None
        observed_dtype = physical_numeric_dtype(field.type)
        dtype_comparison[name] = {
            "declared_dtype": declared_dtype,
            "physical_arrow_dtype": str(field.type),
            "observed_numeric_dtype": observed_dtype,
            "matches": declared_dtype == observed_dtype,
        }

    episode_metadata_row = episodes_table.to_pylist()[0] if episodes_table.num_rows else None
    task_rows = tasks_table.to_pylist()
    data_reference = None
    video_reference = None
    if episode_metadata_row:
        data_reference = info["data_path"].format(
            chunk_index=episode_metadata_row["data/chunk_index"],
            file_index=episode_metadata_row["data/file_index"],
        )
        video_key = "base_0_camera/rgb/image"
        video_reference = info["video_path"].format(
            video_key=video_key,
            chunk_index=episode_metadata_row[f"videos/{video_key}/chunk_index"],
            file_index=episode_metadata_row[f"videos/{video_key}/file_index"],
        )

    task_indices = table_column(episode_table, "task_index")
    valid_task_indices = {row["task_index"] for row in task_rows}
    timestamp_bearing_fields = [
        "base_0_camera/position",
        "base_0_camera/quaternion_xyzw",
        "left_hand/tracks",
        "right_hand/tracks",
    ]
    embedded_timestamp_checks = {
        name: {
            "matches_timestamp_ns_all_rows": all(
                row[0] == timestamp_ns
                for row, timestamp_ns in zip(table_column(episode_table, name), timestamps_ns)
            )
        }
        for name in timestamp_bearing_fields
    }
    action_relationships = {}
    for side in ("left", "right"):
        track_name = f"{side}_hand/tracks"
        action_name = f"action.{side}_hand_tracks"
        tracks = table_column(episode_table, track_name)
        actions = table_column(episode_table, action_name)
        action_relationships[side] = {
            "action_matches_next_row_track_payload": all(
                action == next_track[1:] for action, next_track in zip(actions[:-1], tracks[1:])
            ),
            "last_action_all_zero": all(value == 0 for value in actions[-1]),
        }
    stream = video["streams"][0] if video["streams"] else {}
    declared_video = info["features"]["base_0_camera/rgb/image"]
    comparisons = {
        "declared_total_episodes_vs_metadata_rows": {
            "declared": info["total_episodes"],
            "observed": episodes_table.num_rows,
            "matches": info["total_episodes"] == episodes_table.num_rows,
        },
        "declared_total_tasks_vs_task_rows": {
            "declared": info["total_tasks"],
            "observed": tasks_table.num_rows,
            "matches": info["total_tasks"] == tasks_table.num_rows,
        },
        "declared_total_frames_vs_episode_rows": {
            "declared": info["total_frames"],
            "observed": episode_table.num_rows,
            "matches": info["total_frames"] == episode_table.num_rows,
        },
        "episode_length_vs_episode_rows": {
            "declared": episode_metadata_row["length"] if episode_metadata_row else None,
            "observed": episode_table.num_rows,
            "matches": bool(episode_metadata_row and episode_metadata_row["length"] == episode_table.num_rows),
        },
        "declared_total_videos_vs_local_video": {
            "declared": info["total_videos"],
            "observed": 1,
            "matches": info["total_videos"] == 1,
        },
        "data_reference": {
            "relative_path": data_reference,
            "exists": bool(data_reference and (dataset_root / data_reference).is_file()),
        },
        "video_reference": {
            "relative_path": video_reference,
            "exists": bool(video_reference and (dataset_root / video_reference).is_file()),
        },
        "task_references": {
            "observed_unique_task_indices": sorted(set(task_indices)),
            "valid_task_indices": sorted(valid_task_indices),
            "unresolved_task_indices": sorted(set(task_indices) - valid_task_indices),
            "episode_task_labels": episode_metadata_row["tasks"] if episode_metadata_row else [],
            "task_table_labels": [row["task"] for row in task_rows],
            "labels_resolve": bool(
                episode_metadata_row
                and set(episode_metadata_row["tasks"]) <= {row["task"] for row in task_rows}
            ),
        },
        "custom_metadata_episode_references": {
            "row_count": len(custom_rows),
            "episode_indices": sorted({int(row["episode_index"]) for row in custom_rows}),
            "episode_ids": sorted({row["episode_id"] for row in custom_rows}),
            "resolve": all(
                int(row["episode_index"]) == 0 and row["episode_id"] == "episode_000000"
                for row in custom_rows
            ),
        },
        "embedded_timestamp_fields": embedded_timestamp_checks,
        "action_relationships": action_relationships,
        "frame_index_equals_global_index": {
            "matches_all_rows": frame_indices == table_column(episode_table, "index"),
        },
        "episode_index_values": {
            "unique_values": sorted(set(table_column(episode_table, "episode_index"))),
        },
        "video_resolution": {
            "declared": declared_video["shape"][0:2],
            "observed": [stream.get("height"), stream.get("width")],
            "matches": declared_video["shape"][0:2] == [stream.get("height"), stream.get("width")],
        },
        "video_codec": {
            "declared": declared_video.get("encoding"),
            "observed": stream.get("codec_name"),
            "matches_h264": stream.get("codec_name") == "h264",
        },
        "video_frame_count": {
            "declared": info["total_frames"],
            "header": stream.get("header_frame_count"),
            "matches": stream.get("header_frame_count") == info["total_frames"],
        },
        "video_fps": {
            "declared": info["fps"],
            "observed_average_rate": stream.get("average_rate"),
            "difference": abs(stream.get("average_rate") - info["fps"])
            if stream.get("average_rate") is not None
            else None,
        },
        "video_timestamp_bounds": {
            "metadata_from_timestamp": episode_metadata_row.get(
                "videos/base_0_camera/rgb/image/from_timestamp"
            )
            if episode_metadata_row
            else None,
            "metadata_to_timestamp": episode_metadata_row.get(
                "videos/base_0_camera/rgb/image/to_timestamp"
            )
            if episode_metadata_row
            else None,
            "episode_first_timestamp": timestamps[0],
            "episode_last_timestamp": timestamps[-1],
            "video_packet_first_pts": video["packet_scan"]["pts_min_seconds"],
            "video_packet_last_pts": video["packet_scan"]["pts_max_seconds"],
            "video_packet_end": video["packet_scan"]["pts_end_max_seconds"],
            "container_duration_seconds": video["container_duration_seconds"],
        },
    }

    return finite_or_label(
        {
            "tool_versions": {"pyarrow": pa.__version__, "av": av.__version__},
            "dataset_root": dataset_root.as_posix(),
            "info_json": {
                "path": info_path.as_posix(),
                "document_count": 1,
                "null_value_count_recursive": sum(
                    1 for value in walk_json(info) if value is None
                ),
                "structure": json_structure(info),
                "metadata_declarations": info,
            },
            "custom_metadata_csv": csv_profile(custom_path),
            "episodes_parquet": episodes,
            "tasks_parquet": tasks,
            "episode_parquet": episode,
            "temporal": temporal,
            "shape_comparison": shape_comparison,
            "dtype_comparison": dtype_comparison,
            "video": video,
            "comparisons": comparisons,
        }
    )


def walk_json(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_json(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase-2-schema.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        json.dumps(
            {
                "episode_rows": report["episode_parquet"]["row_count"],
                "frame_index": report["temporal"]["frame_index"],
                "timestamp": report["temporal"]["timestamp"],
                "video": report["video"]["streams"],
                "comparisons": report["comparisons"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
