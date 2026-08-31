"""Read-only local access to a LeRobot 3.0 dataset."""

from __future__ import annotations

import bisect
import csv
import json
import math
import string
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import (
    DatasetMetadataError,
    DatasetReferenceError,
    EpisodeNotFoundError,
    FieldNotFoundError,
    FrameNotFoundError,
    InvalidDatasetRootError,
    TimestampLookupError,
    TimestampOutOfRangeError,
    VideoNotFoundError,
    VideoReferenceError,
)
from .models import (
    DatasetMetadata,
    EpisodeMetadata,
    FeatureDeclaration,
    FeatureSchema,
    FrameRecord,
    TaskMetadata,
    TimeSeriesData,
    TimestampMatch,
    TimingInfo,
    VideoResource,
)

_REQUIRED_IDENTIFIER_FIELDS = frozenset({"episode_index", "frame_index", "timestamp"})
_IDENTIFIER_FIELDS = _REQUIRED_IDENTIFIER_FIELDS | {"timestamp_ns"}
_EPISODE_METADATA_COLUMNS = frozenset(
    {"episode_index", "length", "tasks", "data/chunk_index", "data/file_index"}
)
_TASK_COLUMNS = frozenset({"task_index", "task"})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _parse_csv_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        return value


def _normalize_fields(fields: Iterable[str] | str | None, available: tuple[str, ...]) -> tuple[str, ...]:
    if fields is None:
        return available
    requested = (fields,) if isinstance(fields, str) else tuple(fields)
    missing = [field for field in requested if field not in available]
    if missing:
        raise FieldNotFoundError(f"Unknown episode field(s): {', '.join(missing)}")
    return tuple(dict.fromkeys(requested))


@dataclass
class _EpisodeCache:
    episode_index: int
    table: pa.Table
    frame_to_row: dict[int, int]
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    timestamps_ns: tuple[int, ...] | None


class LeRobotDataset:
    """Small, bounded, local-only access layer over a LeRobot dataset root."""

    def __init__(self, dataset_root: str | Path):
        root = Path(dataset_root).expanduser()
        if not root.exists() or not root.is_dir():
            raise InvalidDatasetRootError(f"Dataset root is not a directory: {root}")
        self._root = root.resolve()
        self._info = self._read_info()
        self._validate_template(self._info.get("data_path"), {"chunk_index", "file_index"}, "data")
        self._validate_template(
            self._info.get("video_path"), {"video_key", "chunk_index", "file_index"}, "video"
        )
        self._feature_declarations = self._build_feature_declarations()
        self._video_keys = tuple(
            name
            for name, declaration in self._info.get("features", {}).items()
            if isinstance(declaration, dict) and declaration.get("dtype") == "video"
        )
        self._episode_rows = self._read_indexed_parquet(
            self._root / "meta" / "episodes.parquet",
            _EPISODE_METADATA_COLUMNS,
            "episode_index",
            "episode metadata",
        )
        self._task_rows = self._read_indexed_parquet(
            self._root / "meta" / "tasks.parquet",
            _TASK_COLUMNS,
            "task_index",
            "task metadata",
        )
        self._custom_rows = self._read_custom_metadata()
        self._validate_declared_totals()
        self._validate_task_references()
        unresolved_custom = set(self._custom_rows) - set(self._episode_rows)
        if unresolved_custom:
            raise DatasetMetadataError(
                f"Custom metadata references unknown episode indices: {sorted(unresolved_custom)}"
            )
        self.metadata = DatasetMetadata(
            codebase_version=str(self._required_info("codebase_version")),
            robot_type=str(self._required_info("robot_type")),
            fps=self._required_float("fps"),
            total_episodes=self._required_int("total_episodes"),
            total_frames=self._required_int("total_frames"),
            total_tasks=self._required_int("total_tasks"),
            total_videos=self._required_int("total_videos"),
            splits=_freeze(self._info.get("splits", {})),
            features=MappingProxyType(dict(self._feature_declarations)),
            raw=_freeze(self._info),
        )
        self._cache: _EpisodeCache | None = None

    def _required_info(self, name: str) -> Any:
        if name not in self._info:
            raise DatasetMetadataError(f"info.json is missing required field: {name}")
        return self._info[name]

    def _required_int(self, name: str) -> int:
        try:
            return int(self._required_info(name))
        except (TypeError, ValueError) as exc:
            raise DatasetMetadataError(f"info.json field {name} must be an integer") from exc

    def _required_float(self, name: str) -> float:
        try:
            return float(self._required_info(name))
        except (TypeError, ValueError) as exc:
            raise DatasetMetadataError(f"info.json field {name} must be numeric") from exc

    def _read_info(self) -> dict[str, Any]:
        path = self._root / "meta" / "info.json"
        if not path.is_file():
            raise DatasetMetadataError(f"Missing required metadata file: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetMetadataError(f"Could not parse metadata file: {path}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("features"), dict):
            raise DatasetMetadataError("info.json must contain a features object")
        return value

    def _build_feature_declarations(self) -> dict[str, FeatureDeclaration]:
        declarations = {}
        for name, value in self._info["features"].items():
            if not isinstance(value, dict):
                continue
            declaration = value
            raw_shape = declaration.get("shape")
            try:
                shape = tuple(int(item) for item in raw_shape) if isinstance(raw_shape, list) else None
            except (TypeError, ValueError) as exc:
                raise DatasetMetadataError(f"Feature {name} has an invalid declared shape") from exc
            declarations[name] = FeatureDeclaration(
                name=name,
                dtype=declaration.get("dtype"),
                shape=shape,
                description=declaration.get("description"),
                frame=declaration.get("frame"),
                storage=declaration.get("storage"),
                raw=_freeze(declaration),
            )
        return declarations

    def _read_indexed_parquet(
        self,
        path: Path,
        required_columns: frozenset[str],
        index_column: str,
        label: str,
    ) -> dict[int, dict[str, Any]]:
        if not path.is_file():
            raise DatasetMetadataError(f"Missing required {label} file: {path}")
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise DatasetMetadataError(f"Could not read {label}: {path}") from exc
        missing = sorted(required_columns - set(table.column_names))
        if missing:
            raise DatasetMetadataError(f"{label} is missing required columns: {', '.join(missing)}")
        rows: dict[int, dict[str, Any]] = {}
        for row in table.to_pylist():
            try:
                index = int(row[index_column])
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetMetadataError(f"Invalid {index_column} in {label}") from exc
            if index in rows:
                raise DatasetMetadataError(f"Duplicate {index_column} {index} in {label}")
            rows[index] = row
        return rows

    def _read_custom_metadata(self) -> dict[int, Mapping[str, Any]]:
        path = self._root / "meta" / "custom_metadata.csv"
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or ())
                if "episode_index" not in columns:
                    raise DatasetMetadataError("custom metadata is missing required column: episode_index")
                result = {}
                for row in reader:
                    index = int(row["episode_index"])
                    if index in result:
                        raise DatasetMetadataError(f"Duplicate episode_index {index} in custom metadata")
                    result[index] = _freeze({key: _parse_csv_value(value) for key, value in row.items()})
                return result
        except DatasetMetadataError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            raise DatasetMetadataError(f"Could not parse custom metadata: {path}") from exc

    def _validate_template(self, template: Any, allowed: set[str], label: str) -> None:
        if not isinstance(template, str) or not template:
            raise DatasetReferenceError(f"Missing or invalid {label}_path template")
        try:
            fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
        except ValueError as exc:
            raise DatasetReferenceError(f"Malformed {label}_path template") from exc
        if fields != allowed:
            raise DatasetReferenceError(
                f"{label}_path template fields must be {sorted(allowed)}, found {sorted(fields)}"
            )
        sample = {"chunk_index": 0, "file_index": 0, "video_key": "video"}
        try:
            rendered = template.format(**sample)
        except (KeyError, ValueError) as exc:
            raise DatasetReferenceError(f"Malformed {label}_path template") from exc
        self._safe_local_path(rendered, label)

    def _safe_local_path(self, relative: str, label: str) -> Path:
        if not relative or "\\" in relative:
            raise DatasetReferenceError(f"{label} reference must use a non-empty portable relative path")
        posix = PurePosixPath(relative)
        windows = PureWindowsPath(relative)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
            raise DatasetReferenceError(f"Unsafe {label} reference: {relative}")
        candidate = self._root.joinpath(*posix.parts).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise DatasetReferenceError(f"{label} reference escapes dataset root: {relative}") from exc
        return candidate

    def _validate_declared_totals(self) -> None:
        expected_episodes = self._required_int("total_episodes")
        expected_tasks = self._required_int("total_tasks")
        if expected_episodes != len(self._episode_rows):
            raise DatasetMetadataError(
                f"Declared episode total {expected_episodes} does not match metadata rows {len(self._episode_rows)}"
            )
        if expected_tasks != len(self._task_rows):
            raise DatasetMetadataError(
                f"Declared task total {expected_tasks} does not match metadata rows {len(self._task_rows)}"
            )

    def _validate_task_references(self) -> None:
        labels = {row["task"] for row in self._task_rows.values()}
        for index, row in self._episode_rows.items():
            tasks = row["tasks"]
            if tasks is not None and not isinstance(tasks, list):
                raise DatasetMetadataError(f"Episode {index} tasks must be a list")
            unresolved = set(tasks or ()) - labels
            if unresolved:
                raise DatasetMetadataError(
                    f"Episode {index} has unresolved task label(s): {sorted(unresolved)}"
                )

    def _episode_row(self, episode_index: int) -> dict[str, Any]:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise EpisodeNotFoundError(f"Unknown episode index: {episode_index}")
        try:
            return self._episode_rows[episode_index]
        except (KeyError, TypeError) as exc:
            raise EpisodeNotFoundError(f"Unknown episode index: {episode_index}") from exc

    def list_episode_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._episode_rows))

    def get_episode_metadata(self, episode_index: int) -> EpisodeMetadata:
        row = self._episode_row(episode_index)
        available_video_keys = tuple(
            key
            for key in self._video_keys
            if f"videos/{key}/chunk_index" in row and f"videos/{key}/file_index" in row
        )
        try:
            return EpisodeMetadata(
                episode_index=episode_index,
                declared_length=int(row["length"]),
                tasks=tuple(row["tasks"] or ()),
                video_keys=available_video_keys,
                custom=self._custom_rows.get(episode_index, MappingProxyType({})),
                raw=_freeze(row),
            )
        except (TypeError, ValueError) as exc:
            raise DatasetMetadataError(f"Malformed episode metadata for episode {episode_index}") from exc

    def list_tasks(self) -> tuple[TaskMetadata, ...]:
        return tuple(self.get_task_metadata(index) for index in sorted(self._task_rows))

    def get_task_metadata(self, task_index: int) -> TaskMetadata:
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise DatasetMetadataError(f"Unknown task index: {task_index}")
        try:
            row = self._task_rows[task_index]
        except (KeyError, TypeError) as exc:
            raise DatasetMetadataError(f"Unknown task index: {task_index}") from exc
        description = row.get("description")
        return TaskMetadata(
            task_index=task_index,
            task=str(row["task"]),
            description=None if description is None else str(description),
            raw=_freeze(row),
        )

    def _episode_path(self, episode_index: int) -> Path:
        row = self._episode_row(episode_index)
        try:
            relative = self._info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetReferenceError(f"Malformed data reference for episode {episode_index}") from exc
        path = self._safe_local_path(relative, "data")
        if not path.is_file():
            raise DatasetReferenceError(f"Episode data file does not exist: {relative}")
        return path

    def _load_episode(self, episode_index: int) -> _EpisodeCache:
        self._episode_row(episode_index)
        if self._cache is not None and self._cache.episode_index == episode_index:
            return self._cache
        path = self._episode_path(episode_index)
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise DatasetMetadataError(f"Could not read episode {episode_index}: {path.name}") from exc
        missing = sorted(_REQUIRED_IDENTIFIER_FIELDS - set(table.column_names))
        if missing:
            raise DatasetMetadataError(
                f"Episode {episode_index} is missing required columns: {', '.join(missing)}"
            )
        frame_indices = tuple(int(value) for value in table["frame_index"].to_pylist())
        timestamps = tuple(float(value) for value in table["timestamp"].to_pylist())
        timestamps_ns = None
        if "timestamp_ns" in table.column_names:
            try:
                timestamps_ns = tuple(int(value) for value in table["timestamp_ns"].to_pylist())
            except (TypeError, ValueError) as exc:
                raise DatasetMetadataError(
                    f"Episode {episode_index} contains invalid timestamp_ns values"
                ) from exc
            if any(right <= left for left, right in zip(timestamps_ns, timestamps_ns[1:])):
                raise DatasetMetadataError(
                    f"Episode {episode_index} timestamp_ns values are not strictly increasing"
                )
        if len(frame_indices) != len(set(frame_indices)):
            raise DatasetMetadataError(f"Episode {episode_index} contains duplicate frame indices")
        if not timestamps or any(not math.isfinite(value) for value in timestamps):
            raise DatasetMetadataError(f"Episode {episode_index} contains invalid timestamps")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise DatasetMetadataError(f"Episode {episode_index} timestamps are not strictly increasing")
        observed_episode_indices = set(table["episode_index"].to_pylist())
        if observed_episode_indices != {episode_index}:
            raise DatasetMetadataError(
                f"Episode {episode_index} contains mismatched episode_index values: {sorted(observed_episode_indices)}"
            )
        if "task_index" in table.column_names:
            unresolved_tasks = set(table["task_index"].to_pylist()) - set(self._task_rows)
            if unresolved_tasks:
                raise DatasetMetadataError(
                    f"Episode {episode_index} contains unresolved task indices: {sorted(unresolved_tasks)}"
                )
        self._cache = _EpisodeCache(
            episode_index=episode_index,
            table=table,
            frame_to_row={frame: row for row, frame in enumerate(frame_indices)},
            frame_indices=frame_indices,
            timestamps=timestamps,
            timestamps_ns=timestamps_ns,
        )
        return self._cache

    def get_frame_count(self, episode_index: int) -> int:
        return self._load_episode(episode_index).table.num_rows

    def get_frame(
        self, episode_index: int, frame_index: int, fields: Iterable[str] | str | None = None
    ) -> FrameRecord:
        cache = self._load_episode(episode_index)
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index not in cache.frame_to_row:
            raise FrameNotFoundError(f"Unknown frame index {frame_index} in episode {episode_index}")
        row_index = cache.frame_to_row[frame_index]
        selected = _normalize_fields(fields, tuple(cache.table.column_names))
        values = {
            field: _freeze(cache.table[field][row_index].as_py())
            for field in selected
            if field not in _IDENTIFIER_FIELDS
        }
        return FrameRecord(
            episode_index=int(cache.table["episode_index"][row_index].as_py()),
            frame_index=frame_index,
            timestamp=cache.timestamps[row_index],
            timestamp_ns=cache.timestamps_ns[row_index] if cache.timestamps_ns is not None else None,
            values=MappingProxyType(values),
        )

    def get_nearest_frame(
        self,
        episode_index: int,
        timestamp: float,
        out_of_range: Literal["error", "clamp"] = "error",
        tie_break: Literal["earlier", "later"] = "earlier",
    ) -> TimestampMatch:
        if out_of_range not in {"error", "clamp"}:
            raise TimestampLookupError(f"Unsupported out_of_range policy: {out_of_range}")
        if tie_break not in {"earlier", "later"}:
            raise TimestampLookupError(f"Unsupported tie_break policy: {tie_break}")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise TimestampLookupError(f"Timestamp must be finite, got: {timestamp}")
        requested = float(timestamp)
        cache = self._load_episode(episode_index)
        first, last = cache.timestamps[0], cache.timestamps[-1]
        if requested < first or requested > last:
            if out_of_range == "error":
                raise TimestampOutOfRangeError(
                    f"Timestamp {requested} is outside episode {episode_index} range [{first}, {last}]"
                )
            row = 0 if requested < first else len(cache.timestamps) - 1
        else:
            right = bisect.bisect_left(cache.timestamps, requested)
            if right < len(cache.timestamps) and cache.timestamps[right] == requested:
                row = right
            elif right == 0:
                row = 0
            elif right == len(cache.timestamps):
                row = len(cache.timestamps) - 1
            else:
                left = right - 1
                left_distance = requested - cache.timestamps[left]
                right_distance = cache.timestamps[right] - requested
                row = left if left_distance < right_distance else right
                if left_distance == right_distance:
                    row = left if tie_break == "earlier" else right
        frame = self.get_frame(episode_index, cache.frame_indices[row])
        return TimestampMatch(
            requested_timestamp=requested,
            matched_timestamp=frame.timestamp,
            delta_seconds=frame.timestamp - requested,
            exact=frame.timestamp == requested,
            frame=frame,
        )

    def get_series(self, episode_index: int, fields: Iterable[str] | str) -> TimeSeriesData:
        cache = self._load_episode(episode_index)
        selected = _normalize_fields(fields, tuple(cache.table.column_names))
        schemas = self.get_feature_schemas(episode_index)
        values = {
            field: tuple(_freeze(value) for value in cache.table[field].to_pylist())
            for field in selected
            if field not in _IDENTIFIER_FIELDS
        }
        return TimeSeriesData(
            episode_index=episode_index,
            frame_indices=cache.frame_indices,
            timestamps=cache.timestamps,
            timestamps_ns=cache.timestamps_ns,
            values=MappingProxyType(values),
            schemas=MappingProxyType({field: schemas[field] for field in selected}),
        )

    def get_timing(self, episode_index: int) -> TimingInfo:
        cache = self._load_episode(episode_index)
        return TimingInfo(
            episode_index=episode_index,
            frame_count=len(cache.timestamps),
            timestamp_start=cache.timestamps[0],
            timestamp_end=cache.timestamps[-1],
            frame_indices=cache.frame_indices,
            timestamps=cache.timestamps,
            timestamps_ns=cache.timestamps_ns,
            metadata_declared_fps=self.metadata.fps,
        )

    def get_feature_schemas(self, episode_index: int) -> Mapping[str, FeatureSchema]:
        cache = self._load_episode(episode_index)
        schemas = {}
        for field in cache.table.schema:
            observed_lengths = None
            if pa.types.is_list(field.type) or pa.types.is_large_list(field.type) or pa.types.is_fixed_size_list(field.type):
                observed_lengths = tuple(
                    sorted({len(value) for value in cache.table[field.name].to_pylist() if value is not None})
                )
            schemas[field.name] = FeatureSchema(
                name=field.name,
                declaration=self._feature_declarations.get(field.name),
                physical_arrow_dtype=str(field.type),
                observed_lengths=observed_lengths,
            )
        return MappingProxyType(schemas)

    def get_video_resource(self, episode_index: int, video_key: str | None = None) -> VideoResource:
        row = self._episode_row(episode_index)
        keys = self.get_episode_metadata(episode_index).video_keys
        if video_key is None:
            if len(keys) != 1:
                raise VideoReferenceError(
                    f"Episode {episode_index} has {len(keys)} video references; video_key is required"
                )
            video_key = keys[0]
        if video_key not in self._video_keys:
            raise VideoReferenceError(f"Unknown video key for episode {episode_index}: {video_key}")
        prefix = f"videos/{video_key}"
        required = [f"{prefix}/chunk_index", f"{prefix}/file_index", f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"]
        missing = [name for name in required if name not in row or row[name] is None]
        if missing:
            raise VideoReferenceError(
                f"Episode {episode_index} is missing video reference fields: {', '.join(missing)}"
            )
        try:
            return VideoResource(
                resource_id=f"episode:{episode_index}:video:{video_key}",
                episode_index=episode_index,
                video_key=video_key,
                from_timestamp=float(row[f"{prefix}/from_timestamp"]),
                to_timestamp=float(row[f"{prefix}/to_timestamp"]),
            )
        except (TypeError, ValueError) as exc:
            raise VideoReferenceError(f"Malformed video timestamps for episode {episode_index}") from exc

    def resolve_video_path(self, video_resource: VideoResource) -> Path:
        expected = self.get_video_resource(video_resource.episode_index, video_resource.video_key)
        if expected != video_resource:
            raise VideoReferenceError("Video resource does not match dataset metadata")
        row = self._episode_row(video_resource.episode_index)
        prefix = f"videos/{video_resource.video_key}"
        try:
            relative = self._info["video_path"].format(
                video_key=video_resource.video_key,
                chunk_index=int(row[f"{prefix}/chunk_index"]),
                file_index=int(row[f"{prefix}/file_index"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VideoReferenceError(
                f"Malformed video reference for episode {video_resource.episode_index}"
            ) from exc
        try:
            path = self._safe_local_path(relative, "video")
        except DatasetReferenceError as exc:
            raise VideoReferenceError(str(exc)) from exc
        if not path.is_file():
            raise VideoNotFoundError(
                f"Video file does not exist for resource {video_resource.resource_id}: {relative}"
            )
        return path
