"""Immutable public data models for the local access layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureDeclaration:
    name: str
    dtype: str | None
    shape: tuple[int, ...] | None
    raw: Mapping[str, Any]
    description: str | None = None
    frame: str | None = None
    storage: str | None = None


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    declaration: FeatureDeclaration | None
    physical_arrow_dtype: str
    observed_lengths: tuple[int, ...] | None


@dataclass(frozen=True)
class DatasetMetadata:
    codebase_version: str
    robot_type: str
    fps: float
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    splits: Mapping[str, Any]
    features: Mapping[str, FeatureDeclaration]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_index: int
    declared_length: int
    tasks: tuple[str, ...]
    video_keys: tuple[str, ...]
    custom: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TaskMetadata:
    task_index: int
    task: str
    description: str | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class FrameRecord:
    episode_index: int
    frame_index: int
    timestamp: float
    timestamp_ns: int | None
    values: Mapping[str, Any]


@dataclass(frozen=True)
class TimestampMatch:
    requested_timestamp: float
    matched_timestamp: float
    delta_seconds: float
    exact: bool
    frame: FrameRecord


@dataclass(frozen=True)
class TimeSeriesData:
    episode_index: int
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    timestamps_ns: tuple[int, ...] | None
    values: Mapping[str, tuple[Any, ...]]
    schemas: Mapping[str, FeatureSchema]


@dataclass(frozen=True)
class TimingInfo:
    episode_index: int
    frame_count: int
    timestamp_start: float
    timestamp_end: float
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    timestamps_ns: tuple[int, ...] | None
    metadata_declared_fps: float


@dataclass(frozen=True)
class VideoResource:
    resource_id: str
    episode_index: int
    video_key: str
    from_timestamp: float
    to_timestamp: float
