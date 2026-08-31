from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_visualizer import (
    DatasetMetadataError,
    DatasetReferenceError,
    EpisodeNotFoundError,
    FieldNotFoundError,
    FrameNotFoundError,
    InvalidDatasetRootError,
    LeRobotDataset,
    TimestampLookupError,
    TimestampOutOfRangeError,
    VideoNotFoundError,
    VideoReferenceError,
)


def build_dataset(
    root: Path,
    *,
    include_video_columns: bool = True,
    include_custom_metadata: bool = True,
    include_timestamp_ns: bool = True,
    include_task_description: bool = True,
) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    video_dir = root / "videos" / "camera" / "chunk-000"
    video_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": 1.0,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "description": "seconds"},
            "timestamp_ns": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "vector": {
                "dtype": "float32",
                "shape": [2],
                "description": "test declaration",
                "frame": "declared-frame",
                "unit": "test-units",
                "future_key": {"preserved": True},
            },
            "camera": {"dtype": "video", "shape": [2, 2, 3], "storage": "MP4"},
            "camera/intrinsics": [[1.0, 0.0], [0.0, 1.0]],
            "translation_unit": "meter",
        },
        "total_episodes": 1,
        "total_frames": 4,
        "total_tasks": 1,
        "total_videos": 1,
        "splits": {"train": "0:1"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episode_metadata = {
        "episode_index": [0],
        "length": [4],
        "tasks": [["test task"]],
        "data/chunk_index": [0],
        "data/file_index": [0],
        "extra_episode_value": ["preserved"],
    }
    if include_video_columns:
        episode_metadata.update(
            {
                "videos/camera/chunk_index": [0],
                "videos/camera/file_index": [0],
                "videos/camera/from_timestamp": [0.0],
                "videos/camera/to_timestamp": [4.0],
            }
        )
    pq.write_table(pa.table(episode_metadata), root / "meta" / "episodes.parquet")
    task_metadata = {"task_index": [0], "task": ["test task"], "extra_task_value": ["preserved"]}
    if include_task_description:
        task_metadata["description"] = ["metadata label"]
    pq.write_table(pa.table(task_metadata), root / "meta" / "tasks.parquet")
    if include_custom_metadata:
        (root / "meta" / "custom_metadata.csv").write_text(
            "episode_index,is_eval_episode,episode_id,success\n0,False,episode_000000,True\n",
            encoding="utf-8",
        )
    episode_data = {
        "timestamp": pa.array([0.0, 1.0, 2.0, 3.0], type=pa.float32()),
        "frame_index": [0, 2, 5, 9],
        "episode_index": [0, 0, 0, 0],
        "index": [10, 11, 12, 13],
        "task_index": [0, 0, 0, 0],
        "vector": pa.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]),
    }
    if include_timestamp_ns:
        episode_data["timestamp_ns"] = [0, 1_000_000_000, 2_000_000_000, 3_000_000_000]
    pq.write_table(
        pa.table(episode_data),
        root / "data" / "chunk-000" / "episode_000000.parquet",
    )
    (video_dir / "episode_000000.mp4").write_bytes(b"test-video-reference")


def build_multi_episode_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": 2.0,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "some_new_sensor/value": {
                "dtype": "float32",
                "shape": [2],
                "components": ["a", "b"],
            },
            "front/rgb": {"dtype": "video", "shape": [2, 2, 3]},
            "wrist/rgb": {"dtype": "video", "shape": [2, 2, 3]},
        },
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "total_videos": 4,
        "splits": {"train": "0:2"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episode_indices = [3, 8]
    lengths = [2, 3]
    episode_metadata: dict[str, list[object]] = {
        "episode_index": episode_indices,
        "length": lengths,
        "tasks": [["task three"], ["task eight"]],
        "data/chunk_index": [0, 0],
        "data/file_index": [0, 1],
        "session_label": ["alpha", "beta"],
    }
    for video_key in ("front/rgb", "wrist/rgb"):
        prefix = f"videos/{video_key}"
        episode_metadata[f"{prefix}/chunk_index"] = [0, 0]
        episode_metadata[f"{prefix}/file_index"] = [0, 1]
        episode_metadata[f"{prefix}/from_timestamp"] = [0.0, 1.0]
        episode_metadata[f"{prefix}/to_timestamp"] = [1.0, 2.5]
    pq.write_table(pa.table(episode_metadata), root / "meta" / "episodes.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": [30, 80],
                "task": ["task three", "task eight"],
                "category": ["alpha", "beta"],
            }
        ),
        root / "meta" / "tasks.parquet",
    )

    episode_values = (
        (0, 3, [4, 9], [0.0, 0.5], 30),
        (1, 8, [100, 105, 111], [1.0, 1.5, 2.0], 80),
    )
    for file_index, episode_index, frame_indices, timestamps, task_index in episode_values:
        count = len(frame_indices)
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.float32()),
                    "frame_index": frame_indices,
                    "episode_index": [episode_index] * count,
                    "task_index": [task_index] * count,
                    "some_new_sensor/value": [[float(offset), float(offset + 1)] for offset in range(count)],
                }
            ),
            root / "data" / "chunk-000" / f"episode_{file_index:06d}.parquet",
        )
        for video_key in ("front/rgb", "wrist/rgb"):
            video_dir = root / "videos" / Path(video_key) / "chunk-000"
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / f"episode_{file_index:06d}.mp4").write_bytes(b"video")


class DatasetUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        build_dataset(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dataset(self) -> LeRobotDataset:
        return LeRobotDataset(self.root)

    def test_open_valid_dataset_and_metadata(self) -> None:
        dataset = self.dataset()
        self.assertEqual(dataset.metadata.codebase_version, "v3.0")
        self.assertEqual(dataset.metadata.total_frames, 4)

    def test_invalid_dataset_root(self) -> None:
        with self.assertRaises(InvalidDatasetRootError):
            LeRobotDataset(self.root / "missing")

    def test_missing_metadata(self) -> None:
        (self.root / "meta" / "tasks.parquet").unlink()
        with self.assertRaises(DatasetMetadataError):
            self.dataset()

    def test_malformed_metadata(self) -> None:
        (self.root / "meta" / "info.json").write_text("{", encoding="utf-8")
        with self.assertRaises(DatasetMetadataError):
            self.dataset()

    def test_episode_listing_and_metadata(self) -> None:
        dataset = self.dataset()
        self.assertEqual(dataset.list_episode_indices(), (0,))
        metadata = dataset.get_episode_metadata(0)
        self.assertEqual(metadata.declared_length, 4)
        self.assertEqual(metadata.tasks, ("test task",))
        self.assertTrue(metadata.custom["success"])
        self.assertEqual(metadata.raw["extra_episode_value"], "preserved")
        with self.assertRaises(TypeError):
            metadata.raw["extra_episode_value"] = "changed"  # type: ignore[index]
        with self.assertRaises(EpisodeNotFoundError):
            dataset.get_episode_metadata(99)

    def test_custom_metadata_is_optional(self) -> None:
        other = Path(self.temp.name) / "without-custom"
        build_dataset(other, include_custom_metadata=False)
        self.assertEqual(dict(LeRobotDataset(other).get_episode_metadata(0).custom), {})

    def test_custom_metadata_preserves_arbitrary_columns(self) -> None:
        path = self.root / "meta" / "custom_metadata.csv"
        path.write_text("episode_index,site_code,attempt\n0,lab-a,7\n", encoding="utf-8")
        custom = self.dataset().get_episode_metadata(0).custom
        self.assertEqual(custom["site_code"], "lab-a")
        self.assertEqual(custom["attempt"], 7)
        self.assertNotIn("success", custom)
        with self.assertRaises(TypeError):
            custom["site_code"] = "changed"  # type: ignore[index]

    def test_episode_loading_is_lazy(self) -> None:
        data_path = self.root / "data" / "chunk-000" / "episode_000000.parquet"
        data_path.write_bytes(b"not parquet")
        dataset = self.dataset()
        self.assertEqual(dataset.list_episode_indices(), (0,))
        with self.assertRaises(DatasetMetadataError):
            dataset.get_frame_count(0)

    def test_task_listing_and_metadata(self) -> None:
        dataset = self.dataset()
        self.assertEqual(len(dataset.list_tasks()), 1)
        task = dataset.get_task_metadata(0)
        self.assertEqual(task.task, "test task")
        self.assertEqual(task.description, "metadata label")
        self.assertEqual(task.raw["extra_task_value"], "preserved")

    def test_task_description_is_optional_and_raw_row_is_preserved(self) -> None:
        other = Path(self.temp.name) / "without-task-description"
        build_dataset(other, include_task_description=False)
        task = LeRobotDataset(other).get_task_metadata(0)
        self.assertIsNone(task.description)
        self.assertEqual(task.raw["extra_task_value"], "preserved")
        with self.assertRaises(TypeError):
            task.raw["extra_task_value"] = "changed"  # type: ignore[index]

    def test_frames_use_frame_index_mapping(self) -> None:
        dataset = self.dataset()
        self.assertEqual(dataset.get_frame(0, 0).frame_index, 0)
        self.assertEqual(dataset.get_frame(0, 5).values["index"], 12)
        self.assertEqual(dataset.get_frame(0, 9).frame_index, 9)
        with self.assertRaises(FrameNotFoundError):
            dataset.get_frame(0, 1)
        with self.assertRaises(FrameNotFoundError):
            dataset.get_frame(0, -1)

    def test_exact_and_midpoint_timestamp_lookup(self) -> None:
        dataset = self.dataset()
        exact = dataset.get_nearest_frame(0, 2.0)
        self.assertTrue(exact.exact)
        self.assertEqual(exact.frame.frame_index, 5)
        earlier = dataset.get_nearest_frame(0, 1.5)
        later = dataset.get_nearest_frame(0, 1.5, tie_break="later")
        self.assertEqual(earlier.frame.frame_index, 2)
        self.assertEqual(later.frame.frame_index, 5)

    def test_timestamp_bounds_and_clamping(self) -> None:
        dataset = self.dataset()
        with self.assertRaises(TimestampOutOfRangeError):
            dataset.get_nearest_frame(0, -0.1)
        with self.assertRaises(TimestampOutOfRangeError):
            dataset.get_nearest_frame(0, 3.1)
        self.assertEqual(dataset.get_nearest_frame(0, -0.1, out_of_range="clamp").frame.frame_index, 0)
        self.assertEqual(dataset.get_nearest_frame(0, 3.1, out_of_range="clamp").frame.frame_index, 9)

    def test_invalid_timestamp_and_policies(self) -> None:
        dataset = self.dataset()
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(TimestampLookupError):
                dataset.get_nearest_frame(0, value)
        with self.assertRaises(TimestampLookupError):
            dataset.get_nearest_frame(0, 1.0, out_of_range="unknown")  # type: ignore[arg-type]
        with self.assertRaises(TimestampLookupError):
            dataset.get_nearest_frame(0, 1.0, tie_break="unknown")  # type: ignore[arg-type]

    def test_selected_fields_and_missing_field(self) -> None:
        dataset = self.dataset()
        frame = dataset.get_frame(0, 2, fields=["vector"])
        self.assertEqual(tuple(frame.values), ("vector",))
        self.assertIsInstance(frame.values["vector"], tuple)
        series = dataset.get_series(0, ["vector"])
        self.assertEqual(series.frame_indices, (0, 2, 5, 9))
        self.assertEqual(len(series.values["vector"]), 4)
        with self.assertRaises(FieldNotFoundError):
            dataset.get_frame(0, 0, fields=["missing"])

    def test_schema_preserves_declaration_and_physical_dtype(self) -> None:
        schema = self.dataset().get_feature_schemas(0)["vector"]
        self.assertEqual(schema.declaration.dtype, "float32")  # type: ignore[union-attr]
        self.assertEqual(schema.physical_arrow_dtype, "list<element: double>")
        self.assertEqual(schema.observed_lengths, (2,))

    def test_feature_declaration_preserves_unknown_keys_and_skips_static_metadata(self) -> None:
        dataset = self.dataset()
        declaration = dataset.metadata.features["vector"]
        self.assertEqual(declaration.raw["unit"], "test-units")
        self.assertTrue(declaration.raw["future_key"]["preserved"])
        self.assertNotIn("camera/intrinsics", dataset.metadata.features)
        self.assertNotIn("translation_unit", dataset.metadata.features)
        self.assertEqual(dataset.metadata.raw["features"]["translation_unit"], "meter")
        self.assertEqual(dataset.metadata.raw["features"]["camera/intrinsics"][0], (1.0, 0.0))
        with self.assertRaises(TypeError):
            declaration.raw["unit"] = "changed"  # type: ignore[index]

    def test_timing(self) -> None:
        timing = self.dataset().get_timing(0)
        self.assertEqual(timing.frame_count, 4)
        self.assertEqual(timing.frame_indices, (0, 2, 5, 9))
        self.assertEqual(timing.timestamps, (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(
            timing.timestamps_ns,
            (0, 1_000_000_000, 2_000_000_000, 3_000_000_000),
        )
        self.assertEqual(
            tuple(zip(timing.frame_indices, timing.timestamps)),
            ((0, 0.0), (2, 1.0), (5, 2.0), (9, 3.0)),
        )

    def test_timestamp_ns_is_optional_and_not_synthesized(self) -> None:
        other = Path(self.temp.name) / "without-timestamp-ns"
        build_dataset(other, include_timestamp_ns=False)
        dataset = LeRobotDataset(other)
        self.assertIsNone(dataset.get_frame(0, 5, fields=[]).timestamp_ns)
        self.assertIsNone(dataset.get_series(0, ["vector"]).timestamps_ns)
        timing = dataset.get_timing(0)
        self.assertIsNone(timing.timestamps_ns)
        self.assertEqual(timing.frame_indices, (0, 2, 5, 9))
        self.assertEqual(dataset.get_nearest_frame(0, 1.5).frame.frame_index, 2)

    def test_timestamp_ns_is_validated_when_present(self) -> None:
        path = self.root / "data" / "chunk-000" / "episode_000000.parquet"
        table = pq.read_table(path)
        timestamp_ns_index = table.column_names.index("timestamp_ns")
        table = table.set_column(timestamp_ns_index, "timestamp_ns", pa.array([0, 2, 1, 3]))
        pq.write_table(table, path)
        with self.assertRaises(DatasetMetadataError):
            self.dataset().get_timing(0)

    def test_video_resource_and_local_resolution(self) -> None:
        dataset = self.dataset()
        resource = dataset.get_video_resource(0)
        self.assertEqual(resource.video_key, "camera")
        self.assertFalse(hasattr(resource, "path"))
        path = dataset.resolve_video_path(resource)
        self.assertTrue(path.is_file())
        self.assertTrue(path.is_relative_to(self.root.resolve()))

    def test_missing_video_reference(self) -> None:
        other = Path(self.temp.name) / "other"
        build_dataset(other, include_video_columns=False)
        dataset = LeRobotDataset(other)
        with self.assertRaises(VideoReferenceError):
            dataset.get_video_resource(0)

    def test_missing_video_file(self) -> None:
        dataset = self.dataset()
        resource = dataset.get_video_resource(0)
        (self.root / "videos" / "camera" / "chunk-000" / "episode_000000.mp4").unlink()
        with self.assertRaises(VideoNotFoundError):
            dataset.resolve_video_path(resource)

    def test_malformed_and_traversing_paths(self) -> None:
        info_path = self.root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["video_path"] = "../outside/{video_key}/{chunk_index}/{file_index}.mp4"
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaises(DatasetReferenceError):
            self.dataset()

        info["video_path"] = "videos/{unknown}/episode.mp4"
        info_path.write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaises(DatasetReferenceError):
            self.dataset()

    def test_multiple_non_contiguous_episodes_lengths_videos_and_arbitrary_feature(self) -> None:
        other = Path(self.temp.name) / "multi"
        build_multi_episode_dataset(other)
        dataset = LeRobotDataset(other)

        self.assertEqual(dataset.list_episode_indices(), (3, 8))
        self.assertEqual(dataset.get_frame_count(3), 2)
        self.assertEqual(dataset.get_frame_count(8), 3)
        self.assertEqual(dataset.get_timing(3).frame_indices, (4, 9))
        self.assertEqual(dataset.get_timing(8).frame_indices, (100, 105, 111))
        self.assertIsNone(dataset.get_timing(8).timestamps_ns)
        self.assertEqual(dataset.get_episode_metadata(8).raw["session_label"], "beta")
        self.assertEqual(dataset.get_task_metadata(80).raw["category"], "beta")
        self.assertIsNone(dataset.get_task_metadata(80).description)

        schema = dataset.get_feature_schemas(8)["some_new_sensor/value"]
        self.assertEqual(schema.observed_lengths, (2,))
        self.assertEqual(schema.declaration.raw["components"], ("a", "b"))  # type: ignore[union-attr]
        self.assertEqual(
            dataset.get_episode_metadata(3).video_keys,
            ("front/rgb", "wrist/rgb"),
        )
        with self.assertRaises(VideoReferenceError):
            dataset.get_video_resource(3)
        resource = dataset.get_video_resource(3, "wrist/rgb")
        self.assertEqual(resource.video_key, "wrist/rgb")
        self.assertTrue(dataset.resolve_video_path(resource).is_file())


class LocalDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.environ.get("LEROBOT_DATASET_ROOT")
        cls.root = Path(configured) if configured else Path("data/raw")
        if not cls.root.is_dir():
            raise unittest.SkipTest(f"Local integration dataset is unavailable: {cls.root}")
        cls.dataset = LeRobotDataset(cls.root)

    def test_episode_and_frame_counts(self) -> None:
        self.assertEqual(self.dataset.list_episode_indices(), (0,))
        self.assertEqual(self.dataset.get_frame_count(0), 3029)

    def test_first_middle_last_and_invalid_frames(self) -> None:
        for frame_index in (0, 1514, 3028):
            self.assertEqual(self.dataset.get_frame(0, frame_index, fields=[]).frame_index, frame_index)
        for frame_index in (-1, 3029):
            with self.assertRaises(FrameNotFoundError):
                self.dataset.get_frame(0, frame_index)

    def test_exact_timestamp_lookup(self) -> None:
        frame = self.dataset.get_frame(0, 1514, fields=[])
        match = self.dataset.get_nearest_frame(0, frame.timestamp)
        self.assertTrue(match.exact)
        self.assertEqual(match.frame.frame_index, 1514)

    def test_selected_series_and_shapes(self) -> None:
        series = self.dataset.get_series(0, ["left_hand/tracks", "action.left_hand_tracks"])
        self.assertEqual(len(series.timestamps), 3029)
        self.assertEqual(series.schemas["left_hand/tracks"].observed_lengths, (64,))
        self.assertEqual(series.schemas["action.left_hand_tracks"].observed_lengths, (63,))

    def test_task_resolution(self) -> None:
        episode = self.dataset.get_episode_metadata(0)
        task = self.dataset.get_task_metadata(0)
        self.assertIn(task.task, episode.tasks)

    def test_video_resolution(self) -> None:
        resource = self.dataset.get_video_resource(0)
        self.assertEqual(resource.video_key, "base_0_camera/rgb/image")
        self.assertTrue(self.dataset.resolve_video_path(resource).is_file())

    def test_declared_and_physical_dtype(self) -> None:
        schema = self.dataset.get_feature_schemas(0)["left_hand/tracks"]
        self.assertEqual(schema.declaration.dtype, "float32")  # type: ignore[union-attr]
        self.assertEqual(schema.physical_arrow_dtype, "list<element: double>")

    def test_current_dataset_metadata_is_preserved(self) -> None:
        raw_features = self.dataset.metadata.raw["features"]
        video_declaration = self.dataset.metadata.features["base_0_camera/rgb/image"]
        self.assertIn("encoding", video_declaration.raw)
        self.assertIn("base_0_camera/intrinsics", raw_features)
        self.assertIn("base_0_camera/distortion", raw_features)
        self.assertIn("base_0_camera/extrinsics", raw_features)
        self.assertEqual(raw_features["translation_unit"], "meter")

        episode = self.dataset.get_episode_metadata(0)
        self.assertIn("videos/base_0_camera/rgb/image/chunk_index", episode.raw)
        self.assertEqual(episode.custom["episode_id"], "episode_000000")
        self.assertIn("success", episode.custom)

        task = self.dataset.get_task_metadata(0)
        self.assertIsNotNone(task.description)
        self.assertEqual(task.raw["description"], task.description)

        timing = self.dataset.get_timing(0)
        self.assertEqual(len(timing.frame_indices), 3029)
        self.assertEqual(len(timing.timestamps), 3029)
        self.assertEqual(len(timing.timestamps_ns or ()), 3029)

    def test_missing_field(self) -> None:
        with self.assertRaises(FieldNotFoundError):
            self.dataset.get_series(0, ["missing"])


if __name__ == "__main__":
    unittest.main()
