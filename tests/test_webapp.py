"""Flask test-client smoke tests for webapp.py (in-process, no real server/browser)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_visualizer.webapp import create_app

VIDEO_KEY = "base_0_camera/rgb/image"
FRAME_COUNT = 5


def _hand_track_row(ts_ns: int, value: float) -> list[float]:
    return [float(ts_ns)] + [value] * 63


def _action_row(value: float | None) -> list[float]:
    return [0.0] * 63 if value is None else [value] * 63


def _mano_row(hand_pose_value: float) -> list[float]:
    return [0.0] * 3 + [hand_pose_value] * 45 + [0.0] * 10 + [0.0] * 3


def build_semantic_dataset(root: Path) -> None:
    """A tiny synthetic dataset shaped like the real one, for webapp route tests.

    Distinct from test_dataset.py's build_dataset(), which exercises the generic
    access layer with an arbitrary "vector" field -- webapp.py depends on the
    specific hand-track/action/camera-pose field names, so this fixture provides
    those directly rather than overloading the generic fixture with extra flags.
    """
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    video_dir = root / "videos" / VIDEO_KEY / "chunk-000"
    video_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": 10.0,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "timestamp_ns": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            VIDEO_KEY: {"dtype": "video", "shape": [2, 2, 3], "storage": "MP4"},
            "base_0_camera/position": {"dtype": "float32", "shape": [4]},
            "base_0_camera/quaternion_xyzw": {"dtype": "float32", "shape": [5]},
            "left_hand/tracks": {"dtype": "float32", "shape": [64]},
            "right_hand/tracks": {"dtype": "float32", "shape": [64]},
            "action.left_hand_tracks": {"dtype": "float32", "shape": [63]},
            "action.right_hand_tracks": {"dtype": "float32", "shape": [63]},
            "observation.left_hand_mano": {"dtype": "float32", "shape": [61]},
            "observation.right_hand_mano": {"dtype": "float32", "shape": [61]},
        },
        "total_episodes": 1,
        "total_frames": FRAME_COUNT,
        "total_tasks": 1,
        "total_videos": 1,
        "splits": {"train": "0:1"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "length": [FRAME_COUNT],
                "tasks": [["test task"]],
                "data/chunk_index": [0],
                "data/file_index": [0],
                f"videos/{VIDEO_KEY}/chunk_index": [0],
                f"videos/{VIDEO_KEY}/file_index": [0],
                f"videos/{VIDEO_KEY}/from_timestamp": [0.0],
                f"videos/{VIDEO_KEY}/to_timestamp": [float(FRAME_COUNT) / 10.0],
            }
        ),
        root / "meta" / "episodes.parquet",
    )
    pq.write_table(
        pa.table({"task_index": [0], "task": ["test task"], "description": ["a test task"]}),
        root / "meta" / "tasks.parquet",
    )

    timestamps = [i / 10.0 for i in range(FRAME_COUNT)]
    timestamps_ns = [i * 100_000_000 for i in range(FRAME_COUNT)]
    left_tracks = [_hand_track_row(ts_ns, float(i)) for i, ts_ns in enumerate(timestamps_ns)]
    right_tracks = [_hand_track_row(ts_ns, float(i) * 2) for i, ts_ns in enumerate(timestamps_ns)]
    left_actions = [
        _action_row(float(i + 1)) if i + 1 < FRAME_COUNT else _action_row(None) for i in range(FRAME_COUNT)
    ]
    right_actions = [
        _action_row(float(i + 1) * 2) if i + 1 < FRAME_COUNT else _action_row(None) for i in range(FRAME_COUNT)
    ]
    camera_positions = [[float(ts_ns), 0.1 * i, 0.2 * i, 0.3 * i] for i, ts_ns in enumerate(timestamps_ns)]
    camera_quaternions = [[float(ts_ns), 0.0, 0.0, 0.0, 1.0] for ts_ns in timestamps_ns]
    left_mano = [_mano_row(float(i)) for i in range(FRAME_COUNT)]
    right_mano = [_mano_row(0.0) for _ in range(FRAME_COUNT)]

    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "timestamp_ns": timestamps_ns,
                "frame_index": list(range(FRAME_COUNT)),
                "episode_index": [0] * FRAME_COUNT,
                "index": list(range(FRAME_COUNT)),
                "task_index": [0] * FRAME_COUNT,
                "base_0_camera/position": camera_positions,
                "base_0_camera/quaternion_xyzw": camera_quaternions,
                "left_hand/tracks": left_tracks,
                "right_hand/tracks": right_tracks,
                "action.left_hand_tracks": left_actions,
                "action.right_hand_tracks": right_actions,
                "observation.left_hand_mano": left_mano,
                "observation.right_hand_mano": right_mano,
            }
        ),
        root / "data" / "chunk-000" / "episode_000000.parquet",
    )
    (video_dir / "episode_000000.mp4").write_bytes(b"0123456789" * 10)


class WebAppRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_semantic_dataset(self.root)
        self.app = create_app(self.root)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dataset_info_exposes_root_metadata(self) -> None:
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["robot_type"], "test")
        self.assertEqual(body["fps"], 10.0)
        self.assertEqual(body["codebase_version"], "v3.0")

    def test_list_episodes(self) -> None:
        response = self.client.get("/api/episodes")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body["episodes"]), 1)
        self.assertEqual(body["episodes"][0]["episode_index"], 0)
        self.assertEqual(body["episodes"][0]["frame_count"], FRAME_COUNT)

    def test_episode_summary_includes_sync_diagnostics(self) -> None:
        response = self.client.get("/api/episodes/0")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("sync_diagnostics", body)
        self.assertEqual(body["sync_diagnostics"]["frame_count"], FRAME_COUNT)

    def test_frame_payload_is_semantic_not_raw_field_names(self) -> None:
        response = self.client.get("/api/episodes/0/frames/2")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body["left_hand"]["points"]), 21)
        self.assertEqual(len(body["right_hand"]["points"]), 21)
        self.assertEqual(body["task"], "test task")
        self.assertNotIn("left_hand/tracks", body)

    def test_frame_payload_includes_raw_drill_down_data(self) -> None:
        response = self.client.get("/api/episodes/0/frames/2")
        body = response.get_json()
        self.assertIn("raw", body)
        self.assertEqual(len(body["raw"]["left_hand/tracks"]), 64)
        self.assertEqual(len(body["raw"]["action.left_hand_tracks"]), 63)

    def test_frame_payload_last_frame_action_target_is_null(self) -> None:
        response = self.client.get(f"/api/episodes/0/frames/{FRAME_COUNT - 1}")
        body = response.get_json()
        self.assertIsNone(body["left_action_target"]["points"])
        self.assertIsNone(body["right_action_target"]["points"])

    def test_unknown_frame_returns_structured_404(self) -> None:
        response = self.client.get("/api/episodes/0/frames/999")
        self.assertEqual(response.status_code, 404)
        body = response.get_json()
        self.assertEqual(body["error"]["type"], "FrameNotFoundError")

    def test_unknown_episode_returns_structured_404(self) -> None:
        response = self.client.get("/api/episodes/7")
        self.assertEqual(response.status_code, 404)
        body = response.get_json()
        self.assertEqual(body["error"]["type"], "EpisodeNotFoundError")

    def test_metrics_arrays_have_expected_length_and_frame_zero_is_null(self) -> None:
        response = self.client.get("/api/episodes/0/metrics")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body["left_hand_speed"]), FRAME_COUNT)
        self.assertIsNone(body["left_hand_speed"][0])
        self.assertIsNone(body["camera_speed"][0])
        self.assertEqual(len(body["left_hand_articulation_rate"]), FRAME_COUNT)
        self.assertIsNone(body["left_hand_articulation_rate"][0])
        self.assertGreater(body["left_hand_articulation_rate"][1], 0.0)
        self.assertEqual(body["right_hand_articulation_rate"][1], 0.0)

    def test_nearest_frame_lookup(self) -> None:
        response = self.client.get("/api/episodes/0/nearest?timestamp=0.22")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["frame_index"], 2)

    def test_video_full_request(self) -> None:
        response = self.client.get("/api/episodes/0/video")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"0123456789" * 10)

    def test_video_range_request_returns_partial_content(self) -> None:
        response = self.client.get("/api/episodes/0/video", headers={"Range": "bytes=0-9"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b"0123456789")
        self.assertTrue(response.headers.get("Content-Range", "").startswith("bytes 0-9/"))
        self.assertEqual(response.headers.get("Accept-Ranges"), "bytes")


class AnalysisRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_semantic_dataset(self.root)
        self.summary_path = self.root / "artifacts" / "analysis" / "summary.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _client(self):
        app = create_app(self.root, analysis_summary_path=self.summary_path)
        return app.test_client()

    def test_missing_artifact_returns_structured_404(self) -> None:
        client = self._client()
        response = client.get("/api/analysis")
        self.assertEqual(response.status_code, 404)
        body = response.get_json()
        self.assertEqual(body["error"]["type"], "AnalysisNotComputed")

    def test_present_artifact_is_served_verbatim(self) -> None:
        self.summary_path.parent.mkdir(parents=True)
        payload = {"episode_count": 1, "episodes": []}
        self.summary_path.write_text(json.dumps(payload), encoding="utf-8")
        client = self._client()
        response = client.get("/api/analysis")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)

    def test_analysis_page_is_served(self) -> None:
        client = self._client()
        response = client.get("/analysis.html")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
