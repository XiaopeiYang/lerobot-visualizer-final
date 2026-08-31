"""Unit tests for whole-episode derived temporal metrics, against hand-computed values."""

from __future__ import annotations

import math
import unittest

from lerobot_visualizer import metrics


def _uniform_hand_track_row(ts_ns: int, point: tuple[float, float, float]) -> list[float]:
    """A hand-track row where all 21 landmarks sit at the same point (centroid == point, span == 0)."""
    flat = [float(ts_ns)]
    for _ in range(21):
        flat.extend(point)
    return flat


def _outlier_hand_track_row(ts_ns: int, outlier: tuple[float, float, float]) -> list[float]:
    """20 landmarks at the origin, 1 at `outlier` -- max pairwise distance == |outlier|."""
    flat = [float(ts_ns)]
    for _ in range(20):
        flat.extend((0.0, 0.0, 0.0))
    flat.extend(outlier)
    return flat


def _action_row(point: tuple[float, float, float] | None) -> list[float]:
    if point is None:
        return [0.0] * 63
    flat: list[float] = []
    for _ in range(21):
        flat.extend(point)
    return flat


def _mano_row(hand_pose_value: float) -> list[float]:
    """A 61-value MANO row with every `hand_pose` component set to `hand_pose_value`
    and everything else zero -- isolates the articulation-rate computation from
    global_orient/betas/transl, which this metric doesn't read."""
    return [0.0] * 3 + [hand_pose_value] * 45 + [0.0] * 10 + [0.0] * 3


class HelperFormulaTests(unittest.TestCase):
    def test_centroid_of_uniform_points(self) -> None:
        points = [(1.0, 2.0, 3.0)] * 21
        self.assertEqual(metrics._centroid(points), (1.0, 2.0, 3.0))

    def test_distance(self) -> None:
        self.assertAlmostEqual(metrics._distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), 5.0)

    def test_max_pairwise_distance_with_one_outlier(self) -> None:
        points = [(0.0, 0.0, 0.0)] * 20 + [(5.0, 0.0, 0.0)]
        self.assertAlmostEqual(metrics._max_pairwise_distance(points), 5.0)

    def test_speed_series_frame_zero_is_none(self) -> None:
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 2.0)]
        timestamps = [0.0, 1.0, 2.0]
        series = metrics._speed_series(positions, timestamps)
        self.assertIsNone(series[0])
        self.assertAlmostEqual(series[1], 1.0)
        self.assertAlmostEqual(series[2], math.sqrt(0 + 4 + 4))


class ComputeEpisodeMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        # 3 synthetic frames, 1-second apart, all hand landmarks moving together
        # (centroid speed is exact) with a per-frame outlier point (span is exact).
        self.frame_indices = (0, 1, 2)
        self.timestamps = (0.0, 1.0, 2.0)
        left_moving_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 2.0)]
        self.left_hand_tracks = [
            _uniform_hand_track_row(i * 1_000_000_000, point) for i, point in enumerate(left_moving_points)
        ]
        self.right_hand_tracks = [_outlier_hand_track_row(i * 1_000_000_000, (float(d), 0.0, 0.0)) for i, d in enumerate([3.0, 4.0, 5.0])]
        # Left action target for frame i exactly equals frame i+1's track point (residual == 0),
        # except the final frame, whose action row is the documented all-zero convention row.
        self.left_action_tracks = [_action_row(left_moving_points[1]), _action_row(left_moving_points[2]), _action_row(None)]
        self.right_action_tracks = [_action_row(None), _action_row(None), _action_row(None)]
        self.camera_positions = [
            [float(i * 1_000_000_000), 0.0, 0.0, float(i)] for i in range(3)
        ]
        # hand_pose component value increases by 1.0 each frame -> per-component
        # delta of 1.0, so ||delta|| over 45 components is sqrt(45) at 1-second
        # intervals.
        self.left_hand_mano = [_mano_row(float(i)) for i in range(3)]
        self.right_hand_mano = [_mano_row(0.0) for _ in range(3)]  # stays still: rate == 0
        self.result = metrics.compute_episode_metrics(
            self.frame_indices,
            self.timestamps,
            self.left_hand_tracks,
            self.right_hand_tracks,
            self.left_action_tracks,
            self.right_action_tracks,
            self.camera_positions,
            self.left_hand_mano,
            self.right_hand_mano,
        )

    def test_left_hand_speed(self) -> None:
        self.assertIsNone(self.result.left_hand_speed[0])
        self.assertAlmostEqual(self.result.left_hand_speed[1], 1.0)
        self.assertAlmostEqual(self.result.left_hand_speed[2], math.sqrt(8))

    def test_right_hand_span_tracks_the_outlier_distance(self) -> None:
        self.assertAlmostEqual(self.result.right_hand_span[0], 3.0)
        self.assertAlmostEqual(self.result.right_hand_span[1], 4.0)
        self.assertAlmostEqual(self.result.right_hand_span[2], 5.0)

    def test_left_hand_span_is_zero_for_uniform_points(self) -> None:
        self.assertEqual(self.result.left_hand_span, (0.0, 0.0, 0.0))

    def test_camera_speed(self) -> None:
        self.assertIsNone(self.result.camera_speed[0])
        self.assertAlmostEqual(self.result.camera_speed[1], 1.0)
        self.assertAlmostEqual(self.result.camera_speed[2], 1.0)

    def test_left_action_residual_zero_where_action_matches_next_track_and_none_at_terminal_row(self) -> None:
        self.assertAlmostEqual(self.result.left_action_residual[0], 0.0)
        self.assertAlmostEqual(self.result.left_action_residual[1], 0.0)
        self.assertIsNone(self.result.left_action_residual[2])

    def test_right_action_residual_is_none_when_action_is_the_zero_convention_row(self) -> None:
        self.assertEqual(self.result.right_action_residual, (None, None, None))

    def test_left_hand_articulation_rate_reflects_changing_pose(self) -> None:
        self.assertIsNone(self.result.left_hand_articulation_rate[0])
        self.assertAlmostEqual(self.result.left_hand_articulation_rate[1], math.sqrt(45))
        self.assertAlmostEqual(self.result.left_hand_articulation_rate[2], math.sqrt(45))

    def test_right_hand_articulation_rate_is_zero_for_a_static_pose(self) -> None:
        self.assertIsNone(self.result.right_hand_articulation_rate[0])
        self.assertAlmostEqual(self.result.right_hand_articulation_rate[1], 0.0)
        self.assertAlmostEqual(self.result.right_hand_articulation_rate[2], 0.0)

    def test_sync_diagnostics_reports_no_gaps_or_duplicates_for_uniform_timestamps(self) -> None:
        diagnostics = self.result.sync_diagnostics
        self.assertEqual(diagnostics.frame_count, 3)
        self.assertAlmostEqual(diagnostics.median_interval_seconds, 1.0)
        self.assertEqual(diagnostics.large_interval_count, 0)
        self.assertEqual(diagnostics.duplicate_timestamp_count, 0)
        self.assertEqual(diagnostics.negative_interval_count, 0)
        self.assertEqual(diagnostics.frame_index_discontinuity_count, 0)


class SyncDiagnosticsFrameLocationTests(unittest.TestCase):
    """A large-gap check alone would miss a timestamp regression -- these tests
    pin down the frame-indexed detail that analysis.py's verified-issue
    detectors rely on to locate each occurrence, not just count it."""

    def test_negative_interval_is_flagged_and_located(self) -> None:
        # Frame 2's timestamp (0.5) is earlier than frame 1's (1.0) -- a regression.
        diagnostics = metrics._sync_diagnostics((0, 1, 2, 3), (0.0, 1.0, 0.5, 1.5))
        self.assertEqual(diagnostics.negative_interval_count, 1)
        self.assertEqual(diagnostics.negative_interval_frame_indices, (2,))

    def test_large_gap_is_located_at_the_frame_after_the_gap(self) -> None:
        diagnostics = metrics._sync_diagnostics((0, 1, 2, 3, 4), (0.0, 1.0, 2.0, 3.0, 10.0))
        self.assertEqual(diagnostics.large_interval_count, 1)
        self.assertEqual(diagnostics.large_interval_frame_indices, (4,))

    def test_duplicate_timestamp_is_located(self) -> None:
        diagnostics = metrics._sync_diagnostics((0, 1, 2), (0.0, 1.0, 1.0))
        self.assertEqual(diagnostics.duplicate_timestamp_count, 1)
        self.assertEqual(diagnostics.duplicate_timestamp_frame_indices, (2,))

    def test_frame_index_discontinuity_is_located(self) -> None:
        # frame_index jumps from 1 to 3 -- frame 2 is missing from the series.
        diagnostics = metrics._sync_diagnostics((0, 1, 3, 4), (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(diagnostics.frame_index_discontinuity_count, 1)
        self.assertEqual(diagnostics.frame_index_discontinuity_frame_indices, (3,))

    def test_strictly_sequential_frame_indices_report_no_discontinuity(self) -> None:
        diagnostics = metrics._sync_diagnostics((5, 6, 7, 8), (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(diagnostics.frame_index_discontinuity_count, 0)


if __name__ == "__main__":
    unittest.main()
