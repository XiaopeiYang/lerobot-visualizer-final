"""Unit tests for pure per-frame semantic adapters (no dataset/tempfile needed)."""

from __future__ import annotations

import unittest

from lerobot_visualizer import semantics


def _hand_track_row(ts_ns: int, points: list[tuple[float, float, float]]) -> list[float]:
    flat: list[float] = [float(ts_ns)]
    for point in points:
        flat.extend(point)
    return flat


def _action_row(points: list[tuple[float, float, float]]) -> list[float]:
    flat: list[float] = []
    for point in points:
        flat.extend(point)
    return flat


class ParseHandTracksTests(unittest.TestCase):
    def test_strips_leading_timestamp_and_reshapes_to_21_points(self) -> None:
        points = [(float(i), float(i) * 2, float(i) * 3) for i in range(21)]
        row = _hand_track_row(ts_ns=1_234, points=points)
        result = semantics.parse_hand_tracks(row)
        self.assertEqual(len(result.points), 21)
        self.assertEqual(result.points[0], (0.0, 0.0, 0.0))
        self.assertEqual(result.points[5], (5.0, 10.0, 15.0))
        self.assertEqual(result.points[20], (20.0, 40.0, 60.0))

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_hand_tracks([0.0] * 63)
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_hand_tracks([0.0] * 65)


class ParseActionTargetsTests(unittest.TestCase):
    def test_nonzero_row_reshapes_to_21_points(self) -> None:
        points = [(1.0, 2.0, 3.0)] * 21
        row = _action_row(points)
        result = semantics.parse_action_targets(row)
        self.assertIsNotNone(result.points)
        assert result.points is not None
        self.assertEqual(len(result.points), 21)
        self.assertEqual(result.points[0], (1.0, 2.0, 3.0))

    def test_all_zero_row_is_none_not_fabricated_points(self) -> None:
        row = [0.0] * 63
        result = semantics.parse_action_targets(row)
        self.assertIsNone(result.points)

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_action_targets([0.0] * 62)


class ParseCameraPoseTests(unittest.TestCase):
    def test_strips_leading_timestamp_from_both_fields(self) -> None:
        position = [999.0, 1.0, 2.0, 3.0]
        quaternion = [999.0, 0.1, 0.2, 0.3, 0.9]
        result = semantics.parse_camera_pose(position, quaternion)
        self.assertEqual(result.position, (1.0, 2.0, 3.0))
        self.assertEqual(result.quaternion_xyzw, (0.1, 0.2, 0.3, 0.9))

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_camera_pose([1.0, 2.0, 3.0], [999.0, 0.1, 0.2, 0.3, 0.9])
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_camera_pose([999.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.9])


class ParseManoHandTests(unittest.TestCase):
    def test_decomposes_into_global_orient_hand_pose_betas_transl(self) -> None:
        row = [1.0, 2.0, 3.0] + [10.0 + i for i in range(45)] + [float(i) for i in range(10)] + [7.0, 8.0, 9.0]
        result = semantics.parse_mano_hand(row)
        self.assertEqual(result.global_orient, (1.0, 2.0, 3.0))
        self.assertEqual(len(result.hand_pose), 45)
        self.assertEqual(result.hand_pose[0], 10.0)
        self.assertEqual(result.hand_pose[-1], 54.0)
        self.assertEqual(len(result.betas), 10)
        self.assertEqual(result.betas, tuple(float(i) for i in range(10)))
        self.assertEqual(result.transl, (7.0, 8.0, 9.0))

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_mano_hand([0.0] * 60)
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_mano_hand([0.0] * 62)


class ParseKeypointQuaternionsTests(unittest.TestCase):
    def test_reshapes_84_values_into_21_quaternions(self) -> None:
        row = []
        for i in range(21):
            row.extend([float(i), 0.0, 0.0, 1.0])
        result = semantics.parse_keypoint_quaternions(row)
        self.assertEqual(len(result), 21)
        self.assertEqual(result[0], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(result[20], (20.0, 0.0, 0.0, 1.0))

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_keypoint_quaternions([0.0] * 83)
        with self.assertRaises(semantics.SemanticShapeError):
            semantics.parse_keypoint_quaternions([0.0] * 85)


class ParseFrameSemanticsTests(unittest.TestCase):
    def test_assembles_all_fields_from_a_values_mapping(self) -> None:
        points21 = [(0.0, 0.0, 0.0)] * 21
        values = {
            "left_hand/tracks": _hand_track_row(1, points21),
            "right_hand/tracks": _hand_track_row(2, points21),
            "action.left_hand_tracks": _action_row([(1.0, 1.0, 1.0)] * 21),
            "action.right_hand_tracks": _action_row([(0.0, 0.0, 0.0)] * 21),
            "base_0_camera/position": [3, 0.1, 0.2, 0.3],
            "base_0_camera/quaternion_xyzw": [3, 0.0, 0.0, 0.0, 1.0],
        }
        frame = semantics.parse_frame_semantics(values)
        self.assertEqual(len(frame.left_hand.points), 21)
        self.assertEqual(len(frame.right_hand.points), 21)
        self.assertIsNotNone(frame.left_action_target.points)
        self.assertIsNone(frame.right_action_target.points)
        self.assertEqual(frame.camera_pose.position, (0.1, 0.2, 0.3))
        self.assertEqual(frame.camera_pose.quaternion_xyzw, (0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
