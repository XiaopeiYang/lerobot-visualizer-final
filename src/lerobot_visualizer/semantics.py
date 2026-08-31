"""Pure per-frame semantic adapters over already-fetched dataset field values.

These functions turn raw stored Parquet field values (already retrieved through
``LeRobotDataset``) into physically meaningful shapes: 21-landmark point sets for
hand tracks and action targets, and position+quaternion for camera pose. They do
no I/O and do not import the dataset access layer, so they can be reused directly
by future analysis code or a Rerun-based logger against the same semantic data.

Field shapes and coordinate frames used here are METADATA DECLARATIONS from
``data/raw/meta/info.json``, cross-checked structurally in
``docs/phase-2-schema.md`` (Phase 2):

- ``left_hand/tracks`` / ``right_hand/tracks``: ``[ts_ns, 21x(x,y,z)]`` (64 values),
  world frame. VERIFIED: component 0 exactly equals ``timestamp_ns`` on every row.
- ``action.left_hand_tracks`` / ``action.right_hand_tracks``: ``21x(x,y,z)``
  (63 values, no leading ``ts_ns``), world frame, next-frame track target. VERIFIED:
  equals the following row's track payload exactly, except the final episode frame,
  whose action row is an all-zero convention row (not a real spatial target).
- ``base_0_camera/position``: ``[ts_ns,x,y,z]`` (4 values), world frame.
- ``base_0_camera/quaternion_xyzw``: ``[ts_ns,x,y,z,w]`` (5 values), world frame.
- ``observation.*_keypoints_quaternion``: ``21x(x,y,z,w)`` (84 values, no leading
  ``ts_ns``), one per-landmark orientation quaternion per hand-track landmark.
  VERIFIED empirically against the local dataset (not just a metadata
  declaration): every one of the 21 quaternions has unit norm, on every frame.
- ``observation.*_hand_mano``: 61 values decomposing as
  ``global_orient(3) + hand_pose(45) + betas(10) + transl(3)``, camera frame (a
  standard MANO parametric hand-model layout). VERIFIED empirically against the
  local dataset: ``betas`` (shape/personalization) are exactly all-zero on every
  frame of the one local episode -- the shape channel is unpopulated, a
  canonical hand shape is assumed throughout, not a parsing bug; ``transl`` sits
  a bounded 0.06-0.21m from the corresponding ``*_hand/tracks`` landmark-0
  position, consistent with a rigid camera-frame/world-frame offset rather than
  an unrelated quantity.

Full MANO mesh reconstruction (turning ``hand_pose``/``betas`` into a 3D hand
surface) is out of scope -- it needs the licensed MANO body-model assets this
project doesn't ship, and buys little for a data-quality/analysis deliverable.
The parsed values themselves are used directly instead, e.g. as an articulation
signal in ``metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

LANDMARK_COUNT = 21
Point3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

MANO_GLOBAL_ORIENT_LEN = 3
MANO_HAND_POSE_LEN = 45
MANO_BETAS_LEN = 10
MANO_TRANSL_LEN = 3
MANO_TOTAL_LEN = MANO_GLOBAL_ORIENT_LEN + MANO_HAND_POSE_LEN + MANO_BETAS_LEN + MANO_TRANSL_LEN  # 61

KEYPOINT_QUATERNION_LEN = LANDMARK_COUNT * 4  # 84


class SemanticShapeError(ValueError):
    """A raw field did not have the length this adapter expects."""


@dataclass(frozen=True)
class HandTracks:
    points: tuple[Point3, ...]  # length LANDMARK_COUNT


@dataclass(frozen=True)
class ActionTargets:
    # None on the dataset's documented final-frame all-zero convention row --
    # reported as "no target" rather than as 21 fabricated points at the origin.
    points: tuple[Point3, ...] | None


@dataclass(frozen=True)
class CameraPose:
    position: Point3
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class ManoHand:
    """One frame's ``observation.*_hand_mano`` row, decomposed per the standard
    MANO parametric layout -- camera frame, see module docstring for the
    empirical verification behind this decomposition."""

    global_orient: Point3  # axis-angle, camera frame
    hand_pose: tuple[float, ...]  # 45 values: 15 joints x 3 axis-angle
    betas: tuple[float, ...]  # 10 shape parameters; all-zero throughout this dataset
    transl: Point3  # camera-frame translation


@dataclass(frozen=True)
class SemanticFrame:
    left_hand: HandTracks
    right_hand: HandTracks
    left_action_target: ActionTargets
    right_action_target: ActionTargets
    camera_pose: CameraPose


def _chunk_points(flat: Sequence[float]) -> tuple[Point3, ...]:
    return tuple(
        (float(flat[index]), float(flat[index + 1]), float(flat[index + 2]))
        for index in range(0, len(flat), 3)
    )


def parse_hand_tracks(raw: Sequence[float]) -> HandTracks:
    """Parse a stored ``*_hand/tracks`` row: ``[ts_ns, 21x(x,y,z)]`` (64 values)."""
    expected = 1 + LANDMARK_COUNT * 3
    if len(raw) != expected:
        raise SemanticShapeError(f"Expected {expected} values for a hand-track row, got {len(raw)}")
    return HandTracks(points=_chunk_points(raw[1:]))


def parse_action_targets(raw: Sequence[float]) -> ActionTargets:
    """Parse a stored ``action.*_hand_tracks`` row: ``21x(x,y,z)`` (63 values, no ts_ns)."""
    expected = LANDMARK_COUNT * 3
    if len(raw) != expected:
        raise SemanticShapeError(f"Expected {expected} values for an action-track row, got {len(raw)}")
    if all(value == 0 for value in raw):
        return ActionTargets(points=None)
    return ActionTargets(points=_chunk_points(raw))


def parse_camera_pose(position_raw: Sequence[float], quaternion_raw: Sequence[float]) -> CameraPose:
    """Parse ``base_0_camera/position`` (4) and ``base_0_camera/quaternion_xyzw`` (5)."""
    if len(position_raw) != 4:
        raise SemanticShapeError(f"Expected 4 values for camera position, got {len(position_raw)}")
    if len(quaternion_raw) != 5:
        raise SemanticShapeError(f"Expected 5 values for camera quaternion, got {len(quaternion_raw)}")
    return CameraPose(
        position=(float(position_raw[1]), float(position_raw[2]), float(position_raw[3])),
        quaternion_xyzw=(
            float(quaternion_raw[1]),
            float(quaternion_raw[2]),
            float(quaternion_raw[3]),
            float(quaternion_raw[4]),
        ),
    )


def parse_mano_hand(raw: Sequence[float]) -> ManoHand:
    """Parse a stored ``observation.*_hand_mano`` row (61 values, camera frame).

    See the module docstring for the empirical basis of this decomposition.
    """
    if len(raw) != MANO_TOTAL_LEN:
        raise SemanticShapeError(f"Expected {MANO_TOTAL_LEN} values for a MANO hand row, got {len(raw)}")
    global_orient_end = MANO_GLOBAL_ORIENT_LEN
    hand_pose_end = global_orient_end + MANO_HAND_POSE_LEN
    betas_end = hand_pose_end + MANO_BETAS_LEN
    return ManoHand(
        global_orient=(float(raw[0]), float(raw[1]), float(raw[2])),
        hand_pose=tuple(float(v) for v in raw[global_orient_end:hand_pose_end]),
        betas=tuple(float(v) for v in raw[hand_pose_end:betas_end]),
        transl=(float(raw[betas_end]), float(raw[betas_end + 1]), float(raw[betas_end + 2])),
    )


def parse_keypoint_quaternions(raw: Sequence[float]) -> tuple[Quaternion, ...]:
    """Parse a stored ``observation.*_keypoints_quaternion`` row: 21x(x,y,z,w)
    (84 values, no leading ``ts_ns``) -- see module docstring for the empirical
    unit-norm verification behind this shape.
    """
    if len(raw) != KEYPOINT_QUATERNION_LEN:
        raise SemanticShapeError(
            f"Expected {KEYPOINT_QUATERNION_LEN} values for a keypoint-quaternion row, got {len(raw)}"
        )
    return tuple(
        (float(raw[index]), float(raw[index + 1]), float(raw[index + 2]), float(raw[index + 3]))
        for index in range(0, len(raw), 4)
    )


def parse_frame_semantics(values: Mapping[str, Sequence[float]]) -> SemanticFrame:
    """Parse one frame's full semantic content from its raw stored field values.

    ``values`` is expected to be shaped like ``FrameRecord.values`` (i.e. include
    the raw ``left_hand/tracks``, ``right_hand/tracks``, ``action.left_hand_tracks``,
    ``action.right_hand_tracks``, ``base_0_camera/position``, and
    ``base_0_camera/quaternion_xyzw`` fields). A plain dict with the same keys and
    shapes works equally well -- this function has no dependency on the dataset
    access layer's types.
    """
    return SemanticFrame(
        left_hand=parse_hand_tracks(values["left_hand/tracks"]),
        right_hand=parse_hand_tracks(values["right_hand/tracks"]),
        left_action_target=parse_action_targets(values["action.left_hand_tracks"]),
        right_action_target=parse_action_targets(values["action.right_hand_tracks"]),
        camera_pose=parse_camera_pose(
            values["base_0_camera/position"], values["base_0_camera/quaternion_xyzw"]
        ),
    )
