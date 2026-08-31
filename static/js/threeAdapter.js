// The single, isolated axis-remap boundary between the dataset's semantic JSON and
// Three.js's rendering convention. This is a PRESENTATION convention only: the JSON
// API and semantics.py keep the dataset's native world-frame numbers untouched, so a
// future Rerun consumer (or any other consumer of the same semantic payload) can apply
// its own convention. Only this file, and scenePanel.js which calls it, know about the
// Three.js coordinate flip.
//
// info.json._coordinate_note (METADATA DECLARATION, data/raw/meta/info.json):
//   "per-feature frame: hand tracks/keypoint_quaternions/action=WORLD;
//    base_0_camera pose=WORLD; MANO=CAMERA. axes +x right +y down +z forward"
//
// The dataset's world frame (+x right, +y down, +z forward) is a right-handed
// image/camera-style convention (x cross y = z). Three.js uses a right-handed
// +y-up convention with the camera looking down -z by default. Converting between
// them by flipping Y and Z -- (x, y, z) -> (x, -y, -z) -- is a proper 180 degree
// rotation about the X axis (determinant +1), which preserves handedness/orientation.
// A single-axis flip would instead be an improper reflection (determinant -1) and
// would silently produce mirrored geometry, so both axes are flipped together.
//
// Because this is a proper rotation, the same (x, -y, -z) transform applies
// consistently to both position vectors and a rotation quaternion's vector part
// (the quaternion's w/scalar component is invariant under a change of basis).

import * as THREE from "../vendor/three/three.module.js";

export function worldToThreeVector3([x, y, z]) {
  return new THREE.Vector3(x, -y, -z);
}

export function worldQuaternionToThree([x, y, z, w]) {
  // THREE.Quaternion's constructor order is already (x, y, z, w), matching the
  // stored `quaternion_xyzw` order once the leading ts_ns component is stripped --
  // no component reordering needed, only the axis remap below.
  return new THREE.Quaternion(x, -y, -z, w);
}
