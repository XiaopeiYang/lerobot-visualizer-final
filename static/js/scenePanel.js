// Semantic 3D spatial view: current hand tracks (solid points), action targets
// (hollow wireframe markers), current -> target vectors (arrows), and an optional
// camera-pose gizmo + hand-skeleton overlay. Subscribes to viewerState; does not
// originate navigation itself (no 3D-click-to-seek in this first cut).
//
// Kept modular (semantic frame in, Three.js objects out) so a future Rerun-based
// debug viewer could log the same left_hand/right_hand/action_target/camera_pose
// semantic data without depending on any of this rendering code.

import * as THREE from "../vendor/three/three.module.js";
import { OrbitControls } from "../vendor/three/examples/jsm/controls/OrbitControls.js";
import { worldToThreeVector3, worldQuaternionToThree } from "./threeAdapter.js";
import { HAND_CONNECTIONS } from "./handSkeleton.js";
import { createLatestFrameQueue } from "./latestFrameQueue.mjs";
import * as api from "./api.js";

const LEFT_COLOR = 0x4da3ff;
const RIGHT_COLOR = 0xff9d4d;
const LANDMARK_COUNT = 21;

function makeSolidPoints(color, { opacity = 1 } = {}) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(LANDMARK_COUNT * 3), 3));
  const material = new THREE.PointsMaterial({
    color,
    size: 0.014,
    sizeAttenuation: true,
    transparent: opacity < 1,
    opacity,
  });
  const points = new THREE.Points(geometry, material);
  points.visible = false;
  return points;
}

function setSolidPoints(pointsObject, points) {
  if (!points) {
    pointsObject.visible = false;
    return;
  }
  pointsObject.visible = true;
  const positions = pointsObject.geometry.attributes.position.array;
  points.forEach((p, i) => {
    const v = worldToThreeVector3(p);
    positions[i * 3] = v.x;
    positions[i * 3 + 1] = v.y;
    positions[i * 3 + 2] = v.z;
  });
  pointsObject.geometry.attributes.position.needsUpdate = true;
  pointsObject.geometry.computeBoundingSphere();
}

// Hollow wireframe markers for action targets -- deliberately a different visual
// style (outline only, not filled) from the solid current-track points, and dimmer
// so the current/target distinction reads clearly without relying on color alone.
function makeHollowTargetGroup(color) {
  const group = new THREE.Group();
  const geometry = new THREE.IcosahedronGeometry(0.009, 0);
  const material = new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.7 });
  for (let i = 0; i < LANDMARK_COUNT; i++) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.visible = false;
    group.add(mesh);
  }
  return group;
}

function setHollowTargets(group, points) {
  if (!points) {
    group.visible = false;
    return;
  }
  group.visible = true;
  points.forEach((p, i) => {
    group.children[i].position.copy(worldToThreeVector3(p));
    group.children[i].visible = true;
  });
}

function setArrows(group, currentPoints, targetPoints, color) {
  group.clear();
  if (!currentPoints || !targetPoints) return;
  for (let i = 0; i < currentPoints.length; i++) {
    const from = worldToThreeVector3(currentPoints[i]);
    const to = worldToThreeVector3(targetPoints[i]);
    const offset = new THREE.Vector3().subVectors(to, from);
    const length = offset.length();
    if (length < 1e-5) continue;
    const direction = offset.clone().normalize();
    const headLength = Math.min(0.008, length * 0.4);
    const headWidth = Math.min(0.005, length * 0.3);
    group.add(new THREE.ArrowHelper(direction, from, length, color, headLength, headWidth));
  }
}

function setSkeleton(lineSegments, points) {
  if (!points) {
    lineSegments.visible = false;
    return;
  }
  const positions = new Float32Array(HAND_CONNECTIONS.length * 6);
  HAND_CONNECTIONS.forEach(([a, b], edgeIndex) => {
    const pa = worldToThreeVector3(points[a]);
    const pb = worldToThreeVector3(points[b]);
    const offset = edgeIndex * 6;
    positions[offset] = pa.x;
    positions[offset + 1] = pa.y;
    positions[offset + 2] = pa.z;
    positions[offset + 3] = pb.x;
    positions[offset + 4] = pb.y;
    positions[offset + 5] = pb.z;
  });
  lineSegments.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  lineSegments.geometry.attributes.position.needsUpdate = true;
  lineSegments.visible = true;
}

export function createScenePanel({ container, viewerState }) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14171f);

  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 50);
  camera.position.set(0.3, 0.3, 0.6);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();

  scene.add(new THREE.AxesHelper(0.1));
  scene.add(new THREE.GridHelper(1, 10, 0x333844, 0x22252e));
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));

  const leftPoints = makeSolidPoints(LEFT_COLOR);
  const rightPoints = makeSolidPoints(RIGHT_COLOR);
  // "Next-frame track" -- a real fetch of frame i+1's own track (not a relabeling
  // of the action-target markers) so it visually demonstrates, rather than
  // assumes, docs/analysis-report.md's finding that action[i] == track[i+1].
  // Distinct translucent style so it reads as "ghost" data, default hidden.
  const nextLeftPoints = makeSolidPoints(LEFT_COLOR, { opacity: 0.4 });
  const nextRightPoints = makeSolidPoints(RIGHT_COLOR, { opacity: 0.4 });
  const leftTargets = makeHollowTargetGroup(LEFT_COLOR);
  const rightTargets = makeHollowTargetGroup(RIGHT_COLOR);
  const leftArrows = new THREE.Group();
  const rightArrows = new THREE.Group();
  const leftSkeleton = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: LEFT_COLOR })
  );
  const rightSkeleton = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: RIGHT_COLOR })
  );
  leftSkeleton.visible = false;
  rightSkeleton.visible = false;
  const cameraGizmo = new THREE.AxesHelper(0.05);

  scene.add(
    leftPoints,
    rightPoints,
    nextLeftPoints,
    nextRightPoints,
    leftTargets,
    rightTargets,
    leftArrows,
    rightArrows,
    leftSkeleton,
    rightSkeleton,
    cameraGizmo
  );

  let skeletonEnabled = false;
  let showCamera = true;
  let showCurrentEnabled = true;
  let showTargetEnabled = true;
  let showNextEnabled = false;
  let lastFrame = null;

  function hideNextTrack() {
    nextLeftPoints.visible = false;
    nextRightPoints.visible = false;
  }

  function applyNextFrame(frame) {
    if (!showNextEnabled) return; // toggled off again while the fetch was in flight
    setSolidPoints(nextLeftPoints, frame.left_hand.points);
    setSolidPoints(nextRightPoints, frame.right_hand.points);
  }

  const nextFrameQueue = createLatestFrameQueue({
    loadFrame: api.getFrame,
    onFrame: applyNextFrame,
    // Most commonly a 404 for the episode's terminal frame (there is no i+1) --
    // hide rather than fabricate, same "no target" convention semantics.py
    // already applies to the terminal action row.
    onError: hideNextTrack,
  });

  function applyFrame(frame) {
    lastFrame = frame;
    setSolidPoints(leftPoints, showCurrentEnabled ? frame.left_hand.points : null);
    setSolidPoints(rightPoints, showCurrentEnabled ? frame.right_hand.points : null);
    setHollowTargets(leftTargets, showTargetEnabled ? frame.left_action_target.points : null);
    setHollowTargets(rightTargets, showTargetEnabled ? frame.right_action_target.points : null);
    // Arrows always originate from the real current point regardless of whether
    // the "Show current track" dots are separately visible; only "Show action
    // target" gates whether an arrow is drawn at all.
    setArrows(leftArrows, frame.left_hand.points, showTargetEnabled ? frame.left_action_target.points : null, LEFT_COLOR);
    setArrows(rightArrows, frame.right_hand.points, showTargetEnabled ? frame.right_action_target.points : null, RIGHT_COLOR);

    if (skeletonEnabled) {
      setSkeleton(leftSkeleton, frame.left_hand.points);
      setSkeleton(rightSkeleton, frame.right_hand.points);
    } else {
      leftSkeleton.visible = false;
      rightSkeleton.visible = false;
    }

    cameraGizmo.visible = showCamera;
    if (showCamera) {
      cameraGizmo.position.copy(worldToThreeVector3(frame.camera_pose.position));
      cameraGizmo.setRotationFromQuaternion(worldQuaternionToThree(frame.camera_pose.quaternion_xyzw));
    }

    if (showNextEnabled) {
      nextFrameQueue.requestFrame(frame.episode_index, frame.frame_index + 1);
    } else {
      hideNextTrack();
    }
  }

  const frameQueue = createLatestFrameQueue({
    loadFrame: api.getFrame,
    onFrame: applyFrame,
    onError(error, request) {
      console.error(`Failed to load 3D frame ${request.episodeIndex}/${request.frameIndex}`, error);
    },
  });

  viewerState.subscribe(({ episodeIndex, frameIndex }) => {
    frameQueue.requestFrame(episodeIndex, frameIndex);
  });

  function setSkeletonEnabled(enabled) {
    skeletonEnabled = enabled;
    if (lastFrame !== null) applyFrame(lastFrame);
  }

  function setCameraGizmoEnabled(enabled) {
    showCamera = enabled;
    cameraGizmo.visible = enabled;
  }

  function setShowCurrent(enabled) {
    showCurrentEnabled = enabled;
    if (lastFrame !== null) applyFrame(lastFrame);
  }

  function setShowTarget(enabled) {
    showTargetEnabled = enabled;
    if (lastFrame !== null) applyFrame(lastFrame);
  }

  function setShowNext(enabled) {
    showNextEnabled = enabled;
    if (!enabled) hideNextTrack();
    if (lastFrame !== null) applyFrame(lastFrame);
  }

  function handleResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
  window.addEventListener("resize", handleResize);

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();

  return { setSkeletonEnabled, setCameraGizmoEnabled, setShowCurrent, setShowTarget, setShowNext };
}
