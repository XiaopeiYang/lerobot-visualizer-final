// Optional hand-skeleton connectivity, default OFF.
//
// This dataset does NOT define landmark connectivity anywhere in its metadata --
// only 21 unconnected (x, y, z) points per hand. The 21-landmark count matches the
// widely published MediaPipe Hands topology, so that topology is used here as an
// external, well-known convention for an *optional* display overlay -- it is not a
// fact about this dataset and must never be presented as one.
//
// Source (fetched and verified directly, not recalled from memory):
//   https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/python/solutions/hands_connections.py
// Landmark 0 = wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.

const HAND_PALM_CONNECTIONS = [
  [0, 1], [0, 5], [9, 13], [13, 17], [5, 9], [0, 17],
];
const HAND_THUMB_CONNECTIONS = [[1, 2], [2, 3], [3, 4]];
const HAND_INDEX_FINGER_CONNECTIONS = [[5, 6], [6, 7], [7, 8]];
const HAND_MIDDLE_FINGER_CONNECTIONS = [[9, 10], [10, 11], [11, 12]];
const HAND_RING_FINGER_CONNECTIONS = [[13, 14], [14, 15], [15, 16]];
const HAND_PINKY_FINGER_CONNECTIONS = [[17, 18], [18, 19], [19, 20]];

export const HAND_CONNECTIONS = [
  ...HAND_PALM_CONNECTIONS,
  ...HAND_THUMB_CONNECTIONS,
  ...HAND_INDEX_FINGER_CONNECTIONS,
  ...HAND_MIDDLE_FINGER_CONNECTIONS,
  ...HAND_RING_FINGER_CONNECTIONS,
  ...HAND_PINKY_FINGER_CONNECTIONS,
];

export const HAND_SKELETON_LABEL =
  "Show hand skeleton (MediaPipe Hands 21-point topology — external convention, not derived from this dataset)";
