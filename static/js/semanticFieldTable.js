// Static documentation of the raw-shape -> semantic-shape mapping for the
// fields `analysis.py`'s `_ANALYSIS_FIELDS` operates on. This is NOT computed
// data -- it's a hand-authored description of `semantics.py`'s parsing plus
// `data/raw/meta/info.json`'s own per-field metadata, reproduced here so the
// live Analysis page doesn't require also reading docs/analysis-report.md's
// §2. If `semantics.py`'s parsing or info.json's `features` dict changes,
// update this table by hand to match -- it is intentionally not derived from
// the JSON artifact, since field semantics are dataset-schema-level facts,
// not a per-run measurement (unlike findingsData.js's numbers, which ARE
// pulled live from the analysis artifact for exactly this reason).
//
// Kept intentionally compact: per-field verification detail (e.g. "component
// 0 exactly equals timestamp_ns") lives in the code that checked it, not
// here -- this table's job is naming what each field means, not re-proving
// it. Genuinely important exceptions/assumptions go in EXTERNAL_CONVENTIONS
// below instead of being folded into this table.

export const SEMANTIC_FIELD_TABLE = [
  {
    field: "left_hand/tracks, right_hand/tracks",
    semanticMeaning: "21 hand landmarks",
    shape: "21 × 3",
    frame: "world",
  },
  {
    field: "action.left_hand_tracks, action.right_hand_tracks",
    semanticMeaning: "next-frame hand target",
    shape: "21 × 3",
    frame: "world",
  },
  {
    field: "base_0_camera/position",
    semanticMeaning: "head-camera position",
    shape: "3",
    frame: "world",
  },
  {
    field: "base_0_camera/quaternion_xyzw",
    semanticMeaning: "head-camera orientation",
    shape: "4",
    frame: "world",
  },
  {
    field: "observation.*_hand_mano",
    semanticMeaning: "MANO pose parameters",
    shape: "61",
    frame: "camera",
  },
  {
    field: "observation.*_keypoints_quaternion",
    semanticMeaning: "joint orientations",
    shape: "21 × 4",
    frame: "world",
  },
];

// Assumptions this project relies on that data/raw/meta/info.json never
// states anywhere -- distinct from the per-field table above, because each of
// these is either a cross-field correspondence or an external naming
// convention, not a property of one field in isolation. Tier "unknown" is
// used (not "inference") where the dataset genuinely provides no way to
// establish the claim at all, per the 5-tier scheme's own definition of
// "unknown": not established by the available evidence.
export const EXTERNAL_CONVENTIONS = [
  {
    title: "Hand landmark identity / skeletal topology",
    tier: "unknown",
    text:
      "Which of the 21 points is the wrist vs. a fingertip, and how they connect, is not stated anywhere; the optional skeleton overlay assumes MediaPipe Hands topology only because the point count matches.",
  },
  {
    title: "MANO hand_pose's internal 15-joint ordering",
    tier: "inference",
    text:
      "The 45-value hand_pose segment is assumed to follow the standard MANO joint ordering, an external convention this dataset doesn't itemize. It's only ever used pooled (articulation rate), never decomposed per-joint.",
  },
  {
    title: "keypoints_quaternion[i] ↔ *_hand/tracks[i] correspondence",
    tier: "inference",
    text:
      "Index i is assumed to name the same landmark in both fields, since both are declared as 21-element hand-landmark arrays -- info.json never states this explicitly.",
  },
];

// Carried forward from Phase 2, not re-derived by this page's data pipeline.
export const SCHEMA_DISCREPANCY_NOTE =
  "All ten nested numeric fields are declared float32 in info.json but physically stored as float64. Cause not investigated by this analysis pass.";
