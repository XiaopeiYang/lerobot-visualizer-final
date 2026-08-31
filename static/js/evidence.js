// Pure, DOM-free resolution of "given episode/frame/metric state, what's the
// evidence." Shared by the viewer (frameInspector.js, chartsPanel.js via app.js)
// and the Dataset Analysis dashboard (analysisPanel.js) so this logic survives a
// future reorganization of either page's layout/DOM without being rewritten --
// every function here takes and returns plain data (arrays, numbers, strings),
// never touches `document`, and the episode/frame/metric URL contract
// (`buildLink`/`parseParams`) is the single place either page's navigation goes
// through, so the pages can link to each other without depending on each
// other's markup.
//
// Metric names used throughout match `analysis.py`'s `named_series`/`global`
// keys exactly (`hand_speed`, `hand_span`, `camera_speed`, `quaternion_norm`,
// `action_residual_k1`), so a motion event's `metric` field, a
// `report.global[metric]` distribution summary, and a per-frame evidence row
// are always the same string.
//
// "Motion event" (not "anomaly"): a frame where one metric crosses a
// statistical threshold is evidence of unusual motion, not by itself evidence
// of a data defect -- see analysis.py's module docstring for the full 3-tier
// evidence system (motion event / suspicious event / verified issue). This
// file only ever reads `report.motion_events` (never invents its own
// tiering), so the Viewer and the Analysis page always agree on which frames
// are flagged and why.

export const METRIC_LABELS = {
  hand_speed: "Hand centroid speed (m/s)",
  hand_span: "Hand span (m)",
  camera_speed: "Camera speed (m/s)",
  quaternion_norm: "Camera quaternion norm",
  action_residual_k1: "Action-to-next-track residual (m)",
};

export function filterMotionEventsByEpisode(report, episodeIndex) {
  if (!report) return [];
  return report.motion_events.filter((e) => e.episode_index === episodeIndex);
}

export function filterMotionEventsByFrame(motionEvents, frameIndex) {
  return motionEvents.filter((e) => e.frame_index === frameIndex);
}

// Map<metric, MotionEvent[]>, for panels that render per-metric (e.g. one
// chart per metric) without needing to know the full motion-event list shape.
export function groupMotionEventsByMetric(motionEvents) {
  const groups = new Map();
  for (const event of motionEvents) {
    if (!groups.has(event.metric)) groups.set(event.metric, []);
    groups.get(event.metric).push(event);
  }
  return groups;
}

function quaternionNorm([x, y, z, w]) {
  return Math.sqrt(x * x + y * y + z * z + w * w);
}

function maxAbsDefined(values) {
  const defined = values.filter((v) => v !== null && v !== undefined);
  return defined.length ? Math.max(...defined.map(Math.abs)) : null;
}

// The per-metric comparison rows for one frame: current value (from the
// episode's already-fetched metrics arrays and the current frame payload) vs.
// the dataset-wide baseline (median, from /api/analysis's `global` section),
// plus whether this exact frame/metric pair is a flagged motion event (and if
// so, which tier). `metrics` is the object returned by
// GET /api/episodes/<id>/metrics (already fetched once per episode by every
// caller); `frame` is one GET /api/episodes/<id>/frames/<n> payload;
// `globalStats` is `report.global` from GET /api/analysis (`null` if the
// analysis artifact hasn't been computed yet -- every row's `baselineMedian`
// is then `null` and callers should render "n/a", not omit the row).
export function computeFrameEvidence({ metrics, frame, globalStats, motionEventsForFrame = [] }) {
  if (!metrics || !frame) return [];
  const position = metrics.frame_indices.indexOf(frame.frame_index);
  if (position === -1) return [];

  // `hand_speed`/`hand_span`/`action_residual_k1` pool both hands into one named
  // series server-side (analysis.py), so a single frame can produce two motion
  // events sharing the same (episode, frame, metric) key -- one per hand.
  // Keep the more significant one (larger |robust z|) rather than whichever
  // happened to appear last.
  const flaggedByMetric = new Map();
  for (const event of motionEventsForFrame) {
    const existing = flaggedByMetric.get(event.metric);
    if (!existing || Math.abs(event.robust_z) > Math.abs(existing.robust_z)) {
      flaggedByMetric.set(event.metric, event);
    }
  }
  const values = {
    hand_speed: maxAbsDefined([metrics.left_hand_speed[position], metrics.right_hand_speed[position]]),
    hand_span: Math.max(metrics.left_hand_span[position], metrics.right_hand_span[position]),
    camera_speed: metrics.camera_speed[position] ?? null,
    quaternion_norm: quaternionNorm(frame.camera_pose.quaternion_xyzw),
    action_residual_k1: maxAbsDefined([metrics.left_action_residual[position], metrics.right_action_residual[position]]),
  };

  return Object.keys(METRIC_LABELS).map((metric) => {
    const flagged = flaggedByMetric.get(metric) ?? null;
    return {
      metric,
      label: METRIC_LABELS[metric],
      value: values[metric],
      baselineMedian: globalStats?.[metric]?.median ?? null,
      flagged: flagged !== null,
      tier: flagged?.tier ?? null,
      robustZ: flagged?.robust_z ?? null,
    };
  });
}

// Central URL contract for every analysis<->viewer interaction. `page` is a
// path relative to the current origin (e.g. "/" for the viewer, "/analysis.html"
// for the dashboard); any param left `undefined`/`null` is omitted.
export function buildLink(page, { episodeIndex, frameIndex, metric } = {}) {
  const url = new URL(page, window.location.href);
  if (episodeIndex !== undefined && episodeIndex !== null) url.searchParams.set("episode", episodeIndex);
  if (frameIndex !== undefined && frameIndex !== null) url.searchParams.set("frame", frameIndex);
  if (metric) url.searchParams.set("metric", metric);
  return url.toString();
}

export function parseParams(search = window.location.search) {
  const params = new URLSearchParams(search);
  return {
    episodeIndex: params.has("episode") ? Number(params.get("episode")) : undefined,
    frameIndex: params.has("frame") ? Number(params.get("frame")) : undefined,
    metric: params.get("metric") || undefined,
  };
}
