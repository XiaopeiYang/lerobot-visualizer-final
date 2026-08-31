// Frame Inspector: the ML-engineer-facing "what is this frame, exactly" panel.
// Structured into Overview / Semantics / Diagnostics / Evidence, plus a
// collapsible Raw Frame Data drill-down. Semantic rows show their source Parquet
// field name; clicking it opens and scrolls to that field in the raw drill-down --
// this keeps the primary view semantic (per the architecture's "don't reinterpret
// raw fields in the UI" rule) while still making the underlying raw values
// reachable for debugging.
//
// The Evidence group and evidenceLink() resolve entirely through evidence.js's
// episode/frame/metric state contract (see that module's header comment) -- this
// panel only renders whatever evidence.js computes, so a future reorganization of
// this page's DOM doesn't require touching evidence logic. evidenceLink() builds a
// URL back into this exact episode/frame; the Dataset Analysis dashboard
// (analysisPanel.js) is the current consumer via its own "Open in Viewer" links.

import * as api from "./api.js";
import * as evidence from "./evidence.js";

// Display-only heuristic thresholds for the Diagnostics "Status" readout. These do
// not assert anything new about the dataset -- they just flag when this frame's
// already-computed Δt or action-residual is unusually large, using reference values
// (median interval, large-interval threshold) that metrics.py already computes.
const RESIDUAL_CHECK_THRESHOLD = 1e-3;

function formatVector(values) {
  return `[${values.map((v) => v.toFixed(4)).join(", ")}]`;
}

function row(label, value, sourceField, extraAttrs = "") {
  const sourceHtml = sourceField
    ? `<button class="sourceLink" data-source-field="${sourceField}">source: ${sourceField}</button>`
    : "";
  return `<div class="inspectorRow"${extraAttrs}><span>${label}</span><span class="value">${value}${sourceHtml}</span></div>`;
}

function renderRawField(name, value) {
  return `
    <div class="rawField" data-raw-field="${name}">
      <span class="rawFieldName">${name}</span>
      <pre>${JSON.stringify(value)}</pre>
    </div>
  `;
}

function formatEvidenceValue(value) {
  return value === null || value === undefined ? "n/a" : Number(value).toPrecision(4);
}

// Evidence group: per-metric current-value-vs-dataset-median comparison for this
// exact frame, plus a link to the Dataset Analysis dashboard when this frame is a
// flagged motion event for one of those metrics. Resolves entirely through
// evidence.js's episode/frame/metric contract -- no dependency on this panel's own
// DOM layout, so it survives a future reorganization of either page.
function renderEvidence(episodeIndex, frameIndex, rows) {
  if (!rows.length) return "";
  const rowsHtml = rows
    .map((entry) => {
      const status = entry.flagged
        ? `<span class="statusCheck">${entry.tier} (z=${entry.robustZ.toFixed(1)})</span>`
        : `<span class="statusNormal">normal</span>`;
      const link = entry.flagged
        ? ` <a class="evidenceLink" href="${evidence.buildLink("/analysis.html", { episodeIndex, frameIndex, metric: entry.metric })}">view in analysis</a>`
        : "";
      return row(
        entry.label,
        `${formatEvidenceValue(entry.value)} (median ${formatEvidenceValue(entry.baselineMedian)}) ${status}${link}`,
        undefined,
        ` data-metric="${entry.metric}"`
      );
    })
    .join("");
  return `
    <details class="inspectorGroup" open>
      <summary>Evidence</summary>
      ${rowsHtml}
    </details>
  `;
}

export function createFrameInspector({ container, viewerState }) {
  let latestRequestId = 0;
  let metrics = null; // set via setMetrics(): { timestamps, left_action_residual, right_action_residual, sync_diagnostics }
  let motionEventsForEpisode = [];
  let globalStats = null;
  // Set via focusMetric() for an incoming "Open in Viewer" deep link from a
  // motion-event row; consumed (and cleared) on the next render() so it only
  // ever highlights the frame that link actually pointed at, not later frames.
  let pendingMetricFocus = null;

  function diagnosticsFor(frameIndex) {
    if (!metrics) return null;
    const position = metrics.frame_indices.indexOf(frameIndex);
    if (position === -1) return null;
    const deltaSeconds = position > 0 ? metrics.timestamps[position] - metrics.timestamps[position - 1] : null;
    const leftResidual = metrics.left_action_residual[position];
    const rightResidual = metrics.right_action_residual[position];
    const residual = [leftResidual, rightResidual].filter((v) => v !== null && v !== undefined);
    const maxResidual = residual.length ? Math.max(...residual.map(Math.abs)) : null;
    const threshold = metrics.sync_diagnostics.large_interval_threshold_seconds;
    const isCheck =
      (deltaSeconds !== null && threshold !== null && deltaSeconds > threshold) ||
      (maxResidual !== null && maxResidual > RESIDUAL_CHECK_THRESHOLD);
    return { deltaSeconds, maxResidual, isCheck };
  }

  async function render(episodeIndex, frameIndex) {
    if (episodeIndex === null || episodeIndex === undefined || frameIndex === null || frameIndex === undefined) {
      return;
    }
    const requestId = ++latestRequestId;
    const frame = await api.getFrame(episodeIndex, frameIndex);
    if (requestId !== latestRequestId) return;

    const diagnostics = diagnosticsFor(frameIndex);

    const overview = `
      <details class="inspectorGroup" open>
        <summary>Overview</summary>
        ${row("Frame", `${frame.frame_index}`)}
        ${row("Timestamp", `${frame.timestamp.toFixed(6)} s${frame.timestamp_ns !== null ? ` (${frame.timestamp_ns} ns)` : ""}`)}
        ${row("Task", frame.task ?? "n/a")}
      </details>
    `;

    const semanticsHtml = `
      <details class="inspectorGroup" open>
        <summary>Semantics</summary>
        ${row("Left hand track", "timestamp, 21 &times; (x,y,z)", "left_hand/tracks")}
        ${row("Right hand track", "timestamp, 21 &times; (x,y,z)", "right_hand/tracks")}
        ${row(
          "Left action target",
          frame.left_action_target.points !== null ? "21 &times; (x,y,z)" : "none (terminal frame)",
          "action.left_hand_tracks"
        )}
        ${row(
          "Right action target",
          frame.right_action_target.points !== null ? "21 &times; (x,y,z)" : "none (terminal frame)",
          "action.right_hand_tracks"
        )}
        ${row("Camera position", formatVector(frame.camera_pose.position), "base_0_camera/position")}
        ${row("Camera quaternion (xyzw)", formatVector(frame.camera_pose.quaternion_xyzw), "base_0_camera/quaternion_xyzw")}
      </details>
    `;

    const diagnosticsHtml = diagnostics
      ? `
      <details class="inspectorGroup" open>
        <summary>Diagnostics</summary>
        ${row("&Delta;t (previous frame)", diagnostics.deltaSeconds !== null ? `${(diagnostics.deltaSeconds * 1000).toFixed(3)} ms` : "n/a (first frame)")}
        ${row("Action residual (max |L|,|R|)", diagnostics.maxResidual !== null ? diagnostics.maxResidual.toExponential(2) : "n/a")}
        ${row("Status", `<span class="${diagnostics.isCheck ? "statusCheck" : "statusNormal"}">${diagnostics.isCheck ? "Check" : "Normal"}</span>`)}
      </details>
    `
      : "";

    const motionEventsForFrame = evidence.filterMotionEventsByFrame(motionEventsForEpisode, frameIndex);
    const evidenceRows = evidence.computeFrameEvidence({ metrics, frame, globalStats, motionEventsForFrame });
    const evidenceHtml = renderEvidence(episodeIndex, frameIndex, evidenceRows);

    const rawFieldsHtml = Object.entries(frame.raw)
      .map(([name, value]) => renderRawField(name, value))
      .join("");

    container.innerHTML = `
      ${overview}
      ${semanticsHtml}
      ${diagnosticsHtml}
      ${evidenceHtml}
      <details id="rawFrameData">
        <summary>&#9654; Raw Frame Data</summary>
        ${rawFieldsHtml}
      </details>
    `;

    const details = container.querySelector("#rawFrameData");
    container.querySelectorAll(".sourceLink").forEach((button) => {
      button.addEventListener("click", () => {
        const fieldName = button.dataset.sourceField;
        details.open = true;
        const target = container.querySelector(`[data-raw-field="${CSS.escape(fieldName)}"]`);
        if (!target) return;
        target.scrollIntoView({ block: "nearest", behavior: "smooth" });
        target.classList.add("flash");
        setTimeout(() => target.classList.remove("flash"), 900);
      });
    });

    if (pendingMetricFocus) {
      const metricRow = container.querySelector(`[data-metric="${CSS.escape(pendingMetricFocus)}"]`);
      if (metricRow) {
        metricRow.scrollIntoView({ block: "center", behavior: "smooth" });
        metricRow.classList.add("flash");
        setTimeout(() => metricRow.classList.remove("flash"), 900);
      }
      pendingMetricFocus = null;
    }
  }

  viewerState.subscribe(({ episodeIndex, frameIndex }) => render(episodeIndex, frameIndex));

  return {
    // Called once per episode load (mirrors videoPanel.setTimeline) so Diagnostics
    // can be computed locally without a network round trip per frame.
    setMetrics(newMetrics) {
      metrics = newMetrics;
    },
    // Called once per episode load (mirrors setMetrics) with this episode's
    // motion events (possibly empty if /api/analysis isn't computed yet) and
    // the dataset-wide `global` distribution summaries used as the Evidence
    // group's comparison baseline.
    setMotionEvents(newMotionEventsForEpisode, newGlobalStats) {
      motionEventsForEpisode = newMotionEventsForEpisode;
      globalStats = newGlobalStats;
    },
    // Called from an incoming "Open in Viewer" deep link that names a specific
    // metric (e.g. from the Dataset Analysis Motion Events table) -- scrolls to
    // and flashes that metric's Evidence row on the next render() only.
    focusMetric(metric) {
      pendingMetricFocus = metric;
    },
    // Deep-links directly to this episode/frame -- e.g. for the Dataset Analysis
    // dashboard to open a specific frame's context here.
    evidenceLink(episodeIndex, frameIndex) {
      return evidence.buildLink("/", { episodeIndex, frameIndex });
    },
  };
}
