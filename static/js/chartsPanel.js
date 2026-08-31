// Temporal charts: 4 primary panels (hand centroid speed, hand pose articulation
// rate, hand span, camera speed), each overlaying left/right as two series on one
// chart so the whole episode fits in a small number of meaningful panels rather
// than one line per raw dimension. Centroid speed and articulation rate are two
// separate charts (crowded together on one dual-axis chart, they were hard to
// read) but are grouped in the DOM under one "Hand Motion" heading, since both
// describe hand motion -- see index.html's .chartGroup. Fetched
// once per episode (not per frame) via /api/episodes/<id>/metrics. A small QA/
// diagnostics readout (action-to-next-track residual, timestamp interval/gap counts)
// is rendered as text near the charts rather than as full chart panels, since Phase 2
// already verified those are expected-flat -- see docs/phase-4-visualizer.md.

import * as api from "./api.js";

const LEFT_COLOR = "#4da3ff";
const RIGHT_COLOR = "#ff9d4d";

const frameCursorPlugin = {
  id: "frameCursor",
  afterDraw(chart) {
    const t = chart.$currentTimestamp;
    if (t === null || t === undefined || !chart.scales.x) return;
    const xScale = chart.scales.x;
    const x = xScale.getPixelForValue(t);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = "#e8e8e8";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function toXY(timestamps, values) {
  return timestamps.map((t, i) => ({ x: t, y: values[i] }));
}

function makeLineChart(canvas, title, series, { viewerState, frameIndices, timestamps }) {
  const scales = {
    x: { type: "linear", title: { display: true, text: "timestamp (s)" } },
    y: { title: { display: true, text: title } },
  };
  // eslint-disable-next-line no-undef -- Chart is the vendored Chart.js UMD global
  const chart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      datasets: series.map(({ label, color, values }) => ({
        label,
        data: toXY(timestamps, values),
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        spanGaps: false,
        yAxisID: "y",
      })),
    },
    options: {
      animation: false,
      parsing: false,
      normalized: true,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      scales,
      plugins: {
        title: { display: true, text: title },
        legend: { display: series.length > 1, labels: { boxWidth: 12 } },
      },
    },
    plugins: [frameCursorPlugin],
  });

  canvas.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect();
    const xPixel = event.clientX - rect.left;
    const clickedTimestamp = chart.scales.x.getValueForPixel(xPixel);
    let nearestIndex = 0;
    let nearestDistance = Infinity;
    for (let i = 0; i < timestamps.length; i++) {
      const distance = Math.abs(timestamps[i] - clickedTimestamp);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = i;
      }
    }
    viewerState.setFrame({
      frameIndex: frameIndices[nearestIndex],
      timestamp: timestamps[nearestIndex],
      source: "chart",
    });
  });

  return chart;
}

function summarizeAbs(values) {
  const finite = values.filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  if (finite.length === 0) return { max: null, mean: null, count: 0 };
  const abs = finite.map(Math.abs);
  const max = Math.max(...abs);
  const mean = abs.reduce((a, b) => a + b, 0) / abs.length;
  return { max, mean, count: finite.length };
}

function renderQaReadout(container, metrics) {
  const leftResidual = summarizeAbs(metrics.left_action_residual);
  const rightResidual = summarizeAbs(metrics.right_action_residual);
  const diagnostics = metrics.sync_diagnostics;
  container.innerHTML = `
    <div><strong>Sync diagnostics</strong> (computed once at episode load):
      ${diagnostics.frame_count} frames,
      median interval ${(diagnostics.median_interval_seconds * 1000).toFixed(3)} ms,
      ${diagnostics.large_interval_count} large gap(s),
      ${diagnostics.duplicate_timestamp_count} duplicate timestamp(s)</div>
    <div><strong>Action-to-next-track residual</strong> (expected ~0; Phase 2 verified
      action[i] == track[i+1] exactly): left max ${leftResidual.max?.toExponential(2) ?? "n/a"},
      right max ${rightResidual.max?.toExponential(2) ?? "n/a"}</div>
  `;
}

export function createChartsPanel({ canvases, qaContainer, viewerState }) {
  let frameIndices = [];
  let timestamps = [];
  let charts = [];

  async function loadEpisode(episodeIndex) {
    const metrics = await api.getMetrics(episodeIndex);
    frameIndices = metrics.frame_indices;
    timestamps = metrics.timestamps;

    for (const chart of charts) chart.destroy();
    charts = [
      // Centroid speed and articulation rate are two separate charts (each
      // easier to read alone than overlaid) but stay grouped in the DOM under
      // one "Hand Motion" heading (see index.html's .chartGroup) -- same
      // semantic group, not a 4th unrelated chart.
      makeLineChart(
        canvases.handCentroidSpeed,
        "Hand Centroid Speed (m/s)",
        [
          { label: "left speed", color: LEFT_COLOR, values: metrics.left_hand_speed },
          { label: "right speed", color: RIGHT_COLOR, values: metrics.right_hand_speed },
        ],
        { viewerState, frameIndices, timestamps }
      ),
      makeLineChart(
        canvases.handArticulation,
        "Hand Pose Articulation Rate (rad/s)",
        [
          { label: "left articulation", color: LEFT_COLOR, values: metrics.left_hand_articulation_rate },
          { label: "right articulation", color: RIGHT_COLOR, values: metrics.right_hand_articulation_rate },
        ],
        { viewerState, frameIndices, timestamps }
      ),
      makeLineChart(
        canvases.handSpan,
        "Hand span (m)",
        [
          { label: "left", color: LEFT_COLOR, values: metrics.left_hand_span },
          { label: "right", color: RIGHT_COLOR, values: metrics.right_hand_span },
        ],
        { viewerState, frameIndices, timestamps }
      ),
      makeLineChart(
        canvases.cameraSpeed,
        "Camera speed (m/s)",
        [{ label: "camera", color: "#9d7bff", values: metrics.camera_speed }],
        { viewerState, frameIndices, timestamps }
      ),
    ];

    renderQaReadout(qaContainer, metrics);
    updateCursor(viewerState.get().timestamp);
    return metrics;
  }

  function updateCursor(timestamp) {
    for (const chart of charts) {
      chart.$currentTimestamp = timestamp;
      chart.update("none");
    }
  }

  viewerState.subscribe(({ timestamp }) => updateCursor(timestamp));

  return { loadEpisode };
}
