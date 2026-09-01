// Dataset Analysis page: renders the artifact produced by
// `scripts/analyze_dataset.py` (served verbatim by GET /api/analysis). Read-only,
// dataset-wide -- distinct from index.html's per-frame viewer. Episode rows and
// motion-event rows link into the viewer via evidence.js's shared episode/frame/
// metric URL contract, and the viewer's Frame Inspector Evidence group links
// back here for a flagged frame -- see evidence.js for the full bidirectional-
// link design.
//
// Tab layout mirrors the deliverable's own rubric (structure/scale/semantics,
// statistics + motion regions, cross-modal relationships, quality/completeness/
// consistency/sync, and training/eval/collection/governance findings).
// High-motion regions live inside Statistics rather than their own tab: a
// statistically extreme hand/camera/articulation reading is evidence of
// unusual motion, not by itself evidence of a data defect -- see analysis.py's
// module docstring. Only hard invariant violations (report.verified_issues)
// belong in the Data Quality tab.
//
// Every claim beyond a raw measurement is tagged with evidenceTag.js's
// fact/metadata/inference/hypothesis/unknown scheme. The legend renders once
// (top of Overview); elsewhere only non-"fact" tiers get an inline badge --
// badging every already-obvious verified fact just becomes visual noise.

import * as api from "./api.js";
import * as evidence from "./evidence.js";
import { initTabs } from "./tabs.js";
import { evidenceTag, evidenceLegend } from "./evidenceTag.js";
import { SEMANTIC_FIELD_TABLE, EXTERNAL_CONVENTIONS, SCHEMA_DISCREPANCY_NOTE } from "./semanticFieldTable.js";
import { FINDINGS } from "./findingsData.js";

const METRIC_LABELS = {
  ...evidence.METRIC_LABELS,
  articulation_rate: "Hand pose articulation rate (rad/s)",
};

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

function fmtUs(seconds) {
  return seconds === null || seconds === undefined ? "n/a" : `${(seconds * 1e6).toFixed(2)} µs`;
}

function evidenceLink(episodeIndex, frameIndex, metric) {
  return evidence.buildLink("/", { episodeIndex, frameIndex, metric });
}

function renderNotComputed(container, message) {
  container.innerHTML = `
    <div class="panel notComputed">
      <h2>Dataset Analysis</h2>
      <p>${message}</p>
      <p>Run the analysis script first, then reload this page:</p>
      <p><code>python scripts/analyze_dataset.py --dataset-root data/raw</code></p>
    </div>
  `;
}

// Merges the former separate "Dataset at a glance" blurb, "Dataset Overview"
// card row, and per-episode FPS cross-check table into one compact block: a
// stat card row plus a handful of takeaway bullets computed live from
// `report` (mirrors findingsData.js's (report) => string pattern so the
// bullets can't drift from summary.json). The FPS cross-check's own
// per-episode table is dropped -- at N=1 it's one row, and the "FPS agrees"
// conclusion is what mattered, not the row itself.
function renderOverviewSummary(report, datasetMeta) {
  const totalFrames = report.episodes.reduce((sum, e) => sum + e.overview.frame_count, 0);
  const totalDuration = report.episodes.reduce((sum, e) => sum + e.overview.duration_seconds, 0);
  const lag = report.global.lag_residuals;
  const firstOverview = report.episodes[0]?.overview;
  const maxSyncError = Math.max(
    ...report.episodes.map((e) => e.sync_validation.summary.max_abs_sync_error_seconds ?? 0)
  );
  const verifiedCount = report.verified_issues.length;

  const cards = [
    { value: report.episode_count, label: "Episodes" },
    { value: totalFrames.toLocaleString(), label: "Frames" },
    { value: `${(totalDuration / 60).toFixed(1)} min`, label: "Duration" },
    { value: fmt(datasetMeta?.fps ?? firstOverview?.declared_fps, 2), label: "FPS" },
  ];
  if (datasetMeta?.robot_type) cards.push({ value: datasetMeta.robot_type, label: "Robot type" });
  const cardsHtml = cards
    .map((c) => `<div class="card"><div class="cardValue">${c.value}</div><div class="cardLabel">${c.label}</div></div>`)
    .join("");

  return `
    <h2>Dataset overview</h2>
    <div class="cardRow">${cardsHtml}</div>
    ${evidenceLegend()}
    <ul>
      <li>Action targets are an exact next-frame hand-track target (lag k=+1 residual median
        ${fmt(lag["1"].median, 4)} m, smallest of any tested offset).</li>
      <li>Video and dataset timestamps agree within ${(maxSyncError * 1000).toFixed(3)} ms.</li>
      <li>${
        verifiedCount === 0
          ? "No hard data-integrity violations were found."
          : `${verifiedCount} hard-invariant data-quality issue${verifiedCount === 1 ? "" : "s"} found -- see Data Quality.`
      }</li>
      <li>${
        report.limitations?.note ??
        `Analysis scope is limited to ${report.episode_count} episode${report.episode_count === 1 ? "" : "s"}; cross-episode conclusions are not yet supported.`
      }</li>
    </ul>
  `;
}

function renderSemanticFieldTable() {
  const rows = SEMANTIC_FIELD_TABLE.map(
    (row) => `
      <tr>
        <td><code>${row.field}</code></td>
        <td>${row.semanticMeaning}</td>
        <td class="numeric">${row.shape}</td>
        <td>${row.frame}</td>
      </tr>
    `
  ).join("");
  const conventionRows = EXTERNAL_CONVENTIONS.map(
    (item) => `
      <div class="conventionCard">
        <strong>${item.title}</strong> ${evidenceTag(item.tier)}
        <p>${item.text}</p>
      </div>
    `
  ).join("");
  return `
    <h3>Key fields</h3>
    <table class="dataTable">
      <thead><tr><th>Field</th><th>Semantic meaning</th><th class="numeric">Shape</th><th>Frame</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="sectionNote">
      Shapes and declared semantics were checked against stored values; important exceptions
      and assumptions are listed below. ${SCHEMA_DISCREPANCY_NOTE}
    </p>

    <h3>Assumptions &amp; unknowns</h3>
    <div class="conventionsGrid">${conventionRows}</div>
  `;
}

function renderOverviewCards(report, datasetMeta) {
  return `
    <section class="panel" data-tab-panel="overview">
      ${renderOverviewSummary(report, datasetMeta)}
      ${renderSemanticFieldTable()}
    </section>
  `;
}

function summaryRow(label, summary) {
  if (!summary) return "";
  return `
    <tr>
      <td>${label}</td>
      <td class="numeric">${summary.count}</td>
      <td class="numeric">${fmt(summary.median, 4)}</td>
      <td class="numeric">${fmt(summary.p95, 4)}</td>
      <td class="numeric">${fmt(summary.p99, 4)}</td>
      <td class="numeric">${fmt(summary.max, 4)}</td>
    </tr>
  `;
}

function statTermsNote() {
  return `
    <p class="sectionNote">
      How to read — Median is the 50th percentile; P95/P99 mean that 95%/99%
      of valid samples are at or below that value. Max is shown only as an
      extreme reference. Count is the number of valid samples for that metric;
      bilateral hand metrics pool left and right hands. Percentiles describe
      the distribution and are not data-quality or anomaly thresholds.
    </p>
  `;
}

function renderGlobalStatsTable(report) {
  const g = report.global;
  const rows = [
    summaryRow(METRIC_LABELS.hand_speed, g.hand_speed),
    summaryRow(METRIC_LABELS.hand_span, g.hand_span),
    summaryRow(METRIC_LABELS.camera_speed, g.camera_speed),
    summaryRow(METRIC_LABELS.quaternion_norm, g.quaternion_norm),
    summaryRow(METRIC_LABELS.action_residual_k1, g.action_residual_k1),
    summaryRow(METRIC_LABELS.articulation_rate, g.articulation_rate),
  ].join("");
  return `
    <h3>Statistical profile (pooled across all episodes)</h3>
    <table class="dataTable">
      <thead><tr><th>Metric</th><th class="numeric">Count</th><th class="numeric">Median</th><th class="numeric">P95</th><th class="numeric">P99</th><th class="numeric">Max</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${statTermsNote()}
    <p class="limitationNote">
      Hand span and camera quaternion norm have no histogram below -- both are tightly or
      near-degenerately distributed.
    </p>
  `;
}

function makeHistogramChart(canvas, title, histogram) {
  if (!histogram || !histogram.bin_edges.length) return null;
  const labels = histogram.counts.map((_, i) => ((histogram.bin_edges[i] + histogram.bin_edges[i + 1]) / 2).toFixed(3));
  // eslint-disable-next-line no-undef -- Chart is the vendored Chart.js UMD global
  return new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: { labels, datasets: [{ label: title, data: histogram.counts, backgroundColor: "#4da3ff" }] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { title: { display: true, text: title }, legend: { display: false } },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: "frame count" } } },
    },
  });
}

// Per-episode left/right hand_speed summaries have been computed by
// analysis.py's run_full_analysis all along -- collapsed to one sentence
// instead of a full table, since no downstream finding depends on the
// per-episode breakdown at N=1.
function handRoleSummaryLine(report) {
  const first = report.episodes[0];
  if (!first) return "";
  const leftSpeed = first.hand_speed.left.median;
  const rightSpeed = first.hand_speed.right.median;
  if (leftSpeed === null || rightSpeed === null || leftSpeed === rightSpeed) return "";
  const higher = leftSpeed > rightSpeed ? "left" : "right";
  const ratio = Math.max(leftSpeed, rightSpeed) / Math.min(leftSpeed, rightSpeed);
  return `
    <p class="sectionNote">
      ${evidenceTag("inference", "Inference")}: ${higher}-hand median speed is ${ratio.toFixed(2)}&times;
      the other hand's in episode ${first.overview.episode_index}; no role interpretation is assigned.
    </p>
  `;
}

// Folded in from the former standalone Motion Events tab (see the module
// header comment): a statistically extreme reading is evidence of unusual
// motion, not by itself evidence of a defect, so it belongs alongside the
// distributions that explain it rather than as a headline tab of its own.
// Only metrics that actually produce flagged points are shown -- hand_span,
// quaternion_norm, and action_residual_k1 are tightly/near-degenerately
// distributed and never cross the threshold.
function highMotionRegionsTable(report) {
  const groups = evidence.groupMotionEventsByMetric(report.motion_events);
  const totalPooled = {
    hand_speed: report.global.hand_speed.count,
    camera_speed: report.global.camera_speed.count,
    articulation_rate: report.global.articulation_rate.count,
  };
  const rows = Object.keys(totalPooled)
    .map((metric) => {
      const events = groups.get(metric) ?? [];
      if (!events.length) return "";
      const total = totalPooled[metric] ?? 0;
      const pct = total ? ((events.length / total) * 100).toFixed(1) : "0.0";
      return `
        <tr>
          <td>${METRIC_LABELS[metric]}</td>
          <td class="numeric">${events.length.toLocaleString()}</td>
          <td class="numeric">${pct}%</td>
        </tr>
      `;
    })
    .join("");
  if (!rows) return "";
  return `
    <h3>High-motion regions</h3>
    <table class="dataTable">
      <thead><tr><th>Signal</th><th class="numeric">Frames flagged</th><th class="numeric">Rate</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="limitationNote">
      Statistical high-motion flags identify rapid-motion regions for inspection; they are
      <strong>not</strong> data-quality errors.
    </p>
  `;
}

function renderDistributions(report) {
  return `
    <section class="panel" data-tab-panel="statistics">
      <h2>Statistics &amp; Distributions</h2>
      ${renderGlobalStatsTable(report)}
      ${handRoleSummaryLine(report)}
      <h3>Motion speed</h3>
      <div class="histogramGrid">
        <div class="chartWrapper"><canvas id="histHandSpeed"></canvas></div>
        <div class="chartWrapper"><canvas id="histCameraSpeed"></canvas></div>
      </div>
      <p class="limitationNote">
        Hand centroid speed measures whole-hand translation from the mean position
        of the 21 landmarks; it does not measure finger articulation.
      </p>
      <p class="limitationNote">
        ${evidenceTag("inference", "Inference")}: hand/camera speed are strongly right-skewed,
        consistent with a task alternating low-motion and high-motion phases.
      </p>
      <h3>Articulation</h3>
      <div class="histogramGrid">
        <div class="chartWrapper"><canvas id="histArticulationRate"></canvas></div>
      </div>
      <p class="limitationNote">
        Articulation rate measures change in the MANO pose vector over time, not
        semantic actions such as grasping.
      </p>
      ${highMotionRegionsTable(report)}
    </section>
  `;
}

function renderRelationships(report) {
  const lagRows = Object.entries(report.global.lag_residuals)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(
      ([lag, summary]) =>
        `<tr${Number(lag) === 1 ? ' class="highlightRow"' : ""}><td>${lag > 0 ? "+" + lag : lag}</td><td class="numeric">${fmt(summary.median, 6)}</td><td class="numeric">${fmt(summary.p95, 6)}</td><td class="numeric">${fmt(summary.max, 6)}</td></tr>`
    )
    .join("");

  const syncRows = report.episodes
    .map(
      (e) => `
        <tr>
          <td>${e.overview.episode_index}</td>
          <td class="numeric">${fmtUs(e.sync_validation.summary.max_abs_sync_error_seconds)}</td>
          <td class="numeric">${fmtUs(e.sync_validation.summary.mean_abs_sync_error_seconds)}</td>
          <td class="numeric">${e.sync_validation.summary.resolved_count}/${e.sync_validation.summary.target_count}</td>
        </tr>
      `
    )
    .join("");
  const maxSyncError = Math.max(
    ...report.episodes.map((e) => e.sync_validation.summary.max_abs_sync_error_seconds ?? 0)
  );

  const correlationRows = report.episodes
    .map(
      (e) => `
        <tr>
          <td>${e.overview.episode_index}</td>
          <td class="numeric">${fmt(e.camera_hand_correlation.left_hand, 3)}</td>
          <td class="numeric">${fmt(e.camera_hand_correlation.right_hand, 3)}</td>
        </tr>
      `
    )
    .join("");
  const correlationValues = report.episodes
    .flatMap((e) => [e.camera_hand_correlation.left_hand, e.camera_hand_correlation.right_hand])
    .filter((v) => v !== null && v !== undefined);
  const rLo = correlationValues.length ? fmt(Math.min(...correlationValues), 2) : "n/a";
  const rHi = correlationValues.length ? fmt(Math.max(...correlationValues), 2) : "n/a";

  return `
    <section class="panel" data-tab-panel="relationships">
      <h2>Cross-Modal Relationships</h2>

      <h3>Action &harr; hand track, multi-lag residual</h3>
      <p class="sectionNote">
        residual(action<sub>i</sub>, track<sub>i+k</sub>) across k=-2..+2, both hands pooled.
      </p>
      <table class="dataTable">
        <thead><tr><th>Lag k</th><th class="numeric">Median residual (m)</th><th class="numeric">P95</th><th class="numeric">Max</th></tr></thead>
        <tbody>${lagRows}</tbody>
      </table>
      <p class="limitationNote">
        A near-zero median only at k=+1 is evidence (not an assumption) that
        <code>action[i]</code> encodes the next-frame hand track --
        ${evidenceTag("inference", "well-supported inference")}.
      </p>

      <h3>Video &harr; dataset timestamp synchronization</h3>
      <table class="dataTable">
        <thead><tr><th>Episode</th><th class="numeric">Max abs sync error</th><th class="numeric">Mean abs sync error</th><th class="numeric">Samples resolved</th></tr></thead>
        <tbody>${syncRows}</tbody>
      </table>
      <p class="limitationNote">
        Sampled (15 frames per episode, not exhaustive); maximum timestamp-to-video error
        ${fmtUs(maxSyncError)}. Training pipelines should use the stored per-frame timestamp,
        not a fixed FPS assumption, to reconstruct frame times.
      </p>

      <h3>Camera motion &harr; hand motion co-variation</h3>
      <table class="dataTable">
        <thead><tr><th>Episode</th><th class="numeric">Pearson r (left hand)</th><th class="numeric">Pearson r (right hand)</th></tr></thead>
        <tbody>${correlationRows}</tbody>
      </table>
      <p class="limitationNote">
        ${evidenceTag("inference", "Inference")}: moderate correlation (r&asymp;${rLo}&ndash;${rHi});
        association does not establish causality -- both are plausibly driven by a third factor
        (e.g. the wearer's overall body motion during an egocentric recording).
      </p>
    </section>
  `;
}

function verifiedIssueRows(issues) {
  return issues
    .map(
      (issue) => `
        <tr>
          <td>${issue.kind}</td>
          <td class="numeric">${issue.episode_index}</td>
          <td>${issue.frame_index ?? "n/a (episode-level)"}</td>
          <td>${issue.detail}</td>
          <td>${issue.frame_index !== null ? `<a class="evidenceLink" href="${evidenceLink(issue.episode_index, issue.frame_index)}">Open in Viewer</a>` : ""}</td>
        </tr>
      `
    )
    .join("");
}

function renderVerifiedIssues(report) {
  const issues = report.verified_issues;
  const summarySentence = issues.length
    ? `<p class="sectionNote">${issues.length} hard invariant violation${issues.length === 1 ? "" : "s"} found (physically or format-impossible, not a statistical judgment call).</p>`
    : `<p class="limitationNote">No hard data-integrity violations found across NaN/Inf, timestamps, duplicates, quaternion validity, terminal sentinel, and sampled video synchronization checks.</p>`;
  const table = issues.length
    ? `<table class="dataTable">
        <thead><tr><th>Kind</th><th class="numeric">Episode</th><th>Frame</th><th>Detail</th><th></th></tr></thead>
        <tbody>${verifiedIssueRows(issues)}</tbody>
      </table>`
    : "";
  return `
    <h3>Verified issues</h3>
    ${summarySentence}
    ${table}
    <p class="limitationNote">
      High-motion regions (Statistics tab) are treated separately as descriptive motion events,
      not data-quality defects.
    </p>
  `;
}

// Synthesizes facts already established elsewhere on this page (MANO betas
// completeness, action-track lag redundancy, quaternion-norm degeneracy)
// into one "does this channel carry independent signal" view -- no new
// computation, just consolidation.
function renderFeatureInformativeness(report) {
  const fieldQuality = report.observation_field_quality;
  const lag = report.global.lag_residuals;
  const otherMedians = ["-2", "-1", "0", "2"].map((k) => lag[k].median);
  const items = [];
  if (fieldQuality) {
    items.push(`
      <li><strong>Uninformative:</strong> MANO betas are
        ${fieldQuality.mano_betas_all_zero ? "zero on all frames" : `non-zero on ${fieldQuality.mano_betas_nonzero_frame_count} frame(s)`}.</li>
    `);
  }
  items.push(`
    <li><strong>Redundant:</strong> action hand tracks duplicate the next-frame observation
      target (lag k=+1 residual median ${fmt(lag["1"].median, 6)} m vs.
      ${fmt(Math.min(...otherMedians), 4)}&ndash;${fmt(Math.max(...otherMedians), 4)} m at every
      other tested offset).</li>
  `);
  items.push(`
    <li><strong>Low-information magnitude:</strong> quaternion norm is always 1; orientation
      components, not magnitude, contain information.</li>
  `);
  return `
    <h3>Feature informativeness / redundancy</h3>
    <ul>${items.join("")}</ul>
  `;
}

function renderQualitySummary(report) {
  const largeGaps = report.episodes.reduce((sum, e) => sum + e.sync_diagnostics.large_interval_count, 0);
  const duplicates = report.episodes.reduce((sum, e) => sum + e.sync_diagnostics.duplicate_timestamp_count, 0);
  const schemaDiffs = report.schema_audit.differences;
  const verifiedIssueCount = report.verified_issues.length;
  const maxSyncError = Math.max(
    ...report.episodes.map((e) => e.sync_validation.summary.max_abs_sync_error_seconds ?? 0)
  );
  const fieldQuality = report.observation_field_quality;

  const badge = (ok, label) => `<span class="badge ${ok ? "ok" : "warn"}">${ok ? "✓" : "⚠"} ${label}</span>`;

  return `
    <section class="panel" data-tab-panel="quality">
      <h2>Data Quality: Completeness, Consistency, Sync, Verified Issues</h2>
      <div class="badgeRow">
        ${badge(schemaDiffs.length === 0, `schema consistency (${schemaDiffs.length} diff${schemaDiffs.length === 1 ? "" : "s"})`)}
        ${badge(largeGaps === 0, `${largeGaps} large timestamp gap${largeGaps === 1 ? "" : "s"}`)}
        ${badge(duplicates === 0, `${duplicates} duplicate timestamp${duplicates === 1 ? "" : "s"}`)}
        ${badge(maxSyncError < 0.01, `max video/timestamp sync error ${(maxSyncError * 1000).toFixed(3)} ms`)}
        ${badge(verifiedIssueCount === 0, `${verifiedIssueCount} verified data-quality issue${verifiedIssueCount === 1 ? "" : "s"}`)}
        ${fieldQuality ? badge(fieldQuality.keypoint_quaternion_invalid_frame_count === 0, `keypoint orientation validity (${fieldQuality.keypoint_quaternion_invalid_frame_count} invalid frame${fieldQuality.keypoint_quaternion_invalid_frame_count === 1 ? "" : "s"})`) : ""}
        ${fieldQuality ? `<span class="badge ${fieldQuality.mano_betas_all_zero ? "warn" : "ok"}">${fieldQuality.mano_betas_all_zero ? "⚠" : "✓"} MANO shape (betas) ${fieldQuality.mano_betas_all_zero ? "unpopulated (all-zero)" : "populated"}</span>` : ""}
      </div>

      <h3>MANO &amp; keypoint/camera-orientation field quality</h3>
      ${
        fieldQuality
          ? `<table class="dataTable">
              <thead><tr><th>Check</th><th class="numeric">Result</th></tr></thead>
              <tbody>
                <tr><td>Keypoint quaternion invalid frames</td><td class="numeric">${fieldQuality.keypoint_quaternion_invalid_frame_count}</td></tr>
                <tr><td>Camera quaternion invalid frames</td><td class="numeric">${fieldQuality.camera_quaternion_invalid_frame_count}</td></tr>
                <tr><td>MANO beta non-zero frames</td><td class="numeric">${fieldQuality.mano_betas_nonzero_frame_count}</td></tr>
              </tbody>
            </table>
            <p class="limitationNote">
              MANO betas are unpopulated, so shape/person-specific information is unavailable.
            </p>`
          : `<p class="limitationNote">Not available in this artifact.</p>`
      }

      ${renderFeatureInformativeness(report)}
      ${renderVerifiedIssues(report)}
    </section>
  `;
}

function resolveFindingText(value, report) {
  return typeof value === "function" ? value(report) : value;
}

function renderFindings(report) {
  const cards = FINDINGS.map(
    (item) => `
      <article class="findingCard">
        <h3>${item.title}</h3>
        <p><strong>Finding</strong> ${item.findingTier !== "fact" ? evidenceTag(item.findingTier) : ""}<br>${resolveFindingText(item.finding, report)}</p>
        <p><strong>Recommendation</strong><br>${resolveFindingText(item.recommendation, report)}</p>
        <button type="button" class="findingSourceLink" data-tab-target="${item.sourceTab}">See supporting evidence &rarr;</button>
      </article>
    `
  ).join("");
  return `
    <section class="panel" data-tab-panel="findings">
      <h2>Findings &amp; Recommendations</h2>
      <div class="findingsGrid">${cards}</div>
    </section>
  `;
}

async function init() {
  const container = document.getElementById("analysisContent");
  let report;
  try {
    report = await api.getAnalysis();
  } catch (error) {
    renderNotComputed(container, error.message);
    return;
  }

  // Completes the Viewer -> Analysis direction of the bidirectional deep link
  // (Analysis -> Viewer already exists via evidenceLink()/"Open in Viewer"): a
  // frame flagged in the Frame Inspector's Evidence panel links back here with
  // `?episode=&frame=&metric=`, resolved purely through evidence.js's state
  // contract, independent of this page's own table layout.
  const highlight = evidence.parseParams();

  // Best-effort, mirroring the report fetch above: dataset-root metadata is a
  // nice-to-have overview card, not something that should block the rest of the
  // page from rendering if /api/dataset is unreachable.
  let datasetMeta = null;
  try {
    datasetMeta = await api.getDatasetMetadata();
  } catch {
    datasetMeta = null;
  }

  container.innerHTML = [
    renderOverviewCards(report, datasetMeta),
    renderDistributions(report),
    renderRelationships(report),
    renderQualitySummary(report),
    renderFindings(report),
  ].join("");

  const tabsNav = document.getElementById("analysisTabs");
  tabsNav.hidden = false;
  const tabs = initTabs({
    nav: tabsNav,
    getPanel: (target) => container.querySelector(`[data-tab-panel="${target}"]`),
  });
  // Findings cards link back to the tab that supports them; same activation
  // path as the nav buttons, just triggered from inside a panel.
  container.querySelectorAll("[data-tab-target].findingSourceLink").forEach((button) => {
    button.addEventListener("click", () => tabs.activate(button.dataset.tabTarget));
  });
  // initTabs() only wires up click handling -- it never hides panels itself,
  // so without an explicit first activate() call here every panel would
  // render visible at once until the user's first click. A deep link naming
  // a specific frame/metric (e.g. the Viewer's "view in analysis" Evidence
  // link) always points at the Statistics tab, since that's where high-motion
  // regions now live -- resolved via evidence.js's contract, independent of
  // tab order/labels above; otherwise default to Overview, matching the
  // "isActive" button already set in analysis.html.
  if (highlight.episodeIndex !== undefined && highlight.frameIndex !== undefined && highlight.metric) {
    tabs.activate("statistics");
  } else {
    tabs.activate("overview");
  }

  makeHistogramChart(document.getElementById("histHandSpeed"), "Hand centroid speed (m/s)", report.histograms.hand_speed);
  makeHistogramChart(document.getElementById("histCameraSpeed"), "Camera speed (m/s)", report.histograms.camera_speed);
  makeHistogramChart(
    document.getElementById("histArticulationRate"),
    "Hand pose articulation rate (rad/s)",
    report.histograms.articulation_rate
  );
}

init();
