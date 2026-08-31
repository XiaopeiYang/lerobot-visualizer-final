// Structured port of docs/analysis-report.md §6 "ML implications" -- each
// item traces a specific finding (with its evidence tier) to a concrete
// recommendation for training/eval/collection/governance, plus which tab of
// this page has the supporting evidence. Kept as data, separate from
// analysisPanel.js's rendering, so the findings can be reviewed/edited without
// touching render logic.
//
// Only `finding` carries an evidence tier -- a Recommendation is a
// prescriptive action, not a claim about the data, so tagging it
// fact/inference/hypothesis would be a category error.
//
// `finding`/`recommendation` may be a plain string or a `(report) => string`
// function. Use a function whenever the text states a specific number that's
// also computed live elsewhere on this page (lag residuals, motion-event flag
// rates, correlation, FPS) -- interpolating from the same `report` object
// GET /api/analysis returns keeps this page's one written narrative in sync
// with its own numbers instead of drifting if summary.json is regenerated.
//
// Kept to 4 cards, one line each for finding/recommendation: the supporting
// tabs already show the full evidence, so a card here should point at it, not
// re-argue it. A 5th former card (timestamp/FPS handling) was folded into a
// one-line recommendation on the Relationships tab instead of a standalone
// card, since it wasn't a novel finding beyond what that tab's sync table
// already shows.

import { groupMotionEventsByMetric } from "./evidence.js";

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

export const FINDINGS = [
  {
    title: "Training",
    findingTier: "fact",
    finding: (report) => {
      const lag = report.global.lag_residuals;
      return `action[i] matches track[i+1] almost exactly (median residual ${fmt(lag["1"].median, 4)} m) and diverges at every other tested offset.`;
    },
    recommendation:
      "Preserve this one-frame offset in any data loader or training pipeline -- an off-by-one would silently change the supervision target from \"predict the next frame\" to something else.",
    sourceTab: "relationships",
  },
  {
    title: "Evaluation",
    findingTier: "fact",
    finding: (report) => {
      const groups = groupMotionEventsByMetric(report.motion_events);
      const rate = (metric) => {
        const count = groups.get(metric)?.length ?? 0;
        const total = report.global[metric]?.count ?? 0;
        return total ? fmt((count / total) * 100, 1) : "0.0";
      };
      const suspiciousCount = report.motion_events.filter((e) => e.tier === "suspicious").length;
      return `${report.motion_events.length.toLocaleString()} frames cross a statistical motion threshold (~${rate("hand_speed")}% of hand-speed points); only ${suspiciousCount.toLocaleString()} are corroborated by a second signal, and ${report.verified_issues.length} are verified hard-invariant issues.`;
    },
    recommendation:
      "Treat high-motion regions as behavioral variation worth inspecting, not automatic defects to discard -- raising the evidence bar (motion → suspicious → verified) is the right lever, not raising the z-score cutoff.",
    sourceTab: "statistics",
  },
  {
    title: "Feature selection",
    findingTier: "fact",
    finding: (report) => {
      const fieldQuality = report.observation_field_quality;
      const betasNote = fieldQuality?.mano_betas_all_zero
        ? "MANO betas are uninformative (all-zero every frame)"
        : "MANO betas carry some non-zero frames";
      return `${betasNote}; action.*_hand_tracks is redundant with the next-frame track; quaternion norm is constant at 1.0, so only orientation carries information.`;
    },
    recommendation:
      "Don't spend model capacity or normalization effort on MANO betas or quaternion norm as if they varied, and don't treat action targets as adding information beyond a +1 frame shift of the observation.",
    sourceTab: "quality",
  },
  {
    title: "Dataset governance",
    findingTier: "fact",
    finding: (report) =>
      report.episode_count === 1
        ? "Only one episode is available locally -- every cross-episode statistic (task balance, outlier ranking, schema drift) is implemented but structurally vacant at N=1."
        : `${report.episode_count} episodes are available locally; cross-episode statistics are now populated -- see Data Quality.`,
    recommendation:
      "Don't read \"0 schema differences\" as dataset-wide consistency while N=1. If more episodes are collected, re-run scripts/analyze_dataset.py unchanged -- the per-episode checks become meaningful immediately.",
    sourceTab: "quality",
  },
];
