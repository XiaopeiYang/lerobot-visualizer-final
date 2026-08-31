// Shared evidence-tier badge, matching the 5-tier scheme used throughout
// docs/analysis-report.md (VERIFIED FACT / METADATA DECLARATION / INFERENCE /
// HYPOTHESIS-NEEDS-VERIFICATION / UNKNOWN). The live analysis page reuses this
// single source of truth instead of the old one-size-fits-all `.limitationNote`
// paragraph, so a reader can tell at a glance whether a number on screen is a
// directly-measured fact or an interpretation that still needs checking.

const TIERS = {
  fact: { icon: "✓", label: "Verified fact", cssClass: "tierFact" },
  metadata: { icon: "Ⓜ", label: "Metadata declaration", cssClass: "tierMetadata" },
  inference: { icon: "\u{1F4A1}", label: "Inference", cssClass: "tierInference" },
  hypothesis: { icon: "?", label: "Hypothesis — needs verification", cssClass: "tierHypothesis" },
  unknown: { icon: "❔", label: "Unknown", cssClass: "tierUnknown" },
};

// `tier` is one of TIERS' keys; `text` overrides the default tier label (used
// when a badge should read e.g. "Inference" but the surrounding prose already
// says what it's an inference about).
export function evidenceTag(tier, text) {
  const spec = TIERS[tier] ?? TIERS.unknown;
  const label = text ?? spec.label;
  return `<span class="evidenceTag ${spec.cssClass}" title="${spec.label}"><span aria-hidden="true">${spec.icon}</span> ${label}</span>`;
}

export function evidenceLegend() {
  return `
    <div class="evidenceLegend">
      ${Object.keys(TIERS)
        .map((tier) => evidenceTag(tier))
        .join("")}
    </div>
  `;
}
