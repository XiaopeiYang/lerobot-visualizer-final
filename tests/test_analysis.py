"""Unit tests for Phase 5/6 dataset-wide analysis helpers, against synthetic data.

Mirrors tests/test_metrics.py's style: hand-computed values on small synthetic
inputs, plus one bounded integration test against the real local dataset (skipped
if data/raw is absent, same pattern as tests/test_dataset.py).
"""

from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path

from lerobot_visualizer import analysis, metrics

_DATASET_ROOT = Path(os.environ.get("LEROBOT_DATASET_ROOT", "data/raw"))


class SummarizeDistributionTests(unittest.TestCase):
    def test_basic_quantiles(self) -> None:
        summary = analysis.summarize_distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary.count, 5)
        self.assertEqual(summary.finite_count, 5)
        self.assertEqual(summary.min, 1.0)
        self.assertEqual(summary.max, 5.0)
        self.assertAlmostEqual(summary.median, 3.0)
        self.assertAlmostEqual(summary.mean, 3.0)

    def test_nan_and_inf_are_counted_but_excluded_from_finite_stats(self) -> None:
        summary = analysis.summarize_distribution([1.0, 2.0, float("nan"), float("inf")])
        self.assertEqual(summary.count, 4)
        self.assertEqual(summary.finite_count, 2)
        self.assertEqual(summary.nan_count, 1)
        self.assertEqual(summary.inf_count, 1)
        self.assertEqual(summary.min, 1.0)
        self.assertEqual(summary.max, 2.0)

    def test_empty_series_returns_none_stats(self) -> None:
        summary = analysis.summarize_distribution([])
        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.median)


class ComputeHistogramTests(unittest.TestCase):
    def test_counts_sum_to_finite_value_count(self) -> None:
        histogram = analysis.compute_histogram([1.0, 2.0, 3.0, float("nan")], bin_count=5)
        self.assertEqual(sum(histogram["counts"]), 3)
        self.assertEqual(len(histogram["bin_edges"]), 6)

    def test_constant_series_collapses_to_one_bin(self) -> None:
        histogram = analysis.compute_histogram([2.0, 2.0, 2.0])
        self.assertEqual(histogram["counts"], [3])

    def test_empty_series_returns_empty_histogram(self) -> None:
        self.assertEqual(analysis.compute_histogram([]), {"bin_edges": [], "counts": []})


class QuaternionNormsTests(unittest.TestCase):
    def test_unit_quaternion_has_norm_one(self) -> None:
        rows = [[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]]
        norms = analysis.quaternion_norms(rows)
        self.assertAlmostEqual(norms[0], 1.0)
        self.assertAlmostEqual(norms[1], 1.0)

    def test_non_unit_quaternion_is_detected(self) -> None:
        rows = [[0.0, 2.0, 0.0, 0.0, 0.0]]
        self.assertAlmostEqual(analysis.quaternion_norms(rows)[0], 2.0)


def _keypoint_quaternion_row(deviations: list[float] | None = None) -> list[float]:
    """21x(x,y,z,w) row; `deviations[i]` (default 0) scales landmark i's w
    component away from 1.0, so norm deviates from 1 by that same amount."""
    deviations = deviations or [0.0] * 21
    row: list[float] = []
    for i in range(21):
        row.extend([0.0, 0.0, 0.0, 1.0 + deviations[i]])
    return row


def _mano_row(betas: list[float] | None = None) -> list[float]:
    return [0.0] * 3 + [0.0] * 45 + (betas or [0.0] * 10) + [0.0] * 3


class KeypointQuaternionMaxDeviationTests(unittest.TestCase):
    def test_all_unit_quaternions_have_zero_deviation(self) -> None:
        rows = [_keypoint_quaternion_row(), _keypoint_quaternion_row()]
        self.assertEqual(analysis.keypoint_quaternion_max_deviation(rows), (0.0, 0.0))

    def test_reports_the_largest_deviation_in_the_row(self) -> None:
        deviations = [0.0] * 21
        deviations[5] = 0.5  # landmark 5's quaternion now has norm 1.5, deviation 0.5
        rows = [_keypoint_quaternion_row(deviations)]
        self.assertAlmostEqual(analysis.keypoint_quaternion_max_deviation(rows)[0], 0.5)


class ManoBetasNonzeroFrameCountTests(unittest.TestCase):
    def test_all_zero_betas_count_as_zero_frames(self) -> None:
        rows = [_mano_row(), _mano_row(), _mano_row()]
        self.assertEqual(analysis.mano_betas_nonzero_frame_count(rows), 0)

    def test_any_nonzero_beta_counts_the_frame(self) -> None:
        rows = [_mano_row(), _mano_row([0.1] + [0.0] * 9), _mano_row()]
        self.assertEqual(analysis.mano_betas_nonzero_frame_count(rows), 1)


def _points_at(value: float) -> tuple[tuple[float, float, float], ...]:
    return tuple((value, 0.0, 0.0) for _ in range(21))


class ComputeLagResidualsTests(unittest.TestCase):
    def setUp(self) -> None:
        # track[i] sits at value i; action[i] is defined as track[i+1]'s point exactly
        # (the dataset's documented next-frame-target convention), except the last
        # frame (no target). This makes lag +1 the only lag with a zero residual.
        n = 5
        self.track_points = [_points_at(float(i)) for i in range(n)]
        self.action_points = [_points_at(float(i + 1)) if i + 1 < n else None for i in range(n)]

    def test_lag_plus_one_has_zero_residual_other_lags_do_not(self) -> None:
        result = analysis.compute_lag_residuals([(self.action_points, self.track_points)])
        self.assertAlmostEqual(result[1].mean, 0.0)
        self.assertGreater(result[-1].mean, 0.0)
        self.assertGreater(result[0].mean, 0.0)
        self.assertGreater(result[2].mean, 0.0)

    def test_pools_across_multiple_hands(self) -> None:
        result = analysis.compute_lag_residuals(
            [
                (self.action_points, self.track_points),
                (self.action_points, self.track_points),
            ]
        )
        # Pooling the same synthetic hand twice should double the sample count.
        single = analysis.compute_lag_residuals([(self.action_points, self.track_points)])
        self.assertEqual(result[1].count, single[1].count * 2)


class PearsonCorrelationTests(unittest.TestCase):
    def test_perfectly_correlated_series(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(analysis.pearson_correlation(xs, ys), 1.0)

    def test_no_variance_returns_none(self) -> None:
        self.assertIsNone(analysis.pearson_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))

    def test_too_few_points_returns_none(self) -> None:
        self.assertIsNone(analysis.pearson_correlation([1.0], [1.0]))


class DetectMotionEventsTests(unittest.TestCase):
    """`detect_motion_events` replaces the old `detect_anomalies` -- same MAD
    robust-z math, renamed because a statistically extreme motion reading is
    evidence of unusual motion, not by itself evidence of a data defect (see
    analysis.py's module docstring). Every new event starts at tier "motion";
    promotion to "suspicious" is `promote_suspicious_events`'s job, tested
    separately below."""

    def test_planted_outlier_is_flagged_in_distribution_points_are_not(self) -> None:
        in_distribution = [1.0 + 0.01 * i for i in range(20)]
        values = in_distribution + [100.0]
        triples = [(0, i, v) for i, v in enumerate(values)]
        events = analysis.detect_motion_events({"metric": triples}, threshold=6.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].frame_index, 20)
        self.assertAlmostEqual(events[0].value, 100.0)
        self.assertEqual(events[0].tier, "motion")
        self.assertEqual(events[0].corroboration, ())

    def test_no_outliers_when_values_are_uniform(self) -> None:
        triples = [(0, i, 1.0) for i in range(10)]
        events = analysis.detect_motion_events({"metric": triples}, threshold=6.0)
        self.assertEqual(events, [])


class DetectNanOrInfIssuesTests(unittest.TestCase):
    def test_nan_and_inf_values_are_each_flagged(self) -> None:
        named_series = {"hand_speed": [(0, 0, 1.0), (0, 1, float("nan")), (0, 2, float("inf"))]}
        issues = analysis.detect_nan_or_inf_issues(named_series)
        self.assertEqual({issue.frame_index for issue in issues}, {1, 2})
        self.assertTrue(all(issue.kind == "nan_or_inf_value" for issue in issues))

    def test_finite_values_produce_no_issues(self) -> None:
        named_series = {"hand_speed": [(0, 0, 1.0), (0, 1, 2.0)]}
        self.assertEqual(analysis.detect_nan_or_inf_issues(named_series), [])


class DetectSchemaMismatchIssuesTests(unittest.TestCase):
    def test_each_difference_becomes_one_located_issue(self) -> None:
        schema_audit = {
            "baseline_episode": 0,
            "differences": [
                {
                    "field": "left_hand/tracks",
                    "episode_index": 1,
                    "baseline_dtype": "float32",
                    "dtype": "float64",
                    "baseline_lengths": [64],
                    "lengths": [64],
                }
            ],
        }
        issues = analysis.detect_schema_mismatch_issues(schema_audit)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "schema_mismatch")
        self.assertEqual(issues[0].episode_index, 1)
        self.assertIsNone(issues[0].frame_index)


class DetectTimestampIssuesTests(unittest.TestCase):
    def test_each_sync_diagnostic_category_becomes_a_located_issue(self) -> None:
        diagnostics = metrics.SyncDiagnostics(
            frame_count=5,
            median_interval_seconds=1.0,
            large_interval_threshold_seconds=1.5,
            large_interval_count=1,
            large_interval_frame_indices=(3,),
            duplicate_timestamp_count=1,
            duplicate_timestamp_frame_indices=(2,),
            negative_interval_count=1,
            negative_interval_frame_indices=(1,),
            frame_index_discontinuity_count=1,
            frame_index_discontinuity_frame_indices=(4,),
        )
        issues = analysis.detect_timestamp_issues(0, diagnostics)
        kinds = {(issue.kind, issue.frame_index) for issue in issues}
        self.assertEqual(
            kinds,
            {
                ("timestamp_regression", 1),
                ("timestamp_duplicate", 2),
                ("timestamp_large_gap", 3),
                ("frame_index_discontinuity", 4),
            },
        )


class DetectVideoSyncIssuesTests(unittest.TestCase):
    def test_error_past_half_the_median_interval_is_flagged(self) -> None:
        sync_validation = {
            "summary": {"median_frame_interval_seconds": 0.033},
            "results": [
                {"frame_index": 0, "abs_sync_error_seconds": 0.001},
                {"frame_index": 1, "abs_sync_error_seconds": 0.02},  # > 0.0165 (half the median)
            ],
        }
        issues = analysis.detect_video_sync_issues(0, sync_validation)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].frame_index, 1)

    def test_no_median_interval_produces_no_issues(self) -> None:
        sync_validation = {"summary": {"median_frame_interval_seconds": None}, "results": []}
        self.assertEqual(analysis.detect_video_sync_issues(0, sync_validation), [])


class DetectQuaternionViolationIssuesTests(unittest.TestCase):
    def test_deviation_past_tolerance_is_flagged(self) -> None:
        deviations = [0.0, analysis.QUATERNION_NORM_TOLERANCE * 10]
        issues = analysis.detect_quaternion_violation_issues(0, "field", deviations, [0, 1])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].frame_index, 1)

    def test_deviation_within_tolerance_is_not_flagged(self) -> None:
        deviations = [0.0, analysis.QUATERNION_NORM_TOLERANCE / 10]
        self.assertEqual(analysis.detect_quaternion_violation_issues(0, "field", deviations, [0, 1]), [])


class DetectTerminalSentinelIssuesTests(unittest.TestCase):
    def test_all_zero_row_on_non_terminal_frame_is_flagged(self) -> None:
        action_points = [((0.0, 0.0, 0.0),), None, ((0.0, 0.0, 0.0),)]
        issues = analysis.detect_terminal_sentinel_issues(0, "field", action_points, [0, 1, 2])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].frame_index, 1)

    def test_all_zero_row_on_terminal_frame_is_not_flagged(self) -> None:
        action_points = [((0.0, 0.0, 0.0),), None]
        issues = analysis.detect_terminal_sentinel_issues(0, "field", action_points, [0, 1])
        self.assertEqual(issues, [])


class PromoteSuspiciousEventsTests(unittest.TestCase):
    def test_two_metrics_on_the_same_frame_are_promoted(self) -> None:
        events = [
            analysis.MotionEvent(episode_index=0, frame_index=5, metric="hand_speed", value=1.0, robust_z=7.0),
            analysis.MotionEvent(episode_index=0, frame_index=5, metric="camera_speed", value=2.0, robust_z=8.0),
            analysis.MotionEvent(episode_index=0, frame_index=9, metric="hand_speed", value=1.0, robust_z=6.5),
        ]
        promoted = analysis.promote_suspicious_events(events, [])
        by_frame = {event.frame_index: event for event in promoted}
        self.assertEqual(by_frame[5].tier, "suspicious")
        self.assertEqual(by_frame[9].tier, "motion")

    def test_a_verified_issue_at_the_same_frame_promotes_a_lone_motion_event(self) -> None:
        events = [analysis.MotionEvent(episode_index=0, frame_index=5, metric="hand_speed", value=1.0, robust_z=7.0)]
        issues = [analysis.VerifiedIssue(kind="timestamp_regression", episode_index=0, frame_index=5, detail="x")]
        promoted = analysis.promote_suspicious_events(events, issues)
        self.assertEqual(promoted[0].tier, "suspicious")
        self.assertIn("coincides with a verified data-quality issue at this frame", promoted[0].corroboration)

    def test_lone_event_with_no_corroboration_stays_motion_tier(self) -> None:
        events = [analysis.MotionEvent(episode_index=0, frame_index=5, metric="hand_speed", value=1.0, robust_z=7.0)]
        promoted = analysis.promote_suspicious_events(events, [])
        self.assertEqual(promoted[0].tier, "motion")
        self.assertEqual(promoted[0].corroboration, ())


@unittest.skipUnless(_DATASET_ROOT.is_dir(), f"Local dataset not found at {_DATASET_ROOT}")
class RunFullAnalysisIntegrationTests(unittest.TestCase):
    """Bounded, structural-only checks against the real local dataset.

    Never asserts specific dataset values as if they were test fixtures -- only
    that the pipeline runs end-to-end, is deterministic, and produces a
    JSON-serializable, internally consistent structure.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = analysis.run_full_analysis(_DATASET_ROOT, seed=20260830, random_count=5)

    def test_is_json_serializable(self) -> None:
        json.dumps(self.report)  # raises on failure

    def test_episode_count_matches_payload_length(self) -> None:
        self.assertEqual(self.report["episode_count"], len(self.report["episodes"]))

    def test_deterministic_across_runs(self) -> None:
        second = analysis.run_full_analysis(_DATASET_ROOT, seed=20260830, random_count=5)
        self.assertEqual(
            self.report["global"]["lag_residuals"]["1"]["mean"],
            second["global"]["lag_residuals"]["1"]["mean"],
        )

    def test_lag_plus_one_residual_is_near_zero_on_real_data(self) -> None:
        # Cross-checks the Phase 2/4-verified action[i] == track[i+1] relationship.
        mean_residual = self.report["global"]["lag_residuals"]["1"]["mean"]
        self.assertIsNotNone(mean_residual)
        self.assertLess(mean_residual, 1e-3)

    def test_quaternion_norms_are_finite_and_reported(self) -> None:
        summary = self.report["global"]["quaternion_norm"]
        self.assertEqual(summary["nan_count"], 0)
        self.assertGreater(summary["finite_count"], 0)

    def test_articulation_rate_is_finite_and_nonnegative_on_real_data(self) -> None:
        summary = self.report["global"]["articulation_rate"]
        self.assertGreater(summary["finite_count"], 0)
        self.assertGreaterEqual(summary["min"], 0.0)

    def test_keypoint_quaternions_are_verified_unit_norm_on_real_data(self) -> None:
        # Cross-checks the semantics.py-documented empirical finding: every
        # keypoint quaternion in the local dataset has unit norm.
        quality = self.report["observation_field_quality"]
        self.assertEqual(quality["keypoint_quaternion_invalid_frame_count"], 0)
        self.assertLess(quality["keypoint_quaternion_max_deviation"]["max"], 1e-3)

    def test_mano_betas_are_all_zero_on_real_data(self) -> None:
        # Cross-checks the semantics.py-documented empirical finding: the
        # shape/personalization channel is unpopulated in the local dataset.
        self.assertTrue(self.report["observation_field_quality"]["mano_betas_all_zero"])

    def test_motion_events_are_concentrated_in_the_expected_skewed_metrics(self) -> None:
        # Structural, not a hardcoded count (which would go stale the moment
        # the dataset or threshold changes): this cross-checks that motion
        # events come only from the metrics independently found to be
        # right-skewed (Statistics tab), never from the tightly/near-
        # degenerately distributed ones.
        metrics_with_events = {event["metric"] for event in self.report["motion_events"]}
        for tight_metric in ("hand_span", "quaternion_norm", "action_residual_k1"):
            self.assertNotIn(tight_metric, metrics_with_events)

    def test_motion_event_tiers_are_valid_and_corroboration_matches_tier(self) -> None:
        for event in self.report["motion_events"]:
            self.assertIn(event["tier"], ("motion", "suspicious"))
            if event["tier"] == "suspicious":
                self.assertGreater(len(event["corroboration"]), 0)
            else:
                self.assertEqual(len(event["corroboration"]), 0)

    def test_verified_issues_have_required_fields(self) -> None:
        for issue in self.report["verified_issues"]:
            for key in ("kind", "episode_index", "frame_index", "detail"):
                self.assertIn(key, issue)

    def test_no_timestamp_regressions_or_frame_index_discontinuities_on_real_data(self) -> None:
        # Cross-checks Phase 2's independently-verified finding that stored
        # timestamps are strictly increasing with no gaps in this dataset --
        # now backed by a dedicated detector instead of only a hand-run check.
        kinds = {issue["kind"] for issue in self.report["verified_issues"]}
        self.assertNotIn("timestamp_regression", kinds)
        self.assertNotIn("frame_index_discontinuity", kinds)


if __name__ == "__main__":
    unittest.main()
