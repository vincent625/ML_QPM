"""Safety and directionality regressions for low-read hypothesis generation."""
from __future__ import annotations

import copy
import json
import unittest

from csa_root_cause_hypotheses import (
    AMBIGUOUS, PINNED, RAWTRACE, build_root_cause_assessment, hypothesis_html,
)


MEDIAN = "plugin.rawtrace.image.pixel_median.std"
Q90 = "plugin.rawtrace.image.pixel_q90.std"
ENTROPY = "plugin.rawtrace.image.entropy.std"
TRIGGER = {"below_chip_lane_read_criterion": True}


def contributor(feature=MEDIAN, rank=1, sign="negative"):
    return dict(feature_name=feature, rank=rank, component="plugin:rawtrace", tier="C",
        heldout_mae_increase=1_000_000, importance_ci_low=500_000,
        importance_ci_high=1_500_000, outer_folds=5, selection_count=5, positive_folds=5,
        coefficient_positive_folds=5 if sign == "positive" else 0,
        coefficient_negative_folds=5 if sign == "negative" else 0,
        direction=sign + "_conditional_association", source_file="plugin", source_field=feature)


def metric(feature=MEDIAN, value=3.0, *, family="rawtrace"):
    return dict(feature=feature, label=RAWTRACE.get(feature, feature), family=family,
        status="available", target_value=value, peer_q05=1.0, peer_median=1.5, peer_q95=2.0,
        reference_physical_runs=8, reference_mode="same_instrument_same_lane",
        source_file="rawTrace/* images", source_field="SD across grayscale medians", unit="normalized image luminance",
        # Deliberately stale: the engine must recompute the actual comparison.
        deviation="within_empirical_range")


def pinned(value=3.0):
    row = metric(PINNED, value, family="chip_diagnostics")
    row.update(unit="%", source_field="offset.PercPinned", source_file="CSA/chipDiagnostics/results.json")
    return row


class DirectionAndEvidenceTests(unittest.TestCase):
    def assess(self, measurements=None, ranks=None, trigger=None):
        return build_root_cause_assessment(
            [metric()] if measurements is None else measurements,
            [contributor()] if ranks is None else ranks,
            TRIGGER if trigger is None else trigger)

    def test_single_adverse_top_metric_produces_one_guarded_hypothesis(self):
        result = self.assess()
        self.assertEqual(result["status"], "single_source_descriptive_clue")
        self.assertEqual(result["distinct_source_family_count"], 1)
        self.assertEqual(result["hypotheses"][0]["hypothesis_id"], "signal_trace_review")
        record = result["contributor_assessments"][0]
        self.assertEqual(record["assessment_status"], "model_concordant_low_read_clue")
        self.assertEqual(record["deviation"], "above_empirical_range")
        self.assertEqual(record["global_heldout_mae_increase"], 1_000_000)
        self.assertEqual(record["coefficient_negative_folds"], 5)

    def test_opposite_direction_is_counter_evidence_not_rootcause(self):
        result = self.assess([metric(value=0)])
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["hypotheses"], [])
        self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "anomaly_direction_discordant")

    def test_positive_conditional_direction_requires_lower_value(self):
        result = self.assess([metric(ENTROPY, .5)], [contributor(ENTROPY, 7, "positive")])
        self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "model_concordant_low_read_clue")
        result = self.assess([metric(ENTROPY, 3)], [contributor(ENTROPY, 7, "positive")])
        self.assertEqual(result["status"], "unresolved")

    def test_all_folds_and_ci_support_required(self):
        for update in [
            {"selection_count": 4}, {"positive_folds": 4},
            {"coefficient_positive_folds": 1, "coefficient_negative_folds": 4},
            {"importance_ci_low": 0}, {"importance_ci_low": -1},
            {"heldout_mae_increase": 0}, {"coefficient_negative_folds": None},
            {"direction": "positive_conditional_association"},
            {"outer_folds": 2, "selection_count": 2, "positive_folds": 2,
                "coefficient_negative_folds": 2},
        ]:
            with self.subTest(update=update):
                row = {**contributor(), **update}
                result = self.assess(ranks=[row])
                self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "anomaly_direction_unresolved")
                self.assertEqual(result["hypotheses"], [])

    def test_correlated_rawtrace_metrics_are_one_source_not_three(self):
        rows = [contributor(), contributor(Q90, 5), contributor(ENTROPY, 7, "positive")]
        values = [metric(), metric(Q90), metric(ENTROPY, .5)]
        result = self.assess(values, rows)
        self.assertEqual(result["status"], "single_source_descriptive_clue")
        self.assertEqual(result["distinct_source_family_count"], 1)
        self.assertEqual(len(result["hypotheses"]), 1)
        self.assertEqual(len(result["hypotheses"][0]["supporting_features"]), 3)
        self.assertNotIn("total_importance", result["hypotheses"][0])

    def test_distinct_chip_source_converges_without_claiming_causality(self):
        result = self.assess([metric(), pinned()])
        self.assertEqual(result["status"], "convergent_descriptive_clues")
        self.assertEqual(result["distinct_source_family_count"], 2)
        self.assertFalse(result["mechanisms_confirmed"])
        chip = result["hypotheses"][1]
        self.assertEqual(chip["evidence_role"], "supplemental_not_top_ranked")
        self.assertIsNone(chip["strongest_global_rank"])

    def test_chip_low_and_high_components_do_not_duplicate_sources(self):
        extra = pinned()
        extra["feature"] = PINNED + "_low"
        result = self.assess([pinned(), extra], [])
        self.assertEqual(len(result["supplemental_assessments"]), 1)
        self.assertEqual(result["distinct_source_family_count"], 1)
        self.assertEqual(result["hypotheses"][0]["hypothesis_id"], "offset_calibration_review")

    def test_low_pinned_percentage_not_an_adverse_explanation(self):
        result = self.assess([pinned(0)], [])
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["supplemental_assessments"][0]["assessment_status"], "supplemental_anomaly_not_adverse")

    def test_original_percentage_identity_and_range_are_required(self):
        for update in [{"unit": "count"}, {"source_field": "offset.perc_pinned"},
                       {"target_value": 101}, {"peer_q95": 101}]:
            result = self.assess([{**pinned(), **update}], [])
            self.assertEqual(result["status"], "unresolved")
            self.assertFalse(result["supplemental_assessments"][0]["supports_hypothesis"])

    def test_one_concordant_and_one_discordant_retains_counter_evidence(self):
        result = self.assess([metric(), metric(Q90, 0)], [contributor(), contributor(Q90, 5)])
        self.assertEqual(result["hypotheses"][0]["counter_evidence_features"], [Q90])
        self.assertEqual(result["hypotheses"][0]["supporting_features"], [MEDIAN])


class AbstentionTests(unittest.TestCase):
    def test_context_metadata_and_ambiguous_counts_always_withheld(self):
        features = ["context.analyzed_lane_count", "csa.ion_params_00.exp_json_log_chipminorrev", *AMBIGUOUS]
        contributors = [contributor(feature, index + 1) for index, feature in enumerate(features)]
        result = build_root_cause_assessment([metric(feature, 1e12) for feature in features], contributors, TRIGGER)
        self.assertEqual(result["hypotheses"], [])
        statuses = [row["assessment_status"] for row in result["contributor_assessments"]]
        self.assertEqual(statuses[:2], ["withheld_context_or_configuration"] * 2)
        self.assertEqual(statuses[2:], ["withheld_ambiguous_count_proxy"] * 2)

    def test_forbidden_tier_or_family_cannot_be_rescued_by_extreme_value(self):
        for row in [{**contributor(), "tier": "D"},
                    {**contributor(), "component": "plugin:true_loading"},
                    contributor("plugin.edge_effects.image.gradient_x.mean")]:
            result = build_root_cause_assessment([metric(row["feature_name"], 1e12)], [row], TRIGGER)
            self.assertEqual(result["hypotheses"], [])
            self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "withheld_leakage_or_forbidden_family")

    def test_missing_peer_history_and_nonfinite_or_invalid_ranges_abstain(self):
        updates = [
            {"status": "target_metric_unavailable"}, {"reference_physical_runs": 7},
            {"reference_physical_runs": 100.5}, {"reference_mode": "none"},
            {"target_value": float("inf")}, {"target_value": float("nan")},
            {"peer_q05": None}, {"peer_median": 5}, {"peer_q95": 0},
            {"target_value": True}, {"peer_q95": "not_a_number"},
        ]
        for update in updates:
            with self.subTest(update=update):
                result = build_root_cause_assessment([{**metric(), **update}], [contributor()], TRIGGER)
                self.assertEqual(result["hypotheses"], [])
                self.assertEqual(result["status"], "unresolved")

    def test_inside_and_boundary_are_not_outliers(self):
        for value in [1, 1.5, 2]:
            result = build_root_cause_assessment([metric(value=value)], [contributor()], TRIGGER)
            self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "within_empirical_range")
            self.assertEqual(result["hypotheses"], [])

    def test_missing_adapter_and_duplicate_records_abstain(self):
        result = build_root_cause_assessment([], [contributor()], TRIGGER)
        self.assertEqual(result["status"], "unresolved")
        result = build_root_cause_assessment([metric(), metric()], [contributor()], TRIGGER)
        self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "unresolved_duplicate_feature")
        result = build_root_cause_assessment([metric()], [contributor(), contributor()], TRIGGER)
        self.assertEqual(result["hypotheses"], [])

    def test_fixed_top_ten_policy_cannot_expand_to_target_outlier(self):
        for rank in [None, 0, 11, 100]:
            result = build_root_cause_assessment([metric()], [{**contributor(), "rank": rank}], TRIGGER)
            self.assertEqual(result["hypotheses"], [])
            self.assertEqual(result["contributor_assessments"][0]["assessment_status"], "withheld_outside_fixed_top_ten")

    def test_length_accuracy_only_no_lowread_hypotheses_and_disabled_abstains(self):
        for trigger in [{}, {"low_outcomes": ["mean_read_length_bp"]}, {"low_outcomes": ["raw_read_accuracy_pct"]}]:
            result = build_root_cause_assessment([metric()], [contributor()], trigger)
            self.assertEqual(result["status"], "not_triggered")
            self.assertEqual(result["hypotheses"], [])
        result = build_root_cause_assessment([metric()], [contributor()], TRIGGER, enabled=False)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["hypotheses"], [])
        result = build_root_cause_assessment([metric()], [contributor()], {"low_outcomes": ["total_reads"]})
        self.assertTrue(result["triggered"])

    def test_inputs_are_not_mutated_and_outputs_deterministic(self):
        metrics = [metric(Q90), metric()]
        ranks = [contributor(Q90, 5), contributor()]
        before = copy.deepcopy((metrics, ranks, TRIGGER))
        first = build_root_cause_assessment(metrics, ranks, TRIGGER)
        second = build_root_cause_assessment(metrics[::-1], ranks[::-1], TRIGGER)
        self.assertEqual((metrics, ranks, TRIGGER), before)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        self.assertEqual([row["global_rank"] for row in first["contributor_assessments"]], [1, 5])


class PresentationTests(unittest.TestCase):
    def test_html_escapes_archive_and_model_labels(self):
        row = metric()
        row.update(label="<script>alert(1)</script>", source_file="<img onerror='x'>")
        result = build_root_cause_assessment([row], [contributor()], TRIGGER)
        page = hypothesis_html(result)
        self.assertNotIn("<script>", page)
        self.assertNotIn("<img onerror=", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&lt;img onerror=", page)
        self.assertIn("How the plugin reached this assessment", page)
        self.assertIn("Contributor-by-contributor case audit", page)

    def test_never_invents_causal_confidence_lost_reads_or_settings(self):
        result = build_root_cause_assessment([metric(), pinned()], [contributor()], TRIGGER)
        self.assertEqual(result["causal_status"], "candidate_explanations_not_proven_causes")
        self.assertFalse(result["mechanisms_confirmed"])
        self.assertFalse(result["model_retrained"])
        self.assertFalse(result["instrument_settings_changed"])
        self.assertIsNone(result["estimated_reads_lost"])
        self.assertIsNone(result["expected_read_gain"])
        self.assertNotIn("causal_probability", result)
        self.assertNotIn("combined_importance", result)
        page = hypothesis_html(result)
        self.assertIn("not confirmed root causes", page)
        self.assertIn("not a local causal contribution", page)
        self.assertEqual(len(result["decision_flow"]), 8)

    def test_unresolved_and_untriggered_html_are_explicit(self):
        result = build_root_cause_assessment([], [contributor()], TRIGGER)
        self.assertIn("No supported explanation", hypothesis_html(result))
        result = build_root_cause_assessment([], [], {})
        self.assertIn("not_triggered", hypothesis_html(result))
        self.assertNotIn("Contributor-by-contributor", hypothesis_html(result))


if __name__ == "__main__":
    unittest.main()
