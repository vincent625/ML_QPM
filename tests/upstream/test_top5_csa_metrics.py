"""Fixed global total-read importance is not a local causal explanation."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import csa_low_performance_diagnostics as diagnostics

RAW="plugin.rawtrace.image.pixel_median.std"
LANES="context.analyzed_lane_count"
COUNT="plugin.chip_diagnostics.structured.results_offset_pixel_in_range.mean"
EDGE="plugin.chip_diagnostics.structured.results_edge_metrics_pixels_n.std"
Q90="plugin.rawtrace.image.pixel_q90.std"
FEATURES=[RAW,LANES,COUNT,EDGE,Q90]

def ranking_rows():
    rows=[]
    for i,feature in enumerate(FEATURES):
        rows.append(dict(feature_name=feature,component="context" if feature==LANES else "plugin:rawtrace" if ".rawtrace." in feature else "plugin:chip_diagnostics",
            outcome="total_reads",model="elastic_net",validation_scheme="grouped",tier="B" if feature==LANES else "C",
            heldout_mae_increase=7e6-i*1e6,importance_ci_low=6e6-i*1e6,importance_ci_high=8e6-i*1e6,
            outer_folds=5,positive_folds=5,selection_frequency=1.,support_status="supported_diagnostic_association_not_causal",
            uncertainty_method="outer_fold_bootstrap_exploratory_not_physical_run_ci",source_file="plugin",source_field=feature,
            unit="source_native_no_unverified_conversion"))
    return rows

class TopFiveRankingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/"feature_rankings.csv"
    def write(self,rows=None):
        pd.DataFrame(ranking_rows() if rows is None else rows).to_csv(self.path,index=False)
    def test_fixed_outcome_model_and_scheme_not_cherry_picked(self):
        rows=ranking_rows()
        rows += [{**rows[0],"model":"extra_trees","heldout_mae_increase":1e12},
                 {**rows[0],"validation_scheme":"chronological","heldout_mae_increase":1e13},
                 {**rows[0],"outcome":"mean_read_length_bp","heldout_mae_increase":1e14}]
        self.write(list(reversed(rows)))
        records,policy=diagnostics._top5_feature_priority(self.path)
        self.assertEqual([r["feature_name"] for r in records],FEATURES)
        self.assertEqual(policy["status"],"available")
        self.assertEqual(policy["validation_scheme"],"grouped")
        self.assertNotIn("plugin:chip_diagnostics",[r["feature_name"] for r in records])
    def test_missing_or_insufficient_ranking_has_no_component_substitute(self):
        self.assertEqual(diagnostics._top5_feature_priority(self.path)[0],[])
        self.write(ranking_rows()[:4])
        records,policy=diagnostics._top5_feature_priority(self.path)
        self.assertEqual(records,[]);self.assertIn("fewer than five",policy["reason"])
    def test_duplicate_and_nonfinite_rankings_fail_closed(self):
        for rows in [ranking_rows()+[ranking_rows()[0]],[{**r,"heldout_mae_increase":float("inf")} if i==0 else r for i,r in enumerate(ranking_rows())]]:
            self.write(rows)
            records,policy=diagnostics._top5_feature_priority(self.path)
            self.assertFalse(records);self.assertEqual(policy["status"],"unavailable")
    def test_forbidden_feature_cannot_fall_through_to_sixth(self):
        rows=ranking_rows()
        rows[0].update(feature_name="plugin.true_loading.structured.x.mean",component="plugin:true_loading")
        rows.append({**rows[-1],"feature_name":"sixth_safe_feature","heldout_mae_increase":1.})
        self.write(rows)
        records,policy=diagnostics._top5_feature_priority(self.path)
        self.assertFalse(records);self.assertIn("forbidden",policy["reason"])
    def test_tier_d_is_withheld(self):
        rows=ranking_rows();rows[0]["tier"]="D";self.write(rows)
        self.assertEqual(diagnostics._top5_feature_priority(self.path)[1]["status"],"unavailable")
    def test_ranking_and_metadata_files_are_fingerprinted(self):
        self.write();config={"csa_diagnostic_model_study":str(self.root)}
        first=diagnostics.diagnostic_input_fingerprints(self.root,config)
        self.assertEqual(first[str(self.path.resolve())],diagnostics._sha(self.path))
        self.path.write_text(self.path.read_text()+"\n")
        second=diagnostics.diagnostic_input_fingerprints(self.root,config)
        self.assertNotEqual(first[str(self.path.resolve())],second[str(self.path.resolve())])
        for name in ["specific_feature_details.csv","feature_dictionary.csv"]:
            self.assertIn(str((self.root/name).resolve()),first)

class TopFiveCaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        path=Path(self.temp.name)/"feature_rankings.csv"
        pd.DataFrame(ranking_rows()).to_csv(path,index=False)
        self.rows,self.policy=diagnostics._top5_feature_priority(path)
    def annotate(self,metrics=None,trigger=None,fresh=None,frozen=None):
        return diagnostics._top5_case_evidence(self.rows,metrics or [],trigger or {},fresh or {},{},frozen or {})
    def test_rawtrace_uses_guarded_local_comparison_not_local_attribution(self):
        metric=dict(feature=RAW,status="available",target_value=.023,peer_median=.020,peer_q05=.018,peer_q95=.021,
            reference_physical_runs=13,reference_mode="same_instrument_all_lanes",reference_instruments=1,deviation="above_empirical_range")
        row=self.annotate(metrics=[metric],fresh={RAW:.023})[0]
        self.assertEqual(row["target_value"],.023);self.assertTrue(row["local_interpretation_allowed"])
        self.assertIn("Rendered-image proxy",row["case_evidence"]);self.assertIn("not a cause",row["case_evidence"])
    def test_missing_rawtrace_is_unresolved(self):
        row=self.annotate()[0]
        self.assertIsNone(row["target_value"]);self.assertFalse(row["local_interpretation_allowed"])
        self.assertEqual(row["case_status"],"target_metric_unavailable")
    def test_lanes_are_adjustment_not_a_failure_cause(self):
        row=self.annotate(trigger={"chip_lane_read_criterion":{"evaluated_lane_count":2}})[1]
        self.assertEqual(row["target_value"],2.)
        self.assertEqual(row["case_status"],"adjustment_context_not_root_cause")
        self.assertFalse(row["local_interpretation_allowed"])
    def test_ambiguous_counts_ranked_but_no_local_health_inference(self):
        rows=self.annotate(frozen={COUNT:123456789.,EDGE:4500000.})
        for row in rows[2:4]:
            self.assertIsNone(row["target_value"]);self.assertIsNotNone(row["frozen_model_input_value"])
            self.assertFalse(row["local_interpretation_allowed"])
            self.assertEqual(row["case_status"],"interpretation_withheld_ambiguous_count_proxy")
        self.assertEqual(rows[2]["feature_name"],COUNT)
        self.assertNotIn(diagnostics.PIN_PREFIX+"perc_pinned",[r["feature_name"] for r in rows])
    def test_html_exactly_five_individual_features_and_guardrails(self):
        page=diagnostics.top5_metrics_html(dict(top5_ml_policy=self.policy,top5_ml_metrics=self.annotate()))
        self.assertEqual(page.count("<code>"),5)
        for text in ["physical-run-grouped Elastic Net","not a failure cause","not replacements","not physical-run or causal-effect"]:
            self.assertIn(text,page)
    def test_html_escapes_feature_and_source_content(self):
        rows=self.annotate();rows[0]["feature_name"]="<script>bad</script>";rows[0]["source_file"]="<img onerror='bad'>"
        page=diagnostics.top5_metrics_html(dict(top5_ml_policy=self.policy,top5_ml_metrics=rows))
        self.assertNotIn("<script>",page);self.assertNotIn("<img onerror",page);self.assertIn("&lt;script&gt;",page)

if __name__=="__main__":unittest.main()
