"""CSA diagnostic reporting/cache guards do not alter QPM classifications."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from qpm_reference_bank import file_sha256
from run_flagged_qpm_batch import (
    completed_report_cache_valid, csa_scope_fields, diagnostic_plan_inputs,
    scope_result, summarize,
)


class CSALowPerformanceBatchTests(unittest.TestCase):
    def setUp(self):
        gallery = patch("qpm_same_run_curves.cache_artifacts_valid", return_value=True)
        gallery.start()
        self.addCleanup(gallery.stop)

    def test_missing_section_is_unresolved_not_normal(self):
        fields = csa_scope_fields({})
        self.assertEqual(fields["csa_diagnostic_status"], "not_recorded")
        self.assertIsNone(fields["csa_diagnostic_triggered"])
        self.assertIn("unresolved", fields["csa_diagnostic_summary"])

    def test_only_available_empirical_outliers_are_counted(self):
        result = csa_scope_fields(dict(status="diagnostic_clues_found", triggered=True,
            summary="Descriptive clues", metrics=[
                dict(label="Pinned", status="available", deviation="above_peer_q95"),
                dict(label="RawTrace", status="available", deviation="within_empirical_range"),
                dict(label="Missing", status="insufficient_history", deviation="above_peer_q95"),
                dict(label="Unknown", status="available", deviation=None)]))
        self.assertEqual(result["csa_assessed_metrics"], 4)
        self.assertEqual(result["csa_available_metrics"], 3)
        self.assertEqual(result["csa_abnormal_metrics"], 1)
        self.assertEqual(result["csa_abnormal_metric_names"], "Pinned")
        self.assertEqual(result["csa_abnormal_metric_findings"], "Pinned: above_peer_q95")

    def completed_fixture(self, directory, diagnostics=None):
        root = Path(directory)
        log = dict(run_metadata=dict(run_date="2026-10-01", run_start_time="2026-10-01T00:00:00Z",
            instrument="instrument", assay_type="assay", chip_type="GX9"),
            trace_quality=dict(target_eligible_waveforms=0, target_er72_excluded_waveforms=1),
            report_branch="low_performance", plugin_version="test")
        (root / "plugin_run_log.json").write_text(json.dumps(log))
        conclusion = dict(materials={})
        if diagnostics is not None:
            conclusion["csa_low_performance_diagnostics"] = diagnostics
        (root / "initial_failure_candidates.json").write_text(json.dumps(conclusion))
        pd.DataFrame([dict(baseline_id=baseline, status="unresolved", scored_operations=0,
            positive_operations=0, earliest_operation_index=None, reference_physical_runs=0,
            reference_instruments=0) for baseline in ["within_run", "same_instrument", "cross_instrument"]]
            ).to_csv(root / "qpm_baseline_comparison_summary.csv", index=False)
        pd.DataFrame(columns=["evidence_role"]).to_csv(root / "qpm_shared_path_evidence.csv", index=False)
        pd.DataFrame(columns=["available", "baseline_id", "slot", "operation_index", "step", "evidence_role"]
            ).to_csv(root / "qpm_displayed_evidence.csv", index=False)
        return dict(observation_id="observation", physical_run_id="run", flagged=True,
            chip_lane="1", report_dir=str(root))

    def test_completed_qpm_only_fixture_retains_its_old_qpm_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.completed_fixture(directory)
            result = scope_result(task, dict(status="completed"))
            self.assertEqual(result["diagnostic_status"], "no_usable_qpm_waveforms")
            self.assertEqual(result["csa_diagnostic_status"], "not_recorded")
            self.assertIsNone(result["earliest_operation_index"])

    def test_csa_clue_cannot_become_qpm_localization_or_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.completed_fixture(directory, dict(status="diagnostic_clues_found", triggered=True,
                summary="One empirical chip outlier; not causal", metrics=[
                    dict(label="Pinned", status="available", deviation="above_peer_q95")]))
            result = scope_result(task, dict(status="completed"))
            self.assertEqual(result["diagnostic_status"], "no_usable_qpm_waveforms")
            self.assertEqual(result["csa_diagnostic_status"], "diagnostic_clues_found")
            self.assertEqual(result["csa_abnormal_metrics"], 1)
            self.assertIsNone(result["earliest_operation_index"])
            self.assertEqual(result["causal_interpretation"], "Not a proven root cause")

    def test_pending_and_failed_csa_statuses_are_explicit(self):
        task = dict(observation_id="o", physical_run_id="p", flagged=True)
        self.assertEqual(scope_result(task, {})["csa_diagnostic_status"], "pending")
        failed = scope_result(task, dict(status="failed", error="missing input"))
        self.assertEqual(failed["csa_diagnostic_status"], "processing_failed")
        self.assertIsNone(failed["csa_abnormal_metrics"])

    def cache_fixture(self, directory, hashes=True):
        root = Path(directory)
        code = Path(__import__("csa_low_performance_diagnostics").__file__)
        (root / "qpm_dropout_report.html").write_text("report")
        (root / "diagnostic_workflow.json").write_text(json.dumps(dict(version="csa-diagnostic-workflow-1.0.0")))
        (root / "plugin_run_log.json").write_text(json.dumps(dict(code_sha256=
            {code.name: file_sha256(code), "qpm_same_run_curves.py": file_sha256(code.parent / "qpm_same_run_curves.py")} if hashes else {})))
        return root

    def test_cache_rejects_old_qpm_only_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.cache_fixture(directory, hashes=False)
            with patch("csa_low_performance_diagnostics.cache_artifacts_valid", return_value=True) as freshness:
                self.assertFalse(completed_report_cache_valid(root))
                freshness.assert_not_called()

    def test_cache_requires_explicit_diagnostic_workflow_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.cache_fixture(directory)
            (root / "diagnostic_workflow.json").unlink()
            with patch("csa_low_performance_diagnostics.cache_artifacts_valid", return_value=True):
                self.assertFalse(completed_report_cache_valid(root))

    def test_cache_rejects_stale_same_run_curve_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.cache_fixture(directory)
            with patch("csa_low_performance_diagnostics.cache_artifacts_valid", return_value=True), \
                 patch("qpm_same_run_curves.cache_artifacts_valid", return_value=False):
                self.assertFalse(completed_report_cache_valid(root))

    def test_cache_requires_helper_freshness_for_all_manifest_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.cache_fixture(directory)
            manifest = root / "csa_low_performance_manifest.json"
            for status in ["disabled", "not_triggered", "unresolved", "diagnostic_clues_found", "no_outlier_in_available_views"]:
                manifest.write_text(json.dumps(dict(status=status)))
                for fresh in [False, True]:
                    with self.subTest(status=status, fresh=fresh):
                        with patch("csa_low_performance_diagnostics.cache_artifacts_valid", return_value=fresh) as freshness:
                            self.assertEqual(completed_report_cache_valid(root), fresh)
                            freshness.assert_called_once_with(manifest)

    def test_cache_rejects_changed_source_before_accepting_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.cache_fixture(directory)
            (root / "plugin_run_log.json").write_text(json.dumps(dict(code_sha256=
                {"csa_low_performance_diagnostics.py": "not-current"})))
            with patch("csa_low_performance_diagnostics.cache_artifacts_valid", return_value=True) as freshness:
                self.assertFalse(completed_report_cache_valid(root))
                freshness.assert_not_called()

    def test_external_inputs_pinned_once_per_distinct_project_including_disabled_mode(self):
        tasks = [dict(project="/project/bank"), dict(project="/project/manual"), dict(project="/project/bank")]
        for enabled in [True, False]:
            config = dict(enable_csa_low_performance_diagnostics=enabled)
            with patch("csa_low_performance_diagnostics.diagnostic_input_fingerprints",
                       side_effect=lambda project, settings: {str(project / "data"): str(settings["enable_csa_low_performance_diagnostics"])}) as hashes:
                result = diagnostic_plan_inputs(tasks, config)
                self.assertEqual(hashes.call_count, 2)
                self.assertEqual(set(result), {"/project/bank", "/project/manual"})
                self.assertEqual(result["/project/bank"]["/project/bank/data"], str(enabled))

    def test_cohort_outputs_keep_csa_clues_in_separate_columns_and_counts(self):
        original = dict(physical_run_id="run", run_date="2026-10-01", instrument="instrument",
            run_name="test run", assay_type="assay", assay_panel="panel", chip_type="GX9", flags="low reads")
        row = dict(physical_run_id="run", observation_id="scope", execution_status="completed",
            flagged=True, diagnostic_status="no_usable_qpm_waveforms", earliest_operation_index=None,
            chip_lane="1", report="", error="")
        row.update(csa_scope_fields(dict(status="diagnostic_clues_found", triggered=True,
            summary="One empirical outlier; not a cause", metrics=[
                dict(label="Pinned", status="available", deviation="above_peer_q95")])))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("run_flagged_qpm_batch.scope_result", return_value=row):
                progress = summarize(output, pd.DataFrame([original]), [dict(observation_id="scope")], {},
                    dict(flagged_scopes=1, unflagged_sibling_scopes=0, reference_bank="frozen", plugin_version="test"))
            self.assertEqual(progress["diagnostic_scope_counts"], {"no_usable_qpm_waveforms": 1})
            self.assertEqual(progress["csa_diagnostic_scope_counts"], {"diagnostic_clues_found": 1})
            self.assertEqual(progress["csa_scopes_with_diagnostic_clues"], 1)
            scope = pd.read_csv(output / "scope_qpm_results.csv").iloc[0]
            self.assertEqual(scope["csa_abnormal_metrics"], 1)
            run = pd.read_csv(output / "physical_run_qpm_results.csv").iloc[0]
            self.assertEqual(run["qpm_findings"], "no_usable_qpm_waveforms")
            self.assertEqual(run["csa_diagnostic_findings"], "diagnostic_clues_found")
            self.assertEqual(run["csa_abnormal_metrics"], 1)
            self.assertIn("Separate CSA diagnostic status", (output / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
