"""Detection routes must not turn total-read model metrics into barcode causes."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from qpm_dropout_plugin import (
    build_diagnostic_workflow, diagnostic_workflow_html, make_html_report,
)


def lineage(classes=None):
    classes = classes or ["surviving"] * 8
    return pd.DataFrame([
        dict(is_control=False, material_type="DNA" if i < 4 else "RNA",
             planned_order=i + 1, read_count=1000 if kind == "surviving" else 0,
             outcome_class=kind)
        for i, kind in enumerate(classes)
    ])


def trigger(*, dropout=False, imbalance=False, absolute_low=False, low_outcomes=None,
            absolute_status="passed"):
    low_outcomes = list(low_outcomes or [])
    reasons = (["barcode_dropout"] if dropout else []) + (["barcode_imbalance"] if imbalance else [])
    if absolute_low:
        reasons.append("below_chip_lane_total_read_criterion")
    if low_outcomes:
        reasons.append("low_peer_relative_performance")
    return dict(has_dropout=dropout, has_imbalance=imbalance,
                below_chip_lane_read_criterion=absolute_low, low_outcomes=low_outcomes,
                reasons=reasons, run_qpm=bool(reasons), performance_boundary="q05",
                chip_lane_read_criterion=dict(total_read_criterion_status=absolute_status,
                    low_performance_trigger=absolute_low, chip_type="GX9", evaluated_lane_count=1,
                    total_reads_value=10_000_000 if absolute_low else 60_000_000,
                    total_reads_threshold=50_000_000))


def csa(status="diagnostic_clues_found", count=5):
    return dict(status=status, triggered=True,
                top5_ml_metrics=[dict(feature=f"feature_{i}", rank=i + 1) for i in range(count)],
                top5_ml_policy=dict(model="elastic_net", validation_scheme="grouped"))


class DiagnosticWorkflowTests(unittest.TestCase):
    def assert_qpm_only_barcode_reasons(self, workflow):
        for key in ["barcode_dropout", "barcode_imbalance"]:
            self.assertEqual(workflow[key]["explanation_sources"], ["qpm"])
            self.assertFalse(workflow[key]["ml_metrics_allowed_as_barcode_reason"])
            self.assertEqual(workflow[key]["evidence_scope"], "barcode_path_qpm_only")
        self.assertTrue(workflow["no_ml_substitution_for_missing_barcode_qpm"])
        self.assertTrue(workflow["missing_evidence_does_not_establish_normality"])
        self.assertFalse(workflow["settings_modified"])

    def assert_two_low_performance_parts(self, workflow):
        low = workflow["low_performance"]
        self.assertTrue(low["triggered"])
        self.assertEqual(low["report_parts"], [1, 2])
        self.assertEqual(low["explanation_sources"], ["qpm", "total_read_ml_metrics"])
        parts = {part["part"]: part for part in workflow["report_parts"]}
        self.assertTrue(parts[1]["required"])
        self.assertTrue(parts[2]["required"])
        self.assertEqual(parts[1]["id"], "qpm-investigation")
        self.assertEqual(parts[2]["id"], "csa-low-performance-diagnostics")

    def test_barcode_dropout_without_low_performance_uses_qpm_only(self):
        workflow = build_diagnostic_workflow(lineage(["dropout"] + ["surviving"] * 7),
                                             trigger(dropout=True), csa())
        self.assertEqual(workflow["barcode_dropout"]["status"], "detected")
        self.assert_qpm_only_barcode_reasons(workflow)
        self.assertFalse(workflow["low_performance"]["triggered"])
        self.assertEqual(workflow["low_performance"]["report_parts"], [])
        self.assertFalse(next(p for p in workflow["report_parts"] if p["part"] == 2)["required"])

    def test_imbalance_without_dropout_uses_qpm_only(self):
        workflow = build_diagnostic_workflow(lineage(["partial_imbalance"] + ["surviving"] * 7),
                                             trigger(imbalance=True), csa())
        self.assertEqual(workflow["barcode_imbalance"]["status"], "detected")
        self.assertFalse(workflow["barcode_dropout"]["detected"])
        self.assert_qpm_only_barcode_reasons(workflow)

    def test_absolute_low_reads_requires_both_parts_without_a_barcode_issue(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(absolute_low=True, absolute_status="failed"), csa())
        self.assert_two_low_performance_parts(workflow)
        self.assertFalse(workflow["barcode_dropout"]["detected"])
        self.assertFalse(workflow["barcode_imbalance"]["detected"])
        self.assertIn("below_chip_lane_total_read_criterion", workflow["low_performance"]["triggers"])

    def test_peer_relative_low_outcome_also_requires_both_parts(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(low_outcomes=["total_reads"]), csa())
        self.assert_two_low_performance_parts(workflow)
        self.assertIn("total_reads", workflow["low_performance"]["triggers"])

    def test_read_length_trigger_does_not_relabel_total_read_ml_as_length_explanation(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(low_outcomes=["mean_read_length_bp"]), csa())
        self.assert_two_low_performance_parts(workflow)
        self.assertIn("not a read-length/accuracy explanation", workflow["low_performance"]["ml_evidence_scope"])

    def test_coexisting_dropout_and_low_reads_keep_explanation_scopes_separate(self):
        workflow = build_diagnostic_workflow(lineage(["dropout", "partial_imbalance"] + ["surviving"] * 6),
            trigger(dropout=True, imbalance=True, absolute_low=True, absolute_status="failed"), csa())
        self.assert_qpm_only_barcode_reasons(workflow)
        self.assert_two_low_performance_parts(workflow)
        self.assertEqual(workflow["barcode_dropout"]["status"], "detected")
        self.assertEqual(workflow["barcode_imbalance"]["status"], "detected")

    def test_evaluable_unflagged_run_keeps_qpm_audit_without_ml_part(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(), csa(status="not_triggered", count=0))
        self.assertEqual(workflow["barcode_coverage"]["status"], "complete")
        self.assertEqual(workflow["barcode_dropout"]["status"], "not_detected_in_evaluable_barcodes")
        self.assertEqual(workflow["low_performance"]["status"], "not_triggered_by_available_checks")
        self.assertTrue(workflow["report_parts"][0]["required"])
        self.assertFalse(workflow["report_parts"][1]["required"])

    def test_missing_barcodes_do_not_become_a_negative_dropout_call(self):
        workflow = build_diagnostic_workflow(pd.DataFrame(), trigger(absolute_low=True), csa())
        self.assertEqual(workflow["barcode_coverage"]["status"], "unresolved")
        self.assertEqual(workflow["barcode_dropout"]["status"], "unresolved_or_partial_barcode_coverage")
        self.assertEqual(workflow["barcode_imbalance"]["status"], "unresolved_or_partial_barcode_coverage")
        self.assert_two_low_performance_parts(workflow)

    def test_partial_barcode_outcomes_are_not_reported_as_complete(self):
        frame = lineage()
        frame.loc[0, "outcome_class"] = "unresolved"
        frame.loc[0, "read_count"] = float("nan")
        workflow = build_diagnostic_workflow(frame, trigger(), {})
        self.assertEqual(workflow["barcode_coverage"]["status"], "partial")
        self.assertEqual(workflow["barcode_coverage"]["evaluable_noncontrol_barcodes"], 7)
        self.assertEqual(workflow["barcode_coverage"]["unresolved_rows"], 1)
        self.assertEqual(workflow["barcode_dropout"]["status"], "unresolved_or_partial_barcode_coverage")

    def test_unknown_control_status_cannot_be_counted_as_evaluable(self):
        frame = lineage()
        frame["is_control"] = frame.is_control.astype("boolean")
        frame.loc[0, "is_control"] = pd.NA
        workflow = build_diagnostic_workflow(frame, trigger(), {})
        self.assertEqual(workflow["barcode_coverage"]["status"], "partial")
        self.assertEqual(workflow["barcode_coverage"]["evaluable_noncontrol_barcodes"], 7)
        self.assertEqual(workflow["barcode_dropout"]["status"], "unresolved_or_partial_barcode_coverage")

    def test_single_sample_in_one_material_cannot_establish_absence_of_imbalance(self):
        frame = lineage()
        frame["material_type"] = ["DNA"] + ["RNA"] * 7
        workflow = build_diagnostic_workflow(frame, trigger(), {})
        self.assertEqual(workflow["barcode_imbalance"]["status"], "unresolved_insufficient_material_replication")
        self.assertEqual(workflow["barcode_dropout"]["status"], "not_detected_in_evaluable_barcodes")

    def test_detected_dropout_remains_detected_despite_partial_other_coverage(self):
        frame = lineage(["dropout", "unresolved"] + ["surviving"] * 6)
        workflow = build_diagnostic_workflow(frame, trigger(dropout=True), {})
        self.assertEqual(workflow["barcode_dropout"]["status"], "detected")
        self.assertEqual(workflow["barcode_coverage"]["status"], "partial")
        self.assert_qpm_only_barcode_reasons(workflow)

    def test_disabled_ml_still_leaves_required_part_two_explicit(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(absolute_low=True), csa(status="disabled", count=0))
        self.assert_two_low_performance_parts(workflow)
        self.assertEqual(workflow["low_performance"]["ml_status"], "disabled")
        self.assertEqual(workflow["low_performance"]["top5_feature_count"], 0)

    def test_missing_ml_does_not_invent_ranked_features_or_resolve_low_performance(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(absolute_low=True), {})
        self.assert_two_low_performance_parts(workflow)
        self.assertEqual(workflow["low_performance"]["ml_status"], "unresolved")
        self.assertEqual(workflow["low_performance"]["top5_feature_count"], 0)
        self.assertEqual(workflow["low_performance"]["causal_status"], "candidate_explanations_not_proven_causes")

    def test_five_global_features_recorded_without_claiming_a_local_cause(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(absolute_low=True), csa())
        self.assertEqual(workflow["low_performance"]["top5_feature_count"], 5)
        self.assertEqual(workflow["low_performance"]["top5_ml_policy"]["validation_scheme"], "grouped")
        self.assertIn("Global total-read model importance", workflow["low_performance"]["ml_evidence_scope"])
        self.assert_qpm_only_barcode_reasons(workflow)

    def test_unresolved_chip_read_criterion_is_not_a_pass(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(absolute_status="unresolved"), {})
        self.assertEqual(workflow["low_performance"]["status"], "unresolved_absolute_read_criterion")
        self.assertFalse(workflow["low_performance"]["triggered"])

    def test_html_escapes_status_text_and_explains_scope(self):
        workflow = build_diagnostic_workflow(lineage(), trigger(), {})
        workflow["barcode_dropout"]["status"] = '<script>alert("bad")</script>'
        fragment = diagnostic_workflow_html(workflow)
        self.assertNotIn("<script>", fragment)
        self.assertIn("&lt;script&gt;", fragment)
        self.assertIn("QPM only", fragment)
        self.assertIn("No qualifying evidence means the reason remains unresolved", fragment)

    def test_html_navigation_contains_ml_part_only_for_low_performance(self):
        normal = diagnostic_workflow_html(build_diagnostic_workflow(lineage(), trigger(dropout=True), csa()))
        low = diagnostic_workflow_html(build_diagnostic_workflow(lineage(), trigger(absolute_low=True), csa()))
        self.assertIn('href="#qpm-investigation"', normal)
        self.assertNotIn('href="#csa-low-performance-diagnostics"', normal)
        self.assertIn('href="#csa-low-performance-diagnostics"', low)


class ReportPartLayoutTests(unittest.TestCase):
    def render(self, flags, frame=None, diagnostic_status="diagnostic_clues_found"):
        frame = lineage() if frame is None else frame
        diagnostics = csa(status=diagnostic_status, count=0 if diagnostic_status == "disabled" else 5)
        workflow = build_diagnostic_workflow(frame, flags, diagnostics)
        conclusions = dict(materials={}, shared_path_reference=dict(earliest_reference_deviation=None),
            qpm_trigger=flags, report_branch="within_run_barcode_comparison" if flags["has_dropout"] or flags["has_imbalance"] else "shared_path_reference",
            causal_guardrail="Potential diagnostic evidence is not a proven cause.", baseline_comparison={},
            csa_low_performance_diagnostics=diagnostics, diagnostic_workflow=workflow)
        marker = '<section id="csa-low-performance-diagnostics"><h2>Part 2 — Total-read ML metrics</h2><p>UNIQUE_ML_PART</p></section>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure = root / "figure.png"
            figure.write_bytes(b"test-report-image")
            with patch("qpm_dropout_plugin.baseline_html_section", return_value='<div id="baseline-audit">THREE_BASELINE_AUDIT</div>'), \
                 patch("qpm_dropout_plugin.run_metadata_html", return_value='<section id="run-information">metadata</section>'), \
                 patch("qpm_dropout_plugin.csa_low_performance_html", return_value=marker) as ml:
                path = make_html_report(root, "obs_test", frame, None, pd.DataFrame(), conclusions, [], figure,
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)
                document = path.read_text()
                calls = ml.call_count
        return document, calls

    def test_barcode_only_report_does_not_render_ml_explanation_part(self):
        document, calls = self.render(trigger(dropout=True))
        self.assertEqual(calls, 0)
        self.assertIn('id="qpm-investigation"', document)
        self.assertIn("THREE_BASELINE_AUDIT", document)
        self.assertNotIn('id="csa-low-performance-diagnostics"', document)
        self.assertNotIn("UNIQUE_ML_PART", document)

    def test_low_report_has_qpm_first_and_ml_second(self):
        document, calls = self.render(trigger(absolute_low=True))
        self.assertEqual(calls, 1)
        qpm = document.index('id="qpm-investigation"')
        baseline = document.index("THREE_BASELINE_AUDIT")
        volume = document.index("ML nominal-volume signature")
        ml = document.index('id="csa-low-performance-diagnostics"')
        self.assertLess(qpm, baseline)
        self.assertLess(baseline, ml)
        self.assertLess(volume, ml)
        self.assertIn("Part 1", document)
        self.assertIn("Part 2", document)

        class SectionAncestry(HTMLParser):
            def __init__(self):
                super().__init__()
                self.stack = []
                self.parents = {}

            def handle_starttag(self, tag, attrs):
                if tag == "section":
                    identifier = dict(attrs).get("id")
                    if identifier:
                        self.parents[identifier] = tuple(self.stack)
                    self.stack.append(identifier)

            def handle_endtag(self, tag):
                if tag == "section":
                    self.stack.pop()

        parser = SectionAncestry()
        parser.feed(document)
        self.assertNotIn("qpm-investigation", parser.parents["csa-low-performance-diagnostics"])

    def test_coexisting_issues_keep_two_parts_and_the_qpm_only_barcode_guard(self):
        document, calls = self.render(trigger(dropout=True, imbalance=True, absolute_low=True))
        self.assertEqual(calls, 1)
        self.assertIn("For dropout or imbalance, potential reasons come from QPM evidence only", document)
        self.assertIn("UNIQUE_ML_PART", document)

    def test_no_trigger_keeps_qpm_audit_without_ml_part(self):
        document, calls = self.render(trigger())
        self.assertEqual(calls, 0)
        self.assertIn("THREE_BASELINE_AUDIT", document)
        self.assertNotIn("UNIQUE_ML_PART", document)

    def test_disabled_ml_still_renders_explicit_second_part_for_low_performance(self):
        document, calls = self.render(trigger(absolute_low=True), diagnostic_status="disabled")
        self.assertEqual(calls, 1)
        self.assertIn('id="csa-low-performance-diagnostics"', document)

    def test_missing_barcodes_do_not_suppress_low_performance_second_part(self):
        document, calls = self.render(trigger(absolute_low=True), frame=pd.DataFrame())
        self.assertEqual(calls, 1)
        self.assertIn("Missing barcode outcomes do not mean no dropout", document)
        self.assertIn('id="csa-low-performance-diagnostics"', document)


if __name__ == "__main__":
    unittest.main()
