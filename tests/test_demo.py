"""The offline toy example must preserve the real core's safety rules."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_demo", ROOT/"examples/run_synthetic_demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class SyntheticDemoTests(unittest.TestCase):
    def test_seed_reproduces_all_synthetic_waveforms(self):
        _, left_labels, left = demo.build_case("replicated")
        _, right_labels, right = demo.build_case("replicated")
        pd.testing.assert_frame_equal(left_labels, right_labels)
        np.testing.assert_array_equal(np.vstack(left.observed_profile), np.vstack(right.observed_profile))

    def test_replicated_tests_are_calculated_and_error_group_excluded(self):
        _, labels, mapped, tests = demo.analyze_case("replicated")
        self.assertEqual(int(labels.path_status.eq("failed").sum()), 4)
        self.assertEqual(int(mapped.qpm_has_er72.sum()), 8)
        self.assertNotIn("synthetic_excluded_error_operation", tests.group_id.tolist())
        self.assertTrue(tests.eligible_internal_comparison.all())
        self.assertTrue(tests.permutation_p.notna().all())
        first = tests.loc[tests.evidence_role.eq("initial_candidate")]
        self.assertEqual(first.first_operation_index.tolist(), [100])
        self.assertTrue(tests.loc[tests.first_operation_index.eq(10), "evidence_role"].ne("initial_candidate").all())

    def test_single_affected_path_has_no_invented_statistics(self):
        _, labels, _, tests = demo.analyze_case("sparse")
        self.assertEqual(int(labels.path_status.eq("failed").sum()), 1)
        self.assertFalse(tests.eligible_internal_comparison.any())
        self.assertTrue(tests.failed_path_deviation_auc.isna().all())
        self.assertTrue(tests.permutation_p.isna().all())
        self.assertTrue(tests.bh_fdr_q.isna().all())

    def test_invalid_scenario_fails(self):
        with self.assertRaises(ValueError):
            demo.build_case("not_a_scenario")

    def test_report_rendering_keeps_history_unavailable_and_demo_notice_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = demo.write_case(root, "sparse")
            report = (root/"report.html").read_text()
            summary = pd.read_csv(root/"qpm_baseline_comparison_summary.csv")
            self.assertIn(demo.NOTICE, report)
            self.assertIn("not a diagnostic accuracy estimate", report)
            self.assertEqual(result["historical_reference_runs"], 0)
            self.assertFalse(result["scientific_validation"])
            self.assertTrue(summary.reference_physical_runs.eq(0).all())
            self.assertTrue(all(row["evidence_role"] == "descriptive_insufficient_replication"
                                for row in result["selected_same_run_events"]))
            self.assertTrue((root/"qpm_same_run_curves.html").is_file())

    def test_unrelated_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            protected = path/"user_data.txt"
            protected.write_text("preserve me")
            with self.assertRaises(ValueError):
                demo.generate_demo(path)
            self.assertEqual(protected.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
