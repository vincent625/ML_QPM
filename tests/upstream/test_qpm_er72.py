"""Er72 cannot enter scores, templates, retries, or evidence selection."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from qpm_trace_quality import parse_trace_quality, eligible_waveforms, er72_mask
from qpm_baseline_comparison import _display_rows, select_path_reference_display
from qpm_dropout_plugin import select_shared_display
from qpm_internal_comparison import collapse_paths, internal_operation_frame
from qpm_waveform_reference import scan_references
from qpm_cross_instrument_reference import scan_cross_instrument_fallback
from test_qpm_parallel_baselines import examples


class TraceStatusTests(unittest.TestCase):
    def parse(self, lines):
        return parse_trace_quality(BytesIO("\n".join(lines).encode()))

    def test_mixed_case_error_is_sticky_across_successful_retry(self):
        rows, stats = self.parse([
            "in PIP.Dispense id=2 vol=31",
            "PipSndV1 id:2,pipCmd:GDid0010di00310",
            "pipette response, id = 2, ret = GDid0010yl0524eR72",
            "PipSndV1 id:2,pipCmd:GDid0011di00310",
            "pipette response, id = 2, ret = GDid0011er00",
        ])
        self.assertTrue(rows[0]["qpm_has_er72"])
        self.assertFalse(rows[0]["qpm_waveform_eligible"])
        self.assertEqual(rows[0]["qpm_er72_command_ids"], "GD:10")
        self.assertEqual(stats["unmatched_er72_responses"], 0)

    def test_delayed_reply_uses_sent_command_not_latest_operation(self):
        rows, _ = self.parse([
            "in PIP.Aspirate id=2 vol=10", "PipSndV1 id:2,pipCmd:GAid0010ai00100",
            "in PIP.Dispense id=2 vol=10", "PipSndV1 id:2,pipCmd:GDid0011di00100",
            "pipette response, id = 2, ret = GAid0010er72",
        ])
        self.assertTrue(rows[0]["qpm_has_er72"])
        self.assertFalse(rows[1]["qpm_has_er72"])

    def test_pipette_and_command_id_are_both_required(self):
        rows, _ = self.parse([
            "in PIP.Aspirate id=1 vol=10", "PipSndV1 id:1,pipCmd:GAid0010ai00100",
            "in PIP.Aspirate id=2 vol=10", "PipSndV1 id:2,pipCmd:GAid0010ai00100",
            "pipette response, id = 2, ret = GAid0010er72",
        ])
        self.assertFalse(rows[0]["qpm_has_er72"])
        self.assertTrue(rows[1]["qpm_has_er72"])

    def test_unassigned_structured_error_quarantines_archive(self):
        rows, stats = self.parse(["in PIP.Aspirate id=2 vol=10",
                                  "pipette response, id = 2, ret = GAid9999er72"])
        self.assertEqual(stats["unmatched_er72_responses"], 1)
        self.assertFalse(rows[0]["qpm_waveform_eligible"])
        self.assertEqual(rows[0]["qpm_trace_quality_status"], "unresolved_er72_association")

    def test_er720_or_retry_echo_is_not_a_structured_er72_reply(self):
        rows, stats = self.parse(["in PIP.Aspirate id=2 vol=10", "PipSndV1 id:2,pipCmd:GAid0010ai00100",
                                  "pipette response, id = 2, ret = GAid0010er720",
                                  "pip 2. Retrying command after er72"])
        self.assertEqual(stats["er72_responses"], 0)
        self.assertTrue(rows[0]["qpm_waveform_eligible"])

    def test_future_raw_parser_preserves_error_on_finished_waveform(self):
        from train_multimodal_csa import parse_qpm_stream
        lines = b"\n".join([b"in PIP.Aspirate id=2 vol=10", b"PipSndV1 id:2,pipCmd:GAid0010ai00100",
                             b"pipette recv id = 2, ret = RXid0011rx00010 1 2 3 4",
                             b"pipette response, id = 2, ret = GAid0010er72"])
        rows = parse_qpm_stream(BytesIO(lines), dict(archive_id="test", run_start_time="2026-10-01"))
        self.assertTrue(rows[0]["qpm_available"])
        self.assertFalse(rows[0]["qpm_waveform_eligible"])
        self.assertTrue(rows[0]["qpm_has_er72"])

    def test_scoring_guards_remove_error_and_unverified_status(self):
        frame = pd.DataFrame(dict(qpm_has_er72=[True, False, False], qpm_waveform_eligible=[False, False, True]))
        self.assertEqual(eligible_waveforms(frame).index.tolist(), [2])
        self.assertEqual(er72_mask(frame).tolist(), [True, False, False])

    def test_checked_no_er72_is_a_valid_trace_not_a_substring_error(self):
        rows, _ = self.parse(["in PIP.Aspirate id=2 vol=10", "PipSndV1 id:2,pipCmd:GAid0010ai00100",
                              "pipette response, id = 2, ret = GAid0010er00"])
        frame = pd.DataFrame(rows)
        self.assertFalse(er72_mask(frame).any())
        self.assertEqual(len(eligible_waveforms(frame)), 1)

    def test_er72_retry_cannot_bias_collapsed_path_waveform(self):
        curve = np.arange(32, dtype=float)
        rows = pd.DataFrame([dict(operation_id=f"id{i}", planned_order=3, location="pcr_c1", path_status="failed",
                                  observed_profile=json.dumps((curve * scale).tolist()), qpm_has_er72=bad)
                             for i, scale, bad in [(1, 1, False), (2, 1000, True)]])
        np.testing.assert_array_equal(collapse_paths(rows).observed_profile.iloc[0], curve)

    def test_er72_references_cannot_satisfy_minimum_physical_runs(self):
        frame = examples()
        frame["qpm_has_er72"] = False
        frame.loc[frame.observation_id.isin(["same_0", "other_0"]), "qpm_has_er72"] = True
        config = dict(minimum_reference_physical_runs=8, waveform_reference_policy_overrides={})
        with tempfile.TemporaryDirectory() as temp:
            same, coverage, _, _ = scan_references(Path(temp), "target", config, (frame, dict(parser_sha256="test", archives=[])))
            cross, _, _, _ = scan_cross_instrument_fallback(Path(temp), "target", {"operation_target"}, config,
                                                           (frame, {}), coverage, independent=True)
        self.assertTrue(same.empty)
        self.assertTrue(cross.empty)
        self.assertEqual(int(coverage.maximum_available_reference_runs.iloc[0]), 7)

    def test_er72_target_is_not_scored_as_internal_operation(self):
        frame = examples()
        frame["qpm_has_er72"] = frame.observation_id.eq("target")
        self.assertTrue(internal_operation_frame(frame, "target", pd.DataFrame(), pd.DataFrame()).empty)

    def test_er72_cannot_be_earliest_or_later_reference_evidence(self):
        rows = pd.DataFrame([dict(operation_id=f"op{i}", operation_index=i, process_step=f"step{i}",
                                  planned_order=3, qpm_has_er72=(i in [10, 30]),
                                  reference_positive=True, evidence_role="later_reference_anomaly")
                             for i in [10, 20, 30, 40, 50, 60, 70]])
        for selection in [select_shared_display(rows), select_path_reference_display(rows),
                          _display_rows(pd.DataFrame(), rows, "same_instrument")]:
            self.assertEqual([row.operation_index for row in selection], [20, 40, 50, 60, 70])


if __name__ == "__main__":
    unittest.main()
