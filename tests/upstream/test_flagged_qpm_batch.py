"""Tests for conservative candidate filtering and missing-barcode handling."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_flagged_qpm_batch import CandidateIndex, CANDIDATE_KEYS, scope_result, safe
from qpm_dropout_plugin import build_barcode_lineage, DEFAULT_CONFIG
from qpm_diagnostic_extensions import diagnostic_trigger


class FlaggedBatchTests(unittest.TestCase):
    def profiles(self):
        rows=[]
        for index in range(7):
            row={key:"a" for key in CANDIDATE_KEYS}
            row.update(observation_id="target" if index<2 else "reference", operation_id=f"op{index}",
                       programmed_volume_ul=10., reference_eligible=index not in [1,4], command_context_status="command_verified")
            if index in [1,3]:
                row["liquid_class_fingerprint"]="other"
            if index in [5,6]:
                row["programmed_volume_ul"]=20.
            rows.append(row)
        return pd.DataFrame(rows),dict(trace_quality_exclusions=[dict(observation_id="target")])

    def test_candidate_filter_preserves_all_targets_and_all_compatible_history(self):
        profiles=self.profiles()
        frame,metadata=CandidateIndex(profiles).view("target")
        self.assertEqual(set(frame.operation_id),{"op0","op1","op2","op3","op4"})
        self.assertEqual(len(frame[frame.observation_id.eq("target")]),2)
        self.assertEqual(metadata["trace_quality_exclusions"],profiles[1]["trace_quality_exclusions"])
        self.assertTrue(metadata["batch_candidate_prefilter"]["full_reference_matching_and_exclusions_unchanged"])

    def test_unqualified_history_is_not_accidentally_promoted_or_reclassified(self):
        frame,_=CandidateIndex(self.profiles()).view("target")
        self.assertFalse(frame.loc[frame.operation_id.eq("op4"),"reference_eligible"].iloc[0])

    def test_missing_target_returns_empty_view_with_quality_provenance(self):
        frame,metadata=CandidateIndex(self.profiles()).view("absent")
        self.assertEqual(len(frame),0)
        self.assertIn("trace_quality_exclusions",metadata)

    def test_qualified_prefilter_removes_only_ineligible_history_not_targets(self):
        frame,note=CandidateIndex(self.profiles(),qualified_references_only=True).view("target")
        self.assertEqual(set(frame.operation_id),{"op0","op1","op2","op3"})
        self.assertIn("op1",set(frame.operation_id))  # An ineligible target is still diagnosed.
        self.assertTrue(note["batch_candidate_prefilter"]["historically_qualified_candidates_only"])

    def test_view_modifications_do_not_mutate_another_target_or_parent(self):
        original=self.profiles(); index=CandidateIndex(original)
        frame,_=index.view("target")
        frame.loc[:,"liquid_class_fingerprint"]="changed"
        self.assertNotIn("changed",set(original[0].liquid_class_fingerprint))
        self.assertNotIn("changed",set(index.view("reference")[0].liquid_class_fingerprint))

    def test_pending_and_failed_tasks_have_no_invented_failure_step(self):
        task=dict(observation_id="o",physical_run_id="p",flagged=True)
        pending=scope_result(task,{})
        self.assertEqual(pending["diagnostic_status"],"pending")
        failed=scope_result(task,dict(status="failed",error="missing input"))
        self.assertEqual(failed["diagnostic_status"],"processing_failed")
        self.assertIsNone(failed["earliest_operation_index"])

    def test_json_safe_does_not_retain_nan(self):
        self.assertEqual(json.dumps(safe(dict(a=np.nan,b=np.int64(3),c=pd.NA)),allow_nan=False),'{"a": null, "b": 3, "c": null}')

    def test_missing_barcode_outcomes_do_not_block_low_read_qpm_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"qpm_failure_examples").mkdir()
            pd.DataFrame([dict(observation_id="other",archive_id="a",is_control=False,read_count=100,material_type="DNA",planned_order=1)]).to_csv(root/"qpm_failure_examples/dna_rna_barcode_counts.csv",index=False)
            pd.DataFrame([dict(observation_id="target",archive_id="a")]).to_parquet(root/"csa_ml_dataset.parquet",index=False)
            archive=root/"dummy.CSA";archive.write_bytes(b"not an archive")
            pd.DataFrame([dict(archive_id="a",local_path=str(archive))]).to_csv(root/"csa_transfer_manifest.csv",index=False)
            lineage,warnings,_,_=build_barcode_lineage(root,"target",DEFAULT_CONFIG)
            self.assertTrue(lineage.empty)
            self.assertTrue(any("Barcode outcomes are unavailable" in value for value in warnings))
            outcome=pd.DataFrame(columns=["status"])
            trigger=diagnostic_trigger(lineage,outcome,DEFAULT_CONFIG,dict(low_performance_trigger=True))
            self.assertTrue(trigger["run_qpm"])
            self.assertFalse(trigger["has_dropout"])
            self.assertFalse(trigger["has_imbalance"])


if __name__ == "__main__":
    unittest.main()
