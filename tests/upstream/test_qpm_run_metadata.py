"""Regression checks for run labels without changes to diagnostic decisions."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from qpm_run_metadata import build_run_metadata, run_metadata_html


class RunMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def inputs(self, changes=None, inventory=None):
        row = dict(observation_id="obs_target", archive_id="a1", physical_run_id="run_target",
                   instrument_id="inst_hash", hostname_for_report="gnxs-0073",
                   run_start_time="2026-10-19T16:33:57Z", chip_lane="1_2_3_4",
                   analysis_lane_scope="combined_multilane")
        row.update({"ion_params_00.chip_info_chip_type": "GX8",
                    "global.analysis_application_type": "RDX Ampliseq Germline Template",
                    "global.analysis_regions_file": "/private/server/AssayImport/Assay-v1/workflow-input/Bed/Panel.bed",
                    "planned_run.object_script_name": "workflow_GX8_v1"})
        row.update(changes or {})
        pd.DataFrame([row]).to_parquet(self.root / "csa_ml_dataset.parquet", index=False)
        if inventory is not None:
            pd.DataFrame(inventory).to_csv(self.root / "csa_inventory.csv", index=False)
        return build_run_metadata(self.root, "obs_target")

    def test_exact_recorded_fields_and_provenance(self):
        result = self.inputs(inventory=[dict(archive_id="a1", hostname="gnxs-0073", run_name="Recorded run")])
        self.assertEqual(result["run_date"], "2026-10-19")
        self.assertEqual(result["instrument"], "gnxs-0073")
        self.assertEqual(result["chip_type"], "GX8")
        self.assertEqual(result["chip_lane"], "1_2_3_4")
        self.assertEqual(result["assay_panel"], "Assay-v1")
        self.assertEqual(result["run_name"], "Recorded run")
        self.assertIn("run_start_time", result["sources"]["run_date"])
        self.assertNotIn("/private/server", json.dumps(result))

    def test_panel_filename_fallback_not_guessed_assay(self):
        result = self.inputs({"global.analysis_regions_file": "/server/TargetRegion/OMA-DNA-v3D5.bed"})
        self.assertEqual(result["assay_panel"], "OMA-DNA-v3D5.bed")
        self.assertEqual(result["assay_panel_basis"], "recorded panel filename")
        self.assertNotEqual(result["assay_panel"], result["workflow"])

    def test_missing_metadata_does_not_become_nan_or_workflow_as_assay(self):
        result = self.inputs({"hostname_for_report": None, "global.analysis_application_type": "nan",
                              "global.analysis_regions_file": None, "run_start_time": pd.NA,
                              "ion_params_00.chip_info_chip_type": None})
        for field in ["instrument", "assay_type", "assay_panel", "run_date", "chip_type"]:
            self.assertEqual(result[field], "Not recorded")
        self.assertEqual(result["workflow"], "workflow_GX8_v1")
        self.assertNotIn("inst_hash", json.dumps(result))

    def test_instrument_fallback_only_uses_matching_archive(self):
        result = self.inputs({"hostname_for_report": None}, [
            dict(archive_id="a2", hostname="wrong-instrument", run_name="wrong-run"),
            dict(archive_id="a1", hostname="gnxs-0018.example.internal", run_name="right-run"),
            dict(archive_id="a1", hostname="gnxs-0018.example.internal", run_name="right-run")])
        self.assertEqual(result["instrument"], "gnxs-0018.example.internal")
        self.assertEqual(result["run_name"], "right-run")

    def test_conflicting_inventory_does_not_pick_first(self):
        result = self.inputs({"hostname_for_report": None}, [
            dict(archive_id="a1", hostname="a", run_name="one"),
            dict(archive_id="a1", hostname="b", run_name="two")])
        self.assertEqual(result["instrument"], "Not recorded")
        self.assertEqual(result["run_name"], "Not recorded")
        self.assertEqual(len(result["warnings"]), 2)

    def test_html_escapes_source_values_and_contains_all_requested_fields(self):
        result = self.inputs({"hostname_for_report": '<script>alert("x")</script>'})
        fragment = run_metadata_html(result)
        self.assertNotIn("<script>", fragment)
        self.assertIn("&lt;script&gt;", fragment)
        for field in ["run_date", "instrument", "assay_type", "assay_panel", "chip_type", "chip_lane"]:
            self.assertIn(f'data-run-field="{field}"', fragment)
        self.assertIn("run_metadata.json", fragment)

    def test_invalid_date_is_unresolved_and_warned(self):
        result = self.inputs({"run_start_time": "unparseable"})
        self.assertEqual(result["run_date"], "Not recorded")
        self.assertEqual(len(result["warnings"]), 1)

    def test_dna_rna_package_and_panel_identities_deduplicate(self):
        result = self.inputs({"global.analysis_fusions_file": "/s/AssayImport/Assay-v1/workflow-input/Fusion/RNA.bed"})
        self.assertEqual(result["assay_panel"], "Assay-v1")
        self.assertEqual(result["panel_files"], "Panel.bed | RNA.bed")

    def test_ambiguous_target_rejected(self):
        self.inputs()
        row = pd.read_parquet(self.root / "csa_ml_dataset.parquet")
        pd.concat([row, row]).to_parquet(self.root / "csa_ml_dataset.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_run_metadata(self.root, "obs_target")


if __name__ == "__main__":
    unittest.main()
