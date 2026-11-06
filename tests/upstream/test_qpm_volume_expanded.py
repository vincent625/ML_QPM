"""Expanded-bank waveform eligibility and bounded sampling regression tests."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa

from audit_qpm_volume_scope import audit_scope
from retrain_qpm_volume_models import load_audited, select_cohort
from qpm_nominal_volume_inference import NominalVolumeRecognizer, predict_nominal_volume


class ExpandedVolumeTests(unittest.TestCase):
    def test_er72_and_unresolved_quality_never_enter_training(self):
        waves = pd.DataFrame([
            dict(observation_id=f"o{i}", physical_run_id=f"r{i}", instrument_id="instrument", archive_id=f"a{i}",
                 chip_lane="1", run_start_time="2026-01-01", operation_id=f"op{i}", operation_index=1,
                 operation="aspirate", pipette="PIP2", programmed_volume_ul=10., qpm_available=True,
                 qpm_profile=np.arange(32, dtype=float), pressure_unit="raw_sensor_counts", liquid_class_index=999,
                 workflow_stage="prep", process_step="step", loc="A1") for i in range(3)])
        context = pd.DataFrame([dict(archive_id=f"a{i}", operation_index=1, pipette="PIP2", command_operation="aspirate",
                                     command_volume_ul=10., command_context_status="command_verified", command_liquid_class_index=49)
                                for i in range(3)])
        quality = pd.DataFrame([
            dict(archive_id="a0", operation_index=1, pipette="PIP2", qpm_waveform_eligible=True, qpm_has_er72=False),
            dict(archive_id="a1", operation_index=1, pipette="PIP2", qpm_waveform_eligible=False, qpm_has_er72=True)])
        def parquet(path):
            return quality if Path(path).name == "qpm_trace_quality.parquet" else context
        with patch("retrain_qpm_volume_models.pq.read_table", return_value=pa.Table.from_pandas(waves)), patch("retrain_qpm_volume_models.pd.read_parquet", side_effect=parquet):
            result, audit = load_audited(Path("/unused"))
        self.assertEqual(result.operation_id.tolist(), ["op0"])
        self.assertEqual(result.liquid_class_index.tolist(), [49])
        self.assertEqual(audit["quality_excluded_operations"], 2)
        self.assertEqual(audit["er72_operations_excluded"], 1)

    def population(self):
        rows = []
        for run in range(120):
            for volume in [5., 10., 20.]:
                for repeat in range(3):
                    rows.append(dict(physical_run_id=f"r{run}", instrument_id=f"i{run % 3}", observation_id=f"o{run}",
                                     operation_id=f"op{run}_{volume}_{repeat}", liquid_class_index=49, operation="aspirate", pipette="PIP2",
                                     liquid_class_fingerprint="same_definition", liquid_class_name="same_liquid",
                                     programmed_volume_ul=volume, qpm_profile=np.arange(32) * volume,
                                     run_start_time=pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=run)))
        return pd.DataFrame(rows)

    def test_sampling_has_run_volume_cap_and_does_not_screen_outcomes(self):
        population = self.population()
        first, _, info = select_cohort(population, 49, "aspirate", 1, 30)
        second, _, _ = select_cohort(population.iloc[::-1], 49, "aspirate", 1, 30)
        self.assertEqual(set(first.operation_id), set(second.operation_id))
        self.assertEqual(first.physical_run_id.nunique(), 30)
        self.assertTrue(first.groupby(["physical_run_id", "programmed_volume_ul"]).size().le(1).all())
        self.assertEqual(first.groupby("instrument_id").physical_run_id.nunique().tolist(), [10, 10, 10])
        self.assertFalse(info["health_screen_applied"])

    def test_coverage_never_licenses_unobserved_31_ul(self):
        population = self.population()
        tasks, volumes = audit_scope(population)
        self.assertTrue(tasks.eligible_three_or_more_levels.all())
        self.assertNotIn(31., volumes.programmed_volume_ul.tolist())

    def test_nominal_inference_refuses_er72_and_unknown_quality(self):
        for operation, expected in [({"qpm_has_er72": True}, "Er72_operation_or_retry_excluded"),
                                    ({"qpm_has_er72": False}, "trace_quality_unverified")]:
            result = predict_nominal_volume(Path("/unused"), operation)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], expected)
            self.assertIsNone(result["measured_transfer_deficit_ul"])
            self.assertFalse(result["actual_volume_prediction_validated"])

    def test_nominal_scope_requires_volume_geometry_settings_cooccurrence(self):
        task = "lc49_aspirate"
        context_fields = ["pressure_unit", "container_geometry_fingerprint", "deck_geometry_fingerprint", "command_settings_fingerprint"]
        operation = dict(qpm_has_er72=False, qpm_waveform_eligible=True, qpm_trace_quality_status="checked_no_er72",
                         command_context_status="command_verified", liquid_class_index=49, operation="aspirate",
                         qpm_profile=np.arange(32), liquid_class_fingerprint="lc", pipette="PIP2", programmed_volume_ul=10.,
                         pressure_unit="counts", container_geometry_fingerprint="cg", deck_geometry_fingerprint="dg",
                         command_settings_fingerprint="settings", physical_run_id="new_run")
        performance = pd.DataFrame([dict(task_id=task, kind="regression", scheme=scheme, model=model, mae_ul=error)
                                    for scheme in ["grouped_run", "unseen_instrument", "chronological"]
                                    for model, error in [("Nested-selected procedure", 1.), ("Median baseline", 3.)]])
        payload = dict(cohort=dict(pipette="PIP2", liquid_class_fingerprint="lc", volumes_ul=[5., 10.]),
                       estimator=SimpleNamespace(predict=lambda x: np.array([5.])), model_name="test_model")
        artifacts = dict(log=dict(models=[dict(task_id=task, kind="regression", path="unused")]),
                         payloads={(task, "regression"): payload}, run_ids={task: {"training_run"}},
                         context_fields=context_fields, contexts={task: {("counts", "cg", "dg", "settings", 5.)}}, performance=performance)
        result = predict_nominal_volume(Path("/unused"), operation, _artifacts=artifacts)
        self.assertEqual(result["reason"], "nominal_volume_unit_geometry_or_command_settings_combination_outside_training_scope")
        operation["programmed_volume_ul"] = 5.
        result = predict_nominal_volume(Path("/unused"), operation, _artifacts=artifacts)
        self.assertEqual(result["status"], "nominal_recognition_only")
        self.assertEqual(result["nominal_equivalent_volume_ul"], 5.)
        self.assertIsNone(result["measured_transfer_deficit_ul"])
        operation["physical_run_id"] = "training_run"
        result = predict_nominal_volume(Path("/unused"), operation, _artifacts=artifacts)
        self.assertEqual(result["reason"], "target_physical_run_present_in_model_training_use_heldout_predictions")
        operation["physical_run_id"] = "new_run"
        for invalid in [np.nan, np.inf, -np.inf]:
            artifacts["performance"].loc[artifacts["performance"].scheme.eq("grouped_run"), "mae_ul"] = invalid
            result = predict_nominal_volume(Path("/unused"), operation, _artifacts=artifacts)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], "nonfinite_grouped_run_validation")

    def test_cached_study_refuses_modified_performance_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = pd.DataFrame([dict(task_id="lc49_aspirate", physical_run_id="run", pressure_unit="counts",
                                          container_geometry_fingerprint="cg", deck_geometry_fingerprint="dg",
                                          command_settings_fingerprint="settings", programmed_volume_ul=5.)])
            training.to_parquet(root / "training_cohort.parquet", index=False)
            pd.DataFrame([dict(task_id="lc49_aspirate", mae_ul=1.)]).to_csv(root / "model_performance.csv", index=False)
            (root / "analysis_run_log.json").write_text(json.dumps(dict(models=[])))
            names = ["analysis_run_log.json", "training_cohort.parquet", "model_performance.csv"]
            pins = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
            (root / "artifact_validation.json").write_text(json.dumps(dict(status="passed", artifact_sha256=pins)))
            first = NominalVolumeRecognizer(root)
            self.assertEqual(first.artifacts["performance"].mae_ul.iloc[0], 1.)
            pd.DataFrame([dict(task_id="lc49_aspirate", mae_ul=99999.)]).to_csv(root / "model_performance.csv", index=False)
            with self.assertRaisesRegex(ValueError, "artifact_hash_mismatch"):
                NominalVolumeRecognizer(root)


if __name__ == "__main__":
    unittest.main()
