"""Read-only integration regression checks for the nominal-volume registry branch."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from qpm_diagnostic_extensions import add_volume_proxies


class NominalPluginIntegrationTests(unittest.TestCase):
    def registry(self, root):
        (root / "qpm_nominal_volume_model.json").write_text(json.dumps(dict(
            study_path="/trusted/local/study", analysis_run_log_sha256="pinned_log",
            artifact_validation_sha256="pinned_validation")))

    def operations(self, count=1, profile_column="observed_profile"):
        rows = []
        for index in range(count):
            row = dict(operation_id=f"op{index}", physical_run_id="new_physical_run", operation="aspirate",
                       pipette="PIP2", liquid_class_index=49, programmed_volume_ul=10.,
                       command_context_status="command_verified", qpm_has_er72=False,
                       qpm_waveform_eligible=True, qpm_trace_quality_status="checked_no_er72")
            row[profile_column] = np.arange(32, dtype=float)
            rows.append(row)
        return pd.DataFrame(rows)

    def nominal(self):
        return dict(status="nominal_recognition_only", nominal_equivalent_volume_ul=6., model_name="mock_nominal_model",
                    actual_volume_prediction_validated=False, measured_transfer_deficit_ul=None,
                    reason="not_an_independent_measurement_of_delivered_volume_or_transfer_error")

    def test_nominal_signature_is_never_labeled_measured_transfer_deficit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.registry(root)
            recognizer = MagicMock()
            recognizer.predict.return_value = self.nominal()
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer", return_value=recognizer) as constructor:
                frame, metadata = add_volume_proxies(root, self.operations(), "target", {"enable_volume_proxy": True})
            constructor.assert_called_once_with(Path("/trusted/local/study"), expected_log_sha256="pinned_log",
                                                expected_validation_sha256="pinned_validation")
            np.testing.assert_array_equal(recognizer.predict.call_args.args[0]["qpm_profile"], np.arange(32))
            self.assertEqual(frame.waveform_equivalent_volume_ul.iloc[0], 6.)
            self.assertEqual(frame.equivalent_volume_delta_ul.iloc[0], -4.)
            self.assertEqual(frame.volume_proxy_direction.iloc[0], "lower_nominal_signature")
            self.assertEqual(frame.delivered_volume_status.iloc[0], "not_identifiable_without_measured_volume_calibration")
            self.assertFalse(metadata["actual_delivered_volume_validated"])
            self.assertEqual(metadata["predicted_operations"], 1)

    def test_raw_qpm_array_profile_is_forwarded_without_observed_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.registry(root)
            recognizer = MagicMock()
            recognizer.predict.return_value = self.nominal()
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer", return_value=recognizer):
                frame, _ = add_volume_proxies(root, self.operations(profile_column="qpm_profile"), "target", {"enable_volume_proxy": True})
            self.assertTrue(frame.volume_proxy_status.str.startswith("nominal_recognition_only").all())
            np.testing.assert_array_equal(recognizer.predict.call_args.args[0]["qpm_profile"], np.arange(32))

    def test_out_of_scope_liquid_class_abstains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.registry(root)
            recognizer = MagicMock()
            recognizer.predict.return_value = dict(status="unavailable", reason="liquid_class_operation_outside_validated_study")
            operations = self.operations()
            operations["liquid_class_index"] = 47
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer", return_value=recognizer):
                frame, metadata = add_volume_proxies(root, operations, "target", {"enable_volume_proxy": True})
            self.assertTrue(frame.waveform_equivalent_volume_ul.isna().all())
            self.assertTrue(frame.equivalent_volume_delta_ul.isna().all())
            self.assertEqual(frame.volume_proxy_model.iloc[0], "none")
            self.assertEqual(metadata["predicted_operations"], 0)

    def test_invalid_registry_clears_existing_values_and_never_uses_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.registry(root)
            operations = self.operations()
            operations["waveform_equivalent_volume_ul"] = 999.
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer", side_effect=ValueError("registry_pin_mismatch")), patch("qpm_diagnostic_extensions._dtw_predict") as legacy:
                frame, metadata = add_volume_proxies(root, operations, "target", {"enable_volume_proxy": True})
            legacy.assert_not_called()
            self.assertTrue(frame.waveform_equivalent_volume_ul.isna().all())
            self.assertTrue(frame.equivalent_volume_delta_ul.isna().all())
            self.assertTrue(frame.volume_proxy_status.eq("withheld_invalid_nominal_model_registry").all())
            self.assertEqual(metadata["registry_error"], "ValueError")

    def test_later_exception_clears_model_names_as_well_as_earlier_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.registry(root)
            recognizer = MagicMock()
            recognizer.predict.side_effect = [self.nominal(), ValueError("late_validation_error")]
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer", return_value=recognizer):
                frame, _ = add_volume_proxies(root, self.operations(2), "target", {"enable_volume_proxy": True})
            self.assertTrue(frame.waveform_equivalent_volume_ul.isna().all())
            self.assertTrue(frame.volume_proxy_model.eq("none").all())

    def test_malformed_serialized_waveform_fails_closed_instead_of_aborting_plugin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.registry(root)
            operations = self.operations()
            operations["observed_profile"] = "[broken waveform"
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer"):
                frame, _ = add_volume_proxies(root, operations, "target", {"enable_volume_proxy": True})
            self.assertTrue(frame.waveform_equivalent_volume_ul.isna().all())


if __name__ == "__main__":
    unittest.main()
