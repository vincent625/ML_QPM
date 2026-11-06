"""Fail-closed tests for pinning a validated nominal study into a new bank."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from build_qualified_reference_bank import nominal_registry, register_validated_nominal_study
from build_expanded_reference_bank import sha256


class RegistryTests(unittest.TestCase):
    def bank(self, root, status="passed"):
        bank=root/"bank";bank.mkdir()
        data=bank/"qualification.csv";data.write_text("eligible\ntrue\n")
        manifest=dict(validation_status=status, artifacts={data.name:dict(size_bytes=data.stat().st_size,sha256=sha256(data))})
        (bank/"reference_bank_manifest.json").write_text(json.dumps(manifest))
        return bank,data

    def test_reject_unvalidated_bank(self):
        with tempfile.TemporaryDirectory() as td:
            bank,_=self.bank(Path(td),"failed")
            with self.assertRaisesRegex(ValueError,"validated bank"):
                register_validated_nominal_study(bank,Path(td)/"study")
            self.assertFalse((bank/"qpm_nominal_volume_model.json").exists())

    def test_reject_changed_bank_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            bank,data=self.bank(Path(td));data.write_text("eligible\nfalse\n")
            with self.assertRaisesRegex(ValueError,"artifact changed"):
                register_validated_nominal_study(bank,Path(td)/"study")

    def test_registration_preserves_reference_data(self):
        with tempfile.TemporaryDirectory() as td:
            bank,data=self.bank(Path(td));before=sha256(data)
            record=dict(study_path=str(Path(td)/"study"),actual_volume_prediction_validated=False,
                        label_basis="nominal_command_not_measured_delivery")
            with patch("build_qualified_reference_bank.nominal_registry",return_value=record):
                register_validated_nominal_study(bank,Path(td)/"study")
            self.assertEqual(before,sha256(data))
            manifest=json.loads((bank/"reference_bank_manifest.json").read_text())
            self.assertEqual(manifest["artifacts"]["qpm_nominal_volume_model.json"]["sha256"],sha256(bank/"qpm_nominal_volume_model.json"))
            self.assertTrue(json.loads((bank/"nominal_model_registration.json").read_text())["reference_tables_unchanged"])
            self.assertFalse(json.loads((bank/"qpm_nominal_volume_model.json").read_text())["actual_volume_prediction_validated"])

    def test_reject_unvalidated_study(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"artifact_validation.json").write_text(json.dumps(dict(status="failed")))
            with self.assertRaisesRegex(ValueError,"artifact validation"):
                nominal_registry(root)

    def test_reject_modified_model_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"artifact_validation.json").write_text(json.dumps(dict(status="passed")))
            (root/"analysis_run_log.json").write_text("{}")
            (root/"model.joblib").write_bytes(b"changed")
            mock=SimpleNamespace(artifacts={"log":{"models":[dict(path="model.joblib",sha256="0"*64)]}})
            with patch("qpm_nominal_volume_inference.NominalVolumeRecognizer",return_value=mock):
                with self.assertRaisesRegex(ValueError,"payload changed"):
                    nominal_registry(root)


if __name__=="__main__":
    unittest.main()
