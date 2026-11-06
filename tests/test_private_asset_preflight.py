"""Integrity preflight must not trust traversal or changed private artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("asset_preflight", ROOT/"tools/check_private_assets.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PrivateAssetPreflightTests(unittest.TestCase):
    def test_parent_and_absolute_manifest_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ["../outside", str(Path(directory).parent/"outside")]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    preflight.safe_member(directory, name)

    def test_external_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root/"bank"
            bank.mkdir()
            outside = root/"not_in_bank.txt"
            outside.write_text("not an asset")
            (bank/"alias").symlink_to(outside)
            with self.assertRaises(ValueError):
                preflight.safe_member(bank, "alias")

    def test_manifest_hash_check_fails_after_same_size_change(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory)
            asset = bank/"table.csv"
            asset.write_bytes(b"valid")
            manifest = dict(validation_status="passed", artifacts={"table.csv": dict(
                size_bytes=5, sha256=hashlib.sha256(b"valid").hexdigest())})
            (bank/"reference_bank_manifest.json").write_text(json.dumps(manifest))
            _, before = preflight.check_bank(preflight.Audit(), bank)
            self.assertTrue(before["ready"])
            asset.write_bytes(b"other")
            audit = preflight.Audit()
            _, after = preflight.check_bank(audit, bank)
            self.assertFalse(after["ready"])
            self.assertEqual(audit.errors["bank_artifact_hash_mismatch"], 1)

    def test_default_summary_excludes_detailed_private_paths(self):
        audit = preflight.Audit()
        audit.issue("raw_archive_missing_or_unreadable", path=Path("/private/customer/archive.txz"), component="archives")
        safe = json.dumps(dict(error_counts=dict(audit.errors)))
        self.assertNotIn("customer", safe)
        self.assertNotIn("archive.txz", safe)
        self.assertIn("sensitive_local_path", audit.details[0])


if __name__ == "__main__":
    unittest.main()
