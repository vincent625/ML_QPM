"""Small packaging guards; no production data, network or model loading."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from validate_package import files_for_upload, forbidden_upload_file, ignored, markdown_links, scan_text
from run_tests import source_snapshot


class PackagingTests(unittest.TestCase):
    def test_private_root_folders_ignored_but_aggregate_docs_retained(self):
        self.assertTrue(ignored(Path("private_artifacts/model.json")))
        self.assertTrue(ignored(Path("outputs/demo/report.html")))
        self.assertFalse(ignored(Path("docs/data/outcome_performance.csv")))

    def test_secret_shape_is_detected_without_returning_its_value(self):
        value = "gh"+"p_"+"a"*36
        findings = scan_text(value, "example.txt")
        self.assertEqual(findings[0]["category"], "github_token_shape")
        self.assertNotIn(value, str(findings))

    def test_machine_path_detected_without_returning_its_value(self):
        value = "/"+"rhome"+"/example_user/private"
        findings = scan_text(value, "example.txt")
        self.assertEqual(findings[0]["category"], "machine_specific_home_path")
        self.assertNotIn(value, str(findings))

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"a.txt").write_text("safe")
            (root/"alias.txt").symlink_to(root/"a.txt")
            with self.assertRaises(ValueError):
                list(files_for_upload(root))

    def test_relative_markdown_link_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"target.md").write_text("# Target")
            readme = root/"README.md"
            readme.write_text("[ok](target.md)\n[missing](absent.md)")
            errors = markdown_links(root, readme)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["target"], "absent.md")

    def test_private_data_and_credential_names_are_blocked(self):
        for value in ["reads.fastq.gz", "reads.fq", "variants.vcf.bgz", "secret.pem",
                      "secret.key", "secret.p12", "secret.pfx", "transfer.partial",
                      "known_hosts", "ssh_config", "id_rsa.pub", "id_ed25519",
                      "active_reference_bank.json", "config.local.json", "model.onnx"]:
            with self.subTest(value=value):
                self.assertTrue(forbidden_upload_file(Path(value)))

    def test_only_reviewed_aggregate_csv_directory_is_allowed(self):
        self.assertFalse(forbidden_upload_file(Path("docs/data/results.csv")))
        self.assertTrue(forbidden_upload_file(Path("pipeline/customer_results.csv")))

    def test_test_fingerprint_changes_with_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"pipeline").mkdir()
            script = root/"pipeline"/"example.py"
            script.write_text("VALUE = 1\n")
            before = source_snapshot(root)
            script.write_text("VALUE = 2\n")
            self.assertNotEqual(before, source_snapshot(root))
            policy = root/"pipeline"/"example_policy.json"
            policy.write_text('{"minimum": 8}\n')
            configured = source_snapshot(root)
            self.assertIn("pipeline/example_policy.json", configured)
            policy.write_text('{"minimum": 9}\n')
            self.assertNotEqual(configured, source_snapshot(root))


if __name__ == "__main__":
    unittest.main()

