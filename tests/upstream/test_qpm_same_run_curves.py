"""Descriptive same-run curves are available independently of test power."""
from __future__ import annotations

import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import qpm_same_run_curves as gallery


PALETTE = dict(ink="#264653", teal="#2A9D8F", coral="#E76F51", gray="#8A918B",
               gold="#E9C46A", paper="#FCFCF8", grid="#D9DED9")


def data(*, failed=1, survivors=7, material="DNA", group="g0", start=10, label=None):
    rows = []
    shape = -100 * np.sin(np.linspace(0, np.pi, 32))
    for index in range(failed + survivors):
        status = "failed" if index < failed else "survivor"
        row = dict(observation_id="target", physical_run_id="target_physical", instrument_id="target_instrument",
                   archive_id="target_archive", chip_lane="1", group_id=group, material_type=material,
                   family="reaction_transfer", operation="aspirate", pipette="PIP1", programmed_volume_ul=20.,
                   liquid_class_index=55., planned_order=index+1, location=f"reaction_well_{index+1}",
                   operation_id=f"{group}_operation_{index+1}", operation_index=start+index,
                   workflow_stage="library_preparation", process_step="reaction_transfer",
                   event_time="2026-10-01T00:00:00Z", pressure_unit="raw_sensor_counts",
                   internal_context_verified=True, internal_context_fingerprint="verified_context",
                   liquid_class_fingerprint="liquid_class", container_geometry_fingerprint="container",
                   deck_geometry_fingerprint="deck", command_settings_fingerprint="command_settings",
                   qpm_has_er72=False, qpm_waveform_eligible=True,
                   qpm_trace_quality_status="checked_no_er72", qpm_exclusion_reason="", path_status=status,
                   observed_profile=shape * (2 if status == "failed" else 1) + index)
        rows.append(row)
    stat = dict(material_type=material, group_id=group, family="reaction_transfer", operation="aspirate",
                pipette="PIP1", programmed_volume_ul=20., liquid_class_index=55.,
                first_operation_index=start, last_operation_index=start+max(0, len(rows)-1),
                n_failed=failed, n_survivor=survivors, eligible_internal_comparison=False,
                failed_path_deviation_auc=np.nan, permutation_p=np.nan, bh_fdr_q=np.nan,
                screen_label="insufficient_replication", evidence_role="insufficient_replication",
                step_label=label or f"Transfer {group} · aspirate · 20 µL")
    return pd.DataFrame([stat]), pd.DataFrame(rows)


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.polylines = []
        self.details = []
        self.svgs = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "polyline": self.polylines.append(attrs)
        if tag == "details": self.details.append(attrs)
        if tag == "svg": self.svgs += 1


class SameRunCurveGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def build(self, tests, mapped, name="out"):
        return gallery.build_same_run_curves(self.root/name, tests, mapped, PALETTE)

    def parse(self, result):
        text = Path(result["outputs"]["fragment"]).read_text()
        parser = Elements(); parser.feed(text)
        return text, parser

    def test_one_affected_seven_survivors_has_both_classes_and_medians(self):
        tests, mapped = data()
        result = self.build(tests, mapped)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["curve_groups"], 1)
        self.assertEqual(result["descriptive_only_groups"], 1)
        self.assertEqual(result["independent_path_group_curves"], 8)
        text, parser = self.parse(result)
        individuals = [p for p in parser.polylines if p.get("data-curve-kind") == "individual"]
        self.assertEqual(sum(p.get("data-path-class") == "failed" for p in individuals), 1)
        self.assertEqual(sum(p.get("data-path-class") == "survivor" for p in individuals), 7)
        self.assertEqual(sum(p.get("data-curve-kind") == "median" for p in parser.polylines), 2)
        self.assertIn("Descriptive only", text)
        self.assertTrue(result["formal_statistical_minimum_unchanged"])
        self.assertFalse(result["new_inferential_statistics"])

    def test_one_versus_one_has_no_invented_reference_interval(self):
        result = self.build(*data(failed=1, survivors=1))
        text, parser = self.parse(result)
        self.assertEqual(result["independent_path_group_curves"], 2)
        self.assertEqual(len(parser.polylines), 4)
        self.assertNotIn("5th–95th", text)
        self.assertNotIn("<polygon", text)
        self.assertIn("Surviving paths n=1", text)

    def test_export_preserves_every_sample_and_source_unit(self):
        tests, mapped = data()
        mapped["pressure_unit"] = "kPa"
        result = self.build(tests, mapped)
        ledger = pd.read_csv(result["outputs"]["curves"])
        self.assertEqual(len(ledger), 8)
        for record in ledger.to_dict("records"):
            source = mapped.loc[mapped.planned_order.eq(record["planned_order"])].iloc[0]
            np.testing.assert_array_equal(json.loads(record["observed_profile"]), source.observed_profile)
            self.assertEqual(record["pressure_unit"], "kPa")
            self.assertEqual(record["source_operation_ids"], source.operation_id)
            self.assertFalse(record["formal_statistics_recomputed"])
        text, parser = self.parse(result)
        self.assertIn("QPM source unit: kPa", text)
        for line in parser.polylines:
            self.assertEqual(len(line["points"].split()), 32)

    def test_retry_waveforms_are_one_median_vote_per_path(self):
        tests, mapped = data()
        retry = mapped.iloc[0].copy()
        retry["operation_id"] = "retry_failed_path"
        retry["operation_index"] = 99
        retry["observed_profile"] = np.asarray(retry.observed_profile) * 3
        source = mapped.iloc[0].observed_profile.copy()
        mapped = pd.concat([mapped, pd.DataFrame([retry])], ignore_index=True)
        result = self.build(tests, mapped)
        ledger = pd.read_csv(result["outputs"]["curves"])
        self.assertEqual(len(ledger), 8)
        failed = ledger.loc[ledger.path_status.eq("failed")].iloc[0]
        np.testing.assert_array_equal(json.loads(failed.observed_profile), np.median(np.vstack([source, retry.observed_profile]), axis=0))
        self.assertIn("retry_failed_path", failed.source_operation_ids)

    def test_unknown_outcomes_are_not_survivors(self):
        tests, mapped = data()
        unknown = mapped.iloc[0].copy()
        unknown["operation_id"] = "unknown_operation"
        unknown["planned_order"] = 99
        unknown["path_status"] = "unresolved"
        mapped = pd.concat([mapped, pd.DataFrame([unknown])], ignore_index=True)
        result = self.build(tests, mapped)
        ledger = pd.read_csv(result["outputs"]["curves"])
        self.assertEqual(set(ledger.path_status), {"failed", "survivor"})
        self.assertEqual(len(ledger), 8)
        self.assertNotIn(99, ledger.planned_order.tolist())

    def test_er72_cannot_supply_the_only_affected_curve(self):
        tests, mapped = data()
        mapped.loc[mapped.path_status.eq("failed"), "qpm_has_er72"] = True
        mapped.loc[mapped.path_status.eq("failed"), "qpm_waveform_eligible"] = False
        result = self.build(tests, mapped)
        self.assertEqual(result["status"], "unavailable_no_paired_verified_curves")
        _, parser = self.parse(result)
        self.assertEqual(parser.svgs, 0)
        self.assertTrue(pd.read_csv(result["outputs"]["curves"]).empty)

    def test_unverified_or_invalid_raw_configuration_does_not_get_a_curve(self):
        for name in ["context", "raw", "mixed_unit"]:
            with self.subTest(name=name):
                tests, mapped = data()
                if name == "context": mapped.loc[0, "internal_context_verified"] = False
                elif name == "raw": mapped.at[0, "observed_profile"] = np.zeros(31)
                else: mapped.loc[0, "pressure_unit"] = "psi"
                result = self.build(tests, mapped, name)
                self.assertEqual(result["curve_groups"], 0)
                self.assertEqual(self.parse(result)[1].svgs, 0)

    def test_missing_counterpart_is_explicit_not_healthy(self):
        for failed, survivors in [(1, 0), (0, 7)]:
            result = self.build(*data(failed=failed, survivors=survivors), name=f"pair_{failed}_{survivors}")
            text, parser = self.parse(result)
            self.assertEqual(result["curve_groups"], 0)
            self.assertEqual(parser.svgs, 0)
            self.assertIn("Missing curves do not establish normality", text)

    def test_empty_input_writes_explicit_artifacts(self):
        result = self.build(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(result["status"], "unavailable_no_paired_verified_curves")
        self.assertTrue(Path(result["manifest_path"]).is_file())
        self.assertTrue(pd.read_csv(result["outputs"]["curves"]).empty)
        self.assertEqual(self.parse(result)[1].svgs, 0)

    def test_all_groups_and_materials_survive_beyond_five_display_slots(self):
        groups = [data(failed=1, survivors=1, material="DNA", group=f"dna_{i}", start=10+i*20) for i in range(7)]
        groups += [data(failed=2, survivors=2, material="RNA", group=f"rna_{i}", start=20+i*30) for i in range(2)]
        tests = pd.concat([t for t, _ in groups], ignore_index=True).iloc[::-1].reset_index(drop=True)
        mapped = pd.concat([m for _, m in groups], ignore_index=True)
        result = self.build(tests, mapped)
        text, parser = self.parse(result)
        self.assertEqual(result["curve_groups"], 9)
        self.assertEqual(result["materials"]["DNA"]["groups"], 7)
        self.assertEqual(result["materials"]["RNA"]["groups"], 2)
        self.assertEqual(parser.svgs, 9)
        self.assertEqual(len(parser.details), 9)
        self.assertEqual(sum("open" in attrs for attrs in parser.details), 2)
        ledger = pd.read_csv(result["outputs"]["curves"])
        indices = ledger.drop_duplicates("group_id").first_operation_index.tolist()
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(ledger), 22)

    def test_inputs_and_all_formal_statistics_are_unchanged(self):
        tests, mapped = data()
        original_tests = tests.copy(deep=True)
        original_mapped = copy.deepcopy(mapped)
        original_profiles = {index: np.array(curve, copy=True) for index, curve in mapped.observed_profile.items()}
        self.build(tests, mapped)
        pd.testing.assert_frame_equal(tests, original_tests)
        pd.testing.assert_frame_equal(mapped, original_mapped)
        for index, curve in mapped.observed_profile.items():
            np.testing.assert_array_equal(curve, original_profiles[index])
        for field in ["failed_path_deviation_auc", "permutation_p", "bh_fdr_q"]:
            self.assertTrue(tests[field].isna().all())

    def test_html_escapes_recorded_step_text(self):
        result = self.build(*data(label='<script>bad("x")</script> & unsafe'))
        text, _ = self.parse(result)
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("&amp; unsafe", text)
        self.assertEqual(gallery.html_section(result), text)

    def test_generated_artifact_hashes_match_the_manifest(self):
        result = self.build(*data())
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        for path, expected in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)
        for path, expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)
        self.assertFalse(manifest["historical_reference_required"])
        self.assertFalse(manifest["new_inferential_statistics"])

    def test_cache_rejects_modified_display_artifacts(self):
        result = self.build(*data())
        manifest = Path(result["manifest_path"])
        self.assertTrue(gallery.cache_artifacts_valid(manifest))
        report = Path(result["outputs"]["report"])
        report.write_text(report.read_text() + "\n<!-- changed -->")
        self.assertFalse(gallery.cache_artifacts_valid(manifest))

    def test_flat_recorded_curves_remain_finite_and_unmodified(self):
        tests, mapped = data(failed=1, survivors=1)
        for index in mapped.index: mapped.at[index, "observed_profile"] = np.zeros(32)
        result = self.build(tests, mapped)
        text, parser = self.parse(result)
        for line in parser.polylines:
            points = np.array([[float(v) for v in p.split(",")] for p in line["points"].split()])
            self.assertTrue(np.isfinite(points).all())
        ledger = pd.read_csv(result["outputs"]["curves"])
        for value in ledger.observed_profile: np.testing.assert_array_equal(json.loads(value), np.zeros(32))


if __name__ == "__main__": unittest.main()
