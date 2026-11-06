"""Focused safety/identifiability regression tests for CSA diagnostic evidence."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

import csa_low_performance_diagnostics as diagnostics
from outcome_reference import DEFAULT_POLICY, SOURCES


FEATURE = diagnostics.PIN_PREFIX + "perc_pinned"


def png(value: int) -> bytes:
    out = io.BytesIO()
    Image.new("L", (20, 10), value).save(out, format="PNG")
    return out.getvalue()


def write_zip(path: Path, *, raw=True, reseq=False, offset=None, extra=None):
    if offset is None:
        offset = {"PercPinned": 10, "PercPinnedLow": 6, "PercPinnedHigh": 4,
            "PixelCount": 100, "PixelLow": 6, "PixelHigh": 4, "PixelInRange": 90}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("CSA/chipDiagnostics/results.json", json.dumps({"offset": offset}))
        if raw:
            archive.writestr("CSA/rawTrace/keyBkgSub.inlet.png", png(0))
            archive.writestr("CSA/rawTrace/flowResidual.outlet.bead.png", png(255))
        if reseq:
            archive.writestr("CSA/RawTrace_Reseq/keyBkgSub.inlet.png", png(40))
        for name, data in (extra or {}).items(): archive.writestr(name, data)


def peer_table(n=8, *, instrument="inst_A", lanes=False):
    rows = []
    for i in range(n):
        row = dict(observation_id=f"peer_{i}", physical_run_id=f"run_{i}", instrument_id=instrument,
            archive_id=f"archive_{i}", chip_lane=f"ChipLane{1 + (i % 2 if lanes else 0)}",
            run_start_time=f"2026-10-{i + 1:02d}T00:00:00Z", reference_eligible=True,
            panel_signature="panel_one", dna_barcode_count="8", rna_barcode_count="8", evaluated_lane_count="1")
        row.update({name: f"matching_{name}" for name in SOURCES})
        row[FEATURE] = float(i + 1)
        rows.append(row)
    target = rows[0].copy() if rows else dict()
    target.update(observation_id="target", physical_run_id="target_run", instrument_id="inst_A",
        archive_id="target_archive", chip_lane="ChipLane1", run_start_time="2026-10-30T00:00:00Z", reference_eligible=False)
    target.update({name: f"matching_{name}" for name in SOURCES})
    target.update(panel_signature="panel_one", dna_barcode_count="8", rna_barcode_count="8", evaluated_lane_count="1")
    target[FEATURE] = 12.0
    return pd.DataFrame([*rows, target])


class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_member_paths_are_never_extracted(self):
        for unsafe in ["../escape.png", "/absolute.png", "a/../../bad", "a\\bad", "a\x00bad"]:
            self.assertIsNone(diagnostics._safe_member(unsafe))
        self.assertEqual(diagnostics._safe_member("./CSA/rawTrace/x.png"), "CSA/rawTrace/x.png")
        path = self.root / "unsafe.txz"
        write_zip(path, extra={"../rawTrace/keyBkgSub.inlet.png": png(80)})
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertEqual(got["rawtrace_images"], 2)
        self.assertTrue(any("Unsafe" in s for s in got["warnings"]))
        self.assertFalse((self.root / "rawTrace").exists())
        self.assertEqual(len(got["native_images"]), 2)
        for image in got["native_images"]:
            self.assertEqual(diagnostics._sha(Path(image["path"])), image["sha256"])

    def test_pinned_original_case_not_merged_count_alias(self):
        path = self.root / "valid.txz"
        write_zip(path, offset={"PercPinned": 10, "PercPinnedLow": 6, "PercPinnedHigh": 4,
            "PixelCount": 100, "PixelLow": 6, "PixelHigh": 4, "PixelInRange": 90,
            "perc_pinned": 99, "pixel_in_range": 1000000})
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertEqual(got["values"][FEATURE], 10)
        self.assertEqual(got["source_fields"][FEATURE]["source_field"], "offset.PercPinned")
        self.assertNotIn("results_offset_pixel_in_range.mean", got["values"])

    def test_lowercase_percentage_is_not_substituted(self):
        path = self.root / "alias.zip"
        write_zip(path, raw=False, offset={"perc_pinned": 10, "PixelCount": 100, "PixelLow": 6, "PixelHigh": 4})
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertNotIn(FEATURE, got["values"])
        self.assertEqual(got["chip_status"], "invalid_or_unverified_chip_denominator")

    def test_invalid_percentage_or_denominator_is_unresolved(self):
        for i, offset in enumerate([
            {"PercPinned": 1000, "PixelCount": 100, "PixelLow": 6, "PixelHigh": 4},
            {"PercPinned": 10, "PixelCount": 0, "PixelLow": 6, "PixelHigh": 4},
            {"PercPinned": 10, "PixelCount": 100, "PixelLow": 6, "PixelHigh": 4, "PixelInRange": 5},
            {"PercPinned": 30, "PixelCount": 100, "PixelLow": 6, "PixelHigh": 4},
        ]):
            path = self.root / f"invalid_{i}.zip"
            write_zip(path, raw=False, offset=offset)
            got = diagnostics.extract_diagnostics(path, self.root / f"out_{i}")
            self.assertNotIn(FEATURE, got["values"])

    def test_missing_rawtrace_is_missing_not_healthy(self):
        path = self.root / "missing.zip"
        write_zip(path, raw=False)
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertEqual(got["rawtrace_status"], "missing_rawtrace")
        self.assertEqual(got["native_images"], [])
        self.assertFalse(any(k.startswith("plugin.rawtrace.image") for k in got["values"]))

    def test_nonobject_offset_is_unresolved_not_a_plugin_crash(self):
        path = self.root / "invalid_offset.zip"
        write_zip(path, raw=False, offset=[])
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertNotIn(FEATURE, got["values"])
        self.assertTrue(any("not an object" in warning for warning in got["warnings"]))

    def test_mixed_resequencing_has_no_image_proxy(self):
        path = self.root / "mixed.zip"
        write_zip(path, reseq=True)
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertEqual(got["rawtrace_status"], "mixed_or_resequencing_rawtrace_excluded")
        self.assertFalse(any(k.startswith("plugin.rawtrace.image") for k in got["values"]))
        self.assertEqual(len(got["native_images"]), 2)

    def test_quarantined_plugins_never_supply_case_values(self):
        path = self.root / "banned.zip"
        write_zip(path, raw=False, extra={"CSA/TrueLoading/results.json": b'{"loading":100}',
            "CSA/EdgeEffects/results.json": b'{"bad_wells":10000000}',
            "CSA/plugins/NovelPlugin_1.0/results.json": b'{"secret":99}'})
        got = diagnostics.extract_diagnostics(path, self.root / "out")
        self.assertEqual(got["plugin_counts"]["true_loading"], 1)
        self.assertEqual(got["plugin_counts"]["edge_effects"], 1)
        self.assertTrue(any(k.startswith("unvalidated_plugin_") for k in got["plugin_counts"]))
        self.assertTrue(all(k.startswith(diagnostics.PIN_PREFIX) for k in got["values"]))


class PeerSelectionTests(unittest.TestCase):
    def test_eight_independent_runs_and_strict_prior_required(self):
        table = peer_table()
        extra = table.iloc[0].to_dict()
        extra.update(observation_id="future", physical_run_id="future", run_start_time="2026-10-31T00:00:00Z")
        sibling = extra.copy()
        sibling.update(observation_id="sibling", physical_run_id="target_run", run_start_time="2026-10-01T00:00:00Z")
        mismatch = extra.copy()
        mismatch.update(observation_id="mismatched", physical_run_id="different", run_start_time="2026-10-01T00:00:00Z", chip_type="different_chip")
        table = pd.concat([table, pd.DataFrame([extra, sibling, mismatch])], ignore_index=True)
        got, members, audit = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertEqual(got["status"], "available")
        self.assertEqual(got["reference_physical_runs"], 8)
        self.assertEqual(len(members), 8)
        self.assertFalse(any(a["peer_observation_id"] == "sibling" for a in audit))
        self.assertFalse(next(a for a in audit if a["peer_observation_id"] == "future")["eligible"])
        self.assertIn("chip_type", next(a for a in audit if a["peer_observation_id"] == "mismatched")["metadata_mismatches"])

    def test_per_metric_missingness_cannot_borrow_healthy_status(self):
        table = peer_table()
        table.loc[table.observation_id.eq("peer_0"), FEATURE] = np.nan
        got, members, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertEqual(got["status"], "insufficient_compatible_measured_peers")
        self.assertEqual(got["peer_missing_observations"], 1)
        self.assertEqual(members, [])
        self.assertEqual(got["deviation"], "unresolved")

    def test_replicate_scopes_are_one_run_vote(self):
        table = peer_table()
        table.loc[table.observation_id.ne("target"), "physical_run_id"] = [f"run_{i // 2}" for i in range(8)]
        got, _, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertNotEqual(got["status"], "available")
        table = peer_table()
        duplicate = table.iloc[0].to_dict()
        duplicate.update(observation_id="peer_0_reanalysis")
        duplicate[FEATURE] = 3
        table = pd.concat([table, pd.DataFrame([duplicate])], ignore_index=True)
        got, members, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertEqual(got["reference_physical_runs"], 8)
        self.assertEqual(next(m for m in members if m["peer_physical_run_id"] == "run_0")["value"], 2)

    def test_cross_reference_is_other_only_and_two_instruments(self):
        table = peer_table(instrument="inst_B")
        got, _, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertNotEqual(got["status"], "available")
        table.loc[table.observation_id.isin(["peer_0", "peer_1"]), "instrument_id"] = "inst_C"
        got, members, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertEqual(got["reference_mode"], "other_instruments")
        self.assertEqual(got["reference_instruments"], 2)
        self.assertTrue(all(m["instruments"] != "inst_A" for m in members))

    def test_missing_protocol_or_target_metric_abstains(self):
        table = peer_table()
        got, _, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, None, [])
        self.assertEqual(got["status"], "target_metric_unavailable")
        table.loc[table.observation_id.eq("target"), "sequencing_kit"] = None
        got, _, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
        self.assertEqual(got["status"], "missing_required_target_metadata")

    def test_matching_image_count_and_render_signature_required(self):
        table = peer_table()
        table[diagnostics.REFERENCE_IMAGE_COUNT] = 43
        table.loc[table.observation_id.eq("peer_0"), diagnostics.REFERENCE_IMAGE_COUNT] = 86
        got, _, audit = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [], composition={diagnostics.REFERENCE_IMAGE_COUNT: 43})
        self.assertNotEqual(got["status"], "available")
        self.assertFalse(next(a for a in audit if a["peer_observation_id"] == "peer_0")["render_signature_matches"])

    def test_nonfinite_and_impossible_peer_percentages_are_missing(self):
        for bad in [np.inf, -1, 101]:
            table = peer_table()
            table.loc[table.observation_id.eq("peer_0"), FEATURE] = bad
            got, _, audit = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [])
            self.assertNotEqual(got["status"], "available")
            row = next(a for a in audit if a["peer_observation_id"] == "peer_0")
            self.assertFalse(row["metric_available"])
            self.assertEqual(row["metric_validation"], "missing_nonfinite_or_impossible_value")

    def test_cross_instrument_disabled_policy_is_respected(self):
        table = peer_table(instrument="inst_B")
        table.loc[table.observation_id.eq("peer_0"), "instrument_id"] = "inst_C"
        got, _, _ = diagnostics.select_peer_measurements(table, "target", FEATURE, 12, [], allow_cross_instrument=False)
        self.assertNotEqual(got["status"], "available")
        self.assertIn("disabled_by_policy", got["coverage_by_tier"])


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"csa_diagnostic_model_study": str(self.root / "study"),
            "csa_diagnostic_plugin_inventory": str(self.root / "inventory.csv")}

    def test_not_triggered_and_disabled_write_valid_manifests(self):
        result = diagnostics.build_low_performance_diagnostics(self.root, "target", self.root / "missing.zip", self.root / "out", {}, self.config)
        self.assertEqual(result["status"], "not_triggered")
        self.assertTrue(Path(result["manifest_path"]).is_file())
        self.assertTrue(diagnostics.cache_artifacts_valid(Path(result["manifest_path"])))
        result = diagnostics.build_low_performance_diagnostics(self.root, "target", self.root / "missing.zip", self.root / "disabled", {"low_outcomes": ["total_reads"]}, {**self.config, "enable_csa_low_performance_diagnostics": False})
        self.assertEqual(result["status"], "disabled")
        self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_builder_narrow_reads_and_forces_prior_qualified_policy(self):
        archive = self.root / "target.txz"
        write_zip(archive)
        extracted = diagnostics.extract_diagnostics(archive, self.root / "fresh")
        table = peer_table()
        rows = []
        for r in table.to_dict("records"):
            values = {k: extracted["values"][k] for k in [*diagnostics.METRICS, *diagnostics.RENDER_FIELDS]}
            if r["observation_id"] != "target":
                values.update({k: 0.1 for k in diagnostics.METRICS})
            rows.append(dict(observation_id=r["observation_id"], archive_id=r["archive_id"], **values, ignored_massive_column=123))
        study = self.root / "study"
        study.mkdir()
        pd.DataFrame(rows).to_parquet(study / "training_dataset.parquet", index=False)
        pd.DataFrame([dict(observation_id=r["observation_id"], archive_id=r["archive_id"], plugin_family="rawtrace", extension=".png", file_count=2) for r in rows]).to_csv(self.root / "inventory.csv", index=False)
        policy = copy.deepcopy(DEFAULT_POLICY)
        policy.update(time_policy="retrospective", minimum_physical_runs=3, require_outcome_qualified_reference=False, extra_match_columns=["evaluated_lane_count"])
        parquet_reader = pd.read_parquet
        seen_columns = []
        def track_read(path, **kwargs):
            seen_columns.extend(kwargs.get("columns", []))
            return parquet_reader(path, **kwargs)
        with patch.object(diagnostics, "load_policy", return_value=policy), patch.object(diagnostics, "load_outcome_table", return_value=table.drop(columns=[FEATURE])) as load_table, patch.object(diagnostics.pd, "read_parquet", side_effect=track_read), patch.object(diagnostics, "_plot", return_value=[]):
            result = diagnostics.build_low_performance_diagnostics(self.root, "target", archive, self.root / "out", {"below_chip_lane_read_criterion": True}, self.config)
        used_policy = load_table.call_args.args[1]
        self.assertEqual(used_policy["time_policy"], "prior_only")
        self.assertEqual(used_policy["minimum_physical_runs"], 8)
        self.assertTrue(used_policy["require_outcome_qualified_reference"])
        self.assertNotIn("ignored_massive_column", seen_columns)
        self.assertEqual(result["status"], "diagnostic_clues_found")
        self.assertEqual(len([m for m in result["metrics"] if m["status"] == "available"]), 6)
        self.assertFalse(result["models_retrained"])
        self.assertFalse(result["instrument_settings_changed"])
        self.assertTrue(result["investigation_hypotheses"])
        self.assertTrue(diagnostics.cache_artifacts_valid(Path(result["manifest_path"])))
        Path(result["outputs"]["csa_diagnostic_metrics"]).write_text("changed\n")
        self.assertFalse(diagnostics.cache_artifacts_valid(Path(result["manifest_path"])))

    def test_html_escapes_data_and_does_not_claim_actual_volume(self):
        result = dict(status="unresolved", triggered=True, summary="<script>bad</script>", warnings=["<img onerror='x'>"], metrics=[])
        html = diagnostics.html_section(result)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img onerror", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Missing is not normal", html)
        self.assertIn("No delivered-volume error", html)

    def test_inventory_sum_retains_single_image_extensions_and_schema(self):
        path = self.root / "counts.csv"
        pd.DataFrame([dict(archive_id="one", plugin_family="rawtrace", extension=".png", file_count=43),
            dict(archive_id="one", plugin_family="rawtrace", extension=".jpg", file_count=1)]).to_csv(path, index=False)
        got = diagnostics._reference_image_counts(path)
        self.assertEqual(got.iloc[0][diagnostics.REFERENCE_IMAGE_COUNT], 44)
        # An observation-annotated inventory is also supported without double
        # voting for an archive shared by lanes.
        rows = pd.read_csv(path)
        rows["observation_id"] = "obs_one"
        rows = pd.concat([rows, rows.assign(observation_id="obs_sibling")])
        rows.to_csv(path, index=False)
        self.assertEqual(diagnostics._reference_image_counts(path).iloc[0][diagnostics.REFERENCE_IMAGE_COUNT], 44)

    def test_corrupt_optional_archive_is_explicitly_unresolved(self):
        archive = self.root / "bad.txz"
        archive.write_bytes(b"not a valid archive")
        result = diagnostics.build_low_performance_diagnostics(self.root, "target", archive, self.root / "bad", {"low_outcomes": ["total_reads"]}, self.config)
        self.assertEqual(result["status"], "unresolved")
        self.assertIn("parse_error", result)
        self.assertTrue(Path(result["manifest_path"]).exists())


class PeerPlotTests(unittest.TestCase):
    def test_whiskers_equal_the_interpolated_table_percentiles(self):
        values = np.arange(12, dtype=float)
        result = diagnostics._peer_box_statistics(values)
        self.assertAlmostEqual(result["whislo"], .55)
        self.assertAlmostEqual(result["whishi"], 10.45)
        self.assertAlmostEqual(result["med"], 5.5)
        self.assertNotIn(result["whislo"], values)
        self.assertNotIn(result["whishi"], values)


if __name__ == "__main__": unittest.main()
