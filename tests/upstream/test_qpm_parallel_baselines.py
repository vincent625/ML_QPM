"""Guard against population mixing, hidden fallback and manufactured evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from qpm_baseline_comparison import enrich_history, summarize_baselines, write_comparison, MODES, nominal_volume_note
from qpm_cross_instrument_reference import scan_cross_instrument_fallback
from qpm_waveform_reference import PROFILE_COLUMNS, scan_references


def examples():
    curve = -100 * np.sin(np.linspace(0, np.pi, 32))
    curve[:3] = 0
    def row(label, instrument, physical, date="2026-10-01T00:00:00Z", scale=1):
        item = dict(observation_id=label, physical_run_id=physical, instrument_id=instrument,
                    archive_id="archive_" + label, chip_lane="1", run_start_time=date,
                    operation_id="operation_" + label, operation_index=1, operation="aspirate",
                    pipette="PIP1", programmed_volume_ul=10., event_time=date,
                    workflow_stage="preparation", process_step="sample_transfer", source_file="CSA/debug",
                    pressure_unit="raw_sensor_counts", liquid_class_index=1., legacy_liquid_class_index=1.,
                    location="pcr_a1", command_context_status="command_verified",
                    container_geometry_index=1, command_container_geometry_index=1,
                    deck_geometry_index=1, command_deck_geometry_index=1,
                    liquid_class_fingerprint="lc_full_settings", container_geometry_fingerprint="cg_full_settings",
                    deck_geometry_fingerprint="dg_full_settings", command_settings_fingerprint="command_modes",
                    container_geometry_name="pcr_well", container_model="pcr", deck_geometry_name="deck",
                    liquid_class_name="water", chip_type="GX5", chip_fluidic_context="not_chip_contact")
        item.update(qpm_has_er72=False, qpm_trace_quality_status="checked_no_er72",
                    qpm_waveform_eligible=True, qpm_exclusion_reason="", reference_eligible=True)
        item.update(dict(zip(PROFILE_COLUMNS, curve * scale)))
        return item
    rows = [row("target", "instrument_target", "physical_target", "2026-10-20T00:00:00Z")]
    rows += [row(f"same_{i}", "instrument_target", f"physical_same_{i}") for i in range(8)]
    rows += [row(f"other_{i}", f"instrument_other_{i % 2}", f"physical_other_{i}", scale=i + 1) for i in range(8)]
    return pd.DataFrame(rows)


class ParallelBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name)
        self.config = dict(minimum_reference_physical_runs=8, waveform_reference_policy_overrides={},
                           shared_reference_minimum_percentile=.95, shared_reference_minimum_outside_fraction=.25)
        self.frame = examples()
        self.cache = {"parser_sha256": "test", "archives": []}

    def tearDown(self):
        self.temp.cleanup()

    def same(self, frame=None):
        return scan_references(self.project, "target", self.config, (self.frame if frame is None else frame, self.cache))

    def test_historical_volume_annotation_retained_without_replacing_raw_curve(self):
        same, _, _, _ = self.same()
        same["waveform_equivalent_volume_ul"] = 12.
        same["aggregate_validation_mae_ul"] = 4.
        same["volume_proxy_status"] = "nominal_recognition_only:not_measured_delivery"
        result = enrich_history(self.frame[self.frame.observation_id.eq("target")],same,self.config)
        self.assertEqual(result.waveform_equivalent_volume_ul.iloc[0],12.)
        np.testing.assert_allclose(json.loads(result.observed_profile.iloc[0]),self.frame.iloc[0][PROFILE_COLUMNS].to_numpy(float))
        note = nominal_volume_note(result.iloc[0])
        self.assertIn("+2.0 µL",note)
        self.assertIn("held-out MAE 4.0 µL",note)
        self.assertIn("Not a measured",note)

    def test_unavailable_volume_annotation_explains_abstention(self):
        note=nominal_volume_note(pd.Series(dict(volume_proxy_status="unavailable:target_physical_run_present_in_model_training_use_heldout_predictions")))
        self.assertIn("unavailable",note)
        self.assertIn("target physical run",note)

    def cross(self, frame=None, independent=True, coverage=None):
        frame = self.frame if frame is None else frame
        if coverage is None:
            _, coverage, _, _ = self.same(frame)
        return scan_cross_instrument_fallback(self.project, "target", {"operation_target"}, self.config,
                                               (frame, self.cache), coverage, independent=independent)

    def test_cross_is_evaluated_despite_adequate_same_history(self):
        same, coverage, members, _ = self.same()
        self.assertEqual(int(same.reference_physical_runs.iloc[0]), 8)
        cross, _, other_members, meta = self.cross(coverage=coverage)
        self.assertEqual(len(cross), 1)
        self.assertEqual(int(cross.reference_physical_runs.iloc[0]), 8)
        self.assertEqual(meta["scope"], "all_target_operations_independent")
        self.assertEqual(set(members.reference_instrument_id), {"instrument_target"})
        self.assertNotIn("instrument_target", set(other_members.reference_instrument_id))

    def test_unqualified_history_cannot_satisfy_minimum_in_either_mode(self):
        frame = self.frame.copy()
        frame.loc[frame.observation_id.isin(["same_0", "other_0"]), "reference_eligible"] = False
        same, coverage, _, _ = self.same(frame)
        cross, _, _, _ = self.cross(frame, coverage=coverage)
        self.assertTrue(same.empty)
        self.assertTrue(cross.empty)
        self.assertEqual(int(coverage.maximum_available_reference_runs.iloc[0]), 7)

    def test_bad_target_is_scored_against_qualified_history(self):
        frame = self.frame.copy()
        frame.loc[frame.observation_id.eq("target"), "reference_eligible"] = False
        same, coverage, _, _ = self.same(frame)
        cross, _, _, _ = self.cross(frame, coverage=coverage)
        self.assertEqual(len(same), 1)
        self.assertEqual(len(cross), 1)

    def test_unverified_historical_qualification_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "eligibility has not been verified"):
            self.same(self.frame.drop(columns="reference_eligible"))

    def test_cross_builds_chip_context_without_same_scan_side_effects(self):
        frame = self.frame.drop(columns="chip_fluidic_context")
        cross, _, _, _ = self.cross(frame)
        self.assertEqual(len(cross), 1)
        self.assertNotIn("chip_fluidic_context", frame.columns)

    def test_legacy_fallback_still_skips_adequate_same_history(self):
        cross, coverage, _, _ = self.cross(independent=False)
        self.assertTrue(cross.empty)
        self.assertTrue(coverage.empty)

    def test_cross_uses_dimensionless_shape_not_raw_amplitude(self):
        cross, _, _, _ = self.cross()
        self.assertEqual(cross.reference_profile_unit.iloc[0], "dimensionless_shape")
        self.assertTrue(np.isnan(cross.amplitude_ratio_to_reference.iloc[0]))
        self.assertAlmostEqual(np.ptp(json.loads(cross.reference_median_profile.iloc[0])), 1.)
        self.assertAlmostEqual(np.ptp(json.loads(cross.reference_observed_profile.iloc[0])), 1.)

    def test_target_and_sibling_physical_run_are_excluded(self):
        siblings = self.frame[self.frame.instrument_id.ne("instrument_target")].copy()
        siblings["physical_run_id"] = "physical_target"
        siblings["observation_id"] = "sibling_" + siblings.observation_id
        siblings["operation_id"] = "sibling_" + siblings.operation_id
        expanded = pd.concat([self.frame, siblings], ignore_index=True)
        cross, _, members, _ = self.cross(expanded)
        self.assertEqual(int(cross.reference_physical_runs.iloc[0]), 8)
        self.assertNotIn("physical_target", set(members.reference_physical_run_id))

    def test_retries_and_sibling_lanes_do_not_inflate_effective_n(self):
        duplicate = self.frame[self.frame.observation_id.str.startswith("other_")].copy()
        duplicate["observation_id"] = "sibling_" + duplicate.observation_id
        duplicate["operation_id"] = "sibling_" + duplicate.operation_id
        duplicate["chip_lane"] = "2"
        expanded = pd.concat([self.frame, duplicate], ignore_index=True)
        cross, _, members, _ = self.cross(expanded)
        self.assertEqual(int(cross.reference_physical_runs.iloc[0]), 8)
        self.assertEqual(len(members), 8)

    def test_physical_configuration_cannot_be_relaxed(self):
        frame = self.frame.copy()
        frame.loc[frame.instrument_id.ne("instrument_target"), "container_geometry_fingerprint"] = "incompatible_geometry"
        cross, coverage, _, _ = self.cross(frame)
        self.assertTrue(cross.empty)
        self.assertEqual(int(coverage.cross_instrument_available_runs.iloc[0]), 0)

    def test_units_cannot_be_pooled(self):
        frame = self.frame.copy()
        frame.loc[frame.instrument_id.ne("instrument_target"), "pressure_unit"] = "kPa"
        cross, _, _, _ = self.cross(frame)
        self.assertTrue(cross.empty)

    def test_missing_target_configuration_is_unresolved(self):
        frame = self.frame.copy()
        frame.loc[frame.observation_id.eq("target"), "command_context_status"] = "missing_or_conflicting_configuration_definition"
        cross, coverage, _, _ = self.cross(frame)
        self.assertTrue(cross.empty)
        self.assertFalse(bool(coverage.cross_instrument_attempted.iloc[0]))

    def test_flat_target_does_not_manufacture_shape(self):
        frame = self.frame.copy()
        frame.loc[frame.observation_id.eq("target"), PROFILE_COLUMNS] = 0.
        cross, coverage, _, _ = self.cross(frame)
        self.assertTrue(cross.empty)
        self.assertEqual(coverage.cross_instrument_status.iloc[0], "unresolved_flat_target_waveform")

    def test_minimum_cross_instrument_diversity_is_enforced(self):
        frame = self.frame.copy()
        frame.loc[frame.instrument_id.ne("instrument_target"), "instrument_id"] = "only_one_other_instrument"
        cross, coverage, _, _ = self.cross(frame)
        self.assertTrue(cross.empty)
        self.assertEqual(coverage.cross_instrument_status.iloc[0], "insufficient_reference_instruments")

    def test_prior_only_policy_excludes_future_runs(self):
        self.config["waveform_reference_policy_overrides"] = {"time_policy": "prior_only"}
        future = self.frame[self.frame.instrument_id.ne("instrument_target")].copy()
        future["physical_run_id"] = "future_" + future.physical_run_id
        future["observation_id"] = "future_" + future.observation_id
        future["operation_id"] = "future_" + future.operation_id
        future["run_start_time"] = "2026-10-25T00:00:00Z"
        cross, _, members, _ = self.cross(pd.concat([self.frame, future], ignore_index=True))
        self.assertEqual(int(cross.reference_physical_runs.iloc[0]), 8)
        self.assertFalse(members.reference_includes_future.any())

    def test_explicit_same_instrument_only_policy_is_not_overridden(self):
        self.config["waveform_reference_policy_overrides"] = {"instrument_policy": "same_instrument_only"}
        cross, _, _, metadata = self.cross()
        self.assertTrue(cross.empty)
        self.assertFalse(metadata["enabled"])

    def test_enrichment_keeps_historical_populations_separate(self):
        same, _, _, _ = self.same()
        cross, _, _, _ = self.cross()
        target = self.frame[self.frame.observation_id.eq("target")]
        a, b = enrich_history(target, same, self.config), enrich_history(target, cross, self.config)
        self.assertEqual(a.reference_scope.iloc[0], "same_instrument")
        self.assertEqual(b.reference_scope.iloc[0], "cross_instrument_shape")
        self.assertNotEqual(a.reference_profile_unit.iloc[0], b.reference_profile_unit.iloc[0])

    def test_three_rows_are_present_when_every_comparator_is_unavailable(self):
        empty = pd.DataFrame()
        target = self.frame.iloc[:0]
        histories = {m: enrich_history(target, empty, self.config) for m in MODES[1:]}
        lineage = pd.DataFrame({"path_status": ["survivor"]})
        summary = summarize_baselines(lineage, empty, empty, target, histories,
                                     {m: empty for m in MODES[1:]}, {m: empty for m in MODES[1:]},
                                     {m: {} for m in MODES[1:]})
        self.assertEqual(summary.baseline_id.tolist(), MODES)
        self.assertEqual(summary.status.iloc[0], "not_testable_no_affected_paths")
        self.assertTrue(summary.status.iloc[1:].eq("unavailable_no_usable_target_waveforms").all())

    def test_report_exports_all_three_modes_without_outcome_trigger(self):
        from qpm_dropout_plugin import PALETTE
        same, same_coverage, same_members, same_metadata = self.same()
        cross, cross_coverage, cross_members, cross_metadata = self.cross()
        lineage = pd.DataFrame({"path_status": ["survivor"]})
        output = self.project / "report"
        output.mkdir()
        payload = write_comparison(output, lineage, {}, {}, self.frame[self.frame.observation_id.eq("target")],
                                   same, same_coverage, same_members, same_metadata,
                                   cross, cross_coverage, cross_members, cross_metadata, self.config, PALETTE)
        summary = pd.read_csv(output / "qpm_baseline_comparison_summary.csv")
        self.assertEqual(summary.baseline_id.tolist(), MODES)
        self.assertEqual(summary.status.iloc[0], "not_testable_no_affected_paths")
        self.assertEqual(summary.scored_operations.iloc[1:].tolist(), [1, 1])
        self.assertTrue(payload["always_evaluated_independently"])
        for mode in MODES[1:]:
            for kind in ["evidence", "coverage", "members"]:
                self.assertTrue((output / f"qpm_baseline_{mode}_{kind}.csv").is_file())
        self.assertTrue((output / "qpm_baseline_comparison.png").stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()
