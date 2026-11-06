#!/usr/bin/env python3
"""Regression tests for user-defined chip/lane outcome-qualified references."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from qpm_reference_qualification import (
    CHIP_COLUMN, DEFAULT_POLICY, POLICY_FILENAME, build_reference_qualification, classify_barcode_outcomes,
    lane_tokens, load_policy, load_reference_qualification, normalize_chip_type,
    write_qualification,
)


def observation(obs="obs1", physical="run1", lane="1", chip="GX7", reads=25_000_000):
    return dict(observation_id=obs, physical_run_id=physical, instrument_id="inst1",
                archive_id="archive_" + obs, chip_lane=lane,
                analysis_lane_scope="combined_multilane" if "_" in str(lane) else "single_lane",
                total_reads=reads, **{CHIP_COLUMN: chip})


def barcodes(obs="obs1", counts=(1000, 1000, 1000, 1000), material="DNA"):
    return pd.DataFrame([dict(observation_id=obs, planned_order=i + 1, material_type=material,
                              is_control=False, read_count=count) for i, count in enumerate(counts)])


class QualificationTests(unittest.TestCase):
    def screen(self, record=None, bars=None):
        return build_reference_qualification(pd.DataFrame([record or observation()]),
                                             bars if bars is not None else barcodes()).iloc[0]

    def test_threshold_equality_passes(self):
        row = self.screen()
        self.assertTrue(row.reference_eligible)
        self.assertFalse(row.low_performance_trigger)
        self.assertEqual(row.total_reads_threshold, 25_000_000)

    def test_just_below_threshold_triggers(self):
        row = self.screen(observation(reads=24_999_999))
        self.assertFalse(row.reference_eligible)
        self.assertTrue(row.low_performance_trigger)
        self.assertNotIn("sibling", row.qualification_reasons)

    def test_all_chips_and_lane_counts(self):
        for chip, minimum in {"GX5": 12_500_000, "GX7": 25_000_000, "GX8": 25_000_000, "GX9": 50_000_000}.items():
            for n in range(1, 5):
                with self.subTest(chip=chip, lanes=n):
                    row = self.screen(observation(lane="_".join(map(str, range(1, n + 1))),
                                                  chip=chip, reads=minimum * n))
                    self.assertTrue(row.reference_eligible)
                    self.assertEqual(row.total_reads_threshold, minimum * n)

    def test_lane_aliases_and_duplicate_rejection(self):
        for value in [1, "1", "ChipLane1", "lane1", 1.0]:
            self.assertEqual(lane_tokens(value), (1,))
        self.assertEqual(lane_tokens("ChipLane4_2_1"), (1, 2, 4))
        for value in [True, None, np.nan, "lane_unknown", "1_1", "1_5", "1_2_3_4_5", ""]:
            self.assertIsNone(lane_tokens(value))

    def test_chip_aliases_and_unknown_chip_not_substituted(self):
        for value in ["GX7", "gx7", "GX-7", "Genexus GX7", "GX7v1"]:
            self.assertEqual(normalize_chip_type(value), "GX7")
        for value in ["GX8", "gx8", "GX-8", "GX 8", "Genexus GX8", "GX8v1"]:
            self.assertEqual(normalize_chip_type(value), "GX8")
            row = self.screen(observation(chip=value, reads=25_000_000))
            self.assertEqual(row.chip_type, "GX8")
            self.assertTrue(row.reference_eligible)
        row = self.screen(observation(chip="GX6", reads=500_000_000))
        self.assertFalse(row.reference_eligible)
        self.assertTrue(pd.isna(row.low_performance_trigger))
        self.assertIn("without_user_approved", row.qualification_reasons)

    def test_gx8_all_lane_boundary_values(self):
        for lanes in range(1, 5):
            scope = "_".join(map(str, range(1, lanes + 1)))
            threshold = 25_000_000 * lanes
            for difference in [-1, 0, 1]:
                with self.subTest(lanes=lanes, difference=difference):
                    row = self.screen(observation(lane=scope, chip="GX8", reads=threshold + difference))
                    self.assertEqual(row.total_reads_threshold, threshold)
                    self.assertEqual(bool(row.low_performance_trigger), difference < 0)
                    self.assertEqual(bool(row.reference_eligible), difference >= 0)

    def test_gx8_read_pass_does_not_override_barcode_health_screen(self):
        for counts in [(0, 1000, 1000, 1000), (300, 1000, 1000, 1000),
                       (3000, 1000, 1000, 1000), (np.nan, 1000, 1000, 1000)]:
            with self.subTest(counts=counts):
                row = self.screen(observation(chip="GX8", reads=100_000_000), barcodes(counts=counts))
                self.assertEqual(row.total_read_criterion_status, "passed")
                self.assertFalse(row.reference_eligible)

    def test_explicit_legacy_policy_does_not_acquire_gx8(self):
        legacy = {**DEFAULT_POLICY,
                  "policy_version": "user-chip-lane-qualified-reference-20261027-1.0",
                  "minimum_reads_per_lane": {"GX5": 12_500_000, "GX7": 25_000_000, "GX9": 50_000_000}}
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            policy_path = project / POLICY_FILENAME
            text = json.dumps(legacy)
            policy_path.write_text(text)
            loaded = load_policy(project)
            self.assertNotIn("GX8", loaded["minimum_reads_per_lane"])
            self.assertEqual(loaded["policy_version"], legacy["policy_version"])
            table = build_reference_qualification(pd.DataFrame([observation(chip="GX8", reads=500_000_000)]), barcodes(), loaded)
            self.assertFalse(table.iloc[0].reference_eligible)
            self.assertTrue(pd.isna(table.iloc[0].low_performance_trigger))
            self.assertEqual(policy_path.read_text(), text)

    def test_new_explicit_policy_and_defaults_support_gx8_without_mutation(self):
        self.assertEqual(load_policy()["minimum_reads_per_lane"]["GX8"], 25_000_000)
        loaded = load_policy()
        loaded["minimum_reads_per_lane"]["GX8"] = 1
        self.assertEqual(DEFAULT_POLICY["minimum_reads_per_lane"]["GX8"], 25_000_000)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / POLICY_FILENAME).write_text(json.dumps(DEFAULT_POLICY))
            new_policy = load_policy(project)
            table = build_reference_qualification(pd.DataFrame([observation(chip="GX8", lane="1_2_3_4", reads=100_000_000)]), barcodes(), new_policy)
            self.assertTrue(table.iloc[0].reference_eligible)
            self.assertEqual(table.iloc[0].total_reads_threshold, 100_000_000)
            self.assertEqual(table.iloc[0].qualification_policy_version, DEFAULT_POLICY["policy_version"])

    def test_explicit_override_map_is_not_implicitly_merged(self):
        policy = load_policy(overrides={"minimum_reads_per_lane": {"GX7": 25_000_000}})
        self.assertEqual(policy["minimum_reads_per_lane"], {"GX7": 25_000_000})
        with self.assertRaises(ValueError):
            load_policy(overrides={"minimum_reads_per_lane": {"GX10": 25_000_000}})

    def test_missing_chip_scope_or_reads_unresolved(self):
        for changed in [{CHIP_COLUMN: None}, {"chip_lane": "lane_unknown"}, {"total_reads": np.nan},
                        {"total_reads": -1}, {"total_reads": np.inf}]:
            record = observation(); record.update(changed)
            row = self.screen(record)
            self.assertFalse(row.reference_eligible)
            self.assertTrue(pd.isna(row.low_performance_trigger))

    def test_conflicting_analysis_scope_unresolved(self):
        record = observation(lane="1_2", reads=50_000_000)
        record["analysis_lane_scope"] = "single_lane"
        row = self.screen(record)
        self.assertFalse(row.reference_eligible)
        self.assertIn("conflicting_analysis_lane_scope", row.qualification_reasons)

    def test_measured_active_lanes_do_not_replace_analyzed_scope(self):
        record = observation()
        record["active_lane_count"] = 4
        row = self.screen(record)
        self.assertTrue(row.reference_eligible)
        self.assertEqual(row.evaluated_lane_count, 1)

    def test_dropout_and_both_imbalance_directions_excluded(self):
        for counts, field in [((0, 1000, 1000, 1000), "dropout_barcode_count"),
                              ((300, 1000, 1000, 1000), "partial_imbalance_barcode_count"),
                              ((3000, 1000, 1000, 1000), "overrepresented_barcode_count")]:
            with self.subTest(counts=counts):
                row = self.screen(bars=barcodes(counts=counts))
                self.assertFalse(row.reference_eligible)
                self.assertEqual(row[field], 1)
                self.assertFalse(row.low_performance_trigger)

    def test_dna_and_rna_not_combined(self):
        bars = pd.concat([barcodes(counts=(1_000_000,) * 4),
                          barcodes(counts=(10_000,) * 4, material="RNA").assign(planned_order=lambda x: x.planned_order + 4)])
        self.assertTrue(self.screen(bars=bars).reference_eligible)

    def test_concatenated_material_indices_do_not_cross_contaminate_calls(self):
        bars = pd.concat([barcodes(counts=(0, 1_000_000, 1_000_000, 1_000_000)),
                          barcodes(counts=(10_000,) * 4, material="RNA").assign(planned_order=lambda x: x.planned_order + 4)])
        calls = classify_barcode_outcomes(bars)
        self.assertEqual(calls[calls.material_type.eq("DNA")].outcome_class.eq("dropout").sum(), 1)
        self.assertTrue(calls[calls.material_type.eq("RNA")].outcome_class.eq("surviving").all())

    def test_depressed_median_not_make_survivors_high(self):
        bars = barcodes(counts=(0, 0, 0, 0, 1000, 1000, 1000, 1000))
        calls = classify_barcode_outcomes(bars)
        self.assertEqual(calls.outcome_class.eq("dropout").sum(), 4)
        self.assertEqual(calls.outcome_class.eq("overrepresented").sum(), 0)

    def test_controls_excluded(self):
        bars = barcodes()
        control = bars.iloc[[0]].assign(planned_order=9, read_count=0, is_control=True)
        row = self.screen(bars=pd.concat([bars, control], ignore_index=True))
        self.assertTrue(row.reference_eligible)
        self.assertEqual(row.noncontrol_barcode_count, 4)

    def test_missing_or_negative_barcode_counts_unresolved(self):
        for bad in [np.nan, -1, np.inf]:
            row = self.screen(bars=barcodes(counts=(bad, 1000, 1000, 1000)))
            self.assertFalse(row.reference_eligible)
            self.assertEqual(row.unresolved_barcode_count, 1)

    def test_no_barcodes_never_healthy(self):
        row = self.screen(bars=pd.DataFrame())
        self.assertFalse(row.reference_eligible)
        self.assertEqual(row.barcode_screen_status, "unresolved")

    def test_one_sample_cannot_certify_no_imbalance(self):
        row = self.screen(bars=barcodes(counts=(1000,)))
        self.assertFalse(row.reference_eligible)
        self.assertIn("fewer_than_two_samples", row.qualification_reasons)

    def test_unknown_control_and_material_excluded(self):
        bars = barcodes(); bars["is_control"] = bars["is_control"].astype("boolean"); bars.loc[0, "is_control"] = None
        self.assertFalse(self.screen(bars=bars).reference_eligible)
        bars = barcodes(); bars.loc[0, "material_type"] = "UNKNOWN"
        self.assertFalse(self.screen(bars=bars).reference_eligible)

    def test_duplicate_barcode_rows_not_replication(self):
        bars = barcodes(); bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
        row = self.screen(bars=bars)
        self.assertFalse(row.reference_eligible)
        self.assertIn("duplicate_planned_barcode_rows", row.qualification_reasons)

    def test_bad_sibling_vetoes_historical_run_not_target_flag(self):
        core = pd.DataFrame([observation(), observation("obs2", lane="2", reads=1_000_000)])
        bars = pd.concat([barcodes(), barcodes("obs2")])
        rows = build_reference_qualification(core, bars).set_index("observation_id")
        self.assertFalse(rows.loc["obs1", "reference_eligible"])
        self.assertFalse(rows.loc["obs1", "low_performance_trigger"])
        self.assertTrue(rows.loc["obs2", "low_performance_trigger"])
        self.assertTrue(rows.loc["obs1", "observation_reference_eligible"])
        self.assertIn("failed_sibling", rows.loc["obs1", "qualification_reasons"])

    def test_missing_sibling_vetoes_historical_run(self):
        core = pd.DataFrame([observation(), observation("obs2", lane="2", reads=np.nan)])
        bars = pd.concat([barcodes(), barcodes("obs2")])
        rows = build_reference_qualification(core, bars)
        self.assertFalse(rows.reference_eligible.any())
        self.assertIn("unresolved_sibling", rows.iloc[0].qualification_reasons)

    def test_identical_disjoint_totals_not_assumed_independent(self):
        core = pd.DataFrame([observation(), observation("obs2", lane="2")])
        bars = pd.concat([barcodes(), barcodes("obs2")])
        rows = build_reference_qualification(core, bars)
        self.assertFalse(rows.reference_eligible.any())
        self.assertIn("identical_total_reads_on_disjoint", rows.iloc[0].qualification_reasons)
        self.assertTrue(rows.low_performance_trigger.isna().all())
        self.assertTrue(rows.total_read_criterion_status.eq("unresolved").all())

    def test_overlapping_exports_not_added(self):
        core = pd.DataFrame([observation(), observation("obs2", lane="1_2", reads=50_000_000)])
        bars = pd.concat([barcodes(), barcodes("obs2")])
        rows = build_reference_qualification(core, bars)
        self.assertTrue(rows.reference_eligible.all())
        self.assertEqual(rows.total_reads_threshold.tolist(), [25_000_000, 50_000_000])
        self.assertEqual(rows.total_reads_value.tolist(), [25_000_000, 50_000_000])

    def test_duplicate_observation_ids_rejected(self):
        with self.assertRaises(ValueError):
            build_reference_qualification(pd.DataFrame([observation(), observation()]), barcodes())

    def test_missing_physical_run_id_unresolved(self):
        record = observation(); record["physical_run_id"] = None
        row = self.screen(record)
        self.assertFalse(row.reference_eligible)
        self.assertIn("unresolved_physical_run_id", row.qualification_reasons)

    def test_load_is_readonly_writer_is_explicit_and_new_target_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); (project / "qpm_failure_examples").mkdir()
            pd.DataFrame([observation()]).to_parquet(project / "csa_ml_dataset.parquet", index=False)
            barcodes().to_csv(project / "qpm_failure_examples/dna_rna_barcode_counts.csv", index=False)
            before = set(project.rglob("*"))
            table, metadata = load_reference_qualification(project)
            self.assertEqual(before, set(project.rglob("*")))
            self.assertTrue(table.iloc[0].reference_eligible)
            output = project / "new_report"; write_qualification(output, table, metadata)
            self.assertTrue((output / "csa_reference_qualification.csv").is_file())
            self.assertTrue(json.loads((output / "csa_reference_qualification_manifest.json").read_text())["table_sha256"])
            pd.DataFrame([observation(reads=1)]).to_parquet(project / "csa_ml_dataset.parquet", index=False)
            current, currentmeta = load_reference_qualification(project)
            self.assertFalse(current.iloc[0].reference_eligible)
            self.assertTrue(current.iloc[0].low_performance_trigger)
            self.assertNotEqual(metadata["input_sha256"], currentmeta["input_sha256"])


class QualificationIntegrationTests(unittest.TestCase):
    def test_historical_pool_filters_bad_sources_but_does_not_remove_bad_target(self):
        from qpm_waveform_reference import prepare_reference_pool
        records = [
            dict(observation_id="target", physical_run_id="run0", reference_eligible=False),
            dict(observation_id="sibling", physical_run_id="run0", reference_eligible=True),
            dict(observation_id="good", physical_run_id="run1", reference_eligible=True),
            dict(observation_id="bad", physical_run_id="run2", reference_eligible=False),
            dict(observation_id="unknown", physical_run_id="run3", reference_eligible=pd.NA),
            dict(observation_id="unverified", physical_run_id="run4", reference_eligible=True,
                 command_context_status="unresolved"),
            dict(observation_id="er72", physical_run_id="run5", reference_eligible=True, qpm_has_er72=True),
        ]
        defaults = dict(command_context_status="command_verified", qpm_has_er72=False, qpm_waveform_eligible=True)
        frame = pd.DataFrame([{**defaults, **record} for record in records])
        pool = prepare_reference_pool(frame, frame.iloc[0], {"time_policy": "retrospective"})
        self.assertEqual(pool.observation_id.tolist(), ["good"])
        self.assertIn("target", frame.observation_id.tolist())

    def test_historical_pool_missing_qualification_fails_closed(self):
        from qpm_waveform_reference import prepare_reference_pool
        frame = pd.DataFrame([dict(physical_run_id="other", command_context_status="command_verified")])
        with self.assertRaises(ValueError):
            prepare_reference_pool(frame, pd.Series({"physical_run_id": "target"}), {"time_policy": "retrospective"})

    def test_absolute_trigger_operates_without_peers_and_missing_criterion_stays_unresolved(self):
        from qpm_diagnostic_extensions import diagnostic_trigger
        lineage = pd.DataFrame({"is_control": [False, False], "outcome_class": ["surviving", "surviving"]})
        outcomes = pd.DataFrame([{"status": "insufficient_peers", "outcome": "total_reads"}])
        config = {"low_performance_reference": "q05"}
        low = diagnostic_trigger(lineage, outcomes, config,
                                 {"low_performance_trigger": True, "total_read_criterion_status": "below_threshold"})
        self.assertTrue(low["run_qpm"])
        self.assertTrue(low["below_chip_lane_read_criterion"])
        unknown = diagnostic_trigger(lineage, outcomes, config,
                                     {"low_performance_trigger": None, "total_read_criterion_status": "unresolved"})
        self.assertFalse(unknown["below_chip_lane_read_criterion"])
        self.assertEqual(unknown["chip_lane_read_criterion"]["total_read_criterion_status"], "unresolved")
        self.assertTrue(unknown["missing_outcome_reference_is_not_normal"])


if __name__ == "__main__":
    unittest.main()
