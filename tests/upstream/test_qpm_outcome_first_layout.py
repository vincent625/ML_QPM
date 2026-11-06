"""Outcome-first presentation must not change calls or evidence selection."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from qpm_baseline_comparison import (
    MODES, _display_rows, display_internal_tests, figure_run_header, html_section,
    plot_barcode_audit, plot_comparison, write_comparison,
)
from test_qpm_sparse_display import example


PALETTE = dict(teal="#2A9D8F", coral="#E76F51", ink="#264653", gold="#E9C46A",
               gray="#8A918B", blue="#457B9D", rna="#8F5D8A", orange="#F4A261")


def barcode_rows():
    return pd.DataFrame([
        dict(material_type="DNA", planned_order=2, read_count=151637., is_control=False,
             path_status="failed", apparent_dropout=True),
        dict(material_type="DNA", planned_order=4, read_count=15708488., is_control=False,
             path_status="survivor", apparent_dropout=False),
        dict(material_type="DNA", planned_order=6, read_count=17699563., is_control=False,
             path_status="survivor", apparent_dropout=False),
        dict(material_type="RNA", planned_order=11, read_count=590123., is_control=False,
             path_status="survivor", apparent_dropout=False),
        dict(material_type="DNA", planned_order=99, read_count=1., is_control=True,
             path_status="failed", apparent_dropout=True),
    ])


def history(offset):
    return pd.DataFrame([
        dict(operation_index=offset + i*10, operation_id=f"h{offset}_{i}",
             process_step=f"step_{i}", reference_positive=True,
             qpm_anomaly_percentile=.99, evidence_role="later_reference_anomaly",
             qpm_has_er72=False, qpm_waveform_eligible=True)
        for i in range(5)
    ])


def summary_rows():
    return pd.DataFrame([
        dict(baseline_id=mode, baseline=mode, status="unresolved", testable_groups=0,
             operation_groups=5, positive_groups=0, scored_operations=5,
             target_operations=5, positive_operations=5, reference_physical_runs=8,
             reference_instruments=1 if mode == "same_instrument" else 2)
        for mode in MODES
    ])


class BarcodeAuditRenderingTests(unittest.TestCase):
    def axis(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        return fig, ax

    def test_single_dropout_and_recorded_reads_stay_visible(self):
        _, ax = self.axis()
        frame = barcode_rows()
        before = frame.copy(deep=True)
        plot_barcode_audit(ax, frame, PALETTE)
        self.assertEqual(len(ax.patches), 4)
        np.testing.assert_allclose([p.get_height() for p in ax.patches],
                                   [151637, 15708488, 17699563, 590123])
        self.assertEqual(ax.get_yscale(), "log")
        self.assertTrue(any("151,637" in t.get_text() for t in ax.texts))
        self.assertTrue(any("1 affected" in t.get_text() for t in ax.texts))
        self.assertNotEqual(ax.patches[0].get_linewidth(), ax.patches[1].get_linewidth())
        pd.testing.assert_frame_equal(frame, before)

    def test_controls_never_appear_as_samples_or_affected_paths(self):
        _, ax = self.axis()
        plot_barcode_audit(ax, barcode_rows(), PALETTE)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertNotIn("DNA 99", labels)
        self.assertEqual(labels, ["DNA 2", "DNA 4", "DNA 6", "RNA 11"])
        self.assertTrue(any("1 controls excluded" in t.get_text() for t in ax.texts))

    def test_dna_and_rna_keep_separate_material_colors(self):
        _, ax = self.axis()
        plot_barcode_audit(ax, barcode_rows(), PALETTE)
        self.assertNotEqual(ax.patches[0].get_facecolor(), ax.patches[-1].get_facecolor())
        labels = ax.get_legend_handles_labels()[1]
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertTrue({"DNA", "RNA"}.issubset({t.get_text() for t in legend.texts}))

    def test_missing_reads_are_unresolved_not_zero(self):
        _, ax = self.axis()
        frame = barcode_rows().iloc[:2].copy()
        frame.loc[0, "read_count"] = np.nan
        frame.loc[0, "path_status"] = "unresolved"
        plot_barcode_audit(ax, frame, PALETTE)
        self.assertEqual(len(ax.patches), 1)
        self.assertEqual(ax.patches[0].get_height(), 15708488)
        self.assertTrue(any("unresolved" in t.get_text().lower() for t in ax.texts))

    def test_recorded_zero_remains_explicit_zero_on_log_scale(self):
        _, ax = self.axis()
        frame = barcode_rows().iloc[:1].assign(read_count=0.)
        plot_barcode_audit(ax, frame, PALETTE)
        self.assertEqual(ax.patches[0].get_height(), .5)
        self.assertIn("0", [t.get_text() for t in ax.texts])

    def test_absent_outcomes_have_explicit_unresolved_panel(self):
        for frame in [None, pd.DataFrame(), pd.DataFrame({"path_status": ["failed"]})]:
            with self.subTest(frame_type=type(frame).__name__):
                _, ax = self.axis()
                plot_barcode_audit(ax, frame, PALETTE)
                self.assertEqual(len(ax.patches), 0)
                self.assertTrue(any("unresolved" in t.get_text().lower() for t in ax.texts))

    def test_control_only_input_does_not_imply_sample_dropout(self):
        _, ax = self.axis()
        plot_barcode_audit(ax, barcode_rows().query("is_control == True"), PALETTE)
        self.assertEqual(len(ax.patches), 0)
        self.assertTrue(any("non-control" in t.get_text() for t in ax.texts))


class OutcomeFirstCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        pairs = [example(group=f"g{i}", index=10*i) for i in range(1, 6)]
        self.raw_tests = pd.concat([p[0] for p in pairs], ignore_index=True)
        self.mapped = pd.concat([p[1] for p in pairs], ignore_index=True)
        self.tests = display_internal_tests(self.raw_tests, self.mapped)
        self.histories = {"same_instrument": history(100), "cross_instrument": history(300)}

    def compose(self):
        snapshots = []
        destinations = []

        def save(fig, filename, *args, **kwargs):
            fig.canvas.draw()
            snapshots.append([(ax.get_position().frozen(), len(ax.patches)) for ax in fig.axes])
            destinations.append(Path(filename).suffix)

        with patch.object(Figure, "savefig", autospec=True, side_effect=save), \
                patch("qpm_baseline_comparison._waveform") as waveform:
            selected = plot_comparison(self.output, summary_rows(), self.tests, self.mapped,
                                       self.histories, PALETTE, lineage=barcode_rows())
        return snapshots, destinations, selected, waveform.call_args_list

    def test_full_width_outcome_panel_is_above_three_scans_and_five_evidence_rows(self):
        snapshots, destinations, _, calls = self.compose()
        axes = snapshots[0]
        self.assertEqual(len(axes), 19)
        top, bar_count = axes[0]
        self.assertEqual(bar_count, 4)
        self.assertTrue(all(top.y0 > box.y1 for box, _ in axes[1:]))
        self.assertGreater(top.width, 2.5 * axes[1][0].width)
        scan_boxes = [item[0] for item in axes[1:4]]
        self.assertLess(scan_boxes[0].x0, scan_boxes[1].x0)
        self.assertLess(scan_boxes[1].x0, scan_boxes[2].x0)
        for row in range(5):
            previous = axes[1 + 3*row][0]
            current = axes[1 + 3*(row + 1)][0]
            self.assertGreater(previous.y0, current.y1)
        self.assertEqual(len(calls), 15)
        self.assertEqual(set(destinations), {".png", ".pdf", ".svg"})

    def test_independent_column_selections_and_inference_inputs_unchanged(self):
        before_tests, before_mapped = self.tests.copy(deep=True), self.mapped.copy(deep=True)
        before_history = {mode: frame.copy(deep=True) for mode, frame in self.histories.items()}
        _, _, selected, calls = self.compose()
        for mode in MODES:
            expected = _display_rows(self.tests, self.histories.get(mode, pd.DataFrame()), mode)
            field = "first_operation_index" if mode == "within_run" else "operation_index"
            self.assertEqual([r["operation_index"] for r in selected[mode]],
                             [int(row[field]) for row in expected])
            self.assertEqual(len(selected[mode]), 5)
        self.assertNotEqual(selected["same_instrument"], selected["cross_instrument"])
        pd.testing.assert_frame_equal(self.tests, before_tests)
        pd.testing.assert_frame_equal(self.mapped, before_mapped)
        for mode in self.histories:
            pd.testing.assert_frame_equal(self.histories[mode], before_history[mode])
        self.assertTrue(self.tests[["failed_path_deviation_auc", "permutation_p", "bh_fdr_q"]].isna().all().all())

    def test_long_abstention_notes_remain_complete_without_collapsing_layout(self):
        reason = ("target_physical_run_present_in_model_training_use_heldout_predictions_"
                  "and_no_compatible_independent_measured_volume_calibration_"
                  "do_not_interpret_nominal_signature_as_actual_over_or_under_transfer")
        curve = np.linspace(0, -100, 32)
        histories = {}
        for mode, original in self.histories.items():
            frame = original.copy(deep=True)
            for key, value in dict(
                reference_observed_profile=json.dumps(curve.tolist()),
                reference_median_profile=json.dumps((curve*.9).tolist()),
                reference_q05_profile=json.dumps((curve*1.1).tolist()),
                reference_q95_profile=json.dumps((curve*.8).tolist()),
                operation="aspirate", programmed_volume_ul=20., pipette="PIP1",
                liquid_class_index=55, location="reaction_well_A1",
                workflow_stage="library_preparation", reference_physical_runs=8,
                outside_90pct_envelope_fraction=.7, matched_reference_level="same_step_location",
                waveform_equivalent_volume_ul=np.nan,
                volume_proxy_status="unavailable:"+reason,
            ).items():
                frame[key] = value
            histories[mode] = frame
        snapshots = []

        def capture(fig, filename, *args, **kwargs):
            if snapshots:
                return
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            notes = []
            for ax in fig.axes[1:]:
                for annotation in ax.texts:
                    if "Volume estimate unavailable" in annotation.get_text():
                        notes.append(annotation.get_text())
                        self.assertFalse(annotation.get_in_layout())
                        self.assertLessEqual(max(map(len, annotation.get_text().splitlines())), 52)
                        self.assertLessEqual(annotation.get_window_extent(renderer).width,
                                             ax.get_window_extent(renderer).width)
            snapshots.append(notes)

        with warnings.catch_warnings(record=True) as seen, \
                patch.object(Figure, "savefig", autospec=True, side_effect=capture):
            warnings.simplefilter("always")
            plot_comparison(self.output, summary_rows(), self.tests, self.mapped,
                            histories, PALETTE, lineage=barcode_rows())
        self.assertEqual(len(snapshots[0]), 10)
        complete_reason = reason.replace("_", " ")
        for note in snapshots[0]:
            self.assertIn(complete_reason, " ".join(note.split()))
        self.assertFalse(any("constrained_layout not applied" in str(item.message).lower()
                             for item in seen))

    def test_header_shows_recorded_run_details_and_safe_fallback(self):
        self.assertIn(self.output.name, figure_run_header(self.output))
        metadata = dict(run_date="2026-10-06", instrument="GNXS-test", chip_type="GX9",
                        chip_lane="2", assay_type="Test DNA/RNA assay", observation_id="obs_test")
        (self.output / "run_metadata.json").write_text(json.dumps(metadata))
        title = figure_run_header(self.output)
        for value in metadata.values():
            self.assertIn(value, title)
        (self.output / "run_metadata.json").write_text("not JSON")
        self.assertIn(self.output.name, figure_run_header(self.output))

    def test_writer_records_outcome_first_layout_without_modifying_statistical_table(self):
        empty = pd.DataFrame()
        with patch("qpm_baseline_comparison.plot_comparison", return_value={}) as renderer:
            payload = write_comparison(
                self.output, barcode_rows(), {"DNA": self.raw_tests}, {"DNA": self.mapped},
                empty, empty, empty, empty, {}, empty, empty, empty, {},
                dict(shared_reference_minimum_percentile=.95,
                     shared_reference_minimum_outside_fraction=.25), PALETTE)
        actual = pd.read_csv(self.output / "qpm_baseline_within_run_steps.csv")
        pd.testing.assert_frame_equal(actual, self.raw_tests, check_dtype=False)
        layout = payload["outcome_first_layout"]
        self.assertTrue(layout["barcode_outcomes_first"])
        self.assertTrue(layout["controls_excluded"])
        self.assertTrue(layout["evidence_selection_unchanged"])
        self.assertEqual((layout["scan_columns"], layout["evidence_rows"], layout["total_panels"]), (3, 5, 19))
        self.assertTrue(payload["figure_svg"].endswith(".svg"))
        supplied = renderer.call_args.kwargs.get("lineage", renderer.call_args.args[-1])
        pd.testing.assert_frame_equal(supplied, barcode_rows())

    def test_html_places_combined_figure_before_long_summary_tables(self):
        payload = dict(summary=summary_rows().to_dict("records"), figure_png="placeholder.png")
        with patch.object(Path, "read_bytes", return_value=b"image"), \
                patch("qpm_baseline_comparison.pd.read_parquet", return_value=pd.DataFrame()):
            rendered = html_section(self.output, payload)
        self.assertLess(rendered.index("<img"), rendered.index("<table"))
        self.assertIn("qpm_baseline_comparison.svg", rendered)


if __name__ == "__main__":
    unittest.main()
