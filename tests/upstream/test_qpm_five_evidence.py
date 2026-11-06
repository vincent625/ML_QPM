"""Five display slots must not manufacture support or change statistical roles."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from qpm_baseline_comparison import (_display_rows, display_slots, later_evidence,
                                     select_path_reference_display, MODES)
from qpm_dropout_plugin import select_shared_display, make_figure, DEFAULT_CONFIG


def history(indices=range(10, 100, 10)):
    return pd.DataFrame([dict(operation_index=i, operation_id=f"op{i}",
                              process_step=f"step{i // 20}", reference_positive=True,
                              evidence_role="earliest_covered_reference_anomaly" if i == 10 else "later_reference_anomaly",
                              planned_order=3) for i in indices])


def internal():
    return pd.DataFrame([dict(first_operation_index=i, group_id=f"group{i}", step_label=f"step{i}",
                              family=f"family{i}", eligible_internal_comparison=True,
                              evidence_role="initial_candidate" if i == 10 else "downstream_support",
                              screen_label="positive") for i in range(10, 90, 10)])


class FiveEvidenceTests(unittest.TestCase):
    def test_history_caps_at_five_and_keeps_true_earliest(self):
        rows = _display_rows(pd.DataFrame(), history().sample(frac=1, random_state=42), "cross_instrument")
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].operation_index, 10)
        indices = [r.operation_index for r in rows]
        self.assertEqual(indices, sorted(set(indices)))

    def test_later_selection_favors_distinct_steps_before_repeats(self):
        frame = history()
        frame.loc[frame.operation_index.lt(50), "process_step"] = "first"
        rows = later_evidence(frame.iloc[0], frame, "operation_index", "operation_id", "process_step", limit=2)
        self.assertEqual([r.operation_index for r in rows], [50, 60])

    def test_duplicate_operations_cannot_fill_slots(self):
        frame = pd.concat([history([10, 20]), history([10, 20])], ignore_index=True)
        rows = _display_rows(pd.DataFrame(), frame, "same_instrument")
        self.assertEqual([r.operation_index for r in rows], [10, 20])

    def test_no_positives_returns_only_labeled_comparison_not_five_claims(self):
        frame = history()
        frame["reference_positive"] = False
        frame["evidence_role"] = "not_reference_positive"
        rows = _display_rows(pd.DataFrame(), frame, "same_instrument")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].reference_positive)

    def test_qualified_internal_chain_is_not_padded_with_borderlines(self):
        frame = internal()
        frame.loc[frame.first_operation_index.gt(20), "evidence_role"] = "not_selective"
        frame.loc[frame.first_operation_index.gt(20), "screen_label"] = "borderline"
        rows = _display_rows(frame, pd.DataFrame(), "within_run")
        self.assertEqual([r.first_operation_index for r in rows], [10, 20])

    def test_internal_illustrations_stay_illustrative(self):
        frame = internal()
        frame["evidence_role"] = "not_selective"
        frame["screen_label"] = "borderline"
        slots = display_slots(frame, {})
        shown = slots[slots.baseline_id.eq("within_run") & slots.available]
        self.assertEqual(len(shown), 5)
        self.assertTrue(shown.interpretation.eq("illustrative_not_support").all())
        self.assertFalse(shown.evidence_role.eq("initial_candidate").any())

    def test_all_modes_have_five_explicit_slots_when_unavailable(self):
        slots = display_slots(pd.DataFrame(), {})
        self.assertEqual(len(slots), 15)
        for mode in MODES:
            self.assertEqual(slots[slots.baseline_id.eq(mode)].slot.tolist(), [1, 2, 3, 4, 5])
        self.assertFalse(slots.available.any())

    def test_reference_fallback_never_borrows_another_barcode(self):
        frame = history()
        frame.loc[frame.operation_index.gt(30), "planned_order"] = 8
        rows = select_path_reference_display(frame)
        self.assertEqual([r.operation_index for r in rows], [10, 20, 30])
        self.assertTrue(all(r.planned_order == 3 for r in rows))

    def test_shared_selector_never_inserts_prior_or_reference_like_phase(self):
        frame = history()
        earlier = frame.iloc[:1].copy()
        earlier["operation_index"] = 9
        earlier["operation_id"] = "prior_pair"
        earlier["evidence_role"] = "not_reference_positive"
        earlier["reference_positive"] = False
        rows = select_shared_display(pd.concat([earlier, frame], ignore_index=True))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].operation_index, 10)
        self.assertTrue(all(r.reference_positive for r in rows))

    def test_primary_figure_has_five_waveform_calls_despite_legacy_config(self):
        frame = internal()
        config = dict(DEFAULT_CONFIG, support_waveform_panels=2)
        with tempfile.TemporaryDirectory() as temp, \
             patch("qpm_dropout_plugin.plot_barcode_outcomes"), \
             patch("qpm_dropout_plugin.plot_process_logic"), \
             patch("qpm_dropout_plugin.plot_auc_scan"), \
             patch("qpm_dropout_plugin.plot_group_waveforms") as waveform, \
             patch("matplotlib.figure.Figure.savefig"):
            make_figure(Path(temp), "test", pd.DataFrame(), "DNA", pd.DataFrame(), frame, config)
            self.assertEqual(waveform.call_count, 5)
            self.assertIn("Evidence 1/5", waveform.call_args_list[0].args[3])
            self.assertIn("Evidence 5/5", waveform.call_args_list[-1].args[3])


if __name__ == "__main__":
    unittest.main()
