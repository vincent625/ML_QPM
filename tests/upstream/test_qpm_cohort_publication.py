import unittest
import pandas as pd

from qpm_cohort_publication import reconcile_historical_label


class CohortPublicationTests(unittest.TestCase):
    def displayed(self):
        return pd.DataFrame([
            dict(baseline_id="same_instrument", available=True, operation_index=50, step="Loading dispense"),
            dict(baseline_id="cross_instrument", available=True, operation_index=20, step="Templating aspiration"),
            dict(baseline_id="within_run", available=True, operation_index=10, step="Borderline comparison")])

    def test_independent_cross_signal_not_lost_when_shared_scan_is_negative(self):
        row=dict(diagnostic_status="no_qualifying_qpm_divergence",cross_instrument_positive_operations=3,
                 cross_instrument_earliest_operation_index=20,flagged=True)
        result=reconcile_historical_label(row,pd.DataFrame(),self.displayed())
        self.assertEqual(result["diagnostic_status"],"historical_reference_only_anomaly")
        self.assertEqual(result["earliest_operation_index"],20)
        self.assertIn("normalized shape",result["earliest_step"])
        self.assertEqual(row["diagnostic_status"],"no_qualifying_qpm_divergence")

    def test_shared_role_is_authoritative_not_nonexistent_boolean(self):
        shared=pd.DataFrame([dict(operation_index=42,evidence_role="earliest_reference_deviation",workflow_stage="Templating",process_step="Transfer",operation="dispense")])
        result=reconcile_historical_label(dict(diagnostic_status="no_qualifying_qpm_divergence"),shared,self.displayed())
        self.assertEqual(result["diagnostic_status"],"historical_reference_only_anomaly")
        self.assertEqual(result["earliest_operation_index"],42)

    def test_displayed_borderline_is_not_promoted_without_a_positive_flag(self):
        result=reconcile_historical_label(dict(diagnostic_status="no_qualifying_qpm_divergence",same_instrument_positive_operations=0),pd.DataFrame(),self.displayed())
        self.assertEqual(result["diagnostic_status"],"no_qualifying_qpm_divergence")

    def test_preserve_barcode_selective_priority_over_earlier_generic_anomaly(self):
        row=dict(diagnostic_status="within_run_selective_divergence",earliest_operation_index=100,
                 same_instrument_positive_operations=10,same_instrument_earliest_operation_index=50)
        self.assertEqual(reconcile_historical_label(row,pd.DataFrame(),self.displayed()),row)

    def test_same_and_cross_comparators_are_named_not_averaged(self):
        row=dict(diagnostic_status="no_qualifying_qpm_divergence",same_instrument_positive_operations=2,
                 same_instrument_earliest_operation_index=50,cross_instrument_positive_operations=3,
                 cross_instrument_earliest_operation_index=20)
        result=reconcile_historical_label(row,pd.DataFrame(),self.displayed())
        self.assertEqual(result["earliest_operation_index"],20)
        self.assertIn("not barcode-selective",result["evidence_basis"])

    def test_unflagged_sibling_is_context_only(self):
        row=dict(diagnostic_status="no_qualifying_qpm_divergence",flagged=False,
                 same_instrument_positive_operations=2,same_instrument_earliest_operation_index=50)
        result=reconcile_historical_label(row,pd.DataFrame(),self.displayed())
        self.assertTrue(result["evidence_basis"].startswith("Unflagged sibling scope"))


if __name__ == "__main__":
    unittest.main()
