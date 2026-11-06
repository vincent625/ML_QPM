"""Sparse raw curves are visible, without weakening inferential tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qpm_baseline_comparison import (
    MODES, _display_rows, _waveform, display_internal_tests, display_slots,
    summarize_baselines, write_comparison,
)


PALETTE = dict(teal="#2A9D8F", coral="#E76F51", ink="#264653", gold="#E9C46A", gray="#8A918B")


def example(group="g1", failed=1, survivors=7, index=10, eligible=False,
            role="insufficient_replication", step=None):
    rows = []
    wave = -100 * np.sin(np.linspace(0, np.pi, 32))
    for n in range(failed + survivors):
        rows.append(dict(group_id=group, material_type="DNA", planned_order=n+1,
            operation_id=f"{group}_op{n}", operation_index=index+n,
            path_status="failed" if n < failed else "survivor", location=f"well_{n}",
            observed_profile=json.dumps((wave*(2 if n < failed else 1)).tolist()),
            internal_context_verified=True, internal_context_fingerprint="same_verified_context",
            operation="aspirate", pipette="PIP1", programmed_volume_ul=20.,
            pressure_unit="raw_sensor_counts", liquid_class_fingerprint="same_lc",
            container_geometry_fingerprint="same_geometry", deck_geometry_fingerprint="same_deck",
            command_settings_fingerprint="same_commands", qpm_has_er72=False,
            qpm_waveform_eligible=True, qpm_trace_quality_status="checked_no_er72"))
    tests = pd.DataFrame([dict(group_id=group, material_type="DNA", first_operation_index=index,
        last_operation_index=index+max(0, failed+survivors-1),
        step_label=step or f"step_{group}", n_failed=failed, n_survivor=survivors,
        eligible_internal_comparison=eligible, evidence_role=role,
        screen_label="positive" if eligible else "insufficient_replication",
        failed_path_deviation_auc=.99 if eligible else np.nan,
        permutation_p=.01 if eligible else np.nan, bh_fdr_q=.02 if eligible else np.nan)])
    return tests, pd.DataFrame(rows)


class SparseCurveAvailabilityTests(unittest.TestCase):
    def test_one_failed_seven_survivors_available_without_new_statistics(self):
        tests, mapped = example()
        before_tests, before_mapped = tests.copy(deep=True), mapped.copy(deep=True)
        result = display_internal_tests(tests, mapped)
        self.assertTrue(result.curve_available.iloc[0])
        self.assertTrue(result.curve_descriptive_only.iloc[0])
        self.assertEqual((result.curve_n_failed.iloc[0], result.curve_n_survivor.iloc[0]), (1,7))
        pd.testing.assert_frame_equal(result[tests.columns], before_tests)
        pd.testing.assert_frame_equal(tests, before_tests)
        pd.testing.assert_frame_equal(mapped, before_mapped)
        self.assertTrue(result.permutation_p.isna().all())

    def test_one_vs_one_display_is_allowed(self):
        tests, mapped = example(survivors=1)
        result = display_internal_tests(tests, mapped)
        self.assertTrue(result.curve_available.iloc[0])
        self.assertEqual(result.curve_n_survivor.iloc[0], 1)

    def test_no_counterpart_is_explicit_not_fabricated(self):
        for failed, survivors, reason in [(0,7,"no_affected_path"),(1,0,"no_surviving_path")]:
            tests, mapped = example(failed=failed,survivors=survivors)
            result = display_internal_tests(tests,mapped)
            self.assertFalse(result.curve_available.iloc[0])
            self.assertEqual(result.curve_unavailable_reason.iloc[0],reason)

    def test_unknown_outcome_is_not_a_survivor(self):
        tests,mapped=example(survivors=1)
        mapped.loc[mapped.path_status.eq("survivor"),"path_status"]="unresolved"
        result=display_internal_tests(tests,mapped)
        self.assertFalse(result.curve_available.iloc[0])
        self.assertEqual(result.curve_n_survivor.iloc[0],0)
        self.assertEqual(result.curve_ignored_unresolved_operations.iloc[0],1)

    def test_retries_and_duplicate_operations_do_not_inflate_counts(self):
        tests,mapped=example()
        retry=mapped.iloc[:1].copy()
        retry["operation_id"]="new_attempt_same_path"
        combined=pd.concat([mapped,retry,mapped.iloc[:1]],ignore_index=True)
        result=display_internal_tests(tests,combined)
        self.assertTrue(result.curve_available.iloc[0])
        self.assertEqual(result.curve_n_failed.iloc[0],1)

    def test_er72_excluded_path_cannot_supply_affected_curve(self):
        tests,mapped=example()
        mapped.loc[mapped.path_status.eq("failed"),"qpm_has_er72"]=True
        result=display_internal_tests(tests,mapped)
        self.assertFalse(result.curve_available.iloc[0])
        self.assertEqual(result.curve_n_failed.iloc[0],0)

    def test_unresolved_trace_quality_cannot_supply_curves(self):
        tests,mapped=example()
        for frame in [mapped.drop(columns="qpm_trace_quality_status"),
                      mapped.assign(qpm_trace_quality_status="unresolved")]:
            result=display_internal_tests(tests,frame)
            self.assertFalse(result.curve_available.iloc[0])
            self.assertEqual(result.curve_unavailable_reason.iloc[0],"trace_quality_unresolved")

    def test_unknown_configuration_is_not_relaxed_for_sparse_display(self):
        tests,mapped=example()
        for frame in [mapped.assign(internal_context_verified=False),
                      mapped.drop(columns="internal_context_fingerprint")]:
            result=display_internal_tests(tests,frame)
            self.assertFalse(result.curve_available.iloc[0])
            self.assertEqual(result.curve_unavailable_reason.iloc[0],"physical_configuration_unverified")

    def test_mixed_material_or_units_or_geometry_cannot_be_overlaid(self):
        tests,mapped=example()
        for field,value in [("material_type","RNA"),("pressure_unit","kPa"),
                            ("container_geometry_fingerprint","different")]:
            frame=mapped.copy()
            frame.loc[0,field]=value
            result=display_internal_tests(tests,frame)
            self.assertFalse(result.curve_available.iloc[0])
            self.assertEqual(result.curve_unavailable_reason.iloc[0],"mixed_material_or_physical_configuration")

    def test_invalid_raw_profile_is_not_a_zero_curve(self):
        tests,mapped=example()
        for raw in ["[]","[1,2]",json.dumps([0]*31+[float('nan')])]:
            frame=mapped.copy()
            frame.loc[0,"observed_profile"]=raw
            result=display_internal_tests(tests,frame)
            self.assertFalse(result.curve_available.iloc[0])
            self.assertIn("raw_waveform",result.curve_unavailable_reason.iloc[0])

    def test_path_with_conflicting_outcomes_abstains(self):
        tests,mapped=example(survivors=1)
        mapped.loc[1,"planned_order"]=mapped.loc[0,"planned_order"]
        result=display_internal_tests(tests,mapped)
        self.assertFalse(result.curve_available.iloc[0])
        self.assertEqual(result.curve_unavailable_reason.iloc[0],"inconsistent_path_labels_or_raw_waveform")

    def test_missing_mapping_and_identity_have_explicit_reasons(self):
        tests,mapped=example()
        for frame in [pd.DataFrame(),mapped.drop(columns="operation_id")]:
            result=display_internal_tests(tests,frame)
            self.assertFalse(result.curve_available.iloc[0])
            self.assertTrue(result.curve_unavailable_reason.iloc[0])


class SparseDisplaySelectionTests(unittest.TestCase):
    def test_empty_enriched_schema_still_has_five_explicit_unavailable_slots(self):
        tests = display_internal_tests(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(_display_rows(tests,pd.DataFrame(),"within_run"),[])
        slots = display_slots(tests,{})
        self.assertEqual(len(slots),15)
        self.assertFalse(slots.available.any())

    def groups(self):
        pairs=[example(group=f"g{i}",index=10*i,step=f"step_{i//2}") for i in range(1,8)]
        tests=pd.concat([x[0] for x in pairs],ignore_index=True)
        mapped=pd.concat([x[1] for x in pairs],ignore_index=True)
        return tests,mapped

    def test_sparse_fallback_is_chronological_and_never_claims_support(self):
        tests,mapped=self.groups()
        result=display_internal_tests(tests,mapped)
        rows=_display_rows(result,pd.DataFrame(),"within_run")
        self.assertEqual(len(rows),5)
        self.assertEqual(rows[0].first_operation_index,10)
        self.assertEqual([r.first_operation_index for r in rows],sorted(r.first_operation_index for r in rows))
        self.assertTrue(all(r.evidence_role=="descriptive_insufficient_replication" for r in rows))
        self.assertTrue(result.evidence_role.eq("insufficient_replication").all())
        slots=display_slots(result,{})
        shown=slots[slots.baseline_id.eq("within_run") & slots.available]
        self.assertEqual(len(shown),5)
        self.assertTrue(shown.curve_descriptive_only.all())
        self.assertTrue(shown.interpretation.eq("illustrative_not_support").all())
        for mode in MODES:self.assertEqual(slots.loc[slots.baseline_id.eq(mode),"slot"].tolist(),[1,2,3,4,5])

    def test_qualified_chain_is_unchanged_and_not_padded_with_sparse_rows(self):
        pairs=[example(group="sparse",index=1),
               example(group="initial",failed=3,survivors=3,index=20,eligible=True,role="initial_candidate"),
               example(group="support",failed=3,survivors=3,index=30,eligible=True,role="downstream_support")]
        tests=pd.concat([p[0] for p in pairs],ignore_index=True)
        mapped=pd.concat([p[1] for p in pairs],ignore_index=True)
        result=display_internal_tests(tests,mapped)
        rows=_display_rows(result,pd.DataFrame(),"within_run")
        self.assertEqual([r.group_id for r in rows],["initial","support"])
        self.assertEqual([r.evidence_role for r in rows],["initial_candidate","downstream_support"])

    def test_one_survivor_has_individual_curves_and_medians_but_no_range_band(self):
        tests,mapped=example(survivors=1)
        result=display_internal_tests(tests,mapped)
        row=_display_rows(result,pd.DataFrame(),"within_run")[0]
        fig,ax=plt.subplots()
        self.addCleanup(plt.close,fig)
        _waveform(ax,row,"within_run",mapped,PALETTE,0)
        self.assertEqual(len(ax.lines),4)
        self.assertEqual(len(ax.collections),0)
        self.assertIn("Descriptive only",ax.get_title(loc="left"))
        labels=[t.get_text() for t in ax.texts]
        self.assertTrue(any("AUC / p / BH q not evaluated" in text for text in labels))
        self.assertTrue(any("1 affected / 1 surviving" in text for text in labels))

    def test_two_survivors_show_raw_paths_without_percentile_band(self):
        tests,mapped=example(survivors=2)
        result=display_internal_tests(tests,mapped)
        row=_display_rows(result,pd.DataFrame(),"within_run")[0]
        fig,ax=plt.subplots()
        self.addCleanup(plt.close,fig)
        _waveform(ax,row,"within_run",mapped,PALETTE,0)
        self.assertEqual(len(ax.lines),5)
        self.assertEqual(len(ax.collections),0)

    def test_write_exports_original_tests_and_additive_curve_coverage(self):
        tests,mapped=example()
        lineage=pd.DataFrame({"path_status":["failed","survivor"]})
        empty=pd.DataFrame()
        with tempfile.TemporaryDirectory() as directory, patch("qpm_baseline_comparison.plot_comparison",return_value={}):
            output=Path(directory)
            payload=write_comparison(output,lineage,{"DNA":tests},{"DNA":mapped},empty,
                empty,empty,empty,{},empty,empty,empty,{},
                dict(shared_reference_minimum_percentile=.95,shared_reference_minimum_outside_fraction=.25),PALETTE)
            saved=pd.read_csv(output/"qpm_baseline_within_run_steps.csv")
            pd.testing.assert_frame_equal(saved,tests,check_dtype=False)
            coverage=pd.read_csv(output/"qpm_same_run_curve_coverage.csv")
            self.assertTrue(coverage.curve_available.iloc[0])
            self.assertEqual(coverage.curve_n_failed.iloc[0],1)
            summary=pd.read_csv(output/"qpm_baseline_comparison_summary.csv")
            within=summary[summary.baseline_id.eq("within_run")].iloc[0]
            self.assertEqual(within.testable_groups,0)
            self.assertEqual(within.positive_groups,0)
            self.assertEqual(within.available_curve_groups,1)
            self.assertTrue(pd.isna(within.earliest_operation_index))
            self.assertTrue(payload["same_run_curve_display"]["statistical_replication_thresholds_unchanged"])


if __name__=="__main__":unittest.main()
