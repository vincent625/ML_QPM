"""Focused leakage and chronology regression tests for expanded outcome models."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from train_expanded_csa_outcomes import FoldSelector, direct_target_copies, extract_one, forbidden, pipeline, qpm_features, splits, transformed_feature_audit


class OutcomeTests(unittest.TestCase):
    def test_forbidden_target_proxies(self):
        for name in ["csa.final_library_reads", "plugin.read_group_metrics.abc", "csa.full.num_reads", "plugin.mean_read_length", "csa.operator", "csa.sample_name", "csa.serial_number", "csa.filename", "csa.exp_json_log_final_reads"]:
            self.assertTrue(forbidden(name), name)
        for name in ["qpm.aspirate.pip1.wave_area_absolute", "libprep.temperature.mean", "csa.flow_count", "plugin.flowrate.flow.mean"]:
            self.assertFalse(forbidden(name), name)

    def test_strict_source_family_and_nested_field_quarantine(self):
        for name in ["plugin.true_loading.image.edge_mean", "plugin.edge_effects.image.fft_entropy", "plugin.lane_diagnostics.structured.results_lane_1_bcmatch_length_mean", "plugin.reseq_diagnostics.image.mean", "plugin.trace_noise.image.mean", "plugin.autocal.structured.results_cal_chip_hist_p90", "plugin.rawtrace.structured.results_pass_filter_mean", "csa.ion_params_00.exp_json_plan", "csa.ion_params_00.exp_json_rep_result", "csa.ion_params_00.experiment_analysis_settings_experiment", "csa.ion_params_00.exp_json_ftp_status", "csa.ion_params_00.exp_json_chef_progress", "csa.ion_params_00.exp_json_log_currentstate", "csa.report_summary.object_quantitation_result_array_length", "csa.ion_params_00.failed_to_reach_temperature2026_10_01_00", "plugin.rawtrace.structured.startplugin_runinfo_plugin_user_input_cellularity_pct", "plugin.flowrate.structured.startplugin_runinfo_flows"]:
            self.assertTrue(forbidden(name),name)
        for name in ["plugin.rawtrace.image.fft_entropy", "plugin.chip_diagnostics.structured.results_gain_mean", "plugin.autocal.structured.results_cal_gain_mean", "plugin.nucstep_rendered.image.edge_mean", "qpm.liquidclass.lc_123.aspirate.pip1.wave_p95", "csa.ion_params_00.exp_json_log_dynamicrangeafterbf"]:
            self.assertFalse(forbidden(name),name)

    def test_selector_heldout_categories_not_learned(self):
        x=pd.DataFrame({"a":np.arange(20.),"b":np.arange(20.),"c":np.ones(20),"kind":["known"]*10+["other"]*10})
        fit=FoldSelector(k=10).fit(x,np.arange(20.))
        self.assertEqual(len([s for s in fit.selected_names_ if s in {"a","b"}]),1)
        before=list(fit.categories_["kind"])
        heldout=pd.DataFrame({"a":[100.],"b":[100.],"c":[5.],"kind":["heldout_only"]})
        self.assertEqual(fit.transform(heldout).shape[0],1)
        self.assertEqual(fit.categories_["kind"],before)
        self.assertNotIn("heldout_only",fit.categories_["kind"])

    def test_context_instruments_not_lost_to_rare_level_filter(self):
        x=pd.DataFrame({"context.instrument":[f"i{i}" for i in range(43) for _ in range(3)],"signal":np.arange(129.)})
        fitted=FoldSelector(k=192).fit(x,np.arange(129.))
        self.assertEqual(len(fitted.categories_["context.instrument"]),43)
        self.assertEqual(len([n for n in fitted.selected_names_ if n.startswith("context.instrument.")]),43)

    def test_group_and_chronological_sibling_protection(self):
        frame=pd.DataFrame([dict(physical_run_id=f"r{i}",instrument_id=f"i{i%3}",run_start_time=f"2026-01-{1+i//2:02d}T00:00:00Z") for i in range(40) for lane in range(2)])
        rows=splits(frame)
        self.assertTrue(any(r[0]=="chronological" for r in rows))
        dates=pd.to_datetime(frame.run_start_time,utc=True)
        for scheme,_,tr,te in rows:
            self.assertFalse(set(frame.iloc[tr].physical_run_id)&set(frame.iloc[te].physical_run_id))
            if scheme=="chronological":
                self.assertLess(dates.iloc[tr].max(),dates.iloc[te].min())

    def test_no_final_reports_or_sensitive_values(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); archive=root/"example.CSA.txz"
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("CSA/ion_params_00.json",json.dumps(dict(flows=800,sample_name="SECRET",operator="SECRET",total_reads=1234)))
                z.writestr("CSA/outputs/BaseCallingActor-00/BaseCaller.json",json.dumps(dict(Filtering=dict(LibraryReport=dict(final_library_reads=1234)),mystery=1234)))
                z.writestr("flowRate/summary.json",json.dumps(dict(flow_rate=1.5,mean_read_length=99)))
            result=extract_one(dict(archive_id="a",archive_sha256="abc",local_path=str(archive),container_format="zip"),root)
            encoded=json.dumps(result)
            self.assertNotIn("SECRET",encoded)
            self.assertNotIn("mystery",encoded)
            self.assertIn("csa.ion_params_00.flows",result["features"])
            self.assertEqual(result["warnings"],[])

    def test_predictive_pipeline_fit(self):
        x=pd.DataFrame(dict(signal=np.arange(40.),noise=np.sin(np.arange(40.))))
        model,_=pipeline("elastic_net")
        model.fit(x,np.arange(40.)*2)
        self.assertEqual(model.predict(x.iloc[:2]).shape,(2,))

    def test_raw_feature_permutation_equivalent_with_missingness_and_categories(self):
        x=pd.DataFrame(dict(signal=np.arange(60,dtype=float),kind=["one","two","three"]*20,noise=np.sin(np.arange(60.))))
        x.loc[::7,"signal"]=np.nan
        y=np.arange(60.)*2+np.sin(np.arange(60.))
        model,_=pipeline("elastic_net");model.fit(x.iloc[:45],y[:45])
        test=x.iloc[45:].copy();prediction=model.predict(test)
        rows,coefficients=transformed_feature_audit(model,test,y[45:],prediction,"total_reads","grouped","one","elastic_net")
        rng=np.random.default_rng(20261027)
        for row in rows:
            deltas=[]
            for _ in range(5):
                shuffled=test.copy();shuffled[row["feature_name"]]=shuffled[row["feature_name"]].iloc[rng.permutation(len(test))].to_numpy()
                deltas.append(float(np.mean(np.abs(y[45:]-model.predict(shuffled))))-float(np.mean(np.abs(y[45:]-prediction))))
            self.assertAlmostEqual(row["heldout_mae_increase"],np.mean(deltas),places=5)
            self.assertEqual(row["permutation_repeats"],5)
        self.assertTrue(coefficients)
        tree,_=pipeline("extra_trees");tree.fit(x.iloc[:45],y[:45]);tree_prediction=tree.predict(test)
        tree_rows,_=transformed_feature_audit(tree,test,y[45:],tree_prediction,"total_reads","grouped","one","extra_trees")
        rng=np.random.default_rng(20261027)
        for row in tree_rows:
            deltas=[]
            for _ in range(5):
                shuffled=test.copy();shuffled[row["feature_name"]]=shuffled[row["feature_name"]].iloc[rng.permutation(len(test))].to_numpy()
                deltas.append(float(np.mean(np.abs(y[45:]-tree.predict(shuffled))))-float(np.mean(np.abs(y[45:]-tree_prediction))))
            self.assertAlmostEqual(row["heldout_mae_increase"],np.mean(deltas),places=8)

    def test_deterministic_outcome_copy_not_ordinary_correlation(self):
        y=np.arange(40.)*13+7
        frame=pd.DataFrame(dict(total_reads=y,mean_read_length_bp=np.nan,raw_read_accuracy_pct=np.nan,
                                hidden_copy=y*.5+3,ordinary_signal=y+np.sin(np.arange(40.))))
        copies=direct_target_copies(frame)
        self.assertIn("hidden_copy",set(copies.feature_name))
        self.assertNotIn("ordinary_signal",set(copies.feature_name))

    def test_qpm_quality_and_verified_liquid_class(self):
        with tempfile.TemporaryDirectory() as td:
            bank=Path(td)/"bank";bank.mkdir();out=Path(td)/"out";out.mkdir()
            rows=[];quality=[];context=[]
            for i,bad in [(1,False),(2,True)]:
                rows.append(dict(archive_id="a",observation_id="o",operation_index=i,pipette="PIP1",operation="aspirate",programmed_volume_ul=10.,qpm_available=True,qpm_profile=list(np.arange(32.)*(100 if bad else 1)),workflow_stage="prep",process_step="test"))
                quality.append(dict(archive_id="a",operation_index=i,pipette="PIP1",qpm_has_er72=bad,qpm_waveform_eligible=not bad,qpm_trace_quality_status="er72_command_error" if bad else "checked_no_er72",qpm_exclusion_reason="Er72" if bad else ""))
                context.append(dict(archive_id="a",operation_index=i,pipette="PIP1",command_operation="aspirate",command_volume_ul=10.,command_context_status="command_verified",liquid_class_fingerprint="fixed_definition"))
            pd.DataFrame(rows).to_parquet(bank/"qpm_operation_waveforms.parquet",index=False)
            pd.DataFrame(quality).to_parquet(bank/"qpm_trace_quality.parquet",index=False)
            pd.DataFrame(context).to_parquet(bank/"qpm_physical_operation_context.parquet",index=False)
            result=qpm_features(bank,out)
            self.assertAlmostEqual(result.iloc[0]["qpm.aspirate.pip1.wave_amplitude.max"],31.)
            self.assertTrue(any("qpm.liquidclass.lc_fixed_defini" in c for c in result))
            manifest=json.loads((out/"qpm_run_descriptors.manifest.json").read_text())
            self.assertIn("trace_quality_sha256",manifest["input_signature"])


if __name__=="__main__":
    unittest.main()
