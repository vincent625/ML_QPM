import json
import re
import shutil
import subprocess
import unittest
import numpy as np
import pandas as pd

from screen_all_csa_runs import assay_context,script_json,summarize_runs,barcode_material_summary,HTML
from qpm_reference_qualification import classify_barcode_outcomes


class CohortScreenTests(unittest.TestCase):
    def obs(self,scope='1',reads=10e6,status='below_threshold'):
        return dict(physical_run_id='run1',observation_id='obs'+scope,instrument_id='inst1',instrument='gnxs-0001',
            run_start_time='2026-10-01T00:00:00Z',run_name='run name',chip_type='GX7',chip_lane=scope,
            assay_panel_label='panel',assay_type='DNA',workflow_script='workflow',qpm_report='',
            total_reads_value=reads,total_reads_threshold=25e6,total_read_criterion_status=status,
            reads_fraction_of_threshold=reads/25e6,barcode_screen_reasons='',
            dropout_barcode_count=0,partial_imbalance_barcode_count=0,overrepresented_barcode_count=0,
            **{m+'_'+kind+'_count':0 for m in ['dna','rna'] for kind in ['dropout','partial_imbalance','overrepresented']})

    def test_physical_runs_not_lane_rows_and_no_read_sum(self):
        a,b=self.obs(),self.obs('2',40e6,'passed')
        result=summarize_runs(pd.DataFrame([a,b])).iloc[0]
        self.assertTrue(result.has_low_read_scope)
        self.assertEqual(result.retained_observation_scopes,2)
        self.assertEqual(result.available_lane_tokens,'1_2')
        self.assertIn('10.00M',result.reads_by_scope)
        self.assertIn('40.00M',result.reads_by_scope)
        self.assertNotIn('total_reads',result.index)

    def test_overlap_is_flagged_and_not_added(self):
        a,b=self.obs('1_2'),self.obs('1')
        result=summarize_runs(pd.DataFrame([a,b])).iloc[0]
        self.assertTrue(result.overlapping_analyzed_scopes)
        self.assertEqual(result.available_lane_tokens,'1_2')

    def test_unknown_is_not_passing(self):
        a=self.obs(status='unresolved');a['barcode_screen_reasons']='no_planned_barcode_rows'
        result=summarize_runs(pd.DataFrame([a])).iloc[0]
        self.assertFalse(result.flagged)
        self.assertTrue(result.screen_incomplete)
        self.assertEqual(result.review_status,'unresolved_no_known_flag')

    def test_known_flag_with_unknown_fields_remains_flagged(self):
        a=self.obs();a['barcode_screen_reasons']='missing_or_invalid_sample_read_counts'
        result=summarize_runs(pd.DataFrame([a])).iloc[0]
        self.assertTrue(result.flagged)
        self.assertEqual(result.review_status,'flagged_with_unresolved_fields')

    def test_material_medians_not_pooled(self):
        bars=pd.DataFrame([dict(observation_id='o',material_type=material,is_control=False,read_count=reads,planned_order=i)
            for material,reads in [('DNA',1e6),('RNA',1e4)] for i in range(1,5)])
        classified=classify_barcode_outcomes(bars)
        self.assertTrue(classified.outcome_class.eq('surviving').all())
        table=barcode_material_summary(classified,['o']).iloc[0]
        self.assertEqual(table.dna_dropout_count,0)
        self.assertEqual(table.rna_dropout_count,0)

    def test_assay_uses_recorded_identity_not_workflow_guess(self):
        r=assay_context({'global.analysis_application_type':'DNA','planned_run.object_script_name':'workflow'})
        self.assertEqual(r['assay_panel_label'],'Not recorded')
        r=assay_context({'global.analysis_regions_file':'/share/apps/IR/AssayImport/Assay-v2/workflow/Bed/panel.bed'})
        self.assertEqual(r['assay_panel_label'],'Assay-v2')
        self.assertEqual(r['panel_files'],'panel.bed')
        self.assertNotIn('/share',json.dumps(r))

    def test_script_json_is_safe_and_lossless(self):
        value={'name':'x </script> <b> & real'}
        encoded=script_json(value)
        self.assertNotIn('<',encoded)
        self.assertEqual(json.loads(encoded),value)

    @unittest.skipUnless(shutil.which('node'),'Node is optional')
    def test_report_javascript_parses(self):
        script=re.findall(r'<script>(.*?)</script>',HTML,re.S)[0]
        result=subprocess.run(['node','--check'],input=script,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
