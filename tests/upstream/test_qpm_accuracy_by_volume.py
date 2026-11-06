import unittest
import numpy as np
import pandas as pd

from plot_qpm_accuracy_by_volume import per_run_errors,summarize_volume_runs,major_volumes,choose_definitions,FAMILIES


class VolumeAccuracyTests(unittest.TestCase):
    def predictions(self):
        rows=[]
        for run,values in [('run1',[10.]*9+[20.]),('run2',[20.])]:
            for n,pred in enumerate(values):
                rows.append(dict(task_id='t',scheme='unseen_instrument',model='Extra Trees',physical_run_id=run,instrument_id=run,
                    programmed_volume_ul=10.,predicted_volume_ul=pred,nearest_training_level_ul=10. if pred==10. else 20.))
        return pd.DataFrame(rows)

    def test_run_weighting_not_repetition_weighting(self):
        result=summarize_volume_runs(per_run_errors(self.predictions()))
        self.assertAlmostEqual(result['accuracy_within_10pct'],.45)
        self.assertAlmostEqual(result['mae_ul'],5.5)
        self.assertEqual(result['physical_runs'],2)
        self.assertEqual(result['waveforms'],11)

    def test_fixed_tolerance_boundaries_and_not_one_minus_mae(self):
        data=self.predictions().iloc[:2].copy()
        data['predicted_volume_ul']=[9.,11.]
        result=summarize_volume_runs(per_run_errors(data))
        self.assertEqual(result['accuracy_within_10pct'],1.)
        self.assertEqual(result['mae_ul'],1.)

    def test_too_few_runs_have_no_interval(self):
        result=summarize_volume_runs(per_run_errors(self.predictions()))
        self.assertTrue(np.isnan(result['accuracy_ci_low']))
        self.assertTrue(result['low_support'])

    def test_bootstrap_is_repeatable_and_within_bounds(self):
        run=per_run_errors(self.predictions())
        frame=pd.concat([run.assign(physical_run_id=run.physical_run_id+'_'+str(i)) for i in range(10)])
        a=summarize_volume_runs(frame);b=summarize_volume_runs(frame)
        self.assertEqual(a,b)
        self.assertGreaterEqual(a['accuracy_ci_low'],0.)
        self.assertLessEqual(a['accuracy_ci_high'],1.)

    def test_display_selection_uses_counts_not_success_and_keeps_extremes(self):
        data=pd.DataFrame(dict(programmed_volume_ul=[1.,2.,5.,10.,20.,50.],physical_runs=[30,90,100,90,40,31],instruments=[3]*6,accuracy_within_10pct=[0,1,1,1,1,0]))
        self.assertEqual(major_volumes(data,4),[1.,2.,5.,50.])
        data['accuracy_within_10pct']=1-data.accuracy_within_10pct
        self.assertEqual(major_volumes(data,4),[1.,2.,5.,50.])

    def test_sparse_display_levels_are_not_promoted(self):
        data=pd.DataFrame(dict(programmed_volume_ul=[1.,10.,100.],physical_runs=[29,40,50],instruments=[5,2,3]))
        self.assertEqual(major_volumes(data),[100.])

    def test_numeric_alias_does_not_mix_blowout_with_transfer(self):
        rows=[]
        for family,name in FAMILIES.items():
            rows += [dict(liquid_class_name=name,liquid_class_fingerprint=str(family)+'main',waveforms=100),
                dict(liquid_class_name=name,liquid_class_fingerprint=str(family)+'rare',waveforms=1)]
        rows.append(dict(liquid_class_name='blowout_postrun',liquid_class_fingerprint='blowout',waveforms=10000))
        selected,inventory=choose_definitions(pd.DataFrame(rows))
        self.assertEqual(len(selected),3)
        self.assertNotIn('blowout',set(selected.liquid_class_fingerprint))
        self.assertEqual(len(inventory),6)


if __name__=='__main__':unittest.main()
