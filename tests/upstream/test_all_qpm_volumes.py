"""Small-data regression checks for the all-event nominal-volume study."""
import unittest
import numpy as np
import pandas as pd

from train_all_qpm_volumes import (training_weights, evaluation_weights, fit_model,
    predict_model, group_splits, outer_splits, nearest_level, metrics, context_baseline)


class AllQPMVolumeTests(unittest.TestCase):
    def frame(self, runs=24):
        rows=[]
        for r in range(runs):
            for volume,repeats in [(10.,1),(20.,4),(40.,2)]:
                for repeat in range(repeats):
                    rows.append(dict(physical_run_id=f"r{r:03d}",instrument_id=f"i{r%3}",
                        operation_id=f"r{r}_{volume}_{repeat}",programmed_volume_ul=volume,
                        run_start_time=pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(days=r),
                        container_geometry_fingerprint="cg",deck_geometry_fingerprint="dg",workflow_stage="prep"))
        return pd.DataFrame(rows)

    def test_training_weights_equalize_runs_and_nominal_levels(self):
        f=self.frame();w=training_weights(f)
        by_run=f.assign(weight=w).groupby("physical_run_id").weight.sum()
        self.assertTrue(np.allclose(by_run,by_run.iloc[0]))
        by_level=f.assign(weight=w).groupby(["physical_run_id","programmed_volume_ul"]).weight.sum()
        self.assertTrue(np.allclose(by_level,by_level.iloc[0]))

    def test_evaluation_has_one_total_vote_per_physical_run(self):
        f=self.frame();w=evaluation_weights(f)
        self.assertTrue(np.allclose(f.assign(weight=w).groupby("physical_run_id").weight.sum(),1.))

    def test_splits_respect_physical_runs_instruments_and_time(self):
        f=self.frame()
        splits=outer_splits(f)
        self.assertEqual(sum(s=='grouped_run' for s,*_ in splits),5)
        self.assertEqual(sum(s=='unseen_instrument' for s,*_ in splits),3)
        self.assertEqual(sum(s=='chronological' for s,*_ in splits),1)
        for scheme,_,tr,te in splits:
            self.assertFalse(set(f.iloc[tr].physical_run_id)&set(f.iloc[te].physical_run_id))
            if scheme=='unseen_instrument':
                self.assertFalse(set(f.iloc[tr].instrument_id)&set(f.iloc[te].instrument_id))
            if scheme=='chronological':
                self.assertLess(f.iloc[tr].run_start_time.max(),f.iloc[te].run_start_time.min())

    def test_no_event_sampling_or_averaging_in_grouped_evaluation(self):
        f=self.frame();splits=group_splits(f)
        observed=np.concatenate([te for _,_,te in splits])
        np.testing.assert_array_equal(np.sort(observed),np.arange(len(f)))

    def test_regressors_predict_from_waveforms_only(self):
        f=self.frame(10);y=f.programmed_volume_ul.to_numpy();w=training_weights(f)
        x=y[:,None]*np.linspace(-1,1,32)[None,:]
        for name in ['Ridge','Elastic Net','Extra Trees']:
            model=fit_model(name,x,y,w)
            pred=predict_model(model,x[:3])
            self.assertEqual(pred.shape,(3,));self.assertTrue(np.isfinite(pred).all())
            self.assertEqual(model['estimator'].n_features_in_,32)

    def test_nominal_level_assignment_is_from_training_levels(self):
        np.testing.assert_array_equal(nearest_level(np.array([1.,15.,16.,100.]),np.array([10.,20.,40.])),[10.,10.,20.,40.])

    def test_prediction_errors_and_tolerances_recompute(self):
        f=self.frame(1).iloc[:2].copy();f['programmed_volume_ul']=[10.,20.]
        result=metrics(f,np.array([11.,18.]),np.array([10.,20.]))
        self.assertAlmostEqual(result['mae_ul'],1.5)
        self.assertAlmostEqual(result['mape_pct'],10.)
        self.assertEqual(result['within_5pct'],0.)
        self.assertEqual(result['within_10pct'],1.)
        self.assertEqual(result['nominal_level_accuracy'],1.)

    def test_unseen_context_uses_training_median(self):
        f=self.frame();test=f.iloc[:3].copy();test['container_geometry_fingerprint']='new_geometry'
        median,pred=context_baseline(f,test)
        self.assertTrue(np.allclose(pred,median))

    def test_pooled_report_gives_runs_not_events_equal_weight(self):
        from report_all_qpm_volumes import collapse_run_errors
        rows=[]
        for run,n,error,inside in [('r1',2,0.,2),('r1',2,40.,0),('r2',1,0.,1)]:
            rows.append(dict(scheme='unseen_instrument',model='Extra Trees',physical_run_id=run,waveforms=n,
                absolute_error=error,squared_error=error**2,relative_error=error/100,
                inside_5pct=inside,inside_10pct=inside,correct_level=inside))
        result=collapse_run_errors(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.test_physical_runs,2)
        self.assertEqual(result.test_waveforms,5)
        self.assertEqual(result.mae_ul,5.)
        self.assertEqual(result.within_10pct,.75)


if __name__=='__main__':
    unittest.main()
