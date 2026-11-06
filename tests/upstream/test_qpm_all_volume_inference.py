"""Fail-closed scope and honest-label checks for the all-event registry format."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from qpm_nominal_volume_inference import NominalVolumeRecognizer
from train_all_qpm_volumes import fit_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AllVolumeInferenceTests(unittest.TestCase):
    def fixture(self, root, support=8, deploy=True):
        y=np.tile([10.,20.,30.],8)
        shape=np.linspace(-1.,1.,32)
        x=y[:,None]*(shape-np.median(shape[:3]))[None,:]
        payload=fit_model('Ridge',x,y,np.ones(len(y)))
        payload.update(training_waveform_q01=x.min(axis=0),training_waveform_q99=x.max(axis=0))
        joblib.dump(payload,root/'model.joblib')
        pd.DataFrame([dict(task_id='task',physical_run_id='training_run',instrument_id='inst')]).to_parquet(root/'training_run_membership.parquet',index=False)
        pd.DataFrame([dict(task_id='task',pressure_unit='raw_sensor_count_unresolved',container_geometry_fingerprint='cg',
            deck_geometry_fingerprint='dg',command_settings_fingerprint='settings',programmed_volume_ul=20.,physical_runs=support,instruments=3)]).to_parquet(root/'inference_context_scope.parquet',index=False)
        pd.DataFrame([dict(task_id='task',model='Nested-selected procedure',scheme=scheme,mae_ul=2.,within_10pct=.8,nominal_level_accuracy=.7)
            for scheme in ['grouped_run','unseen_instrument','chronological']]).to_csv(root/'model_performance.csv',index=False)
        record=dict(task_id='task',liquid_class_fingerprint='definition',pipette='PIP1',operation='aspirate',
            pressure_unit='raw_sensor_count_unresolved',deployment_eligible=deploy,model_name='Ridge',path='model.joblib',
            sha256=sha(root/'model.joblib'),volumes_ul=[10.,20.,30.])
        (root/'analysis_run_log.json').write_text(json.dumps(dict(format_version='all-operations-nominal-v1',models=[record])))
        names=['analysis_run_log.json','training_run_membership.parquet','inference_context_scope.parquet','model_performance.csv']
        (root/'artifact_validation.json').write_text(json.dumps(dict(status='passed',artifact_sha256={name:sha(root/name) for name in names})))
        return dict(qpm_has_er72=False,qpm_waveform_eligible=True,qpm_trace_quality_status='checked_no_er72',
            command_context_status='command_verified',qpm_profile=20.*shape,programmed_volume_ul=20.,
            liquid_class_index=43,liquid_class_fingerprint='definition',pipette='PIP1',operation='aspirate',
            pressure_unit='raw_sensor_count_unresolved',container_geometry_fingerprint='cg',deck_geometry_fingerprint='dg',
            command_settings_fingerprint='settings',physical_run_id='new_run')

    def test_broader_pipette_and_definition_not_numeric_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);operation=self.fixture(root);r=NominalVolumeRecognizer(root)
            result=r.predict(operation)
            self.assertEqual(result['status'],'nominal_recognition_only')
            self.assertFalse(result['actual_volume_prediction_validated'])
            self.assertIsNone(result['measured_transfer_deficit_ul'])
            self.assertEqual(result['validation_mae_ul'],2.)
            self.assertTrue(r.all_operations_format)
            self.assertEqual(len(r.predict_many([operation,operation])),2)

    def test_seen_run_is_not_graded_by_its_final_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);operation=self.fixture(root);operation['physical_run_id']='training_run'
            self.assertIn('present_in_model_training',NominalVolumeRecognizer(root).predict(operation)['reason'])

    def test_all_physical_settings_and_at_least_eight_runs_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);operation=self.fixture(root,support=7)
            self.assertIn('fewer_than_8',NominalVolumeRecognizer(root).predict(operation)['reason'])
        for field,value in [('pipette','PIP2'),('liquid_class_fingerprint','other_definition'),('pressure_unit','kPa')]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);operation=self.fixture(root);operation[field]=value
                self.assertIn('outside_trained_study',NominalVolumeRecognizer(root).predict(operation)['reason'])
        for field,value in [('programmed_volume_ul',25.),('container_geometry_fingerprint','other_cg'),('command_settings_fingerprint','other_command')]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);operation=self.fixture(root);operation[field]=value
                self.assertIn('fewer_than_8',NominalVolumeRecognizer(root).predict(operation)['reason'])

    def test_unvalidated_task_and_er72_abstain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);operation=self.fixture(root,deploy=False)
            self.assertIn('uncertain_heldout_validation',NominalVolumeRecognizer(root).predict(operation)['reason'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);operation=self.fixture(root);operation['qpm_has_er72']=True
            self.assertEqual(NominalVolumeRecognizer(root).predict(operation)['reason'],'Er72_operation_or_retry_excluded')

    def test_modified_models_and_scope_fail_before_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);operation=self.fixture(root);r=NominalVolumeRecognizer(root)
            (root/'model.joblib').write_bytes(b'changed_payload')
            self.assertEqual(r.predict(operation)['reason'],'model_hash_mismatch')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'model_performance.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact_hash_mismatch'):
                NominalVolumeRecognizer(root)


if __name__=='__main__':
    unittest.main()
