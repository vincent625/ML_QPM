#!/usr/bin/env python3
"""Independently check the complete all-operation nominal-volume study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VERSION = "all-qpm-volume-validation-1.0.0"
MODELS = ["Median baseline", "Context-only median", "Ridge", "Elastic Net", "Extra Trees", "Nested-selected procedure"]
PROFILE = [f"profile_{i:02d}" for i in range(32)]
CONTEXT = ["container_geometry_fingerprint", "deck_geometry_fingerprint", "command_settings_fingerprint"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def independent_metrics(frame):
    y = frame.programmed_volume_ul.to_numpy(float)
    p = frame.predicted_volume_ul.to_numpy(float)
    w = 1/frame.groupby("physical_run_id").operation_id.transform("size").to_numpy(float)
    mean = lambda a: float(np.sum(w*a)/np.sum(w))
    err = np.abs(p-y)
    denominator = mean((y-mean(y))**2)
    correct = np.isclose(y, frame.nearest_training_level_ul.to_numpy(), rtol=0, atol=1e-6)
    return dict(mae_ul=mean(err), rmse_ul=np.sqrt(mean(err**2)), mape_pct=100*mean(err/y),
        within_5pct=mean(err <= .05*y), within_10pct=mean(err <= .1*y),
        r2=1-mean((p-y)**2)/denominator if denominator > 0 else np.nan,
        nominal_level_accuracy=mean(correct),
        nominal_level_balanced_accuracy=float(np.mean([np.average(correct[y == level], weights=w[y == level]) for level in np.unique(y)])))


def assert_metrics(frame, record):
    for key, value in independent_metrics(frame).items():
        if not np.isclose(value, record[key], rtol=1e-7, atol=1e-7, equal_nan=True):
            raise AssertionError(f"Metric does not recompute: {key}: {value} vs {record[key]}")


def validate(root):
    started = time.monotonic()
    log = json.loads((root/"analysis_run_log.json").read_text())
    prep = json.loads((root/"preparation_audit.json").read_text())
    assert log["status"] == "complete" and log["actual_delivered_volume_labels"] == 0
    assert log["preparation_audit_sha256"] == sha(root/"preparation_audit.json")
    assert log["code_sha256"] == sha(root/"train_all_qpm_volumes.executed.py")
    assert prep["code_sha256"] == sha(root/"prepare_all_qpm_volumes.executed.py")
    for name, value in prep["artifact_sha256"].items(): assert sha(root/name) == value
    source = Path(prep["source_bank"])
    assert sha(source/"reference_bank_manifest.json") == prep["source_manifest_sha256"]
    for name,value in prep["source_sha256"].items(): assert sha(source/name) == value
    transfers = pd.read_csv(source/"csa_transfer_manifest.csv", usecols=["archive_id","sha256"]).set_index("archive_id")
    assert all(transfers.loc[key,"sha256"] == value for key,value in prep["archive_hashes"].items())
    assert prep["no_run_or_event_cap"] and prep["no_outcome_or_healthy_reference_filter"]
    all_data = pd.read_parquet(root/"all_audited_operations.parquet")
    assert not all_data.operation_id.duplicated().any()
    event_keys=["physical_run_id","operation_index","event_time","workflow_stage","process_step","loc","programmed_volume_ul","waveform_sha256",
        "liquid_class_fingerprint","pipette","operation","pressure_unit",*CONTEXT]
    assert not (all_data.duplicated(event_keys) & all_data.event_time.notna()).any()
    duplicates=pd.read_parquet(root/"duplicate_operation_exports.parquet")
    assert len(duplicates)==prep["duplicate_export_operations"] and not duplicates.operation_id.duplicated().any()
    assert not set(duplicates.operation_id)&set(all_data.operation_id)
    assert len(all_data) == prep["unique_audited_operations"]
    assert len(all_data) + prep["duplicate_export_operations"] == prep["physical_definition_and_unit_checked_operations"]
    assert np.isfinite(all_data[PROFILE].to_numpy()).all()
    coverage = pd.read_csv(root/"all_task_coverage.csv")
    assert set(coverage.task_id) == set(all_data.task_id)
    assert coverage.waveforms.sum() == len(all_data)
    assert (coverage.trainable == (coverage.physical_runs.ge(16) & coverage.observed_volume_levels.ge(2))).all()
    requested = set(coverage.loc[coverage.trainable,"task_id"])
    assert requested == {row["task_id"] for row in log["models"]}
    assert len(requested) == prep["trainable_tasks"]
    root_metrics = pd.read_csv(root/"model_performance.csv")
    membership = pd.read_parquet(root/"training_run_membership.parquet")
    scope = pd.read_parquet(root/"inference_context_scope.parquet")
    data = all_data[all_data.task_id.isin(requested)]
    assert len(data) == prep["trainable_waveforms"]
    expected_membership = data[["task_id","physical_run_id","instrument_id"]].drop_duplicates()
    pd.testing.assert_frame_equal(membership.sort_values(list(membership)).reset_index(drop=True),
        expected_membership.sort_values(list(expected_membership)).reset_index(drop=True))
    keys = ["task_id","pressure_unit",*CONTEXT,"programmed_volume_ul"]
    expected_scope = data.groupby(keys,sort=False).agg(physical_runs=("physical_run_id","nunique"), instruments=("instrument_id","nunique")).reset_index()
    pd.testing.assert_frame_equal(scope.sort_values(keys).reset_index(drop=True), expected_scope.sort_values(keys).reset_index(drop=True))
    del all_data, data
    checked, files = [], {}
    total_folds = 0
    for row in log["models"]:
        task = row["task_id"]
        folder = root/"tasks"/task
        for name,value in row["artifact_sha256"].items(): assert sha(folder/name) == value
        assert sha(root/row["path"]) == row["sha256"]
        frame = pq.read_table(root/"all_audited_operations.parquet",filters=[("task_id","=",task)],use_threads=False).to_pandas()
        assert len(frame) == row["train_waveforms"]
        assert frame.physical_run_id.nunique() == row["physical_runs"]
        assert sorted(frame.programmed_volume_ul.unique()) == row["volumes_ul"]
        assert sorted(frame.liquid_class_index.unique()) == row["liquid_class_indices"]
        assert frame.liquid_class_fingerprint.nunique() == frame.pipette.nunique() == frame.operation.nunique() == frame.pressure_unit.nunique() == 1
        times = pd.to_datetime(frame.run_start_time,utc=True).groupby(frame.physical_run_id).min()
        task_metrics = pd.read_csv(folder/"model_performance.csv")
        pd.testing.assert_frame_equal(task_metrics.reset_index(drop=True),root_metrics[root_metrics.task_id.eq(task)].reset_index(drop=True))
        parts=[]
        for fold_path in sorted(folder.glob("fold_*.json")):
            audit = json.loads(fold_path.read_text())
            signature = audit["input_signature"]
            assert signature["code_sha256"] == log["code_sha256"]
            assert signature["dataset_sha256"] == prep["artifact_sha256"]["all_audited_operations.parquet"]
            train,test=set(audit["train_physical_runs"]),set(audit["test_physical_runs"])
            assert not train & test and train|test <= set(frame.physical_run_id)
            if signature["scheme"] in ["grouped_run","unseen_instrument"]: assert train|test == set(frame.physical_run_id)
            if signature["scheme"] == "unseen_instrument":
                assert not set(audit["train_instruments"]) & set(audit["test_instruments"])
                assert len(audit["test_instruments"]) == 1
            if signature["scheme"] == "chronological": assert times.loc[list(train)].max() < times.loc[list(test)].min()
            pred_path=folder/f"predictions_{signature['scheme']}_{signature['fold']}.parquet"
            assert sha(pred_path) == audit["predictions_sha256"]
            prediction = pd.read_parquet(pred_path)
            assert set(prediction.model) == set(MODELS)
            expected_ids = set(frame.loc[frame.physical_run_id.isin(test),"operation_id"])
            assert set(prediction.physical_run_id) == test
            means = [record for record in audit["selection"] if record["inner_fold"] == "mean"]
            for record in means:
                values=[m["mae_ul"] for m in audit["selection"] if m["model"]==record["model"] and m["inner_fold"] != "mean"]
                assert len(values)==2 and np.isclose(record["mae_ul"],np.mean(values))
            selected = min(means,key=lambda record:(record["mae_ul"],["Ridge","Elastic Net","Extra Trees"].index(record["model"])))["model"]
            a=prediction[prediction.model.eq(selected)].sort_values("operation_id")
            b=prediction[prediction.model.eq("Nested-selected procedure")].sort_values("operation_id")
            assert np.array_equal(a.predicted_volume_ul.to_numpy(),b.predicted_volume_ul.to_numpy())
            levels=np.sort(frame.loc[frame.physical_run_id.isin(train),"programmed_volume_ul"].unique())
            for model,part in prediction.groupby("model"):
                assert len(part) == len(expected_ids) and set(part.operation_id) == expected_ids
                assert not part.operation_id.duplicated().any()
                assert np.isfinite(part.predicted_volume_ul).all()
                # Independent brute nearest-level calculation in bounded chunks.
                values=part.predicted_volume_ul.to_numpy()
                nearest=np.concatenate([levels[np.argmin(np.abs(chunk[:,None]-levels),axis=1)] for chunk in np.array_split(values,max(1,int(np.ceil(len(values)/4096))))])
                assert np.allclose(nearest,part.nearest_training_level_ul.to_numpy(),rtol=0,atol=1e-8)
                assert_metrics(part,next(r for r in audit["metrics"] if r["model"]==model))
            parts.append(prediction)
            files[str(fold_path.relative_to(root))]=sha(fold_path)
            files[str(pred_path.relative_to(root))]=audit["predictions_sha256"]
            total_folds+=1
        joined=pd.concat(parts,ignore_index=True)
        for scheme,part in joined.groupby("scheme"):
            baseline=part[part.model.eq("Median baseline")].set_index("operation_id")
            for model,subset in part.groupby("model"):
                assert not subset.operation_id.duplicated().any()
                if scheme == "grouped_run": assert set(subset.operation_id) == set(frame.operation_id)
                record=task_metrics[task_metrics.scheme.eq(scheme)&task_metrics.model.eq(model)].iloc[0]
                assert_metrics(subset,record)
                aligned=baseline.loc[subset.operation_id]
                errors=pd.DataFrame(dict(run=subset.physical_run_id.to_numpy(),m=abs(subset.predicted_volume_ul.to_numpy()-subset.programmed_volume_ul.to_numpy()),
                    b=abs(aligned.predicted_volume_ul.to_numpy()-subset.programmed_volume_ul.to_numpy()))).groupby("run")[["m","b"]].mean().to_numpy()
                rng=np.random.default_rng(log["random_seed"])
                samples=errors[rng.integers(0,len(errors),size=(250,len(errors)))].mean(axis=1)
                gain=100*(samples[:,1]-samples[:,0])/np.maximum(samples[:,1],1e-12)
                assert np.allclose([record.improvement_ci_low_pct,record.improvement_ci_high_pct],np.quantile(gain,[.025,.975]))
        supported=task_metrics[task_metrics.model.eq("Nested-selected procedure")&task_metrics.improvement_ci_low_pct.gt(0)]
        assert row["deployment_eligible"] == {"grouped_run","unseen_instrument","chronological"}.issubset(set(supported.scheme))
        model=joblib.load(root/row["path"])
        assert model["features"] == PROFILE and not model["actual_volume_prediction_validated"]
        assert model["estimator"].n_features_in_ == 32
        assert model["model_name"] == row["model_name"]
        checked.append(dict(task_id=task,waveforms=len(frame),folds=len(parts),deployment_eligible=row["deployment_eligible"]))
        print(f"Validated {len(checked)}/{len(log['models'])}: {task}",flush=True)
    for path in root.glob("*.csv"):files[path.name]=sha(path)
    for name in ["analysis_run_log.json","preparation_audit.json","training_run_membership.parquet","inference_context_scope.parquet",
                 "train_all_qpm_volumes.executed.py","prepare_all_qpm_volumes.executed.py"]: files[name]=sha(root/name)
    for row in log["models"]:
        for name,value in row["artifact_sha256"].items(): files[f"tasks/{row['task_id']}/{name}"]=value
        files[f"tasks/{row['task_id']}/task_complete.json"]=sha(root/"tasks"/row["task_id"]/"task_complete.json")
    result=dict(version=VERSION,status="passed",validated_utc=datetime.now(timezone.utc).isoformat(),tasks=checked,
        training_physical_runs=int(membership.physical_run_id.nunique()),training_instruments=int(membership.instrument_id.nunique()),
        verified_outer_folds=total_folds,all_trainable_operations_used=True,no_event_or_run_caps=True,
        no_train_test_physical_run_overlap=True,heldout_instrument_isolation_verified=True,chronological_order_verified=True,
        independent_metric_recomputation=True,nested_selection_verified=True,actual_delivered_volume_labels=0,
        artifact_sha256=files,validator_sha256=sha(Path(__file__)),execution_seconds=time.monotonic()-started)
    (root/"artifact_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ["tasks","artifact_sha256"]},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    validate(parser.parse_args().study.resolve())
