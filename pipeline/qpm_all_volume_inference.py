"""Scoped inference for the all-operation nominal-volume study.

Predictions express waveform resemblance to programmed volume. They are not
measurements of delivered volume, transfer deficits or a failure's cause.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qpm_trace_quality import er72_mask

FORMAT_VERSION = "all-operations-nominal-v1"
CONTEXT_FIELDS = ["pressure_unit", "container_geometry_fingerprint", "deck_geometry_fingerprint", "command_settings_fingerprint"]


class AllQPMNominalRecognizer:
    def __init__(self, study, *, expected_log_sha256=None, expected_validation_sha256=None):
        self.study = Path(study)
        log_bytes = (self.study/"analysis_run_log.json").read_bytes()
        seal_bytes = (self.study/"artifact_validation.json").read_bytes()
        log, seal = json.loads(log_bytes), json.loads(seal_bytes)
        if log.get("format_version") != FORMAT_VERSION or seal.get("status") != "passed":
            raise ValueError("all_operation_nominal_study_not_validated")
        log_hash, seal_hash = hashlib.sha256(log_bytes).hexdigest(), hashlib.sha256(seal_bytes).hexdigest()
        if expected_log_sha256 is not None and expected_log_sha256 != log_hash:
            raise ValueError("nominal_volume_study_registry_hash_mismatch")
        if expected_validation_sha256 is not None and expected_validation_sha256 != seal_hash:
            raise ValueError("nominal_volume_study_registry_hash_mismatch")
        if seal.get("artifact_sha256", {}).get("analysis_run_log.json") != log_hash:
            raise ValueError("nominal_study_artifact_hash_mismatch")
        content = {}
        for name in ["training_run_membership.parquet", "inference_context_scope.parquet", "model_performance.csv"]:
            raw = (self.study/name).read_bytes()
            if seal.get("artifact_sha256", {}).get(name) != hashlib.sha256(raw).hexdigest():
                raise ValueError("nominal_study_artifact_hash_mismatch")
            content[name] = raw
        membership = pd.read_parquet(io.BytesIO(content["training_run_membership.parquet"]))
        scope = pd.read_parquet(io.BytesIO(content["inference_context_scope.parquet"]))
        performance = pd.read_csv(io.BytesIO(content["model_performance.csv"]))
        models = {}
        records = {}
        for record in log["models"]:
            key = (record["liquid_class_fingerprint"], record["pipette"], record["operation"], record["pressure_unit"])
            if key in models:
                raise ValueError("ambiguous_full_definition_volume_task")
            models[key] = record
            records[record["task_id"]] = record
        run_ids = {task:set(group.physical_run_id) for task,group in membership.groupby("task_id")}
        contexts = {}
        for task,group in scope.groupby("task_id"):
            contexts[task] = {tuple(row[:-2])+(round(float(row[-2]),3),):int(row[-1])
                for row in group[CONTEXT_FIELDS+["programmed_volume_ul","physical_runs"]].itertuples(index=False,name=None)}
        self.artifacts = dict(log=log, performance=performance, models=models, records=records,
            contexts=contexts, run_ids=run_ids, payloads={}, log_sha256=log_hash, validation_sha256=seal_hash)

    def guard(self, operation):
        result = dict(status="unavailable",actual_volume_prediction_validated=False,
            measured_transfer_deficit_ul=None,label_basis="nominal_command_not_measured_delivery")
        def unavailable(reason):
            return {**result,"reason":reason},None,None
        if er72_mask(pd.DataFrame([operation])).iloc[0]:
            return unavailable("Er72_operation_or_retry_excluded")
        if str(operation.get("qpm_waveform_eligible","")).lower() not in {"true","1","1.0"} or operation.get("qpm_trace_quality_status") != "checked_no_er72":
            return unavailable("trace_quality_unverified")
        if operation.get("command_context_status") != "command_verified":
            return unavailable("command_context_unverified")
        try:
            wave=np.asarray(operation["qpm_profile"],dtype=float)
            volume=float(operation["programmed_volume_ul"])
        except (KeyError,TypeError,ValueError):
            return unavailable("invalid_waveform_or_volume_metadata")
        if wave.shape != (32,) or not np.isfinite(wave).all() or not np.isfinite(volume) or volume <= 0:
            return unavailable("invalid_32_point_waveform_or_volume")
        key=tuple(operation.get(field) for field in ["liquid_class_fingerprint","pipette","operation","pressure_unit"])
        record=self.artifacts["models"].get(key)
        if record is None:
            return unavailable("full_definition_channel_operation_unit_outside_trained_study")
        task=record["task_id"]
        if not record["deployment_eligible"]:
            return unavailable("task_has_insufficient_or_uncertain_heldout_validation")
        if not operation.get("physical_run_id"):
            return unavailable("missing_physical_run_identity")
        if operation["physical_run_id"] in self.artifacts["run_ids"].get(task,set()):
            return unavailable("target_physical_run_present_in_model_training_use_heldout_predictions")
        for field in CONTEXT_FIELDS:
            if not operation.get(field):
                return unavailable("missing_"+field)
        context=tuple(operation[field] for field in CONTEXT_FIELDS)+(round(volume,3),)
        count=self.artifacts["contexts"].get(task,{}).get(context,0)
        if count < 8:
            return unavailable("exact_nominal_geometry_command_context_has_fewer_than_8_training_runs")
        result["exact_context_training_physical_runs"] = count
        return result,record,wave-np.median(wave[:3])

    def predict_many(self, operations, *, kind="regression"):
        if kind != "regression":
            return [dict(status="unavailable",reason="all_event_study_provides_regression_only",
                actual_volume_prediction_validated=False,measured_transfer_deficit_ul=None) for _ in operations]
        entries=[self.guard(operation) for operation in operations]
        output=[result for result,_,_ in entries]
        by_task={}
        for i,(_,record,_) in enumerate(entries):
            if record is not None:by_task.setdefault(record["task_id"],[]).append(i)
        for task,indices in by_task.items():
            record=self.artifacts["records"][task]
            if task not in self.artifacts["payloads"]:
                raw=(self.study/record["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
                    for i in indices:output[i]={**output[i],"reason":"model_hash_mismatch"}
                    continue
                self.artifacts["payloads"][task]=joblib.load(io.BytesIO(raw))
            payload=self.artifacts["payloads"][task]
            x=np.vstack([entries[i][2] for i in indices])
            transformed=payload["scaler"].transform(x) if payload["scaler"] is not None else x
            predicted=np.asarray(payload["estimator"].predict(transformed))*payload["y_scale"]+payload["y_mean"]
            part=self.artifacts["performance"]
            part=part[part.task_id.eq(task)&part.model.eq("Nested-selected procedure")]
            by_scheme={row.scheme:row for row in part.itertuples(index=False)}
            for i,value,wave in zip(indices,predicted,x):
                if not np.isfinite(value) or not min(record["volumes_ul"])<=value<=max(record["volumes_ul"]):
                    output[i]={**output[i],"reason":"predicted_nominal_equivalent_outside_observed_range"}
                    continue
                outside=float(np.mean((wave<payload["training_waveform_q01"])|(wave>payload["training_waveform_q99"])))
                external=by_scheme["unseen_instrument"]
                output[i]={**output[i],"status":"nominal_recognition_only","nominal_equivalent_volume_ul":float(value),
                    "task_id":task,"model_name":record["model_name"],"validation_mae_ul":float(external.mae_ul),
                    "validation_mae_unseen_run_ul":float(by_scheme["grouped_run"].mae_ul),
                    "validation_mae_unseen_instrument_ul":float(external.mae_ul),
                    "validation_mae_chronological_ul":float(by_scheme["chronological"].mae_ul),
                    "validation_within_10pct":float(external.within_10pct),
                    "validation_nominal_level_accuracy":float(external.nominal_level_accuracy),
                    "waveform_outside_training_98pct_fraction":outside,
                    "reason":"not_an_independent_measurement_of_delivered_volume_or_transfer_error"}
        return output

    def predict(self, operation, *, kind="regression"):
        return self.predict_many([operation],kind=kind)[0]
