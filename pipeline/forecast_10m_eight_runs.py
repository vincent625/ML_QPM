#!/usr/bin/env python3
"""Offline <10M forecasting from eight as-of-time, same-instrument CSA runs.

No instrument, plugin, baseline bank, or older analysis is changed. Threshold,
history summaries, model grid, and validation rules are fixed before fitting.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss

import forecast_next_run_csa as previous
from outcome_reference import load_policy, load_outcome_table

VERSION = "csa-eight-run-10m-forecast-1.0.2"
SEED = 20261023
THRESHOLD = 10_000_000
LOOKBACK = 8
PARAMS = [("logistic", 8, .05), ("logistic", 20, .5),
          ("extra_trees", 8, 3), ("extra_trees", 20, 5)]
FAMILIES = ["history_only", "history_fluidics", "history_spatial_plugins", "history_all_csa"]
say = previous.say
sha256 = previous.sha256
write_json = previous.write_json


def load_wide_history(path):
    # This is our own just-generated ~196k-column file. Its schema exceeds
    # Arrow's default Thrift metadata cap; no limits are relaxed for archives.
    binary = Path(path).with_suffix(".joblib")
    if binary.exists():
        return joblib.load(binary)  # Only this workflow's own generated artifact.
    return pq.ParquetFile(path, thrift_string_size_limit=2**30,
                          thrift_container_size_limit=4_000_000).read(use_threads=False).to_pandas()


def event_label(s):
    """Any retained lane <10M; negative requires all retained lanes measured."""
    s = pd.to_numeric(s, errors="coerce")
    if s.lt(THRESHOLD).any():
        return 1.
    return 0. if s.notna().all() else np.nan


def numeric_history(a):
    """Deterministic within-window summaries; missing stays missing.

    a is [targets, eight older-to-newer runs, descriptors]. Mean/trend need
    four measured source runs. Slopes are per observed run, not per day.
    """
    # Float32 accumulation makes a constant field acquire spurious slopes when
    # missing runs differ. Accumulate in float64 before storing the result.
    a = a.astype(np.float64, copy=False)
    valid = np.isfinite(a)
    n = valid.sum(axis=1)
    values = np.where(valid, a, 0.)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis=1) / n
        t = np.arange(a.shape[1], dtype=np.float64)[None, :, None]
        st = (valid*t).sum(axis=1)
        stt = (valid*t*t).sum(axis=1)
        sty = (values*t).sum(axis=1)
        slope = (sty-st*mean)/(stt-st*st/n)
    mean[n < 4] = np.nan
    slope[n < 4] = np.nan
    return a[:, -1, :], mean, slope


def rebuild_precision(source_out, out):
    """Reuse audited cohort/source medians; regenerate only history transforms.

    A new output directory preserves every preliminary artifact. No fitted
    model, selected features, target labels, or model decisions are reused.
    """
    import shutil
    if (out / "preparation_log.json").exists():
        raise ValueError("Refusing to replace an existing stable-precision experiment")
    names = ["physical_source_numeric.parquet", "physical_run_timeline.csv", "lane_label_audit.csv",
             "forecast_cohort.csv", "model_cohort.csv", "history_members.csv", "source_feature_policy.csv",
             "model_feature_inventory.csv", "eight_run_categorical_features.parquet", "cohort_counts.csv", "input_archive_hashes.csv"]
    for name in names:
        shutil.copy2(source_out / name, out / name)
    frame = pd.read_csv(out / "model_cohort.csv")
    original = load_wide_history(source_out / "eight_run_numeric_features.parquet")
    history = original[[c for c in original if c.startswith("history.")]].copy()
    del original
    gc.collect()
    source = pd.read_parquet(out / "physical_source_numeric.parquet")
    pos = np.asarray([[source.index.get_loc(i) for i in t.split(";")] for t in frame.history_ids])
    summaries = {name: [] for name in ["latest", "mean8", "slope8"]}
    a = source.to_numpy(dtype=np.float32)
    for left in range(0, a.shape[1], 2048):
        for name, values in zip(summaries, numeric_history(a[pos, left:left+2048])):
            summaries[name].append(values.astype(np.float32))
    x = pd.concat([history, *[pd.DataFrame(np.concatenate(summaries[name], axis=1),
                                         columns=[f"csa8.{name}.{c}" for c in source.columns]) for name in summaries]], axis=1)
    joblib.dump(x, out / "eight_run_numeric_features.joblib", compress=3)
    log = json.loads((source_out / "preparation_log.json").read_text())
    log.update(version=VERSION, precision="float64 accumulation; float32 stored summaries",
               precision_rebuild_source=str(source_out), precision_rebuild_source_log_sha256=sha256(source_out / "preparation_log.json"),
               numeric_history_storage="trusted_workflow_generated_joblib; no untrusted pickle inputs",
               history_matrix_sha256=sha256(out / "eight_run_numeric_features.joblib"))
    log["input_sha256"][str(Path(__file__))] = sha256(__file__)
    log["warnings"].append("Preliminary float32 trend artifacts were identified by constant-feature regression tests. All CSA means/trends were rebuilt with float64 accumulation and every model was refit in a separate directory. No preliminary result is used as a final score.")
    write_json(out / "preparation_log.json", log)
    say("Stable-precision histories rebuilt; preliminary analyses preserved separately")


def prepare(project, out):
    begun = time.monotonic()
    bank = json.loads((project / "reference_bank_refresh.json").read_text())
    multi_dir = project / "refreshes" / bank["bank_id"] / "project"
    policy_path = project / "next_run_forecast_20261021/model_feature_inventory.csv"
    policy = pd.read_csv(policy_path)
    source_policy = policy[~policy.feature_name.str.startswith("history.")].copy()
    assert not source_policy.feature_name.str.contains(previous.FORBIDDEN).any()
    # Reuse only a target-independent, reviewed semantic allowlist, not prior
    # selected features, feature importance, learned transforms, labels or fits.
    core_path = project / "csa_ml_dataset.parquet"
    multi_path = multi_dir / "multimodal_training_dataset.parquet"
    core = pd.read_parquet(core_path)
    inventory = pd.read_csv(project / "csa_inventory.csv")
    keep = ["observation_id", "physical_run_id", "instrument_id", "archive_id",
            "chip_lane", "run_start_time", "total_reads", "mean_read_length_bp"]
    lanes = core[keep].merge(inventory[["archive_id", "hostname", "remote_mtime_utc"]],
                            on="archive_id", validate="one_to_one")
    lanes["start"] = pd.to_datetime(lanes.run_start_time, utc=True)
    lanes["available"] = pd.to_datetime(lanes.remote_mtime_utc, utc=True)
    lanes["low_10m"] = lanes.total_reads.lt(THRESHOLD).astype(float).where(lanes.total_reads.notna())
    assert lanes.start.notna().all() and lanes.available.notna().all()
    assert lanes.available.ge(lanes.start).all()
    assert not lanes.observation_id.duplicated().any()
    assert lanes.groupby("physical_run_id").instrument_id.nunique().max() == 1
    physical = lanes.groupby("physical_run_id", sort=False).agg(
        instrument_id=("instrument_id", "first"), hostname=("hostname", "first"),
        start=("start", "min"), available=("available", "max"),
        retained_lanes=("chip_lane", "size"), total_reads=("total_reads", "median"),
        minimum_lane_reads=("total_reads", "min"),
        mean_read_length_bp=("mean_read_length_bp", "median"),
        low_10m=("total_reads", event_label),
        source_archive_ids=("archive_id", lambda s: ";".join(s)),
        observation_ids=("observation_id", lambda s: ";".join(s)),
    ).reset_index().sort_values(["instrument_id", "start"]).reset_index(drop=True)
    context = load_outcome_table(project, load_policy(project))
    context_cols = ["chip_type", "assay_type", "panel_technology", "workflow_script",
                    "run_mode", "template_size", "flow_count", "instrument_software"]
    def common(s):
        a = s.dropna().astype(str).unique()
        return a[0] if len(a) == 1 else "__missing_or_mixed__"
    pc = context.groupby("physical_run_id")[context_cols].agg(common)
    physical = physical.merge(pc, on="physical_run_id", validate="one_to_one")
    physical.to_csv(out / "physical_run_timeline.csv", index=False)
    lanes.to_csv(out / "lane_label_audit.csv", index=False)

    members, cohort, history_rows = [], [], []
    source_ids = []
    for _, group in physical.groupby("instrument_id"):
        for _, target in group.iterrows():
            earlier = group[group.start.lt(target.start)].sort_values("start")
            available = earlier[earlier.available.lt(target.start)]
            r = target.to_dict()
            r.update(n_prior_observed=len(earlier), n_prior_available=len(available),
                     eligible=len(available) >= LOOKBACK,
                     strict_immediate_eight=(len(earlier) >= LOOKBACK and
                                             earlier.tail(LOOKBACK).available.lt(target.start).all()))
            if not r["eligible"]:
                cohort.append(r)
                continue
            hist = available.tail(LOOKBACK)
            ids = hist.physical_run_id.tolist()
            r.update(source_physical_run_id=ids[-1], source_start=hist.start.iloc[-1],
                     source_available=hist.available.iloc[-1],
                     history_ids=";".join(ids), history_span_days=(target.start-hist.start.iloc[0]).total_seconds()/86400,
                     last_gap_days=(target.start-hist.start.iloc[-1]).total_seconds()/86400,
                     min_availability_margin_hours=(target.start-hist.available.max()).total_seconds()/3600,
                     previous_bad=hist.low_10m.iloc[-1],
                     skipped_more_recent_unavailable=int(sum(earlier.start.ge(hist.start.iloc[0]) & earlier.available.ge(target.start))))
            r["known_bad_in_history8"] = int(hist.low_10m.eq(1).sum())
            r["known_labels_in_history8"] = int(hist.low_10m.notna().sum())
            r["all_history_same_chip"] = hist.chip_type.eq(target.chip_type).all()
            r["all_history_same_assay"] = hist.assay_type.eq(target.assay_type).all()
            for j, (_, src) in enumerate(hist.iterrows()):
                members.append({"target_id": target.physical_run_id, "target_start": target.start,
                                "target_instrument": target.instrument_id, "lag": LOOKBACK-j,
                                "source_id": src.physical_run_id, "source_start": src.start,
                                "source_available": src.available, "source_instrument": src.instrument_id,
                                "total_reads": src.total_reads, "low_10m": src.low_10m,
                                "minimum_lane_reads": src.minimum_lane_reads,
                                "same_chip_as_target": src.chip_type == target.chip_type,
                                "same_assay_as_target": src.assay_type == target.assay_type})
            f = {"physical_run_id": target.physical_run_id,
                 "history.gap_days": r["last_gap_days"],
                 "history.span_days": r["history_span_days"],
                 "history.archive_age_hours": r["min_availability_margin_hours"]}
            for name in ["total_reads", "minimum_lane_reads", "mean_read_length_bp", "low_10m"]:
                a = hist[name].to_numpy(dtype=float)
                latest, mean, slope = numeric_history(a[None, :, None])
                f[f"history.latest.{name}"] = float(latest[0, 0])
                f[f"history.mean8.{name}"] = float(mean[0, 0])
                f[f"history.slope8.{name}"] = float(slope[0, 0])
                f[f"history.count8.{name}"] = int(np.isfinite(a).sum())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    f[f"history.std8.{name}"] = float(np.nanstd(a))
                    f[f"history.min8.{name}"] = float(np.nanmin(a)) if np.isfinite(a).any() else np.nan
                    f[f"history.delta3.{name}"] = float(np.nanmean(a[-3:])-np.nanmean(a[:3]))
            for name in ["total_reads", "minimum_lane_reads"]:
                a = np.log1p(hist[name].clip(lower=0).to_numpy(dtype=float))
                latest, mean, slope = numeric_history(a[None, :, None])
                for label, val in zip(["latest", "mean8", "slope8"], [latest, mean, slope]):
                    f[f"history.{label}.log1p_{name}"] = float(val[0, 0])
            source_ids.append((target.physical_run_id, ids))
            history_rows.append(f)
            cohort.append(r)
    cohort = pd.DataFrame(cohort).sort_values("start").reset_index(drop=True)
    cohort.to_csv(out / "forecast_cohort.csv", index=False)
    pairs = cohort[cohort.eligible & cohort.low_10m.notna()].reset_index(drop=True)
    pairs.to_csv(out / "model_cohort.csv", index=False)
    members = pd.DataFrame(members)
    members.to_csv(out / "history_members.csv", index=False)
    hist = pd.DataFrame(history_rows).set_index("physical_run_id").loc[pairs.physical_run_id].astype(np.float32)
    hist.index = pd.RangeIndex(len(pairs))

    numeric_policy = source_policy[source_policy.data_type.eq("numeric")]
    cat_policy = source_policy[~source_policy.data_type.eq("numeric")]
    core_nums = [n[5:] for n in numeric_policy.feature_name if n.startswith("core.")]
    multi_nums = [n[6:] for n in numeric_policy.feature_name if n.startswith("multi.")]
    core_cats = [n[5:] for n in cat_policy.feature_name if n.startswith("core.")]
    num = core[core_nums].apply(pd.to_numeric, errors="coerce")
    num.index = core.physical_run_id
    num = num.replace([np.inf, -np.inf], np.nan).groupby(level=0).median().astype(np.float32)
    num.columns = "core."+num.columns
    multi = pd.read_parquet(multi_path, columns=["observation_id", "archive_id", *multi_nums]).set_index("observation_id")
    assert set(multi.index) == set(core.observation_id)
    assert multi.archive_id.sort_index().equals(core.set_index("observation_id").archive_id.sort_index())
    multi = multi.loc[core.observation_id, multi_nums]
    multi.index = core.physical_run_id
    multi = multi.replace([np.inf, -np.inf], np.nan).groupby(level=0).median().astype(np.float32)
    multi.columns = "multi."+multi.columns
    source = pd.concat([num, multi], axis=1)
    source.to_parquet(out / "physical_source_numeric.parquet")
    id_map = dict(source_ids)
    positions = np.asarray([[source.index.get_loc(i) for i in id_map[t]] for t in pairs.physical_run_id])
    # Process source columns in chunks: no target-dependent global selection.
    say(f"Building eight-run trajectories: {len(pairs)} known targets, {int(pairs.low_10m.sum())} <10M events, {len(source.columns)} source descriptors")
    summaries = {name: [] for name in ["latest", "mean8", "slope8"]}
    a = source.to_numpy(dtype=np.float32)
    for left in range(0, a.shape[1], 2048):
        for name, values in zip(summaries, numeric_history(a[positions, left:left+2048])):
            summaries[name].append(values.astype(np.float32))
    x = pd.concat([hist, *[pd.DataFrame(np.concatenate(summaries[name], axis=1),
                                     columns=[f"csa8.{name}.{c}" for c in source.columns]) for name in summaries]], axis=1)
    x.to_parquet(out / "eight_run_numeric_features.parquet", index=False)
    cat_source = core[["physical_run_id", *core_cats]].groupby("physical_run_id")[core_cats].agg(common)
    cat_source.columns = "core."+cat_source.columns
    # Only prior context can enter X; current-run protocol is audit/subgroup data.
    for c in context_cols:
        cat_source[f"context.{c}"] = pc[c]
    categorical = cat_source.loc[pairs.source_physical_run_id].reset_index(drop=True).fillna("__missing__").astype(str)
    unsafe = re.compile(r"[/\\]|@|(?:\d{1,3}\.){3}\d{1,3}|Auto_R_|gnxs-", re.I)
    for c in categorical:
        categorical[c] = categorical[c].mask(categorical[c].str.contains(unsafe), "__redacted_identifier_or_path__")
    categorical.to_parquet(out / "eight_run_categorical_features.parquet", index=False)
    source_policy.to_csv(out / "source_feature_policy.csv", index=False)
    feature_rows = [{"feature_name": c, "component": "outcome_history", "source_feature": c} for c in hist]
    source_info = source_policy.set_index("feature_name")
    for method in summaries:
        feature_rows.extend({"feature_name": f"csa8.{method}.{c}", "source_feature": c,
                             "component": source_info.at[c, "component"], "summary": method} for c in source.columns)
    pd.DataFrame(feature_rows).to_csv(out / "model_feature_inventory.csv", index=False)
    counts = []
    for pop, frame in [("all_physical_runs", physical), ("eight_available", pairs),
                       ("eight_immediate_available", pairs[pairs.strict_immediate_eight])]:
        for host, g in [("ALL", frame), *list(frame.groupby("hostname"))]:
            counts.append({"population": pop, "hostname": host, "n": len(g),
                           "known": int(g.low_10m.notna().sum()), "events": int(g.low_10m.eq(1).sum())})
    pd.DataFrame(counts).to_csv(out / "cohort_counts.csv", index=False)
    manifest = pd.read_csv(project / "csa_transfer_manifest.csv")
    manifest[manifest.archive_id.isin(lanes.archive_id)][["archive_id", "sha256", "state"]].to_csv(out / "input_archive_hashes.csv", index=False)
    inputs = [core_path, multi_path, policy_path, project / "csa_inventory.csv",
              project / "csa_transfer_manifest.csv", Path(__file__), project / "forecast_next_run_csa.py"]
    log = {"version": VERSION, "seed": SEED, "threshold_reads": THRESHOLD, "lookback_runs": LOOKBACK,
           "bank_id": bank["bank_id"], "source_descriptor_count": len(source.columns),
           "numeric_history_columns": len(x.columns), "categorical_columns": len(categorical.columns),
           "physical_runs": len(physical), "lane_observations": len(lanes), "known_targets_with_history": len(pairs),
           "bad_targets_with_history": int(pairs.low_10m.sum()), "input_sha256": {str(p): sha256(p) for p in inputs},
           "prepare_seconds": round(time.monotonic()-begun, 2),
           "warnings": [
               "<10M is a user-defined per-lane low-read threshold, not an assay-approved failure limit. A physical run is bad if ANY retained lane is <10M; negative requires all retained lanes known. Lanes are never summed.",
               "Exactly eight earlier CSA-available physical runs on the SAME instrument are used. Missing or late archives can skip more recent observed runs; strict immediate-eight sensitivity is reported.",
               "CSA remote mtime is an unverified availability proxy; run-start timezone and clock offsets are not verified. Earlier exports and missing intervening real instrument runs are not reconstructed.",
               "History windows overlap. Uncertainty resamples eight-target blocks within instruments and is conditional on the fitted models; only six instruments and few events remain.",
               "Target own diagnostics, reads, and protocol metadata never enter predictors. Prior outcomes are allowed only as history. Earlier Tier C values are legitimate as-of-time forecasting inputs, not actionable settings.",
               "Candidate semantic allowlist is reused from the earlier leakage review, not learned feature ranks or transforms. All imputation, filtering, scaling, encoding, and feature/model selection are refit inside each training fold.",
               "Inherited extracted multimodal source descriptors had target-independent sparse/constant filtering. This tests available numeric descriptors and safe context, not every raw CSA byte.",
               "Raw read accuracy is unavailable in this six-instrument bank. <10M may include aborted or deliberately low-yield runs; failure causes are not verified.",
               "This is a post-hoc follow-up hypothesis using the same archive bank, not an independent prospective replication. No plugin deployment or instrument change is made.",
           ]}
    write_json(out / "preparation_log.json", log)
    say("Preparation and leakage review complete")


def fit_predict(pre, param, xx, cc, train, test, y):
    k = min(param[1], len(pre.names))
    model = previous.make_model(param)
    model.set_params(random_state=SEED)
    model.fit(pre.transform(xx.loc[train], cc.loc[train])[:, -k:], y[train])
    return model, model.predict_proba(pre.transform(xx.loc[test], cc.loc[test])[:, -k:])[:, 1], k


def train(out):
    begun = time.monotonic()
    frame = pd.read_csv(out / "model_cohort.csv")
    for c in ["start", "available", "source_available"]:
        frame[c] = pd.to_datetime(frame[c], utc=True)
    x = load_wide_history(out / "eight_run_numeric_features.parquet")
    cats = pd.read_parquet(out / "eight_run_categorical_features.parquet")
    info = pd.read_csv(out / "model_feature_inventory.csv").set_index("feature_name")
    comp = info.component.to_dict()
    history = [c for c in x if c.startswith("history.")]
    fluidics = [c for c in x if comp.get(c) in {"qpm_debug", "vacuum_steps", "libprep_telemetry"} or
                c.startswith("csa8.") and re.search(r"pressure|vacuum|pipett|libprep|liquid|templating", c, re.I)]
    spatial = [c for c in x if comp.get(c, "").startswith("plugin:") or comp.get(c) == "nucstep_spatial"]
    cols = dict(zip(FAMILIES, [history, list(dict.fromkeys(history+fluidics)), list(dict.fromkeys(history+spatial)), list(x.columns)]))
    y = frame.low_10m.to_numpy(dtype=int)
    performance, predictions, tuning, selected, audits, skipped = [], [], [], [], [], []
    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    for scheme, fold, train, test, cutoff in previous.splits(frame):
        if len(train) < 40 or y[train].sum() < 5 or len(np.unique(y[train])) != 2 or len(test) < 3:
            skipped.append({"scheme": scheme, "fold": fold, "train_n": len(train), "train_events": int(y[train].sum()), "test_n": len(test), "reason": "insufficient_training_or_test_data"})
            continue
        assert frame.loc[train, "available"].lt(cutoff).all()
        assert frame.loc[train, "start"].lt(frame.loc[test, "start"].min()).all()
        assert not set(frame.loc[train, "physical_run_id"]) & set(frame.loc[test, "physical_run_id"])
        if scheme == "unseen_instrument_and_future":
            assert not set(frame.loc[train, "instrument_id"]) & set(frame.loc[test, "instrument_id"])
        say(f"{scheme} {fold}: train {len(train)}/{int(y[train].sum())} bad; test {len(test)}/{int(y[test].sum())} bad")
        audits.append({"scheme": scheme, "fold": fold, "cutoff": cutoff, "train_n": len(train), "train_events": int(y[train].sum()),
                       "test_n": len(test), "test_events": int(y[test].sum()), "train_max_available": frame.loc[train, "available"].max(),
                       "train_ids": ";".join(frame.loc[train, "physical_run_id"]), "test_ids": ";".join(frame.loc[test, "physical_run_id"])})
        prior = (int(y[train].sum())+1)/(len(train)+2)
        inst = frame.loc[train].assign(label=y[train]).groupby("instrument_id").label.agg(["sum", "count"])
        ip = frame.loc[test, "instrument_id"].map((inst["sum"]+5*prior)/(inst["count"]+5)).fillna(prior).to_numpy()
        pp = {"historical_event_rate": np.full(len(test), prior), "instrument_event_rate": ip,
              "previous_run_status": .5*x.loc[test, "history.latest.low_10m"].fillna(prior).to_numpy()+.5*prior,
              "eight_run_event_rate": (x.loc[test, "history.mean8.low_10m"].fillna(prior).to_numpy()*
                                       x.loc[test, "history.count8.low_10m"].to_numpy()+2*prior)/
                                      (x.loc[test, "history.count8.low_10m"].to_numpy()+2)}
        choices = []
        for family in FAMILIES:
            say(f"  fitting {family}")
            xx = x[cols[family]]
            cc = cats if family == "history_all_csa" else cats.iloc[:, :0]
            scores = [[] for _ in PARAMS]
            for inner_train, inner_test in previous.inner_splits(frame, train):
                if y[inner_train].sum() < 3 or len(np.unique(y[inner_train])) != 2:
                    continue
                pre = previous.FoldFeatures(20).fit(xx.loc[inner_train], cc.loc[inner_train], y[inner_train])
                if not len(pre.names):
                    continue
                for pi, param in enumerate(PARAMS):
                    _, pred, _ = fit_predict(pre, param, xx, cc, inner_train, inner_test, y)
                    scores[pi].append((len(inner_test), brier_score_loss(y[inner_test], pred)))
                del pre
            averages = [sum(n*v for n, v in s)/sum(n for n, _ in s) if s else np.inf for s in scores]
            for pi, param in enumerate(PARAMS):
                tuning.append({"scheme": scheme, "fold": fold, "family": family, "model": param[0],
                               "features": param[1], "regularization_or_depth": param[2], "inner_folds": len(scores[pi]),
                               "inner_brier": averages[pi] if np.isfinite(averages[pi]) else np.nan})
            pre = previous.FoldFeatures(20).fit(xx.loc[train], cc.loc[train], y[train])
            for model_name in ["logistic", "extra_trees"]:
                candidates = [i for i, p in enumerate(PARAMS) if p[0] == model_name]
                best = min(candidates, key=lambda i: averages[i])
                param = PARAMS[best]
                model, pred, k = fit_predict(pre, param, xx, cc, train, test, y)
                name = f"{family}__{model_name}"
                pp[name] = pred
                choices.append((averages[best], name, family))
                selected.extend({"scheme": scheme, "fold": fold, "model": name, "feature": f,
                                 "training_filtered_features": pre.filtered_count,
                                 "interpretation": "training_selected_not_validated_biomarker"} for f in pre.names[-k:])
                if scheme == "final_chronological_holdout":
                    joblib.dump({"preprocessor": pre, "model": model, "k": k, "numeric_columns": cols[family],
                                 "categorical_columns": list(cc.columns), "cutoff": str(cutoff), "version": VERSION}, model_dir / f"{name}.joblib")
            del xx, cc, pre
            gc.collect()
        for output, allowed in [("history_nested_selected", {"history_only"}),
                                ("csa_nested_selected", set(FAMILIES)-{"history_only"})]:
            chosen = min((a for a in choices if a[2] in allowed), key=lambda a: (a[0], a[1]))
            pp[output] = pp[chosen[1]]
            tuning.append({"scheme": scheme, "fold": fold, "family": output, "model": chosen[1],
                           "inner_brier": chosen[0] if np.isfinite(chosen[0]) else np.nan,
                           "fallback": not np.isfinite(chosen[0])})
        for name, pred in pp.items():
            performance.append({"scheme": scheme, "fold": fold, "model": name, "train_n": len(train),
                                "train_events": int(y[train].sum()), **previous.metric(y[test], pred)})
            for i, prob in zip(test, pred):
                predictions.append({"scheme": scheme, "fold": fold, "model": name, "physical_run_id": frame.at[i, "physical_run_id"],
                                    "instrument_id": frame.at[i, "instrument_id"], "hostname": frame.at[i, "hostname"],
                                    "start": frame.at[i, "start"], "observed": int(y[i]), "probability": float(prob)})
        for filename, rows in [("heldout_predictions.csv", predictions), ("model_performance_folds.csv", performance),
                               ("nested_model_selection.csv", tuning), ("fold_selected_features.csv", selected), ("validation_split_audit.csv", audits)]:
            pd.DataFrame(rows).to_csv(out / filename, index=False)
    pd.DataFrame(skipped).to_csv(out / "skipped_tests.csv", index=False)
    log = json.loads((out / "preparation_log.json").read_text())
    log.update(executed_utc=datetime.now(timezone.utc).isoformat(), model_seconds=round(time.monotonic()-begun, 2),
               model_code_version=VERSION,
               execution_warnings=["Initial pre-fit read exceeded Arrow's default metadata cap for the generated 195730-column history file. Raised the metadata cap only for this trusted local derivative; no cohort, features, outcomes, model choices or archive limits changed."],
               code_sha256=sha256(__file__), sklearn=sklearn.__version__, numpy=np.__version__,
               models={"families": FAMILIES, "parameter_grid": PARAMS, "selection": "nested_inner_time_validation_Brier",
                       "final_holdout": "latest20pct_of_250_known_eligible_targets", "minimum_outer_training_events": 5,
                       "minimum_inner_training_events": 3, "probability_threshold_reported": .2,
                       "history": "latest,mean8,slope8 for every numeric CSA descriptor; richer outcome history",
                       "summaries_minimum_source_count": 4, "probabilities": "uncalibrated; no resampling/class reweighting"})
    write_json(out / "analysis_run_log.json", log)
    say("Nested model fitting complete")


def block_comparison(g, baseline, model, rng):
    w = g.pivot(index="physical_run_id", columns="model", values="probability")
    meta = g.drop_duplicates("physical_run_id").set_index("physical_run_id").loc[w.index]
    valid = w[[baseline, model]].notna().all(axis=1)
    w, meta = w[valid], meta[valid]
    delta = ((w[baseline]-meta.observed)**2-(w[model]-meta.observed)**2).to_numpy()
    blocks = []
    # Moving blocks of eight consecutive held-out targets within instrument.
    # Resample instruments too; wide uncertainty is expected with six hosts.
    for _, inst in meta.assign(position=np.arange(len(meta))).groupby("instrument_id"):
        ii = inst.sort_values("start").position.to_numpy()
        blocks.append([ii[j:j+8] for j in range(max(1, len(ii)-7))])
    boots = []
    for _ in range(2000):
        sample = []
        for j in rng.integers(0, len(blocks), len(blocks)):
            host_blocks = blocks[j]
            required = sum(len(b) for b in host_blocks)/len(host_blocks) + len(host_blocks)-1
            picks = []
            while len(picks) < required:
                picks.extend(host_blocks[rng.integers(len(host_blocks))])
            sample.extend(picks[:int(required)])
        boots.append(float(delta[sample].mean()))
    return {"baseline": baseline, "model": model, "n": len(delta), "events": int(meta.observed.sum()),
            "brier_improvement": float(delta.mean()), "ci_low": float(np.quantile(boots, .025)),
            "ci_high": float(np.quantile(boots, .975)), "uncertainty": "hierarchical_instrument_moving_eight_target_block_bootstrap_conditional_on_models"}


def summarize(out):
    p = pd.read_csv(out / "heldout_predictions.csv")
    cohort = pd.read_csv(out / "model_cohort.csv").set_index("physical_run_id")
    summaries, by_inst, subgroups, comparisons = [], [], [], []
    rng = np.random.default_rng(SEED)
    for (scheme, name), g in p.groupby(["scheme", "model"]):
        summaries.append({"scheme": scheme, "model": name, "folds": g.fold.nunique(), **previous.metric(g.observed, g.probability)})
        for host, z in g.groupby("hostname"):
            by_inst.append({"scheme": scheme, "model": name, "hostname": host, **previous.metric(z.observed, z.probability)})
        for label, mask in [
            ("all", np.ones(len(g), dtype=bool)),
            ("strict_immediate_eight", g.physical_run_id.map(cohort.strict_immediate_eight).to_numpy()),
            ("previous_run_ge10m", g.physical_run_id.map(cohort.previous_bad).eq(0).to_numpy()),
            ("all_eight_known_ge10m", (g.physical_run_id.map(cohort.known_bad_in_history8).eq(0) &
                                     g.physical_run_id.map(cohort.known_labels_in_history8).eq(8)).to_numpy()),
            ("GX9_chip", g.physical_run_id.map(cohort.chip_type).eq("GX9").to_numpy()),
            ("all_history_same_chip", g.physical_run_id.map(cohort.all_history_same_chip).to_numpy()),
            ("all_history_same_assay", g.physical_run_id.map(cohort.all_history_same_assay).to_numpy()),
            ("availability_margin_ge6h", g.physical_run_id.map(cohort.min_availability_margin_hours).ge(6).to_numpy()),
            ("history_span_le30d", g.physical_run_id.map(cohort.history_span_days).le(30).to_numpy()),
        ]:
            z = g[mask]
            if len(z):
                subgroups.append({"scheme": scheme, "model": name, "subgroup": label, **previous.metric(z.observed, z.probability)})
    for scheme, g in p.groupby("scheme"):
        for baseline in ["historical_event_rate", "instrument_event_rate", "eight_run_event_rate", "history_nested_selected"]:
            for name in ["history_nested_selected", "csa_nested_selected", "history_all_csa__logistic", "history_all_csa__extra_trees"]:
                if name != baseline:
                    comparisons.append({"scheme": scheme, **block_comparison(g, baseline, name, rng)})
    for filename, rows in [("model_performance_summary.csv", summaries), ("per_instrument_performance.csv", by_inst),
                           ("sensitivity_performance.csv", subgroups), ("incremental_forecast_value.csv", comparisons)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    return p, pd.DataFrame(summaries), pd.DataFrame(comparisons)


def validate(out):
    cohort = pd.read_csv(out / "model_cohort.csv").set_index("physical_run_id")
    members = pd.read_csv(out / "history_members.csv")
    lanes = pd.read_csv(out / "lane_label_audit.csv")
    model_members = members[members.target_id.isin(cohort.index)]
    assert model_members.groupby("target_id").size().eq(8).all()
    assert model_members.groupby("target_id").source_id.nunique().eq(8).all()
    assert model_members.target_id.ne(model_members.source_id).all()
    assert model_members.source_instrument.eq(model_members.target_instrument).all()
    assert pd.to_datetime(model_members.source_available, utc=True).lt(pd.to_datetime(model_members.target_start, utc=True)).all()
    independently = lanes.groupby("physical_run_id").total_reads.agg(lambda a: 1. if any(pd.notna(v) and v < THRESHOLD for v in a) else 0. if a.notna().all() else np.nan)
    np.testing.assert_array_equal(cohort.low_10m.to_numpy(), independently.loc[cohort.index].to_numpy())
    p = pd.read_csv(out / "heldout_predictions.csv")
    scores = pd.read_csv(out / "model_performance_summary.csv")
    for _, row in scores.iterrows():
        z = p[p.scheme.eq(row.scheme) & p.model.eq(row.model)]
        assert not z.physical_run_id.duplicated().any()
        assert np.allclose(((z.probability-z.observed)**2).mean(), row.brier, atol=1e-12)
        np.testing.assert_array_equal(z.observed.to_numpy(), cohort.loc[z.physical_run_id].low_10m.to_numpy())
    for _, r in pd.read_csv(out / "validation_split_audit.csv").iterrows():
        assert not set(r.train_ids.split(";")) & set(r.test_ids.split(";"))
        assert pd.Timestamp(r.train_max_available) < pd.Timestamp(r.cutoff)
    log = json.loads((out / "analysis_run_log.json").read_text())
    for path, digest in log["input_sha256"].items():
        # Code hash at prepare may change only for non-model presentation repairs;
        # such a repair is recorded separately and never silently accepted.
        if Path(path).name != Path(__file__).name:
            assert sha256(path) == digest, path
    x = load_wide_history(out / "eight_run_numeric_features.parquet")
    cat = pd.read_parquet(out / "eight_run_categorical_features.parquet")
    reloads = []
    for path in sorted((out / "models").glob("*.joblib")):
        payload = joblib.load(path)
        name = path.stem
        z = p[p.scheme.eq("final_chronological_holdout") & p.model.eq(name)]
        ii = cohort.reset_index().reset_index().set_index("physical_run_id").loc[z.physical_run_id, "index"].to_numpy()
        pre, k = payload["preprocessor"], payload["k"]
        predicted = payload["model"].predict_proba(pre.transform(x.loc[ii, payload["numeric_columns"]], cat.loc[ii, payload["categorical_columns"]])[:, -k:])[:, 1]
        np.testing.assert_allclose(predicted, z.probability.to_numpy(), atol=1e-12)
        reloads.append({"model": name, "n": len(z), "maximum_error": float(np.max(np.abs(predicted-z.probability.to_numpy())))})
    pd.DataFrame(reloads).to_csv(out / "saved_model_reload_checks.csv", index=False)
    result = {"status": "passed", "known_eligible_targets": len(cohort), "bad_targets": int(cohort.low_10m.sum()),
              "history_links_checked": len(model_members), "prediction_rows_checked": len(p), "saved_models_checked": len(reloads),
              "code_sha256_at_validation": sha256(__file__), "validated_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out / "artifact_validation.json", result)
    say(f"Artifact validation passed: {result}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=["prepare", "models", "summary", "validate", "all", "rebuild_precision"], default="all")
    ap.add_argument("--rebuild-from", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "rebuild_precision":
        if args.rebuild_from is None:
            raise SystemExit("--rebuild-from is required")
        rebuild_precision(args.rebuild_from, args.out)
    if args.stage in {"prepare", "all"}:
        if (args.out / "preparation_log.json").exists():
            raise SystemExit("Refusing to overwrite an existing experiment preparation")
        prepare(args.project, args.out)
    if args.stage in {"models", "all"}:
        train(args.out)
    if args.stage in {"summary", "all"}:
        summarize(args.out)
    if args.stage in {"validate", "all"}:
        validate(args.out)


if __name__ == "__main__":
    main()
