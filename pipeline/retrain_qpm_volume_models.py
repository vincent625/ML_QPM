#!/usr/bin/env python3
"""Six-instrument, command-audited QPM volume recognition (NOT metrology).

Each example is a single operation, not an averaged run-volume waveform.
Only the 32 baseline-centred waveform values enter predictive models. Commanded
volume supplies the label only. All learned transformations and family selection
are fold-local; physical runs never straddle training and validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
import multiprocessing
from pathlib import Path
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

VERSION = "qpm-command-audited-volume-2.0.0-expanded-er72"
SEED = 20261027
CLASSIFIERS = ["Robust template", "Logistic regression", "Linear SVM", "PCA + logistic", "Extra Trees"]
REGRESSORS = ["Elastic Net", "Linear SVR", "PCA + Ridge", "Extra Trees regressor"]
CONTEXT_COLUMNS = ["container_geometry_fingerprint", "deck_geometry_fingerprint", "workflow_stage"]
COLORS = ["#009E73", "#0072B2", "#56B4E9", "#CC79A7", "#D55E00", "#E69F00", "#666666", "#222222"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_center(profiles: np.ndarray) -> np.ndarray:
    return profiles - np.median(profiles[:, :3], axis=1, keepdims=True)


def load_audited(project: Path) -> tuple[pd.DataFrame, dict]:
    from qpm_trace_quality import eligible_waveforms
    columns = ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane", "run_start_time",
               "operation_id", "operation_index", "operation", "pipette", "programmed_volume_ul", "qpm_available",
               "qpm_profile", "pressure_unit", "liquid_class_index", "workflow_stage", "process_step", "loc"]
    wave = pq.read_table(project / "qpm_operation_waveforms.parquet", columns=columns).to_pandas()
    total_operations = len(wave)
    wave = wave[wave.qpm_available.fillna(False) & wave.programmed_volume_ul.gt(0)].copy()
    quality = pd.read_parquet(project / "qpm_trace_quality.parquet")
    wave = wave.merge(quality, on=["archive_id", "operation_index", "pipette"], how="left", validate="one_to_one")
    # No error-free assumption when command/response quality is unresolved.
    wave["qpm_waveform_eligible"] = wave.qpm_waveform_eligible.fillna(False)
    er72_count = int(wave.qpm_has_er72.fillna(False).sum())
    before_quality = len(wave)
    wave = eligible_waveforms(wave)
    context = pd.read_parquet(project / "qpm_physical_operation_context.parquet")
    frame = wave.merge(context, on=["archive_id", "operation_index", "pipette"], how="left", validate="one_to_one")
    alignment = frame.operation.eq(frame.command_operation) & np.isclose(frame.programmed_volume_ul, frame.command_volume_ul, equal_nan=False)
    profiles_valid = frame.qpm_profile.map(lambda v: len(v) == 32 and np.isfinite(v).all())
    keep = frame.qpm_available.fillna(False) & profiles_valid & alignment & frame.command_context_status.eq("command_verified") & frame.command_volume_ul.gt(0)
    audit = {"total_operations": total_operations, "qpm_available_positive_volume_operations": before_quality,
             "er72_operations_excluded": er72_count, "quality_excluded_operations": before_quality-len(wave),
             "quality_policy": "verified trace status required; entire Er72 operation/retries excluded",
             "verified_usable_operations": int(keep.sum()), "command_alignment_mismatches": int((~alignment).sum()),
             "liquid_class_labels_corrected_vs_current_table": int((frame.liquid_class_index.ne(frame.command_liquid_class_index) & keep).sum()),
             "context_status_counts": frame.command_context_status.fillna("missing_context").value_counts().to_dict(),
             "actual_delivered_volume_labels": 0}
    frame = frame.loc[keep].copy()
    frame["liquid_class_index"] = frame.command_liquid_class_index.astype(int)
    frame["programmed_volume_ul"] = frame.command_volume_ul.round(3)
    frame["run_start_time"] = pd.to_datetime(frame.run_start_time, utc=True, errors="coerce")
    audit["invalid_run_start_times"] = int(frame.run_start_time.isna().sum())
    return frame, audit


def select_cohort(frame: pd.DataFrame, liquid_class: int, operation: str, cap: int, max_runs: int = 600) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    task = frame[frame.liquid_class_index.eq(liquid_class) & frame.operation.eq(operation) & frame.pipette.eq("PIP2")].copy()
    definitions = task.groupby("liquid_class_fingerprint").agg(runs=("physical_run_id", "nunique"), instruments=("instrument_id", "nunique"), waves=("operation_id", "size"))
    definition = definitions.sort_values(["instruments", "runs", "waves"], ascending=False).index[0]
    matched = task[task.liquid_class_fingerprint.eq(definition)].copy()
    coverage = matched.groupby("programmed_volume_ul").agg(physical_runs=("physical_run_id", "nunique"), instruments=("instrument_id", "nunique"), waves=("operation_id", "size"))
    host_counts = matched.groupby(["programmed_volume_ul", "instrument_id"]).physical_run_id.nunique().unstack(fill_value=0)
    coverage["minimum_runs_on_each_instrument"] = host_counts.min(axis=1)
    # Coverage-only inclusion: do not require every rare instrument to provide
    # every volume. This would discard otherwise useful 43-instrument history.
    coverage["instruments_with_eight_runs"] = host_counts.ge(8).sum(axis=1)
    eligible = coverage[coverage.physical_runs.ge(100) & coverage.instruments.ge(3) & coverage.instruments_with_eight_runs.ge(3)]
    volumes = sorted(eligible.sort_values(["physical_runs", "waves"], ascending=False).head(5).index.tolist())
    if len(volumes) < 3:
        raise ValueError(f"LC{liquid_class} {operation}: only {len(volumes)} sufficiently supported volumes")
    selected = matched[matched.programmed_volume_ul.isin(volumes)].copy()
    raw_count = len(selected)
    population_runs = selected.physical_run_id.nunique()
    # Deterministic, equally allocated instrument sampling, independent of
    # waveform/outcomes. Retain all volume rows of each selected physical run.
    run_table = selected[["physical_run_id", "instrument_id"]].drop_duplicates()
    run_table["sampling_key"] = run_table.physical_run_id.map(lambda v: hashlib.sha256(f"{SEED}:{v}".encode()).hexdigest())
    run_table = run_table.sort_values("sampling_key")
    run_table["instrument_rank"] = run_table.groupby("instrument_id").cumcount()
    run_ids = run_table.sort_values(["instrument_rank", "sampling_key"]).head(max_runs).physical_run_id
    selected = selected[selected.physical_run_id.isin(run_ids)].copy()
    # A stable hash provides waveform/outcome-independent sampling. Capping
    # repetition prevents a repeated mixing operation from dominating the study.
    selected["sampling_key"] = selected.operation_id.map(lambda v: hashlib.sha256(f"{SEED}:{v}".encode()).hexdigest())
    selected = selected.sort_values("sampling_key").groupby(["physical_run_id", "programmed_volume_ul"], group_keys=False).head(cap)
    selected = selected.sort_values(["run_start_time", "physical_run_id", "programmed_volume_ul", "sampling_key"], kind="stable").reset_index(drop=True)
    matrix = baseline_center(np.vstack(selected.qpm_profile.to_numpy()))
    selected[[f"profile_{i:02d}" for i in range(32)]] = matrix
    task_id = f"lc{liquid_class}_{operation}"
    selected.insert(0, "task_id", task_id)
    coverage = coverage.reset_index()
    coverage.insert(0, "task_id", task_id)
    coverage["selected_volume"] = coverage.programmed_volume_ul.isin(volumes)
    cohort = {"task_id": task_id, "liquid_class_index": liquid_class, "operation": operation, "pipette": "PIP2",
              "volumes_ul": volumes, "liquid_class_fingerprint": definition, "liquid_class_name": matched.liquid_class_name.iloc[0],
              "total_verified_class_operation_waves": len(task), "other_full_liquid_class_definition_waves_excluded": len(task) - len(matched),
              "eligible_raw_waveforms": raw_count, "model_single_operation_waveforms": len(selected),
              "physical_runs": selected.physical_run_id.nunique(), "observations": selected.observation_id.nunique(),
              "instruments": selected.instrument_id.nunique(), "maximum_operations_per_physical_run_volume": cap,
              "population_supported_physical_runs": population_runs, "maximum_physical_runs_sampled": max_runs,
              "health_screen_applied": False,
              "aggregation": "none: each example is a single-operation waveform"}
    return selected, coverage, cohort


def dtw_matrix(x: np.ndarray, radius: int = 3) -> np.ndarray:
    """Fixed pairwise metric: no scaling, fitting, or labels outside a fold."""
    n, points = x.shape
    result = np.empty((n, n), np.float32)
    for i, query in enumerate(x):
        previous = np.full((n, points + 1), np.inf)
        previous[:, 0] = 0
        for a in range(1, points + 1):
            current = np.full_like(previous, np.inf)
            for b in range(max(1, a - radius), min(points, a + radius) + 1):
                current[:, b] = (x[:, b - 1] - query[a - 1]) ** 2 + np.minimum(np.minimum(previous[:, b], current[:, b - 1]), previous[:, b - 1])
            previous = current
        result[i] = np.sqrt(previous[:, points] / points)
    result = (result + result.T) / 2
    np.fill_diagonal(result, 0)
    return result


def make_estimator(name: str):
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=.2, class_weight="balanced", max_iter=3000, random_state=SEED))
    if name == "Linear SVM":
        return make_pipeline(StandardScaler(), LinearSVC(C=.1, class_weight="balanced", dual="auto", max_iter=10000, random_state=SEED))
    if name == "PCA + logistic":
        return make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver="full"), LogisticRegression(C=.2, class_weight="balanced", max_iter=3000, random_state=SEED))
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=2)
    if name == "Elastic Net":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=.1, l1_ratio=.5, max_iter=10000, random_state=SEED))
    if name == "Linear SVR":
        return make_pipeline(StandardScaler(), LinearSVR(C=10, epsilon=1, dual="auto", max_iter=10000, random_state=SEED))
    if name == "PCA + Ridge":
        return make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver="full"), Ridge(alpha=10))
    if name == "Extra Trees regressor":
        return ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, max_features=.7, random_state=SEED, n_jobs=2)
    raise KeyError(name)


def fit_predict(name: str, x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, distances: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
    if name == "Majority baseline":
        labels, counts = np.unique(y[train], return_counts=True)
        return np.full(len(test), labels[np.argmax(counts)])
    if name == "Median baseline":
        return np.full(len(test), np.median(y[train]))
    if name == "Context-only control":
        model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=.2, class_weight="balanced", max_iter=3000, random_state=SEED))
        model.fit(context[train], y[train].astype(str))
        return model.predict(context[test]).astype(float)
    if name == "Robust template":
        levels = np.unique(y[train])
        scores = []
        for value in levels:
            members = x[train][y[train] == value]
            median = np.median(members, axis=0)
            scale = np.maximum(1.4826 * np.median(np.abs(members - median), axis=0), 5.)
            scores.append(np.sqrt(np.mean(((x[test] - median) / scale) ** 2, axis=1)))
        return levels[np.argmin(np.column_stack(scores), axis=1)]
    if name in {"DTW 5-NN", "DTW regression"}:
        d = distances[np.ix_(test, train)]
        k = min(5, len(train))
        nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
        weights = 1 / np.maximum(np.take_along_axis(d, nearest, axis=1), 1e-6)
        labels = y[train][nearest]
        if name == "DTW regression":
            return np.sum(weights * labels, axis=1) / weights.sum(axis=1)
        levels = np.unique(y[train])
        votes = np.column_stack([np.sum(weights * (labels == v), axis=1) for v in levels])
        return levels[np.argmax(votes, axis=1)]
    model = make_estimator(name)
    model.fit(x[train], y[train])
    return model.predict(x[test])


def physical_weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1 / counts[inverse]


def metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, kind: str, weights: np.ndarray | None = None) -> dict:
    weight = physical_weights(groups) if weights is None else weights
    result = {"mae_ul": float(mean_absolute_error(y, pred, sample_weight=weight)),
              "rmse_ul": float(np.sqrt(mean_squared_error(y, pred, sample_weight=weight))),
              "r2": float(r2_score(y, pred, sample_weight=weight)) if len(y) > 1 else np.nan,
              "spearman": float(spearmanr(y, pred).statistic) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else np.nan,
              "within_10pct": float(np.average(np.abs(pred - y) <= .1 * y, weights=weight))}
    if kind == "classification":
        result.update(accuracy=float(accuracy_score(y, pred, sample_weight=weight)),
                      balanced_accuracy=float(balanced_accuracy_score(y, pred, sample_weight=weight)))
    return result


def grouped_splits(frame: pd.DataFrame, folds: int = 5) -> list[tuple[str, np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return [(f"fold_{i+1}", tr, te) for i, (tr, te) in enumerate(cv.split(np.zeros((len(frame), 1)), frame.programmed_volume_ul, frame.physical_run_id))]


def outer_splits(frame: pd.DataFrame) -> dict:
    instruments = frame.instrument_id.to_numpy()
    result = {"grouped_run": grouped_splits(frame),
              "unseen_instrument": [(str(host), np.flatnonzero(instruments != host), np.flatnonzero(instruments == host)) for host in sorted(np.unique(instruments))]}
    times = frame.groupby("physical_run_id").run_start_time.min().dropna().sort_values(kind="stable")
    cutoff = times.iloc[max(0, int(len(times) * .8) - 1)]
    train_ids = times[times <= cutoff].index
    test_ids = times[times > cutoff].index
    train = np.flatnonzero(frame.physical_run_id.isin(train_ids))
    test = np.flatnonzero(frame.physical_run_id.isin(test_ids))
    if len(test_ids) >= 10 and len(train_ids) >= 30:
        result["chronological"] = [("latest_20pct", train, test)]
    for scheme, splits in result.items():
        for _, tr, te in splits:
            if set(frame.physical_run_id.iloc[tr]) & set(frame.physical_run_id.iloc[te]):
                raise AssertionError(f"Physical-run leakage in {scheme}")
            if not set(frame.programmed_volume_ul.iloc[te]).issubset(frame.programmed_volume_ul.iloc[tr]):
                raise ValueError(f"Unseen commanded-volume level in {scheme}; do not silently extrapolate")
            if scheme == "chronological" and frame.run_start_time.iloc[tr].max() >= frame.run_start_time.iloc[te].min():
                raise AssertionError("Chronological split is not strictly ordered")
    return result


def inner_selection(frame: pd.DataFrame, train: np.ndarray, x: np.ndarray, y: np.ndarray, distances: np.ndarray) -> tuple[dict, list]:
    local = frame.iloc[train].reset_index(drop=True)
    inner = grouped_splits(local, 3)
    rows = []
    choices = {}
    for kind, candidates in [("classification", CLASSIFIERS), ("regression", REGRESSORS)]:
        scores = []
        for name in candidates:
            fold_scores = []
            for _, tr, te in inner:
                prediction = fit_predict(name, x, y, train[tr], train[te], distances)
                score = metrics(y[train[te]], prediction, local.physical_run_id.iloc[te].to_numpy(), kind)
                fold_scores.append(score["balanced_accuracy"] if kind == "classification" else -score["mae_ul"])
            score = float(np.mean(fold_scores))
            rows.append({"kind": kind, "model": name, "inner_selection_score": score})
            scores.append(score)
        choices[kind] = candidates[int(np.argmax(scores))]
    return choices, rows


def cluster_interval(part: pd.DataFrame, kind: str, reps: int = 100) -> tuple[float, float]:
    """Conditional uncertainty of held-out predictions; NOT model refitting."""
    rng = np.random.default_rng(SEED)
    groups, inverse = np.unique(part.physical_run_id, return_inverse=True)
    values = []
    truth = part.programmed_volume_ul.to_numpy()
    prediction = part.predicted_volume_ul.to_numpy()
    folds = part.fold.to_numpy()
    for _ in range(reps):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weight = physical_weights(part.physical_run_id.to_numpy()) * counts[inverse]
        fold_values = []
        for fold in np.unique(folds):
            mask = (folds == fold) & (weight > 0)
            if mask.sum() < 2:
                continue
            score = metrics(truth[mask], prediction[mask], part.physical_run_id.to_numpy()[mask], kind, weight[mask])
            fold_values.append(score["balanced_accuracy"] if kind == "classification" else score["mae_ul"])
        if fold_values:
            values.append(np.mean(fold_values))
    return tuple(np.quantile(values, [.025, .975]).tolist())


def evaluate(frame: pd.DataFrame, distances: np.ndarray, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = frame[[f"profile_{i:02d}" for i in range(32)]].to_numpy()
    y = frame.programmed_volume_ul.to_numpy()
    groups = frame.physical_run_id.to_numpy()
    context = frame[CONTEXT_COLUMNS].fillna("unknown").to_numpy(str)
    predictions, fold_rows, selections = [], [], []
    task_id = frame.task_id.iloc[0]
    for scheme, splits in outer_splits(frame).items():
        for fold, train, test in splits:
            print(f"{task_id} {scheme} {fold}: nested selection; train={len(train)}, test={len(test)}", flush=True)
            choices, inner = inner_selection(frame, train, x, y, distances)
            selections.extend({"task_id": task_id, "scheme": scheme, "fold": fold, **row, "selected": row["model"] == choices[row["kind"]]} for row in inner)
            for kind, names in [("classification", ["Majority baseline", "Context-only control"] + CLASSIFIERS), ("regression", ["Median baseline"] + REGRESSORS)]:
                cached = {}
                for name in names:
                    started = time.perf_counter()
                    pred = fit_predict(name, x, y, train, test, distances, context)
                    cached[name] = pred
                    score = metrics(y[test], pred, groups[test], kind)
                    fold_rows.append({"task_id": task_id, "scheme": scheme, "fold": fold, "kind": kind, "model": name,
                                      "test_waveforms": len(test), "test_physical_runs": len(np.unique(groups[test])),
                                      "fit_predict_seconds": time.perf_counter() - started, **score})
                    part = frame.iloc[test][["task_id", "operation_id", "physical_run_id", "instrument_id", "run_start_time", "programmed_volume_ul"]].copy()
                    part["predicted_volume_ul"] = pred
                    part["scheme"], part["fold"], part["kind"], part["model"] = scheme, fold, kind, name
                    predictions.append(part)
                chosen = choices[kind]
                score = metrics(y[test], cached[chosen], groups[test], kind)
                fold_rows.append({"task_id": task_id, "scheme": scheme, "fold": fold, "kind": kind, "model": "Nested-selected procedure",
                                  "selected_model": chosen, "test_waveforms": len(test), "test_physical_runs": len(np.unique(groups[test])), **score})
                part = predictions[-len(names) + names.index(chosen)].copy()
                part["model"] = "Nested-selected procedure"
                part["selected_model"] = chosen
                predictions.append(part)
        pd.DataFrame(fold_rows).to_csv(output / f"{task_id}_fold_checkpoint.csv", index=False)
    preds = pd.concat(predictions, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    summary = []
    numeric = ["mae_ul", "rmse_ul", "r2", "spearman", "within_10pct", "accuracy", "balanced_accuracy"]
    for (scheme, kind, name), part in preds.groupby(["scheme", "kind", "model"], sort=False):
        rows = folds[(folds.scheme == scheme) & (folds.kind == kind) & (folds.model == name)]
        lo, hi = cluster_interval(part, kind)
        pooled = metrics(part.programmed_volume_ul.to_numpy(), part.predicted_volume_ul.to_numpy(), part.physical_run_id.to_numpy(), kind)
        result = {"task_id": task_id, "scheme": scheme, "kind": kind, "model": name,
                  "test_waveforms": len(part), "test_physical_runs": part.physical_run_id.nunique(), "test_instruments": part.instrument_id.nunique(),
                  "folds": len(rows), "ci_low": lo, "ci_high": hi,
                  "ci_metric": "balanced_accuracy" if kind == "classification" else "mae_ul",
                  **{metric: float(rows[metric].mean()) for metric in numeric if metric in rows},
                  **{f"pooled_{key}": value for key, value in pooled.items()}}
        summary.append(result)
    return pd.DataFrame(summary), folds, preds, pd.DataFrame(selections)


def save_final_models(frame: pd.DataFrame, distances: np.ndarray, output: Path, cohort: dict) -> list[dict]:
    x = frame[[f"profile_{i:02d}" for i in range(32)]].to_numpy()
    y = frame.programmed_volume_ul.to_numpy()
    choices, scores = inner_selection(frame, np.arange(len(frame)), x, y, distances)
    result = []
    for kind, name in choices.items():
        payload = {"version": VERSION, "model_name": name, "kind": kind, "cohort": cohort,
                   "label_basis": "verified_commanded_volume_not_measured_delivered_volume",
                   "features": [f"profile_{i:02d}" for i in range(32)], "baseline": "per_waveform_first_three_points_median",
                   "actual_volume_prediction_validated": False, "deployment_enabled": False}
        if name in {"Robust template", "DTW 5-NN", "DTW regression"}:
            payload.update(training_profiles=x, training_commanded_volumes=y)
        else:
            model = make_estimator(name)
            model.fit(x, y)
            payload["estimator"] = model
        path = output / f"{cohort['task_id']}_{kind}_model.joblib"
        joblib.dump(payload, path)
        result.append({"task_id": cohort["task_id"], "kind": kind, "model": name, "path": path.name, "sha256": sha256(path),
                       "selection_basis": "3-fold physical-run CV using all development data; nested outer results reported separately"})
    return result


def execute_task(task: pd.DataFrame, cohort: dict, output: Path) -> tuple:
    task_output = output / "tasks" / cohort["task_id"]
    task_output.mkdir(parents=True)
    captured = []
    warnings.showwarning = lambda message, category, filename, lineno, *args, **kwargs: captured.append(f"{category.__name__}: {message}")
    x = task[[f"profile_{i:02d}" for i in range(32)]].to_numpy()
    started = time.perf_counter()
    distances = dtw_matrix(x) if "DTW 5-NN" in CLASSIFIERS else np.empty((0, 0))
    cohort["dtw_seconds"] = time.perf_counter() - started
    results = evaluate(task, distances, task_output)
    for name, table in zip(["model_performance.csv", "fold_performance.csv", "heldout_predictions.csv", "nested_model_selections.csv"], results):
        table.to_csv(task_output / name, index=False)
    models = save_final_models(task, distances, output, cohort)
    (task_output / "warnings.json").write_text(json.dumps(sorted(set(captured)), indent=2))
    return (*results, models, cohort, sorted(set(captured)))


def create_figure(performance: pd.DataFrame, predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    tasks = sorted(performance.task_id.unique())
    labels = [task.replace("lc", "LC").replace("_aspirate", " aspiration").replace("_dispense", " dispense") for task in tasks]
    candidates = performance[~performance.model.isin(["Majority baseline", "Median baseline", "Context-only control", "Nested-selected procedure"])]
    best = []
    for (task, kind), group in candidates[candidates.scheme.eq("unseen_instrument")].groupby(["task_id", "kind"]):
        row = group.sort_values("balanced_accuracy", ascending=False).iloc[0] if kind == "classification" else group.sort_values("mae_ul").iloc[0]
        best.append(row)
    winners = pd.DataFrame(best)
    for ax, kind, models, title in [(axes[0, 0], "classification", CLASSIFIERS, "A  Unseen-instrument volume-level recognition"),
                                    (axes[0, 1], "regression", REGRESSORS, "B  Unseen-instrument continuous-volume prediction")]:
        subset = performance[performance.scheme.eq("unseen_instrument") & performance.kind.eq(kind)]
        metric = "balanced_accuracy" if kind == "classification" else "mae_ul"
        matrix = subset.pivot(index="task_id", columns="model", values=metric).reindex(index=tasks, columns=models).to_numpy()
        shown = matrix * 100 if kind == "classification" else matrix
        image = ax.imshow(shown, cmap="cividis" if kind == "classification" else "cividis_r", aspect="auto", vmin=0, vmax=100 if kind == "classification" else None)
        ax.set_xticks(np.arange(len(models)), [name.replace(" regression", "\nregression").replace(" regressor", "\nregressor").replace(" + ", " +\n") for name in models], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(tasks)), labels)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
        for i in range(len(tasks)):
            for j in range(len(models)):
                val = shown[i, j]
                threshold = 55 if kind == "classification" else np.nanmax(shown) * .5
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black" if (val > threshold if kind == "classification" else val < threshold) else "white", fontsize=10)
        fig.colorbar(image, ax=ax, shrink=.72, label="Balanced accuracy (%)" if kind == "classification" else "MAE vs commanded volume (µL); lower is better")
    ax = axes[1, 0]
    for index, scheme in enumerate(["grouped_run", "unseen_instrument", "chronological"]):
        subset = performance[performance.scheme.eq(scheme) & performance.kind.eq("classification") & performance.model.eq("Nested-selected procedure")].set_index("task_id").reindex(tasks)
        val = subset.balanced_accuracy.to_numpy() * 100
        low, high = subset.ci_low.to_numpy() * 100, subset.ci_high.to_numpy() * 100
        pos = np.arange(len(tasks)) + (index - 1) * .18
        ax.errorbar(val, pos, xerr=np.vstack([np.maximum(0, val-low), np.maximum(0, high-val)]), fmt="o", capsize=3,
                    color=COLORS[index], label={"grouped_run": "Unseen run", "unseen_instrument": "Unseen instrument", "chronological": "Later runs"}[scheme])
    ax.axvline(20, ls=":", color="#777777", label="Five-level chance")
    ax.set_yticks(np.arange(len(tasks)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Balanced accuracy (%); nested family selection")
    ax.set_title("C  Model choice graded on untouched outer folds", loc="left", fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[1, 1]
    host_map = sorted(predictions.instrument_id.unique())
    # Predictions store privacy-preserving instrument IDs; the run log supplies
    # host mapping, without exposing serials or any sample-identifying fields.
    mapping = json.loads((output / "instrument_mapping.json").read_text())
    nested = predictions[predictions.scheme.eq("unseen_instrument") & predictions.kind.eq("regression") & predictions.model.eq("Nested-selected procedure")]
    values = np.full((len(tasks), len(host_map)), np.nan)
    for i, task in enumerate(tasks):
        for j, host in enumerate(host_map):
            part = nested[nested.task_id.eq(task) & nested.instrument_id.eq(host)]
            if not part.empty:
                values[i, j] = metrics(part.programmed_volume_ul.to_numpy(), part.predicted_volume_ul.to_numpy(), part.physical_run_id.to_numpy(), "regression")["mae_ul"]
    im = ax.imshow(values, cmap="cividis_r", aspect="auto", vmin=0)
    ax.set_xticks(range(len(host_map)), [mapping.get(host, host).split(".")[0].replace("gnxs-", "") for host in host_map], rotation=90, fontsize=6)
    ax.set_yticks(range(len(tasks)), labels)
    for i in range(len(tasks)):
        for j in range(len(host_map)):
            if len(host_map) <= 10:
                ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", color="black" if values[i,j] < np.nanmax(values)/2 else "white")
    ax.set_xlabel("Held-out instrument")
    ax.set_title("D  Volume error by unseen instrument", loc="left", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=.72, label="MAE vs commanded volume (µL)")
    fig.suptitle(f"QPM volume models retrained across {len(host_map)} Genexus instruments\nVerified commands · Er72 excluded · physical-run-separated validation", fontsize=19, fontweight="bold", color="#264653")
    fig.supxlabel("Labels are commanded volumes—not measured delivery. No actual under/over-transfer accuracy is established.", fontsize=11, color="#264653")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"qpm_six_instrument_comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return winners


def write_report(output: Path, performance: pd.DataFrame, winners: pd.DataFrame, cohorts: list, audit: dict, run_log: dict) -> None:
    table = winners[["task_id", "kind", "model", "balanced_accuracy", "mae_ul", "rmse_ul", "r2", "ci_low", "ci_high"]].copy()
    comparison = performance[performance.scheme.eq("unseen_instrument") & performance.kind.eq("classification") & performance.model.isin(["Majority baseline", "Context-only control", "Nested-selected procedure"])][["task_id", "model", "balanced_accuracy", "accuracy", "mae_ul"]]
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>Expanded-bank QPM volume models</title>
    <style>body{{font:16px Arial;background:#fbfcf8;color:#264653;max-width:1400px;margin:35px auto;padding:20px}} h1,h2{{color:#264653}} .warning{{background:#fff0d2;padding:18px;border-left:5px solid #E69F00}} img{{width:100%}} table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:8px;border-bottom:1px solid #dde5df;text-align:left}} a{{color:#0072B2}}</style></head><body>
    <h1>Expanded-bank QPM volume-model retraining</h1><p>{html.escape(VERSION)} · {run_log['completed_utc']}</p>
    <div class='warning'><strong>What is—and is not—predicted:</strong> the label is the volume in the verified GA/GD aspiration/dispense command. There are zero independent actual-delivery labels. A low commanded-volume MAE does not measure under/over delivery, air intake, or transferred sample mass. Deployment and actual-volume claims remain disabled.</div>
    <h2>Comparison</h2><img src='qpm_six_instrument_comparison.png'><p>Heatmaps A/B compare fixed model families; selecting the largest outer score post hoc is descriptive, not independent validation of that winner. Panel C grades family selection within nested CV. Panel D reports nested-selected regression error for each unseen instrument. Equal physical-run weights within folds; equal outer-fold means. Intervals resample physical runs using already held-out predictions (conditional uncertainty, not full model-refit uncertainty).</p>
    <h2>Best observed family per liquid class and operation</h2>{table.to_html(index=False, float_format=lambda v: f'{v:.4f}')}
    <h2>Baseline and geometry/workflow shortcut control</h2>{comparison.to_html(index=False, float_format=lambda v: f'{v:.4f}')}
    <p>The context-only model uses container geometry, deck geometry, and broad workflow stage, not step names or locations that can encode volume. High context performance is evidence of a programmed-protocol shortcut, not volume metrology. See <a href='context_volume_support.csv'>context–volume support</a>. The same target volume can occur in different liquids/operations; liquid identity and actual transfer quality remain unverified.</p>
    <h2>Dataset</h2>{pd.DataFrame(cohorts).to_html(index=False)}
    <p>PIP2 only, LC49/55/61, aspiration and dispense analyzed separately; up to five supported discrete commanded-volume levels per task. Only one full liquid-class definition per task. Command configuration and trace quality must be verified; Er72 operations/retries are excluded. Individual waveforms are hash-sampled with a per-run/volume cap, and physical runs are bounded by an instrument-balanced deterministic cap (see dataset table). No mean/median waveform aggregation. Levels require ≥100 physical runs and ≥3 instruments with ≥8 physical runs each. Level selection uses coverage only, not curve shapes, run outcomes, or accuracy. Healthy-reference screening does not restrict this predictive cohort: good and bad runs are included.</p>
    <h2>Leakage and validation</h2><ul><li>Only 32 baseline-centred waveform samples enter volume models. No command volume, step number/name, path/ID, timing field, sample identity, outcomes, or target-derived metadata enters X.</li><li>Five outer grouped physical-run folds; leave-one-instrument-out folds; latest 20% physical runs held out chronologically. Every model/scaler/PCA and three-fold inner family selection is fit only on the training partition. Small held-out instruments remain in the audit and have imprecise estimates.</li><li>All lanes/reanalyses/repeated traces of one physical run stay together. Fixed hyperparameters were declared before execution. Model families are chosen independently for each outer fold.</li><li>Raw sensor counts are unresolved, not physical pressure units. Cross-instrument predictive tests assess empirical transportability, not calibration or universal raw-count references.</li></ul>
    <h2>Limitations and next validation</h2><p>The trained task is five-level command recognition; continuous regression between these levels is not a calibrated actual-volume measurement. Geometry, fluid composition, tip state, and step can co-vary with command volume. Chronological tests use the historical collection, not a prospective experiment. These results do not validate classifying an abnormal transfer as a measured volume deficit.</p>
    <p>Next: an engineering-approved bench study with gravimetric or validated dye-based measured-volume labels. Block by liquid class, container/tip geometry, operation, pipette and instrument; randomize approved low/mid/high volumes and include controlled transfer faults. Reserve entire runs and an instrument for validation; evaluate µL bias, MAE, repeatability, and coverage of prediction intervals. Do not deliberately introduce faults into sample/customer runs.</p>
    <h2>Files</h2><p><a href='model_performance.csv'>All model performance</a> · <a href='heldout_predictions.parquet'>Held-out predictions</a> · <a href='fold_performance.csv'>Per-fold performance</a> · <a href='nested_model_selections.csv'>Nested selections</a> · <a href='analysis_run_log.json'>Provenance/run log</a></p></body></html>"""
    (output / "qpm_volume_model_report.html").write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operations-per-run-volume", type=int, default=2)
    parser.add_argument("--maximum-physical-runs-per-task", type=int, default=450)
    parser.add_argument("--include-dtw", action="store_true", help="Quadratic-memory optional comparator; omitted by default at expanded-bank scale")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.include_dtw:
        CLASSIFIERS.append("DTW 5-NN")
        REGRESSORS.append("DTW regression")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    warning_messages = []
    warnings.showwarning = lambda message, category, filename, lineno, *args, **kwargs: warning_messages.append(f"{category.__name__}: {message}")
    frame, audit = load_audited(args.project)
    print("Audited inputs:", json.dumps(audit), flush=True)
    inventory = pd.read_csv(args.project / "csa_inventory.csv", usecols=["archive_id", "hostname"])
    inventory = inventory.drop_duplicates("archive_id")
    mapping = frame[["archive_id", "instrument_id"]].drop_duplicates().merge(inventory, on="archive_id", validate="one_to_one").groupby("instrument_id").hostname.unique()
    if any(len(hosts) != 1 for hosts in mapping):
        raise ValueError("Instrument-to-host identity is ambiguous")
    mapping = {instrument: hosts[0] for instrument, hosts in mapping.items()}
    (output / "instrument_mapping.json").write_text(json.dumps(mapping, indent=2))
    all_performance, all_folds, all_predictions, all_selections, all_coverage, cohorts, models = [], [], [], [], [], [], []
    members, context_support = [], []
    jobs = []
    for liquid_class in [49, 55, 61]:
        for operation in ["aspirate", "dispense"]:
            task, coverage, cohort = select_cohort(frame, liquid_class, operation, args.operations_per_run_volume, args.maximum_physical_runs_per_task)
            cohorts.append(cohort)
            all_coverage.append(coverage)
            members.append(task.drop(columns=["qpm_profile"]).copy())
            support = task.groupby(CONTEXT_COLUMNS + ["programmed_volume_ul"], dropna=False).agg(physical_runs=("physical_run_id", "nunique"), waveforms=("operation_id", "size")).reset_index()
            support.insert(0, "task_id", cohort["task_id"])
            context_support.append(support)
            print("Cohort:", json.dumps(cohort), flush=True)
            jobs.append((task, cohort))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(execute_task, task, cohort, output) for task, cohort in jobs]
        for future in as_completed(futures):
            performance, folds, predictions, selections, task_models, finished_cohort, captured = future.result()
            all_performance.append(performance)
            all_folds.append(folds)
            all_predictions.append(predictions)
            all_selections.append(selections)
            models.extend(task_models)
            warning_messages.extend(captured)
            for cohort in cohorts:
                if cohort["task_id"] == finished_cohort["task_id"]:
                    cohort.update(finished_cohort)
            pd.concat(all_performance).to_csv(output / "model_performance.csv", index=False)
            pd.concat(all_folds).to_csv(output / "fold_performance.csv", index=False)
            print(f"Completed {finished_cohort['task_id']} in {time.perf_counter()-started:.1f}s cumulative", flush=True)
    performance = pd.concat(all_performance, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(output / "heldout_predictions.parquet", index=False)
    predictions.to_csv(output / "heldout_predictions.csv", index=False)
    pd.concat(all_selections).to_csv(output / "nested_model_selections.csv", index=False)
    pd.concat(all_coverage).to_csv(output / "volume_coverage.csv", index=False)
    pd.concat(context_support).to_csv(output / "context_volume_support.csv", index=False)
    pd.concat(members).to_parquet(output / "training_cohort.parquet", index=False)
    pd.DataFrame(cohorts).to_csv(output / "cohorts.csv", index=False)
    per_host = []
    for keys, part in predictions.groupby(["task_id", "scheme", "kind", "model", "instrument_id"]):
        per_host.append(dict(zip(["task_id", "scheme", "kind", "model", "instrument_id"], keys), hostname=mapping[keys[-1]],
                             physical_runs=part.physical_run_id.nunique(), **metrics(part.programmed_volume_ul.to_numpy(), part.predicted_volume_ul.to_numpy(), part.physical_run_id.to_numpy(), keys[2])))
    pd.DataFrame(per_host).to_csv(output / "per_instrument_performance.csv", index=False)
    winners = create_figure(performance, predictions, output)
    winners.to_csv(output / "best_observed_models.csv", index=False)
    manifest = args.project / "reference_bank_manifest.json"
    bank = json.loads((manifest if manifest.exists() else args.project / "reference_bank_refresh.json").read_text())
    transfers = pd.read_csv(args.project / "csa_transfer_manifest.csv", usecols=["archive_id", "sha256"])
    archive_ids = set(pd.concat(members).archive_id)
    log = {"version": VERSION, "code_sha256": sha256(Path(__file__)), "random_seed": SEED,
           "completed_utc": datetime.now(timezone.utc).isoformat(), "execution_seconds": time.perf_counter()-started,
           "bank_id": bank["bank_id"], "source_sha256": {name: sha256(args.project / name) for name in ["qpm_operation_waveforms.parquet", "qpm_physical_operation_context.parquet", "qpm_trace_quality.parquet", "csa_ml_dataset.parquet"]},
           "input_archive_hashes": dict(zip(transfers[transfers.archive_id.isin(archive_ids)].archive_id, transfers[transfers.archive_id.isin(archive_ids)].sha256)),
           "feature_policy_version": VERSION, "features": "32 baseline-centred waveform samples only", "audit": audit,
           "cohorts": cohorts, "models": models, "model_settings": {name: str(make_estimator(name)) for name in CLASSIFIERS + REGRESSORS if name not in {"Robust template", "DTW 5-NN", "DTW regression"}},
           "validation": "5 grouped folds / leave-one-instrument-out folds / latest 20% chronology; inner 3-fold grouped family selection",
           "uncertainty": "100 physical-run bootstrap replicates of fixed held-out predictions, no refitting", "actual_volume_prediction_validated": False,
           "plugin_deployment_enabled": False, "warnings": sorted(set(warning_messages)),
           "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    (output / "analysis_run_log.json").write_text(json.dumps(log, indent=2))
    write_report(output, performance, winners, cohorts, audit, log)
    print("COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
