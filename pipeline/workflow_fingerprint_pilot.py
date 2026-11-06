#!/usr/bin/env python3
"""Multimodal CSA pilot: pressure-waveform and spatial workflow fingerprints.

This is an unsupervised review-prioritization pipeline.  It preserves raw
waveform order and spatial block topology, evaluates anomaly models only on
held-out physical runs, and never presents an anomaly as a proven root cause.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import shutil
import sys
import tarfile
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import analyze_csa as core


SEED = core.SEED
VERSION = "csa-multimodal-fingerprint-pilot-1.3"
PROFILE_POINTS = 64
MIN_REFERENCE_TRACES = 8
MIN_REFERENCE_RUNS = 6

PRESSURE_RE = re.compile(
    r"(?:^|/)(pipPressure_lane\d+_try\d+_PIP\d+|"
    r"pipInCoupler_lane\d+_try\d+_PIP\d+|"
    r"pipInCouplerDsp_lane\d+_try\d+_PIP\d+|"
    r"Vacuum_PressureDataLANE\d+_PIP\d+|"
    r"blockedTipPressureCurve_\d+)\.csv$",
    re.I,
)
SPATIAL_RE = re.compile(
    r"^CSA/outputs/SigProcActor-00/block_X(?P<x>\d+)_Y(?P<y>\d+)/"
    r"NucStep/NucStep_(?P<location>inlet|middle|outlet)_step\.txt$",
    re.I,
)
TIMELINE_BASES = {"scriptstatus.csv", "vacuum_log.csv"}
SAFE_CONTEXT = (
    "global.analysis_application_type",
    "ion_params_00.chip_info_chip_type",
    "ion_params_00.experiment_analysis_settings_library_kit_name",
    "planned_run.object_script_name",
)
ENGINEERED = (
    "level_start", "level_end", "mean", "median", "std", "minimum", "maximum",
    "q05", "q95", "amplitude", "area_signed", "area_absolute",
    "peak_abs_value", "time_to_peak_fraction", "initial_slope", "final_slope",
    "largest_drop", "largest_rise", "recovery_error", "derivative_noise",
    "second_difference_noise", "derivative_sign_changes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: Any, prefix: str = "") -> str:
    text = "|".join("" if x is None else str(x) for x in parts)
    return prefix + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def normalize_name(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")).lower()


def safe_float(value: Any) -> float:
    try:
        result = float(str(value).strip())
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def selected_member(name: str) -> bool:
    clean = name.replace("\\", "/")
    base = Path(clean).name.lower()
    return bool(PRESSURE_RE.search(clean) or SPATIAL_RE.match(clean) or base in TIMELINE_BASES)


def archive_members(path: Path, max_bytes: int = 20_000_000) -> Iterator[tuple[int, str, int, str, bytes]]:
    """Stream selected top-level members; CSA payloads here are not nested."""
    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        with zipfile.ZipFile(path) as archive:
            for order, entry in enumerate(archive.infolist()):
                if entry.is_dir() or not selected_member(entry.filename):
                    continue
                if entry.file_size > max_bytes:
                    raise ValueError(f"selected member too large: {entry.filename}")
                stamp = datetime(*entry.date_time, tzinfo=timezone.utc).isoformat()
                yield order, entry.filename, entry.file_size, stamp, archive.read(entry)
        return
    if not (signature.startswith(b"\xfd7zXZ\x00") or signature.startswith(b"\x1f\x8b")):
        raise ValueError(f"unrecognized archive signature: {path.name}")
    with tarfile.open(path, mode="r|*") as archive:
        for order, entry in enumerate(archive):
            if not entry.isfile() or not selected_member(entry.name):
                continue
            if entry.size > max_bytes:
                raise ValueError(f"selected member too large: {entry.name}")
            handle = archive.extractfile(entry)
            if handle is None:
                continue
            with handle:
                data = handle.read(max_bytes + 1)
            stamp = datetime.fromtimestamp(entry.mtime, timezone.utc).isoformat()
            yield order, entry.name, entry.size, stamp, data


def choose_pilot(project: Path, per_instrument: int) -> tuple[pd.DataFrame, dict[str, str]]:
    duplicates = pd.read_csv(project / "duplicate_archives.csv", dtype={"archive_id": str})
    manifest = pd.read_csv(project / "csa_transfer_manifest.csv", dtype={"archive_id": str})
    columns = [
        "archive_id", "physical_run_id", "observation_id", "instrument_id",
        "chip_lane", "run_start_time", "total_reads", "mean_read_length_bp",
        *SAFE_CONTEXT,
    ]
    dataset = pd.read_parquet(project / "csa_ml_dataset.parquet", columns=columns)
    for frame in (duplicates, manifest, dataset):
        frame["archive_id"] = frame["archive_id"].astype(str)
    primary = duplicates[duplicates["retained_primary"].astype(bool)].copy()
    primary = primary.merge(
        dataset.drop_duplicates("archive_id"),
        on=["archive_id", "physical_run_id", "observation_id"],
        suffixes=("_duplicate", ""),
    )
    primary = primary.merge(
        manifest[["archive_id", "local_path", "size_bytes", "sha256", "container_format"]],
        on="archive_id",
        how="inner",
    )
    primary = primary.sort_values(
        ["instrument_id", "physical_run_id", "completeness_score"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates("physical_run_id")
    selected = []
    for _, group in primary.groupby("instrument_id", sort=True):
        group = group.sort_values(["run_start_time", "physical_run_id"], kind="stable").reset_index(drop=True)
        count = min(per_instrument, len(group))
        indices = np.unique(np.round(np.linspace(0, len(group) - 1, count)).astype(int))
        selected.append(group.iloc[indices])
    pilot = pd.concat(selected, ignore_index=True)
    pilot["pilot_order"] = np.arange(1, len(pilot) + 1)
    pilot["workflow_context"] = (
        pilot["planned_run.object_script_name"].fillna("unknown").astype(str)
        + "|" + pilot["global.analysis_application_type"].fillna("unknown").astype(str)
        + "|" + pilot["ion_params_00.chip_info_chip_type"].fillna("unknown").astype(str)
    )
    paths = dict(zip(pilot["archive_id"], pilot["local_path"]))
    public = pilot.drop(columns=["local_path"]).copy()
    return public, paths


def pressure_metadata(name: str) -> dict[str, Any]:
    base = Path(name).name
    lower = base.lower()
    if lower.startswith("pippressure"):
        family, operation = "pipette_pressure", "operation_unresolved"
    elif lower.startswith("pipincouplerdsp"):
        family, operation = "pipette_in_coupler_dispense", "dispense"
    elif lower.startswith("pipincoupler"):
        family, operation = "pipette_in_coupler", "coupler_operation_unresolved"
    elif lower.startswith("vacuum_pressure"):
        family, operation = "vacuum_pressure", "vacuum"
    else:
        family, operation = "blocked_tip_pressure", "blocked_tip_test"
    lane_match = re.search(r"lane(\d+)", base, re.I)
    pip_match = re.search(r"pip(\d+)", base, re.I)
    attempt_match = re.search(r"try(\d+)", base, re.I)
    if family == "blocked_tip_pressure":
        pip_match = re.search(r"_(\d+)\.csv$", base, re.I)
    return {
        "trace_family": family,
        "operation": operation,
        "lane": int(lane_match.group(1)) if lane_match else np.nan,
        "pipette": f"PIP{pip_match.group(1)}" if pip_match else "unresolved",
        "attempt": int(attempt_match.group(1)) if attempt_match else 0,
        "attempt_bucket": (
            "retry" if attempt_match and int(attempt_match.group(1)) > 0
            else "first_or_only"
        ),
    }


def parse_pressure(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    meta = pressure_metadata(name)
    text = data.decode("utf-8-sig", "replace")
    times: list[float] = []
    values: list[float] = []
    if meta["trace_family"] == "vacuum_pressure":
        for row in csv.reader(io.StringIO(text)):
            pairs = {str(row[i]).strip().lower(): row[i + 1] for i in range(0, len(row) - 1, 2)}
            t = safe_float(pairs.get("ts"))
            value = safe_float(pairs.get("pressure"))
            if np.isfinite(t) and np.isfinite(value):
                times.append(t)
                values.append(value)
        return np.asarray(times), np.asarray(values), "relative_seconds", "unresolved_pressure_unit"
    for index, line in enumerate(text.splitlines()):
        token = re.split(r"[,\s]+", line.strip())[0] if line.strip() else ""
        value = safe_float(token)
        if np.isfinite(value):
            times.append(float(index))
            values.append(value)
    return np.asarray(times), np.asarray(values), "sample_index", "raw_sensor_count_unresolved"


def resample_profile(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    t, y = times[order], values[order]
    unique, index = np.unique(t, return_index=True)
    t, y = unique, y[index]
    if len(y) < 2:
        raw = np.repeat(y[0] if len(y) else 0.0, PROFILE_POINTS)
    else:
        normalized = (t - t[0]) / max(t[-1] - t[0], 1e-12)
        raw = np.interp(np.linspace(0, 1, PROFILE_POINTS), normalized, y)
    baseline = float(np.median(raw[: max(3, PROFILE_POINTS // 10)]))
    centered = raw - baseline
    scale = float(np.quantile(centered, .75) - np.quantile(centered, .25))
    if scale < 1e-9:
        scale = float(np.std(centered))
    shape = centered / scale if scale >= 1e-9 else centered
    return raw.astype(float), shape.astype(float)


def engineered_features(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    order = np.argsort(times, kind="stable")
    t, y = times[order], values[order]
    duration = max(float(t[-1] - t[0]), 1.0)
    tn = (t - t[0]) / duration
    dy = np.diff(y)
    dt = np.diff(t)
    derivative = np.divide(dy, dt, out=np.zeros_like(dy), where=np.abs(dt) > 1e-12)
    window = max(2, len(y) // 10)
    peak_index = int(np.argmax(np.abs(y - np.median(y[:window]))))
    sign_changes = np.sum(np.diff(np.sign(derivative)) != 0) if len(derivative) > 1 else 0
    return {
        "n_points": float(len(y)),
        "duration": duration,
        "level_start": float(np.median(y[:window])),
        "level_end": float(np.median(y[-window:])),
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "std": float(np.std(y)),
        "minimum": float(np.min(y)),
        "maximum": float(np.max(y)),
        "q05": float(np.quantile(y, .05)),
        "q95": float(np.quantile(y, .95)),
        "amplitude": float(np.max(y) - np.min(y)),
        "area_signed": float(np.trapezoid(y, tn)),
        "area_absolute": float(np.trapezoid(np.abs(y), tn)),
        "peak_abs_value": float(y[peak_index]),
        "time_to_peak_fraction": float(tn[peak_index]),
        "initial_slope": float(np.polyfit(tn[:window], y[:window], 1)[0]) if window >= 2 else np.nan,
        "final_slope": float(np.polyfit(tn[-window:], y[-window:], 1)[0]) if window >= 2 else np.nan,
        "largest_drop": float(np.min(dy)) if len(dy) else 0.0,
        "largest_rise": float(np.max(dy)) if len(dy) else 0.0,
        "recovery_error": float(abs(np.median(y[-window:]) - np.median(y[:window]))),
        "derivative_noise": float(np.std(derivative)) if len(derivative) else 0.0,
        "second_difference_noise": float(np.std(np.diff(y, n=2))) if len(y) > 2 else 0.0,
        "derivative_sign_changes": float(sign_changes),
    }


def parse_spatial(data: bytes, match: re.Match[str], base: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for line in data.decode("utf-8-sig", "replace").splitlines():
        fields = line.strip().split()
        if len(fields) < 3:
            continue
        flow = safe_float(fields[0])
        value = safe_float(fields[2])
        nucleotide = str(fields[1]).upper()
        if np.isfinite(flow) and np.isfinite(value) and nucleotide in {"A", "C", "G", "T"}:
            rows.append({
                **base,
                "block_x": int(match.group("x")),
                "block_y": int(match.group("y")),
                "location": match.group("location").lower(),
                "flow_index": int(flow),
                "nucleotide": nucleotide,
                "step_height": value,
                "unit": "signal_unit_unresolved",
                "source_file": source,
            })
    return rows


def parse_timeline(data: bytes, name: str, base: dict[str, Any]) -> list[dict[str, Any]]:
    base_name = Path(name).name.lower()
    text = data.decode("utf-8-sig", "replace")
    rows: list[dict[str, Any]] = []
    if base_name == "scriptstatus.csv":
        frame = pd.read_csv(io.StringIO(text), skipinitialspace=True)
        frame.columns = [normalize_name(x) for x in frame.columns]
        for sequence, record in frame.iterrows():
            raw_time = str(record.get("time", "")).strip()
            timestamp = pd.to_datetime(raw_time, format="%Y_%m_%d-%H:%M:%S", utc=True, errors="coerce")
            module = normalize_name(record.get("module", ""))
            submodule = normalize_name(record.get("submodule", ""))
            status = normalize_name(record.get("status", ""))
            if not module and not submodule:
                continue
            rows.append({
                **base, "event_sequence": int(sequence), "event_time": timestamp.isoformat() if pd.notna(timestamp) else "",
                "event_type": "workflow_state", "module": module, "submodule": submodule,
                "status": status, "source_file": "CSA/ScriptStatus.csv",
                "parse_confidence": "high",
            })
    elif base_name == "vacuum_log.csv":
        for sequence, record in enumerate(csv.reader(io.StringIO(text))):
            pairs = {str(record[i]).strip(): record[i + 1] for i in range(0, len(record) - 1, 2)}
            timestamp = pd.to_datetime(pairs.get("TS", ""), format="%Y_%m_%d-%H:%M:%S", utc=True, errors="coerce")
            comment = normalize_name(pairs.get("Comment", ""))
            if not comment:
                continue
            rows.append({
                **base, "event_sequence": sequence, "event_time": timestamp.isoformat() if pd.notna(timestamp) else "",
                "event_type": "vacuum_summary", "module": comment.split("_", 1)[0],
                "submodule": comment, "status": "measured",
                "target_pressure": safe_float(pairs.get("TargetPressure")),
                "mean_pressure": safe_float(pairs.get("MeanPressure")),
                "time_to_target": safe_float(pairs.get("TimeToTarget")),
                "pressure_oscillation": safe_float(pairs.get("AmplitudeOfOscillation")),
                "source_file": "CSA/vacuum_log.csv", "parse_confidence": "high",
            })
    return rows


def extract_pilot(
    pilot: pd.DataFrame,
    paths: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    waveform_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    warnings_out: list[str] = []
    for count, record in enumerate(pilot.to_dict("records"), 1):
        archive_id = str(record["archive_id"])
        base = {
            "observation_id": record["observation_id"],
            "physical_run_id": record["physical_run_id"],
            "instrument_id": record["instrument_id"],
            "archive_id": archive_id,
            "chip_lane": record["chip_lane"],
            "run_start_time": record["run_start_time"],
            "workflow_context": record["workflow_context"],
        }
        path = Path(paths[archive_id])
        pressure_count = spatial_count = 0
        short_pressure_count = 0
        try:
            for member_order, name, size, member_time, data in archive_members(path):
                clean_name = name.replace("\\", "/")
                spatial_match = SPATIAL_RE.match(clean_name)
                if spatial_match:
                    extracted = parse_spatial(data, spatial_match, base, clean_name)
                    spatial_rows.extend(extracted)
                    spatial_count += len(extracted)
                    continue
                if PRESSURE_RE.search(clean_name):
                    times, values, time_unit, pressure_unit = parse_pressure(data, clean_name)
                    valid = np.isfinite(times) & np.isfinite(values)
                    times, values = times[valid], values[valid]
                    if len(values) < 20:
                        short_pressure_count += 1
                        continue
                    meta = pressure_metadata(clean_name)
                    trace_id = stable_id(archive_id, clean_name, prefix="trace_")
                    raw_profile, shape_profile = resample_profile(times, values)
                    features = engineered_features(times, values)
                    trace_rows.append({
                        **base, **meta, **features,
                        "trace_id": trace_id,
                        "source_file": "CSA/" + clean_name.split("CSA/", 1)[-1],
                        "source_size_bytes": size,
                        "archive_member_order": member_order,
                        "archive_member_mtime": member_time,
                        "time_unit": time_unit,
                        "pressure_unit": pressure_unit,
                        "operation_timestamp_available": False,
                        "temporal_order_confidence": "unresolved_export_timestamp_only",
                        **{f"profile_raw_{i:02d}": value for i, value in enumerate(raw_profile)},
                        **{f"profile_shape_{i:02d}": value for i, value in enumerate(shape_profile)},
                    })
                    t0, span = float(times.min()), max(float(times.max() - times.min()), 1e-12)
                    waveform_rows.extend({
                        **base, **meta,
                        "trace_id": trace_id, "point_index": point,
                        "time_value": float(t), "time_normalized": float((t - t0) / span),
                        "pressure_value": float(value), "time_unit": time_unit,
                        "pressure_unit": pressure_unit,
                        "source_file": "CSA/" + clean_name.split("CSA/", 1)[-1],
                    } for point, (t, value) in enumerate(zip(times, values)))
                    pressure_count += 1
                    continue
                if Path(clean_name).name.lower() in TIMELINE_BASES:
                    timeline_rows.extend(parse_timeline(data, clean_name, base))
            if pressure_count == 0:
                warnings_out.append(f"{archive_id}: no usable pressure traces")
            if spatial_count == 0:
                warnings_out.append(f"{archive_id}: no usable spatial rows")
            if short_pressure_count:
                warnings_out.append(
                    f"{archive_id}: excluded {short_pressure_count} pressure files with fewer than 20 points"
                )
        except Exception as exc:
            warnings_out.append(f"{archive_id}/{path.name}: {type(exc).__name__}: {exc}")
        print(f"[{count:02d}/{len(pilot)}] {archive_id}: pressure={pressure_count}, spatial_rows={spatial_count}", flush=True)
    return (
        pd.DataFrame(waveform_rows),
        pd.DataFrame(trace_rows),
        pd.DataFrame(spatial_rows),
        pd.DataFrame(timeline_rows),
        warnings_out,
    )


def empirical_percentile(train_scores: np.ndarray, value: float) -> float:
    clean = train_scores[np.isfinite(train_scores)]
    if not len(clean):
        return np.nan
    tolerance = max(1e-12, np.nanmax(np.abs(clean)) * 1e-10)
    less = np.sum(clean < value - tolerance)
    equal = np.sum(np.abs(clean - value) <= tolerance)
    return float((0.5 + less + 0.5 * equal) / (len(clean) + 1))


def fit_anomaly_reference(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    profile_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median().fillna(0)
    x_train = x_train.fillna(medians).to_numpy(float)
    x_test = test[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(float)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    total_variance = float(np.var(train_scaled, axis=0).sum())
    if total_variance < 1e-12:
        component_count = 0
        train_pc = np.zeros((len(train), 1))
        test_pc = np.zeros((len(test), 1))
        train_reconstructed = train_scaled.copy()
        test_reconstructed = test_scaled.copy()
        explained_variance = 1.0
    else:
        component_count = max(1, min(8, len(train) - 1, train_scaled.shape[1]))
        pca = PCA(n_components=component_count, random_state=SEED)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.decomposition")
            train_pc = pca.fit_transform(train_scaled)
        test_pc = pca.transform(test_scaled)
        train_reconstructed = pca.inverse_transform(train_pc)
        test_reconstructed = pca.inverse_transform(test_pc)
        explained_variance = float(np.nansum(pca.explained_variance_ratio_))
    forest = IsolationForest(
        n_estimators=200, max_samples=min(256, len(train)), contamination="auto",
        random_state=SEED, n_jobs=1,
    )
    forest.fit(train_pc)
    train_iso = -forest.decision_function(train_pc)
    test_iso = -forest.decision_function(test_pc)
    train_re = np.mean((train_scaled - train_reconstructed) ** 2, axis=1)
    test_re = np.mean((test_scaled - test_reconstructed) ** 2, axis=1)
    # A held-out trace is calibrated against leave-self-out neighbor distances
    # from the same reference representation.  Unlike in-sample IF or PCA
    # residuals, this does not make every new record anomalous by construction.
    k_neighbors = max(1, min(3, len(train) - 1))
    train_distances = pairwise_distances(train_pc)
    np.fill_diagonal(train_distances, np.inf)
    train_knn = np.mean(np.partition(train_distances, k_neighbors - 1, axis=1)[:, :k_neighbors], axis=1)
    test_distances = pairwise_distances(test_pc, train_pc)
    test_knn = np.mean(np.partition(test_distances, k_neighbors - 1, axis=1)[:, :k_neighbors], axis=1)
    rows = []
    for index in range(len(test)):
        iso_pct = empirical_percentile(train_iso, float(test_iso[index]))
        re_pct = empirical_percentile(train_re, float(test_re[index]))
        knn_pct = empirical_percentile(train_knn, float(test_knn[index]))
        contribution = np.abs(test_scaled[index])
        top = np.argsort(contribution)[::-1][:3]
        rows.append({
            "isolation_forest_score": float(test_iso[index]),
            "isolation_forest_percentile": iso_pct,
            "pca_reconstruction_error": float(test_re[index]),
            "pca_reconstruction_percentile": re_pct,
            "knn_distance": float(test_knn[index]),
            "knn_conformal_percentile": knn_pct,
            "anomaly_percentile": knn_pct,
            "calibration_method": "tie_aware_leave_self_out_knn_in_training_pca_space",
            "top_deviation_features": "|".join(feature_columns[i] for i in top),
        })
    reference: dict[str, Any] = {
        "reference_n": len(train),
        "reference_physical_runs": train["physical_run_id"].nunique(),
        "pca_components": component_count,
        "pca_explained_variance": explained_variance,
    }
    if profile_columns:
        matrix = train[profile_columns].to_numpy(float)
        reference["profile_median"] = np.median(matrix, axis=0).tolist()
        reference["profile_q05"] = np.quantile(matrix, .05, axis=0).tolist()
        reference["profile_q95"] = np.quantile(matrix, .95, axis=0).tolist()
    return pd.DataFrame(rows, index=test.index), reference


def waveform_anomalies(traces: pd.DataFrame) -> pd.DataFrame:
    profile = [f"profile_shape_{i:02d}" for i in range(PROFILE_POINTS)]
    features = list(ENGINEERED) + profile
    clean = traces.copy()
    for column in ("instrument_id", "trace_family", "pipette", "attempt_bucket"):
        clean[column] = clean[column].fillna("unresolved").astype(str)
    runs = clean["physical_run_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(5, clean["physical_run_id"].nunique()))
    result_rows = []
    levels = [
        ("instrument_family_pipette_attempt", ["instrument_id", "trace_family", "pipette", "attempt_bucket"]),
        ("instrument_family_pipette", ["instrument_id", "trace_family", "pipette"]),
        ("instrument_family", ["instrument_id", "trace_family"]),
        ("fleet_family_pipette", ["trace_family", "pipette"]),
        ("fleet_family", ["trace_family"]),
    ]
    for fold, (train_index, test_index) in enumerate(splitter.split(clean, groups=runs), 1):
        train, test = clean.iloc[train_index], clean.iloc[test_index]
        assigned: dict[tuple[str, tuple[Any, ...]], list[int]] = {}
        unscored = []
        for index, row in test.iterrows():
            found = False
            for level_name, columns in levels:
                key = tuple(row[column] for column in columns)
                mask = np.ones(len(train), dtype=bool)
                for column, value in zip(columns, key):
                    mask &= train[column].to_numpy() == value
                reference = train.loc[mask]
                if len(reference) >= MIN_REFERENCE_TRACES and reference["physical_run_id"].nunique() >= MIN_REFERENCE_RUNS:
                    assigned.setdefault((level_name + "|" + ",".join(columns), key), []).append(index)
                    found = True
                    break
            if not found:
                unscored.append(index)
        for (level_spec, key), indices in assigned.items():
            level_name, columns_text = level_spec.split("|", 1)
            columns = columns_text.split(",")
            mask = np.ones(len(train), dtype=bool)
            for column, value in zip(columns, key):
                mask &= train[column].to_numpy() == value
            reference_frame = train.loc[mask]
            test_frame = test.loc[indices]
            scores, reference = fit_anomaly_reference(reference_frame, test_frame, features, profile)
            for index, score in scores.iterrows():
                row = clean.loc[index]
                result_rows.append({
                    "trace_id": row["trace_id"], "observation_id": row["observation_id"],
                    "physical_run_id": row["physical_run_id"], "instrument_id": row["instrument_id"],
                    "chip_lane": row["chip_lane"], "trace_family": row["trace_family"],
                    "operation": row["operation"], "pipette": row["pipette"],
                    "attempt": row["attempt"], "attempt_bucket": row["attempt_bucket"],
                    "source_file": row["source_file"], "validation_fold": fold,
                    "reference_level": level_name, **reference, **score.to_dict(),
                    "operation_timestamp_available": False,
                    "temporal_interpretation": "ranked anomaly only; operation order unresolved",
                    "observed_profile": json.dumps([float(row[x]) for x in profile]),
                    "reference_profile_median": json.dumps(reference["profile_median"]),
                    "reference_profile_q05": json.dumps(reference["profile_q05"]),
                    "reference_profile_q95": json.dumps(reference["profile_q95"]),
                })
        for index in unscored:
            row = clean.loc[index]
            result_rows.append({
                "trace_id": row["trace_id"], "observation_id": row["observation_id"],
                "physical_run_id": row["physical_run_id"], "instrument_id": row["instrument_id"],
                "chip_lane": row["chip_lane"], "trace_family": row["trace_family"],
                "operation": row["operation"], "pipette": row["pipette"],
                "attempt": row["attempt"], "attempt_bucket": row["attempt_bucket"],
                "source_file": row["source_file"], "validation_fold": fold,
                "reference_level": "unscored_insufficient_reference",
                "reference_n": 0, "reference_physical_runs": 0,
                "anomaly_percentile": np.nan,
                "temporal_interpretation": "ranked anomaly unavailable; operation order unresolved",
            })
    return pd.DataFrame(result_rows)


def neighbor_roughness(frame: pd.DataFrame) -> float:
    grid = frame.groupby(["block_x", "block_y"])["block_value"].median().to_dict()
    xs, ys = sorted(frame["block_x"].unique()), sorted(frame["block_y"].unique())
    diffs = []
    for (x, y), value in grid.items():
        xi, yi = xs.index(x), ys.index(y)
        for neighbor in ((xs[xi + 1], y) if xi + 1 < len(xs) else None,
                         (x, ys[yi + 1]) if yi + 1 < len(ys) else None):
            if neighbor in grid:
                diffs.append(abs(value - grid[neighbor]))
    return float(np.median(diffs)) if diffs else np.nan


def spatial_features(spatial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = (
        spatial.groupby(
            ["observation_id", "physical_run_id", "instrument_id", "chip_lane",
             "workflow_context", "block_x", "block_y", "location", "nucleotide"],
            as_index=False,
        )["step_height"].agg(block_value="median", flow_iqr=lambda x: x.quantile(.75) - x.quantile(.25), flow_count="size")
    )
    blocks["block_x_index"] = -1
    blocks["block_y_index"] = -1
    for _, indices in blocks.groupby("observation_id").groups.items():
        index_list = list(indices)
        x_values = sorted(blocks.loc[index_list, "block_x"].unique())
        y_values = sorted(blocks.loc[index_list, "block_y"].unique())
        blocks.loc[index_list, "block_x_index"] = blocks.loc[index_list, "block_x"].map(
            {value: rank for rank, value in enumerate(x_values)}
        )
        blocks.loc[index_list, "block_y_index"] = blocks.loc[index_list, "block_y"].map(
            {value: rank for rank, value in enumerate(y_values)}
        )
    rows = []
    for keys, group in blocks.groupby(
        ["observation_id", "physical_run_id", "instrument_id", "chip_lane", "workflow_context"],
        sort=False,
    ):
        row = dict(zip(["observation_id", "physical_run_id", "instrument_id", "chip_lane", "workflow_context"], keys))
        for nucleotide, nuc in group.groupby("nucleotide"):
            location_means = nuc.groupby("location")["block_value"].mean()
            for location in ("inlet", "middle", "outlet"):
                part = nuc[nuc["location"] == location]
                prefix = f"{nucleotide}_{location}"
                row[f"{prefix}_mean"] = part["block_value"].mean()
                row[f"{prefix}_block_std"] = part["block_value"].std()
                row[f"{prefix}_roughness"] = neighbor_roughness(part)
                if len(part) >= 3 and part["block_x"].nunique() >= 2:
                    row[f"{prefix}_x_slope"] = stats.linregress(part["block_x"], part["block_value"]).slope
                if len(part) >= 3 and part["block_y"].nunique() >= 2:
                    row[f"{prefix}_y_slope"] = stats.linregress(part["block_y"], part["block_value"]).slope
            row[f"{nucleotide}_inlet_outlet_gradient"] = location_means.get("outlet", np.nan) - location_means.get("inlet", np.nan)
            row[f"{nucleotide}_inlet_middle_gradient"] = location_means.get("middle", np.nan) - location_means.get("inlet", np.nan)
        rows.append(row)
    return pd.DataFrame(rows), blocks


def spatial_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    ids = {"observation_id", "physical_run_id", "instrument_id", "chip_lane", "workflow_context"}
    numeric = [x for x in features.columns if x not in ids]
    runs = features["physical_run_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(5, features["physical_run_id"].nunique()))
    rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(features, groups=runs), 1):
        train, test = features.iloc[train_index], features.iloc[test_index]
        for index, record in test.iterrows():
            exact = train[
                (train["instrument_id"] == record["instrument_id"])
                & (train["workflow_context"] == record["workflow_context"])
            ]
            if len(exact) >= MIN_REFERENCE_TRACES:
                reference, level = exact, "instrument_workflow"
            else:
                instrument = train[train["instrument_id"] == record["instrument_id"]]
                if len(instrument) >= MIN_REFERENCE_TRACES:
                    reference, level = instrument, "instrument"
                else:
                    reference, level = train, "fleet"
            score, ref = fit_anomaly_reference(reference, test.loc[[index]], numeric)
            rows.append({
                "observation_id": record["observation_id"],
                "physical_run_id": record["physical_run_id"],
                "instrument_id": record["instrument_id"],
                "chip_lane": record["chip_lane"],
                "workflow_context": record["workflow_context"],
                "validation_fold": fold, "reference_level": level,
                **ref, **score.iloc[0].to_dict(),
            })
    return pd.DataFrame(rows)


def spatial_block_anomalies(blocks: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    fold_map = {}
    groups = features["physical_run_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(5, features["physical_run_id"].nunique()))
    for fold, (_, test_index) in enumerate(splitter.split(features, groups=groups), 1):
        fold_map.update({features.iloc[i]["physical_run_id"]: fold for i in test_index})
    blocks = blocks.copy()
    blocks["validation_fold"] = blocks["physical_run_id"].map(fold_map)
    rows = []
    keys = ["instrument_id", "block_x_index", "block_y_index", "location", "nucleotide"]
    for fold in sorted(blocks["validation_fold"].dropna().unique()):
        train, test = blocks[blocks["validation_fold"] != fold], blocks[blocks["validation_fold"] == fold]
        grouped = {key: group for key, group in train.groupby(keys, sort=False)}
        for _, record in test.iterrows():
            key = tuple(record[x] for x in keys)
            reference = grouped.get(key)
            if reference is None or reference["physical_run_id"].nunique() < MIN_REFERENCE_RUNS:
                continue
            values = reference["block_value"].to_numpy(float)
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            scale = 1.4826 * mad
            if scale < 1e-9:
                scale = max(float(np.std(values)), 1e-9)
            z = (float(record["block_value"]) - center) / scale
            absolute_train = np.abs((values - center) / scale)
            rows.append({
                "observation_id": record["observation_id"], "physical_run_id": record["physical_run_id"],
                "instrument_id": record["instrument_id"], "chip_lane": record["chip_lane"],
                "block_x": record["block_x"], "block_y": record["block_y"],
                "block_x_index": record["block_x_index"], "block_y_index": record["block_y_index"],
                "location": record["location"], "nucleotide": record["nucleotide"],
                "block_value": record["block_value"], "robust_z": z,
                "anomaly_percentile": empirical_percentile(absolute_train, abs(z)),
                "reference_n": len(reference), "validation_fold": fold,
            })
    return pd.DataFrame(rows)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    return core.benjamini_hochberg(values)


def fuse_evidence(
    pilot: pd.DataFrame,
    wave_scores: pd.DataFrame,
    spatial_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_wave = wave_scores.dropna(subset=["anomaly_percentile"]).copy()
    summary = scored_wave.groupby("observation_id").agg(
        waveform_trace_count=("trace_id", "size"),
        waveform_anomaly_p95=("anomaly_percentile", lambda x: x.quantile(.95)),
        waveform_anomaly_max=("anomaly_percentile", "max"),
        waveform_fraction_ge_95=("anomaly_percentile", lambda x: (x >= .95).mean()),
    ).reset_index()
    summary["waveform_extreme_trace_count"] = (
        summary["waveform_trace_count"] * summary["waveform_fraction_ge_95"]
    ).round().astype(int)
    summary["waveform_run_anomaly_percentile"] = [
        (
            0.0 if extreme == 0
            else float(1.0 - stats.binom.sf(extreme - 1, int(total), .05))
        )
        for extreme, total in zip(
            summary["waveform_extreme_trace_count"], summary["waveform_trace_count"]
        )
    ]
    top_indices = scored_wave.groupby("observation_id")["anomaly_percentile"].idxmax()
    top = scored_wave.loc[top_indices, [
        "observation_id", "trace_id", "trace_family", "operation", "pipette",
        "attempt", "anomaly_percentile", "top_deviation_features", "source_file",
    ]].rename(columns={x: "top_" + x for x in [
        "trace_id", "trace_family", "operation", "pipette", "attempt",
        "anomaly_percentile", "top_deviation_features", "source_file",
    ]})
    result = pilot.merge(summary, on="observation_id", how="left").merge(top, on="observation_id", how="left")
    result = result.merge(
        spatial_scores[[
            "observation_id", "anomaly_percentile", "top_deviation_features",
            "reference_level", "reference_n",
        ]].rename(columns={
            "anomaly_percentile": "spatial_anomaly_percentile",
            "top_deviation_features": "spatial_top_deviation_features",
            "reference_level": "spatial_reference_level",
            "reference_n": "spatial_reference_n",
        }),
        on="observation_id", how="left",
    )
    result["agreement_score"] = result[
        ["waveform_run_anomaly_percentile", "spatial_anomaly_percentile"]
    ].min(axis=1)
    result["independent_evidence_agreement_90"] = (
        (result["waveform_run_anomaly_percentile"] >= .90)
        & (result["spatial_anomaly_percentile"] >= .90)
    )
    result["evidence_label"] = np.select(
        [
            (result["waveform_run_anomaly_percentile"] >= .95)
            & (result["spatial_anomaly_percentile"] >= .95),
            result["independent_evidence_agreement_90"],
            result["waveform_run_anomaly_percentile"] >= .95,
            result["spatial_anomaly_percentile"] >= .95,
        ],
        ["joint_multimodal_review_priority_ge95", "joint_multimodal_review_priority_ge90",
         "waveform_only_review_priority", "spatial_only_review_priority"],
        default="no_joint_extreme",
    )
    result["interpretation"] = (
        "Review priority only; pattern consistency does not establish a fluidic root cause."
    )
    result = result.sort_values(
        ["independent_evidence_agreement_90", "agreement_score"],
        ascending=[False, False], kind="stable",
    ).reset_index(drop=True)
    result["multimodal_rank"] = np.arange(1, len(result) + 1)

    association_rows = []
    for score in ("waveform_run_anomaly_percentile", "spatial_anomaly_percentile", "agreement_score"):
        for outcome in ("total_reads", "mean_read_length_bp"):
            part = result[[score, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(part) >= 10 and part[score].nunique() >= 3 and part[outcome].nunique() >= 3:
                test = stats.spearmanr(part[score], part[outcome])
                association_rows.append({
                    "anomaly_score": score, "outcome": outcome, "usable_n": len(part),
                    "spearman_rho": float(test.statistic), "p_value": float(test.pvalue),
                    "fdr_q_value": np.nan,
                    "interpretation": "exploratory enrichment only; outcomes were not used to fit anomaly models",
                })
    associations = pd.DataFrame(association_rows)
    if not associations.empty:
        associations["fdr_q_value"] = benjamini_hochberg(associations["p_value"])
    return result, associations


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def html_table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "<p class='muted'>No rows available.</p>"
    shown = frame[[x for x in columns if x in frame.columns]].head(limit).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(fmt)
    return shown.to_html(index=False, escape=True, border=0)


def svg_trace(row: pd.Series) -> str:
    try:
        observed = np.asarray(json.loads(row["observed_profile"]), float)
        median = np.asarray(json.loads(row["reference_profile_median"]), float)
        low = np.asarray(json.loads(row["reference_profile_q05"]), float)
        high = np.asarray(json.loads(row["reference_profile_q95"]), float)
    except Exception:
        return ""
    values = np.concatenate([observed, low, high])
    y0, y1 = float(np.min(values)), float(np.max(values))
    span = max(y1 - y0, 1e-9)
    width, height, left, top = 480, 230, 42, 34
    def points(arr: np.ndarray) -> str:
        return " ".join(
            f"{left + i / (len(arr)-1) * (width-2*left):.1f},{height-left-(v-y0)/span*(height-left-top):.1f}"
            for i, v in enumerate(arr)
        )
    polygon = points(low) + " " + " ".join(reversed(points(high).split(" ")))
    title = f"{row['trace_family']} · {row['pipette']} · anomaly {row['anomaly_percentile']:.2f}"
    return (
        f"<svg viewBox='0 0 {width} {height}'><text x='{left}' y='20' class='svg-title'>{html.escape(title)}</text>"
        f"<polygon points='{polygon}' fill='{core.PALETTE['mist']}' opacity='.8'/>"
        f"<polyline points='{points(median)}' fill='none' stroke='{core.PALETTE['forest']}' stroke-width='2'/>"
        f"<polyline points='{points(observed)}' fill='none' stroke='{core.PALETTE['clay']}' stroke-width='2.4'/>"
        f"<line x1='{left}' y1='{height-left}' x2='{width-left}' y2='{height-left}' stroke='{core.PALETTE['ink']}'/>"
        f"<text x='{width/2}' y='{height-10}' text-anchor='middle'>normalized trace time</text></svg>"
    )


def svg_heatmap(block_scores: pd.DataFrame, observation_id: str, title: str) -> str:
    part = block_scores[block_scores["observation_id"] == observation_id]
    if part.empty:
        return ""
    part = part.groupby(["block_x", "block_y"], as_index=False)["anomaly_percentile"].max()
    xs, ys = sorted(part["block_x"].unique()), sorted(part["block_y"].unique())
    width, height, margin = 430, 300, 50
    cell_w = (width - 2 * margin) / max(len(xs), 1)
    cell_h = (height - 2 * margin) / max(len(ys), 1)
    rects = []
    for row in part.itertuples():
        ix, iy = xs.index(row.block_x), ys.index(row.block_y)
        p = float(row.anomaly_percentile)
        red = int(230 - 60 * (1-p)); green = int(240 - 150 * p); blue = int(210 - 150 * p)
        rects.append(
            f"<rect x='{margin+ix*cell_w:.1f}' y='{height-margin-(iy+1)*cell_h:.1f}' "
            f"width='{cell_w-1:.1f}' height='{cell_h-1:.1f}' fill='rgb({red},{green},{blue})'>"
            f"<title>X{row.block_x} Y{row.block_y}: {p:.3f}</title></rect>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}'><text x='{margin}' y='22' class='svg-title'>{html.escape(title)}</text>"
        + "".join(rects)
        + f"<text x='{width/2}' y='{height-8}' text-anchor='middle'>block X</text>"
        + f"<text transform='translate(14 {height/2}) rotate(-90)' text-anchor='middle'>block Y</text></svg>"
    )


def make_report(
    pilot: pd.DataFrame,
    waveforms: pd.DataFrame,
    traces: pd.DataFrame,
    spatial: pd.DataFrame,
    timeline: pd.DataFrame,
    wave_scores: pd.DataFrame,
    spatial_scores: pd.DataFrame,
    block_scores: pd.DataFrame,
    rankings: pd.DataFrame,
    associations: pd.DataFrame,
    warnings_out: list[str],
) -> str:
    scored = wave_scores.dropna(subset=["anomaly_percentile"]).sort_values("anomaly_percentile", ascending=False)
    trace_plots = "".join(svg_trace(row) for _, row in scored.head(8).iterrows())
    heatmaps = "".join(
        svg_heatmap(block_scores, row["observation_id"], f"Spatial held-out anomaly map · rank {row['multimodal_rank']}")
        for _, row in rankings.head(6).iterrows()
    )
    joint = int(rankings["independent_evidence_agreement_90"].sum())
    operations = wave_scores[wave_scores["anomaly_percentile"] >= .95].merge(
        rankings[["observation_id", "multimodal_rank", "evidence_label"]],
        on="observation_id", how="left",
    ).sort_values(["multimodal_rank", "anomaly_percentile"], ascending=[True, False])
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>CSA multimodal workflow fingerprints</title><style>
:root{{--paper:{core.PALETTE['paper']};--ink:{core.PALETTE['ink']};--forest:{core.PALETTE['forest']};--ocean:{core.PALETTE['ocean']};--amber:{core.PALETTE['amber']};--mist:{core.PALETTE['mist']};}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:34px}}
h1,h2{{color:var(--forest)}}h1{{border-bottom:5px solid var(--ocean);padding-bottom:10px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:white;border-left:5px solid var(--forest);padding:14px;border-radius:5px}}.big{{font-size:26px;font-weight:700;color:var(--ocean)}}
.note{{background:#fff6dc;border-left:5px solid var(--amber);padding:12px}}table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}
th,td{{padding:6px;border-bottom:1px solid var(--mist);text-align:left}}th{{background:var(--mist)}}svg{{background:white;margin:8px;max-width:490px}}.svg-title{{font-weight:700;fill:var(--ink)}}.muted{{color:#667}}</style></head>
<body><main><h1>CSA multimodal workflow-fingerprint pilot</h1>
<p>Generated {html.escape(utc_now())}. Unsupervised anomaly models were evaluated on held-out physical runs. Outcomes were used only for exploratory enrichment after scoring.</p>
<div class='cards'><div class='card'><div class='big'>{pilot['physical_run_id'].nunique()}</div>independent physical runs</div>
<div class='card'><div class='big'>{pilot['instrument_id'].nunique()}</div>instruments</div>
<div class='card'><div class='big'>{traces['trace_id'].nunique()}</div>pressure traces</div>
<div class='card'><div class='big'>{len(waveforms):,}</div>ordered pressure points</div>
<div class='card'><div class='big'>{len(spatial):,}</div>spatial nucleotide-step rows</div>
<div class='card'><div class='big'>{joint}</div>runs with ≥0.90 agreement</div></div>
<h2>Interpretation boundary</h2><div class='note'>Scores prioritize records for expert review. They do not prove a root cause, identify a tunable setting, or authorize an instrument change. Pipette trace files lack operational timestamps; they can be ranked by unusual shape but cannot be placed honestly as the first workflow deviation.</div>
<h2>Highest multimodal review priorities</h2>
{html_table(rankings, ['multimodal_rank','physical_run_id','instrument_id','chip_lane','waveform_trace_count','waveform_extreme_trace_count','waveform_run_anomaly_percentile','spatial_anomaly_percentile','agreement_score','evidence_label','top_trace_family','top_pipette','top_attempt','top_deviation_features','spatial_top_deviation_features'], 30)}
<h2>Ranked unusual operations</h2>
{html_table(operations, ['multimodal_rank','physical_run_id','trace_id','trace_family','operation','pipette','attempt','anomaly_percentile','top_deviation_features','source_file','reference_physical_runs','evidence_label'], 40)}
<h2>Unusual waveform overlays</h2><p>Orange is the held-out trace; green is the condition-matched reference median; the shaded band is the reference 5th–95th percentile.</p><div>{trace_plots}</div>
<h2>Spatial block evidence</h2><p>Color represents the maximum held-out anomaly percentile across inlet/middle/outlet and nucleotide within each block.</p><div>{heatmaps}</div>
<h2>Exploratory outcome enrichment</h2>
{html_table(associations, ['anomaly_score','outcome','usable_n','spearman_rho','p_value','fdr_q_value','interpretation'])}
<h2>Coverage and model safeguards</h2><ul>
<li>Exactly one retained lane archive was selected from each physical run; selection was date-stratified within instrument.</li>
<li>Five-fold validation separated complete physical runs, so traces from a held-out run never entered its reference model.</li>
<li>Reference matching preferred instrument × trace family × pipette × attempt, then used documented fallbacks when support was insufficient.</li>
<li>Pressure units remain unresolved where the CSA did not document a conversion.</li>
<li>Spatial topology came directly from block coordinates and inlet/middle/outlet step files, not rendered images.</li>
<li>Structured ScriptStatus and vacuum summaries produced {len(timeline):,} timeline events. Events.log and wflog were excluded because free text can contain credential/network messages.</li></ul>
<h2>Limitations</h2><ul><li>No adjudicated failure labels were available.</li><li>Rare workflow/trace conditions may fall back to fleet-level references.</li>
<li>Sixty runs are appropriate for a pilot but not for a production alarm threshold.</li><li>Multiple unusual traces from one run are correlated and are not independent biological observations.</li>
<li>Block/location coverage differs by instrument; heatmaps show only cells with at least six held-out reference runs.</li>
<li>Extraction/model warnings: {len(warnings_out)}.</li></ul>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("genexus_csa_project"))
    parser.add_argument("--per-instrument", type=int, default=15)
    parser.add_argument("--reuse-extracted", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    started = time.time()
    warnings_out: list[str] = []
    pilot, paths = choose_pilot(project, args.per_instrument)
    pilot.to_csv(project / "fingerprint_pilot_archive_selection.csv", index=False)

    extraction_files = {
        "waveforms": project / "operation_waveforms.parquet",
        "traces": project / "operation_trace_features.parquet",
        "spatial": project / "spatial_diagnostics.parquet",
        "timeline": project / "process_timeline.csv",
    }
    if args.reuse_extracted and all(path.exists() for path in extraction_files.values()):
        waveforms = pd.read_parquet(extraction_files["waveforms"])
        traces = pd.read_parquet(extraction_files["traces"])
        spatial = pd.read_parquet(extraction_files["spatial"])
        timeline = pd.read_csv(extraction_files["timeline"], low_memory=False)
        previous_log_path = project / "workflow_fingerprint_run_log.json"
        if previous_log_path.exists():
            previous_warnings = json.loads(previous_log_path.read_text()).get("warnings", [])
            if any(
                "fewer than 20" in str(item) or "prior raw extraction excluded" in str(item)
                for item in previous_warnings
            ):
                warnings_out.append(
                    "short pressure files with fewer than 20 points were excluded during raw extraction"
                )
        missing_pressure = set(pilot["observation_id"]) - set(traces["observation_id"])
        missing_spatial = set(pilot["observation_id"]) - set(spatial["observation_id"])
        if missing_pressure:
            warnings_out.append(f"{len(missing_pressure)} pilot archives have no usable pressure traces")
        if missing_spatial:
            warnings_out.append(f"{len(missing_spatial)} pilot archives have no usable spatial rows")
    else:
        waveforms, traces, spatial, timeline, extraction_warnings = extract_pilot(pilot, paths)
        warnings_out.extend(extraction_warnings)
        waveforms.to_parquet(extraction_files["waveforms"], index=False)
        traces.to_parquet(extraction_files["traces"], index=False)
        spatial.to_parquet(extraction_files["spatial"], index=False)
        timeline.to_csv(extraction_files["timeline"], index=False)
    if traces.empty or spatial.empty:
        raise RuntimeError(f"extraction insufficient: traces={len(traces)}, spatial={len(spatial)}")

    public_trace_columns = [
        x for x in traces.columns
        if not x.startswith("profile_raw_") and not x.startswith("profile_shape_")
    ]
    traces[public_trace_columns].to_csv(project / "operation_trace_features.csv", index=False)
    wave_scores = waveform_anomalies(traces)
    wave_scores = wave_scores.sort_values(
        ["anomaly_percentile", "physical_run_id", "trace_id"],
        ascending=[False, True, True], kind="stable", na_position="last",
    ).reset_index(drop=True)
    wave_scores["trace_review_rank"] = np.arange(1, len(wave_scores) + 1)
    spatial_run, spatial_blocks = spatial_features(spatial)
    spatial_scores = spatial_anomalies(spatial_run)
    block_scores = spatial_block_anomalies(spatial_blocks, spatial_run)
    rankings, associations = fuse_evidence(pilot, wave_scores, spatial_scores)
    review_operations = wave_scores[wave_scores["anomaly_percentile"] >= .95].merge(
        rankings[["observation_id", "multimodal_rank", "evidence_label", "agreement_score"]],
        on="observation_id", how="left",
    ).sort_values(["multimodal_rank", "anomaly_percentile"], ascending=[True, False])
    review_operations = review_operations[[
        "multimodal_rank", "physical_run_id", "observation_id", "instrument_id",
        "chip_lane", "trace_review_rank", "trace_id", "trace_family", "operation",
        "pipette", "attempt", "attempt_bucket", "anomaly_percentile",
        "knn_distance", "top_deviation_features", "source_file", "reference_level",
        "reference_n", "reference_physical_runs", "evidence_label", "agreement_score",
        "temporal_interpretation",
    ]]
    review_blocks = block_scores[block_scores["anomaly_percentile"] >= .95].merge(
        rankings[["observation_id", "multimodal_rank", "evidence_label", "agreement_score"]],
        on="observation_id", how="left",
    ).sort_values(["multimodal_rank", "anomaly_percentile"], ascending=[True, False])

    wave_scores.to_csv(project / "waveform_anomaly_scores.csv", index=False)
    spatial_run.to_csv(project / "spatial_run_features.csv", index=False)
    spatial_scores.to_csv(project / "spatial_anomaly_scores.csv", index=False)
    block_scores.to_csv(project / "spatial_block_anomaly_scores.csv", index=False)
    review_operations.to_csv(project / "review_priority_operations.csv", index=False)
    review_blocks.to_csv(project / "review_priority_spatial_blocks.csv", index=False)
    rankings.to_csv(project / "multimodal_evidence_rankings.csv", index=False)
    associations.to_csv(project / "multimodal_outcome_associations.csv", index=False)
    report = make_report(
        pilot, waveforms, traces, spatial, timeline, wave_scores, spatial_scores,
        block_scores, rankings, associations, warnings_out,
    )
    (project / "workflow_fingerprint_report.html").write_text(report, encoding="utf-8")

    script_path = Path(__file__).resolve()
    project_script = project / "workflow_fingerprint_pilot.py"
    if script_path != project_script:
        shutil.copy2(script_path, project_script)
    input_files = [
        project / "duplicate_archives.csv", project / "csa_transfer_manifest.csv",
        project / "csa_ml_dataset.parquet",
    ]
    log = {
        "version": VERSION, "generated_at_utc": utc_now(), "random_seed": SEED,
        "elapsed_seconds": time.time() - started,
        "code_sha256": sha256_file(project_script),
        "input_hashes": {path.name: sha256_file(path) for path in input_files},
        "experimental_unit": "physical sequencing run",
        "pilot_physical_runs": int(pilot["physical_run_id"].nunique()),
        "instrument_coverage": int(pilot["instrument_id"].nunique()),
        "pressure_traces": int(traces["trace_id"].nunique()),
        "ordered_pressure_points": int(len(waveforms)),
        "spatial_rows": int(len(spatial)),
        "timeline_events": int(len(timeline)),
        "scored_waveforms": int(wave_scores["anomaly_percentile"].notna().sum()),
        "spatial_runs_scored": int(spatial_scores["anomaly_percentile"].notna().sum()),
        "multimodal_agreement_ge_90": int(rankings["independent_evidence_agreement_90"].sum()),
        "validation": "5-fold GroupKFold by physical_run_id; no trace from a held-out run enters its reference",
        "models": "condition-matched StandardScaler + PCA + IsolationForest diagnostics; tie-aware kNN conformal percentile is the calibrated anomaly score",
        "waveform_run_aggregation": "binomial excess rate of trace conformal percentiles >=0.95, adjusted for traces per run",
        "outcome_use": "post-hoc exploratory enrichment only; no outcome-trained anomaly model",
        "excluded_free_text": ["Events.log", "wflog"],
        "temporal_limit": "pipette trace operational timestamps unavailable; rank unusual operations but do not claim first temporal deviation",
        "warnings": warnings_out,
    }
    (project / "workflow_fingerprint_run_log.json").write_text(
        json.dumps(log, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pilot_runs": log["pilot_physical_runs"], "pressure_traces": log["pressure_traces"],
        "pressure_points": log["ordered_pressure_points"], "spatial_rows": log["spatial_rows"],
        "scored_waveforms": log["scored_waveforms"],
        "multimodal_agreement_ge_90": log["multimodal_agreement_ge_90"],
        "warnings": len(warnings_out), "elapsed_seconds": round(log["elapsed_seconds"], 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
