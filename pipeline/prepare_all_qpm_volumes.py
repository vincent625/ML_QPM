#!/usr/bin/env python3
"""Prepare every audited QPM operation for nominal-volume regression.

No waveform/run sampling, outcome filtering, or dominant-definition selection.
Only exact repeated raw operations across sibling exports are deduplicated.
Programmed volume is the supervised label, never measured delivered volume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from retrain_qpm_volume_models import load_audited, baseline_center, sha256

VERSION = "all-qpm-volume-preparation-1.1.0-event-identity"
SEED = 20261027
PROFILE_COLUMNS = [f"profile_{i:02d}" for i in range(32)]
TASK_KEYS = ["liquid_class_fingerprint", "pipette", "operation", "pressure_unit"]
CONTEXT_KEYS = ["container_geometry_fingerprint", "deck_geometry_fingerprint", "command_settings_fingerprint"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def prepare(bank, output):
    started = time.monotonic()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output is not empty; preserve it and use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = bank / "reference_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("validation_status") != "passed":
        raise ValueError("Source reference bank has not passed validation")
    names = ["qpm_operation_waveforms.parquet", "qpm_physical_operation_context.parquet", "qpm_trace_quality.parquet", "csa_ml_dataset.parquet"]
    hashes = {}
    for name in names:
        hashes[name] = sha256(bank / name)
        if manifest["artifacts"][name]["sha256"] != hashes[name]:
            raise ValueError("Source artifact does not match active bank: " + name)
    print("Loading all command-audited, Er72-cleared QPM events", flush=True)
    frame, audit = load_audited(bank)
    event_times = pq.read_table(bank / "qpm_operation_waveforms.parquet",
        columns=["operation_id", "event_time"], use_threads=False).to_pandas()
    frame = frame.merge(event_times, on="operation_id", how="left", validate="one_to_one")
    usable = len(frame)
    required = TASK_KEYS + CONTEXT_KEYS
    # Preserve exact source-count identities as separate task keys. The legacy
    # parser explicitly labels QPM counts' physical pressure units unresolved;
    # that does not license conversion to psi/kPa or pooling distinct identities.
    complete = frame[required].notna().all(axis=1) & frame.pressure_unit.isin(
        ["raw_sensor_counts", "raw_sensor_count_unresolved"])
    audit["missing_physical_definition_or_unrecognized_sensor_unit"] = int((~complete).sum())
    frame = frame[complete].copy()
    print(f"Audited source operations: {usable:,}; complete physical context: {len(frame):,}", flush=True)
    if frame.empty:
        raise ValueError("No operation has verified physical context and an accepted source-count identity")
    matrix = baseline_center(np.vstack(frame.qpm_profile.to_numpy())).astype("<f8")
    frame["waveform_sha256"] = [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix]
    frame[PROFILE_COLUMNS] = matrix.astype("<f4")
    # Same physical run + raw debug operation + identical curve/configuration
    # is an export duplicate, not another independent transfer. Distinct raw
    # operation indices remain even when their curves are numerically equal.
    duplicate_keys = ["physical_run_id", "operation_index", "event_time", "workflow_stage", "process_step", "loc",
                      "programmed_volume_ul", "waveform_sha256", *TASK_KEYS, *CONTEXT_KEYS]
    frame = frame.sort_values(["physical_run_id", "operation_index", "archive_id"], kind="stable")
    duplicate = frame.duplicated(duplicate_keys, keep="first") & frame.event_time.notna()
    duplicate_rows = frame.loc[duplicate, ["operation_id", "physical_run_id", "archive_id", "operation_index", "event_time", "pipette", "waveform_sha256"]]
    duplicate_rows.to_parquet(output / "duplicate_operation_exports.parquet", index=False)
    frame = frame.loc[~duplicate].copy()
    task_signatures = frame[TASK_KEYS].astype(str).agg("|".join, axis=1)
    frame["task_id"] = "qpm_" + task_signatures.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    volume = frame.groupby(["task_id", "programmed_volume_ul"], sort=False).agg(
        waveforms=("operation_id", "size"), physical_runs=("physical_run_id", "nunique"), instruments=("instrument_id", "nunique")).reset_index()
    volume["minimum_independent_level_support_met"] = volume.physical_runs.ge(8)
    rows = []
    for task_id, group in frame.groupby("task_id", sort=False):
        levels = volume[volume.task_id.eq(task_id)]
        n_runs = int(group.physical_run_id.nunique())
        n_levels = len(levels)
        trainable = n_runs >= 16 and n_levels >= 2
        reason = "trainable_all_events_no_sampling" if trainable else "only_one_observed_volume" if n_levels < 2 else "fewer_than_16_independent_runs"
        aliases = sorted(int(v) for v in group.liquid_class_index.unique())
        first = group.iloc[0]
        rows.append(dict(task_id=task_id, liquid_class_indices=json.dumps(aliases), pipette=first.pipette,
            operation=first.operation, liquid_class_fingerprint=first.liquid_class_fingerprint,
            liquid_class_name=first.get("liquid_class_name"), pressure_unit=first.pressure_unit,
            waveforms=len(group), physical_runs=n_runs, instruments=group.instrument_id.nunique(),
            observed_volume_levels=n_levels, levels_with_at_least_8_runs=int(levels.minimum_independent_level_support_met.sum()),
            minimum_volume_ul=float(group.programmed_volume_ul.min()), maximum_volume_ul=float(group.programmed_volume_ul.max()),
            trainable=trainable, training_status=reason))
    tasks = pd.DataFrame(rows).sort_values(["pipette", "operation", "liquid_class_indices"])
    columns = ["task_id", "operation_id", "physical_run_id", "observation_id", "instrument_id", "archive_id", "chip_lane", "run_start_time",
        "operation_index", "event_time", "operation", "pipette", "programmed_volume_ul", "liquid_class_index", "liquid_class_fingerprint",
        "liquid_class_name", "pressure_unit", "workflow_stage", "process_step", "loc", "waveform_sha256", *CONTEXT_KEYS, *PROFILE_COLUMNS]
    frame = frame[columns].sort_values(["task_id", "run_start_time", "physical_run_id", "operation_index"], kind="stable").reset_index(drop=True)
    frame.to_parquet(output / "all_audited_operations.parquet", index=False, compression="zstd")
    tasks.to_csv(output / "all_task_coverage.csv", index=False)
    volume.to_csv(output / "all_volume_level_coverage.csv", index=False)
    transfers = pd.read_csv(bank / "csa_transfer_manifest.csv", usecols=["archive_id", "sha256"])
    transfers = transfers[transfers.archive_id.isin(frame.archive_id)].drop_duplicates("archive_id")
    audit.update(version=VERSION, prepared_utc=datetime.now(timezone.utc).isoformat(), source_bank=str(bank),
        source_manifest_sha256=sha256(manifest_path), source_sha256=hashes,
        archive_hashes=dict(zip(transfers.archive_id, transfers.sha256)),
        code_sha256=sha256(Path(__file__)), inherited_loader_sha256=hashlib.sha256(inspect.getsource(load_audited).encode()).hexdigest(),
        physical_definition_and_unit_checked_operations=int(complete.sum()),
        duplicate_export_operations=int(duplicate.sum()), unique_audited_operations=len(frame),
        physical_runs=int(frame.physical_run_id.nunique()), observations=int(frame.observation_id.nunique()),
        instruments=int(frame.instrument_id.nunique()), definition_channel_operation_unit_tasks=len(tasks),
        trainable_tasks=int(tasks.trainable.sum()), trainable_waveforms=int(tasks.loc[tasks.trainable, "waveforms"].sum()),
        untrainable_tasks=int((~tasks.trainable).sum()), untrainable_waveforms=int(tasks.loc[~tasks.trainable, "waveforms"].sum()),
        no_run_or_event_cap=True, no_outcome_or_healthy_reference_filter=True,
        random_seed=SEED, elapsed_seconds=time.monotonic()-started,
        source_qpm_resolution="all available stored 32-point QPM profiles; not a new full-rate debug re-extraction",
        actual_delivered_volume_labels=0)
    (output / "prepare_all_qpm_volumes.executed.py").write_bytes(Path(__file__).read_bytes())
    audit["artifact_sha256"] = {name:sha256(output/name) for name in ["all_audited_operations.parquet", "all_task_coverage.csv", "all_volume_level_coverage.csv", "duplicate_operation_exports.parquet"]}
    write_json(output / "preparation_audit.json", audit)
    print(json.dumps({k:audit[k] for k in ["unique_audited_operations", "duplicate_export_operations", "physical_runs", "instruments", "definition_channel_operation_unit_tasks", "trainable_tasks", "trainable_waveforms", "untrainable_waveforms", "elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    prepare(args.bank.resolve(), args.output.resolve())
