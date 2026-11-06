"""Physical-operation QPM references, independent of outcome-peer eligibility.

Raw sensor counts are compared within an instrument/pipette only. Command-time
geometry and liquid-class settings are recovered from CSA/debug, not inherited
from an earlier aspiration. References are empirical, not certified healthy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import re
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from qpm_trace_quality import QUALITY_FIELDS, eligible_waveforms, ensure_trace_quality_cache

VERSION = "qpm-physical-reference-1.4.0-outcome-qualified"
# Reference scoring can evolve without invalidating unchanged debug parsing.
CONTEXT_PARSER_VERSION = "qpm-physical-reference-1.0.0"
DEFAULT_POLICY = {
    "version": "qpm-waveform-reference-policy-1.1.0",
    "time_policy": "retrospective",
    "minimum_physical_runs": 8,
    "preferred_physical_runs": 20,
    "instrument_policy": "same_instrument_then_cross_instrument_shape",
    "minimum_cross_instrument_instruments": 2,
    "cross_instrument_normalization": "baseline_center_then_unit_peak_to_peak",
    "allow_cross_step_fallback": True,
    "chip_context_for_coupler": True,
    "require_outcome_qualified_reference": True,
}
OP_RE = re.compile(r"\bin PIP\.(Aspirate|Dispense) id\s*=\s*(\d+) vol\s*=\s*([-+\d.]+)", re.I)
SEND_RE = re.compile(r"PipSndV1 id:(\d+),pipCmd:(GA|GD)id(\d+)([^,\s]*)(.*)", re.I)
TABLES = {"LiquidClassCommonTbl.json": "lc", "ContainerGeometryCommonTbl.json": "cg",
          "ContainerGeometryWorkflowTbl.json": "cg", "DeckGeometryTbl.json": "dg"}
PROFILE_COLUMNS = [f"profile_{i:02d}" for i in range(32)]
PHYSICAL_KEYS = ["instrument_id", "pipette", "operation", "programmed_volume_ul", "pressure_unit",
                 "liquid_class_fingerprint", "container_geometry_fingerprint", "deck_geometry_fingerprint",
                 "command_settings_fingerprint", "chip_fluidic_context"]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_policy(project: Path, overrides: dict | None = None) -> dict:
    policy = dict(DEFAULT_POLICY)
    path = project / "qpm_waveform_reference_policy.json"
    if path.exists():
        policy.update(json.loads(path.read_text()))
    if overrides:
        unknown = set(overrides) - set(policy)
        if unknown:
            raise ValueError(f"Unknown waveform policy fields: {sorted(unknown)}")
        policy.update(overrides)
    if policy["time_policy"] not in {"prior_only", "retrospective"}:
        raise ValueError("Waveform time_policy must be prior_only or retrospective")
    if policy["instrument_policy"] not in {"same_instrument_only", "same_instrument_then_cross_instrument_shape"}:
        raise ValueError("Only same-instrument raw references or normalized cross-instrument shape fallback are supported")
    if policy["cross_instrument_normalization"] != "baseline_center_then_unit_peak_to_peak":
        raise ValueError("Cross-instrument raw-count pooling is not enabled")
    instrument_minimum = policy["minimum_cross_instrument_instruments"]
    if isinstance(instrument_minimum, bool) or not isinstance(instrument_minimum, int) or instrument_minimum < 2:
        raise ValueError("Cross-instrument shape references require at least two reference instruments")
    if int(policy["minimum_physical_runs"]) < 8:
        raise ValueError("Waveform references require at least eight independent physical runs")
    return policy


def parse_command_context(handle) -> list[dict]:
    """Pair command payloads to operation index and pipette, without storing debug text."""
    rows, pending = [], {}
    index = 0
    for raw in handle:
        line = raw.decode("utf-8", "replace")
        if "in PIP." not in line and "PipSndV1" not in line:
            continue
        match = OP_RE.search(line)
        if match:
            index += 1
            pipette = int(match[2])
            # Debug messages occasionally concatenate a date onto vol=... .
            # Keep operation numbering aligned with the waveform parser, but
            # do not invent a volume or discard the entire archive.
            try:
                volume = float(match[3])
            except ValueError:
                volume = np.nan
            row = {"operation_index": index, "pipette": f"PIP{pipette}",
                   "command_operation": match[1].lower(), "command_volume_ul": volume,
                   "command_context_status": "missing_command"}
            rows.append(row)
            pending[pipette] = row
        match = SEND_RE.search(line)
        if not match or int(match[1]) not in pending:
            continue
        row = pending[int(match[1])]
        expected = "aspirate" if match[2].upper() == "GA" else "dispense"
        if row["command_operation"] != expected:
            continue
        tokens = re.findall(r"([a-z]{2})([-+]?\d+)", match[4].lower())
        fields = dict(tokens)
        if not all(key in fields for key in ["ge", "go", "lq"]):
            row["command_context_status"] = "missing_command_geometry_or_class"
            continue
        row.update(command_container_geometry_index=int(fields["ge"]),
                   command_deck_geometry_index=int(fields["go"]), command_liquid_class_index=int(fields["lq"]),
                   command_context_status="command_verified")
        # ge/go/lq are replaced by full definition fingerprints, not raw IDs.
        # cf/ce/te are absolute calibrated positions: match requested modes,
        # immersion/mix/surface-follow settings instead of calibration offsets.
        fixed = [(key, value) for key, value in tokens if key not in {"ge", "go", "lq", "cf", "ce", "te"}]
        row["command_settings_fingerprint"] = digest([expected, fixed])
        row["command_settings_codes"] = json.dumps(fixed)
    return rows


def definition_maps(tables: dict) -> dict:
    maps = {"lc": {}, "cg": {}, "dg": {}}
    for filename, table in tables.items():
        kind = TABLES[filename]
        if not isinstance(table, dict):
            continue
        for name, definition in table.items():
            if kind == "dg" and isinstance(definition, list) and len(definition) >= 3:
                number = int(definition[1])
                payload = {"encoded_geometry": definition[0], "retraction_height": definition[2]}
            elif isinstance(definition, dict):
                id_key = "lqNum" if kind == "lc" else "cgNum"
                if id_key not in definition:
                    continue
                number = int(definition[id_key])
                payload = {key: value for key, value in definition.items() if key != id_key}
                required = "lcdef" if kind == "lc" else "cgStr"
                if not payload.get(required):
                    continue
            else:
                continue
            entry = {"name": name, "fingerprint": digest(payload), "definition": payload,
                     "source_file": f"CSA/{filename}"}
            if number in maps[kind] and maps[kind][number]["fingerprint"] != entry["fingerprint"]:
                # Conflicting definitions are unresolved, never silently pick one.
                entry = {"name": name, "fingerprint": None, "definition": {}, "source_file": f"CSA/{filename}"}
            maps[kind][number] = entry
    return maps


def extract_archive(record: dict) -> tuple[list[dict], dict]:
    path = Path(record["local_path"])
    tables, rows = {}, []
    source = {key: record[key] for key in ["archive_id", "sha256"]}
    source.update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    base = Path(info.filename).name
                    if base in TABLES and info.file_size <= 2_000_000:
                        tables[base] = json.loads(archive.read(info))
                    elif info.filename.rstrip("/").endswith("CSA/debug"):
                        with archive.open(info) as handle:
                            rows = parse_command_context(handle)
        else:
            # Stream once: no extraction, no repeated decompression/seeking.
            with tarfile.open(path, "r|*") as archive:
                for info in archive:
                    base = Path(info.name).name
                    if info.isfile() and (base in TABLES or info.name.rstrip("/").endswith("CSA/debug")):
                        handle = archive.extractfile(info)
                        if handle is None:
                            continue
                        with handle:
                            if base in TABLES and info.size <= 2_000_000:
                                tables[base] = json.load(handle)
                            elif info.name.rstrip("/").endswith("CSA/debug"):
                                rows = parse_command_context(handle)
        maps = definition_maps(tables)
        for row in rows:
            row["archive_id"] = record["archive_id"]
            for kind, prefix, id_field in [
                ("lc", "liquid_class", "command_liquid_class_index"),
                ("cg", "container_geometry", "command_container_geometry_index"),
                ("dg", "deck_geometry", "command_deck_geometry_index")]:
                entry = maps[kind].get(row.get(id_field), {})
                row[f"{prefix}_fingerprint"] = entry.get("fingerprint")
                row[f"{prefix}_name"] = entry.get("name")
                row[f"{prefix}_source"] = entry.get("source_file")
                if kind == "cg":
                    row["container_model"] = entry.get("definition", {}).get("tubeModel")
            if row["command_context_status"] == "command_verified" and any(not row.get(f"{prefix}_fingerprint")
                    for prefix in ["liquid_class", "container_geometry", "deck_geometry"]):
                row["command_context_status"] = "missing_or_conflicting_configuration_definition"
        source.update(status="parsed", operations=len(rows), config_sha256=digest(tables))
    except Exception as exc:
        source.update(status="failed", error=f"{type(exc).__name__}: {exc}", operations=0)
        rows = []
    return rows, source


def ensure_context_cache(project: Path, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    path = project / "qpm_physical_operation_context.parquet"
    manifest_path = project / "qpm_physical_operation_context_manifest.json"
    qpm_ids = pq.read_table(project / "qpm_operation_waveforms.parquet", columns=["archive_id"], use_threads=False).to_pandas()["archive_id"].unique()
    transfers = pd.read_csv(project / "csa_transfer_manifest.csv", dtype={"archive_id": str})
    records = transfers[transfers["archive_id"].isin(qpm_ids)].drop_duplicates("archive_id").to_dict("records")
    def input_signature(record):
        info=Path(record["local_path"]).stat()
        return record["archive_id"],[record["sha256"],info.st_size,info.st_mtime_ns]
    with ThreadPoolExecutor(max_workers=16) as readers:
        signatures=dict(readers.map(input_signature,records))
    code_hash = digest([CONTEXT_PARSER_VERSION, str(OP_RE.pattern), str(SEND_RE.pattern), TABLES,
                        inspect.getsource(parse_command_context), inspect.getsource(definition_maps),
                        inspect.getsource(extract_archive)])
    if path.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("input_signatures") == signatures and previous.get("parser_version") == CONTEXT_PARSER_VERSION and previous.get("parser_sha256") == code_hash:
            return pq.read_table(path, use_threads=False).to_pandas(), previous
    rows, sources = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(extract_archive, row): row["archive_id"] for row in records}
        for i, future in enumerate(as_completed(jobs), 1):
            batch, source = future.result()
            rows.extend(batch)
            sources.append(source)
            if i % 25 == 0 or i == len(records):
                print(f"Verified physical QPM context: {i}/{len(records)} archives", flush=True)
    context = pd.DataFrame(rows)
    if context.duplicated(["archive_id", "operation_index", "pipette"]).any():
        raise ValueError("Ambiguous command context keys; refusing reference matching")
    context.to_parquet(path, index=False)
    manifest = {"parser_version": CONTEXT_PARSER_VERSION, "parser_sha256": code_hash, "input_signatures": signatures,
                "archives": sorted(sources, key=lambda row: row["archive_id"]), "operations": len(context),
                "sensitive_debug_text_stored": False,
                "geometry_source": "actual_GA_GD_command_ge_go_and_full_configuration_definitions",
                "liquid_class_source": "actual_GA_GD_command_lq_and_full_lcDef_configuration"}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return context, manifest


def load_profiles(project: Path) -> tuple[pd.DataFrame, dict]:
    fields = ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane", "run_start_time",
              "operation_id", "operation_index", "operation", "pipette", "programmed_volume_ul", "event_time",
              "workflow_stage", "process_step", "source_file", "pressure_unit", "liquid_class_index", "loc",
              "qpm_available", "qpm_profile", "container_geometry_index", "deck_geometry_index"]
    qpm = pq.read_table(project / "qpm_operation_waveforms.parquet", columns=fields, use_threads=False).to_pandas()
    qpm = qpm[qpm["qpm_available"].fillna(False) & qpm["programmed_volume_ul"].gt(0)
              & qpm["qpm_profile"].map(lambda value: len(value) == 32)].copy()
    quality, quality_manifest = ensure_trace_quality_cache(project)
    qpm = qpm.merge(quality, on=["archive_id", "operation_index", "pipette"], how="left", validate="one_to_one")
    # Old waveform files did not retain command errors. A missing audit or
    # unmatched operation is unresolved and cannot silently enter a template.
    qpm["qpm_waveform_eligible"] = qpm["qpm_waveform_eligible"].fillna(False)
    qpm["qpm_trace_quality_status"] = qpm["qpm_trace_quality_status"].fillna("unresolved_trace_status")
    qpm["qpm_exclusion_reason"] = qpm["qpm_exclusion_reason"].fillna("trace_status_not_verified")
    accepted = eligible_waveforms(qpm)
    audit_fields = ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane",
                    "operation_id", "operation_index", "operation", "pipette", "programmed_volume_ul",
                    "event_time", "workflow_stage", "process_step", "loc", "source_file", *QUALITY_FIELDS]
    exclusions = qpm.loc[~qpm.operation_id.isin(accepted.operation_id), audit_fields].to_dict("records")
    qpm = accepted
    matrix = np.vstack(qpm["qpm_profile"].to_numpy()) if not qpm.empty else np.empty((0, 32))
    qpm[PROFILE_COLUMNS] = matrix - np.median(matrix[:, :3], axis=1, keepdims=True)
    qpm = qpm[np.isfinite(qpm[PROFILE_COLUMNS]).all(axis=1)].copy()
    context, cache = ensure_context_cache(project)
    cache = {**cache, "trace_quality": quality_manifest, "trace_quality_exclusions": exclusions}
    qpm = qpm.merge(context, on=["archive_id", "operation_index", "pipette"], how="left", validate="one_to_one")
    qpm["legacy_liquid_class_index"] = qpm["liquid_class_index"]
    qpm["liquid_class_index"] = qpm["command_liquid_class_index"]
    mismatch = qpm["operation"].ne(qpm["command_operation"]) | ~np.isclose(qpm["programmed_volume_ul"], qpm["command_volume_ul"], equal_nan=False)
    qpm.loc[mismatch, "command_context_status"] = "command_waveform_alignment_mismatch"
    chip = pq.read_table(project / "csa_ml_dataset.parquet", columns=["observation_id", "ion_params_00.chip_info_chip_type"], use_threads=False).to_pandas()
    qpm = qpm.merge(chip.rename(columns={"ion_params_00.chip_info_chip_type": "chip_type"}), on="observation_id", how="left", validate="many_to_one")
    qpm["location"] = qpm["loc"].fillna("unknown")
    for col in ["workflow_stage", "process_step"]:
        qpm[col] = qpm[col].fillna("unknown")
    qpm["programmed_volume_ul"] = qpm["programmed_volume_ul"].round(3)
    from qpm_reference_qualification import load_reference_qualification
    qualification, qualification_metadata = load_reference_qualification(project)
    qpm = qpm.merge(qualification[["observation_id", "reference_eligible"]],
                    on="observation_id", how="left", validate="many_to_one")
    qpm["reference_eligible"] = qpm["reference_eligible"].fillna(False).astype(bool)
    cache["reference_qualification"] = qualification_metadata
    return qpm, cache


def prepare_reference_pool(qpm: pd.DataFrame, target: pd.Series, policy: dict) -> pd.DataFrame:
    qpm = eligible_waveforms(qpm)
    pool = qpm[qpm["command_context_status"].eq("command_verified") & ~qpm["physical_run_id"].eq(target["physical_run_id"])].copy()
    if policy.get("require_outcome_qualified_reference", True):
        if "reference_eligible" not in pool:
            raise ValueError("Historical waveform eligibility has not been verified against chip/lane reads and barcode balance")
        pool = pool[pool["reference_eligible"].eq(True)].copy()
    if policy["time_policy"] == "prior_only":
        times = pd.to_datetime(pool["run_start_time"], utc=True, errors="coerce")
        target_time = pd.to_datetime(target["run_start_time"], utc=True, errors="coerce")
        pool = pool[times.lt(target_time) & times.notna()]
    return pool


def reference_template(reference: pd.DataFrame, target_time, *, shape_only=False) -> dict:
    """Cacheable cohort statistics; every physical run still has one vote.

    This is identical to the operation-by-operation calculation, but dates,
    medians and audit members are calculated once for a shared cohort.
    """
    grouped = reference.groupby(["physical_run_id", "observation_id"], sort=False)[PROFILE_COLUMNS].median()
    matrix = grouped.groupby(level="physical_run_id", sort=False).median().to_numpy(float)
    median = np.median(matrix, axis=0)
    q05, q95 = np.quantile(matrix, [.05, .95], axis=0)
    amplitude = float(np.median(np.ptp(matrix, axis=1)))
    floor = .02 if shape_only else max(amplitude*.02, 1.0)
    scale = np.maximum(1.4826*np.median(np.abs(matrix-median), axis=0), floor)
    ref_scores = np.sqrt(np.mean(np.clip((matrix-median)/scale, -10, 10)**2, axis=1))
    dates = pd.to_datetime(reference.run_start_time, utc=True, errors="coerce", format="mixed")
    target_time = pd.to_datetime(target_time, utc=True, errors="coerce")
    # Avoid constructing one small DataFrame per reference run per operation.
    # The large bank otherwise spends most scan time building provenance rows.
    by_run=reference.groupby("physical_run_id",sort=False)
    audit=reference.drop_duplicates("physical_run_id").set_index("physical_run_id")[["instrument_id","run_start_time"]].copy()
    audit=audit.rename(columns={"instrument_id":"reference_instrument_id","run_start_time":"reference_run_start_time"})
    def joined_ids(column):
        distinct=reference[["physical_run_id",column]].drop_duplicates()
        if not distinct.physical_run_id.duplicated().any():
            return distinct.set_index("physical_run_id")[column]
        return distinct.sort_values(column,kind="stable").groupby("physical_run_id",sort=False)[column].agg("|".join)
    audit["reference_observation_ids"]=joined_ids("observation_id")
    audit["reference_archive_ids"]=joined_ids("archive_id")
    audit["reference_operation_count"]=by_run.size()
    audit["reference_includes_future"]=dates.ge(target_time).groupby(reference.physical_run_id.to_numpy(),sort=False).any()
    audit["weight_per_physical_run"]=1
    audit["reference_profile_unit"]="dimensionless_shape" if shape_only else "raw_sensor_counts"
    audit_members=audit.reset_index().rename(columns={"physical_run_id":"reference_physical_run_id"}).to_dict("records")
    return dict(matrix=matrix, median=median, q05=q05, q95=q95, amplitude=amplitude,
                scale=scale, ref_scores=ref_scores, audit_members=audit_members,
                instrument_ids=sorted(reference.instrument_id.unique()))


def scan_references(project: Path, target_id: str, config: dict,
                    profiles: tuple[pd.DataFrame, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    policy = load_policy(project, {"minimum_physical_runs": config.get("minimum_reference_physical_runs", 8),
                                   **config.get("waveform_reference_policy_overrides", {})})
    qpm, cache = profiles if profiles is not None else load_profiles(project)
    qpm = eligible_waveforms(qpm)
    target = qpm[qpm["observation_id"].eq(target_id)]
    if target.empty:
        raise ValueError("No usable target QPM waveforms")
    coupler = qpm["container_model"].fillna("").str.contains("chipCoupler", case=False) | qpm["location"].str.contains("chip_coupler", case=False)
    qpm["chip_fluidic_context"] = "not_chip_contact"
    if policy["chip_context_for_coupler"]:
        qpm.loc[coupler, "chip_fluidic_context"] = qpm.loc[coupler, "chip_type"].fillna("missing_chip_context").astype(str)
    target = qpm[qpm["observation_id"].eq(target_id)]
    pool = prepare_reference_pool(qpm, target.iloc[0], policy)
    levels = [("physical_same_instrument_step_location", PHYSICAL_KEYS + ["workflow_stage", "process_step", "location"])]
    if policy["allow_cross_step_fallback"]:
        levels.append(("physical_same_instrument_cross_step", PHYSICAL_KEYS))
    lookups = [(name, keys, pool.groupby(keys, dropna=False, sort=False).indices) for name, keys in levels]
    scores, coverage, members = [], [], []
    template_cache = {}
    minimum = int(policy["minimum_physical_runs"])
    for _, row in target.sort_values("operation_index").iterrows():
        cover = {key: row[key] for key in ["operation_id", "operation_index", "operation", "programmed_volume_ul", "liquid_class_index", "legacy_liquid_class_index", "workflow_stage", "process_step", "location", "command_context_status", "container_geometry_index", "command_container_geometry_index", "deck_geometry_index", "command_deck_geometry_index"]}
        cover.update(reference_status="unresolved", reference_physical_runs=0, maximum_available_reference_runs=0,
                     reason="insufficient_compatible_physical_runs", matched_reference_level="none",
                     reference_scope="unresolved", same_instrument_available_runs=0,
                     cross_instrument_available_runs=0, cross_instrument_available_instruments=0,
                     cross_instrument_attempted=False, cross_instrument_status="not_requested")
        if row["command_context_status"] != "command_verified" or row["chip_fluidic_context"] == "missing_chip_context":
            cover["reason"] = str(row["command_context_status"]) if row["command_context_status"] != "command_verified" else "missing_chip_context_for_chip_contact"
            coverage.append(cover)
            continue
        selected = None
        for level, keys, lookup in lookups:
            indices = lookup.get(tuple(row[key] for key in keys))
            if indices is None:
                continue
            candidates = pool.iloc[indices]
            n = candidates["physical_run_id"].nunique()
            cover["maximum_available_reference_runs"] = max(cover["maximum_available_reference_runs"], n)
            cover["same_instrument_available_runs"] = cover["maximum_available_reference_runs"]
            if n >= minimum:
                selected = level, candidates
                break
        if selected is None:
            coverage.append(cover)
            continue
        level, reference = selected
        # Median within observation first, then physical run: siblings/retries do
        # not gain extra weight; never average unrelated physical runs together.
        cache_key = (level, tuple(row[k] for k in dict(levels)[level]))
        if cache_key not in template_cache:
            template_cache[cache_key] = reference_template(reference, row["run_start_time"])
        template = template_cache[cache_key]
        matrix = template["matrix"]
        observed = row[PROFILE_COLUMNS].to_numpy(float)
        median, q05, q95 = template["median"], template["q05"], template["q95"]
        amplitude, scale = template["amplitude"], template["scale"]
        shape = float(np.sqrt(np.mean(np.clip((observed - median) / scale, -10, 10) ** 2)))
        ref_scores = template["ref_scores"]
        percentile = float(((ref_scores <= shape).sum() + 1) / (len(matrix) + 1))
        outside = float(np.mean((observed < q05) | (observed > q95)))
        ratio = float(np.ptp(observed) / amplitude) if amplitude else np.nan
        record = {key: row[key] for key in ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane", "operation_id", "operation_index", "operation", "pipette", "programmed_volume_ul", "event_time", "workflow_stage", "process_step", "source_file", "pressure_unit", "liquid_class_index", "legacy_liquid_class_index", "location", "container_geometry_name", "container_model", "deck_geometry_name", "liquid_class_name", "command_context_status"] + PHYSICAL_KEYS[5:]}
        record.update({key: row[key] for key in QUALITY_FIELDS if key in row})
        record.update(matched_reference_level=level, reference_physical_runs=len(matrix), reference_time_policy=policy["time_policy"],
                      reference_scope="same_instrument", reference_profile_unit="raw_sensor_counts",
                      reference_normalization="baseline_center_only", reference_instruments=1,
                      reference_instrument_ids=str(row["instrument_id"]), reference_shape_only=False,
                      reference_observed_profile=json.dumps(observed.tolist()),
                      reference_precision="sparse_exploratory" if len(matrix) < policy["preferred_physical_runs"] else "empirical_not_certified_healthy",
                      qpm_shape_score=shape, qpm_anomaly_percentile=percentile, outside_90pct_envelope_fraction=outside,
                      amplitude_ratio_to_reference=ratio, waveform_morphology="amplified_waveform" if ratio > 1.5 else "attenuated_waveform" if ratio < .65 else "shape_deviation" if outside >= .25 else "subtle_or_mixed_deviation",
                      pool_transfer_relevant_step=bool(re.search(r"sample|pool|library|ligat|transfer|amplicon", str(row["process_step"]) + str(row["location"]), re.I)),
                      observed_profile=json.dumps(observed.tolist()), reference_median_profile=json.dumps(median.tolist()),
                      reference_q05_profile=json.dumps(q05.tolist()), reference_q95_profile=json.dumps(q95.tolist()))
        scores.append(record)
        cover.update(reference_status="scored", reference_physical_runs=len(matrix), reason="matched_verified_physical_configuration",
                     matched_reference_level=level, reference_scope="same_instrument")
        coverage.append(cover)
        for member in template["audit_members"]:
            members.append({"target_operation_id": row["operation_id"], "target_operation_index": row["operation_index"],
                            **member, "matched_reference_level": level})
    coverage_frame = pd.DataFrame(coverage)
    corrections = int(target["legacy_liquid_class_index"].ne(target["liquid_class_index"]).sum())
    metadata = {"version": VERSION, "policy": policy, "target_physical_run_excluded": True,
                "reference_qualification": cache.get("reference_qualification", {}),
                "independent_of_outcome_peer_filter": True, "usable_target_operations": len(target), "scored_operations": len(scores),
                "liquid_class_labels_corrected": corrections,
                "context_cache_parser_sha256": cache["parser_sha256"], "input_archives": cache["archives"],
                "Er72_excluded_from_target_and_references": True,
                "trace_quality_version": cache.get("trace_quality", {}).get("version"),
                "warnings": [f"Physical QPM scan: {len(scores)}/{len(target)} usable operations have >= {minimum} compatible physical runs; uncovered operations are unresolved.",
                             f"{corrections} target liquid-class labels differ from legacy inherited labels; actual GA/GD command labels are used.",
                             "Waveform references are independent of strict outcome peers. Same instrument/pipette only; no uncalibrated cross-instrument raw-count pooling.",
                             "Retrospective waveform references may include later runs; not prospective validation." if policy["time_policy"] == "retrospective" else "Waveform references use prior physical runs only.",
                             "Liquid-class settings do not verify actual liquid identity, tip wetting, input quality, or delivered volume. Cross-step fallback is exploratory and explicitly labeled.",
                             "Sensor calibration history, controller firmware, and hardware replacements are not verified; same-instrument matching does not prove sensor invariance.",
                             "Empirical reference bands and shape percentiles are exploratory; not calibrated false-positive probabilities or certified healthy references."]}
    failed = [archive for archive in cache["archives"] if archive["status"] == "failed"]
    if failed:
        metadata["warnings"].append(f"{len(failed)} command-context archives failed parsing and cannot supply references; see input archive audit.")
    return pd.DataFrame(scores), coverage_frame, pd.DataFrame(members, columns=["target_operation_id", "target_operation_index", "reference_physical_run_id", "reference_observation_ids", "reference_archive_ids", "reference_instrument_id", "reference_run_start_time", "reference_operation_count", "reference_includes_future", "matched_reference_level", "weight_per_physical_run", "reference_profile_unit"]), metadata


def write_reference_audit(output: Path, coverage: pd.DataFrame, members: pd.DataFrame, metadata: dict) -> None:
    coverage.to_csv(output / "qpm_waveform_reference_coverage.csv", index=False)
    members.to_csv(output / "qpm_waveform_reference_members.csv", index=False)
    (output / "qpm_waveform_reference_manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-context", action="store_true")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.build_context:
        table, manifest = ensure_context_cache(args.project, args.workers)
        print(f"Command context ready: {len(table)} operations; {sum(row['status'] == 'failed' for row in manifest['archives'])} failed archives")
