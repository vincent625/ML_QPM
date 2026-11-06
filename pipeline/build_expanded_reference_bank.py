#!/usr/bin/env python3
"""Build a versioned, privacy-minimized plugin reference bank from local CSA.

Reads only verified local copies. Raw archives remain intact; no feature-model
training and no instrument writes. Earlier ML datasets are never replaced.
"""
from __future__ import annotations

import argparse
import configparser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import io
import json
import lzma
import multiprocessing
from pathlib import Path
import re
import shutil
import tarfile
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "expanded-csa-reference-builder-1.0.2"
WAVE_FIELDS = ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane", "run_start_time",
               "operation_id", "operation_index", "operation", "pipette", "programmed_volume_ul", "event_time",
               "workflow_stage", "process_step", "source_file", "pressure_unit", "liquid_class_index", "loc",
               "qpm_available", "qpm_profile", "container_geometry_index", "deck_geometry_index"]
ID_FIELDS = ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane", "run_start_time"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    partial = Path(str(path)+".partial")
    partial.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    partial.replace(path)


def instrument_label(value):
    match = re.search(r"(?i)(?:GNXS[-_])?(VC\d+|\d{3}[\dA-Z])(?:\.|[-_]|$)", str(value))
    return "gnxs-"+match[1].lower() if match else None


def declared_origin(raw):
    match = re.search(r"(?i)_ion(?:admin|service)_GNXS[-_](VC\d+|[0-9A-Z]{4})(?:[-_]|$)", raw)
    return "gnxs-"+match[1].lower() if match else None


def canonical_hostname(label, legacy_hosts):
    matches = [h for h in legacy_hosts if h.split(".")[0].lower() == label]
    return matches[0] if len(matches) == 1 else label


def clean_member(name):
    return name.replace("\\", "/").lstrip("./")


def members(path, fmt):
    """Read every member and compression trailer, validating ZIP CRC/XZ/GZIP.

    No extraction to the filesystem and no unsafe path traversal.
    """
    if fmt == "zip":
        with Path(path).open("rb",buffering=8*2**20) as raw, zipfile.ZipFile(raw) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as handle:
                    yield clean_member(info.filename), info.file_size, handle
                    for _ in iter(lambda: handle.read(2**20), b""):
                        pass  # CRC of even unselected members must be checked.
        return
    opener = {"xz_tar": lzma.open, "gzip_tar": gzip.open}.get(fmt)
    if opener is None and fmt != "tar":
        raise ValueError("Unrecognized container format")
    with ExitStack() as stack:
        raw=stack.enter_context(Path(path).open("rb",buffering=8*2**20))
        compressed=raw if fmt=="tar" else stack.enter_context(opener(raw,"rb"))
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for info in archive:
                if info.isfile():
                    with archive.extractfile(info) as handle:
                        yield clean_member(info.name), info.size, handle
        for _ in iter(lambda: compressed.read(2**20), b""):
            pass  # Force decompressor trailer / checksum validation.


def resolve_metadata(documents):
    import analyze_csa as core
    from outcome_reference import SOURCES, PANEL_COLUMNS, normalize
    needed = set(SOURCES.values()) | set(PANEL_COLUMNS)
    values = {k: [] for k in needed}
    for source, obj in documents.items():
        source_key = core.normalize_name(Path(source).stem)
        for field, value in core.flatten_json(obj):
            name = f"{source_key}.{core.normalize_name(field)}"
            if name in needed and value is not None:
                number = core.parse_number(value)
                values[name].append(normalize(number if number is not None else value))
    resolved, conflicts = {}, []
    for name, candidates in values.items():
        distinct = sorted(set(v for v in candidates if v is not None))
        resolved[name] = distinct[0] if len(distinct) == 1 else None
        if len(distinct) > 1:
            conflicts.append(name)
    return resolved, conflicts


def parse_global_ini(data):
    """Support both sectioned INI and the established dotted key=value form."""
    text = data.decode("utf-8-sig", "replace")
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text)
        return {section: dict(parser.items(section)) for section in parser.sections()}
    except configparser.MissingSectionHeaderError:
        return {key.strip(): value.strip() for line in text.splitlines()
                if "=" in line and not line.lstrip().startswith(("#", ";", "import "))
                for key, value in [line.split("=", 1)]}


def resolve_canonical_outcomes(candidates):
    """Equal-priority contradictory outcome values are unresolved, not averaged."""
    import analyze_csa as core
    values, provenance = core.resolve_outcomes(candidates)
    for outcome in (*core.TARGETS, core.SECONDARY_TARGET):
        options = [r for r in candidates if r["outcome"] == outcome]
        if not options:
            continue
        priority = min(r["priority"] for r in options)
        if len({r["value"] for r in options if r["priority"] == priority}) > 1:
            values[outcome] = np.nan
            values[outcome+"__source_file"] = "unresolved_equal_priority_conflict"
            for r in provenance:
                if r["outcome"] == outcome:
                    r["selected"] = False
    # Invalid outcomes cannot supply a benchmark.
    for name, (low, high) in {"total_reads": (0, float("inf")), "mean_read_length_bp": (0, 10000),
                               "aq20_mean_read_length_bp": (0, 10000), "raw_read_accuracy_pct": (0, 100)}.items():
        if pd.notna(values.get(name)) and not low <= values[name] <= high:
            values[name] = np.nan
            values[name+"__source_file"] = "invalid_value_excluded"
    return values, provenance


def attach_context(rows, tables, archive_id):
    import qpm_waveform_reference as physical
    maps = physical.definition_maps(tables)
    for row in rows:
        row["archive_id"] = archive_id
        for kind, prefix, field in [("lc", "liquid_class", "command_liquid_class_index"),
                                     ("cg", "container_geometry", "command_container_geometry_index"),
                                     ("dg", "deck_geometry", "command_deck_geometry_index")]:
            entry = maps[kind].get(row.get(field), {})
            row[prefix+"_fingerprint"] = entry.get("fingerprint")
            row[prefix+"_name"] = entry.get("name")
            row[prefix+"_source"] = entry.get("source_file")
            if kind == "cg":
                row["container_model"] = entry.get("definition", {}).get("tubeModel")
        if row["command_context_status"] == "command_verified" and any(not row.get(p+"_fingerprint")
                for p in ["liquid_class", "container_geometry", "deck_geometry"]):
            row["command_context_status"] = "missing_or_conflicting_configuration_definition"
    return rows


def ambiguous_acquisition_keys(results, legacy_core):
    """Names with the same instrument/start time must not get extra votes.

    Without a verified alias map, quarantine new conflicting raw names rather
    than treating renamed exports as independent acquisitions.
    """
    rows=legacy_core[["instrument_id","run_start_time","physical_run_id"]].to_dict("records")
    rows += [{k:r["row"].get(k) for k in ["instrument_id","run_start_time","physical_run_id"]}
             for r in results if r.get("run_time_source")=="raw_run_header_assumed_UTC"]
    frame=pd.DataFrame(rows)
    frame["acquisition_time"]=pd.to_datetime(frame.run_start_time,utc=True,errors="coerce",format="mixed")
    counts=frame.dropna(subset=["acquisition_time"]).groupby(["instrument_id","acquisition_time"]).physical_run_id.nunique()
    return set(counts[counts.gt(1)].index)


def parse_one(record, output, legacy_hosts):
    import analyze_csa as core
    import train_multimodal_csa as multi
    import qpm_waveform_reference as physical
    from qpm_trace_quality import QUALITY_FIELDS
    from analyze_dna_rna_barcode_balance import safe_barcode_plan, attach_read_counts, classify_barcodes
    from find_qpm_failure_examples import BARCODE_RE, normalize_barcode
    aid, path = record["archive_id"], Path(record["local_path"])
    cache = Path(output) / ".reference_parse_cache" / aid
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = dict(version=VERSION, archive_id=aid, sha256=record["sha256"], status="failed",
                  warnings=[], member_count=0, raw_debug_text_stored=False)
    documents, tables, candidates, samples, stats, debug = {}, {}, [], {}, {}, None
    try:
        for name, size, handle in members(path, record["container_format"]):
            result["member_count"] += 1
            base = Path(name).name
            if name == "CSA/debug":
                if debug is not None or size > 1_000_000_000:
                    raise ValueError("Ambiguous or oversized root debug")
                debug = handle.read()
            elif name == "CSA/analysisSamples.json":
                samples = json.load(handle)
            elif base in physical.TABLES and size <= 2_000_000:
                value = json.load(handle)
                if base in tables and tables[base] != value:
                    raise ValueError("Conflicting repeated physical configuration table")
                tables[base] = value
            elif name in {"CSA/ion_params_00.json", "CSA/planned_run.json", "CSA/global.json"}:
                if size > 64_000_000 or name in documents:
                    raise ValueError("Ambiguous or oversized root metadata")
                documents[name] = json.load(handle)
            elif base.lower() == "global.ini" and size <= 10_000_000:
                documents[name] = parse_global_ini(handle.read())
            elif name.lower() in {"csa/outputs/basecallingactor-00/ionstats_basecaller.json",
                                 "csa/outputs/basecallingactor-00/basecaller.json"}:
                value = json.load(handle)
                candidates.extend(core.extract_outcomes_from_json(value, name))
                if base.lower() == "ionstats_basecaller.json":
                    stats = value
            elif base.lower() == "info.csv" and size <= 10_000_000:
                candidates.extend(core.parse_info_csv(handle.read(), name))
        result["container_validation"] = "full_content_and_compression_checksum_passed"
        params = documents.get("CSA/ion_params_00.json", {})
        raw = str(params.get("expName") or "").strip()
        if not raw:
            raise ValueError("Raw physical-run key missing")
        folder_label = instrument_label(record["source_instrument_folder"])
        origin = declared_origin(raw)
        if not origin and not folder_label:
            raise ValueError("Instrument identity unresolved")
        hostname = canonical_hostname(origin or folder_label, legacy_hosts)
        lane = re.search(r"(?i)ChipLane(\d+(?:_\d+)*)", record["filename"])
        if not lane:
            raise ValueError("Lane scope unresolved")
        lane = lane[1]
        inv = dict(archive_id=aid, hostname=hostname, ssh_user="", remote_path=record["source_path"],
                   run_name=re.sub(r"(?i)\.ChipLane.*$", "", record["filename"]),
                   physical_experiment_name="raw_key_sha256:"+hashlib.sha256(raw.encode()).hexdigest(),
                   chip_lane=lane, plugin_version="local_downloaded_collection", filename=record["filename"],
                   size_bytes=record["size_bytes"], remote_mtime_utc=record["source_mtime_utc"],
                   match_reason="user_authorized_local_collection", source_instrument_folder=record["source_instrument_folder"])
        manifest = dict(archive_id=aid, hostname=hostname, remote_path=inv["remote_path"], local_path=str(path),
                        sha256=record["sha256"], state="already_present", size_bytes=record["size_bytes"],
                        container_format=record["container_format"], validation_result=result["container_validation"])
        row = core.derive_observation(inv, manifest)
        row["physical_run_id"] = core.stable_hash(f"{hostname}|{raw}", "run_")
        row["observation_id"] = core.stable_hash(f"{row['physical_run_id']}|{lane}", "obs_")
        match = re.search(r"R_(20\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", raw)
        if match:
            row["run_start_time"] = datetime(*map(int, match.groups()), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            result["run_time_source"] = "raw_run_header_assumed_UTC"
        else:
            row["run_start_time"] = None
            result["warnings"].append("Run time unresolved; excluded from prior-only outcome peers")
        row.pop("raw_experiment_name", None)
        row["analysis_lane_scope"] = "combined_multilane" if "_" in lane else "single_lane"
        metadata, conflicts = resolve_metadata(documents)
        row.update(metadata)
        result["metadata_conflicts"] = conflicts
        outcomes, provenance = resolve_canonical_outcomes(candidates)
        row.update(outcomes)
        result.update(identity_source="canonical_raw_header" if origin else "source_folder_only",
                      source_folder_origin_mismatch=bool(origin and folder_label and origin != folder_label),
                      raw_run_key_hash=hashlib.sha256(raw.encode()).hexdigest(),
                      instrument_label=origin or folder_label)
        quality_stats, qpm_rows, context_rows = {}, [], []
        if debug is not None:
            base = {k: row[k] for k in ID_FIELDS}
            qpm_rows = multi.parse_qpm_stream(io.BytesIO(debug), base, compact=True, quality_stats=quality_stats)
            context_rows = attach_context(physical.parse_command_context(io.BytesIO(debug)), tables, aid)
            del debug
        else:
            result["warnings"].append("Root CSA/debug missing; no waveform reference contribution")
        qpm = pd.DataFrame(qpm_rows)
        qpm.reindex(columns=WAVE_FIELDS).to_parquet(cache / "waves.parquet", index=False)
        quality = qpm.reindex(columns=["archive_id", "operation_index", "pipette", *QUALITY_FIELDS])
        quality.to_parquet(cache / "quality.parquet", index=False)
        pd.DataFrame(context_rows).to_parquet(cache / "context.parquet", index=False)
        plan = pd.DataFrame(safe_barcode_plan(aid, samples))
        bars = pd.DataFrame()
        balance = {"observation_id": row["observation_id"]}
        counts = {}
        for key, value in stats.get("read_group_metrics", {}).items():
            matches = BARCODE_RE.findall(key)
            if len(matches) == 1 and isinstance(value, dict) and value.get("read_count") is not None:
                token = normalize_barcode(matches[0])
                counts[token] = counts.get(token, 0) + float(value["read_count"])
        if not plan.empty:
            for tokens in plan.barcode_tokens:
                for token in tokens:
                    counts.setdefault(token, np.nan)
            count_table = pd.DataFrame([{"archive_id": aid, **counts}]).set_index("archive_id")
            bars = attach_read_counts(pd.DataFrame([row]), plan, count_table, pd.DataFrame([inv]))
            bars, _ = classify_barcodes(bars)
        for material in ["dna", "rna"]:
            g = bars[(~bars.is_control) & bars.material_type.eq(material.upper())] if not bars.empty else bars
            balance[material+"_n_planned_noncontrol_barcodes"] = len(g) if not plan.empty else np.nan
            balance[material+"_median_reads"] = g.read_count.median() if len(g) else np.nan
            balance[material+"_apparent_dropout_count"] = int(g.apparent_dropout.sum()) if len(g) else 0
        bars.to_parquet(cache / "barcodes.parquet", index=False)
        result.update(status="parsed", row=row, inventory=inv, manifest=manifest, balance=balance,
                      outcome_provenance=provenance, operations=len(qpm), context_operations=len(context_rows),
                      waveform_candidates=int(qpm.qpm_available.sum()) if len(qpm) else 0,
                      trace_quality_stats=quality_stats, config_sha256=physical.digest(tables),
                      barcode_rows=len(bars), parse_finished_utc=utc_now())
    except Exception as exc:
        # Parser exceptions can echo raw configuration lines. Keep only the
        # exception class; raw archive text is never written to an audit log.
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        result.update(error_type=type(exc).__name__, error_location=f"{Path(frame.filename).name}:{frame.lineno}",
                      error="Archive parse failed; exception class retained without raw source text")
    write_json(cache / "result.json", result)
    return {k: result.get(k) for k in ["archive_id", "status", "operations", "waveform_candidates", "error_type"]}


def parse_staged(output, legacy, workers=4, follow=False):
    output, legacy = Path(output), Path(legacy)
    (output / ".reference_parse_cache").mkdir(exist_ok=True, mode=0o700)
    old = pd.read_csv(legacy / "csa_transfer_manifest.csv", dtype={"archive_id": str})
    old_hashes = set(pd.read_parquet(legacy/"csa_ml_dataset.parquet",columns=["archive_sha256"]).archive_sha256.dropna())
    hosts = pd.read_csv(legacy / "csa_inventory.csv").hostname.dropna().unique().tolist()
    if (output/"staging_summary.json").exists():
        records=pd.read_csv(output/"source_transfer_manifest.csv",dtype={"archive_id":str})
        records=records[records.state.isin(["transferred","already_present"])&~records.sha256.isin(old_hashes)].drop_duplicates("archive_id").to_dict("records")
        def cached_result(path):
            try:
                value=json.loads(path.read_text())
                return value["archive_id"] if value.get("version")==VERSION else None
            except (OSError,json.JSONDecodeError):
                return None
        with ThreadPoolExecutor(max_workers=16) as readers:
            cached=set(readers.map(cached_result,(output/".reference_parse_cache").glob("*/result.json")))-{None}
        jobs=[r for r in records if r["archive_id"] not in cached]
        print(json.dumps(dict(stage="bulk_reference_parse",cached_terminal_archives=len(cached),remaining_archives=len(jobs))),flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            pending=[pool.submit(parse_one,r,output,hosts) for r in jobs]
            for index,future in enumerate(as_completed(pending),1):
                result=future.result()
                if index%25==0 or result["status"]!="parsed" or index==len(jobs):
                    print(json.dumps(dict(stage="bulk_reference_parse",completed_this_session=index,total_this_session=len(jobs),**result)),flush=True)
        return
    submitted, futures, seen_receipts = set(), {}, set()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while True:
            for receipt in sorted((output / ".stage_receipts").glob("*.json")):
                # On network storage, reopening every old receipt after each
                # completed task makes scheduling quadratic and idles workers.
                if receipt.name in seen_receipts:
                    continue
                if len(futures) >= workers*3:
                    break
                try:
                    r = json.loads(receipt.read_text())
                except json.JSONDecodeError:
                    continue  # A receipt may still be finishing its write.
                seen_receipts.add(receipt.name)
                aid = r.get("archive_id")
                if not aid or aid in submitted or r.get("state") not in {"transferred", "already_present"}:
                    continue
                cache = output / ".reference_parse_cache" / aid / "result.json"
                if r["sha256"] in old_hashes or cache.exists() and json.loads(cache.read_text()).get("version") == VERSION:
                    submitted.add(aid)
                    continue
                # Bound the pending queue; source copying continues concurrently.
                submitted.add(aid)
                futures[pool.submit(parse_one, r, output, hosts)] = aid
            if futures:
                done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    del futures[future]
                    completed += 1
                    if completed % 10 == 0 or result["status"] != "parsed":
                        print(json.dumps(dict(stage="compact_reference_parse", completed_this_session=completed,
                                              submitted=len(submitted), **result)), flush=True)
            else:
                staged_done = (output / "staging_summary.json").exists()
                if not follow or staged_done:
                    break
                time.sleep(10)
    print(json.dumps(dict(stage="compact_parse_complete", completed_this_session=completed,
                          unique_archive_hashes_seen=len(submitted))), flush=True)


def concat_parquet(paths, destination, fields=None):
    """Bounded prefetch avoids thousands of serial network-storage opens."""
    with ThreadPoolExecutor(max_workers=16) as readers:
        raw_schemas=list(readers.map(pq.read_schema,paths))
    schemas=[s if fields is None else pa.schema([s.field(c) for c in fields]) for s in raw_schemas if len(s)]
    if not schemas:
        raise ValueError(f"No schema for {destination}")
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    partial = Path(str(destination)+".partial")
    active_paths=[p for p,s in zip(paths,raw_schemas) if len(s)]
    def read_table(path):
        return pq.read_table(path,columns=fields,use_threads=False)
    with pq.ParquetWriter(partial, schema, compression="zstd") as writer, ThreadPoolExecutor(max_workers=16) as readers:
        for start in range(0,len(active_paths),16):
            for table in readers.map(read_table,active_paths[start:start+16]):
                arrays=[table[c].cast(schema.field(c).type) if c in table.column_names else pa.nulls(len(table), type=schema.field(c).type)
                        for c in schema.names]
                writer.write_table(pa.Table.from_arrays(arrays,schema=schema))
    partial.replace(destination)


def finalize(output, legacy):
    import qpm_waveform_reference as physical
    import qpm_trace_quality as quality
    from outcome_reference import SOURCES, PANEL_COLUMNS, normalize
    output, legacy = Path(output), Path(legacy)
    staging = json.loads((output / "staging_summary.json").read_text())
    receipts = pd.read_csv(output / "source_transfer_manifest.csv", dtype={"archive_id": str})
    old_core = pd.read_parquet(legacy / "csa_ml_dataset.parquet", columns=[*ID_FIELDS, "analysis_id", "archive_sha256",
                       "total_reads", "mean_read_length_bp", "raw_read_accuracy_pct", *SOURCES.values(), *PANEL_COLUMNS])
    old_core["analysis_lane_scope"] = "single_lane"
    old_transfers = pd.read_csv(legacy / "csa_transfer_manifest.csv", dtype={"archive_id": str})
    identity_hashes=pd.concat([receipts[["archive_id","sha256"]],old_transfers[["archive_id","sha256"]]],ignore_index=True).dropna()
    if identity_hashes.groupby("archive_id").sha256.nunique().gt(1).any():
        raise ValueError("Truncated archive-ID collision detected; refusing any silent overwrite")
    old_inventory = pd.read_csv(legacy / "csa_inventory.csv", dtype={"archive_id": str})
    old_wave_aids = set(pq.read_table(legacy / "qpm_operation_waveforms.parquet", columns=["archive_id"]).to_pandas().archive_id)
    # Match complete hashes, not truncated IDs or filenames.
    old_hashes = set(old_core.archive_sha256.dropna())
    source_ids=receipts.loc[receipts.state.isin(["transferred","already_present"])&~receipts.sha256.isin(old_hashes),"archive_id"].drop_duplicates()
    with ThreadPoolExecutor(max_workers=16) as readers:
        results=list(readers.map(lambda aid: json.loads((output/".reference_parse_cache"/aid/"result.json").read_text()),source_ids))
    results = [r for r in results if r.get("version") == VERSION]
    successful = [r for r in results if r["status"] == "parsed"]
    physical_keys=pd.DataFrame([dict(physical_run_id=r["row"]["physical_run_id"],raw_run_key_hash=r["raw_run_key_hash"]) for r in successful])
    if len(physical_keys) and physical_keys.groupby("physical_run_id").raw_run_key_hash.nunique().gt(1).any():
        raise ValueError("Physical-run-ID collision detected")
    expected = set(receipts.loc[receipts.state.isin(["transferred", "already_present"]), "sha256"]) - old_hashes
    if expected - {r["sha256"] for r in results}:
        raise ValueError("Some staged archives have not reached a parse terminal state")
    audit = [{k: r.get(k) for k in ["archive_id", "sha256", "status", "member_count", "container_validation", "operations",
                 "waveform_candidates", "barcode_rows", "identity_source", "instrument_label", "source_folder_origin_mismatch",
                 "raw_run_key_hash", "error_type", "error"]} for r in results]
    pd.DataFrame(audit).to_csv(output / "archive_parse_audit.csv", index=False)
    # Ambiguous cross-instrument raw keys cannot vote in either reference bank.
    raw_hosts = {}
    for r in successful:
        raw_hosts.setdefault(r["raw_run_key_hash"], set()).add(r["row"]["instrument_id"])
    ambiguous = {key for key, hosts in raw_hosts.items() if len(hosts) > 1}
    excluded = [r for r in successful if r["raw_run_key_hash"] in ambiguous]
    successful = [r for r in successful if r["raw_run_key_hash"] not in ambiguous]
    acquisition_aliases=ambiguous_acquisition_keys(successful,old_core)
    timestamp_conflicts=[]
    for r in successful:
        timestamp=pd.to_datetime(r["row"].get("run_start_time"),utc=True,errors="coerce")
        if (r["row"]["instrument_id"],timestamp) in acquisition_aliases:
            timestamp_conflicts.append(r)
    excluded += timestamp_conflicts
    conflict_ids={r["archive_id"] for r in timestamp_conflicts}
    successful=[r for r in successful if r["archive_id"] not in conflict_ids]
    # Preserve the audited legacy primary observations. Fresh exports of those
    # raw run/lane observations are recorded but do not change their identity.
    legacy_obs = set(old_core.observation_id)
    legacy_wave_counts = pq.read_table(legacy / "qpm_operation_waveforms.parquet", columns=["archive_id", "qpm_available"], use_threads=False).to_pandas()
    legacy_wave_counts = legacy_wave_counts.groupby("archive_id").qpm_available.sum().to_dict()
    legacy_barcode_counts=pd.read_csv(legacy/"qpm_failure_examples/dna_rna_barcode_counts.csv",usecols=["archive_id"]).archive_id.value_counts().to_dict()
    metadata_fields=[*SOURCES.values(),*PANEL_COLUMNS]
    legacy_completeness = {r["observation_id"]: (sum(pd.notna(r.get(y)) for y in ["total_reads", "mean_read_length_bp", "raw_read_accuracy_pct"]),
                            int(legacy_wave_counts.get(r["archive_id"],0)),sum(normalize(r.get(c)) is not None for c in metadata_fields),
                            int(legacy_barcode_counts.get(r["archive_id"],0))) for r in old_core.to_dict("records")}
    grouped = {}
    for r in successful:
        grouped.setdefault(r["row"]["observation_id"], []).append(r)
    selected, duplicates, replaced_legacy_obs = [], [], set()
    for obs, candidates in grouped.items():
        candidates.sort(key=lambda r: (sum(pd.notna(r["row"].get(y)) for y in ["total_reads", "mean_read_length_bp", "raw_read_accuracy_pct"]),
                                         r["waveform_candidates"], r["context_operations"], r["barcode_rows"], r["member_count"],
                                         r["inventory"]["remote_mtime_utc"]), reverse=True)
        for index, r in enumerate(candidates):
            new_score = (sum(pd.notna(r["row"].get(y)) for y in ["total_reads", "mean_read_length_bp", "raw_read_accuracy_pct"]),
                         r["waveform_candidates"],sum(normalize(r["row"].get(c)) is not None for c in metadata_fields),r["barcode_rows"])
            replaces_legacy = obs in legacy_obs and new_score > legacy_completeness[obs]
            keep = index == 0 and (obs not in legacy_obs or replaces_legacy)
            duplicates.append(dict(observation_id=obs, physical_run_id=r["row"]["physical_run_id"], archive_id=r["archive_id"],
                                   retained_primary=keep, duplicate_type="legacy_observation_reexport" if obs in legacy_obs else "unique" if len(candidates)==1 else "reanalysis_or_reexport",
                                   selection_reason="more_complete_export_replaces_legacy_primary" if keep and replaces_legacy else "audited_legacy_primary_preserved_on_completeness_tie" if obs in legacy_obs else "most_complete_then_latest" if keep else "same_raw_run_lane_duplicate"))
            if keep:
                selected.append(r)
                if replaces_legacy:
                    replaced_legacy_obs.add(obs)
    pd.DataFrame(duplicates).to_csv(output / "duplicate_archives.csv", index=False)
    pd.DataFrame([{**{k:r.get(k) for k in ["archive_id", "raw_run_key_hash", "instrument_label"]},
                   "reason":"same_instrument_time_conflicting_raw_identity" if r["archive_id"] in conflict_ids else "same_raw_key_multiple_instruments"}
                  for r in excluded],columns=["archive_id","raw_run_key_hash","instrument_label","reason"]).to_csv(output / "quarantined_identity_archives.csv", index=False)
    retained_legacy = old_core[~old_core.observation_id.isin(replaced_legacy_obs)].copy()
    core = pd.concat([retained_legacy, pd.DataFrame([r["row"] for r in selected])], ignore_index=True)
    for column in [*SOURCES.values(), *PANEL_COLUMNS]:
        core[column] = core[column].map(normalize)
    if core.observation_id.duplicated().any() or core.duplicated(["physical_run_id", "chip_lane"]).any():
        raise ValueError("Duplicate independent observation in expanded bank")
    core.to_parquet(output / "csa_ml_dataset.parquet", index=False)
    core.drop(columns=[c for c in core if c.endswith("__source_file") or c.endswith("__source_field")], errors="ignore").to_csv(output / "reference_observations.csv", index=False)
    transfers = pd.concat([old_transfers, pd.DataFrame([r["manifest"] for r in successful])], ignore_index=True).drop_duplicates("archive_id")
    inventory = pd.concat([old_inventory, pd.DataFrame([r["inventory"] for r in successful])], ignore_index=True).drop_duplicates("archive_id")
    transfers.to_csv(output / "csa_transfer_manifest.csv", index=False)
    inventory.to_csv(output / "csa_inventory.csv", index=False)
    paths = [output / ".reference_parse_cache" / r["archive_id"] for r in selected if r["operations"]]
    retained_legacy_aids = set(retained_legacy.archive_id)
    filtered_legacy = output / ".retained_legacy"
    filtered_legacy.mkdir(exist_ok=True)
    for name in ["qpm_operation_waveforms.parquet", "qpm_physical_operation_context.parquet", "qpm_trace_quality.parquet"]:
        data = pq.read_table(legacy / name, columns=WAVE_FIELDS if name == "qpm_operation_waveforms.parquet" else None, use_threads=False).to_pandas()
        data[data.archive_id.isin(retained_legacy_aids)].to_parquet(filtered_legacy/name, index=False)
    concat_parquet([filtered_legacy / "qpm_operation_waveforms.parquet", *[p/"waves.parquet" for p in paths]], output / "qpm_operation_waveforms.parquet", WAVE_FIELDS)
    concat_parquet([filtered_legacy / "qpm_physical_operation_context.parquet", *[p/"context.parquet" for p in paths]], output / "qpm_physical_operation_context.parquet")
    concat_parquet([filtered_legacy / "qpm_trace_quality.parquet", *[p/"quality.parquet" for p in paths]], output / "qpm_trace_quality.parquet")
    bar_dir = output / "qpm_failure_examples"
    bar_dir.mkdir(exist_ok=True)
    bars = [pd.read_csv(legacy / "qpm_failure_examples/dna_rna_barcode_counts.csv")]
    balances = [pd.read_csv(legacy / "qpm_failure_examples/dna_rna_balance_by_run.csv")]
    bars[0] = bars[0][~bars[0].observation_id.isin(replaced_legacy_obs)]
    balances[0] = balances[0][~balances[0].observation_id.isin(replaced_legacy_obs)]
    for r in selected:
        b = pd.read_parquet(output / ".reference_parse_cache" / r["archive_id"] / "barcodes.parquet")
        if not b.empty:
            bars.append(b)
        balances.append(pd.DataFrame([r["balance"]]))
    pd.concat(bars, ignore_index=True).to_csv(bar_dir / "dna_rna_barcode_counts.csv", index=False)
    pd.concat(balances, ignore_index=True).drop_duplicates("observation_id").to_csv(bar_dir / "dna_rna_balance_by_run.csv", index=False)
    for name in ["outcome_reference_policy.json", "qpm_waveform_reference_policy.json", "qpm_dropout_plugin_config.json"]:
        shutil.copy2(legacy / name, output / name)
    policy = json.loads((output / "outcome_reference_policy.json").read_text())
    policy["extra_match_columns"] = sorted(set(policy.get("extra_match_columns", []) + ["analysis_lane_scope"]))
    write_json(output / "outcome_reference_policy.json", policy)
    all_wave_aids = (old_wave_aids & retained_legacy_aids) | {r["archive_id"] for r in selected if r["operations"]}
    manifest_rows = transfers[transfers.archive_id.isin(all_wave_aids)].drop_duplicates("archive_id")
    signatures = {r.archive_id: [r.sha256, Path(r.local_path).stat().st_size, Path(r.local_path).stat().st_mtime_ns]
                  for r in manifest_rows.itertuples()}
    ctx = json.loads((legacy / "qpm_physical_operation_context_manifest.json").read_text())
    ctx["archives"] = [r for r in ctx["archives"] if r["archive_id"] in retained_legacy_aids]
    ctx.update(input_signatures=signatures, operations=pq.ParquetFile(output / "qpm_physical_operation_context.parquet").metadata.num_rows)
    ctx["archives"] += [dict(archive_id=r["archive_id"], sha256=r["sha256"], status="parsed", size_bytes=r["manifest"]["size_bytes"],
                             mtime_ns=Path(r["manifest"]["local_path"]).stat().st_mtime_ns, operations=r["context_operations"],
                             config_sha256=r["config_sha256"]) for r in selected if r["operations"]]
    write_json(output / "qpm_physical_operation_context_manifest.json", ctx)
    qc = json.loads((legacy / "qpm_trace_quality_manifest.json").read_text())
    qc["archives"] = [r for r in qc["archives"] if r["archive_id"] in retained_legacy_aids]
    qc.update(input_signatures=signatures, operations=pq.ParquetFile(output / "qpm_trace_quality.parquet").metadata.num_rows)
    qc["archives"] += [dict(archive_id=r["archive_id"], sha256=r["sha256"], status="parsed", operations=r["operations"],
                             **r["trace_quality_stats"]) for r in selected if r["operations"]]
    qc["er72_operations"] = sum(a.get("er72_operations", 0) for a in qc["archives"])
    write_json(output / "qpm_trace_quality_manifest.json", qc)
    # Provenance is kept for new files; inherited results retain the legacy
    # source project and its archive hashes rather than invented provenance.
    provenance = [{**p, "archive_id": r["archive_id"], "observation_id": r["row"]["observation_id"]}
                  for r in successful for p in r["outcome_provenance"]]
    pd.DataFrame(provenance).to_csv(output / "new_outcome_source_provenance.csv", index=False)
    coverage = core.groupby("instrument_id").agg(observations=("observation_id", "size"), physical_runs=("physical_run_id", "nunique"),
                 first_run=("run_start_time", "min"), last_run=("run_start_time", "max"), reads_available=("total_reads", "count"),
                 mean_length_available=("mean_read_length_bp", "count"), accuracy_available=("raw_read_accuracy_pct", "count")).reset_index()
    host_lookup = inventory.assign(instrument_id=inventory.hostname.map(lambda x: __import__('analyze_csa').stable_hash(x, "inst_"))).drop_duplicates("instrument_id")
    coverage = coverage.merge(host_lookup[["instrument_id", "hostname"]], on="instrument_id", how="left", validate="one_to_one")
    coverage.to_csv(output / "instrument_coverage.csv", index=False)
    log = dict(version=VERSION, completed_utc=utc_now(), source_collection=staging, legacy_project=str(legacy),
               legacy_observations=len(old_core), legacy_physical_runs=old_core.physical_run_id.nunique(),
               newly_parsed_unique_archives=sum(r["status"]=="parsed" for r in results), identity_cleared_unique_archives=len(successful),
               parse_failures=sum(r["status"] != "parsed" for r in results),
               byte_identical_to_legacy=int(receipts.sha256.isin(old_hashes).sum()),
               new_primary_observations=len(selected)-len(replaced_legacy_obs), more_complete_legacy_exports_replaced=len(replaced_legacy_obs),
               observations=len(core), physical_runs=core.physical_run_id.nunique(),
               instruments=core.instrument_id.nunique(), ambiguous_identity_archives=len(excluded),
               same_instrument_time_alias_keys=len(acquisition_aliases),
               waveform_operation_rows=pq.ParquetFile(output / "qpm_operation_waveforms.parquet").metadata.num_rows,
               trace_quality_operations=qc["operations"], er72_operations=qc["er72_operations"],
               no_model_retraining=True, source_collection_read_only=True, activation_status="built_pending_validation",
               code_sha256={p.name: sha256(p) for p in [Path(__file__), *[Path(__file__).parent/name for name in
                              ["analyze_csa.py", "train_multimodal_csa.py", "qpm_waveform_reference.py", "qpm_trace_quality.py", "outcome_reference.py", "analyze_dna_rna_barcode_balance.py"]]] if p.exists()},
               limitations=["Waveform references are empirical, not certified healthy.", "Cross-instrument references compare normalized shape, not uncalibrated raw amplitudes.",
                            "Source-folder-only identities are retained but explicitly audited.", "Run-header times are assumed UTC; missing run times never enter prior-only peers.",
                            "Historical ML and volume models have not been retrained.", "Combined-lane outcomes match only combined-lane peers."])
    write_json(output / "reference_bank_build_log.json", log)
    print(json.dumps(json_safe(log), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--legacy", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stage", choices=["parse", "finalize"], required=True)
    p.add_argument("--follow-staging", action="store_true")
    args = p.parse_args()
    if args.stage == "parse":
        parse_staged(args.output, args.legacy, args.workers, args.follow_staging)
    else:
        finalize(args.output, args.legacy)


if __name__ == "__main__":
    main()
