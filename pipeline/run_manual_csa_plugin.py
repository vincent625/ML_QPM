#!/usr/bin/env python3
"""Isolated local CSA-bundle replay. Never promote targets into the baseline bank."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
import re
import inspect
import io
import time
import gc

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VERSION = "manual-csa-plugin-replay-1.0.6-qualified-reference"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def unpack_inputs(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "inputs"
    dest.mkdir(exist_ok=True)
    rows = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.lower().endswith((".txz", ".zip", ".tgz", ".tar.xz", ".tar.gz")):
            continue
        outer_hash = digest(path)
        record = {"input_filename": path.name, "input_path": str(path), "input_sha256": outer_hash,
                  "input_size_bytes": path.stat().st_size, "version": VERSION}
        if not zipfile.is_zipfile(path):
            record.update(local_path=str(path), archive_sha256=outer_hash, container="tar", embedded_member=None)
        else:
            with zipfile.ZipFile(path) as z:
                invalid = z.testzip()
                if invalid:
                    raise ValueError("ZIP CRC failed for supplied archive")
                nested = [i for i in z.infolist() if i.filename.lower().endswith((".txz", ".tar.xz", ".tar.gz", ".tgz"))]
                if len(nested) > 1:
                    raise ValueError("Multiple embedded archives require an explicit selection")
                if not nested:
                    record.update(local_path=str(path), archive_sha256=outer_hash, container="zip", embedded_member=None)
                else:
                    info = nested[0]
                    target = dest / f"{outer_hash[:16]}.CSA.txz"
                    if not target.exists():
                        if shutil.disk_usage(dest).free < info.file_size*1.1:
                            raise ValueError("Insufficient local space to stage embedded CSA")
                        partial = target.with_suffix(target.suffix+".partial")
                        with z.open(info) as source, partial.open("xb") as sink:
                            shutil.copyfileobj(source, sink, length=8*1024*1024)
                        partial.rename(target)
                    inner_hash = digest(target)
                    record.update(local_path=str(target), archive_sha256=inner_hash, container="tar",
                                  embedded_member=info.filename, embedded_size_bytes=info.file_size)
        rows.append(record)
        print(f"Validated {path.name}; staged raw CSA {record['archive_sha256'][:12]}", flush=True)
    (out / "input_bundles.json").write_text(json.dumps({"version": VERSION, "executed_utc": datetime.now(timezone.utc).isoformat(), "archives": rows}, indent=2))
    return rows


def inspect_inputs(rows):
    wanted = {"CSA/ion_params_00.json", "CSA/analysisSamples.json", "CSA/planned_run.json",
              "CSA/outputs/BaseCallingActor-00/ionstats_basecaller.json", "CSA/outputs/BaseCallingActor-00/BaseCaller.json"}
    for r in rows:
        found = {}
        with tarfile.open(r["local_path"], "r|*") as t:
            for m in t:
                if m.name.lstrip("./") in wanted and m.isfile():
                    f = t.extractfile(m)
                    if f:
                        found[m.name.lstrip("./")] = json.load(f)
        print("\nINPUT", r["input_filename"], flush=True)
        p = found.get("CSA/ion_params_00.json", {})
        print("ion params safe metadata", {k:p.get(k) for k in ["expName", "chipType", "chipLane", "flows", "flowOrder", "analysisName"]})
        exp=p.get("exp_json", {}); exp=json.loads(exp) if isinstance(exp,str) else exp
        print("experiment metadata",{k:exp.get(k) for k in ["expName", "chipType", "flows"]})
        log=exp.get("log",{}); log=json.loads(log) if isinstance(log,str) else log
        print("instrument fields", {k:v for k,v in log.items() if any(x in k.lower() for x in ["hostname", "device_name", "devicename", "platform", "instrumentname"])})
        a=found.get("CSA/analysisSamples.json", {})
        print("barcode arrays", {k:len(a.get(k,[])) for k in ["barcodeIds","relations","mergedBarcodeIds","nucleicAcidTypes"]})
        planned=found.get("CSA/planned_run.json",{}).get("object",{})
        print("planned contexts", {k:planned.get(k) for k in ["runType","scriptName","requiredChipLanes","templatingSize","flows"]})
        print("deck lane contexts", [(i+1,d.get("chipLane"),len(d.get("sampleBarcodeMapping",[]))) for i,d in enumerate(planned.get("deckConfig",[]))])
        for key,d in found.items():
            if key.endswith("BaseCaller.json"):
                print("Final library reads",d.get("Filtering",{}).get("LibraryReport",{}).get("final_library_reads"))
            if key.endswith("ionstats_basecaller.json"):
                print("ionstats full",{k:d.get("full",{}).get(k) for k in ["num_reads","mean_read_length"]},"read groups",len(d.get("read_group_metrics",{})))


def root_params(path):
    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.name.lstrip("./") == "CSA/ion_params_00.json":
                return json.load(t.extractfile(m))
    raise ValueError("Root ion_params_00.json missing")


def parse_target(record, baseline, out):
    import analyze_csa as core
    import train_multimodal_csa as multi
    import qpm_waveform_reference as physical
    from analyze_dna_rna_barcode_balance import safe_barcode_plan, attach_read_counts, classify_barcodes
    from find_qpm_failure_examples import read_root_analysis_samples, normalize_barcode
    from outcome_reference import SOURCES, PANEL_COLUMNS

    params = root_params(record["local_path"])
    expname = params.get("expName", "")
    device = re.search(r"GNXS[-_](VC\d+|\d{4})", expname, re.I)
    if not device:
        raise ValueError("Instrument identity cannot be recovered safely")
    label = "gnxs-"+device.group(1).lower()
    hosts = pd.read_csv(baseline / "csa_inventory.csv").hostname.unique()
    matches = [h for h in hosts if h.split(".")[0].lower() == label]
    hostname = matches[0] if len(matches) == 1 else label
    lane_match = re.search(r"ChipLane(\d+(?:_\d+)*)", record["input_filename"], re.I)
    lane = lane_match.group(1) if lane_match else "lane_unknown"
    archive_id = record["archive_sha256"][:20]
    inv = {"archive_id": archive_id, "hostname": hostname, "ssh_user": "", "remote_path": record["local_path"],
           "run_name": re.sub(r"\.ChipLane.*$", "", record["input_filename"]),
           "physical_experiment_name": expname, "chip_lane": lane, "plugin_version": "local_bundle",
           "filename": record["input_filename"], "size_bytes": Path(record["local_path"]).stat().st_size,
           "remote_mtime_utc": datetime.fromtimestamp(Path(record["input_path"]).stat().st_mtime, timezone.utc).isoformat(),
           "match_reason": "explicit_user_supplied_local_CSA_bundle"}
    manifest = {"archive_id": archive_id, "hostname": hostname, "remote_path": record["local_path"],
                "local_path": record["local_path"], "sha256": record["archive_sha256"], "state": "already_present",
                "size_bytes": inv["size_bytes"], "container_format": "xz_tar", "validation_result": "bundle_CRC_and_embedded_content_validated"}
    print(f"Parsing {hostname}, lane scope {lane}", flush=True)
    parsed = core.parse_archive(inv, manifest)
    obs = parsed["observation"]
    fields = list(SOURCES.values())+PANEL_COLUMNS
    row = dict(obs)
    for name in fields:
        f = [x for x in parsed["features"] if x["feature_name"] == name]
        vals = [x["numeric_value"] if x["data_type"] == "numeric" else x["value"] for x in f]
        distinct = pd.Series(vals, dtype=object).dropna().astype(str).unique()
        row[name] = vals[0] if len(distinct) == 1 else None
    row["analysis_lane_scope"] = "combined_multilane" if "_" in lane else "single_lane"
    raw_warnings = len(parsed["parse_warnings"])
    pd.DataFrame(parsed["vacuum"]).to_csv(out / f"{archive_id}_vacuum_step_features.csv", index=False)
    pd.DataFrame(parsed["pipette"]).to_csv(out / f"{archive_id}_measured_pressure_features.csv", index=False)
    # Keep exact provenance for the selected canonical outcomes; no sample names.
    pd.DataFrame([r for r in parsed["outcome_provenance"] if r.get("selected")]).to_csv(out / f"{archive_id}_canonical_outcome_provenance.csv", index=False)
    del parsed
    gc.collect()
    plan_document = read_root_analysis_samples(record["local_path"])
    plan = pd.DataFrame(safe_barcode_plan(archive_id, plan_document))
    del plan_document
    base = {k: obs[k] for k in ["observation_id","physical_run_id","instrument_id","archive_id","chip_lane","run_start_time"]}
    qpm_rows, context_rows, tables, stats = [], [], {}, None
    with tarfile.open(record["local_path"], "r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            clean = m.name.lstrip("./")
            basename = Path(clean).name
            if clean == "CSA/debug":
                # Parse the existing waveform and command parsers from the same
                # bytes, transiently in memory. Never write raw debug text.
                if m.size > 1_000_000_000:
                    raise ValueError("Raw debug member exceeds local replay size limit")
                raw = t.extractfile(m).read()
                qpm_rows = multi.parse_qpm_stream(io.BytesIO(raw), base)
                context_rows = physical.parse_command_context(io.BytesIO(raw))
                del raw
            elif basename in physical.TABLES and m.size <= 2_000_000:
                tables[basename] = json.load(t.extractfile(m))
            elif clean.lower() == "csa/outputs/basecallingactor-00/ionstats_basecaller.json":
                stats = json.load(t.extractfile(m))
    maps = physical.definition_maps(tables)
    for r in context_rows:
        r["archive_id"] = archive_id
        for kind, prefix, key in [("lc","liquid_class","command_liquid_class_index"),
                                  ("cg","container_geometry","command_container_geometry_index"),
                                  ("dg","deck_geometry","command_deck_geometry_index")]:
            entry = maps[kind].get(r.get(key), {})
            r[f"{prefix}_fingerprint"] = entry.get("fingerprint")
            r[f"{prefix}_name"] = entry.get("name")
            r[f"{prefix}_source"] = entry.get("source_file")
            if kind == "cg":
                r["container_model"] = entry.get("definition",{}).get("tubeModel")
        if r["command_context_status"] == "command_verified" and any(not r.get(f"{p}_fingerprint") for p in ["liquid_class","container_geometry","deck_geometry"]):
            r["command_context_status"] = "missing_or_conflicting_configuration_definition"
    counts = {}
    for key, value in (stats or {}).get("read_group_metrics", {}).items():
        match = re.search(r"(?i)(ion[_-]?(?:hd)?dual[_-]?\d{4})", key)
        if match and isinstance(value, dict):
            token = normalize_barcode(match.group(1))
            n = value.get("read_count")
            if n is not None:
                counts[token] = counts.get(token, 0)+float(n)
    required_tokens = {t for v in plan.barcode_tokens for t in v}
    for token in required_tokens:
        counts.setdefault(token, np.nan)
    barcode_table = pd.DataFrame([{"archive_id": archive_id, **counts}]).set_index("archive_id")
    barcodes = attach_read_counts(pd.DataFrame([row]), plan, barcode_table, pd.DataFrame([inv]))
    barcodes, mat = classify_barcodes(barcodes)
    balance = {"observation_id": obs["observation_id"]}
    for material in ["dna", "rna"]:
        g = barcodes[(~barcodes.is_control) & barcodes.material_type.eq(material.upper())]
        balance[f"{material}_n_planned_noncontrol_barcodes"] = len(g)
        balance[f"{material}_median_reads"] = g.read_count.median() if len(g) else np.nan
        balance[f"{material}_apparent_dropout_count"] = int(g.apparent_dropout.sum())
    # Read the bundle's repeated run-QC values only when all explicit Raw Read
    # Accuracy entries agree. Raw mean length never substitutes AQ20/sample QC.
    accuracies = []
    with zipfile.ZipFile(record["input_path"]) as z:
        for name in z.namelist():
            if name.endswith("/Info.csv"):
                for candidate in core.parse_info_csv(z.read(name), "bundle_Info.csv"):
                    if candidate["outcome"] == "raw_read_accuracy_pct":
                        accuracies.append(candidate["value"])
    accuracy_values = sorted(set(accuracies))
    if len(accuracy_values) == 1:
        row["raw_read_accuracy_pct"] = accuracy_values[0]
        row["raw_read_accuracy_pct__source_file"] = "outer_bundle_Info.csv_repeated_run_QC_consensus"
        row["raw_read_accuracy_pct__source_field"] = "Run QC Evaluation Metrics / Raw Read Accuracy"
    metadata = {**record, "observation_id": obs["observation_id"], "physical_run_id": obs["physical_run_id"],
                "hostname": hostname, "chip_lane": lane, "analysis_scope": row["analysis_lane_scope"],
                "parse_warning_count": raw_warnings, "qpm_operations": len(qpm_rows),
                "qpm_waveforms": sum(r.get("qpm_available",False) for r in qpm_rows),
                "context_operations": len(context_rows), "barcode_rows": len(barcodes),
                "noncontrol_barcode_read_counts_known": int(barcodes.loc[~barcodes.is_control,"read_count"].notna().sum()),
                "explicit_accuracy_values": accuracy_values,
                "combined_lane_warning": "Combined four-lane outcomes and barcode counts; lane-specific failure localization is not established." if "_" in lane else None}
    joblib.dump({"row":row,"inventory":inv,"manifest":manifest,"qpm":pd.DataFrame(qpm_rows),
                 "context":pd.DataFrame(context_rows),"barcodes":barcodes,"balance":balance,"metadata":metadata}, out / f"{archive_id}_parsed.joblib", compress=3)
    print(f"Parsed {hostname}: {len(barcodes)} barcodes; {metadata['qpm_waveforms']}/{len(qpm_rows)} QPM waves", flush=True)
    return archive_id


def manual_target_physical_runs(core, records):
    """Remove renamed exports too: raw IDs, exact archives or acquisition time."""
    excluded={r["row"]["physical_run_id"] for r in records}
    ids={r["row"]["archive_id"] for r in records}
    excluded.update(core.loc[core.archive_id.isin(ids),"physical_run_id"])
    times=pd.to_datetime(core.run_start_time,utc=True,errors="coerce",format="mixed")
    for r in records:
        time=pd.to_datetime(r["row"].get("run_start_time"),utc=True,errors="coerce")
        if pd.notna(time):
            mask=core.instrument_id.eq(r["row"]["instrument_id"])&times.eq(time)
            excluded.update(core.loc[mask,"physical_run_id"])
    return excluded


def prepare_cases(baseline, out):
    from qpm_reference_bank import resolve_reference_bank
    baseline = resolve_reference_bank(baseline)
    import qpm_waveform_reference as physical
    from outcome_reference import SOURCES, PANEL_COLUMNS, normalize
    from analyze_csa import stable_hash
    rows = json.loads((out / "input_bundles.json").read_text())["archives"]
    ids=[]
    for r in rows:
        aid=r["archive_sha256"][:20]
        if not (out / f"{aid}_parsed.joblib").exists():
            parse_target(r,baseline,out)
        ids.append(aid)
    meta_fields=["observation_id","physical_run_id","instrument_id","archive_id","chip_lane","run_start_time", "total_reads","mean_read_length_bp","raw_read_accuracy_pct",*SOURCES.values(),*PANEL_COLUMNS]
    if "analysis_lane_scope" in pq.read_schema(baseline / "csa_ml_dataset.parquet").names:
        meta_fields.append("analysis_lane_scope")
    core=pd.read_parquet(baseline / "csa_ml_dataset.parquet",columns=meta_fields)
    if "analysis_lane_scope" not in core:
        core["analysis_lane_scope"]="single_lane"
    wave_fields=["observation_id","physical_run_id","instrument_id","archive_id","chip_lane","run_start_time",
                 "operation_id","operation_index","operation","pipette","programmed_volume_ul","event_time",
                 "workflow_stage","process_step","source_file","pressure_unit","liquid_class_index","loc",
                 "qpm_available","qpm_profile","container_geometry_index","deck_geometry_index"]
    waves=pd.read_parquet(baseline / "qpm_operation_waveforms.parquet",columns=wave_fields)
    context=pd.read_parquet(baseline / "qpm_physical_operation_context.parquet")
    context_meta=json.loads((baseline / "qpm_physical_operation_context_manifest.json").read_text())
    inv=pd.read_csv(baseline / "csa_inventory.csv",dtype={"archive_id":str})
    transfers=pd.read_csv(baseline / "csa_transfer_manifest.csv",dtype={"archive_id":str})
    bars=pd.read_csv(baseline / "qpm_failure_examples/dna_rna_barcode_counts.csv")
    balance=pd.read_csv(baseline / "qpm_failure_examples/dna_rna_balance_by_run.csv")
    parser_hash=physical.digest([physical.CONTEXT_PARSER_VERSION,str(physical.OP_RE.pattern),str(physical.SEND_RE.pattern),physical.TABLES,
                  inspect.getsource(physical.parse_command_context),inspect.getsource(physical.definition_maps),inspect.getsource(physical.extract_archive)])
    if context_meta["parser_sha256"] != parser_hash:
        raise ValueError("Baseline context cache does not match current parser; rebuild must be separately authorized")
    case_rows=[]
    case_records={aid:joblib.load(out / f"{aid}_parsed.joblib") for aid in ids}
    manual_physical_runs=manual_target_physical_runs(core,list(case_records.values()))
    for aid in ids:
        r=case_records[aid]
        # Normalize the filename's exact lane token. A separator before Final
        # is not an additional lane. This only repairs local ingest metadata.
        lane=re.search(r"ChipLane(\d+(?:_\d+)*)",r["metadata"]["input_filename"],re.I).group(1)
        corrected_obs=stable_hash(f"{r['row']['physical_run_id']}|{lane}","obs_")
        if r["row"]["chip_lane"] != lane:
            previous_id=r["row"]["observation_id"]
            for name in ["row","metadata"]:
                r[name]["observation_id"]=corrected_obs
                r[name]["chip_lane"]=lane
            for name in ["qpm","barcodes"]:
                r[name]["observation_id"]=corrected_obs
                r[name]["chip_lane"]=lane
            r["inventory"]["chip_lane"]=lane
            r["balance"]["observation_id"]=corrected_obs
            r["row"]["analysis_lane_scope"]="combined_multilane" if "_" in lane else "single_lane"
            r["metadata"]["analysis_scope"]=r["row"]["analysis_lane_scope"]
            r["metadata"]["combined_lane_warning"]=("Combined four-lane outcomes and barcode counts; lane-specific failure localization is not established." if "_" in lane else None)
            r["metadata"]["ingestion_corrections"]=["Removed filename separator from lane token before any plugin diagnostic run."]
            for tail in ["vacuum_step_features.csv","measured_pressure_features.csv","canonical_outcome_provenance.csv"]:
                path=out / f"{aid}_{tail}"
                frame=pd.read_csv(path)
                if "observation_id" in frame:
                    frame["observation_id"]=corrected_obs
                frame.to_csv(path,index=False)
            joblib.dump(r,out / f"{aid}_parsed.joblib",compress=3)
        case=out / r["row"]["observation_id"]
        project=case / "project"
        project.mkdir(parents=True,exist_ok=True)
        (project / "qpm_failure_examples").mkdir(exist_ok=True)
        # Reuse the audited bank status; only this target needs an incremental
        # debug scan. Targets still never enter the shared reference archive bank.
        for name in ["qpm_trace_quality.parquet", "qpm_trace_quality_manifest.json"]:
            if (baseline / name).exists():
                shutil.copy2(baseline / name, project / name)
        for name in ["outcome_reference_policy.json","qpm_waveform_reference_policy.json","qpm_dropout_plugin_config.json"]:
            shutil.copy2(baseline/name,project/name)
        qualification_policy = baseline / "reference_qualification_policy.json"
        if qualification_policy.is_file():
            shutil.copy2(qualification_policy, project / qualification_policy.name)
        volume_registry = baseline / "qpm_nominal_volume_model.json"
        if volume_registry.is_file():
            shutil.copy2(volume_registry, project / volume_registry.name)
        new_row=dict(r["row"])
        for column in [*SOURCES.values(),*PANEL_COLUMNS]:
            if isinstance(new_row.get(column),(bool,np.bool_)):
                # The established core parser serializes booleans as title-case
                # strings. Preserve that representation for exact peer matching.
                new_row[column]=str(bool(new_row[column]))
        # A newly supplied CSA can already occur in the expanded bank. Remove
        # ALL its physical-run lanes/reanalyses before appending the target;
        # manual targets must never become their own (or sibling) references.
        target_physical = manual_physical_runs
        case_core = core[~core.physical_run_id.isin(target_physical)].copy()
        case_waves = waves[~waves.physical_run_id.isin(target_physical)].copy()
        case_archive_ids = set(case_waves.archive_id)
        case_context = context[context.archive_id.isin(case_archive_ids)].copy()
        case_bars = bars[~bars.physical_run_id.isin(target_physical)].copy()
        case_balance = balance[~balance.observation_id.isin(set(core.loc[core.physical_run_id.isin(target_physical), "observation_id"]))].copy()
        combined_core=pd.concat([case_core,pd.DataFrame([new_row])],ignore_index=True)
        # The reference normalizer already maps booleans/numbers consistently;
        # apply it before Arrow sees mixed older-string/new-boolean contexts.
        for column in [*SOURCES.values(),*PANEL_COLUMNS]:
            combined_core[column]=combined_core[column].map(normalize)
        combined_core.to_parquet(project / "csa_ml_dataset.parquet",index=False)
        pd.concat([inv,pd.DataFrame([r["inventory"]])],ignore_index=True).to_csv(project / "csa_inventory.csv",index=False)
        merged_transfers=pd.concat([transfers,pd.DataFrame([r["manifest"]])],ignore_index=True)
        merged_transfers.to_csv(project / "csa_transfer_manifest.csv",index=False)
        pd.concat([case_bars,r["barcodes"]],ignore_index=True).to_csv(project / "qpm_failure_examples/dna_rna_barcode_counts.csv",index=False)
        pd.concat([case_balance,pd.DataFrame([r["balance"]])],ignore_index=True).to_csv(project / "qpm_failure_examples/dna_rna_balance_by_run.csv",index=False)
        pd.concat([case_waves,r["qpm"].reindex(columns=wave_fields)],ignore_index=True).to_parquet(project / "qpm_operation_waveforms.parquet",index=False)
        pd.concat([case_context,r["context"]],ignore_index=True).to_parquet(project / "qpm_physical_operation_context.parquet",index=False)
        ids_in_waves=set(case_waves.archive_id)|{aid}
        selected=merged_transfers[merged_transfers.archive_id.isin(ids_in_waves)].drop_duplicates("archive_id")
        cache=json.loads(json.dumps(context_meta))
        # Bank signatures were checked at validation; the normal cache loader
        # rechecks all actual files (in parallel) when scoring starts.
        def signature(a):
            old=context_meta["input_signatures"].get(a.archive_id)
            if a.archive_id != aid and old and old[0]==a.sha256:
                return old
            info=Path(a.local_path).stat()
            return [a.sha256,info.st_size,info.st_mtime_ns]
        cache["input_signatures"]={a.archive_id:signature(a) for a in selected.itertuples()}
        cache["archives"]=[entry for entry in cache["archives"] if entry["archive_id"] in case_archive_ids and entry["archive_id"] != aid]
        cache["archives"].append({"archive_id":aid,"sha256":r["manifest"]["sha256"],"size_bytes":r["inventory"]["size_bytes"],
                                  "mtime_ns":Path(r["manifest"]["local_path"]).stat().st_mtime_ns,"status":"parsed","operations":len(r["context"])})
        cache["operations"]=len(case_context)+len(r["context"])
        (project / "qpm_physical_operation_context_manifest.json").write_text(json.dumps(cache,indent=2))
        cfg=json.loads((project / "qpm_dropout_plugin_config.json").read_text())
        if r["row"]["analysis_lane_scope"] == "combined_multilane":
            cfg.setdefault("outcome_reference_policy_overrides",{})["extra_match_columns"]=["analysis_lane_scope"]
        (case / "case_config.json").write_text(json.dumps(cfg,indent=2))
        r["metadata"].update(project=str(project),report_dir=str(case / "report"),config=str(case / "case_config.json"),
                             reference_bank_target_count=1, reference_bank_observations=len(case_core),
                             scope_guard="Each isolated view excludes all manually supplied physical runs, including renamed exports matched by instrument/acquisition time, and appends ONLY its own target.")
        (case / "input_audit.json").write_text(json.dumps(r["metadata"],indent=2))
        from qpm_reference_bank import reference_bank_info
        origin = reference_bank_info(baseline)
        origin.update(isolated_target_observation_id=r["row"]["observation_id"],
                      target_physical_run_removed_from_bank=True, other_manual_targets_excluded=True)
        (project / "reference_bank_origin.json").write_text(json.dumps(origin,indent=2))
        case_rows.append(r["metadata"])
        print(f"Isolated project ready for {r['metadata']['observation_id']}",flush=True)
    (out / "cases.json").write_text(json.dumps(case_rows,indent=2))


def report_cache_matches(log, row, config, report):
    """A cached pre-update report must not hide newly required comparators."""
    from qpm_dropout_plugin import VERSION as PLUGIN_VERSION
    comparison = log.get("baseline_comparison", {})
    if (log.get("plugin_version") != PLUGIN_VERSION
            or log.get("archive_sha256") != row["archive_sha256"]
            or log.get("config") != config
            or not comparison.get("always_evaluated_independently")
            or {x.get("baseline_id") for x in comparison.get("summary", [])}
               != {"within_run", "same_instrument", "cross_instrument"}):
        return False
    hashes = log.get("code_sha256", {})
    if not {"qpm_baseline_comparison.py", "qpm_trace_quality.py", "qpm_reference_qualification.py", "qpm_run_metadata.py",
            "csa_low_performance_diagnostics.py", "qpm_same_run_curves.py"} <= set(hashes):
        return False
    code_root = Path(__file__).resolve().parent
    if any(not (code_root / name).is_file() or digest(code_root / name) != value
           for name, value in hashes.items()):
        return False
    # A helper may be imported lazily. Conservatively replay a report if any
    # protected source changed during its run, even when its final file hash
    # happens to match the currently installed source.
    try:
        generated = pd.Timestamp(log["generated_utc"])
        started = generated.timestamp() - float(log["runtime_seconds"])
        if any((code_root / name).stat().st_mtime > started for name in hashes):
            return False
    except (ValueError, KeyError, TypeError):
        return False
    for filename in ["qpm_baseline_comparison_summary.csv", "qpm_baseline_comparison_manifest.json",
                     "qpm_baseline_comparison.png", "qpm_baseline_comparison.pdf", "qpm_baseline_comparison.svg",
                     "qpm_dropout_report.html", "qpm_excluded_traces.csv",
                     "qpm_trace_quality_manifest.json", "qpm_displayed_evidence.csv", "run_metadata.json",
                     "csa_low_performance_manifest.json", "diagnostic_workflow.json",
                     "qpm_same_run_curve_manifest.json", "qpm_same_run_curves.html",
                     "qpm_same_run_curve_fragment.html", "qpm_same_run_curves.csv", "qpm_same_run_curve_coverage.csv"]:
        if not (report / filename).is_file():
            return False
    from csa_low_performance_diagnostics import cache_artifacts_valid
    if not cache_artifacts_valid(report / "csa_low_performance_manifest.json"):
        return False
    from qpm_same_run_curves import cache_artifacts_valid as same_run_cache_valid
    if not same_run_cache_valid(report / "qpm_same_run_curve_manifest.json"):
        return False
    if not all((report / name).is_file() for name in ["reference_qualification_manifest.json", "historical_reference_qualification.csv"]):
        return False
    inputs = log.get("outcome_reference", {}).get("input_and_code_sha256", {})
    project = Path(row["project"])
    from qpm_reference_qualification import load_reference_qualification
    _, qualification = load_reference_qualification(project)
    if log.get("reference_qualification", {}).get("policy_sha256") != qualification["policy_sha256"]:
        return False
    volume_registry = project / "qpm_nominal_volume_model.json"
    recorded_registry = log.get("volume_proxy", {}).get("registry_sha256")
    if volume_registry.is_file() and recorded_registry != digest(volume_registry):
        return False
    if recorded_registry and not volume_registry.is_file():
        return False
    for filename, path in [("csa_ml_dataset.parquet", project / "csa_ml_dataset.parquet"),
                           ("dna_rna_balance_by_run.csv", project / "qpm_failure_examples/dna_rna_balance_by_run.csv"),
                           ("dna_rna_barcode_counts.csv", project / "qpm_failure_examples/dna_rna_barcode_counts.csv")]:
        if inputs.get(filename) != digest(path):
            return False
    return True


def run_reports(out):
    from qpm_dropout_plugin import run_plugin,load_config
    rows=json.loads((out / "cases.json").read_text())
    logs=[]
    for row in rows:
        report=Path(row["report_dir"])
        config=load_config(Path(row["config"]))
        log=None
        if (report / "plugin_run_log.json").exists():
            candidate=json.loads((report / "plugin_run_log.json").read_text())
            if report_cache_matches(candidate,row,config,report):
                log=candidate
            else:
                print(f"Refreshing stale plugin report for {row['hostname']}",flush=True)
        if log is None:
            print(f"Running current plugin for {row['hostname']} / {row['observation_id']}",flush=True)
            log=run_plugin(Path(row["project"]),row["observation_id"],report,config)
        logs.append({"observation_id":row["observation_id"],"report":log["outputs"]["report"],"branch":log["report_branch"],"warnings":log["warnings"]})
        print(f"Report complete: {log['report_branch']}",flush=True)
    (out / "replay_report_index.json").write_text(json.dumps(logs,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--stage",choices=["inspect","prepare","reports","all"],default="inspect")
    p.add_argument("--baseline",type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    if not (args.out / "input_bundles.json").exists():
        unpack_inputs(args.input_dir,args.out)
    if args.stage == "inspect":
        inspect_inputs(json.loads((args.out / "input_bundles.json").read_text())["archives"])
    if args.stage in {"prepare","all"}:
        prepare_cases(args.baseline,args.out)
    if args.stage in {"reports","all"}:
        run_reports(args.out)


if __name__ == "__main__":
    main()
