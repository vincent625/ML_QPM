#!/usr/bin/env python3
"""Resumable QPM diagnostics for every scope of the flagged physical runs.

The plugin's exact three-comparator analysis is retained. A conservative
physical-key prefilter removes only rows that cannot match ANY target operation;
all target rows remain, including unknown configurations. Full matching, target
run exclusion, trace quality, replication and significance guards run unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa

from qpm_reference_bank import resolve_reference_bank, file_sha256, MANIFEST
from qpm_run_metadata import clean_text
from qpm_cohort_publication import reconcile_historical_label

VERSION = "flagged-qpm-cohort-batch-1.6.0-low-read-hypotheses"
CANDIDATE_KEYS = ["pipette", "operation", "programmed_volume_ul", "pressure_unit",
                  "liquid_class_fingerprint", "container_geometry_fingerprint",
                  "deck_geometry_fingerprint", "command_settings_fingerprint"]
_INDEX = None
_BANK = None
_QUALIFIED_ONLY = False


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(safe(payload), indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


class CandidateIndex:
    """Immutable input index. Hash collisions only admit extra candidate rows."""
    def __init__(self, profiles, *, qualified_references_only=False):
        self.profiles = profiles
        self.qualified_references_only = qualified_references_only
        frame = profiles[0]
        self.hashes = pd.util.hash_pandas_object(frame[CANDIDATE_KEYS], index=False).to_numpy()
        self.targets = frame.groupby("observation_id", sort=False).indices
        if qualified_references_only:
            if "reference_eligible" not in frame:
                raise ValueError("Outcome-qualified prefilter requires verified historical eligibility")
            positions = np.flatnonzero(frame.reference_eligible.eq(True) & frame.command_context_status.eq("command_verified"))
        else:
            positions = np.arange(len(frame))
        groups = pd.DataFrame({"key": self.hashes[positions]}).groupby("key", sort=False).indices
        self.candidates = {key: positions[indices] for key, indices in groups.items()}

    def view(self, observation_id):
        frame, metadata = self.profiles
        targets = self.targets.get(observation_id, np.array([], dtype=int))
        if len(targets):
            parts = [targets] + [self.candidates.get(key, np.array([], dtype=int)) for key in np.unique(self.hashes[targets])]
            positions = np.unique(np.concatenate(parts))
        else:
            # An empty target must still produce an unresolved report; its
            # trace exclusions and full bank provenance remain available.
            positions = np.array([], dtype=int)
        result = frame.iloc[positions].copy()
        note = dict(method="exact_physical_key_superset", keys=CANDIDATE_KEYS,
                    original_profile_rows=len(frame), candidate_profile_rows=len(result),
                    target_profile_rows=len(targets), all_target_rows_preserved=True,
                    historically_qualified_candidates_only=self.qualified_references_only,
                    full_reference_matching_and_exclusions_unchanged=True,
                    driver_sha256=file_sha256(Path(__file__)))
        return result, {**metadata, "batch_candidate_prefilter": note}


def task_roster(screen, bank, output):
    runs = pd.read_csv(screen / "flagged_runs.csv")
    observations = pd.read_csv(screen / "all_observations.csv")
    scopes = observations.loc[observations.physical_run_id.isin(runs.physical_run_id)].copy()
    if scopes.observation_id.duplicated().any() or scopes.physical_run_id.nunique() != len(runs):
        raise ValueError("Flagged physical-run/scope roster is not one-to-one reconciled")
    core = pd.read_parquet(bank / "csa_ml_dataset.parquet", columns=["observation_id", "physical_run_id", "archive_sha256"])
    known = core.set_index("observation_id")
    manual = pd.read_csv(screen / "manual_CSA_reconciliation.csv").set_index("observation_id")
    rows = []
    for row in scopes.to_dict("records"):
        obs = row["observation_id"]
        if obs in known.index:
            if known.loc[obs, "physical_run_id"] != row["physical_run_id"] or known.loc[obs, "archive_sha256"] != row["archive_sha256"]:
                raise ValueError("Screen and bank target identity/hash disagree")
            row["project"] = str(bank)
        elif obs in manual.index:
            row["project"] = str(manual.loc[obs, "project"])
        else:
            raise ValueError("Screen target has no validated parsed project")
        row["report_dir"] = str(output / "reports" / obs)
        rows.append(safe(row))
    # Barcode failures first, then low reads, with unflagged siblings retained.
    rows.sort(key=lambda r: (not r["has_dropout"], not r["has_imbalance"], not r["flagged"], r["run_start_time"] or ""))
    return runs, rows


def assert_full_scan_equivalence(report, canonical):
    for name in ["barcode_dropout_calls.csv", "qpm_baseline_comparison_summary.csv", "qpm_displayed_evidence.csv", "qpm_step_evidence.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(report/name), pd.read_csv(canonical/name), check_dtype=False, rtol=1e-10, atol=1e-10)
    for mode in ["same_instrument", "cross_instrument"]:
        name = f"qpm_baseline_{mode}_waveforms.parquet"
        a = pd.read_parquet(report/name).sort_values("operation_id").reset_index(drop=True)
        b = pd.read_parquet(canonical/name).sort_values("operation_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-10, atol=1e-10)
    return dict(status="passed", canonical_report=str(canonical), full_scores_calls_and_evidence_identical=True)


def completed_report_cache_valid(output):
    """A completed QPM task also needs fresh, explicit CSA diagnostic coverage.

    Disabled and not-triggered analyses still write a manifest. Their absence
    must not silently reuse an older QPM-only report, nor may changed external
    diagnostic tables survive a code-only cache check.
    """
    from csa_low_performance_diagnostics import cache_artifacts_valid
    from qpm_same_run_curves import cache_artifacts_valid as same_run_cache_valid
    output = Path(output)
    try:
        log_path = output / "plugin_run_log.json"
        if (not log_path.is_file() or not (output / "qpm_dropout_report.html").is_file()
                or not (output / "diagnostic_workflow.json").is_file()):
            return False
        log = json.loads(log_path.read_text())
        hashes = log.get("code_sha256", {})
        if not {"csa_low_performance_diagnostics.py", "qpm_same_run_curves.py"} <= set(hashes):
            return False
        if not all((Path(__file__).parent / name).is_file() and
                   file_sha256(Path(__file__).parent / name) == sha
                   for name, sha in hashes.items()):
            return False
        return (cache_artifacts_valid(output / "csa_low_performance_manifest.json")
                and same_run_cache_valid(output / "qpm_same_run_curve_manifest.json"))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def csa_scope_fields(diagnostics):
    """Keep CSA clues separate from QPM localization and its evidence labels."""
    metrics = diagnostics.get("metrics", [])
    available = [metric for metric in metrics if metric.get("status") == "available"]
    abnormal = [metric for metric in available if metric.get("deviation") not in
                {None, "", "within_empirical_range"}]
    root_cause = diagnostics.get("root_cause_assessment", {})
    return dict(
        csa_diagnostic_status=diagnostics.get("status", "not_recorded"),
        csa_diagnostic_triggered=diagnostics.get("triggered"),
        csa_diagnostic_summary=diagnostics.get("summary", "CSA diagnostic section was not recorded in this historical report; unresolved."),
        csa_assessed_metrics=len(metrics), csa_available_metrics=len(available),
        csa_abnormal_metrics=len(abnormal),
        csa_abnormal_metric_names=" | ".join(str(m.get("label", m.get("feature", ""))) for m in abnormal),
        csa_abnormal_metric_findings=" | ".join(
            f"{m.get('label', m.get('feature', ''))}: {m['deviation']}" for m in abnormal),
        csa_root_cause_status=root_cause.get("status", "not_recorded"),
        csa_root_cause_hypothesis_count=len(root_cause.get("hypotheses", [])),
        csa_diagnostic_manifest=diagnostics.get("manifest_path", ""))


def diagnostic_plan_inputs(tasks, config):
    """Pin external diagnostic tables for every distinct bank/isolated view."""
    from csa_low_performance_diagnostics import diagnostic_input_fingerprints
    return {project: diagnostic_input_fingerprints(Path(project), config)
            for project in sorted({task["project"] for task in tasks})}


def execute_task(task, config, fingerprint, audit_directory=None):
    from qpm_dropout_plugin import run_plugin, json_ready
    output = Path(task["report_dir"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completion = output / "batch_completion.json"
    if completion.is_file():
        result = json.loads(completion.read_text())
        if result.get("batch_fingerprint") == fingerprint and result.get("status") == "completed":
            if completed_report_cache_valid(output):
                return {**result, "cached": True}
    try:
        project = Path(task["project"])
        if project == _BANK:
            profiles = _INDEX.view(task["observation_id"])
        else:
            from qpm_waveform_reference import load_profiles
            profiles = CandidateIndex(load_profiles(project), qualified_references_only=_QUALIFIED_ONLY).view(task["observation_id"])
        log = run_plugin(project, task["observation_id"], output, config, preloaded_profiles=profiles)
        equivalence = None
        if audit_directory:
            canonical = Path(audit_directory) / "reports" / task["observation_id"]
            if (canonical / "plugin_run_log.json").is_file():
                equivalence = assert_full_scan_equivalence(output, canonical)
        result = dict(status="completed", observation_id=task["observation_id"],
                      physical_run_id=task["physical_run_id"], report=log["outputs"]["report"],
                      batch_fingerprint=fingerprint, runtime_seconds=round(time.time()-started, 2),
                      plugin_version=log["plugin_version"], archive_sha256=log["archive_sha256"],
                      full_scan_equivalence=equivalence,
                      generated_utc=now(), cached=False)
    except Exception as exc:
        result = dict(status="failed", observation_id=task["observation_id"],
                      physical_run_id=task["physical_run_id"], batch_fingerprint=fingerprint,
                      error_type=type(exc).__name__, error=str(exc)[:1200],
                      runtime_seconds=round(time.time()-started, 2), generated_utc=now())
        (output / "batch_error.txt").write_text(traceback.format_exc())
    write_json(completion, result)
    return result


def scope_result(task, result):
    row = {k: task.get(k) for k in ["observation_id", "physical_run_id", "run_start_time", "instrument", "instrument_id",
           "run_name", "assay_type", "assay_panel_label", "chip_type", "chip_lane", "total_reads_value", "total_reads_threshold",
           "has_dropout", "has_imbalance", "has_low_read_scope", "flagged"]}
    row.update(execution_status=result.get("status", "pending"),
               report=result.get("report", ""), error=result.get("error", ""),
               diagnostic_status="pending", earliest_step="", earliest_operation_index=None,
               evidence_basis="", causal_interpretation="Not a proven root cause",
               csa_diagnostic_status="pending", csa_diagnostic_triggered=None,
               csa_diagnostic_summary="", csa_assessed_metrics=None, csa_available_metrics=None,
               csa_abnormal_metrics=None, csa_abnormal_metric_names="", csa_abnormal_metric_findings="",
               csa_diagnostic_manifest="")
    if row["execution_status"] == "failed":
        row["diagnostic_status"] = "processing_failed"
        row["csa_diagnostic_status"] = "processing_failed"
        return row
    if row["execution_status"] != "completed":
        return row
    report = Path(task["report_dir"])
    log = json.loads((report / "plugin_run_log.json").read_text())
    conclusion = json.loads((report / "initial_failure_candidates.json").read_text())
    row.update(csa_scope_fields(conclusion.get("csa_low_performance_diagnostics",
                                             log.get("csa_low_performance_diagnostics", {}))))
    for key in ["run_date", "run_start_time", "instrument", "assay_type", "chip_type"]:
        row[key] = log["run_metadata"][key]
    row["target_eligible_qpm_operations"] = log["trace_quality"]["target_eligible_waveforms"]
    row["target_er72_excluded_waveforms"] = log["trace_quality"]["target_er72_excluded_waveforms"]
    row["report_branch"] = log["report_branch"]
    row["plugin_version"] = log["plugin_version"]
    candidates, path_candidates = [], []
    for material, finding in conclusion["materials"].items():
        initial = finding.get("initial_candidate")
        if initial:
            candidates.append({**initial, "material": material})
        fallback = finding.get("reference_fallback", {})
        for initial in fallback.get("earliest_path_anomalies", []):
            path_candidates.append({**initial, "material": material})
    summaries = pd.read_csv(report / "qpm_baseline_comparison_summary.csv")
    for baseline in ["within_run", "same_instrument", "cross_instrument"]:
        part = summaries.loc[summaries.baseline_id.eq(baseline)]
        if len(part) != 1:
            raise ValueError("Each completed scope must retain all three baseline summaries")
        values = part.iloc[0]
        for field in ["status", "scored_operations", "positive_operations", "earliest_operation_index", "reference_physical_runs", "reference_instruments"]:
            row[f"{baseline}_{field}"] = safe(values.get(field))
    shared = pd.read_csv(report / "qpm_shared_path_evidence.csv")
    positive = shared.loc[shared.evidence_role.isin(["earliest_reference_deviation", "later_reference_anomaly"])] if "evidence_role" in shared else shared.iloc[:0]
    if candidates:
        first = min(candidates, key=lambda c: c["operation_index"])
        row.update(diagnostic_status="within_run_selective_divergence", earliest_step=first["step"],
                   earliest_operation_index=first["operation_index"], evidence_basis="Replicated affected-versus-surviving paths",
                   material=first["material"], failed_path_deviation_auc=first["failed_path_deviation_auc"],
                   permutation_p=first["permutation_p"], bh_fdr_q=first["bh_fdr_q"])
    elif path_candidates:
        first = min(path_candidates, key=lambda c: c["operation_index"])
        row.update(diagnostic_status="barcode_path_reference_anomaly", earliest_step=first.get("step_label", first.get("step", "")),
                   earliest_operation_index=first["operation_index"], evidence_basis="Affected-path historical reference; insufficient internal replication",
                   material=first["material"])
    elif len(positive):
        first = positive.sort_values("operation_index").iloc[0]
        row.update(diagnostic_status="historical_reference_only_anomaly",
                   earliest_step=" / ".join(clean_text(first.get(k)) for k in ["workflow_stage", "process_step", "operation"]),
                   earliest_operation_index=int(first.operation_index), evidence_basis="Shared-path historical deviation; not barcode-selective")
    elif not row["target_eligible_qpm_operations"]:
        row["diagnostic_status"] = "no_usable_qpm_waveforms"
    elif not any((row.get(f"{b}_scored_operations") or 0) > 0 for b in ["within_run", "same_instrument", "cross_instrument"]):
        row["diagnostic_status"] = "insufficient_comparable_qpm_evidence"
    else:
        row["diagnostic_status"] = "no_qualifying_qpm_divergence"
    # Separate explorations in unflagged siblings from proposed explanations.
    if not row["flagged"]:
        row["evidence_basis"] = "Unflagged sibling scope (context only). " + row["evidence_basis"]
    displayed = pd.read_csv(report / "qpm_displayed_evidence.csv")
    row["displayed_evidence_panels"] = int(displayed.available.eq(True).sum())
    later = []
    for baseline, part in displayed.loc[displayed.available.eq(True)].groupby("baseline_id", sort=False):
        first = part.sort_values("slot").iloc[0]
        row[f"{baseline}_first_displayed_step"] = first["step"]
        row[f"{baseline}_first_displayed_evidence_role"] = first["evidence_role"]
        for record in part.loc[part.slot.gt(1)].sort_values("slot").to_dict("records"):
            later.append(f"{baseline}: {record['step']} (op {int(record['operation_index'])}; {record['evidence_role']})")
    row["later_displayed_evidence"] = " | ".join(later)
    return safe(reconcile_historical_label(row, shared, displayed))


def summarize(output, runs, tasks, results, plan):
    scopes = [scope_result(task, results.get(task["observation_id"], {})) for task in tasks]
    table = pd.DataFrame(scopes)
    table.to_csv(output / "scope_qpm_results.csv", index=False)
    run_rows, cards = [], []
    for original in runs.to_dict("records"):
        group = table.loc[table.physical_run_id.eq(original["physical_run_id"])]
        complete = group.execution_status.eq("completed")
        failed = group.execution_status.eq("failed")
        status = "completed" if complete.all() else "completed_with_errors" if (complete | failed).all() else "running"
        flagged = group.loc[group.flagged.eq(True)]
        findings = sorted(set(flagged.diagnostic_status) - {"pending"})
        selected = flagged.loc[flagged.earliest_operation_index.notna()].sort_values("earliest_operation_index")
        descriptions = [f"Lane(s) {r.chip_lane}: {r.earliest_step} [op {int(r.earliest_operation_index)}; {r.diagnostic_status}]"
                        for r in selected.itertuples()]
        # Operation indices belong to individual scope logs. They are not a
        # cross-lane physical clock; list per scope without claiming run-wide ordering.
        record = {k: original[k] for k in ["physical_run_id", "run_date", "instrument", "run_name", "assay_type", "assay_panel", "chip_type", "flags"]}
        record.update(execution_status=status, retained_scopes=len(group), completed_scopes=int(complete.sum()),
                      failed_scopes=int(failed.sum()), qpm_findings=" | ".join(findings),
                      potential_steps_by_scope=" | ".join(descriptions), no_causal_root_cause_established=True,
                      csa_diagnostic_findings=" | ".join(sorted(set(flagged.csa_diagnostic_status) - {"pending"})),
                      csa_abnormal_metrics=int(pd.to_numeric(flagged.csa_abnormal_metrics, errors="coerce").sum()),
                      csa_scopes_with_diagnostic_clues=int(flagged.csa_diagnostic_status.eq("diagnostic_clues_found").sum()),
                      csa_diagnostic_summaries=" | ".join(
                          f"Lane(s) {r.chip_lane}: {r.csa_diagnostic_summary}" for r in flagged.itertuples()
                          if r.csa_diagnostic_summary))
        run_rows.append(record)
        links = []
        for r in group.to_dict("records"):
            label = f"Lane(s) {r['chip_lane']}: {r['diagnostic_status']}"
            if r.get("report"):
                relative = os.path.relpath(r["report"], output)
                links.append(f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>')
            else:
                links.append(html.escape(label + (": " + r["error"] if r.get("error") else "")))
        details = "<br>".join(html.escape(d) for d in descriptions) or "No qualifying step available yet / unresolved."
        search = " ".join(str(v) for v in record.values()).lower()
        cells = [original["run_date"], original["instrument"], original["chip_type"], original["assay_panel"], original["flags"],
                 f"{int(complete.sum())}/{len(group)} scopes", " | ".join(findings), record["csa_diagnostic_findings"]]
        cells_html = "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
        cards.append(f'<tr data-search="{html.escape(search, quote=True)}">{cells_html}<td><details><summary>{html.escape(original["run_name"])}</summary><p>{details}</p>{"<br>".join(links)}</details></td></tr>')
    pd.DataFrame(run_rows).to_csv(output / "physical_run_qpm_results.csv", index=False)
    counts = table.execution_status.value_counts().to_dict()
    status = "completed" if counts.get("completed", 0) == len(tasks) else "completed_with_errors" if counts.get("completed", 0)+counts.get("failed", 0)==len(tasks) else "running"
    progress = dict(status=status, updated_utc=now(), physical_runs=len(runs), total_scopes=len(tasks),
                    completed_scopes=counts.get("completed", 0), failed_scopes=counts.get("failed", 0),
                    pending_scopes=counts.get("pending", 0),
                    diagnostic_scope_counts=table.loc[table.flagged.eq(True)].diagnostic_status.value_counts().to_dict(),
                    csa_diagnostic_scope_counts=table.loc[table.flagged.eq(True)].csa_diagnostic_status.value_counts().to_dict(),
                    csa_scopes_with_diagnostic_clues=int((table.flagged.eq(True) & table.csa_diagnostic_status.eq("diagnostic_clues_found")).sum()),
                    input_plan=plan)
    write_json(output / "progress.json", progress)
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flagged-run QPM investigation</title><style>body{{font:15px Arial,sans-serif;background:#fcfcf8;color:#264653;margin:24px;overflow-wrap:anywhere}}h1{{margin-bottom:8px}}a{{color:#16796e}}.guard{{padding:14px;background:#fbf2d8;border-left:5px solid #e9c46a}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d9ded9;vertical-align:top}}th{{background:#e8f3ef}}td{{max-width:320px;overflow-wrap:anywhere}}input{{padding:10px;width:min(700px,90%);margin:18px 0}}summary{{cursor:pointer}}details p{{min-width:280px}}.muted{{color:#65716d}}</style></head><body>
<h1>QPM investigation of all {len(runs)} flagged physical runs</h1>
<p id="progress">{progress['completed_scopes']} / {len(tasks)} scope reports complete; {progress['failed_scopes']} processing failures; status: {status}.</p>
<p>{plan['flagged_scopes']} flagged scopes plus {plan['unflagged_sibling_scopes']} unflagged sibling scopes. The original flags overlap: 101 apparent dropout runs, 134 imbalance runs, 563 with a low-read analyzed scope.</p>
<p class="guard">Potential failure-localization clues, not proven root causes. Replicated affected-versus-surviving evidence, affected-path historical anomalies, and shared-path reference-only anomalies are distinct evidence levels. A reference-only anomaly does not explain a barcode dropout by itself. Earliest means earliest qualifying <em>covered</em> operation; uncovered earlier steps remain possible. Thousands of exploratory comparisons can generate chance anomalies. Missing QPM/history is unresolved, not normal. No delivered-volume error or instrument setting change is inferred.</p>
<p>Each report retains all three baseline comparisons and up to five evidence panels per comparison, with Er72 exclusions and full physical matching. Operation indices are scope-local, not comparable clocks across lanes. Outcome-qualified history is empirical, not certified fault-free.</p>
<p>CSA-wide low-performance clues (for example RawTrace and chip diagnostics) are reported separately from QPM localization. An empirical metric outlier is not a proven cause or a per-run estimate of lost reads; missing views and inadequate compatible history remain unresolved.</p>
<p><a href="physical_run_qpm_results.csv">Physical-run results CSV</a> · <a href="scope_qpm_results.csv">Scope results CSV</a> · <a href="progress.json">Batch status and provenance</a></p>
<label for="search">Filter by date, instrument, chip, assay, flag, or QPM finding</label><br><input id="search" type="search" placeholder="e.g. gnxs-0073, dropout, insufficient, GX9"><p id="counter">{len(runs)} physical runs</p>
<div class="scroll"><table><thead><tr>{''.join(f'<th>{label}</th>' for label in ['Date','Instrument','Chip','Assay / panel','Outcome flags','Progress','QPM evidence category','Separate CSA diagnostic status','Run, potential steps & reports'])}</tr></thead><tbody id="rows">{''.join(cards)}</tbody></table></div>
<p class="muted">Updated {progress['updated_utc']}. Refresh this page while the batch runs. Frozen reference bank: {html.escape(plan['reference_bank'])}. Plugin: {html.escape(plan['plugin_version'])}.</p>
<script>document.getElementById('search').addEventListener('input',function(){{const q=this.value.toLowerCase();let count=0;for(const r of document.querySelectorAll('#rows tr')){{r.hidden=!r.dataset.search.includes(q);if(!r.hidden)count++}}document.getElementById('counter').textContent=count+' physical runs'}});</script></body></html>'''
    temporary = output / "index.html.partial"
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(output / "index.html")
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--observations", nargs="+", help="Restrict an audit pilot; full roster remains reconciled")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--unfiltered", action="store_true", help="Canonical full-profile comparison for audit pilots")
    parser.add_argument("--audit-reference-dir", type=Path, help="Require exact equivalence where unfiltered pilot reports exist")
    args = parser.parse_args()
    from qpm_dropout_plugin import VERSION as PLUGIN_VERSION, load_config
    from qpm_waveform_reference import load_profiles
    global _BANK, _INDEX, _QUALIFIED_ONLY
    _BANK = resolve_reference_bank(args.project)
    args.output.mkdir(parents=True, exist_ok=True)
    runs, tasks = task_roster(args.screen, _BANK, args.output)
    config = load_config(_BANK / "qpm_dropout_plugin_config.json")
    plan = dict(version=VERSION, plugin_version=PLUGIN_VERSION, reference_bank=str(_BANK),
                reference_manifest_sha256=file_sha256(_BANK/MANIFEST),
                screen_roster_sha256=file_sha256(args.screen/"all_observations.csv"),
                flagged_physical_runs_sha256=file_sha256(args.screen/"flagged_runs.csv"),
                flagged_scopes=sum(t["flagged"] for t in tasks),
                unflagged_sibling_scopes=sum(not t["flagged"] for t in tasks),
                physical_runs=len(runs), scopes=len(tasks), config=config,
                driver_sha256=file_sha256(Path(__file__)),
                candidate_prefilter_enabled=not args.unfiltered,
                full_scan_audit_reference=str(args.audit_reference_dir) if args.audit_reference_dir else None,
                full_matching_and_trace_gates_unchanged=True, no_instrument_settings_modified=True,
                no_combined_root_cause_confidence_or_actual_volume_claim=True)
    # Fingerprint every runtime dependency, not just the driver or version.
    from audit_expanded_reference_bank import CODE_FILES
    dependencies = list(dict.fromkeys([*CODE_FILES, "csa_low_performance_diagnostics.py", "csa_root_cause_hypotheses.py", "train_multimodal_csa.py",
                                      "qpm_same_run_curves.py", "test_qpm_sparse_display.py", "test_qpm_same_run_curves.py",
                                      "test_qpm_outcome_first_layout.py",
                                      "test_csa_low_performance_diagnostics.py", "test_csa_root_cause_hypotheses.py", "test_top5_csa_metrics.py",
                                      "test_csa_low_performance_batch.py", "test_qpm_diagnostic_routing.py",
                                      "qpm_internal_comparison.py", "qpm_barcode_reference.py",
                                      "plot_dropout_workflow_curves.py", "find_qpm_failure_examples.py"]))
    # Runtime snapshot excludes test sources now stored under tests/upstream.
    dependencies = [name for name in dependencies if not name.startswith("test_")]
    plan["code_sha256"] = {name:file_sha256(Path(__file__).parent/name) for name in dependencies}
    plan["csa_diagnostic_input_sha256_by_project"] = diagnostic_plan_inputs(tasks, config)
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    existing = args.output / "batch_plan.json"
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError("Output belongs to different code, roster, settings or bank; choose a new output directory")
    write_json(existing, plan)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, expected in plan["code_sha256"].items():
        destination = snapshot / name
        if destination.is_file():
            if file_sha256(destination) != expected:
                raise ValueError("Frozen batch source differs from its input fingerprint")
        else:
            shutil.copy2(Path(__file__).parent / name, destination)
    pd.DataFrame(tasks).to_csv(args.output / "task_roster.csv", index=False)
    results = {}
    for task in tasks:
        path = Path(task["report_dir"]) / "batch_completion.json"
        if path.is_file():
            result = json.loads(path.read_text())
            if result.get("batch_fingerprint") == fingerprint and (
                    result.get("status") != "completed" or completed_report_cache_valid(Path(task["report_dir"]))):
                results[task["observation_id"]] = result
    progress = summarize(args.output, runs, tasks, results, plan)
    print(json.dumps({k:v for k,v in progress.items() if k != "input_plan"}), flush=True)
    if args.summary_only:
        return
    selected = [t for t in tasks if results.get(t["observation_id"], {}).get("status") != "completed"]
    if args.observations:
        requested = set(args.observations)
        if not requested <= {t["observation_id"] for t in tasks}:
            raise ValueError("Requested pilot scope is not part of the flagged-run roster")
        selected = [t for t in selected if t["observation_id"] in requested]
    if args.audit_reference_dir:
        selected.sort(key=lambda t: not (args.audit_reference_dir / "reports" / t["observation_id"] / "plugin_run_log.json").is_file())
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        return
    if not 1 <= args.workers <= 32:
        raise ValueError("Batch concurrency must be between 1 and 32")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    print("Loading validated QPM profiles once before read-only worker fork", flush=True)
    profiles = load_profiles(_BANK)
    from qpm_waveform_reference import load_policy
    _QUALIFIED_ONLY = load_policy(_BANK, config.get("waveform_reference_policy_overrides"))['require_outcome_qualified_reference']
    _INDEX = CandidateIndex(profiles, qualified_references_only=_QUALIFIED_ONLY)
    if args.unfiltered:
        _INDEX.view = lambda observation_id: profiles
    print(f"Ready: {len(profiles[0])} quality-cleared waveforms; {len(selected)} queued scopes", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {pool.submit(execute_task, task, config, fingerprint, str(args.audit_reference_dir) if args.audit_reference_dir else None):task for task in selected}
        for number, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            result = future.result()
            results[task["observation_id"]] = result
            print(json.dumps({k:v for k,v in result.items() if k not in {"batch_fingerprint"}}), flush=True)
            if number % 8 == 0 or number == len(selected):
                progress = summarize(args.output, runs, tasks, results, plan)
                print(json.dumps(dict(completed=progress["completed_scopes"], failed=progress["failed_scopes"],
                                     total=len(tasks), elapsed_seconds=round(time.time()-started, 1))), flush=True)
    if any(file_sha256(Path(__file__).parent/name) != value for name,value in plan["code_sha256"].items()):
        raise ValueError("Protected plugin source changed during the batch")
    if file_sha256(_BANK/MANIFEST) != plan["reference_manifest_sha256"]:
        raise ValueError("Reference bank activation changed during the batch")
    if diagnostic_plan_inputs(tasks, config) != plan["csa_diagnostic_input_sha256_by_project"]:
        raise ValueError("Protected external CSA diagnostic inputs changed during the batch")
    summarize(args.output, runs, tasks, results, plan)


if __name__ == "__main__":
    main()
