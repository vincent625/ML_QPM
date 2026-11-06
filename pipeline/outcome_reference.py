#!/usr/bin/env python3
"""Auditable outcome benchmarks with strict protocol metadata matching.

Outcome peers and QPM operation templates are different reference populations.
This module constructs the former, with no chip/assay/protocol relaxation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VERSION = "outcome-reference-1.1.0-outcome-qualified"
OUTCOMES = ["dna_median_reads", "rna_median_reads", "total_reads",
            "mean_read_length_bp", "raw_read_accuracy_pct"]
SOURCES = {
    "chip_type": "ion_params_00.chip_info_chip_type",
    "assay_type": "global.analysis_application_type",
    "panel_technology": "global.analysis_panel_technology",
    "workflow_script": "planned_run.object_script_name",
    "run_mode": "planned_run.object_run_type",
    "template_size": "planned_run.object_templating_size",
    "flow_count": "ion_params_00.flows",
    "flow_order": "ion_params_00.flow_order",
    "resequencing": "planned_run.object_resequencing",
    "instrument_software": "ion_params_00.exp_json_log_genexus_release_version",
    "analysis_software": "global.ir_version_number",
    "sequencing_kit": "ion_params_00.exp_json_sequencekitname",
}
PANEL_COLUMNS = ["global.analysis_regions_file", "global.analysis_fusions_file"]
DEFAULT_POLICY = {
    "version": "outcome-reference-policy-1.0.0",
    "minimum_physical_runs": 8,
    "preferred_physical_runs": 20,
    "time_policy": "prior_only",
    "allow_cross_instrument": True,
    "require_fields": list(SOURCES) + ["panel_signature", "dna_barcode_count", "rna_barcode_count"],
    "extra_match_columns": [],
    "require_outcome_qualified_reference": True,
}


def normalize(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (float, int, np.number)):
        return format(float(value), ".12g")
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower()
    if text.lower() in {"", "none", "null", "nan", "unknown", "n/a"}:
        return None
    return text


def panel_signature(row: pd.Series) -> str | None:
    # Names plus the assay package/version identify the panel independently of
    # the instrument's installation prefix. Never output the original paths.
    members = []
    for column in PANEL_COLUMNS:
        raw = normalize(row.get(column))
        if raw is None:
            members.append(None)
            continue
        parts = Path(raw).parts
        package = parts[parts.index("AssayImport") + 1] if "AssayImport" in parts else ""
        members.append([package, Path(raw).name])
    if not any(members):
        return None
    return hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest()[:20]


def load_policy(project: Path, overrides: dict | None = None) -> dict:
    policy = dict(DEFAULT_POLICY)
    path = project / "outcome_reference_policy.json"
    if path.exists():
        policy.update(json.loads(path.read_text(encoding="utf-8")))
    if overrides:
        policy.update(overrides)
    if policy["time_policy"] not in {"prior_only", "retrospective"}:
        raise ValueError("time_policy must be prior_only or retrospective")
    if int(policy["minimum_physical_runs"]) < 3:
        raise ValueError("minimum_physical_runs must be at least three")
    if not {"chip_type", "assay_type", "panel_signature"}.issubset(policy["require_fields"]):
        raise ValueError("Chip type, assay type, and panel identity must always be required")
    # Combined two-, three- and four-lane read totals are not comparable even
    # when all other protocol metadata and barcode counts happen to match.
    policy["extra_match_columns"] = list(dict.fromkeys([
        *policy["extra_match_columns"], "evaluated_lane_count"]))
    return policy


def load_outcome_table(project: Path, policy: dict) -> pd.DataFrame:
    path = project / "csa_ml_dataset.parquet"
    schema = set(pq.read_schema(path).names)
    requested = ["observation_id", "physical_run_id", "instrument_id", "chip_lane",
                 "run_start_time", "archive_id", "total_reads", "mean_read_length_bp",
                 "raw_read_accuracy_pct"] + list(SOURCES.values()) + PANEL_COLUMNS
    requested += list(policy["extra_match_columns"])
    table = pd.read_parquet(path, columns=[column for column in dict.fromkeys(requested) if column in schema])
    for name, source in SOURCES.items():
        table[name] = table[source].map(normalize) if source in table else None
    for column in policy["extra_match_columns"]:
        if column == "evaluated_lane_count":
            continue
        table[column] = table[column].map(normalize) if column in table else None
    table["panel_signature"] = table.apply(panel_signature, axis=1)
    balance = pd.read_csv(project / "qpm_failure_examples" / "dna_rna_balance_by_run.csv")
    fields = ["observation_id", "dna_n_planned_noncontrol_barcodes",
              "rna_n_planned_noncontrol_barcodes", "dna_median_reads", "rna_median_reads"]
    if balance["observation_id"].duplicated().any():
        raise ValueError("Barcode balance table contains duplicate observation IDs")
    table = table.merge(balance[fields], on="observation_id", how="left", validate="one_to_one")
    for material in ["dna", "rna"]:
        # Absent material can only be represented as zero if the balance row
        # exists and the other material has explicit planned noncontrol samples.
        other = "rna" if material == "dna" else "dna"
        raw = table[f"{material}_n_planned_noncontrol_barcodes"].copy()
        known_absent = raw.isna() & table[f"{other}_n_planned_noncontrol_barcodes"].notna()
        raw.loc[known_absent] = 0
        table[f"{material}_barcode_count"] = raw.map(normalize)
    from qpm_reference_qualification import load_reference_qualification
    qualification, _ = load_reference_qualification(project)
    table = table.drop(columns=["reference_eligible", "qualification_reasons", "evaluated_lane_count"], errors="ignore")
    table = table.merge(qualification[["observation_id", "reference_eligible", "qualification_reasons", "evaluated_lane_count"]],
                        on="observation_id", how="left", validate="one_to_one")
    table["evaluated_lane_count"] = table["evaluated_lane_count"].map(normalize)
    return table


def benchmark_table(table: pd.DataFrame, target_id: str, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    target_rows = table[table["observation_id"].eq(target_id)]
    if len(target_rows) != 1:
        raise ValueError("Target must resolve to exactly one observation")
    target = target_rows.iloc[0]
    fields = list(dict.fromkeys(policy["require_fields"] + policy["extra_match_columns"]))
    missing = [field for field in fields if field not in table or normalize(target.get(field)) is None]
    context = pd.DataFrame([{
        "field": field, "target_value": normalize(target.get(field)),
        "required": True, "status": "missing" if field in missing else "matched_exactly",
        "source": SOURCES.get(field, "panel file identities" if field == "panel_signature" else "barcode balance / extra metadata"),
    } for field in fields])
    candidates = table[~table["physical_run_id"].eq(target["physical_run_id"])].copy()
    audit_columns = ["observation_id", "physical_run_id", "instrument_id", "chip_lane", "run_start_time"]
    audit = candidates[audit_columns].copy()
    mismatch = pd.Series("", index=candidates.index, dtype=object)
    for field in fields:
        if field not in candidates or field in missing:
            mismatch += f"{field}:missing_target_or_column;"
        else:
            matches = candidates[field].map(normalize).eq(normalize(target[field]))
            mismatch.loc[~matches] += field + ";"
    timestamps = pd.to_datetime(candidates["run_start_time"], utc=True, errors="coerce")
    target_time = pd.to_datetime(target["run_start_time"], utc=True, errors="coerce")
    time_ok = timestamps.notna() & timestamps.lt(target_time) if policy["time_policy"] == "prior_only" else pd.Series(True, index=candidates.index)
    audit["metadata_matches"] = mismatch.eq("")
    audit["mismatched_fields"] = mismatch
    audit["time_eligible"] = time_ok
    audit["eligible_peer"] = audit["metadata_matches"] & time_ok
    if policy.get("require_outcome_qualified_reference", True):
        if "reference_eligible" not in candidates:
            raise ValueError("Outcome-peer read and barcode eligibility has not been verified")
        audit["reference_eligible"] = candidates.reference_eligible.eq(True)
        audit["qualification_reasons"] = candidates.get("qualification_reasons", "")
        audit["eligible_peer"] &= audit["reference_eligible"]
    audit["same_instrument"] = candidates["instrument_id"].eq(target["instrument_id"])
    audit["same_lane"] = candidates["chip_lane"].eq(target["chip_lane"])
    eligible = candidates[audit["eligible_peer"]].copy()
    warnings = [
        "Matching is exact on recorded metadata. Library loading concentration, reference/spike amount, reagent lots, input quality, and unrecorded assay/workflow changes are not verified.",
        "The 5th–95th percentiles describe the empirical peer distribution, not confidence intervals or approved acceptance limits.",
    ]
    if missing:
        warnings.append("Required target metadata missing: " + ", ".join(missing))
    minimum = int(policy["minimum_physical_runs"])
    tiers = [("same_instrument_same_lane", True, True), ("same_instrument_all_lanes", True, False)]
    if policy["allow_cross_instrument"]:
        tiers.append(("cross_instrument_all_lanes", False, False))
    summaries = []
    for outcome in OUTCOMES:
        target_value = pd.to_numeric(pd.Series([target.get(outcome)]), errors="coerce").iloc[0]
        base = {
            "outcome": outcome, "target_value": target_value,
            "reference_median": np.nan, "reference_q05": np.nan, "reference_q95": np.nan,
            "target_pct_of_reference_median": np.nan, "target_empirical_percentile": np.nan,
            "reference_physical_runs": 0, "reference_instruments": 0,
            "match_level": "none", "status": "insufficient_peers", "time_policy": policy["time_policy"],
            "percentile_precision": "unavailable", "unverified_context": "library_concentration;reference_spike_amount;reagent_lots;sample_input_quality;unrecorded_workflow_changes",
        }
        if missing:
            base["status"] = "missing_required_metadata"
        elif not pd.notna(target_value):
            base["status"] = "target_outcome_missing"
        selected = None
        for level, same_instrument, same_lane in tiers:
            group = eligible
            if same_instrument:
                group = group[group["instrument_id"].eq(target["instrument_id"])]
            if same_lane:
                group = group[group["chip_lane"].eq(target["chip_lane"])]
            values = pd.to_numeric(group[outcome], errors="coerce")
            group = group[values.notna()].copy()
            group[outcome] = values.loc[group.index]
            # Median of retained lanes/versions prevents one physical run from
            # receiving more weight than another. Target siblings are excluded.
            run_values = group.groupby("physical_run_id")[outcome].median()
            if len(run_values) < minimum or missing or not pd.notna(target_value):
                continue
            selected = (level, group, run_values)
            break
        audit[f"selected_for_{outcome}"] = False
        if selected is not None:
            level, group, values = selected
            median = float(values.median())
            base.update({
                "reference_median": median, "reference_q05": float(values.quantile(0.05)),
                "reference_q95": float(values.quantile(0.95)),
                "target_pct_of_reference_median": float(target_value / median * 100) if median else np.nan,
                "target_empirical_percentile": float(((values < target_value).sum() + 0.5 * (values == target_value).sum()) / len(values)),
                "reference_physical_runs": int(len(values)), "reference_instruments": int(group["instrument_id"].nunique()),
                "match_level": level, "status": "available",
                "percentile_precision": "sparse_exploratory" if len(values) < int(policy["preferred_physical_runs"]) else "empirical",
            })
            audit.loc[group.index, f"selected_for_{outcome}"] = True
        summaries.append(base)
    metadata = {
        "version": VERSION, "policy": policy, "target_observation_id": target_id,
        "target_physical_run_id": target["physical_run_id"], "target_excluded": True,
        "eligible_metadata_matched_observations": len(eligible),
        "eligible_metadata_matched_physical_runs": eligible["physical_run_id"].nunique(),
        "same_instrument_eligible_physical_runs": eligible.loc[
            eligible["instrument_id"].eq(target["instrument_id"]), "physical_run_id"
        ].nunique(),
        "warnings": warnings,
    }
    return pd.DataFrame(summaries), audit.reset_index(drop=True), context, metadata


def build_outcome_benchmark(project: Path, target_id: str, overrides: dict | None = None):
    policy = load_policy(project, overrides)
    result = benchmark_table(load_outcome_table(project, policy), target_id, policy)
    inputs = [project / "csa_ml_dataset.parquet",
              project / "qpm_failure_examples" / "dna_rna_balance_by_run.csv",
              project / "qpm_failure_examples" / "dna_rna_barcode_counts.csv",
              Path(__file__).resolve().parent / "qpm_reference_qualification.py",
              Path(__file__).resolve()]
    qualification_policy = project / "reference_qualification_policy.json"
    if qualification_policy.exists():
        inputs.append(qualification_policy)
    policy_path = project / "outcome_reference_policy.json"
    if policy_path.exists():
        inputs.append(policy_path)
    hashes = {}
    for path in inputs:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    result[3]["input_and_code_sha256"] = hashes
    return result


def write_benchmark(output_dir: Path, result: tuple) -> None:
    summary, audit, context, metadata = result
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "outcome_reference_summary.csv", index=False)
    audit.to_csv(output_dir / "outcome_peer_audit.csv", index=False)
    context.to_csv(output_dir / "outcome_matching_context.csv", index=False)
    (output_dir / "outcome_reference_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_outcome_benchmark(args.project, args.observation_id)
    write_benchmark(args.output_dir, result)
    print(result[0].to_string(index=False))


if __name__ == "__main__":
    main()
