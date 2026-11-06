#!/usr/bin/env python3
"""Derive an outcome-qualified local reference snapshot without altering its source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from audit_expanded_reference_bank import catalog, regression_tests, activate, CODE_FILES
from build_expanded_reference_bank import sha256, write_json
from qpm_reference_bank import MANIFEST
from qpm_reference_qualification import (DEFAULT_POLICY, POLICY_FILENAME,
    TABLE_FILENAME, MANIFEST_FILENAME, load_reference_qualification, write_qualification)
from qpm_waveform_reference import load_profiles, PHYSICAL_KEYS, PROFILE_COLUMNS
from qpm_trace_quality import er72_mask

VERSION = "qualified-reference-bank-build-1.1.0-gx7-gx8"


def nominal_registry(volume_study: Path) -> dict:
    """Validate a local nominal-only study before admitting its registry pin."""
    from qpm_nominal_volume_inference import NominalVolumeRecognizer
    volume_study = volume_study.resolve()
    validation = volume_study / "artifact_validation.json"
    if json.loads(validation.read_text()).get("status") != "passed":
        raise ValueError("Nominal-volume models have not passed artifact validation")
    log = volume_study / "analysis_run_log.json"
    record = dict(study_path=str(volume_study), analysis_run_log_sha256=sha256(log),
        artifact_validation_sha256=sha256(validation),
        label_basis="nominal_command_not_measured_delivery", actual_volume_prediction_validated=False)
    checked = NominalVolumeRecognizer(volume_study,
        expected_log_sha256=record["analysis_run_log_sha256"],
        expected_validation_sha256=record["artifact_validation_sha256"])
    for model in checked.artifacts["log"]["models"]:
        if sha256(volume_study / model["path"]) != model["sha256"]:
            raise ValueError("Nominal-volume model payload changed after validation")
    return record


def register_validated_nominal_study(output: Path, volume_study: Path):
    """Seal a new nominal registry without rebuilding unchanged reference tables."""
    output = output.resolve()
    path = output / MANIFEST
    manifest = json.loads(path.read_text())
    if manifest.get("validation_status") != "passed":
        raise ValueError("Registration requires a validated bank")
    for name, record in manifest["artifacts"].items():
        if sha256(output / name) != record["sha256"]:
            raise ValueError(f"Existing bank artifact changed before registration: {name}")
    registry = nominal_registry(volume_study)
    registry_path = output / "qpm_nominal_volume_model.json"
    previous_manifest_sha256 = sha256(path)
    write_json(registry_path, registry)
    manifest["artifacts"][registry_path.name] = dict(size_bytes=registry_path.stat().st_size,
                                                     sha256=sha256(registry_path))
    manifest["nominal_registry_updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    write_json(output / "nominal_model_registration.json", dict(
        previous_manifest_sha256=previous_manifest_sha256, manifest_sha256=sha256(path),
        registry=registry, reference_tables_unchanged=True, no_settings_modified=True))
    print("Validated nominal study registered; reference population is unchanged", flush=True)


def derive(source: Path, output: Path, volume_study: Path | None = None):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("A qualified snapshot must not overwrite its source")
    source_manifest = json.loads((source / MANIFEST).read_text())
    if source_manifest["validation_status"] != "passed":
        raise ValueError("The source bank has not passed validation")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "derived_snapshot_source.json"
    receipt = dict(source_bank=str(source), source_manifest_sha256=sha256(source / MANIFEST))
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Derived snapshot belongs to another source; use a new output")
    write_json(receipt_path, receipt)
    names = [n for n in source_manifest["artifacts"] if n not in {
        "reference_bank_validation.json", "reference_bank_regression_tests.json",
        "instrument_reference_coverage.csv"}]
    names += [n for n in ["instrument_coverage.csv", "source_archive_status.csv", "duplicate_archives.csv",
                          "archive_parse_audit.csv", "reference_observations.csv"] if (source / n).exists()]
    for name in dict.fromkeys(names):
        origin, destination = source / name, output / name
        expected = source_manifest["artifacts"].get(name, {}).get("sha256") or sha256(origin)
        if sha256(origin) != expected:
            raise ValueError(f"Source artifact no longer matches validation: {name}")
        if destination.exists():
            # Local policies are intentionally different in the derived bank.
            if name.endswith("policy.json"):
                continue
            if sha256(destination) != expected:
                raise ValueError(f"Existing derived artifact differs: {name}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copy2(origin, temporary)
        if sha256(temporary) != expected:
            raise ValueError(f"Copied artifact failed verification: {name}")
        temporary.replace(destination)
        print(f"Verified local copy: {name}", flush=True)
    write_json(output / POLICY_FILENAME, DEFAULT_POLICY)
    for name in ["qpm_waveform_reference_policy.json", "outcome_reference_policy.json"]:
        policy = json.loads((output / name).read_text())
        policy["require_outcome_qualified_reference"] = True
        if name == "outcome_reference_policy.json":
            policy["extra_match_columns"] = list(dict.fromkeys([
                *policy.get("extra_match_columns", []), "evaluated_lane_count"]))
        write_json(output / name, policy)
    if volume_study is not None:
        write_json(output / "qpm_nominal_volume_model.json", nominal_registry(volume_study))
    return source_manifest


def audit(output: Path, source: Path, source_manifest: dict):
    qualification, qualification_metadata = load_reference_qualification(output)
    write_qualification(output, qualification, qualification_metadata)
    print(json.dumps(qualification_metadata["summary"]), flush=True)
    regression_tests(output)
    print("Auditing outcome-qualified physical waveform populations", flush=True)
    profiles, metadata = load_profiles(output)
    coupler = profiles.container_model.fillna("").str.contains("chipCoupler", case=False) | profiles.location.str.contains("chip_coupler", case=False)
    profiles["chip_fluidic_context"] = "not_chip_contact"
    profiles.loc[coupler, "chip_fluidic_context"] = profiles.loc[coupler, "chip_type"].fillna("missing_chip_context").astype(str)
    physical = profiles[profiles.command_context_status.eq("command_verified") & profiles.chip_fluidic_context.ne("missing_chip_context")].copy()
    qualified = physical[physical.reference_eligible.eq(True)].copy()
    external = qualified[np.ptp(qualified[PROFILE_COLUMNS].to_numpy(float), axis=1) > 1e-8].copy()
    cross_keys = [k for k in PHYSICAL_KEYS if k != "instrument_id"]
    same, cross = catalog(qualified, PHYSICAL_KEYS), catalog(external, cross_keys)
    same["available_for_unqualified_target"] = same.physical_runs
    same["available_after_holding_out_qualified_target"] = same.physical_runs - 1
    same.to_csv(output / "qpm_same_instrument_reference_catalog.csv", index=False)
    cross.to_csv(output / "qpm_cross_instrument_reference_catalog.csv", index=False)
    for name, frame, keys in [("same", qualified, PHYSICAL_KEYS), ("cross", external, cross_keys)]:
        catalog(frame, keys + ["workflow_stage", "process_step", "location"]).to_parquet(
            output / f"qpm_{name}_step_location_reference_catalog.parquet", index=False)
    targets = catalog(physical, PHYSICAL_KEYS)
    sources = catalog(external, PHYSICAL_KEYS).rename(columns={
        "physical_runs": "local_reference_physical_runs", "instruments": "local_reference_instruments"})
    coverage = targets.merge(cross[cross_keys + ["physical_runs", "instruments"]].rename(columns={
        "physical_runs": "all_reference_physical_runs", "instruments": "all_reference_instruments"}),
        on=cross_keys, how="left", validate="many_to_one")
    coverage = coverage.merge(sources[PHYSICAL_KEYS + ["local_reference_physical_runs", "local_reference_instruments"]],
                              on=PHYSICAL_KEYS, how="left", validate="one_to_one")
    fields = ["all_reference_physical_runs", "all_reference_instruments", "local_reference_physical_runs", "local_reference_instruments"]
    coverage[fields] = coverage[fields].fillna(0).astype(int)
    coverage["other_instrument_physical_runs"] = coverage.all_reference_physical_runs - coverage.local_reference_physical_runs
    coverage["other_reference_instruments"] = coverage.all_reference_instruments - coverage.local_reference_instruments
    coverage["other_instrument_minimum_met"] = coverage.other_instrument_physical_runs.ge(8) & coverage.other_reference_instruments.ge(2)
    coverage.to_parquet(output / "qpm_cross_instrument_coverage_by_target_instrument.parquet", index=False)
    instruments = qualification.groupby("instrument_id", dropna=False).agg(
        observations=("observation_id", "size"), physical_runs=("physical_run_id", "nunique"),
        below_read_criterion=("low_performance_trigger", "sum"),
        qualified_reference_observations=("reference_eligible", "sum")).reset_index()
    qualified_runs = qualification[qualification.reference_eligible].groupby("instrument_id").physical_run_id.nunique()
    instruments["qualified_reference_physical_runs"] = instruments.instrument_id.map(qualified_runs).fillna(0).astype(int)
    instruments["qualified_reference_waveforms"] = instruments.instrument_id.map(qualified.groupby("instrument_id").size()).fillna(0).astype(int)
    instruments["same_physical_signatures_with_at_least_8_source_runs"] = instruments.instrument_id.map(
        same[same.physical_runs.ge(8)].groupby("instrument_id").size()).fillna(0).astype(int)
    instruments["cross_instrument_covered_target_signatures"] = instruments.instrument_id.map(
        coverage[coverage.other_instrument_minimum_met].groupby("instrument_id").size()).fillna(0).astype(int)
    instrument_names = pd.read_csv(output / "instrument_coverage.csv")[["instrument_id", "hostname"]].drop_duplicates("instrument_id")
    instruments = instruments.merge(instrument_names, on="instrument_id", how="left", validate="one_to_one")
    instruments.to_csv(output / "instrument_reference_coverage.csv", index=False)
    core = pd.read_parquet(output / "csa_ml_dataset.parquet")
    excluded = pd.DataFrame(metadata.get("trace_quality_exclusions", []))
    excluded.to_parquet(output / "qpm_bank_excluded_traces.parquet", index=False)
    checks = dict(
        all_source_observations_retained=len(core) == source_manifest["observations"],
        unique_observations=not core.observation_id.duplicated().any(),
        all_reference_sources_outcome_qualified=qualified.reference_eligible.all(),
        no_er72_in_reference_sources=not er72_mask(qualified).any(),
        target_candidates_not_filtered_by_outcome_gate=len(profiles) >= len(qualified),
        qualified_source_physical_runs_have_no_veto=qualification.groupby("physical_run_id").reference_eligible.nunique().le(1).all(),
        physical_runs_not_operations_as_votes=same.physical_runs.le(same.observations).all(),
        same_cross_sources_respect_identical_gate=set(external.operation_id).issubset(set(qualified.operation_id)),
        source_manifest_unchanged=sha256(source / MANIFEST) == json.loads((output / "derived_snapshot_source.json").read_text())["source_manifest_sha256"],
        regression_tests_passed=json.loads((output / "reference_bank_regression_tests.json").read_text())["status"] == "passed",
        regression_matches_current_code=all(sha256(Path(__file__).parent/name) == value for name,value in
            json.loads((output / "reference_bank_regression_tests.json").read_text())["code_sha256"].items()),
        regression_matches_current_tests=all(
            sha256(Path(__file__).resolve().parent.parent/"tests"/"upstream"/name) == value
            for name,value in json.loads((output / "reference_bank_regression_tests.json").read_text())["test_sha256"].items()))
    validation = dict(version=VERSION, status="passed" if all(checks.values()) else "failed",
        checks={k: bool(v) for k, v in checks.items()}, **qualification_metadata["summary"],
        target_waveforms=len(profiles), physically_matchable_target_waveforms=len(physical),
        outcome_qualified_reference_waveforms=len(qualified), cross_shape_reference_waveforms=len(external),
        same_physical_signatures=len(same), cross_physical_signatures=len(cross),
        minimum_reference_physical_runs=8, minimum_cross_reference_instruments=2,
        scope_warning=qualification_metadata["scope_warning"])
    write_json(output / "reference_bank_validation.json", validation)
    artifact_names = [n for n in source_manifest["artifacts"] if (output / n).is_file()]
    artifact_names += [POLICY_FILENAME, TABLE_FILENAME, MANIFEST_FILENAME, "qpm_same_instrument_reference_catalog.csv",
        "qpm_cross_instrument_reference_catalog.csv", "qpm_same_step_location_reference_catalog.parquet",
        "qpm_cross_step_location_reference_catalog.parquet", "qpm_cross_instrument_coverage_by_target_instrument.parquet",
        "qpm_bank_excluded_traces.parquet", "derived_snapshot_source.json"]
    if (output / "qpm_nominal_volume_model.json").is_file():
        artifact_names.append("qpm_nominal_volume_model.json")
    manifest = dict(bank_id=output.name, version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        validation_status=validation["status"], observations=len(core), physical_runs=int(core.physical_run_id.nunique()),
        instruments=int(core.instrument_id.nunique()), eligible_waveforms=len(qualified),
        source_archives=source_manifest["source_archives"], outcome_qualified_reference=True,
        qualification_summary=qualification_metadata["summary"], source_bank=str(source), source_collection_read_only=True,
        no_settings_modified=True, artifacts={n: dict(size_bytes=(output/n).stat().st_size, sha256=sha256(output/n)) for n in dict.fromkeys(artifact_names)})
    write_json(output / MANIFEST, manifest)
    summary = qualification_metadata["summary"]
    policy = qualification_metadata["policy"]
    thresholds = pd.DataFrame({"Chip": list(policy["minimum_reads_per_lane"]), **{
        f"{n} lane{'s' if n > 1 else ''} (M reads)": [value*n/1e6 for value in policy["minimum_reads_per_lane"].values()] for n in range(1, 5)}})
    report = f'''<!doctype html><html><head><meta charset="utf-8"><title>Outcome-qualified QPM bank</title><style>
body{{font:16px system-ui;color:#264653;background:#fcfcf8;max-width:1400px;margin:32px auto;padding:20px}}h1,h2{{color:#245c53}}
table{{border-collapse:collapse;background:white;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #d9ded9;text-align:left}}
.hero{{background:#e8f3ef;padding:22px;border-left:6px solid #2a9d8f}}.note{{background:#fbf2d8;padding:16px}}</style></head><body>
<h1>Outcome-qualified historical QPM references</h1><p>{html.escape(output.name)} · validation {validation['status']}</p>
<div class="hero">{summary['reference_eligible_physical_runs']:,} qualifying physical runs · {summary['reference_eligible_observations']:,} run/lane observations · {summary['reference_eligible_instruments']} instruments · {len(qualified):,} physically matched, Er72-cleared source waveforms</div>
<h2>The same read criterion screens references and triggers low-performance investigation</h2>{thresholds.to_html(index=False,border=0)}
<p>Read equality passes. GX7 and GX8 share the read cutoff, but retain distinct chip identities for outcome-peer and chip-contact physical matching. In addition, references must have no detected DNA/RNA noncontrol sample dropout or imbalance, complete barcode counts, and enough samples to assess balance. Known failed or unresolved sibling observations veto a source physical run. Missing chip, analyzed-lane scope or outcome remains unresolved.</p>
<p>{summary['below_chip_lane_read_criterion']:,} observations are below their read criterion; {summary['unresolved_chip_lane_read_criterion']:,} have unresolved read criteria; {summary['sample_dropout_or_imbalance_observations']:,} have detected dropout/imbalance. These sets can overlap.</p>
<p class="note">{html.escape(qualification_metadata['scope_warning'])} This is a user-defined outcome screen, not certified fault-free controls or manufacturer validation. Bad target runs remain diagnosable. Predictive models use both good and bad runs and do not use this source-only gate as a training filter.</p>
<h2>Three independent comparisons</h2><ol><li>Same-run affected versus surviving barcode paths; history not required.</li><li>Qualified same-instrument history; baseline-centered raw sensor counts.</li><li>Qualified other-instrument history; dimensionless shapes, never raw-count pooling.</li></ol>
<p>Full liquid-class/geometry/command settings, operation and volume must match. Minimum eight independent source physical runs; external history requires at least two other instruments. Every physical run has one vote. Missing compatible qualified history remains unresolved; no rejected source is restored to fill the quota.</p>
<h2>Instrument coverage</h2>{instruments.sort_values('hostname').to_html(index=False,border=0)}
<p>Signature coverage is not a count of proven faults. Same-instrument minimum counts above include all qualifying source runs; a qualifying target must additionally be held out, which can reduce a boundary cohort below eight.</p>
<h2>Audit outputs</h2><p><a href="{TABLE_FILENAME}">Every observation and exclusion reason</a> · <a href="{MANIFEST_FILENAME}">Policy and input hashes</a> · <a href="reference_bank_validation.json">Validation</a> · <a href="instrument_reference_coverage.csv">Coverage</a></p>
<p>Original archives, prior snapshots and reports are preserved. No instrument settings are modified. QPM deviations are diagnostic hypotheses, not measured delivery errors or proven causes.</p></body></html>'''
    (output / "reference_bank_report.html").write_text(report)
    print(json.dumps(validation, indent=2), flush=True)
    if validation["status"] != "passed":
        raise ValueError("Qualified bank failed validation; activation prohibited")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activate-project", type=Path)
    parser.add_argument("--nominal-volume-study", type=Path)
    parser.add_argument("--register-only", action="store_true", help="Register validated nominal models into an existing validated bank")
    args = parser.parse_args()
    if args.register_only:
        if args.nominal_volume_study is None:
            parser.error("--register-only requires --nominal-volume-study")
        register_validated_nominal_study(args.output, args.nominal_volume_study)
    else:
        if args.source is None:
            parser.error("--source is required when constructing a bank")
        source_manifest = derive(args.source, args.output, args.nominal_volume_study)
        audit(args.output, args.source, source_manifest)
    if args.activate_project:
        activate(args.output, args.activate_project)


if __name__ == "__main__":
    main()
