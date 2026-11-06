"""Path-specific historical QPM fallback for under-replicated comparisons.

Reuse the independently audited physical-reference scores. Do not relax matching,
invent AUC/p-values, treat attempts as independent cases, or attribute a shared
operation to an individual barcode without verified lineage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qpm_internal_comparison import independent_path_ids, profile
from qpm_trace_quality import QUALITY_FIELDS, eligible_waveforms

VERSION = "qpm-barcode-reference-fallback-1.1.0"
EVIDENCE_COLUMNS = [
    "material_type", "planned_order", "operation_id", "operation_index", "group_id",
    "step_label", "family", "workflow_stage", "process_step", "location", "operation",
    "pipette", "programmed_volume_ul", "liquid_class_index", "event_time", "mapping_scope",
    "source_file", "pressure_unit", "n_failed", "n_survivor", "fallback_reason",
    "fallback_status", "exclusion_reasons", "reference_physical_runs",
    "reference_scope", "reference_profile_unit", "reference_normalization", "reference_shape_only",
    "reference_instruments", "reference_instrument_ids", "same_instrument_available_runs",
    "cross_instrument_available_runs", "cross_instrument_available_instruments", "cross_instrument_attempted", "cross_instrument_status",
    "maximum_available_reference_runs", "matched_reference_level", "reference_time_policy",
    "qpm_shape_score", "qpm_anomaly_percentile", "outside_90pct_envelope_fraction",
    "amplitude_ratio_to_reference", "reference_positive", "surviving_paths_reference_scored",
    "surviving_paths_reference_positive", "survivor_context", "survivor_context_time_confounded",
    "evidence_role", "evidence_basis", "interpretation",
] + QUALITY_FIELDS
SUMMARY_COLUMNS = [
    "material_type", "planned_order", "fallback_operations", "reference_scored_operations",
    "same_instrument_scored_operations", "cross_instrument_shape_scored_operations",
    "unresolved_operations", "reference_anomaly_operations", "earliest_reference_operation_index",
    "earliest_reference_step", "later_reference_anomaly_operations", "status", "interpretation",
]


def _number(row: pd.Series, column: str, default: float = np.nan) -> float:
    value = row.get(column, default)
    return float(value) if pd.notna(value) else default


def _valid_profile(value: object) -> bool:
    try:
        curve = profile(value)
        return curve.shape == (32,) and bool(np.isfinite(curve).all())
    except (ValueError, TypeError, SyntaxError):
        return False


def reference_unit(row: pd.Series) -> str:
    value = row.get("reference_profile_unit")
    return str(value) if pd.notna(value) else "raw_sensor_counts"


def reference_curve(row: pd.Series) -> object:
    value = row.get("reference_observed_profile")
    if reference_unit(row) == "dimensionless_shape":
        return value  # Never put raw counts onto a normalized reference band.
    return value if _valid_profile(value) else row.get("observed_profile")


def _exclusions(row: pd.Series, config: dict, require_lineage: bool = True) -> list[str]:
    reasons = []
    if require_lineage and pd.isna(row.get("planned_order", np.nan)):
        reasons.append("barcode_lineage_unresolved")
    if row.get("command_context_status") != "command_verified":
        reasons.append("physical_configuration_unverified")
    if _number(row, "reference_physical_runs", 0) < config["minimum_reference_physical_runs"]:
        reasons.append("insufficient_compatible_historical_runs")
    if reference_unit(row) == "dimensionless_shape":
        minimum_instruments = config.get("waveform_reference_policy_overrides", {}).get("minimum_cross_instrument_instruments", 2)
        if _number(row, "reference_instruments", 0) < minimum_instruments:
            reasons.append("insufficient_cross_instrument_diversity")
        if row.get("reference_normalization") != "baseline_center_then_unit_peak_to_peak":
            reasons.append("unverified_cross_instrument_normalization")
    if not all(np.isfinite(_number(row, column)) for column in
               ["qpm_shape_score", "qpm_anomaly_percentile", "outside_90pct_envelope_fraction"]):
        reasons.append("historical_scores_unavailable")
    if not _valid_profile(reference_curve(row)) or not all(_valid_profile(row.get(column)) for column in
               ["reference_median_profile", "reference_q05_profile", "reference_q95_profile"]):
        reasons.append("waveform_or_reference_profile_unavailable")
    return reasons


def _positive(row: pd.Series, config: dict) -> bool:
    return bool(_number(row, "qpm_anomaly_percentile") >= config["shared_reference_minimum_percentile"]
                and _number(row, "outside_90pct_envelope_fraction") >= config["shared_reference_minimum_outside_fraction"])


def build_barcode_reference_fallback(mapped: pd.DataFrame, steps: pd.DataFrame,
                                     config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every affected operation in insufficient-replication groups.

    Qualified positive operations are nominated separately per material/barcode.
    Matching survivor paths provide descriptive specificity context, not a
    replicated group test. Unknown earlier steps remain explicit coverage gaps.
    """
    mapped = eligible_waveforms(mapped)
    if mapped.empty or steps.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS), pd.DataFrame(columns=SUMMARY_COLUMNS)
    underpowered = steps[steps["screen_label"].eq("insufficient_replication")]
    records = []
    for _, step in underpowered.iterrows():
        group = mapped[mapped["group_id"].eq(step["group_id"])].drop_duplicates("operation_id")
        affected = group[group["path_status"].eq("failed")]
        if affected.empty:
            continue
        survivors = group[group["path_status"].eq("survivor")].copy()
        scored_survivors = survivors[survivors.apply(lambda row: not _exclusions(row, config), axis=1)].copy() if not survivors.empty else survivors
        time_gap = _number(step, "failed_survivor_time_gap_minutes")
        time_confounded = np.isfinite(time_gap) and time_gap > config["maximum_failed_survivor_time_gap_minutes"]
        for _, row in affected.iterrows():
            # Do not equate raw-amplitude and normalized-shape screens. Report
            # surviving reference context only on the same comparison scale.
            comparable = scored_survivors[scored_survivors.apply(reference_unit, axis=1).eq(reference_unit(row))].copy() if not scored_survivors.empty else scored_survivors
            survivor_n = survivor_positive_n = 0
            if not comparable.empty:
                comparable["path_id"] = independent_path_ids(comparable)
                comparable["positive"] = comparable.apply(lambda entry: _positive(entry, config), axis=1)
                survivor_n = comparable["path_id"].nunique()
                survivor_positive_n = int(comparable.groupby("path_id")["positive"].any().sum())
            context = ("time_confounded_survivor_context" if time_confounded else
                       "no_qualified_surviving_path_references" if survivor_n == 0 else
                       "also_seen_in_surviving_paths" if survivor_positive_n else
                       "not_seen_in_covered_surviving_paths")
            reasons = _exclusions(row, config)
            scored = not reasons
            positive = scored and _positive(row, config)
            record = {column: row.get(column, np.nan) for column in EVIDENCE_COLUMNS}
            record.update(
                step_label=step["step_label"], n_failed=int(step["n_failed"]), n_survivor=int(step["n_survivor"]),
                fallback_reason="insufficient_replication", fallback_status="reference_scored" if scored else "unresolved",
                reference_profile_unit=reference_unit(row),
                reference_scope=(row.get("reference_scope") if pd.notna(row.get("reference_scope")) else
                                 "same_instrument" if scored else "unresolved"),
                exclusion_reasons=";".join(reasons), reference_positive=positive,
                reference_physical_runs=int(_number(row, "reference_physical_runs", 0)),
                maximum_available_reference_runs=max(int(_number(row, "maximum_available_reference_runs", 0)),
                                                     int(_number(row, "reference_physical_runs", 0))),
                surviving_paths_reference_scored=int(survivor_n), surviving_paths_reference_positive=survivor_positive_n,
                survivor_context=context, survivor_context_time_confounded=bool(time_confounded),
                evidence_role="unresolved_reference_coverage" if not scored else
                              "not_reference_positive" if not positive else "affected_path_reference_anomaly",
                evidence_basis="historical_reference_fallback_not_replicated_group_evidence",
                interpretation="exploratory_path_anomaly_not_causal_not_delivered_volume_not_calibrated_failure_probability",
            )
            records.append(record)
    evidence = pd.DataFrame(records, columns=EVIDENCE_COLUMNS)
    if evidence.empty:
        return evidence, pd.DataFrame(columns=SUMMARY_COLUMNS)
    evidence = evidence.sort_values(["material_type", "operation_index", "operation_id"]).reset_index(drop=True)
    summaries = []
    for (material, order), rows in evidence.groupby(["material_type", "planned_order"], dropna=False, sort=False):
        positives = rows[rows["reference_positive"]].sort_values("operation_index")
        first = positives.iloc[0] if not positives.empty else None
        if first is not None:
            evidence.loc[positives.index, "evidence_role"] = "later_affected_path_reference_anomaly"
            evidence.loc[first.name, "evidence_role"] = "earliest_affected_path_reference_anomaly"
        scored = int(rows["fallback_status"].eq("reference_scored").sum())
        summaries.append(dict(
            material_type=material, planned_order=order, fallback_operations=len(rows),
            reference_scored_operations=scored, unresolved_operations=len(rows) - scored,
            same_instrument_scored_operations=int((rows.fallback_status.eq("reference_scored") & rows.reference_scope.eq("same_instrument")).sum()),
            cross_instrument_shape_scored_operations=int((rows.fallback_status.eq("reference_scored") & rows.reference_scope.eq("cross_instrument_shape")).sum()),
            reference_anomaly_operations=len(positives),
            earliest_reference_operation_index=int(first["operation_index"]) if first is not None else np.nan,
            earliest_reference_step=first["step_label"] if first is not None else "",
            later_reference_anomaly_operations=max(0, len(positives) - 1),
            status="unresolved_barcode_lineage" if pd.isna(order) else
                   "reference_anomalies_detected" if first is not None else
                   "reference_scored_no_qualifying_anomaly" if scored else "unresolved_reference_coverage",
            interpretation="earliest_qualifying_covered_path_anomaly_only; uncovered_steps_unresolved; no_group_AUC_or_p_value; non_causal",
        ))
    return evidence, pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def fallback_payload(evidence: pd.DataFrame, summary: pd.DataFrame, material: str) -> dict:
    rows = evidence[evidence["material_type"].eq(material)]
    paths = summary[summary["material_type"].eq(material)]
    return {
        "version": VERSION,
        "requested": not rows.empty,
        "fallback_groups": int(rows["group_id"].nunique()),
        "affected_operations": len(rows),
        "reference_scored_operations": int(rows["fallback_status"].eq("reference_scored").sum()),
        "same_instrument_scored_operations": int((rows.fallback_status.eq("reference_scored") & rows.reference_scope.eq("same_instrument")).sum()),
        "cross_instrument_shape_scored_operations": int((rows.fallback_status.eq("reference_scored") & rows.reference_scope.eq("cross_instrument_shape")).sum()),
        "unresolved_operations": int(rows["fallback_status"].eq("unresolved").sum()),
        "reference_anomaly_operations": int(rows["reference_positive"].sum()),
        "paths": paths.to_dict("records"),
        "earliest_path_anomalies": rows[rows["evidence_role"].eq("earliest_affected_path_reference_anomaly")].to_dict("records"),
        "later_path_anomalies": rows[rows["evidence_role"].eq("later_affected_path_reference_anomaly")].to_dict("records"),
        "evidence_basis": "historical_reference_fallback_not_replicated_group_evidence",
        "guardrail": "No group AUC, permutation p, BH q, delivery error or causal attribution is inferred from this fallback. Repeated attempts are not independent cases.",
    }


def report_route(has_affected: bool, has_selective_initial: bool, fallback_requested: bool, run_qpm: bool) -> str:
    if has_affected:
        return "barcode_path_reference_fallback" if fallback_requested and not has_selective_initial else "within_run_barcode_comparison"
    return "shared_path_reference" if run_qpm else "no_qpm_trigger"
