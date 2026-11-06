"""Shape-only external history, after under-replicated same-instrument history.

Never combine uncalibrated raw-count amplitudes. Full physical settings remain
matched, target/sibling runs are excluded, and repeated operations have one
physical-run vote. Cross-instrument shape screening is exploratory, not sensor
calibration, delivered-volume measurement, or a replicated barcode group test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from qpm_trace_quality import eligible_waveforms

from qpm_waveform_reference import PHYSICAL_KEYS, PROFILE_COLUMNS, load_policy, prepare_reference_pool, reference_template

VERSION = "qpm-cross-instrument-shape-reference-1.3.0-expanded-bank"
REFERENCE_FIELDS = [
    "matched_reference_level", "reference_physical_runs", "reference_time_policy",
    "reference_scope", "reference_profile_unit", "reference_normalization",
    "reference_instruments", "reference_instrument_ids", "reference_shape_only",
    "reference_observed_profile", "reference_median_profile", "reference_q05_profile",
    "reference_q95_profile", "reference_precision", "qpm_shape_score", "qpm_anomaly_percentile",
    "outside_90pct_envelope_fraction", "amplitude_ratio_to_reference", "waveform_morphology",
]
COVERAGE_FIELDS = [
    "operation_id", "reference_status", "reason", "matched_reference_level", "reference_physical_runs",
    "same_instrument_available_runs", "cross_instrument_available_runs", "cross_instrument_available_instruments",
    "cross_instrument_attempted", "cross_instrument_status", "reference_scope",
]
MEMBER_FIELDS = [
    "target_operation_id", "target_operation_index", "reference_physical_run_id", "reference_observation_ids",
    "reference_archive_ids", "reference_instrument_id", "reference_run_start_time", "reference_operation_count",
    "reference_includes_future", "matched_reference_level", "weight_per_physical_run", "reference_profile_unit",
]


def normalize_shape(curve: np.ndarray) -> np.ndarray:
    """Remove baseline and gain; do not manufacture a shape for flat traces."""
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (32,) or not np.isfinite(curve).all():
        raise ValueError("Shape comparison requires 32 finite waveform samples")
    amplitude = float(np.ptp(curve))
    if amplitude <= 1e-8:
        raise ValueError("Flat waveforms cannot supply a cross-instrument shape reference")
    return (curve - np.median(curve[:3])) / amplitude


def overlay_by_operation(frame: pd.DataFrame, updates: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Update historical annotations without replacing raw/internal waveforms."""
    if updates.empty or frame.empty:
        return frame.copy()
    if updates.operation_id.duplicated().any():
        raise ValueError("Ambiguous operation-level reference update")
    result = frame.copy()
    lookup = updates.set_index("operation_id")
    selected = result.operation_id.isin(lookup.index)
    for column in columns:
        if column not in lookup:
            continue
        if column not in result:
            result[column] = pd.Series([None] * len(result), index=result.index, dtype=object)
        elif result[column].dtype != object:
            result[column] = result[column].astype(object)
        result.loc[selected, column] = result.loc[selected, "operation_id"].map(lookup[column]).to_numpy()
    return result


def scan_cross_instrument_fallback(project: Path, target_id: str, operation_ids: set[str], config: dict,
                                   profiles: tuple[pd.DataFrame, dict], same_coverage: pd.DataFrame,
                                   *, independent: bool = False,
                                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Scan requested operations against OTHER instruments only.

    Legacy fallback skips operations with sufficient same-instrument history.
    Independent mode evaluates every requested operation regardless of internal
    replication or same-instrument coverage, keeping the populations separate.

    Compatible references come exclusively from other instruments. Two or more
    reference instruments and eight independent runs are required. Same-history
    availability is retained separately; missing configuration is never bypassed.
    """
    policy = load_policy(project, {"minimum_physical_runs": config.get("minimum_reference_physical_runs", 8),
                                   **config.get("waveform_reference_policy_overrides", {})})
    minimum = int(policy["minimum_physical_runs"])
    enabled = policy["instrument_policy"] == "same_instrument_then_cross_instrument_shape"
    metadata = {"version": VERSION, "enabled": enabled,
                "reference_qualification": profiles[1].get("reference_qualification", {}),
                "scope": "all_target_operations_independent" if independent else "insufficient_replication_groups_only",
                "matching": "other_instruments_only_full_physical_keys_except_instrument_id",
                "normalization": policy["cross_instrument_normalization"], "shape_only": True,
                "minimum_physical_runs": minimum, "minimum_reference_instruments": policy["minimum_cross_instrument_instruments"],
                "target_and_sibling_runs_excluded": True, "absolute_amplitude_comparable": False,
                "calibration_validated": False, "requested_operations": len(operation_ids),
                "attempted_operations": 0, "scored_operations": 0, "warnings": []}
    if not enabled or not operation_ids:
        return (pd.DataFrame(columns=["operation_id"] + REFERENCE_FIELDS), pd.DataFrame(columns=COVERAGE_FIELDS),
                pd.DataFrame(columns=MEMBER_FIELDS), metadata)
    qpm = eligible_waveforms(profiles[0])
    # This scan must not depend on scan_references mutating a shared frame.
    # The Er72 gate intentionally copies frames before filtering.
    if "chip_fluidic_context" not in qpm:
        models = qpm.get("container_model", pd.Series("", index=qpm.index)).fillna("")
        coupler = models.str.contains("chipCoupler", case=False) | qpm.location.str.contains("chip_coupler", case=False)
        qpm["chip_fluidic_context"] = "not_chip_contact"
        if policy["chip_context_for_coupler"]:
            chips = qpm.get("chip_type", pd.Series(None, index=qpm.index, dtype=object)).fillna("missing_chip_context")
            qpm.loc[coupler, "chip_fluidic_context"] = chips.loc[coupler].astype(str)
    target = qpm[qpm.observation_id.eq(target_id) & qpm.operation_id.isin(operation_ids)]
    if target.empty:
        return (pd.DataFrame(columns=["operation_id"] + REFERENCE_FIELDS), pd.DataFrame(columns=COVERAGE_FIELDS),
                pd.DataFrame(columns=MEMBER_FIELDS), metadata)
    pool = prepare_reference_pool(qpm, target.iloc[0], policy)
    pool = pool[pool.instrument_id.ne(target.iloc[0].instrument_id)].copy()
    amplitudes = np.ptp(pool[PROFILE_COLUMNS].to_numpy(float), axis=1)
    pool = pool[np.isfinite(amplitudes) & (amplitudes > 1e-8)].copy()
    if not pool.empty:
        matrix = pool[PROFILE_COLUMNS].to_numpy(float)
        pool[PROFILE_COLUMNS] = (matrix-np.median(matrix[:, :3], axis=1, keepdims=True))/np.ptp(matrix, axis=1, keepdims=True)
    keys = [key for key in PHYSICAL_KEYS if key != "instrument_id"]
    levels = [("physical_cross_instrument_step_location_shape", keys + ["workflow_stage", "process_step", "location"])]
    if policy["allow_cross_step_fallback"]:
        levels.append(("physical_cross_instrument_cross_step_shape", keys))
    lookups = [(name, fields, pool.groupby(fields, dropna=False, sort=False).indices) for name, fields in levels]
    prior = same_coverage.set_index("operation_id") if not same_coverage.empty else pd.DataFrame()
    scores, coverage, members = [], [], []
    template_cache = {}
    for _, row in target.sort_values("operation_index").iterrows():
        old = prior.loc[row.operation_id] if row.operation_id in prior.index else pd.Series(dtype=object)
        if not independent and (old.get("reference_status") == "scored" or int(old.get("reference_physical_runs", 0)) >= minimum):
            continue  # Same-instrument history always takes precedence.
        cover = {"operation_id": row.operation_id, "same_instrument_available_runs": int(old.get("maximum_available_reference_runs", 0)),
                 "cross_instrument_available_runs": 0, "cross_instrument_available_instruments": 0,
                 "cross_instrument_attempted": False, "cross_instrument_status": "unresolved_configuration",
                 "reference_scope": "unresolved", "reference_status": "unresolved", "reference_physical_runs": 0,
                 "matched_reference_level": "none", "reason": str(old.get("reason", "physical_configuration_unverified"))}
        if row.command_context_status != "command_verified" or row.chip_fluidic_context == "missing_chip_context":
            coverage.append(cover)
            continue
        cover.update(cross_instrument_attempted=True, cross_instrument_status="insufficient_other_instrument_history",
                     reason="insufficient_cross_instrument_physical_runs")
        metadata["attempted_operations"] += 1
        try:
            observed = normalize_shape(row[PROFILE_COLUMNS].to_numpy(float))
        except ValueError:
            cover.update(cross_instrument_status="unresolved_flat_target_waveform", reason="flat_target_no_cross_instrument_shape")
            coverage.append(cover)
            continue
        selected = None
        for level, fields, lookup in lookups:
            indices = lookup.get(tuple(row[field] for field in fields))
            if indices is None:
                continue
            candidates = pool.iloc[indices]
            n = candidates.physical_run_id.nunique()
            instruments = candidates.instrument_id.nunique()
            cover["cross_instrument_available_runs"] = max(cover["cross_instrument_available_runs"], n)
            cover["cross_instrument_available_instruments"] = max(cover["cross_instrument_available_instruments"], instruments)
            if n >= minimum and instruments >= policy["minimum_cross_instrument_instruments"]:
                selected = level, candidates
                break
        if selected is None:
            if cover["cross_instrument_available_runs"] >= minimum:
                cover.update(cross_instrument_status="insufficient_reference_instruments", reason="insufficient_cross_instrument_diversity")
            coverage.append(cover)
            continue
        level, reference = selected
        # Normalize each operation before medians; retries/sibling lanes cannot
        # increase effective n or give one physical run extra influence.
        cache_key = (level, tuple(row[k] for k in dict(levels)[level]))
        if cache_key not in template_cache:
            template_cache[cache_key] = reference_template(reference, row.run_start_time, shape_only=True)
        template = template_cache[cache_key]
        matrix = template["matrix"]
        median, q05, q95, scale = template["median"], template["q05"], template["q95"], template["scale"]
        shape = float(np.sqrt(np.mean(np.clip((observed - median) / scale, -10, 10) ** 2)))
        ref_scores = template["ref_scores"]
        percentile = float(((ref_scores <= shape).sum() + 1) / (len(matrix) + 1))
        outside = float(np.mean((observed < q05) | (observed > q95)))
        instrument_ids = template["instrument_ids"]
        record = {"operation_id": row.operation_id, "matched_reference_level": level,
                  "reference_physical_runs": len(matrix), "reference_time_policy": policy["time_policy"],
                  "reference_scope": "cross_instrument_shape", "reference_profile_unit": "dimensionless_shape",
                  "reference_normalization": policy["cross_instrument_normalization"], "reference_instruments": len(instrument_ids),
                  "reference_instrument_ids": "|".join(instrument_ids), "reference_shape_only": True,
                  "reference_observed_profile": json.dumps(observed.tolist()),
                  "reference_median_profile": json.dumps(median.tolist()), "reference_q05_profile": json.dumps(q05.tolist()),
                  "reference_q95_profile": json.dumps(q95.tolist()), "reference_precision": "exploratory_cross_instrument_shape_uncalibrated",
                  "qpm_shape_score": shape, "qpm_anomaly_percentile": percentile,
                  "outside_90pct_envelope_fraction": outside, "amplitude_ratio_to_reference": np.nan,
                  "waveform_morphology": "normalized_shape_only_absolute_amplitude_unassessed"}
        scores.append(record)
        cover.update(reference_status="scored", reference_physical_runs=len(matrix), reference_scope="cross_instrument_shape",
                     matched_reference_level=level, cross_instrument_status="scored_shape_only",
                     reason="matched_other_instruments_normalized_shape_not_amplitude_calibration")
        coverage.append(cover)
        for member in template["audit_members"]:
            members.append({"target_operation_id": row.operation_id, "target_operation_index": row.operation_index,
                            **member, "matched_reference_level": level})
    metadata["scored_operations"] = len(scores)
    metadata["warnings"] = [
        (f"Independent cross-instrument scan: {len(scores)}/{len(target)} target operations received qualified other-instrument shape references; same-instrument availability does not suppress this comparison."
         if independent else f"Under-replicated path fallback: {len(scores)}/{metadata['attempted_operations']} operations without qualified same-instrument history received normalized other-instrument shape references."),
        "Cross-instrument references compare normalized morphology only. Baseline/gain normalization is not sensor calibration; absolute-amplitude failures may be invisible. No amplitude ratio or delivered-volume estimate is inferred.",
        "Peak-to-peak normalization can magnify noise in low-dynamic-range traces; there is no validated instrument-specific noise floor for this shape screen.",
        "Cross-instrument shape percentiles are exploratory, not validated failure probabilities. Hardware/firmware shape differences and unequal instrument representation may remain confounded.",
    ]
    return (pd.DataFrame(scores, columns=["operation_id"] + REFERENCE_FIELDS),
            pd.DataFrame(coverage, columns=COVERAGE_FIELDS), pd.DataFrame(members, columns=MEMBER_FIELDS), metadata)
