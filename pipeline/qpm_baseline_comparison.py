"""Parallel QPM comparators: internal, same instrument, other instruments.

Never overlay the two historical populations or turn their agreement into
independent replication. This audit does not replace the primary plugin route.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from qpm_cross_instrument_reference import REFERENCE_FIELDS
from qpm_internal_comparison import collapse_paths, profile
from qpm_waveform_reference import PROFILE_COLUMNS
from qpm_trace_quality import QUALITY_FIELDS, eligible_waveforms

VERSION = "qpm-parallel-baselines-1.4.0-outcome-first-layout"
EVIDENCE_WAVEFORM_SLOTS = 5
MODES = ["within_run", "same_instrument", "cross_instrument"]
LABELS = {
    "within_run": "1. Same run — no historical baseline",
    "same_instrument": "2. Same-instrument historical baseline",
    "cross_instrument": "3. Other-instrument historical baseline",
}
CONTEXT = ["observation_id", "physical_run_id", "instrument_id", "archive_id", "chip_lane",
           "operation_id", "operation_index", "operation", "pipette", "programmed_volume_ul",
           "event_time", "workflow_stage", "process_step", "location", "liquid_class_index",
           "liquid_class_name", "container_geometry_name", "command_context_status", "pressure_unit"] + QUALITY_FIELDS


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def positive_history(frame: pd.DataFrame, config: dict) -> pd.Series:
    return (frame.qpm_anomaly_percentile.ge(config["shared_reference_minimum_percentile"])
            & frame.outside_90pct_envelope_fraction.ge(config["shared_reference_minimum_outside_fraction"]))


def enrich_history(target: pd.DataFrame, scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    target = eligible_waveforms(target)
    scores = eligible_waveforms(scores)
    context = target[[c for c in CONTEXT if c in target]].copy()
    context["observed_profile"] = [json.dumps(v.tolist()) for v in target[PROFILE_COLUMNS].to_numpy(float)] if not target.empty else []
    fields = list(dict.fromkeys([c for c in REFERENCE_FIELDS if c in scores] +
        [c for c in scores if c.startswith(("volume_proxy_", "waveform_equivalent_", "equivalent_volume_", "aggregate_validation_", "delivered_volume_"))]))
    result = context.merge(scores[["operation_id", *fields]], on="operation_id", how="inner", validate="one_to_one") if not scores.empty else context.iloc[:0].copy()
    if result.empty:
        # Stable schemas even when no instrument history exists.
        for c in REFERENCE_FIELDS:
            if c not in result:
                result[c] = pd.Series(dtype=object)
        result["reference_positive"] = pd.Series(dtype=bool)
        result["evidence_role"] = pd.Series(dtype=str)
        return result
    result = result.sort_values("operation_index").reset_index(drop=True)
    result["reference_positive"] = positive_history(result, config)
    result["evidence_role"] = np.where(result.reference_positive, "later_reference_anomaly", "not_reference_positive")
    indices = result.index[result.reference_positive]
    if len(indices):
        result.loc[indices[0], "evidence_role"] = "earliest_covered_reference_anomaly"
    result["interpretation"] = "exploratory_reference_only_not_barcode_selective_not_causal"
    return result


def summarize_baselines(lineage, tests, mapped, target, histories, coverages, members, metadata):
    """Always return exactly three rows, including unavailable comparators."""
    rows = []
    eligible = tests[tests.eligible_internal_comparison.astype(bool)] if not tests.empty else tests
    positive = tests[tests.evidence_role.isin(["initial_candidate", "same_transaction_confirmation", "downstream_support"])] if not tests.empty else tests
    affected = lineage[lineage.path_status.eq("failed")]
    within = dict(baseline_id="within_run", baseline=LABELS["within_run"],
                  status="selective_candidate_detected" if not positive.empty else
                         "tested_no_qualifying_selective_step" if not eligible.empty else
                         "not_testable_no_affected_paths" if affected.empty else "unresolved_internal_comparison",
                  target_operations=len(target), mapped_operations=mapped.operation_id.nunique() if not mapped.empty else 0,
                  operation_groups=len(tests), testable_groups=len(eligible),
                  untestable_groups=len(tests) - len(eligible), positive_groups=len(positive),
                  fdr_supported_groups=int(eligible.bh_fdr_q.le(.05).sum()) if not eligible.empty else 0,
                  scored_operations=mapped.loc[mapped.group_id.isin(eligible.group_id), "operation_id"].nunique() if not mapped.empty and not eligible.empty else 0,
                  reference_physical_runs=0, reference_instruments=0, profile_unit="raw_sensor_counts",
                  earliest_operation_index=int(positive.first_operation_index.min()) if not positive.empty else None,
                  comparison_basis="leave_one_path_out_same_run_survivors; no_historical_baseline",
                  limitation="Surviving paths are not certified healthy; block/timing confounding and sparse replication remain possible.")
    rows.append(within)
    for mode in MODES[1:]:
        frame, coverage, cohort = histories[mode], coverages[mode], members[mode]
        positives = frame[frame.reference_positive] if not frame.empty else frame
        refs = cohort.reference_physical_run_id.nunique() if not cohort.empty else 0
        instruments = cohort.reference_instrument_id.nunique() if not cohort.empty else 0
        available_max = coverage.get("maximum_available_reference_runs", coverage.get("cross_instrument_available_runs", pd.Series(dtype=float))).max()
        rows.append(dict(baseline_id=mode, baseline=LABELS[mode],
                         status="reference_anomalies_detected" if not positives.empty else
                                "scored_no_qualifying_reference_anomaly" if not frame.empty else
                                "disabled_by_explicit_policy" if metadata[mode].get("enabled") is False else
                                "unavailable_no_usable_target_waveforms" if target.empty else "unavailable_insufficient_compatible_history",
                         target_operations=len(target), scored_operations=len(frame),
                         unresolved_operations=max(0, len(target) - len(frame)), positive_operations=len(positives),
                         reference_physical_runs=refs, reference_instruments=instruments,
                         reference_runs_per_operation_min=int(frame.reference_physical_runs.min()) if not frame.empty else 0,
                         reference_runs_per_operation_max=int(frame.reference_physical_runs.max()) if not frame.empty else 0,
                         maximum_available_compatible_runs=int(available_max) if pd.notna(available_max) else 0,
                         cross_step_scored_operations=int(frame.matched_reference_level.str.contains("cross_step").sum()) if not frame.empty else 0,
                         future_reference_member_groups=int(cohort.reference_includes_future.astype(str).str.lower().eq("true").sum()) if not cohort.empty else 0,
                         profile_unit="raw_sensor_counts" if mode == "same_instrument" else "dimensionless_shape",
                         earliest_operation_index=int(positives.operation_index.min()) if not positives.empty else None,
                         comparison_basis="same_instrument_and_pipette_only" if mode == "same_instrument" else "other_instruments_only; baseline_center_then_unit_peak_to_peak",
                         limitation="Empirical, not certified healthy; reference anomalies are not barcode-selective evidence, calibrated failure probabilities or delivered-volume errors."))
    return pd.DataFrame(rows)


def later_evidence(first, candidates, index_column, identity_column, step_column,
                   limit=EVIDENCE_WAVEFORM_SLOTS - 1):
    """Prefer distinct later steps, then fill repeats, without changing evidence.

    The caller supplies qualifying rows only (or explicitly illustrative rows).
    Display diversity is not a causal or statistical ranking. Returned rows are
    chronological, unique, and strictly after the first displayed operation.
    """
    candidates = eligible_waveforms(candidates)
    if candidates.empty or limit <= 0:
        return []
    later = candidates[candidates[index_column].gt(first[index_column])].sort_values(
        index_column, kind="stable").drop_duplicates(identity_column)
    selected, used_steps, used_ids = [], {str(first[step_column])}, {str(first[identity_column])}
    for diverse_only in (True, False):
        for _, row in later.iterrows():
            identity, step = str(row[identity_column]), str(row[step_column])
            if identity in used_ids or (diverse_only and step in used_steps):
                continue
            selected.append(row)
            used_ids.add(identity)
            used_steps.add(step)
            if len(selected) == limit:
                return sorted(selected, key=lambda r: r[index_column])
    return sorted(selected, key=lambda r: r[index_column])


def select_path_reference_display(evidence):
    """Never borrow another barcode's anomaly as support for the first path."""
    evidence = eligible_waveforms(evidence)
    if evidence.empty:
        return []
    positives = evidence[evidence.reference_positive.astype(bool)].sort_values("operation_index", kind="stable")
    if positives.empty:
        return []
    first = positives.iloc[0]
    same_path = positives[positives.planned_order.eq(first.planned_order)]
    return [first] + later_evidence(first, same_path, "operation_index", "operation_id", "process_step")


def _display_curve_paths(mapped: pd.DataFrame, group_id) -> tuple[pd.DataFrame, str, int]:
    """Return existing independent raw paths, not statistical eligibility.

    A sparse curve pair can be displayed with one affected and one surviving
    path. Quality and physical compatibility are not relaxed. The statistical
    summarizer retains its independent replication requirement unchanged.
    """
    if mapped.empty or "group_id" not in mapped:
        return mapped.iloc[:0].copy(), "no_mapped_raw_operations", 0
    selected = eligible_waveforms(mapped.loc[mapped.group_id.eq(group_id)]).copy()
    if selected.empty:
        return selected, "no_quality_cleared_mapped_operations", 0
    if "qpm_trace_quality_status" not in selected or not selected.qpm_trace_quality_status.eq("checked_no_er72").all():
        return selected.iloc[:0], "trace_quality_unresolved", 0
    if "path_status" not in selected:
        return selected.iloc[:0], "outcome_labels_unresolved", len(selected)
    ignored = int((~selected.path_status.isin(["failed", "survivor"])).sum())
    # Unknown outcomes are not surviving controls merely because they are not
    # explicitly failed. Controls and unresolved paths cannot provide a curve.
    if "operation_id" not in selected:
        return selected.iloc[:0], "raw_waveform_or_path_identity_unavailable", ignored
    selected = selected.loc[selected.path_status.isin(["failed", "survivor"])].drop_duplicates("operation_id")
    if selected.empty:
        return selected, "no_evaluable_affected_or_surviving_paths", ignored
    verified = selected.get("internal_context_verified", pd.Series(False, index=selected.index))
    verified = verified.astype(str).str.lower().isin(["true", "1", "1.0"])
    if not verified.all() or "internal_context_fingerprint" not in selected:
        return selected.iloc[:0], "physical_configuration_unverified", ignored
    matching = ["material_type", "internal_context_fingerprint", "operation", "pipette", "programmed_volume_ul",
                "pressure_unit", "liquid_class_fingerprint", "container_geometry_fingerprint",
                "deck_geometry_fingerprint", "command_settings_fingerprint"]
    for field in matching:
        if field not in selected:
            continue
        values = selected[field]
        if values.isna().any() or values.astype(str).str.strip().str.lower().isin(["", "unknown", "unresolved", "nan", "none"]).any():
            return selected.iloc[:0], "physical_configuration_unverified", ignored
        if values.nunique(dropna=False) != 1:
            return selected.iloc[:0], "mixed_material_or_physical_configuration", ignored
    if not {"planned_order", "location", "observed_profile", "operation_id"}.issubset(selected):
        return selected.iloc[:0], "raw_waveform_or_path_identity_unavailable", ignored
    try:
        paths = collapse_paths(selected)
        raw_valid = (not paths.empty and "observed_profile" in paths and
                     all(profile(value).shape == (32,) and np.isfinite(profile(value)).all()
                         for value in paths.observed_profile))
    except (ValueError, TypeError, SyntaxError, KeyError):
        return selected.iloc[:0], "inconsistent_path_labels_or_raw_waveform", ignored
    if not raw_valid:
        return paths.iloc[:0], "raw_waveform_unavailable_or_invalid", ignored
    return paths, "", ignored


def display_internal_tests(tests: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive curve availability without changing statistical fields.

    All input test statistics, screen labels and evidence roles are preserved.
    A display-only selector can use these extra fields when no replicated test
    is available. Counts are independent barcode/location paths, not retries.
    """
    result = tests.copy(deep=True)
    defaults = dict(curve_available=False, curve_unavailable_reason="no_mapped_raw_operations",
                    curve_n_failed=0, curve_n_survivor=0, curve_descriptive_only=False,
                    curve_ignored_unresolved_operations=0)
    for key, default in defaults.items():
        result[key] = default
    if result.empty:
        return result
    if "group_id" not in result:
        result["curve_unavailable_reason"] = "test_group_identity_unavailable"
        return result
    for index, row in result.iterrows():
        paths, reason, ignored = _display_curve_paths(mapped, row.group_id)
        failed = int(paths.path_status.eq("failed").sum()) if "path_status" in paths else 0
        survivors = int(paths.path_status.eq("survivor").sum()) if "path_status" in paths else 0
        available = not reason and failed >= 1 and survivors >= 1
        if not reason and not available:
            reason = "no_affected_path" if not failed else "no_surviving_path"
        statistical = str(row.get("eligible_internal_comparison", False)).lower() in {"true", "1", "1.0"}
        result.loc[index, list(defaults)] = [bool(available), reason, failed, survivors,
                                            bool(available and not statistical), ignored]
    return result


def _display_rows(tests, history, mode):
    tests, history = eligible_waveforms(tests), eligible_waveforms(history)
    if mode == "within_run":
        available = tests[tests.curve_available.eq(True)] if "curve_available" in tests else tests
        eligible = available[available.eligible_internal_comparison.astype(bool)].sort_values("first_operation_index") if not available.empty else available
        if eligible.empty:
            # Availability is not a significance test: show chronological raw
            # pairs with 1+ affected and 1+ surviving path without inventing an
            # AUC, p-value, q-value or qualifying initial/support role.
            if available.empty or "curve_available" not in tests:
                return []
            descriptive = available.sort_values("first_operation_index", kind="stable")
            if descriptive.empty:
                return []
            first = descriptive.iloc[0]
            rows = [first] + later_evidence(first, descriptive, "first_operation_index", "group_id", "step_label")
            output = []
            for row in rows:
                displayed = row.copy()
                displayed["evidence_role"] = "descriptive_insufficient_replication"
                output.append(displayed)
            return output
        qualified = eligible[eligible.evidence_role.eq("initial_candidate")]
        illustrative = eligible[eligible.screen_label.eq("borderline")]
        first = (qualified if not qualified.empty else illustrative if not illustrative.empty else eligible).iloc[0]
        # Do not pad a qualifying chain with borderline or unselected curves.
        support = (eligible[eligible.evidence_role.isin(["same_transaction_confirmation", "downstream_support"])]
                   if not qualified.empty else eligible[eligible.screen_label.eq("borderline")])
        return [first] + later_evidence(first, support, "first_operation_index", "group_id", "step_label")
    if history.empty:
        return []
    positives = history[history.reference_positive].sort_values("operation_index")
    if positives.empty:
        return [history.sort_values("operation_index").iloc[0]]
    first = positives.iloc[0]
    return [first] + later_evidence(first, positives, "operation_index", "operation_id", "process_step")


def display_slots(tests, histories):
    """Five auditable slots per mode; absent evidence is explicit, not padded."""
    records = []
    for mode in MODES:
        selected = _display_rows(tests, histories.get(mode, pd.DataFrame()), mode)
        for position in range(EVIDENCE_WAVEFORM_SLOTS):
            row = selected[position] if position < len(selected) else None
            role = row.evidence_role if row is not None else "unavailable"
            illustrative = row is not None and mode == "within_run" and role not in {
                "initial_candidate", "same_transaction_confirmation", "downstream_support"}
            records.append(dict(baseline_id=mode, slot=position + 1, available=row is not None,
                                operation_index=int(row.first_operation_index if mode == "within_run" else row.operation_index) if row is not None else None,
                                evidence_role=role, interpretation=("illustrative_not_support" if illustrative else
                                "selective_candidate_not_causal" if mode == "within_run" and row is not None else
                                "reference_only_not_proven_support" if row is not None else "unresolved_no_additional_eligible_evidence"),
                                step=(str(row.step_label if mode == "within_run" else row.process_step) if row is not None else ""),
                                curve_descriptive_only=bool(row.get("curve_descriptive_only", False)) if row is not None and mode == "within_run" else False,
                                curve_n_failed=int(row.get("curve_n_failed", row.get("n_failed", 0))) if row is not None and mode == "within_run" else None,
                                curve_n_survivor=int(row.get("curve_n_survivor", row.get("n_survivor", 0))) if row is not None and mode == "within_run" else None))
    return pd.DataFrame(records)


def nominal_volume_note(selected):
    """Single-operation estimate only; validation MAE is not an individual CI."""
    value = selected.get("waveform_equivalent_volume_ul", np.nan)
    if pd.notna(value) and np.isfinite(float(value)):
        commanded = float(selected.programmed_volume_ul)
        delta = float(value) - commanded
        text = f"Command {commanded:g} µL → QPM-equivalent {float(value):.1f} µL\nNominal-signature Δ {delta:+.1f} µL ({100*delta/commanded:+.1f}%)"
        mae = selected.get("aggregate_validation_mae_ul", np.nan)
        if pd.notna(mae):
            text += f"; held-out MAE {float(mae):.1f} µL"
        return text + "\nNot a measured over/under-transfer amount"
    reason = str(selected.get("volume_proxy_status", "not_evaluated"))
    reason = reason.split(":", 1)[-1].replace("_", " ")
    return "Volume estimate unavailable: " + reason


def _waveform(ax, selected, mode, mapped, palette, position):
    x = np.linspace(0, 100, 32)
    if mode == "within_run":
        group, unavailable, _ = _display_curve_paths(mapped, selected.group_id)
        if unavailable or group.empty or not group.path_status.eq("failed").any() or not group.path_status.eq("survivor").any():
            ax.text(.5, .5, "Same-run curve pair unavailable\n" + (unavailable or "Affected or surviving paths absent").replace("_", " "),
                    ha="center", va="center", transform=ax.transAxes, wrap=True, fontsize=9)
            ax.set_axis_off()
            return
        matrix = np.vstack(group.observed_profile.map(profile))
        failed = group.path_status.eq("failed").to_numpy()
        healthy = matrix[~failed]
        for offset, curve in enumerate(healthy):
            ax.plot(x, curve, color=palette["teal"], alpha=.3, lw=.7, label="Individual surviving path" if offset == 0 else None)
        for offset, curve in enumerate(matrix[failed]):
            ax.plot(x, curve, color=palette["coral"], alpha=.4, lw=.8, label="Individual affected path" if offset == 0 else None)
        if len(healthy) >= 3:
            q05, q95 = np.quantile(healthy, [.05, .95], axis=0)
            ax.fill_between(x, q05, q95, color=palette["teal"], alpha=.12, label="Empirical survivor 5th–95th%")
        ax.plot(x, np.median(healthy, axis=0), color=palette["teal"], ls="--", lw=1.4, label="Same-run survivor median")
        ax.plot(x, np.median(matrix[failed], axis=0), color=palette["coral"], lw=2, label="Affected-path median")
        index = int(selected.first_operation_index)
        qualified = selected.evidence_role in {"initial_candidate", "same_transaction_confirmation", "downstream_support"}
        descriptive = bool(selected.get("curve_descriptive_only", False))
        role = ("Earliest selective candidate" if position == 0 else "Later selective contrast") if qualified else "Descriptive only: insufficient replication" if descriptive else "Illustrative only: no qualifying initial" if position == 0 else "Later illustrative contrast, not support"
        name = selected.step_label
        note = ("AUC / p / BH q not evaluated; descriptive curves only" if descriptive else
                f"AUC {selected.failed_path_deviation_auc:.3f}; BH q {selected.bh_fdr_q:.3g}")
        note += f"\n{int(failed.sum())} affected / {len(healthy)} surviving independent paths"
        note += "\nIndividual paths and group medians; retries are not replication"
        unit = "Baseline-centered raw sensor counts"
    else:
        curve = profile(selected.reference_observed_profile)
        ax.fill_between(x, profile(selected.reference_q05_profile), profile(selected.reference_q95_profile), color=palette["gold"], alpha=.3, label="Historical 5th–95th%")
        ax.plot(x, profile(selected.reference_median_profile), color=palette["ink"], ls="--", lw=1.4, label="Historical median")
        ax.plot(x, curve, color=palette["coral"], lw=2, label="Target operation")
        index = int(selected.operation_index)
        role = ("Earliest covered reference anomaly" if position == 0 else "Later reference anomaly, not proven support") if selected.reference_positive else "Reference-scored example: no anomaly"
        stage = str(selected.get("workflow_stage", "")).replace("_", " ")
        name = f"{stage} · {selected.process_step.replace('_', ' ')} · {selected.operation} {selected.programmed_volume_ul:g} µL · LC{selected.get('liquid_class_index', '?')} {selected.get('pipette', '?')} · {selected.location}"
        note = f"{int(selected.reference_physical_runs)} reference runs; percentile {selected.qpm_anomaly_percentile:.2f}\nOutside band {selected.outside_90pct_envelope_fraction:.0%}; " + ("cross-step match" if "cross_step" in selected.matched_reference_level else "step/location match")
        note += "\n" + nominal_volume_note(selected)
        unit = "Normalized shape (dimensionless)" if mode == "cross_instrument" else "Baseline-centered raw sensor counts"
    title = "\n".join(textwrap.wrap(f"Evidence {position + 1}/5 · {role} · op{index}", 53))
    title += "\n" + "\n".join(textwrap.wrap(name, 53)[:3])
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=8)
    ax.set(xlim=(0, 100), xlabel="Normalized operation progress (%)", ylabel=unit)
    # Wrap the complete annotation, including its prefix. A long abstention
    # reason must not collapse the three-column constrained layout or overflow
    # into its neighbour; keep the full reason rather than truncating it.
    note = "\n".join("\n".join(textwrap.wrap(line, 52)) for line in note.splitlines())
    annotation = ax.text(.02, .03, note, transform=ax.transAxes, fontsize=6.3, va="bottom",
                         bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    annotation.set_in_layout(False)
    ax.legend(fontsize=6.3, loc="best", framealpha=.9)


def figure_run_header(output: Path) -> str:
    """Use recorded report metadata without turning a display label into data."""
    path = output / "run_metadata.json"
    try:
        metadata = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    fields = [metadata.get("run_date"), metadata.get("instrument"),
              metadata.get("chip_type"),
              (f"lane {metadata['chip_lane']}" if metadata.get("chip_lane") else None),
              metadata.get("assay_type"), metadata.get("observation_id", output.name)]
    details = " · ".join(str(value).replace("\n", " ") for value in fields if value)
    return "\n".join(textwrap.wrap(details, width=160))


def plot_barcode_audit(ax, lineage, palette):
    """Render existing barcode calls, including a sparse single-dropout case.

    This is a view of the recorded outcome audit, not a reclassification.
    Missing reads are never drawn as zero reads. Log-scale zero placeholders
    are labeled with their actual recorded value.
    """
    ink = palette.get("ink", "#264653")
    ax.set_title("a  Barcode outcome audit · DNA and RNA evaluated separately",
                 loc="left", fontweight="bold", color=ink, fontsize=12, pad=12)
    required = {"planned_order", "material_type", "read_count", "is_control"}
    if lineage is None or lineage.empty or not required.issubset(lineage.columns):
        ax.text(.5, .5, "Barcode outcomes unavailable or unresolved\nNo sample dropout or survival pattern can be inferred",
                transform=ax.transAxes, ha="center", va="center", color=ink, fontsize=11)
        ax.set_axis_off()
        return
    controls = lineage.is_control.astype(str).str.lower().isin(["true", "1"])
    rows = lineage.loc[~controls].sort_values(["material_type", "planned_order"], kind="stable").copy()
    if rows.empty:
        ax.text(.5, .5, "No evaluable non-control sample barcodes\nControls are excluded from dropout and imbalance screening",
                transform=ax.transAxes, ha="center", va="center", color=ink, fontsize=11)
        ax.set_axis_off()
        return
    counts = pd.to_numeric(rows.read_count, errors="coerce")
    known = counts.notna() & np.isfinite(counts) & counts.ge(0)
    positions = np.arange(len(rows))
    material_colors = {"DNA": palette.get("blue", "#457B9D"), "RNA": palette.get("rna", "#8F5D8A")}
    colors = rows.material_type.map(material_colors).fillna(palette.get("gray", "#8A918B"))
    status = rows.get("path_status", pd.Series("unresolved", index=rows.index)).fillna("unresolved")
    affected = status.eq("failed")
    dropout = rows.get("apparent_dropout", pd.Series(False, index=rows.index)).astype(str).str.lower().isin(["true", "1"])
    edges = np.where(dropout, palette.get("coral", "#E76F51"),
                     np.where(affected, palette.get("orange", "#F4A261"), ink))
    if known.any():
        heights = counts.loc[known].clip(lower=.5).to_numpy(float)
        ax.bar(positions[known.to_numpy()], heights, color=colors.loc[known],
               edgecolor=edges[known.to_numpy()], linewidth=np.where(affected.loc[known], 2.2, .5), width=.72)
        for index in positions[known.to_numpy()]:
            if len(rows) <= 24 or affected.iloc[index]:
                value = float(counts.iloc[index])
                ax.text(index, max(value, .5) * 1.30, f"{value:,.0f}", ha="center", va="bottom",
                        fontsize=7.3, fontweight="bold" if affected.iloc[index] else "normal",
                        color=palette.get("coral", "#E76F51") if affected.iloc[index] else ink)
        maximum = max(float(counts.loc[known].max()), 1.)
    else:
        maximum = 1_000.
    for index in positions[~known.to_numpy()]:
        ax.text(index, .55, "unresolved", rotation=90, ha="center", va="bottom", fontsize=7,
                color=palette.get("gray", "#8A918B"))
    labels = [f"{str(row.material_type)} {row.planned_order:g}" if isinstance(row.planned_order, (int, float, np.number))
              else f"{row.material_type} {row.planned_order}" for row in rows.itertuples(index=False)]
    ax.set_xticks(positions, labels, fontsize=8)
    if len(rows) > 24:
        ax.tick_params(axis="x", rotation=90)
    ax.set(xlabel="Planned barcode order · controls excluded", ylabel="Final assigned reads (log scale)",
           yscale="log", ylim=(.35, max(maximum * 15, 1_000)), xlim=(-.65, len(rows) - .35))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.legend(handles=[Line2D([0], [0], color=material_colors["DNA"], lw=7, label="DNA"),
                       Line2D([0], [0], color=material_colors["RNA"], lw=7, label="RNA"),
                       Line2D([0], [0], color=palette.get("coral", "#E76F51"), lw=2, label="dropout outline"),
                       Line2D([0], [0], color=palette.get("orange", "#F4A261"), lw=2, label="imbalance outline")],
              fontsize=8, loc="upper right", framealpha=.9, ncol=4)
    n_failed = int(affected.sum())
    n_survivor = int(status.eq("survivor").sum())
    unresolved = int((~known | ~status.isin(["failed", "survivor"])).sum())
    ax.text(0, -.30, f"{len(rows)} non-control barcodes · {n_failed} affected / {n_survivor} surviving / {unresolved} unresolved · "
            f"{int(controls.sum())} controls excluded. Recorded calls unchanged; apparent dropout is not confirmed sample loss.",
            transform=ax.transAxes, fontsize=8, color=ink)


def plot_comparison(output, summary, tests, mapped, histories, palette, lineage=None):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(18, 30), layout="constrained", facecolor=palette.get("paper", "white"))
    grid = fig.add_gridspec(2 + EVIDENCE_WAVEFORM_SLOTS, 3,
                            height_ratios=[1.15, 1, *([1.1] * EVIDENCE_WAVEFORM_SLOTS)])
    outcome_axis = fig.add_subplot(grid[0, :])
    axes = np.array([[fig.add_subplot(grid[row + 1, column]) for column in range(3)]
                     for row in range(1 + EVIDENCE_WAVEFORM_SLOTS)])
    fig.get_layout_engine().set(w_pad=.12, h_pad=.14, wspace=.06, hspace=.08)
    fig.suptitle("Barcode outcomes, workflow scans and five QPM evidence events\n" + figure_run_header(output)
                 + "\nEr72 traces excluded · independent comparison views · sparse curve pairs are descriptive",
                 fontsize=14, fontweight="bold", color=palette["ink"])
    plot_barcode_audit(outcome_axis, lineage, palette)
    selections = {}
    for column, mode in enumerate(MODES):
        info = summary[summary.baseline_id.eq(mode)].iloc[0]
        ax = axes[0, column]
        ax.set_title(f"{'bcd'[column]}  " + LABELS[mode], loc="left", fontsize=11, fontweight="bold", color=palette["ink"])
        if mode == "within_run":
            eligible = tests[tests.eligible_internal_comparison.astype(bool)] if not tests.empty else tests
            if not eligible.empty:
                colors = np.where(eligible.evidence_role.isin(["initial_candidate", "same_transaction_confirmation", "downstream_support"]), palette["coral"], palette["gray"])
                ax.scatter(eligible.first_operation_index, eligible.failed_path_deviation_auc, c=colors, s=18)
                ax.axhline(.875, color=palette["gold"], ls=":", lw=1)
                ax.axhline(.5, color=palette["ink"], ls="--", lw=.8)
                ax.set(xlabel="Workflow operation group start", ylabel="Affected-path deviation AUC", ylim=(-.03, 1.03))
            else:
                descriptive_count = int(tests.curve_available.sum()) if "curve_available" in tests else 0
                explanation = (f"Statistical comparison under-replicated\n{descriptive_count} descriptive raw curve pairs available below\nNo AUC, p-value or q-value is manufactured"
                               if descriptive_count else "No available affected-versus-surviving curve pair\n" + info.status.replace("_", " "))
                ax.text(.5, .5, explanation, ha="center", va="center", transform=ax.transAxes, fontsize=9, wrap=True)
                ax.set_axis_off()
            curve_count = int(tests.curve_available.sum()) if "curve_available" in tests else 0
            note = f"{int(info.testable_groups)}/{int(info.operation_groups)} groups statistically testable; {int(info.positive_groups)} qualifying\n{curve_count} descriptive pairs available; history not required"
        else:
            frame = histories[mode]
            if not frame.empty:
                ax.scatter(frame.operation_index, frame.qpm_anomaly_percentile, c=np.where(frame.reference_positive, palette["coral"], palette["gray"]), s=12, alpha=.65)
                ax.axhline(.95, color=palette["gold"], ls=":", lw=1)
                ax.set(xlabel="Workflow operation index", ylabel="Historical shape-score percentile", ylim=(-.03, 1.03))
            else:
                ax.text(.5, .5, "No qualified reference population\n" + info.status.replace("_", " "), ha="center", va="center", transform=ax.transAxes, fontsize=9, wrap=True)
                ax.set_axis_off()
            note = f"{int(info.scored_operations)}/{int(info.target_operations)} operations covered; {int(info.positive_operations)} anomalies\n{int(info.reference_physical_runs)} unique reference runs / {int(info.reference_instruments)} instruments"
        ax.text(0, -.25, note, transform=ax.transAxes, fontsize=8, color=palette["ink"], va="top")
        selected = _display_rows(tests, histories.get(mode, pd.DataFrame()), mode)
        selections[mode] = []
        for position in range(EVIDENCE_WAVEFORM_SLOTS):
            panel = axes[position + 1, column]
            if position < len(selected):
                row = selected[position]
                _waveform(panel, row, mode, mapped, palette, position)
                selections[mode].append({"operation_index": int(row.first_operation_index if mode == "within_run" else row.operation_index),
                                         "evidence_role": row.evidence_role})
            else:
                panel.text(.5, .5, f"Evidence {position + 1}/5 — unavailable\nNo additional eligible waveform\nUncovered operations remain unresolved", ha="center", va="center", transform=panel.transAxes, fontsize=9)
                panel.set_axis_off()
            panel.grid(axis="y", alpha=.15)
    fig.supxlabel("Raw-count and normalized-shape panels have different units. No measured volume or failure probability is inferred.\nLater display favors distinct steps, then repeats, in chronological order; this is not a causal ranking.", fontsize=8, color=palette["ink"])
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"qpm_baseline_comparison.{extension}", dpi=180)
    plt.close(fig)
    return selections


def write_comparison(output_dir, lineage, evidence_by_material, mapped_by_material, target,
                     same_scores, same_coverage, same_members, same_metadata,
                     cross_scores, cross_coverage, cross_members, cross_metadata, config, palette):
    tests = pd.concat(evidence_by_material.values(), ignore_index=True) if evidence_by_material else pd.DataFrame()
    mapped = pd.concat(mapped_by_material.values(), ignore_index=True) if mapped_by_material else pd.DataFrame()
    histories = {"same_instrument": enrich_history(target, same_scores, config),
                 "cross_instrument": enrich_history(target, cross_scores, config)}
    coverages = {"same_instrument": same_coverage, "cross_instrument": cross_coverage}
    members = {"same_instrument": same_members, "cross_instrument": cross_members}
    metadata = {"same_instrument": same_metadata, "cross_instrument": cross_metadata}
    summary = summarize_baselines(lineage, tests, mapped, target, histories, coverages, members, metadata)
    display_tests = display_internal_tests(tests, mapped)
    available_curves = display_tests.curve_available.eq(True)
    summary.loc[summary.baseline_id.eq("within_run"), "available_curve_groups"] = int(available_curves.sum())
    summary.loc[summary.baseline_id.eq("within_run"), "descriptive_only_curve_groups"] = int(display_tests.curve_descriptive_only.eq(True).sum())
    summary.to_csv(output_dir / "qpm_baseline_comparison_summary.csv", index=False)
    tests.to_csv(output_dir / "qpm_baseline_within_run_steps.csv", index=False)
    display_tests.to_csv(output_dir / "qpm_same_run_curve_coverage.csv", index=False)
    for mode in MODES[1:]:
        frame = histories[mode]
        frame.to_parquet(output_dir / f"qpm_baseline_{mode}_waveforms.parquet", index=False)
        compact = frame.drop(columns=[c for c in frame if "profile" in c and c != "reference_profile_unit"], errors="ignore")
        compact.to_csv(output_dir / f"qpm_baseline_{mode}_evidence.csv", index=False)
        coverages[mode].to_csv(output_dir / f"qpm_baseline_{mode}_coverage.csv", index=False)
        members[mode].to_csv(output_dir / f"qpm_baseline_{mode}_members.csv", index=False)
    selections = plot_comparison(output_dir, summary, display_tests, mapped, histories, palette, lineage=lineage)
    display_slots(display_tests, histories).to_csv(output_dir / "qpm_displayed_evidence.csv", index=False)
    payload = {"version": VERSION, "always_evaluated_independently": True,
               "same_and_cross_history_never_overlaid_in_this_audit": True,
               "cross_instrument_excludes_target_instrument": True,
               "primary_outcome_thresholds_and_internal_selection_unchanged": False,
               "internal_selection_unchanged": True,
               "same_run_curve_display": dict(version=VERSION, minimum_affected_paths_for_display=1,
                    minimum_surviving_paths_for_display=1, statistical_replication_thresholds_unchanged=True,
                    displayed_raw_curves_do_not_establish_significance=True,
                    available_groups=int(available_curves.sum()),
                    descriptive_only_groups=int(display_tests.curve_descriptive_only.eq(True).sum()),
                    coverage_file=str(output_dir / "qpm_same_run_curve_coverage.csv"),
                    display_policy="qualified_chain_unchanged; if no statistically eligible groups, chronological_descriptive_pairs_then_distinct_later_steps"),
               "reference_populations_outcome_qualified_not_certified_healthy": True,
               "outcome_first_layout": dict(barcode_outcomes_first=True, scan_columns=3, evidence_rows=5,
                    total_panels=19, controls_excluded=True, barcode_source="barcode_lineage.csv",
                    evidence_selection_unchanged=True),
               "evidence_waveform_slots_per_baseline": EVIDENCE_WAVEFORM_SLOTS,
               "display_selection_policy": "earliest_then_distinct_later_steps_then_repeats; chronological; never_pad_qualified_with_illustrative",
               "summary": summary.to_dict("records"), "displayed_operations": selections,
               "same_instrument": same_metadata, "cross_instrument": cross_metadata,
               "figure_png": str(output_dir / "qpm_baseline_comparison.png"),
               "figure_pdf": str(output_dir / "qpm_baseline_comparison.pdf"),
               "figure_svg": str(output_dir / "qpm_baseline_comparison.svg")}
    (output_dir / "qpm_baseline_comparison_manifest.json").write_text(json.dumps(_json(payload), indent=2, allow_nan=False))
    return payload


def html_section(output: Path, payload: dict) -> str:
    summary = pd.DataFrame(payload["summary"])
    columns = ["baseline", "status", "target_operations", "scored_operations", "testable_groups",
               "positive_groups", "positive_operations", "reference_physical_runs", "reference_instruments",
               "earliest_operation_index", "profile_unit"]
    table = summary.reindex(columns=columns).fillna("").to_html(index=False, border=0, classes="data-table", escape=True)
    encoded = base64.b64encode(Path(payload["figure_png"]).read_bytes()).decode("ascii")
    sections = []
    for mode in MODES[1:]:
        frame = pd.read_parquet(output / f"qpm_baseline_{mode}_waveforms.parquet")
        positives = frame[frame.reference_positive.astype(bool)] if not frame.empty else frame
        fields = ["operation_index", "workflow_stage", "process_step", "location", "operation", "programmed_volume_ul",
                  "liquid_class_name", "reference_physical_runs", "reference_instruments", "matched_reference_level",
                  "qpm_anomaly_percentile", "outside_90pct_envelope_fraction", "evidence_role"]
        detail = positives.reindex(columns=fields).head(50).to_html(index=False, border=0, classes="data-table", escape=True) if not positives.empty else "<p>No qualified reference anomalies. Inspect coverage before interpreting this as evidence of normality.</p>"
        sections.append(f"<h3>{html.escape(LABELS[mode])}</h3><p>Showing up to 50 anomalies; complete scores, unresolved operations and membership are downloadable.</p>{detail}<p class='files'><a href='qpm_baseline_{mode}_evidence.csv'>All scored operations</a><a href='qpm_baseline_{mode}_coverage.csv'>Coverage and exclusions</a><a href='qpm_baseline_{mode}_members.csv'>Exact reference runs</a></p>")
    return f"""<section id="parallel-baselines"><h2>All three QPM comparisons — evaluated independently</h2>
<p>The same-run analysis uses only this run's surviving paths, with no historical baseline. Same-instrument history and other-instrument history are both evaluated for every usable target operation, even with adequate same-instrument history, adequate internal replication, or no outcome trigger. Unavailable populations remain explicit. This audit does not substitute historical anomalies for barcode-selective evidence or replace the primary route.</p>
<p class="guard">Same-instrument curves use baseline-centered raw sensor counts. Other instruments contribute normalized dimensionless shapes only; the target instrument is excluded. Both historical populations must first pass the user-specified chip/lane read minimum and have no detected barcode dropout or imbalance. Matching definitions, command modes, geometry, pipette, operation, volume and sensor units remain required. Cross-step matches are labeled. Target/sibling runs are excluded; a physical run has one vote. Outcome-qualified does not mean certified fault-free. AUC, empirical percentiles and reference counts are different quantities—not interchangeable significance tests or measured delivery errors.</p>
<img src="data:image/png;base64,{encoded}" alt="Barcode outcomes above workflow scans and five QPM evidence rows for independent same-run, same-instrument and cross-instrument comparisons">
<h3>Comparison coverage and statistical availability</h3>{table}
<p>Each column can nominate a different earliest <em>covered</em> anomaly. These are separate hypotheses, not a causal ranking or three independent votes. The same-run column can display individual affected and surviving curves and their medians with as few as one independent path per group. Sparse plots are explicitly descriptive, without manufactured AUC, p-value, q-value, or statistical support. Missing affected/surviving curves or unverified physical configuration remain unresolved. No compatible outcome peers is not normal performance.</p>
<p><strong>Five evidence slots per comparison:</strong> earliest qualifying result plus up to four later steps. Later selection favors distinct workflow steps before repeated steps, then displays them chronologically. Qualified chains are not padded with descriptive curves. If no within-run group is statistically eligible, available raw curve pairs are shown chronologically as descriptive examples, not earliest failure or later support. Empty slots are not evidence of normality. Historical anomalies are reference-only, not a demonstrated barcode failure chain.</p>
<p class="files"><a href="qpm_displayed_evidence.csv">All 15 evidence slots, roles and unavailable results</a></p>
<p class="files"><a href="qpm_same_run_curve_coverage.csv">Every same-run group's curve availability, path counts and exclusion reason</a></p>
<p class="files"><a href="qpm_baseline_comparison_summary.csv">Three-comparison summary</a><a href="qpm_baseline_comparison_manifest.json">Policies and provenance</a><a href="qpm_baseline_comparison.pdf">PDF figure</a><a href="qpm_baseline_comparison.svg">Editable SVG figure</a><a href="qpm_baseline_comparison.png">PNG figure</a><a href="qpm_baseline_within_run_steps.csv">All within-run step tests</a></p>
{''.join(sections)}</section>"""
