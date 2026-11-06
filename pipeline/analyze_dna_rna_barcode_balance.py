#!/usr/bin/env python3
"""Analyze planned DNA and RNA barcode dropout separately.

Only privacy-safe hashed barcode/pair identifiers are written.  Material type
comes from the per-row ``relations`` field in CSA/analysisSamples.json.  This is
important for DNA_RNA workflows because the merged barcode joins a DNA barcode
and an RNA barcode; summing that pair would hide material-specific dropout.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from find_qpm_failure_examples import (
    BARCODE_RE,
    barcode_count_table,
    is_control_value,
    load_run_data,
    normalize_barcode,
    read_root_analysis_samples,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "qpm_failure_examples"
OUT.mkdir(exist_ok=True)


def list_value(document: dict, key: str, index: int) -> object:
    values = document.get(key) or []
    return values[index] if index < len(values) else None


def material_type(document: dict, index: int) -> str:
    # ``relations`` is authoritative for paired DNA_RNA workflows.  In sampled
    # archives nucleicAcidTypes was sometimes "DNA" for both paired rows.
    relation = str(list_value(document, "relations", index) or "").strip().upper()
    if relation in {"DNA", "RNA"}:
        return relation
    nucleic = str(list_value(document, "nucleicAcidTypes", index) or "").strip().upper()
    return nucleic if nucleic in {"DNA", "RNA"} else "UNKNOWN"


def safe_barcode_plan(archive_id: str, document: dict) -> list[dict]:
    barcode_ids = document.get("barcodeIds") or []
    merged_ids = document.get("mergedBarcodeIds") or []
    rows: dict[tuple[str, str], dict] = {}
    for index, raw_barcode in enumerate(barcode_ids):
        barcode_tokens = tuple(
            dict.fromkeys(normalize_barcode(item) for item in BARCODE_RE.findall(str(raw_barcode)))
        )
        if not barcode_tokens:
            continue
        material = material_type(document, index)
        merged_raw = merged_ids[index] if index < len(merged_ids) else raw_barcode
        pair_tokens = tuple(
            dict.fromkeys(normalize_barcode(item) for item in BARCODE_RE.findall(str(merged_raw)))
        ) or barcode_tokens

        sensitive_control_fields = [
            list_value(document, key, index)
            for key in ("sampleTypes", "sampleExternalIds", "sampleNames", "controlNames")
        ]
        control = any(is_control_value(value) for value in sensitive_control_fields)
        barcode_hash = hashlib.sha256(
            f"{archive_id}|{material}|{'|'.join(barcode_tokens)}".encode("utf-8")
        ).hexdigest()[:12]
        merged_pool_hash = hashlib.sha256(
            f"{archive_id}|{'|'.join(sorted(pair_tokens))}".encode("utf-8")
        ).hexdigest()[:12]
        key = (material, "|".join(barcode_tokens))
        rows[key] = {
            "archive_id": archive_id,
            "barcode_id": f"barcode_{barcode_hash}",
            "merged_pool_id": f"pool_{merged_pool_hash}",
            "material_type": material,
            "planned_order": index + 1,
            "barcode_tokens": list(barcode_tokens),
            "is_control": bool(control),
        }
    return list(rows.values())


def extract_one(record: dict) -> tuple[list[dict], dict | None]:
    try:
        document = read_root_analysis_samples(record["local_path"])
        return safe_barcode_plan(record["archive_id"], document), None
    except Exception as exc:
        return [], {
            "archive_id": record["archive_id"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def extract_plans(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    failures: list[dict] = []
    records = primary[["archive_id", "local_path"]].to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_one, record) for record in records]
        for completed, future in enumerate(as_completed(futures), start=1):
            plan, failure = future.result()
            rows.extend(plan)
            if failure is not None:
                failures.append(failure)
            if completed % 25 == 0 or completed == len(records):
                print(f"DNA/RNA plan metadata: {completed}/{len(records)} archives")
    return pd.DataFrame(rows), pd.DataFrame(failures)


def attach_read_counts(
    runs: pd.DataFrame, plan: pd.DataFrame, barcode_counts: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    run_lookup = runs.set_index("archive_id", drop=False)
    inventory_lookup = inventory.set_index("archive_id")
    rows = []
    for item in plan.itertuples(index=False):
        if item.archive_id not in barcode_counts.index or item.archive_id not in run_lookup.index:
            continue
        values = [barcode_counts.at[item.archive_id, token] for token in item.barcode_tokens]
        numeric = [float(value) for value in values if pd.notna(value)]
        read_count = float(np.sum(numeric)) if numeric else np.nan
        run = run_lookup.loc[item.archive_id]
        inv = inventory_lookup.loc[item.archive_id] if item.archive_id in inventory_lookup.index else None
        rows.append(
            {
                "observation_id": run["observation_id"],
                "physical_run_id": run["physical_run_id"],
                "instrument_id": run["instrument_id"],
                "chip_lane": run["chip_lane"],
                "run_start_time": run["run_start_time"],
                "archive_id": item.archive_id,
                "hostname": inv["hostname"] if inv is not None else "",
                "run_name": inv["run_name"] if inv is not None else "",
                "barcode_id": item.barcode_id,
                "merged_pool_id": item.merged_pool_id,
                "material_type": item.material_type,
                "planned_order": item.planned_order,
                "is_control": item.is_control,
                "read_count": read_count,
                "count_status": "reported_by_ionstats" if numeric else "planned_barcode_absent_from_ionstats",
                "source": "CSA/analysisSamples.json relations + barcodeIds; ionstats_basecaller read_group_metrics",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Ionstats can omit a planned barcode entirely instead of emitting an
    # explicit zero. Treat that absence as zero only when the same archive and
    # material has at least one *non-control* planned barcode with a valid
    # count. An all-missing material group remains unresolved rather than being
    # manufactured into an all-zero failure.
    noncontrol = result[~result["is_control"]]
    reported_by_group = noncontrol.groupby(["archive_id", "material_type"])[
        "read_count"
    ].apply(lambda values: values.notna().any())
    group_keys = pd.MultiIndex.from_frame(result[["archive_id", "material_type"]])
    same_material_has_counts = pd.Series(
        reported_by_group.reindex(group_keys, fill_value=False).to_numpy(),
        index=result.index,
    )
    assumed_zero = result["read_count"].isna() & same_material_has_counts
    result.loc[assumed_zero, "read_count"] = 0.0
    result.loc[assumed_zero, "count_status"] = "planned_barcode_absent_assumed_zero"
    return result


def shannon_evenness(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) < 2 or values.sum() <= 0:
        return np.nan
    proportions = values / values.sum()
    proportions = proportions[proportions > 0]
    return float(-(proportions * np.log(proportions)).sum() / np.log(len(values)))


def classify_barcodes(barcodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = barcodes.copy()
    data["apparent_dropout"] = False
    summaries = []
    eligible = data[
        (~data["is_control"])
        & data["material_type"].isin(["DNA", "RNA"])
        & data["read_count"].notna()
    ]
    for (observation_id, material), group in eligible.groupby(
        ["observation_id", "material_type"], sort=False
    ):
        values = group["read_count"].to_numpy(float)
        median = float(np.median(values))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        threshold = max(100.0, 0.15 * median)
        dropout_mask = group["read_count"] < threshold
        data.loc[group.index, "apparent_dropout"] = dropout_mask
        mad = float(np.median(np.abs(values - median)))
        summaries.append(
            {
                "observation_id": observation_id,
                "material_type": material,
                "n_planned_noncontrol_barcodes": len(values),
                "total_reads": float(values.sum()),
                "median_reads": median,
                "minimum_reads": minimum,
                "maximum_reads": maximum,
                "dropout_threshold_reads": threshold,
                "apparent_dropout_count": int(dropout_mask.sum()),
                "min_to_median_ratio": minimum / median if median > 0 else np.nan,
                "max_to_min_ratio": maximum / minimum if minimum > 0 else np.inf,
                "robust_cv": 1.4826 * mad / median if median > 0 else np.nan,
                "shannon_evenness": shannon_evenness(values),
            }
        )
    return data, pd.DataFrame(summaries)


def merged_pool_table(barcodes: pd.DataFrame) -> pd.DataFrame:
    data = barcodes[
        (~barcodes["is_control"])
        & barcodes["material_type"].isin(["DNA", "RNA"])
        & barcodes["read_count"].notna()
    ].copy()
    if data.empty:
        return pd.DataFrame()
    pools = (
        data.groupby(["observation_id", "merged_pool_id", "material_type"], sort=False)
        .agg(
            n_planned_barcodes=("barcode_id", "nunique"),
            merged_pool_reads=("read_count", "sum"),
            minimum_barcode_reads=("read_count", "min"),
            maximum_barcode_reads=("read_count", "max"),
            barcode_dropout_count=("apparent_dropout", "sum"),
            absent_assumed_zero_count=(
                "count_status",
                lambda values: int((values == "planned_barcode_absent_assumed_zero").sum()),
            ),
        )
        .reset_index()
    )
    pools["dropout_pattern"] = np.select(
        [pools["barcode_dropout_count"] == 1, pools["barcode_dropout_count"] >= 2],
        ["single_barcode", "multiple_barcodes"],
        default="none",
    )
    return pools


def run_summary(
    material_summary: pd.DataFrame, merged_pools: pd.DataFrame, barcodes: pd.DataFrame
) -> pd.DataFrame:
    dna = material_summary[material_summary["material_type"] == "DNA"].drop(columns="material_type")
    rna = material_summary[material_summary["material_type"] == "RNA"].drop(columns="material_type")
    dna = dna.rename(columns={c: f"dna_{c}" for c in dna.columns if c != "observation_id"})
    rna = rna.rename(columns={c: f"rna_{c}" for c in rna.columns if c != "observation_id"})
    summary = dna.merge(rna, on="observation_id", how="outer")
    identity = barcodes[
        [
            "observation_id",
            "physical_run_id",
            "instrument_id",
            "chip_lane",
            "run_start_time",
            "archive_id",
            "hostname",
            "run_name",
        ]
    ].drop_duplicates("observation_id")
    summary = identity.merge(summary, on="observation_id", how="right")
    summary["rna_to_dna_median_ratio"] = (
        summary["rna_median_reads"] / summary["dna_median_reads"].replace(0, np.nan)
    )
    summary["rna_to_dna_total_ratio"] = (
        summary["rna_total_reads"] / summary["dna_total_reads"].replace(0, np.nan)
    )
    if not merged_pools.empty:
        pattern_counts = (
            merged_pools.groupby(["observation_id", "material_type", "dropout_pattern"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        pattern_counts.columns = [
            column if column in {"observation_id", "material_type"} else f"merged_pool_{column}_count"
            for column in pattern_counts.columns
        ]
        for material in ["DNA", "RNA"]:
            material_patterns = pattern_counts[pattern_counts["material_type"] == material].drop(
                columns="material_type"
            )
            material_patterns = material_patterns.rename(
                columns={
                    column: f"{material.lower()}_{column}"
                    for column in material_patterns.columns
                    if column != "observation_id"
                }
            )
            summary = summary.merge(material_patterns, on="observation_id", how="left")
    for material in ["dna", "rna"]:
        counts = summary[f"{material}_apparent_dropout_count"].fillna(0)
        summary[f"{material}_dropout_class"] = np.select(
            [counts == 1, counts >= 2], ["single", "multiple"], default="none"
        )
    return summary


def make_examples(summary: pd.DataFrame) -> pd.DataFrame:
    examples = []
    for material in ["dna", "rna"]:
        for dropout_class in ["single", "multiple"]:
            subset = summary[summary[f"{material}_dropout_class"] == dropout_class].copy()
            if subset.empty:
                continue
            row = subset.sort_values(
                [f"{material}_min_to_median_ratio", f"{material}_shannon_evenness"],
                ascending=[True, True],
            ).iloc[0]
            examples.append(
                {
                    **row.to_dict(),
                    "example_type": f"{dropout_class}_{material}_dropout",
                    "material": material.upper(),
                    "selection_metric": "min_to_median_ratio",
                    "selection_value": row[f"{material}_min_to_median_ratio"],
                }
            )
        subset = summary[
            (summary[f"{material}_apparent_dropout_count"].fillna(0) == 0)
            & (summary[f"{material}_n_planned_noncontrol_barcodes"] >= 4)
        ].copy()
        if not subset.empty:
            row = subset.sort_values(f"{material}_max_to_min_ratio", ascending=False).iloc[0]
            examples.append(
                {
                    **row.to_dict(),
                    "example_type": f"{material}_imbalance_without_dropout",
                    "material": material.upper(),
                    "selection_metric": "max_to_min_ratio",
                    "selection_value": row[f"{material}_max_to_min_ratio"],
                }
            )

    paired = summary[summary["rna_to_dna_median_ratio"].notna()].copy()
    if not paired.empty:
        low = paired.sort_values("rna_to_dna_median_ratio").iloc[0]
        high = paired.sort_values("rna_to_dna_median_ratio", ascending=False).iloc[0]
        for label, row in [
            ("lowest_rna_to_dna_median_ratio", low),
            ("highest_rna_to_dna_median_ratio", high),
        ]:
            examples.append(
                {
                    **row.to_dict(),
                    "example_type": label,
                    "material": "DNA_RNA",
                    "selection_metric": "rna_to_dna_median_ratio",
                    "selection_value": row["rna_to_dna_median_ratio"],
                }
            )
    result = pd.DataFrame(examples)
    if (OUT / "qpm_anomaly_burden.csv").exists() and not result.empty:
        burden = pd.read_csv(OUT / "qpm_anomaly_burden.csv")
        result = result.merge(burden, on="observation_id", how="left")
    return result


def make_report(summary: pd.DataFrame, examples: pd.DataFrame) -> None:
    lines = [
        "# DNA/RNA barcode dropout and imbalance",
        "",
        "DNA and RNA are analyzed separately using the individual planned barcode and the `relations` field. Merged barcode pools are same-material pools and are not used to combine DNA with RNA.",
        "",
        "## Definitions",
        "",
        "- Only planned, non-control barcodes are evaluated.",
        "- A planned barcode absent from ionstats is counted as zero only if the same archive has another reported non-control barcode of the same material; the assumption is retained in `count_status`. All-missing material groups remain unresolved.",
        "- Apparent dropout: reads below max(100, 15% of the median for the same material in the same observation).",
        "- Single dropout: exactly one apparent dropout for that material; multiple dropout: two or more.",
        "- Imbalance without dropout examples maximize max/min reads among runs with at least four planned barcodes and no apparent dropout.",
        "- These are read-count phenotypes. They do not prove a pipetting or QPM cause, and intended unequal input proportions were unavailable.",
        "",
        "## Coverage",
        "",
        f"- Observations with DNA metrics: {summary['dna_n_planned_noncontrol_barcodes'].notna().sum()}",
        f"- Observations with RNA metrics: {summary['rna_n_planned_noncontrol_barcodes'].notna().sum()}",
        f"- DNA single-dropout observations: {(summary['dna_dropout_class'] == 'single').sum()}",
        f"- DNA multiple-dropout observations: {(summary['dna_dropout_class'] == 'multiple').sum()}",
        f"- RNA single-dropout observations: {(summary['rna_dropout_class'] == 'single').sum()}",
        f"- RNA multiple-dropout observations: {(summary['rna_dropout_class'] == 'multiple').sum()}",
        "",
        "## Most extreme examples",
        "",
    ]
    for row in examples.itertuples(index=False):
        lines.extend(
            [
                f"### {row.example_type}: `{row.observation_id}`",
                "",
                f"- Run: `{row.run_name}`",
                f"- Selection metric: {row.selection_metric} = {row.selection_value:.4g}",
                f"- DNA: n={row.dna_n_planned_noncontrol_barcodes:.0f}, dropouts={row.dna_apparent_dropout_count:.0f}, median={row.dna_median_reads:,.0f}, min={row.dna_minimum_reads:,.0f}, max/min={row.dna_max_to_min_ratio:.2f}",
                f"- RNA: n={row.rna_n_planned_noncontrol_barcodes:.0f}, dropouts={row.rna_apparent_dropout_count:.0f}, median={row.rna_median_reads:,.0f}, min={row.rna_minimum_reads:,.0f}, max/min={row.rna_max_to_min_ratio:.2f}",
                "",
            ]
        )
    (OUT / "dna_rna_balance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    duplicate = pd.read_csv(ROOT / "duplicate_archives.csv")
    manifest = pd.read_csv(ROOT / "csa_transfer_manifest.csv")
    inventory = pd.read_csv(ROOT / "csa_inventory.csv")
    retained = duplicate[duplicate["retained_primary"].astype(str).str.lower().eq("true")]
    primary = retained[["archive_id", "observation_id"]].merge(
        manifest[["archive_id", "local_path", "state"]], on="archive_id", how="left"
    )
    primary = primary[primary["state"].isin(["transferred", "already_present"])]

    plan, failures = extract_plans(primary)
    runs, count_columns = load_run_data()
    counts = barcode_count_table(runs, count_columns)
    barcodes = attach_read_counts(runs, plan, counts, inventory)
    barcodes, material_summary = classify_barcodes(barcodes)
    merged_pools = merged_pool_table(barcodes)
    summary = run_summary(material_summary, merged_pools, barcodes)
    examples = make_examples(summary)

    # Drop internal raw barcode tokens before writing any table.
    plan.drop(columns=["barcode_tokens"], errors="ignore").to_csv(
        OUT / "dna_rna_planned_barcodes.csv", index=False
    )
    barcodes.to_csv(OUT / "dna_rna_barcode_counts.csv", index=False)
    material_summary.to_csv(OUT / "dna_rna_material_balance.csv", index=False)
    merged_pools.to_csv(OUT / "dna_rna_merged_pool_balance.csv", index=False)
    # Retain the earlier filename as a compatibility copy, now with corrected
    # same-material merged-pool semantics.
    merged_pools.to_csv(OUT / "dna_rna_paired_barcode_balance.csv", index=False)
    summary.to_csv(OUT / "dna_rna_balance_by_run.csv", index=False)
    examples.to_csv(OUT / "dna_rna_dropout_examples.csv", index=False)
    failures.to_csv(OUT / "dna_rna_parse_failures.csv", index=False)
    make_report(summary, examples)
    print((OUT / "dna_rna_balance_report.md").read_text())


if __name__ == "__main__":
    main()
