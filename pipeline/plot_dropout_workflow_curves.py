#!/usr/bin/env python3
"""Match barcode dropout outcomes to positional QPM workflow curves.

This analysis deliberately separates three levels of evidence:

1. direct barcode-well evidence (adapter-ligation operations executed at the
   exact barcode plate position assigned to a planned barcode),
2. workflow-block evidence (target-amplification operations shared by a deck
   configuration / primer-pool block), and
3. run-level evidence (operations that cannot be assigned to one barcode).

Identifying sample fields are used only in memory to validate the positional
join and are never written.  QPM units remain raw sensor counts because the CSA
does not provide a validated physical-unit conversion.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tarfile
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from find_qpm_failure_examples import read_root_analysis_samples


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "qpm_failure_examples"

OBSERVATIONS = [
    "obs_fef8b0992599",
    "obs_32eade743d80",
    "obs_327d7cbf172b",
]

PALETTE = {
    "ink": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#457B9D",
    "rna": "#8F5D8A",
    "gray": "#8A918B",
    "pale": "#F4F1DE",
}


def read_json_member(path: Path, wanted: str) -> dict:
    """Read one JSON member from a ZIP or compressed tar without extraction."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist() if name.lstrip("./") == wanted
            ]
            if not candidates:
                raise KeyError(f"{wanted} not found in {path}")
            return json.loads(archive.read(candidates[0]))
    with tarfile.open(path, "r:*") as archive:
        member = next(
            (item for item in archive.getmembers() if item.name.lstrip("./") == wanted),
            None,
        )
        if member is None:
            raise KeyError(f"{wanted} not found in {path}")
        handle = archive.extractfile(member)
        if handle is None:
            raise OSError(f"could not read {wanted} from {path}")
        return json.load(handle)


def archive_paths() -> dict[str, Path]:
    barcodes = pd.read_csv(OUT / "dna_rna_barcode_counts.csv")
    manifest = pd.read_csv(ROOT / "csa_transfer_manifest.csv")
    selected = barcodes[barcodes["observation_id"].isin(OBSERVATIONS)][
        ["observation_id", "archive_id"]
    ].drop_duplicates()
    selected = selected.merge(
        manifest[["archive_id", "local_path"]], on="archive_id", how="left", validate="one_to_one"
    )
    if selected["local_path"].isna().any():
        raise ValueError("missing local CSA path for at least one dropout observation")
    return {
        row.observation_id: Path(row.local_path)
        for row in selected.itertuples(index=False)
    }


def positional_mapping(observation_id: str, path: Path) -> pd.DataFrame:
    """Build and validate the planned-order to physical-well mapping.

    Root ``analysisSamples.json`` and ``planned_run.json`` have aligned rows.
    The equality of analysisSamples.sampleExternalIds and
    planned_run.sampleBarcodeMapping.sampleId is checked at every position,
    but neither identifying field is retained.
    """
    analysis = read_root_analysis_samples(str(path))
    planned = read_json_member(path, "CSA/planned_run.json").get("object", {})
    rows = []
    flat_index = 0
    external_ids = analysis.get("sampleExternalIds") or []
    for block_index, deck in enumerate(planned.get("deckConfig") or [], start=1):
        for within_block_index, mapping in enumerate(
            deck.get("sampleBarcodeMapping") or [], start=1
        ):
            external = external_ids[flat_index] if flat_index < len(external_ids) else None
            planned_id = mapping.get("sampleId")
            rows.append(
                {
                    "observation_id": observation_id,
                    "planned_order": flat_index + 1,
                    "workflow_block": block_index,
                    "within_block_order": within_block_index,
                    "planned_material_type": str(mapping.get("sampleType") or "").upper(),
                    "barcode_plate_well": str(mapping.get("barcodePlatePosition") or "").upper(),
                    "sample_plate_well": str(mapping.get("samplePlatePosition") or "").upper(),
                    "primer_pool_position": str(deck.get("primerPoolPosition") or "").upper(),
                    "positional_join_validated": bool(
                        external is not None
                        and planned_id is not None
                        and str(external) == str(planned_id)
                    ),
                    "mapping_source": (
                        "CSA/analysisSamples.json sampleExternalIds row order validated against "
                        "CSA/planned_run.json deckConfig.sampleBarcodeMapping.sampleId; "
                        "identifiers not retained"
                    ),
                }
            )
            flat_index += 1
    mapping = pd.DataFrame(rows)
    if mapping.empty or not mapping["positional_join_validated"].all():
        raise ValueError(f"unvalidated planned barcode/well mapping for {observation_id}")
    return mapping


def parse_profile(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        return np.array([], dtype=float)
    parsed = ast.literal_eval(str(value))
    return np.asarray(parsed, dtype=float)


def prepare_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    barcodes = pd.read_csv(OUT / "dna_rna_barcode_counts.csv")
    scores = pd.read_csv(OUT / "dna_rna_dropout_qpm_scores.csv")
    paths = archive_paths()

    mappings = pd.concat(
        [positional_mapping(observation_id, path) for observation_id, path in paths.items()],
        ignore_index=True,
    )
    selected = barcodes[
        barcodes["observation_id"].isin(OBSERVATIONS) & ~barcodes["is_control"].astype(bool)
    ].copy()
    selected = selected.merge(
        mappings,
        on=["observation_id", "planned_order"],
        how="left",
        validate="many_to_one",
    )
    if selected["barcode_plate_well"].isna().any():
        raise ValueError("at least one planned barcode did not map to a barcode plate well")
    selected["qpm_location"] = "barcode_" + selected["barcode_plate_well"].str.lower()

    ligation = scores[
        scores["observation_id"].isin(OBSERVATIONS)
        & scores["process_step"].str.contains(
            r"ligate_adapters_(?:dispense_digested_sample_to_barcode|aspirate_digested_sample_and_resuspended_barcode)",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()
    ligation = ligation.merge(
        selected[
            [
                "observation_id",
                "barcode_id",
                "material_type",
                "planned_order",
                "read_count",
                "apparent_dropout",
                "workflow_block",
                "barcode_plate_well",
                "sample_plate_well",
                "qpm_location",
            ]
        ],
        left_on=["observation_id", "location"],
        right_on=["observation_id", "qpm_location"],
        how="inner",
        validate="many_to_one",
    )
    ligation["match_scope"] = "direct_barcode_plate_well"
    ligation["candidate_step"] = np.where(
        ligation["operation"].eq("dispense"),
        "adapter_ligation.dispense_sample_into_barcode_well_and_mix",
        "adapter_ligation.aspirate_ligated_sample_from_barcode_well",
    )

    block = scores[
        scores["observation_id"].isin(OBSERVATIONS)
        & scores["process_step"].str.contains(
            r"target_amp_transfer_(?:21u_l_of_dna|15u_l_of_rna)_at_mastermix_from_dnainline_ctrl_[1-4]",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()
    block["workflow_block"] = pd.to_numeric(
        block["process_step"].str.extract(r"dnainline_ctrl_([1-4])", expand=False),
        errors="coerce",
    ).astype("Int64")
    block["material_type"] = np.where(
        block["process_step"].str.contains("_dna_", case=False, na=False), "DNA", "RNA"
    )
    block["match_scope"] = "workflow_block"
    block["candidate_step"] = np.where(
        block["material_type"].eq("DNA"),
        "target_amplification.DNA_AT_mastermix_transfer",
        "target_amplification.RNA_AT_mastermix_transfer",
    )
    dropout_by_block = (
        selected.groupby(["observation_id", "workflow_block", "material_type"], as_index=False)
        .agg(
            planned_barcodes=("barcode_id", "size"),
            dropout_barcodes=("apparent_dropout", "sum"),
            median_reads=("read_count", "median"),
        )
    )
    block = block.merge(
        dropout_by_block,
        on=["observation_id", "workflow_block", "material_type"],
        how="left",
        validate="many_to_one",
    )
    block["apparent_dropout"] = block["dropout_barcodes"].fillna(0).gt(0)

    # The second DNA block in obs_fef8b0992599 has an earlier selective event:
    # UDG1 feeds reaction wells 1-6 (block 1) and UDG2 feeds wells 7-12
    # (block 2). These operations precede target-amplification mastermix and
    # adapter ligation.
    udg = scores[
        scores["observation_id"].isin(OBSERVATIONS)
        & scores["process_step"].str.contains(
            r"udg_transfer_5u_l_from_udg_[12]_to_at_reaction_block1_well_",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()
    udg["workflow_block"] = pd.to_numeric(
        udg["process_step"].str.extract(r"from_udg_([12])", expand=False),
        errors="coerce",
    ).astype("Int64")
    udg["material_type"] = "DNA"
    udg["match_scope"] = "workflow_block"
    udg["candidate_step"] = "target_amplification.UDG_transfer_to_DNA_reaction_block"
    udg = udg.merge(
        dropout_by_block,
        on=["observation_id", "workflow_block", "material_type"],
        how="left",
        validate="many_to_one",
    )
    udg["apparent_dropout"] = udg["dropout_barcodes"].fillna(0).gt(0)

    # The debug log also preserves the RNA reaction-well source at adapter
    # ligation. PCR A9-E9 are workflow block 3 and PCR G9-C10 are block 4.
    reaction_ligation = scores[
        scores["observation_id"].isin(OBSERVATIONS)
        & scores["process_step"].str.contains(
            "ligate_adapters_aspirate_all_digested_sample_and_oil_from_well_at_reac",
            case=False,
            regex=False,
            na=False,
        )
    ].copy()
    reaction_ligation["workflow_block"] = pd.Series(
        pd.NA, index=reaction_ligation.index, dtype="Int64"
    )
    block3_wells = {"pcr_a9", "pcr_b9", "pcr_c9", "pcr_d9", "pcr_e9"}
    block4_wells = {"pcr_g9", "pcr_h9", "pcr_a10", "pcr_b10", "pcr_c10"}
    reaction_ligation.loc[
        reaction_ligation["location"].isin(block3_wells), "workflow_block"
    ] = 3
    reaction_ligation.loc[
        reaction_ligation["location"].isin(block4_wells), "workflow_block"
    ] = 4
    reaction_ligation = reaction_ligation[
        reaction_ligation["workflow_block"].notna()
    ].copy()
    reaction_ligation["material_type"] = "RNA"
    reaction_ligation["match_scope"] = "workflow_block_reaction_well"
    reaction_ligation["candidate_step"] = (
        "adapter_ligation.aspirate_from_RNA_reaction_well"
    )
    reaction_ligation = reaction_ligation.merge(
        dropout_by_block,
        on=["observation_id", "workflow_block", "material_type"],
        how="left",
        validate="many_to_one",
    )
    reaction_ligation["apparent_dropout"] = (
        reaction_ligation["dropout_barcodes"].fillna(0).gt(0)
    )

    selected.to_csv(OUT / "dropout_barcode_well_mapping.csv", index=False)
    ligation.to_csv(OUT / "dropout_barcode_matched_qpm_operations.csv", index=False)
    block.to_csv(OUT / "dropout_block_matched_qpm_operations.csv", index=False)
    udg.to_csv(OUT / "dropout_udg_block_matched_qpm_operations.csv", index=False)
    reaction_ligation.to_csv(
        OUT / "dropout_reaction_ligation_block_matched_qpm_operations.csv",
        index=False,
    )
    return selected, ligation, block, udg, reaction_ligation


def metric_summary(
    data: pd.DataFrame,
    group_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    values = data.copy()
    values["absolute_log_amplitude_ratio"] = np.abs(
        np.log(values["amplitude_ratio_to_reference"].clip(lower=1e-9))
    )
    summary = (
        values.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            operations=("operation_id", "size"),
            median_anomaly_percentile=("qpm_anomaly_percentile", "median"),
            median_outside_reference_fraction=("outside_90pct_envelope_fraction", "median"),
            median_amplitude_ratio=("amplitude_ratio_to_reference", "median"),
            median_absolute_log_amplitude_ratio=("absolute_log_amplitude_ratio", "median"),
        )
    )
    summary.to_csv(output_path, index=False)
    return summary


def reference_arrays(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q05 = np.vstack(rows["reference_q05_profile"].map(parse_profile))
    median = np.vstack(rows["reference_median_profile"].map(parse_profile))
    q95 = np.vstack(rows["reference_q95_profile"].map(parse_profile))
    return np.nanmedian(q05, axis=0), np.nanmedian(median, axis=0), np.nanmedian(q95, axis=0)


def plot_waveforms(
    ax: plt.Axes,
    rows: pd.DataFrame,
    failure_mask: pd.Series,
    title: str,
    failed_label: str,
    survivor_label: str,
) -> None:
    rows = rows.copy()
    failure_mask = failure_mask.reindex(rows.index).fillna(False).astype(bool)
    if rows.empty:
        ax.text(0.5, 0.5, "No matched QPM curves", ha="center", va="center")
        ax.set_title(title)
        return
    q05, reference, q95 = reference_arrays(rows)
    x = np.linspace(0, 1, len(reference))
    ax.fill_between(x, q05, q95, color=PALETTE["gold"], alpha=0.30, lw=0)
    ax.plot(x, reference, color=PALETTE["ink"], lw=1.7, ls="--")
    for index, row in rows.loc[~failure_mask].iterrows():
        profile = parse_profile(row["observed_profile"])
        ax.plot(
            np.linspace(0, 1, len(profile)),
            profile,
            color=PALETTE["teal"],
            alpha=0.34,
            lw=1.0,
        )
    for index, row in rows.loc[failure_mask].iterrows():
        profile = parse_profile(row["observed_profile"])
        ax.plot(
            np.linspace(0, 1, len(profile)),
            profile,
            color=PALETTE["coral"],
            alpha=0.86,
            lw=1.45,
        )
    ax.axhline(0, color=PALETTE["gray"], lw=0.5, alpha=0.5)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_xlabel("Normalized operation time")
    ax.set_ylabel("QPM response\n(raw counts)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DED9", lw=0.5, alpha=0.65)
    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["coral"], lw=2, label=failed_label),
            Line2D([0], [0], color=PALETTE["teal"], lw=2, label=survivor_label),
            Line2D([0], [0], color=PALETTE["ink"], lw=1.7, ls="--", label="matched reference median"),
            Line2D([0], [0], color=PALETTE["gold"], lw=7, alpha=0.35, label="reference 5th–95th%"),
        ],
        fontsize=6.5,
        frameon=False,
        loc="best",
    )


def plot_barcode_counts(ax: plt.Axes, rows: pd.DataFrame, title: str) -> None:
    rows = rows.sort_values("planned_order").copy()
    heights = rows["read_count"].fillna(0).clip(lower=0.5)
    colors = rows["material_type"].map({"DNA": PALETTE["blue"], "RNA": PALETTE["rna"]})
    edges = np.where(rows["apparent_dropout"], PALETTE["coral"], PALETTE["ink"])
    widths = np.where(rows["apparent_dropout"], 2.2, 0.6)
    ax.bar(rows["planned_order"], heights, color=colors, edgecolor=edges, linewidth=widths)
    for row in rows.itertuples(index=False):
        if row.apparent_dropout:
            ax.text(
                row.planned_order,
                max(float(row.read_count), 0.5) * 1.8,
                f"{row.barcode_plate_well}\n{int(row.read_count):,}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=PALETTE["coral"],
                fontweight="bold",
            )
    ax.set_yscale("log")
    ax.set_ylim(0.35, max(heights.max() * 15, 1_000))
    ax.set_xticks(rows["planned_order"])
    ax.set_xticklabels(rows["planned_order"].astype(int), fontsize=6)
    ax.set_xlabel("Planned barcode order")
    ax.set_ylabel("Reads (log scale)")
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", color="#D9DED9", lw=0.5, alpha=0.65)
    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["blue"], lw=7, label="DNA"),
            Line2D([0], [0], color=PALETTE["rna"], lw=7, label="RNA"),
            Line2D([0], [0], color=PALETTE["coral"], lw=2.2, label="dropout outline"),
        ],
        fontsize=6.5,
        frameon=False,
        loc="upper right",
    )


def run_labels(barcodes: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for observation_id, rows in barcodes.groupby("observation_id", sort=False):
        row = rows.iloc[0]
        labels[observation_id] = f"{row['hostname']} · {row['chip_lane']} · {observation_id}"
    return labels


def make_figure(
    barcodes: pd.DataFrame,
    ligation: pd.DataFrame,
    block: pd.DataFrame,
    udg: pd.DataFrame,
    reaction_ligation: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = run_labels(barcodes)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15.2, 12.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.92, 1.05, 1.05]},
    )
    fig.suptitle(
        "Barcode dropout matched to workflow-position QPM curves",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.5,
        0.972,
        "Red curves are assigned to failed barcode wells or failed workflow blocks; green curves are same-run survivors. "
        "Gold bands are matched cross-run reference envelopes.",
        ha="center",
        va="top",
        fontsize=8,
        color=PALETTE["ink"],
    )

    # Row 1: second DNA workflow block failed; show its earliest selective QPM deviation.
    obs = "obs_fef8b0992599"
    b = barcodes[barcodes["observation_id"].eq(obs)]
    plot_barcode_counts(axes[0, 0], b, f"{labels[obs]}\nSecond DNA block: orders 6–9 all dropped")
    for col, operation in [(1, "aspirate"), (2, "dispense")]:
        q = udg[
            udg["observation_id"].eq(obs)
            & udg["operation"].eq(operation)
            & udg["workflow_block"].isin([1, 2])
        ]
        plot_waveforms(
            axes[0, col],
            q,
            q["workflow_block"].eq(2),
            f"UDG {operation}: 3 µL into DNA reaction block\nUDG2/wells 7–12 failed; UDG1/wells 1–6 survived",
            "UDG2 · failed DNA block",
            "UDG1 · surviving DNA block",
        )

    # Row 2: direct positional evidence at each failed barcode well.
    obs = "obs_32eade743d80"
    b = barcodes[barcodes["observation_id"].eq(obs)]
    plot_barcode_counts(axes[1, 0], b, f"{labels[obs]}\nDNA order 6 + RNA orders 11–14 dropped")
    q = reaction_ligation[
        reaction_ligation["observation_id"].eq(obs)
        & reaction_ligation["workflow_block"].isin([3, 4])
    ]
    plot_waveforms(
        axes[1, 1],
        q,
        q["workflow_block"].eq(3),
        "Adapter ligation: aspirate from RNA reaction wells\nfailed block 3 vs surviving paired block 4",
        "failed RNA block 3",
        "surviving RNA block 4",
    )
    q = ligation[
        ligation["observation_id"].eq(obs) & ligation["operation"].eq("dispense")
    ]
    plot_waveforms(
        axes[1, 2],
        q,
        q["apparent_dropout"],
        "Adapter ligation: 20 µL into assigned barcode well\ndirect well-to-barcode match",
        "dropout barcode wells",
        "surviving barcode wells",
    )

    # Row 3: show that direct ligation curves do not separate scattered dropouts.
    obs = "obs_327d7cbf172b"
    b = barcodes[barcodes["observation_id"].eq(obs)]
    plot_barcode_counts(axes[2, 0], b, f"{labels[obs]}\nScattered DNA and RNA dropout")
    for col, operation in [(1, "dispense"), (2, "aspirate")]:
        q = ligation[
            ligation["observation_id"].eq(obs) & ligation["operation"].eq(operation)
        ]
        volume = "20 µL into" if operation == "dispense" else "25 µL from"
        plot_waveforms(
            axes[2, col],
            q,
            q["apparent_dropout"],
            f"Adapter ligation: {volume} barcode well\ndropout and survivor curves overlap",
            "dropout barcode wells",
            "surviving barcode wells",
        )

    for label, ax in zip("abcdefghi", axes.ravel()):
        ax.text(
            -0.12,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            color=PALETTE["ink"],
        )

    png = OUT / "dropout_workflow_matched_curves.png"
    pdf = OUT / "dropout_workflow_matched_curves.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def value(summary: pd.DataFrame, **conditions: object) -> pd.Series:
    rows = summary
    for column, expected in conditions.items():
        rows = rows[rows[column].eq(expected)]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {conditions}, found {len(rows)}")
    return rows.iloc[0]


def make_conclusions(
    barcodes: pd.DataFrame,
    ligation_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    udg_summary: pd.DataFrame,
    reaction_ligation_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Run 1: UDG2 is the earliest strongly separated operation in failed DNA block 2.
    obs = "obs_fef8b0992599"
    failed_a = value(
        udg_summary,
        observation_id=obs,
        workflow_block=2,
        operation="aspirate",
    )
    good_a = value(
        udg_summary,
        observation_id=obs,
        workflow_block=1,
        operation="aspirate",
    )
    failed_d = value(
        udg_summary,
        observation_id=obs,
        workflow_block=2,
        operation="dispense",
    )
    good_d = value(
        udg_summary,
        observation_id=obs,
        workflow_block=1,
        operation="dispense",
    )
    rows.append(
        {
            "observation_id": obs,
            "rank": 1,
            "candidate_step": "UDG2 transfer into DNA reaction wells 7–12, workflow block 2",
            "match_scope": "workflow_block",
            "causal_assessment": "most_likely_observed_contributor",
            "confidence": "moderate",
            "supporting_evidence": (
                f"All four non-control DNA barcodes in block 2 dropped. All 12 UDG2 aspiration/dispense "
                f"operations were at or above the 95th anomaly percentile. Median fraction outside the matched "
                f"reference envelope was {failed_a.median_outside_reference_fraction:.3f} vs "
                f"{good_a.median_outside_reference_fraction:.3f} for UDG aspirations and "
                f"{failed_d.median_outside_reference_fraction:.3f} vs "
                f"{good_d.median_outside_reference_fraction:.3f} for dispenses; median dispense amplitude was "
                f"{failed_d.median_amplitude_ratio:.2f}× vs {good_d.median_amplitude_ratio:.2f}× reference."
            ),
            "limitation": "Block-level match; it does not distinguish a UDG reagent condition, aspiration/delivery problem, reaction-well condition, or QPM response artifact.",
        }
    )
    rows.append(
        {
            "observation_id": obs,
            "rank": 2,
            "candidate_step": "Later DNA target-amplification and adapter-ligation operations",
            "match_scope": "direct_barcode_plate_well",
            "causal_assessment": "supporting_but_not_specific",
            "confidence": "low",
            "supporting_evidence": "Several later operations in the failed block remained abnormal.",
            "limitation": "They occur after the highly selective UDG2 divergence; adapter curves do not separate failed DNA wells from surviving DNA wells.",
        }
    )

    # Run 2: direct positional adapter-ligation match is strongest.
    obs = "obs_32eade743d80"
    failed_rxn = value(
        reaction_ligation_summary,
        observation_id=obs,
        workflow_block=3,
        operation="aspirate",
    )
    good_rxn = value(
        reaction_ligation_summary,
        observation_id=obs,
        workflow_block=4,
        operation="aspirate",
    )
    failed_a = value(ligation_summary, observation_id=obs, operation="aspirate", apparent_dropout=True)
    good_a = value(ligation_summary, observation_id=obs, operation="aspirate", apparent_dropout=False)
    failed_d = value(ligation_summary, observation_id=obs, operation="dispense", apparent_dropout=True)
    good_d = value(ligation_summary, observation_id=obs, operation="dispense", apparent_dropout=False)
    rows.append(
        {
            "observation_id": obs,
            "rank": 1,
            "candidate_step": "Adapter ligation: mix in and recover from each assigned barcode well",
            "match_scope": "direct_barcode_plate_well",
            "causal_assessment": "most_likely_observed_contributor",
            "confidence": "moderate_to_strong_diagnostic",
            "supporting_evidence": (
                f"The failed RNA branch was already abnormal when ligation aspirated from reaction wells: median "
                f"amplitude was {failed_rxn.median_amplitude_ratio:.2f}× reference vs "
                f"{good_rxn.median_amplitude_ratio:.2f}× for the surviving paired branch. All five failed "
                f"barcode positions then directly matched extreme ligation operations. Failed-well "
                f"aspirations had median amplitude ratio {failed_a.median_amplitude_ratio:.2f} vs "
                f"{good_a.median_amplitude_ratio:.2f} in surviving wells and median outside-envelope fraction "
                f"{failed_a.median_outside_reference_fraction:.3f} vs {good_a.median_outside_reference_fraction:.3f}. "
                f"Failed-well dispenses had median amplitude ratio {failed_d.median_amplitude_ratio:.2f} vs "
                f"{good_d.median_amplitude_ratio:.2f}."
            ),
            "limitation": "Observational association; the QPM signal may reflect a condition in the well rather than a pipetting fault.",
        }
    )
    rows.append(
        {
            "observation_id": obs,
            "rank": 2,
            "candidate_step": "RNA target-amplification mastermix transfer before adapter ligation",
            "match_scope": "workflow_block",
            "causal_assessment": "not_supported_as_unique_cause",
            "confidence": "low",
            "supporting_evidence": "The failed RNA barcodes occupy workflow block 3.",
            "limitation": "The surviving RNA block 4 had comparable or larger QPM amplitude deviations, so this step does not discriminate failure.",
        }
    )

    # Run 3: no direct QPM localization.
    obs = "obs_327d7cbf172b"
    failed_a = value(ligation_summary, observation_id=obs, operation="aspirate", apparent_dropout=True)
    good_a = value(ligation_summary, observation_id=obs, operation="aspirate", apparent_dropout=False)
    failed_d = value(ligation_summary, observation_id=obs, operation="dispense", apparent_dropout=True)
    good_d = value(ligation_summary, observation_id=obs, operation="dispense", apparent_dropout=False)
    rows.append(
        {
            "observation_id": obs,
            "rank": 1,
            "candidate_step": "No barcode-specific QPM step localized",
            "match_scope": "direct_and_run_level_review",
            "causal_assessment": "unresolved",
            "confidence": "high_confidence_in_nonlocalization",
            "supporting_evidence": (
                f"At adapter ligation, failed wells were not more abnormal than survivors: median outside-envelope "
                f"fractions were {failed_a.median_outside_reference_fraction:.3f} vs "
                f"{good_a.median_outside_reference_fraction:.3f} for aspiration and "
                f"{failed_d.median_outside_reference_fraction:.3f} vs "
                f"{good_d.median_outside_reference_fraction:.3f} for dispense."
            ),
            "limitation": "The run had broad templating and library-preparation anomalies, but those are shared across barcodes and cannot explain which barcode dropped.",
        }
    )
    conclusions = pd.DataFrame(rows)
    conclusions.to_csv(OUT / "dropout_workflow_step_ranking.csv", index=False)
    return conclusions


def make_report(
    barcodes: pd.DataFrame,
    conclusions: pd.DataFrame,
    ligation_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
) -> None:
    identity = (
        barcodes[["observation_id", "hostname", "chip_lane", "run_name"]]
        .drop_duplicates("observation_id")
        .set_index("observation_id")
    )
    lines = [
        "# Dropout barcode–workflow curve matching",
        "",
        "## What is directly matched",
        "",
        "The row order in `CSA/analysisSamples.json` was validated against `CSA/planned_run.json` by comparing `sampleExternalIds` with each `deckConfig.sampleBarcodeMapping.sampleId`. Identifiers were used only for equality checks and were not retained. This provides a direct mapping from each hashed barcode outcome to its barcode-plate well, sample-plate well, material, and workflow block.",
        "",
        "Adapter-ligation operations explicitly naming that barcode-plate well are therefore direct positional matches. Target-amplification operations are block-level matches. Templating and sequencing operations are run-level only.",
        "",
        "QPM values remain baseline-relative raw sensor counts; no psi/kPa conversion is implied.",
        "",
        "![Matched dropout workflow curves](dropout_workflow_matched_curves.png)",
        "",
        "## Conclusions",
        "",
    ]
    for obs in OBSERVATIONS:
        info = identity.loc[obs]
        lines.extend(
            [
                f"### `{obs}` — {info.hostname}, {info.chip_lane}",
                "",
                f"Run: `{info.run_name}`",
                "",
            ]
        )
        for row in conclusions[conclusions["observation_id"].eq(obs)].sort_values("rank").itertuples(index=False):
            lines.extend(
                [
                    f"{row.rank}. **{row.candidate_step}** — `{row.causal_assessment}` ({row.confidence}).",
                    f"   Evidence: {row.supporting_evidence}",
                    f"   Limitation: {row.limitation}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "A positional QPM match increases mechanistic plausibility, but it does not establish that pressure caused dropout. A curve can change because of liquid properties, an empty/partly empty well, bubbles, tip behavior, or a preceding failure. Controlled replay with known input and randomized well positions is required to distinguish those explanations.",
            "",
        ]
    )
    (OUT / "dropout_workflow_curve_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    barcodes, ligation, block, udg, reaction_ligation = prepare_tables()
    ligation_summary = metric_summary(
        ligation,
        ["observation_id", "operation", "apparent_dropout"],
        OUT / "dropout_ligation_group_summary.csv",
    )
    block_summary = metric_summary(
        block,
        ["observation_id", "material_type", "workflow_block", "operation"],
        OUT / "dropout_target_amp_block_summary.csv",
    )
    udg_summary = metric_summary(
        udg,
        ["observation_id", "workflow_block", "operation"],
        OUT / "dropout_udg_block_summary.csv",
    )
    reaction_ligation_summary = metric_summary(
        reaction_ligation,
        ["observation_id", "workflow_block", "operation"],
        OUT / "dropout_reaction_ligation_block_summary.csv",
    )
    conclusions = make_conclusions(
        barcodes,
        ligation_summary,
        block_summary,
        udg_summary,
        reaction_ligation_summary,
    )
    make_figure(barcodes, ligation, block, udg, reaction_ligation)
    make_report(barcodes, conclusions, ligation_summary, block_summary)
    print(f"Wrote {OUT / 'dropout_workflow_matched_curves.png'}")
    print(f"Wrote {OUT / 'dropout_workflow_step_ranking.csv'}")
    print(f"Wrote {OUT / 'dropout_workflow_curve_report.md'}")


if __name__ == "__main__":
    main()
