#!/usr/bin/env python3
"""Compare QPM waveform classifiers across three liquid classes and two operations."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import sklearn
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from analyze_qpm_volume_diagnostics import RANDOM_SEED
from compare_qpm_models import (
    MODEL_COLORS,
    MODEL_ORDER,
    constrained_dtw_distance_matrix,
    estimator_factory,
)


PROJECT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT / "qpm_class_operation_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_DEFINITIONS = {
    49: {
        "name": "lc_general_simple_surface_part_10uLps",
        "config_feature": "liquid_class_common_tbl.lc_general_simple_surface_part_10u_lps_lq_num",
        "short": "LC49 / 10uLps",
    },
    55: {
        "name": "lc_general_simple_surface_part_50uLps",
        "config_feature": "liquid_class_common_tbl.lc_general_simple_surface_part_50u_lps_lq_num",
        "short": "LC55 / 50uLps",
    },
    61: {
        "name": "lc_general_simple_surface_part_100uLps",
        "config_feature": "liquid_class_common_tbl.lc_general_simple_surface_part_100u_lps_lq_num",
        "short": "LC61 / 100uLps",
    },
}
OPERATIONS = ["aspirate", "dispense"]


def load_task(class_index: int, operation: str) -> tuple[pd.DataFrame, list[int], int]:
    definition = CLASS_DEFINITIONS[class_index]
    columns = [
        "observation_id",
        "physical_run_id",
        "instrument_id",
        "chip_lane",
        "run_start_time",
        "operation_id",
        "operation_index",
        "programmed_volume_ul",
        "workflow_stage",
        "process_step",
        "loc",
        "pressure_unit",
        "qpm_profile",
    ]
    frame = pq.read_table(
        PROJECT / "qpm_operation_waveforms.parquet",
        columns=columns,
        filters=[
            ("liquid_class_index", "=", float(class_index)),
            ("operation", "=", operation),
            ("pipette", "=", "PIP2"),
            ("qpm_available", "=", True),
        ],
    ).to_pandas()
    configuration = pd.read_parquet(
        PROJECT / "csa_ml_dataset.parquet",
        columns=["observation_id", definition["config_feature"]],
    )
    frame = frame.merge(configuration, on="observation_id", how="left")
    frame = frame[
        frame[definition["config_feature"]].eq(float(class_index))
        & frame["programmed_volume_ul"].gt(0)
    ].copy()
    raw_waveforms = len(frame)

    coverage = (
        frame.groupby("programmed_volume_ul")
        .agg(
            observations=("observation_id", "nunique"),
            physical_runs=("physical_run_id", "nunique"),
            instruments=("instrument_id", "nunique"),
            waveforms=("operation_id", "size"),
        )
        .reset_index()
    )
    eligible = coverage[
        coverage["observations"].ge(100) & coverage["instruments"].eq(4)
    ].sort_values(
        ["observations", "waveforms", "programmed_volume_ul"],
        ascending=[False, False, True],
    )
    if len(eligible) < 5:
        raise RuntimeError(
            f"Only {len(eligible)} well-covered volumes for class {class_index} {operation}"
        )
    volumes = sorted(eligible.head(5)["programmed_volume_ul"].astype(int).tolist())
    frame = frame[frame["programmed_volume_ul"].isin(volumes)].copy()
    frame["programmed_volume_ul"] = frame["programmed_volume_ul"].astype(int)

    rows: list[dict[str, object]] = []
    group_columns = [
        "observation_id",
        "physical_run_id",
        "instrument_id",
        "chip_lane",
        "run_start_time",
        "programmed_volume_ul",
    ]
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        profiles = np.vstack(group["qpm_profile"].map(lambda value: np.asarray(value, dtype=float)))
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "qpm_profile": np.median(profiles, axis=0),
                "raw_waveform_count": int(len(group)),
                "workflow_step_count": int(group["process_step"].nunique(dropna=True)),
                "source_location_count": int(group["loc"].nunique(dropna=True)),
            }
        )
        rows.append(row)
    aggregated = pd.DataFrame(rows).sort_values(
        ["run_start_time", "observation_id", "programmed_volume_ul"], kind="stable"
    ).reset_index(drop=True)
    return aggregated, volumes, raw_waveforms


def baseline_correct(frame: pd.DataFrame) -> np.ndarray:
    profiles = np.vstack(frame["qpm_profile"].to_numpy())
    return profiles - np.median(profiles[:, :3], axis=1)[:, None]


def majority_prediction(y_train: np.ndarray, n_test: int) -> np.ndarray:
    values, counts = np.unique(y_train, return_counts=True)
    return np.full(n_test, values[np.argmax(counts)], dtype=int)


def robust_template_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    volumes: list[int],
) -> np.ndarray:
    distances = []
    for volume in volumes:
        members = x_train[y_train == volume]
        median = np.median(members, axis=0)
        mad = 1.4826 * np.median(np.abs(members - median), axis=0)
        scale = np.maximum(mad, 5.0)
        distances.append(np.sqrt(np.mean(((x_test - median) / scale) ** 2, axis=1)))
    return np.asarray(volumes, dtype=int)[np.argmin(np.column_stack(distances), axis=1)]


def dtw_prediction(
    distances: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    volumes: list[int],
    neighbors: int = 5,
) -> np.ndarray:
    result = []
    for test_index in test:
        test_distances = distances[test_index, train]
        k = min(neighbors, len(train))
        nearest = np.argpartition(test_distances, k - 1)[:k]
        labels = y_train[nearest]
        weights = 1.0 / np.maximum(test_distances[nearest], 1e-6)
        votes = {
            volume: float(weights[labels == volume].sum()) for volume in volumes
        }
        result.append(max(volumes, key=lambda volume: (votes[volume], -volume)))
    return np.asarray(result, dtype=int)


def fit_predict(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    volumes: list[int],
    dtw_distances: np.ndarray,
) -> np.ndarray:
    if model_name == "Majority baseline":
        return majority_prediction(y[train], len(test))
    if model_name == "Robust template":
        return robust_template_prediction(x[train], y[train], x[test], volumes)
    if model_name == "DTW 5-NN":
        return dtw_prediction(dtw_distances, train, test, y[train], volumes)
    model = estimator_factory(model_name)
    model.fit(x[train], y[train])
    return model.predict(x[test]).astype(int)


def metric_row(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    recalls = recall_score(
        y_true, prediction, labels=np.unique(y_true), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "minimum_volume_recall": float(np.min(recalls)),
        "maximum_volume_recall": float(np.max(recalls)),
    }


def evaluate_task(
    class_index: int,
    operation: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    frame, volumes, total_class_operation_waveforms = load_task(class_index, operation)
    x = baseline_correct(frame)
    y = frame["programmed_volume_ul"].to_numpy(dtype=int)
    groups = frame["physical_run_id"].to_numpy()
    instruments = frame["instrument_id"].to_numpy()
    task_id = f"lc{class_index}_{operation}"
    print(
        f"{task_id}: {len(frame)} aggregated profiles, "
        f"{int(frame['raw_waveform_count'].sum())} selected raw waveforms, volumes {volumes}",
        flush=True,
    )

    dtw_started = time.perf_counter()
    dtw_distances = constrained_dtw_distance_matrix(x, radius=3)
    dtw_seconds = time.perf_counter() - dtw_started

    grouped_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    grouped_splits = [
        (f"group_fold_{index}", train, test)
        for index, (train, test) in enumerate(
            grouped_splitter.split(x, y, groups), start=1
        )
    ]
    loio_splits = []
    for instrument in sorted(np.unique(instruments)):
        test = np.flatnonzero(instruments == instrument)
        train = np.flatnonzero(instruments != instrument)
        loio_splits.append((str(instrument), train, test))

    predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []

    for scheme, splits in (
        ("grouped", grouped_splits),
        ("leave_one_instrument_out", loio_splits),
    ):
        for model_name in MODEL_ORDER:
            model_prediction = np.full(len(y), -1, dtype=int)
            model_seconds = 0.0
            for fold_name, train, test in splits:
                started = time.perf_counter()
                fold_prediction = fit_predict(
                    model_name,
                    x,
                    y,
                    train,
                    test,
                    volumes,
                    dtw_distances,
                )
                model_seconds += time.perf_counter() - started
                model_prediction[test] = fold_prediction
                fold_rows.append(
                    {
                        "task_id": task_id,
                        "liquid_class_index": class_index,
                        "operation": operation,
                        "validation_scheme": scheme,
                        "fold": fold_name,
                        "model": model_name,
                        "test_n": int(len(test)),
                        **metric_row(y[test], fold_prediction),
                    }
                )
            if np.any(model_prediction < 0):
                raise RuntimeError(f"Missing predictions for {task_id}/{scheme}/{model_name}")
            predictions[(scheme, model_name)] = model_prediction
            recalls = recall_score(
                y, model_prediction, labels=volumes, average=None, zero_division=0
            )
            performance_rows.append(
                {
                    "task_id": task_id,
                    "liquid_class_index": class_index,
                    "liquid_class_name": CLASS_DEFINITIONS[class_index]["name"],
                    "operation": operation,
                    "validation_scheme": scheme,
                    "model": model_name,
                    "n_profiles": int(len(frame)),
                    "n_raw_selected_waveforms": int(frame["raw_waveform_count"].sum()),
                    "n_observations": int(frame["observation_id"].nunique()),
                    "n_physical_runs": int(frame["physical_run_id"].nunique()),
                    "n_instruments": int(frame["instrument_id"].nunique()),
                    "volumes_ul": ";".join(map(str, volumes)),
                    **metric_row(y, model_prediction),
                    "fit_predict_seconds": model_seconds,
                }
            )
            for volume, recall in zip(volumes, recalls):
                confusion_rows.append(
                    {
                        "task_id": task_id,
                        "liquid_class_index": class_index,
                        "operation": operation,
                        "validation_scheme": scheme,
                        "model": model_name,
                        "metric": "volume_recall",
                        "actual_volume_ul": volume,
                        "predicted_volume_ul": volume,
                        "count": int((y == volume).sum()),
                        "row_fraction": float(recall),
                    }
                )
            counts = confusion_matrix(y, model_prediction, labels=volumes)
            fractions = confusion_matrix(
                y, model_prediction, labels=volumes, normalize="true"
            )
            for row, actual in enumerate(volumes):
                for col, predicted in enumerate(volumes):
                    confusion_rows.append(
                        {
                            "task_id": task_id,
                            "liquid_class_index": class_index,
                            "operation": operation,
                            "validation_scheme": scheme,
                            "model": model_name,
                            "metric": "confusion_cell",
                            "actual_volume_ul": actual,
                            "predicted_volume_ul": predicted,
                            "count": int(counts[row, col]),
                            "row_fraction": float(fractions[row, col]),
                        }
                    )
            part = frame[
                [
                    "observation_id",
                    "physical_run_id",
                    "instrument_id",
                    "chip_lane",
                    "run_start_time",
                    "programmed_volume_ul",
                    "raw_waveform_count",
                ]
            ].copy()
            part.insert(0, "task_id", task_id)
            part["validation_scheme"] = scheme
            part["model"] = model_name
            part["predicted_volume_ul"] = model_prediction
            prediction_rows.append(part)

    cohort = {
        "task_id": task_id,
        "liquid_class_index": class_index,
        "liquid_class_name": CLASS_DEFINITIONS[class_index]["name"],
        "operation": operation,
        "pipette": "PIP2",
        "volumes_ul": volumes,
        "n_profiles": int(len(frame)),
        "n_selected_raw_waveforms": int(frame["raw_waveform_count"].sum()),
        "n_all_positive_volume_waveforms_for_class_operation": int(total_class_operation_waveforms),
        "n_observations": int(frame["observation_id"].nunique()),
        "n_physical_runs": int(frame["physical_run_id"].nunique()),
        "n_instruments": int(frame["instrument_id"].nunique()),
        "aggregation": "pointwise median within observation and programmed volume",
        "dtw_precompute_seconds": dtw_seconds,
    }
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(fold_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(confusion_rows),
        cohort,
    )


def task_label(task_id: str) -> str:
    class_index = int(task_id.split("_")[0].replace("lc", ""))
    operation = task_id.split("_", 1)[1]
    return f"{CLASS_DEFINITIONS[class_index]['short']}\n{operation}"


def annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    vmin: float = 0.2,
    vmax: float = 1.0,
) -> None:
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="YlGnBu", aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > 0.72 else "#24352A",
                fontweight="bold" if value > 0.82 else "normal",
            )
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=34, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontweight="bold")
    plt.colorbar(image, ax=ax, shrink=0.7, label="Balanced accuracy")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        zorder=20,
    )


def make_summary_figure(performance: pd.DataFrame) -> pd.DataFrame:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    tasks = [
        f"lc{class_index}_{operation}"
        for class_index in CLASS_DEFINITIONS
        for operation in OPERATIONS
    ]
    labels = [task_label(task) for task in tasks]
    fig = plt.figure(figsize=(183 / 25.4, 175 / 25.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.95])
    axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(2)]

    matrices: dict[str, np.ndarray] = {}
    for scheme, ax, title, panel in (
        ("grouped", axes[0], "Held-out physical-run balanced accuracy (%)", "a"),
        (
            "leave_one_instrument_out",
            axes[1],
            "Unseen-instrument balanced accuracy (%)",
            "b",
        ),
    ):
        subset = performance[performance["validation_scheme"].eq(scheme)]
        pivot = subset.pivot(index="task_id", columns="model", values="balanced_accuracy")
        matrix = pivot.reindex(index=tasks, columns=MODEL_ORDER).to_numpy()
        matrices[scheme] = matrix
        annotated_heatmap(ax, matrix, labels, MODEL_ORDER, title)
        add_panel_label(ax, panel)

    ax = axes[2]
    transport_drop = matrices["grouped"] - matrices["leave_one_instrument_out"]
    image = ax.imshow(transport_drop, vmin=-0.1, vmax=0.35, cmap="RdYlGn_r", aspect="auto")
    for row in range(transport_drop.shape[0]):
        for col in range(transport_drop.shape[1]):
            value = transport_drop[row, col]
            ax.text(
                col,
                row,
                f"{value * 100:+.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > 0.22 else "#24352A",
            )
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=34, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Transport loss: grouped minus unseen-instrument (%)", fontweight="bold")
    plt.colorbar(image, ax=ax, shrink=0.7, label="Accuracy-point loss")
    add_panel_label(ax, "c")

    ax = axes[3]
    loio = performance[
        performance["validation_scheme"].eq("leave_one_instrument_out")
    ]
    positions = np.arange(len(MODEL_ORDER))
    offsets = {"aspirate": -0.14, "dispense": 0.14}
    operation_colors = {"aspirate": "#2A9D8F", "dispense": "#E76F51"}
    for operation in OPERATIONS:
        values = []
        for model_name in MODEL_ORDER:
            task_values = loio[
                loio["operation"].eq(operation) & loio["model"].eq(model_name)
            ]["balanced_accuracy"].to_numpy()
            values.append(float(np.mean(task_values)))
            ax.scatter(
                np.full(len(task_values), positions[MODEL_ORDER.index(model_name)] + offsets[operation]),
                task_values,
                s=15,
                color=operation_colors[operation],
                alpha=0.6,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        ax.plot(
            positions + offsets[operation],
            values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=operation_colors[operation],
            label=f"{operation} mean",
        )
    ax.set_xticks(positions, MODEL_ORDER, rotation=34, ha="right")
    ax.set_ylim(0.15, 1.02)
    ax.set_ylabel("Unseen-instrument balanced accuracy")
    ax.set_title("Mean transport across three liquid classes", fontweight="bold")
    ax.grid(axis="y", color="#D8E2DC", linewidth=0.55, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "d")

    fig.suptitle(
        "QPM model comparison across liquid classes and operations",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "PIP2; five most widely represented positive volumes per task; repeated traces collapsed to one median profile per observation-volume. "
        "All settings fixed before evaluation.",
        ha="center",
        fontsize=6.3,
        color="#4F5D4F",
    )
    fig.savefig(
        OUTPUT_DIR / "qpm_class_operation_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    candidates = loio[~loio["model"].eq("Majority baseline")]
    best = (
        candidates.sort_values(
            ["task_id", "balanced_accuracy", "accuracy"],
            ascending=[True, False, False],
        )
        .drop_duplicates("task_id")
        .sort_values(["liquid_class_index", "operation"])
    )
    return best


def make_best_confusion_figure(
    best: pd.DataFrame, confusion: pd.DataFrame, cohorts: list[dict[str, object]]
) -> None:
    cohort_map = {str(row["task_id"]): row for row in cohorts}
    tasks = [
        f"lc{class_index}_{operation}"
        for class_index in CLASS_DEFINITIONS
        for operation in OPERATIONS
    ]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(183 / 25.4, 190 / 25.4),
        constrained_layout=True,
    )
    for ax, task, panel in zip(axes.ravel(), tasks, "abcdef"):
        model_name = str(best.loc[best["task_id"].eq(task), "model"].iloc[0])
        volumes = list(map(int, cohort_map[task]["volumes_ul"]))
        selected = confusion[
            confusion["task_id"].eq(task)
            & confusion["validation_scheme"].eq("leave_one_instrument_out")
            & confusion["model"].eq(model_name)
            & confusion["metric"].eq("confusion_cell")
        ]
        matrix = (
            selected.pivot(
                index="actual_volume_ul",
                columns="predicted_volume_ul",
                values="row_fraction",
            )
            .reindex(index=volumes, columns=volumes)
            .to_numpy()
        )
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > 0.55 else "#24352A",
                    fontweight="bold" if row == col else "normal",
                )
        score = float(
            best.loc[best["task_id"].eq(task), "balanced_accuracy"].iloc[0]
        )
        ax.set_xticks(range(len(volumes)), volumes)
        ax.set_yticks(range(len(volumes)), volumes)
        ax.set_xlabel("Predicted volume (µL)")
        ax.set_ylabel("Actual volume (µL)")
        ax.set_title(
            f"{task_label(task).replace(chr(10), ' — ')}\n{model_name}, balanced accuracy {score * 100:.1f}%",
            fontweight="bold",
        )
        add_panel_label(ax, panel)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.45, label="Row fraction")
    fig.suptitle(
        "Best unseen-instrument model for each QPM task",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(
        OUTPUT_DIR / "qpm_class_operation_best_confusions.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


def main() -> None:
    started = time.perf_counter()
    performance_parts = []
    fold_parts = []
    prediction_parts = []
    confusion_parts = []
    cohorts: list[dict[str, object]] = []
    for class_index in CLASS_DEFINITIONS:
        for operation in OPERATIONS:
            performance, folds, predictions, confusion, cohort = evaluate_task(
                class_index, operation
            )
            performance_parts.append(performance)
            fold_parts.append(folds)
            prediction_parts.append(predictions)
            confusion_parts.append(confusion)
            cohorts.append(cohort)

    performance = pd.concat(performance_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    confusion = pd.concat(confusion_parts, ignore_index=True)
    cohort_table = pd.DataFrame(
        [
            {
                **row,
                "volumes_ul": ";".join(map(str, row["volumes_ul"])),
            }
            for row in cohorts
        ]
    )

    performance.to_csv(
        OUTPUT_DIR / "qpm_class_operation_model_performance.csv", index=False
    )
    folds.to_csv(
        OUTPUT_DIR / "qpm_class_operation_fold_performance.csv", index=False
    )
    predictions.to_csv(
        OUTPUT_DIR / "qpm_class_operation_predictions.csv", index=False
    )
    confusion.to_csv(
        OUTPUT_DIR / "qpm_class_operation_confusion.csv", index=False
    )
    cohort_table.to_csv(
        OUTPUT_DIR / "qpm_class_operation_cohorts.csv", index=False
    )

    best = make_summary_figure(performance)
    best.to_csv(
        OUTPUT_DIR / "qpm_class_operation_best_models.csv", index=False
    )
    make_best_confusion_figure(best, confusion, cohorts)

    nonbaseline = performance[~performance["model"].eq("Majority baseline")]
    loio = nonbaseline[
        nonbaseline["validation_scheme"].eq("leave_one_instrument_out")
    ]
    mean_ranking = (
        loio.groupby("model", as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            minimum_task_balanced_accuracy=("balanced_accuracy", "min"),
            maximum_task_balanced_accuracy=("balanced_accuracy", "max"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    mean_ranking.to_csv(
        OUTPUT_DIR / "qpm_class_operation_model_ranking.csv", index=False
    )

    run_log = {
        "analysis": "QPM model comparison across classes 49, 55, 61 and aspiration/dispense",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "random_seed": RANDOM_SEED,
        "class_definitions": CLASS_DEFINITIONS,
        "operations": OPERATIONS,
        "pipette": "PIP2",
        "cohorts": cohorts,
        "models": MODEL_ORDER,
        "validation": {
            "grouped": "5-fold StratifiedGroupKFold by physical_run_id",
            "leave_one_instrument_out": "four held-out-instrument folds",
            "volume_selection": "top five positive volumes by observation coverage, requiring >=100 observations and all four instruments",
            "aggregation": "pointwise median within observation-volume before validation",
            "target_guided_model_tuning": False,
        },
        "mean_unseen_instrument_ranking": mean_ranking.to_dict(orient="records"),
        "execution_seconds": time.perf_counter() - started,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (OUTPUT_DIR / "qpm_class_operation_run_log.json").write_text(
        json.dumps(run_log, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
