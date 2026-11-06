#!/usr/bin/env python3
"""Re-summarize sealed held-out QPM predictions by exact nominal volume.

No training, model selection, plugin edits or reference-bank changes occur.
Accuracy means agreement within a fixed 10% of commanded nominal volume;
it does not validate delivered volume or choose a recommended transfer volume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from qpm_reference_bank import file_sha256 as sha

VERSION="qpm-volume-stratified-performance-1.0.0"
SEED=20261027
FAMILIES={49:"lc_general_simple_surface_part_10uLps",55:"lc_general_simple_surface_part_50uLps",61:"lc_general_simple_surface_part_100uLps"}
SCHEMES=["grouped_run","unseen_instrument","chronological"]
MODELS=["Median baseline","Context-only median","Ridge","Elastic Net","Extra Trees"]
NAMES=["Median baseline","Context only","Ridge","Elastic Net","Extra Trees"]
PALETTE=dict(ink="#264653",teal="#218c80",gold="#d8b25c",blue="#477f9d",plum="#8d628d",paper="#f7f9f3",gray="#8c9690")


def write_json(path,record):
    path.write_text(json.dumps(record,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


def choose_definitions(coverage):
    """Choose dominant full definitions by source count, never by accuracy."""
    chosen=[]
    inventories=[]
    for family,name in FAMILIES.items():
        subset=coverage[coverage.liquid_class_name.eq(name)].copy()
        if subset.empty:raise ValueError(f"Main LC{family} definition absent")
        fingerprints=subset.groupby("liquid_class_fingerprint").waveforms.sum().sort_values(ascending=False)
        fingerprint=fingerprints.index[0]
        subset["canonical_family"]=family
        subset["chosen_definition"]=subset.liquid_class_fingerprint.eq(fingerprint)
        inventories.append(subset)
        chosen.append(subset[subset.chosen_definition])
    return pd.concat(chosen,ignore_index=True),pd.concat(inventories,ignore_index=True)


def per_run_errors(data):
    error=np.abs(data.predicted_volume_ul.to_numpy()-data.programmed_volume_ul.to_numpy())
    y=data.programmed_volume_ul.to_numpy()
    part=data[["task_id","scheme","model","physical_run_id","instrument_id","programmed_volume_ul"]].copy()
    part["waveforms"]=1
    part["absolute_error"]=error
    part["squared_error"]=error**2
    part["signed_error"]=data.predicted_volume_ul.to_numpy()-y
    part["inside_5pct"]=error <= .05*y
    part["inside_10pct"]=error <= .1*y
    part["correct_nominal_level"]=np.isclose(y,data.nearest_training_level_ul.to_numpy(),rtol=0,atol=1e-6)
    keys=["task_id","scheme","model","physical_run_id","instrument_id","programmed_volume_ul"]
    grouped=part.groupby(keys,sort=False).sum(numeric_only=True).reset_index()
    for name in ["absolute_error","squared_error","signed_error","inside_5pct","inside_10pct","correct_nominal_level"]:
        grouped[name]=grouped[name]/grouped.waveforms
    return grouped


def summarize_volume_runs(group,reps=500):
    group=group.sort_values("physical_run_id")
    volume=float(group.programmed_volume_ul.iloc[0])
    result=dict(programmed_volume_ul=volume,physical_runs=len(group),instruments=group.instrument_id.nunique(),
        waveforms=int(group.waveforms.sum()),accuracy_within_10pct=float(group.inside_10pct.mean()),
        agreement_within_5pct=float(group.inside_5pct.mean()),nominal_level_accuracy=float(group.correct_nominal_level.mean()),
        mae_ul=float(group.absolute_error.mean()),rmse_ul=float(np.sqrt(group.squared_error.mean())),
        signed_bias_ul=float(group.signed_error.mean()),mape_pct=100*float(group.absolute_error.mean())/volume,
        low_support=len(group)<30 or group.instrument_id.nunique()<3)
    for field in ["accuracy_ci_low","accuracy_ci_high","mae_ci_low_ul","mae_ci_high_ul"]:result[field]=np.nan
    if len(group)>=8:
        # Identical physical-run resampling across model families within a
        # task/scheme/volume; fixed OOF predictions, not model-refit intervals.
        key=f"{group.task_id.iloc[0]}|{group.scheme.iloc[0]}|{volume:.3f}"
        seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)+SEED
        rng=np.random.default_rng(seed)
        indices=rng.integers(0,len(group),size=(reps,len(group)))
        a=group.inside_10pct.to_numpy()[indices].mean(axis=1)
        m=group.absolute_error.to_numpy()[indices].mean(axis=1)
        result.update(accuracy_ci_low=float(np.quantile(a,.025)),accuracy_ci_high=float(np.quantile(a,.975)),
            mae_ci_low_ul=float(np.quantile(m,.025)),mae_ci_high_ul=float(np.quantile(m,.975)))
    return result


def major_volumes(group,maximum=14):
    """Coverage-only display rule, with the supported low/high ends retained."""
    available=group[group.physical_runs.ge(30)&group.instruments.ge(3)].copy()
    if available.empty:return []
    values=available.sort_values("programmed_volume_ul")
    extremes=[float(values.programmed_volume_ul.iloc[0]),float(values.programmed_volume_ul.iloc[-1])]
    ranked=available.sort_values(["physical_runs","programmed_volume_ul"],ascending=[False,True]).programmed_volume_ul
    kept=list(dict.fromkeys(extremes+[float(v) for v in ranked]))[:maximum]
    return sorted(kept)


def metrics(study,output):
    if output.exists() and any(output.iterdir()):raise ValueError("Use a fresh output folder for metrics")
    output.mkdir(parents=True,exist_ok=True)
    seal=json.loads((study/"artifact_validation.json").read_text())
    assert seal["status"]=="passed"
    for name in ["analysis_run_log.json","all_task_coverage.csv","all_volume_level_coverage.csv"]:
        assert sha(study/name)==seal["artifact_sha256"][name]
    log=json.loads((study/"analysis_run_log.json").read_text())
    coverage=pd.read_csv(study/"all_task_coverage.csv")
    tasks,inventory=choose_definitions(coverage)
    tasks.to_csv(output/"selected_main_transfer_tasks.csv",index=False)
    inventory.to_csv(output/"liquid_class_definition_inventory.csv",index=False)
    level_coverage=pd.read_csv(study/"all_volume_level_coverage.csv")
    source_performance=pd.read_csv(study/"model_performance.csv")
    summaries=[]
    run_tables=[]
    sources={}
    checks=[]
    for task in tasks[tasks.trainable].itertuples():
        folder=study/"tasks"/task.task_id
        for scheme in SCHEMES:
            paths=sorted(folder.glob(f"predictions_{scheme}_*.parquet"))
            if not paths:continue
            parts=[]
            for path in paths:
                relative=str(path.relative_to(study))
                assert sha(path)==seal["artifact_sha256"][relative]
                sources[relative]=seal["artifact_sha256"][relative]
                part=pd.read_parquet(path)
                fold=part.fold.iloc[0]
                audit_path=folder/f"fold_{scheme}_{fold}.json"
                assert sha(audit_path)==seal["artifact_sha256"][str(audit_path.relative_to(study))]
                audit=json.loads(audit_path.read_text())
                assert not set(audit["train_physical_runs"])&set(audit["test_physical_runs"])
                assert set(part.physical_run_id)==set(audit["test_physical_runs"])
                if scheme=="unseen_instrument":assert not set(audit["train_instruments"])&set(part.instrument_id)
                if scheme=="chronological":assert pd.Timestamp(audit["train_acquisition_max"])<pd.Timestamp(audit["test_acquisition_min"])
                parts.append(part)
            data=pd.concat(parts,ignore_index=True)
            assert not data.duplicated(["model","operation_id"]).any()
            assert np.isfinite(data.predicted_volume_ul).all()
            expected=source_performance[source_performance.task_id.eq(task.task_id)&source_performance.scheme.eq(scheme)]
            assert set(data.model)==set(expected.model)
            for model,part in data.groupby("model",sort=False):
                row=expected[expected.model.eq(model)].iloc[0]
                assert len(part)==row.test_waveforms
                if scheme in ["grouped_run","unseen_instrument"]:
                    source=level_coverage[level_coverage.task_id.eq(task.task_id)].set_index("programmed_volume_ul")
                    actual=part.groupby("programmed_volume_ul").agg(waveforms=("operation_id","size"),physical_runs=("physical_run_id","nunique"),instruments=("instrument_id","nunique"))
                    pd.testing.assert_frame_equal(actual,source.reindex(index=actual.index)[list(actual)],check_dtype=False,check_names=False)
            run=per_run_errors(data)
            for (model,volume),group in run.groupby(["model","programmed_volume_ul"],sort=False):
                summaries.append(dict(task_id=task.task_id,canonical_family=task.canonical_family,pipette=task.pipette,
                    operation=task.operation,liquid_class_name=task.liquid_class_name,liquid_class_fingerprint=task.liquid_class_fingerprint,
                    liquid_class_indices=task.liquid_class_indices,scheme=scheme,model=model,**summarize_volume_runs(group)))
            # Reconstruct the original task-wide run-balanced metric exactly.
            totals=run.copy()
            totals["hits"]=totals.inside_10pct*totals.waveforms
            totals["errors"]=totals.absolute_error*totals.waveforms
            totals=totals.groupby(["model","physical_run_id"])[["waveforms","hits","errors"]].sum().reset_index()
            for model,part in totals.groupby("model"):
                row=expected[expected.model.eq(model)].iloc[0]
                assert np.isclose((part.hits/part.waveforms).mean(),row.within_10pct)
                assert np.isclose((part.errors/part.waveforms).mean(),row.mae_ul)
            run_tables.append(run)
            checks.append(dict(task_id=task.task_id,scheme=scheme,prediction_files=len(paths),models=data.model.nunique(),waveforms_per_model=len(data)//data.model.nunique()))
            print(f"Checked {task.pipette} LC{task.canonical_family} {task.operation} {scheme}: {len(data):,} predictions",flush=True)
    result=pd.DataFrame(summaries).sort_values(["pipette","canonical_family","operation","scheme","programmed_volume_ul","model"])
    result.to_csv(output/"volume_level_model_performance.csv",index=False)
    pd.concat(run_tables,ignore_index=True).to_parquet(output/"per_run_volume_error_summaries.parquet",index=False,compression="zstd")
    selections=[]
    for task,group in result[result.scheme.eq("unseen_instrument")&result.model.eq("Extra Trees")].groupby("task_id"):
        volumes=major_volumes(group)
        selection=group[group.programmed_volume_ul.isin(volumes)].copy()
        selections.append(selection[["task_id","canonical_family","pipette","operation","programmed_volume_ul","physical_runs","instruments","waveforms"]])
    pd.concat(selections,ignore_index=True).to_csv(output/"major_volume_display_selection.csv",index=False)
    (output/"plot_qpm_accuracy_by_volume.executed.py").write_bytes(Path(__file__).read_bytes())
    names=["selected_main_transfer_tasks.csv","liquid_class_definition_inventory.csv","volume_level_model_performance.csv",
        "per_run_volume_error_summaries.parquet","major_volume_display_selection.csv","plot_qpm_accuracy_by_volume.executed.py"]
    receipt=dict(version=VERSION,status="passed",generated_utc=datetime.now(timezone.utc).isoformat(),source_study=str(study),
        source_study_validation_sha256=sha(study/"artifact_validation.json"),source_analysis_log_sha256=sha(study/"analysis_run_log.json"),
        source_prediction_sha256=sources,source_code_sha256=sha(Path(__file__)),checks=checks,
        definition_selection="dominant full fingerprint per named 10/50/100 uL/s general-transfer family; numeric aliases alone never pooled",
        display_selection="at most 14 exact volumes: >=30 physical runs and >=3 instruments; retain supported extremes then highest run coverage; no accuracy-based selection",
        accuracy_definition="mean across physical runs of the within-run fraction with abs(prediction-command)<=0.1*command at this exact volume",
        ci_definition="500 physical-run bootstrap resamples of fixed held-out predictions; not model-refit intervals or corrected simultaneous intervals",
        actual_delivered_volume_labels=0,models_retrained=False,plugin_and_reference_bank_unchanged=True,
        artifact_sha256={name:sha(output/name) for name in names})
    write_json(output/"metrics_validation.json",receipt)
    print(json.dumps(dict(status="passed",output=str(output),trained_tasks=len(tasks[tasks.trainable]),metric_rows=len(result),selected_definitions=3)),flush=True)


def setup():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":PALETTE["ink"],"axes.labelcolor":PALETTE["ink"],
        "axes.titlecolor":PALETTE["ink"],"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})


def heatmaps(frame,tasks,selections,pipette,metric):
    cmap=LinearSegmentedColormap.from_list("forest_accuracy",["#f3f5eb",PALETTE["gold"],PALETTE["teal"],"#235746"])
    fig,axes=plt.subplots(3,2,figsize=(17,10.5),layout="constrained")
    fig.set_facecolor(PALETTE["paper"])
    fig.get_layout_engine().set(hspace=.10,wspace=.03,h_pad=.10)
    accuracy=metric=="accuracy_within_10pct"
    title=("Accuracy within ±10% of commanded volume" if accuracy else "Mean absolute nominal-volume prediction error (µL)")
    fig.suptitle(f"{pipette} · {title}\nMajor transfer liquid classes · unseen-instrument predictions · volumes low → high",fontsize=18,fontweight="bold")
    scale_max=100 if accuracy else max(5,float(frame.loc[frame.task_id.isin(tasks.loc[tasks.pipette.eq(pipette),"task_id"]),metric].quantile(.95)))
    for row,family in enumerate(FAMILIES):
        for col,operation in enumerate(["aspirate","dispense"]):
            ax=axes[row,col]
            task=tasks[tasks.canonical_family.eq(family)&tasks.pipette.eq(pipette)&tasks.operation.eq(operation)].iloc[0]
            direction="Aspiration" if operation=="aspirate" else "Dispense"
            ax.set_title(f"{chr(97+row*2+col)}  LC{family} family · {direction}",loc="left",fontweight="bold",fontsize=12,pad=8)
            if not bool(task.trainable):
                ax.set_axis_off()
                ax.text(.5,.55,f"Only one commanded volume: {task.minimum_volume_ul:g} µL\n{task.physical_runs:,} runs · {task.waveforms:,} waveforms\nVolume-discrimination model not identifiable\nNo artificial 100% accuracy reported",ha="center",va="center",transform=ax.transAxes,color=PALETTE["gray"],fontsize=12)
                continue
            chosen=selections[selections.task_id.eq(task.task_id)].sort_values("programmed_volume_ul")
            volumes=chosen.programmed_volume_ul.tolist()
            matrix=frame[frame.task_id.eq(task.task_id)&frame.model.isin(MODELS)].pivot(index="model",columns="programmed_volume_ul",values=metric).reindex(index=MODELS,columns=volumes)
            values=matrix.to_numpy()*(100 if accuracy else 1)
            im=ax.imshow(values,cmap=cmap if accuracy else "cividis_r",vmin=0,vmax=scale_max,aspect="auto")
            ax.set_yticks(np.arange(len(MODELS)),NAMES)
            labels=[f"{volume:g}\n{int(n):,}" for volume,n in zip(volumes,chosen.physical_runs)]
            ax.set_xticks(np.arange(len(volumes)),labels,fontsize=8)
            ax.set_xlabel("Commanded volume (µL) / independent physical runs (n)",fontsize=9)
            for i in range(len(MODELS)):
                for j in range(len(volumes)):
                    value=values[i,j]
                    cell_cmap=cmap if accuracy else plt.get_cmap("cividis_r")
                    rgb=np.asarray(cell_cmap(np.clip(value/scale_max,0,1))[:3])
                    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                    luminance=float(linear@np.asarray([.2126,.7152,.0722]))
                    color="white" if luminance<.179 else "black"
                    ax.text(j,i,f"{value:.0f}" if accuracy else f"{value:.1f}",ha="center",va="center",fontsize=8,color=color)
            ax.tick_params(axis="both",length=0,pad=7)
    colorbar=fig.colorbar(im,ax=axes,location="right",shrink=.65,pad=.015,aspect=35)
    colorbar.set_label("Within ±10% agreement (%)" if accuracy else "MAE (µL); scale clipped at shown maximum",fontsize=10)
    note=("Accuracy = run-balanced agreement, not 1 − MAE. ±10% allows ±1 µL at 10 µL but ±10 µL at 100 µL.\n" if accuracy else "Average absolute error against the nominal command; not a measured transfer deficit or individual uncertainty interval.\n")
    fig.supxlabel(note+"Columns: ≤14 common volumes with ≥30 runs and ≥3 instruments; selected by coverage only. All volumes and confidence intervals are in the report.\nCells rounded to whole percent: 0 may mean <0.5%, not missing data. Full liquid-class definition and pipette remain separate.\nCSA labels are commanded—not independently measured delivered—volumes.",fontsize=9,color=PALETTE["ink"])
    return fig


def render(output):
    receipt=json.loads((output/"metrics_validation.json").read_text())
    assert receipt["status"]=="passed"
    for name,value in receipt["artifact_sha256"].items():assert sha(output/name)==value
    metrics=pd.read_csv(output/"volume_level_model_performance.csv")
    tasks=pd.read_csv(output/"selected_main_transfer_tasks.csv")
    selections=pd.read_csv(output/"major_volume_display_selection.csv")
    external=metrics[metrics.scheme.eq("unseen_instrument")]
    setup()
    outputs=[]
    with PdfPages(output/"qpm_accuracy_and_error_by_volume.pdf") as pdf:
        for measure in ["accuracy_within_10pct","mae_ul"]:
            for pipette in ["PIP2","PIP1"]:
                fig=heatmaps(external,tasks,selections,pipette,measure)
                name=f"qpm_{pipette.lower()}_{'accuracy' if measure=='accuracy_within_10pct' else 'mae'}_by_volume.png"
                fig.savefig(output/name,dpi=220,facecolor=fig.get_facecolor())
                pdf.savefig(fig,facecolor=fig.get_facecolor());plt.close(fig)
                outputs.append(name)
    selected_metrics=external.merge(selections[["task_id","programmed_volume_ul"]],on=["task_id","programmed_volume_ul"],validate="many_to_one")
    selected_metrics.to_csv(output/"major_volume_model_comparison.csv",index=False)
    panels=[]
    for pipette in ["PIP2","PIP1"]:
        cards=[]
        for task in tasks[tasks.pipette.eq(pipette)].sort_values(["canonical_family","operation"]).itertuples():
            selected=selected_metrics[selected_metrics.task_id.eq(task.task_id)&selected_metrics.model.isin(MODELS)].copy()
            for key in ["accuracy_within_10pct","accuracy_ci_low","accuracy_ci_high","agreement_within_5pct","nominal_level_accuracy"]:selected[key]*=100
            selected=selected.sort_values(["programmed_volume_ul","model"])
            table=selected[["programmed_volume_ul","model","accuracy_within_10pct","accuracy_ci_low","accuracy_ci_high","mae_ul","mae_ci_low_ul","mae_ci_high_ul","signed_bias_ul","physical_runs","instruments","waveforms"]].to_html(index=False,border=0,float_format=lambda v:f"{v:,.2f}")
            cards.append(f"<details><summary>LC{task.canonical_family} · {task.operation} · aliases {html.escape(task.liquid_class_indices)} · {task.physical_runs:,} source runs</summary><p>{html.escape(task.liquid_class_name)} · full fingerprint {html.escape(task.liquid_class_fingerprint)}</p>{table if len(selected) else '<p>Single nominal volume; no trained volume-discrimination model.</p>'}</details>")
        panels.append(f"<section><h2>{pipette}: accuracy by exact volume</h2><img src='qpm_{pipette.lower()}_accuracy_by_volume.png' alt='{pipette} volume-specific accuracy by model'><p>Read volumes left to right; second tick line is independent physical-run count. Every cell is a percentage rounded to the nearest whole percent; a displayed zero may mean less than 0.5%, not missing data. Expand the tables for decimal values and intervals.</p>{''.join(cards)}<details><summary>Companion MAE plot</summary><img src='qpm_{pipette.lower()}_mae_by_volume.png' alt='{pipette} nominal-volume MAE'></details></section>")
    families=tasks[["canonical_family","liquid_class_name","liquid_class_fingerprint"]].drop_duplicates().to_html(index=False,border=0)
    report=f'''<!doctype html><html><head><meta charset="utf-8"><title>QPM model accuracy from low to high volume</title><style>body{{font:16px system-ui;line-height:1.5;max-width:1800px;margin:28px auto;padding:0 20px;background:#f7f9f3;color:#264653}}section{{background:white;padding:20px;margin:20px 0;border-radius:8px;overflow:auto}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:12px}}th,td{{text-align:left;padding:7px;border-bottom:1px solid #dbe3d7}}details{{padding:12px;border-bottom:1px solid #dbe3d7}}summary{{cursor:pointer;font-weight:600}}a{{color:#218c80}}.warning{{border-left:5px solid #d8b25c;padding:15px;background:#fff8e8}}</style></head><body>
<h1>Where do the volume models work? Accuracy from low to high volume</h1>
<p class="warning"><strong>Accuracy means prediction within ±10% of the programmed command.</strong> It is not 1−MAE, a delivered-volume measurement, or an actual over/under-transfer estimate. At 10 µL, the acceptance band is 9–11 µL; at 100 µL it is 90–110 µL. Larger volumes therefore have a wider absolute tolerance.</p>
<p>No retraining or plugin/reference changes. Only the existing independently held-out predictions are re-summarized. The main plots use leave-one-instrument-out results; grouped physical-run and chronological results are also in the complete CSV.</p>
{''.join(panels)}
<section><h2>Definitions and honest comparison</h2><p>LC49, LC55 and LC61 denote the main general-transfer families, selected by their full definitions and names—not numerical IDs alone. Numerical IDs shift across software; aliases are recorded in the tables. For example, blowout_postrun can also carry ID49, but it is not pooled with the 10 µL/s general-transfer class. The earlier broad overview's largest numeric-ID cohort is not necessarily this main transfer family.</p>{families}<p>Other full-definition variants, including sparse untrainable definitions, are preserved in the definition inventory. PIP1 and PIP2 and aspiration/dispense are never pooled. PIP1 LC61 dispense contains only 250 µL in the dominant definition, so its volume-discrimination accuracy is not identifiable; no artificial constant-model 100% score is supplied.</p>
<p>Each exact-volume/model cell first averages correctness across that run's repeated operations, then gives each physical run one vote. More waveforms in one run do not create more independent evidence. Per-volume averages cannot simply be averaged to recover the original overall score: volumes and run participation differ.</p>
<p>A baseline can score 100% at a particular volume simply because its fixed prediction falls inside that volume's tolerance. For example, always predicting 50 µL is perfect in the 50 µL column but fails in a 10 µL column. This is not universal volume discrimination. The geometry/stage-only control can likewise identify nominal recipe volumes without demonstrating that a pressure waveform measures physical delivery.</p>
<p>Only display selection uses a support rule: ≥30 physical runs and ≥3 instruments, at most 14 exact volumes, retaining the supported low/high endpoints and then the most frequent volumes. It never selects the best-performing volumes. Every observed volume remains in the full CSV, including poorly supported levels. Missing cells mean no model or inadequate display coverage, not normal performance. No interpolation or smoothing invents untested volumes.</p>
<p>Confidence intervals use 500 physical-run bootstrap draws of fixed held-out predictions. They do not refit model selection, resample instruments as clusters, or correct for simultaneous subgroup comparisons. Sparse levels with fewer than eight physical runs have no interval. Results are descriptive, not confirmation that a particular volume should be used.</p>
<p>Five model rows include both controls and all three learned regressors. The nested-selection procedure is preserved as a sixth method in the CSV; it selected Extra Trees in these held-out-instrument tasks, so it is not duplicated in the plot.</p></section>
<section><h2>Downloads and audit</h2><p><a href="qpm_accuracy_and_error_by_volume.pdf">Four-page presentation PDF</a> · <a href="major_volume_model_comparison.csv">Displayed volumes: all model results</a> · <a href="volume_level_model_performance.csv">Every exact volume and all three validation schemes</a> · <a href="liquid_class_definition_inventory.csv">Full definitions and aliases</a> · <a href="metrics_validation.json">Source hashes, validation and method</a> · <a href="major_volume_display_selection.csv">Coverage-only display selection</a></p></section></body></html>'''
    (output/"qpm_volume_accuracy_report.html").write_text(report)
    outputs += ["qpm_accuracy_and_error_by_volume.pdf","major_volume_model_comparison.csv","qpm_volume_accuracy_report.html"]
    write_json(output/"render_validation.json",dict(status="passed",generated_utc=datetime.now(timezone.utc).isoformat(),
        metrics_validation_sha256=sha(output/"metrics_validation.json"),renderer_sha256=sha(Path(__file__)),
        artifact_sha256={name:sha(output/name) for name in outputs}))
    print(json.dumps(dict(report=str(output/"qpm_volume_accuracy_report.html"),figure=str(output/"qpm_pip2_accuracy_by_volume.png"))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=Path(__file__).resolve().parent/"qpm_volume_all_operations_20261027_v2")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=["metrics","render","all"],default="all")
    args=p.parse_args()
    if args.stage in ["metrics","all"]:metrics(args.study.resolve(),args.output.resolve())
    if args.stage in ["render","all"]:render(args.output.resolve())
