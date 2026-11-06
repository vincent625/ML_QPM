#!/usr/bin/env python3
"""Offline demonstration of real QPM Sentinel logic on artificial data.

This exercises barcode classification, path-level inference, Er72 exclusion and
report rendering. It does NOT parse a real CSA, load a trained model/reference
bank, measure delivered liquid or establish scientific diagnostic performance.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
PIPELINE = REPOSITORY / "pipeline"
SEED = 1729
PRODUCER = "qpm-sentinel-synthetic-component-demo-v1"
NOTICE = "SYNTHETIC DEMO — NOT SCIENTIFIC VALIDATION"


def core_modules():
    """Load the packaged flat modules, never a user's live reference project."""
    if not (PIPELINE / "qpm_dropout_plugin.py").is_file():
        raise RuntimeError("Packaged pipeline/ modules are missing; run this example from a complete repository.")
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import qpm_dropout_plugin as plugin
    import qpm_baseline_comparison as comparison
    import qpm_same_run_curves as curves
    from qpm_reference_qualification import classify_barcode_outcomes
    from qpm_waveform_reference import PROFILE_COLUMNS
    return plugin, comparison, curves, classify_barcode_outcomes, PROFILE_COLUMNS


def build_case(scenario="replicated", seed=SEED):
    """Create known synthetic path labels and physically matched toy traces.

    'replicated' has four affected/four surviving paths. 'sparse' has one
    affected/seven surviving paths and must abstain from inferential tests.
    Error-coded operations are deliberately conspicuous but remain excluded.
    """
    if scenario not in {"replicated", "sparse"}:
        raise ValueError("scenario must be replicated or sparse")
    plugin, _, _, classify, _ = core_modules()
    config = dict(plugin.DEFAULT_CONFIG)
    config.update(enable_volume_proxy=False, enable_csa_low_performance_diagnostics=False)
    rng = np.random.default_rng(seed)
    affected = 4 if scenario == "replicated" else 1
    observation = f"synthetic_{scenario}"
    lineage = pd.DataFrame([
        dict(observation_id=observation, archive_id="synthetic_archive", physical_run_id="synthetic_run",
             instrument_id="synthetic_instrument", run_name="ARTIFICIAL_DEMO_NOT_A_REAL_RUN",
             material_type="DNA", planned_order=path+1, is_control=False,
             read_count=float(1000+path*100 if path < affected else 8_000_000+path*50_000))
        for path in range(8)
    ])
    lineage = classify(lineage, config)
    t = np.linspace(0, 1, 32)
    records = []
    definitions = [
        (10, "upstream_control", "aspirate", 0.0, False),
        (50, "excluded_error_operation", "aspirate", 3.0, True),
        (100, "library_transfer", "aspirate", 1.0, False),
        (200, "barcode_delivery", "dispense", 1.0, False),
        (300, "barcode_recovery", "aspirate", 1.0, False),
        (400, "return_transfer", "dispense", 1.0, False),
        (500, "later_cleanup", "aspirate", 1.0, False),
    ]
    for start, step, operation, effect, er72 in definitions:
        sign = -1 if operation == "aspirate" else 1
        normal = sign*(85*(1-np.exp(-10*t))-20*t)
        for path in range(8):
            is_affected = lineage.iloc[path].path_status == "failed"
            curve = normal+rng.normal(0, 1.1, 32)
            if is_affected and effect:
                if operation == "aspirate":
                    curve = normal*.22+7*np.sin(4*np.pi*t)+rng.normal(0, .8, 32)
                else:
                    curve = normal*(1.0+1.2*t)+rng.normal(0, 1.1, 32)
                    curve[t>.66] *= .12
            if er72:
                curve *= 3
            records.append(dict(
                observation_id=observation, archive_id="synthetic_archive", physical_run_id="synthetic_run",
                instrument_id="synthetic_instrument", chip_lane="1", material_type="DNA",
                group_id=f"synthetic_{step}", family=f"Synthetic {step.replace('_', ' ')}",
                operation_id=f"synthetic_{start}_{path+1}", operation_index=start+path,
                planned_order=path+1, location=f"synthetic_well_{path+1}",
                path_status=lineage.iloc[path].path_status,
                operation=operation, pipette="PIP1", programmed_volume_ul=20.0, liquid_class_index=55,
                liquid_class_name="SYNTHETIC_LIQUID_CLASS", container_geometry_name="SYNTHETIC_WELL",
                workflow_stage="synthetic_workflow", process_step=f"synthetic_{step}",
                event_time=f"2020-01-01T00:{start//100:02d}:{path:02d}Z",
                reference_physical_runs=0, mapping_scope="synthetic_explicit_path_fixture",
                pressure_unit="raw_sensor_counts", command_context_status="command_verified",
                internal_context_verified=True, internal_context_fingerprint=f"synthetic_context_{operation}",
                liquid_class_fingerprint="synthetic_lc", container_geometry_fingerprint="synthetic_geometry",
                deck_geometry_fingerprint="synthetic_deck", command_settings_fingerprint=f"synthetic_{operation}",
                qpm_has_er72=er72, qpm_waveform_eligible=not er72,
                qpm_trace_quality_status="excluded_er72" if er72 else "checked_no_er72",
                qpm_exclusion_reason="Er72_synthetic_demo_fixture" if er72 else "",
                observed_profile=curve,
            ))
    return config, lineage, pd.DataFrame(records)


def analyze_case(scenario="replicated", seed=SEED):
    plugin, _, _, _, _ = core_modules()
    config, lineage, mapped = build_case(scenario, seed)
    tests = plugin.summarize_step_evidence(mapped, "DNA", config)
    return config, lineage, mapped, tests


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_case(output, scenario, seed=SEED):
    plugin, comparison, curves, _, profile_columns = core_modules()
    config, lineage, mapped, tests = analyze_case(scenario, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(run_date="SYNTHETIC", instrument="SYNTHETIC_INSTRUMENT", chip_type="SYNTHETIC",
                    chip_lane="1", assay_type="ARTIFICIAL DNA-ONLY EXAMPLE",
                    observation_id=f"synthetic_{scenario}")
    (output/"run_metadata.json").write_text(json.dumps(metadata, indent=2))
    lineage.to_csv(output/"barcode_lineage.csv", index=False)
    exported = mapped.copy()
    exported["observed_profile"] = exported.observed_profile.map(lambda v: json.dumps(v.tolist()))
    exported.to_csv(output/"synthetic_operation_ledger.csv", index=False)
    target = mapped.copy()
    target[profile_columns] = np.vstack(mapped.observed_profile)
    empty = pd.DataFrame()
    coverage = pd.DataFrame(columns=["operation_id", "reference_status", "reason"])
    members = pd.DataFrame(columns=["reference_physical_run_id", "reference_instrument_id"])
    no_history = dict(enabled=True, available=False, synthetic_demo=True,
                      reason="No historical data is supplied in this component demo.")
    payload = comparison.write_comparison(
        output, lineage, {"DNA": tests}, {"DNA": mapped}, target,
        empty, coverage, members, no_history,
        empty, coverage, members, no_history, config, plugin.PALETTE,
    )
    gallery = curves.build_same_run_curves(output, tests, mapped, plugin.PALETTE)
    comparison_html = comparison.html_section(output, payload)
    gallery_fragment = Path(gallery["outputs"]["fragment"]).read_text()
    table = tests[["first_operation_index", "step_label", "n_failed", "n_survivor",
                   "eligible_internal_comparison", "failed_path_deviation_auc", "permutation_p",
                   "bh_fdr_q", "evidence_role"]].to_html(index=False, escape=True, na_rep="Not evaluated")
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic QPM Sentinel demo</title>
<style>body{{font:16px/1.55 Arial,sans-serif;max-width:1240px;margin:30px auto;padding:0 20px;color:#264653;background:#FCFCF8}}
.notice{{padding:18px;background:#FBF2D8;border-left:6px solid #E9C46A;font-weight:bold}}h1,h2,h3{{color:#264653}}img,svg{{max-width:100%;height:auto}}
table{{border-collapse:collapse;font-size:13px;display:block;overflow:auto}}td,th{{padding:8px;border-bottom:1px solid #D9DED9;text-align:left}}
.guard,.muted{{padding:12px;background:#E8F3EF}}a{{color:#226F68}}.files a{{display:inline-block;margin:6px}}details{{background:white;margin:18px 0;padding:12px}}</style>
<div class="notice">{NOTICE}</div><h1>QPM Sentinel · {html.escape(scenario)} component demo</h1>
<p>All barcode counts, path identifiers, physical-context definitions and waveforms on this page are artificial.
The packaged <code>classify_barcode_outcomes</code>, <code>summarize_step_evidence</code>,
<code>write_comparison</code> and <code>build_same_run_curves</code> functions generated these results.</p>
<p class="guard">No customer data, CSA archive, historical reference, trained model or network access is used.
This is a reproducible software demonstration—not a diagnostic accuracy estimate, biological experiment,
full CSA-ingestion test, delivered-volume measurement or causal validation.</p>
<p><a href="../index.html">Back to synthetic-demo index</a> · <a href="barcode_lineage.csv">Artificial barcode outcomes</a> ·
<a href="synthetic_operation_ledger.csv">Artificial operation ledger</a></p>
<h2>Computed step tests</h2>{table}
<p>The replicated example permits formal path-level tests. The sparse example has one affected path and must
leave AUC/p/q unavailable. The deliberately abnormal Er72 group at operation50 is excluded in both cases.
Historical columns remain explicitly unavailable rather than inventing a reference population.</p>
{comparison_html}<h2>All same-run curve pairs</h2>{gallery_fragment}</html>"""
    (output/"report.html").write_text(document, encoding="utf-8")
    selected = payload["displayed_operations"].get("within_run", [])
    result = dict(scenario=scenario, synthetic=True, scientific_validation=False, seed=seed,
                  affected_paths=int(lineage.path_status.eq("failed").sum()),
                  surviving_paths=int(lineage.path_status.eq("survivor").sum()),
                  raw_operation_rows=len(mapped), er72_rows_excluded=int(mapped.qpm_has_er72.sum()),
                  computed_test_groups=len(tests), historical_reference_runs=0,
                  selected_same_run_events=selected, volume_models_loaded=False,
                  instrument_settings_changed=False, report="report.html",
                  component_api="real packaged core functions; no inference functions mocked")
    (output/"synthetic_case_manifest.json").write_text(json.dumps(_json(result), indent=2, allow_nan=False))
    return result


def generate_demo(output, scenario="all", seed=SEED):
    output = Path(output)
    marker = output/"synthetic_demo_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get("producer") != PRODUCER:
            raise ValueError("Choose an empty output folder; this command will not overwrite unrelated user files.")
    output.mkdir(parents=True, exist_ok=True)
    scenarios = ["replicated", "sparse"] if scenario == "all" else [scenario]
    # Only known demo output files are overwritten on repeat runs; nothing is deleted.
    results = [write_case(output/name, name, seed) for name in scenarios]
    links = "".join(f'<li><a href="{name}/report.html">{html.escape(name)} synthetic component report</a></li>' for name in scenarios)
    index = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>QPM Sentinel synthetic demo</title>
<style>body{{max-width:900px;margin:40px auto;padding:20px;font:18px/1.6 Arial;color:#264653}}.notice{{background:#FBF2D8;padding:18px}}</style>
<h1>QPM Sentinel · offline synthetic demo</h1><p class="notice">{NOTICE}</p>
<p>Artificial path counts and curves demonstrate the production core's report and abstention logic.
No raw archive, trained model, historical reference, remote instrument or API service is required.</p><ul>{links}</ul>
<p><strong>Replicated:</strong> four affected/four surviving paths; tests are calculated by the real core.
<strong>Sparse:</strong> one affected/seven surviving paths; curves remain visible, but inferential statistics are withheld.</p>
<p>Both examples include an abnormal error-coded group that must be excluded. Historical and volume-model evidence is unavailable by design.
Observed synthetic separation is not real-world diagnostic accuracy or evidence of causality.</p></html>"""
    (output/"index.html").write_text(index, encoding="utf-8")
    manifests = dict(producer=PRODUCER, seed=seed, synthetic=True, scientific_validation=False,
                     scenarios=results, notice=NOTICE, data_origin="deterministic toy generator; no customer inputs")
    (output/"synthetic_demo_manifest.json").write_text(json.dumps(_json(manifests), indent=2, allow_nan=False))
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--scenario", choices=["all", "replicated", "sparse"], default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = generate_demo(args.output, args.scenario, args.seed)
    print(json.dumps(dict(report=str(args.output/"index.html"), synthetic=True,
                          scientific_validation=False, scenarios=[r["scenario"] for r in result["scenarios"]]), indent=2))


if __name__ == "__main__":
    main()
