"""All physically comparable same-run curves, independent of test power.

This module renders saved QPM observations only. It does not calculate a test,
rank a failure, replace historical evidence, or measure transferred volume.
Inline SVG keeps a large, offline step gallery compact without new libraries.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpm_internal_comparison import collapse_paths, profile
from qpm_trace_quality import eligible_waveforms

VERSION = "qpm-same-run-curves-1.0.0"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _value(value):
    if isinstance(value, dict): return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_value(v) for v in value]
    if isinstance(value, np.generic): return _value(value.item())
    if value is pd.NA or value is pd.NaT: return None
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _curves_svg(group: pd.DataFrame, palette: dict, unit: str) -> str:
    """Draw all independent paths and medians; never synthesize a counterpart."""
    failed = group.path_status.eq("failed").to_numpy()
    survivor = group.path_status.eq("survivor").to_numpy()
    assert failed.any() and survivor.any() and (failed | survivor).all()
    matrix = np.vstack(group.observed_profile.map(profile))
    assert matrix.shape[1] == 32 and np.isfinite(matrix).all()
    width, height, left, right, top, bottom = 960, 370, 82, 20, 43, 58
    lo, hi = float(matrix.min()), float(matrix.max())
    pad = max((hi - lo) * .09, 1.)
    lo -= pad; hi += pad
    xx = np.linspace(left, width - right, 32)
    yy = lambda values: top + (hi - values) / (hi - lo) * (height - top - bottom)
    esc = lambda value: html.escape(str(value), quote=True)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Same-run affected and surviving independent QPM path curves" viewBox="0 0 {width} {height}" style="width:100%;height:auto;background:white;font-family:Arial,sans-serif">']
    for tick in np.linspace(lo, hi, 5):
        y = float(yy(tick))
        pieces.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#D9DED9" stroke-width=".7"/>')
        pieces.append(f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#264653">{tick:.3g}</text>')
    for tick in [0, 25, 50, 75, 100]:
        x = left + tick / 100 * (width - left - right)
        pieces.append(f'<text x="{x:.2f}" y="{height-bottom+21}" text-anchor="middle" font-size="12" fill="#264653">{tick}</text>')
    for values, color, path_class in [(matrix[failed], palette["coral"], "failed"), (matrix[survivor], palette["teal"], "survivor")]:
        for curve in values:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xx, yy(curve)))
            pieces.append(f'<polyline data-curve-kind="individual" data-path-class="{path_class}" points="{points}" fill="none" stroke="{esc(color)}" stroke-width="1" opacity=".3"/>')
        median = np.median(values, axis=0)
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xx, yy(median)))
        pieces.append(f'<polyline data-curve-kind="median" points="{points}" fill="none" stroke="{esc(color)}" stroke-width="2.7"/>')
    pieces.append(f'<text x="{left}" y="22" font-size="14" fill="{esc(palette["coral"])}">Affected paths n={int(failed.sum())}</text>')
    pieces.append(f'<text x="{left+240}" y="22" font-size="14" fill="{esc(palette["teal"])}">Surviving paths n={int(survivor.sum())}</text>')
    pieces.append(f'<text x="{left+480}" y="22" font-size="12" fill="#65716D">thin: individual paths; thick: medians</text>')
    pieces.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-8}" text-anchor="middle" font-size="13" fill="#264653">Normalized operation progress (%)</text>')
    label = "Baseline-centered raw sensor counts" if unit == "raw_sensor_counts" else "QPM source unit: " + unit
    pieces.append(f'<text transform="translate(18,{(top+height-bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12" fill="#264653">{esc(label)}</text></svg>')
    return "".join(pieces)


def build_same_run_curves(output_dir: Path, tests: pd.DataFrame, mapped: pd.DataFrame, palette: dict) -> dict:
    """Export every verified paired group, including 1-versus-1 comparisons."""
    from qpm_baseline_comparison import display_internal_tests
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    display = display_internal_tests(tests, mapped)
    available = display.loc[display.curve_available.eq(True)].copy()
    if not available.empty:
        available = available.sort_values("first_operation_index", kind="stable")
    parts, ledger, materials = [], [], {}
    esc = lambda value: html.escape(str(value), quote=True)
    for _, record in available.iterrows():
        group_id = record["group_id"]
        operations = eligible_waveforms(mapped.loc[mapped.group_id.eq(group_id)]).copy()
        operations = operations.loc[operations.path_status.isin(["failed", "survivor"])].drop_duplicates("operation_id")
        group = collapse_paths(operations)
        assert int(group.path_status.eq("failed").sum()) == int(record["curve_n_failed"])
        assert int(group.path_status.eq("survivor").sum()) == int(record["curve_n_survivor"])
        material = str(record.get("material_type", "unresolved"))
        first_for_material = material not in materials
        materials.setdefault(material, dict(groups=0, descriptive_only_groups=0))
        materials[material]["groups"] += 1
        descriptive = bool(record["curve_descriptive_only"])
        materials[material]["descriptive_only_groups"] += int(descriptive)
        start = int(record["first_operation_index"])
        title = str(record.get("step_label", record.get("family", "Recorded workflow group")))
        badge = "Descriptive only — insufficient replication for formal testing" if descriptive else "Statistically testable group — see existing test results; not a proven cause"
        note = (f"{int(record['curve_n_failed'])} affected / {int(record['curve_n_survivor'])} surviving independent paths. "
                "Statistical tests retain the configured minimum; no AUC, permutation p-value or BH q-value is created here.")
        unit_values = group.pressure_unit.dropna().astype(str).unique() if "pressure_unit" in group else []
        unit = str(unit_values[0]) if len(unit_values) == 1 else "unresolved"
        ident = hashlib.sha256(str(group_id).encode()).hexdigest()[:16]
        parts.append(f'<details class="same-run-curve" id="same-run-group-{ident}"{" open" if first_for_material else ""}><summary>{esc(material)} · op {start} · {esc(title)} · n={int(record["curve_n_failed"])}/{int(record["curve_n_survivor"])}</summary><p><strong>{esc(badge)}</strong>. {esc(note)}</p>{_curves_svg(group, palette, unit)}<p class="muted">Class: {esc(record.get("screen_label", "unresolved"))}; workflow ordering is not a claim of earliest failure. Source units are retained; no transferred-volume error is inferred.</p></details>')
        for _, path in group.iterrows():
            source_ids = operations.loc[operations.planned_order.eq(path.get("planned_order")), "operation_id"].astype(str).tolist() if pd.notna(path.get("planned_order")) else operations.loc[operations.location.eq(path.get("location")), "operation_id"].astype(str).tolist()
            ledger.append(dict(material_type=material, group_id=str(group_id), first_operation_index=start,
                step_label=title, independent_path=str(path["independent_path"]), path_status=str(path["path_status"]),
                planned_order=_value(path.get("planned_order")), pressure_unit=unit,
                source_operation_ids="|".join(source_ids), observed_profile=json.dumps(profile(path["observed_profile"]).tolist()),
                descriptive_only=descriptive, formal_statistics_recomputed=False))
    count = len(available)
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata_text = " · ".join(str(metadata.get(key, "Not recorded")) for key in ["instrument", "run_date", "chip_type", "chip_lane", "assay_type"])
    intro = f'''<section id="same-run-affected-survivor-curves"><h2>Same-run affected and surviving curves — visible at any replication</h2>
<p class="muted" style="overflow-wrap:anywhere">{esc(metadata_text)}<br>{esc(metadata.get('run_name', 'Run name not recorded'))}</p>
<p><strong>{count} physically comparable workflow groups</strong>. Curves require at least one affected and one surviving independent path, not three per class. DNA/RNA stay separate. Thin lines show individual independent paths and thick lines show medians; retries are collapsed to one median vote per path.</p>
<p class="guard">These are descriptive views of recorded data, not extra statistical evidence. The formal minimum remains unchanged. Under-replicated groups receive no new AUC, permutation p-value, BH q-value, earliest-failure nomination or causal claim. Historical comparisons remain independent. Er72, unverified configuration and missing counterparts remain excluded/unavailable.</p>
<p>Every available group is listed chronologically; the first group for each material is expanded. This ordering is not a failure ranking. Open any workflow group to inspect its curves.</p>'''
    if not count:
        intro += '<p><strong>No paired, physically verified affected/surviving curves are available.</strong> Missing curves do not establish normality. Inspect the coverage ledger for missing groups, configuration, trace-quality and counterpart limitations.</p>'
    fragment = intro + "".join(parts) + '''<p class="files"><a href="qpm_same_run_curves.html">Open all same-run curves separately</a> · <a href="qpm_same_run_curves.csv">Exact independent-path waveforms</a> · <a href="qpm_same_run_curve_coverage.csv">Display availability and exclusions</a> · <a href="qpm_same_run_curve_manifest.json">Display policy and provenance</a></p></section>'''
    fragment_path = output_dir / "qpm_same_run_curve_fragment.html"
    fragment_path.write_text(fragment, encoding="utf-8")
    standalone = output_dir / "qpm_same_run_curves.html"
    standalone.write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Same-run affected/surviving QPM curves</title><style>body{font:15px/1.5 Arial,sans-serif;color:#264653;background:#fcfcf8;max-width:1150px;margin:25px auto;padding:20px}summary{cursor:pointer;font-weight:bold}details{background:white;border:1px solid #d9ded9;border-radius:5px;padding:12px;margin:12px 0}.guard{background:#fbf2d8;padding:12px;border-left:5px solid #e9c46a}.muted{color:#65716d;font-size:13px}a{color:#16796e}svg{max-width:100%;height:auto}</style></head><body>''' + fragment + "</body></html>", encoding="utf-8")
    ledger_path = output_dir / "qpm_same_run_curves.csv"
    columns = ["material_type", "group_id", "first_operation_index", "step_label", "independent_path", "path_status", "planned_order", "pressure_unit", "source_operation_ids", "observed_profile", "descriptive_only", "formal_statistics_recomputed"]
    pd.DataFrame(ledger, columns=columns).to_csv(ledger_path, index=False)
    coverage_path = output_dir / "qpm_same_run_curve_coverage.csv"
    if not coverage_path.is_file():
        # Supports isolated diagnostic use; the main plugin normally writes
        # this exact display-policy table alongside the three-comparator audit.
        display.to_csv(coverage_path, index=False)
    outputs = dict(fragment=str(fragment_path), report=str(standalone), curves=str(ledger_path), coverage=str(coverage_path))
    payload = dict(version=VERSION, status="available" if count else "unavailable_no_paired_verified_curves",
        curve_groups=count, descriptive_only_groups=int(available.curve_descriptive_only.sum()),
        independent_path_group_curves=len(ledger), materials=materials, all_available_groups_retained=True,
        display_minimum_affected_paths=1, display_minimum_surviving_paths=1,
        formal_statistical_minimum_unchanged=True, statistical_scores_and_nominations_unchanged=True,
        selection_policy="all_verified_paired_groups_chronological_first_material_group_open_not_failure_ranking",
        er72_excluded=True, unknown_paths_not_treated_as_survivors=True, historical_reference_required=False,
        individual_paths_and_medians_shown=True, new_inferential_statistics=False,
        outputs=outputs, source_sha256={str(Path(__file__)): _sha(Path(__file__))},
        output_sha256={path: _sha(path) for path in outputs.values()})
    for name in ["qpm_step_evidence.csv", "qpm_operation_ledger.parquet"]:
        path = output_dir / name
        if path.is_file(): payload["source_sha256"][str(path)] = _sha(path)
    if metadata_path.is_file(): payload["source_sha256"][str(metadata_path)] = _sha(metadata_path)
    for name in ["qpm_baseline_comparison.py", "qpm_internal_comparison.py", "qpm_trace_quality.py"]:
        path = Path(__file__).resolve().parent / name
        payload["source_sha256"][str(path)] = _sha(path)
    manifest = output_dir / "qpm_same_run_curve_manifest.json"
    payload["manifest_path"] = str(manifest)
    manifest.write_text(json.dumps(_value(payload), indent=2, allow_nan=False), encoding="utf-8")
    return _value(payload)


def html_section(payload: dict) -> str:
    path = payload.get("outputs", {}).get("fragment")
    if path and Path(path).is_file(): return Path(path).read_text(encoding="utf-8")
    return '<section id="same-run-affected-survivor-curves"><h2>Same-run affected and surviving curves</h2><p>Display unavailable or unresolved; missing evidence is not normality.</p></section>'


def cache_artifacts_valid(manifest_path: Path) -> bool:
    path = Path(manifest_path)
    if path.is_dir(): path /= "qpm_same_run_curve_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != VERSION or not payload.get("source_sha256") or not payload.get("output_sha256"):
            return False
        for field in ["source_sha256", "output_sha256"]:
            if any(not Path(name).is_file() or _sha(name) != expected for name, expected in payload[field].items()):
                return False
        return all(Path(name).is_file() for name in payload.get("outputs", {}).values())
    except (OSError, ValueError, KeyError, TypeError):
        return False
