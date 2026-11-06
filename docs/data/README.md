# Frozen aggregate result tables

These files support the README figures. They contain aggregate statistics, not
customer CSA archives, individual-run records, individual waveform traces, trained model
objects, patient data or sample identifiers.

| File | Content / interpretation |
| --- | --- |
| `cohort_summary.json` | Screened archive/run/instrument counts; reference source pools; earlier batch and 0.18.0 release counts. Units and cohorts differ. |
| `outcome_model_performance.csv` | 30 frozen model/outcome/validation rows. Read-count MAE is in reads; read-length MAE is in base pairs. These are retrospective same-run diagnostic models. |
| `qpm_volume_classification.csv` | Six tasks × eight model procedures; values are balanced-accuracy fractions, displayed as percentages. |
| `qpm_classification_tasks.csv` | Five commanded volume classes per task, operation, pipette, and task-level denominator. All six tasks use 450 physical runs and 43 instruments. |
| `qpm_volume_regression.csv` | 18 run-balanced nominal-volume regression summaries. MAE is in µL; it is not an accuracy percentage or measured delivered-volume error. |
| `diagnostic_feature_importance.csv` | Ten frozen held-out individual-feature permutation results, with exploratory outer-fold intervals and interpretation limits. |
| `forecast_holdout.csv` | Six aggregate rows from a **separate**, earlier six-instrument forecasting study. Each endpoint has 54 final chronological-holdout runs; lower Brier score is better. |
| `barcode_workflow_group_summaries.csv` | Observed, de-identified pointwise median/IQR QPM summaries: two selected workflow steps × two outcome-defined groups × 32 progress points. No individual paths or traces are exported. |
| `barcode_workflow_example_summary.json` | Outcome counts, step descriptions and saved within-run post-hoc separation scores for the group-summary example. No instrument, run, sample, barcode or exact date/time identifiers. |
| `source_provenance.json` | Relative original source labels and SHA-256 hashes, bundled-table hashes, and interpretation limits. No raw source data are bundled. |

Regenerate the five figures without training or access to private source data:

```bash
python tools/render_results.py
```

The renderer checks bundled table hashes before plotting, preserves all input
tables, and writes `docs/assets/render_validation.json`. Figures use a shared
nature-inspired palette and retain units, denominators, baselines and caveats.

## Boundaries

- Reference qualification does **not** certify a run as fault-free. A target
  operation uses only its compatible subset, not all 910 usable source runs.
- The 763 completed scope reports came from an earlier cohort pass; the 0.18.0
  release regenerated three examples, not the whole cohort. Zero processing
  failures is not validated diagnostic sensitivity or specificity.
- Nominal-volume targets are programmed commands, not independently measured
  delivered liquid. Do not interpret `1 - MAE` as accuracy.
- Sequencing models use context and during/post-run diagnostics. Forecasting
  results are a different experiment and did not establish reliable added value
  from broad earlier-CSA features over recent outcome history.
- Feature importance is not causality or reads lost. RawTrace metrics are
  rendered-image proxies. Daggered count/geometry proxies are withheld as
  loading or chip-health measurements.
- The barcode example is observed, not synthetic, but includes only de-identified
  aggregate counts and group median/IQR curves. Groups are defined after outcomes
  are known. Within-run AUC 1.00 / 0.9583 is post-hoc path-score separation, **not**
  held-out diagnostic sensitivity. Earlier untestable/borderline groups existed;
  earliest qualifying evidence is not necessarily the first fault or a cause.
