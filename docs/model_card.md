# Model and diagnostic-evidence card

## Intended use

Retrospective research/engineering investigation of apparent barcode dropout,
imbalance and low sequencing output. The system prioritizes evidence for human
review. It is not a validated clinical tool, autonomous root-cause engine,
delivered-volume meter or instrument-setting optimizer.

## Distinct tasks

| Task | Inputs and target | Main limitation |
|---|---|---|
| Barcode-aware QPM investigation | Recorded outcomes plus physically mapped aspiration/dispense waveforms | Outcomes are already known; same-run separation is not held-out diagnostic sensitivity |
| Nominal-volume recognition | Quality-cleared waveforms; programmed command/class as label | No independent delivered-volume ground truth |
| Sequencing outcome regression | Approved context and during/post-run diagnostics; total reads or mean raw read length | Same-run retrospective associations, not future-run prognosis |
| Prior-run forecasting experiments | Earlier run outcomes/CSA features predicting a later event | Reliable added CSA value was not established in the tested experiment |

Raw read accuracy was a requested outcome but lacks adequate usable labels.
Do not substitute an invented accuracy endpoint or AQ20 length for raw length.

## Population and validation

- Screened cohort: 2,630 archive entries, 1,789 physical runs, 43 instruments.
  These are not interchangeable with model-specific labeled sample counts.
- Frozen unseen-instrument outcome benchmarks: 1,665 read-count and 1,663
  read-length physical runs, across 42 instruments.
- The six-task PIP2 volume-classification comparison uses 450 physical runs per
  task, five commanded volume classes per task and 43 instruments.
- All-operation volume regression uses a different event-level dataset and
  run-balanced evaluation; see the exported aggregate regression table for its
  denominator. It must not be merged with the six-task classification result.
- Related lanes, exports and reanalyses remain grouped by physical run.
- Outer views include physical-run grouped, held-out instrument and chronological
  splits. Model/feature selection and preprocessing belong inside training folds,
  with nested selection in the studies that use it.
- Compare context-only controls as well as median regression/majority
  classification baselines. Much of the outcome-model gain is explained by
  context alone.

The chronological read-length improvement beyond context was uncertain
(approximately 7.42%, bootstrap interval −5.25% to 19.03%). Do not generalize the
positive held-out-instrument comparison to every temporal setting.

## Feature policy and leakage

- **Tier A:** explicitly commanded pre-step settings. No credible actionable
  tuning parameter/setpoint effect is established by these studies.
- **Tier B:** instrument, assay, chip/lane, workflow, date/lot and other adjustment
  context. These are not tuning recommendations or fault mechanisms.
- **Tier C:** measured pressures, process diagnostics, signal/image descriptors
  and other during/post-run evidence. Interpret as diagnostics only.
- **Tier D:** targets, derived/rounded target copies, outcome reports or labels,
  identifiers and sensitive fields. Exclude from predictor models.

The strict reviewed policy quarantines 13,749 descriptors, including the
TrueLoading, EdgeEffects, LaneDiagnostics and ReseqDiagnostics families.
Do not reuse earlier, withdrawn loading rankings as validated results.
Ambiguous chip pixel-count/geometry proxies are not validated health/loading
percentages. Original pinned-pixel percentages require their source fields and
checked denominator; they are separate supporting evidence, not replacements
for a different ranked feature.

## Waveform and reference guards

- Exclude Er72, missing/invalid traces and unverified command context from
  scoring and selected evidence.
- Match liquid class, volume, pipette, physical geometry, command settings and
  units. Respect additional workflow/location/chip constraints where configured.
- Retain sparse same-run curves when physically comparable, but require at
  least three independent affected and three surviving paths for the formal
  selective statistical branch. Retries are not extra independent paths.
- Historical scoring requires at least eight compatible physical runs for
  that operation. Keep same-instrument and other-instrument results distinct.
- One run contributes a median vote where the reference calculation requires
  independent run-level votes. “Outcome-qualified” is not proven fault-free.
- Retrospective QPM/model snapshots may contain later historical source dates.
  Do not automatically present those examples as a prospective, strictly-prior
  deployment. Low-read metric peer checks use their own recorded temporal rules.

## Interpretation boundaries

Held-out permutation importance measures prediction-error sensitivity to a
shuffle. It is not causal effect, reads lost, individual attribution or a simple
correlation. Correlated descriptors can split or mask importance.

RawTrace image descriptors are rendered-image proxies, not calibrated signal,
SNR or a specific nucleotide/salt diagnosis. Measured vacuum/pipette summaries
are not commanded setpoints. Units are not interchangeable without source-based
conversion. An anomaly percentile is not a validated probability of failure.

Earliest evidence means earliest **covered/qualifying** event, not necessarily
the first abnormal operation or causal origin. Later abnormalities may be
downstream consequences or unrelated. A report can abstain or retain counter-
evidence. No automatic setting change is authorized.

## Required next validation

1. Blinded expert-reviewed cases and controls, with assay/material intent known.
2. Independently measured delivered volumes under approved operating conditions.
3. Representative instruments, software, assays, chip types and maintenance states.
4. A prospective shadow-mode pilot with verified availability timestamps.
5. False-positive, miss, abstention and compatible-reference coverage monitoring.
6. Measured investigation time, repeat-run cost and user-experience outcomes.

No validated root-cause accuracy, actual-volume accuracy, prospective forecasting
benefit, causal tuning improvement or measured cost saving is claimed here.

