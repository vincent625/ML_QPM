# Data, units and provenance contract

## Independent observations

A retained analysis scope is a physical sequencing run plus analyzed chip lane
or lane set. Preserve `physical_run_id`, `instrument_id`, `chip_lane`,
`run_start_time`, `analysis_id` and `observation_id` for grouping/provenance—not
as predictive IDs. Repeated exports, sibling lanes and reanalyses of the same
raw run must never leak across training and validation groups.

For a private reference/model snapshot, its own manifest and policy files are
authoritative. Different studies retain different cohorts; a screened archive,
physical run, lane observation, operation and waveform are not interchangeable.

## Outcomes

| Output | Preferred source / field | Unit / restriction |
|---|---|---|
| `total_reads` | `BaseCaller.json`: `Filtering.LibraryReport.final_library_reads`; explicit `Info.csv` final-read fallback | Read count; retain source and discrepancies |
| `mean_read_length_bp` | `ionstats_basecaller.json`: `full.mean_read_length` | Base pairs; raw length, not AQ20 length |
| `aq20_mean_read_length_bp` | `ionstats_basecaller.json`: `Q20.mean_read_length` | Separate secondary outcome |
| `raw_read_accuracy_pct` | `Info.csv`: Run QC Evaluation Metrics / Raw Read Accuracy | Percent; currently insufficient labels for validated modeling |

Compare duplicate sources instead of averaging inconsistent values. Missing is
not zero. Retain the original source member, field, parsing status and any
discrepancy warning.

## QPM and workflow context

- Debug/pressure logs provide aspiration and dispense responses. Parse recorded
  operation, workflow step, liquid class, pipette, volume, attempt and time.
- Preserve source/destination and geometry for physical mapping, with sensitive
  identifying material kept only in approved local data stores.
- Match compatible commands and physical definitions, not just operation number.
- The nominal-volume models use 32-point, baseline-centered waveforms when the
  applicable quality/context/model guards pass. Normalized progress is not
  elapsed seconds or calibrated flow.
- QPM raw sensor counts are not interchangeable with psi, kPa or mbar. Convert
  only when source units and a valid conversion are known.
- Repeated attempts are dependent events; Er72 traces are excluded from scoring
  and selected evidence. Missing/invalid traces remain unresolved.

## Vacuum, preparation and plugin evidence

`vacuum_log.csv`, `libPrep_log.csv`, workflow transitions, configuration tables,
spatial nucleotide/block diagnostics and plugin outputs can contribute context
and diagnostic features. Their presence and schemas vary by archive/software.

Only explicit commanded pre-step settings belong to the tunable tier. Measured
vacuum/pipette traces, temperatures, retries and outcome-dependent summaries
remain diagnostics. A measured pressure cannot be relabeled as a setpoint.

RawTrace descriptors summarize rendered images; they are not direct physical
signal/SNR or chemistry measurements. Pixel counts require source-specific
denominators and geometry interpretation. Strict leakage exclusions override
“use all parameters.” See the [model card](model_card.md).

## Safe repository fixtures

The upload repository contains aggregate benchmark tables, deidentified
group-level measured waveform summaries and synthetic software-test inputs.
It contains no raw CSA containers, individual waveform traces, sample/barcode
records or trained production model payloads. Full scientific training requires
the original approved private inputs, not these summary tables.

