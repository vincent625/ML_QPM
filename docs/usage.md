# Usage and workflow entry points

Run commands from the repository root in the tested Python environment. Keep
approved data and private snapshots outside the tracked source folder.

## Offline demonstration and tests

```bash
python examples/run_synthetic_demo.py --output outputs/demo
python tools/run_tests.py
python tools/validate_package.py
python tools/render_results.py
```

The demonstration exercises actual diagnostic-core functions using generated
data. It does not exercise the complete CSA parser, production histories or
registered volume models. Unit tests are software checks—not evidence of
clinical, delivered-volume or causal diagnostic accuracy.

## Local CSA replay

```bash
python pipeline/run_manual_csa_plugin.py \
  --input-dir /path/to/approved/csa_folder \
  --baseline /path/to/approved/reference_snapshot \
  --out outputs/manual_replay \
  --stage all
```

The runner's stages are `inspect`, `prepare`, `reports` and `all`. The baseline
must be compatible and correctly relocated/validated. New targets are not
automatically promoted into the historical good-run bank.

For an existing parsed project, use:

```bash
python pipeline/qpm_dropout_plugin.py \
  --project /path/to/approved/project \
  --observation-id YOUR_OBSERVATION_ID \
  --config configs/qpm_dropout_plugin_config.json \
  --output-dir outputs/my_run
```

Replace `--observation-id` with `--archive-id` or `--run-name` if appropriate;
exactly one selector is required. Check the report's source, temporal eligibility,
reference coverage, trace exclusions and unresolved states before interpretation.

## End-to-end processing and studies

The flat `pipeline/` layout preserves local module imports used by the prototype.
The following entry points are included; inspect their own `--help` output before
using them on an approved dataset.

| Stage | Entry points |
|---|---|
| Optional read-only collection | `collect_csa.py` |
| Stage locally downloaded archives | `stage_downloaded_csa.py` |
| Parse archives and source provenance | `analyze_csa.py` (`--skip-models` for extraction only) |
| Fleet outcome screening | `screen_all_csa_runs.py` |
| Workflow/QPM/spatial feature extraction | `workflow_fingerprint_pilot.py` |
| Physical operation definitions and QPM context | `analyze_qpm_volume_diagnostics.py` |
| Historical bank creation/qualification | `build_expanded_reference_bank.py`, `build_qualified_reference_bank.py` |
| Nominal-volume model studies | `retrain_qpm_volume_models.py`, `prepare_all_qpm_volumes.py`, `train_all_qpm_volumes.py` |
| Sequencing outcome models | `train_multimodal_csa.py`, `train_expanded_csa_outcomes.py` |
| Exploratory prior-run forecasting | `forecast_next_run_csa.py`, `forecast_10m_eight_runs.py` |
| Single-run diagnostics | `qpm_dropout_plugin.py`, `run_manual_csa_plugin.py` |
| Checkpointed cohort diagnostics | `run_flagged_qpm_batch.py` |

For example, parsing without model fitting is separate from downstream studies:

```bash
python pipeline/analyze_csa.py --project /path/to/approved/project --skip-models
```

Training scripts consume the required prepared tables and policies; they do not
reconstruct the complete private study from the aggregate README CSVs. Model
deserialization and source-study registration require the trusted private
assets described in [the artifact guide](private_assets.md).

## Collector safety

The packaged collector is separate from diagnostic execution. Supply authorized
SSH targets explicitly; no production host roster or credentials are included.
Use existing SSH configuration, keys or an agent. Remote operations must remain
read-only: no `sudo`, permission changes, remote extraction, renaming or deletion.

Start with inventory only, using your own authorized host:

```bash
python pipeline/collect_csa.py \
  --targets user@instrument.example \
  --project /path/to/approved/collection \
  --mode inventory
```

Only after reviewing inventory and available storage, select `--mode all` or
`--mode transfer` if authorized. An explicit `--jump-host user@jump.example` is
optional; the repository contains no fleet roster or default jump host.

The workflow limits discovery to configured analysis-output roots, validates
container content locally, uses partial/resumable transfers and records local
hashes. Remote collection should be reviewed and authorized for the instruments
being accessed. The local demo and ordinary unit tests never contact them.

## Outputs to review

- The main HTML report and its run-information/source labels.
- Material-aware barcode calls and unresolved/controlled sample exclusions.
- Three separate reference views and their operation-level coverage.
- Earliest covered/qualifying evidence and later supporting events.
- Full descriptive same-run curve gallery, including under-replicated cases.
- Low-read QPM and CSA-metric sections; retained counter-evidence and abstentions.
- Configuration, source/model versions, input hashes and diagnostic warnings.

An anomaly is a troubleshooting clue, not a causal verdict. Do not automatically
adjust settings or infer actual delivered volume from nominal-volume resemblance.

