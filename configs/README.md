# Configuration examples

These configuration files preserve the source snapshot's screening and
reference rules. They are **investigation screens, not manufacturer-approved
acceptance limits or operating settings**. No instrument setting is changed.

| File | Used by |
| --- | --- |
| `qpm_dropout_plugin_config.json` | Pass explicitly as `--config` to `pipeline/qpm_dropout_plugin.py`. Unlisted options retain the module's defaults. |
| `outcome_reference_policy.json` | Outcome peers; loaded from the parsed project/reference-bank directory. |
| `qpm_waveform_reference_policy.json` | Physically compatible waveform references; loaded from the project/reference-bank directory. |
| `reference_qualification_policy.json` | Source-identical chip/analyzed-lane and barcode qualification policy for constructing a new reference bank. |

Do not edit policies inside a sealed existing reference bank. Use the supplied
reference-building tools to create and validate a new versioned bank. An
operation still needs its own sufficient compatible historical subset; merely
having many source runs does not make all operations scoreable.

## Optional private sequencing-outcome model study

The code-only repository does not contain an outcome-model training dataset,
fitted payloads or historical reference bank. If an approved private study is
installed separately, point the runtime to it before generating a report:

```bash
export QPM_CSA_MODEL_STUDY="$PWD/private_artifacts/outcome_models"
```

Without that environment variable, the portable default is
`private_artifacts/outcome_models/` relative to the repository. Missing model
evidence remains unavailable; it is never silently called normal. Do not add
`csa_diagnostic_model_study` to the plugin JSON: the original plugin deliberately
rejects unknown configuration fields, and the portable environment variable
avoids weakening that validation.

Nominal-volume models use a separate hash-pinned registry in the reference
bank. Use the model registration/release helpers rather than editing model
paths or hashes by hand.

## Network collection is opt-in

No SSH host, jump host or credentials are embedded. The collection command
requires explicit `--targets user@host` arguments and uses the caller's existing
SSH keys/agent and known-host configuration. Optional `--jump-host user@host`
applies only to the supplied targets. No collection happens by importing a
module or running the synthetic demo.
