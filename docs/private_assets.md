# Production reference and model assets

The Git repository contains source, tests, policies, a synthetic demo and
aggregate scientific results. **It intentionally does not contain production
reference banks, trained model payloads or customer-derived run tables.**

## Why these are separate

The active derived reference bank is hundreds of MiB, and the associated QPM
and sequencing-model study folders total several GiB. More importantly, these
are private derived datasets with run metadata—not ordinary source files.

For a real historical/ML investigation you need an approved compatible snapshot:

1. Parsed run/observation outcomes and provenance.
2. Quality-cleared QPM waveforms and physical operation context.
3. Outcome/reference qualification data and matching policies.
4. Registered nominal-volume metadata and trusted model payloads when requested.
5. Sequencing-model rankings, reviewed feature policies and supporting metric
   data for the low-read section.

The synthetic demo needs none of these. It must not be confused with a complete
reproduction of the private study or a parser benchmark.

## Integrity and relocation

Existing production snapshots use validated bank manifests, activation pointers,
artifact hashes and model registrations. Some registration metadata contains
absolute local study paths. **Do not copy a bank, change one path and leave its
integrity metadata stale.**

Keep the original snapshot immutable. Use an approved export/restore procedure
that verifies the original bundle, records any relocation-only edits, updates
affected manifests/pointers consistently and validates the restored runtime.
Do not mark an unvalidated reconstruction as a scientifically revalidated bank.

**Original CSA access can still be required.** Some historical context and
manual-replay paths check the source archive's existence, size, modification
time and parser fingerprint before accepting a cache. A copied derived bank
and model folder alone therefore do not guarantee portable historical replay.
The offline demonstration is independent of this requirement. Use
`python tools/check_private_assets.py --help` for the privacy-preserving asset
preflight, and read [the detailed contract](private_asset_contract.md).

`active_reference_bank.json` is machine-specific and intentionally ignored by
Git. Explicit `--project`/`--baseline` paths select approved local snapshots;
an invalid activation pointer must fail closed, not silently pick an older bank.

Only load trusted model payloads. A correct path is not proof that a joblib/pickle
artifact is safe. Verify hashes and ownership before deserialization.

The low-read model study is selected separately:

```bash
export QPM_CSA_MODEL_STUDY="/path/to/approved/outcome_model_study"
```

Without that environment variable the packaged runtime looks under
`private_artifacts/outcome_models/`. See [configuration notes](../configs/README.md).

## Rebuilding from approved inputs

The `pipeline/` snapshot includes archive parsing, screening, reference-building,
qualification and model-training entry points. Use their `--help` commands and
[usage guide](usage.md). Full rebuilds require approved private inputs and
appropriate compute; README aggregate tables alone cannot reproduce training.

Archive collection is optional and read-only on instruments. No stored
credentials, fleet key-deployment helper or production host roster is included.

