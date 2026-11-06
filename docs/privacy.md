# Data handling and release boundaries

This source package is intended for a **private/internal repository**. It does
not by itself authorize publication of code, data or derived instrument assets.

## Included

- Runtime and analysis source code, portable configurations and unit tests.
- A clearly synthetic, offline software demonstration.
- Aggregate benchmark tables and newly rendered summary figures.
- Source/version/hash provenance and interpretation/validation documentation.

## Excluded from the Git upload

- CSA containers or extracted archive members.
- Patient/sample identifiers, barcode-level outcomes and individual-run reports.
- Raw QPM traces, run-level feature matrices, private instrument addresses and
  instrument access/SSH configuration.
- Trained joblib/pickle payloads, production reference banks and related
  private source-study datasets.
- Corporate slide templates/logos, full screenshots of identifiable reports,
  and the presentation's illustrative replacement RawTrace box.

The scientific figures in this repository show actual frozen aggregate results.
The synthetic demo is labeled separately and is never used as evidence for
model accuracy or root-cause performance.

## Your own data

Use approved storage outside the tracked source tree and review `.gitignore`
before adding files. Generated reports can contain run metadata and sensitive
provenance, even if they do not contain the original CSA container.

If you export private reference/model artifacts, keep them separate from the
Git upload. Check internal access policy, integrity and model-deserialization
safety before sharing them with another user or machine.

## Scanner limitations

The packaging checker detects selected common secret formats, machine-specific
paths, internal host patterns and accidental large/private file types. It does
not guarantee anonymization or detect every sensitive value. A clean automated
scan is not a substitute for owner and data-governance review.

The ZIP builder uses an explicit exclusion/forbidden-file policy in addition to
the repository's Git ignore rules. Reviewed aggregate CSVs belong under
`docs/data/`; run-level CSVs do not belong in the source upload. Adding a custom
`.gitignore` rule alone does not change the ZIP builder's policy. Inspect the
resulting manifest and rerun the checker before distribution.

