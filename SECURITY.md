# Security and sensitive data

- Do not put CSA archives, patient/sample identifiers, customer reports,
  instrument credentials, internal network details or model payloads into issues,
  pull requests, public repositories or CI logs.
- Report sensitive findings to the maintainers through an approved internal
  channel. Do not create a public issue containing the underlying data.
- Only use authorized archives and read-only instrument access. This package
  does not provision SSH keys or change instrument settings.
- Treat archives as untrusted inputs. Process them with local resource limits
  and least-privilege filesystem access; compressed input can be unexpectedly
  large or malformed.
- Only load trusted and integrity-checked joblib/pickle artifacts. Python model
  deserialization can execute code; a filename or extension is not a safety check.
- Output reports and production-derived reference assets can themselves contain
  sensitive metadata even when the input archive is not retained.

The package scanner checks selected common patterns; it is **not** a complete
data-loss-prevention or security review. Review access controls and publication
approvals independently.

