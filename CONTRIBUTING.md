# Contributing internally

1. Work in a private/internal repository and use approved synthetic or
   appropriately authorized data.
2. Preserve physical-run grouping, source provenance, units and missing-data
   states. Never turn sparse or absent evidence into a confirmed diagnosis.
3. Keep same-run, same-instrument and cross-instrument reference views separate.
4. Add a unit/regression test for changes to calls, eligibility, matching,
   evidence selection, volume guards or report interpretation.
5. Run `python tools/run_tests.py` and `python tools/validate_package.py`.
6. If frozen results change, version the dataset/model/policy, rerun the relevant
   held-out validation and update the model card and aggregate result provenance.

Do not retune thresholds using held-out targets, silently change historical
bank membership, revive quarantined leakage features, add patient data to a test
fixture, or describe nominal-volume resemblance as measured delivery.

