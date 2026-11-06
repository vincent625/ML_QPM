# Private-asset integrity and portability contract

The source repository and offline synthetic demo do not contain customer CSA
archives, historical waveform banks or trained model payloads. An approved
private deployment can supply these separately. A private GitHub repository is
not a reason to put customer archives or unreviewed sensitive metadata into Git.

## Check before claiming a deployment is ready

```bash
python tools/check_private_assets.py \
  --bank /approved/private/reference_bank \
  --outcome-study /approved/private/outcome_models \
  --plugin-inventory /approved/private/plugin_inventory.csv
```

Use paths belonging to your approved deployment. The tool prints counts and
error codes, not run identifiers or local paths. Exit code `0` means the
supplied integrity/path checks passed; `2` means attention is required. It does
not establish scientific validity, model suitability for a particular command
context, a measured transfer volume or a clinical diagnostic claim.

The check is read-only and does not deserialize models, extract archives,
rewrite registries or rebuild caches. Its separate readiness fields distinguish
historical references, nominal-volume model integrity and optional CSA-metric
assets. The check may read large registered files to calculate SHA-256 hashes.

## What the current runtime actually requires

| Asset | Contract | Missing or changed asset |
| --- | --- | --- |
| Reference bank | `reference_bank_manifest.json` has `validation_status: passed`; every listed artifact has its recorded size and SHA-256. | Unresolved; never silently choose an older bank. |
| Physical context and Er72 caches | Cached parser fingerprints match the packaged parser and trace-quality implementation. | A separately authorized cache rebuild is needed. |
| Original archive store | Every required archive has a manifest record and readable `local_path`; current size/mtime signatures agree with both caches. | A derived-bank copy alone is not sufficient for current historical loaders. |
| Nominal-volume registry | Registered study location exists; analysis-log and validation-seal hashes agree; required sealed tables and registered model payloads match their hashes. | Nominal recognition is withheld; no legacy model fallback. |
| CSA metric references | Frozen feature/ranking tables and compatible plugin-image inventory are available. | The corresponding metric comparison remains unresolved, not normal. |

Archive signatures mirror current runtime cache checks: a recorded archive hash
plus current size and modification time. Raw bytes are **not** rehashed by
default. `--verify-raw-hashes` explicitly performs that more expensive check.
No archive contents are extracted by either mode.

Historical references are outcome-qualified, not certified fault-free. A
successful preflight does not create an eight-run compatible population for
every operation, establish an assay match, authorize extrapolation or certify
an individual inference. Those are separate runtime guards.

## Relocating approved private assets

Do not copy an old absolute-path pointer into a new checkout and assume it is
portable. Current reference pointers require an absolute bank directory and a
pinned hash of that bank's manifest. The nominal-volume registry also names a
study location. A source bank can additionally refer to an existing raw-CSA
store; moving derived tables does not move that store.

If an authorized deployment relocates files, it must:

1. Preserve the original manifests and an auditable record of the relocation.
2. Copy approved derived assets and trusted model payloads without changing
   frozen model bytes, study logs, seals or their existing hash pins.
3. In the **copy only**, update the nominal registry's study location; update its
   size/hash entry in the copied bank manifest and the bank's study-location
   metadata. Recalculate the activation pointer's manifest hash. Do not claim
   the new manifest is byte-identical to the old one.
4. Provide approved raw-archive paths. Path, timestamp or parser changes can
   invalidate context/Er72 caches; the present tool deliberately does not weaken
   those checks or invent a successful rebuild.
5. Set the documented portable CSA-study/plugin-inventory paths, then rerun
   preflight before diagnostic use.

`--volume-study` can audit a proposed destination, but a mismatch with the
registered path remains an error; the tool does not silently activate it.
The preflight is not an export or restore utility.

## Trust and privacy

SHA-256 agreement proves consistency with the supplied records, not that an
unknown supplier is trustworthy. Loading an untrusted pickle/joblib model can
execute code. Use only assets obtained through the approved trusted channel.

Default output suppresses sensitive paths. If needed, `--details-output` writes
an explicit **sensitive local-path** report outside the repository; existing
files are never overwritten. Do not commit that file, customer inventories,
raw archives, identifying report outputs or private model/feature snapshots.

The repository's aggregate benchmark figures and artificial examples are
different from these restricted operational assets.
