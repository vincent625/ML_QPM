"""Command-linked trace exclusions. Raw debug text is never retained.

Any Er72 response on an operation or its retries excludes the whole recorded
operation from waveform scoring and evidence. Unknown status is not normal.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import re
import tarfile
import zipfile

import pandas as pd
import pyarrow.parquet as pq

VERSION = "qpm-trace-quality-er72-1.0.0"
OP_RE = re.compile(r"\bin PIP\.(Aspirate|Dispense) id\s*=\s*(\d+) vol\s*=\s*([-+\d.]+)", re.I)
SEND_RE = re.compile(r"PipSndV1 id:(\d+),pipCmd:(GA|GD|RX)id(\d+)", re.I)
REPLY_RE = re.compile(r"\bret\s*[=:]\s*(GA|GD|RX)id(\d+)([^\s,]*)", re.I)
CHANNEL_RE = re.compile(r"\bid\s*[=:]\s*(\d+)", re.I)
ER72_RE = re.compile(r"er\s*0*72(?!\d)", re.I)
QUALITY_FIELDS = ["qpm_has_er72", "qpm_er72_command_ids", "qpm_er72_source_lines",
                  "qpm_trace_quality_status", "qpm_waveform_eligible", "qpm_exclusion_reason"]


class TraceQualityTracker:
    """Bind replies by pipette AND sent command ID, not the latest operation."""
    def __init__(self):
        self.rows, self.pending, self.commands = [], {}, {}
        self.line_number = 0
        self.er72_responses = 0
        self.unmatched_er72_responses = 0

    def feed(self, line):
        self.line_number += 1
        op = OP_RE.search(line)
        if op:
            channel = int(op[2])
            row = dict(operation_index=len(self.rows) + 1, pipette=f"PIP{channel}",
                       qpm_has_er72=False, qpm_er72_command_ids="", qpm_er72_source_lines="",
                       qpm_trace_quality_status="checked_no_er72", qpm_waveform_eligible=True,
                       qpm_exclusion_reason="")
            self.rows.append(row)
            self.pending[channel] = row
        sent = SEND_RE.search(line)
        if sent and int(sent[1]) in self.pending:
            self.commands[(int(sent[1]), sent[2].upper(), int(sent[3]))] = self.pending[int(sent[1])]
        reply = REPLY_RE.search(line)
        if not reply or not ER72_RE.search(reply[3]):
            return
        self.er72_responses += 1
        channel = CHANNEL_RE.search(line[:reply.start()])
        if channel:
            row = self.commands.get((int(channel[1]), reply[1].upper(), int(reply[2])))
        else:
            options = [v for (pipette, command, number), v in self.commands.items()
                       if (command, number) == (reply[1].upper(), int(reply[2]))]
            row = options[0] if len(options) == 1 else None
        if row is None:
            self.unmatched_er72_responses += 1
            return
        row.update(qpm_has_er72=True, qpm_trace_quality_status="excluded_er72",
                   qpm_waveform_eligible=False, qpm_exclusion_reason="Er72_command_or_retry_response")
        command = f"{reply[1].upper()}:{int(reply[2])}"
        row["qpm_er72_command_ids"] = "|".join(sorted(set(filter(None, row["qpm_er72_command_ids"].split("|") + [command]))))
        row["qpm_er72_source_lines"] = "|".join(filter(None, [row["qpm_er72_source_lines"], str(self.line_number)]))

    def annotations(self):
        if self.unmatched_er72_responses:
            # Unassigned structured errors could otherwise contaminate a pool.
            for row in self.rows:
                if not row["qpm_has_er72"]:
                    row.update(qpm_trace_quality_status="unresolved_er72_association", qpm_waveform_eligible=False,
                               qpm_exclusion_reason="unmatched_Er72_response_in_archive")
        return self.rows


def parse_trace_quality(handle):
    tracker = TraceQualityTracker()
    for raw in handle:
        tracker.feed(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
    rows = tracker.annotations()
    return rows, dict(er72_responses=tracker.er72_responses,
                      unmatched_er72_responses=tracker.unmatched_er72_responses,
                      er72_operations=sum(row["qpm_has_er72"] for row in rows))


def er72_mask(frame):
    mask = pd.Series(False, index=frame.index)
    if "qpm_has_er72" in frame:
        mask |= frame.qpm_has_er72.astype(str).str.lower().isin(["true", "1", "1.0"])
    if "qpm_trace_quality_status" in frame:
        # The normal label itself contains "er72"; substring matching would
        # incorrectly reject checked_no_er72 traces.
        mask |= frame.qpm_trace_quality_status.fillna("").astype(str).str.lower().isin(["er72", "excluded_er72"])
    for name in ["qpm_error_code", "qpm_exclusion_reason"]:
        if name in frame:
            values = frame[name].fillna("").astype(str)
            mask |= values.str.contains(ER72_RE) | values.str.fullmatch(r"0*72(?:\.0)?")
    return mask


def eligible_waveforms(frame):
    """Defense in depth for scoring/selection, including injected data frames."""
    allowed = ~er72_mask(frame)
    if "qpm_waveform_eligible" in frame:
        allowed &= frame.qpm_waveform_eligible.astype(str).str.lower().isin(["true", "1", "1.0"])
    return frame.loc[allowed].copy()


def extract_quality(record):
    path = Path(record["local_path"])
    source = dict(archive_id=record["archive_id"], sha256=record["sha256"], status="failed",
                  operations=0, er72_operations=0, er72_responses=0, unmatched_er72_responses=0)
    try:
        rows, stats, found = [], {}, False
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                debug = [info for info in archive.infolist() if info.filename.rstrip("/").endswith("CSA/debug")]
                if len(debug) != 1:
                    raise ValueError("Expected exactly one root CSA/debug")
                with archive.open(debug[0]) as handle:
                    rows, stats = parse_trace_quality(handle)
                    found = True
        else:
            with tarfile.open(path, "r|*") as archive:
                for info in archive:
                    if info.isfile() and info.name.rstrip("/").endswith("CSA/debug"):
                        with archive.extractfile(info) as handle:
                            rows, stats = parse_trace_quality(handle)
                        found = True
                        break
        if not found:
            raise ValueError("Root CSA/debug not found")
        for row in rows:
            row["archive_id"] = record["archive_id"]
        source.update(stats, status="parsed", operations=len(rows))
        return rows, source
    except Exception as exc:
        source.update(error_type=type(exc).__name__)
        return [], source


def ensure_trace_quality_cache(project, workers=8):
    """Incremental cache; input hashes and parser code invalidate stale status."""
    project = Path(project)
    cache_path, manifest_path = project / "qpm_trace_quality.parquet", project / "qpm_trace_quality_manifest.json"
    ids = set(pq.read_table(project / "qpm_operation_waveforms.parquet", columns=["archive_id"], use_threads=False)
              .to_pandas().archive_id.unique())
    transfers = pd.read_csv(project / "csa_transfer_manifest.csv", dtype={"archive_id": str})
    records = transfers[transfers.archive_id.isin(ids)].drop_duplicates("archive_id").to_dict("records")
    def input_signature(record):
        info=Path(record["local_path"]).stat()
        return record["archive_id"],[record["sha256"],info.st_size,info.st_mtime_ns]
    with ThreadPoolExecutor(max_workers=16) as readers:
        signatures=dict(readers.map(input_signature,records))
    parser_hash = hashlib.sha256((VERSION + inspect.getsource(TraceQualityTracker) + inspect.getsource(extract_quality)
                                 + inspect.getsource(parse_trace_quality)
                                 + str([OP_RE.pattern, SEND_RE.pattern, REPLY_RE.pattern, CHANNEL_RE.pattern, ER72_RE.pattern])).encode()).hexdigest()
    previous, table = {}, pd.DataFrame(columns=["archive_id", "operation_index", "pipette", *QUALITY_FIELDS])
    if cache_path.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("parser_sha256") == parser_hash:
            table = pq.read_table(cache_path, use_threads=False).to_pandas()
        else:
            previous = {}
    old_sources = {row["archive_id"]: row for row in previous.get("archives", [])}
    reusable = {aid for aid, signature in signatures.items()
                if previous.get("input_signatures", {}).get(aid) == signature and old_sources.get(aid, {}).get("status") == "parsed"}
    jobs = [record for record in records if record["archive_id"] not in reusable]
    missing = ids - set(signatures)
    if not jobs and not missing and previous.get("input_signatures") == signatures:
        return table, previous
    batches = [table[table.archive_id.isin(reusable)]]
    sources = [old_sources[aid] for aid in reusable]
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(extract_quality, record) for record in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                rows, source = future.result()
                if rows:
                    batches.append(pd.DataFrame(rows))
                sources.append(source)
                if index % 25 == 0 or index == len(jobs):
                    print(f"Er72 trace-status audit: {index}/{len(jobs)} archives", flush=True)
    table = pd.concat(batches, ignore_index=True)
    if table.duplicated(["archive_id", "operation_index", "pipette"]).any():
        raise ValueError("Ambiguous trace-status association; refusing scoring")
    manifest = dict(version=VERSION, parser_sha256=parser_hash, input_signatures=signatures,
                    archives=sorted(sources, key=lambda row: row["archive_id"]),
                    missing_archive_records=sorted(missing), operations=len(table),
                    er72_operations=int(table.qpm_has_er72.fillna(False).sum()),
                    raw_debug_text_stored=False, successful_retry_does_not_clear_exclusion=True)
    table.to_parquet(cache_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return table, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    table, manifest = ensure_trace_quality_cache(args.project, args.workers)
    print(json.dumps({k: manifest[k] for k in ["version", "operations", "er72_operations", "missing_archive_records"]}), flush=True)
