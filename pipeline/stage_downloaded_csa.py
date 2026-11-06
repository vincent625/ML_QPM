#!/usr/bin/env python3
"""Copy an existing CSA collection into a private, resumable local snapshot.

The source tree is strictly read-only. Copies are hashed, verified and renamed
atomically. Paths are collision-safe and independent of archive filename length.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil

VERSION = "local-csa-staging-1.0.0"
EXTENSIONS = (".txz", ".tar.xz", ".zip", ".tgz", ".tar.gz")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identify(header):
    if header.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "zip"
    if header.startswith(b"\xfd7zXZ\x00"):
        return "xz_tar"
    if header.startswith(b"\x1f\x8b"):
        return "gzip_tar"
    if header[257:262] == b"ustar":
        return "tar"
    return "unrecognized_format"


def copy_one(record):
    source, dest, receipt = map(Path, [record["source_path"], record["local_path"], record["receipt_path"]])
    result = dict(record, state="failed", sha256="", archive_id="", validation_result="not_validated")
    result.pop("receipt_path")
    try:
        if receipt.exists() and dest.exists():
            old = json.loads(receipt.read_text())
            if old["size_bytes"] == source.stat().st_size and old["source_mtime_ns"] == source.stat().st_mtime_ns:
                if dest.stat().st_size == old["size_bytes"] and digest(dest) == old["sha256"]:
                    return dict(old, state="already_present")
        if dest.exists():
            raise FileExistsError("Existing finalized copy has no matching verified receipt")
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        partial = dest.with_name(dest.name + ".partial")
        # Resume only after verifying the entire existing prefix against source.
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > record["size_bytes"]:
            raise ValueError("Partial copy exceeds source size")
        h = hashlib.sha256()
        with source.open("rb") as origin:
            header = origin.read(512)
            origin.seek(0)
            if prefix:
                with partial.open("rb") as previous:
                    remaining = prefix
                    while remaining:
                        old_block = previous.read(min(8 * 1024 * 1024, remaining))
                        src_block = origin.read(len(old_block))
                        if not old_block or old_block != src_block:
                            raise ValueError("Partial prefix does not match source; preserved for inspection")
                        h.update(src_block)
                        remaining -= len(src_block)
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "ab") as sink:
                for block in iter(lambda: origin.read(8 * 1024 * 1024), b""):
                    sink.write(block)
                    h.update(block)
                sink.flush()
                os.fsync(sink.fileno())
        current = source.stat()
        if current.st_size != record["size_bytes"] or current.st_mtime_ns != record["source_mtime_ns"]:
            raise ValueError("Source changed during copy")
        checksum = h.hexdigest()
        if partial.stat().st_size != record["size_bytes"] or digest(partial) != checksum:
            raise ValueError("Local copy failed SHA-256 verification")
        partial.rename(dest)
        fmt = identify(header)
        result.update(state="transferred" if fmt != "unrecognized_format" else fmt,
                      sha256=checksum, archive_id=checksum[:20], container_format=fmt,
                      validation_result="sha256_and_signature_verified_content_parse_pending", copied_utc=utc_now())
        receipt.write_text(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        return result


def stage(source, output, workers=8):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and destination must be separate trees")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipts = output / ".stage_receipts"
    receipts.mkdir(exist_ok=True, mode=0o700)
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.name.lower().endswith(EXTENSIONS))
    rows = []
    for path in files:
        if not path.resolve().is_relative_to(source):
            raise ValueError("Archive symlink escapes the authorized source collection")
        stat = path.stat()
        relative = path.relative_to(source)
        key = hashlib.sha256(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
        rows.append(dict(source_path=str(path), source_relative_path=str(relative), source_instrument_folder=relative.parts[0],
                         filename=path.name, size_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                         source_mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                         local_path=str(output / "archives" / relative.parts[0] / f"{key}.CSA.txz"),
                         receipt_path=str(receipts / f"{key}.json")))
    required = sum(r["size_bytes"] for r in rows if not Path(r["local_path"]).exists())
    free = shutil.disk_usage(output).free
    if free < required * 1.1:
        raise OSError("Free space is below required copy size plus 10% safety margin")
    (output / "source_discovery.json").write_text(json.dumps(dict(version=VERSION, source=str(source), generated_utc=utc_now(),
                                                                archives=len(rows), total_bytes=sum(r["size_bytes"] for r in rows),
                                                                additional_required_bytes=required, free_bytes=free,
                                                                source_read_only=True, source_instrument_folders=len({r["source_instrument_folder"] for r in rows})), indent=2))
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(copy_one, row) for row in rows]
        for number, job in enumerate(as_completed(jobs), 1):
            result = job.result()
            results.append(result)
            if number % 25 == 0 or number == len(jobs):
                print(json.dumps(dict(stage="local_copy", complete=number, total=len(rows),
                                      failed=sum(r["state"] == "failed" for r in results),
                                      verified_bytes=sum(r["size_bytes"] for r in results if r["state"] != "failed"))), flush=True)
    fields = list(dict.fromkeys(k for r in results for k in r))
    with (output / "source_transfer_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["source_relative_path"]))
    (output / "staging_summary.json").write_text(json.dumps(dict(version=VERSION, generated_utc=utc_now(), discovered=len(rows),
                                                                successful=sum(r["state"] in {"transferred", "already_present"} for r in results),
                                                                failed=sum(r["state"] == "failed" for r in results),
                                                                unrecognized=sum(r["state"] == "unrecognized_format" for r in results)), indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stage(args.source, args.output, args.workers)
