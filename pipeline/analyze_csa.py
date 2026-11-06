#!/usr/bin/env python3
"""Parse Genexus CSA archives and run leakage-guarded small-data analyses.

The pipeline is deliberately conservative: only explicit pre-run commands or
settings enter Tier A, context enters Tier B, measured/process values enter
Tier C, and outcomes/identifiers/sensitive fields enter Tier D.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import shlex
import sys
import tarfile
import tempfile
import time
import traceback
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260925
POLICY_VERSION = "csa-feature-policy-1.1"
TARGETS = ("total_reads", "mean_read_length_bp", "raw_read_accuracy_pct")
SECONDARY_TARGET = "aq20_mean_read_length_bp"
ID_COLUMNS = (
    "observation_id", "physical_run_id", "instrument_id", "chip_lane",
    "run_start_time", "analysis_id", "archive_id", "archive_sha256",
)
PALETTE = {
    "ocean": "#0072B2", "forest": "#009E73", "amber": "#E69F00",
    "clay": "#D55E00", "sky": "#56B4E9", "orchid": "#CC79A7",
    "ink": "#243238", "paper": "#F5F3EA", "mist": "#DFE8E2",
    "grey": "#7A8487", "white": "#FFFFFF",
}

SENSITIVE_RE = re.compile(
    r"(patient|subject|mrn|operator|username|password|passwd|credential|token|"
    r"secret|api[_-]?key|network|ip[_-]?address|email|phone|"
    r"sample(?![_ .-]?(?:count|composition|type)))",
    re.I,
)
IDENTIFIER_RE = re.compile(
    r"(^|[._])(path|filename|file_name|uuid|guid|hash|serial|barcode|id|"
    r"timestamp|starttime|endtime|date|time|runid)($|[._])",
    re.I,
)
TARGET_RE = re.compile(
    r"(final[_ .-]?library[_ .-]?reads|final[_ .-]?reads|total[_ .-]?reads|"
    r"mean[_ .-]?read[_ .-]?length|raw[_ .-]?read[_ .-]?accuracy|"
    r"q20.*mean[_ .-]?read[_ .-]?length|read[_ .-]?length[_ .-]?hist|"
    r"quality[_ .-]?hist|success[_ .-]?label|qc[_ .-]?pass)",
    re.I,
)
ACTION_RE = re.compile(
    r"(setpoint|target|threshold|tolerance|limit|minimum|maximum|(?:^|[._])min(?:$|[._])|"
    r"(?:^|[._])max(?:$|[._])|(?:^|[._])flows?(?:$|[._])|"
    r"flow[_ .-]?order|num(?:ber)?[_ .-]?flows?|"
    r"template[_ .-]?size|basecaller.*arg|prebasecaller.*arg|beadfind.*arg|"
    r"temperature|temp[_ .-]?(?:set|target)|volume|concentration|ratio|"
    r"cycles?|duration|hold|delay|ramp|speed|height|air[_ .-]?gap|mix|"
    r"blowout|aspirat|dispens|blocked[_ .-]?tip|pressure[_ .-]?(?:set|target|"
    r"min|max|tol|limit|threshold)|wash|incubat|frequency|amplitude)",
    re.I,
)
CONTEXT_RE = re.compile(
    r"(instrument|lane|software|version|chip[_ .-]?(type|lot)|reagent|"
    r"consumable|lot|assay|panel|sample[_ .-]?count|library[_ .-]?prep|"
    r"workflow|maintenance|clean|run[_ .-]?date|application|kit|chemistry)",
    re.I,
)
DIAGNOSTIC_RE = re.compile(
    r"(loading|key[_ .-]?signal|snr|bead|filter|phasing|measured|actual|"
    r"calibrat|retr(?:y|ies)|error|warning|elapsed|process[_ .-]?time|"
    r"pressure|vacuum|pipette|temperature|sensor|signal|noise|count)",
    re.I,
)
CONFIG_SOURCE_RE = re.compile(
    r"(planned_run\.json|ion_params_00\.json|\.ini$|liquidclass.*\.json|"
    r"containergeometry.*\.json|workflow.*(?:\.json|\.csv|\.ini|\.txt))",
    re.I,
)
PRESSURE_FILE_RE = re.compile(
    r"(vacuum_log\.csv|Vacuum_PressureDataLANE.*\.csv|"
    r"pipPressure_lane.*\.csv|pipInCoupler.*\.csv|blockedTipPressureCurve.*\.csv)",
    re.I,
)
PIPETTE_FILE_RE = re.compile(
    r"(pipPressure_lane|pipInCoupler|blockedTipPressureCurve|bottomFindEvents|"
    r"tubeVolHistory|TubeBottomLog|Vacuum_PressureDataLANE.*PIP)",
    re.I,
)
EVENT_FILE_RE = re.compile(
    r"(libPrep_log|ScriptStatus|wflog|Events\.log|bottomFindEvents|"
    r"tubeVolHistory|TubeBottomLog)",
    re.I,
)
INTERESTING_CSV_RE = re.compile(
    r"(info\.csv|pressure|vacuum|pip|liquid|workflow|scriptstatus|"
    r"tube|bottom|container|config)",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: str, prefix: str = "") -> str:
    return prefix + hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


def normalize_name(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", value)[:180]


def parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip().replace(",", "")
    # Do not turn command lines, dates, identifiers, or arbitrary labels into
    # numbers merely because they contain a digit.  A scalar may carry a
    # documented unit, but otherwise the whole string must be numeric.
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*(?:%|psi|kpa|mbar|bar|pa|mmhg|°?c|ul|µl|ml|nl|"
        r"seconds?|secs?|ms|minutes?|mins?|hours?|bp)?\s*",
        text,
        re.I,
    )
    if not match:
        return None
    try:
        result = float(match.group(1))
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def infer_unit(name: str, value: Any = None) -> str:
    text = f"{name} {value if isinstance(value, str) else ''}".lower()
    bracket = re.search(r"[\[(]\s*(psi|kpa|mbar|bar|pa|mmhg|c|°c|ul|µl|ml|nl|"
                        r"seconds?|secs?|ms|minutes?|mins?|hours?|bp|percent|%)\s*[\])]", text)
    if bracket:
        return bracket.group(1).replace("°", "")
    checks = [
        (r"(?:^|[_ .])psi(?:$|[_ .])", "psi"),
        (r"(?:^|[_ .])kpa(?:$|[_ .])", "kPa"),
        (r"(?:^|[_ .])mbar(?:$|[_ .])", "mbar"),
        (r"(?:^|[_ .])mmhg(?:$|[_ .])", "mmHg"),
        (r"(temperature|temp).*c(?:$|[_ .])", "C"),
        (r"(duration|hold|delay|time).*ms(?:$|[_ .])", "ms"),
        (r"(duration|hold|delay).*min(?:$|[_ .])", "min"),
        (r"(duration|hold|delay|time).*(?:_s|seconds?|secs?)(?:$|[_ .])", "s"),
        (r"(volume|vol).*(?:ul|µl)(?:$|[_ .])", "uL"),
        (r"(volume|vol).*ml(?:$|[_ .])", "mL"),
        (r"(accuracy|percent|pct|rate)(?:$|[_ .])", "%"),
    ]
    for pattern, unit in checks:
        if re.search(pattern, text, re.I):
            return unit
    return ""


def flatten_json(value: Any, prefix: str = "", max_list: int = 64) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(child, child_prefix, max_list=max_list)
    elif isinstance(value, list):
        if len(value) <= max_list and all(not isinstance(x, (dict, list)) for x in value):
            if value and all(parse_number(x) is not None for x in value):
                arr = np.asarray([parse_number(x) for x in value], dtype=float)
                for metric, val in (
                    ("count", len(arr)), ("mean", np.mean(arr)), ("std", np.std(arr)),
                    ("min", np.min(arr)), ("q05", np.quantile(arr, .05)),
                    ("median", np.median(arr)), ("q95", np.quantile(arr, .95)),
                    ("max", np.max(arr)), ("iqr", np.quantile(arr, .75) - np.quantile(arr, .25)),
                ):
                    yield f"{prefix}.__array_{metric}", float(val)
            else:
                yield prefix, "|".join(map(str, value))
        else:
            yield f"{prefix}.__array_length", len(value)
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield prefix, value


def source_basename(name: str) -> str:
    return Path(name).name


def is_interesting_member(name: str) -> bool:
    lower = name.lower()
    base = source_basename(name)
    if lower.endswith(".json"):
        primary_output = bool(re.search(
            r"/outputs/basecallingactor-00/(basecaller|ionstats_basecaller)\.json$",
            lower.replace("\\", "/"),
        ))
        return (
            primary_output
            or bool(CONFIG_SOURCE_RE.search(base))
            or bool(re.search(r"(bead|calibrat|qc|summary|metrics)", base, re.I))
        )
    return (
        lower.endswith(".ini")
        or bool(INTERESTING_CSV_RE.search(base))
        or bool(EVENT_FILE_RE.search(base))
    )


def is_nested_archive(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".txz", ".tar.xz", ".zip", ".tgz", ".tar.gz"))


def should_parse_ini(source: str) -> bool:
    normalized = source.split("!")[-1].replace("\\", "/").lower()
    base = source_basename(normalized).lower()
    if "_libprep" in normalized or "/tertiary/" in normalized:
        return False
    if normalized.count("/") <= 1 and base == "analysis.ini":
        return True
    return (
        "/secondary/" in normalized
        and base in {
            "basecallingactor.ini", "sigprocactor.ini", "global.ini",
            "beadfindactor.ini", "inputcontroller.ini",
        }
    )


@dataclass
class Member:
    name: str
    size: int
    data: bytes


class ArchiveView:
    def __init__(self, path: Path):
        self.path = path
        self.kind = ""
        self.handle: Any = None

    def __enter__(self) -> "ArchiveView":
        with self.path.open("rb") as signature_handle:
            signature = signature_handle.read(8)
        if signature.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            self.kind = "zip"
            self.handle = zipfile.ZipFile(self.path)
        elif signature.startswith(b"\xfd7zXZ\x00") or signature.startswith(b"\x1f\x8b"):
            self.kind = "tar"
            self.handle = tarfile.open(self.path, mode="r:*")
        else:
            raise ValueError("unrecognized archive signature")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()

    def entries(self) -> list[tuple[str, int, Any]]:
        if self.kind == "zip":
            return [(x.filename, x.file_size, x) for x in self.handle.infolist() if not x.is_dir()]
        return [(x.name, x.size, x) for x in self.handle.getmembers() if x.isfile()]

    def read_entry(self, entry: Any, max_bytes: int = 512 * 1024 * 1024) -> bytes:
        size = entry.file_size if self.kind == "zip" else entry.size
        if size > max_bytes:
            raise ValueError(f"member exceeds {max_bytes} byte safety limit")
        if self.kind == "zip":
            return self.handle.read(entry)
        handle = self.handle.extractfile(entry)
        if handle is None:
            return b""
        with handle:
            return handle.read(max_bytes + 1)

    def copy_entry(self, entry: Any, destination: Path) -> None:
        if self.kind == "zip":
            source = self.handle.open(entry)
        else:
            source = self.handle.extractfile(entry)
        if source is None:
            return
        with source, destination.open("wb") as out:
            while chunk := source.read(8 * 1024 * 1024):
                out.write(chunk)


def iter_archive_members(path: Path, warnings_out: list[str], max_depth: int = 2) -> Iterator[Member]:
    with tempfile.TemporaryDirectory(prefix="csa_nested_") as temp_dir:
        temp_root = Path(temp_dir)

        def walk(current: Path, prefix: str, depth: int) -> Iterator[Member]:
            try:
                with ArchiveView(current) as archive:
                    for name, size, entry in archive.entries():
                        qualified = f"{prefix}!{name}" if prefix else name
                        if is_nested_archive(name) and depth < max_depth:
                            nested_path = temp_root / f"{stable_hash(qualified)}_{Path(name).name}"
                            try:
                                archive.copy_entry(entry, nested_path)
                                yield from walk(nested_path, qualified, depth + 1)
                            except Exception as exc:
                                warnings_out.append(f"nested archive {qualified}: {exc}")
                        if is_interesting_member(name):
                            try:
                                yield Member(qualified, size, archive.read_entry(entry))
                            except Exception as exc:
                                warnings_out.append(f"member {qualified}: {exc}")
            except Exception as exc:
                warnings_out.append(f"archive {current.name}: {exc}")

        yield from walk(path, "", 0)


def parse_cli_arguments(value: str) -> list[tuple[str, Any]]:
    if not isinstance(value, str) or "--" not in value:
        return []
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = value.split()
    result: list[tuple[str, Any]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if "=" in key:
                key, val = key.split("=", 1)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                i += 1
                val = tokens[i]
            else:
                val = True
            result.append((key, val))
        i += 1
    return result


def classify_feature(source: str, field: str, value: Any, measured: bool = False) -> tuple[str, str, str]:
    key = f"{source}.{field}"
    if SENSITIVE_RE.search(key):
        return "D", "unavailable", "sensitive_or_operator_field"
    if TARGET_RE.search(key):
        return "D", "post_run", "outcome_or_direct_target_derivative"
    if IDENTIFIER_RE.search(field) or field.lower().endswith(("_id", ".id")):
        return "D", "unavailable", "unique_identifier_path_or_time"
    if measured:
        return "C", "during_or_post_run", "measured_process_diagnostic"
    if (
        source_basename(source).lower() == "ion_params_00.json"
        and re.search(r"^exp_json\.log(?:\.|$)", field, re.I)
    ):
        return "C", "during_or_post_run", "embedded_acquisition_log_not_pre_run_configuration"
    if CONFIG_SOURCE_RE.search(source_basename(source)):
        if ACTION_RE.search(field):
            return "A", "pre_run", ""
        if CONTEXT_RE.search(field):
            return "B", "pre_run", "context_not_tunable"
        return "B", "pre_run", "configuration_metadata_not_explicitly_actionable"
    if DIAGNOSTIC_RE.search(key):
        return "C", "during_or_post_run", "during_or_post_run_diagnostic"
    return "C", "during_or_post_run", "availability_not_proven_pre_run"


def feature_record(
    observation_id: str,
    instrument_id: str,
    source: str,
    field: str,
    value: Any,
    measured: bool = False,
    force_name: str | None = None,
    force_tier: str | None = None,
    force_reason: str = "",
) -> dict[str, Any]:
    tier, availability, reason = classify_feature(source, field, value, measured=measured)
    if force_tier:
        tier = force_tier
        availability = {"A": "pre_run", "B": "pre_run", "C": "during_or_post_run", "D": "unavailable"}[tier]
        reason = force_reason
    source_key = normalize_name(Path(source.split("!")[-1]).stem)
    name = force_name or f"{source_key}.{normalize_name(field)}"
    numeric = parse_number(value)
    if tier == "D" and (SENSITIVE_RE.search(f"{source}.{field}") or IDENTIFIER_RE.search(field)):
        stored_value: Any = None
        numeric = None
    elif re.search(r"instrument.*(?:name|serial)", field, re.I):
        stored_value = stable_hash(str(value), "instrument_")
        numeric = None
    else:
        stored_value = value if not isinstance(value, (dict, list)) else json.dumps(value, sort_keys=True)
    return {
        "observation_id": observation_id,
        "instrument_id": instrument_id,
        "feature_name": name,
        "value": stored_value,
        "numeric_value": numeric,
        "unit": infer_unit(field, value),
        "data_type": "numeric" if numeric is not None else "categorical",
        "source_file": source,
        "source_field": field,
        "availability_time": availability,
        "actionability_tier": tier,
        "exclusion_reason": reason,
    }


def parse_info_csv(data: bytes, source: str) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    found: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        cleaned = [str(x).strip() for x in row]
        for i, cell in enumerate(cleaned):
            lower = cell.lower()
            outcome = None
            if re.search(r"\bfinal\s*reads?\b", lower):
                outcome = "total_reads"
            elif re.search(r"\braw\s*read\s*accuracy\b", lower):
                outcome = "raw_read_accuracy_pct"
            elif re.search(r"\bmean\s*(raw\s*)?read\s*length\b", lower) and "q20" not in lower:
                outcome = "mean_read_length_bp"
            if outcome:
                candidates = cleaned[i + 1:] + cleaned[:i]
                value = next((parse_number(x) for x in candidates if parse_number(x) is not None), None)
                if value is not None:
                    found.append({
                        "outcome": outcome, "value": value, "source_file": source,
                        "source_field": f"row_{row_number}:{cell}", "priority": 1,
                    })
    return found


def extract_outcomes_from_json(obj: Any, source: str) -> list[dict[str, Any]]:
    flat = dict(flatten_json(obj))
    result: list[dict[str, Any]] = []
    lower_source = source.split("!")[-1].replace("\\", "/").lower()
    run_basecaller = lower_source.endswith("/outputs/basecallingactor-00/basecaller.json")
    run_ionstats = lower_source.endswith("/outputs/basecallingactor-00/ionstats_basecaller.json")
    for field, value in flat.items():
        norm = field.lower()
        outcome = None
        priority = 5
        if (
            run_basecaller
            and norm.endswith("filtering.libraryreport.final_library_reads")
        ):
            outcome, priority = "total_reads", 0
        elif run_ionstats and norm == "full.mean_read_length":
            outcome, priority = "mean_read_length_bp", 0
        elif run_ionstats and norm == "q20.mean_read_length":
            outcome, priority = SECONDARY_TARGET, 0
        elif re.search(r"final[_ .]?library[_ .]?reads$", norm):
            outcome, priority = "total_reads", 2
        elif norm.endswith("raw_read_accuracy"):
            outcome, priority = "raw_read_accuracy_pct", 2
        if outcome and parse_number(value) is not None:
            result.append({
                "outcome": outcome, "value": parse_number(value),
                "source_file": source, "source_field": field, "priority": priority,
            })
    return result


def parse_vacuum_log(
    data: bytes,
    source: str,
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig", "replace")
    rows: list[dict[str, Any]] = []
    configured = {
        "targetpressure", "waitpressure", "vacuumtime", "maxchargetime",
        "bleedtime", "ventatend",
    }
    measured = {
        "timetoinit", "initalpressure", "initialpressure", "timetotarget",
        "meanpressure", "amplitudeofoscillation", "pumpdutycycle",
    }
    for sequence, raw in enumerate(csv.reader(io.StringIO(text)), 1):
        pairs = {}
        for index in range(0, len(raw) - 1, 2):
            pairs[normalize_name(raw[index])] = raw[index + 1]
        if not pairs:
            continue
        step_raw = pairs.get("comment", f"vacuum_step_{sequence}")
        lane_value = pairs.get("lane", observation.get("chip_lane", ""))
        lane = f"ChipLane{lane_value}" if str(lane_value).isdigit() else str(lane_value)
        timestamp = parse_datetime_like(pairs.get("ts"))
        for key, value in pairs.items():
            compact = key.replace("_", "")
            if compact not in configured | measured:
                continue
            is_setting = compact in configured
            field_name = {
                "targetpressure": "vacuum_pressure_setpoint",
                "waitpressure": "vacuum_wait_pressure_setpoint",
                "vacuumtime": "vacuum_hold_duration",
                "maxchargetime": "vacuum_max_charge_duration",
                "bleedtime": "vacuum_bleed_duration",
                "ventatend": "vacuum_vent_at_end",
                "timetoinit": "time_to_initialize",
                "initalpressure": "initial_pressure",
                "initialpressure": "initial_pressure",
                "timetotarget": "time_to_reach_setpoint",
                "meanpressure": "mean_pressure",
                "amplitudeofoscillation": "pressure_oscillation_amplitude",
                "pumpdutycycle": "pump_duty_cycle",
            }[compact]
            rows.append({
                **{k: observation.get(k, "") for k in ("observation_id", "physical_run_id", "instrument_id")},
                "chip_lane": lane, "workflow_name": "templating",
                "step_number": sequence, "step_name_raw": step_raw,
                "step_name_normalized": normalize_name(step_raw),
                "pipette": "", "attempt": "", "operation": "vacuum",
                "source": "", "destination": "", "liquid_class": "",
                "programmed_volume": "",
                "feature_name": f"templating.{normalize_name(step_raw)}.{field_name}",
                "value": parse_number(value) if parse_number(value) is not None else value,
                "unit": infer_unit(key), "setpoint_or_measured": "setpoint" if is_setting else "measured",
                "feature_tier": "A" if is_setting else "C",
                "classification": "explicit_commanded_setpoint" if is_setting else "diagnostic_only",
                "source_file": source, "source_field": key,
                "start_time": timestamp.isoformat() if timestamp is not None else "",
                "end_time": timestamp.isoformat() if timestamp is not None else "",
                "parse_confidence": "high",
                "exclusion_reason": "" if is_setting else "measured_pressure_is_not_a_pre_run_setting",
            })
    return rows


def parse_datetime_like(value: Any) -> pd.Timestamp | None:
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if isinstance(parsed, pd.DatetimeIndex):
            parsed = parsed[0] if len(parsed) else pd.NaT
        if pd.isna(parsed):
            return None
        return parsed
    except Exception:
        return None


def event_rows(data: bytes, source: str) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig", "replace")
    result: list[dict[str, Any]] = []
    timestamp_re = re.compile(
        r"(20\d{2}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    )
    for line in text.splitlines():
        if not re.search(r"(step|aspirat|dispens|vacuum|pressure|pipette|workflow|transfer)", line, re.I):
            continue
        match = timestamp_re.search(line)
        if not match:
            continue
        stamp = parse_datetime_like(match.group(1).replace(",", "."))
        if stamp is not None:
            result.append({
                "time": stamp, "raw": line[:500], "normalized": normalize_name(line[:200]),
                "source": source,
            })
    return result


def dataframe_from_bytes(data: bytes) -> pd.DataFrame:
    for kwargs in (
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": "\t"},
        {"sep": ";"},
    ):
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding_errors="replace", on_bad_lines="skip", **kwargs)
            if frame.shape[1] > 0:
                return frame
        except Exception:
            pass
    return pd.DataFrame()


def trace_context(frame: pd.DataFrame, source: str) -> dict[str, Any]:
    base = source_basename(source)
    lane_match = re.search(r"lane[_ -]?(\d+)", base, re.I)
    pip_match = re.search(r"(PIP\d+)", base, re.I)
    attempt_match = re.search(r"(?:try|attempt)[_ -]?(\d+)", base, re.I)
    operation = (
        "aspiration" if re.search(r"asp|incoupler(?!dsp)", base, re.I)
        else "dispense" if re.search(r"dsp|disp", base, re.I)
        else ""
    )
    constants: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if 0 < series.nunique() <= 3:
            constants[normalize_name(str(column))] = str(series.iloc[0])[:200]
    operation = constants.get("operation", operation)
    return {
        "chip_lane": f"ChipLane{lane_match.group(1)}" if lane_match else "",
        "pipette": pip_match.group(1).upper() if pip_match else constants.get("pipette", ""),
        "attempt": int(attempt_match.group(1)) if attempt_match else parse_number(constants.get("attempt")),
        "operation": operation,
        "source": constants.get("source", constants.get("source_container", "")),
        "destination": constants.get("destination", constants.get("destination_container", "")),
        "liquid_class": constants.get("liquid_class", constants.get("reagent", "")),
        "programmed_volume": parse_number(constants.get("volume", constants.get("programmed_volume"))),
        "workflow_name": constants.get("workflow", constants.get("workflow_name", "")),
        "step_name_raw": constants.get("step", constants.get("step_name", Path(base).stem)),
        "step_number": parse_number(constants.get("step_number", constants.get("step"))),
    }


def identify_time_axis(frame: pd.DataFrame) -> tuple[np.ndarray, str, str, str]:
    for column in frame.columns:
        if re.search(r"(timestamp|date[_ ]?time|^time$)", str(column), re.I):
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if parsed.notna().sum() >= max(3, int(len(frame) * .5)):
                origin = parsed.dropna().iloc[0]
                seconds = (parsed - origin).dt.total_seconds().to_numpy(dtype=float)
                return seconds, str(origin.isoformat()), str(parsed.dropna().iloc[-1].isoformat()), str(column)
    for column in frame.columns:
        if re.search(r"(elapsed|seconds|time)", str(column), re.I):
            numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(numeric).sum() >= 3:
                return numeric, "", "", str(column)
    return np.arange(len(frame), dtype=float), "", "", "row_index"


def robust_trace_metrics(values: np.ndarray, time_axis: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values) & np.isfinite(time_axis)
    y, x = values[mask], time_axis[mask]
    if len(y) < 3:
        return {}
    order = np.argsort(x)
    x, y = x[order], y[order]
    q05, q25, med, q75, q95 = np.quantile(y, [.05, .25, .5, .75, .95])
    centered_x = x - x[0]
    slope = float(np.polyfit(centered_x, y, 1)[0]) if np.ptp(centered_x) > 0 else 0.0
    diffs = np.diff(y)
    baseline_count = max(3, len(y) // 10)
    baseline = float(np.median(y[:baseline_count]))
    mad = float(np.median(np.abs(y - med)))
    peak_abs_index = int(np.argmax(np.abs(y - baseline)))
    peak_index = int(np.argmax(y))
    trough_index = int(np.argmin(y))
    integral = float(np.trapezoid(y, centered_x)) if np.ptp(centered_x) > 0 else float(np.sum(y))
    tolerance = max(3 * mad, abs(baseline) * .05, np.finfo(float).eps)
    recovery = np.nan
    after = np.where(np.abs(y[peak_abs_index:] - baseline) <= tolerance)[0]
    if len(after):
        recovery = float(centered_x[peak_abs_index + after[0]] - centered_x[peak_abs_index])
    return {
        "count": float(len(y)), "baseline": baseline, "mean": float(np.mean(y)),
        "median": float(med), "minimum": float(np.min(y)), "maximum": float(np.max(y)),
        "std": float(np.std(y, ddof=1)), "q05": float(q05), "q95": float(q95),
        "iqr": float(q75 - q25), "start": float(y[0]), "end": float(y[-1]),
        "slope": slope, "integral": integral,
        "largest_drop": float(np.min(diffs)) if len(diffs) else 0.0,
        "largest_rise": float(np.max(diffs)) if len(diffs) else 0.0,
        "peak_negative": float(y[trough_index]), "peak_positive": float(y[peak_index]),
        "time_to_peak_s": float(centered_x[peak_abs_index]),
        "recovery_time_s": recovery,
        "noise_mad": mad,
        "oscillation": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "sudden_pressure_drops": float(np.sum(diffs < -max(5 * mad, np.std(y) * 2))),
        "suspected_air_intake": float(np.min(diffs) < -max(8 * mad, np.std(y) * 3)),
        "suspected_clog": float((q75 - q25) > max(10 * mad, abs(med) * .5)),
    }


def pressure_trace_rows(
    frame: pd.DataFrame,
    source: str,
    observation: dict[str, Any],
    kind: str,
) -> list[dict[str, Any]]:
    if frame.empty or len(frame) < 3:
        return []
    context = trace_context(frame, source)
    x, start_time, end_time, time_column = identify_time_axis(frame)
    numeric_columns: list[tuple[str, np.ndarray]] = []
    for column in frame.columns:
        if str(column) == time_column:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(numeric).sum() >= max(3, int(len(frame) * .5)):
            numeric_columns.append((str(column), numeric))
    pressure_columns = [
        (c, a) for c, a in numeric_columns
        if re.search(r"(pressure|vacuum|sensor|psi|kpa|mbar)", c, re.I)
    ] or numeric_columns
    setpoint_columns = {
        c: a for c, a in pressure_columns
        if re.search(r"(setpoint|target|command)", c, re.I)
    }
    measured_columns = [
        (c, a) for c, a in pressure_columns
        if c not in setpoint_columns
    ]
    rows: list[dict[str, Any]] = []
    for column, values in measured_columns:
        metrics = robust_trace_metrics(values, x)
        unit = infer_unit(column)
        for metric, value in metrics.items():
            rows.append({
                **{k: observation.get(k, "") for k in ("observation_id", "physical_run_id", "instrument_id")},
                "chip_lane": context["chip_lane"] or observation.get("chip_lane", ""),
                "workflow_name": context["workflow_name"],
                "step_number": context["step_number"],
                "step_name_raw": context["step_name_raw"],
                "step_name_normalized": normalize_name(context["step_name_raw"]),
                "pipette": context["pipette"], "attempt": context["attempt"],
                "operation": context["operation"], "source": context["source"],
                "destination": context["destination"], "liquid_class": context["liquid_class"],
                "programmed_volume": context["programmed_volume"],
                "feature_name": f"{kind}.{normalize_name(context['step_name_raw'])}.{normalize_name(column)}.{metric}",
                "value": value, "unit": "s" if metric.endswith("_s") else unit,
                "setpoint_or_measured": "measured", "feature_tier": "C",
                "classification": "diagnostic_only", "source_file": source,
                "source_field": f"{column}:{metric}", "start_time": start_time,
                "end_time": end_time, "parse_confidence": "medium",
                "exclusion_reason": "measured_pressure_is_not_a_pre_run_setting",
            })
    for column, values in setpoint_columns.items():
        unique = pd.Series(values[np.isfinite(values)]).drop_duplicates()
        if 0 < len(unique) <= 10:
            for value in unique:
                rows.append({
                    **{k: observation.get(k, "") for k in ("observation_id", "physical_run_id", "instrument_id")},
                    "chip_lane": context["chip_lane"] or observation.get("chip_lane", ""),
                    "workflow_name": context["workflow_name"], "step_number": context["step_number"],
                    "step_name_raw": context["step_name_raw"],
                    "step_name_normalized": normalize_name(context["step_name_raw"]),
                    "pipette": context["pipette"], "attempt": context["attempt"],
                    "operation": context["operation"], "source": context["source"],
                    "destination": context["destination"], "liquid_class": context["liquid_class"],
                    "programmed_volume": context["programmed_volume"],
                    "feature_name": f"{kind}.{normalize_name(context['step_name_raw'])}.{normalize_name(column)}",
                    "value": float(value), "unit": infer_unit(column),
                    "setpoint_or_measured": "setpoint", "feature_tier": "A",
                    "classification": "explicit_commanded_setpoint",
                    "source_file": source, "source_field": column,
                    "start_time": start_time, "end_time": end_time,
                    "parse_confidence": "high", "exclusion_reason": "",
                })
    if setpoint_columns and measured_columns:
        set_col, set_values = next(iter(setpoint_columns.items()))
        meas_col, meas_values = measured_columns[0]
        mask = np.isfinite(set_values) & np.isfinite(meas_values)
        if mask.sum() >= 3:
            error = meas_values[mask] - set_values[mask]
            for metric, value in {
                "actual_minus_setpoint_mean": float(np.mean(error)),
                "actual_minus_setpoint_mae": float(np.mean(np.abs(error))),
                "time_outside_5pct_tolerance_fraction": float(
                    np.mean(np.abs(error) > np.maximum(np.abs(set_values[mask]) * .05, np.finfo(float).eps))
                ),
            }.items():
                rows.append({
                    **{k: observation.get(k, "") for k in ("observation_id", "physical_run_id", "instrument_id")},
                    "chip_lane": context["chip_lane"] or observation.get("chip_lane", ""),
                    "workflow_name": context["workflow_name"], "step_number": context["step_number"],
                    "step_name_raw": context["step_name_raw"],
                    "step_name_normalized": normalize_name(context["step_name_raw"]),
                    "pipette": context["pipette"], "attempt": context["attempt"],
                    "operation": context["operation"], "source": context["source"],
                    "destination": context["destination"], "liquid_class": context["liquid_class"],
                    "programmed_volume": context["programmed_volume"],
                    "feature_name": f"{kind}.{normalize_name(context['step_name_raw'])}.{metric}",
                    "value": value, "unit": infer_unit(meas_col),
                    "setpoint_or_measured": "measured", "feature_tier": "C",
                    "classification": "diagnostic_only", "source_file": source,
                    "source_field": f"{meas_col}-minus-{set_col}", "start_time": start_time,
                    "end_time": end_time, "parse_confidence": "high",
                    "exclusion_reason": "measured_setpoint_error_is_a_diagnostic",
                })
    return rows


def align_pressure_steps(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    if not events:
        return
    events = sorted(events, key=lambda x: x["time"])
    times = np.array([x["time"].value for x in events], dtype=np.int64)
    for row in rows:
        start = parse_datetime_like(row.get("start_time"))
        if start is None:
            continue
        index = int(np.argmin(np.abs(times - start.value)))
        nearest = events[index]
        row["step_name_raw"] = nearest["raw"]
        row["step_name_normalized"] = nearest["normalized"]
        row["parse_confidence"] = "high" if abs((nearest["time"] - start).total_seconds()) <= 300 else "medium"


def settings_pressure_rows(
    features: list[dict[str, Any]],
    observation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vacuum: list[dict[str, Any]] = []
    pipette: list[dict[str, Any]] = []
    for feature in features:
        key = f"{feature['source_file']}.{feature['source_field']}"
        if not re.search(r"(pressure|vacuum|aspirat|dispens|blocked.tip)", key, re.I):
            continue
        explicit = bool(re.search(r"(setpoint|target|threshold|tolerance|limit|min|max|ramp|hold)", key, re.I))
        tier = "A" if explicit and feature["actionability_tier"] == "A" else "D"
        classification = "explicit_commanded_setpoint" if tier == "A" else "unresolved_pressure_parameter"
        reason = "" if tier == "A" else "commanded_vs_measured_status_not_resolved"
        target = pipette if re.search(r"(pip|aspirat|dispens|blocked.tip|liquidclass)", key, re.I) else vacuum
        step = normalize_name(feature["source_field"].rsplit(".", 1)[0])
        target.append({
            **{k: observation.get(k, "") for k in ("observation_id", "physical_run_id", "instrument_id", "chip_lane")},
            "workflow_name": "", "step_number": "", "step_name_raw": feature["source_field"],
            "step_name_normalized": step, "pipette": "", "attempt": "",
            "operation": "", "source": "", "destination": "", "liquid_class": "",
            "programmed_volume": "",
            "feature_name": (
                f"{'liquid_transfer' if target is pipette else 'templating'}."
                f"{step}.{normalize_name(feature['source_field'].rsplit('.', 1)[-1])}"
            ),
            "value": feature["numeric_value"] if feature["numeric_value"] is not None else feature["value"],
            "unit": feature["unit"], "setpoint_or_measured": "setpoint" if tier == "A" else "unresolved",
            "feature_tier": tier, "classification": classification,
            "source_file": feature["source_file"], "source_field": feature["source_field"],
            "start_time": "", "end_time": "", "parse_confidence": "high" if tier == "A" else "low",
            "exclusion_reason": reason,
        })
    return vacuum, pipette


def derive_observation(inventory_row: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, Any]:
    hostname = inventory_row["hostname"]
    instrument_id = stable_hash(hostname, "inst_")
    experiment = inventory_row.get("physical_experiment_name") or inventory_row.get("run_name", "")
    timestamp_match = re.search(r"(20\d{15,17})", experiment)
    run_start = ""
    if timestamp_match:
        raw = timestamp_match.group(1)[:14]
        try:
            run_start = datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            run_start = ""
    canonical = re.sub(r"_20\d{15,17}.*$", "", experiment)
    canonical = re.sub(r"(?:_re(?:analysis|analyzed)?\d*)+$", "", canonical, flags=re.I)
    if not canonical:
        canonical = inventory_row.get("run_name", "")
    physical_run_id = stable_hash(f"{hostname}|{canonical}", "run_")
    lane = inventory_row.get("chip_lane", "") or "lane_unknown"
    observation_id = stable_hash(f"{physical_run_id}|{lane}", "obs_")
    return {
        "observation_id": observation_id,
        "physical_run_id": physical_run_id,
        "instrument_id": instrument_id,
        "chip_lane": lane,
        "run_start_time": run_start or inventory_row.get("remote_mtime_utc", ""),
        "analysis_id": stable_hash(inventory_row["remote_path"], "ana_"),
        "archive_id": inventory_row["archive_id"],
        "archive_sha256": manifest_row.get("sha256", ""),
        "hostname_for_report": hostname,
        "raw_experiment_name": experiment,
    }


def resolve_outcomes(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    for outcome in (*TARGETS, SECONDARY_TARGET):
        options = [x for x in candidates if x["outcome"] == outcome]
        if not options:
            resolved[outcome] = np.nan
            resolved[f"{outcome}__source_file"] = ""
            resolved[f"{outcome}__source_field"] = ""
            continue
        options.sort(key=lambda x: (x["priority"], x["source_file"], x["source_field"]))
        chosen = options[0]
        resolved[outcome] = chosen["value"]
        resolved[f"{outcome}__source_file"] = chosen["source_file"]
        resolved[f"{outcome}__source_field"] = chosen["source_field"]
        values = sorted({round(float(x["value"]), 10) for x in options})
        resolved[f"{outcome}__discrepancy"] = len(values) > 1
        for option in options:
            provenance.append({
                **option, "selected": option is chosen,
                "discrepancy": len(values) > 1,
                "all_values": "|".join(map(str, values)),
            })
    return resolved, provenance


def parse_archive(
    inventory_row: dict[str, Any],
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    observation = derive_observation(inventory_row, manifest_row)
    result = {
        "observation": observation, "features": [], "vacuum": [], "pipette": [],
        "outcome_provenance": [], "parse_warnings": [], "member_count": 0,
    }
    path = Path(manifest_row["local_path"])
    if not path.exists():
        result["parse_warnings"].append("local archive missing")
        return result
    outcome_candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    pressure_rows_vacuum: list[dict[str, Any]] = []
    pressure_rows_pipette: list[dict[str, Any]] = []
    raw_experiment_key = ""
    for member in iter_archive_members(path, result["parse_warnings"]):
        result["member_count"] += 1
        source = member.name
        base = source_basename(source)
        lower = base.lower()
        if lower == "info.csv":
            outcome_candidates.extend(parse_info_csv(member.data, source))
        if EVENT_FILE_RE.search(base):
            events.extend(event_rows(member.data, source))
        if PRESSURE_FILE_RE.search(base):
            if lower == "vacuum_log.csv":
                pressure_rows_vacuum.extend(parse_vacuum_log(member.data, source, observation))
            elif PIPETTE_FILE_RE.search(base):
                frame = dataframe_from_bytes(member.data)
                pressure_rows_pipette.extend(
                    pressure_trace_rows(frame, source, observation, "liquid_transfer")
                )
            else:
                frame = dataframe_from_bytes(member.data)
                pressure_rows_vacuum.extend(
                    pressure_trace_rows(frame, source, observation, "templating")
                )
        if lower.endswith(".json"):
            normalized_source = source.split("!")[-1].replace("\\", "/").lower()
            primary_output = bool(re.search(
                r"/outputs/basecallingactor-00/(basecaller|ionstats_basecaller)\.json$",
                normalized_source,
            ))
            config_json = bool(CONFIG_SOURCE_RE.search(base))
            diagnostic_json = bool(re.search(r"(bead|calibrat|qc|summary|metrics)", base, re.I))
            if not (primary_output or config_json or diagnostic_json):
                continue
            try:
                obj = json.loads(member.data.decode("utf-8-sig", "replace"))
            except Exception as exc:
                result["parse_warnings"].append(f"json {source}: {exc}")
                continue
            outcome_candidates.extend(extract_outcomes_from_json(obj, source))
            parse_features = config_json or primary_output or diagnostic_json
            if parse_features:
                for field, value in flatten_json(obj):
                    if (
                        source_basename(source).lower() == "ion_params_00.json"
                        and field.lower() == "expname"
                        and value
                    ):
                        raw_experiment_key = str(value)
                    record = feature_record(
                        observation["observation_id"], observation["instrument_id"],
                        source, field, value,
                    )
                    result["features"].append(record)
                    if record["actionability_tier"] == "A" and isinstance(value, str):
                        for arg, arg_value in parse_cli_arguments(value):
                            result["features"].append(feature_record(
                                observation["observation_id"], observation["instrument_id"],
                                source, f"{field}.arg.{arg}", arg_value,
                            ))
        elif lower.endswith(".ini") and should_parse_ini(source):
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                parser.read_string(member.data.decode("utf-8-sig", "replace"))
                for section in parser.sections():
                    for key, value in parser.items(section):
                        field = f"{section}.{key}"
                        result["features"].append(feature_record(
                            observation["observation_id"], observation["instrument_id"],
                            source, field, value,
                        ))
                        for arg, arg_value in parse_cli_arguments(value):
                            result["features"].append(feature_record(
                                observation["observation_id"], observation["instrument_id"],
                                source, f"{field}.arg.{arg}", arg_value,
                            ))
            except configparser.MissingSectionHeaderError:
                text = member.data.decode("utf-8-sig", "replace")
                for line in text.splitlines():
                    if "=" not in line or line.lstrip().startswith(("#", ";", "import ")):
                        continue
                    key, value = line.split("=", 1)
                    result["features"].append(feature_record(
                        observation["observation_id"], observation["instrument_id"],
                        source, key.strip(), value.strip(),
                    ))
            except Exception as exc:
                result["parse_warnings"].append(f"ini {source}: {exc}")
        elif lower.endswith(".csv") and CONFIG_SOURCE_RE.search(base) and lower != "info.csv":
            frame = dataframe_from_bytes(member.data)
            if not frame.empty and len(frame) <= 10000:
                for column in frame.columns:
                    values = frame[column].dropna().drop_duplicates()
                    if 0 < len(values) <= 20:
                        for index, value in enumerate(values):
                            result["features"].append(feature_record(
                                observation["observation_id"], observation["instrument_id"],
                                source, f"{column}.level_{index}", value,
                            ))
    align_pressure_steps(pressure_rows_vacuum, events)
    align_pressure_steps(pressure_rows_pipette, events)
    setting_vacuum, setting_pipette = settings_pressure_rows(result["features"], observation)
    result["vacuum"] = setting_vacuum + pressure_rows_vacuum
    result["pipette"] = setting_pipette + pressure_rows_pipette
    outcomes, provenance = resolve_outcomes(outcome_candidates)
    result["observation"].update(outcomes)
    if raw_experiment_key:
        old_observation_id = result["observation"]["observation_id"]
        hostname = result["observation"]["hostname_for_report"]
        physical_run_id = stable_hash(f"{hostname}|{raw_experiment_key}", "run_")
        observation_id = stable_hash(
            f"{physical_run_id}|{result['observation']['chip_lane']}", "obs_"
        )
        result["observation"]["physical_run_id"] = physical_run_id
        result["observation"]["observation_id"] = observation_id
        timestamp_match = re.search(
            r"R_(20\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})",
            raw_experiment_key,
        )
        if timestamp_match:
            try:
                parsed_start = datetime(
                    *map(int, timestamp_match.groups()), tzinfo=timezone.utc
                )
                result["observation"]["run_start_time"] = (
                    parsed_start.isoformat().replace("+00:00", "Z")
                )
            except ValueError:
                pass
        for collection in (
            result["features"], pressure_rows_vacuum, pressure_rows_pipette,
            setting_vacuum, setting_pipette,
        ):
            for row in collection:
                if row.get("observation_id") == old_observation_id:
                    row["observation_id"] = observation_id
                row["physical_run_id"] = physical_run_id
    for row in provenance:
        row["archive_id"] = observation["archive_id"]
        row["observation_id"] = result["observation"]["observation_id"]
    result["outcome_provenance"] = provenance
    result["completeness_score"] = (
        sum(pd.notna(outcomes.get(x)) for x in (*TARGETS, SECONDARY_TARGET)) * 100000
        + len(result["features"]) + len(result["vacuum"]) + len(result["pipette"])
        + result["member_count"]
    )
    return result


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_dataframe(rows: list[dict[str, Any]], columns: Iterable[str] = ()) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=list(columns))


def select_primary_archives(
    parsed: list[dict[str, Any]],
    inventory_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        groups[item["observation"]["observation_id"]].append(item)
    retained: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for observation_id, items in groups.items():
        items.sort(
            key=lambda x: (
                int(x.get("completeness_score", 0)),
                inventory_by_id.get(x["observation"]["archive_id"], {}).get("remote_mtime_utc", ""),
            ),
            reverse=True,
        )
        chosen = items[0]
        retained.append(chosen)
        raw_names = {x["observation"].get("raw_experiment_name", "") for x in items}
        for rank, item in enumerate(items):
            archive_id = item["observation"]["archive_id"]
            inv = inventory_by_id.get(archive_id, {})
            duplicate_rows.append({
                "observation_id": observation_id,
                "physical_run_id": item["observation"]["physical_run_id"],
                "chip_lane": item["observation"]["chip_lane"],
                "archive_id": archive_id,
                "remote_path": inv.get("remote_path", ""),
                "analysis_id": item["observation"]["analysis_id"],
                "completeness_score": item.get("completeness_score", 0),
                "member_count": item.get("member_count", 0),
                "retained_primary": rank == 0,
                "duplicate_type": (
                    "unique" if len(items) == 1 else
                    "reanalysis_or_repeated_export" if len(raw_names) > 1 else
                    "repeated_export"
                ),
                "selection_reason": (
                    "most_complete_then_latest" if rank == 0
                    else "excluded_from_primary_same_physical_run_and_lane"
                ),
            })
    retained.sort(key=lambda x: x["observation"]["observation_id"])
    return retained, safe_dataframe(duplicate_rows, (
        "observation_id", "physical_run_id", "chip_lane", "archive_id",
        "remote_path", "analysis_id", "completeness_score", "member_count",
        "retained_primary", "duplicate_type", "selection_reason",
    ))


def feature_value(row: pd.Series) -> Any:
    if pd.notna(row.get("numeric_value")):
        return float(row["numeric_value"])
    value = row.get("value")
    return np.nan if value is None or str(value).strip() == "" else str(value)[:1000]


def append_pressure_features(
    features: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        value = row.get("value")
        numeric = parse_number(value)
        tier = row.get("feature_tier", "C")
        features.append({
            "observation_id": row.get("observation_id", ""),
            "instrument_id": row.get("instrument_id", ""),
            "feature_name": row.get("feature_name", ""),
            "value": value,
            "numeric_value": numeric,
            "unit": row.get("unit", ""),
            "data_type": "numeric" if numeric is not None else "categorical",
            "source_file": row.get("source_file", ""),
            "source_field": row.get("source_field", ""),
            "availability_time": "pre_run" if tier == "A" else "during_or_post_run",
            "actionability_tier": tier,
            "exclusion_reason": row.get("exclusion_reason", ""),
        })


def build_wide_dataset(
    retained: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.DataFrame([x["observation"] for x in retained])
    all_features: list[dict[str, Any]] = []
    vacuum_rows: list[dict[str, Any]] = []
    pipette_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for item in retained:
        all_features.extend(item["features"])
        vacuum_rows.extend(item["vacuum"])
        pipette_rows.extend(item["pipette"])
        provenance_rows.extend(item["outcome_provenance"])
    append_pressure_features(all_features, vacuum_rows)
    append_pressure_features(all_features, pipette_rows)
    features = safe_dataframe(all_features, (
        "observation_id", "instrument_id", "feature_name", "value", "numeric_value",
        "unit", "data_type", "source_file", "source_field", "availability_time",
        "actionability_tier", "exclusion_reason",
    ))
    if features.empty:
        wide = observations.copy()
    else:
        features["effective_value"] = features.apply(feature_value, axis=1)
        collapsed = (
            features.groupby(["observation_id", "feature_name"], dropna=False)["effective_value"]
            .agg(lambda x: next(iter(x.dropna()), np.nan) if x.dropna().nunique() <= 1
                 else "|".join(sorted({str(v) for v in x.dropna()})[:20]))
            .reset_index()
        )
        pivot = collapsed.pivot(index="observation_id", columns="feature_name", values="effective_value")
        pivot.columns.name = None
        wide = observations.merge(pivot.reset_index(), on="observation_id", how="left")
    run_dates = pd.to_datetime(wide.get("run_start_time"), utc=True, errors="coerce")
    wide["run_date_ordinal"] = run_dates.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    return (
        wide,
        features,
        safe_dataframe(vacuum_rows),
        safe_dataframe(pipette_rows),
        safe_dataframe(provenance_rows),
    )


def encoded_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(series.notna().sum() * .9)):
        return numeric
    return series.fillna("__MISSING__").astype(str)


def exact_duplicate_columns(dataset: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for column in columns:
        series = encoded_series(dataset[column])
        signature = hashlib.sha256(
            pd.util.hash_pandas_object(series, index=False).values.tobytes()
        ).hexdigest()
        if signature in hashes and series.equals(encoded_series(dataset[hashes[signature]])):
            duplicates[column] = hashes[signature]
        else:
            hashes[signature] = column
    return duplicates


def context_confounder(
    dataset: pd.DataFrame,
    feature: str,
    context_columns: list[str],
) -> tuple[bool, str]:
    x = encoded_series(dataset[feature])
    for context in context_columns:
        if context not in dataset or context == feature:
            continue
        c = encoded_series(dataset[context])
        valid = dataset[feature].notna() & dataset[context].notna()
        if valid.sum() < 5 or x[valid].nunique() < 2 or c[valid].nunique() < 2:
            continue
        x_given_c = pd.DataFrame({"x": x[valid], "c": c[valid]}).groupby("c")["x"].nunique().max()
        c_given_x = pd.DataFrame({"x": x[valid], "c": c[valid]}).groupby("x")["c"].nunique().max()
        if x_given_c == 1 and c_given_x == 1:
            return True, context
    return False, ""


def cross_instrument_overlap(
    dataset: pd.DataFrame,
    feature: str,
    instrument_indices: dict[str, np.ndarray] | None = None,
) -> float:
    numeric = pd.to_numeric(dataset[feature], errors="coerce")
    if instrument_indices is None:
        instrument_indices = {
            str(instrument): np.flatnonzero(dataset["instrument_id"].to_numpy() == instrument)
            for instrument in dataset["instrument_id"].dropna().unique()
        }
    ranges = []
    for indices in instrument_indices.values():
        part = numeric.iloc[indices].dropna()
        if part.nunique() >= 2:
            ranges.append((part.min(), part.max()))
    if len(ranges) < 2:
        return np.nan
    intersection = min(x[1] for x in ranges) - max(x[0] for x in ranges)
    union = max(x[1] for x in ranges) - min(x[0] for x in ranges)
    return max(0.0, float(intersection / union)) if union > 0 else 0.0


def build_feature_policy(
    dataset: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(dataset)
    if features.empty:
        columns = [
            "feature_name", "unit", "data_type", "source_file", "source_field",
            "availability_time", "actionability_tier", "missingness",
            "number_unique", "instruments_on_which_varies", "actionability_label",
            "exclusion_reason",
        ]
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy(), empty.copy()
    meta_rows: list[dict[str, Any]] = []
    tier_order = {"D": 4, "C": 3, "B": 2, "A": 1}
    feature_names = [x for x in sorted(features["feature_name"].dropna().unique()) if x in dataset]
    print(f"Feature-policy audit over {len(feature_names)} extracted feature columns", flush=True)
    duplicates = exact_duplicate_columns(dataset, feature_names)
    instrument_values = dataset["instrument_id"].to_numpy()
    instrument_indices = {
        str(instrument): np.flatnonzero(instrument_values == instrument)
        for instrument in dataset["instrument_id"].dropna().unique()
    }
    b_context = []
    for name, group in features.groupby("feature_name"):
        if group["actionability_tier"].mode().iloc[0] == "B" and name in dataset:
            unique = dataset[name].nunique(dropna=True)
            if 1 < unique <= min(30, max(3, n // 3)):
                b_context.append(name)
    context_columns = ["instrument_id", "chip_lane", "run_date_ordinal"] + b_context[:30]
    for name, group in features.groupby("feature_name", sort=True):
        if name not in dataset:
            continue
        tiers = sorted(group["actionability_tier"].dropna().unique(), key=lambda x: tier_order.get(x, 9))
        tier = tiers[0] if len(tiers) == 1 else max(tiers, key=lambda x: tier_order.get(x, 9))
        nonmissing = dataset[name].notna().sum()
        unique = dataset[name].nunique(dropna=True)
        top_frequency = (
            dataset[name].value_counts(dropna=True, normalize=True).iloc[0]
            if nonmissing else 1.0
        )
        varies_by_instrument = sum(
            dataset[name].iloc[indices].nunique(dropna=True) >= 2
            for indices in instrument_indices.values()
        )
        units = sorted({str(x) for x in group["unit"].dropna() if str(x)})
        data_type = (
            "numeric" if pd.to_numeric(dataset[name], errors="coerce").notna().sum()
            >= max(1, int(nonmissing * .9)) else "categorical"
        )
        preliminary_missingness = 1 - nonmissing / n if n else 1.0
        tied, tied_to = (
            context_confounder(dataset, name, context_columns)
            if (
                tier == "A" and unique >= 3 and top_frequency < .95
                and preliminary_missingness <= .60 and varies_by_instrument >= 2
            )
            else (False, "")
        )
        reasons = sorted({str(x) for x in group["exclusion_reason"].dropna() if str(x)})
        flags: list[str] = []
        if unique <= 1:
            flags.append("constant")
        elif top_frequency >= .95:
            flags.append("near_constant")
        if unique < 3:
            flags.append("fewer_than_three_distinct_values")
        if varies_by_instrument <= 1 and unique >= 2:
            flags.append("varies_on_at_most_one_instrument")
        if len(units) > 1:
            flags.append("unit_inconsistency")
        if name in duplicates:
            flags.append(f"duplicate_of:{duplicates[name]}")
        if tied:
            flags.append(f"perfectly_tied_to:{tied_to}")
        missingness = preliminary_missingness
        if missingness > .60:
            flags.append("missingness_over_60pct")
        numeric = pd.to_numeric(dataset[name], errors="coerce")
        impossible = False
        lower_name = name.lower()
        if numeric.notna().any():
            if re.search(r"(volume|duration|hold|cycles|flows|count|reads)", lower_name) and (numeric < 0).any():
                impossible = True
            if re.search(r"(accuracy|percent|pct)", lower_name) and ((numeric < 0) | (numeric > 100)).any():
                impossible = True
            if re.search(r"(temperature|temp)", lower_name) and ((numeric < -20) | (numeric > 150)).any():
                impossible = True
        if impossible:
            flags.append("impossible_value_detected")
        eligible = (
            tier in {"A", "B", "C"}
            and unique >= 3
            and top_frequency < .95
            and missingness <= .60
            and name not in duplicates
            and not tied
            and not impossible
            and len(units) <= 1
        )
        if tier == "A" and varies_by_instrument <= 1:
            eligible = False
        if tier == "D":
            eligible = False
        action_label = (
            "eligible_actionable_candidate" if tier == "A" and eligible
            else "diagnostic_only" if tier == "C"
            else "context_only" if tier == "B"
            else "forbidden_leakage" if tier == "D"
            else "insufficient_variation"
        )
        meta_rows.append({
            "feature_name": name,
            "unit": "|".join(units),
            "data_type": data_type,
            "source_file": "|".join(sorted(set(group["source_file"].astype(str)))[:20]),
            "source_field": "|".join(sorted(set(group["source_field"].astype(str)))[:20]),
            "availability_time": "|".join(sorted(set(group["availability_time"].astype(str)))),
            "missingness": missingness,
            "number_unique": unique,
            "instruments_observed": dataset.loc[dataset[name].notna(), "instrument_id"].nunique(),
            "instruments_on_which_varies": varies_by_instrument,
            "cross_instrument_range_overlap": (
                cross_instrument_overlap(dataset, name, instrument_indices)
                if data_type == "numeric" and missingness <= .60 else np.nan
            ),
            "actionability_tier": tier,
            "actionability_label": action_label,
            "modeling_eligible": eligible,
            "tied_context_variable": tied_to,
            "duplicate_of": duplicates.get(name, ""),
            "top_value_frequency": top_frequency,
            "observed_min": numeric.min() if numeric.notna().any() else "",
            "observed_max": numeric.max() if numeric.notna().any() else "",
            "exclusion_reason": ";".join(reasons + flags),
        })
    policy = pd.DataFrame(meta_rows).sort_values(
        ["actionability_tier", "feature_name"], kind="stable"
    )
    policy = pd.concat([
        policy,
        pd.DataFrame([
            {
                "feature_name": "instrument_id", "unit": "", "data_type": "categorical",
                "source_file": "derived inventory", "source_field": "hashed hostname",
                "availability_time": "pre_run", "missingness": 0.0,
                "number_unique": dataset["instrument_id"].nunique(),
                "instruments_observed": dataset["instrument_id"].nunique(),
                "instruments_on_which_varies": 0, "cross_instrument_range_overlap": np.nan,
                "actionability_tier": "B", "actionability_label": "context_only",
                "modeling_eligible": True, "tied_context_variable": "",
                "duplicate_of": "", "top_value_frequency": dataset["instrument_id"].value_counts(normalize=True).max(),
                "observed_min": "", "observed_max": "", "exclusion_reason": "context_not_tunable",
            },
            {
                "feature_name": "chip_lane", "unit": "", "data_type": "categorical",
                "source_file": "derived inventory", "source_field": "chip lane",
                "availability_time": "pre_run", "missingness": 0.0,
                "number_unique": dataset["chip_lane"].nunique(),
                "instruments_observed": dataset["instrument_id"].nunique(),
                "instruments_on_which_varies": 0, "cross_instrument_range_overlap": np.nan,
                "actionability_tier": "B", "actionability_label": "context_only",
                "modeling_eligible": True, "tied_context_variable": "",
                "duplicate_of": "", "top_value_frequency": dataset["chip_lane"].value_counts(normalize=True).max(),
                "observed_min": "", "observed_max": "", "exclusion_reason": "context_not_tunable",
            },
            {
                "feature_name": "run_date_ordinal", "unit": "day", "data_type": "numeric",
                "source_file": "derived inventory", "source_field": "run_start_time",
                "availability_time": "pre_run", "missingness": dataset["run_date_ordinal"].isna().mean(),
                "number_unique": dataset["run_date_ordinal"].nunique(),
                "instruments_observed": dataset["instrument_id"].nunique(),
                "instruments_on_which_varies": dataset.groupby("instrument_id")["run_date_ordinal"].nunique().ge(2).sum(),
                "cross_instrument_range_overlap": cross_instrument_overlap(dataset, "run_date_ordinal"),
                "actionability_tier": "B", "actionability_label": "context_only",
                "modeling_eligible": True, "tied_context_variable": "",
                "duplicate_of": "", "top_value_frequency": dataset["run_date_ordinal"].value_counts(normalize=True).max(),
                "observed_min": dataset["run_date_ordinal"].min(), "observed_max": dataset["run_date_ordinal"].max(),
                "exclusion_reason": "date_context_not_tunable",
            },
        ])
    ], ignore_index=True)
    excluded = policy.loc[
        (policy["actionability_tier"] == "D") | (~policy["modeling_eligible"].astype(bool))
    ].copy()
    dictionary = policy.rename(columns={
        "actionability_tier": "tier",
        "number_unique": "unique_values",
    }).copy()
    return policy, excluded, dictionary


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    if not m:
        return result
    adjusted = np.minimum.accumulate((valid.to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def grouped_bootstrap_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    groups = frame["physical_run_id"].dropna().unique()
    values: list[float] = []
    if len(groups) < 4:
        return np.nan, np.nan
    grouped = {g: frame[frame["physical_run_id"] == g] for g in groups}
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([grouped[g] for g in sampled], ignore_index=True)
        if boot[x].nunique() >= 3 and boot[y].nunique() >= 3:
            rho = stats.spearmanr(boot[x], boot[y], nan_policy="omit").statistic
            if np.isfinite(rho):
                values.append(float(rho))
    if len(values) < max(30, iterations // 10):
        return np.nan, np.nan
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def adjusted_iqr_effect(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    clean = frame[[x, y, "instrument_id"]].dropna().copy()
    if len(clean) < 5:
        return np.nan, np.nan
    x_iqr = clean[x].quantile(.75) - clean[x].quantile(.25)
    if not np.isfinite(x_iqr) or x_iqr == 0:
        return np.nan, np.nan
    clean["x_centered"] = clean[x] - clean.groupby("instrument_id")[x].transform("mean")
    clean["y_centered"] = clean[y] - clean.groupby("instrument_id")[y].transform("mean")
    if np.isclose(np.sum(clean["x_centered"] ** 2), 0):
        slope = stats.linregress(clean[x], clean[y]).slope
    else:
        slope = float(np.sum(clean["x_centered"] * clean["y_centered"]) / np.sum(clean["x_centered"] ** 2))
    effect = slope * x_iqr
    influential = (
        np.abs(stats.zscore(clean[y], nan_policy="omit")) > 3
        if len(clean) >= 8 and clean[y].std() > 0 else np.zeros(len(clean), dtype=bool)
    )
    sensitivity = np.nan
    if (~influential).sum() >= 5 and clean.loc[~influential, x].nunique() >= 3:
        reduced = clean.loc[~influential]
        reduced_iqr = reduced[x].quantile(.75) - reduced[x].quantile(.25)
        if not np.isclose(np.sum(reduced["x_centered"] ** 2), 0):
            reduced_slope = np.sum(reduced["x_centered"] * reduced["y_centered"]) / np.sum(reduced["x_centered"] ** 2)
            sensitivity = float(reduced_slope * reduced_iqr)
    return float(effect), sensitivity


def categorical_adjusted_comparison(
    frame: pd.DataFrame,
    x: str,
    y: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Compare adequately represented levels after removing instrument means."""
    clean = frame[[x, y, "instrument_id", "physical_run_id"]].dropna().copy()
    clean[x] = clean[x].astype(str)
    counts = clean[x].value_counts()
    rare = set(counts[counts < 5].index)
    if rare:
        clean.loc[clean[x].isin(rare), x] = "__RARE__"
    counts = clean[x].value_counts()
    if not (2 <= len(counts) <= 10) or counts.min() < 5:
        return None
    clean["adjusted_y"] = clean[y] - clean.groupby("instrument_id")[y].transform("mean")
    means = clean.groupby(x)["adjusted_y"].mean().sort_values()
    low_level, high_level = str(means.index[0]), str(means.index[-1])
    effect = float(means.iloc[-1] - means.iloc[0])
    groups = [part["adjusted_y"].to_numpy(dtype=float) for _, part in clean.groupby(x)]
    p_value = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan

    run_ids = clean["physical_run_id"].unique()
    by_run = {run_id: clean[clean["physical_run_id"] == run_id] for run_id in run_ids}
    boot_effects: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(run_ids, size=len(run_ids), replace=True)
        boot = pd.concat([by_run[run_id] for run_id in sampled], ignore_index=True)
        boot_means = boot.groupby(x)["adjusted_y"].mean()
        if low_level in boot_means and high_level in boot_means:
            boot_effects.append(float(boot_means[high_level] - boot_means[low_level]))
    if len(boot_effects) >= max(30, iterations // 10):
        ci_low, ci_high = np.quantile(boot_effects, [.025, .975])
    else:
        ci_low, ci_high = np.nan, np.nan

    within: list[float] = []
    for _, part in clean.groupby("instrument_id"):
        level_counts = part[x].value_counts()
        if level_counts.get(low_level, 0) >= 3 and level_counts.get(high_level, 0) >= 3:
            level_means = part.groupby(x)[y].mean()
            within.append(float(level_means[high_level] - level_means[low_level]))
    return {
        "p_value": p_value,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "within_instrument_rho_median": float(np.median(within)) if within else np.nan,
        "within_instrument_count": len(within),
        "categorical_adjusted_difference": effect,
        "reference_level": low_level,
        "comparison_level": high_level,
        "levels_summary": "|".join(f"{level}:{count}" for level, count in counts.items()),
    }


def correlation_analysis(
    dataset: pd.DataFrame,
    policy: pd.DataFrame,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    numeric_candidates = policy[
        (policy["actionability_tier"] == "A")
        & (policy["data_type"] == "numeric")
        & (policy["modeling_eligible"].astype(bool))
    ]["feature_name"].tolist()
    categorical_candidates = policy[
        (policy["actionability_tier"] == "A")
        & (policy["data_type"] != "numeric")
        & (policy["modeling_eligible"].astype(bool))
    ]["feature_name"].tolist()
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for feature in numeric_candidates:
            frame = dataset[[feature, target, "instrument_id", "physical_run_id"]].copy()
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            frame[target] = pd.to_numeric(frame[target], errors="coerce")
            frame = frame.dropna()
            if len(frame) < 8 or frame[feature].nunique() < 3 or frame[target].nunique() < 3:
                continue
            test = stats.spearmanr(frame[feature], frame[target])
            low, high = grouped_bootstrap_spearman(
                frame, feature, target, bootstrap_iterations, rng
            )
            within = []
            for _, part in frame.groupby("instrument_id"):
                if len(part) >= 5 and part[feature].nunique() >= 3 and part[target].nunique() >= 3:
                    rho = stats.spearmanr(part[feature], part[target]).statistic
                    if np.isfinite(rho):
                        within.append(float(rho))
            effect, sensitivity = adjusted_iqr_effect(frame, feature, target)
            rows.append({
                "feature_name": feature, "outcome": target, "test_type": "spearman",
                "usable_n": len(frame), "unique_x": frame[feature].nunique(),
                "spearman_rho": float(test.statistic), "p_value": float(test.pvalue),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "within_instrument_rho_median": np.median(within) if within else np.nan,
                "within_instrument_count": len(within),
                "x_iqr": frame[feature].quantile(.75) - frame[feature].quantile(.25),
                "estimated_y_change_per_x_iqr": effect,
                "sensitivity_effect_without_outlying_y": sensitivity,
                "observed_x_min": frame[feature].min(), "observed_x_max": frame[feature].max(),
            })
        for feature in categorical_candidates:
            frame = dataset[[feature, target, "instrument_id", "physical_run_id"]].copy()
            frame[target] = pd.to_numeric(frame[target], errors="coerce")
            frame = frame.dropna()
            if len(frame) < 12 or frame[feature].nunique() < 2 or frame[target].nunique() < 3:
                continue
            comparison = categorical_adjusted_comparison(
                frame, feature, target, bootstrap_iterations, rng
            )
            if comparison is None:
                continue
            rows.append({
                "feature_name": feature, "outcome": target,
                "test_type": "categorical_instrument_adjusted_kruskal",
                "usable_n": len(frame), "unique_x": frame[feature].nunique(),
                "spearman_rho": np.nan,
                "x_iqr": np.nan, "estimated_y_change_per_x_iqr": np.nan,
                "sensitivity_effect_without_outlying_y": np.nan,
                "observed_x_min": "", "observed_x_max": "",
                **comparison,
            })
    result = safe_dataframe(rows, (
        "feature_name", "outcome", "test_type", "usable_n", "unique_x",
        "spearman_rho", "p_value", "fdr_q_value", "bootstrap_ci_low",
        "bootstrap_ci_high", "within_instrument_rho_median",
        "within_instrument_count", "x_iqr", "estimated_y_change_per_x_iqr",
        "sensitivity_effect_without_outlying_y", "observed_x_min", "observed_x_max",
        "categorical_adjusted_difference", "reference_level", "comparison_level",
        "levels_summary",
    ))
    if not result.empty:
        result["fdr_q_value"] = result.groupby("outcome", group_keys=False)["p_value"].apply(benjamini_hochberg)
        result = result.sort_values(["outcome", "fdr_q_value", "p_value"], kind="stable")
    return result


def choose_predictors(
    dataset: pd.DataFrame,
    policy: pd.DataFrame,
    dataset_kind: str,
    max_features: int,
) -> tuple[list[str], list[str], list[str]]:
    tiers = {"A", "B"} if dataset_kind == "actionable" else {"A", "B", "C"}
    eligible = policy[
        policy["actionability_tier"].isin(tiers)
        & policy["modeling_eligible"].astype(bool)
        & policy["feature_name"].isin(dataset.columns)
    ].copy()
    eligible["priority"] = eligible["actionability_tier"].map({"A": 0, "B": 1, "C": 2})
    eligible = eligible.sort_values(
        ["priority", "missingness", "number_unique", "feature_name"], kind="stable"
    )
    if len(eligible) > max_features:
        mandatory = eligible[eligible["actionability_tier"] == "A"]
        remainder = eligible[eligible["actionability_tier"] != "A"]
        slots = max(0, max_features - len(mandatory))
        eligible = pd.concat([mandatory, remainder.head(slots)], ignore_index=True)
        if len(eligible) > max_features:
            eligible = eligible.head(max_features)
    predictors = eligible["feature_name"].tolist()
    numeric = eligible.loc[eligible["data_type"] == "numeric", "feature_name"].tolist()
    categorical = [x for x in predictors if x not in numeric]
    return predictors, numeric, categorical


def stable_f_regression(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Run fold-local F scoring while silencing harmless roundoff warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in sqrt", category=RuntimeWarning
        )
        return f_regression(x, y)


def make_model_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_name: str,
) -> tuple[Pipeline, dict[str, list[Any]]]:
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=2,
                    sparse_output=True,
                )),
            ]),
            categorical,
        ))
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=.3)
    if model_name == "elastic_net":
        estimator = ElasticNet(max_iter=20000, random_state=SEED)
        grid = {
            "select__percentile": [50, 100],
            "model__alpha": [.1, 1.0],
            "model__l1_ratio": [.5],
        }
    elif model_name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=100, random_state=SEED, n_jobs=1,
            min_samples_leaf=2,
        )
        grid = {
            "select__percentile": [50, 100],
            "model__max_depth": [3, None],
            "model__min_samples_leaf": [2],
            "model__max_features": ["sqrt"],
        }
    else:
        raise ValueError(model_name)
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("select", SelectPercentile(score_func=stable_f_regression, percentile=100)),
        ("model", estimator),
    ])
    return pipeline, grid


def validation_splits(frame: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
    groups = frame["physical_run_id"].astype(str).to_numpy()
    if scheme == "grouped":
        unique = np.unique(groups)
        if len(unique) < 3:
            return []
        cv = GroupKFold(n_splits=min(5, len(unique)))
        return [(train, test, f"group_fold_{i + 1}") for i, (train, test) in enumerate(cv.split(frame, groups=groups))]
    if scheme == "leave_one_instrument_out":
        splits = []
        for instrument in sorted(frame["instrument_id"].dropna().unique()):
            test = np.flatnonzero(frame["instrument_id"].to_numpy() == instrument)
            train = np.flatnonzero(frame["instrument_id"].to_numpy() != instrument)
            if len(test) >= 2 and len(train) >= 10:
                splits.append((train, test, f"heldout_{instrument}"))
        return splits
    if scheme == "chronological":
        dates = pd.to_datetime(frame["run_start_time"], utc=True, errors="coerce")
        group_dates = (
            pd.DataFrame({"group": groups, "date": dates})
            .groupby("group")["date"].min()
            .sort_values()
        )
        valid_groups = group_dates.dropna().index.to_numpy()
        if len(valid_groups) < 5:
            return []
        test_count = max(1, int(math.ceil(len(valid_groups) * .20)))
        test_groups = set(valid_groups[-test_count:])
        test = np.flatnonzero(np.isin(groups, list(test_groups)))
        train = np.flatnonzero(~np.isin(groups, list(test_groups)))
        return [(train, test, "newest_20pct_physical_runs")] if len(test) >= 2 and len(train) >= 10 else []
    raise ValueError(scheme)


def fit_nested_model(
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    warnings_out: list[str],
) -> Any:
    unique_groups = groups_train.nunique()
    if unique_groups >= 3 and len(y_train) >= 12:
        inner = GroupKFold(n_splits=min(3, unique_groups))
        search = GridSearchCV(
            pipeline, grid, scoring="neg_mean_absolute_error", cv=inner,
            n_jobs=min(4, os.cpu_count() or 1), refit=True, error_score=np.nan,
            pre_dispatch=4,
        )
        try:
            search.fit(x_train, y_train, groups=groups_train)
            if not hasattr(search, "best_estimator_"):
                raise ValueError("inner search produced no valid estimator")
            return search
        except Exception as exc:
            warnings_out.append(f"inner model selection fallback: {exc}")
    fallback = clone(pipeline)
    fallback.fit(x_train, y_train)
    return fallback


def heldout_permutation_importance(
    fitted: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    baseline = mean_absolute_error(y_test, fitted.predict(x_test))
    rows = []
    for feature in features:
        deltas = []
        for _ in range(repeats):
            shuffled = x_test.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            deltas.append(mean_absolute_error(y_test, fitted.predict(shuffled)) - baseline)
        rows.append({
            "feature_name": feature,
            "heldout_importance": float(np.mean(deltas)),
            "repeat_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        })
    return rows


def run_models(
    dataset: pd.DataFrame,
    policy: pd.DataFrame,
    max_features: int,
    permutation_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_warnings: list[str] = []
    rng = np.random.default_rng(SEED)
    policy_map = policy.set_index("feature_name")
    for dataset_kind in ("actionable", "diagnostic"):
        predictors, numeric, categorical = choose_predictors(
            dataset, policy, dataset_kind, max_features
        )
        if not predictors:
            model_warnings.append(f"{dataset_kind}: no eligible predictors")
            continue
        x_all = dataset[predictors].copy()
        for column in numeric:
            x_all[column] = pd.to_numeric(x_all[column], errors="coerce")
        for column in categorical:
            series = x_all[column]
            x_all[column] = series.map(
                lambda value: np.nan if value is None or pd.isna(value) else str(value)
            ).astype(object)
        tier_a = [
            x for x in predictors
            if x in policy_map.index and policy_map.loc[x, "actionability_tier"] == "A"
        ]
        tier_c_pressure = [
            x for x in predictors
            if x in policy_map.index and policy_map.loc[x, "actionability_tier"] == "C"
            and re.search(r"pressure|vacuum|pipette", x, re.I)
        ]
        importance_features = tier_a if dataset_kind == "actionable" else tier_c_pressure
        for target in TARGETS:
            valid = pd.to_numeric(dataset[target], errors="coerce").notna()
            frame = dataset.loc[valid, [
                "observation_id", "physical_run_id", "instrument_id",
                "chip_lane", "run_start_time", target,
            ]].reset_index(drop=True)
            x = x_all.loc[valid].reset_index(drop=True)
            y = pd.to_numeric(frame[target], errors="coerce")
            if len(frame) < 20 or frame["physical_run_id"].nunique() < 10 or frame["instrument_id"].nunique() < 2:
                model_warnings.append(
                    f"{dataset_kind}/{target}: modeling skipped; n={len(frame)}, "
                    f"physical_runs={frame['physical_run_id'].nunique()}, "
                    f"instruments={frame['instrument_id'].nunique()}"
                )
                continue
            for scheme in ("grouped", "leave_one_instrument_out", "chronological"):
                splits = validation_splits(frame, scheme)
                if not splits:
                    model_warnings.append(f"{dataset_kind}/{target}/{scheme}: no valid splits")
                    continue
                for train_index, test_index, fold_name in splits:
                    train_y, test_y = y.iloc[train_index], y.iloc[test_index]
                    median = float(train_y.median())
                    baseline_prediction = np.repeat(median, len(test_index))
                    for row_index, prediction in zip(test_index, baseline_prediction):
                        prediction_rows.append({
                            "dataset": dataset_kind, "outcome": target,
                            "validation_scheme": scheme, "validation_fold": fold_name,
                            "model": "median_baseline",
                            "observation_id": frame.loc[row_index, "observation_id"],
                            "physical_run_id": frame.loc[row_index, "physical_run_id"],
                            "instrument_id": frame.loc[row_index, "instrument_id"],
                            "chip_lane": frame.loc[row_index, "chip_lane"],
                            "run_start_time": frame.loc[row_index, "run_start_time"],
                            "observed": float(test_y.loc[row_index]),
                            "predicted": float(prediction),
                            "residual": float(test_y.loc[row_index] - prediction),
                        })
                    for model_name in ("elastic_net", "extra_trees"):
                        pipeline, grid = make_model_pipeline(numeric, categorical, model_name)
                        try:
                            with warnings.catch_warnings(record=True) as captured:
                                warnings.simplefilter("always")
                                fitted = fit_nested_model(
                                    pipeline, grid, x.iloc[train_index], train_y,
                                    frame.loc[train_index, "physical_run_id"],
                                    model_warnings,
                                )
                                predictions = fitted.predict(x.iloc[test_index])
                                for warning in captured[:5]:
                                    model_warnings.append(
                                        f"{dataset_kind}/{target}/{scheme}/{fold_name}/{model_name}: "
                                        f"{warning.message}"
                                    )
                        except Exception as exc:
                            model_warnings.append(
                                f"{dataset_kind}/{target}/{scheme}/{fold_name}/{model_name}: {exc}"
                            )
                            continue
                        for row_index, prediction in zip(test_index, predictions):
                            observed = float(y.loc[row_index])
                            prediction_rows.append({
                                "dataset": dataset_kind, "outcome": target,
                                "validation_scheme": scheme, "validation_fold": fold_name,
                                "model": model_name,
                                "observation_id": frame.loc[row_index, "observation_id"],
                                "physical_run_id": frame.loc[row_index, "physical_run_id"],
                                "instrument_id": frame.loc[row_index, "instrument_id"],
                                "chip_lane": frame.loc[row_index, "chip_lane"],
                                "run_start_time": frame.loc[row_index, "run_start_time"],
                                "observed": observed, "predicted": float(prediction),
                                "residual": observed - float(prediction),
                            })
                        if importance_features and (
                            dataset_kind == "actionable" or scheme == "grouped"
                        ):
                            try:
                                importance = heldout_permutation_importance(
                                    fitted, x.iloc[test_index], test_y, importance_features,
                                    permutation_repeats, rng,
                                )
                                for row in importance:
                                    importance_rows.append({
                                        **row, "dataset": dataset_kind, "outcome": target,
                                        "validation_scheme": scheme,
                                        "validation_fold": fold_name, "model": model_name,
                                        "test_n": len(test_index),
                                    })
                            except Exception as exc:
                                model_warnings.append(
                                    f"permutation {dataset_kind}/{target}/{scheme}/{fold_name}/{model_name}: {exc}"
                                )
    predictions = safe_dataframe(prediction_rows, (
        "dataset", "outcome", "validation_scheme", "validation_fold", "model",
        "observation_id", "physical_run_id", "instrument_id", "chip_lane",
        "run_start_time", "observed", "predicted", "residual",
    ))
    importance = safe_dataframe(importance_rows, (
        "feature_name", "heldout_importance", "repeat_std", "dataset", "outcome",
        "validation_scheme", "validation_fold", "model", "test_n",
    ))
    performance = summarize_performance(predictions)
    return performance, predictions, importance, model_warnings


def metric_row(group: pd.DataFrame) -> dict[str, Any]:
    observed = group["observed"].to_numpy(dtype=float)
    predicted = group["predicted"].to_numpy(dtype=float)
    spearman = (
        stats.spearmanr(observed, predicted).statistic
        if len(group) >= 3 and np.unique(observed).size >= 2 and np.unique(predicted).size >= 2
        else np.nan
    )
    return {
        "n_observations": len(group),
        "n_physical_runs": group["physical_run_id"].nunique(),
        "instrument_coverage": group["instrument_id"].nunique(),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": math.sqrt(mean_squared_error(observed, predicted)),
        "r2": r2_score(observed, predicted) if len(group) >= 2 else np.nan,
        "predicted_observed_spearman": spearman,
    }


def summarize_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=(
            "dataset", "outcome", "validation_scheme", "validation_fold", "model",
            "n_observations", "n_physical_runs", "instrument_coverage",
            "mae", "rmse", "r2", "predicted_observed_spearman",
            "baseline_mae", "mae_improvement_over_baseline", "beats_baseline",
        ))
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "outcome", "validation_scheme", "validation_fold", "model"]
    for group_key, group in predictions.groupby(keys, sort=True):
        rows.append(dict(zip(keys, group_key)) | metric_row(group))
    aggregate_keys = ["dataset", "outcome", "validation_scheme", "model"]
    for group_key, group in predictions.groupby(aggregate_keys, sort=True):
        rows.append(dict(zip(aggregate_keys, group_key)) | {"validation_fold": "aggregate"} | metric_row(group))
    performance = pd.DataFrame(rows)
    baseline = (
        performance[performance["model"] == "median_baseline"][
            ["dataset", "outcome", "validation_scheme", "validation_fold", "mae"]
        ].rename(columns={"mae": "baseline_mae"})
    )
    performance = performance.merge(
        baseline,
        on=["dataset", "outcome", "validation_scheme", "validation_fold"],
        how="left",
    )
    performance["mae_improvement_over_baseline"] = performance["baseline_mae"] - performance["mae"]
    performance["beats_baseline"] = performance["mae"] < performance["baseline_mae"]
    return performance.sort_values(
        ["dataset", "outcome", "validation_scheme", "validation_fold", "model"],
        kind="stable",
    )


def consistent_model_evidence(
    performance: pd.DataFrame,
    dataset_kind: str,
    target: str,
    model: str,
) -> bool:
    subset = performance[
        (performance["dataset"] == dataset_kind)
        & (performance["outcome"] == target)
        & (performance["model"] == model)
        & (performance["validation_fold"] == "aggregate")
    ]
    required = {"grouped", "leave_one_instrument_out"}
    if not required.issubset(set(subset["validation_scheme"])):
        return False
    return bool(subset["beats_baseline"].all())


def bootstrap_mean_ci(values: pd.Series, rng: np.random.Generator, iterations: int = 1000) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) < 2:
        return (float(arr[0]), float(arr[0])) if len(arr) else (np.nan, np.nan)
    means = np.mean(rng.choice(arr, size=(iterations, len(arr)), replace=True), axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def build_feature_rankings(
    policy: pd.DataFrame,
    correlations: pd.DataFrame,
    importance: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    importance = ensure_columns(importance, (
        "feature_name", "heldout_importance", "repeat_std", "dataset",
        "outcome", "validation_scheme", "validation_fold", "model", "test_n",
    ))
    candidates = policy[policy["actionability_tier"] == "A"].copy()
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for _, candidate in candidates.iterrows():
            feature = candidate["feature_name"]
            corr = correlations[
                (correlations["outcome"] == target)
                & (correlations["feature_name"] == feature)
            ]
            corr_row = corr.iloc[0] if len(corr) else pd.Series(dtype=object)
            imp = importance[
                (importance["outcome"] == target)
                & (importance["feature_name"] == feature)
                & (importance["validation_scheme"] == "grouped")
            ].copy()
            mean_importance = imp["heldout_importance"].mean() if len(imp) else np.nan
            low, high = bootstrap_mean_ci(imp["heldout_importance"], rng) if len(imp) else (np.nan, np.nan)
            if len(imp):
                imp["rank"] = imp.groupby(["model", "validation_fold"])["heldout_importance"].rank(
                    ascending=False, method="average"
                )
                rank_stability = 1 / (1 + imp["rank"].std(ddof=0))
            else:
                rank_stability = np.nan
            by_model = imp.groupby("model")["heldout_importance"].mean() if len(imp) else pd.Series(dtype=float)
            agreement = bool(
                {"elastic_net", "extra_trees"}.issubset(set(by_model.index))
                and (by_model[["elastic_net", "extra_trees"]] > 0).all()
            )
            models_beating = [
                model for model in ("elastic_net", "extra_trees")
                if consistent_model_evidence(performance, "actionable", target, model)
            ]
            effect = corr_row.get("estimated_y_change_per_x_iqr", np.nan)
            categorical_effect = corr_row.get("categorical_adjusted_difference", np.nan)
            if pd.notna(effect):
                direction = "increase" if effect > 0 else "decrease"
            elif pd.notna(categorical_effect):
                direction = (
                    f"{corr_row.get('comparison_level', '')}_higher_than_"
                    f"{corr_row.get('reference_level', '')}"
                )
            else:
                direction = "unresolved"
            within_rho = corr_row.get("within_instrument_rho_median", np.nan)
            overall_rho = corr_row.get("spearman_rho", np.nan)
            overall_direction = overall_rho if pd.notna(overall_rho) else categorical_effect
            cross_consistent = bool(
                pd.notna(within_rho) and pd.notna(overall_direction)
                and np.sign(within_rho) == np.sign(overall_direction)
                and corr_row.get("within_instrument_count", 0) >= 2
            )
            reason = str(candidate.get("exclusion_reason", ""))
            if not bool(candidate.get("modeling_eligible", False)) or candidate.get("number_unique", 0) < 3:
                label = "insufficient_variation"
            elif candidate.get("tied_context_variable") or "perfectly_tied_to" in reason:
                label = "confounded"
            elif not models_beating:
                label = "no_evidence"
            elif (
                pd.notna(mean_importance) and mean_importance > 0
                and pd.notna(low) and low > 0
                and agreement and cross_consistent
                and corr_row.get("fdr_q_value", 1.0) <= .10
            ):
                label = "evidence_supported"
            elif pd.notna(mean_importance) and mean_importance > 0:
                label = "promising_but_uncertain"
            else:
                label = "no_evidence"
            rows.append({
                "feature_name": feature, "outcome": target, "direction": direction,
                "observed_min": candidate.get("observed_min", ""),
                "observed_max": candidate.get("observed_max", ""),
                "approved_range": "not_available_in_CSA_manual_approval_required",
                "unique_values": candidate.get("number_unique", np.nan),
                "instruments_with_variation": candidate.get("instruments_on_which_varies", np.nan),
                "cross_instrument_range_overlap": candidate.get("cross_instrument_range_overlap", np.nan),
                "heldout_importance_mean_mae_increase": mean_importance,
                "heldout_importance_ci_low": low, "heldout_importance_ci_high": high,
                "rank_stability": rank_stability,
                "spearman_rho": overall_rho,
                "fdr_q_value": corr_row.get("fdr_q_value", np.nan),
                "within_instrument_effect": within_rho,
                "estimated_y_change_per_parameter_iqr": effect,
                "categorical_adjusted_difference": categorical_effect,
                "categorical_reference_level": corr_row.get("reference_level", ""),
                "categorical_comparison_level": corr_row.get("comparison_level", ""),
                "exploratory_effect_ci_low": corr_row.get("bootstrap_ci_low", np.nan),
                "exploratory_effect_ci_high": corr_row.get("bootstrap_ci_high", np.nan),
                "effect_sensitivity_without_outlying_y": corr_row.get(
                    "sensitivity_effect_without_outlying_y", np.nan
                ),
                "cross_instrument_consistency": cross_consistent,
                "confounding_warnings": reason,
                "elastic_net_tree_agreement": agreement,
                "models_consistently_beating_baseline": "|".join(models_beating),
                "evidence_label": label,
            })
    result = safe_dataframe(rows)
    if not result.empty:
        label_order = {
            "evidence_supported": 0, "promising_but_uncertain": 1,
            "confounded": 2, "insufficient_variation": 3, "no_evidence": 4,
        }
        result["_order"] = result["evidence_label"].map(label_order)
        result = result.sort_values(
            ["outcome", "_order", "heldout_importance_mean_mae_increase"],
            ascending=[True, True, False], kind="stable",
        ).drop(columns="_order")
    return result


def add_core_policy_rows(
    policy: pd.DataFrame,
    excluded: pd.DataFrame,
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing = set(policy["feature_name"]) if not policy.empty else set()
    rows: list[dict[str, Any]] = []
    for name in [*TARGETS, SECONDARY_TARGET]:
        if name in existing or name not in dataset:
            continue
        source_file = ""
        source_field = ""
        if f"{name}__source_file" in dataset:
            values = dataset[f"{name}__source_file"].dropna().astype(str)
            source_file = values.iloc[0] if len(values) else ""
        if f"{name}__source_field" in dataset:
            values = dataset[f"{name}__source_field"].dropna().astype(str)
            source_field = values.iloc[0] if len(values) else ""
        rows.append({
            "feature_name": name, "unit": "%" if "accuracy" in name else "bp" if "length" in name else "reads",
            "data_type": "numeric", "source_file": source_file,
            "source_field": source_field,
            "availability_time": "post_run", "missingness": dataset[name].isna().mean(),
            "number_unique": dataset[name].nunique(dropna=True),
            "instruments_observed": dataset.loc[dataset[name].notna(), "instrument_id"].nunique(),
            "instruments_on_which_varies": 0, "cross_instrument_range_overlap": np.nan,
            "actionability_tier": "D", "actionability_label": "forbidden_leakage",
            "modeling_eligible": False, "tied_context_variable": "", "duplicate_of": "",
            "top_value_frequency": dataset[name].value_counts(normalize=True).max() if dataset[name].notna().any() else np.nan,
            "observed_min": pd.to_numeric(dataset[name], errors="coerce").min(),
            "observed_max": pd.to_numeric(dataset[name], errors="coerce").max(),
            "exclusion_reason": "outcome_variable",
        })
    for name in ("observation_id", "physical_run_id", "analysis_id", "archive_id", "archive_sha256", "run_start_time"):
        if name in existing or name not in dataset:
            continue
        rows.append({
            "feature_name": name, "unit": "", "data_type": "categorical",
            "source_file": "derived provenance", "source_field": name,
            "availability_time": "unavailable", "missingness": dataset[name].isna().mean(),
            "number_unique": dataset[name].nunique(dropna=True),
            "instruments_observed": dataset["instrument_id"].nunique(),
            "instruments_on_which_varies": 0, "cross_instrument_range_overlap": np.nan,
            "actionability_tier": "D", "actionability_label": "forbidden_leakage",
            "modeling_eligible": False, "tied_context_variable": "", "duplicate_of": "",
            "top_value_frequency": dataset[name].value_counts(normalize=True).max() if dataset[name].notna().any() else np.nan,
            "observed_min": "", "observed_max": "",
            "exclusion_reason": "unique_identifier_path_hash_or_timestamp",
        })
    if rows:
        core = pd.DataFrame(rows)
        policy = pd.concat([policy, core], ignore_index=True)
        excluded = pd.concat([excluded, core], ignore_index=True)
        dictionary_core = core.rename(columns={
            "actionability_tier": "tier", "number_unique": "unique_values",
        })
        dictionary = pd.concat([dictionary, dictionary_core], ignore_index=True)
    return policy, excluded, dictionary


def normalize_dataset_types(dataset: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    type_map = policy.drop_duplicates("feature_name").set_index("feature_name")["data_type"].to_dict()
    for column in result.columns:
        declared = type_map.get(column)
        if declared == "numeric" or column in (*TARGETS, SECONDARY_TARGET, "run_date_ordinal"):
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: np.nan if value is None or pd.isna(value)
                else str(value)
            ).astype(object)
    return result


def pareto_analysis(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "observation_id", "physical_run_id", "instrument_id", "chip_lane",
        "run_start_time", *TARGETS,
    ]
    frame = dataset[columns].copy()
    complete = frame[list(TARGETS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    frame["pareto_eligible"] = complete
    frame["pareto_optimal"] = False
    values = frame.loc[complete, list(TARGETS)].to_numpy(dtype=float)
    eligible_indices = frame.index[complete].to_list()
    for local_index, global_index in enumerate(eligible_indices):
        dominated = np.any(
            np.all(values >= values[local_index], axis=1)
            & np.any(values > values[local_index], axis=1)
        )
        frame.loc[global_index, "pareto_optimal"] = not bool(dominated)
    return frame


def add_multioutcome_pattern(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        rankings["multi_outcome_pattern"] = []
        return rankings
    result = rankings.copy()
    patterns: dict[str, str] = {}
    for feature, group in result.groupby("feature_name"):
        effects = group.set_index("outcome")["estimated_y_change_per_parameter_iqr"].reindex(TARGETS)
        if effects.isna().any():
            pattern = "insufficient_multi_outcome_evidence"
        elif (effects > 0).all():
            pattern = "improves_all_three_observationally"
        elif (effects >= 0).all() and (effects > 0).any():
            pattern = "improves_one_without_observed_harm"
        elif (effects > 0).any() and (effects < 0).any():
            pattern = "trade_off"
        else:
            pattern = "no_observed_improvement"
        patterns[feature] = pattern
    result["multi_outcome_pattern"] = result["feature_name"].map(patterns)
    return result


def propose_experiments(
    dataset: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "candidate", "status", "primary_outcome", "secondary_outcomes",
        "current_observed_range", "approved_range", "test_low", "test_middle",
        "test_high", "replicates_per_level_per_block", "instrument_blocking",
        "lane_blocking", "variables_to_hold_constant", "expected_effect_per_iqr",
        "expected_effect_confidence_interval", "stopping_criteria",
        "failure_criteria", "interpretation",
    ]
    if rankings.empty:
        return pd.DataFrame([{
            "candidate": "engineering_selected_approved_Tier_A_setting",
            "status": "identifiability_pilot_not_performance_recommendation",
            "primary_outcome": "total_reads",
            "secondary_outcomes": "mean_read_length_bp|raw_read_accuracy_pct",
            "current_observed_range": "constant_or_confounded; see feature_policy.csv",
            "approved_range": "must_be_supplied_by_manufacturer_assay_and_engineering_review",
            "test_low": "approved_low_not_inferable_from_CSA",
            "test_middle": "approved_current_or_mid_not_inferable_from_CSA",
            "test_high": "approved_high_not_inferable_from_CSA",
            "replicates_per_level_per_block": 4,
            "instrument_blocking": "Randomize all three approved levels within each of the four reachable instruments",
            "lane_blocking": "Balance levels across lanes; keep sibling lanes in one physical-run analysis block",
            "variables_to_hold_constant": "assay,panel,chip,reagent_lot,library_batch,sample_input,software",
            "expected_effect_per_iqr": "not_estimable_from_current_CSA",
            "expected_effect_confidence_interval": "not_estimable; pilot creates variation for estimation",
            "stopping_criteria": "Pause for any manufacturer QC excursion or approved-limit breach; review after each complete instrument block",
            "failure_criteria": "Abort if safe low/mid/high levels cannot be approved or two consecutive blocks breach prespecified QC limits",
            "interpretation": "Three-level blocked randomized pilot (48 physical runs if four instruments x three levels x four replicates); preregister analysis and capture raw read accuracy",
        }], columns=columns)
    supported = rankings[rankings["evidence_label"] == "evidence_supported"]
    if supported.empty:
        supported = rankings[rankings["evidence_label"] == "promising_but_uncertain"]
    if supported.empty:
        return pd.DataFrame([{
            "candidate": "engineering_selected_approved_Tier_A_setting",
            "status": "identifiability_pilot_not_performance_recommendation",
            "primary_outcome": "total_reads",
            "secondary_outcomes": "mean_read_length_bp|raw_read_accuracy_pct",
            "current_observed_range": "constant_or_confounded; see feature_policy.csv",
            "approved_range": "must_be_supplied_by_manufacturer_assay_and_engineering_review",
            "test_low": "approved_low_not_inferable_from_CSA",
            "test_middle": "approved_current_or_mid_not_inferable_from_CSA",
            "test_high": "approved_high_not_inferable_from_CSA",
            "replicates_per_level_per_block": 4,
            "instrument_blocking": "Randomize all three approved levels within each of the four reachable instruments",
            "lane_blocking": "Balance levels across lanes; keep sibling lanes in one physical-run analysis block",
            "variables_to_hold_constant": "assay,panel,chip,reagent_lot,library_batch,sample_input,software",
            "expected_effect_per_iqr": "not_estimable_from_current_CSA",
            "expected_effect_confidence_interval": "not_estimable; pilot creates variation for estimation",
            "stopping_criteria": "Pause for any manufacturer QC excursion or approved-limit breach; review after each complete instrument block",
            "failure_criteria": "Abort if safe low/mid/high levels cannot be approved or two consecutive blocks breach prespecified QC limits",
            "interpretation": "Three-level blocked randomized pilot (48 physical runs if four instruments x three levels x four replicates); preregister analysis and capture raw read accuracy",
        }], columns=columns)
    supported = supported.sort_values(
        "heldout_importance_mean_mae_increase", ascending=False
    ).drop_duplicates("feature_name").head(3)
    rows = []
    for _, row in supported.iterrows():
        feature = row["feature_name"]
        values = pd.to_numeric(dataset[feature], errors="coerce").dropna()
        if values.nunique() < 3:
            continue
        low, middle, high = values.quantile([.25, .5, .75])
        rows.append({
            "candidate": feature,
            "status": "hypothesis_requires_engineering_assay_and_domain_approval",
            "primary_outcome": row["outcome"],
            "secondary_outcomes": "|".join(x for x in TARGETS if x != row["outcome"]),
            "current_observed_range": f"{values.min():.8g} to {values.max():.8g}",
            "approved_range": "unknown_from_CSA; intersect proposed levels with manufacturer/assay-approved limits",
            "test_low": f"{low:.8g} (use only if approved)",
            "test_middle": f"{middle:.8g} (use only if approved)",
            "test_high": f"{high:.8g} (use only if approved)",
            "replicates_per_level_per_block": 4,
            "instrument_blocking": "Randomize each approved level within every participating instrument; minimum 3 instruments",
            "lane_blocking": "Balance levels across chip lanes and keep sibling lanes in the same analysis block",
            "variables_to_hold_constant": "assay,panel,chip_type,chip_lot,reagent_lot,library_batch,sample_input,workflow,software,other_Tier_A_settings",
            "expected_effect_per_iqr": row.get("estimated_y_change_per_parameter_iqr", ""),
            "expected_effect_confidence_interval": "observational estimate only; confirm with preregistered mixed-effects analysis",
            "stopping_criteria": "Pause for any manufacturer QC excursion, pressure/temperature limit breach, or prespecified adverse shift >2 SD",
            "failure_criteria": "Abort candidate if two consecutive blocks breach QC/failure limits or approved operating bounds cannot be verified",
            "interpretation": "Three-level blocked randomized validation; no automatic setting changes",
        })
    return pd.DataFrame(rows, columns=columns)


def outlier_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        values = pd.to_numeric(dataset[target], errors="coerce")
        median = values.median()
        mad = np.median(np.abs(values.dropna() - median)) if values.notna().any() else np.nan
        if pd.isna(mad) or mad == 0:
            flags = pd.Series(False, index=dataset.index)
        else:
            robust_z = .6745 * (values - median) / mad
            flags = robust_z.abs() > 3.5
        for _, row in dataset.loc[flags, ["observation_id", "instrument_id", target]].iterrows():
            rows.append({
                "outcome": target, "observation_id": row["observation_id"],
                "instrument_id": row["instrument_id"], "value": row[target],
                "flag": "robust_z_over_3.5_not_automatically_deleted",
            })
    return safe_dataframe(rows, ("outcome", "observation_id", "instrument_id", "value", "flag"))


def make_data_quality_report(
    project: Path,
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    retained: list[dict[str, Any]],
    dataset: pd.DataFrame,
    policy: pd.DataFrame,
    correlations: pd.DataFrame,
    outliers: pd.DataFrame,
    parse_warnings: list[str],
) -> str:
    missing_outcomes = {target: int(dataset[target].isna().sum()) for target in TARGETS}
    state_counts = manifest["state"].value_counts().to_dict() if not manifest.empty else {}
    tier_counts = policy["actionability_tier"].value_counts().to_dict() if not policy.empty else {}
    flag_counts = Counter()
    for reason in policy.get("exclusion_reason", pd.Series(dtype=str)).fillna(""):
        for flag in str(reason).split(";"):
            if flag:
                flag_counts[flag.split(":", 1)[0]] += 1
    lines = [
        "# CSA data quality and identifiability report", "",
        f"Generated: {utc_now()}", "",
        "## Collection and parsing", "",
        f"- Discovered archives: {len(inventory)}",
        f"- Valid local archives eligible for parsing: {int(manifest['validation_result'].eq('valid').sum()) if not manifest.empty else 0}",
        f"- Successfully parsed archives retained before duplicate selection: {sum(1 for x in retained if x.get('member_count', 0) > 0)}",
        f"- Independent primary physical-run/lane observations: {len(dataset)}",
        f"- Transfer states: {json.dumps(state_counts, sort_keys=True)}",
        f"- Parse warnings: {len(parse_warnings)}", "",
        "## Missing outcomes", "",
    ]
    lines.extend(f"- {key}: {value}/{len(dataset)} missing" for key, value in missing_outcomes.items())
    lines.extend([
        "", "## Feature tiers and leakage", "",
        f"- Tier counts: {json.dumps(tier_counts, sort_keys=True)}",
        "- Tier D outcomes, unique identifiers, paths, timestamps, sensitive fields, and direct target derivatives were excluded from modeling.",
        "- Tier B context and Tier C diagnostics are never presented as tuning parameters.", "",
        "## Variation and identifiability flags", "",
    ])
    lines.extend(f"- {key}: {value}" for key, value in flag_counts.most_common())
    lines.extend([
        "", "## Units and impossible values", "",
        f"- Unit inconsistencies flagged: {flag_counts.get('unit_inconsistency', 0)}",
        f"- Impossible values flagged: {flag_counts.get('impossible_value_detected', 0)}",
        "- Pressure units were not converted unless explicitly encoded; raw counts, psi, kPa, mbar, and other units remain separate.", "",
        "## Outliers and sensitivity", "",
        f"- Influential outcome observations flagged but retained: {len(outliers)}",
        "- Correlation results include an effect estimate with and without robustly flagged outcome observations when estimable.",
        "- No outlier was automatically deleted.", "",
        "## Identifiability conclusion", "",
    ])
    eligible = policy[
        (policy["actionability_tier"] == "A") & policy["modeling_eligible"].astype(bool)
    ] if not policy.empty else pd.DataFrame()
    if eligible.empty:
        lines.append("No Tier A parameter has sufficient non-confounded variation for actionable ranking.")
    else:
        lines.append(f"{len(eligible)} Tier A parameters passed pre-model identifiability screening.")
    if correlations.empty:
        lines.append("No non-degenerate Tier A numeric correlations could be estimated.")
    lines.extend(["", "## Parse warnings (first 100)", ""])
    lines.extend(f"- {warning}" for warning in parse_warnings[:100])
    text = "\n".join(lines) + "\n"
    (project / "data_quality_report.md").write_text(text, encoding="utf-8")
    return text


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value)


def html_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "<p class='muted'>No rows available.</p>"
    shown = frame.head(max_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(fmt)
    return shown.to_html(index=False, escape=True, border=0, classes="dataframe")


def svg_histogram(values: pd.Series, title: str, color: str) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return f"<p class='muted'>{html.escape(title)}: insufficient values.</p>"
    counts, edges = np.histogram(clean, bins=min(14, max(5, int(np.sqrt(len(clean))))))
    width, height, margin = 520, 230, 45
    max_count = max(counts.max(), 1)
    bars = []
    inner_w, inner_h = width - 2 * margin, height - 2 * margin
    for i, count in enumerate(counts):
        x = margin + i * inner_w / len(counts)
        bar_w = inner_w / len(counts) - 2
        bar_h = count / max_count * inner_h
        bars.append(f"<rect x='{x:.1f}' y='{height-margin-bar_h:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' fill='{color}' opacity='.85'/>")
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<text x='{margin}' y='22' class='svg-title'>{html.escape(title)}</text>"
        f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='{PALETTE['ink']}'/>"
        + "".join(bars)
        + f"<text x='{margin}' y='{height-12}'>{edges[0]:.4g}</text>"
        + f"<text x='{width-margin}' y='{height-12}' text-anchor='end'>{edges[-1]:.4g}</text></svg>"
    )


def svg_scatter(
    x_values: pd.Series,
    y_values: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
) -> str:
    frame = pd.DataFrame({
        "x": pd.to_numeric(x_values, errors="coerce"),
        "y": pd.to_numeric(y_values, errors="coerce"),
    }).dropna()
    if len(frame) < 2:
        return f"<p class='muted'>{html.escape(title)}: insufficient values.</p>"
    width, height, margin = 520, 300, 55
    x_min, x_max = frame["x"].min(), frame["x"].max()
    y_min, y_max = frame["y"].min(), frame["y"].max()
    x_span, y_span = max(x_max - x_min, 1e-12), max(y_max - y_min, 1e-12)
    points = []
    for x, y in frame.itertuples(index=False):
        px = margin + (x - x_min) / x_span * (width - 2 * margin)
        py = height - margin - (y - y_min) / y_span * (height - 2 * margin)
        points.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.2' fill='{color}' opacity='.65'/>")
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<text x='{margin}' y='22' class='svg-title'>{html.escape(title)}</text>"
        f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='{PALETTE['ink']}'/>"
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='{PALETTE['ink']}'/>"
        + "".join(points)
        + f"<text x='{width/2}' y='{height-10}' text-anchor='middle'>{html.escape(x_label)}</text>"
        + f"<text transform='translate(14 {height/2}) rotate(-90)' text-anchor='middle'>{html.escape(y_label)}</text>"
        + "</svg>"
    )


def svg_performance(performance: pd.DataFrame) -> str:
    frame = performance[
        (performance["dataset"] == "actionable")
        & (performance["validation_fold"] == "aggregate")
        & (performance["validation_scheme"] == "grouped")
    ].copy()
    if frame.empty:
        return "<p class='muted'>Models were not estimable.</p>"
    labels = [f"{row.outcome[:10]} · {row.model}" for row in frame.itertuples()]
    values = frame["mae"].to_numpy(dtype=float)
    colors = [
        PALETTE["grey"] if model == "median_baseline"
        else PALETTE["ocean"] if model == "elastic_net"
        else PALETTE["forest"]
        for model in frame["model"]
    ]
    width, row_h, margin = 720, 28, 170
    height = 40 + row_h * len(values)
    scale = (width - margin - 30) / max(values.max(), 1e-12)
    rows = []
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 25 + i * row_h
        rows.append(
            f"<text x='{margin-8}' y='{y+13}' text-anchor='end'>{html.escape(label)}</text>"
            f"<rect x='{margin}' y='{y}' width='{value*scale:.1f}' height='17' fill='{color}'/>"
            f"<text x='{margin+value*scale+5:.1f}' y='{y+13}'>{value:.4g}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Grouped held-out MAE'>{''.join(rows)}</svg>"


def collection_status(
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    host_status: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    targets = host_status["ssh_target"].tolist() if not host_status.empty else sorted(inventory.get("ssh_target", []))
    for target in targets:
        hostname = str(target).split("@")[-1]
        inv = inventory[inventory["hostname"] == hostname] if not inventory.empty else pd.DataFrame()
        man = manifest[manifest["hostname"] == hostname] if not manifest.empty else pd.DataFrame()
        status_row = host_status[host_status["ssh_target"] == target]
        rows.append({
            "instrument": hostname,
            "status": status_row.iloc[0]["status"] if len(status_row) else "unknown",
            "discovered": len(inv), "bytes": pd.to_numeric(inv.get("size_bytes"), errors="coerce").sum() if len(inv) else 0,
            "transferred": int(man.get("state", pd.Series(dtype=str)).eq("transferred").sum()),
            "already_present": int(man.get("state", pd.Series(dtype=str)).eq("already_present").sum()),
            "valid": int(man.get("validation_result", pd.Series(dtype=str)).eq("valid").sum()),
            "failed": int(man.get("state", pd.Series(dtype=str)).eq("failed").sum()),
            "unrecognized": int(man.get("state", pd.Series(dtype=str)).eq("unrecognized_format").sum()),
            "note": status_row.iloc[0]["note"] if len(status_row) else "",
        })
    return pd.DataFrame(rows)


def model_claim_table(performance: pd.DataFrame) -> pd.DataFrame:
    subset = performance[
        (performance["dataset"] == "actionable")
        & (performance["validation_fold"] == "aggregate")
        & (performance["model"] != "median_baseline")
    ].copy()
    if subset.empty:
        return subset
    return subset[[
        "outcome", "validation_scheme", "model", "n_observations",
        "instrument_coverage", "mae", "baseline_mae",
        "mae_improvement_over_baseline", "r2",
        "predicted_observed_spearman", "beats_baseline",
    ]].sort_values(["outcome", "validation_scheme", "mae"], kind="stable")


def render_html_report(
    project: Path,
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    host_status: pd.DataFrame,
    dataset: pd.DataFrame,
    duplicates: pd.DataFrame,
    policy: pd.DataFrame,
    vacuum: pd.DataFrame,
    pipette: pd.DataFrame,
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
    rankings: pd.DataFrame,
    pareto: pd.DataFrame,
    experiments: pd.DataFrame,
    parse_success: int,
    parse_warnings: list[str],
) -> None:
    collection = collection_status(inventory, manifest, host_status)
    overview = pd.DataFrame([{
        "archives_discovered": len(inventory),
        "archives_validated": int(manifest.get("validation_result", pd.Series(dtype=str)).eq("valid").sum()),
        "archives_successfully_parsed": parse_success,
        "primary_observations": len(dataset),
        "physical_runs": dataset["physical_run_id"].nunique(),
        "instruments": dataset["instrument_id"].nunique(),
        "lanes": dataset["chip_lane"].nunique(),
    }])
    coverage = (
        dataset.groupby(["instrument_id", "chip_lane"], dropna=False)
        .agg(observations=("observation_id", "nunique"), physical_runs=("physical_run_id", "nunique"))
        .reset_index()
    )
    missing = pd.DataFrame({
        "field": [*TARGETS, SECONDARY_TARGET],
        "missing_n": [dataset[x].isna().sum() for x in (*TARGETS, SECONDARY_TARGET)],
        "missing_fraction": [dataset[x].isna().mean() for x in (*TARGETS, SECONDARY_TARGET)],
    })
    tier_summary = (
        policy.groupby(["actionability_tier", "actionability_label"], dropna=False)
        .size().rename("features").reset_index()
    ) if not policy.empty else pd.DataFrame()
    identifiability = policy[policy["actionability_tier"] == "A"][[
        "feature_name", "number_unique", "instruments_on_which_varies",
        "cross_instrument_range_overlap", "modeling_eligible", "exclusion_reason",
    ]].sort_values(["modeling_eligible", "number_unique"], ascending=[False, False]).head(40) if not policy.empty else pd.DataFrame()
    vacuum_summary = (
        vacuum.groupby(["classification", "feature_tier", "setpoint_or_measured"], dropna=False)
        .size().rename("rows").reset_index()
    ) if not vacuum.empty else pd.DataFrame()
    pipette_summary = (
        pipette.groupby(["classification", "feature_tier", "setpoint_or_measured"], dropna=False)
        .size().rename("rows").reset_index()
    ) if not pipette.empty else pd.DataFrame()
    outcome_plots = "".join(
        svg_histogram(dataset[target], target, color)
        for target, color in zip(TARGETS, (PALETTE["ocean"], PALETTE["forest"], PALETTE["amber"]))
    )
    calibration_plots = []
    residual_plots = []
    if not predictions.empty:
        for target in TARGETS:
            candidates = performance[
                (performance["dataset"] == "actionable")
                & (performance["outcome"] == target)
                & (performance["validation_scheme"] == "grouped")
                & (performance["validation_fold"] == "aggregate")
                & (performance["model"] != "median_baseline")
            ].sort_values("mae")
            if candidates.empty:
                continue
            model = candidates.iloc[0]["model"]
            plot = predictions[
                (predictions["dataset"] == "actionable")
                & (predictions["outcome"] == target)
                & (predictions["validation_scheme"] == "grouped")
                & (predictions["model"] == model)
            ]
            calibration_plots.append(svg_scatter(
                plot["observed"], plot["predicted"],
                f"{target}: observed vs predicted ({model})",
                "Observed", "Predicted", PALETTE["ocean"],
            ))
            residual_plots.append(svg_scatter(
                plot["predicted"], plot["residual"],
                f"{target}: held-out residuals ({model})",
                "Predicted", "Observed − predicted", PALETTE["clay"],
            ))
    response_plots = []
    if not rankings.empty:
        top = rankings[
            rankings["evidence_label"].isin(["evidence_supported", "promising_but_uncertain"])
        ].drop_duplicates(["feature_name", "outcome"]).head(6)
        for _, row in top.iterrows():
            if row["feature_name"] in dataset:
                response_plots.append(svg_scatter(
                    dataset[row["feature_name"]], dataset[row["outcome"]],
                    f"Observed-range response: {row['feature_name']} → {row['outcome']}",
                    row["feature_name"], row["outcome"], PALETTE["forest"],
                ))
    pareto_plot = svg_scatter(
        pareto["total_reads"], pareto["mean_read_length_bp"],
        "Observed multi-objective trade-off", "Total reads", "Mean read length (bp)",
        PALETTE["orchid"],
    )
    stable = rankings[
        rankings["evidence_label"].isin(["evidence_supported", "promising_but_uncertain"])
    ][[
        "feature_name", "outcome", "direction",
        "estimated_y_change_per_parameter_iqr", "heldout_importance_mean_mae_increase",
        "heldout_importance_ci_low", "heldout_importance_ci_high", "rank_stability",
        "evidence_label", "multi_outcome_pattern", "confounding_warnings",
    ]].head(50) if not rankings.empty else pd.DataFrame()
    model_claims = model_claim_table(performance)
    supported = rankings[rankings["evidence_label"] == "evidence_supported"] if not rankings.empty else pd.DataFrame()
    conclusion = (
        "No tunable parameter met the prespecified evidence-supported criteria."
        if supported.empty else
        f"{supported['feature_name'].nunique()} tunable parameters met the prespecified evidence-supported criteria."
    )
    css = f"""
    :root {{ --paper:{PALETTE['paper']}; --ink:{PALETTE['ink']}; --ocean:{PALETTE['ocean']};
      --forest:{PALETTE['forest']}; --amber:{PALETTE['amber']}; --mist:{PALETTE['mist']}; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.55 Arial,sans-serif; }}
    main {{ max-width:1180px; margin:auto; padding:34px; }}
    h1,h2,h3 {{ color:#164A50; line-height:1.2; }}
    h1 {{ border-bottom:5px solid var(--forest); padding-bottom:12px; }}
    section {{ background:#fff; padding:22px; margin:18px 0; border-radius:10px; box-shadow:0 2px 9px #24323818; }}
    .callout {{ border-left:6px solid var(--amber); background:#fff8df; padding:14px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:15px; }}
    table {{ border-collapse:collapse; width:100%; font-size:12px; display:block; overflow-x:auto; }}
    th {{ background:#DDEBE6; position:sticky; top:0; }}
    th,td {{ border:1px solid #CAD4D0; padding:6px 8px; text-align:left; vertical-align:top; }}
    tr:nth-child(even) {{ background:#F5F8F6; }}
    svg {{ width:100%; max-width:720px; background:#fff; border:1px solid #DDE5E1; margin:8px 0; }}
    svg text {{ font:11px Arial,sans-serif; fill:{PALETTE['ink']}; }}
    svg .svg-title {{ font-size:14px; font-weight:bold; }}
    .muted {{ color:{PALETTE['grey']}; }} code {{ overflow-wrap:anywhere; }}
    """
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Genexus CSA parameter analysis</title><style>{css}</style></head><body><main>
<h1>Genexus CSA parameter analysis</h1>
<p>Generated {html.escape(utc_now())}. Palette: CVD-safe ocean, forest, amber, and neutral earth tones.</p>
<div class="callout"><strong>Guarded conclusion.</strong> {html.escape(conclusion)}
Associations are observational and non-causal. No instrument setting was changed.</div>

<section><h2>Collection status by instrument</h2>{html_table(collection, 20)}</section>
<section><h2>Dataset and observation overview</h2>{html_table(overview)}
<h3>Instrument and lane coverage</h3>{html_table(coverage, 50)}
<h3>Missing outcomes</h3>{html_table(missing)}</section>
<section><h2>Duplicate and reanalysis handling</h2>
<p>One most-complete, then latest archive was retained per physical run and lane. Sibling lanes and reanalyses share a physical-run CV group.</p>
{html_table(duplicates, 40)}</section>
<section><h2>Parameter variation and identifiability</h2>{html_table(identifiability, 40)}</section>
<section><h2>Feature tiers and leakage review</h2>{html_table(tier_summary, 30)}
<p>Tier A is the only recommendable tier. Tier B adjusts context; Tier C supports diagnostics only; Tier D never enters a model.</p></section>
<section><h2>Vacuum-pressure step analysis</h2>{html_table(vacuum_summary, 30)}
<p>Only explicit commanded setpoints, thresholds, or tolerances can be Tier A. Measured traces remain diagnostic.</p></section>
<section><h2>Pipette-pressure liquid-transfer analysis</h2>{html_table(pipette_summary, 30)}
<p>Pressure units were kept as recorded; no conversion across raw counts, psi, kPa, or mbar was assumed.</p></section>
<section><h2>Outcome distributions</h2><div class="grid">{outcome_plots}</div></section>
<section><h2>Held-out model performance</h2>{svg_performance(performance)}
{html_table(model_claims, 60)}</section>
<section><h2>Residual and calibration plots</h2><div class="grid">{''.join(calibration_plots + residual_plots) or "<p class='muted'>Not estimable.</p>"}</div></section>
<section><h2>Stable feature rankings</h2>{html_table(stable, 50)}</section>
<section><h2>Response plots restricted to observed ranges</h2>
<div class="grid">{''.join(response_plots) or "<p class='muted'>No evidence-supported or promising candidate was eligible for response plotting.</p>"}</div></section>
<section><h2>Multi-objective trade-offs</h2>{pareto_plot}
<h3>Pareto-optimal observed runs</h3>{html_table(pareto[pareto['pareto_optimal']], 40)}</section>
<section><h2>Proposed controlled experiments</h2>{html_table(experiments, 10)}
<p>No value may be used until manufacturer/assay-approved limits and engineering safety review are documented.</p></section>
<section><h2>Limitations and non-causal interpretation</h2>
<ul><li>CSA data are observational; correlations and feature importance do not prove that changing a setting changes an outcome.</li>
<li>Unreachable instruments and missing outcomes limit external validity.</li>
<li>Perfectly confounded, constant, near-constant, unit-inconsistent, and target-derived fields were not ranked as actionable.</li>
<li>Measured vacuum/pipette pressure, retries, calibration, and final QC are diagnostic, not pre-run tuning recommendations.</li>
<li>Models are interpreted only when they beat the median baseline on grouped unseen runs and leave-one-instrument-out validation.</li>
<li>Parse warnings recorded: {len(parse_warnings)}. See data_quality_report.md and analysis_run_log.json.</li></ul></section>
</main></body></html>"""
    (project / "csa_ml_report.html").write_text(document, encoding="utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cached(
    inventory_row: dict[str, Any],
    manifest_row: dict[str, Any],
    cache_dir: Path,
    refresh: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{inventory_row['archive_id']}.joblib"
    key = f"{POLICY_VERSION}|{manifest_row.get('sha256', '')}"
    if cache_path.exists() and not refresh:
        try:
            cached = joblib.load(cache_path)
            if cached.get("cache_key") == key:
                return cached["result"]
        except Exception:
            pass
    result = parse_archive(inventory_row, manifest_row)
    joblib.dump({"cache_key": key, "result": result}, cache_path, compress=3)
    return result


PRESSURE_COLUMNS = (
    "observation_id", "physical_run_id", "instrument_id", "chip_lane",
    "workflow_name", "step_number", "step_name_raw", "step_name_normalized",
    "pipette", "attempt", "operation", "source", "destination", "liquid_class",
    "programmed_volume", "feature_name", "value", "unit",
    "setpoint_or_measured", "feature_tier", "classification", "source_file",
    "source_field", "start_time", "end_time", "parse_confidence",
    "exclusion_reason",
)


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.Series(dtype=object)
    return result[list(columns) + [x for x in result.columns if x not in columns]]


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frame.to_csv(path, index=False)


def code_versions(script_dir: Path) -> dict[str, Any]:
    result = {}
    for name in ("collect_csa.py", "analyze_csa.py", "requirements.txt"):
        path = script_dir / name
        if path.exists():
            result[name] = sha256_path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("genexus_csa_project"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--max-model-features", type=int, default=200)
    parser.add_argument("--refresh-parse-cache", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--exclude-from-observations", nargs="+", default=[],
                        help="Parse but quarantine specified archive IDs from primary run/lane observations.")
    args = parser.parse_args()
    started = time.time()
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    run_log_path = project / "analysis_run_log.json"
    run_log: dict[str, Any] = {
        "status": "running", "started_utc": utc_now(), "random_seed": SEED,
        "feature_policy_version": POLICY_VERSION,
        "model_settings": {
            "baseline": "training-fold median",
            "models": ["ElasticNet", "ExtraTreesRegressor"],
            "validation": ["grouped physical_run_id", "leave-one-instrument-out", "chronological newest 20%"],
            "nested_inner_cv": "GroupKFold up to 3 folds",
            "max_model_features_unsupervised": args.max_model_features,
            "permutation_repeats": args.permutation_repeats,
            "bootstrap_iterations": args.bootstrap_iterations,
            "elastic_net_grid": {
                "feature_percentile": [50, 100], "alpha": [0.1, 1.0],
                "l1_ratio": [0.5],
            },
            "extra_trees_grid": {
                "n_estimators": 100, "feature_percentile": [50, 100],
                "max_depth": [3, None], "min_samples_leaf": [2],
                "max_features": ["sqrt"],
            },
        },
        "warnings": [],
    }
    try:
        inventory = pd.read_csv(project / "csa_inventory.csv", dtype=str, keep_default_na=False)
        manifest = pd.read_csv(project / "csa_transfer_manifest.csv", dtype=str, keep_default_na=False)
        host_status = (
            pd.read_csv(project / "host_status.csv", dtype=str, keep_default_na=False)
            if (project / "host_status.csv").exists() else pd.DataFrame()
        )
        inventory["size_bytes"] = pd.to_numeric(inventory["size_bytes"], errors="coerce")
        inventory_by_id = inventory.set_index("archive_id", drop=False).to_dict("index")
        manifest_by_id = manifest.set_index("archive_id", drop=False).to_dict("index")
        eligible = manifest[
            manifest["state"].isin(["transferred", "already_present"])
            & manifest["validation_result"].eq("valid")
        ]
        input_pairs = [
            (inventory_by_id[row.archive_id], manifest_by_id[row.archive_id])
            for row in eligible.itertuples()
            if row.archive_id in inventory_by_id
        ]
        run_log["input_archive_hashes"] = {
            row["archive_id"]: row.get("sha256", "") for row in eligible.to_dict("records")
        }
        print(f"Parsing {len(input_pairs)} validated CSA archives with {args.jobs} workers", flush=True)
        parsed = joblib.Parallel(n_jobs=max(1, args.jobs), prefer="threads", verbose=10)(
            joblib.delayed(parse_cached)(
                inv, man, project / ".parse_cache", args.refresh_parse_cache
            ) for inv, man in input_pairs
        )
        parse_success_items = [x for x in parsed if x.get("member_count", 0) > 0]
        parse_failures = [
            {
                "archive_id": x["observation"]["archive_id"],
                "warnings": "|".join(x.get("parse_warnings", [])),
            }
            for x in parsed if x.get("member_count", 0) == 0
        ]
        parse_warnings = [
            f"{item['observation']['archive_id']}: {warning}"
            for item in parsed for warning in item.get("parse_warnings", [])
        ]
        retained, duplicates = select_primary_archives(parse_success_items, inventory_by_id)
        quarantined_ids = set(args.exclude_from_observations)
        if quarantined_ids:
            if not quarantined_ids.issubset(inventory_by_id):
                raise ValueError("Observation quarantine contains unknown archive IDs")
            retained = [item for item in retained if item["observation"]["archive_id"] not in quarantined_ids]
            mask = duplicates["archive_id"].isin(quarantined_ids)
            duplicates.loc[mask, "retained_primary"] = False
            duplicates.loc[mask, "selection_reason"] = "explicit_observation_quarantine_not_independent_run_lane"
            run_log["observation_quarantine_archive_ids"] = sorted(quarantined_ids)
            parse_warnings.extend(f"{archive_id}: parsed but excluded from independent run/lane observations" for archive_id in sorted(quarantined_ids))
        dataset, features, vacuum, pipette, provenance = build_wide_dataset(retained)
        dataset = dataset.drop(columns=["hostname_for_report", "raw_experiment_name"], errors="ignore")
        policy, excluded, dictionary = build_feature_policy(dataset, features)
        policy, excluded, dictionary = add_core_policy_rows(policy, excluded, dictionary, dataset)
        dataset = normalize_dataset_types(dataset, policy)
        correlations = correlation_analysis(dataset, policy, args.bootstrap_iterations)
        outliers = outlier_summary(dataset)
        make_data_quality_report(
            project, inventory, manifest, retained, dataset, policy,
            correlations, outliers, parse_warnings,
        )
        if args.skip_models:
            performance = summarize_performance(pd.DataFrame())
            predictions = safe_dataframe([], (
                "dataset", "outcome", "validation_scheme", "validation_fold", "model",
                "observation_id", "physical_run_id", "instrument_id", "chip_lane",
                "run_start_time", "observed", "predicted", "residual",
            ))
            importance = safe_dataframe([], (
                "feature_name", "heldout_importance", "repeat_std", "dataset",
                "outcome", "validation_scheme", "validation_fold", "model", "test_n",
            ))
            model_warnings = ["models skipped by command-line request"]
        else:
            print("Running nested grouped, instrument, and chronological validation", flush=True)
            performance, predictions, importance, model_warnings = run_models(
                dataset, policy, args.max_model_features, args.permutation_repeats
            )
        rankings = add_multioutcome_pattern(
            build_feature_rankings(policy, correlations, importance, performance)
        )
        pareto = pareto_analysis(dataset)
        experiments = propose_experiments(dataset, rankings)
        vacuum = ensure_columns(vacuum, PRESSURE_COLUMNS)
        pipette = ensure_columns(pipette, PRESSURE_COLUMNS)
        write_frame(dataset, project / "csa_ml_dataset.csv")
        dataset.to_parquet(project / "csa_ml_dataset.parquet", index=False)
        write_frame(dictionary, project / "data_dictionary.csv")
        write_frame(policy, project / "feature_policy.csv")
        write_frame(excluded, project / "excluded_features.csv")
        write_frame(duplicates, project / "duplicate_archives.csv")
        write_frame(vacuum, project / "templating_vacuum_step_features.csv")
        write_frame(pipette, project / "liquid_transfer_pipette_pressure_features.csv")
        write_frame(correlations, project / "correlation_results.csv")
        write_frame(performance, project / "model_performance.csv")
        write_frame(predictions, project / "heldout_predictions.csv")
        write_frame(rankings, project / "feature_rankings.csv")
        write_frame(importance, project / "heldout_permutation_importance.csv")
        write_frame(pareto, project / "pareto_runs.csv")
        write_frame(experiments, project / "proposed_experiments.csv")
        write_frame(provenance, project / "outcome_provenance.csv")
        write_frame(outliers, project / "outlier_flags.csv")
        write_frame(pd.DataFrame(parse_failures), project / "parse_failures.csv")
        render_html_report(
            project, inventory, manifest, host_status, dataset, duplicates,
            policy, vacuum, pipette, performance, predictions, rankings, pareto,
            experiments, len(parse_success_items), parse_warnings,
        )
        script_dir = Path(__file__).resolve().parent
        run_log.update({
            "status": "complete", "completed_utc": utc_now(),
            "execution_seconds": time.time() - started,
            "code_version": code_versions(script_dir),
            "counts": {
                "archives_discovered": len(inventory),
                "archives_validated": int(manifest["validation_result"].eq("valid").sum()),
                "archives_successfully_parsed": len(parse_success_items),
                "parse_failures": len(parse_failures),
                "independent_observations": len(dataset),
                "physical_runs": dataset["physical_run_id"].nunique(),
                "instruments_analyzed": dataset["instrument_id"].nunique(),
                "tier_a_features": int(policy["actionability_tier"].eq("A").sum()),
                "eligible_tier_a_features": int(
                    (policy["actionability_tier"].eq("A") & policy["modeling_eligible"].astype(bool)).sum()
                ),
            },
            "warnings": parse_warnings + model_warnings,
        })
        run_log_path.write_text(json.dumps(run_log, indent=2, default=str), encoding="utf-8")
        print(json.dumps(run_log["counts"], indent=2), flush=True)
        return 0
    except Exception as exc:
        run_log.update({
            "status": "failed", "completed_utc": utc_now(),
            "execution_seconds": time.time() - started,
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        run_log_path.write_text(json.dumps(run_log, indent=2, default=str), encoding="utf-8")
        raise


if __name__ == "__main__":
    sys.exit(main())
