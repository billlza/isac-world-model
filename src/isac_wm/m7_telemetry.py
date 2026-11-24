"""Read Mooring's scalar M7 history without inventing RF or arrival times.

This is a local, offline data adapter. It neither contacts the router nor feeds
``Observation.radio`` or the calibrated complex-RF occupancy model. The CSV's
timestamp is the host poll start; RF capture and response availability are
unknown. The pinned source below defines the export contract, not the version
that necessarily collected a user's file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA_VERSION = "m7-link-telemetry-v1"
SCHEMA_REFERENCE_COMMIT = "0d6361d3bca8ea76a08cd8ef1cac6606dd730958"
SCHEMA_REFERENCE_URL = (
    f"https://github.com/billlza/mooring/blob/{SCHEMA_REFERENCE_COMMIT}/crates/daemon-core/src/radio_table.rs"
)
SAMPLES_HEADER = (
    "at_unix,at_utc,outcome,failure,uptime_secs,temperature_c,thermal,service,"
    "network_service,data_service,band,ca_secondary_count,"
    "lte_rsrp_dbm,lte_rsrq_db,lte_sinr_db,lte_rssi_dbm,"
    "nr_rsrp_dbm,nr_rsrq_db,nr_sinr_db,plmn,session_secs"
)
COLUMNS = tuple(SAMPLES_HEADER.split(","))
RF_FIELDS = COLUMNS[12:19]
CONTEXT_FIELDS = COLUMNS[4:12] + COLUMNS[19:]
FAILURES = frozenset(
    (
        "not_configured",
        "unreachable",
        "auth_failed",
        "identity_mismatch",
        "unsupported",
        "malformed",
        "local_failure",
    )
)
# These are the pinned Mooring router.rs value-type bounds, not new claims
# about a device's physical limits. RF values are represented in tenths.
RF_BOUNDS_TENTHS = {
    "lte_rsrp_dbm": (-1560, -310),
    "lte_rsrq_db": (-430, 200),
    "lte_sinr_db": (-230, 400),
    "lte_rssi_dbm": (-1500, 0),
    "nr_rsrp_dbm": (-1560, -310),
    "nr_rsrq_db": (-430, 200),
    "nr_sinr_db": (-230, 400),
}
INTEGER_BOUNDS = {
    "at_unix": (0, 2**64 - 1),
    "uptime_secs": (0, 2**64 - 1),
    "temperature_c": (-40, 125),
    "ca_secondary_count": (0, 255),
    "session_secs": (0, 2**64 - 1),
}
NOMINAL_POLL_INTERVAL_S = 60
_UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_INT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_READING = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]\Z")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


def _integer(value: str, name: str, *, signed: bool = False) -> int:
    if not (_INT if signed else _UINT).fullmatch(value):
        raise ValueError(f"{name} must be a canonical whole number")
    number = int(value)
    lower, upper = INTEGER_BOUNDS[name]
    if not lower <= number <= upper:
        raise ValueError(f"{name} must be within the source contract's [{lower}, {upper}] range")
    return number


def _reading(value: str, name: str) -> float | None:
    if not value:
        return None
    if not _READING.fullmatch(value):
        raise ValueError(f"{name} must be a finite reading with one decimal or empty")
    # Parse the exact tenths scale before checking the named source value type.
    tenths = int(value.replace(".", ""))
    lower, upper = RF_BOUNDS_TENTHS[name]
    if not lower <= tenths <= upper:
        raise ValueError(
            f"{name} must be within the source contract's [{lower / 10:.1f}, {upper / 10:.1f}] range"
        )
    return tenths / 10.0


def _record(cells: list[str], source_row: int) -> dict[str, Any]:
    if len(cells) != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, got {len(cells)}")
    if any(cell != cell.strip() or '"' in cell for cell in cells):
        raise ValueError("whitespace or quoting is not part of this CSV export contract")
    row = dict(zip(COLUMNS, cells, strict=True))
    poll_start = _integer(row["at_unix"], "at_unix")
    if not _UTC.fullmatch(row["at_utc"]):
        raise ValueError("at_utc must have the exported UTC second-resolution form")
    try:
        utc = datetime.strptime(row["at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid at_utc calendar time") from exc
    if int(utc.timestamp()) != poll_start:
        raise ValueError("at_unix and at_utc disagree")

    if row["outcome"] == "failed":
        if row["failure"] not in FAILURES:
            raise ValueError("failed poll requires a known failure reason")
        if any(cells[4:]):
            raise ValueError("failed poll cannot carry a successful reading")
    elif row["outcome"] == "read":
        if row["failure"]:
            raise ValueError("successful read cannot carry a poll failure")
        if row["thermal"] != "normal" or row["service"] not in {"attached", "no_service"}:
            raise ValueError("unknown thermal or service state")
        for field in ("uptime_secs", "temperature_c"):
            _integer(row[field], field, signed=field == "temperature_c")
        if row["service"] == "no_service":
            if any(cells[8:]):
                raise ValueError("no_service cannot carry attachment fields or RF readings")
        else:
            if row["network_service"] not in {"nr5g", "lte"}:
                raise ValueError("unknown network_service")
            if row["data_service"] not in {"nr_sub6", "lte", "lte_plus"}:
                raise ValueError("unknown data_service")
            if not re.fullmatch(r"[Bn][1-9][0-9]{0,2}", row["band"]):
                raise ValueError("band must use B or n followed by a number within 1..999")
            if not re.fullmatch(r"[0-9]{3}-[0-9]{2,3}", row["plmn"]):
                raise ValueError("invalid PLMN network-code form")
            for field in ("ca_secondary_count", "session_secs"):
                _integer(row[field], field)
    else:
        raise ValueError("outcome must be read or failed")

    rf = {field: _reading(row[field], field) for field in RF_FIELDS}
    context: dict[str, Any] = {field: row[field] or None for field in CONTEXT_FIELDS}
    for field in ("uptime_secs", "temperature_c", "ca_secondary_count", "session_secs"):
        if context[field] is not None:
            context[field] = _integer(row[field], field, signed=field == "temperature_c")
    return {
        "source_row": source_row,
        "host_poll_start_unix_s": poll_start,
        "host_poll_start_utc": row["at_utc"],
        "rf_capture_unix_s": None,
        "available_unix_s": None,
        "outcome": row["outcome"],
        "failure": row["failure"] or None,
        "rf_metrics": rf,
        "rf_missing": {field: value is None for field, value in rf.items()},
        "context": context,
    }


def _quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [
        later["host_poll_start_unix_s"] - earlier["host_poll_start_unix_s"]
        for earlier, later in zip(records, records[1:])
    ]
    return {
        "records": len(records),
        "poll_outcomes": dict(Counter(row["outcome"] for row in records)),
        "poll_failure_reasons": dict(Counter(row["failure"] for row in records if row["failure"])),
        "read_without_service": sum(row["context"]["service"] == "no_service" for row in records),
        "attached_reads": sum(row["context"]["service"] == "attached" for row in records),
        "records_with_any_rf_metric": sum(
            any(v is not None for v in row["rf_metrics"].values()) for row in records
        ),
        "rf_missing_counts": {field: sum(row["rf_missing"][field] for row in records) for field in RF_FIELDS},
        "poll_start_intervals_s": {
            "count": len(intervals),
            "minimum": min(intervals) if intervals else None,
            "median": median(intervals) if intervals else None,
            "maximum": max(intervals) if intervals else None,
        },
        "gaps_over_2x_nominal": [
            {
                "previous_source_row": records[i]["source_row"],
                "source_row": records[i + 1]["source_row"],
                "interval_s": interval,
            }
            for i, interval in enumerate(intervals)
            if interval > 2 * NOMINAL_POLL_INTERVAL_S
        ],
        "gap_interpretation": "observed poll-start spacing; not a packet-loss estimate",
        "rows_reordered_or_dropped": 0,
    }


def read_mooring_samples_csv(
    path: str | Path,
    *,
    data_origin: str = "unknown",
    collector_commit: str | None = None,
    firmware: str | None = None,
) -> dict[str, Any]:
    """Validate the entire export and return records, manifest and quality.

    ``data_origin``, ``collector_commit`` and ``firmware`` are declarations by
    the caller, not properties authenticated by the CSV. A header-only export
    is valid and reports ``empty_export``. Invalid rows fail the entire import.
    No timestamp is reinterpreted as acquisition or availability.
    """
    if data_origin not in {"real_device", "synthetic", "unknown"}:
        raise ValueError("data_origin must be real_device, synthetic or unknown")
    if collector_commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", collector_commit):
        raise ValueError("collector_commit must be a full 40-character Git commit or omitted")
    if firmware is not None and (
        not firmware.strip() or len(firmware) > 128 or any(ord(c) < 32 for c in firmware)
    ):
        raise ValueError("firmware must be a nonempty single-line version of at most 128 characters")
    path = Path(path)
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("CSV must end with a newline; refusing an empty or possibly truncated export")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8") from exc
    # Permit CRLF exports without hiding other carriage returns or blank rows.
    lines = text.replace("\r\n", "\n").split("\n")[:-1]
    if not lines or lines[0] != SAMPLES_HEADER:
        raise ValueError("unknown samples header; export mooringctl router history --csv samples")
    records: list[dict[str, Any]] = []
    for source_row, line in enumerate(lines[1:], 2):
        try:
            record = _record(line.split(","), source_row)
            if records and record["host_poll_start_unix_s"] <= records[-1]["host_poll_start_unix_s"]:
                raise ValueError("poll-start timestamps must strictly increase; duplicate/out-of-order row")
        except ValueError as exc:
            raise ValueError(f"CSV row {source_row}: {exc}") from exc
        records.append(record)
    quality = _quality(records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "imported_offline_telemetry" if records else "empty_export",
        "device_profile": "NETGEAR MR7450-100EUS / Mooring samples CSV",
        "source": {
            "filename": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "format": "mooring_radio_samples_csv",
            "schema_reference_commit": SCHEMA_REFERENCE_COMMIT,
            "schema_reference_url": SCHEMA_REFERENCE_URL,
            "declared_collector_commit": collector_commit,
            "declared_firmware": firmware,
            "declared_data_origin": data_origin,
            "declarations_verified_by_csv": False,
            "upstream_history_completeness": "unknown; CSV omits incomplete_days",
        },
        "timing": {
            "timestamp_semantics": "host_poll_start",
            "nominal_collector_poll_interval_s": NOMINAL_POLL_INTERVAL_S,
            "rf_capture_time_available": False,
            "response_availability_time_available": False,
            "timestamp_unit": "seconds since Unix epoch, UTC",
        },
        "capabilities": {
            "scope": "verified CSV export interface, not a hardware impossibility claim",
            "offline_scalar_link_telemetry": True,
            "online_causal_replay": False,
            "calibrated_3d_rf_input": False,
            "complex_csi": False,
            "raw_iq": False,
        },
        "rf_fields": list(RF_FIELDS),
        "rf_missing_mask": "true means not reported; observed zero is not missing",
        "poll_failure_semantics": "failed to read the router; not evidence of cellular outage",
        "records": len(records),
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    return {"records": records, "manifest": manifest, "quality": quality}


def write_telemetry_import(result: dict[str, Any], output: str | Path) -> Path:
    """Write a new local directory only; never replace previous results.

    Serialization is completed before reserving the directory. The manifest is
    written last. A write failure removes only files created by this invocation.
    """
    payloads = {
        "telemetry.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result["records"]
        ),
        "quality.json": json.dumps(result["quality"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "manifest.json": json.dumps(result["manifest"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    try:
        for name, payload in payloads.items():
            target = output / name
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                written.append(target)
                handle.write(payload)
    except Exception:
        for target in written:
            target.unlink(missing_ok=True)
        try:
            output.rmdir()
        except OSError:
            pass
        raise
    return output
