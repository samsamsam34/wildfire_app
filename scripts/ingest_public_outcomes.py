#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_public_structure_damage import normalize_public_damage_rows

SCHEMA_VERSION = "1.0.0"
SUPPORTED_INPUT_SUFFIXES = {".csv", ".json", ".geojson"}
DEFAULT_SOURCE_MANIFEST_PATH = Path("config/public_outcome_sources.json")


@dataclass(frozen=True)
class OutcomeSourceSpec:
    path: Path
    source_name: str | None = None
    default_state: str = "CA"


def _timestamp_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _deterministic_generated_at(run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    return datetime.now(tz=timezone.utc).isoformat()


def _classify_match_confidence(label_confidence: float | None) -> str:
    if label_confidence is None:
        return "unknown"
    if label_confidence >= 0.8:
        return "high"
    if label_confidence >= 0.5:
        return "moderate"
    return "low"


def _damage_severity_class(damage_label: str) -> str:
    mapping = {
        "no_damage": "none",
        "minor_damage": "minor",
        "major_damage": "major",
        "destroyed": "destroyed",
    }
    return mapping.get(str(damage_label or "").strip().lower(), "unknown")


def _canonical_adverse_binary(damage_label: str, structure_loss_or_major_damage: int | None) -> bool | None:
    if structure_loss_or_major_damage in (0, 1):
        return bool(structure_loss_or_major_damage)
    severity = _damage_severity_class(damage_label)
    if severity in {"major", "destroyed"}:
        return True
    if severity in {"none", "minor"}:
        return False
    return None


def _source_status(spec: OutcomeSourceSpec, status: str, reason: str | None = None) -> dict[str, Any]:
    payload = {
        "source_name": spec.source_name or spec.path.stem,
        "source_path": str(spec.path),
        "status": status,
    }
    if reason:
        payload["reason"] = reason
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_name_key(row: dict[str, Any]) -> str:
    event_id = str(row.get("event_id") or "").strip()
    event_name = str(row.get("event_name") or "").strip()
    event_year = str(row.get("event_year") or "").strip()
    if event_id:
        return event_id
    if event_name:
        return f"{event_name}|{event_year or 'unknown'}"
    return "unknown_event"


def _record_quality_score(row: dict[str, Any]) -> tuple[int, float, int]:
    has_binary = 1 if row.get("structure_loss_or_major_damage") in (0, 1) else 0
    label_confidence = _safe_float(row.get("label_confidence")) or 0.0
    has_address = 1 if str(row.get("address_text") or "").strip() else 0
    return (has_binary, label_confidence, has_address)


def _normalize_key_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_key(row: dict[str, Any]) -> str:
    source_name = str(row.get("source_name") or "").strip()
    source_record_id = str(row.get("source_record_id") or "").strip()
    record_id = str(row.get("record_id") or "").strip()
    event_key = _event_name_key(row)
    damage_label = str(row.get("damage_label") or "").strip().lower()
    address_norm = _normalize_key_text(row.get("address_text"))
    lat = _safe_float(row.get("latitude"))
    lon = _safe_float(row.get("longitude"))
    coord = f"{round(lat, 6):.6f},{round(lon, 6):.6f}" if lat is not None and lon is not None else ""
    if event_key != "unknown_event" and coord and address_norm:
        return f"event_coord_addr:{event_key}|{coord}|{address_norm}"
    if event_key != "unknown_event" and coord and damage_label:
        return f"event_coord_label:{event_key}|{coord}|{damage_label}"
    if source_name and source_record_id:
        return f"src:{source_name}|{source_record_id}"
    if source_name and record_id:
        return f"record:{source_name}|{record_id}"
    if event_key != "unknown_event" and source_record_id:
        return f"event_src:{event_key}|{source_record_id}"
    if event_key != "unknown_event" and record_id:
        return f"event_record:{event_key}|{record_id}"
    return f"fallback:{source_name}|{record_id}|{coord}"


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    label = str(raw.get("damage_label") or "unknown")
    source_binary = raw.get("structure_loss_or_major_damage")
    structure_loss_binary = int(source_binary) if source_binary in (0, 1) else None
    adverse_binary = _canonical_adverse_binary(label, structure_loss_binary)
    label_confidence = _safe_float(raw.get("label_confidence"))
    match_confidence = _classify_match_confidence(label_confidence)
    severity = _damage_severity_class(label)
    lat = _safe_float(raw.get("latitude"))
    lon = _safe_float(raw.get("longitude"))
    return {
        "record_id": raw.get("record_id"),
        "source_record_id": raw.get("source_record_id"),
        "source_name": raw.get("source_name"),
        "source_path": raw.get("source_path"),
        "event_id": raw.get("event_id"),
        "event_name": raw.get("event_name"),
        "event_date": raw.get("event_date"),
        "event_year": raw.get("event_year"),
        "latitude": lat,
        "longitude": lon,
        "address_text": raw.get("address_text"),
        "locality": raw.get("locality"),
        "state": raw.get("state"),
        "postal_code": raw.get("postal_code"),
        "parcel_identifier": None,
        "damage_label": label,
        "damage_severity_class": severity,
        "structure_loss_or_major_damage": structure_loss_binary,
        "adverse_outcome_binary": adverse_binary,
        "adverse_outcome_label": (
            "yes" if adverse_binary is True else ("no" if adverse_binary is False else "unknown")
        ),
        "source_native_label": raw.get("raw_damage_label"),
        "label_confidence": label_confidence,
        "geospatial_confidence": "high" if lat is not None and lon is not None else "unknown",
        "match_confidence": match_confidence,
        "provenance_notes": list(raw.get("source_quality_flags") or []),
    }


def _summarize_records(
    *,
    raw_record_count: int,
    deduped_records: list[dict[str, Any]],
    deduplicated_count: int,
    configured_source_count: int,
    ignored_duplicate_source_paths: list[str],
    included_sources: list[dict[str, Any]],
    excluded_sources: list[dict[str, Any]],
    dropped_invalid_coordinate_count: int,
) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_event: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_adverse: dict[str, int] = {}
    match_confidence_dist: dict[str, int] = {}
    missing_address_count = 0
    unknown_label_count = 0

    for row in deduped_records:
        source_name = str(row.get("source_name") or "unknown_source")
        by_source[source_name] = by_source.get(source_name, 0) + 1
        event_key = _event_name_key(row)
        by_event[event_key] = by_event.get(event_key, 0) + 1
        damage_label = str(row.get("damage_label") or "unknown")
        by_label[damage_label] = by_label.get(damage_label, 0) + 1
        severity = str(row.get("damage_severity_class") or "unknown")
        by_severity[severity] = by_severity.get(severity, 0) + 1
        adverse_label = str(row.get("adverse_outcome_label") or "unknown")
        by_adverse[adverse_label] = by_adverse.get(adverse_label, 0) + 1
        confidence_bucket = str(row.get("match_confidence") or "unknown")
        match_confidence_dist[confidence_bucket] = match_confidence_dist.get(confidence_bucket, 0) + 1
        if not str(row.get("address_text") or "").strip():
            missing_address_count += 1
        if severity == "unknown":
            unknown_label_count += 1

    final_count = len(deduped_records)
    event_count = len(by_event)
    return {
        "configured_source_count": int(configured_source_count),
        "included_source_count": len(included_sources),
        "excluded_source_count": len(excluded_sources),
        "ignored_duplicate_source_count": len(ignored_duplicate_source_paths),
        "ignored_duplicate_source_paths": sorted(ignored_duplicate_source_paths),
        "raw_record_count": raw_record_count,
        "final_record_count": final_count,
        "total_dataset_size": final_count,
        "event_count": event_count,
        "deduplicated_record_count": deduplicated_count,
        "deduplication_rate": round(deduplicated_count / float(raw_record_count), 4) if raw_record_count else 0.0,
        "dropped_invalid_coordinate_count": dropped_invalid_coordinate_count,
        "missing_address_count": missing_address_count,
        "missing_address_rate": round(missing_address_count / float(final_count), 4) if final_count else 0.0,
        "unknown_label_count": unknown_label_count,
        "unknown_label_rate": round(unknown_label_count / float(final_count), 4) if final_count else 0.0,
        "match_confidence_distribution": dict(sorted(match_confidence_dist.items())),
        "count_by_source": dict(sorted(by_source.items())),
        "count_by_event": dict(sorted(by_event.items())),
        "records_per_event": dict(sorted(by_event.items())),
        "count_by_damage_label": dict(sorted(by_label.items())),
        "count_by_damage_severity_class": dict(sorted(by_severity.items())),
        "count_by_adverse_outcome_label": dict(sorted(by_adverse.items())),
        "included_sources": included_sources,
        "excluded_sources": excluded_sources,
    }


def _build_markdown_report(
    *,
    run_id: str,
    generated_at: str,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Public Outcomes Ingestion Report",
        "",
        "- This artifact normalizes public observed structure-damage outcomes for directional evaluation and calibration.",
        "- It is not insurer claims truth and not a carrier-grade validation artifact by itself.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Final normalized records: `{summary.get('final_record_count')}`",
        f"- Total dataset size: `{summary.get('total_dataset_size')}`",
        f"- Event count: `{summary.get('event_count')}`",
        f"- Raw records before dedupe: `{summary.get('raw_record_count')}`",
        f"- Deduplicated records removed: `{summary.get('deduplicated_record_count')}`",
        f"- Unknown label rate: `{summary.get('unknown_label_rate')}`",
        f"- Missing address rate: `{summary.get('missing_address_rate')}`",
        f"- Configured sources: `{summary.get('configured_source_count')}`",
        f"- Included sources: `{summary.get('included_source_count')}`",
        f"- Excluded sources: `{summary.get('excluded_source_count')}`",
        f"- Ignored duplicate paths: `{summary.get('ignored_duplicate_source_count')}`",
        "",
        "## Included Sources",
    ]
    included = summary.get("included_sources") if isinstance(summary.get("included_sources"), list) else []
    if included:
        for row in included:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('source_name')}`: status={row.get('status')}, "
                f"records={row.get('record_count')}, dropped_invalid_coordinates={row.get('dropped_invalid_coordinate_count')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Excluded Sources")
    excluded = summary.get("excluded_sources") if isinstance(summary.get("excluded_sources"), list) else []
    if excluded:
        for row in excluded:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('source_name')}`: status={row.get('status')}, reason={row.get('reason')}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Outcome Distribution",
            f"- Damage label counts: `{summary.get('count_by_damage_label')}`",
            f"- Severity class counts: `{summary.get('count_by_damage_severity_class')}`",
            f"- Adverse outcome counts: `{summary.get('count_by_adverse_outcome_label')}`",
            f"- Records per event: `{summary.get('records_per_event')}`",
            "",
            "## Confidence and Quality",
            f"- Match confidence distribution: `{summary.get('match_confidence_distribution')}`",
            f"- Invalid coordinates dropped: `{summary.get('dropped_invalid_coordinate_count')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "record_id",
        "source_record_id",
        "source_name",
        "source_path",
        "event_id",
        "event_name",
        "event_date",
        "event_year",
        "latitude",
        "longitude",
        "address_text",
        "locality",
        "state",
        "postal_code",
        "parcel_identifier",
        "damage_label",
        "damage_severity_class",
        "structure_loss_or_major_damage",
        "adverse_outcome_binary",
        "adverse_outcome_label",
        "source_native_label",
        "label_confidence",
        "geospatial_confidence",
        "match_confidence",
        "provenance_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["provenance_notes"] = ";".join(str(token) for token in (row.get("provenance_notes") or []))
            writer.writerow(out)


def run_public_outcomes_ingestion(
    *,
    sources: list[OutcomeSourceSpec],
    output_root: Path = Path("benchmark/public_outcomes/normalized"),
    run_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one input source is required.")
    run_token = str(run_id or _timestamp_id())
    generated_at = _deterministic_generated_at(run_id)
    run_dir = Path(output_root).expanduser() / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    included_sources: list[dict[str, Any]] = []
    excluded_sources: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    dropped_invalid_coordinates_total = 0

    deduped_specs: list[OutcomeSourceSpec] = []
    seen_paths: set[str] = set()
    ignored_duplicate_source_paths: list[str] = []
    for spec in sources:
        key = str(spec.path.expanduser().resolve())
        if key in seen_paths:
            ignored_duplicate_source_paths.append(str(spec.path))
            continue
        seen_paths.add(key)
        deduped_specs.append(spec)

    for spec in deduped_specs:
        if not spec.path.exists():
            excluded_sources.append(_source_status(spec, status="not_found", reason="input_path_not_found"))
            continue
        try:
            payload = normalize_public_damage_rows(
                input_path=spec.path,
                source_name=(spec.source_name or None),
                default_state=spec.default_state,
            )
        except Exception as exc:
            excluded_sources.append(_source_status(spec, status="parse_failed", reason=str(exc)))
            continue
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        dropped_invalid = int(payload.get("dropped_missing_or_invalid_coordinates") or 0)
        dropped_invalid_coordinates_total += dropped_invalid
        included_sources.append(
            {
                "source_name": payload.get("source_name") or spec.source_name or spec.path.stem,
                "source_path": str(spec.path),
                "status": "included",
                "record_count": len(records),
                "dropped_invalid_coordinate_count": dropped_invalid,
            }
        )
        for row in records:
            if not isinstance(row, dict):
                continue
            raw_records.append(_normalize_record(row))

    if not raw_records:
        raise ValueError(
            "No usable normalized records were produced. Check input paths and source formats."
        )

    dedupe_index: dict[str, dict[str, Any]] = {}
    dedupe_quality: dict[str, tuple[int, float, int]] = {}
    deduplicated_count = 0
    for row in raw_records:
        key = _dedupe_key(row)
        quality = _record_quality_score(row)
        if key in dedupe_index:
            deduplicated_count += 1
            if quality > dedupe_quality[key]:
                dedupe_index[key] = row
                dedupe_quality[key] = quality
            continue
        dedupe_index[key] = row
        dedupe_quality[key] = quality
    deduped_records = sorted(
        dedupe_index.values(),
        key=lambda row: (
            str(row.get("source_name") or ""),
            str(row.get("event_id") or ""),
            str(row.get("record_id") or ""),
        ),
    )
    summary = _summarize_records(
        raw_record_count=len(raw_records),
        deduped_records=deduped_records,
        deduplicated_count=deduplicated_count,
        configured_source_count=len(deduped_specs),
        ignored_duplicate_source_paths=ignored_duplicate_source_paths,
        included_sources=included_sources,
        excluded_sources=excluded_sources,
        dropped_invalid_coordinate_count=dropped_invalid_coordinates_total,
    )

    normalized_json = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_token,
        "generated_at": generated_at,
        "records": deduped_records,
    }
    normalized_json_path = run_dir / "normalized_outcomes.json"
    normalized_json_path.write_text(json.dumps(normalized_json, indent=2, sort_keys=True), encoding="utf-8")
    normalized_jsonl_path = run_dir / "normalized_outcomes.jsonl"
    _write_jsonl(normalized_jsonl_path, deduped_records)
    normalized_csv_path = run_dir / "normalized_outcomes.csv"
    _write_csv(normalized_csv_path, deduped_records)

    source_summary_path = run_dir / "source_summary_counts.json"
    source_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report_path = run_dir / "normalization_report.md"
    report_path.write_text(
        _build_markdown_report(run_id=run_token, generated_at=generated_at, summary=summary),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_token,
        "generated_at": generated_at,
        "output_root": str(Path(output_root).expanduser()),
        "artifacts": {
            "normalized_outcomes_json": str(normalized_json_path),
            "normalized_outcomes_jsonl": str(normalized_jsonl_path),
            "normalized_outcomes_csv": str(normalized_csv_path),
            "source_summary_counts_json": str(source_summary_path),
            "normalization_report_markdown": str(report_path),
        },
        "summary": summary,
        "caveat": (
            "Public observed outcomes are directional validation signals and not equivalent to insurer claims truth."
        ),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_id": run_token,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "record_count": len(deduped_records),
        "excluded_source_count": len(excluded_sources),
        "deduplicated_record_count": deduplicated_count,
    }


def _build_source_specs(inputs: list[str], source_names: list[str], default_state: str) -> list[OutcomeSourceSpec]:
    specs: list[OutcomeSourceSpec] = []
    for idx, token in enumerate(inputs):
        source_name = source_names[idx] if idx < len(source_names) and source_names[idx] else None
        specs.append(
            OutcomeSourceSpec(
                path=Path(token).expanduser(),
                source_name=source_name,
                default_state=default_state,
            )
        )
    return specs


def _load_source_specs_from_manifest(path: Path, default_state: str) -> list[OutcomeSourceSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("sources") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Source manifest must be a list or object with 'sources': {path}")
    specs: list[OutcomeSourceSpec] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        if row.get("enabled") is False:
            continue
        source_path = row.get("path")
        if not source_path:
            continue
        state = str(row.get("default_state") or default_state)
        specs.append(
            OutcomeSourceSpec(
                path=Path(str(source_path)).expanduser(),
                source_name=(str(row.get("source_name")) if row.get("source_name") else None),
                default_state=state,
            )
        )
    return specs


def _discover_input_files(
    *,
    input_dirs: list[str],
    input_glob: str,
) -> list[Path]:
    paths: list[Path] = []
    pattern = str(input_glob or "**/*")
    for token in input_dirs:
        root = Path(str(token)).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
                paths.append(candidate)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest and normalize public observed structure-damage outcomes into a reproducible "
            "multi-source artifact bundle for directional model validation."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input CSV/JSON/GeoJSON path. May be supplied multiple times.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help=(
            "Directory containing input outcome files. All files matching --input-glob "
            "with .csv/.json/.geojson suffixes are discovered."
        ),
    )
    parser.add_argument(
        "--input-glob",
        default="**/*",
        help="Glob pattern used with each --input-dir (default: **/*).",
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        default=[],
        help=(
            "Optional JSON manifest path containing source rows: "
            "{'sources':[{'path':'...','source_name':'...','default_state':'...','enabled':true}]}"
        ),
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=[],
        help="Optional source name aligned by index with each --input.",
    )
    parser.add_argument(
        "--default-state",
        default="CA",
        help="Default state code when source rows do not contain state.",
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/public_outcomes/normalized",
        help="Root directory for timestamped normalized outcome artifacts.",
    )
    parser.add_argument("--run-id", default="", help="Optional fixed run id for deterministic output naming.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite run directory if it already exists.")
    args = parser.parse_args()

    specs = _build_source_specs(
        inputs=[str(row) for row in (args.input or []) if str(row).strip()],
        source_names=[str(row) for row in (args.source_name or [])],
        default_state=str(args.default_state),
    )
    discovered_paths = _discover_input_files(
        input_dirs=[str(row) for row in (args.input_dir or []) if str(row).strip()],
        input_glob=str(args.input_glob or "**/*"),
    )
    if discovered_paths:
        specs.extend(
            _build_source_specs(
                inputs=[str(path) for path in discovered_paths],
                source_names=[],
                default_state=str(args.default_state),
            )
        )
    for token in [str(row) for row in (args.source_manifest or []) if str(row).strip()]:
        specs.extend(
            _load_source_specs_from_manifest(
                Path(token).expanduser(),
                default_state=str(args.default_state),
            )
        )
    if not specs and DEFAULT_SOURCE_MANIFEST_PATH.exists():
        specs.extend(
            _load_source_specs_from_manifest(
                DEFAULT_SOURCE_MANIFEST_PATH,
                default_state=str(args.default_state),
            )
        )
    result = run_public_outcomes_ingestion(
        sources=specs,
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
