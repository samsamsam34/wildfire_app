#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANONICAL_LABELS = {
    "no_damage": {"none", "no damage", "undamaged", "no_known_damage", "affected_not_damaged"},
    "minor_damage": {"minor", "minor damage", "affected", "affected_minor"},
    "major_damage": {"major", "major damage", "severe", "substantial", "heavy damage"},
    "destroyed": {"destroyed", "totaled", "total loss", "complete loss"},
}


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_label(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return "unknown"
    for canonical, aliases in CANONICAL_LABELS.items():
        if text == canonical or text in aliases:
            return canonical
    # Common DINS-like phrasing.
    if "destroy" in text or "total" in text:
        return "destroyed"
    if "major" in text or "severe" in text or "substantial" in text:
        return "major_damage"
    if "minor" in text or "affected" in text:
        return "minor_damage"
    if "no damage" in text or "undamaged" in text or "none" == text:
        return "no_damage"
    return "unknown"


def _binary_target(label: str) -> int | None:
    if label in {"major_damage", "destroyed"}:
        return 1
    if label in {"no_damage", "minor_damage"}:
        return 0
    return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
            rows: list[dict[str, Any]] = []
            for feature in payload.get("features", []):
                if not isinstance(feature, dict):
                    continue
                props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
                geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
                coords = geom.get("coordinates") if isinstance(geom.get("coordinates"), list) else []
                row = dict(props)
                if geom.get("type") == "Point" and len(coords) >= 2:
                    row.setdefault("longitude", coords[0])
                    row.setdefault("latitude", coords[1])
                rows.append(row)
            return rows
        records = payload.get("records")
        if isinstance(records, list):
            return [row for row in records if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON payload shape in {path}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in {".json", ".geojson"}:
        return _load_json(path)
    raise ValueError(f"Unsupported input format for {path}; use CSV/JSON/GeoJSON.")


def normalize_public_damage_rows(
    *,
    input_path: Path,
    source_name: str | None = None,
    default_state: str = "CA",
) -> dict[str, Any]:
    rows = _load_rows(input_path)
    normalized: list[dict[str, Any]] = []
    dropped_missing_coords = 0

    for idx, row in enumerate(rows, start=1):
        lat = _to_float(_first(row, ["latitude", "lat", "y", "structure_latitude"]))
        lon = _to_float(_first(row, ["longitude", "lon", "lng", "x", "structure_longitude"]))
        if lat is None or lon is None:
            dropped_missing_coords += 1
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            dropped_missing_coords += 1
            continue

        raw_label = _first(
            row,
            [
                "damage_label",
                "damage_state",
                "damage_status",
                "structure_status",
                "status",
                "outcome_label",
            ],
        )
        damage_label = _normalize_label(raw_label)
        binary = _binary_target(damage_label)

        event_name = _first(row, ["event_name", "incident_name", "fire_name"])
        event_id = _first(row, ["event_id", "incident_id", "fire_id"])
        event_date = _first(row, ["event_date", "incident_date", "inspection_date", "date"])
        event_year = _first(row, ["event_year", "year"]) or (
            str(event_date)[:4] if event_date else None
        )

        locality = _first(row, ["city", "locality", "town"])
        postal_code = _first(row, ["zip", "zip_code", "postal_code"])
        state = _first(row, ["state", "state_code"]) or default_state
        source_record_id = _first(row, ["record_id", "id", "global_id", "objectid"])
        address_text = _first(
            row,
            [
                "address",
                "site_address",
                "full_address",
                "formatted_address",
                "situs_address",
            ],
        )

        label_confidence = _to_float(_first(row, ["label_confidence", "confidence", "quality_score"]))
        source_quality_flags: list[str] = []
        if damage_label == "unknown":
            source_quality_flags.append("unknown_damage_label")
        if label_confidence is not None and label_confidence < 0.5:
            source_quality_flags.append("low_label_confidence")

        normalized.append(
            {
                "record_id": f"{input_path.stem}_{idx:06d}",
                "source_record_id": str(source_record_id) if source_record_id is not None else None,
                "source_name": source_name or input_path.stem,
                "source_path": str(input_path),
                "event_id": str(event_id) if event_id is not None else None,
                "event_name": str(event_name) if event_name is not None else None,
                "event_date": str(event_date) if event_date is not None else None,
                "event_year": int(str(event_year)[:4]) if str(event_year or "").isdigit() else None,
                "address_text": str(address_text) if address_text is not None else None,
                "locality": str(locality) if locality is not None else None,
                "state": str(state) if state is not None else None,
                "postal_code": str(postal_code) if postal_code is not None else None,
                "latitude": round(lat, 7),
                "longitude": round(lon, 7),
                "damage_label": damage_label,
                "structure_loss_or_major_damage": binary,
                "label_confidence": label_confidence,
                "source_quality_flags": source_quality_flags,
                "raw_damage_label": str(raw_label) if raw_label is not None else None,
            }
        )

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_name": source_name or input_path.stem,
        "input_path": str(input_path),
        "record_count": len(normalized),
        "dropped_missing_or_invalid_coordinates": dropped_missing_coords,
        "records": normalized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize public wildfire structure-damage records (DINS-style) into calibration-ready schema."
    )
    parser.add_argument("--input", required=True, help="Input CSV/JSON/GeoJSON with structure damage outcomes.")
    parser.add_argument(
        "--output-json",
        default="benchmark/calibration/public_structure_damage_normalized.json",
        help="Normalized output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional normalized output CSV path.",
    )
    parser.add_argument("--source-name", default="", help="Optional source name override.")
    parser.add_argument("--default-state", default="CA")
    args = parser.parse_args()

    payload = normalize_public_damage_rows(
        input_path=Path(args.input).expanduser(),
        source_name=(args.source_name or None),
        default_state=args.default_state,
    )

    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = payload.get("records") or []
        fieldnames = [
            "record_id",
            "source_record_id",
            "source_name",
            "event_id",
            "event_name",
            "event_date",
            "event_year",
            "address_text",
            "locality",
            "state",
            "postal_code",
            "latitude",
            "longitude",
            "damage_label",
            "structure_loss_or_major_damage",
            "label_confidence",
            "source_quality_flags",
            "raw_damage_label",
        ]
        with output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["source_quality_flags"] = ";".join(row.get("source_quality_flags") or [])
                writer.writerow(out)

    print(
        json.dumps(
            {
                "input": args.input,
                "output_json": str(output_json),
                "output_csv": str(Path(args.output_csv).expanduser()) if args.output_csv else None,
                "record_count": payload.get("record_count"),
                "dropped_missing_or_invalid_coordinates": payload.get(
                    "dropped_missing_or_invalid_coordinates"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
