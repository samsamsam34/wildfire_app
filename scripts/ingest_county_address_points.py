#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from backend.address_resolution import normalize_address_for_matching


LAT_KEYS = ("latitude", "lat", "y", "lat_dd", "lat_wgs84")
LON_KEYS = ("longitude", "lon", "lng", "x", "lon_dd", "lon_wgs84")
HOUSE_KEYS = ("house_number", "housenumber", "number", "st_num", "addr_num")
STREET_KEYS = ("street", "street_name", "road", "rd_name", "street_full", "road_name")
CITY_KEYS = ("city", "town", "locality", "municipality")
STATE_KEYS = ("state", "st", "province")
ZIP_KEYS = ("zip", "zipcode", "postal", "postcode")
ADDRESS_KEYS = (
    "address",
    "full_address",
    "formatted_address",
    "site_address",
    "situs_address",
    "prop_addr",
    "prop_address",
    "addr",
    "addr_full",
)


def _first_non_empty(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        if key in lowered and str(lowered[key]).strip():
            return str(lowered[key]).strip()
    return ""


def _coerce_lon_lat(record: dict[str, Any]) -> tuple[float, float] | None:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for lat_key in LAT_KEYS:
        for lon_key in LON_KEYS:
            lat_raw = lowered.get(lat_key)
            lon_raw = lowered.get(lon_key)
            if lat_raw is None or lon_raw is None:
                continue
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except (TypeError, ValueError):
                continue
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return float(lon), float(lat)
    return None


def _zip5(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:5] if len(digits) >= 5 else ""


def _normalize_address_components(address: str, house: str, street: str, city: str, state: str, postal: str) -> dict[str, str]:
    if not address:
        address = ", ".join([part for part in [f"{house} {street}".strip(), city, state, postal] if part])
    normalized_address = normalize_address_for_matching(address)
    return {
        "address": address,
        "house_number": normalize_address_for_matching(house),
        "street": normalize_address_for_matching(street),
        "city": normalize_address_for_matching(city),
        "state": normalize_address_for_matching(state),
        "postal": _zip5(postal),
        "normalized_address": normalized_address,
    }


def _iter_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _iter_geojson_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        rows: list[dict[str, Any]] = []
        for feature in payload["features"]:
            if not isinstance(feature, dict):
                continue
            props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            coords = None
            geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
            if geom.get("type") == "Point" and isinstance(geom.get("coordinates"), list) and len(geom["coordinates"]) >= 2:
                try:
                    lon = float(geom["coordinates"][0])
                    lat = float(geom["coordinates"][1])
                    coords = (lon, lat)
                except (TypeError, ValueError):
                    coords = None
            row = dict(props)
            if coords is not None:
                row["longitude"] = coords[0]
                row["latitude"] = coords[1]
            rows.append(row)
        return rows
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError("Unsupported JSON structure; expected GeoJSON FeatureCollection or list of records.")


def _build_features(records: list[dict[str, Any]], source_name: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    features: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    seen_normalized: dict[str, tuple[float, float]] = {}

    for idx, row in enumerate(records):
        lon_lat = _coerce_lon_lat(row)
        if lon_lat is None:
            warnings.append(f"row[{idx}] missing valid coordinates; skipped")
            continue
        address = _first_non_empty(row, ADDRESS_KEYS)
        house = _first_non_empty(row, HOUSE_KEYS)
        street = _first_non_empty(row, STREET_KEYS)
        city = _first_non_empty(row, CITY_KEYS)
        state = _first_non_empty(row, STATE_KEYS)
        postal = _first_non_empty(row, ZIP_KEYS)

        normalized = _normalize_address_components(address, house, street, city, state, postal)
        if not normalized["house_number"] or not normalized["street"]:
            warnings.append(f"row[{idx}] missing house_number/street; skipped")
            continue

        normalized_key = normalized["normalized_address"]
        lat = float(lon_lat[1])
        lon = float(lon_lat[0])
        existing = seen_normalized.get(normalized_key)
        if existing is not None:
            prev_lat, prev_lon = existing
            if abs(prev_lat - lat) > 1e-5 or abs(prev_lon - lon) > 1e-5:
                errors.append(
                    f"row[{idx}] normalized_address '{normalized_key}' conflicts with prior coordinates ({prev_lat},{prev_lon}) vs ({lat},{lon})"
                )
                continue
            warnings.append(f"row[{idx}] duplicate normalized_address '{normalized_key}' with same coordinates")
            continue
        seen_normalized[normalized_key] = (lat, lon)

        source_id = _first_non_empty(row, ("address_id", "site_id", "parcel_id", "objectid", "id")) or str(idx)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "address": normalized["address"],
                    "house_number": normalized["house_number"],
                    "street": normalized["street"],
                    "city": normalized["city"],
                    "state": normalized["state"],
                    "postal": normalized["postal"],
                    "normalized_address": normalized_key,
                    "source_name": source_name,
                    "source_type": "county_address_dataset",
                    "source_record_id": source_id,
                },
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )

    return features, warnings, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest county address points into normalized GeoJSON format.")
    parser.add_argument("--input", required=True, help="Input file path (CSV, GeoJSON, or JSON records).")
    parser.add_argument(
        "--output",
        default="data/address_points/okanogan/okanogan_address_points.geojson",
        help="Output GeoJSON path.",
    )
    parser.add_argument("--source-name", default="okanogan_county_addressing", help="Source name metadata.")
    parser.add_argument("--fail-on-warnings", action="store_true", help="Treat warnings as fatal.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        records = _iter_csv_records(input_path)
    elif suffix in {".json", ".geojson"}:
        records = _iter_geojson_records(input_path)
    else:
        raise SystemExit("Unsupported input format. Use CSV, JSON, or GeoJSON.")

    features, warnings, errors = _build_features(records, str(args.source_name))
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source_name": str(args.source_name),
            "record_count": len(features),
            "warning_count": len(warnings),
            "error_count": len(errors),
        },
        "features": features,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "record_count": len(features),
        "warning_count": len(warnings),
        "error_count": len(errors),
        "warnings": warnings[:10],
        "errors": errors[:10],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if errors:
        return 2
    if warnings and args.fail_on_warnings:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
