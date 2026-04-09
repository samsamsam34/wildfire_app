#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from shapely.geometry import shape as shapely_shape
except Exception:  # pragma: no cover
    shapely_shape = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.address_resolution import normalize_address_for_matching


ADDRESS_KEYS = (
    "situs_address",
    "site_address",
    "full_address",
    "address",
    "addr",
    "prop_addr",
)


def _first_non_empty(props: dict[str, Any], keys: tuple[str, ...]) -> str:
    lowered = {str(k).lower(): v for k, v in props.items()}
    for key in keys:
        value = lowered.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_address(props: dict[str, Any]) -> str:
    return _first_non_empty(props, ADDRESS_KEYS)


def _has_house_and_street(address: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(address or "").strip())
    if not cleaned:
        return False
    house_match = re.match(r"^\d+[a-zA-Z0-9-]*\b", cleaned)
    if not house_match:
        return False
    remainder = re.sub(r"^\d+[a-zA-Z0-9-]*\s+", "", cleaned, count=1)
    return bool(remainder.strip())


def _flatten_coords(values: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if isinstance(values, (list, tuple)):
        if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values[:2]):
            out.append((float(values[0]), float(values[1])))
        else:
            for item in values:
                out.extend(_flatten_coords(item))
    return out


def _representative_lon_lat(geometry: dict[str, Any]) -> tuple[float, float] | None:
    if not isinstance(geometry, dict):
        return None
    coords = geometry.get("coordinates")
    geom_type = str(geometry.get("type") or "")
    if geom_type == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            return float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            return None
    if shapely_shape is not None:
        try:
            geom = shapely_shape(geometry)
            if not geom.is_valid:
                geom = geom.buffer(0)
            point = geom.representative_point()
            return float(point.x), float(point.y)
        except Exception:
            pass
    flat = _flatten_coords(coords)
    if not flat:
        return None
    lons = [row[0] for row in flat]
    lats = [row[1] for row in flat]
    return float((min(lons) + max(lons)) / 2.0), float((min(lats) + max(lats)) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a parcel_address_points GeoJSON from parcel polygons by generating "
            "a representative point per unique complete address."
        )
    )
    parser.add_argument(
        "--region-dir",
        default="data/regions/winthrop_pilot",
        help="Prepared region directory containing parcel_polygons and parcel_address_points.",
    )
    parser.add_argument(
        "--input",
        default="parcel_polygons.geojson",
        help="Input polygon dataset path (absolute or relative to --region-dir).",
    )
    parser.add_argument(
        "--output",
        default="parcel_address_points.geojson",
        help="Output point dataset path (absolute or relative to --region-dir).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file before overwriting output.",
    )
    args = parser.parse_args()

    region_dir = Path(args.region_dir).expanduser()
    if not region_dir.is_absolute():
        region_dir = Path.cwd() / region_dir

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = region_dir / input_path

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = region_dir / output_path

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else []
    if not isinstance(features, list):
        raise ValueError("Input payload is not a FeatureCollection with a 'features' array.")

    repaired_features: list[dict[str, Any]] = []
    seen_addresses: set[str] = set()
    skipped_incomplete = 0
    skipped_coords = 0
    duplicate_addresses = 0

    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        address = _extract_address(props)
        if not _has_house_and_street(address):
            skipped_incomplete += 1
            continue
        normalized = normalize_address_for_matching(address)
        if not normalized:
            skipped_incomplete += 1
            continue
        if normalized in seen_addresses:
            duplicate_addresses += 1
            continue
        geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
        lon_lat = _representative_lon_lat(geometry)
        if lon_lat is None:
            skipped_coords += 1
            continue
        seen_addresses.add(normalized)
        repaired_features.append(
            {
                "type": "Feature",
                "properties": dict(props),
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon_lat[0]), float(lon_lat[1])],
                },
            }
        )

    repaired_payload = {
        "type": "FeatureCollection",
        "name": "parcel_address_points_repaired",
        "features": repaired_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.no_backup:
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        backup_path.write_bytes(output_path.read_bytes())

    output_path.write_text(json.dumps(repaired_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "input_feature_count": len(features),
                "output_feature_count": len(repaired_features),
                "skipped_incomplete_address": skipped_incomplete,
                "skipped_missing_coordinates": skipped_coords,
                "skipped_duplicate_address": duplicate_addresses,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
