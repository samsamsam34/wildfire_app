#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_prep.region_prep import SUPPORTED_PARCEL_SOURCES, fetch_parcels_for_region
from backend.region_registry import get_region_data_dir, load_region_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Region preparation helper (parcel/address source orchestration).",
    )
    parser.add_argument("--region", required=True, help="Region id (for example: missoula_pilot).")
    parser.add_argument("--regions-root", default=None, help="Override region data root (default: WF_REGION_DATA_DIR or data/regions).")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help="Bounding box override. If omitted, script uses existing region manifest bounds.",
    )
    parser.add_argument("--display-name", default=None, help="Optional display name used by county-search query hints.")
    parser.add_argument("--state", default=None, help="Optional state override (MT/WA/OR/CA/CO).")
    parser.add_argument("--fetch-parcels", action="store_true", help="Run parcel polygon source resolution and fetch.")
    parser.add_argument(
        "--parcel-source",
        default="auto",
        choices=list(SUPPORTED_PARCEL_SOURCES),
        help="Parcel source strategy. 'auto' runs Regrid -> Overture -> state_gis -> county_search.",
    )
    parser.add_argument(
        "--state-gis-registry",
        default=None,
        help="Path to state GIS registry json (default: config/state_gis_registry.json).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff-seconds", type=float, default=1.5)
    return parser.parse_args()


def _resolve_bounds(*, region_id: str, regions_root: Path, bbox_args: list[float] | None) -> dict[str, float]:
    if bbox_args is not None:
        min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_args]
        return {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }

    manifest = load_region_manifest(region_id, base_dir=str(regions_root))
    if manifest and isinstance(manifest.get("bounds"), dict):
        bounds = manifest["bounds"]
        return {
            "min_lon": float(bounds["min_lon"]),
            "min_lat": float(bounds["min_lat"]),
            "max_lon": float(bounds["max_lon"]),
            "max_lat": float(bounds["max_lat"]),
        }
    raise ValueError(
        "Bounds were not provided and no existing region manifest with bounds was found. "
        "Pass --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT."
    )


def main() -> int:
    args = _parse_args()
    regions_root = (
        Path(args.regions_root).expanduser()
        if args.regions_root
        else get_region_data_dir()
    )
    region_id = str(args.region).strip()
    bounds = _resolve_bounds(region_id=region_id, regions_root=regions_root, bbox_args=args.bbox)
    region_dir = regions_root / region_id
    region_dir.mkdir(parents=True, exist_ok=True)

    if not args.fetch_parcels:
        raise ValueError("No action selected. Use --fetch-parcels.")

    result = fetch_parcels_for_region(
        region_id=region_id,
        bounds=bounds,
        region_dir=region_dir,
        parcel_source=str(args.parcel_source).strip().lower(),
        state_code=(str(args.state).strip().upper() if args.state else None),
        display_name=(str(args.display_name).strip() if args.display_name else None),
        state_registry_path=(Path(args.state_gis_registry).expanduser() if args.state_gis_registry else None),
        timeout_seconds=float(args.timeout_seconds),
        retries=int(args.retries),
        backoff_seconds=float(args.backoff_seconds),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
