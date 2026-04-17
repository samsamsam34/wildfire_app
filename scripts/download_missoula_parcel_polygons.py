#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.data_prep.parcel_polygons import (
    MISSOULA_PILOT_BBOX,
    download_and_clip_missoula_parcel_polygons,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download + clip Missoula parcel polygons into prepared region data.",
    )
    parser.add_argument("--region-id", default="missoula_pilot")
    parser.add_argument("--regions-root", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=[
            MISSOULA_PILOT_BBOX["min_lon"],
            MISSOULA_PILOT_BBOX["min_lat"],
            MISSOULA_PILOT_BBOX["max_lon"],
            MISSOULA_PILOT_BBOX["max_lat"],
        ],
    )
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff-seconds", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    min_lon, min_lat, max_lon, max_lat = args.bbox
    bounds = {
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
    }
    result = download_and_clip_missoula_parcel_polygons(
        region_id=str(args.region_id),
        regions_root=(Path(args.regions_root).expanduser() if args.regions_root else None),
        endpoint=(str(args.endpoint).strip() if args.endpoint else None),
        bounds=bounds,
        timeout_seconds=float(args.timeout_seconds),
        retries=int(args.retries),
        backoff_seconds=float(args.backoff_seconds),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
