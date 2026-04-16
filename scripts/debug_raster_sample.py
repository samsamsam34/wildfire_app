#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_data_adapters import _sample_raster_point_detailed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a single raster sample at a coordinate.")
    parser.add_argument("--raster", required=True, help="Path to raster (for example: data/regions/missoula_pilot/whp.tif)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude in decimal degrees")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in decimal degrees")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raster_path = Path(args.raster).expanduser()
    result = _sample_raster_point_detailed(str(raster_path), float(args.lat), float(args.lon))

    output = {
        "raster": str(raster_path),
        "status": result.get("status"),
        "raw_value": result.get("value"),
        "axis_order": result.get("axis_info") or [],
        "sample_coords": result.get("sample_coords"),
        "coords_swapped": bool(result.get("coords_swapped")),
        "within_bounds": bool(result.get("within_bounds")),
        "reason": result.get("reason"),
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    print(f"Raster: {output['raster']}")
    print(f"Status: {output['status']}")
    print(f"Raw sampled value: {output['raw_value']}")
    print(f"Axis order: {output['axis_order']}")
    print(f"Sample coords used: {output['sample_coords']} (swapped={output['coords_swapped']})")
    print(f"Within raster bounds: {output['within_bounds']}")
    if output.get("reason"):
        print(f"Reason: {output['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
