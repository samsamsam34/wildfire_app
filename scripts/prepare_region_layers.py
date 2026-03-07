from __future__ import annotations

import argparse
import json

from backend.data_prep.prepare_region import prepare_region_layers


def _parse_bbox(value: str) -> dict[str, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be min_lon,min_lat,max_lon,max_lat")
    try:
        min_lon, min_lat, max_lon, max_lat = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be numeric") from exc
    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local wildfire layers for one region.")
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--bbox", required=True, type=_parse_bbox, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--region-data-dir", default=None)
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing region directory")

    parser.add_argument("--dem", required=True)
    parser.add_argument("--slope", required=True)
    parser.add_argument("--fuel", required=True)
    parser.add_argument("--canopy", required=True)
    parser.add_argument("--fire-perimeters", required=True)
    parser.add_argument("--building-footprints", required=True)

    parser.add_argument("--burn-probability", default=None)
    parser.add_argument("--wildfire-hazard", default=None)
    parser.add_argument("--moisture", default=None)
    parser.add_argument("--aspect", default=None)

    args = parser.parse_args()

    layer_sources = {
        "dem": args.dem,
        "slope": args.slope,
        "fuel": args.fuel,
        "canopy": args.canopy,
        "fire_perimeters": args.fire_perimeters,
        "building_footprints": args.building_footprints,
    }
    if args.burn_probability:
        layer_sources["burn_probability"] = args.burn_probability
    if args.wildfire_hazard:
        layer_sources["wildfire_hazard"] = args.wildfire_hazard
    if args.moisture:
        layer_sources["moisture"] = args.moisture
    if args.aspect:
        layer_sources["aspect"] = args.aspect

    manifest = prepare_region_layers(
        region_id=args.region_id,
        display_name=args.display_name,
        bounds=args.bbox,
        layer_sources=layer_sources,
        region_data_dir=args.region_data_dir,
        crs=args.crs,
        copy_files=args.copy,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
