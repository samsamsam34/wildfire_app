from __future__ import annotations

import argparse
import json

from backend.data_prep.prepare_region import parse_bbox, prepare_region_layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one pilot region under data/regions/<region_id>/ for runtime scoring.\n"
            "This is an offline/admin workflow: runtime API scoring does not download large GIS datasets."
        )
    )
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--bbox", required=True, type=parse_bbox, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--out-dir", dest="region_data_dir", default=None)
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing region directory")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Disable URL downloads (useful for local-source prep mode).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial manifest with warnings/errors instead of failing hard.",
    )

    parser.add_argument("--dem", default=None, help="Local DEM source path")
    parser.add_argument("--dem-url", default=None, help="DEM source URL (pilot path)")
    parser.add_argument("--slope", default=None, help="Optional local slope source path")
    parser.add_argument("--slope-url", default=None, help="Optional slope source URL")
    parser.add_argument("--fuel", default=None, help="Local fuel raster source path")
    parser.add_argument("--fuel-url", default=None, help="Optional fuel source URL (LANDFIRE automation is partial)")
    parser.add_argument("--canopy", default=None, help="Local canopy raster source path")
    parser.add_argument("--canopy-url", default=None, help="Optional canopy source URL (LANDFIRE automation is partial)")
    parser.add_argument("--fire-perimeters", default=None, help="Local fire perimeter GeoJSON source path")
    parser.add_argument("--fire-perimeters-url", default=None, help="Optional fire perimeter URL")
    parser.add_argument("--building-footprints", default=None, help="Local building footprints GeoJSON source path")
    parser.add_argument("--building-footprints-url", default=None, help="Optional building footprints URL")

    parser.add_argument("--burn-probability", default=None)
    parser.add_argument("--burn-probability-url", default=None)
    parser.add_argument("--wildfire-hazard", default=None)
    parser.add_argument("--wildfire-hazard-url", default=None)
    parser.add_argument("--moisture", default=None)
    parser.add_argument("--moisture-url", default=None)
    parser.add_argument("--aspect", default=None)
    parser.add_argument("--aspect-url", default=None)

    args = parser.parse_args()
    display_name = args.display_name or args.region_id.replace("_", " ").title()

    layer_sources = {
        "dem": args.dem,
        "slope": args.slope,
        "fuel": args.fuel,
        "canopy": args.canopy,
        "fire_perimeters": args.fire_perimeters,
        "building_footprints": args.building_footprints,
    }
    optional_local = {
        "burn_probability": args.burn_probability,
        "wildfire_hazard": args.wildfire_hazard,
        "moisture": args.moisture,
        "aspect": args.aspect,
    }
    for key, value in optional_local.items():
        if value:
            layer_sources[key] = value

    layer_urls = {
        "dem": args.dem_url,
        "slope": args.slope_url,
        "fuel": args.fuel_url,
        "canopy": args.canopy_url,
        "fire_perimeters": args.fire_perimeters_url,
        "building_footprints": args.building_footprints_url,
        "burn_probability": args.burn_probability_url,
        "wildfire_hazard": args.wildfire_hazard_url,
        "moisture": args.moisture_url,
        "aspect": args.aspect_url,
    }
    layer_sources = {k: v for k, v in layer_sources.items() if v}
    layer_urls = {k: v for k, v in layer_urls.items() if v}

    manifest = prepare_region_layers(
        region_id=args.region_id,
        display_name=display_name,
        bounds=args.bbox,
        layer_sources=layer_sources,
        layer_urls=layer_urls,
        region_data_dir=args.region_data_dir,
        crs=args.crs,
        copy_files=args.copy,
        force=args.force,
        skip_download=args.skip_download,
        allow_partial=args.allow_partial,
    )

    prepared = [k for k, v in manifest.get("layers", {}).items() if v.get("validation_status") == "ok"]
    skipped = [k for k, v in manifest.get("layers", {}).items() if v.get("validation_status") == "missing"]
    failed = [k for k, v in manifest.get("layers", {}).items() if v.get("validation_status") == "error"]
    print(
        json.dumps(
            {
                "region_id": manifest.get("region_id"),
                "preparation_status": manifest.get("preparation_status"),
                "prepared_layers": prepared,
                "skipped_layers": skipped,
                "failed_layers": failed,
                "warnings": manifest.get("warnings", []),
                "errors": manifest.get("errors", []),
                "output_dir": str((args.region_data_dir or "data/regions")),
                "manifest_path": f"{(args.region_data_dir or 'data/regions')}/{args.region_id}/manifest.json",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
