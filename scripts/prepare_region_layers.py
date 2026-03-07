from __future__ import annotations

import argparse
import json
from typing import Sequence

from backend.data_prep.prepare_region import parse_bbox, prepare_region_layers


def _parse_bbox_args(values: Sequence[str]) -> dict[str, float]:
    if len(values) == 1:
        return parse_bbox(values[0])
    if len(values) == 4:
        return parse_bbox(",".join(values))
    raise ValueError("--bbox expects either one comma-delimited value or four numbers")


def _build_source_metadata_from_args(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    checksum_flags = {
        "dem": "dem_checksum",
        "slope": "slope_checksum",
        "fuel": "fuel_checksum",
        "canopy": "canopy_checksum",
        "fire_perimeters": "fire_perimeters_checksum",
        "building_footprints": "building_footprints_checksum",
        "burn_probability": "burn_probability_checksum",
        "wildfire_hazard": "wildfire_hazard_checksum",
        "moisture": "moisture_checksum",
        "aspect": "aspect_checksum",
    }
    source_metadata: dict[str, dict[str, str]] = {}
    for layer_key, attr_name in checksum_flags.items():
        checksum = getattr(args, attr_name, None)
        if checksum:
            source_metadata[layer_key] = {"checksum": checksum}
    return source_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one pilot region under data/regions/<region_id>/ for runtime scoring.\n"
            "This is an offline/admin workflow: runtime API scoring does not download large GIS datasets."
        )
    )
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument(
        "--bbox",
        nargs="+",
        required=True,
        help="Either 'min_lon,min_lat,max_lon,max_lat' or four values: min_lon min_lat max_lon max_lat",
    )
    parser.add_argument("--out-dir", dest="region_data_dir", default=None)
    parser.add_argument("--crs", default="EPSG:4326")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing region directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and report plan without writing outputs.")
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
    parser.add_argument(
        "--no-auto-discovery",
        action="store_true",
        help="Disable automatic dataset discovery and require manual source paths/URLs.",
    )
    parser.add_argument("--download-timeout", type=float, default=45.0, help="Per-request timeout in seconds.")
    parser.add_argument("--download-retries", type=int, default=2, help="Retry count for URL downloads.")
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.5,
        help="Base backoff in seconds; retries use exponential backoff.",
    )
    parser.add_argument(
        "--keep-temp-on-failure",
        action="store_true",
        help="Keep temporary downloaded/extracted files when preparation fails.",
    )
    parser.add_argument(
        "--clean-download-cache",
        action="store_true",
        default=False,
        help="Clear cached downloads for this run and remove staging folders after completion.",
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

    parser.add_argument("--dem-checksum", default=None, help="Optional checksum (sha256:<hex>) for DEM source")
    parser.add_argument("--slope-checksum", default=None, help="Optional checksum (sha256:<hex>) for slope source")
    parser.add_argument("--fuel-checksum", default=None, help="Optional checksum (sha256:<hex>) for fuel source")
    parser.add_argument("--canopy-checksum", default=None, help="Optional checksum (sha256:<hex>) for canopy source")
    parser.add_argument(
        "--fire-perimeters-checksum",
        default=None,
        help="Optional checksum (sha256:<hex>) for fire perimeter source",
    )
    parser.add_argument(
        "--building-footprints-checksum",
        default=None,
        help="Optional checksum (sha256:<hex>) for building-footprints source",
    )
    parser.add_argument(
        "--burn-probability-checksum",
        default=None,
        help="Optional checksum (sha256:<hex>) for burn-probability source",
    )
    parser.add_argument(
        "--wildfire-hazard-checksum",
        default=None,
        help="Optional checksum (sha256:<hex>) for wildfire-hazard source",
    )
    parser.add_argument("--moisture-checksum", default=None, help="Optional checksum (sha256:<hex>) for moisture source")
    parser.add_argument("--aspect-checksum", default=None, help="Optional checksum (sha256:<hex>) for aspect source")

    args = parser.parse_args()
    display_name = args.display_name or args.region_id.replace("_", " ").title()
    bbox = _parse_bbox_args(args.bbox)
    source_metadata = _build_source_metadata_from_args(args)

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
        bounds=bbox,
        layer_sources=layer_sources,
        layer_urls=layer_urls,
        region_data_dir=args.region_data_dir,
        crs=args.crs,
        copy_files=args.copy,
        source_metadata=source_metadata,
        force=args.force,
        skip_download=args.skip_download,
        allow_partial=args.allow_partial,
        auto_discover=not args.no_auto_discovery,
        download_timeout=args.download_timeout,
        download_retries=args.download_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        dry_run=args.dry_run,
        keep_temp_on_failure=args.keep_temp_on_failure,
        clean_download_cache=args.clean_download_cache,
    )

    prepared = manifest.get("prepared_layers", [])
    attempted = manifest.get("attempted_layers", [])
    discovered = manifest.get("discovered_layers", [])
    skipped = manifest.get("skipped_layers", [])
    unsupported = manifest.get("unsupported_auto_discovery_layers", [])
    required_blockers = manifest.get("required_blockers", [])
    failed = manifest.get("failed_layers", [])
    out_root = args.region_data_dir or "data/regions"
    manifest_path = None if args.dry_run else f"{out_root}/{args.region_id}/manifest.json"
    print(
        json.dumps(
            {
                "region_id": manifest.get("region_id"),
                "final_status": manifest.get("final_status"),
                "preparation_status": manifest.get("preparation_status"),
                "prepared_layers": prepared,
                "attempted_layers": attempted,
                "discovered_layers": discovered,
                "skipped_layers": skipped,
                "unsupported_auto_discovery_layers": unsupported,
                "failed_layers": failed,
                "required_blockers": required_blockers,
                "slope_derived": manifest.get("slope_derived", False),
                "archives_extracted": manifest.get("archives_extracted", False),
                "auto_discovery_used": manifest.get("download_config", {}).get("auto_discover", False),
                "cache_dir": manifest.get("download_config", {}).get("cache_dir"),
                "warnings": manifest.get("warnings", []),
                "errors": manifest.get("errors", []),
                "output_dir": str(out_root),
                "manifest_path": manifest_path,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
