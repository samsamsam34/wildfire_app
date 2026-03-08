from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from backend.data_prep.prepare_region import parse_bbox, prepare_region_layers
from backend.data_prep.sources import (
    default_cache_root,
    default_data_root,
    load_latest_staged_assets,
    stage_landfire_assets,
)


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


def _staged_layer_path(staged_meta: dict[str, object] | None, layer_key: str) -> str | None:
    if not isinstance(staged_meta, dict):
        return None
    layers = staged_meta.get("layers")
    if not isinstance(layers, dict):
        return None
    meta = layers.get(layer_key)
    if not isinstance(meta, dict):
        return None
    extracted = meta.get("extracted_path")
    if isinstance(extracted, str) and extracted and Path(extracted).exists():
        return extracted
    archive = meta.get("source_archive_path")
    if isinstance(archive, str) and archive and Path(archive).exists():
        return archive
    return None


def _apply_landfire_cache_policy(
    *,
    layer_sources: dict[str, str],
    layer_urls: dict[str, str],
    cache_root: Path,
    cache_only_landfire: bool,
    stage_landfire_first: bool,
    timeout: float,
    retries: int,
    backoff: float,
    fuel_checksum: str | None,
    canopy_checksum: str | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, object] | None]:
    staged: dict[str, object] | None = load_latest_staged_assets(cache_root)
    stage_result: dict[str, object] | None = None

    if stage_landfire_first:
        fuel_url = layer_urls.get("fuel")
        canopy_url = layer_urls.get("canopy")
        if not fuel_url and not canopy_url:
            # If URLs are not provided, use previously staged assets when available.
            if not staged:
                raise ValueError(
                    "--stage-landfire-first requested but no fuel/canopy URL provided and no staged metadata exists."
                )
        else:
            if not fuel_url:
                raise ValueError("--stage-landfire-first currently requires --fuel-url for LANDFIRE staging.")
            stage_result = stage_landfire_assets(
                fuel_url=fuel_url,
                canopy_url=canopy_url,
                cache_root=cache_root,
                timeout_seconds=timeout,
                retries=retries,
                retry_backoff_seconds=backoff,
                checksums={"fuel": fuel_checksum, "canopy": canopy_checksum},
            )
            staged = stage_result

    for key in ("fuel", "canopy"):
        if layer_sources.get(key):
            continue
        staged_path = _staged_layer_path(staged, key)
        if staged_path:
            layer_sources[key] = staged_path
            layer_urls.pop(key, None)
        elif cache_only_landfire:
            raise ValueError(
                f"--cache-only-landfire set but no staged cached {key} asset is available. "
                "Run scripts/stage_landfire_assets.py first."
            )

    return layer_sources, layer_urls, stage_result


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
    parser.add_argument("--cache-root", default=None)
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
        "--landfire-only",
        action="store_true",
        help="Prepare only LANDFIRE fuel/canopy layers for debugging pilot AOIs.",
    )
    parser.add_argument(
        "--no-auto-discovery",
        action="store_true",
        help="Disable automatic dataset discovery and require manual source paths/URLs.",
    )
    parser.add_argument(
        "--cache-only-landfire",
        action="store_true",
        help="Use only staged LANDFIRE cache assets for fuel/canopy; fail instead of downloading when unavailable.",
    )
    parser.add_argument(
        "--stage-landfire-first",
        action="store_true",
        help="Run LANDFIRE staging workflow before region prep when fuel/canopy URLs are provided.",
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
    cache_root = Path(args.cache_root).expanduser() if args.cache_root else default_cache_root()
    region_data_dir = Path(args.region_data_dir).expanduser() if args.region_data_dir else (default_data_root() / "regions")

    layer_sources, layer_urls, stage_result = _apply_landfire_cache_policy(
        layer_sources=layer_sources,
        layer_urls=layer_urls,
        cache_root=cache_root,
        cache_only_landfire=bool(args.cache_only_landfire),
        stage_landfire_first=bool(args.stage_landfire_first),
        timeout=float(args.download_timeout),
        retries=max(0, int(args.download_retries)),
        backoff=max(0.0, float(args.retry_backoff_seconds)),
        fuel_checksum=args.fuel_checksum,
        canopy_checksum=args.canopy_checksum,
    )

    manifest = prepare_region_layers(
        region_id=args.region_id,
        display_name=display_name,
        bounds=bbox,
        layer_sources=layer_sources,
        layer_urls=layer_urls,
        region_data_dir=region_data_dir,
        cache_dir=cache_root,
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
        landfire_only=args.landfire_only,
    )

    prepared = manifest.get("prepared_layers", [])
    attempted = manifest.get("attempted_layers", [])
    discovered = manifest.get("discovered_layers", [])
    skipped = manifest.get("skipped_layers", [])
    unsupported = manifest.get("unsupported_auto_discovery_layers", [])
    required_blockers = manifest.get("required_blockers", [])
    failed = manifest.get("failed_layers", [])
    out_root = str(region_data_dir)
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
                "landfire_only": manifest.get("download_config", {}).get("landfire_only", False),
                "cache_dir": manifest.get("download_config", {}).get("cache_dir"),
                "warnings": manifest.get("warnings", []),
                "errors": manifest.get("errors", []),
                "progress_log": manifest.get("progress_log", []),
                "landfire_stage": stage_result,
                "output_dir": str(out_root),
                "manifest_path": manifest_path,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
