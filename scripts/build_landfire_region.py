from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.data_prep.prepare_region import prepare_region_layers
from backend.data_prep.sources import (
    default_cache_root,
    default_data_root,
    default_tmp_root,
    load_latest_staged_assets,
)

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover
    rasterio = None
    from_bounds = None
    transform_bounds = None


def _estimate_cells(path: Path, bounds: dict[str, float]) -> int | None:
    if rasterio is None or from_bounds is None or transform_bounds is None:
        return None
    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        bbox_ds = transform_bounds(
            "EPSG:4326",
            ds.crs,
            bounds["min_lon"],
            bounds["min_lat"],
            bounds["max_lon"],
            bounds["max_lat"],
        )
        window = from_bounds(*bbox_ds, transform=ds.transform).round_offsets().round_lengths()
        return int(max(0, int(window.width)) * max(0, int(window.height)))


def _resolve_staged_layer_path(
    staged: dict[str, object] | None,
    layer_key: str,
) -> str | None:
    if not isinstance(staged, dict):
        return None
    layers = staged.get("layers")
    if not isinstance(layers, dict):
        return None
    layer_meta = layers.get(layer_key)
    if not isinstance(layer_meta, dict):
        return None
    extracted = layer_meta.get("extracted_path")
    if isinstance(extracted, str) and extracted:
        return extracted
    archive = layer_meta.get("source_archive_path")
    if isinstance(archive, str) and archive:
        return archive
    return None


def _can_reuse_existing_region(region_dir: Path, required_layers: list[str]) -> tuple[bool, str | None]:
    manifest_path = region_dir / "manifest.json"
    if not manifest_path.exists():
        return False, None
    with open(manifest_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_files = existing.get("files") if isinstance(existing, dict) else {}
    if not isinstance(existing_files, dict):
        return False, None
    if all((region_dir / str(existing_files.get(k, ""))).exists() for k in required_layers):
        return True, str(manifest_path)
    return False, str(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a region-ready LANDFIRE subset from staged cache assets.\n"
            "Intended for local now and VM/batch later with path/config overrides."
        )
    )
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--bbox", nargs=4, required=True, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--cache-root", default=None, help="Cache root (default: WILDFIRE_APP_CACHE_ROOT or data/cache).")
    parser.add_argument("--out-dir", default=None, help="Region output dir root (default: WILDFIRE_APP_DATA_ROOT/regions).")
    parser.add_argument("--fuel-cache-path", default=None, help="Optional explicit staged fuel source path.")
    parser.add_argument("--canopy-cache-path", default=None, help="Optional explicit staged canopy source path.")
    parser.add_argument(
        "--prefer-cached-extracted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer staged extracted cache paths when explicit cache paths are not provided.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", default="DEFLATE")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-expected-cells", type=int, default=40_000_000)
    args = parser.parse_args()

    min_lon, min_lat, max_lon, max_lat = args.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise SystemExit("Invalid bbox: require min < max for lon/lat.")
    bounds = {
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
    }

    cache_root = Path(args.cache_root).expanduser() if args.cache_root else default_cache_root()
    data_root = Path(args.out_dir).expanduser() if args.out_dir else (default_data_root() / "regions")
    tmp_root = default_tmp_root()
    staged = load_latest_staged_assets(cache_root) if args.prefer_cached_extracted or not (args.fuel_cache_path or args.canopy_cache_path) else None

    fuel_path = args.fuel_cache_path or _resolve_staged_layer_path(staged, "fuel")
    canopy_path = args.canopy_cache_path or _resolve_staged_layer_path(staged, "canopy")

    layer_sources: dict[str, str] = {}
    warnings: list[str] = []
    if fuel_path:
        layer_sources["fuel"] = fuel_path
    else:
        warnings.append("Fuel cache path was not resolved from args or staged metadata.")
    if canopy_path:
        layer_sources["canopy"] = canopy_path
    else:
        warnings.append("Canopy cache path was not resolved from args or staged metadata.")

    if not layer_sources:
        raise SystemExit(
            "No staged LANDFIRE sources resolved. Provide --fuel-cache-path/--canopy-cache-path "
            "or run scripts/stage_landfire_assets.py first."
        )

    region_dir = Path(data_root) / args.region_id
    if region_dir.exists() and not args.overwrite:
        required = [k for k in ("fuel", "canopy") if k in layer_sources]
        reusable, manifest_path = _can_reuse_existing_region(region_dir, required_layers=required)
        if reusable:
            print(
                json.dumps(
                    {
                        "region_id": args.region_id,
                        "final_status": "success",
                        "preparation_status": "reused_existing",
                        "prepared_layers": required,
                        "warnings": ["Region already prepared; rerun skipped. Use --overwrite to rebuild."],
                        "manifest_path": manifest_path,
                        "tmp_root": str(tmp_root),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        raise SystemExit(
            f"Region directory already exists: {region_dir}. Use --overwrite to rebuild."
        )

    estimated_cells: dict[str, int | None] = {}
    for key, path in layer_sources.items():
        p = Path(path)
        if p.exists():
            est = _estimate_cells(p, bounds=bounds)
            estimated_cells[key] = est
            if est and args.max_expected_cells and est > args.max_expected_cells:
                raise SystemExit(
                    f"{key} bbox appears too large for local workflow ({est} cells > limit {args.max_expected_cells}). "
                    "Use a smaller pilot bbox or pre-clipped regional source."
                )
            if est and args.max_expected_cells and est > int(args.max_expected_cells * 0.75):
                warnings.append(f"{key} bbox is large ({est} cells); local run may be slow.")

    display_name = args.display_name or args.region_id.replace("_", " ").title()
    source_metadata: dict[str, dict[str, str]] = {}
    if isinstance(staged, dict):
        for layer_key in ("fuel", "canopy"):
            layer = (staged.get("layers") or {}).get(layer_key) if isinstance(staged.get("layers"), dict) else None
            if isinstance(layer, dict):
                meta: dict[str, str] = {}
                if layer.get("source_url"):
                    meta["dataset_source"] = str(layer["source_url"])
                if layer.get("handler_version"):
                    meta["dataset_version"] = str(layer["handler_version"])
                if meta:
                    source_metadata[layer_key] = meta

    manifest = prepare_region_layers(
        region_id=args.region_id,
        display_name=display_name,
        bounds=bounds,
        layer_sources=layer_sources,
        source_metadata=source_metadata,
        region_data_dir=data_root,
        cache_dir=cache_root,
        force=bool(args.overwrite),
        allow_partial=bool(args.allow_partial),
        auto_discover=False,
        landfire_only=True,
        clean_download_cache=False,
        raster_compression=args.compression,
        tile_size=max(16, int(args.tile_size)),
        max_expected_cells=int(args.max_expected_cells) if args.max_expected_cells else None,
    )

    out = {
        "region_id": manifest.get("region_id"),
        "final_status": manifest.get("final_status"),
        "preparation_status": manifest.get("preparation_status"),
        "prepared_layers": manifest.get("prepared_layers", []),
        "failed_layers": manifest.get("failed_layers", []),
        "warnings": sorted(set((manifest.get("warnings", []) or []) + warnings)),
        "estimated_cells": estimated_cells,
        "manifest_path": str(Path(data_root) / args.region_id / "manifest.json"),
        "tmp_root": str(tmp_root),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
