#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from backend.building_footprints import compute_structure_rings
from backend.naip_features import (
    NAIP_FEATURES_FILENAME,
    RING_KEYS,
    build_quantiles,
    percentile_from_quantiles,
    structure_feature_key,
)

try:
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.mask import mask as raster_mask
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - optional geo stack in constrained envs
    np = None
    rasterio = None
    Transformer = None
    raster_mask = None
    mapping = None
    shape = None
    shapely_transform = None


def _require_geo_stack() -> None:
    if not all([np is not None, rasterio is not None, Transformer is not None, raster_mask is not None, mapping is not None, shape is not None, shapely_transform is not None]):
        raise RuntimeError("rasterio/numpy/shapely/pyproj are required to prepare NAIP structure features.")


def _load_manifest(region_dir: Path) -> dict[str, Any]:
    manifest_path = region_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Region manifest must be a JSON object.")
    payload["_manifest_path"] = str(manifest_path)
    return payload


def _resolve_source_file(region_dir: Path, manifest: dict[str, Any], layer_keys: list[str]) -> Path | None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return None
    for key in layer_keys:
        rel = files.get(key)
        if not rel:
            continue
        candidate = Path(str(rel))
        if not candidate.is_absolute():
            candidate = region_dir / candidate
        if candidate.exists():
            return candidate
    return None


def _extract_structure_id(props: dict[str, Any]) -> str | None:
    for key in (
        "structure_id",
        "building_id",
        "id",
        "objectid",
        "OBJECTID",
        "globalid",
        "GlobalID",
        "fid",
        "FID",
    ):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _normalize_bands(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    # Expect NAIP-like order: R,G,B,(NIR). When only RGB is available, NIR=None.
    band_count = int(arr.shape[0])
    red = arr[0].astype(np.float32)
    green = arr[1 if band_count > 1 else 0].astype(np.float32)
    blue = arr[2 if band_count > 2 else 0].astype(np.float32)
    nir = arr[3].astype(np.float32) if band_count > 3 else None

    max_value = float(np.nanmax(arr)) if arr.size else 1.0
    scale = 255.0 if max_value > 2.5 else 1.0
    red = np.clip(red / scale, 0.0, 1.0)
    green = np.clip(green / scale, 0.0, 1.0)
    blue = np.clip(blue / scale, 0.0, 1.0)
    if nir is not None:
        nir = np.clip(nir / scale, 0.0, 1.0)
    return red, green, blue, nir


def _continuity_proxy(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return 100.0 if float(mask.mean()) >= 0.5 else 0.0
    h_both = np.logical_and(mask[:, :-1], mask[:, 1:]).sum()
    h_any = np.logical_or(mask[:, :-1], mask[:, 1:]).sum()
    v_both = np.logical_and(mask[:-1, :], mask[1:, :]).sum()
    v_any = np.logical_or(mask[:-1, :], mask[1:, :]).sum()
    denom = float(h_any + v_any)
    if denom <= 0.0:
        return 100.0 if float(mask.mean()) >= 0.5 else 0.0
    return round(float((h_both + v_both) / denom * 100.0), 2)


def _to_dataset_geom(ds: Any, geom_wgs84: Any) -> Any:
    if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        return geom_wgs84
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    return shapely_transform(transformer.transform, geom_wgs84)


def _summarize_ring(ds: Any, ring_geom_wgs84: Any) -> dict[str, Any] | None:
    geom_ds = _to_dataset_geom(ds, ring_geom_wgs84)
    if geom_ds.is_empty:
        return None
    try:
        clipped, _transform = raster_mask(ds, [mapping(geom_ds)], crop=True, filled=False)
    except Exception:
        return None
    if clipped is None or clipped.size == 0:
        return None
    if hasattr(clipped, "mask"):
        valid_mask = ~clipped.mask[0]
        data = clipped.data
    else:
        valid_mask = np.isfinite(clipped[0])
        data = clipped
    if int(valid_mask.sum()) == 0:
        return None

    data = np.where(np.broadcast_to(valid_mask, data.shape), data, np.nan)
    red, green, blue, nir = _normalize_bands(data)
    brightness = (red + green + blue) / 3.0

    if nir is not None:
        ndvi = (nir - red) / np.maximum(1e-6, nir + red)
        vegetation = ndvi >= 0.18
        canopy = ndvi >= 0.42
        high_fuel = ndvi >= 0.52
        ndvi_mean = float(np.nanmean(ndvi[valid_mask]))
    else:
        exg = (2.0 * green) - red - blue
        vegetation = exg >= 0.06
        canopy = exg >= 0.16
        high_fuel = exg >= 0.22
        ndvi_mean = None

    vegetation = np.logical_and(vegetation, valid_mask)
    canopy = np.logical_and(canopy, valid_mask)
    high_fuel = np.logical_and(high_fuel, valid_mask)
    impervious_low_fuel = np.logical_and(np.logical_not(vegetation), np.logical_and(valid_mask, brightness >= 0.60))

    valid_count = float(valid_mask.sum())
    veg_pct = float(vegetation.sum() / valid_count * 100.0)
    canopy_pct = float(canopy.sum() / valid_count * 100.0)
    high_fuel_pct = float(high_fuel.sum() / valid_count * 100.0)
    impervious_pct = float(impervious_low_fuel.sum() / valid_count * 100.0)

    return {
        "vegetation_cover_pct": round(veg_pct, 2),
        "canopy_proxy_pct": round(canopy_pct, 2),
        "high_fuel_proxy_pct": round(high_fuel_pct, 2),
        "impervious_low_fuel_pct": round(impervious_pct, 2),
        "vegetation_continuity_pct": _continuity_proxy(vegetation.astype(bool)),
        "ndvi_or_exg_mean": round(ndvi_mean, 4) if ndvi_mean is not None else None,
        "pixel_count": int(valid_count),
        "vegetation_density_proxy": round(veg_pct, 1),
    }


def _nearest_high_fuel_distance_ft(ring_metrics: dict[str, dict[str, Any]]) -> float | None:
    for ring_key, approx_ft in (
        ("ring_0_5_ft", 3.0),
        ("ring_5_30_ft", 18.0),
        ("ring_30_100_ft", 65.0),
        ("ring_100_300_ft", 180.0),
    ):
        metrics = ring_metrics.get(ring_key) or {}
        high_fuel = metrics.get("high_fuel_proxy_pct")
        try:
            if high_fuel is not None and float(high_fuel) >= 12.0:
                return float(approx_ft)
        except (TypeError, ValueError):
            continue
    return None


def run(
    *,
    region_id: str,
    regions_root: Path,
    naip_path: Path | None,
    output_filename: str,
    overwrite: bool,
    max_structures: int | None,
    update_manifest: bool,
) -> dict[str, Any]:
    _require_geo_stack()
    region_dir = regions_root / region_id
    if not region_dir.exists():
        raise FileNotFoundError(f"Region directory not found: {region_dir}")

    manifest = _load_manifest(region_dir)
    footprints_path = _resolve_source_file(
        region_dir,
        manifest,
        ["building_footprints_overture", "building_footprints", "fema_structures"],
    )
    if footprints_path is None:
        raise ValueError("No building footprint source found in region manifest.")

    imagery_path = naip_path or _resolve_source_file(region_dir, manifest, ["naip_imagery", "naip_rgb", "naip"])
    if imagery_path is None:
        raise ValueError(
            "No NAIP imagery source provided. Use --naip-path or add naip_imagery/naip_rgb in manifest files."
        )
    if not imagery_path.exists():
        raise FileNotFoundError(f"NAIP imagery file not found: {imagery_path}")

    output_path = region_dir / output_filename
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Re-run with --overwrite to replace.")

    payload = json.loads(footprints_path.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError(f"Invalid footprint GeoJSON: {footprints_path}")

    features_by_key: dict[str, dict[str, Any]] = {}
    keys_by_structure_id: dict[str, str] = {}

    metric_buckets: dict[str, list[float]] = {}
    processed = 0
    skipped = 0

    with rasterio.open(imagery_path) as ds:
        for feature in features:
            if max_structures is not None and processed >= max_structures:
                break
            if not isinstance(feature, dict):
                skipped += 1
                continue
            geom_raw = feature.get("geometry")
            if not isinstance(geom_raw, dict):
                skipped += 1
                continue
            try:
                footprint = shape(geom_raw)
            except Exception:
                skipped += 1
                continue
            if footprint.is_empty or footprint.geom_type not in {"Polygon", "MultiPolygon"}:
                skipped += 1
                continue

            props = feature.get("properties")
            props = props if isinstance(props, dict) else {}
            structure_id = _extract_structure_id(props)

            rings, _assumptions = compute_structure_rings(footprint)
            if not rings:
                skipped += 1
                continue
            centroid = footprint.centroid
            feature_key = structure_feature_key(
                structure_id=structure_id,
                centroid_lat=float(centroid.y),
                centroid_lon=float(centroid.x),
            )
            if not feature_key:
                skipped += 1
                continue

            ring_metrics: dict[str, dict[str, Any]] = {}
            for ring_key in RING_KEYS:
                ring_geom = rings.get(ring_key)
                if ring_geom is None:
                    continue
                summary = _summarize_ring(ds, ring_geom)
                if summary is None:
                    continue
                ring_metrics[ring_key] = summary

            if not ring_metrics:
                skipped += 1
                continue

            nearest_high_fuel_patch_distance_ft = _nearest_high_fuel_distance_ft(ring_metrics)
            canopy_adjacency_proxy_pct = (ring_metrics.get("ring_0_5_ft") or {}).get("canopy_proxy_pct")
            continuity_values = [
                (ring_metrics.get("ring_5_30_ft") or {}).get("vegetation_continuity_pct"),
                (ring_metrics.get("ring_30_100_ft") or {}).get("vegetation_continuity_pct"),
            ]
            continuity_values = [float(v) for v in continuity_values if v is not None]
            vegetation_continuity_proxy_pct = (
                round(sum(continuity_values) / len(continuity_values), 2) if continuity_values else None
            )

            row = {
                "feature_key": feature_key,
                "structure_id": structure_id,
                "centroid": {
                    "latitude": round(float(centroid.y), 7),
                    "longitude": round(float(centroid.x), 7),
                },
                "ring_metrics": ring_metrics,
                "near_structure_vegetation_0_5_pct": (ring_metrics.get("ring_0_5_ft") or {}).get("vegetation_cover_pct"),
                "canopy_adjacency_proxy_pct": canopy_adjacency_proxy_pct,
                "vegetation_continuity_proxy_pct": vegetation_continuity_proxy_pct,
                "nearest_high_fuel_patch_distance_ft": nearest_high_fuel_patch_distance_ft,
                "method": "naip_ring_threshold_proxies",
                "source_raster": str(imagery_path),
            }
            features_by_key[feature_key] = row
            if structure_id:
                keys_by_structure_id[str(structure_id)] = feature_key

            for ring_key, metrics in ring_metrics.items():
                for metric_key in (
                    "vegetation_cover_pct",
                    "canopy_proxy_pct",
                    "high_fuel_proxy_pct",
                    "impervious_low_fuel_pct",
                    "vegetation_continuity_pct",
                ):
                    metric_value = metrics.get(metric_key)
                    if metric_value is None:
                        continue
                    try:
                        metric_buckets.setdefault(f"{ring_key}.{metric_key}", []).append(float(metric_value))
                    except (TypeError, ValueError):
                        pass
            for top_level_key in (
                "near_structure_vegetation_0_5_pct",
                "canopy_adjacency_proxy_pct",
                "vegetation_continuity_proxy_pct",
            ):
                value = row.get(top_level_key)
                if value is None:
                    continue
                try:
                    metric_buckets.setdefault(top_level_key, []).append(float(value))
                except (TypeError, ValueError):
                    pass

            processed += 1

    quantiles: dict[str, dict[str, float]] = {
        metric_name: build_quantiles(values)
        for metric_name, values in metric_buckets.items()
        if values
    }

    # Attach local percentile context to each ring metric for deterministic runtime normalization.
    for row in features_by_key.values():
        ring_metrics = row.get("ring_metrics") or {}
        for ring_key, metrics in ring_metrics.items():
            if not isinstance(metrics, dict):
                continue
            for metric_key in (
                "vegetation_cover_pct",
                "canopy_proxy_pct",
                "high_fuel_proxy_pct",
                "impervious_low_fuel_pct",
                "vegetation_continuity_pct",
            ):
                q_key = f"{ring_key}.{metric_key}"
                percentile = percentile_from_quantiles(
                    metrics.get(metric_key),
                    quantiles.get(q_key),
                )
                if percentile is not None:
                    metrics[f"{metric_key}_local_percentile"] = percentile

    output_payload = {
        "schema_version": "1.0.0",
        "region_id": region_id,
        "source": {
            "imagery_path": str(imagery_path),
            "imagery_type": "naip",
            "footprints_path": str(footprints_path),
        },
        "feature_count": len(features_by_key),
        "keys_by_structure_id": keys_by_structure_id,
        "features_by_key": features_by_key,
        "quantiles": quantiles,
        "ring_keys": list(RING_KEYS),
        "notes": [
            "Derived from open NAIP imagery using transparent threshold proxies.",
            "Runtime scoring can blend these ring features with existing canopy/fuel/ring evidence.",
        ],
    }

    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")

    if update_manifest:
        files = manifest.get("files")
        if isinstance(files, dict):
            files["naip_structure_features"] = output_filename
        layers = manifest.get("layers")
        if isinstance(layers, dict):
            layers["naip_structure_features"] = {
                "source_name": "NAIP imagery derived ring features",
                "source_type": "derived_open_data",
                "source_mode": "derived",
                "source_url": None,
                "validation_status": "ok",
                "notes": "Generated by scripts/prepare_naip_structure_features.py",
            }
        prepared_layers = manifest.get("prepared_layers")
        if isinstance(prepared_layers, list) and "naip_structure_features" not in prepared_layers:
            prepared_layers.append("naip_structure_features")
            prepared_layers[:] = sorted(set(str(v) for v in prepared_layers))
        (region_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "region_id": region_id,
        "output_path": str(output_path),
        "feature_count": len(features_by_key),
        "processed_structures": processed,
        "skipped_structures": skipped,
        "imagery_path": str(imagery_path),
        "footprints_path": str(footprints_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare NAIP-derived near-structure ring features for a prepared region.")
    parser.add_argument("--region-id", required=True, help="Prepared region id under data/regions.")
    parser.add_argument("--regions-root", default="data/regions", help="Prepared regions root directory.")
    parser.add_argument("--naip-path", default=None, help="Path to NAIP raster (GeoTIFF).")
    parser.add_argument("--output-filename", default=NAIP_FEATURES_FILENAME, help="Output filename inside the region directory.")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional processing limit for quick iteration.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write naip_structure_features reference into region manifest files map.",
    )
    args = parser.parse_args()

    result = run(
        region_id=str(args.region_id),
        regions_root=Path(args.regions_root).expanduser(),
        naip_path=Path(args.naip_path).expanduser() if args.naip_path else None,
        output_filename=str(args.output_filename),
        overwrite=bool(args.overwrite),
        max_structures=int(args.max_structures) if args.max_structures is not None else None,
        update_manifest=bool(args.update_manifest),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

