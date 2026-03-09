from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - optional in constrained envs
    np = None
    rasterio = None
    from_bounds = None
    transform_bounds = None

try:
    from shapely.geometry import box, mapping, shape
except Exception:  # pragma: no cover - optional in constrained envs
    box = None
    mapping = None
    shape = None

from backend.data_prep.prepare_region import prepare_region_layers
from backend.data_prep.sources import acquire_layer_from_config, resolve_landfire_raster


CATALOG_CORE_RASTER_LAYERS = ("dem", "fuel", "canopy")
CATALOG_DERIVED_RASTER_LAYERS = ("slope",)
CATALOG_CORE_VECTOR_LAYERS = ("fire_perimeters", "building_footprints")
CATALOG_OPTIONAL_LAYERS = ("roads", "whp", "mtbs_severity", "gridmet_dryness")

LAYER_TYPES: dict[str, str] = {
    "dem": "raster",
    "fuel": "raster",
    "canopy": "raster",
    "slope": "raster",
    "burn_probability": "raster",
    "wildfire_hazard": "raster",
    "moisture": "raster",
    "aspect": "raster",
    "whp": "raster",
    "mtbs_severity": "raster",
    "gridmet_dryness": "raster",
    "fire_perimeters": "vector",
    "building_footprints": "vector",
    "roads": "vector",
}


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def default_catalog_root() -> Path:
    env = os.getenv("WILDFIRE_APP_CATALOG_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("data") / "catalog"


def default_cache_root() -> Path:
    env = os.getenv("WILDFIRE_APP_CACHE_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("data") / "cache"


def _layer_dir(catalog_root: Path, layer_name: str, layer_type: str) -> Path:
    top = "rasters" if layer_type == "raster" else "vectors"
    return catalog_root / top / layer_name


def _metadata_dir(catalog_root: Path, layer_name: str) -> Path:
    return catalog_root / "metadata" / layer_name


def _index_path(catalog_root: Path) -> Path:
    return catalog_root / "index" / "catalog_index.json"


def _ensure_catalog_dirs(catalog_root: Path, layer_name: str, layer_type: str) -> None:
    _layer_dir(catalog_root, layer_name, layer_type).mkdir(parents=True, exist_ok=True)
    _metadata_dir(catalog_root, layer_name).mkdir(parents=True, exist_ok=True)
    _index_path(catalog_root).parent.mkdir(parents=True, exist_ok=True)


def _stable_bbox(bounds: dict[str, float] | None) -> str | None:
    if not bounds:
        return None
    return (
        f"{bounds['min_lon']:.6f},{bounds['min_lat']:.6f},"
        f"{bounds['max_lon']:.6f},{bounds['max_lat']:.6f}"
    )


def _hash_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _looks_like_html(path: Path) -> bool:
    try:
        data = path.read_bytes()[:2048].lower()
    except Exception:
        return False
    return b"<html" in data or b"<!doctype html" in data or b"<body" in data


def _download_file(url: str, target: Path, timeout_seconds: float = 60.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response, open(target, "wb") as out:
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if target.stat().st_size <= 0:
        raise ValueError(f"Downloaded empty payload for {url}")
    if _looks_like_html(target):
        raise ValueError(f"Downloaded HTML/error payload for {url}")


def _ensure_raster_deps() -> None:
    if rasterio is None or np is None or from_bounds is None or transform_bounds is None:
        raise ValueError("rasterio and numpy are required for catalog raster operations.")


def _ensure_vector_deps() -> None:
    if shape is None or mapping is None or box is None:
        raise ValueError("shapely is required for catalog vector operations.")


def _clip_raster_if_requested(src: Path, dst: Path, bounds: dict[str, float] | None, compression: str = "DEFLATE") -> dict[str, Any]:
    _ensure_raster_deps()
    assert rasterio is not None and from_bounds is not None and transform_bounds is not None
    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
        if bounds:
            if ds.crs is None:
                raise ValueError(f"Raster has no CRS: {src}")
            bbox_ds = transform_bounds(
                "EPSG:4326",
                ds.crs,
                bounds["min_lon"],
                bounds["min_lat"],
                bounds["max_lon"],
                bounds["max_lat"],
            )
            window = from_bounds(*bbox_ds, transform=ds.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
            if window.width <= 0 or window.height <= 0:
                raise ValueError(f"Raster does not intersect bbox: {src}")
            arr = ds.read(window=window)
            transform = ds.window_transform(window)
            width = int(window.width)
            height = int(window.height)
        else:
            arr = ds.read()
            transform = ds.transform
            width = ds.width
            height = ds.height
        profile.update(
            width=width,
            height=height,
            transform=transform,
            compress=compression,
        )
        if width >= 16 and height >= 16:
            block = max(16, min(512, width, height))
            block = max(16, (block // 16) * 16)
            profile.update(tiled=True, blockxsize=block, blockysize=block)
        with rasterio.open(dst, "w", **profile) as out:
            out.write(arr)
    with rasterio.open(dst) as out_ds:
        return {
            "source_crs": str(out_ds.crs),
            "stored_crs": str(out_ds.crs),
            "resolution": [abs(out_ds.transform.a), abs(out_ds.transform.e)],
            "bounds": [float(v) for v in out_ds.bounds],
            "storage_format": "GeoTIFF",
        }


def _clip_vector_if_requested(src: Path, dst: Path, bounds: dict[str, float] | None) -> dict[str, Any]:
    _ensure_vector_deps()
    assert shape is not None and mapping is not None and box is not None
    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", []) if isinstance(payload, dict) else []
    if not isinstance(features, list):
        raise ValueError(f"Invalid vector source: {src}")
    aoi = box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"]) if bounds else None
    out_features: list[dict[str, Any]] = []
    bounds_union = None
    for feat in features:
        if not isinstance(feat, dict) or not feat.get("geometry"):
            continue
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        if aoi is not None:
            if not geom.intersects(aoi):
                continue
            geom = geom.intersection(aoi)
            if geom.is_empty:
                continue
        out_features.append(
            {
                "type": "Feature",
                "properties": dict(feat.get("properties") or {}),
                "geometry": mapping(geom),
            }
        )
        bounds_union = geom.bounds if bounds_union is None else (
            min(bounds_union[0], geom.bounds[0]),
            min(bounds_union[1], geom.bounds[1]),
            max(bounds_union[2], geom.bounds[2]),
            max(bounds_union[3], geom.bounds[3]),
        )
    if not out_features:
        raise ValueError(f"Vector source produced no features after normalization: {src}")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": out_features}, f)
    return {
        "source_crs": "EPSG:4326",
        "stored_crs": "EPSG:4326",
        "resolution": None,
        "bounds": [float(v) for v in bounds_union] if bounds_union else None,
        "storage_format": "GeoJSON",
        "feature_count": len(out_features),
    }


def load_catalog_index(catalog_root: Path | None = None) -> dict[str, Any]:
    root = Path(catalog_root or default_catalog_root()).expanduser()
    path = _index_path(root)
    if not path.exists():
        return {"catalog_root": str(root), "updated_at": None, "layers": {}}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {"catalog_root": str(root), "updated_at": None, "layers": {}}
    payload.setdefault("layers", {})
    payload["catalog_root"] = str(root)
    return payload


def _write_catalog_index(index: dict[str, Any], catalog_root: Path) -> None:
    index["updated_at"] = _now()
    path = _index_path(catalog_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _upsert_index_entry(index: dict[str, Any], layer_name: str, layer_type: str, entry: dict[str, Any]) -> None:
    layers = index.setdefault("layers", {})
    layer_bucket = layers.setdefault(layer_name, {"layer_type": layer_type, "entries": []})
    entries = layer_bucket.setdefault("entries", [])
    item_id = entry["item_id"]
    replaced = False
    for i, existing in enumerate(entries):
        if existing.get("item_id") == item_id:
            entries[i] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)


def _entry_for_index(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": meta["item_id"],
        "layer_name": meta["layer_name"],
        "layer_type": meta["layer_type"],
        "catalog_path": meta["catalog_path"],
        "bounds": meta.get("bounds"),
        "resolution": meta.get("resolution"),
        "stored_crs": meta.get("stored_crs"),
        "provider_type": meta.get("provider_type"),
        "acquisition_method": meta.get("acquisition_method"),
        "source_url": meta.get("source_url"),
        "source_endpoint": meta.get("source_endpoint"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
    }


def _write_layer_metadata(catalog_root: Path, layer_name: str, item_id: str, metadata: dict[str, Any]) -> Path:
    meta_dir = _metadata_dir(catalog_root, layer_name)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{item_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return meta_path


def _build_item_id(
    *,
    layer_name: str,
    source_ref: str,
    provider_type: str,
    acquisition_method: str,
    bounds: dict[str, float] | None,
    target_resolution: float | None,
) -> str:
    payload = {
        "layer_name": layer_name,
        "source_ref": source_ref,
        "provider_type": provider_type,
        "acquisition_method": acquisition_method,
        "bbox": _stable_bbox(bounds),
        "target_resolution": target_resolution,
    }
    return _hash_payload(payload)[:20]


def _resolve_ingest_input(
    *,
    layer_name: str,
    layer_type: str,
    source_path: str | None,
    source_url: str | None,
    source_endpoint: str | None,
    provider_type: str,
    bounds: dict[str, float] | None,
    cache_root: Path,
    prefer_bbox_downloads: bool,
    allow_full_download_fallback: bool,
    target_resolution: float | None,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> tuple[Path, dict[str, Any]]:
    warnings: list[str] = []
    if source_path:
        p = Path(source_path).expanduser()
        if not p.exists():
            raise ValueError(f"Source path not found: {p}")
        return p, {
            "provider_type": provider_type or "local_file",
            "acquisition_method": "local_existing",
            "source_url": None,
            "source_endpoint": source_endpoint,
            "cache_hit": False,
            "warnings": warnings,
        }

    if not source_url and not source_endpoint:
        raise ValueError("Either source_path, source_url, or source_endpoint is required.")

    if source_endpoint and bounds:
        result = acquire_layer_from_config(
            layer_key=layer_name,
            layer_type=layer_type,
            layer_config={
                "provider_type": provider_type or ("arcgis_image_service" if layer_type == "raster" else "arcgis_feature_service"),
                "source_endpoint": source_endpoint,
                "source_url": source_url,
                "full_download_url": source_url,
                "supports_bbox_export": True,
            },
            bounds=bounds,
            cache_root=cache_root,
            prefer_bbox_downloads=prefer_bbox_downloads,
            allow_full_download_fallback=allow_full_download_fallback,
            target_resolution=target_resolution,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if result and result.local_path:
            warnings.extend(result.warnings)
            return Path(result.local_path), {
                "provider_type": result.provider_type,
                "acquisition_method": result.acquisition_method,
                "source_url": result.source_url or source_url,
                "source_endpoint": result.source_endpoint or source_endpoint,
                "cache_hit": bool(result.cache_hit),
                "warnings": warnings,
            }
        if result and result.source_url:
            source_url = result.source_url
            warnings.extend(result.warnings)
            provider_type = result.provider_type
            acquisition_method = result.acquisition_method
        else:
            acquisition_method = "full_download_clip"
    else:
        acquisition_method = "full_download_clip"

    assert source_url
    cache_dir = cache_root / "catalog_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(source_url).path).suffix or (".tif" if layer_type == "raster" else ".geojson")
    digest = _hash_payload({"layer_name": layer_name, "source_url": source_url})
    target = cache_dir / f"{digest}{suffix}"
    cache_hit = target.exists() and target.stat().st_size > 0 and not _looks_like_html(target)
    if not cache_hit:
        _download_file(source_url, target, timeout_seconds=timeout_seconds)
    return target, {
        "provider_type": provider_type or "file_download",
        "acquisition_method": "cached_full_download" if cache_hit else acquisition_method,
        "source_url": source_url,
        "source_endpoint": source_endpoint,
        "cache_hit": cache_hit,
        "warnings": warnings,
    }


def ingest_catalog_raster(
    *,
    layer_name: str,
    source_path: str | None = None,
    source_url: str | None = None,
    source_endpoint: str | None = None,
    provider_type: str = "file_download",
    bounds: dict[str, float] | None = None,
    catalog_root: Path | None = None,
    cache_root: Path | None = None,
    prefer_bbox_downloads: bool = False,
    allow_full_download_fallback: bool = True,
    target_resolution: float | None = None,
    timeout_seconds: float = 60.0,
    retries: int = 2,
    backoff_seconds: float = 1.5,
    force: bool = False,
) -> dict[str, Any]:
    if LAYER_TYPES.get(layer_name, "raster") != "raster":
        raise ValueError(f"{layer_name} is not configured as a raster layer.")
    root = Path(catalog_root or default_catalog_root()).expanduser()
    cache = Path(cache_root or default_cache_root()).expanduser()
    _ensure_catalog_dirs(root, layer_name, "raster")

    ingest_path, acquisition_meta = _resolve_ingest_input(
        layer_name=layer_name,
        layer_type="raster",
        source_path=source_path,
        source_url=source_url,
        source_endpoint=source_endpoint,
        provider_type=provider_type,
        bounds=bounds,
        cache_root=cache,
        prefer_bbox_downloads=prefer_bbox_downloads,
        allow_full_download_fallback=allow_full_download_fallback,
        target_resolution=target_resolution,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )

    if ingest_path.suffix.lower() == ".zip" and layer_name in {"fuel", "canopy"}:
        lf = resolve_landfire_raster(
            layer_key=layer_name,
            source_path=ingest_path,
            cache_dir=cache,
            bounds=bounds or {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            progress_log=[],
            warnings=acquisition_meta["warnings"],
        )
        ingest_path = lf.raster_path

    source_ref = source_url or source_endpoint or str(ingest_path.resolve())
    item_id = _build_item_id(
        layer_name=layer_name,
        source_ref=source_ref,
        provider_type=acquisition_meta["provider_type"],
        acquisition_method=acquisition_meta["acquisition_method"],
        bounds=bounds,
        target_resolution=target_resolution,
    )
    out_path = _layer_dir(root, layer_name, "raster") / f"{item_id}.tif"
    meta_path = _metadata_dir(root, layer_name) / f"{item_id}.json"

    created_at = _now()
    if out_path.exists() and meta_path.exists() and not force:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["cache_hit"] = True
        metadata["updated_at"] = _now()
    else:
        clip_meta = _clip_raster_if_requested(ingest_path, out_path, bounds)
        metadata = {
            "item_id": item_id,
            "layer_name": layer_name,
            "layer_type": "raster",
            "provider_type": acquisition_meta["provider_type"],
            "source_url": acquisition_meta["source_url"],
            "source_endpoint": acquisition_meta["source_endpoint"],
            "acquisition_method": acquisition_meta["acquisition_method"],
            "source_crs": clip_meta["source_crs"],
            "stored_crs": clip_meta["stored_crs"],
            "resolution": clip_meta["resolution"],
            "bounds": clip_meta["bounds"],
            "storage_format": clip_meta["storage_format"],
            "catalog_path": str(out_path),
            "bbox_used": _stable_bbox(bounds),
            "warnings": acquisition_meta["warnings"],
            "cache_hit": bool(acquisition_meta["cache_hit"]),
            "version": None,
            "checksum": None,
            "created_at": created_at,
            "updated_at": _now(),
        }
        _write_layer_metadata(root, layer_name, item_id, metadata)

    index = load_catalog_index(root)
    _upsert_index_entry(index, layer_name, "raster", _entry_for_index(metadata))
    _write_catalog_index(index, root)
    return metadata


def ingest_catalog_vector(
    *,
    layer_name: str,
    source_path: str | None = None,
    source_url: str | None = None,
    source_endpoint: str | None = None,
    provider_type: str = "file_download",
    bounds: dict[str, float] | None = None,
    catalog_root: Path | None = None,
    cache_root: Path | None = None,
    prefer_bbox_downloads: bool = False,
    allow_full_download_fallback: bool = True,
    timeout_seconds: float = 60.0,
    retries: int = 2,
    backoff_seconds: float = 1.5,
    force: bool = False,
) -> dict[str, Any]:
    if LAYER_TYPES.get(layer_name, "vector") != "vector":
        raise ValueError(f"{layer_name} is not configured as a vector layer.")
    root = Path(catalog_root or default_catalog_root()).expanduser()
    cache = Path(cache_root or default_cache_root()).expanduser()
    _ensure_catalog_dirs(root, layer_name, "vector")

    ingest_path, acquisition_meta = _resolve_ingest_input(
        layer_name=layer_name,
        layer_type="vector",
        source_path=source_path,
        source_url=source_url,
        source_endpoint=source_endpoint,
        provider_type=provider_type,
        bounds=bounds,
        cache_root=cache,
        prefer_bbox_downloads=prefer_bbox_downloads,
        allow_full_download_fallback=allow_full_download_fallback,
        target_resolution=None,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    source_ref = source_url or source_endpoint or str(ingest_path.resolve())
    item_id = _build_item_id(
        layer_name=layer_name,
        source_ref=source_ref,
        provider_type=acquisition_meta["provider_type"],
        acquisition_method=acquisition_meta["acquisition_method"],
        bounds=bounds,
        target_resolution=None,
    )
    out_path = _layer_dir(root, layer_name, "vector") / f"{item_id}.geojson"
    meta_path = _metadata_dir(root, layer_name) / f"{item_id}.json"
    created_at = _now()

    if out_path.exists() and meta_path.exists() and not force:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["cache_hit"] = True
        metadata["updated_at"] = _now()
    else:
        clip_meta = _clip_vector_if_requested(ingest_path, out_path, bounds)
        metadata = {
            "item_id": item_id,
            "layer_name": layer_name,
            "layer_type": "vector",
            "provider_type": acquisition_meta["provider_type"],
            "source_url": acquisition_meta["source_url"],
            "source_endpoint": acquisition_meta["source_endpoint"],
            "acquisition_method": acquisition_meta["acquisition_method"],
            "source_crs": clip_meta["source_crs"],
            "stored_crs": clip_meta["stored_crs"],
            "resolution": clip_meta["resolution"],
            "bounds": clip_meta["bounds"],
            "storage_format": clip_meta["storage_format"],
            "feature_count": clip_meta.get("feature_count"),
            "catalog_path": str(out_path),
            "bbox_used": _stable_bbox(bounds),
            "warnings": acquisition_meta["warnings"],
            "cache_hit": bool(acquisition_meta["cache_hit"]),
            "version": None,
            "checksum": None,
            "created_at": created_at,
            "updated_at": _now(),
        }
        _write_layer_metadata(root, layer_name, item_id, metadata)

    index = load_catalog_index(root)
    _upsert_index_entry(index, layer_name, "vector", _entry_for_index(metadata))
    _write_catalog_index(index, root)
    return metadata


def _entry_intersects_bbox(entry: dict[str, Any], bounds: dict[str, float]) -> bool:
    eb = entry.get("bounds")
    if not isinstance(eb, list) or len(eb) != 4:
        return False
    minx, miny, maxx, maxy = [float(v) for v in eb]
    return not (maxx < bounds["min_lon"] or minx > bounds["max_lon"] or maxy < bounds["min_lat"] or miny > bounds["max_lat"])


def find_catalog_entry_for_bbox(
    *,
    layer_name: str,
    bounds: dict[str, float],
    catalog_root: Path | None = None,
) -> dict[str, Any] | None:
    index = load_catalog_index(catalog_root)
    layers = index.get("layers", {})
    bucket = layers.get(layer_name, {})
    entries = list(bucket.get("entries", [])) if isinstance(bucket, dict) else []
    if not entries:
        return None
    candidates = [e for e in entries if _entry_intersects_bbox(e, bounds)]
    if not candidates:
        return None
    candidates.sort(key=lambda e: str(e.get("updated_at") or ""), reverse=True)
    return candidates[0]


def _load_layer_sources_from_catalog(
    *,
    bounds: dict[str, float],
    catalog_root: Path,
    skip_optional_layers: bool,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[str]]:
    core_layers = list(CATALOG_CORE_RASTER_LAYERS) + list(CATALOG_CORE_VECTOR_LAYERS)
    optional_layers = [] if skip_optional_layers else list(CATALOG_OPTIONAL_LAYERS)
    selected_layers = core_layers + optional_layers
    sources: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for layer in selected_layers:
        entry = find_catalog_entry_for_bbox(layer_name=layer, bounds=bounds, catalog_root=catalog_root)
        if not entry:
            missing.append(layer)
            continue
        path = Path(str(entry.get("catalog_path") or ""))
        if not path.exists():
            missing.append(layer)
            continue
        sources[layer] = str(path)
        provenance[layer] = entry
    return sources, provenance, missing


def build_region_from_catalog(
    *,
    region_id: str,
    display_name: str,
    bounds: dict[str, float],
    catalog_root: Path | None = None,
    regions_root: Path | None = None,
    overwrite: bool = False,
    require_core_layers: bool = True,
    skip_optional_layers: bool = False,
    allow_partial: bool = False,
    target_resolution: float | None = None,
    validate: bool = False,
    raster_compression: str = "DEFLATE",
    tile_size: int = 512,
    max_expected_cells: int | None = None,
) -> dict[str, Any]:
    cat_root = Path(catalog_root or default_catalog_root()).expanduser()
    region_root = Path(regions_root or (Path("data") / "regions")).expanduser()
    layer_sources, catalog_provenance, missing = _load_layer_sources_from_catalog(
        bounds=bounds,
        catalog_root=cat_root,
        skip_optional_layers=skip_optional_layers,
    )
    core_required = set(CATALOG_CORE_RASTER_LAYERS + CATALOG_CORE_VECTOR_LAYERS)
    missing_core = sorted([k for k in missing if k in core_required])
    if require_core_layers and missing_core:
        raise ValueError("Catalog missing required core layers for bbox: " + ", ".join(missing_core))

    manifest = prepare_region_layers(
        region_id=region_id,
        display_name=display_name,
        bounds=bounds,
        layer_sources=layer_sources,
        layer_urls={},
        region_data_dir=region_root,
        force=overwrite,
        allow_partial=allow_partial,
        auto_discover=False,
        skip_download=True,
        require_core_layers=require_core_layers,
        skip_optional_layers=skip_optional_layers,
        target_resolution=target_resolution,
        raster_compression=raster_compression,
        tile_size=tile_size,
        max_expected_cells=max_expected_cells,
    )
    manifest.setdefault("catalog", {})
    manifest["catalog"].update(
        {
            "catalog_root": str(cat_root),
            "used": True,
            "missing_layers": missing,
            "provenance": catalog_provenance,
        }
    )
    for layer_key, layer_meta in (manifest.get("layers") or {}).items():
        entry = catalog_provenance.get(layer_key)
        if not entry:
            continue
        layer_meta["built_from_catalog"] = True
        layer_meta["catalog_layer_name"] = layer_key
        layer_meta["catalog_source_path"] = entry.get("catalog_path")
        layer_meta["provider_type"] = entry.get("provider_type")
        layer_meta["source_url"] = entry.get("source_url")
        layer_meta["source_endpoint"] = entry.get("source_endpoint")
        layer_meta["subset_bbox"] = _stable_bbox(bounds)
        layer_meta["acquisition_method"] = entry.get("acquisition_method")
        layer_meta["cache_hit"] = True

    region_dir = region_root / region_id
    manifest_path = region_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    if validate:
        from backend.data_prep.validate_region import validate_prepared_region

        validation = validate_prepared_region(region_id=region_id, base_dir=str(region_root), update_manifest=True)
        manifest["catalog"]["validation"] = validation

    return manifest

