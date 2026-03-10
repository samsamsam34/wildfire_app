from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - optional dependency in constrained envs
    np = None
    rasterio = None
    from_bounds = None
    transform_bounds = None

try:
    from shapely.geometry import box, mapping, shape
except Exception:  # pragma: no cover - optional dependency in constrained envs
    box = None
    mapping = None
    shape = None

from backend.region_registry import DEFAULT_REGION_DATA_DIR, REQUIRED_REGION_FILES
from backend.data_prep.sources import (
    LANDFIRE_HANDLER_VERSION,
    SourceAsset,
    acquire_layer_from_config,
    discover_wildfire_sources,
    resolve_landfire_raster,
)


STANDARD_LAYER_FILENAMES = {
    "dem": "dem.tif",
    "slope": "slope.tif",
    "fuel": "fuel.tif",
    "canopy": "canopy.tif",
    "fire_perimeters": "fire_perimeters.geojson",
    "building_footprints": "building_footprints.geojson",
    "burn_probability": "burn_probability.tif",
    "wildfire_hazard": "wildfire_hazard.tif",
    "moisture": "moisture.tif",
    "aspect": "aspect.tif",
    "whp": "whp.tif",
    "mtbs_severity": "mtbs_severity.tif",
    "gridmet_dryness": "gridmet_dryness.tif",
    "roads": "roads.geojson",
    "parcel_polygons": "parcel_polygons.geojson",
    "parcel_address_points": "parcel_address_points.geojson",
}

LAYER_TYPES = {
    "dem": "raster",
    "slope": "raster",
    "fuel": "raster",
    "canopy": "raster",
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
    "parcel_polygons": "vector",
    "parcel_address_points": "vector",
}

AUTOMATION_NOTES = {
    "dem": "Supports local file or URL source. Full source-catalog automation is deferred.",
    "fire_perimeters": "Supports local GeoJSON or URL source. Direct NIFC catalog sync is deferred.",
    "building_footprints": "Supports local GeoJSON or URL source. Direct Microsoft tile index sync is deferred.",
    "fuel": "Supports local/URL source prep. Full LANDFIRE discovery/download automation is deferred.",
    "canopy": "Supports local/URL source prep. Full LANDFIRE discovery/download automation is deferred.",
    "roads": "Supports local file or URL source. Automated regional road extraction is optional.",
    "whp": "Supports local file or URL source. WHP is optional enrichment for hazard context.",
    "mtbs_severity": "Supports local file or URL source. MTBS severity is optional enrichment.",
    "gridmet_dryness": "Supports local file or URL source. gridMET dryness is optional enrichment.",
    "parcel_polygons": "Supports local file or URL source. Parcel polygons improve structure matching confidence.",
    "parcel_address_points": "Supports local file or URL source. Address/parcel points improve property anchors.",
}

OPTIONAL_LAYER_KEYS = (
    "burn_probability",
    "wildfire_hazard",
    "moisture",
    "aspect",
    "roads",
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "parcel_polygons",
    "parcel_address_points",
)
BASE_REQUIRED_KEYS = ("dem", "fuel", "canopy", "fire_perimeters", "building_footprints")
MINIMUM_REQUIRED_KEYS = ("dem", "slope")


@dataclass
class DownloadConfig:
    timeout_seconds: float = 45.0
    retries: int = 2
    retry_backoff_seconds: float = 1.5


@dataclass
class PreparedLayerInput:
    path: Path | None
    source_name: str
    source_type: str
    source_mode: str
    source_url: str | None = None
    downloaded_at: str | None = None
    warnings: list[str] | None = None
    bytes_downloaded: int = 0
    download_status: str = "not_attempted"
    extraction_performed: bool = False
    extracted_path: str | None = None
    checksum_status: str = "not_provided"
    retry_count_used: int = 0
    timeout_seconds: float = 0.0
    cache_hit: bool = False


def parse_bbox(value: str) -> dict[str, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be min_lon,min_lat,max_lon,max_lat")
    try:
        min_lon, min_lat, max_lon, max_lat = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("bbox values must be numeric") from exc
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox must satisfy min < max for lon/lat")
    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_raster_deps() -> None:
    if rasterio is None or np is None or from_bounds is None or transform_bounds is None:
        raise ValueError("rasterio and numpy are required for raster preparation.")


def _ensure_vector_deps() -> None:
    if shape is None or mapping is None or box is None:
        raise ValueError("shapely is required for vector preparation.")


def _bbox_polygon(bounds: dict[str, float]):
    _ensure_vector_deps()
    assert box is not None
    return box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])


def _looks_like_html(path: Path) -> bool:
    try:
        head = path.read_bytes()[:2048].lower()
    except Exception:
        return False
    return b"<html" in head or b"<!doctype html" in head or b"<body" in head


def _expected_extensions(layer_type: str) -> tuple[str, ...]:
    if layer_type == "raster":
        return (".tif", ".tiff", ".zip")
    return (".geojson", ".json", ".zip")


def _sanity_check_source_file(path: Path, layer_key: str, layer_type: str) -> None:
    if not path.exists():
        raise ValueError(f"{layer_key} source missing after resolve: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{layer_key} source is empty: {path}")
    if _looks_like_html(path):
        raise ValueError(f"{layer_key} source looks like HTML/error page: {path}")

    ext = path.suffix.lower()
    if ext and ext not in _expected_extensions(layer_type):
        raise ValueError(f"{layer_key} source extension {ext} is unexpected for {layer_type} input")


def _download_with_retry(
    source_url: str,
    target: Path,
    config: DownloadConfig,
) -> tuple[int, int]:
    last_exc: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            bytes_downloaded = 0
            with urllib.request.urlopen(source_url, timeout=config.timeout_seconds) as response, open(target, "wb") as out:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_downloaded += len(chunk)
            return bytes_downloaded, attempt
        except Exception as exc:  # pragma: no cover - runtime network behavior
            last_exc = exc
            if attempt >= config.retries:
                break
            sleep_seconds = config.retry_backoff_seconds * (2**attempt)
            time.sleep(sleep_seconds)

    raise ValueError(
        f"Download failed for {source_url} after {config.retries + 1} attempts: {last_exc}"
    ) from last_exc


def _cache_path_for_url(source_url: str, cache_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(source_url)
    suffix = Path(parsed.path).suffix.lower() or ".bin"
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}{suffix}"


def _download_with_cache(
    source_url: str,
    *,
    cache_dir: Path,
    config: DownloadConfig,
) -> tuple[Path, int, int, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_url(source_url, cache_dir)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not _looks_like_html(cache_path):
        return cache_path, int(cache_path.stat().st_size), 0, True

    bytes_downloaded, retry_count_used = _download_with_retry(source_url, cache_path, config)
    return cache_path, bytes_downloaded, retry_count_used, False


def _select_archive_member(archive_path: Path, layer_key: str, layer_type: str) -> str:
    suffixes = (".tif", ".tiff") if layer_type == "raster" else (".geojson", ".json")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(name for name in zf.namelist() if not name.endswith("/"))
    candidates = [m for m in members if Path(m).suffix.lower() in suffixes]
    if not candidates:
        raise ValueError(f"{layer_key} archive has no usable {layer_type} files")

    # Deterministic selection:
    # 1) exact expected output filename if present (e.g., "dem.tif")
    # 2) unique filename that includes the layer key
    # 3) only-candidate fallback
    expected_name = STANDARD_LAYER_FILENAMES.get(layer_key, "").lower()
    exact_matches = [m for m in candidates if Path(m).name.lower() == expected_name]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"{layer_key} archive selection is ambiguous; multiple exact matches found")

    key_matches = [m for m in candidates if layer_key in Path(m).name.lower()]
    if len(key_matches) == 1:
        return key_matches[0]
    if len(key_matches) > 1:
        raise ValueError(f"{layer_key} archive selection is ambiguous; multiple layer-matching files found")

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"{layer_key} archive selection is ambiguous; multiple candidates found: "
        + ", ".join(Path(c).name for c in candidates[:5])
    )


def _extract_archive_layer(
    archive_path: Path,
    layer_key: str,
    layer_type: str,
    extract_dir: Path,
) -> Path:
    member = _select_archive_member(archive_path, layer_key=layer_key, layer_type=layer_type)
    with zipfile.ZipFile(archive_path, "r") as zf:
        extracted = zf.extract(member, path=extract_dir)
    return Path(extracted)


class LayerSourceAdapter:
    layer_key = "base"
    preserve_zip_for_handler = False

    def resolve(
        self,
        *,
        source_path: str | None,
        source_url: str | None,
        layer_type: str,
        download_dir: Path,
        cache_dir: Path,
        extraction_dir: Path,
        skip_download: bool,
        dry_run: bool,
        download_config: DownloadConfig,
        checksum: str | None,
    ) -> PreparedLayerInput:
        warnings: list[str] = []
        timeout_seconds = float(download_config.timeout_seconds)
        if source_path:
            path = Path(source_path).expanduser()
            if not path.exists():
                raise ValueError(f"{self.layer_key} local source not found: {path}")
            if dry_run:
                return PreparedLayerInput(
                    path=None,
                    source_name=path.name,
                    source_type="local_file",
                    source_mode="local_file",
                    warnings=warnings,
                    download_status="dry_run",
                    timeout_seconds=timeout_seconds,
                )

            _sanity_check_source_file(path, self.layer_key, layer_type)
            final_path = path
            extraction_performed = False
            extracted_path = None
            if path.suffix.lower() == ".zip" and not self.preserve_zip_for_handler:
                final_path = _extract_archive_layer(path, self.layer_key, layer_type, extraction_dir)
                extraction_performed = True
                extracted_path = str(final_path)
                _sanity_check_source_file(final_path, self.layer_key, layer_type)
            checksum_status = "not_provided"
            if checksum:
                checksum_status = _verify_checksum(final_path, checksum)
            return PreparedLayerInput(
                path=final_path,
                source_name=path.name,
                source_type="local_file",
                source_mode="local_file",
                warnings=warnings,
                download_status="not_attempted",
                extraction_performed=extraction_performed,
                extracted_path=extracted_path,
                checksum_status=checksum_status,
                timeout_seconds=timeout_seconds,
            )

        if source_url:
            if skip_download:
                warnings.append(f"{self.layer_key} download skipped by --skip-download.")
                return PreparedLayerInput(
                    path=None,
                    source_name=self.layer_key,
                    source_type="missing",
                    source_mode="remote_url",
                    source_url=source_url,
                    warnings=warnings,
                    download_status="skipped",
                    timeout_seconds=timeout_seconds,
                )
            if dry_run:
                return PreparedLayerInput(
                    path=None,
                    source_name=self.layer_key,
                    source_type="remote_url",
                    source_mode="remote_url",
                    source_url=source_url,
                    warnings=warnings,
                    download_status="dry_run",
                    timeout_seconds=timeout_seconds,
                )
            parsed = urllib.parse.urlparse(source_url)
            filename = Path(parsed.path).name or f"{self.layer_key}.dat"
            cached_path, bytes_downloaded, retry_count_used, cache_hit = _download_with_cache(
                source_url,
                cache_dir=cache_dir,
                config=download_config,
            )
            _sanity_check_source_file(cached_path, self.layer_key, layer_type)

            final_path = cached_path
            extraction_performed = False
            extracted_path = None
            if cached_path.suffix.lower() == ".zip" and not self.preserve_zip_for_handler:
                final_path = _extract_archive_layer(cached_path, self.layer_key, layer_type, extraction_dir)
                extraction_performed = True
                extracted_path = str(final_path)
                _sanity_check_source_file(final_path, self.layer_key, layer_type)

            checksum_status = "not_provided"
            if checksum:
                checksum_status = _verify_checksum(final_path, checksum)
            return PreparedLayerInput(
                path=final_path,
                source_name=filename,
                source_type="remote_url",
                source_mode="remote_url",
                source_url=source_url,
                downloaded_at=_now(),
                warnings=warnings,
                bytes_downloaded=bytes_downloaded,
                download_status="cache_hit" if cache_hit else "ok",
                extraction_performed=extraction_performed,
                extracted_path=extracted_path,
                checksum_status=checksum_status,
                retry_count_used=retry_count_used,
                timeout_seconds=timeout_seconds,
                cache_hit=cache_hit,
            )

        warnings.append(
            f"No source provided for {self.layer_key}. "
            f"{AUTOMATION_NOTES.get(self.layer_key, 'Provide a local source file or URL.')}"
        )
        return PreparedLayerInput(
            path=None,
            source_name=self.layer_key,
            source_type="missing",
            source_mode="missing",
            warnings=warnings,
            download_status="missing",
            timeout_seconds=timeout_seconds,
        )


class DemSourceAdapter(LayerSourceAdapter):
    layer_key = "dem"


class FirePerimeterSourceAdapter(LayerSourceAdapter):
    layer_key = "fire_perimeters"


class BuildingFootprintSourceAdapter(LayerSourceAdapter):
    layer_key = "building_footprints"


class FuelSourceAdapter(LayerSourceAdapter):
    layer_key = "fuel"
    preserve_zip_for_handler = True


class CanopySourceAdapter(LayerSourceAdapter):
    layer_key = "canopy"
    preserve_zip_for_handler = True


class GenericSourceAdapter(LayerSourceAdapter):
    def __init__(self, layer_key: str):
        self.layer_key = layer_key


ADAPTERS: dict[str, LayerSourceAdapter] = {
    "dem": DemSourceAdapter(),
    "fire_perimeters": FirePerimeterSourceAdapter(),
    "building_footprints": BuildingFootprintSourceAdapter(),
    "fuel": FuelSourceAdapter(),
    "canopy": CanopySourceAdapter(),
}


def _verify_checksum(path: Path, checksum: str) -> str:
    expected = checksum.strip().lower()
    if expected.startswith("sha256:"):
        expected = expected.split(":", 1)[1]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}: expected {expected}, got {actual}")
    return "verified"


def _stage_file(src: Path, dest: Path, *, copy_files: bool) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy_files:
        shutil.copy2(src, dest)
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def _clip_raster_to_bbox(
    src: Path,
    dest: Path,
    bounds: dict[str, float],
    *,
    compression: str | None = None,
    tile_size: int = 512,
    max_expected_cells: int | None = None,
) -> dict[str, Any]:
    _ensure_raster_deps()
    assert rasterio is not None and np is not None and from_bounds is not None and transform_bounds is not None
    with rasterio.open(src) as ds:
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
        if window.width <= 0 or window.height <= 0:
            raise ValueError(f"Raster does not intersect AOI: {src}")

        window = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        if window.width <= 0 or window.height <= 0:
            raise ValueError(f"Raster clip window is empty: {src}")
        expected_cells = int(window.width * window.height)
        if max_expected_cells and expected_cells > max_expected_cells:
            raise ValueError(
                f"Requested bbox is too large for local processing ({expected_cells} cells > limit {max_expected_cells}). "
                "Use a smaller pilot bbox or provide a pre-clipped regional raster source."
            )

        arr = ds.read(window=window)
        if arr.size == 0:
            raise ValueError(f"Raster clip produced empty array: {src}")

        arr_float = arr.astype("float32", copy=False)
        valid_mask = np.isfinite(arr_float)
        if ds.nodata is not None:
            valid_mask &= arr_float != float(ds.nodata)
        if not valid_mask.any():
            raise ValueError(f"Raster clip contains no valid data: {src}")

        profile = ds.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=ds.window_transform(window),
        )
        if compression:
            profile.update(compress=str(compression))
        normalized_tile = max(0, int(tile_size))
        if normalized_tile and int(window.width) >= 16 and int(window.height) >= 16:
            block_size = min(int(window.width), int(window.height), normalized_tile)
            block_size = max(16, (block_size // 16) * 16)
            if block_size >= 16:
                profile.update(
                    tiled=True,
                    blockxsize=block_size,
                    blockysize=block_size,
                )
        with rasterio.open(dest, "w", **profile) as out:
            out.write(arr)

    with rasterio.open(dest) as out_ds:
        return {
            "crs": str(out_ds.crs),
            "resolution": [abs(out_ds.transform.a), abs(out_ds.transform.e)],
            "bounds": list(out_ds.bounds),
            "expected_cells": expected_cells,
        }


def _raster_file_metadata(path: Path) -> dict[str, Any]:
    _ensure_raster_deps()
    assert rasterio is not None
    with rasterio.open(path) as ds:
        return {
            "crs": str(ds.crs),
            "resolution": [abs(ds.transform.a), abs(ds.transform.e)],
            "bounds": list(ds.bounds),
        }


def _derive_slope_from_dem(dem_path: Path, slope_path: Path) -> dict[str, Any]:
    _ensure_raster_deps()
    assert rasterio is not None and np is not None
    with rasterio.open(dem_path) as dem_ds:
        arr = dem_ds.read(1, masked=True).filled(np.nan).astype("float32")
        nodata = dem_ds.nodata

        x_res = abs(dem_ds.transform.a) or 1.0
        y_res = abs(dem_ds.transform.e) or 1.0
        grad_y, grad_x = np.gradient(arr, y_res, x_res)
        slope_rad = np.arctan(np.sqrt((grad_x * grad_x) + (grad_y * grad_y)))
        slope_deg = np.degrees(slope_rad).astype("float32")
        nan_mask = ~np.isfinite(arr)
        if nodata is not None:
            slope_deg = slope_deg.copy()
            slope_deg[nan_mask] = float(nodata)
        else:
            slope_deg = np.nan_to_num(slope_deg, nan=0.0)

        profile = dem_ds.profile.copy()
        profile.update(dtype="float32", count=1)
        with rasterio.open(slope_path, "w", **profile) as out:
            out.write(slope_deg, 1)

        return {
            "crs": str(dem_ds.crs),
            "resolution": [abs(dem_ds.transform.a), abs(dem_ds.transform.e)],
            "derived_from": "dem",
        }


def _clip_geojson_to_bbox(src: Path, dest: Path, bounds: dict[str, float]) -> dict[str, Any]:
    _ensure_vector_deps()
    assert shape is not None and mapping is not None
    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", []) if isinstance(payload, dict) else []
    if not isinstance(features, list):
        raise ValueError(f"Invalid GeoJSON feature collection: {src}")

    aoi = _bbox_polygon(bounds)
    clipped_features: list[dict[str, Any]] = []
    invalid_count = 0
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            shp = shape(geom)
        except Exception:
            invalid_count += 1
            continue
        if not shp.is_valid:
            shp = shp.buffer(0)
        if shp.is_empty or not shp.intersects(aoi):
            continue
        clipped = shp.intersection(aoi)
        if clipped.is_empty:
            continue
        clipped_features.append(
            {
                "type": "Feature",
                "properties": dict(feat.get("properties") or {}),
                "geometry": mapping(clipped),
            }
        )

    if not clipped_features:
        raise ValueError(f"Vector clip produced no intersecting features: {src}")

    out_payload = {"type": "FeatureCollection", "features": clipped_features}
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)
    return {
        "crs": "EPSG:4326",
        "feature_count": len(clipped_features),
        "invalid_geometry_count": invalid_count,
    }


def _validate_prepared_layer(path: Path, layer_type: str, bounds: dict[str, float]) -> None:
    if not path.exists():
        raise ValueError(f"Prepared layer missing: {path}")
    if layer_type == "raster":
        _ensure_raster_deps()
        _ensure_vector_deps()
        assert rasterio is not None and transform_bounds is not None and box is not None
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise ValueError(f"Raster missing CRS: {path}")
            if ds.width <= 0 or ds.height <= 0:
                raise ValueError(f"Raster has invalid shape: {path}")
            bbox_ds = transform_bounds(
                "EPSG:4326",
                ds.crs,
                bounds["min_lon"],
                bounds["min_lat"],
                bounds["max_lon"],
                bounds["max_lat"],
            )
            if not box(*ds.bounds).intersects(box(*bbox_ds)):
                raise ValueError(f"Raster does not intersect AOI: {path}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not features:
            raise ValueError(f"Vector output has no features: {path}")


def _layer_source(
    layer_key: str,
    layer_sources: dict[str, str] | None,
    layer_urls: dict[str, str | list[str]] | None,
) -> tuple[str | None, list[str]]:
    source_path = (layer_sources or {}).get(layer_key)
    source_url = (layer_urls or {}).get(layer_key)
    if isinstance(source_url, list):
        return source_path, [str(u) for u in source_url if u]
    if isinstance(source_url, str) and source_url:
        return source_path, [source_url]
    return source_path, []


def _base_layer_meta(
    *,
    layer_key: str,
    resolved: PreparedLayerInput,
    source_metadata: dict[str, dict[str, Any]],
    asset_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = source_metadata.get(layer_key, {}) or {}
    details = asset_details or {}
    return {
        "source_name": resolved.source_name,
        "source_type": resolved.source_type,
        "source_mode": resolved.source_mode,
        "source_url": resolved.source_url,
        "dataset_source": details.get("dataset_name") or meta.get("dataset_source"),
        "dataset_version": meta.get("dataset_version"),
        "dataset_provider": details.get("dataset_provider") or meta.get("dataset_provider"),
        "tile_ids": details.get("tile_ids", []),
        "download_url": details.get("download_url", resolved.source_url),
        "mosaic_performed": bool(details.get("mosaic_performed", False)),
        "freshness_timestamp": meta.get("freshness_timestamp"),
        "downloaded_at": resolved.downloaded_at,
        "download_status": resolved.download_status,
        "bytes_downloaded": int(resolved.bytes_downloaded or 0),
        "extraction_performed": bool(resolved.extraction_performed),
        "extracted_path": resolved.extracted_path,
        "checksum_status": resolved.checksum_status,
        "retry_count_used": int(resolved.retry_count_used or 0),
        "timeout_seconds": float(resolved.timeout_seconds or 0.0),
        "cache_hit": bool(resolved.cache_hit),
        "discovered_source": bool(details.get("discovered_source", False)),
        "provider_type": meta.get("provider_type"),
        "source_endpoint": meta.get("source_endpoint"),
        "acquisition_method": meta.get("acquisition_method"),
        "bbox_used": meta.get("bbox_used"),
        "output_resolution": meta.get("output_resolution"),
        "bbox_cache_hit": bool(meta.get("bbox_cache_hit", False)),
        "clipped_to_bbox": False,
        "validation_status": "missing",
        "notes": AUTOMATION_NOTES.get(layer_key),
    }


def _normalize_discovered_assets(
    layer_key: str,
    discovered_assets: dict[str, list[SourceAsset]] | None,
) -> list[SourceAsset]:
    if not discovered_assets:
        return []
    return list(discovered_assets.get(layer_key) or [])


def _normalize_source_config(source_config: dict[str, Any] | None) -> dict[str, Any]:
    if not source_config:
        return {}
    if isinstance(source_config.get("layers"), dict):
        return source_config["layers"]
    return source_config


def _apply_source_config_acquisition(
    *,
    bounds: dict[str, float],
    layer_sources: dict[str, str],
    layer_urls: dict[str, str | list[str]],
    source_metadata: dict[str, dict[str, Any]],
    source_config: dict[str, Any] | None,
    cache_root: Path,
    prefer_bbox_downloads: bool,
    allow_full_download_fallback: bool,
    target_resolution: float | None,
    download_config: DownloadConfig,
    notes: list[str],
    progress_log: list[str],
    skip_optional_layers: bool,
) -> None:
    configured_layers = _normalize_source_config(source_config)
    if not configured_layers:
        return
    for layer_key, layer_type in LAYER_TYPES.items():
        if skip_optional_layers and layer_key in OPTIONAL_LAYER_KEYS:
            continue
        if layer_sources.get(layer_key) or layer_urls.get(layer_key):
            continue
        layer_cfg = configured_layers.get(layer_key)
        if not isinstance(layer_cfg, dict):
            continue
        try:
            acquired = acquire_layer_from_config(
                layer_key=layer_key,
                layer_type=layer_type,
                layer_config=layer_cfg,
                bounds=bounds,
                cache_root=cache_root,
                prefer_bbox_downloads=prefer_bbox_downloads,
                allow_full_download_fallback=allow_full_download_fallback,
                target_resolution=target_resolution,
                timeout_seconds=download_config.timeout_seconds,
                retries=download_config.retries,
                backoff_seconds=download_config.retry_backoff_seconds,
            )
        except Exception as exc:
            notes.append(f"{layer_key} source-config acquisition failed: {exc}")
            progress_log.append(f"Source config acquisition failed for {layer_key}: {exc}")
            continue
        if acquired is None:
            continue

        if acquired.local_path:
            layer_sources[layer_key] = acquired.local_path
        elif acquired.source_url:
            layer_urls[layer_key] = acquired.source_url

        source_metadata.setdefault(layer_key, {})
        source_metadata[layer_key].update(
            {
                "provider_type": acquired.provider_type,
                "source_endpoint": acquired.source_endpoint,
                "acquisition_method": acquired.acquisition_method,
                "bbox_used": acquired.bbox_used,
                "output_resolution": acquired.output_resolution,
                "bbox_cache_hit": acquired.cache_hit,
            }
        )
        for warning in acquired.warnings:
            notes.append(warning)
        progress_log.append(
            f"Source-config acquired {layer_key} via {acquired.acquisition_method}"
            + (" (cache hit)" if acquired.cache_hit else "")
        )


def _resolve_inputs_for_layer(
    *,
    layer_key: str,
    layer_type: str,
    source_path: str | None,
    source_urls: list[str],
    discovered_assets: list[SourceAsset],
    adapter: LayerSourceAdapter,
    download_dir: Path,
    cache_dir: Path,
    extraction_dir: Path,
    skip_download: bool,
    dry_run: bool,
    download_config: DownloadConfig,
    checksum: str | None,
    notes: list[str],
) -> tuple[list[PreparedLayerInput], dict[str, Any]]:
    resolved_inputs: list[PreparedLayerInput] = []
    asset_details: dict[str, Any] = {
        "dataset_name": None,
        "dataset_provider": None,
        "tile_ids": [],
        "download_url": None,
        "mosaic_performed": False,
        "discovered_source": False,
    }

    if source_path:
        resolved = adapter.resolve(
            source_path=source_path,
            source_url=None,
            layer_type=layer_type,
            download_dir=download_dir,
            cache_dir=cache_dir,
            extraction_dir=extraction_dir,
            skip_download=skip_download,
            dry_run=dry_run,
            download_config=download_config,
            checksum=checksum,
        )
        resolved_inputs.append(resolved)
        return resolved_inputs, asset_details

    urls_to_fetch = list(source_urls)
    source_assets = discovered_assets if not urls_to_fetch else []
    if not urls_to_fetch and source_assets:
        urls_to_fetch = [asset.url for asset in source_assets if asset.url]
        if source_assets:
            asset_details["dataset_name"] = source_assets[0].dataset_name
            asset_details["dataset_provider"] = source_assets[0].dataset_provider
            asset_details["tile_ids"] = [a.tile_id for a in source_assets if a.tile_id]
            asset_details["download_url"] = urls_to_fetch[0] if urls_to_fetch else None
            asset_details["discovered_source"] = True

    if not urls_to_fetch:
        resolved = adapter.resolve(
            source_path=None,
            source_url=None,
            layer_type=layer_type,
            download_dir=download_dir,
            cache_dir=cache_dir,
            extraction_dir=extraction_dir,
            skip_download=skip_download,
            dry_run=dry_run,
            download_config=download_config,
            checksum=checksum,
        )
        for warning in resolved.warnings or []:
            notes.append(warning)
        return [resolved], asset_details

    for url in urls_to_fetch:
        resolved = adapter.resolve(
            source_path=None,
            source_url=url,
            layer_type=layer_type,
            download_dir=download_dir,
            cache_dir=cache_dir,
            extraction_dir=extraction_dir,
            skip_download=skip_download,
            dry_run=dry_run,
            download_config=download_config,
            checksum=checksum,
        )
        for warning in resolved.warnings or []:
            notes.append(warning)
        resolved_inputs.append(resolved)

    if len(resolved_inputs) > 1:
        asset_details["mosaic_performed"] = True
    return resolved_inputs, asset_details


def _mosaic_rasters(paths: list[Path], output_path: Path) -> Path:
    _ensure_raster_deps()
    assert rasterio is not None
    from rasterio.merge import merge

    datasets = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()
    return output_path


def _merge_vectors(paths: list[Path], output_path: Path) -> Path:
    merged_features: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if isinstance(features, list):
            merged_features.extend([f for f in features if isinstance(f, dict)])
    if not merged_features:
        raise ValueError("Merged vector sources produced no features.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": merged_features}, f)
    return output_path


def _prepare_one_layer(
    *,
    layer_key: str,
    bounds: dict[str, float],
    region_dir: Path,
    download_dir: Path,
    cache_dir: Path,
    extraction_dir: Path,
    staging_dir: Path,
    layer_sources: dict[str, str] | None,
    layer_urls: dict[str, str | list[str]] | None,
    discovered_assets: dict[str, list[SourceAsset]] | None,
    copy_files: bool,
    skip_download: bool,
    dry_run: bool,
    download_config: DownloadConfig,
    source_metadata: dict[str, dict[str, Any]],
    notes: list[str],
    progress_log: list[str],
    raster_compression: str | None,
    tile_size: int,
    max_expected_cells: int | None,
) -> tuple[str | None, dict[str, Any]]:
    layer_type = LAYER_TYPES[layer_key]
    adapter = ADAPTERS.get(layer_key, GenericSourceAdapter(layer_key))
    source_path, source_urls = _layer_source(layer_key, layer_sources, layer_urls)
    discovered_for_layer = _normalize_discovered_assets(layer_key, discovered_assets)
    checksum = (source_metadata.get(layer_key, {}) or {}).get("checksum")

    resolved_inputs, asset_details = _resolve_inputs_for_layer(
        layer_key=layer_key,
        layer_type=layer_type,
        source_path=source_path,
        source_urls=source_urls,
        discovered_assets=discovered_for_layer,
        adapter=adapter,
        download_dir=download_dir,
        cache_dir=cache_dir,
        extraction_dir=extraction_dir,
        skip_download=skip_download,
        dry_run=dry_run,
        download_config=download_config,
        checksum=checksum,
        notes=notes,
    )
    if not resolved_inputs:
        resolved_inputs = [
            PreparedLayerInput(
                path=None,
                source_name=layer_key,
                source_type="missing",
                source_mode="missing",
                download_status="missing",
            )
        ]
    primary_resolved = resolved_inputs[0]
    layer_meta = _base_layer_meta(
        layer_key=layer_key,
        resolved=primary_resolved,
        source_metadata=source_metadata,
        asset_details=asset_details,
    )
    filename = STANDARD_LAYER_FILENAMES[layer_key]
    if dry_run:
        if all(r.source_mode == "missing" for r in resolved_inputs):
            layer_meta.setdefault("acquisition_method", "unavailable")
            layer_meta["validation_status"] = "missing"
            return None, layer_meta
        layer_meta.setdefault(
            "acquisition_method",
            "bbox_export" if layer_meta.get("source_mode") == "local_file" and layer_meta.get("bbox_used") else
            ("full_download_clip" if layer_meta.get("source_mode") == "remote_url" else "local_existing"),
        )
        layer_meta["validation_status"] = "dry_run"
        layer_meta["clipped_to_bbox"] = True
        layer_meta["bytes_downloaded"] = int(sum(r.bytes_downloaded for r in resolved_inputs))
        layer_meta["cache_hit"] = bool(any(r.cache_hit for r in resolved_inputs))
        layer_meta["retry_count_used"] = int(sum(r.retry_count_used for r in resolved_inputs))
        layer_meta["extraction_performed"] = bool(any(r.extraction_performed for r in resolved_inputs))
        return filename, layer_meta

    valid_inputs = [r for r in resolved_inputs if r.path is not None]
    if not valid_inputs:
        return None, layer_meta

    input_paths: list[Path] = [Path(r.path) for r in valid_inputs if r.path is not None]
    landfire_resolutions = []
    landfire_subset_path: Path | None = None
    landfire_subset_reused = False

    if layer_key in {"fuel", "canopy"}:
        processed_inputs: list[Path] = []
        for src_path in input_paths:
            lf = resolve_landfire_raster(
                layer_key=layer_key,
                source_path=src_path,
                cache_dir=cache_dir,
                bounds=bounds,
                progress_log=progress_log,
                warnings=notes,
            )
            processed_inputs.append(lf.raster_path)
            landfire_resolutions.append(lf)
            if lf.subset_cache_path:
                subset_candidate = Path(lf.subset_cache_path)
                if subset_candidate.exists():
                    landfire_subset_path = subset_candidate
                    landfire_subset_reused = True
        input_paths = processed_inputs

    if copy_files:
        staged_inputs: list[Path] = []
        for idx, src_path in enumerate(input_paths):
            staged = staging_dir / f"{layer_key}_{idx}_{src_path.name}"
            _stage_file(src_path, staged, copy_files=True)
            staged_inputs.append(staged)
        input_paths = staged_inputs

    if len(input_paths) == 1:
        input_path = input_paths[0]
    else:
        mosaic_path = staging_dir / f"{layer_key}_mosaic{'.tif' if layer_type == 'raster' else '.geojson'}"
        if layer_type == "raster":
            input_path = _mosaic_rasters(input_paths, mosaic_path)
        else:
            input_path = _merge_vectors(input_paths, mosaic_path)
        layer_meta["mosaic_performed"] = True

    output_path = region_dir / filename
    if layer_key in {"fuel", "canopy"} and landfire_subset_path and landfire_subset_path.exists():
        progress_log.append(f"LANDFIRE: reusing cached clipped subset for {layer_key}")
        shutil.copy2(landfire_subset_path, output_path)
        clip_meta = _raster_file_metadata(output_path)
    elif layer_type == "raster":
        progress_log.append(f"LANDFIRE: clipping raster to bbox for {layer_key}" if layer_key in {"fuel", "canopy"} else f"Clipping raster to bbox for {layer_key}")
        clip_meta = _clip_raster_to_bbox(
            input_path,
            output_path,
            bounds,
            compression=raster_compression,
            tile_size=tile_size,
            max_expected_cells=max_expected_cells,
        )
        if layer_key in {"fuel", "canopy"} and landfire_resolutions:
            subset_target_raw = landfire_resolutions[0].subset_cache_path
            if subset_target_raw:
                subset_target = Path(subset_target_raw)
                subset_target.parent.mkdir(parents=True, exist_ok=True)
                if not subset_target.exists():
                    shutil.copy2(output_path, subset_target)
                    progress_log.append(f"LANDFIRE: cached clipped subset for {layer_key} at {subset_target}")
                landfire_subset_path = subset_target
    else:
        clip_meta = _clip_geojson_to_bbox(input_path, output_path, bounds)

    _validate_prepared_layer(output_path, layer_type, bounds)
    layer_meta.update(clip_meta)
    layer_meta.setdefault(
        "acquisition_method",
        "full_download_clip" if layer_meta.get("source_mode") == "remote_url" else "local_existing",
    )
    layer_meta["clipped_to_bbox"] = True
    layer_meta["validation_status"] = "ok"
    layer_meta["bytes_downloaded"] = int(sum(r.bytes_downloaded for r in valid_inputs))
    layer_meta["cache_hit"] = bool(any(r.cache_hit for r in valid_inputs))
    layer_meta["retry_count_used"] = int(sum(r.retry_count_used for r in valid_inputs))
    layer_meta["extraction_performed"] = bool(any(r.extraction_performed for r in valid_inputs))
    extracted_paths = [r.extracted_path for r in valid_inputs if r.extracted_path]
    layer_meta["extracted_path"] = extracted_paths[0] if len(extracted_paths) == 1 else extracted_paths or None
    layer_meta["download_status"] = (
        "cache_hit"
        if valid_inputs and all(r.download_status in {"cache_hit", "not_attempted"} for r in valid_inputs)
        else "ok"
    )
    if discovered_for_layer:
        layer_meta["dataset_source"] = discovered_for_layer[0].dataset_name
        layer_meta["dataset_provider"] = discovered_for_layer[0].dataset_provider
        if not layer_meta.get("dataset_version"):
            layer_meta["dataset_version"] = discovered_for_layer[0].dataset_version
        layer_meta["tile_ids"] = [a.tile_id for a in discovered_for_layer if a.tile_id]
        layer_meta["download_url"] = [a.url for a in discovered_for_layer]
        layer_meta["discovered_source"] = True

    if layer_key in {"fuel", "canopy"} and landfire_resolutions:
        first = landfire_resolutions[0]
        source_url_hint = next((r.source_url for r in valid_inputs if getattr(r, "source_url", None)), None)
        layer_meta.update(
            {
                "landfire_handler_version": LANDFIRE_HANDLER_VERSION,
                "archive_source_url": source_url_hint,
                "archive_cache_path": first.archive_cache_path,
                "archive_size_bytes": first.archive_size_bytes,
                "archive_index_path": first.index_path,
                "extracted_raster_path": first.extracted_raster_path,
                "subset_cache_path": str(landfire_subset_path) if landfire_subset_path else first.subset_cache_path,
                "subset_reused": bool(landfire_subset_reused),
                "cache_hit_subset": bool(landfire_subset_reused),
                "clipping_bbox": {
                    "min_lon": float(bounds["min_lon"]),
                    "min_lat": float(bounds["min_lat"]),
                    "max_lon": float(bounds["max_lon"]),
                    "max_lat": float(bounds["max_lat"]),
                },
                "landfire_layer_type": layer_key,
                "landfire_processing_notes": first.processing_notes or [],
            }
        )
    return filename, layer_meta


def _cleanup_temp(
    *,
    download_dir: Path,
    extraction_dir: Path,
    staging_dir: Path,
    clean_download_cache: bool,
    keep_temp_on_failure: bool,
    had_errors: bool,
) -> None:
    if keep_temp_on_failure and had_errors:
        return
    if clean_download_cache and download_dir.exists():
        shutil.rmtree(download_dir, ignore_errors=True)
    if clean_download_cache and extraction_dir.exists():
        shutil.rmtree(extraction_dir, ignore_errors=True)
    if clean_download_cache and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def prepare_region_layers(
    *,
    region_id: str,
    display_name: str,
    bounds: dict[str, float],
    layer_sources: dict[str, str] | None = None,
    layer_urls: dict[str, str | list[str]] | None = None,
    region_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    crs: str = "EPSG:4326",
    copy_files: bool = False,
    source_metadata: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
    skip_download: bool = False,
    allow_partial: bool = False,
    auto_discover: bool = True,
    download_timeout: float = 45.0,
    download_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
    dry_run: bool = False,
    keep_temp_on_failure: bool = False,
    clean_download_cache: bool = False,
    landfire_only: bool = False,
    raster_compression: str | None = "DEFLATE",
    tile_size: int = 512,
    max_expected_cells: int | None = None,
    source_config: dict[str, Any] | None = None,
    prefer_bbox_downloads: bool = False,
    allow_full_download_fallback: bool = True,
    require_core_layers: bool = True,
    skip_optional_layers: bool = False,
    target_resolution: float | None = None,
) -> dict[str, Any]:
    if not region_id.strip():
        raise ValueError("region_id is required")
    if not display_name.strip():
        raise ValueError("display_name is required")
    for key in ["min_lon", "min_lat", "max_lon", "max_lat"]:
        if key not in bounds:
            raise ValueError(f"bounds must include {key}")

    root = Path(region_data_dir or os.getenv("WF_REGION_DATA_DIR") or DEFAULT_REGION_DATA_DIR).expanduser()
    cache_root = Path(cache_dir or os.getenv("WF_LAYER_DOWNLOAD_CACHE_DIR") or (Path("data") / "cache")).expanduser()
    layer_sources = dict(layer_sources or {})
    layer_urls = dict(layer_urls or {})
    region_dir = root / region_id
    download_dir = region_dir / "_downloads"
    extraction_dir = region_dir / "_extracted"
    staging_dir = region_dir / "_staging"
    if clean_download_cache and cache_root.exists() and not dry_run:
        shutil.rmtree(cache_root, ignore_errors=True)
    if not dry_run:
        cache_root.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        if region_dir.exists() and not force:
            raise ValueError(f"Region directory already exists: {region_dir} (use force=True to overwrite)")
        if region_dir.exists() and force:
            shutil.rmtree(region_dir)
        region_dir.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)
        extraction_dir.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    layers_meta: dict[str, dict[str, Any]] = {}
    metadata = dict(source_metadata or {})
    warnings: list[str] = []
    errors: list[str] = []
    progress_log: list[str] = []
    slope_derived = False
    archives_extracted = False
    download_config = DownloadConfig(
        timeout_seconds=float(download_timeout),
        retries=max(0, int(download_retries)),
        retry_backoff_seconds=max(0.0, float(retry_backoff_seconds)),
    )

    _apply_source_config_acquisition(
        bounds=bounds,
        layer_sources=layer_sources,
        layer_urls=layer_urls,
        source_metadata=metadata,
        source_config=source_config,
        cache_root=cache_root,
        prefer_bbox_downloads=bool(prefer_bbox_downloads),
        allow_full_download_fallback=bool(allow_full_download_fallback),
        target_resolution=target_resolution,
        download_config=download_config,
        notes=warnings,
        progress_log=progress_log,
        skip_optional_layers=bool(skip_optional_layers),
    )

    discovered_assets: dict[str, list[SourceAsset]] = {}
    if auto_discover:
        unresolved_keys = [
            key
            for key in (
                (["fuel", "canopy"] if landfire_only else list(BASE_REQUIRED_KEYS))
                + list(OPTIONAL_LAYER_KEYS)
            )
            if not layer_sources.get(key) and not layer_urls.get(key)
        ]
        if unresolved_keys:
            discovered_all = discover_wildfire_sources(bounds)
            discovered_assets = {k: discovered_all.get(k, []) for k in unresolved_keys}
            for key, assets in discovered_assets.items():
                if assets:
                    warnings.append(f"Auto-discovered {len(assets)} source asset(s) for {key}.")
                    progress_log.append(f"Auto-discovery resolved {len(assets)} source asset(s) for {key}.")

    base_required_keys = ("fuel", "canopy") if landfire_only else BASE_REQUIRED_KEYS

    def _prepare_layer(layer_key: str) -> None:
        nonlocal archives_extracted
        filename, layer_meta = _prepare_one_layer(
            layer_key=layer_key,
            bounds=bounds,
            region_dir=region_dir,
            download_dir=download_dir,
            cache_dir=cache_root,
            extraction_dir=extraction_dir,
            staging_dir=staging_dir,
            layer_sources=layer_sources,
            layer_urls=layer_urls,
            discovered_assets=discovered_assets,
            copy_files=copy_files,
            skip_download=skip_download,
            dry_run=dry_run,
            download_config=download_config,
            source_metadata=metadata,
            notes=warnings,
            progress_log=progress_log,
            raster_compression=raster_compression,
            tile_size=tile_size,
            max_expected_cells=max_expected_cells,
        )
        if filename:
            files[layer_key] = filename
        layers_meta[layer_key] = layer_meta
        if layer_meta.get("extraction_performed"):
            archives_extracted = True
        if layer_meta.get("validation_status") == "error":
            errors.append(f"{layer_key} preparation error")

    for layer_key in base_required_keys:
        try:
            _prepare_layer(layer_key)
        except Exception as exc:
            errors.append(f"{layer_key} preparation failed: {exc}")
            layers_meta[layer_key] = {
                "source_name": layer_key,
                "source_type": "missing",
                "source_mode": "missing",
                "source_url": (layer_urls or {}).get(layer_key),
                "dataset_version": (metadata.get(layer_key, {}) or {}).get("dataset_version"),
                "provider_type": (metadata.get(layer_key, {}) or {}).get("provider_type"),
                "source_endpoint": (metadata.get(layer_key, {}) or {}).get("source_endpoint"),
                "acquisition_method": (metadata.get(layer_key, {}) or {}).get("acquisition_method", "unavailable"),
                "bbox_used": (metadata.get(layer_key, {}) or {}).get("bbox_used"),
                "output_resolution": (metadata.get(layer_key, {}) or {}).get("output_resolution"),
                "bbox_cache_hit": bool((metadata.get(layer_key, {}) or {}).get("bbox_cache_hit", False)),
                "freshness_timestamp": (metadata.get(layer_key, {}) or {}).get("freshness_timestamp"),
                "downloaded_at": None,
                "download_status": "error",
                "bytes_downloaded": 0,
                "extraction_performed": False,
                "extracted_path": None,
                "checksum_status": "not_provided",
                "retry_count_used": 0,
                "timeout_seconds": float(download_timeout),
                "clipped_to_bbox": False,
                "validation_status": "error",
                "notes": str(exc),
            }

    if not landfire_only:
        # Optional slope source; derive from prepared dem when absent.
        try:
            slope_source_path, slope_source_url = _layer_source("slope", layer_sources, layer_urls)
            if slope_source_path or slope_source_url:
                _prepare_layer("slope")
            else:
                if dry_run:
                    dem_ready = layers_meta.get("dem", {}).get("validation_status") in {"dry_run", "ok"}
                    if dem_ready:
                        files["slope"] = STANDARD_LAYER_FILENAMES["slope"]
                        layers_meta["slope"] = {
                            "source_name": "dem.tif",
                            "source_type": "derived_from_dem",
                            "source_mode": "derived",
                            "source_url": None,
                            "dataset_version": (metadata.get("slope", {}) or {}).get("dataset_version"),
                            "freshness_timestamp": (metadata.get("slope", {}) or {}).get("freshness_timestamp"),
                            "downloaded_at": None,
                            "download_status": "dry_run",
                            "bytes_downloaded": 0,
                            "extraction_performed": False,
                            "extracted_path": None,
                            "checksum_status": "not_provided",
                            "retry_count_used": 0,
                            "timeout_seconds": float(download_timeout),
                            "clipped_to_bbox": True,
                            "validation_status": "dry_run",
                            "acquisition_method": "derived",
                            "notes": "Slope will be derived from dem.tif because no slope source was provided.",
                        }
                    else:
                        layers_meta["slope"] = {
                            "source_name": "slope",
                            "source_type": "missing",
                            "source_mode": "missing",
                            "source_url": None,
                            "dataset_version": (metadata.get("slope", {}) or {}).get("dataset_version"),
                            "freshness_timestamp": (metadata.get("slope", {}) or {}).get("freshness_timestamp"),
                            "downloaded_at": None,
                            "download_status": "missing",
                            "bytes_downloaded": 0,
                            "extraction_performed": False,
                            "extracted_path": None,
                            "checksum_status": "not_provided",
                            "retry_count_used": 0,
                            "timeout_seconds": float(download_timeout),
                            "clipped_to_bbox": False,
                            "validation_status": "missing",
                            "acquisition_method": "unavailable",
                            "notes": "Slope cannot be derived in dry-run because dem is unavailable.",
                        }
                        warnings.append("Slope dry-run skipped because dem was not available for derivation.")
                else:
                    dem_prepared = region_dir / STANDARD_LAYER_FILENAMES["dem"]
                    slope_out = region_dir / STANDARD_LAYER_FILENAMES["slope"]
                    if not dem_prepared.exists():
                        raise ValueError("Cannot derive slope without prepared dem.tif")
                    slope_meta = _derive_slope_from_dem(dem_prepared, slope_out)
                    _validate_prepared_layer(slope_out, "raster", bounds)
                    files["slope"] = STANDARD_LAYER_FILENAMES["slope"]
                    layers_meta["slope"] = {
                        "source_name": "dem.tif",
                        "source_type": "derived_from_dem",
                        "source_mode": "derived",
                        "source_url": None,
                        "dataset_version": (metadata.get("slope", {}) or {}).get("dataset_version"),
                        "freshness_timestamp": (metadata.get("slope", {}) or {}).get("freshness_timestamp"),
                        "downloaded_at": None,
                        "download_status": "derived",
                        "bytes_downloaded": 0,
                        "extraction_performed": False,
                        "extracted_path": None,
                        "checksum_status": "not_provided",
                        "retry_count_used": 0,
                        "timeout_seconds": float(download_timeout),
                        "clipped_to_bbox": True,
                        "validation_status": "ok",
                        "acquisition_method": "derived",
                        "notes": "Slope derived from prepared DEM because no slope source was provided.",
                        **slope_meta,
                    }
                    slope_derived = True
        except Exception as exc:
            errors.append(f"slope preparation failed: {exc}")
            layers_meta["slope"] = {
                "source_name": "slope",
                "source_type": "missing",
                "source_mode": "missing",
                "source_url": (layer_urls or {}).get("slope"),
                "dataset_version": (metadata.get("slope", {}) or {}).get("dataset_version"),
                "freshness_timestamp": (metadata.get("slope", {}) or {}).get("freshness_timestamp"),
                "downloaded_at": None,
                "download_status": "error",
                "bytes_downloaded": 0,
                "extraction_performed": False,
                "extracted_path": None,
                "checksum_status": "not_provided",
                "retry_count_used": 0,
                "timeout_seconds": float(download_timeout),
                "clipped_to_bbox": False,
                "validation_status": "error",
                "acquisition_method": "unavailable",
                "notes": str(exc),
            }

    if not skip_optional_layers:
        for optional_layer in OPTIONAL_LAYER_KEYS:
            source_path, source_url = _layer_source(optional_layer, layer_sources, layer_urls)
            if not source_path and not source_url:
                continue
            try:
                _prepare_layer(optional_layer)
            except Exception as exc:
                errors.append(f"{optional_layer} optional layer preparation failed: {exc}")
                warnings.append(f"{optional_layer} optional layer preparation failed: {exc}")
                layers_meta[optional_layer] = {
                    "source_name": optional_layer,
                    "source_type": "missing",
                    "source_mode": "missing",
                    "source_url": source_url,
                    "dataset_version": (metadata.get(optional_layer, {}) or {}).get("dataset_version"),
                    "provider_type": (metadata.get(optional_layer, {}) or {}).get("provider_type"),
                    "source_endpoint": (metadata.get(optional_layer, {}) or {}).get("source_endpoint"),
                    "acquisition_method": (metadata.get(optional_layer, {}) or {}).get(
                        "acquisition_method", "unavailable"
                    ),
                    "bbox_used": (metadata.get(optional_layer, {}) or {}).get("bbox_used"),
                    "output_resolution": (metadata.get(optional_layer, {}) or {}).get("output_resolution"),
                    "bbox_cache_hit": bool((metadata.get(optional_layer, {}) or {}).get("bbox_cache_hit", False)),
                    "freshness_timestamp": (metadata.get(optional_layer, {}) or {}).get("freshness_timestamp"),
                    "downloaded_at": None,
                    "download_status": "error",
                    "bytes_downloaded": 0,
                    "extraction_performed": False,
                    "extracted_path": None,
                    "checksum_status": "not_provided",
                    "retry_count_used": 0,
                    "timeout_seconds": float(download_timeout),
                    "clipped_to_bbox": False,
                    "validation_status": "error",
                    "notes": str(exc),
                }
    else:
        warnings.append("Optional layers skipped by configuration (--skip-optional-layers).")

    requested_layers = set(layer_sources.keys()) | set(layer_urls.keys())
    prepared_layers = sorted([k for k, v in layers_meta.items() if v.get("validation_status") == "ok"])
    skipped_layers = sorted([k for k, v in layers_meta.items() if v.get("validation_status") == "missing"])
    failed_layers = sorted([k for k, v in layers_meta.items() if v.get("validation_status") == "error"])
    attempted_layers = sorted(
        [
            k
            for k, v in layers_meta.items()
            if v.get("validation_status") in {"ok", "error", "dry_run"} and v.get("source_mode") != "missing"
        ]
    )
    discovered_layers = sorted([k for k, v in layers_meta.items() if bool(v.get("discovered_source"))])
    unsupported_auto_discovery_layers = sorted(
        [
            k
            for k, v in layers_meta.items()
            if auto_discover
            and v.get("validation_status") == "missing"
            and k not in requested_layers
            and not bool(v.get("discovered_source"))
        ]
    )

    if landfire_only:
        explicit_landfire = [k for k in ("fuel", "canopy") if k in requested_layers]
        run_required_layers = explicit_landfire if explicit_landfire else ["fuel", "canopy"]
        minimum_required_layers = explicit_landfire if explicit_landfire else ["fuel"]
    else:
        run_required_layers = list(REQUIRED_REGION_FILES) if require_core_layers else list(MINIMUM_REQUIRED_KEYS)
        minimum_required_layers = list(MINIMUM_REQUIRED_KEYS)

    full_required_missing = [k for k in run_required_layers if k not in files]
    minimum_required_missing = [k for k in minimum_required_layers if k not in files]
    required_blockers = minimum_required_missing if (allow_partial or dry_run) else full_required_missing

    for layer_key in skipped_layers:
        if layer_key in unsupported_auto_discovery_layers:
            warnings.append(
                f"{layer_key} skipped: no source provided and auto-discovery support is unavailable or unresolved."
            )
        elif layer_key not in requested_layers:
            warnings.append(f"{layer_key} skipped: no source provided.")

    if required_blockers:
        blocker_scope = "minimum" if allow_partial else "full"
        for layer_key in required_blockers:
            warnings.append(f"Required ({blocker_scope}) layer missing: {layer_key}.")
            if not dry_run:
                errors.append(f"{layer_key} required output missing after preparation")

    if dry_run:
        preparation_status = "dry_run"
        final_status = "dry_run_ready" if not required_blockers and not errors else "dry_run_partial"
    elif errors and allow_partial and not minimum_required_missing:
        preparation_status = "partial"
        final_status = "partial"
    elif errors:
        preparation_status = "failed"
        final_status = "failed"
    elif allow_partial and full_required_missing:
        preparation_status = "partial"
        final_status = "partial"
    else:
        preparation_status = "prepared"
        final_status = "success"

    if preparation_status == "partial":
        warnings.append(
            "Partial preparation completed. Missing/failed layers are listed in failed_layers and errors."
        )

    manifest = {
        "region_id": region_id,
        "display_name": display_name,
        "prepared_at": _now(),
        "preparation_status": preparation_status,
        "final_status": final_status,
        "crs": crs,
        "bounds": {
            "min_lon": float(bounds["min_lon"]),
            "min_lat": float(bounds["min_lat"]),
            "max_lon": float(bounds["max_lon"]),
            "max_lat": float(bounds["max_lat"]),
        },
        "files": files,
        "layers": layers_meta,
        "prepared_layers": prepared_layers,
        "attempted_layers": attempted_layers,
        "discovered_layers": discovered_layers,
        "skipped_layers": skipped_layers,
        "unsupported_auto_discovery_layers": unsupported_auto_discovery_layers,
        "failed_layers": failed_layers,
        "required_blockers": required_blockers,
        "minimum_required_missing": minimum_required_missing,
        "full_required_missing": full_required_missing,
        "warnings": sorted(set(warnings)),
        "errors": sorted(set(errors)),
        "download_config": {
            "timeout_seconds": download_config.timeout_seconds,
            "retries": download_config.retries,
            "retry_backoff_seconds": download_config.retry_backoff_seconds,
            "skip_download": bool(skip_download),
            "auto_discover": bool(auto_discover),
            "prefer_bbox_downloads": bool(prefer_bbox_downloads),
            "allow_full_download_fallback": bool(allow_full_download_fallback),
            "require_core_layers": bool(require_core_layers),
            "skip_optional_layers": bool(skip_optional_layers),
            "target_resolution": float(target_resolution) if target_resolution is not None else None,
            "source_config_used": bool(source_config),
            "cache_dir": str(cache_root),
            "landfire_only": bool(landfire_only),
            "raster_compression": raster_compression,
            "tile_size": int(tile_size),
            "max_expected_cells": int(max_expected_cells) if max_expected_cells else None,
        },
        "slope_derived": slope_derived,
        "archives_extracted": archives_extracted,
        "dry_run": bool(dry_run),
        "notes": [
            "Regional preparation runs offline/admin-only; runtime assessment does not download large GIS data.",
        ],
        "progress_log": progress_log,
    }

    if not dry_run:
        with open(region_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        _cleanup_temp(
            download_dir=download_dir,
            extraction_dir=extraction_dir,
            staging_dir=staging_dir,
            clean_download_cache=clean_download_cache,
            keep_temp_on_failure=keep_temp_on_failure,
            had_errors=bool(errors),
        )

    if errors and not allow_partial and not dry_run:
        raise ValueError("Region preparation failed: " + "; ".join(sorted(set(errors))))

    return manifest
