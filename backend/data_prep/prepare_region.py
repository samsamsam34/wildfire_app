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
    "fire_perimeters": "vector",
    "building_footprints": "vector",
}

AUTOMATION_NOTES = {
    "dem": "Supports local file or URL source. Full source-catalog automation is deferred.",
    "fire_perimeters": "Supports local GeoJSON or URL source. Direct NIFC catalog sync is deferred.",
    "building_footprints": "Supports local GeoJSON or URL source. Direct Microsoft tile index sync is deferred.",
    "fuel": "Supports local/URL source prep. Full LANDFIRE discovery/download automation is deferred.",
    "canopy": "Supports local/URL source prep. Full LANDFIRE discovery/download automation is deferred.",
}

OPTIONAL_LAYER_KEYS = ("burn_probability", "wildfire_hazard", "moisture", "aspect")
BASE_REQUIRED_KEYS = ("dem", "fuel", "canopy", "fire_perimeters", "building_footprints")


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


def _select_archive_member(archive_path: Path, layer_key: str, layer_type: str) -> str:
    suffixes = (".tif", ".tiff") if layer_type == "raster" else (".geojson", ".json")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(name for name in zf.namelist() if not name.endswith("/"))
    candidates = [m for m in members if Path(m).suffix.lower() in suffixes]
    if not candidates:
        raise ValueError(f"{layer_key} archive has no usable {layer_type} files")
    return candidates[0]


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

    def resolve(
        self,
        *,
        source_path: str | None,
        source_url: str | None,
        layer_type: str,
        download_dir: Path,
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
            if path.suffix.lower() == ".zip":
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
            download_dir.mkdir(parents=True, exist_ok=True)
            parsed = urllib.parse.urlparse(source_url)
            filename = Path(parsed.path).name or f"{self.layer_key}.dat"
            target = download_dir / filename
            bytes_downloaded, retry_count_used = _download_with_retry(source_url, target, download_config)
            _sanity_check_source_file(target, self.layer_key, layer_type)

            final_path = target
            extraction_performed = False
            extracted_path = None
            if target.suffix.lower() == ".zip":
                final_path = _extract_archive_layer(target, self.layer_key, layer_type, extraction_dir)
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
                download_status="ok",
                extraction_performed=extraction_performed,
                extracted_path=extracted_path,
                checksum_status=checksum_status,
                retry_count_used=retry_count_used,
                timeout_seconds=timeout_seconds,
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


class CanopySourceAdapter(LayerSourceAdapter):
    layer_key = "canopy"


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


def _clip_raster_to_bbox(src: Path, dest: Path, bounds: dict[str, float]) -> dict[str, Any]:
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
        with rasterio.open(dest, "w", **profile) as out:
            out.write(arr)

    with rasterio.open(dest) as out_ds:
        return {
            "crs": str(out_ds.crs),
            "resolution": [abs(out_ds.transform.a), abs(out_ds.transform.e)],
            "bounds": list(out_ds.bounds),
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
    layer_urls: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    source_path = (layer_sources or {}).get(layer_key)
    source_url = (layer_urls or {}).get(layer_key)
    return source_path, source_url


def _base_layer_meta(
    *,
    layer_key: str,
    resolved: PreparedLayerInput,
    source_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    meta = source_metadata.get(layer_key, {}) or {}
    return {
        "source_name": resolved.source_name,
        "source_type": resolved.source_type,
        "source_mode": resolved.source_mode,
        "source_url": resolved.source_url,
        "dataset_version": meta.get("dataset_version"),
        "freshness_timestamp": meta.get("freshness_timestamp"),
        "downloaded_at": resolved.downloaded_at,
        "download_status": resolved.download_status,
        "bytes_downloaded": int(resolved.bytes_downloaded or 0),
        "extraction_performed": bool(resolved.extraction_performed),
        "extracted_path": resolved.extracted_path,
        "checksum_status": resolved.checksum_status,
        "retry_count_used": int(resolved.retry_count_used or 0),
        "timeout_seconds": float(resolved.timeout_seconds or 0.0),
        "clipped_to_bbox": False,
        "validation_status": "missing",
        "notes": AUTOMATION_NOTES.get(layer_key),
    }


def _prepare_one_layer(
    *,
    layer_key: str,
    bounds: dict[str, float],
    region_dir: Path,
    download_dir: Path,
    extraction_dir: Path,
    layer_sources: dict[str, str] | None,
    layer_urls: dict[str, str] | None,
    copy_files: bool,
    skip_download: bool,
    dry_run: bool,
    download_config: DownloadConfig,
    source_metadata: dict[str, dict[str, Any]],
    notes: list[str],
) -> tuple[str | None, dict[str, Any]]:
    layer_type = LAYER_TYPES[layer_key]
    adapter = ADAPTERS.get(layer_key, GenericSourceAdapter(layer_key))
    source_path, source_url = _layer_source(layer_key, layer_sources, layer_urls)
    checksum = (source_metadata.get(layer_key, {}) or {}).get("checksum")
    resolved = adapter.resolve(
        source_path=source_path,
        source_url=source_url,
        layer_type=layer_type,
        download_dir=download_dir,
        extraction_dir=extraction_dir,
        skip_download=skip_download,
        dry_run=dry_run,
        download_config=download_config,
        checksum=checksum,
    )
    for warning in resolved.warnings or []:
        notes.append(warning)

    layer_meta = _base_layer_meta(layer_key=layer_key, resolved=resolved, source_metadata=source_metadata)
    filename = STANDARD_LAYER_FILENAMES[layer_key]
    if dry_run:
        if resolved.source_mode == "missing":
            layer_meta["validation_status"] = "missing"
            return None, layer_meta
        layer_meta["validation_status"] = "dry_run"
        layer_meta["clipped_to_bbox"] = True
        return filename, layer_meta

    if resolved.path is None:
        return None, layer_meta

    if copy_files and resolved.source_mode == "local_file":
        staged = download_dir / resolved.path.name
        _stage_file(resolved.path, staged, copy_files=True)
        input_path = staged
    else:
        input_path = resolved.path

    output_path = region_dir / filename
    if layer_type == "raster":
        clip_meta = _clip_raster_to_bbox(input_path, output_path, bounds)
    else:
        clip_meta = _clip_geojson_to_bbox(input_path, output_path, bounds)

    _validate_prepared_layer(output_path, layer_type, bounds)
    layer_meta.update(clip_meta)
    layer_meta["clipped_to_bbox"] = True
    layer_meta["validation_status"] = "ok"
    return filename, layer_meta


def _cleanup_temp(
    *,
    download_dir: Path,
    extraction_dir: Path,
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


def prepare_region_layers(
    *,
    region_id: str,
    display_name: str,
    bounds: dict[str, float],
    layer_sources: dict[str, str] | None = None,
    layer_urls: dict[str, str] | None = None,
    region_data_dir: str | Path | None = None,
    crs: str = "EPSG:4326",
    copy_files: bool = False,
    source_metadata: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
    skip_download: bool = False,
    allow_partial: bool = False,
    download_timeout: float = 45.0,
    download_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
    dry_run: bool = False,
    keep_temp_on_failure: bool = False,
    clean_download_cache: bool = True,
) -> dict[str, Any]:
    if not region_id.strip():
        raise ValueError("region_id is required")
    if not display_name.strip():
        raise ValueError("display_name is required")
    for key in ["min_lon", "min_lat", "max_lon", "max_lat"]:
        if key not in bounds:
            raise ValueError(f"bounds must include {key}")

    root = Path(region_data_dir or os.getenv("WF_REGION_DATA_DIR") or DEFAULT_REGION_DATA_DIR).expanduser()
    region_dir = root / region_id
    download_dir = region_dir / "_downloads"
    extraction_dir = region_dir / "_extracted"
    if not dry_run:
        if region_dir.exists() and not force:
            raise ValueError(f"Region directory already exists: {region_dir} (use force=True to overwrite)")
        if region_dir.exists() and force:
            shutil.rmtree(region_dir)
        region_dir.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)
        extraction_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    layers_meta: dict[str, dict[str, Any]] = {}
    metadata = source_metadata or {}
    warnings: list[str] = []
    errors: list[str] = []
    slope_derived = False
    archives_extracted = False
    download_config = DownloadConfig(
        timeout_seconds=float(download_timeout),
        retries=max(0, int(download_retries)),
        retry_backoff_seconds=max(0.0, float(retry_backoff_seconds)),
    )

    def _prepare_layer(layer_key: str) -> None:
        nonlocal archives_extracted
        filename, layer_meta = _prepare_one_layer(
            layer_key=layer_key,
            bounds=bounds,
            region_dir=region_dir,
            download_dir=download_dir,
            extraction_dir=extraction_dir,
            layer_sources=layer_sources,
            layer_urls=layer_urls,
            copy_files=copy_files,
            skip_download=skip_download,
            dry_run=dry_run,
            download_config=download_config,
            source_metadata=metadata,
            notes=warnings,
        )
        if filename:
            files[layer_key] = filename
        layers_meta[layer_key] = layer_meta
        if layer_meta.get("extraction_performed"):
            archives_extracted = True
        if layer_meta.get("validation_status") in {"missing", "error"}:
            errors.append(f"{layer_key} preparation {layer_meta.get('validation_status')}")

    for layer_key in BASE_REQUIRED_KEYS:
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

    # Optional slope source; derive from prepared dem when absent.
    try:
        slope_source_path, slope_source_url = _layer_source("slope", layer_sources, layer_urls)
        if slope_source_path or slope_source_url:
            _prepare_layer("slope")
        else:
            if dry_run:
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
                    "notes": "Slope will be derived from dem.tif because no slope source was provided.",
                }
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
            "notes": str(exc),
        }

    for optional_layer in OPTIONAL_LAYER_KEYS:
        source_path, source_url = _layer_source(optional_layer, layer_sources, layer_urls)
        if not source_path and not source_url:
            continue
        try:
            _prepare_layer(optional_layer)
        except Exception as exc:
            warnings.append(f"{optional_layer} optional layer preparation failed: {exc}")
            layers_meta[optional_layer] = {
                "source_name": optional_layer,
                "source_type": "missing",
                "source_mode": "missing",
                "source_url": source_url,
                "dataset_version": (metadata.get(optional_layer, {}) or {}).get("dataset_version"),
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

    required_missing = [k for k in REQUIRED_REGION_FILES if k not in files]
    if required_missing:
        errors.extend([f"{k} output missing after preparation" for k in required_missing])

    prepared_layers = sorted([k for k, v in layers_meta.items() if v.get("validation_status") == "ok"])
    skipped_layers = sorted([k for k, v in layers_meta.items() if v.get("validation_status") == "missing"])
    failed_layers = sorted([k for k, v in layers_meta.items() if v.get("validation_status") == "error"])

    if dry_run:
        preparation_status = "dry_run"
        final_status = "failed" if required_missing else "success"
    elif errors and allow_partial:
        preparation_status = "partial"
        final_status = "partial"
    elif errors:
        preparation_status = "failed"
        final_status = "failed"
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
        "skipped_layers": skipped_layers,
        "failed_layers": failed_layers,
        "warnings": sorted(set(warnings)),
        "errors": sorted(set(errors)),
        "download_config": {
            "timeout_seconds": download_config.timeout_seconds,
            "retries": download_config.retries,
            "retry_backoff_seconds": download_config.retry_backoff_seconds,
            "skip_download": bool(skip_download),
        },
        "slope_derived": slope_derived,
        "archives_extracted": archives_extracted,
        "dry_run": bool(dry_run),
        "notes": [
            "Regional preparation runs offline/admin-only; runtime assessment does not download large GIS data.",
        ],
    }

    if not dry_run:
        with open(region_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        _cleanup_temp(
            download_dir=download_dir,
            extraction_dir=extraction_dir,
            clean_download_cache=clean_download_cache,
            keep_temp_on_failure=keep_temp_on_failure,
            had_errors=bool(errors),
        )

    if errors and not allow_partial and not dry_run:
        raise ValueError("Region preparation failed: " + "; ".join(sorted(set(errors))))

    return manifest
