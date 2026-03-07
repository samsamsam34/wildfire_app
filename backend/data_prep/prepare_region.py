from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - handled via runtime validation
    np = None
    rasterio = None
    from_bounds = None
    transform_bounds = None

try:
    from shapely.geometry import box, mapping, shape
except Exception:  # pragma: no cover - handled via runtime validation
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


@dataclass
class PreparedLayerInput:
    path: Path | None
    source_name: str
    source_type: str
    source_url: str | None = None
    downloaded_at: str | None = None
    warnings: list[str] | None = None


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


def _bbox_polygon(bounds: dict[str, float]):
    if box is None:
        raise ValueError("shapely is required for vector clipping.")
    return box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])


class LayerSourceAdapter:
    layer_key = "base"

    def resolve(
        self,
        *,
        source_path: str | None,
        source_url: str | None,
        download_dir: Path,
        skip_download: bool,
    ) -> PreparedLayerInput:
        warnings: list[str] = []
        if source_path:
            path = Path(source_path).expanduser()
            if not path.exists():
                raise ValueError(f"{self.layer_key} local source not found: {path}")
            return PreparedLayerInput(
                path=path,
                source_name=path.name,
                source_type="local_file",
                warnings=warnings,
            )

        if source_url:
            if skip_download:
                warnings.append(f"{self.layer_key} download skipped by --skip-download.")
                return PreparedLayerInput(
                    path=None,
                    source_name=self.layer_key,
                    source_type="missing",
                    source_url=source_url,
                    warnings=warnings,
                )
            downloaded = self._download(source_url, download_dir)
            return PreparedLayerInput(
                path=downloaded,
                source_name=downloaded.name,
                source_type="downloaded_url",
                source_url=source_url,
                downloaded_at=datetime.now(tz=timezone.utc).isoformat(),
                warnings=warnings,
            )

        warnings.append(
            f"No source provided for {self.layer_key}. {AUTOMATION_NOTES.get(self.layer_key, 'Provide a local source file or URL.')}"
        )
        return PreparedLayerInput(
            path=None,
            source_name=self.layer_key,
            source_type="missing",
            warnings=warnings,
        )

    def _download(self, source_url: str, download_dir: Path) -> Path:
        download_dir.mkdir(parents=True, exist_ok=True)
        parsed = urllib.parse.urlparse(source_url)
        filename = Path(parsed.path).name or f"{self.layer_key}.dat"
        target = download_dir / filename
        with urllib.request.urlopen(source_url) as response, open(target, "wb") as out:
            shutil.copyfileobj(response, out)
        return target


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
    if rasterio is None or np is None or from_bounds is None or transform_bounds is None:
        raise ValueError("rasterio and numpy are required for raster preparation.")
    with rasterio.open(src) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {src}")
        bbox_ds = transform_bounds("EPSG:4326", ds.crs, bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])
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
    if rasterio is None or np is None:
        raise ValueError("rasterio and numpy are required to derive slope from DEM.")
    with rasterio.open(dem_path) as dem_ds:
        arr = dem_ds.read(1, masked=True).filled(np.nan).astype("float32")
        nodata = dem_ds.nodata

        x_res = abs(dem_ds.transform.a) or 1.0
        y_res = abs(dem_ds.transform.e) or 1.0
        grad_y, grad_x = np.gradient(arr, y_res, x_res)
        slope_rad = np.arctan(np.sqrt((grad_x * grad_x) + (grad_y * grad_y)))
        slope_deg = np.degrees(slope_rad)
        slope_deg = slope_deg.astype("float32")
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
    if shape is None or mapping is None:
        raise ValueError("shapely is required for vector clipping.")
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

    out_payload = {
        "type": "FeatureCollection",
        "features": clipped_features,
    }
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
        if rasterio is None or transform_bounds is None or box is None:
            raise ValueError("rasterio and shapely are required for raster validation.")
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
            raster_box = box(*ds.bounds)
            if not raster_box.intersects(box(*bbox_ds)):
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


def _prepare_one_layer(
    *,
    layer_key: str,
    bounds: dict[str, float],
    region_dir: Path,
    download_dir: Path,
    layer_sources: dict[str, str] | None,
    layer_urls: dict[str, str] | None,
    copy_files: bool,
    skip_download: bool,
    source_metadata: dict[str, dict[str, Any]],
    notes: list[str],
) -> tuple[str | None, dict[str, Any]]:
    adapter = ADAPTERS.get(layer_key, GenericSourceAdapter(layer_key))
    source_path, source_url = _layer_source(layer_key, layer_sources, layer_urls)
    resolved = adapter.resolve(
        source_path=source_path,
        source_url=source_url,
        download_dir=download_dir,
        skip_download=skip_download,
    )
    for warning in resolved.warnings or []:
        notes.append(warning)

    filename = STANDARD_LAYER_FILENAMES[layer_key]
    output_path = region_dir / filename
    layer_meta = {
        "source_name": resolved.source_name,
        "source_type": resolved.source_type,
        "source_url": resolved.source_url,
        "dataset_version": (source_metadata.get(layer_key, {}) or {}).get("dataset_version"),
        "freshness_timestamp": (source_metadata.get(layer_key, {}) or {}).get("freshness_timestamp"),
        "downloaded_at": resolved.downloaded_at,
        "clipped_to_bbox": False,
        "validation_status": "missing",
        "notes": AUTOMATION_NOTES.get(layer_key),
    }

    if resolved.path is None:
        return None, layer_meta

    if copy_files and resolved.source_type == "local_file":
        staged = download_dir / resolved.path.name
        _stage_file(resolved.path, staged, copy_files=True)
        input_path = staged
    else:
        input_path = resolved.path

    layer_type = LAYER_TYPES[layer_key]
    if layer_type == "raster":
        clip_meta = _clip_raster_to_bbox(input_path, output_path, bounds)
    else:
        clip_meta = _clip_geojson_to_bbox(input_path, output_path, bounds)
    _validate_prepared_layer(output_path, layer_type, bounds)

    layer_meta.update(clip_meta)
    layer_meta["clipped_to_bbox"] = True
    layer_meta["validation_status"] = "ok"
    return filename, layer_meta


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
    if region_dir.exists() and not force:
        raise ValueError(f"Region directory already exists: {region_dir} (use force=True to overwrite)")
    if region_dir.exists() and force:
        shutil.rmtree(region_dir)
    region_dir.mkdir(parents=True, exist_ok=True)
    download_dir = region_dir / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    layers_meta: dict[str, dict[str, Any]] = {}
    metadata = source_metadata or {}
    warnings: list[str] = []
    errors: list[str] = []

    for layer_key in ["dem", "fuel", "canopy", "fire_perimeters", "building_footprints"]:
        try:
            filename, layer_meta = _prepare_one_layer(
                layer_key=layer_key,
                bounds=bounds,
                region_dir=region_dir,
                download_dir=download_dir,
                layer_sources=layer_sources,
                layer_urls=layer_urls,
                copy_files=copy_files,
                skip_download=skip_download,
                source_metadata=metadata,
                notes=warnings,
            )
            if filename:
                files[layer_key] = filename
            layers_meta[layer_key] = layer_meta
            if not filename:
                errors.append(f"{layer_key} source unavailable")
        except Exception as exc:
            errors.append(f"{layer_key} preparation failed: {exc}")
            layers_meta[layer_key] = {
                "source_name": layer_key,
                "source_type": "missing",
                "source_url": (layer_urls or {}).get(layer_key),
                "dataset_version": (metadata.get(layer_key, {}) or {}).get("dataset_version"),
                "freshness_timestamp": (metadata.get(layer_key, {}) or {}).get("freshness_timestamp"),
                "downloaded_at": None,
                "clipped_to_bbox": False,
                "validation_status": "error",
                "notes": str(exc),
            }

    # Optional slope source; auto-derive from prepared DEM when omitted.
    try:
        slope_source_path, slope_source_url = _layer_source("slope", layer_sources, layer_urls)
        if slope_source_path or slope_source_url:
            filename, layer_meta = _prepare_one_layer(
                layer_key="slope",
                bounds=bounds,
                region_dir=region_dir,
                download_dir=download_dir,
                layer_sources=layer_sources,
                layer_urls=layer_urls,
                copy_files=copy_files,
                skip_download=skip_download,
                source_metadata=metadata,
                notes=warnings,
            )
            if filename:
                files["slope"] = filename
            layers_meta["slope"] = layer_meta
            if not filename:
                errors.append("slope source unavailable")
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
                "source_url": None,
                "dataset_version": (metadata.get("slope", {}) or {}).get("dataset_version"),
                "freshness_timestamp": (metadata.get("slope", {}) or {}).get("freshness_timestamp"),
                "downloaded_at": None,
                "clipped_to_bbox": True,
                "validation_status": "ok",
                "notes": "Slope derived from prepared DEM because no slope source was provided.",
                **slope_meta,
            }
    except Exception as exc:
        errors.append(f"slope preparation failed: {exc}")
        layers_meta["slope"] = {
            "source_name": "slope",
            "source_type": "missing",
            "source_url": (layer_urls or {}).get("slope"),
            "dataset_version": (metadata.get("slope", {}) or {}).get("dataset_version"),
            "freshness_timestamp": (metadata.get("slope", {}) or {}).get("freshness_timestamp"),
            "downloaded_at": None,
            "clipped_to_bbox": False,
            "validation_status": "error",
            "notes": str(exc),
        }

    # Optional extended layers (best-effort; not required for prepared-region validation today).
    for optional_layer in ["burn_probability", "wildfire_hazard", "moisture", "aspect"]:
        source_path, source_url = _layer_source(optional_layer, layer_sources, layer_urls)
        if not source_path and not source_url:
            continue
        try:
            filename, layer_meta = _prepare_one_layer(
                layer_key=optional_layer,
                bounds=bounds,
                region_dir=region_dir,
                download_dir=download_dir,
                layer_sources=layer_sources,
                layer_urls=layer_urls,
                copy_files=copy_files,
                skip_download=skip_download,
                source_metadata=metadata,
                notes=warnings,
            )
            if filename:
                files[optional_layer] = filename
            layers_meta[optional_layer] = layer_meta
        except Exception as exc:
            warnings.append(f"{optional_layer} optional layer preparation failed: {exc}")
            layers_meta[optional_layer] = {
                "source_name": optional_layer,
                "source_type": "missing",
                "source_url": source_url,
                "dataset_version": (metadata.get(optional_layer, {}) or {}).get("dataset_version"),
                "freshness_timestamp": (metadata.get(optional_layer, {}) or {}).get("freshness_timestamp"),
                "downloaded_at": None,
                "clipped_to_bbox": False,
                "validation_status": "error",
                "notes": str(exc),
            }

    required_missing = [k for k in REQUIRED_REGION_FILES if k not in files]
    if required_missing:
        errors.extend([f"{k} output missing after preparation" for k in required_missing])

    manifest = {
        "region_id": region_id,
        "display_name": display_name,
        "prepared_at": datetime.now(tz=timezone.utc).isoformat(),
        "preparation_status": "prepared" if not errors else ("partial" if allow_partial else "failed"),
        "crs": crs,
        "bounds": {
            "min_lon": float(bounds["min_lon"]),
            "min_lat": float(bounds["min_lat"]),
            "max_lon": float(bounds["max_lon"]),
            "max_lat": float(bounds["max_lat"]),
        },
        "files": files,
        "layers": layers_meta,
        "warnings": sorted(set(warnings)),
        "errors": sorted(set(errors)),
    }

    with open(region_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    if errors and not allow_partial:
        raise ValueError("Region preparation failed: " + "; ".join(sorted(set(errors))))

    return manifest
