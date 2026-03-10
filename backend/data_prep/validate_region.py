from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.region_registry import (
    REQUIRED_REGION_FILES,
    find_region_for_point,
    load_region_manifest,
    region_contains_point,
    resolve_region_file,
    validate_region_files,
)

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - optional dependency in constrained envs
    rasterio = None
    transform_bounds = None

try:
    from shapely.geometry import box, shape
except Exception:  # pragma: no cover - optional dependency in constrained envs
    box = None
    shape = None


RUNTIME_LAYER_MAP = {
    "dem": "dem",
    "slope": "slope",
    "fuel": "fuel",
    "canopy": "canopy",
    "fire_perimeters": "fire_perimeters",
    "building_footprints": "building_footprints",
}

VECTOR_KEYS = {"fire_perimeters", "building_footprints"}
OPTIONAL_REGION_LAYER_KEYS = {
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "roads",
    "fema_structures",
    "building_footprints_overture",
    "parcel_polygons",
    "parcel_address_points",
}


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _coerce_bounds(bounds: Any) -> tuple[float, float, float, float] | None:
    if isinstance(bounds, dict):
        keys = ("min_lon", "min_lat", "max_lon", "max_lat")
        if not all(k in bounds for k in keys):
            return None
        try:
            return (
                float(bounds["min_lon"]),
                float(bounds["min_lat"]),
                float(bounds["max_lon"]),
                float(bounds["max_lat"]),
            )
        except (TypeError, ValueError):
            return None
    return None


def _validate_raster_layer(path: Path, bounds: tuple[float, float, float, float]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if rasterio is None or transform_bounds is None or box is None:
        warnings.append("Raster validation downgraded: rasterio/shapely not available.")
        return errors, warnings

    with rasterio.open(path) as ds:
        if ds.crs is None:
            errors.append(f"Raster has no CRS: {path}")
            return errors, warnings
        if ds.width <= 0 or ds.height <= 0:
            errors.append(f"Raster has invalid dimensions: {path}")
            return errors, warnings
        layer_bounds = ds.bounds
        bbox_ds = transform_bounds(
            "EPSG:4326",
            ds.crs,
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
        )
        if not box(*layer_bounds).intersects(box(*bbox_ds)):
            errors.append(f"Raster does not intersect manifest bbox: {path}")
    return errors, warnings


def _validate_vector_layer(path: Path, bounds: tuple[float, float, float, float]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", []) if isinstance(payload, dict) else []
    if not isinstance(features, list):
        errors.append(f"Vector payload is not a feature collection: {path}")
        return errors, warnings
    if not features:
        errors.append(f"Vector has no features: {path}")
        return errors, warnings
    if box is None or shape is None:
        warnings.append("Vector geometry intersection validation downgraded: shapely not available.")
        return errors, warnings

    bbox_poly = box(bounds[0], bounds[1], bounds[2], bounds[3])
    intersects = False
    invalid_count = 0
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if not geom:
            continue
        try:
            shp = shape(geom)
            if not shp.is_valid:
                shp = shp.buffer(0)
            if shp.is_valid and shp.intersects(bbox_poly):
                intersects = True
                break
        except Exception:
            invalid_count += 1
            continue
    if invalid_count:
        warnings.append(f"Vector contains {invalid_count} invalid/unreadable geometry item(s): {path.name}")
    if not intersects:
        errors.append(f"Vector does not intersect manifest bbox: {path}")
    return errors, warnings


def _validate_layer_openable_and_intersects(
    layer_key: str,
    layer_path: Path,
    bounds: tuple[float, float, float, float],
) -> tuple[list[str], list[str]]:
    if layer_key in VECTOR_KEYS:
        return _validate_vector_layer(layer_path, bounds)
    return _validate_raster_layer(layer_path, bounds)


def _run_sample_smoke_test(
    *,
    manifest: dict[str, Any],
    base_dir: str | None,
    sample_lat: float,
    sample_lon: float,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not region_contains_point(manifest, lat=sample_lat, lon=sample_lon):
        errors.append("Sample point is outside region manifest bounds.")
        return errors, warnings

    resolved = find_region_for_point(sample_lat, sample_lon, base_dir=base_dir)
    if not resolved:
        errors.append("Sample point did not resolve to any prepared region.")
        return errors, warnings
    if str(resolved.get("region_id")) != str(manifest.get("region_id")):
        errors.append(
            f"Sample point resolved to region '{resolved.get('region_id')}' instead of '{manifest.get('region_id')}'."
        )
        return errors, warnings

    files_map = manifest.get("files") if isinstance(manifest, dict) else {}
    configured_optional = []
    if isinstance(files_map, dict):
        configured_optional = [k for k in OPTIONAL_REGION_LAYER_KEYS if files_map.get(k)]
    sample_layers = list(REQUIRED_REGION_FILES) + configured_optional

    for layer_key in sample_layers:
        resolved_path = resolve_region_file(manifest, layer_key, base_dir=base_dir)
        if not resolved_path:
            errors.append(f"Sample smoke test missing runtime layer mapping for '{layer_key}'.")
            continue
        path = Path(resolved_path)
        if not path.exists():
            errors.append(f"Sample smoke test missing file for layer '{layer_key}': {path}")
            continue
        if layer_key in VECTOR_KEYS:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                features = payload.get("features", []) if isinstance(payload, dict) else []
                if not features:
                    errors.append(f"Sample smoke test vector layer has no features: {layer_key}")
            except Exception as exc:
                errors.append(f"Sample smoke test could not read vector layer '{layer_key}': {exc}")
        else:
            if rasterio is None:
                warnings.append(
                    f"Sample smoke raster check for '{layer_key}' downgraded: rasterio unavailable."
                )
                continue
            try:
                with rasterio.open(path) as ds:
                    if ds.crs is None:
                        errors.append(f"Sample smoke raster '{layer_key}' has no CRS.")
                        continue
                    if transform_bounds is None:
                        warnings.append(
                            f"Sample smoke raster value sampling skipped for '{layer_key}' due to missing transforms."
                        )
                        continue
                    x, y = sample_lon, sample_lat
                    if str(ds.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                        # Avoid optional pyproj dependency in validator; openness check is enough for compatibility.
                        warnings.append(
                            f"Sample smoke raster '{layer_key}' uses non-EPSG:4326 CRS; value sample skipped."
                        )
                    else:
                        _ = next(ds.sample([(x, y)]))
            except Exception as exc:
                errors.append(f"Sample smoke test could not read raster layer '{layer_key}': {exc}")

    return errors, warnings


def _runtime_layer_mapping_issues(manifest: dict[str, Any], base_dir: str | None) -> list[str]:
    issues: list[str] = []
    for runtime_layer, manifest_layer in RUNTIME_LAYER_MAP.items():
        resolved = resolve_region_file(manifest, manifest_layer, base_dir=base_dir)
        if not resolved:
            issues.append(
                f"Runtime layer mapping mismatch: '{runtime_layer}' -> '{manifest_layer}' is not resolvable from manifest files."
            )
    return issues


def _write_manifest_validation_status(
    manifest: dict[str, Any],
    *,
    validation_status: str,
    runtime_compatibility_status: str,
    notes: list[str],
) -> None:
    manifest_path = Path(str(manifest.get("_manifest_path") or ""))
    if not manifest_path:
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return
    payload["validation_run_at"] = _now()
    payload["validation_status"] = validation_status
    payload["runtime_compatibility_status"] = runtime_compatibility_status
    payload["validation_notes"] = notes[:20]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def validate_prepared_region(
    *,
    region_id: str,
    base_dir: str | None = None,
    sample_lat: float | None = None,
    sample_lon: float | None = None,
    update_manifest: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest = load_region_manifest(region_id, base_dir=base_dir)
    if not manifest:
        return {
            "region_id": region_id,
            "ready_for_runtime": False,
            "validation_status": "failed",
            "runtime_compatibility_status": "failed",
            "scoring_readiness": "insufficient_data_behavior_only",
            "blockers": [f"Manifest not found for region '{region_id}'."],
            "warnings": [],
            "footprint_ring_support": "unknown",
            "required_layers_checked": list(REQUIRED_REGION_FILES),
            "sample_test": None,
        }

    bounds = _coerce_bounds(manifest.get("bounds"))
    if bounds is None:
        errors.append("Manifest bounds are missing or invalid.")

    files_ok, missing_issues = validate_region_files(manifest, base_dir=base_dir)
    if not files_ok:
        errors.extend(missing_issues)

    runtime_mapping_issues = _runtime_layer_mapping_issues(manifest, base_dir=base_dir)
    errors.extend(runtime_mapping_issues)

    files_map = manifest.get("files")
    configured_optional_layers: list[str] = []
    if isinstance(files_map, dict):
        configured_optional_layers = [k for k in OPTIONAL_REGION_LAYER_KEYS if files_map.get(k)]
    checked_layers = list(REQUIRED_REGION_FILES) + configured_optional_layers

    if bounds is not None:
        for layer_key in checked_layers:
            resolved = resolve_region_file(manifest, layer_key, base_dir=base_dir)
            if not resolved:
                continue
            path = Path(resolved)
            if not path.exists():
                continue
            layer_errors, layer_warnings = _validate_layer_openable_and_intersects(layer_key, path, bounds)
            errors.extend(layer_errors)
            warnings.extend(layer_warnings)

    sample_summary: dict[str, Any] | None = None
    if sample_lat is not None or sample_lon is not None:
        if sample_lat is None or sample_lon is None:
            errors.append("Sample point requires both sample_lat and sample_lon.")
        else:
            sample_errors, sample_warnings = _run_sample_smoke_test(
                manifest=manifest,
                base_dir=base_dir,
                sample_lat=float(sample_lat),
                sample_lon=float(sample_lon),
            )
            errors.extend(sample_errors)
            warnings.extend(sample_warnings)
            sample_summary = {
                "sample_lat": float(sample_lat),
                "sample_lon": float(sample_lon),
                "status": "failed" if sample_errors else "passed",
                "errors": sample_errors,
                "warnings": sample_warnings,
            }

    footprint_path = resolve_region_file(manifest, "building_footprints", base_dir=base_dir)
    ring_support = "missing"
    if footprint_path:
        ring_support = "available"
        if sample_lat is not None and sample_lon is not None and box is not None and shape is not None:
            try:
                with open(footprint_path, "r", encoding="utf-8") as f:
                    fp_payload = json.load(f)
                features = fp_payload.get("features", []) if isinstance(fp_payload, dict) else []
                has_intersection = False
                sample_pt = box(float(sample_lon), float(sample_lat), float(sample_lon), float(sample_lat))
                for feat in features:
                    geom = feat.get("geometry") if isinstance(feat, dict) else None
                    if not geom:
                        continue
                    try:
                        shp = shape(geom)
                    except Exception:
                        continue
                    if shp.is_valid and shp.intersects(sample_pt):
                        has_intersection = True
                        break
                if not has_intersection:
                    ring_support = "partial"
                    warnings.append(
                        "Building footprints exist but do not intersect sample point; structure rings may fallback to point-based mode."
                    )
            except Exception:
                ring_support = "partial"
                warnings.append("Could not evaluate footprint/sample intersection for ring readiness.")

    unique_errors = sorted(set(errors))
    unique_warnings = sorted(set(warnings))
    ready_for_runtime = len(unique_errors) == 0
    if ready_for_runtime and not unique_warnings:
        scoring_readiness = "full_scoring"
        validation_status = "passed"
    elif ready_for_runtime:
        scoring_readiness = "partial_scoring_only"
        validation_status = "passed_with_warnings"
    else:
        scoring_readiness = "insufficient_data_behavior_only"
        validation_status = "failed"

    runtime_compatibility_status = "pass" if ready_for_runtime else "failed"
    all_notes = unique_errors + unique_warnings
    if update_manifest:
        _write_manifest_validation_status(
            manifest,
            validation_status=validation_status,
            runtime_compatibility_status=runtime_compatibility_status,
            notes=all_notes,
        )

    return {
        "region_id": str(manifest.get("region_id") or region_id),
        "display_name": manifest.get("display_name"),
        "manifest_path": manifest.get("_manifest_path"),
        "ready_for_runtime": ready_for_runtime,
        "validation_status": validation_status,
        "runtime_compatibility_status": runtime_compatibility_status,
        "scoring_readiness": scoring_readiness,
        "blockers": unique_errors,
        "warnings": unique_warnings,
        "required_layers_checked": checked_layers,
        "footprint_ring_support": ring_support,
        "sample_test": sample_summary,
    }
