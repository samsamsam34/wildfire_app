from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.data_prep.layer_definitions import (
    REQUIRED_CORE_RASTER_LAYERS,
    REQUIRED_CORE_VECTOR_LAYERS,
)
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

VECTOR_KEYS = {
    "fire_perimeters",
    "building_footprints",
    "roads",
    "fema_structures",
    "building_footprints_microsoft",
    "building_footprints_overture",
    "parcel_polygons",
    "parcel_polygons_override",
    "parcel_address_points",
}
ARTIFACT_JSON_KEYS = {"naip_structure_features"}
OPTIONAL_REGION_LAYER_KEYS = {
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "roads",
    "fema_structures",
    "building_footprints_microsoft",
    "building_footprints_overture",
    "parcel_polygons",
    "parcel_polygons_override",
    "parcel_address_points",
    "naip_imagery",
    "naip_structure_features",
}
DEFAULT_OPTIONAL_LAYER_KEYS = (
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "roads",
)
ENRICHMENT_LAYER_KEYS = (
    "building_footprints_overture",
    "building_footprints_microsoft",
    "fema_structures",
    "parcel_polygons",
    "parcel_polygons_override",
    "parcel_address_points",
    "naip_imagery",
    "naip_structure_features",
)
REQUIRED_CORE_LAYER_KEYS = tuple(REQUIRED_CORE_RASTER_LAYERS) + tuple(REQUIRED_CORE_VECTOR_LAYERS)
_ADDRESS_DIRECT_KEYS = (
    "address",
    "full_address",
    "formatted_address",
    "site_address",
    "site_addr",
    "situs_address",
    "situs_addr",
    "addr",
    "addr_full",
)
_ADDRESS_HOUSE_KEYS = ("house_number", "housenumber", "number", "st_num", "addr_num")
_ADDRESS_STREET_KEYS = ("street", "street_name", "road", "rd_name", "street_full", "road_name")


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


def _first_non_empty_casefold(props: dict[str, Any], keys: tuple[str, ...]) -> str:
    lowered = {str(k).lower(): v for k, v in props.items()}
    for key in keys:
        lowered_key = str(key).lower()
        if lowered_key not in lowered:
            continue
        value = lowered.get(lowered_key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _has_min_address_components(props: dict[str, Any]) -> bool:
    direct = _first_non_empty_casefold(props, _ADDRESS_DIRECT_KEYS)
    if direct:
        direct_norm = re.sub(r"\s+", " ", direct).strip().lower()
        if re.match(r"^\d+[a-z0-9-]*\s+.+", direct_norm):
            return True
    house = _first_non_empty_casefold(props, _ADDRESS_HOUSE_KEYS)
    street = _first_non_empty_casefold(props, _ADDRESS_STREET_KEYS)
    return bool(house and street)


def _validate_vector_layer(
    path: Path,
    bounds: tuple[float, float, float, float],
    *,
    layer_key: str | None = None,
) -> tuple[list[str], list[str]]:
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

    if layer_key == "parcel_address_points":
        min_point_ratio_raw = str(os.getenv("WF_VALIDATE_PARCEL_ADDRESS_POINTS_MIN_POINT_RATIO", "0.85")).strip()
        try:
            min_point_ratio = max(0.0, min(1.0, float(min_point_ratio_raw)))
        except ValueError:
            min_point_ratio = 0.85

        min_complete_ratio_raw = str(os.getenv("WF_VALIDATE_PARCEL_ADDRESS_POINTS_MIN_COMPLETE_RATIO", "0.1")).strip()
        try:
            min_complete_ratio = max(0.0, min(1.0, float(min_complete_ratio_raw)))
        except ValueError:
            min_complete_ratio = 0.1

        point_like_count = 0
        complete_count = 0
        for feat in features:
            geom = feat.get("geometry") if isinstance(feat, dict) else {}
            geom_type = str((geom or {}).get("type") or "")
            if geom_type in {"Point", "MultiPoint"}:
                point_like_count += 1
            props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
            if _has_min_address_components(props):
                complete_count += 1

        total = max(1, len(features))
        point_ratio = float(point_like_count) / float(total)
        complete_ratio = float(complete_count) / float(total)
        if point_ratio < min_point_ratio:
            errors.append(
                f"parcel_address_points semantic validation failed: point geometry ratio "
                f"{point_ratio:.3f} is below minimum {min_point_ratio:.3f}."
            )
        if complete_ratio < min_complete_ratio:
            errors.append(
                f"parcel_address_points semantic validation failed: complete-address ratio "
                f"{complete_ratio:.3f} is below minimum {min_complete_ratio:.3f}."
            )

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
    if layer_key in ARTIFACT_JSON_KEYS:
        try:
            with open(layer_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return [f"Artifact layer payload is not a JSON object: {layer_path}"], []
            return [], []
        except Exception as exc:
            return [f"Artifact layer could not be read: {layer_path} ({exc})"], []
    if layer_key in VECTOR_KEYS:
        return _validate_vector_layer(layer_path, bounds, layer_key=layer_key)
    return _validate_raster_layer(layer_path, bounds)


def _detect_semantic_duplicate_layers(manifest: dict[str, Any], base_dir: str | None) -> list[str]:
    parcel_address_path = resolve_region_file(manifest, "parcel_address_points", base_dir=base_dir)
    parcel_polygon_path = resolve_region_file(manifest, "parcel_polygons", base_dir=base_dir)
    if not parcel_address_path or not parcel_polygon_path:
        return []
    parcel_address_file = Path(parcel_address_path)
    parcel_polygon_file = Path(parcel_polygon_path)
    if not parcel_address_file.exists() or not parcel_polygon_file.exists():
        return []

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    try:
        address_hash = _sha256(parcel_address_file)
        polygon_hash = _sha256(parcel_polygon_file)
    except Exception as exc:
        return [f"Could not compute semantic duplicate check for parcel layers: {exc}"]

    if address_hash == polygon_hash:
        return [
            "parcel_address_points and parcel_polygons resolve to byte-identical datasets; "
            "address-point layer appears misconfigured."
        ]
    return []


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
        if layer_key in ARTIFACT_JSON_KEYS:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    errors.append(f"Sample smoke test artifact layer is not a JSON object: {layer_key}")
            except Exception as exc:
                errors.append(f"Sample smoke test could not read artifact layer '{layer_key}': {exc}")
        elif layer_key in VECTOR_KEYS:
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
                warnings.append(
                    f"Sample smoke raster check skipped for '{layer_key}' due to unreadable raster payload: {exc}"
                )

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


def _present_and_missing_layers(
    *,
    manifest: dict[str, Any],
    layer_keys: tuple[str, ...] | list[str],
    base_dir: str | None,
) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for layer_key in layer_keys:
        resolved = resolve_region_file(manifest, layer_key, base_dir=base_dir)
        if resolved and Path(resolved).exists():
            present.append(layer_key)
        else:
            missing.append(layer_key)
    return sorted(set(present)), sorted(set(missing))


def summarize_property_specific_readiness(
    *,
    required_layers_present: list[str],
    optional_layers_present: list[str],
    enrichment_layers_present: list[str],
    missing_reason_by_layer: dict[str, str] | None = None,
    configured_anchor_layers: list[str] | None = None,
    configured_anchor_available: bool | None = None,
    source_used_by_layer: dict[str, Any] | None = None,
    catalog_public_record_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing_reason_by_layer = missing_reason_by_layer or {}
    source_used_by_layer = source_used_by_layer or {}
    catalog_public_record_fields = catalog_public_record_fields or {}
    required_set = set(required_layers_present)
    optional_set = set(optional_layers_present)
    enrichment_set = set(enrichment_layers_present)

    configured_anchor_layers = [str(v) for v in (configured_anchor_layers or []) if str(v).strip()]
    building_signal = "building_footprints" in required_set
    roads_signal = "roads" in optional_set
    hazard_signal = bool({"whp", "gridmet_dryness", "mtbs_severity"}.intersection(optional_set))
    vegetation_signal = (
        ("naip_imagery" in enrichment_set)
        or ("naip_structure_features" in enrichment_set)
        or ("canopy" in required_set)
    )
    if configured_anchor_layers:
        anchor_signal = bool(configured_anchor_available)
    else:
        anchor_signal = bool(
            {"parcel_polygons", "parcel_address_points", "building_footprints_overture"}.intersection(
                enrichment_set
            )
        )

    parcel_ready = "parcel_polygons" in enrichment_set
    footprint_ready = building_signal
    naip_ready = "naip_structure_features" in enrichment_set
    structure_enrichment_ready = bool(catalog_public_record_fields) or (
        "parcel_address_points" in enrichment_set
    )

    if parcel_ready and footprint_ready:
        parcel_footprint_linkage_quality = "high"
    elif footprint_ready and anchor_signal:
        parcel_footprint_linkage_quality = "moderate"
    elif footprint_ready or parcel_ready:
        parcel_footprint_linkage_quality = "low"
    else:
        parcel_footprint_linkage_quality = "unavailable"

    if (
        parcel_ready
        and footprint_ready
        and naip_ready
        and parcel_footprint_linkage_quality in {"high", "moderate"}
    ):
        overall_readiness = "property_specific"
    elif footprint_ready and (parcel_ready or anchor_signal):
        overall_readiness = "address_level"
    else:
        overall_readiness = "limited_regional"

    if building_signal and roads_signal and hazard_signal and vegetation_signal and anchor_signal:
        readiness = "property_specific_ready"
    elif building_signal and (roads_signal or anchor_signal):
        readiness = "address_level_only"
    else:
        readiness = "limited_regional_ready"

    missing_supporting_layers: list[str] = []
    if not roads_signal:
        missing_supporting_layers.append("roads")
    if not hazard_signal:
        missing_supporting_layers.append("whp|gridmet_dryness|mtbs_severity")
    if not anchor_signal:
        missing_supporting_layers.append("parcel_polygons|parcel_address_points|building_footprints_overture")
    if not vegetation_signal:
        missing_supporting_layers.append("naip_imagery|naip_structure_features|canopy")
    if not parcel_ready:
        missing_supporting_layers.append("parcel_polygons")
    if not naip_ready:
        missing_supporting_layers.append("naip_structure_features")
    if not structure_enrichment_ready:
        missing_supporting_layers.append("public_record_structure_enrichment")
    if parcel_footprint_linkage_quality in {"low", "unavailable"}:
        missing_supporting_layers.append("parcel_footprint_linkage_quality")
    if configured_anchor_layers and not anchor_signal:
        missing_supporting_layers.extend(configured_anchor_layers)

    return {
        "readiness": readiness,
        "parcel_ready": parcel_ready,
        "footprint_ready": footprint_ready,
        "naip_ready": naip_ready,
        "structure_enrichment_ready": structure_enrichment_ready,
        "parcel_footprint_linkage_quality": parcel_footprint_linkage_quality,
        "overall_readiness": overall_readiness,
        "signals": {
            "building_footprints": building_signal,
            "roads": roads_signal,
            "hazard_context": hazard_signal,
            "near_structure_vegetation": vegetation_signal,
            "property_anchor_context": anchor_signal,
        },
        "configured_anchor_layers": configured_anchor_layers,
        "missing_supporting_layers": sorted(set(missing_supporting_layers)),
        "missing_reason_by_layer": dict(missing_reason_by_layer),
        "source_used_by_layer": dict(source_used_by_layer),
    }


def _validation_manifest_summary(
    *,
    manifest: dict[str, Any],
    base_dir: str | None,
) -> dict[str, Any]:
    catalog = manifest.get("catalog") if isinstance(manifest.get("catalog"), dict) else {}
    required_present, required_missing = _present_and_missing_layers(
        manifest=manifest,
        layer_keys=list(REQUIRED_CORE_LAYER_KEYS),
        base_dir=base_dir,
    )
    optional_present, optional_missing = _present_and_missing_layers(
        manifest=manifest,
        layer_keys=list(DEFAULT_OPTIONAL_LAYER_KEYS),
        base_dir=base_dir,
    )
    enrichment_present, enrichment_missing = _present_and_missing_layers(
        manifest=manifest,
        layer_keys=list(ENRICHMENT_LAYER_KEYS),
        base_dir=base_dir,
    )

    catalog_reasons = catalog.get("missing_reason_by_layer")
    missing_reason_by_layer: dict[str, str] = {}
    for layer in sorted(set(required_missing + optional_missing + enrichment_missing)):
        fallback_reason = "missing_from_manifest_or_files"
        if isinstance(catalog_reasons, dict) and catalog_reasons.get(layer):
            fallback_reason = str(catalog_reasons.get(layer))
        missing_reason_by_layer[layer] = fallback_reason

    source_used_by_layer = (
        dict(catalog.get("source_used_by_layer"))
        if isinstance(catalog.get("source_used_by_layer"), dict)
        else {}
    )
    readiness = summarize_property_specific_readiness(
        required_layers_present=required_present,
        optional_layers_present=optional_present,
        enrichment_layers_present=enrichment_present,
        missing_reason_by_layer=missing_reason_by_layer,
        source_used_by_layer=source_used_by_layer,
        catalog_public_record_fields=(
            dict(catalog.get("public_record_fields"))
            if isinstance(catalog.get("public_record_fields"), dict)
            else {}
        ),
    )
    return {
        "required_layers_present": required_present,
        "required_layers_missing": required_missing,
        "optional_layers_present": optional_present,
        "optional_layers_missing": optional_missing,
        "enrichment_layers_present": enrichment_present,
        "enrichment_layers_missing": enrichment_missing,
        "missing_reason_by_layer": missing_reason_by_layer,
        "source_used_by_layer": source_used_by_layer,
        "property_specific_readiness": readiness,
    }


def _write_manifest_validation_status(
    manifest: dict[str, Any],
    *,
    validation_status: str,
    runtime_compatibility_status: str,
    notes: list[str],
    validation_summary: dict[str, Any] | None = None,
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
    if isinstance(validation_summary, dict):
        payload.setdefault("catalog", {})
        payload["catalog"]["required_layers_present"] = validation_summary.get("required_layers_present", [])
        payload["catalog"]["required_layers_missing"] = validation_summary.get("required_layers_missing", [])
        payload["catalog"]["optional_layers_present"] = validation_summary.get("optional_layers_present", [])
        payload["catalog"]["optional_layers_missing"] = validation_summary.get("optional_layers_missing", [])
        payload["catalog"]["enrichment_layers_present"] = validation_summary.get("enrichment_layers_present", [])
        payload["catalog"]["enrichment_layers_missing"] = validation_summary.get("enrichment_layers_missing", [])
        payload["catalog"]["missing_reason_by_layer"] = validation_summary.get("missing_reason_by_layer", {})
        payload["catalog"]["source_used_by_layer"] = validation_summary.get("source_used_by_layer", {})
        payload["catalog"]["property_specific_readiness"] = validation_summary.get("property_specific_readiness", {})
        payload["catalog"]["validation_summary"] = validation_summary.get("validation_summary", {})
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
        property_specific_readiness = {
            "readiness": "limited_regional_ready",
            "parcel_ready": False,
            "footprint_ready": False,
            "naip_ready": False,
            "structure_enrichment_ready": False,
            "parcel_footprint_linkage_quality": "unavailable",
            "overall_readiness": "limited_regional",
            "signals": {},
            "configured_anchor_layers": [],
            "missing_supporting_layers": [],
            "missing_reason_by_layer": {},
            "source_used_by_layer": {},
        }
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
            "required_layers_present": [],
            "required_layers_missing": list(REQUIRED_CORE_LAYER_KEYS),
            "optional_layers_present": [],
            "optional_layers_missing": list(DEFAULT_OPTIONAL_LAYER_KEYS),
            "enrichment_layers_present": [],
            "enrichment_layers_missing": list(ENRICHMENT_LAYER_KEYS),
            "missing_reason_by_layer": {},
            "source_used_by_layer": {},
            "property_specific_readiness": property_specific_readiness,
            "validation_summary": {
                "validation_status": "failed",
                "runtime_compatibility_status": "failed",
                "ready_for_runtime": False,
                "scoring_readiness": "insufficient_data_behavior_only",
                "blocker_count": 1,
                "warning_count": 0,
                "property_specific_readiness": property_specific_readiness,
            },
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

    errors.extend(_detect_semantic_duplicate_layers(manifest, base_dir=base_dir))

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

    manifest_summary = _validation_manifest_summary(manifest=manifest, base_dir=base_dir)
    property_specific_readiness = dict(manifest_summary.get("property_specific_readiness") or {})

    runtime_compatibility_status = "pass" if ready_for_runtime else "failed"
    all_notes = unique_errors + unique_warnings
    validation_summary = {
        "validation_status": validation_status,
        "runtime_compatibility_status": runtime_compatibility_status,
        "ready_for_runtime": ready_for_runtime,
        "scoring_readiness": scoring_readiness,
        "blocker_count": len(unique_errors),
        "warning_count": len(unique_warnings),
        "property_specific_readiness": property_specific_readiness,
    }
    manifest_summary["validation_summary"] = validation_summary
    if update_manifest:
        _write_manifest_validation_status(
            manifest,
            validation_status=validation_status,
            runtime_compatibility_status=runtime_compatibility_status,
            notes=all_notes,
            validation_summary=manifest_summary,
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
        "required_layers_present": manifest_summary.get("required_layers_present", []),
        "required_layers_missing": manifest_summary.get("required_layers_missing", []),
        "optional_layers_present": manifest_summary.get("optional_layers_present", []),
        "optional_layers_missing": manifest_summary.get("optional_layers_missing", []),
        "enrichment_layers_present": manifest_summary.get("enrichment_layers_present", []),
        "enrichment_layers_missing": manifest_summary.get("enrichment_layers_missing", []),
        "missing_reason_by_layer": manifest_summary.get("missing_reason_by_layer", {}),
        "source_used_by_layer": manifest_summary.get("source_used_by_layer", {}),
        "property_specific_readiness": property_specific_readiness,
        "validation_summary": validation_summary,
    }
