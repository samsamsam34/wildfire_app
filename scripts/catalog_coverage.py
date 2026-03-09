from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from backend.data_prep.catalog import (
    CATALOG_CORE_RASTER_LAYERS,
    CATALOG_CORE_VECTOR_LAYERS,
    CATALOG_OPTIONAL_LAYERS,
    default_catalog_root,
    load_catalog_index,
)

try:
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - optional dependency
    transform_bounds = None

try:
    from shapely.geometry import box
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional dependency
    box = None
    unary_union = None


def required_core_layers() -> list[str]:
    return list(CATALOG_CORE_RASTER_LAYERS) + list(CATALOG_CORE_VECTOR_LAYERS)


def optional_layers() -> list[str]:
    return list(CATALOG_OPTIONAL_LAYERS)


def _to_bbox_tuple(bounds: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(bounds["min_lon"]),
        float(bounds["min_lat"]),
        float(bounds["max_lon"]),
        float(bounds["max_lat"]),
    )


def _bbox_contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _entry_bounds_wgs84(entry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = entry.get("bounds")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        bounds = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None

    stored_crs = str(entry.get("stored_crs") or "EPSG:4326")
    if stored_crs.upper() in {"EPSG:4326", "OGC:CRS84"}:
        return bounds
    if transform_bounds is None:
        return None
    try:
        transformed = transform_bounds(stored_crs, "EPSG:4326", *bounds)
        return (float(transformed[0]), float(transformed[1]), float(transformed[2]), float(transformed[3]))
    except Exception:
        return None


def evaluate_layer_coverage(
    *,
    layer_key: str,
    bounds: dict[str, float],
    catalog_index: dict[str, Any],
) -> dict[str, Any]:
    requested = _to_bbox_tuple(bounds)
    layers = catalog_index.get("layers", {})
    bucket = layers.get(layer_key) if isinstance(layers, dict) else None
    entries = list(bucket.get("entries", [])) if isinstance(bucket, dict) else []

    if not entries:
        return {
            "layer_key": layer_key,
            "coverage_status": "none",
            "entries_total": 0,
            "entries_with_bounds": 0,
            "entries_covering": [],
            "entries_intersecting": [],
            "entries_missing_bounds": [],
            "notes": ["No catalog entries for this layer."],
        }

    covering: list[str] = []
    intersecting: list[str] = []
    missing_bounds: list[str] = []
    intersecting_bounds: list[tuple[float, float, float, float]] = []

    for entry in entries:
        item_id = str(entry.get("item_id") or "unknown")
        normalized = _entry_bounds_wgs84(entry)
        if normalized is None:
            missing_bounds.append(item_id)
            continue
        if _bbox_contains(normalized, requested):
            covering.append(item_id)
        if _bbox_intersects(normalized, requested):
            intersecting.append(item_id)
            intersecting_bounds.append(normalized)

    if covering:
        status = "full"
        notes = ["At least one catalog entry fully covers the requested bbox."]
    elif intersecting:
        status = "partial"
        notes = ["Catalog entries intersect the bbox but do not individually cover it."]
        if box is not None and unary_union is not None:
            try:
                requested_poly = box(*requested)
                merged = unary_union([box(*b) for b in intersecting_bounds])
                if merged.covers(requested_poly):
                    status = "full"
                    notes = ["Union of multiple catalog entries covers the requested bbox."]
            except Exception:
                notes.append("Coverage union check unavailable; using intersection-only classification.")
    else:
        status = "none"
        notes = ["No catalog entry intersects the requested bbox."]

    return {
        "layer_key": layer_key,
        "coverage_status": status,
        "entries_total": len(entries),
        "entries_with_bounds": len(entries) - len(missing_bounds),
        "entries_covering": sorted(set(covering)),
        "entries_intersecting": sorted(set(intersecting)),
        "entries_missing_bounds": sorted(set(missing_bounds)),
        "notes": notes,
    }


def build_catalog_coverage_plan(
    *,
    bounds: dict[str, float],
    required_layers: Sequence[str],
    optional_layer_keys: Sequence[str] | None = None,
    catalog_root: Path | None = None,
    catalog_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = catalog_index or load_catalog_index(catalog_root or default_catalog_root())
    optional_layer_keys = list(optional_layer_keys or [])
    target_layers = list(required_layers) + list(optional_layer_keys)

    layer_results: dict[str, dict[str, Any]] = {}
    for layer_key in target_layers:
        layer_results[layer_key] = evaluate_layer_coverage(
            layer_key=layer_key,
            bounds=bounds,
            catalog_index=index,
        )

    required_missing = sorted(
        [k for k in required_layers if layer_results.get(k, {}).get("coverage_status") == "none"]
    )
    required_partial = sorted(
        [k for k in required_layers if layer_results.get(k, {}).get("coverage_status") == "partial"]
    )
    optional_missing = sorted(
        [k for k in optional_layer_keys if layer_results.get(k, {}).get("coverage_status") == "none"]
    )
    optional_partial = sorted(
        [k for k in optional_layer_keys if layer_results.get(k, {}).get("coverage_status") == "partial"]
    )

    full_count = sum(1 for r in layer_results.values() if r.get("coverage_status") == "full")
    partial_count = sum(1 for r in layer_results.values() if r.get("coverage_status") == "partial")
    none_count = sum(1 for r in layer_results.values() if r.get("coverage_status") == "none")

    recommended_actions: list[str] = []
    if required_missing:
        recommended_actions.append(
            "Ingest missing core layers into catalog: " + ", ".join(required_missing)
        )
    if required_partial:
        recommended_actions.append(
            "Core layers are partially covered; enable gap fill or ingest larger bbox chunks: "
            + ", ".join(required_partial)
        )
    if optional_missing:
        recommended_actions.append(
            "Optional enrichment layers missing: " + ", ".join(optional_missing)
        )

    return {
        "requested_bbox": {
            "min_lon": float(bounds["min_lon"]),
            "min_lat": float(bounds["min_lat"]),
            "max_lon": float(bounds["max_lon"]),
            "max_lat": float(bounds["max_lat"]),
        },
        "catalog_root": str(Path(catalog_root or default_catalog_root()).expanduser()),
        "required_layers": list(required_layers),
        "optional_layers": list(optional_layer_keys),
        "layers": layer_results,
        "summary": {
            "total_layers": len(target_layers),
            "full_count": full_count,
            "partial_count": partial_count,
            "none_count": none_count,
            "required_missing": required_missing,
            "required_partial": required_partial,
            "optional_missing": optional_missing,
            "optional_partial": optional_partial,
            "buildable_from_catalog": len(required_missing) == 0,
            "fully_covered_from_catalog": len(required_missing) == 0 and len(required_partial) == 0,
            "recommended_actions": recommended_actions,
        },
    }


def render_plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, sort_keys=True)
