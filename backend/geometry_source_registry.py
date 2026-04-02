from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY_PATH = _ROOT / "config" / "geometry_source_registry.json"

_BUILTIN_REGISTRY: dict[str, Any] = {
    "version": 1,
    "defaults": {
        "source_order": {
            "parcel_sources": [
                "parcel_polygons",
                "parcel_polygons_override",
                "nearest_parcel_fallback",
            ],
            "footprint_sources": [
                "building_footprints",
                "building_footprints_overture",
                "building_footprints_microsoft",
                "fema_structures",
            ],
        },
        "source_definitions": {
            "parcel_sources": {
                "parcel_polygons": {
                    "display_name": "Prepared/local parcel polygons",
                    "layer_keys": ["parcel_polygons", "parcels"],
                },
                "parcel_polygons_override": {
                    "display_name": "Region-specific parcel override",
                    "layer_keys": ["parcel_polygons_override", "parcel_overrides"],
                },
                "nearest_parcel_fallback": {
                    "display_name": "Nearest-parcel fallback",
                    "layer_keys": [],
                    "fallback_only": True,
                    "explicit_downgrade": True,
                },
            },
            "footprint_sources": {
                "building_footprints": {
                    "display_name": "Prepared/local building footprints",
                    "layer_keys": ["building_footprints", "footprints"],
                },
                "building_footprints_overture": {
                    "display_name": "Overture building footprints",
                    "layer_keys": ["building_footprints_overture", "overture_buildings", "footprints_overture"],
                },
                "building_footprints_microsoft": {
                    "display_name": "Microsoft building footprints",
                    "layer_keys": ["building_footprints_microsoft", "microsoft_buildings", "footprints_microsoft"],
                },
                "fema_structures": {
                    "display_name": "FEMA or equivalent backup structures",
                    "layer_keys": ["fema_structures", "osm_buildings"],
                },
            },
        },
        "schema_normalization_rules": {
            "parcel_sources": {
                "parcel_id_fields": ["parcel_id", "PARCEL_ID", "parcelid", "APN"],
                "geometry_type": "Polygon|MultiPolygon",
            },
            "footprint_sources": {
                "structure_id_fields": ["structure_id", "building_id", "id", "OBJECTID"],
                "geometry_type": "Polygon|MultiPolygon",
            },
        },
        "confidence_weights": {
            "parcel_sources": {
                "parcel_polygons": 0.92,
                "parcel_polygons_override": 0.94,
                "nearest_parcel_fallback": 0.28,
            },
            "footprint_sources": {
                "building_footprints": 0.88,
                "building_footprints_overture": 0.86,
                "building_footprints_microsoft": 0.80,
                "fema_structures": 0.62,
            },
        },
        "known_limitations": [
            "Source precedence governs geometry anchoring confidence only; it does not change wildfire scoring logic.",
            "Nearest-parcel fallback is allowed only with explicit confidence downgrade and cautionary diagnostics.",
        ],
    },
    "regions": {},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_registry_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    env_path = str(os.getenv("WF_GEOMETRY_SOURCE_REGISTRY_PATH", "")).strip()
    if env_path:
        return Path(env_path).expanduser()
    return _DEFAULT_REGISTRY_PATH


def load_geometry_source_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry = dict(_BUILTIN_REGISTRY)
    candidate = _resolve_registry_path(path)
    if not candidate.exists():
        return registry
    with open(candidate, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return registry
    return _deep_merge(registry, payload)


def _resolve_region_config(
    *,
    registry: dict[str, Any],
    region_id: str,
) -> dict[str, Any]:
    defaults = dict(registry.get("defaults") or {})
    regions = dict(registry.get("regions") or {})
    override = dict(regions.get(region_id) or {})
    return _deep_merge(defaults, override)


def _extract_layer_version(layer_meta: dict[str, Any]) -> str | None:
    for key in ("dataset_version", "source_vintage", "freshness_timestamp", "prepared_at"):
        value = layer_meta.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _normalize_source_entries(
    *,
    source_order: list[str],
    source_definitions: dict[str, Any],
    files: dict[str, str],
    layers_meta: dict[str, dict[str, Any]],
    schema_rules: dict[str, Any],
    confidence_weights: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    entries: list[dict[str, Any]] = []
    source_versions: dict[str, str] = {}
    limitations: list[str] = []
    for rank, source_id in enumerate(source_order, start=1):
        source_cfg = dict(source_definitions.get(source_id) or {})
        layer_keys = [str(v).strip() for v in list(source_cfg.get("layer_keys") or []) if str(v).strip()]
        selected_layer_key = None
        selected_file = None
        selected_layer_meta: dict[str, Any] = {}
        for key in layer_keys:
            rel = files.get(key)
            if rel:
                selected_layer_key = key
                selected_file = rel
                selected_layer_meta = dict(layers_meta.get(key) or {})
                break

        fallback_only = bool(source_cfg.get("fallback_only"))
        explicit_downgrade = bool(source_cfg.get("explicit_downgrade"))
        available = bool(selected_layer_key) or fallback_only
        confidence_weight = float(confidence_weights.get(source_id, 0.5) or 0.5)
        entry = {
            "source_id": source_id,
            "display_name": str(source_cfg.get("display_name") or source_id),
            "priority_rank": rank,
            "available": available,
            "selected_layer_key": selected_layer_key,
            "selected_file": selected_file,
            "fallback_only": fallback_only,
            "explicit_downgrade": explicit_downgrade,
            "confidence_weight": round(max(0.0, min(1.0, confidence_weight)), 3),
            "schema_normalization_rules": dict(schema_rules),
            "known_limitations": list(source_cfg.get("known_limitations") or []),
        }
        if selected_layer_meta:
            source_type = str(selected_layer_meta.get("source_type") or "").strip()
            if source_type:
                entry["source_type"] = source_type
            source_name = str(selected_layer_meta.get("source_name") or "").strip()
            if source_name:
                entry["source_name"] = source_name
            version = _extract_layer_version(selected_layer_meta)
            if version:
                source_versions[source_id] = version
        if not available and not fallback_only:
            limitations.append(
                f"{source_id} is not available in this prepared region; downstream geometry matching will use lower-priority sources."
            )
        if fallback_only and explicit_downgrade:
            limitations.append(
                f"{source_id} is fallback-only and requires explicit confidence downgrade when used."
            )
        entries.append(entry)
    return entries, source_versions, limitations


def build_region_geometry_source_manifest(
    *,
    region_id: str,
    files: dict[str, str],
    layers_meta: dict[str, dict[str, Any]] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_registry = (
        registry
        if isinstance(registry, dict) and registry
        else load_geometry_source_registry()
    )
    effective = _resolve_region_config(registry=effective_registry, region_id=region_id)

    source_order_cfg = dict(effective.get("source_order") or {})
    parcel_order = [str(v).strip() for v in list(source_order_cfg.get("parcel_sources") or []) if str(v).strip()]
    footprint_order = [str(v).strip() for v in list(source_order_cfg.get("footprint_sources") or []) if str(v).strip()]

    source_definitions = dict(effective.get("source_definitions") or {})
    parcel_defs = dict(source_definitions.get("parcel_sources") or {})
    footprint_defs = dict(source_definitions.get("footprint_sources") or {})

    schema_rules_cfg = dict(effective.get("schema_normalization_rules") or {})
    parcel_schema_rules = dict(schema_rules_cfg.get("parcel_sources") or {})
    footprint_schema_rules = dict(schema_rules_cfg.get("footprint_sources") or {})

    confidence_cfg = dict(effective.get("confidence_weights") or {})
    parcel_conf = dict(confidence_cfg.get("parcel_sources") or {})
    footprint_conf = dict(confidence_cfg.get("footprint_sources") or {})

    layer_meta = layers_meta if isinstance(layers_meta, dict) else {}
    parcel_entries, parcel_versions, parcel_limitations = _normalize_source_entries(
        source_order=parcel_order,
        source_definitions=parcel_defs,
        files=files,
        layers_meta=layer_meta,
        schema_rules=parcel_schema_rules,
        confidence_weights=parcel_conf,
    )
    footprint_entries, footprint_versions, footprint_limitations = _normalize_source_entries(
        source_order=footprint_order,
        source_definitions=footprint_defs,
        files=files,
        layers_meta=layer_meta,
        schema_rules=footprint_schema_rules,
        confidence_weights=footprint_conf,
    )

    known_limitations = [
        str(v).strip()
        for v in list(effective.get("known_limitations") or [])
        if str(v).strip()
    ]
    known_limitations.extend(parcel_limitations)
    known_limitations.extend(footprint_limitations)

    has_explicit_parcel_source = any(
        bool(entry.get("available")) and not bool(entry.get("fallback_only"))
        for entry in parcel_entries
    )
    if not has_explicit_parcel_source:
        known_limitations.append(
            "No parcel polygon source was available; nearest-parcel fallback is allowed only with explicit downgrade."
        )

    source_versions: dict[str, str] = {}
    source_versions.update(parcel_versions)
    source_versions.update(footprint_versions)

    return {
        "version": int(effective_registry.get("version") or 1),
        "region_id": region_id,
        "parcel_sources": parcel_entries,
        "footprint_sources": footprint_entries,
        "default_source_order": {
            "parcel_sources": parcel_order,
            "footprint_sources": footprint_order,
        },
        "schema_normalization_rules": {
            "parcel_sources": parcel_schema_rules,
            "footprint_sources": footprint_schema_rules,
        },
        "confidence_weights": {
            "parcel_sources": parcel_conf,
            "footprint_sources": footprint_conf,
        },
        "source_versions": source_versions,
        "known_limitations": list(dict.fromkeys(known_limitations)),
    }
