from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _file_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _first_existing_path(candidates: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    for key, candidate in candidates:
        if _file_exists(candidate):
            return key, str(candidate)
    return None, None


@dataclass(frozen=True)
class SourceGroup:
    key: str
    runtime_candidates: tuple[tuple[str, str], ...]
    env_candidates: tuple[tuple[str, str, str], ...]


SOURCE_GROUPS: tuple[SourceGroup, ...] = (
    SourceGroup(
        key="building_footprint",
        runtime_candidates=(
            ("footprints_overture", "overture_buildings"),
            ("footprints_microsoft", "microsoft_buildings"),
            ("footprints", "prepared_building_footprints"),
            ("fema_structures", "fema_structures"),
        ),
        env_candidates=(
            ("WF_ENRICH_OVERTURE_BUILDINGS_PATH", "footprints_overture", "overture_buildings"),
            ("WF_ENRICH_MICROSOFT_BUILDINGS_PATH", "footprints_microsoft", "microsoft_buildings"),
            ("WF_LAYER_BUILDING_FOOTPRINTS_MICROSOFT_GEOJSON", "footprints_microsoft", "microsoft_buildings"),
            ("WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON", "footprints", "prepared_building_footprints"),
        ),
    ),
    SourceGroup(
        key="parcel",
        runtime_candidates=(("parcels", "parcel_polygons"),),
        env_candidates=(
            ("WF_ENRICH_PARCELS_PATH", "parcels", "county_or_regrid_parcels"),
            ("WF_LAYER_PARCEL_POLYGONS_GEOJSON", "parcels", "county_or_regrid_parcels"),
            ("WF_LAYER_PARCELS_GEOJSON", "parcels", "county_or_regrid_parcels"),
        ),
    ),
    SourceGroup(
        key="address_points",
        runtime_candidates=(("address_points", "address_points"),),
        env_candidates=(
            ("WF_ENRICH_ADDRESS_POINTS_PATH", "address_points", "county_address_points"),
            ("WF_LAYER_PARCEL_ADDRESS_POINTS_GEOJSON", "address_points", "county_address_points"),
            ("WF_LAYER_ADDRESS_POINTS_GEOJSON", "address_points", "county_address_points"),
        ),
    ),
    SourceGroup(
        key="vegetation",
        runtime_candidates=(("fuel", "landfire_fuel"),),
        env_candidates=(
            ("WF_ENRICH_LANDFIRE_FUEL_TIF", "fuel", "landfire_fuel"),
            ("WF_LAYER_FUEL_TIF", "fuel", "landfire_fuel"),
        ),
    ),
    SourceGroup(
        key="canopy",
        runtime_candidates=(("canopy", "landfire_canopy"),),
        env_candidates=(
            ("WF_ENRICH_LANDFIRE_CANOPY_TIF", "canopy", "landfire_canopy"),
            ("WF_LAYER_CANOPY_TIF", "canopy", "landfire_canopy"),
        ),
    ),
    SourceGroup(
        key="burn_probability",
        runtime_candidates=(
            ("burn_prob", "burn_probability_raster"),
            ("whp", "whp"),
        ),
        env_candidates=(
            ("WF_ENRICH_BURN_PROBABILITY_TIF", "burn_prob", "burn_probability_raster"),
            ("WF_LAYER_BURN_PROB_TIF", "burn_prob", "burn_probability_raster"),
            ("WF_ENRICH_WHP_TIF", "whp", "whp"),
            ("WF_LAYER_WHP_TIF", "whp", "whp"),
        ),
    ),
    SourceGroup(
        key="historical_fire",
        runtime_candidates=(
            ("perimeters", "fire_perimeters"),
            ("mtbs_severity", "mtbs_severity"),
        ),
        env_candidates=(
            ("WF_ENRICH_MTBS_PERIMETERS_GEOJSON", "perimeters", "fire_perimeters"),
            ("WF_LAYER_FIRE_PERIMETERS_GEOJSON", "perimeters", "fire_perimeters"),
            ("WF_ENRICH_MTBS_SEVERITY_TIF", "mtbs_severity", "mtbs_severity"),
            ("WF_LAYER_MTBS_SEVERITY_TIF", "mtbs_severity", "mtbs_severity"),
        ),
    ),
    SourceGroup(
        key="roads",
        runtime_candidates=(("roads", "osm_roads"),),
        env_candidates=(
            ("WF_ENRICH_OSM_ROADS_GEOJSON", "roads", "osm_roads"),
            ("WF_LAYER_OSM_ROADS_GEOJSON", "roads", "osm_roads"),
        ),
    ),
    SourceGroup(
        key="climate_dryness",
        runtime_candidates=(
            ("gridmet_dryness", "gridmet_dryness"),
            ("moisture", "moisture_raster"),
        ),
        env_candidates=(
            ("WF_ENRICH_GRIDMET_DRYNESS_TIF", "gridmet_dryness", "gridmet_dryness"),
            ("WF_LAYER_GRIDMET_DRYNESS_TIF", "gridmet_dryness", "gridmet_dryness"),
            ("WF_LAYER_MOISTURE_TIF", "moisture", "moisture_raster"),
        ),
    ),
    SourceGroup(
        key="naip_imagery",
        runtime_candidates=(("naip_imagery", "naip_imagery"),),
        env_candidates=(
            ("WF_ENRICH_NAIP_IMAGERY_TIF", "naip_imagery", "naip_imagery"),
            ("WF_LAYER_NAIP_IMAGERY_TIF", "naip_imagery", "naip_imagery"),
        ),
    ),
    SourceGroup(
        key="naip_structure_features",
        runtime_candidates=(("naip_structure_features", "naip_structure_features"),),
        env_candidates=(
            ("WF_ENRICH_NAIP_STRUCTURE_FEATURES_JSON", "naip_structure_features", "naip_structure_features"),
            ("WF_LAYER_NAIP_STRUCTURE_FEATURES_JSON", "naip_structure_features", "naip_structure_features"),
        ),
    ),
)

RUNTIME_PATH_ALIASES: dict[str, str] = {
    "building_footprints_overture": "footprints_overture",
    "building_footprints_microsoft": "footprints_microsoft",
    "building_footprints": "footprints",
    "parcel_polygons": "parcels",
    "parcel": "parcels",
    "parcel_address_points": "address_points",
    "road_network": "roads",
    "road_centerlines": "roads",
    "osm_roads": "roads",
    "burn_probability": "burn_prob",
    "wildfire_hazard": "hazard",
    "fire_history_perimeters": "perimeters",
}


def _normalize_runtime_aliases(runtime_paths: dict[str, str]) -> dict[str, str]:
    normalized = dict(runtime_paths)
    for alias_key, canonical_key in RUNTIME_PATH_ALIASES.items():
        canonical_value = str(normalized.get(canonical_key) or "").strip()
        if canonical_value:
            continue
        alias_value = str(normalized.get(alias_key) or "").strip()
        if alias_value:
            normalized[canonical_key] = alias_value
    return normalized


def _classify_enrichment_runtime_status(
    *,
    layer_row: dict[str, Any] | None,
    consumed: bool,
) -> str:
    if not isinstance(layer_row, dict):
        return "not_configured"
    coverage_status = str(layer_row.get("coverage_status") or "").strip().lower()
    if coverage_status == "not_configured":
        return "not_configured"
    if coverage_status in {"missing_file", "sampling_failed"}:
        return "configured_but_fetch_failed"
    if coverage_status in {"outside_extent", "fallback_used"}:
        return "configured_but_no_coverage"
    if coverage_status in {"observed", "partial"}:
        return "present_and_consumed" if consumed else "present_but_not_consumed"
    return "configured_but_no_coverage"


def _build_enrichment_runtime_status(
    *,
    layer_coverage_audit: list[dict[str, Any]] | None,
    property_level_context: dict[str, Any],
) -> dict[str, str]:
    audit_map: dict[str, dict[str, Any]] = {}
    for row in (layer_coverage_audit or []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("layer_key") or "").strip()
        if key:
            audit_map[key] = row

    hazard_context = property_level_context.get("hazard_context")
    moisture_context = property_level_context.get("moisture_context")
    historical_fire_context = property_level_context.get("historical_fire_context")
    access_context = property_level_context.get("access_context")
    naip_source = str(property_level_context.get("naip_feature_source") or "").strip().lower()

    consumed_map = {
        "whp": isinstance(hazard_context, dict) and str(hazard_context.get("status") or "") == "ok",
        "mtbs_severity": isinstance(historical_fire_context, dict) and str(historical_fire_context.get("status") or "") == "ok",
        "gridmet_dryness": isinstance(moisture_context, dict) and str(moisture_context.get("status") or "") == "ok",
        "roads": isinstance(access_context, dict) and str(access_context.get("status") or "") == "ok",
        "naip_structure_features": naip_source == "prepared_region_naip",
    }

    return {
        layer_key: _classify_enrichment_runtime_status(
            layer_row=audit_map.get(layer_key),
            consumed=bool(consumed_map.get(layer_key)),
        )
        for layer_key in ("whp", "mtbs_severity", "gridmet_dryness", "roads", "naip_structure_features")
    }


def apply_enrichment_source_fallbacks(
    runtime_paths: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[str]]:
    updated = _normalize_runtime_aliases(runtime_paths)
    source_status: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    for group in SOURCE_GROUPS:
        selected_runtime_key: str | None = None
        selected_path: str | None = None
        selected_source: str | None = None
        fallback_applied = False

        runtime_candidates = [
            (runtime_key, str(updated.get(runtime_key) or ""))
            for runtime_key, _source_name in group.runtime_candidates
        ]
        runtime_key_hit, runtime_path_hit = _first_existing_path(runtime_candidates)
        if runtime_key_hit and runtime_path_hit:
            selected_runtime_key = runtime_key_hit
            selected_path = runtime_path_hit
            selected_source = dict(group.runtime_candidates).get(runtime_key_hit)

        if selected_path is None:
            for env_name, runtime_key, source_name in group.env_candidates:
                env_path = str(os.getenv(env_name, "")).strip()
                if not _file_exists(env_path):
                    continue
                updated[runtime_key] = env_path
                selected_runtime_key = runtime_key
                selected_path = env_path
                selected_source = source_name
                fallback_applied = True
                notes.append(f"Enrichment fallback applied: {group.key} sourced from {env_name}.")
                break

        source_status[group.key] = {
            "status": "observed" if selected_path else "missing",
            "source": selected_source,
            "runtime_key": selected_runtime_key,
            "path": selected_path,
            "fallback_applied": fallback_applied,
        }
        if selected_path is None:
            notes.append(f"Enrichment source missing: {group.key}.")

    return updated, source_status, notes


def build_feature_bundle_summary(
    *,
    lat: float,
    lon: float,
    region_context: dict[str, Any],
    property_level_context: dict[str, Any],
    source_status: dict[str, dict[str, Any]],
    runtime_paths: dict[str, str],
    environmental_layer_status: dict[str, str],
    layer_coverage_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ring_metrics = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    ring_metrics = ring_metrics if isinstance(ring_metrics, dict) else {}
    normalized_geometry_basis = str(property_level_context.get("geometry_basis") or "").strip().lower()
    if normalized_geometry_basis in {"footprint", "parcel", "point", "geocode_point"}:
        geometry_basis = "geocode_point" if normalized_geometry_basis == "point" else normalized_geometry_basis
    else:
        geometry_basis = (
            "footprint"
            if bool(property_level_context.get("footprint_used"))
            else ("parcel" if bool(property_level_context.get("parcel_geometry")) else "geocode_point")
        )

    data_sources = {
        "building_footprint": (source_status.get("building_footprint") or {}).get("source") or "missing",
        "parcel": (source_status.get("parcel") or {}).get("source") or "missing",
        "vegetation": (source_status.get("vegetation") or {}).get("source") or "missing",
        "canopy": (source_status.get("canopy") or {}).get("source") or "missing",
        "burn_probability": (source_status.get("burn_probability") or {}).get("source") or "missing",
        "historical_fire": (source_status.get("historical_fire") or {}).get("source") or "missing",
        "roads": (source_status.get("roads") or {}).get("source") or "missing",
        "climate_dryness": (source_status.get("climate_dryness") or {}).get("source") or "missing",
        "naip_imagery": (source_status.get("naip_imagery") or {}).get("source") or "missing",
        "naip_structure_features": (source_status.get("naip_structure_features") or {}).get("source") or "missing",
    }
    coverage_flags = {
        key: value != "missing"
        for key, value in data_sources.items()
    }
    source_observed_count = sum(1 for item in source_status.values() if str(item.get("status") or "") == "observed")
    source_missing_count = sum(1 for item in source_status.values() if str(item.get("status") or "") == "missing")
    source_fallback_count = sum(
        1
        for item in source_status.values()
        if bool(item.get("fallback_applied")) and str(item.get("status") or "") == "observed"
    )

    feature_snapshot = {
        "ring_metric_keys": sorted([key for key in ring_metrics.keys() if key.startswith("ring_")]),
        "nearest_vegetation_distance_ft": property_level_context.get("nearest_vegetation_distance_ft"),
        "near_structure_vegetation_0_5_pct": property_level_context.get("near_structure_vegetation_0_5_pct"),
        "canopy_adjacency_proxy_pct": property_level_context.get("canopy_adjacency_proxy_pct"),
        "vegetation_continuity_proxy_pct": property_level_context.get("vegetation_continuity_proxy_pct"),
        "nearest_high_fuel_patch_distance_ft": property_level_context.get("nearest_high_fuel_patch_distance_ft"),
        "access_exposure_index": ((property_level_context.get("access_context") or {}).get("access_exposure_index")),
    }
    feature_sampling = (
        property_level_context.get("feature_sampling")
        if isinstance(property_level_context.get("feature_sampling"), dict)
        else {}
    )
    observed_feature_count = 0
    inferred_feature_count = 0
    fallback_feature_count = 0
    missing_feature_count = 0
    for feature in feature_sampling.values():
        if not isinstance(feature, dict):
            missing_feature_count += 1
            continue
        scope = str(feature.get("scope") or "unknown").strip().lower()
        value = feature.get("index")
        if scope in {"property_specific", "neighborhood_level", "region_level"} and value is not None:
            observed_feature_count += 1
        elif scope in {"inferred", "estimated"}:
            inferred_feature_count += 1
        elif scope == "fallback":
            fallback_feature_count += 1
        else:
            missing_feature_count += 1

    # Some benchmark and legacy contexts do not provide feature_sampling.
    # Derive a deterministic fallback count profile from available context signals
    # so observed evidence is not misclassified as "zero observed features".
    if not feature_sampling:
        inferred_feature_count = 0
        observed_feature_count = 0
        fallback_feature_count = 0
        missing_feature_count = 0

        for item in source_status.values():
            if not isinstance(item, dict):
                missing_feature_count += 1
                continue
            status = str(item.get("status") or "missing").strip().lower()
            if status == "observed":
                if bool(item.get("fallback_applied")):
                    fallback_feature_count += 1
                else:
                    observed_feature_count += 1
            elif status in {"inferred", "estimated"}:
                inferred_feature_count += 1
            else:
                missing_feature_count += 1

        for key in ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "zone_0_5_ft", "zone_5_30_ft", "zone_30_100_ft"):
            ring = ring_metrics.get(key) if isinstance(ring_metrics, dict) else None
            if isinstance(ring, dict) and ring.get("vegetation_density") is not None:
                observed_feature_count += 1

        for field in (
            "near_structure_vegetation_0_5_pct",
            "canopy_adjacency_proxy_pct",
            "vegetation_continuity_proxy_pct",
            "nearest_high_fuel_patch_distance_ft",
            "access_exposure_index",
        ):
            if feature_snapshot.get(field) is not None:
                observed_feature_count += 1

        if observed_feature_count == 0 and fallback_feature_count == 0 and missing_feature_count == 0:
            for available in coverage_flags.values():
                if available:
                    observed_feature_count += 1
                else:
                    missing_feature_count += 1
    total_feature_count = max(
        1,
        observed_feature_count + inferred_feature_count + fallback_feature_count + missing_feature_count,
    )
    observed_weight_fraction = round(float(observed_feature_count) / float(total_feature_count), 3)
    fallback_dominance_ratio = round(float(fallback_feature_count) / float(total_feature_count), 3)
    geometry_quality_by_basis = {
        "footprint": 0.92,
        "parcel": 0.74,
        "geocode_point": 0.46,
    }
    geometry_quality_score = float(geometry_quality_by_basis.get(geometry_basis, 0.46))
    anchor_quality_score = None
    try:
        anchor_quality_score = float(
            property_level_context.get("property_anchor_quality_score")
            if property_level_context.get("property_anchor_quality_score") is not None
            else property_level_context.get("anchor_quality_score")
        )
    except (TypeError, ValueError):
        anchor_quality_score = None
    structure_geometry_confidence = None
    try:
        structure_geometry_confidence = float(property_level_context.get("structure_geometry_confidence"))
    except (TypeError, ValueError):
        structure_geometry_confidence = None
    if structure_geometry_confidence is not None:
        geometry_quality_score = (
            (geometry_quality_score * 0.5)
            + (max(0.0, min(1.0, structure_geometry_confidence)) * 0.35)
            + (max(0.0, min(1.0, anchor_quality_score)) * 0.15 if anchor_quality_score is not None else 0.0)
        )
    elif anchor_quality_score is not None:
        geometry_quality_score = (geometry_quality_score * 0.7) + (max(0.0, min(1.0, anchor_quality_score)) * 0.3)
    environmental_keys = ("burn_probability", "historical_fire", "roads", "climate_dryness", "vegetation", "canopy")
    environmental_layer_coverage_score = round(
        (
            sum(1 for key in environmental_keys if coverage_flags.get(key))
            / float(len(environmental_keys))
        )
        * 100.0,
        1,
    )
    property_specificity_score = round(
        max(
            0.0,
            min(
                100.0,
                (geometry_quality_score * 45.0)
                + (environmental_layer_coverage_score * 0.35)
                + (observed_weight_fraction * 20.0),
            ),
        ),
        1,
    )
    enrichment_runtime_status = _build_enrichment_runtime_status(
        layer_coverage_audit=layer_coverage_audit,
        property_level_context=property_level_context,
    )
    enrichment_total = max(1, len(enrichment_runtime_status))
    enrichment_consumed_count = sum(
        1 for status in enrichment_runtime_status.values() if status == "present_and_consumed"
    )
    enrichment_present_not_consumed_count = sum(
        1 for status in enrichment_runtime_status.values() if status == "present_but_not_consumed"
    )
    enrichment_missing_count = sum(
        1
        for status in enrichment_runtime_status.values()
        if status in {"not_configured", "configured_but_fetch_failed", "configured_but_no_coverage"}
    )
    regional_enrichment_consumption_score = round(
        (float(enrichment_consumed_count) / float(enrichment_total)) * 100.0,
        1,
    )

    return {
        "bundle_schema_version": "1.0.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "query_point": {"latitude": round(float(lat), 7), "longitude": round(float(lon), 7)},
        "region_id": region_context.get("region_id"),
        "region_status": region_context.get("region_status"),
        "geometry_basis": geometry_basis,
        "data_sources": data_sources,
        "coverage_flags": coverage_flags,
        "coverage_metrics": {
            "source_observed_count": source_observed_count,
            "source_missing_count": source_missing_count,
            "source_fallback_count": source_fallback_count,
            "observed_feature_count": observed_feature_count,
            "inferred_feature_count": inferred_feature_count,
            "fallback_feature_count": fallback_feature_count,
            "missing_feature_count": missing_feature_count,
            "observed_weight_fraction": observed_weight_fraction,
            "fallback_dominance_ratio": fallback_dominance_ratio,
            "structure_geometry_quality_score": round(float(geometry_quality_score), 3),
            "structure_geometry_confidence": (
                round(max(0.0, min(1.0, float(structure_geometry_confidence))), 3)
                if structure_geometry_confidence is not None
                else None
            ),
            "anchor_quality_score": (
                round(max(0.0, min(1.0, float(anchor_quality_score))), 3)
                if anchor_quality_score is not None
                else None
            ),
            "environmental_layer_coverage_score": environmental_layer_coverage_score,
            "regional_enrichment_consumption_score": regional_enrichment_consumption_score,
            "enrichment_layers_consumed_count": enrichment_consumed_count,
            "enrichment_layers_present_not_consumed_count": enrichment_present_not_consumed_count,
            "enrichment_layers_missing_count": enrichment_missing_count,
            "property_specificity_score": property_specificity_score,
        },
        "geometry_provenance": {
            "geometry_basis": geometry_basis,
            "property_anchor_source": property_level_context.get("property_anchor_source"),
            "property_anchor_precision": property_level_context.get("property_anchor_precision"),
            "property_anchor_quality": property_level_context.get("property_anchor_quality"),
            "property_anchor_quality_score": property_level_context.get("property_anchor_quality_score"),
            "property_anchor_selection_method": property_level_context.get("property_anchor_selection_method"),
            "structure_selection_method": (
                property_level_context.get("structure_selection_method")
                or property_level_context.get("structure_match_method")
            ),
            "footprint_source": property_level_context.get("footprint_source"),
            "parcel_lookup_method": property_level_context.get("parcel_lookup_method"),
            "parcel_source": (
                property_level_context.get("parcel_source")
                or property_level_context.get("parcel_source_name")
            ),
        },
        "feature_snapshot": feature_snapshot,
        "enrichment_runtime_status": enrichment_runtime_status,
        "environmental_layer_status": dict(environmental_layer_status or {}),
        "runtime_layer_paths": {k: v for k, v in runtime_paths.items() if str(v or "").strip()},
    }
