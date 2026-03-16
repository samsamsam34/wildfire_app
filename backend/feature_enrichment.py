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


def apply_enrichment_source_fallbacks(
    runtime_paths: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[str]]:
    updated = dict(runtime_paths)
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
) -> dict[str, Any]:
    ring_metrics = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    ring_metrics = ring_metrics if isinstance(ring_metrics, dict) else {}
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

    feature_snapshot = {
        "ring_metric_keys": sorted([key for key in ring_metrics.keys() if key.startswith("ring_")]),
        "nearest_vegetation_distance_ft": property_level_context.get("nearest_vegetation_distance_ft"),
        "near_structure_vegetation_0_5_pct": property_level_context.get("near_structure_vegetation_0_5_pct"),
        "canopy_adjacency_proxy_pct": property_level_context.get("canopy_adjacency_proxy_pct"),
        "vegetation_continuity_proxy_pct": property_level_context.get("vegetation_continuity_proxy_pct"),
        "nearest_high_fuel_patch_distance_ft": property_level_context.get("nearest_high_fuel_patch_distance_ft"),
        "access_exposure_index": ((property_level_context.get("access_context") or {}).get("access_exposure_index")),
    }

    return {
        "bundle_schema_version": "1.0.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "query_point": {"latitude": round(float(lat), 7), "longitude": round(float(lon), 7)},
        "region_id": region_context.get("region_id"),
        "region_status": region_context.get("region_status"),
        "geometry_basis": geometry_basis,
        "data_sources": data_sources,
        "coverage_flags": coverage_flags,
        "feature_snapshot": feature_snapshot,
        "environmental_layer_status": dict(environmental_layer_status or {}),
        "runtime_layer_paths": {k: v for k, v in runtime_paths.items() if str(v or "").strip()},
    }
