from __future__ import annotations

from pathlib import Path

from backend.feature_bundle_cache import FeatureBundleCache
from backend.feature_enrichment import (
    apply_enrichment_source_fallbacks,
    build_feature_bundle_summary,
)
from backend.wildfire_data import WildfireDataClient


def _touch(path: Path, content: str = "{}") -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_apply_enrichment_source_fallbacks_uses_external_paths(monkeypatch, tmp_path):
    ms_path = _touch(tmp_path / "microsoft_buildings.geojson")
    parcels_path = _touch(tmp_path / "parcel_polygons.geojson")
    address_points_path = _touch(tmp_path / "county_address_points.geojson")
    roads_path = _touch(tmp_path / "roads.geojson")
    dryness_path = _touch(tmp_path / "gridmet_dryness.tif")

    monkeypatch.setenv("WF_ENRICH_MICROSOFT_BUILDINGS_PATH", ms_path)
    monkeypatch.setenv("WF_ENRICH_PARCELS_PATH", parcels_path)
    monkeypatch.setenv("WF_ENRICH_ADDRESS_POINTS_PATH", address_points_path)
    monkeypatch.setenv("WF_ENRICH_OSM_ROADS_GEOJSON", roads_path)
    monkeypatch.setenv("WF_ENRICH_GRIDMET_DRYNESS_TIF", dryness_path)

    runtime_paths = {
        "footprints_overture": "",
        "footprints_microsoft": "",
        "footprints": "",
        "fema_structures": "",
        "parcels": "",
        "address_points": "",
        "roads": "",
        "gridmet_dryness": "",
        "moisture": "",
    }
    updated, status, notes = apply_enrichment_source_fallbacks(runtime_paths)

    assert updated["footprints_microsoft"] == ms_path
    assert updated["parcels"] == parcels_path
    assert updated["address_points"] == address_points_path
    assert updated["roads"] == roads_path
    assert updated["gridmet_dryness"] == dryness_path

    assert status["building_footprint"]["status"] == "observed"
    assert status["building_footprint"]["source"] == "microsoft_buildings"
    assert status["parcel"]["status"] == "observed"
    assert status["roads"]["status"] == "observed"
    assert status["climate_dryness"]["status"] == "observed"
    assert any("Enrichment fallback applied" in note for note in notes)


def test_build_feature_bundle_summary_returns_source_map():
    summary = build_feature_bundle_summary(
        lat=48.47,
        lon=-120.18,
        region_context={"region_id": "winthrop_large", "region_status": "prepared"},
        property_level_context={
            "footprint_used": True,
            "structure_selection_method": "parcel_intersection",
            "parcel_source_name": "county_or_regrid_parcels",
            "ring_metrics": {
                "ring_0_5_ft": {"vegetation_density": 66.0},
                "ring_5_30_ft": {"vegetation_density": 72.0},
            },
            "near_structure_vegetation_0_5_pct": 61.0,
            "canopy_adjacency_proxy_pct": 44.0,
            "vegetation_continuity_proxy_pct": 58.0,
            "nearest_high_fuel_patch_distance_ft": 28.0,
            "feature_sampling": {
                "burn_probability": {"index": 72.0, "scope": "region_level"},
                "hazard_severity": {"index": 67.0, "scope": "region_level"},
                "slope": {"index": 51.0, "scope": "property_specific"},
                "fuel_model": {"index": 64.0, "scope": "neighborhood_level"},
                "canopy_cover": {"index": 54.0, "scope": "neighborhood_level"},
            },
        },
        source_status={
            "building_footprint": {"source": "microsoft_buildings", "status": "observed"},
            "parcel": {"source": "county_or_regrid_parcels", "status": "observed"},
            "vegetation": {"source": "landfire_fuel", "status": "observed"},
            "canopy": {"source": "landfire_canopy", "status": "observed"},
            "burn_probability": {"source": "burn_probability_raster", "status": "observed"},
            "historical_fire": {"source": "mtbs_severity", "status": "observed"},
            "roads": {"source": "osm_roads", "status": "observed"},
            "climate_dryness": {"source": "gridmet_dryness", "status": "observed"},
            "naip_imagery": {"source": "naip_imagery", "status": "observed"},
            "naip_structure_features": {"source": "naip_structure_features", "status": "observed"},
        },
        runtime_paths={"fuel": "/tmp/fuel.tif", "canopy": "/tmp/canopy.tif"},
        environmental_layer_status={"fuel": "ok", "canopy": "ok"},
    )

    assert summary["geometry_basis"] == "footprint"
    assert summary["data_sources"]["building_footprint"] == "microsoft_buildings"
    assert summary["data_sources"]["parcel"] == "county_or_regrid_parcels"
    assert summary["coverage_flags"]["roads"] is True
    assert summary["feature_snapshot"]["near_structure_vegetation_0_5_pct"] == 61.0
    metrics = summary["coverage_metrics"]
    assert metrics["observed_feature_count"] >= 4
    assert metrics["fallback_feature_count"] == 0
    assert metrics["environmental_layer_coverage_score"] > 0
    assert metrics["property_specificity_score"] > 0
    geometry = summary["geometry_provenance"]
    assert geometry["geometry_basis"] == "footprint"
    assert geometry["structure_selection_method"] == "parcel_intersection"
    assert geometry["parcel_source"] == "county_or_regrid_parcels"


def test_apply_enrichment_source_fallbacks_normalizes_alias_runtime_keys(tmp_path):
    roads_alias = _touch(tmp_path / "road_network.geojson")
    runtime_paths = {
        "road_network": roads_alias,
        "roads": "",
    }

    updated, status, _notes = apply_enrichment_source_fallbacks(runtime_paths)

    assert updated["roads"] == roads_alias
    assert status["roads"]["status"] == "observed"


def test_build_feature_bundle_summary_reports_enrichment_consumption_status():
    summary = build_feature_bundle_summary(
        lat=46.87,
        lon=-113.99,
        region_context={"region_id": "missoula_pilot", "region_status": "prepared"},
        property_level_context={
            "footprint_used": True,
            "hazard_context": {"status": "ok"},
            "moisture_context": {"status": "missing"},
            "historical_fire_context": {"status": "ok"},
            "access_context": {"status": "sampling_failed"},
            "naip_feature_source": None,
            "feature_sampling": {
                "burn_probability": {"index": 55.0, "scope": "region_level"},
                "hazard_severity": {"index": 52.0, "scope": "region_level"},
                "moisture_dryness": {"index": None, "scope": "fallback"},
            },
        },
        source_status={
            "building_footprint": {"source": "building_footprints", "status": "observed"},
            "parcel": {"source": "county_or_regrid_parcels", "status": "observed"},
            "vegetation": {"source": "landfire_fuel", "status": "observed"},
            "canopy": {"source": "landfire_canopy", "status": "observed"},
            "burn_probability": {"source": "whp", "status": "observed"},
            "historical_fire": {"source": "mtbs_severity", "status": "observed"},
            "roads": {"source": "osm_roads", "status": "observed"},
            "climate_dryness": {"source": "gridmet_dryness", "status": "observed"},
            "naip_imagery": {"source": "naip_imagery", "status": "observed"},
            "naip_structure_features": {"source": "naip_structure_features", "status": "observed"},
        },
        runtime_paths={},
        environmental_layer_status={"hazard": "ok", "fire_history": "ok"},
        layer_coverage_audit=[
            {"layer_key": "whp", "coverage_status": "observed"},
            {"layer_key": "mtbs_severity", "coverage_status": "observed"},
            {"layer_key": "gridmet_dryness", "coverage_status": "outside_extent"},
            {"layer_key": "roads", "coverage_status": "sampling_failed"},
            {"layer_key": "naip_structure_features", "coverage_status": "observed"},
        ],
    )

    statuses = summary["enrichment_runtime_status"]
    assert statuses["whp"] == "present_and_consumed"
    assert statuses["mtbs_severity"] == "present_and_consumed"
    assert statuses["gridmet_dryness"] == "configured_but_no_coverage"
    assert statuses["roads"] == "configured_but_fetch_failed"
    assert statuses["naip_structure_features"] == "present_but_not_consumed"
    metrics = summary["coverage_metrics"]
    assert metrics["regional_enrichment_consumption_score"] >= 30.0
    assert metrics["enrichment_layers_present_not_consumed_count"] >= 1


def test_feature_bundle_cache_roundtrip(tmp_path):
    cache = FeatureBundleCache(cache_dir=str(tmp_path), enabled=True, ttl_seconds=3600)
    layer_path = _touch(tmp_path / "fuel.tif", content="dummy")
    key = cache.build_key(
        lat=47.0,
        lon=-113.9,
        runtime_paths={"fuel": layer_path},
        region_context={"region_status": "prepared", "region_id": "missoula_pilot"},
        extras={"selection_mode": "polygon"},
    )
    payload = {"wildfire_context": {"environmental_index": 52.5}}
    cache.save(key, payload)
    loaded = cache.load(key)
    assert loaded == payload


def test_runtime_building_priority_supports_microsoft(monkeypatch):
    monkeypatch.setenv("WF_BUILDING_SOURCE_PRIORITY", "building_footprints_microsoft,building_footprints")
    paths = WildfireDataClient._resolve_building_source_paths(
        {
            "footprints_microsoft": "/tmp/ms.geojson",
            "footprints": "/tmp/fallback.geojson",
        },
        {"building_sources": []},
    )
    assert paths[0] == "/tmp/ms.geojson"
