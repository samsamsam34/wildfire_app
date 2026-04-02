from __future__ import annotations

from backend.differentiation import build_differentiation_snapshot


def test_local_differentiation_score_is_high_with_full_local_inputs() -> None:
    snapshot = build_differentiation_snapshot(
        feature_coverage_summary={
            "building_footprint_available": True,
            "parcel_polygon_available": True,
            "near_structure_vegetation_available": True,
            "hazard_severity_available": True,
            "burn_probability_available": True,
        },
        preflight={"geometry_basis": "footprint"},
        property_level_context={
            "footprint_used": True,
            "parcel_geometry": {"type": "Polygon", "coordinates": []},
            "naip_feature_source": "prepared_region_naip",
            "near_structure_vegetation_0_5_pct": 42.0,
            "canopy_adjacency_proxy_pct": 36.0,
            "vegetation_continuity_proxy_pct": 31.0,
            "nearest_high_fuel_patch_distance_ft": 22.0,
        },
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        fallback_weight_fraction=0.08,
        missing_inputs=[],
        inferred_inputs=[],
        input_source_metadata={
            "roof_type": {"source_type": "observed"},
            "vent_type": {"source_type": "observed"},
            "defensible_space_ft": {"source_type": "observed"},
            "construction_year": {"source_type": "observed"},
            "zone_0_5_ft": {"source_type": "footprint_derived"},
            "zone_5_30_ft": {"source_type": "footprint_derived"},
            "zone_30_100_ft": {"source_type": "footprint_derived"},
            "near_structure_vegetation_0_5_pct": {"source_type": "observed"},
            "canopy_adjacency_proxy_pct": {"source_type": "observed"},
            "vegetation_continuity_proxy_pct": {"source_type": "observed"},
            "nearest_high_fuel_patch_distance_ft": {"source_type": "observed"},
            "burn_probability": {"source_type": "observed"},
            "wildfire_hazard": {"source_type": "observed"},
            "slope": {"source_type": "observed"},
            "fuel_model": {"source_type": "observed"},
            "canopy_cover": {"source_type": "observed"},
            "historic_fire_distance": {"source_type": "observed"},
            "wildland_distance": {"source_type": "observed"},
            "moisture": {"source_type": "observed"},
        },
        fallback_decisions=[],
    )

    assert snapshot["differentiation_mode"] == "highly_local"
    assert float(snapshot["local_differentiation_score"] or 0.0) >= 70.0
    assert float(snapshot["neighborhood_differentiation_confidence"] or 0.0) >= 70.0


def test_local_differentiation_score_is_low_when_fallback_heavy() -> None:
    snapshot = build_differentiation_snapshot(
        feature_coverage_summary={
            "building_footprint_available": False,
            "parcel_polygon_available": False,
            "near_structure_vegetation_available": False,
            "hazard_severity_available": False,
            "burn_probability_available": False,
        },
        preflight={"geometry_basis": "point_based"},
        property_level_context={
            "footprint_used": False,
            "parcel_geometry": None,
            "fallback_mode": "point_based",
        },
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        fallback_weight_fraction=0.72,
        missing_inputs=["roof_type", "vent_type", "defensible_space_ft"],
        inferred_inputs=["construction_year"],
        input_source_metadata={
            "roof_type": {"source_type": "missing"},
            "vent_type": {"source_type": "missing"},
            "defensible_space_ft": {"source_type": "missing"},
            "construction_year": {"source_type": "public_record_inferred"},
            "burn_probability": {"source_type": "missing"},
            "wildfire_hazard": {"source_type": "missing"},
            "slope": {"source_type": "observed"},
            "fuel_model": {"source_type": "observed"},
            "canopy_cover": {"source_type": "observed"},
            "historic_fire_distance": {"source_type": "heuristic"},
            "wildland_distance": {"source_type": "heuristic"},
        },
        fallback_decisions=[
            {"fallback_type": "point_based_context"},
            {"fallback_type": "layer_proxy"},
        ],
    )

    assert snapshot["differentiation_mode"] == "mostly_regional"
    assert float(snapshot["local_differentiation_score"] or 0.0) <= 35.0
    assert float(snapshot["neighborhood_differentiation_confidence"] or 0.0) <= 35.0
