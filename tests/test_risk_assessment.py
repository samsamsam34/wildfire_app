from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover - optional in constrained test runners
    Polygon = None

import backend.auth as auth
import backend.main as app_main
import backend.data_prep.prepare_region as prep_region_module
import backend.wildfire_data as wildfire_data_module
from backend.building_footprints import BuildingFootprintClient, BuildingFootprintResult, compute_structure_rings
from backend.data_prep.prepare_region import prepare_region_layers
from backend.database import AssessmentStore
from backend.geocoding import GeocodingError
from backend.mitigation import build_mitigation_plan
from backend.models import AssumptionsBlock, PropertyAttributes
from backend.open_data_adapters import (
    GridMETDrynessObservation,
    MTBSSummary,
    OSMAccessSummary,
    WHPObservation,
)
from backend.region_registry import find_region_for_point, list_prepared_regions, load_region_manifest
from backend.version import (
    API_VERSION,
    CALIBRATION_VERSION,
    FACTOR_SCHEMA_VERSION,
    LEGACY_MODEL_VERSION,
    MODEL_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
)
from backend.wildfire_data import WildfireContext, WildfireDataClient, compute_environmental_data_completeness


client = TestClient(app_main.app)
FIXTURE_DIR = Path(__file__).parent / "fixtures"


REQUIRED_SUBMODELS = [
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "slope_topography_risk",
    "ember_exposure_risk",
    "flame_contact_risk",
    "historic_fire_risk",
    "structure_vulnerability_risk",
    "defensible_space_risk",
]


def _require_shapely() -> None:
    if Polygon is None:
        pytest.skip("shapely is not installed in this environment")


def _require_region_prep_deps() -> None:
    if getattr(prep_region_module, "rasterio", None) is None:
        pytest.skip("rasterio is not installed in this environment")
    if getattr(prep_region_module, "shape", None) is None:
        pytest.skip("shapely is not available for region prep tests")


def _require_geo_runtime_deps() -> None:
    if getattr(wildfire_data_module, "rasterio", None) is None:
        pytest.skip("rasterio is not installed in this environment")
    if getattr(wildfire_data_module, "Transformer", None) is None:
        pytest.skip("pyproj is not installed in this environment")


def _ctx(
    env: float,
    wildland: float,
    historic: float,
    ring_metrics: dict[str, dict[str, float | None]] | None = None,
    environmental_layer_status: dict[str, str] | None = None,
) -> WildfireContext:
    ring_metrics = ring_metrics or {}
    environmental_layer_status = environmental_layer_status or {
        "burn_probability": "ok",
        "hazard": "ok",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "ok",
        "fire_history": "ok",
    }
    return WildfireContext(
        environmental_index=env,
        slope_index=env,
        aspect_index=50.0,
        fuel_index=env,
        moisture_index=env,
        canopy_index=env,
        wildland_distance_index=wildland,
        historic_fire_index=historic,
        burn_probability_index=env,
        hazard_severity_index=env,
        burn_probability=env,
        wildfire_hazard=env,
        slope=env,
        fuel_model=env,
        canopy_cover=env,
        historic_fire_distance=1.2,
        wildland_distance=120.0,
        environmental_layer_status=environmental_layer_status,
        data_sources=[
            "Burn probability raster",
            "Wildfire hazard severity raster",
            "Fuel model raster",
            "Historical fire perimeter recurrence",
            "Slope raster",
            "Aspect raster",
        ],
        assumptions=[],
        structure_ring_metrics=ring_metrics,
        property_level_context={
            "footprint_used": bool(ring_metrics),
            "footprint_status": "used" if ring_metrics else "not_found",
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "ring_metrics": ring_metrics,
        },
    )


def _setup(monkeypatch, tmp_path, context: WildfireContext) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_assessments.db")))


def _setup_with_collect_capture(monkeypatch, tmp_path, context: WildfireContext, captured: dict[str, object]) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))

    def _collect(lat, lon, **kwargs):
        captured["lat"] = lat
        captured["lon"] = lon
        captured["kwargs"] = dict(kwargs)
        return context

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", _collect)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_assessments.db")))


def _payload(address: str, attrs: dict, confirmed: list[str] | None = None, tags: list[str] | None = None) -> dict:
    return {
        "address": address,
        "attributes": attrs,
        "confirmed_fields": confirmed or [],
        "audience": "homeowner",
        "tags": tags or [],
    }


def _assert_core_contract(body: dict) -> None:
    required = [
        "assessment_id",
        "address",
        "model_version",
        "generated_at",
        "wildfire_risk_score",
        "overall_wildfire_risk",
        "legacy_weighted_wildfire_risk_score",
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "insurance_readiness_score",
        "home_hardening_readiness",
        "wildfire_risk_score_available",
        "site_hazard_score_available",
        "home_ignition_vulnerability_score_available",
        "insurance_readiness_score_available",
        "home_hardening_readiness_score_available",
        "submodel_scores",
        "weighted_contributions",
        "submodel_explanations",
        "factor_breakdown",
        "score_summaries",
        "property_findings",
        "defensible_space_analysis",
        "top_near_structure_risk_drivers",
        "prioritized_vegetation_actions",
        "defensible_space_limitations_summary",
        "top_risk_drivers",
        "top_risk_drivers_detailed",
        "prioritized_mitigation_actions",
        "confidence_summary",
        "top_recommended_actions",
        "top_protective_factors",
        "explanation_summary",
        "readiness_factors",
        "readiness_blockers",
        "readiness_penalties",
        "readiness_summary",
        "confirmed_inputs",
        "observed_inputs",
        "inferred_inputs",
        "missing_inputs",
        "assumptions_used",
        "assumptions_and_unknowns",
        "confidence_score",
        "data_completeness_score",
        "environmental_data_completeness_score",
        "confidence_tier",
        "use_restriction",
        "low_confidence_flags",
        "mitigation_plan",
        "data_sources",
        "environmental_layer_status",
        "input_source_metadata",
        "direct_data_coverage_score",
        "inferred_data_coverage_score",
        "missing_data_share",
        "data_provenance",
        "site_hazard_input_quality",
        "home_vulnerability_input_quality",
        "insurance_readiness_input_quality",
        "score_evidence_ledger",
        "evidence_quality_summary",
        "feature_coverage_summary",
        "feature_coverage_percent",
        "assessment_specificity_tier",
        "specificity_summary",
        "geometry_resolution",
        "limited_assessment_flag",
        "observed_factor_count",
        "missing_factor_count",
        "fallback_factor_count",
        "observed_feature_count",
        "inferred_feature_count",
        "fallback_feature_count",
        "missing_feature_count",
        "observed_weight_fraction",
        "fallback_dominance_ratio",
        "fallback_weight_fraction",
        "structure_data_completeness",
        "structure_assumption_mode",
        "structure_score_confidence",
        "geometry_quality_score",
        "regional_context_coverage_score",
        "property_specificity_score",
        "layer_coverage_audit",
        "coverage_summary",
        "site_hazard_eligibility",
        "home_vulnerability_eligibility",
        "insurance_readiness_eligibility",
        "coverage_available",
        "resolved_region_id",
        "assessment_status",
        "assessment_blockers",
        "assessment_diagnostics",
        "property_level_context",
        "site_hazard_section",
        "home_ignition_vulnerability_section",
        "insurance_readiness_section",
        "review_status",
        "product_version",
        "api_version",
        "scoring_model_version",
        "rules_logic_version",
        "factor_schema_version",
        "benchmark_pack_version",
        "calibration_version",
        "region_data_version",
        "data_bundle_version",
        "model_governance",
    ]
    for key in required:
        assert key in body

    for score_key, availability_key in [
        ("wildfire_risk_score", "wildfire_risk_score_available"),
        ("site_hazard_score", "site_hazard_score_available"),
        ("home_ignition_vulnerability_score", "home_ignition_vulnerability_score_available"),
        ("insurance_readiness_score", "insurance_readiness_score_available"),
    ]:
        if body[availability_key]:
            assert body[score_key] is not None
            assert 0.0 <= float(body[score_key]) <= 100.0
        else:
            assert body[score_key] is None

    assert 0.0 <= body["environmental_data_completeness_score"] <= 100.0
    assert 0.0 <= body["direct_data_coverage_score"] <= 100.0
    assert 0.0 <= body["inferred_data_coverage_score"] <= 100.0
    assert 0.0 <= body["missing_data_share"] <= 100.0
    coverage_total = round(
        body["direct_data_coverage_score"] + body["inferred_data_coverage_score"] + body["missing_data_share"],
        1,
    )
    assert abs(coverage_total - 100.0) <= 0.2
    assert body["risk_scores"]["site_hazard_score"] == body["site_hazard_score"]
    assert body["risk_scores"]["home_ignition_vulnerability_score"] == body["home_ignition_vulnerability_score"]
    assert body["risk_scores"]["wildfire_risk_score"] == body["wildfire_risk_score"]
    assert body["risk_scores"]["overall_wildfire_risk"] == body["overall_wildfire_risk"]
    assert body["risk_scores"]["insurance_readiness_score"] == body["insurance_readiness_score"]
    assert body["risk_scores"]["home_hardening_readiness"] == body["home_hardening_readiness"]
    assert body["risk_scores"]["site_hazard_score_available"] == body["site_hazard_score_available"]
    assert (
        body["risk_scores"]["home_ignition_vulnerability_score_available"]
        == body["home_ignition_vulnerability_score_available"]
    )
    assert body["risk_scores"]["wildfire_risk_score_available"] == body["wildfire_risk_score_available"]
    assert (
        body["risk_scores"]["insurance_readiness_score_available"]
        == body["insurance_readiness_score_available"]
    )
    assert (
        body["risk_scores"]["home_hardening_readiness_score_available"]
        == body["home_hardening_readiness_score_available"]
    )
    assert set(body["score_summaries"].keys()) == {
        "site_hazard",
        "home_ignition_vulnerability",
        "insurance_readiness",
    }
    assert body["model_governance"]["product_version"] == body["product_version"]
    assert body["model_governance"]["api_version"] == body["api_version"]
    assert body["model_governance"]["scoring_model_version"] == body["scoring_model_version"]
    assert body["model_governance"]["ruleset_version"] == body["ruleset_version"]
    assert body["model_governance"]["rules_logic_version"] == body["rules_logic_version"]
    assert body["model_governance"]["factor_schema_version"] == body["factor_schema_version"]
    assert body["model_governance"]["calibration_version"] == body["calibration_version"]
    for section in body["score_summaries"].values():
        assert "label" in section
        assert "score" in section
        assert "explanation" in section
        assert "top_drivers" in section
        assert "protective_factors" in section
        assert "next_actions" in section
        if section["score"] is not None:
            assert 0.0 <= float(section["score"]) <= 100.0

    assert body["confidence_tier"] in {"high", "moderate", "low", "preliminary"}
    assert isinstance(body["top_recommended_actions"], list)
    assert isinstance(body["top_risk_drivers_detailed"], list)
    assert isinstance(body["prioritized_mitigation_actions"], list)
    assert isinstance(body["confidence_summary"], dict)
    assert isinstance(body["assumptions_and_unknowns"], list)
    assert isinstance(body.get("near_structure_features"), dict)
    assert isinstance(body.get("directional_risk"), dict)
    for key in ("veg_density_0_5", "veg_density_5_30", "canopy_overlap", "hardscape_ratio"):
        assert key in body["near_structure_features"]
    assert body["use_restriction"] in {
        "shareable",
        "homeowner_review_recommended",
        "agent_or_inspector_review_recommended",
        "not_for_underwriting_or_binding",
    }
    assert body["assessment_status"] in {"fully_scored", "partially_scored", "insufficient_data"}

    for sm in REQUIRED_SUBMODELS:
        assert sm in body["submodel_scores"]
        assert sm in body["factor_breakdown"]["submodels"]
        assert "score" in body["submodel_scores"][sm]
        assert "weighted_contribution" in body["submodel_scores"][sm]

    assert "fallback_mode" in body["property_level_context"]
    assert body["property_level_context"]["fallback_mode"] in {"footprint", "point_based"}
    geometry_resolution = body["geometry_resolution"]
    assert isinstance(geometry_resolution, dict)
    for key in [
        "anchor_source",
        "anchor_quality_score",
        "parcel_match_status",
        "footprint_match_status",
        "footprint_source",
        "ring_generation_mode",
        "naip_structure_feature_status",
        "geometry_limitations",
    ]:
        assert key in geometry_resolution
    assert isinstance(geometry_resolution["anchor_source"], str)
    assert 0.0 <= float(geometry_resolution["anchor_quality_score"]) <= 1.0
    assert geometry_resolution["parcel_match_status"] in {
        "matched",
        "not_found",
        "provider_unavailable",
    }
    assert geometry_resolution["footprint_match_status"] in {
        "matched",
        "none",
        "ambiguous",
        "provider_unavailable",
        "error",
    }
    assert geometry_resolution["footprint_source"] is None or isinstance(geometry_resolution["footprint_source"], str)
    assert geometry_resolution["ring_generation_mode"] in {"footprint_aware_rings", "point_annulus_fallback"}
    assert geometry_resolution["naip_structure_feature_status"] in {
        "observed",
        "fallback_or_proxy",
        "missing",
        "present_but_not_consumed",
        "provider_unavailable",
    }
    assert isinstance(geometry_resolution["geometry_limitations"], list)
    footprint_resolution = body.get("footprint_resolution")
    assert isinstance(footprint_resolution, dict)
    for key in [
        "selected_source",
        "confidence_score",
        "candidates_considered",
        "fallback_used",
    ]:
        assert key in footprint_resolution
    assert footprint_resolution["selected_source"] is None or isinstance(
        footprint_resolution["selected_source"], str
    )
    assert 0.0 <= float(footprint_resolution["confidence_score"]) <= 1.0
    assert int(footprint_resolution["candidates_considered"]) >= 0
    assert isinstance(footprint_resolution["fallback_used"], bool)
    for layer in ["burn_probability", "hazard", "slope", "fuel", "canopy", "fire_history"]:
        assert layer in body["environmental_layer_status"]
    assert isinstance(body["input_source_metadata"], dict)
    assert "environmental_inputs_used" in body["data_provenance"]
    assert "property_inputs_used" in body["data_provenance"]
    assert "missing_inputs" in body["data_provenance"]
    assert "inputs" in body["data_provenance"]
    assert "summary" in body["data_provenance"]
    assert "stale_data_share" in body["data_provenance"]["summary"]
    for eligibility_key in [
        "site_hazard_eligibility",
        "home_vulnerability_eligibility",
        "insurance_readiness_eligibility",
    ]:
        eligibility = body[eligibility_key]
        assert "eligible" in eligibility
        assert "eligibility_status" in eligibility
        assert "blocking_reasons" in eligibility
        assert "caveats" in eligibility
        assert eligibility["eligibility_status"] in {"full", "partial", "insufficient"}
    for quality_key in [
        "site_hazard_input_quality",
        "home_vulnerability_input_quality",
        "insurance_readiness_input_quality",
    ]:
        quality = body[quality_key]
        assert "direct_coverage" in quality
        assert "inferred_coverage" in quality
        assert "stale_share" in quality
        assert "missing_share" in quality
        assert "heuristic_count" in quality
    ledger = body["score_evidence_ledger"]
    assert set(ledger.keys()) == {
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "insurance_readiness_score",
        "wildfire_risk_score",
    }
    for family, rows in ledger.items():
        assert isinstance(rows, list)
        for row in rows:
            for key in [
                "factor_key",
                "display_name",
                "category",
                "weight",
                "contribution",
                "direction",
                "evidence_status",
                "notes",
            ]:
                assert key in row
            assert row["evidence_status"] in {"observed", "inferred", "missing", "fallback"}
            assert row["direction"] in {
                "increases_risk",
                "reduces_risk",
                "blocks_readiness",
                "improves_readiness",
                "composes_score",
                "data_quality",
            }
    summary = body["evidence_quality_summary"]
    for key in [
        "observed_factor_count",
        "inferred_factor_count",
        "missing_factor_count",
        "fallback_factor_count",
        "evidence_quality_score",
        "confidence_penalties",
        "use_restriction",
    ]:
        assert key in summary
    assert summary["use_restriction"] in {"consumer_estimate", "screening_only", "review_required"}
    assert 0.0 <= float(summary["evidence_quality_score"]) <= 100.0
    assert isinstance(body["feature_coverage_summary"], dict)
    assert 0.0 <= float(body["feature_coverage_percent"]) <= 100.0
    assert body["assessment_specificity_tier"] in {"property_specific", "address_level", "regional_estimate"}
    assert isinstance(body["limited_assessment_flag"], bool)
    assert int(body["observed_factor_count"]) >= 0
    assert int(body["missing_factor_count"]) >= 0
    assert int(body["fallback_factor_count"]) >= 0
    assert int(body["observed_feature_count"]) >= 0
    assert int(body["inferred_feature_count"]) >= 0
    assert int(body["fallback_feature_count"]) >= 0
    assert int(body["missing_feature_count"]) >= 0
    assert 0.0 <= float(body["observed_weight_fraction"]) <= 1.0
    assert float(body["fallback_dominance_ratio"]) >= 0.0
    assert 0.0 <= float(body["fallback_weight_fraction"]) <= 1.0
    assert 0.0 <= float(body["geometry_quality_score"]) <= 1.0
    assert 0.0 <= float(body["regional_context_coverage_score"]) <= 100.0
    assert 0.0 <= float(body["property_specificity_score"]) <= 100.0
    coverage_rows = body["layer_coverage_audit"]
    assert isinstance(coverage_rows, list)
    for row in coverage_rows:
        for key in [
            "layer_key",
            "display_name",
            "required_for",
            "configured",
            "present_in_region",
            "sample_attempted",
            "sample_succeeded",
            "coverage_status",
            "source_type",
            "notes",
        ]:
            assert key in row
        assert row["coverage_status"] in {
            "observed",
            "missing_file",
            "not_configured",
            "outside_extent",
            "sampling_failed",
            "fallback_used",
            "partial",
        }
        assert row["source_type"] in {"prepared_region", "runtime_env", "derived", "open_data"}
    coverage_summary = body["coverage_summary"]
    for key in [
        "total_layers_checked",
        "observed_count",
        "partial_count",
        "fallback_count",
        "failed_count",
        "not_configured_count",
        "critical_missing_layers",
        "recommended_actions",
    ]:
        assert key in coverage_summary
    assert coverage_summary["total_layers_checked"] == len(coverage_rows)
    diagnostics = body["assessment_diagnostics"]
    for key in [
        "critical_inputs_present",
        "critical_inputs_missing",
        "stale_inputs",
        "inferred_inputs",
        "heuristic_inputs",
        "confidence_downgrade_reasons",
        "trust_tier_blockers",
    ]:
        assert key in diagnostics

    assert "confidence" in body and isinstance(body["confidence"], dict)
    for key in [
        "environmental_data_present",
        "property_context_present",
        "confirmed_fields_count",
        "inferred_fields_count",
        "missing_critical_fields",
        "confidence_drivers",
        "confidence_limiters",
    ]:
        assert key in body["confidence"]
    assert body["confidence"]["confidence_score"] == body["confidence_score"]


def test_assessment_passes_user_selected_structure_override_to_context(monkeypatch, tmp_path):
    context = _ctx(
        52.0,
        40.0,
        34.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 22.0},
            "ring_5_30_ft": {"vegetation_density": 35.0},
            "ring_30_100_ft": {"vegetation_density": 48.0},
            "ring_100_300_ft": {"vegetation_density": 50.0},
        },
    )
    captured: dict[str, object] = {}
    _setup_with_collect_capture(monkeypatch, tmp_path, context, captured)

    selected_geometry = {
        "type": "Feature",
        "properties": {"structure_id": "home-42"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-104.9910, 39.7390],
                [-104.9907, 39.7390],
                [-104.9907, 39.7393],
                [-104.9910, 39.7393],
                [-104.9910, 39.7390],
            ]],
        },
    }
    response = client.post(
        "/risk/assess",
        json={
            **_payload("1500 Market St, Denver, CO 80202", {"roof_type": "metal"}),
            "structure_geometry_source": "user_selected",
            "selected_structure_id": "home-42",
            "selected_structure_geometry": selected_geometry,
        },
    )
    assert response.status_code == 200
    body = response.json()

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("structure_geometry_source") == "user_selected"
    assert kwargs.get("selected_structure_id") == "home-42"
    assert kwargs.get("selected_structure_geometry") == selected_geometry

    plc = body.get("property_level_context") or {}
    assert plc.get("structure_geometry_source") == "user_selected"
    assert plc.get("selected_structure_id") == "home-42"
    assert plc.get("selected_structure_geometry", {}).get("geometry", {}).get("type") == "Polygon"


def test_assessment_passes_user_selected_point_override_to_context(monkeypatch, tmp_path):
    context = _ctx(
        52.0,
        40.0,
        34.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 24.0},
            "ring_5_30_ft": {"vegetation_density": 32.0},
            "ring_30_100_ft": {"vegetation_density": 45.0},
        },
    )
    context.property_level_context.update(
        {
            "selection_mode": "point",
            "user_selected_point": {"latitude": 39.7394, "longitude": -104.9901},
            "final_structure_geometry_source": "user_selected_point_unsnapped",
            "structure_geometry_confidence": 0.42,
            "snapped_structure_distance_m": None,
            "display_point_source": "property_anchor_point",
        }
    )
    captured: dict[str, object] = {}
    _setup_with_collect_capture(monkeypatch, tmp_path, context, captured)

    response = client.post(
        "/risk/assess",
        json={
            **_payload("1500 Market St, Denver, CO 80202", {"roof_type": "metal"}),
            "selection_mode": "point",
            "user_selected_point": {"latitude": 39.7394, "longitude": -104.9901},
        },
    )
    assert response.status_code == 200
    body = response.json()

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("selection_mode") == "point"
    assert kwargs.get("user_selected_point") == {"latitude": 39.7394, "longitude": -104.9901}

    plc = body.get("property_level_context") or {}
    assert plc.get("selection_mode") == "point"
    assert plc.get("user_selected_point") == {"latitude": 39.7394, "longitude": -104.9901}
    assert plc.get("geometry_basis") in {"point", "footprint", "parcel"}
    assert plc.get("geometry_source") in {
        "trusted_building_footprint",
        "parcel_geometry_inferred_home_location",
        "user_selected_map_point_snapped_structure",
        "user_selected_map_point_unsnapped",
        "raw_geocode_point",
    }
    assert isinstance(plc.get("geometry_confidence"), (int, float))
    assert plc.get("ring_generation_mode") in {"footprint_aware_rings", "point_annulus_fallback"}
    assert body.get("geometry_source") == plc.get("geometry_source")
    assert body.get("ring_generation_mode") == plc.get("ring_generation_mode")
    assert plc.get("structure_selection_method")
    assert plc.get("anchor_quality") in {"low", "medium", "high"}
    assert isinstance(plc.get("anchor_quality_score"), (int, float))
    assert body.get("final_structure_geometry_source") == "user_selected_point_unsnapped"


def test_nearby_homes_with_distinct_footprints_surface_distinct_ring_metrics_and_geometry_resolution(
    monkeypatch,
    tmp_path,
):
    auth.API_KEYS = set()

    def _geocode(address: str):
        if "home a" in address.lower():
            return (39.7392, -104.99032, "test-geocoder")
        return (39.7392, -104.99002, "test-geocoder")

    def _collect(lat, lon, **_kwargs):
        dense = float(lon) < -104.99015
        ring_0_5 = 82.0 if dense else 26.0
        ring_5_30 = 74.0 if dense else 33.0
        ring_metrics = {
            "ring_0_5_ft": {"vegetation_density": ring_0_5},
            "ring_5_30_ft": {"vegetation_density": ring_5_30},
            "ring_30_100_ft": {"vegetation_density": 58.0 if dense else 41.0},
        }
        ctx = _ctx(59.0, 63.0, 50.0, ring_metrics=ring_metrics)
        ctx.property_level_context.update(
            {
                "footprint_used": True,
                "footprint_status": "used",
                "structure_match_status": "matched",
                "structure_match_method": "parcel_intersection",
                "structure_match_confidence": 0.93,
                "ring_generation_mode": "footprint_aware_rings",
                "geometry_source": "trusted_building_footprint",
                "geometry_confidence": 0.93,
                "final_structure_geometry_source": "auto_detected",
                "footprint_source_name": "microsoft_buildings",
                "property_anchor_source": "authoritative_address_point",
                "property_anchor_quality_score": 0.95,
                "anchor_quality_score": 0.95,
                "parcel_id": "parcel-a" if dense else "parcel-b",
                "parcel_lookup_method": "contains_point",
            }
        )
        return ctx

    monkeypatch.setattr(app_main.geocoder, "geocode", _geocode)
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", _collect)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_assessments.db")))

    a = _run(_payload("Nearby Home A", {"roof_type": "class a", "vent_type": "ember-resistant"}))
    b = _run(_payload("Nearby Home B", {"roof_type": "class a", "vent_type": "ember-resistant"}))

    assert (
        a["property_level_context"]["ring_metrics"]["ring_0_5_ft"]["vegetation_density"]
        != b["property_level_context"]["ring_metrics"]["ring_0_5_ft"]["vegetation_density"]
    )
    assert a["ring_generation_mode"] == "footprint_aware_rings"
    assert b["ring_generation_mode"] == "footprint_aware_rings"
    assert a["geometry_resolution"]["footprint_match_status"] == "matched"
    assert b["geometry_resolution"]["footprint_match_status"] == "matched"
    assert a["geometry_resolution"]["footprint_source"] == "microsoft_buildings"
    assert b["geometry_resolution"]["footprint_source"] == "microsoft_buildings"
    assert a["geometry_resolution"]["parcel_match_status"] == "matched"
    assert b["geometry_resolution"]["parcel_match_status"] == "matched"
    assert a["geometry_resolution"]["ring_generation_mode"] == "footprint_aware_rings"
    assert b["geometry_resolution"]["ring_generation_mode"] == "footprint_aware_rings"
    assert a["near_structure_features"]["geometry_type"] == "footprint"
    assert b["near_structure_features"]["geometry_type"] == "footprint"
    assert a["near_structure_features"]["veg_density_0_5"] != b["near_structure_features"]["veg_density_0_5"]


def test_low_quality_anchor_geometry_resolution_is_explicitly_cautious(monkeypatch, tmp_path):
    context = _ctx(
        51.0,
        52.0,
        45.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 46.0},
            "ring_5_30_ft": {"vegetation_density": 43.0},
            "ring_30_100_ft": {"vegetation_density": 38.0},
        },
    )
    context.property_level_context.update(
        {
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "structure_match_status": "none",
            "ring_generation_mode": "point_annulus_fallback",
            "geometry_basis": "point",
            "geometry_source": "raw_geocode_point",
            "final_structure_geometry_source": "raw_geocode_point",
            "structure_geometry_confidence": 0.2,
            "property_anchor_source": "approximate_geocode",
            "property_anchor_quality": "low",
            "property_anchor_quality_score": 0.34,
            "anchor_quality_score": 0.34,
            "parcel_id": None,
            "parcel_lookup_method": None,
            "parcel_lookup_distance_m": None,
        }
    )
    _setup(monkeypatch, tmp_path, context)

    assessed = _run(_payload("Low Quality Anchor Ln", {"roof_type": "unknown"}))
    resolution = assessed["geometry_resolution"]

    assert resolution["anchor_source"] == "approximate_geocode"
    assert resolution["anchor_quality_score"] == pytest.approx(0.34, abs=1e-6)
    assert resolution["parcel_match_status"] == "not_found"
    assert resolution["footprint_match_status"] == "none"
    assert resolution["naip_structure_feature_status"] in {"missing", "present_but_not_consumed"}
    assert any("point-based annulus fallback" in str(row).lower() for row in (resolution.get("geometry_limitations") or []))
    assert resolution["ring_generation_mode"] == "point_annulus_fallback"
    assert assessed["ring_generation_mode"] == "point_annulus_fallback"
    assert assessed["final_structure_geometry_source"] == "raw_geocode_point"
    assert assessed["footprint_resolution"]["fallback_used"] is True
    assert assessed["footprint_resolution"]["selected_source"] is None
    assert assessed["assessment_specificity_tier"] in {"address_level", "regional_estimate", "insufficient_data"}
    trust_summary = ((assessed.get("homeowner_summary") or {}).get("trust_summary") or {})
    assert trust_summary.get("geometry_specificity_limited") is True
    condensed = trust_summary.get("geometry_resolution_summary") or {}
    assert condensed.get("ring_generation_mode") == "point_annulus_fallback"


def test_near_structure_features_fallback_marks_lower_confidence_when_imagery_unavailable(monkeypatch, tmp_path):
    context = _ctx(
        52.0,
        56.0,
        49.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 61.0, "coverage_pct": 58.0},
            "ring_5_30_ft": {"vegetation_density": 54.0, "coverage_pct": 51.0},
        },
    )
    context.property_level_context.update(
        {
            "footprint_used": False,
            "ring_generation_mode": "point_annulus_fallback",
            "naip_feature_source": None,
            "near_structure_vegetation_0_5_pct": 61.0,
            "near_structure_vegetation_5_30_pct": 54.0,
            "canopy_adjacency_proxy_pct": 57.0,
        }
    )
    _setup(monkeypatch, tmp_path, context)

    assessed = _run(_payload("Fallback Imagery Missing Ln", {"roof_type": "class a"}))
    near = assessed.get("near_structure_features") or {}

    assert near.get("source") == "fallback_layers"
    assert near.get("imagery_available") is False
    assert near.get("confidence_flag") == "low"
    assert near.get("precision_flag") in {"fallback_point_proxy", "footprint_relative"}


def test_parcel_only_geometry_resolution_surfaces_explicit_status(monkeypatch, tmp_path):
    context = _ctx(
        50.0,
        52.0,
        44.0,
        ring_metrics={},
    )
    context.property_level_context.update(
        {
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "structure_match_status": "none",
            "ring_generation_mode": "point_annulus_fallback",
            "geometry_basis": "parcel",
            "geometry_source": "parcel_geometry_inferred_home_location",
            "parcel_id": "parcel-only-42",
            "parcel_lookup_method": "contains_point",
            "parcel_source_name": "county_parcels",
            "property_anchor_source": "authoritative_address_point",
            "property_anchor_quality_score": 0.88,
            "anchor_quality_score": 0.88,
        }
    )
    _setup(monkeypatch, tmp_path, context)

    assessed = _run(_payload("Parcel Only Resolution Ln", {"roof_type": "class a"}))
    resolution = assessed["geometry_resolution"]
    assert resolution["anchor_source"] == "authoritative_address_point"
    assert resolution["parcel_match_status"] == "matched"
    assert resolution["footprint_match_status"] == "none"
    assert resolution["ring_generation_mode"] == "point_annulus_fallback"
    assert any("building footprint was not matched" in str(row).lower() for row in (resolution.get("geometry_limitations") or []))


def test_geometry_resolution_flags_missing_naip_structure_artifact(monkeypatch, tmp_path):
    context = _ctx(
        57.0,
        59.0,
        48.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 43.0},
            "ring_5_30_ft": {"vegetation_density": 48.0},
            "ring_30_100_ft": {"vegetation_density": 55.0},
        },
    )
    context.property_level_context.update(
        {
            "footprint_used": True,
            "footprint_status": "used",
            "structure_match_status": "matched",
            "ring_generation_mode": "footprint_aware_rings",
            "footprint_source_name": "overture_buildings",
            "property_anchor_source": "authoritative_address_point",
            "property_anchor_quality_score": 0.92,
            "anchor_quality_score": 0.92,
            "near_structure_vegetation_0_5_pct": None,
            "near_structure_vegetation_5_30_pct": None,
            "canopy_adjacency_proxy_pct": None,
            "vegetation_continuity_proxy_pct": None,
            "nearest_high_fuel_patch_distance_ft": None,
            "naip_feature_source": None,
        }
    )
    _setup(monkeypatch, tmp_path, context)

    assessed = _run(_payload("Missing NAIP Geometry Artifact Ln", {"roof_type": "class a"}))
    resolution = assessed["geometry_resolution"]
    assert resolution["footprint_match_status"] == "matched"
    assert resolution["naip_structure_feature_status"] in {"missing", "present_but_not_consumed"}
    assert any(
        "naip-derived near-structure vegetation features were unavailable" in str(row).lower()
        for row in (resolution.get("geometry_limitations") or [])
    )


def test_property_findings_from_ring_metrics_surface_in_assessment(monkeypatch, tmp_path):
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 82.0},
        "ring_5_30_ft": {"vegetation_density": 74.0},
        "ring_30_100_ft": {"vegetation_density": 68.0},
    }
    _setup(monkeypatch, tmp_path, _ctx(env=58.0, wildland=62.0, historic=50.0, ring_metrics=ring_metrics))

    assessed = _run(
        _payload(
            "310 Ring Insight Way",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 18,
            },
        )
    )

    assert len(assessed["property_findings"]) >= 1
    assert any("within 5 feet" in f.lower() for f in assessed["property_findings"])
    assert any("30 feet" in f.lower() for f in assessed["property_findings"])
    assert any("defensible space" in f.lower() for f in assessed["property_findings"])
    assert any(
        d == "dense vegetation close to the home"
        or "0-5 ft zone" in d.lower()
        or "within 5 ft" in d.lower()
        for d in assessed["top_risk_drivers"]
    )
    assert assessed["property_level_context"]["footprint_used"] is True
    assert assessed["property_level_context"]["fallback_mode"] == "footprint"
    assert isinstance(assessed["property_level_context"]["ring_metrics"], dict)


def test_property_findings_fallback_empty_when_ring_data_missing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=45.0, wildland=45.0, historic=40.0, ring_metrics={}))

    assessed = _run(_payload("311 No Ring Data Ln", {"defensible_space_ft": 20}))
    assert "property_findings" in assessed
    assert assessed["property_findings"] == []
    assert assessed["property_level_context"]["footprint_used"] is False
    assert assessed["property_level_context"]["fallback_mode"] == "point_based"
    assert assessed["property_level_context"]["ring_metrics"] in (None, {})


def test_defensible_space_analysis_present_with_footprint_ring_metrics(monkeypatch, tmp_path):
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 78.0, "coverage_pct": 82.0, "fuel_presence_proxy": 71.0},
        "ring_5_30_ft": {"vegetation_density": 69.0, "coverage_pct": 74.0, "fuel_presence_proxy": 66.0},
        "ring_30_100_ft": {"vegetation_density": 61.0, "coverage_pct": 63.0, "fuel_presence_proxy": 58.0},
    }
    _setup(monkeypatch, tmp_path, _ctx(env=58.0, wildland=62.0, historic=50.0, ring_metrics=ring_metrics))

    assessed = _run(
        _payload(
            "312 Defensible Analysis Ave",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 16},
        )
    )
    analysis = assessed["defensible_space_analysis"]
    assert analysis["basis_geometry_type"] == "building_footprint"
    assert analysis["data_quality"]["analysis_status"] in {"complete", "partial"}
    assert analysis["zones"]["zone_0_5_ft"]["vegetation_density"] == pytest.approx(78.0)
    assert len(assessed["top_near_structure_risk_drivers"]) >= 1
    assert isinstance(assessed["prioritized_vegetation_actions"], list)
    assert any(action.get("target_zone") in {"0-5 ft", "5-30 ft", "30-100 ft"} for action in assessed["prioritized_vegetation_actions"])


def test_point_proxy_ring_analysis_runs_without_footprint_and_reduces_confidence(monkeypatch, tmp_path):
    proxy_rings = {
        "zone_0_5_ft": {"vegetation_density": 72.0, "coverage_pct": 76.0, "fuel_presence_proxy": 68.0},
        "zone_5_30_ft": {"vegetation_density": 66.0, "coverage_pct": 69.0, "fuel_presence_proxy": 62.0},
        "zone_30_100_ft": {"vegetation_density": 60.0, "coverage_pct": 61.0, "fuel_presence_proxy": 57.0},
    }
    proxy_ctx = _ctx(env=56.0, wildland=60.0, historic=47.0, ring_metrics=proxy_rings)
    proxy_ctx.property_level_context.update(
        {
            "footprint_used": False,
            "footprint_found": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": proxy_rings,
        }
    )
    _setup(monkeypatch, tmp_path, proxy_ctx)
    proxy_assessed = _run(_payload("313 Point Proxy Ln", {"roof_type": "class a", "vent_type": "ember-resistant"}))

    observed_ctx = _ctx(env=56.0, wildland=60.0, historic=47.0, ring_metrics=proxy_rings)
    observed_ctx.property_level_context.update({"footprint_used": True, "footprint_status": "used", "fallback_mode": "footprint"})
    _setup(monkeypatch, tmp_path, observed_ctx)
    observed_assessed = _run(_payload("313 Point Proxy Ln", {"roof_type": "class a", "vent_type": "ember-resistant"}))

    proxy_analysis = proxy_assessed["defensible_space_analysis"]
    assert proxy_analysis["basis_geometry_type"] == "point_proxy"
    assert proxy_analysis["basis_quality"] in {"derived_proxy", "unavailable"}
    assert any("approximated" in s.lower() for s in proxy_assessed["defensible_space_limitations_summary"])
    assert proxy_assessed["confidence_score"] < observed_assessed["confidence_score"]


def test_partial_or_missing_zone_metrics_are_non_blocking_and_reported(monkeypatch, tmp_path):
    partial_rings = {
        "zone_0_5_ft": {"vegetation_density": 64.0, "coverage_pct": 68.0, "fuel_presence_proxy": 60.0},
        "zone_5_30_ft": {"vegetation_density": None, "coverage_pct": None, "fuel_presence_proxy": None},
        "zone_30_100_ft": {"vegetation_density": None, "coverage_pct": None, "fuel_presence_proxy": None},
    }
    partial_ctx = _ctx(env=50.0, wildland=55.0, historic=45.0, ring_metrics=partial_rings)
    partial_ctx.property_level_context.update({"footprint_used": False, "fallback_mode": "point_based", "ring_metrics": partial_rings})
    _setup(monkeypatch, tmp_path, partial_ctx)
    assessed = _run(_payload("314 Partial Ring Dr", {"roof_type": "composite"}))
    assert assessed["wildfire_risk_score_available"] is True
    analysis = assessed["defensible_space_analysis"]
    assert analysis["data_quality"]["analysis_status"] == "partial"
    assert analysis["data_quality"]["unavailable_zone_count"] >= 1

    missing_ctx = _ctx(env=49.0, wildland=53.0, historic=44.0, ring_metrics={})
    _setup(monkeypatch, tmp_path, missing_ctx)
    missing = _run(_payload("315 Missing Veg Dr", {"roof_type": "composite"}))
    assert missing["wildfire_risk_score_available"] is True
    assert missing["defensible_space_analysis"]["data_quality"]["analysis_status"] == "unavailable"
    assert missing["prioritized_vegetation_actions"] == []


def test_wildfire_data_builds_point_proxy_ring_metrics_when_footprint_unavailable(monkeypatch):
    client = WildfireDataClient()

    class _NoFootprint:
        found = False
        footprint = None
        source = "fixture"
        confidence = 0.0
        assumptions = ["No nearby building footprint found for this location."]

    monkeypatch.setattr(client.footprints, "get_building_footprint", lambda _lat, _lon: _NoFootprint())
    proxy = {
        "zone_0_5_ft": {"vegetation_density": 70.0, "coverage_pct": 74.0, "fuel_presence_proxy": 65.0},
        "zone_5_30_ft": {"vegetation_density": 62.0, "coverage_pct": 66.0, "fuel_presence_proxy": 58.0},
        "zone_30_100_ft": {"vegetation_density": 55.0, "coverage_pct": 60.0, "fuel_presence_proxy": 50.0},
        "ring_0_5_ft": {"vegetation_density": 70.0, "coverage_pct": 74.0, "fuel_presence_proxy": 65.0},
        "ring_5_30_ft": {"vegetation_density": 62.0, "coverage_pct": 66.0, "fuel_presence_proxy": 58.0},
        "ring_30_100_ft": {"vegetation_density": 55.0, "coverage_pct": 60.0, "fuel_presence_proxy": 50.0},
    }
    monkeypatch.setattr(client, "_build_point_proxy_ring_metrics", lambda **_kwargs: proxy)

    ring_context, assumptions, sources = client._compute_structure_ring_metrics(
        lat=46.87,
        lon=-113.99,
        canopy_path="",
        fuel_path="",
    )
    assert ring_context["footprint_used"] is False
    assert ring_context["fallback_mode"] == "point_based"
    assert ring_context["ring_generation_mode"] == "point_annulus_fallback"
    assert ring_context["footprint_resolution"]["selected_source"] is None
    assert ring_context["footprint_resolution"]["fallback_used"] is True
    assert ring_context["footprint_resolution"]["match_status"] in {"none", "ambiguous"}
    assert ring_context["footprint_resolution"]["candidates_considered"] >= 0
    assert ring_context["geometry_source"] in {
        "raw_geocode_point",
        "parcel_geometry_inferred_home_location",
        "user_selected_map_point_unsnapped",
    }
    assert float(ring_context.get("geometry_confidence") or 0.0) < 0.7
    assert isinstance(ring_context["ring_metrics"], dict)
    assert ring_context["ring_metrics"]["zone_0_5_ft"]["vegetation_density"] == pytest.approx(70.0)
    assert any("point-based annulus" in note.lower() for note in assumptions)
    assert any("point-proxy ring vegetation summaries" in s.lower() for s in sources)


def test_wildfire_data_point_selection_snaps_to_nearby_footprint(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    footprint = Polygon(
        [
            (-105.00010, 40.00010),
            (-104.99992, 40.00010),
            (-104.99992, 39.99992),
            (-105.00010, 39.99992),
            (-105.00010, 40.00010),
        ]
    )

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=True,
            footprint=footprint,
            centroid=(40.0, -105.0),
            source="fixture",
            confidence=0.72,
            match_status="matched",
            match_method="nearest_building_fallback",
            matched_structure_id="snap-home",
            match_distance_m=4.2,
            candidate_count=3,
            candidate_summaries=[],
            assumptions=[],
        ),
    )
    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda _geom, canopy_path: {
            "canopy_mean": 52.0,
            "canopy_max": 71.0,
            "coverage_pct": 46.0,
            "vegetation_density": 55.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: 42.0)

    context_blob, assumptions, _sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        selection_mode="point",
        user_selected_point={"latitude": 40.0, "longitude": -105.0},
    )

    assert context_blob["footprint_used"] is True
    assert context_blob["selection_mode"] == "point"
    assert context_blob["final_structure_geometry_source"] == "user_selected_point_snapped"
    assert context_blob["structure_selection_method"] == "point_nearest_footprint_snap"
    assert context_blob["geometry_basis"] == "footprint"
    assert context_blob["structure_geometry_confidence"] >= 0.5
    assert context_blob["snapped_structure_distance_m"] == pytest.approx(4.2)
    assert context_blob["user_selected_point_in_footprint"] is False
    assert context_blob["matched_structure_id"] == "snap-home"
    # High-confidence snapped geometry should drive the display point.
    assert context_blob["display_point_source"] == "matched_structure_centroid"
    assert any("user-selected map point" in note.lower() for note in assumptions)


def test_wildfire_data_point_selection_uses_parcel_intersection_snap_method(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    footprint = Polygon(
        [
            (-105.00010, 40.00010),
            (-104.99992, 40.00010),
            (-104.99992, 39.99992),
            (-105.00010, 39.99992),
            (-105.00010, 40.00010),
        ]
    )

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=True,
            footprint=footprint,
            centroid=(40.0, -105.0),
            source="fixture",
            confidence=0.79,
            match_status="matched",
            match_method="parcel_intersection",
            matched_structure_id="parcel-home",
            match_distance_m=6.2,
            candidate_count=2,
            candidate_summaries=[],
            assumptions=[],
        ),
    )
    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda _geom, canopy_path: {
            "canopy_mean": 49.0,
            "canopy_max": 68.0,
            "coverage_pct": 44.0,
            "vegetation_density": 51.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: 38.0)

    context_blob, _assumptions, _sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        selection_mode="point",
        user_selected_point={"latitude": 40.0, "longitude": -105.0},
        parcel_polygon=footprint,
        use_parcel_association_for_point_mode=True,
    )

    assert context_blob["footprint_used"] is True
    assert context_blob["final_structure_geometry_source"] == "user_selected_point_snapped"
    assert context_blob["structure_selection_method"] == "point_parcel_intersection_snap"
    assert context_blob["geometry_basis"] == "footprint"


def test_wildfire_data_point_selection_missing_footprint_prefers_parcel_inferred_anchor(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    parcel = Polygon(
        [
            (-105.00080, 40.00080),
            (-105.00040, 40.00080),
            (-105.00040, 40.00040),
            (-105.00080, 40.00040),
            (-105.00080, 40.00080),
        ]
    )

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=False,
            footprint=None,
            centroid=None,
            source="fixture_missing",
            confidence=0.0,
            match_status="none",
            match_method="nearest_building_fallback",
            matched_structure_id=None,
            match_distance_m=41.0,
            candidate_count=1,
            candidate_summaries=[],
            assumptions=["No nearby building footprint found for this location."],
        ),
    )

    sampled: dict[str, float] = {}

    def _proxy(**kwargs):
        sampled["lat"] = float(kwargs["lat"])
        sampled["lon"] = float(kwargs["lon"])
        return {
            "zone_0_5_ft": {"vegetation_density": 41.0},
            "ring_0_5_ft": {"vegetation_density": 41.0},
            "zone_5_30_ft": {"vegetation_density": 37.0},
            "ring_5_30_ft": {"vegetation_density": 37.0},
            "zone_30_100_ft": {"vegetation_density": 33.0},
            "ring_30_100_ft": {"vegetation_density": 33.0},
        }

    monkeypatch.setattr(client, "_build_point_proxy_ring_metrics", _proxy)

    context_blob, assumptions, _sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="",
        fuel_path="",
        selection_mode="point",
        user_selected_point={"latitude": 40.00005, "longitude": -105.00005},
        parcel_polygon=parcel,
        use_parcel_association_for_point_mode=True,
    )

    inferred_point = parcel.representative_point()
    assert sampled["lat"] == pytest.approx(float(inferred_point.y), abs=1e-6)
    assert sampled["lon"] == pytest.approx(float(inferred_point.x), abs=1e-6)
    assert context_blob["geometry_source"] == "parcel_geometry_inferred_home_location"
    assert context_blob["geometry_basis"] == "parcel"
    assert context_blob["ring_generation_mode"] == "point_annulus_fallback"
    assert context_blob["final_structure_geometry_source"] == "parcel_inferred_home_location"
    assert context_blob["structure_selection_method"] == "parcel_inferred_home_location"
    assert any("parcel geometry to infer a home location" in note.lower() for note in assumptions)


def test_wildfire_data_point_selection_detects_point_inside_footprint(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    footprint = Polygon(
        [
            (-105.00008, 40.00008),
            (-104.99992, 40.00008),
            (-104.99992, 39.99992),
            (-105.00008, 39.99992),
            (-105.00008, 40.00008),
        ]
    )
    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=True,
            footprint=footprint,
            centroid=(40.0, -105.0),
            source="fixture",
            confidence=0.86,
            match_status="matched",
            match_method="nearest_building_fallback",
            matched_structure_id="inside-home",
            match_distance_m=0.0,
            candidate_count=1,
            candidate_summaries=[],
            assumptions=[],
        ),
    )
    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda _geom, canopy_path: {
            "canopy_mean": 45.0,
            "canopy_max": 63.0,
            "coverage_pct": 40.0,
            "vegetation_density": 48.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: 36.0)

    context_blob, _assumptions, _sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        selection_mode="point",
        user_selected_point={"latitude": 40.0, "longitude": -105.0},
    )
    assert context_blob["final_structure_geometry_source"] == "user_selected_point_snapped"
    assert context_blob["structure_selection_method"] == "point_inside_footprint_snap"
    assert context_blob["user_selected_point_in_footprint"] is True
    assert context_blob["snapped_structure_distance_m"] == pytest.approx(0.0)
    assert context_blob["display_point_source"] == "matched_structure_centroid"


def test_wildfire_data_point_selection_weak_match_remains_unsnapped(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    monkeypatch.setenv("WF_POINT_SELECTION_MIN_SNAP_CONFIDENCE", "0.70")
    weak_footprint = Polygon(
        [
            (-105.00010, 40.00010),
            (-104.99992, 40.00010),
            (-104.99992, 39.99992),
            (-105.00010, 39.99992),
            (-105.00010, 40.00010),
        ]
    )
    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=True,
            footprint=weak_footprint,
            centroid=(40.0, -105.0),
            source="fixture",
            confidence=0.58,
            match_status="matched",
            match_method="nearest_building_fallback",
            matched_structure_id="weak-home",
            match_distance_m=6.0,
            candidate_count=2,
            candidate_summaries=[],
            assumptions=[],
        ),
    )
    proxy = {
        "zone_0_5_ft": {"vegetation_density": 52.0, "coverage_pct": 55.0, "fuel_presence_proxy": 49.0},
        "zone_5_30_ft": {"vegetation_density": 48.0, "coverage_pct": 52.0, "fuel_presence_proxy": 44.0},
        "zone_30_100_ft": {"vegetation_density": 43.0, "coverage_pct": 47.0, "fuel_presence_proxy": 40.0},
        "ring_0_5_ft": {"vegetation_density": 52.0, "coverage_pct": 55.0, "fuel_presence_proxy": 49.0},
        "ring_5_30_ft": {"vegetation_density": 48.0, "coverage_pct": 52.0, "fuel_presence_proxy": 44.0},
        "ring_30_100_ft": {"vegetation_density": 43.0, "coverage_pct": 47.0, "fuel_presence_proxy": 40.0},
    }
    monkeypatch.setattr(client, "_build_point_proxy_ring_metrics", lambda **_kwargs: proxy)

    context_blob, assumptions, _sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="",
        fuel_path="",
        selection_mode="point",
        user_selected_point={"latitude": 40.0, "longitude": -105.0},
    )
    assert context_blob["footprint_used"] is False
    assert context_blob["final_structure_geometry_source"] == "user_selected_point_unsnapped"
    assert context_blob["structure_selection_method"] == "point_unsnapped_low_confidence_or_distance"
    assert context_blob["geometry_basis"] == "point"
    assert context_blob["display_point_source"] == "property_anchor_point"
    assert context_blob["snapped_structure_distance_m"] is None
    assert any("low confidence" in note.lower() for note in assumptions)


def test_wildfire_data_point_selection_unsnapped_falls_back_to_selected_anchor(monkeypatch):
    client = WildfireDataClient()

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=False,
            footprint=None,
            centroid=None,
            source="fixture_missing",
            confidence=0.0,
            match_status="none",
            match_method="nearest_building_fallback",
            matched_structure_id=None,
            match_distance_m=28.0,
            candidate_count=2,
            candidate_summaries=[],
            assumptions=["Nearest structure footprint is too far from the geocoded point; using geocoded point fallback."],
        ),
    )
    proxy = {
        "zone_0_5_ft": {"vegetation_density": 58.0, "coverage_pct": 62.0, "fuel_presence_proxy": 54.0},
        "zone_5_30_ft": {"vegetation_density": 51.0, "coverage_pct": 55.0, "fuel_presence_proxy": 47.0},
        "zone_30_100_ft": {"vegetation_density": 45.0, "coverage_pct": 49.0, "fuel_presence_proxy": 42.0},
        "ring_0_5_ft": {"vegetation_density": 58.0, "coverage_pct": 62.0, "fuel_presence_proxy": 54.0},
        "ring_5_30_ft": {"vegetation_density": 51.0, "coverage_pct": 55.0, "fuel_presence_proxy": 47.0},
        "ring_30_100_ft": {"vegetation_density": 45.0, "coverage_pct": 49.0, "fuel_presence_proxy": 42.0},
    }
    monkeypatch.setattr(client, "_build_point_proxy_ring_metrics", lambda **_kwargs: proxy)

    context_blob, assumptions, _sources = client._compute_structure_ring_metrics(
        46.87,
        -113.99,
        canopy_path="",
        fuel_path="",
        selection_mode="point",
        user_selected_point={"latitude": 46.8702, "longitude": -113.9898},
    )
    assert context_blob["footprint_used"] is False
    assert context_blob["selection_mode"] == "point"
    assert context_blob["final_structure_geometry_source"] == "user_selected_point_unsnapped"
    assert context_blob["structure_selection_method"] == "point_unsnapped_no_match"
    assert context_blob["geometry_source"] == "user_selected_map_point_unsnapped"
    assert context_blob["ring_generation_mode"] == "point_annulus_fallback"
    assert context_blob["geometry_basis"] == "point"
    assert context_blob["structure_geometry_confidence"] == pytest.approx(0.42)
    assert context_blob["snapped_structure_distance_m"] is None
    assert context_blob["user_selected_point"] == {"latitude": 46.8702, "longitude": -113.9898}
    assert context_blob["display_point_source"] == "property_anchor_point"
    assert any("could not be snapped" in note.lower() for note in assumptions)


def test_score_decomposition_and_blended_wildfire_score(monkeypatch, tmp_path):
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 55.0},
        "ring_5_30_ft": {"vegetation_density": 60.0},
        "ring_30_100_ft": {"vegetation_density": 62.0},
    }
    _setup(monkeypatch, tmp_path, _ctx(env=62.0, wildland=67.0, historic=58.0, ring_metrics=ring_metrics))

    assessed = _run(
        _payload(
            "322 Score Split Ave",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 22,
                "construction_year": 2015,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        )
    )

    expected_blended = app_main.risk_engine.compute_blended_wildfire_score(
        assessed["site_hazard_score"],
        assessed["home_ignition_vulnerability_score"],
        assessed["insurance_readiness_score"],
    )
    env_names = set(app_main.ENVIRONMENTAL_SUBMODELS)
    struct_names = set(app_main.STRUCTURAL_SUBMODELS)
    weighted = assessed["weighted_contributions"]
    env_weight = sum(float(weighted[name]["weight"]) for name in env_names)
    struct_weight = sum(float(weighted[name]["weight"]) for name in struct_names)
    expected_site = round(
        sum(weighted[name]["contribution"] for name in env_names) / env_weight,
        1,
    )
    expected_home_structural_base = round(
        sum(weighted[name]["contribution"] for name in struct_names) / struct_weight,
        1,
    )
    assert assessed["site_hazard_score"] == expected_site
    assert assessed["home_ignition_vulnerability_score"] >= expected_home_structural_base
    assert abs(float(assessed["wildfire_risk_score"]) - float(expected_blended)) <= 2.0
    assert assessed["legacy_weighted_wildfire_risk_score"] >= 0
    assert assessed["insurance_readiness_score"] != round(100.0 - assessed["wildfire_risk_score"], 1)


def test_score_evidence_ledger_includes_contribution_evidence_and_blended_composition(monkeypatch, tmp_path):
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 68.0},
        "ring_5_30_ft": {"vegetation_density": 64.0},
        "ring_30_100_ft": {"vegetation_density": 60.0},
    }
    _setup(monkeypatch, tmp_path, _ctx(env=58.0, wildland=61.0, historic=52.0, ring_metrics=ring_metrics))
    assessed = _run(
        _payload(
            "Evidence Ledger Way",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 20,
                "construction_year": 2012,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )

    ledger = assessed["score_evidence_ledger"]
    assert any(row["factor_key"] == "vegetation_intensity_risk" for row in ledger["site_hazard_score"])
    assert any(row["factor_key"] == "structure_vulnerability_risk" for row in ledger["home_ignition_vulnerability_score"])
    assert any(row["factor_key"] == "site_hazard_component" for row in ledger["wildfire_risk_score"])
    assert any(row["factor_key"] == "home_ignition_component" for row in ledger["wildfire_risk_score"])
    for family in ledger.values():
        for row in family:
            assert "contribution" in row
            assert "evidence_status" in row
            assert row["evidence_status"] in {"observed", "inferred", "missing", "fallback"}


def test_confidence_tier_high_and_shareable_when_inputs_are_strong(monkeypatch, tmp_path):
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 20.0},
        "ring_5_30_ft": {"vegetation_density": 25.0},
        "ring_30_100_ft": {"vegetation_density": 30.0},
    }
    strong_ctx = _ctx(env=35.0, wildland=35.0, historic=20.0, ring_metrics=ring_metrics)
    strong_ctx.access_exposure_index = 22.0
    strong_ctx.access_context = {
        "status": "ok",
        "source": "OpenStreetMap road network",
        "distance_to_nearest_road_m": 12.0,
        "road_segments_within_300m": 7,
        "intersections_within_300m": 3,
        "dead_end_indicator": False,
    }
    _setup(monkeypatch, tmp_path, strong_ctx)

    assessed = _run(
        _payload(
            "333 High Confidence Rd",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 35,
                "construction_year": 2018,
                "siding_type": "fiber cement",
                "window_type": "dual pane tempered",
                "vegetation_condition": "maintained",
            },
            confirmed=[
                "roof_type",
                "vent_type",
                "defensible_space_ft",
                "construction_year",
            ],
        )
    )

    assert assessed["confidence_tier"] == "high"
    assert assessed["use_restriction"] == "shareable"


def test_low_confidence_restriction_when_key_layers_missing(monkeypatch, tmp_path):
    degraded = WildfireContext(
        environmental_index=58.0,
        slope_index=58.0,
        aspect_index=50.0,
        fuel_index=58.0,
        moisture_index=58.0,
        canopy_index=58.0,
        wildland_distance_index=60.0,
        historic_fire_index=55.0,
        burn_probability_index=58.0,
        hazard_severity_index=58.0,
        data_sources=["test-geocoder"],
        assumptions=[
            "Burn probability layer unavailable at property location.",
            "Fuel model unavailable within 100m neighborhood.",
            "Historical perimeter layer missing; recurrence defaulted.",
        ],
        structure_ring_metrics={},
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "missing",
            "canopy": "ok",
            "fire_history": "missing",
        },
        property_level_context={
            "footprint_used": False,
            "footprint_status": "provider_unavailable",
            "fallback_mode": "point_based",
            "ring_metrics": None,
        },
    )
    _setup(monkeypatch, tmp_path, degraded)

    assessed = _run(_payload("344 Low Confidence Ct", {"defensible_space_ft": 10}))
    assert assessed["confidence_tier"] in {"low", "preliminary"}
    assert assessed["use_restriction"] == "not_for_underwriting_or_binding"
    assert assessed["evidence_quality_summary"]["use_restriction"] in {"review_required", "screening_only"}
    assert assessed["evidence_quality_summary"]["confidence_penalties"]


@pytest.mark.parametrize(
    ("assumptions", "missing_inputs", "inferred_inputs", "expected_tier", "expected_restriction"),
    [
        ([], [], {}, "high", "shareable"),
        (["one fallback assumption"], ["fuel_model_layer"], {}, "moderate", "homeowner_review_recommended"),
        (
            [],
            ["roof_type", "vent_type", "defensible_space_ft"],
            {"construction_year": "pre_2008_proxy", "roof_type": "composition", "vent_type": "standard", "defensible_space_ft": 15},
            "low",
            "agent_or_inspector_review_recommended",
        ),
        ([], ["roof_type", "vent_type", "defensible_space_ft", "construction_year"], {}, "preliminary", "not_for_underwriting_or_binding"),
    ],
)
def test_confidence_tier_and_use_restriction_mapping(
    assumptions,
    missing_inputs,
    inferred_inputs,
    expected_tier,
    expected_restriction,
):
    block = AssumptionsBlock(
        confirmed_inputs={"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 35},
        observed_inputs={},
        inferred_inputs=inferred_inputs,
        missing_inputs=missing_inputs,
        assumptions_used=["Access exposure remains provisional."] + assumptions,
    )
    layer_status = {
        "burn_probability": "ok",
        "hazard": "ok",
        "slope": "ok",
        "fuel": "missing" if "fuel_model_layer" in missing_inputs else "ok",
        "canopy": "ok",
        "fire_history": "ok",
    }
    env_completeness = compute_environmental_data_completeness(
        _ctx(env=50.0, wildland=50.0, historic=50.0, environmental_layer_status=layer_status)
    )
    conf = app_main._build_confidence(
        block,
        environmental_data_completeness=env_completeness,
        geocode_verified=True,
        property_level_context={"footprint_used": True},
        environmental_layer_status=layer_status,
    )
    # Guardrails: explicit expected buckets driven by deterministic confidence inputs.
    if expected_tier == "high":
        assert conf.confidence_tier in {"high", "moderate"}
    else:
        assert conf.confidence_tier == expected_tier
    if conf.confidence_tier == "high":
        assert conf.use_restriction == "shareable"
    else:
        assert conf.use_restriction == expected_restriction


def test_ring_metrics_increase_home_ignition_vulnerability_score(monkeypatch, tmp_path):
    attrs = {
        "roof_type": "class a",
        "vent_type": "ember-resistant",
        "defensible_space_ft": 20,
        "construction_year": 2015,
    }

    low_ring = {
        "ring_0_5_ft": {"vegetation_density": 10.0},
        "ring_5_30_ft": {"vegetation_density": 20.0},
        "ring_30_100_ft": {"vegetation_density": 30.0},
    }
    high_ring = {
        "ring_0_5_ft": {"vegetation_density": 85.0},
        "ring_5_30_ft": {"vegetation_density": 80.0},
        "ring_30_100_ft": {"vegetation_density": 70.0},
    }

    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=55.0, historic=50.0, ring_metrics=low_ring))
    low = _run(_payload("355 Ring Low", attrs))

    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=55.0, historic=50.0, ring_metrics=high_ring))
    high = _run(_payload("356 Ring High", attrs))

    assert high["home_ignition_vulnerability_score"] > low["home_ignition_vulnerability_score"]


def test_worsening_defensible_space_does_not_improve_readiness(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=52.0, wildland=58.0, historic=47.0))
    safer = _run(
        _payload(
            "Defensible Space Better Ave",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 40,
                "construction_year": 2016,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    worse = _run(
        _payload(
            "Defensible Space Worse Ave",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 5,
                "construction_year": 2016,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    assert worse["insurance_readiness_score"] <= safer["insurance_readiness_score"]


def test_better_roof_and_vents_do_not_increase_home_vulnerability(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=62.0, historic=50.0))
    weaker = _run(
        _payload(
            "Weak Hardening Ln",
            {
                "roof_type": "wood",
                "vent_type": "standard",
                "defensible_space_ft": 20,
                "construction_year": 1995,
            },
        )
    )
    stronger = _run(
        _payload(
            "Strong Hardening Ln",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 20,
                "construction_year": 1995,
            },
            confirmed=["roof_type", "vent_type"],
        )
    )
    assert stronger["home_ignition_vulnerability_score"] <= weaker["home_ignition_vulnerability_score"]


def test_higher_near_structure_vegetation_does_not_reduce_site_hazard(monkeypatch, tmp_path):
    attrs = {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 20}
    low_ring = {
        "ring_0_5_ft": {"vegetation_density": 20.0},
        "ring_5_30_ft": {"vegetation_density": 25.0},
        "ring_30_100_ft": {"vegetation_density": 30.0},
    }
    high_ring = {
        "ring_0_5_ft": {"vegetation_density": 88.0},
        "ring_5_30_ft": {"vegetation_density": 82.0},
        "ring_30_100_ft": {"vegetation_density": 78.0},
    }
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=50.0, historic=45.0, ring_metrics=low_ring))
    low = _run(_payload("Low Near Veg", attrs))
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=50.0, historic=45.0, ring_metrics=high_ring))
    high = _run(_payload("High Near Veg", attrs))
    assert high["site_hazard_score"] >= low["site_hazard_score"]
    assert high["wildfire_risk_score"] >= low["wildfire_risk_score"] + 1.0


def test_0_5_ft_vegetation_mitigation_has_meaningful_delta():
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=20,
        construction_year=2014,
    )
    baseline_ring = {
        "ring_0_5_ft": {"vegetation_density": 78.0},
        "ring_5_30_ft": {"vegetation_density": 78.0},
        "ring_30_100_ft": {"vegetation_density": 74.0},
        "ring_100_300_ft": {"vegetation_density": 70.0},
    }
    mitigate_0_5_ring = {
        "ring_0_5_ft": {"vegetation_density": 22.0},
        "ring_5_30_ft": {"vegetation_density": 78.0},
        "ring_30_100_ft": {"vegetation_density": 74.0},
        "ring_100_300_ft": {"vegetation_density": 70.0},
    }
    mitigate_5_30_ring = {
        "ring_0_5_ft": {"vegetation_density": 78.0},
        "ring_5_30_ft": {"vegetation_density": 34.0},
        "ring_30_100_ft": {"vegetation_density": 74.0},
        "ring_100_300_ft": {"vegetation_density": 70.0},
    }

    def _overall(ctx: WildfireContext) -> tuple[float, float]:
        risk = app_main.risk_engine.score(attrs, 39.7392, -104.9903, ctx)
        site = app_main.risk_engine.compute_site_hazard_score(risk)
        home = app_main.risk_engine.compute_home_ignition_vulnerability_score(risk)
        readiness = app_main.risk_engine.compute_insurance_readiness(attrs, ctx, risk).insurance_readiness_score
        total = app_main.risk_engine.compute_blended_wildfire_score(site, home, readiness, risk)
        return total, readiness

    baseline_total, baseline_readiness = _overall(_ctx(env=60.0, wildland=60.0, historic=54.0, ring_metrics=baseline_ring))
    mitigated_0_5_total, mitigated_0_5_readiness = _overall(
        _ctx(env=60.0, wildland=60.0, historic=54.0, ring_metrics=mitigate_0_5_ring)
    )
    mitigated_5_30_total, mitigated_5_30_readiness = _overall(
        _ctx(env=60.0, wildland=60.0, historic=54.0, ring_metrics=mitigate_5_30_ring)
    )

    delta_0_5 = baseline_total - mitigated_0_5_total
    delta_5_30 = baseline_total - mitigated_5_30_total
    assert delta_0_5 >= 1.5
    assert delta_5_30 > 0.0
    assert delta_0_5 >= delta_5_30
    assert mitigated_0_5_readiness > baseline_readiness
    assert mitigated_5_30_readiness > baseline_readiness


def test_geocoding_failure_does_not_use_synthetic_coordinate_fallback(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (_ for _ in ()).throw(RuntimeError("nominatim unavailable")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=50.0, wildland=50.0, historic=50.0))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_fail.db")))

    res = client.post(
        "/risk/assess",
        json={
            "address": "404 Missing Geocode St",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert detail["error"] == "geocoding_failed"
    assert detail["geocode_status"] == "provider_error"
    assert "temporarily unavailable" in detail["message"].lower()
    assert detail["normalized_address"]


def test_malformed_address_returns_structured_geocode_parser_error(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="parser_error",
                message="Address input is too short for geocoding.",
                submitted_address="??",
                normalized_address="??",
                rejection_reason="input_too_short",
            )
        ),
    )
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_parser.db")))

    res = client.post(
        "/risk/assess",
        json={
            "address": "?? ???",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["error"] == "geocoding_failed"
    assert detail["geocode_status"] == "parser_error"
    assert "could not be parsed" in detail["message"].lower()


def test_ambiguous_address_returns_structured_geocode_ambiguity_error(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="ambiguous_match",
                message="Geocoding returned multiple similarly ranked matches.",
                submitted_address="100 Main St",
                normalized_address="100 Main St",
                rejection_reason="top and second candidates within ambiguity threshold",
            )
        ),
    )
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_ambiguous.db")))

    res = client.post(
        "/risk/assess",
        json={
            "address": "100 Main St",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["error"] == "geocoding_failed"
    assert detail["geocode_status"] == "ambiguous_match"
    assert "multiple possible locations" in detail["message"].lower()


def test_low_confidence_address_returns_structured_geocode_low_confidence_error(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="low_confidence",
                message="Best geocoding match was below the confidence threshold.",
                submitted_address="Rural Route 1 Box 2",
                normalized_address="Rural Route 1 Box 2",
                rejection_reason="importance=0.01 threshold=0.2",
            )
        ),
    )
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_low_conf.db")))

    res = client.post(
        "/risk/assess",
        json={
            "address": "Rural Route 1 Box 2",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["error"] == "geocoding_failed"
    assert detail["geocode_status"] == "low_confidence"
    assert detail["rejection_category"] == "trust_filter_rejected"
    assert "below policy threshold" in detail["message"].lower()


def test_low_confidence_with_candidate_fallback_scores_when_region_is_covered(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_low_conf_fallback.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=44.0, wildland=50.0, historic=35.0))
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="low_confidence",
                message="Best geocoding match was below the confidence threshold.",
                submitted_address="104 Riverside Ave, Winthrop, WA 98862",
                normalized_address="104 Riverside Ave, Winthrop, WA 98862",
                rejection_reason="importance=0.01 threshold=0.2",
                raw_response_preview={
                    "candidate_count": 2,
                    "trust_filter_rule": "min_importance_threshold",
                    "top_candidate": {
                        "display_name": "104 Riverside Ave, Winthrop, WA 98862, USA",
                        "lat": "48.4772",
                        "lon": "-120.1864",
                        "importance": 0.01,
                        "class": "highway",
                        "type": "residential",
                    },
                    "parsed_candidates": [
                        {
                            "display_name": "104 Riverside Ave, Winthrop, WA 98862, USA",
                            "lat": "48.4772",
                            "lon": "-120.1864",
                            "importance": 0.01,
                        }
                    ],
                },
            )
        ),
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
        },
    )

    res = client.post(
        "/risk/assess",
        json={
            "address": "104 Riverside Ave, Winthrop, WA 98862",
            "attributes": {"roof_type": "class a"},
            "confirmed_fields": ["roof_type"],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["geocoding"]["geocode_status"] == "accepted"
    assert body["geocoding"]["geocode_outcome"] == "geocode_succeeded_untrusted"
    assert body["geocoding"]["trusted_match_status"] == "untrusted_fallback"
    assert body["geocoding"]["fallback_eligibility"] is True
    assert body["resolved_region_id"] == "winthrop_pilot"
    assert body["coverage_available"] is True


def test_low_confidence_with_candidate_fallback_reaches_uncovered_response(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_low_conf_uncovered.db")))
    monkeypatch.setenv("WF_REQUIRE_PREPARED_REGION_COVERAGE", "true")
    monkeypatch.setenv("WF_AUTO_QUEUE_REGION_PREP_ON_MISS", "false")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="low_confidence",
                message="Best geocoding match was below the confidence threshold.",
                submitted_address="10 Unknown Rd, Winthrop, WA 98862",
                normalized_address="10 Unknown Rd, Winthrop, WA 98862",
                rejection_reason="importance=0.01 threshold=0.2",
                raw_response_preview={
                    "candidate_count": 1,
                    "trust_filter_rule": "min_importance_threshold",
                    "top_candidate": {
                        "display_name": "10 Unknown Rd, Winthrop, WA 98862, USA",
                        "lat": "48.4760",
                        "lon": "-120.1900",
                        "importance": 0.01,
                    },
                    "parsed_candidates": [
                        {
                            "display_name": "10 Unknown Rd, Winthrop, WA 98862, USA",
                            "lat": "48.4760",
                            "lon": "-120.1900",
                            "importance": 0.01,
                        }
                    ],
                },
            )
        ),
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": False,
            "diagnostics": ["No prepared region bounds contain point."],
            "nearest_region_id": "winthrop_pilot",
            "region_distance_to_boundary_m": 412.4,
        },
    )

    payload = {
        "address": "10 Unknown Rd, Winthrop, WA 98862",
        "attributes": {"roof_type": "class a"},
        "confirmed_fields": ["roof_type"],
        "audience": "homeowner",
    }
    coverage = client.post("/regions/coverage-check", json={"address": payload["address"]})
    debug = client.post("/risk/debug", json=payload)
    assess = client.post("/risk/assess", json=payload)

    assert coverage.status_code == 200
    assert debug.status_code == 200
    coverage_body = coverage.json()
    debug_body = debug.json()
    assert coverage_body["geocode_status"] == "accepted"
    assert coverage_body["geocode_outcome"] == "geocode_succeeded_untrusted"
    assert coverage_body["trusted_match_status"] == "untrusted_fallback"
    assert coverage_body["coverage_available"] is False
    assert coverage_body["reason"] == "no_prepared_region_for_location"
    assert debug_body["geocoding"]["geocode_status"] == "accepted"
    assert debug_body["geocoding"]["geocode_outcome"] == "geocode_succeeded_untrusted"
    assert "region_resolution" in debug_body

    assert assess.status_code == 409
    detail = assess.json()["detail"]
    assert detail["region_not_ready"] is True
    assert detail["geocode_status"] == "accepted"
    assert detail["geocode_outcome"] == "geocode_succeeded_untrusted"
    assert detail["trusted_match_status"] == "untrusted_fallback"
    assert detail["coverage_available"] is False
    assert detail["reason"] == "no_prepared_region_for_location"


def test_primary_no_match_can_resolve_via_secondary_geocoder(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "secondary_geocoder_fallback.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=41.0, wildland=50.0, historic=36.0))
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "true")
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_SEARCH_URL", "https://example.test/geocode")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )

    class _Secondary:
        provider_name = "Secondary Test Geocoder"
        last_result = {
            "geocode_status": "accepted",
            "provider": "Secondary Test Geocoder",
            "matched_address": "6 Pineview Rd, Winthrop, WA 98862, USA",
            "confidence_score": 0.22,
            "candidate_count": 1,
            "geocode_precision": "parcel_or_address_point",
            "raw_response_preview": {"candidate_count": 1},
        }

        def geocode(self, _address):
            return (48.4772, -120.1864, "Secondary Test Geocoder")

    monkeypatch.setattr(app_main, "secondary_geocoder", _Secondary())
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
        },
    )

    response = client.post(
        "/risk/assess",
        json={
            "address": "6 Pineview Rd, Winthrop, WA 98862",
            "attributes": {"roof_type": "class a"},
            "confirmed_fields": ["roof_type"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()
    geocode = body["geocoding"]
    assert geocode["geocode_status"] == "accepted"
    assert geocode["resolution_method"] == "secondary_geocoder"
    assert geocode["fallback_used"] is True
    assert body["resolved_region_id"] == "winthrop_pilot"


def test_no_match_on_providers_can_resolve_via_local_region_fallback(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "local_fallback_geocoder.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=43.0, wildland=48.0, historic=40.0))
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "true")
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_SEARCH_URL", "https://example.test/geocode")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )

    class _SecondaryNoMatch:
        provider_name = "Secondary Test Geocoder"
        last_result = {
            "geocode_status": "no_match",
            "provider": "Secondary Test Geocoder",
            "matched_address": None,
            "candidate_count": 0,
        }

        def geocode(self, _address):
            raise GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                provider="Secondary Test Geocoder",
                rejection_reason="provider returned no candidates",
            )

    monkeypatch.setattr(app_main, "secondary_geocoder", _SecondaryNoMatch())
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _addr: {
            "matched": False,
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": ["no authoritative local match"],
            "failure_reason": "no_local_authoritative_match",
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_fallback_coordinates",
        lambda _addr: {
            "matched": True,
            "confidence": "medium",
            "candidate_count": 1,
            "best_match": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "match_score": 0.84,
                "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
                "source": "address_points",
            },
            "top_candidates": [],
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "diagnostics": ["resolved via test local fallback"],
        },
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
        },
    )

    response = client.post(
        "/risk/assess",
        json={
            "address": "6 Pineview Rd, Winthrop, WA 98862",
            "attributes": {"roof_type": "class a"},
            "confirmed_fields": ["roof_type"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()
    geocode = body["geocoding"]
    assert geocode["geocode_status"] == "accepted"
    assert geocode["resolution_method"] == "explicit_fallback_record"
    assert geocode["fallback_used"] is True
    assert geocode["local_fallback_attempted"] is True
    assert geocode["local_fallback_result"]["matched"] is True


def test_no_match_can_still_assess_with_user_selected_point(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "user_selected_point_fallback.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=42.0, wildland=47.0, historic=39.0))
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setenv("WF_GEOCODE_ENABLE_PROVIDER_BACKOFF_QUERY", "false")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _addr: {
            "matched": False,
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": ["no authoritative local match"],
            "failure_reason": "no_local_authoritative_match",
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_fallback_coordinates",
        lambda _addr: {
            "matched": False,
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": ["no local match"],
        },
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
        },
    )

    response = client.post(
        "/risk/assess",
        json={
            "address": "6 Pineview Rd, Winthrop, WA 98862",
            "attributes": {"roof_type": "class a"},
            "confirmed_fields": ["roof_type"],
            "audience": "homeowner",
            "selection_mode": "point",
            "property_anchor_point": {"latitude": 48.4772, "longitude": -120.1864},
        },
    )
    assert response.status_code == 200
    body = response.json()
    geocode = body["geocoding"]
    assert geocode["resolution_method"] == "user_selected_point"
    assert geocode["fallback_used"] is True
    assert body["coverage_available"] is True


def test_street_only_fallback_candidate_is_not_auto_used(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "street_only_fallback_rejected.db")))
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setenv("WF_GEOCODE_ENABLE_PROVIDER_BACKOFF_QUERY", "false")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _addr: {
            "matched": False,
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": ["no authoritative local match"],
            "failure_reason": "no_local_authoritative_match",
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_fallback_coordinates",
        lambda _addr: {
            "matched": False,
            "confidence": None,
            "candidate_count": 1,
            "best_match": None,
            "best_candidate": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "match_type": "street_only_match",
                "confidence_tier": "low",
                "auto_usable": False,
            },
            "top_candidates": [
                {
                    "latitude": 48.4772,
                    "longitude": -120.1864,
                    "match_type": "street_only_match",
                    "confidence_tier": "low",
                    "auto_usable": False,
                }
            ],
            "diagnostics": ["street-only fallback records are not auto-usable"],
            "failure_reason": "street_only_match",
        },
    )

    response = client.post(
        "/risk/assess",
        json={
            "address": "6 Pineview Rd, Winthrop, WA 98862",
            "attributes": {"roof_type": "class a"},
            "confirmed_fields": ["roof_type"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["geocode_status"] == "low_confidence"
    assert detail["rejection_category"] == "location_needs_confirmation"
    assert "selecting your home" in detail["message"].lower()


def test_winthrop_known_address_no_primary_match_uses_fallback_pipeline(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "winthrop_known_address_fallback.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=44.0, wildland=52.0, historic=41.0))
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
        },
    )

    response = client.post(
        "/risk/assess",
        json={
            "address": "6 Pineview Rd, Winthrop, WA 98862",
            "attributes": {"roof_type": "class a"},
            "confirmed_fields": ["roof_type"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["geocoding"]["resolution_method"] in {
        "local_authoritative_fallback",
        "explicit_fallback_record",
        "provider_backoff_query",
    }
    assert body["geocoding"]["fallback_used"] is True
    assert body["coverage_available"] is True


def test_in_region_local_authoritative_candidate_overrides_outside_primary_geocode(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "winthrop_override_outside_geocode.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=44.0, wildland=52.0, historic=41.0))
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setenv("WF_GEOCODE_ENABLE_PROVIDER_BACKOFF_QUERY", "false")

    def _outside_geocode(_addr):
        app_main.geocoder.last_result = {
            "geocode_status": "accepted",
            "normalized_address": "6 Pineview Rd, Winthrop, WA 98862",
            "provider": "test-geocoder",
            "matched_address": "6 Pineview Rd, Winthrop, WA 98862, USA",
            "confidence_score": 0.45,
            "candidate_count": 1,
            "geocode_precision": "parcel_or_address_point",
            "rejection_reason": None,
        }
        # Intentionally outside prepared Winthrop regions.
        return (47.01, -122.90, "test-geocoder")

    monkeypatch.setattr(app_main.geocoder, "geocode", _outside_geocode)
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _addr: {
            "matched": True,
            "confidence": "medium",
            "candidate_count": 1,
            "best_match": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "match_score": 0.87,
                "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
                "source": "address_points",
            },
            "top_candidates": [],
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "diagnostics": ["resolved via local authoritative test match"],
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_fallback_coordinates",
        lambda _addr: {
            "matched": False,
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": ["not used"],
            "failure_reason": "no_explicit_fallback_match",
        },
    )

    def _lookup(lat, lon, regions_root=None):  # noqa: ARG001
        if abs(lat - 48.4772) < 0.01 and abs(lon + 120.1864) < 0.01:
            return {
                "covered": True,
                "region_id": "winthrop_pilot",
                "display_name": "Winthrop Pilot",
                "diagnostics": [],
                "containing_region_ids": ["winthrop_pilot", "winthrop_large"],
            }
        return {
            "covered": False,
            "diagnostics": ["No prepared region bounds contain point."],
            "nearest_region_id": "winthrop_large",
            "region_distance_to_boundary_m": 180000.0,
            "containing_region_ids": [],
        }

    monkeypatch.setattr(app_main, "lookup_region_for_point", _lookup)

    payload = {
        "address": "6 Pineview Rd, Winthrop, WA 98862",
        "attributes": {"roof_type": "class a"},
        "confirmed_fields": ["roof_type"],
        "audience": "homeowner",
    }
    coverage = client.post("/regions/coverage-check", json={"address": payload["address"]})
    assess = client.post("/risk/assess", json=payload)

    assert coverage.status_code == 200
    assert assess.status_code == 200
    coverage_body = coverage.json()
    assess_body = assess.json()
    assert coverage_body["coverage_available"] is True
    assert coverage_body["resolved_region_id"] == "winthrop_pilot"
    assert coverage_body["resolution_method"] == "local_authoritative_fallback"
    assert assess_body["coverage_available"] is True
    assert assess_body["resolved_region_id"] == "winthrop_pilot"
    assert assess_body["geocoding"]["resolution_method"] == "local_authoritative_fallback"
    assert assess_body["geocoding"]["candidate_regions_containing_point"] == ["winthrop_pilot", "winthrop_large"]


def test_geocode_debug_returns_structured_accepted_payload_with_region_resolution(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_debug_ok.db")))
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (46.8721, -113.9940, "test-geocoder"))
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "missoula_pilot",
            "display_name": "Missoula Pilot",
            "diagnostics": [],
        },
    )

    res = client.post("/risk/geocode-debug", json={"address": "201 W Front St, Missoula, MT 59802"})
    assert res.status_code == 200
    body = res.json()
    assert body["geocode_status"] == "accepted"
    assert body["accepted"] is True
    assert body["resolved_latitude"] == pytest.approx(46.8721)
    assert body["resolved_longitude"] == pytest.approx(-113.9940)
    assert body["region_resolution"]["coverage_available"] is True
    assert body["region_resolution"]["resolved_region_id"] == "missoula_pilot"


def test_geocode_debug_returns_accepted_for_valid_uncovered_address(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_debug_uncovered.db")))

    def _fake_geocode(_addr):
        app_main.geocoder.last_result = {
            "geocode_status": "accepted",
            "normalized_address": "62910 O B Riley Rd, Bend, OR 97703",
            "provider": "test-geocoder",
            "matched_address": "62910 O B Riley Rd, Bend, OR 97703, USA",
            "confidence_score": 0.03,
            "candidate_count": 2,
            "rejection_reason": None,
            "raw_response_preview": {
                "candidate_count": 2,
                "parsed_candidates": [{"display_name": "62910 O B Riley Rd, Bend, OR 97703, USA", "importance": 0.03}],
            },
        }
        return (44.0839, -121.3153, "test-geocoder")

    monkeypatch.setattr(app_main.geocoder, "geocode", _fake_geocode)
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {"covered": False, "diagnostics": ["No prepared region bounds contain point."]},
    )

    res = client.post("/debug/geocode", json={"address": "62910 O B Riley Rd, Bend, OR 97703"})
    assert res.status_code == 200
    body = res.json()
    assert body["geocode_status"] == "accepted"
    assert body["accepted"] is True
    assert body["region_resolution"]["coverage_available"] is False
    assert body["region_resolution"]["reason"] == "no_prepared_region_for_location"
    assert body["parsed_candidates"]


def test_geocode_debug_returns_structured_rejection_for_parser_error(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "geocode_debug_parser.db")))
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="parser_error",
                message="Address input is too short for geocoding.",
                submitted_address="??",
                normalized_address="??",
                rejection_reason="input_too_short",
            )
        ),
    )

    res = client.post("/risk/geocode-debug", json={"address": "??"})
    assert res.status_code == 200
    body = res.json()
    assert body["accepted"] is False
    assert body["geocode_status"] == "parser_error"
    assert body["rejection_category"] == "parser_error"
    assert body["rejection_reason"] == "input_too_short"
    assert body["region_resolution"] is None


def test_route_geocode_and_coverage_are_consistent_for_valid_covered_address(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "route_consistency_covered.db")))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=48.0, wildland=42.0, historic=35.0))
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (46.8721, -113.9940, "test-geocoder"),
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "missoula_pilot",
            "display_name": "Missoula Pilot",
            "diagnostics": [],
        },
    )

    payload = {
        "address": "201 W Front St, Missoula, MT 59802",
        "attributes": {"roof_type": "class a"},
        "confirmed_fields": ["roof_type"],
        "audience": "homeowner",
    }

    coverage = client.post("/regions/coverage-check", json={"address": payload["address"]})
    debug = client.post("/risk/debug", json=payload)
    assess = client.post("/risk/assess", json=payload)

    assert coverage.status_code == 200
    assert debug.status_code == 200
    assert assess.status_code == 200

    coverage_body = coverage.json()
    debug_body = debug.json()
    assess_body = assess.json()

    assert coverage_body["geocode_status"] == "accepted"
    assert coverage_body["coverage_available"] is True
    assert coverage_body["resolved_region_id"] == "missoula_pilot"
    assert debug_body["geocoding"]["geocode_status"] == "accepted"
    assert debug_body["region_resolution"]["coverage_available"] is True
    assert debug_body["region_resolution"]["resolved_region_id"] == "missoula_pilot"
    assert assess_body["geocoding"]["geocode_status"] == "accepted"
    assert assess_body["coverage_available"] is True
    assert assess_body["resolved_region_id"] == "missoula_pilot"
    assert debug_body["geocoding"]["resolved_latitude"] == pytest.approx(coverage_body["latitude"])
    assert debug_body["geocoding"]["resolved_longitude"] == pytest.approx(coverage_body["longitude"])
    assert assess_body["geocoding"]["resolved_latitude"] == pytest.approx(coverage_body["latitude"])
    assert assess_body["geocoding"]["resolved_longitude"] == pytest.approx(coverage_body["longitude"])
    assert assess_body["display_point_source"] in {"property_anchor_point", "matched_structure_centroid"}
    assert assess_body["structure_match_status"] in {"none", "matched", "ambiguous", "provider_unavailable", "error"}


def test_route_geocode_and_coverage_are_consistent_for_valid_uncovered_address(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "route_consistency_uncovered.db")))
    monkeypatch.setenv("WF_REQUIRE_PREPARED_REGION_COVERAGE", "true")
    monkeypatch.setenv("WF_AUTO_QUEUE_REGION_PREP_ON_MISS", "false")
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx(env=45.0, wildland=45.0, historic=45.0))
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (44.0839, -121.3153, "test-geocoder"),
    )
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": False,
            "diagnostics": ["No prepared region bounds contain point."],
        },
    )

    payload = {
        "address": "62910 O B Riley Rd, Bend, OR 97703",
        "attributes": {"roof_type": "class a"},
        "confirmed_fields": ["roof_type"],
        "audience": "homeowner",
    }

    coverage = client.post("/regions/coverage-check", json={"address": payload["address"]})
    debug = client.post("/risk/debug", json=payload)
    assess = client.post("/risk/assess", json=payload)

    assert coverage.status_code == 200
    assert debug.status_code == 200
    assert assess.status_code == 409

    coverage_body = coverage.json()
    debug_body = debug.json()
    assess_detail = assess.json()["detail"]

    assert coverage_body["geocode_status"] == "accepted"
    assert coverage_body["coverage_available"] is False
    assert coverage_body["reason"] == "no_prepared_region_for_location"
    assert debug_body["geocoding"]["geocode_status"] == "accepted"
    assert debug_body["region_resolution"]["coverage_available"] is False
    assert debug_body["region_resolution"]["reason"] == "no_prepared_region_for_location"
    assert assess_detail["region_not_ready"] is True
    assert assess_detail["geocode_status"] == "accepted"
    assert assess_detail["coverage_available"] is False
    assert assess_detail["reason"] == "no_prepared_region_for_location"
    assert assess_detail["resolved_latitude"] == pytest.approx(coverage_body["latitude"])
    assert assess_detail["resolved_longitude"] == pytest.approx(coverage_body["longitude"])
    assert "error" not in assess_detail
    assert assess_detail["normalized_address"]


def test_route_geocode_and_coverage_are_consistent_for_invalid_address(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "route_consistency_invalid.db")))
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="Invalid Input",
                normalized_address="Invalid Input",
                rejection_reason="provider returned no candidates",
            )
        ),
    )

    payload = {
        "address": "Invalid Input",
        "attributes": {"roof_type": "class a"},
        "confirmed_fields": [],
        "audience": "homeowner",
    }

    coverage = client.post("/regions/coverage-check", json={"address": payload["address"]})
    debug = client.post("/risk/debug", json=payload)
    assess = client.post("/risk/assess", json=payload)

    assert coverage.status_code == 422
    assert debug.status_code == 422
    assert assess.status_code == 422

    for response in (coverage, debug, assess):
        detail = response.json()["detail"]
        assert detail["error"] == "geocoding_failed"
        assert detail["geocode_status"] == "no_match"
        assert detail["normalized_address"] == "Invalid Input"
        assert detail["resolution_status"] == "unresolved"
        assert detail["resolution_method"] == "none"
        assert isinstance(detail.get("provider_attempts"), list)
        assert detail.get("local_fallback_attempted") is True


def test_structure_resolution_failure_is_not_reported_as_geocoding_failure(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "structure_failure_not_geocode.db")))
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (48.4772, -120.1864, "test-geocoder"))

    def _collect_context_fail(_lat, _lon, **kwargs):  # noqa: ARG001
        raise RuntimeError("structure matching backend unavailable")

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", _collect_context_fail)
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
        },
    )

    non_raising_client = TestClient(app_main.app, raise_server_exceptions=False)
    response = non_raising_client.post(
        "/risk/assess",
        json={
            "address": "104 Riverside Ave, Winthrop, WA 98862",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert response.status_code == 500
    try:
        parsed = response.json()
    except Exception:
        parsed = {"detail": response.text}
    detail = parsed.get("detail")
    if isinstance(detail, dict):
        assert detail.get("error") != "geocoding_failed"
    elif isinstance(detail, str):
        assert "geocoding failed" not in detail.lower()


def _run(payload: dict) -> dict:
    res = client.post("/risk/assess", json=payload)
    assert res.status_code == 200
    return res.json()


def _subset_assert(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            _subset_assert(actual[key], value)
    elif isinstance(expected, list):
        assert actual == expected
    else:
        assert actual == expected


def _assert_calibration_fixture(body: dict, fixture_name: str) -> None:
    expected = json.loads((FIXTURE_DIR / fixture_name).read_text())
    _subset_assert(body, expected)


def test_assess_and_report_core_shape_match(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=35.0, wildland=40.0, historic=20.0))

    assessed = _run(
        _payload(
            "101 Contract Way, Boulder, CO",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 32,
                "construction_year": 2018,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        )
    )

    report_res = client.get(f"/report/{assessed['assessment_id']}")
    assert report_res.status_code == 200
    report = report_res.json()

    _assert_core_contract(assessed)
    _assert_core_contract(report)

    for key in [
        "model_version",
        "wildfire_risk_score",
        "insurance_readiness_score",
        "submodel_scores",
        "weighted_contributions",
        "factor_breakdown",
        "readiness_blockers",
        "readiness_penalties",
        "mitigation_plan",
    ]:
        assert assessed[key] == report[key]


def test_assessment_with_confirmed_property_facts(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=55.0, historic=50.0))

    unconfirmed = _run(_payload("456 Mid St, Denver, CO", {"defensible_space_ft": 20}, confirmed=[]))
    confirmed = _run(
        _payload(
            "456 Mid St, Denver, CO",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 20,
                "construction_year": 2015,
                "siding_type": "fiber cement",
                "window_type": "dual pane tempered",
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        )
    )

    _assert_core_contract(confirmed)
    assert unconfirmed["confidence_score"] > 0.0
    assert confirmed["confirmed_inputs"]["roof_type"] == "class a"
    assert confirmed["confidence_score"] > unconfirmed["confidence_score"]
    assert confirmed["confidence"]["confirmed_fields_count"] >= unconfirmed["confidence"]["confirmed_fields_count"]
    assert confirmed["confidence"]["environmental_data_present"] is True


def test_simulation_returns_deltas(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=70.0, wildland=85.0, historic=65.0))

    baseline = _run(
        _payload(
            "789 High St, Colorado Springs, CO",
            {
                "roof_type": "wood",
                "vent_type": "standard",
                "defensible_space_ft": 5,
                "construction_year": 1990,
            },
        )
    )

    sim_res = client.post(
        "/risk/simulate",
        json={
            "assessment_id": baseline["assessment_id"],
            "scenario_name": "hardening_upgrade",
            "scenario_overrides": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 35,
            },
            "scenario_confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        },
    )
    assert sim_res.status_code == 200
    sim = sim_res.json()

    assert sim["baseline"]["assessment_id"] != sim["simulated"]["assessment_id"]
    assert sim["delta"]["wildfire_risk_score_delta"] <= 0
    assert sim["delta"]["insurance_readiness_score_delta"] >= 0
    assert "home_hardening_readiness_delta" in sim["delta"]
    assert isinstance(sim.get("simulator_explanations"), dict)
    assert "current_risk_score" in sim["simulator_explanations"]
    assert "simulated_risk_score" in sim["simulator_explanations"]
    assert "estimated_risk_reduction" in sim["simulator_explanations"]
    assert "roof_type" in sim["changed_inputs"]


def test_simulation_ui_enum_roof_values_produce_non_zero_delta(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=70.0, wildland=85.0, historic=65.0))

    baseline = _run(
        _payload(
            "790 Enum Fix Ave, Colorado Springs, CO",
            {
                "roof_type": "wood_shake",
                "vent_type": "standard_vents",
                "defensible_space_ft": 5,
                "construction_year": 1990,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )

    sim_res = client.post(
        "/risk/simulate",
        json={
            "assessment_id": baseline["assessment_id"],
            "scenario_name": "enum_upgrade",
            "scenario_overrides": {
                "roof_type": "class_a_asphalt_composition",
                "vent_type": "ember_resistant_vents",
                "defensible_space_ft": 35,
            },
            "scenario_confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        },
    )
    assert sim_res.status_code == 200
    sim = sim_res.json()

    assert sim["changed_inputs"]["roof_type"]["before"] == "wood_shake"
    assert sim["changed_inputs"]["roof_type"]["after"] == "class_a_asphalt_composition"
    assert sim["changed_inputs"]["vent_type"]["before"] == "standard_vents"
    assert sim["changed_inputs"]["vent_type"]["after"] == "ember_resistant_vents"

    assert sim["delta"]["wildfire_risk_score_delta"] is not None
    assert sim["delta"]["insurance_readiness_score_delta"] is not None
    assert sim["delta"]["wildfire_risk_score_delta"] < 0
    assert sim["delta"]["insurance_readiness_score_delta"] > 0


def test_baseline_confidence_non_zero_with_geospatial_context_and_no_optional_home_facts(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=52.0, wildland=57.0, historic=46.0, ring_metrics={}))

    assessed = _run(_payload("777 Baseline Confidence Ln", {}, confirmed=[]))
    assert assessed["confidence_score"] > 0.0
    assert assessed["confidence"]["environmental_data_present"] is True
    assert assessed["confidence"]["property_context_present"] is True
    assert assessed["confidence"]["inferred_fields_count"] >= 0
    assert assessed["missing_factor_count"] >= 1
    assert "missing_critical_fields" in assessed["confidence"]


def test_frontend_confidence_rendering_does_not_coerce_to_zero():
    ui_path = Path(__file__).resolve().parents[1] / "frontend" / "public" / "index.html"
    html = ui_path.read_text(encoding="utf-8")
    assert "confidence_score || 0" not in html
    assert "Number.isFinite(confidenceRaw)" in html


def test_simulation_history_listing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=65.0, wildland=70.0, historic=60.0))

    baseline = _run(
        _payload(
            "500 Scenario Store Way",
            {
                "roof_type": "wood",
                "vent_type": "standard",
                "defensible_space_ft": 10,
                "construction_year": 1995,
            },
        )
    )

    sim_res = client.post(
        "/risk/simulate",
        json={
            "assessment_id": baseline["assessment_id"],
            "scenario_name": "scenario_saved_test",
            "scenario_overrides": {"roof_type": "class a", "vent_type": "ember-resistant"},
            "scenario_confirmed_fields": ["roof_type", "vent_type"],
        },
    )
    assert sim_res.status_code == 200

    history = client.get(f"/assessments/{baseline['assessment_id']}/scenarios?limit=10")
    assert history.status_code == 200
    rows = history.json()
    assert len(rows) >= 1
    assert rows[0]["assessment_id"] == baseline["assessment_id"]


def test_simulation_from_address_and_attributes(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=68.0, wildland=74.0, historic=61.0))

    res = client.post(
        "/risk/simulate",
        json={
            "address": "123 Main St, Boulder, CO",
            "attributes": {
                "roof_type": "wood shake",
                "vent_type": "standard",
                "defensible_space_ft": 5,
            },
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
            "scenario_name": "homeowner_upgrade",
            "scenario_overrides": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 30,
            },
            "scenario_confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        },
    )
    assert res.status_code == 200
    sim = res.json()

    assert sim["base_assessment_id"]
    assert sim["simulated_assessment_id"]
    assert sim["score_delta"]["wildfire_risk_score_delta"] <= 0
    assert sim["score_delta"]["insurance_readiness_score_delta"] >= 0
    assert "roof_type" in sim["changed_inputs"]
    assert sim["summary"]


def test_simulation_missing_assessment_id_returns_404(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=50.0, historic=50.0))

    res = client.post(
        "/risk/simulate",
        json={
            "assessment_id": "does-not-exist",
            "scenario_name": "bad_reference",
            "scenario_overrides": {"roof_type": "class a"},
            "scenario_confirmed_fields": ["roof_type"],
        },
    )
    assert res.status_code == 404


def test_simulation_missing_address_and_assessment_returns_400(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=50.0, historic=50.0))

    res = client.post(
        "/risk/simulate",
        json={
            "scenario_name": "missing_base",
            "scenario_overrides": {"roof_type": "class a"},
            "scenario_confirmed_fields": ["roof_type"],
        },
    )
    assert res.status_code == 400
    assert "assessment_id or address" in res.text


def test_simulation_confirmed_field_merge(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=64.0, wildland=70.0, historic=58.0))

    base = _run(
        _payload(
            "812 Merge St, Boulder, CO",
            {
                "roof_type": "wood",
                "vent_type": "standard",
                "defensible_space_ft": 8,
            },
            confirmed=["roof_type"],
        )
    )

    res = client.post(
        "/risk/simulate",
        json={
            "assessment_id": base["assessment_id"],
            "attributes": {"vent_type": "standard"},
            "confirmed_fields": ["vent_type"],
            "scenario_name": "merge_confirmed",
            "scenario_overrides": {"defensible_space_ft": 30},
            "scenario_confirmed_fields": ["defensible_space_ft"],
        },
    )
    assert res.status_code == 200
    sim = res.json()
    assert sorted(sim["simulated"]["confirmed_fields"]) == [
        "defensible_space_ft",
        "roof_type",
        "vent_type",
    ]


def test_reassess_from_existing_assessment(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=60.0, wildland=60.0, historic=55.0))

    original = _run(_payload("100 Reassess Ln", {"defensible_space_ft": 15}, tags=["initial"]))

    res = client.post(
        f"/risk/reassess/{original['assessment_id']}",
        json={
            "attributes": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 35,
                "construction_year": 2018,
            },
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
            "audience": "inspector",
        },
    )
    assert res.status_code == 200
    updated = res.json()

    assert updated["assessment_id"] != original["assessment_id"]
    assert updated["audience"] == "inspector"
    assert updated["property_facts"]["roof_type"] == "class a"
    assert updated["tags"] == ["initial"]


def test_report_export_and_view_with_audience(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=45.0, wildland=40.0, historic=30.0))
    assessed = _run(
        _payload(
            "200 Report View Dr",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 30},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )

    report_res = client.get(f"/report/{assessed['assessment_id']}?audience=inspector")
    assert report_res.status_code == 200
    report = report_res.json()
    assert report["report_audience"] == "inspector"
    assert len(report["audience_highlights"]) > 0

    export_res = client.get(f"/report/{assessed['assessment_id']}/export?audience=insurer&include_benchmark_hints=true")
    assert export_res.status_code == 200
    exported = export_res.json()
    assert exported["audience_mode"] == "insurer"
    assert "audience_focus" in exported
    assert "score_evidence_ledger" in exported
    assert "evidence_quality_summary" in exported
    assert "governance_metadata" in exported
    assert "model_governance" in exported
    assert exported["governance_metadata"] == exported["model_governance"]
    assert "benchmark_hints" in exported["assumptions_confidence"]
    assert exported["evidence_quality_summary"]["use_restriction"] in {
        "consumer_estimate",
        "screening_only",
        "review_required",
    }
    assert "evidence_quality_summary" in exported["assumptions_confidence"]

    view_res = client.get(f"/report/{assessed['assessment_id']}/view?audience=agent")
    assert view_res.status_code == 200
    assert "text/html" in view_res.headers.get("content-type", "")
    assert "Audience View agent" in view_res.text


def test_debug_endpoint_includes_evidence_ledger_and_summary(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=49.0, wildland=51.0, historic=44.0))
    res = client.post(
        "/risk/debug",
        json={
            "address": "Debug Evidence Ln",
            "attributes": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 25,
            },
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
            "audience": "insurer",
            "tags": [],
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert "score_evidence_ledger" in payload
    assert "evidence_quality_summary" in payload
    assert "layer_coverage_audit" in payload
    assert "coverage_summary" in payload
    assert isinstance(payload.get("geometry_resolution"), dict)
    assert payload["evidence_quality_summary"]["observed_factor_count"] >= 0


def test_debug_endpoint_optional_benchmark_hints(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=47.0, wildland=44.0, historic=39.0))
    res = client.post(
        "/risk/debug?include_benchmark_hints=true",
        json={
            "address": "Debug Benchmark Hints Ln",
            "attributes": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 26,
            },
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
            "audience": "insurer",
            "tags": [],
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert "benchmark_hints" in payload
    hints = payload["benchmark_hints"]
    assert "benchmark_pack_version" in hints
    assert "benchmark_style_sanity_checks" in hints
    assert "all_sanity_checks_passed" in hints


def test_layer_diagnostics_endpoint_returns_layer_audit(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=58.0, wildland=52.0, historic=43.0, ring_metrics={}))

    res = client.post(
        "/risk/layer-diagnostics",
        json={
            "address": "Layer Diagnostics Way",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
            "tags": [],
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert "coordinates" in payload
    assert "region" in payload
    assert "structure_footprint" in payload
    assert "geometry_resolution" in payload
    assert "layer_coverage_audit" in payload
    assert "coverage_summary" in payload
    assert isinstance(payload["layer_coverage_audit"], list)
    assert isinstance(payload["coverage_summary"], dict)
    assert isinstance(payload["geometry_resolution"], dict)
    assert isinstance(payload["fallback_decisions"], dict)
    assert isinstance(payload["warnings"], list)


def test_layer_coverage_sampling_failure_penalty_and_restriction(monkeypatch, tmp_path):
    degraded = _ctx(env=52.0, wildland=48.0, historic=40.0, ring_metrics={})
    degraded.property_level_context = {
        "footprint_used": False,
        "footprint_status": "provider_unavailable",
        "fallback_mode": "point_based",
        "ring_metrics": None,
        "layer_coverage_audit": [
            {
                "layer_key": "fuel",
                "display_name": "Fuel Model",
                "required_for": ["site_hazard"],
                "configured": True,
                "present_in_region": True,
                "sample_attempted": True,
                "sample_succeeded": False,
                "coverage_status": "sampling_failed",
                "source_type": "prepared_region",
                "failure_reason": "raster read error",
                "notes": [],
            },
            {
                "layer_key": "building_footprints",
                "display_name": "Building Footprints",
                "required_for": ["home_ignition_vulnerability"],
                "configured": True,
                "present_in_region": True,
                "sample_attempted": True,
                "sample_succeeded": False,
                "coverage_status": "outside_extent",
                "source_type": "prepared_region",
                "failure_reason": "outside extent",
                "notes": ["Point-based fallback used."],
            },
        ],
        "coverage_summary": {
            "total_layers_checked": 2,
            "observed_count": 0,
            "partial_count": 0,
            "fallback_count": 0,
            "failed_count": 2,
            "not_configured_count": 0,
            "critical_missing_layers": ["fuel", "building_footprints"],
            "recommended_actions": ["fuel sampling failed; validate raster integrity."],
        },
    }
    _setup(monkeypatch, tmp_path, degraded)

    assessed = _run(_payload("Layer Failure Ct", {}, confirmed=[]))
    assert assessed["use_restriction"] == "not_for_underwriting_or_binding"
    penalties = assessed["evidence_quality_summary"]["confidence_penalties"]
    penalty_keys = {p["penalty_key"] for p in penalties}
    assert "layer_sampling_failures" in penalty_keys
    assert "critical_layer_gaps" in penalty_keys
    assert any(r["layer_key"] == "building_footprints" and r["coverage_status"] == "outside_extent" for r in assessed["layer_coverage_audit"])


def test_optional_layer_not_configured_is_explicit_and_nonfatal(monkeypatch, tmp_path):
    partial = _ctx(env=55.0, wildland=51.0, historic=44.0, ring_metrics={})
    partial.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": None,
        "layer_coverage_audit": [
            {
                "layer_key": "dem",
                "display_name": "Digital Elevation Model",
                "required_for": ["site_hazard"],
                "configured": True,
                "present_in_region": True,
                "sample_attempted": True,
                "sample_succeeded": True,
                "coverage_status": "observed",
                "source_type": "prepared_region",
                "notes": [],
            },
            {
                "layer_key": "gridmet_dryness",
                "display_name": "gridMET Dryness",
                "required_for": ["site_hazard"],
                "configured": False,
                "present_in_region": False,
                "sample_attempted": False,
                "sample_succeeded": False,
                "coverage_status": "not_configured",
                "source_type": "open_data",
                "failure_reason": "No configured source path.",
                "notes": [],
            },
        ],
        "coverage_summary": {
            "total_layers_checked": 2,
            "observed_count": 1,
            "partial_count": 0,
            "fallback_count": 0,
            "failed_count": 0,
            "not_configured_count": 1,
            "critical_missing_layers": [],
            "recommended_actions": ["gridmet_dryness is not configured; add a source path or keep optional."],
        },
    }
    _setup(monkeypatch, tmp_path, partial)

    assessed = _run(_payload("Optional Layer Way", {"defensible_space_ft": 20}, confirmed=["defensible_space_ft"]))
    assert assessed["assessment_status"] in {"fully_scored", "partially_scored"}
    assert assessed["coverage_summary"]["not_configured_count"] == 1
    assert any(r["layer_key"] == "gridmet_dryness" and r["coverage_status"] == "not_configured" for r in assessed["layer_coverage_audit"])
    assert any("not configured" in n.lower() for n in assessed["scoring_notes"])


def test_runtime_region_readiness_limited_forces_stricter_confidence_and_restriction(monkeypatch, tmp_path):
    constrained = _ctx(
        env=56.0,
        wildland=52.0,
        historic=44.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 35.0},
            "ring_5_30_ft": {"vegetation_density": 40.0},
            "ring_30_100_ft": {"vegetation_density": 46.0},
        },
    )
    constrained.property_level_context.update(
        {
            "region_status": "prepared",
            "region_id": "test_region",
            "region_property_specific_readiness": "limited_regional_ready",
            "region_required_layers_missing": ["dem"],
            "region_optional_layers_missing": ["whp", "gridmet_dryness"],
            "region_enrichment_layers_missing": ["parcel_polygons"],
            "region_missing_reason_by_layer": {"dem": "configured_but_outside_coverage"},
            "layer_coverage_audit": [
                {
                    "layer_key": "dem",
                    "display_name": "Digital Elevation Model",
                    "required_for": ["site_hazard"],
                    "configured": True,
                    "present_in_region": True,
                    "sample_attempted": True,
                    "sample_succeeded": True,
                    "coverage_status": "observed",
                    "source_type": "prepared_region",
                    "notes": [],
                }
            ],
            "coverage_summary": {
                "total_layers_checked": 1,
                "observed_count": 1,
                "partial_count": 0,
                "fallback_count": 0,
                "failed_count": 0,
                "not_configured_count": 0,
                "critical_missing_layers": [],
                "recommended_actions": [],
            },
        }
    )
    _setup(monkeypatch, tmp_path, constrained)

    assessed = _run(
        _payload(
            "Limited Readiness Road",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 25},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    assert assessed["use_restriction"] == "not_for_underwriting_or_binding"
    assert assessed["confidence_tier"] == "preliminary"
    assert assessed["assessment_output_state"] in {"limited_regional_estimate", "insufficient_data"}
    assert any("prepared region still reports required-layer gaps" in note.lower() for note in assessed["scoring_notes"])
    limitation_categories = {
        row["category"]
        for row in ((assessed.get("homeowner_summary") or {}).get("assessment_limitations") or [])
        if isinstance(row, dict) and row.get("category")
    }
    assert "prepared_region_readiness" in limitation_categories


def test_portfolio_batch_and_partial_failure(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=55.0, historic=45.0))

    res = client.post(
        "/portfolio/assess",
        json={
            "portfolio_name": "Q2 Carrier Review",
            "items": [
                {
                    "row_id": "A1",
                    "address": "11 Batch Way, Denver, CO",
                    "attributes": {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 30},
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
                    "audience": "insurer",
                    "tags": ["renewal", "co"],
                },
                {
                    "row_id": "A2",
                    "address": "",
                    "attributes": {"roof_type": "wood", "vent_type": "standard", "defensible_space_ft": 8},
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
                    "audience": "insurer",
                    "tags": ["renewal", "co"],
                },
            ],
        },
    )
    assert res.status_code == 200
    body = res.json()

    assert body["total_properties"] == 2
    assert body["completed_count"] == 1
    assert body["failed_count"] == 1
    assert body["total"] == 2
    assert body["succeeded"] == 1
    assert body["failed"] == 1
    assert "high_risk_count" in body
    assert "average_wildfire_risk" in body
    assert any(r["status"] == "failed" for r in body["results"])


def test_portfolio_filters_summary_and_recent(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=65.0, wildland=80.0, historic=60.0))

    strong = _run(
        _payload(
            "400 Strong Structure Ln",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 35},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
            tags=["portfolio_x"],
        )
    )
    weak = _run(
        _payload(
            "401 Weak Structure Ln",
            {"roof_type": "wood", "vent_type": "standard", "defensible_space_ft": 5},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
            tags=["portfolio_x"],
        )
    )

    assert weak["wildfire_risk_score"] >= strong["wildfire_risk_score"]

    filt = client.get(
        "/portfolio?sort_by=wildfire_risk_score&sort_dir=desc&tag=portfolio_x&readiness_blocker=Defensible%20space&recent_days=365&limit=10"
    )
    assert filt.status_code == 200
    pbody = filt.json()
    assert "summary" in pbody
    rows = pbody["items"]
    assert len(rows) >= 1
    assert rows[0]["wildfire_risk_score"] >= rows[-1]["wildfire_risk_score"]
    assert any("Defensible" in b for b in rows[0]["readiness_blockers"])

    summary = client.get("/assessments/summary?tag=portfolio_x")
    assert summary.status_code == 200
    sb = summary.json()["summary"]
    assert sb["total_count"] >= 2
    assert "high_risk_count" in sb


def test_assessment_listing_fields(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=35.0, wildland=35.0, historic=20.0))
    _run(_payload("One Listing St", {"defensible_space_ft": 20}, tags=["renewal"]))
    _run(_payload("Two Listing St", {"defensible_space_ft": 22}, tags=["inspection"]))

    res = client.get("/assessments?limit=5")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) >= 2
    assert "assessment_id" in rows[0]
    assert "created_at" in rows[0]
    assert "model_version" in rows[0]
    assert "confidence_score" in rows[0]
    assert "review_status" in rows[0]


def test_annotations_and_review_status_workflow(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=60.0, historic=50.0))
    assessed = _run(_payload("900 Notes Blvd", {"defensible_space_ft": 20}))

    status_res = client.put(
        f"/assessments/{assessed['assessment_id']}/review-status",
        json={"review_status": "flagged"},
    )
    assert status_res.status_code == 200
    assert status_res.json()["review_status"] == "flagged"

    add = client.post(
        f"/assessments/{assessed['assessment_id']}/annotations",
        json={
            "author_role": "inspector",
            "note": "Observed dense brush in immediate zone 1.",
            "tags": ["inspection", "zone1"],
            "visibility": "shared",
            "review_status": "reviewed",
        },
    )
    assert add.status_code == 200
    row = add.json()
    assert row["assessment_id"] == assessed["assessment_id"]
    assert row["author_role"] == "inspector"
    assert row["review_status"] == "reviewed"

    listed = client.get(f"/assessment/{assessed['assessment_id']}/annotations?visibility=shared")
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) >= 1
    assert rows[0]["visibility"] == "shared"

    status_now = client.get(f"/assessments/{assessed['assessment_id']}/review-status")
    assert status_now.status_code == 200
    assert status_now.json()["review_status"] == "reviewed"


def test_compare_endpoints(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=62.0, wildland=75.0, historic=60.0))

    a = _run(
        _payload(
            "100 Compare One",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 35},
        )
    )
    b = _run(
        _payload(
            "101 Compare Two",
            {"roof_type": "wood", "vent_type": "standard", "defensible_space_ft": 5},
        )
    )

    pair = client.get(f"/assessments/{a['assessment_id']}/compare/{b['assessment_id']}")
    assert pair.status_code == 200
    p = pair.json()
    assert "wildfire_risk_delta" in p
    assert "insurance_readiness_delta" in p
    assert "driver_differences" in p
    assert "blocker_differences" in p
    assert "mitigation_differences" in p
    assert "version_comparison" in p
    assert "directly_comparable" in p["version_comparison"]

    multi = client.get(f"/assessments/compare?ids={a['assessment_id']},{b['assessment_id']}")
    assert multi.status_code == 200
    m = multi.json()
    assert m["requested_ids"] == [a["assessment_id"], b["assessment_id"]]
    assert len(m["comparisons"]) == 1


def test_compare_endpoint_suppresses_adjacent_home_deltas_when_mostly_regional(monkeypatch, tmp_path):
    ctx = _ctx(
        env=50.0,
        wildland=57.0,
        historic=40.0,
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "missing",
            "fire_history": "missing",
        },
        ring_metrics={},
    )
    ctx.burn_probability_index = None
    ctx.hazard_severity_index = None
    ctx.moisture_index = None
    ctx.burn_probability = None
    ctx.wildfire_hazard = None
    ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": {},
        "region_property_specific_readiness": "limited_regional_ready",
        "region_required_layers_missing": [],
        "region_optional_layers_missing": ["whp", "roads", "gridmet_dryness"],
        "region_enrichment_layers_missing": ["parcel_polygons"],
    }
    _setup(monkeypatch, tmp_path, ctx)

    a = _run(_payload("Comparison Safeguard A", {}, confirmed=[]))
    b = _run(_payload("Comparison Safeguard B", {}, confirmed=[]))
    pair = client.get(f"/assessments/{a['assessment_id']}/compare/{b['assessment_id']}")
    assert pair.status_code == 200
    body = pair.json()
    safeguard = (body.get("version_comparison") or {}).get("comparison_precision_safeguard") or {}
    assert safeguard.get("triggered") is True
    assert safeguard.get("message") == "This estimate is not precise enough to compare adjacent homes."
    assert body.get("wildfire_risk_delta") is None
    assert body.get("insurance_readiness_delta") is None
    assert body.get("driver_differences") == {"added": [], "removed": []}
    assert body.get("mitigation_differences") == {"added": [], "removed": []}


def test_provisional_access_not_in_wildfire_score(monkeypatch, tmp_path):
    auth.API_KEYS = set()

    payload = _payload(
        "Access Control",
        {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 20,
            "construction_year": 2015,
        },
        confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
    )

    ctx = _ctx(env=60.0, wildland=60.0, historic=60.0)
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: ctx)

    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder-a"))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "a.db")))
    a = _run(payload)

    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (34.0522, -118.2437, "test-geocoder-b"))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "b.db")))
    b = _run(payload)

    assert a["wildfire_risk_score"] == b["wildfire_risk_score"]
    assert a["factor_breakdown"]["access_included_in_total"] is False


def test_old_rows_without_model_version_are_readable(tmp_path):
    store = AssessmentStore(str(tmp_path / "legacy.db"))
    legacy_payload = {
        "assessment_id": "legacy-1",
        "address": "Legacy Ln",
        "coordinates": {"latitude": 1.1, "longitude": 2.2},
        "risk_scores": {"wildfire_risk_score": 44.0, "insurance_readiness_score": 61.0},
        "risk_drivers": {"environmental": 45.0, "structural": 43.0, "access_exposure": 20.0},
        "mitigation_recommendations": [{"action": "clear brush", "related_factor": "fuel"}],
    }

    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.execute(
        "INSERT INTO assessments (assessment_id, created_at, payload_json, model_version) VALUES (?, datetime('now'), ?, ?)",
        ("legacy-1", json.dumps(legacy_payload), LEGACY_MODEL_VERSION),
    )
    conn.commit()
    conn.close()

    loaded = store.get("legacy-1")
    assert loaded is not None
    assert loaded.model_version == LEGACY_MODEL_VERSION
    assert loaded.assessment_id == "legacy-1"


def test_model_version_current(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=45.0, wildland=45.0, historic=40.0))
    row = _run(_payload("Version Check Rd", {"defensible_space_ft": 20}))
    assert row["model_version"] == MODEL_VERSION
    assert row["product_version"] == PRODUCT_VERSION
    assert row["api_version"] == API_VERSION
    assert row["rules_logic_version"] == RULESET_LOGIC_VERSION
    assert row["factor_schema_version"] == FACTOR_SCHEMA_VERSION
    assert row["calibration_version"] == CALIBRATION_VERSION
    assert row["model_governance"]["scoring_model_version"] == MODEL_VERSION
    loaded = app_main.store.get(row["assessment_id"])
    assert loaded is not None
    assert loaded.model_governance.scoring_model_version == MODEL_VERSION
    assert loaded.model_governance.product_version == PRODUCT_VERSION


def test_health_includes_model_governance():
    auth.API_KEYS = set()
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["product_version"] == PRODUCT_VERSION
    assert body["api_version"] == API_VERSION
    assert body["model_governance"]["product_version"] == PRODUCT_VERSION
    assert body["model_governance"]["api_version"] == API_VERSION


def test_building_footprint_lookup_success(tmp_path):
    _require_shapely()
    footprint_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-105.0002, 40.0002],
                            [-104.9998, 40.0002],
                            [-104.9998, 39.9998],
                            [-105.0002, 39.9998],
                            [-105.0002, 40.0002],
                        ]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "footprints.geojson"
    path.write_text(json.dumps(footprint_geojson))

    client = BuildingFootprintClient(path=str(path))
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is True
    assert result.footprint is not None
    assert result.centroid is not None
    assert result.confidence >= 0.9


def test_building_footprint_lookup_multisource_prefers_plausible_candidate(tmp_path):
    _require_shapely()
    large_source = tmp_path / "regional_large.geojson"
    small_source = tmp_path / "microsoft_small.geojson"
    large_source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": "large"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-105.00030, 40.00030],
                                    [-104.99970, 40.00030],
                                    [-104.99970, 39.99970],
                                    [-105.00030, 39.99970],
                                    [-105.00030, 40.00030],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    small_source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": "small"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-105.000063, 40.000063],
                                    [-104.999937, 40.000063],
                                    [-104.999937, 39.999937],
                                    [-105.000063, 39.999937],
                                    [-105.000063, 40.000063],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )

    client = BuildingFootprintClient(path=str(large_source), extra_paths=[str(small_source)])
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is True
    assert result.match_status == "matched"
    assert result.source == str(small_source)
    assert result.candidate_count == 2
    assert len(result.candidate_summaries) >= 2
    assert result.candidate_summaries[0]["source"] == str(small_source)


def test_building_footprint_lookup_no_source_fallback(tmp_path):
    _require_shapely()
    missing_path = tmp_path / "missing.geojson"
    client = BuildingFootprintClient(path=str(missing_path))
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is False
    assert any("not configured or missing" in note for note in result.assumptions)


def test_structure_ring_generation_correctness():
    _require_shapely()
    footprint = Polygon(
        [
            (-105.00005, 40.00005),
            (-104.99995, 40.00005),
            (-104.99995, 39.99995),
            (-105.00005, 39.99995),
            (-105.00005, 40.00005),
        ]
    )
    rings, assumptions = compute_structure_rings(footprint)

    assert set(rings.keys()) == {"ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "ring_100_300_ft"}
    assert not assumptions

    area_0_5 = rings["ring_0_5_ft"].area
    area_5_30 = rings["ring_5_30_ft"].area
    area_30_100 = rings["ring_30_100_ft"].area
    area_100_300 = rings["ring_100_300_ft"].area
    assert area_0_5 > 0
    assert area_5_30 > area_0_5
    assert area_30_100 > area_5_30
    assert area_100_300 > area_30_100


def test_structure_ring_summary_pipeline(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    footprint = Polygon(
        [
            (-105.00004, 40.00004),
            (-104.99996, 40.00004),
            (-104.99996, 39.99996),
            (-105.00004, 39.99996),
            (-105.00004, 40.00004),
        ]
    )

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon: BuildingFootprintResult(
            found=True,
            footprint=footprint,
            centroid=(40.0, -105.0),
            source="fixture",
            confidence=0.9,
            assumptions=[],
        ),
    )
    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda _geom, canopy_path: {
            "canopy_mean": 62.0,
            "canopy_max": 81.0,
            "coverage_pct": 58.0,
            "vegetation_density": 70.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: 50.0)

    context_blob, assumptions, sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
    )
    metrics = context_blob["ring_metrics"]

    assert context_blob["footprint_used"] is True
    assert context_blob["footprint_status"] == "used"
    for required_key in {
        "ring_0_5_ft",
        "ring_5_30_ft",
        "ring_30_100_ft",
        "ring_100_300_ft",
        "zone_0_5_ft",
        "zone_5_30_ft",
        "zone_30_100_ft",
        "zone_100_300_ft",
    }:
        assert required_key in metrics
    assert metrics["geometry_type"] == "footprint"
    assert metrics["precision_flag"] == "footprint_relative"
    assert isinstance(metrics["_meta"], dict)
    assert metrics["_meta"]["geometry_type"] == "footprint"
    assert metrics["ring_0_5_ft"]["vegetation_density"] == 60.0
    assert metrics["zone_0_5_ft"]["vegetation_density"] == 60.0
    assert metrics["ring_5_30_ft"]["canopy_mean"] == 62.0
    assert metrics["ring_0_5_ft"]["ring_area_sqft"] is not None
    assert metrics["ring_0_5_ft"]["vegetated_overlap_area_sqft"] is not None
    assert "Structure ring vegetation summaries" in sources
    assert assumptions == []


def test_ring_metrics_include_geometry_type_and_precision_for_footprint_vs_point(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    footprint = Polygon(
        [
            (-105.00004, 40.00004),
            (-104.99996, 40.00004),
            (-104.99996, 39.99996),
            (-105.00004, 39.99996),
            (-105.00004, 40.00004),
        ]
    )

    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda _geom, canopy_path: {
            "canopy_mean": 55.0,
            "canopy_max": 72.0,
            "coverage_pct": 52.0,
            "vegetation_density": 55.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: 40.0)

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=True,
            footprint=footprint,
            centroid=(40.0, -105.0),
            source="fixture",
            confidence=0.92,
            match_status="matched",
            match_method="point_in_footprint",
            matched_structure_id="fp-1",
            match_distance_m=0.0,
            candidate_count=1,
            candidate_summaries=[],
            assumptions=[],
        ),
    )
    footprint_ctx, _fp_assumptions, _fp_sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
    )

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=False,
            footprint=None,
            centroid=None,
            source="fixture",
            confidence=0.0,
            match_status="none",
            match_method="nearest_building_fallback",
            matched_structure_id=None,
            match_distance_m=35.0,
            candidate_count=2,
            candidate_summaries=[],
            assumptions=["No nearby building footprint found for this location."],
        ),
    )
    point_ctx, _pt_assumptions, _pt_sources = client._compute_structure_ring_metrics(
        40.0,
        -105.0,
        canopy_path="",
        fuel_path="",
    )

    fp_metrics = footprint_ctx["ring_metrics"]
    pt_metrics = point_ctx["ring_metrics"]
    assert fp_metrics["geometry_type"] == "footprint"
    assert fp_metrics["precision_flag"] == "footprint_relative"
    assert pt_metrics["geometry_type"] == "point"
    assert pt_metrics["precision_flag"] == "fallback_point_proxy"
    assert fp_metrics["ring_0_5_ft"]["ring_area_sqft"] != pt_metrics["ring_0_5_ft"]["ring_area_sqft"]


def test_nearby_distinct_footprints_generate_distinct_ring_metrics(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    small = Polygon(
        [
            (-105.00024, 40.00024),
            (-105.00018, 40.00024),
            (-105.00018, 40.00018),
            (-105.00024, 40.00018),
            (-105.00024, 40.00024),
        ]
    )
    large = Polygon(
        [
            (-105.00004, 40.00024),
            (-104.99982, 40.00024),
            (-104.99982, 40.00002),
            (-105.00004, 40.00002),
            (-105.00004, 40.00024),
        ]
    )

    def _lookup(lat: float, lon: float, **_kwargs):
        chosen = small if float(lon) < -105.0001 else large
        c = chosen.centroid
        return BuildingFootprintResult(
            found=True,
            footprint=chosen,
            centroid=(float(c.y), float(c.x)),
            source="fixture",
            confidence=0.94,
            match_status="matched",
            match_method="nearest_building_fallback",
            matched_structure_id="small-home" if chosen is small else "large-home",
            match_distance_m=2.0,
            candidate_count=1,
            candidate_summaries=[],
            assumptions=[],
        )

    monkeypatch.setattr(client.footprints, "get_building_footprint", _lookup)
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: None)

    def _canopy(ring_geom, canopy_path):
        # Deterministic geometry-driven synthetic canopy proxy for test discrimination.
        area_signal = max(1.0, min(95.0, ring_geom.area * 1_000_000_000.0))
        return {
            "canopy_mean": round(area_signal, 1),
            "canopy_max": round(min(100.0, area_signal + 8.0), 1),
            "coverage_pct": round(area_signal, 1),
            "vegetation_density": round(area_signal, 1),
        }

    monkeypatch.setattr(client, "_summarize_ring_canopy", _canopy)

    a, _a_assumptions, _a_sources = client._compute_structure_ring_metrics(
        40.0002,
        -105.0002,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
    )
    b, _b_assumptions, _b_sources = client._compute_structure_ring_metrics(
        40.0002,
        -104.9999,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
    )

    assert a["footprint_used"] is True
    assert b["footprint_used"] is True
    assert a["geometry_source"] == "trusted_building_footprint"
    assert b["geometry_source"] == "trusted_building_footprint"
    assert a["ring_generation_mode"] == "footprint_aware_rings"
    assert b["ring_generation_mode"] == "footprint_aware_rings"
    assert a["ring_metrics"]["geometry_type"] == "footprint"
    assert b["ring_metrics"]["geometry_type"] == "footprint"
    assert a["ring_metrics"]["precision_flag"] == "footprint_relative"
    assert b["ring_metrics"]["precision_flag"] == "footprint_relative"
    assert a["ring_metrics"]["ring_0_5_ft"]["vegetation_density"] != b["ring_metrics"]["ring_0_5_ft"]["vegetation_density"]
    assert a["ring_metrics"]["ring_0_5_ft"]["ring_area_sqft"] != b["ring_metrics"]["ring_0_5_ft"]["ring_area_sqft"]


def test_asymmetric_vegetation_produces_directional_risk_differentiation(monkeypatch):
    _require_shapely()
    client = WildfireDataClient()
    origin_lat, origin_lon = 40.0, -105.0
    footprint = Polygon(
        [
            (-105.00008, 40.00006),
            (-104.99992, 40.00006),
            (-104.99992, 39.99994),
            (-105.00008, 39.99994),
            (-105.00008, 40.00006),
        ]
    )

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon, **_kwargs: BuildingFootprintResult(
            found=True,
            footprint=footprint,
            centroid=(origin_lat, origin_lon),
            source="fixture",
            confidence=0.94,
            match_status="matched",
            match_method="point_in_footprint",
            matched_structure_id="home-1",
            match_distance_m=0.0,
            candidate_count=1,
            candidate_summaries=[],
            assumptions=[],
        ),
    )
    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda _geom, canopy_path: {
            "canopy_mean": 50.0,
            "canopy_max": 70.0,
            "coverage_pct": 45.0,
            "vegetation_density": 50.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom, fuel_path: 42.0)

    def _asymmetric_veg(*, canopy_path, fuel_path, lat, lon):
        eps = 0.00001
        if lon > origin_lon + eps:
            return 92.0
        if lon < origin_lon - eps:
            return 24.0
        if lat > origin_lat + eps:
            return 48.0
        if lat < origin_lat - eps:
            return 34.0
        return 50.0

    def _asymmetric_slope(path: str, lat: float, lon: float):
        if path != "slope.tif":
            return None
        eps = 0.00001
        if lon > origin_lon + eps:
            return 32.0
        if lon < origin_lon - eps:
            return 8.0
        if lat > origin_lat + eps:
            return 18.0
        if lat < origin_lat - eps:
            return 12.0
        return 15.0

    monkeypatch.setattr(client, "_sample_combined_vegetation_index", _asymmetric_veg)
    monkeypatch.setattr(client, "_sample_raster_point", _asymmetric_slope)

    context_blob, _assumptions, _sources = client._compute_structure_ring_metrics(
        origin_lat,
        origin_lon,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
    )

    directional_risk = context_blob.get("directional_risk") or {}
    sector_scores = directional_risk.get("sector_scores") or {}
    sector_details = context_blob.get("vegetation_directional_sectors") or {}

    assert directional_risk.get("max_risk_direction") == "east"
    assert float(sector_scores.get("east") or 0.0) > float(sector_scores.get("west") or 0.0)
    assert float(sector_scores.get("east") or 0.0) > float(sector_scores.get("south") or 0.0)
    assert sector_details.get("east", {}).get("slope_deg") is not None
    assert sector_details.get("east", {}).get("uphill_fuel_concentration") is not None


def test_context_collect_fallback_when_footprint_unavailable(monkeypatch):
    client = WildfireDataClient()

    monkeypatch.setattr(
        client.footprints,
        "get_building_footprint",
        lambda _lat, _lon: BuildingFootprintResult(
            found=False,
            footprint=None,
            centroid=None,
            source="missing_fixture",
            confidence=0.0,
            assumptions=["No nearby building footprint found for this location."],
        ),
    )

    ctx = client.collect_context(39.7392, -104.9903)
    assert ctx is not None
    assert ctx.property_level_context.get("footprint_used") is False
    assert ctx.property_level_context.get("footprint_status") in {"not_found", "provider_unavailable"}
    assert "ring_metrics" in ctx.property_level_context


def test_whp_adapter_populates_burn_and_hazard_when_direct_layers_missing(monkeypatch):
    _require_geo_runtime_deps()
    client = WildfireDataClient()
    runtime_paths = {k: "" for k in client.base_paths.keys()}
    runtime_paths["whp"] = "whp.tif"

    monkeypatch.setattr(
        client,
        "_resolve_runtime_layer_paths",
        lambda _lat, _lon: (
            runtime_paths,
            {"region_status": "legacy_fallback", "region_id": None, "region_display_name": None, "manifest_path": None},
            [],
            [],
        ),
    )
    monkeypatch.setattr(client, "_sample_layer_value", lambda _path, _lat, _lon: (None, "missing"))
    monkeypatch.setattr(
        client.whp_adapter,
        "sample",
        lambda **_kwargs: WHPObservation(
            status="ok",
            raw_value=3.5,
            hazard_class="high",
            burn_probability_index=78.0,
            hazard_severity_index=74.0,
        ),
    )

    ctx = client.collect_context(39.7392, -104.9903)
    assert ctx.burn_probability_index == 78.0
    assert ctx.hazard_severity_index == 74.0
    assert ctx.environmental_layer_status["burn_probability"] == "ok"
    assert ctx.environmental_layer_status["hazard"] == "ok"
    assert ctx.hazard_context.get("status") == "observed"
    assert "WHP" in str(ctx.hazard_context.get("source"))


def test_gridmet_dryness_populates_moisture_index(monkeypatch):
    _require_geo_runtime_deps()
    client = WildfireDataClient()
    runtime_paths = {k: "" for k in client.base_paths.keys()}
    runtime_paths["gridmet_dryness"] = "gridmet.tif"

    monkeypatch.setattr(
        client,
        "_resolve_runtime_layer_paths",
        lambda _lat, _lon: (
            runtime_paths,
            {"region_status": "legacy_fallback", "region_id": None, "region_display_name": None, "manifest_path": None},
            [],
            [],
        ),
    )
    monkeypatch.setattr(client, "_sample_layer_value", lambda _path, _lat, _lon: (None, "missing"))
    monkeypatch.setattr(
        client.gridmet_adapter,
        "sample_dryness",
        lambda **_kwargs: GridMETDrynessObservation(
            status="ok",
            raw_value=62.5,
            dryness_index=66.0,
        ),
    )

    ctx = client.collect_context(39.7392, -104.9903)
    assert ctx.moisture_index == 66.0
    assert ctx.moisture_context.get("status") == "observed"
    assert "gridMET" in str(ctx.moisture_context.get("source"))
    assert not any("Moisture/fuel dryness context unavailable" in note for note in ctx.assumptions)


def test_mtbs_summary_populates_historical_fire_context(monkeypatch):
    _require_geo_runtime_deps()
    client = WildfireDataClient()
    runtime_paths = {k: "" for k in client.base_paths.keys()}
    runtime_paths["perimeters"] = "mtbs_perimeters.geojson"
    runtime_paths["mtbs_severity"] = "mtbs_severity.tif"

    monkeypatch.setattr(
        client,
        "_resolve_runtime_layer_paths",
        lambda _lat, _lon: (
            runtime_paths,
            {"region_status": "legacy_fallback", "region_id": None, "region_display_name": None, "manifest_path": None},
            [],
            [],
        ),
    )
    monkeypatch.setattr(client, "_sample_layer_value", lambda _path, _lat, _lon: (None, "missing"))
    monkeypatch.setattr(
        client.mtbs_adapter,
        "summarize",
        lambda **_kwargs: MTBSSummary(
            status="ok",
            nearest_perimeter_km=1.3,
            intersects_prior_burn=False,
            nearby_high_severity=True,
            fire_history_index=57.0,
        ),
    )

    ctx = client.collect_context(39.7392, -104.9903)
    assert ctx.historic_fire_index == 57.0
    assert ctx.historical_fire_context.get("status") == "ok"
    assert ctx.environmental_layer_status["fire_history"] == "ok"


def test_osm_access_exposure_replaces_synthetic_placeholder(monkeypatch):
    context = _ctx(env=58.0, wildland=62.0, historic=54.0)
    context.access_exposure_index = 41.0
    context.access_context = {
        "status": "ok",
        "source": "OpenStreetMap road network",
        "distance_to_nearest_road_m": 21.4,
        "road_segments_within_300m": 8,
        "intersections_within_300m": 4,
        "dead_end_indicator": False,
    }
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=25)
    risk = app_main.risk_engine.score(attrs, 39.7392, -104.9903, context)

    assert risk.drivers.access_exposure == 41.0
    assert risk.access_provisional is False
    assert "synthetic placeholder" not in " ".join(risk.assumptions).lower()


def _make_region_sources(tmp_path: Path) -> dict[str, str]:
    _require_region_prep_deps()
    rasterio_mod = prep_region_module.rasterio
    np_mod = prep_region_module.np
    assert rasterio_mod is not None
    assert np_mod is not None

    files = {
        "dem": tmp_path / "src_dem.tif",
        "slope": tmp_path / "src_slope.tif",
        "fuel": tmp_path / "src_fuel.tif",
        "canopy": tmp_path / "src_canopy.tif",
        "fire_perimeters": tmp_path / "src_fire_perimeters.geojson",
        "building_footprints": tmp_path / "src_building_footprints.geojson",
        "burn_probability": tmp_path / "src_burn_probability.tif",
        "wildfire_hazard": tmp_path / "src_hazard.tif",
    }
    files["fire_perimeters"].write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-105.8, 39.7], [-105.0, 39.7], [-105.0, 40.2], [-105.8, 40.2], [-105.8, 39.7]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    files["building_footprints"].write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-105.25, 39.95], [-105.20, 39.95], [-105.20, 40.00], [-105.25, 40.00], [-105.25, 39.95]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    transform = rasterio_mod.transform.from_origin(-106.5, 40.7, 0.01, 0.01)
    for key in ["dem", "slope", "fuel", "canopy", "burn_probability", "wildfire_hazard"]:
        data = np_mod.full((220, 220), 50.0, dtype="float32")
        with rasterio_mod.open(
            files[key],
            "w",
            driver="GTiff",
            width=220,
            height=220,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as ds:
            ds.write(data, 1)
    return {k: str(v) for k, v in files.items()}


def test_prepare_region_manifest_creation_and_discovery(tmp_path):
    _require_region_prep_deps()
    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    manifest = prepare_region_layers(
        region_id="boulder_county_co",
        display_name="Boulder County, CO",
        bounds={"min_lon": -106.0, "min_lat": 39.5, "max_lon": -104.5, "max_lat": 40.5},
        layer_sources=sources,
        region_data_dir=region_root,
    )
    manifest_path = region_root / "boulder_county_co" / "manifest.json"
    assert manifest_path.exists()
    assert manifest["files"]["dem"] == "dem.tif"
    assert manifest["files"]["building_footprints"] == "building_footprints.geojson"

    loaded = load_region_manifest("boulder_county_co", base_dir=str(region_root))
    assert loaded is not None
    assert loaded["region_id"] == "boulder_county_co"

    listed = list_prepared_regions(base_dir=str(region_root))
    assert len(listed) == 1
    assert listed[0]["display_name"] == "Boulder County, CO"

    found = find_region_for_point(40.0, -105.2, base_dir=str(region_root))
    assert found is not None
    assert found["region_id"] == "boulder_county_co"


def test_runtime_resolves_prepared_region_files(tmp_path):
    _require_region_prep_deps()
    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    prepare_region_layers(
        region_id="marin_county_ca",
        display_name="Marin County, CA",
        bounds={"min_lon": -123.0, "min_lat": 37.7, "max_lon": -122.2, "max_lat": 38.3},
        layer_sources=sources,
        region_data_dir=region_root,
    )

    client = WildfireDataClient()
    client.region_data_dir = str(region_root)
    client.use_prepared_regions = True
    client.allow_legacy_layer_fallback = False
    client.base_paths = {k: "" for k in client.base_paths.keys()}
    client.paths = dict(client.base_paths)

    paths, region_context, assumptions, sources_used = client._resolve_runtime_layer_paths(38.0, -122.7)
    assert region_context["region_status"] == "prepared"
    assert region_context["region_id"] == "marin_county_ca"
    assert paths["dem"].endswith("dem.tif")
    assert paths["fuel"].endswith("fuel.tif")
    assert paths["perimeters"].endswith("fire_perimeters.geojson")
    assert not any("region not prepared" in a.lower() for a in assumptions)
    assert any("Prepared region" in src for src in sources_used)


def test_runtime_prefers_smallest_covering_region_for_point(tmp_path):
    _require_region_prep_deps()
    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    prepare_region_layers(
        region_id="large_region",
        display_name="Large Region",
        bounds={"min_lon": -112.0, "min_lat": 45.0, "max_lon": -110.0, "max_lat": 47.0},
        layer_sources=sources,
        region_data_dir=region_root,
    )
    prepare_region_layers(
        region_id="small_region",
        display_name="Small Region",
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        layer_sources=sources,
        region_data_dir=region_root,
    )

    client = WildfireDataClient()
    client.region_data_dir = str(region_root)
    client.use_prepared_regions = True
    client.allow_legacy_layer_fallback = False
    client.base_paths = {k: "" for k in client.base_paths.keys()}
    client.paths = dict(client.base_paths)

    paths, region_context, assumptions, sources_used = client._resolve_runtime_layer_paths(45.67, -111.04)
    assert region_context["region_status"] == "prepared"
    assert region_context["region_id"] == "small_region"
    assert paths["dem"].endswith("dem.tif")
    assert not any("region not prepared" in a.lower() for a in assumptions)
    assert any("Prepared region" in src for src in sources_used)


def test_missoula_prepared_region_assessment_smoke(monkeypatch, tmp_path):
    _require_region_prep_deps()
    _require_geo_runtime_deps()
    auth.API_KEYS = set()

    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    required_only = {
        key: value
        for key, value in sources.items()
        if key in {"dem", "slope", "fuel", "canopy", "fire_perimeters", "building_footprints"}
    }
    prepare_region_layers(
        region_id="missoula_pilot",
        display_name="Missoula Pilot",
        bounds={"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0},
        layer_sources=required_only,
        region_data_dir=region_root,
    )

    runtime_client = WildfireDataClient()
    runtime_client.region_data_dir = str(region_root)
    runtime_client.use_prepared_regions = True
    runtime_client.allow_legacy_layer_fallback = False
    runtime_client.base_paths = {k: "" for k in runtime_client.base_paths.keys()}
    runtime_client.paths = dict(runtime_client.base_paths)

    monkeypatch.setattr(app_main, "wildfire_data", runtime_client)
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (46.8721, -113.9940, "test-geocoder"))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "missoula_smoke.db")))

    response = client.post(
        "/risk/assess",
        json=_payload(
            "201 W Front St, Missoula, MT 59802",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 25,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        ),
        headers=_headers(role="admin"),
    )
    assert response.status_code == 200
    body = response.json()

    property_ctx = body["property_level_context"]
    assert property_ctx["region_status"] == "prepared"
    assert property_ctx["region_id"] == "missoula_pilot"
    assert str(property_ctx["region_manifest_path"]).endswith("missoula_pilot/manifest.json")
    assert body["region_resolution"]["coverage_available"] is True
    assert body["region_resolution"]["resolved_region_id"] == "missoula_pilot"
    assert body["region_resolution"]["reason"] == "prepared_region_found"

    assert body["wildfire_risk_score_available"] is True
    assert body["insurance_readiness_score_available"] is True
    assert isinstance(body["wildfire_risk_score"], (int, float))
    assert isinstance(body["insurance_readiness_score"], (int, float))
    assert body["assessment_status"] in {"fully_scored", "partially_scored"}

    coverage_summary = body["coverage_summary"]
    assert coverage_summary["critical_missing_layers"] == []
    assert coverage_summary["not_configured_count"] >= 1

    layer_rows = body["layer_coverage_audit"]
    required_layers = {"dem", "slope", "fuel", "canopy", "fire_perimeters", "building_footprints"}
    for layer_key in required_layers:
        row = next(r for r in layer_rows if r["layer_key"] == layer_key)
        assert row["coverage_status"] in {"observed", "partial"}

    assert any(r["layer_key"] == "gridmet_dryness" and r["coverage_status"] == "not_configured" for r in layer_rows)
    assert any("optional enrichment layers not configured" in note.lower() for note in body["scoring_notes"])


def test_runtime_region_not_prepared_returns_insufficient_data(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "region_not_prepared.db")))

    runtime_client = WildfireDataClient()
    runtime_client.region_data_dir = str(tmp_path / "empty_regions")
    runtime_client.use_prepared_regions = True
    runtime_client.allow_legacy_layer_fallback = False
    runtime_client.base_paths = {k: "" for k in runtime_client.base_paths.keys()}
    runtime_client.paths = dict(runtime_client.base_paths)
    monkeypatch.setattr(app_main, "wildfire_data", runtime_client)

    response = client.post(
        "/risk/assess",
        json=_payload("No Prepared Region Address", {"roof_type": "class a"}),
        headers=_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["assessment_status"] == "insufficient_data"
    assert body["property_level_context"]["region_status"] == "region_not_prepared"
    assert body["region_resolution"]["coverage_available"] is False
    assert body["region_resolution"]["resolved_region_id"] is None
    assert body["region_resolution"]["reason"] == "no_prepared_region_for_location"
    assert body["region_resolution"]["recommended_action"]
    assert any("region not prepared" in blocker.lower() for blocker in body["assessment_blockers"])
    assert any("region not prepared" in note.lower() for note in body["scoring_notes"])


def test_legacy_layer_env_fallback_when_region_missing(tmp_path):
    legacy = _make_region_sources(tmp_path)
    client = WildfireDataClient()
    client.region_data_dir = str(tmp_path / "missing_regions")
    client.use_prepared_regions = True
    client.allow_legacy_layer_fallback = True
    client.base_paths = {
        "burn_prob": legacy["burn_probability"],
        "hazard": legacy["wildfire_hazard"],
        "slope": legacy["slope"],
        "aspect": "",
        "dem": legacy["dem"],
        "fuel": legacy["fuel"],
        "canopy": legacy["canopy"],
        "moisture": "",
        "perimeters": legacy["fire_perimeters"],
        "footprints": legacy["building_footprints"],
    }
    client.paths = dict(client.base_paths)

    paths, region_context, assumptions, _sources = client._resolve_runtime_layer_paths(40.1, -105.2)
    assert region_context["region_status"] == "legacy_fallback"
    assert paths["dem"] == legacy["dem"]
    assert paths["canopy"] == legacy["canopy"]
    assert any("legacy direct layer paths" in a.lower() for a in assumptions)


def test_ring_metrics_influence_risk_and_mitigation():
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=20,
        construction_year=2015,
    )

    low_ring = {
        "ring_0_5_ft": {"vegetation_density": 15.0},
        "ring_5_30_ft": {"vegetation_density": 25.0},
        "ring_30_100_ft": {"vegetation_density": 35.0},
    }
    high_ring = {
        "ring_0_5_ft": {"vegetation_density": 85.0},
        "ring_5_30_ft": {"vegetation_density": 80.0},
        "ring_30_100_ft": {"vegetation_density": 70.0},
    }

    low_ctx = _ctx(env=55.0, wildland=55.0, historic=50.0, ring_metrics=low_ring)
    high_ctx = _ctx(env=55.0, wildland=55.0, historic=50.0, ring_metrics=high_ring)

    low_risk = app_main.risk_engine.score(attrs, 39.7392, -104.9903, low_ctx)
    high_risk = app_main.risk_engine.score(attrs, 39.7392, -104.9903, high_ctx)

    assert high_risk.submodel_scores["flame_contact_risk"].score > low_risk.submodel_scores["flame_contact_risk"].score
    assert high_risk.submodel_scores["defensible_space_risk"].score > low_risk.submodel_scores["defensible_space_risk"].score

    high_plan = build_mitigation_plan(
        attrs,
        high_ctx,
        {name: item.score for name, item in high_risk.submodel_scores.items()},
        readiness_blockers=[],
    )
    titles = [rec.title.lower() for rec in high_plan]
    assert any("0-5 ft zone" in title for title in titles)
    assert any("5-30 ft zone" in title for title in titles)


def test_nearby_properties_with_distinct_edge_vegetation_get_distinct_vegetation_subscores():
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=24,
        construction_year=2015,
    )
    shared_ring = {
        "ring_0_5_ft": {"vegetation_density": 40.0},
        "ring_5_30_ft": {"vegetation_density": 46.0},
        "ring_30_100_ft": {"vegetation_density": 49.0},
    }

    low_edge_ctx = _ctx(env=56.0, wildland=54.0, historic=48.0, ring_metrics=shared_ring)
    low_edge_ctx.property_level_context.update(
        {
            "near_structure_vegetation_0_5_pct": 12.0,
            "near_structure_vegetation_5_30_pct": 24.0,
            "vegetation_edge_directional_concentration_pct": 18.0,
            "canopy_dense_fuel_asymmetry_pct": 14.0,
            "nearest_continuous_vegetation_distance_ft": 120.0,
            "vegetation_directional_precision": "footprint_boundary",
            "vegetation_directional_precision_score": 0.92,
        }
    )

    high_edge_ctx = _ctx(env=56.0, wildland=54.0, historic=48.0, ring_metrics=shared_ring)
    high_edge_ctx.property_level_context.update(
        {
            "near_structure_vegetation_0_5_pct": 92.0,
            "near_structure_vegetation_5_30_pct": 88.0,
            "vegetation_edge_directional_concentration_pct": 94.0,
            "canopy_dense_fuel_asymmetry_pct": 90.0,
            "nearest_continuous_vegetation_distance_ft": 4.0,
            "vegetation_directional_precision": "footprint_boundary",
            "vegetation_directional_precision_score": 0.92,
        }
    )

    # Nearby homes should still diverge materially when immediate vegetation contexts differ.
    low_risk = app_main.risk_engine.score(attrs, 39.73920, -104.99030, low_edge_ctx)
    high_risk = app_main.risk_engine.score(attrs, 39.73945, -104.99010, high_edge_ctx)

    assert high_risk.submodel_scores["flame_contact_risk"].score - low_risk.submodel_scores["flame_contact_risk"].score >= 12.0
    assert high_risk.submodel_scores["defensible_space_risk"].score - low_risk.submodel_scores["defensible_space_risk"].score >= 10.0
    assert high_risk.submodel_scores["vegetation_intensity_risk"].score - low_risk.submodel_scores["vegetation_intensity_risk"].score >= 9.0
    assert (
        high_risk.submodel_scores["flame_contact_risk"].key_inputs["vegetation_directional_precision"]
        == "footprint_boundary"
    )


def test_vegetation_directional_features_fallback_to_point_proxy_when_geometry_missing():
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=22,
        construction_year=2015,
    )
    point_ctx = _ctx(env=56.0, wildland=54.0, historic=48.0, ring_metrics={})
    point_ctx.property_level_context.update(
        {
            "footprint_used": False,
            "fallback_mode": "point_based",
            "near_structure_vegetation_0_5_pct": 68.0,
            "near_structure_vegetation_5_30_pct": 60.0,
            "vegetation_edge_directional_concentration_pct": 72.0,
            "canopy_dense_fuel_asymmetry_pct": 66.0,
            "nearest_continuous_vegetation_distance_ft": 16.0,
            "vegetation_directional_precision": "point_proxy",
            "vegetation_directional_precision_score": 0.42,
        }
    )

    risk = app_main.risk_engine.score(attrs, 39.7392, -104.9903, point_ctx)
    flame = risk.submodel_scores["flame_contact_risk"]
    defensible = risk.submodel_scores["defensible_space_risk"]

    assert isinstance(risk.total_score, float)
    assert flame.key_inputs["vegetation_directional_precision"] == "point_proxy"
    assert defensible.key_inputs["vegetation_directional_precision"] == "point_proxy"
    assert any("point-proxy sampling" in assumption.lower() for assumption in flame.assumptions)


def _headers(role: str = "admin", org: str = "default_org", user: str = "test_user") -> dict:
    return {
        "X-User-Role": role,
        "X-Organization-Id": org,
        "X-User-Id": user,
    }


def test_organization_scope_enforced(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=52.0, wildland=57.0, historic=50.0))

    created = client.post(
        "/risk/assess",
        json=_payload("Org Scope Test", {"defensible_space_ft": 22}),
        headers=_headers(role="admin", org="org_alpha"),
    )
    assert created.status_code == 200
    assessment_id = created.json()["assessment_id"]

    denied = client.get(
        f"/report/{assessment_id}",
        headers=_headers(role="underwriter", org="org_beta"),
    )
    assert denied.status_code == 403


def test_role_permissions_for_review_status(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=58.0, wildland=62.0, historic=54.0))
    assessed = _run(_payload("Role Test Ave", {"defensible_space_ft": 16}))

    viewer_attempt = client.put(
        f"/assessments/{assessed['assessment_id']}/review-status",
        json={"review_status": "reviewed"},
        headers=_headers(role="viewer"),
    )
    assert viewer_attempt.status_code == 403

    underwriter_attempt = client.put(
        f"/assessments/{assessed['assessment_id']}/review-status",
        json={"review_status": "reviewed"},
        headers=_headers(role="underwriter"),
    )
    assert underwriter_attempt.status_code == 200
    assert underwriter_attempt.json()["review_status"] == "reviewed"


def test_ruleset_selection_and_output(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=88.0, wildland=90.0, historic=86.0))

    res = client.post(
        "/risk/assess",
        json={
            **_payload(
                "Strict Ruleset Ct",
                {
                    "roof_type": "wood",
                    "vent_type": "standard",
                    "defensible_space_ft": 4,
                },
            ),
            "ruleset_id": "strict_carrier_demo",
        },
        headers=_headers(role="underwriter"),
    )
    assert res.status_code == 200
    body = res.json()

    assert body["ruleset_id"] == "strict_carrier_demo"
    assert body["ruleset_name"] == "Strict Carrier Demo"
    assert any("strict_carrier_demo" in b for b in body["readiness_blockers"])


def test_portfolio_job_lifecycle_and_results(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=52.0, historic=44.0))
    headers = _headers(role="broker")

    create = client.post(
        "/portfolio/jobs",
        json={
            "portfolio_name": "Lifecycle",
            "process_immediately": True,
            "items": [
                {
                    "row_id": "1",
                    "address": "1 Job St",
                    "attributes": {"defensible_space_ft": 18},
                    "confirmed_fields": [],
                    "audience": "insurer",
                    "tags": [],
                },
                {
                    "row_id": "2",
                    "address": "",
                    "attributes": {"defensible_space_ft": 10},
                    "confirmed_fields": [],
                    "audience": "insurer",
                    "tags": [],
                },
            ],
        },
        headers=headers,
    )
    assert create.status_code == 200
    job = create.json()
    assert job["status"] in {"completed", "partial", "failed", "running", "queued"}
    job_id = job["job_id"]

    status = client.get(f"/portfolio/jobs/{job_id}", headers=headers)
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id

    results = client.get(f"/portfolio/jobs/{job_id}/results", headers=headers)
    assert results.status_code == 200
    rows = results.json()["results"]
    assert len(rows) == 2
    assert any(r["status"] == "failed" for r in rows)


def test_csv_import_validation_and_export(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=48.0, wildland=49.0, historic=42.0))
    headers = _headers(role="broker")

    csv_blob = (
        "address,roof_type,vent_type,defensible_space_ft,confirmed_fields,tags,audience\n"
        "123 Csv Way,class a,ember-resistant,30,\"roof_type,vent_type,defensible_space_ft\",\"renewal,co\",insurer\n"
        ",wood,standard,8,\"roof_type,vent_type\",bad,insurer\n"
    )
    imported = client.post(
        "/portfolio/import/csv",
        json={
            "csv_text": csv_blob,
            "portfolio_name": "CSV Demo",
            "process_immediately": True,
        },
        headers=headers,
    )
    assert imported.status_code == 200
    body = imported.json()
    assert body["accepted_count"] == 1
    assert body["rejected_count"] == 1
    assert len(body["validation_errors"]) == 1

    job_id = body["job"]["job_id"]
    exported = client.get(f"/portfolio/jobs/{job_id}/export/csv", headers=headers)
    assert exported.status_code == 200
    text = exported.text
    assert "address,assessment_id,status,error,wildfire_risk_score" in text
    assert "123 Csv Way" in text


def test_assignment_and_workflow_transitions(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=57.0, wildland=66.0, historic=55.0))
    assessed = _run(_payload("Workflow Test Rd", {"defensible_space_ft": 14}))
    headers = _headers(role="underwriter")

    assigned = client.post(
        f"/assessment/{assessed['assessment_id']}/assign",
        json={"assigned_reviewer": "uw_17", "assigned_role": "underwriter"},
        headers=headers,
    )
    assert assigned.status_code == 200
    assert assigned.json()["assigned_reviewer"] == "uw_17"

    to_triaged = client.post(
        f"/assessment/{assessed['assessment_id']}/workflow",
        json={"workflow_state": "triaged"},
        headers=headers,
    )
    assert to_triaged.status_code == 200
    assert to_triaged.json()["workflow_state"] == "triaged"

    to_approved = client.post(
        f"/assessment/{assessed['assessment_id']}/workflow",
        json={"workflow_state": "approved"},
        headers=headers,
    )
    assert to_approved.status_code == 400


def test_audit_events_and_admin_summary(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=62.0, wildland=68.0, historic=61.0))
    headers = _headers(role="admin", org="default_org", user="ops_admin")

    assessed = client.post(
        "/risk/assess",
        json=_payload("Audit Row", {"defensible_space_ft": 12}),
        headers=headers,
    )
    assert assessed.status_code == 200

    summary = client.get("/admin/summary?recent_days=365", headers=headers)
    assert summary.status_code == 200
    sb = summary.json()
    assert "assessments_created_recently" in sb
    assert "jobs_summary" in sb

    events = client.get("/audit/events?limit=50", headers=headers)
    assert events.status_code == 200
    rows = events.json()
    assert len(rows) >= 1
    assert any(e["action"] == "assessment_created" for e in rows)


def test_environmental_data_completeness_scoring_helper():
    full = _ctx(env=50.0, wildland=50.0, historic=50.0)
    assert compute_environmental_data_completeness(full) == 100.0

    partial = _ctx(
        env=50.0,
        wildland=50.0,
        historic=50.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "missing",
            "canopy": "ok",
            "fire_history": "missing",
        },
    )
    assert compute_environmental_data_completeness(partial) == 50.0


def test_collect_context_missing_layers_does_not_silently_default_to_neutral():
    client = WildfireDataClient()
    client.paths = {k: "" for k in client.paths.keys()}
    ctx = client.collect_context(40.0, -105.0)

    assert ctx.burn_probability_index is None
    assert ctx.hazard_severity_index is None
    assert ctx.slope_index is None
    assert ctx.fuel_index is None
    assert ctx.canopy_index is None
    assert ctx.historic_fire_index is None
    assert ctx.wildland_distance_index is None
    assert ctx.environmental_layer_status["burn_probability"] in {"missing", "error"}
    assert ctx.environmental_layer_status["fuel"] in {"missing", "error"}


def test_scoring_notes_include_missing_layers_and_provisional_access(monkeypatch, tmp_path):
    degraded = _ctx(
        env=58.0,
        wildland=60.0,
        historic=55.0,
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "missing",
            "canopy": "ok",
            "fire_history": "error",
        },
        ring_metrics={},
    )
    degraded.property_level_context = {
        "footprint_used": False,
        "footprint_status": "provider_unavailable",
        "fallback_mode": "point_based",
        "ring_metrics": None,
    }

    _setup(monkeypatch, tmp_path, degraded)
    assessed = _run(_payload("Notes Transparency Ave", {"defensible_space_ft": 12}))

    notes = " ".join(assessed["scoring_notes"]).lower()
    assert "burn probability layer" in notes
    assert "fuel layer" in notes
    assert "building footprint not found" in notes
    assert "access exposure" in notes
    assert "advisory" in notes


def test_coverage_scores_and_provenance_reflect_missing_vs_user_inputs(monkeypatch, tmp_path):
    degraded = _ctx(
        env=58.0,
        wildland=60.0,
        historic=55.0,
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "missing",
            "canopy": "missing",
            "fire_history": "ok",
        },
        ring_metrics={},
    )
    degraded.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": None,
    }
    _setup(monkeypatch, tmp_path, degraded)

    assessed = _run(
        _payload(
            "Coverage Check Ln",
            {"roof_type": "class a", "vent_type": "ember-resistant"},
            confirmed=["roof_type", "vent_type"],
        )
    )
    assert assessed["missing_data_share"] > 0
    assert assessed["direct_data_coverage_score"] < 100
    assert assessed["inferred_data_coverage_score"] > 0
    assert assessed["data_provenance"]["property_inputs_used"]["roof_type"]["source_type"] == "user_provided"
    assert assessed["data_provenance"]["property_inputs_used"]["zone_0_5_ft"]["source_type"] == "missing"
    assert "summary" in assessed["data_provenance"]
    assert "stale_data_share" in assessed["data_provenance"]["summary"]
    assert "heuristic_input_count" in assessed["data_provenance"]["summary"]
    assert "current_input_count" in assessed["data_provenance"]["summary"]
    burn_meta = assessed["input_source_metadata"]["burn_probability"]
    assert burn_meta["provider_status"] in {"ok", "missing", "error"}
    assert burn_meta["freshness_status"] in {"current", "aging", "stale", "unknown"}
    assert isinstance(burn_meta["used_in_scoring"], bool)
    assert 0.0 <= float(burn_meta["confidence_weight"]) <= 1.0


def test_access_provenance_marks_observed_open_data_when_available(monkeypatch, tmp_path):
    context = _ctx(env=54.0, wildland=58.0, historic=49.0)
    context.access_exposure_index = 36.0
    context.access_context = {
        "status": "ok",
        "source": "OpenStreetMap road network",
        "distance_to_nearest_road_m": 18.2,
        "road_segments_within_300m": 6,
        "intersections_within_300m": 3,
        "dead_end_indicator": False,
    }
    _setup(monkeypatch, tmp_path, context)
    assessed = _run(_payload("Access Provenance Way", {"defensible_space_ft": 18}))

    access_meta = assessed["input_source_metadata"]["access_exposure"]
    assert access_meta["source_type"] == "observed"
    assert "OpenStreetMap" in access_meta["source_name"]
    assert access_meta["provider_status"] == "ok"


def test_confidence_reduces_when_ring_metrics_unavailable(monkeypatch, tmp_path):
    attrs = {
        "roof_type": "class a",
        "vent_type": "ember-resistant",
        "defensible_space_ft": 25,
        "construction_year": 2018,
    }
    with_rings = _ctx(
        env=45.0,
        wildland=45.0,
        historic=35.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 20.0},
            "zone_5_30_ft": {"vegetation_density": 25.0},
            "zone_30_100_ft": {"vegetation_density": 30.0},
        },
    )
    no_rings = _ctx(env=45.0, wildland=45.0, historic=35.0, ring_metrics={})

    _setup(monkeypatch, tmp_path, with_rings)
    assessed_with = _run(_payload("With Ring Metrics", attrs))

    _setup(monkeypatch, tmp_path, no_rings)
    assessed_without = _run(_payload("Without Ring Metrics", attrs))

    assert assessed_with["confidence_score"] >= assessed_without["confidence_score"]
    assert assessed_without["property_level_context"]["fallback_mode"] == "point_based"
    assert assessed_without["evidence_quality_summary"]["use_restriction"] in {"review_required", "screening_only", "consumer_estimate"}


def test_nearby_properties_collapse_toward_similar_scores_when_geometry_and_layers_are_missing(monkeypatch, tmp_path):
    auth.API_KEYS = set()

    def _geocode(address: str):
        if "100 " in address:
            return 46.87210, -113.99400, "test-geocoder"
        return 46.87255, -113.99360, "test-geocoder"

    def _context_for(lat: float) -> WildfireContext:
        # Mild environmental differences between nearby homes are preserved,
        # but missing geometry + partial layers should collapse differentiation.
        env = 58.0 if lat < 46.87230 else 62.0
        slope = 54.0 if lat < 46.87230 else 60.0
        ctx = _ctx(env=env, wildland=48.0, historic=42.0, ring_metrics={})
        ctx.hazard_severity_index = None
        ctx.wildfire_hazard = None
        ctx.moisture_index = None
        ctx.environmental_layer_status = {
            "burn_probability": "ok",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        }
        ctx.property_level_context.update(
            {
                "footprint_used": False,
                "footprint_status": "not_found",
                "fallback_mode": "point_based",
                "ring_metrics": {},
                "parcel_id": None,
                "parcel_geometry": None,
            }
        )
        return ctx

    monkeypatch.setattr(app_main.geocoder, "geocode", _geocode)
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda lat, lon: _context_for(float(lat)))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "nearby_similarity.db")))

    a = _run(_payload("100 Similarity Lane, Missoula, MT 59802", {}, confirmed=[]))
    b = _run(_payload("101 Similarity Lane, Missoula, MT 59802", {}, confirmed=[]))

    assert a["property_level_context"]["fallback_mode"] == "point_based"
    assert b["property_level_context"]["fallback_mode"] == "point_based"
    assert a["assessment_specificity_tier"] == "regional_estimate"
    assert b["assessment_specificity_tier"] == "regional_estimate"
    assert a["home_ignition_vulnerability_score"] is None
    assert b["home_ignition_vulnerability_score"] is None
    assert float(a["fallback_weight_fraction"] or 0.0) >= 0.95
    assert float(b["fallback_weight_fraction"] or 0.0) >= 0.95

    wildfire_delta = abs(float(a["wildfire_risk_score"] or 0.0) - float(b["wildfire_risk_score"] or 0.0))
    assert wildfire_delta <= 3.0

    canonical = [
        "vegetation_intensity_risk",
        "fuel_proximity_risk",
        "slope_topography_risk",
        "ember_exposure_risk",
        "flame_contact_risk",
        "historic_fire_risk",
        "structure_vulnerability_risk",
        "defensible_space_risk",
    ]
    def _effective_weight(payload: dict, key: str) -> float:
        return float(((payload.get("weighted_contributions") or {}).get(key) or {}).get("effective_weight") or 0.0)

    total_effective_a = sum(_effective_weight(a, key) for key in canonical)
    shared_effective_a = _effective_weight(a, "slope_topography_risk") + _effective_weight(a, "historic_fire_risk")
    assert total_effective_a > 0.0
    assert (shared_effective_a / total_effective_a) >= 0.90


def test_observed_property_data_quality_beats_fallback_only(monkeypatch, tmp_path):
    observed_ctx = _ctx(
        env=46.0,
        wildland=43.0,
        historic=34.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 22.0},
            "zone_5_30_ft": {"vegetation_density": 30.0},
            "zone_30_100_ft": {"vegetation_density": 38.0},
        },
    )
    _setup(monkeypatch, tmp_path, observed_ctx)
    observed = _run(
        _payload(
            "Observed Data Ln",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 35,
                "construction_year": 2016,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        )
    )

    fallback_ctx = _ctx(env=46.0, wildland=43.0, historic=34.0, ring_metrics={})
    _setup(monkeypatch, tmp_path, fallback_ctx)
    fallback = _run(_payload("Fallback Data Ln", {}, confirmed=[]))

    assert observed["confidence_score"] > fallback["confidence_score"]
    assert observed["evidence_quality_summary"]["evidence_quality_score"] > fallback["evidence_quality_summary"]["evidence_quality_score"]


def test_score_family_eligibility_full_path(monkeypatch, tmp_path):
    _setup(
        monkeypatch,
        tmp_path,
        _ctx(
            env=42.0,
            wildland=38.0,
            historic=30.0,
            ring_metrics={
                "zone_0_5_ft": {"vegetation_density": 20.0},
                "zone_5_30_ft": {"vegetation_density": 30.0},
                "zone_30_100_ft": {"vegetation_density": 35.0},
            },
        ),
    )
    assessed = _run(
        _payload(
            "Full Eligibility Way",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 35,
                "construction_year": 2018,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        )
    )
    assert assessed["site_hazard_eligibility"]["eligibility_status"] == "full"
    assert assessed["home_vulnerability_eligibility"]["eligibility_status"] == "full"
    assert assessed["insurance_readiness_eligibility"]["eligibility_status"] == "full"
    assert assessed["assessment_status"] == "fully_scored"


def test_score_family_eligibility_partial_path(monkeypatch, tmp_path):
    partial_ctx = _ctx(env=56.0, wildland=50.0, historic=45.0, ring_metrics={})
    partial_ctx.slope_index = None
    partial_ctx.slope = None
    partial_ctx.environmental_layer_status["slope"] = "missing"
    partial_ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": None,
    }
    _setup(monkeypatch, tmp_path, partial_ctx)

    assessed = _run(
        _payload(
            "Partial Eligibility Way",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 25},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    assert assessed["site_hazard_eligibility"]["eligibility_status"] == "partial"
    assert assessed["home_vulnerability_eligibility"]["eligibility_status"] == "partial"
    assert assessed["assessment_status"] == "partially_scored"
    assert assessed["home_vulnerability_eligibility"]["caveats"]


def test_score_family_eligibility_insufficient_path_and_hard_blocker(monkeypatch, tmp_path):
    insufficient_ctx = _ctx(
        env=0.0,
        wildland=0.0,
        historic=0.0,
        ring_metrics={},
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "missing",
            "fuel": "missing",
            "canopy": "missing",
            "fire_history": "missing",
        },
    )
    insufficient_ctx.burn_probability_index = None
    insufficient_ctx.hazard_severity_index = None
    insufficient_ctx.slope_index = None
    insufficient_ctx.fuel_index = None
    insufficient_ctx.canopy_index = None
    insufficient_ctx.historic_fire_index = None
    insufficient_ctx.wildland_distance_index = None
    insufficient_ctx.burn_probability = None
    insufficient_ctx.wildfire_hazard = None
    insufficient_ctx.slope = None
    insufficient_ctx.fuel_model = None
    insufficient_ctx.canopy_cover = None
    insufficient_ctx.historic_fire_distance = None
    insufficient_ctx.wildland_distance = None
    insufficient_ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "provider_unavailable",
        "fallback_mode": "point_based",
        "ring_metrics": None,
    }
    _setup(monkeypatch, tmp_path, insufficient_ctx)

    assessed = _run(_payload("Insufficient Evidence Way", {}, confirmed=[]))
    assert assessed["site_hazard_eligibility"]["eligibility_status"] == "insufficient"
    assert assessed["home_vulnerability_eligibility"]["eligibility_status"] == "insufficient"
    assert assessed["insurance_readiness_eligibility"]["eligibility_status"] == "insufficient"
    assert assessed["assessment_status"] == "insufficient_data"
    assert assessed["use_restriction"] == "not_for_underwriting_or_binding"
    assert assessed["assessment_blockers"]
    assert assessed["wildfire_risk_score_available"] is False
    assert assessed["site_hazard_score_available"] is False
    assert assessed["home_ignition_vulnerability_score_available"] is False
    assert assessed["insurance_readiness_score_available"] is False
    assert assessed["wildfire_risk_score"] is None
    assert assessed["site_hazard_score"] is None
    assert assessed["home_ignition_vulnerability_score"] is None
    assert assessed["insurance_readiness_score"] is None
    notes = " ".join(assessed["scoring_notes"]).lower()
    assert "wildfire score not computed" in notes
    assert "home vulnerability score not computed" in notes
    assert "insurance readiness score not computed" in notes
    assert assessed["confidence_tier"] == "preliminary"
    assert assessed["confidence_score"] == 0.0
    assert assessed["scoring_status"] == "insufficient_data_to_score"
    assert assessed["top_risk_drivers"]
    assert "not enough verified inputs" in assessed["top_risk_drivers"][0].lower()
    assert assessed["top_risk_drivers_detailed"] == []
    assert "preliminary near-structure observation" in str(
        (assessed.get("defensible_space_analysis") or {}).get("summary") or ""
    ).lower()


def test_assessment_diagnostics_and_report_persistence(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=50.0, wildland=48.0, historic=42.0, ring_metrics={}))
    assessed = _run(_payload("Diagnostics Persist Way", {"defensible_space_ft": 12}))
    diagnostics = assessed["assessment_diagnostics"]
    assert "critical_inputs_present" in diagnostics
    assert "critical_inputs_missing" in diagnostics
    assert "trust_tier_blockers" in diagnostics

    report = client.get(f"/report/{assessed['assessment_id']}")
    assert report.status_code == 200
    body = report.json()
    assert "assessment_diagnostics" in body
    assert set(body["assessment_diagnostics"].keys()) == set(diagnostics.keys())
    assert "layer_coverage_audit" in body
    assert "coverage_summary" in body
    assert isinstance(body["layer_coverage_audit"], list)
    assert isinstance(body["coverage_summary"], dict)


def test_missing_burn_probability_and_defensible_space_use_fallback_with_confidence_penalty(monkeypatch, tmp_path):
    full_ctx = _ctx(env=58.0, wildland=52.0, historic=47.0)
    full_ctx.burn_probability_index = 58.0
    full_ctx.environmental_layer_status["burn_probability"] = "ok"
    _setup(monkeypatch, tmp_path, full_ctx)
    full = _run(
        _payload(
            "Fallback Baseline Full",
            {
                "roof_type": "class_a_asphalt_composition",
                "vent_type": "ember_resistant_vents",
                "defensible_space_ft": 30,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )

    fallback_ctx = _ctx(env=58.0, wildland=52.0, historic=47.0)
    fallback_ctx.burn_probability_index = None
    fallback_ctx.burn_probability = None
    fallback_ctx.environmental_layer_status["burn_probability"] = "missing"
    fallback_ctx.hazard_severity_index = 56.0
    fallback_ctx.wildfire_hazard = 56.0
    fallback_ctx.property_level_context["ring_metrics"] = {
        "ring_0_5_ft": {"vegetation_density": 62.0},
        "ring_5_30_ft": {"vegetation_density": 58.0},
        "ring_30_100_ft": {"vegetation_density": 50.0},
    }
    fallback_ctx.property_level_context["footprint_used"] = True
    fallback_ctx.structure_ring_metrics = fallback_ctx.property_level_context["ring_metrics"]
    _setup(monkeypatch, tmp_path, fallback_ctx)

    assessed = _run(
        _payload(
            "Fallback Missing Burn Defensible",
            {
                "roof_type": "class_a_asphalt_composition",
                "vent_type": "ember_resistant_vents",
            },
            confirmed=["roof_type", "vent_type"],
        )
    )
    assert assessed["wildfire_risk_score_available"] is True
    assert assessed["site_hazard_score_available"] is True
    assert assessed["assessment_status"] in {"fully_scored", "partially_scored"}
    assert assessed["confidence_score"] < full["confidence_score"]
    diagnostics = assessed["assessment_diagnostics"]
    fallback_decisions = diagnostics.get("fallback_decisions") or []
    assert any(str(d.get("missing_input")) == "defensible_space_ft" for d in fallback_decisions)
    assert any(str(d.get("missing_input")) == "burn_probability_layer" for d in fallback_decisions)
    assert assessed["assessment_limitations_summary"]


def test_low_coverage_site_evidence_returns_limited_homeowner_estimate(monkeypatch, tmp_path):
    ctx = _ctx(
        env=51.0,
        wildland=57.0,
        historic=40.0,
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "missing",
        },
        ring_metrics={},
    )
    ctx.burn_probability_index = None
    ctx.hazard_severity_index = None
    ctx.moisture_index = None
    ctx.burn_probability = None
    ctx.wildfire_hazard = None
    ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": {},
        "region_property_specific_readiness": "limited_regional_ready",
        "region_required_layers_missing": [],
        "region_optional_layers_missing": ["roads", "gridmet_dryness", "whp"],
        "region_enrichment_layers_missing": ["parcel_polygons", "address_points", "naip_features"],
    }
    _setup(monkeypatch, tmp_path, ctx)

    assessed = _run(_payload("Limited Site Component Way", {}, confirmed=[]))
    assert assessed["assessment_output_state"] == "limited_regional_estimate"
    assert assessed["assessment_mode"] == "limited_regional_estimate"
    assert assessed["scoring_status"] == "limited_homeowner_estimate"
    assert "site_hazard" in (assessed.get("computed_components") or [])
    assert assessed["site_hazard_score_available"] is True
    assert assessed["wildfire_risk_score_available"] is True
    assert assessed["home_ignition_vulnerability_score_available"] is False
    homeowner_summary = assessed.get("homeowner_summary") or {}
    assert homeowner_summary.get("scoring_status") == "limited_homeowner_estimate"


def test_missing_structure_specific_fields_still_scores_home_with_fallbacks(monkeypatch, tmp_path):
    ctx = _ctx(env=54.0, wildland=50.0, historic=46.0)
    ctx.property_level_context["ring_metrics"] = {
        "ring_0_5_ft": {"vegetation_density": 48.0},
        "ring_5_30_ft": {"vegetation_density": 55.0},
        "ring_30_100_ft": {"vegetation_density": 45.0},
    }
    ctx.property_level_context["footprint_used"] = True
    ctx.structure_ring_metrics = ctx.property_level_context["ring_metrics"]
    _setup(monkeypatch, tmp_path, ctx)

    assessed = _run(_payload("Missing Structure Fields", {}, confirmed=[]))
    assert assessed["home_ignition_vulnerability_score_available"] is True
    assert assessed["wildfire_risk_score_available"] is True
    assert assessed["assessment_status"] in {"fully_scored", "partially_scored"}
    fallback_decisions = (assessed["assessment_diagnostics"] or {}).get("fallback_decisions") or []
    assert any(str(d.get("missing_input")) == "roof_type" for d in fallback_decisions)
    assert any(str(d.get("missing_input")) == "vent_type" for d in fallback_decisions)
    assert assessed["structure_data_completeness"] <= 40.0
    assert assessed["structure_assumption_mode"] in {"default_assumed", "mixed"}
    assert assessed["structure_score_confidence"] <= 55.0
    struct_weight = float(
        assessed["weighted_contributions"]["structure_vulnerability_risk"]["effective_weight"] or 0.0
    )
    assert struct_weight < 0.09
    readiness_factor_names = {f["name"] for f in assessed["readiness_factors"]}
    assert "structure_data_completeness" in readiness_factor_names or "structure_evidence_quality" in readiness_factor_names


def test_missing_structure_details_reduce_structure_scoring_confidence_vs_observed(monkeypatch, tmp_path):
    ctx = _ctx(env=52.0, wildland=49.0, historic=44.0)
    ctx.property_level_context["ring_metrics"] = {
        "ring_0_5_ft": {"vegetation_density": 44.0},
        "ring_5_30_ft": {"vegetation_density": 51.0},
        "ring_30_100_ft": {"vegetation_density": 46.0},
    }
    ctx.property_level_context["footprint_used"] = True
    ctx.structure_ring_metrics = ctx.property_level_context["ring_metrics"]
    _setup(monkeypatch, tmp_path, ctx)

    observed = _run(
        _payload(
            "Observed Structure Confidence",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "window_type": "dual pane tempered",
                "construction_year": 2017,
                "defensible_space_ft": 24,
            },
            confirmed=["roof_type", "vent_type", "window_type", "construction_year", "defensible_space_ft"],
        )
    )
    unknown = _run(
        _payload(
            "Unknown Structure Confidence",
            {"defensible_space_ft": 24},
            confirmed=["defensible_space_ft"],
        )
    )

    assert unknown["structure_data_completeness"] < observed["structure_data_completeness"]
    assert unknown["structure_score_confidence"] < observed["structure_score_confidence"]
    assert unknown["structure_score_confidence"] <= 55.0
    assert observed["structure_score_confidence"] >= 70.0
    assert unknown["structure_assumption_mode"] in {"default_assumed", "mixed"}
    assert observed["structure_assumption_mode"] in {"observed", "mixed"}

    observed_weight = float(
        observed["weighted_contributions"]["structure_vulnerability_risk"]["effective_weight"] or 0.0
    )
    unknown_weight = float(
        unknown["weighted_contributions"]["structure_vulnerability_risk"]["effective_weight"] or 0.0
    )
    assert unknown_weight < observed_weight

    unknown_structure_score = float(unknown["submodel_scores"]["structure_vulnerability_risk"]["score"])
    assert 30.0 <= unknown_structure_score <= 70.0


def test_partial_noncritical_layer_coverage_degrades_without_fatal_failure(monkeypatch, tmp_path):
    ctx = _ctx(env=53.0, wildland=49.0, historic=43.0)
    ctx.environmental_layer_status["fire_history"] = "missing"
    ctx.historic_fire_index = None
    ctx.historic_fire_distance = None
    _setup(monkeypatch, tmp_path, ctx)

    assessed = _run(
        _payload(
            "Partial Layer Coverage",
            {"roof_type": "class_a_asphalt_composition", "vent_type": "ember_resistant_vents", "defensible_space_ft": 22},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    assert assessed["wildfire_risk_score_available"] is True
    assert assessed["assessment_status"] in {"fully_scored", "partially_scored"}
    assert "unexpected_hard_failure" not in " ".join(assessed["assessment_blockers"]).lower()
    notes = " ".join(assessed["scoring_notes"]).lower()
    assert "fire history layer missing" in notes or "historical fire perimeter layer" in notes


def test_blended_wildfire_score_available_when_one_component_unavailable(monkeypatch, tmp_path):
    ctx = _ctx(env=55.0, wildland=58.0, historic=41.0)
    ctx.fuel_index = None
    ctx.canopy_index = None
    ctx.fuel_model = None
    ctx.canopy_cover = None
    ctx.environmental_layer_status["fuel"] = "missing"
    ctx.environmental_layer_status["canopy"] = "missing"
    ctx.structure_ring_metrics = {}
    ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": {},
    }
    _setup(monkeypatch, tmp_path, ctx)

    assessed = _run(
        _payload(
            "Single Component Blend",
            {"roof_type": "class_a_asphalt_composition", "vent_type": "ember_resistant_vents"},
            confirmed=["roof_type", "vent_type"],
        )
    )
    assert assessed["site_hazard_score_available"] is True
    assert assessed["home_ignition_vulnerability_score_available"] is False
    assert assessed["wildfire_risk_score_available"] is True
    assert assessed["wildfire_risk_score"] is not None
    assert assessed["assessment_status"] == "partially_scored"
    assert any("component only" in str(note).lower() for note in assessed["scoring_notes"])


def test_provenance_freshness_status_classification(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_LAYER_BURN_PROB_DATE", "2099-01-01")
    monkeypatch.setenv("WF_LAYER_HAZARD_SEVERITY_DATE", "2010-01-01")
    monkeypatch.setenv("WF_LAYER_SLOPE_DATE", "2018-01-01")
    monkeypatch.setenv("WF_LAYER_FUEL_DATE", "2017-01-01")
    monkeypatch.setenv("WF_LAYER_CANOPY_DATE", "2016-01-01")
    monkeypatch.setenv("WF_LAYER_FIRE_PERIMETERS_DATE", "2015-01-01")

    _setup(monkeypatch, tmp_path, _ctx(env=52.0, wildland=57.0, historic=49.0))
    assessed = _run(_payload("Freshness Status Rd", {"roof_type": "class a", "vent_type": "ember-resistant"}))

    burn = assessed["input_source_metadata"]["burn_probability"]
    hazard = assessed["input_source_metadata"]["wildfire_hazard"]
    assert burn["freshness_status"] in {"current", "aging", "stale", "unknown"}
    assert hazard["freshness_status"] in {"current", "aging", "stale", "unknown"}
    assert assessed["data_provenance"]["summary"]["stale_data_share"] >= 0.0


def test_stale_critical_inputs_reduce_confidence_and_block_shareable(monkeypatch, tmp_path):
    attrs = {
        "roof_type": "class a",
        "vent_type": "ember-resistant",
        "defensible_space_ft": 40,
        "construction_year": 2019,
    }
    fresh_ctx = _ctx(
        env=40.0,
        wildland=35.0,
        historic=30.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 12.0},
            "zone_5_30_ft": {"vegetation_density": 22.0},
            "zone_30_100_ft": {"vegetation_density": 30.0},
        },
    )
    _setup(monkeypatch, tmp_path, fresh_ctx)
    fresh = _run(_payload("Fresh Inputs Ave", attrs, confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"]))

    monkeypatch.setenv("WF_LAYER_BURN_PROB_DATE", "2010-01-01")
    monkeypatch.setenv("WF_LAYER_HAZARD_SEVERITY_DATE", "2010-01-01")
    monkeypatch.setenv("WF_LAYER_SLOPE_DATE", "2010-01-01")
    monkeypatch.setenv("WF_LAYER_FUEL_DATE", "2010-01-01")
    monkeypatch.setenv("WF_LAYER_CANOPY_DATE", "2010-01-01")
    monkeypatch.setenv("WF_LAYER_FIRE_PERIMETERS_DATE", "2010-01-01")
    stale_ctx = _ctx(
        env=40.0,
        wildland=35.0,
        historic=30.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 12.0},
            "zone_5_30_ft": {"vegetation_density": 22.0},
            "zone_30_100_ft": {"vegetation_density": 30.0},
        },
    )
    _setup(monkeypatch, tmp_path, stale_ctx)
    stale = _run(_payload("Stale Inputs Ave", attrs, confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"]))

    assert stale["confidence_score"] <= fresh["confidence_score"]
    assert stale["confidence_tier"] in {"moderate", "low", "preliminary"}
    assert stale["use_restriction"] != "shareable"
    assert stale["data_provenance"]["summary"]["stale_data_share"] > 0.0


def test_preflight_coverage_tier_marks_high_vs_limited_specificity(monkeypatch, tmp_path):
    high_ctx = _ctx(
        env=57.0,
        wildland=52.0,
        historic=45.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 28.0},
            "ring_5_30_ft": {"vegetation_density": 33.0},
            "ring_30_100_ft": {"vegetation_density": 42.0},
        },
    )
    high_ctx.access_exposure_index = 24.0
    high_ctx.access_context = {"status": "ok", "source": "osm_road_network"}
    _setup(monkeypatch, tmp_path, high_ctx)
    high = _run(
        _payload(
            "High Coverage Way",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 35},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    assert high["assessment_specificity_tier"] == "property_specific"
    assert high["limited_assessment_flag"] is False
    assert high["feature_coverage_percent"] >= 70.0

    low_ctx = _ctx(env=57.0, wildland=52.0, historic=45.0, ring_metrics={})
    low_ctx.burn_probability_index = None
    low_ctx.hazard_severity_index = None
    low_ctx.moisture_index = None
    low_ctx.access_exposure_index = None
    low_ctx.access_context = {"status": "missing"}
    low_ctx.environmental_layer_status = {
        "burn_probability": "missing",
        "hazard": "missing",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "missing",
        "fire_history": "missing",
    }
    low_ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": {},
    }
    _setup(monkeypatch, tmp_path, low_ctx)
    low = _run(_payload("Low Coverage Way", {}, confirmed=[]))

    assert low["limited_assessment_flag"] is True
    assert low["assessment_specificity_tier"] in {"address_level", "regional_estimate"}
    assert low["score_specificity_warning"]
    assert low["confidence_tier"] in {"moderate", "low", "preliminary"}


def test_specificity_summary_tiers_for_property_address_and_regional(monkeypatch, tmp_path):
    property_ctx = _ctx(
        env=58.0,
        wildland=48.0,
        historic=39.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 18.0},
            "zone_5_30_ft": {"vegetation_density": 28.0},
            "zone_30_100_ft": {"vegetation_density": 36.0},
        },
    )
    property_ctx.access_exposure_index = 25.0
    property_ctx.access_context = {"status": "ok", "source": "osm_road_network"}
    _setup(monkeypatch, tmp_path, property_ctx)
    property_assessed = _run(
        _payload(
            "901 Property Specific Rd",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 30},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )
    property_specificity = property_assessed.get("specificity_summary") or {}
    assert property_specificity.get("specificity_tier") == "property_specific"
    assert property_specificity.get("comparison_allowed") is True
    assert "property-specific" in str(property_specificity.get("headline", "")).lower()

    address_ctx = _ctx(env=58.0, wildland=48.0, historic=39.0, ring_metrics={})
    address_ctx.access_exposure_index = 20.0
    address_ctx.access_context = {"status": "ok"}
    address_ctx.property_level_context.update(
        {
            "parcel_geometry": {
                "type": "Polygon",
                "coordinates": [[[-104.9905, 39.7391], [-104.9901, 39.7391], [-104.9901, 39.7394], [-104.9905, 39.7394], [-104.9905, 39.7391]]],
            },
            "region_property_specific_readiness": "address_level_only",
        }
    )
    _setup(monkeypatch, tmp_path, address_ctx)
    address_assessed = _run(_payload("902 Address Level Rd", {}, confirmed=[]))
    address_specificity = address_assessed.get("specificity_summary") or {}
    assert address_specificity.get("specificity_tier") == "address_level"
    assert isinstance(address_specificity.get("comparison_allowed"), bool)
    assert "address-level" in str(address_specificity.get("headline", "")).lower()

    regional_ctx = _ctx(env=58.0, wildland=48.0, historic=39.0, ring_metrics={})
    regional_ctx.burn_probability_index = None
    regional_ctx.hazard_severity_index = None
    regional_ctx.moisture_index = None
    regional_ctx.access_exposure_index = None
    regional_ctx.access_context = {"status": "missing"}
    regional_ctx.environmental_layer_status = {
        "burn_probability": "missing",
        "hazard": "missing",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "missing",
        "fire_history": "missing",
    }
    regional_ctx.property_level_context.update(
        {
            "footprint_used": False,
            "fallback_mode": "point_based",
            "ring_metrics": {},
            "region_property_specific_readiness": "limited_regional_ready",
        }
    )
    _setup(monkeypatch, tmp_path, regional_ctx)
    regional_assessed = _run(_payload("903 Regional Estimate Rd", {}, confirmed=[]))
    regional_specificity = regional_assessed.get("specificity_summary") or {}
    assert regional_specificity.get("specificity_tier") == "regional_estimate"
    assert regional_specificity.get("comparison_allowed") is False
    assert "nearby homes may appear similar" in str(
        regional_specificity.get("what_this_means", "")
    ).lower()


def test_low_coverage_homeowner_summary_uses_limited_mode_and_grouped_limitations(monkeypatch, tmp_path):
    low_ctx = _ctx(env=56.0, wildland=52.0, historic=45.0, ring_metrics={})
    low_ctx.burn_probability_index = None
    low_ctx.hazard_severity_index = None
    low_ctx.moisture_index = None
    low_ctx.access_exposure_index = None
    low_ctx.access_context = {"status": "missing"}
    low_ctx.environmental_layer_status = {
        "burn_probability": "missing",
        "hazard": "missing",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "missing",
        "fire_history": "missing",
    }
    low_ctx.property_level_context = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
        "ring_metrics": {},
        "region_property_specific_readiness": "limited_regional_ready",
        "region_required_layers_missing": [],
        "region_optional_layers_missing": ["whp", "roads", "gridmet_dryness"],
        "region_enrichment_layers_missing": ["parcel_polygons"],
    }
    _setup(monkeypatch, tmp_path, low_ctx)
    assessed = _run(_payload("Limited Output Lane", {}, confirmed=[]))

    assert assessed["assessment_output_state"] in {"limited_regional_estimate", "insufficient_data"}
    assert assessed["assessment_mode"] in {"limited_regional_estimate", "insufficient_data"}

    homeowner_summary = assessed.get("homeowner_summary") or {}
    assert homeowner_summary.get("assessment_mode") == assessed["assessment_mode"]
    assert homeowner_summary.get("home_hardening_readiness_precision") in {"provisional", "stable"}
    evidence_snapshot = homeowner_summary.get("evidence_snapshot") or {}
    assert isinstance(evidence_snapshot, dict)
    assert "observed_feature_count" in evidence_snapshot
    assert "regional_enrichment_consumption_score" in evidence_snapshot
    limitations = homeowner_summary.get("assessment_limitations") or []
    assert isinstance(limitations, list)
    assert len(limitations) <= 5
    categories = [row.get("category") for row in limitations if isinstance(row, dict)]
    assert len(categories) == len(set(categories))

    confidence_summary = homeowner_summary.get("confidence_summary") or {}
    assert isinstance(confidence_summary, dict)
    assert confidence_summary.get("assessment_type") in {
        "limited regional estimate",
        "insufficient data",
    }
    why_limited = confidence_summary.get("why_confidence_is_limited") or []
    assert isinstance(why_limited, list)
    assert len(why_limited) <= 4
    assert all("underwriting threshold" not in str(reason).lower() for reason in why_limited)
    how_to_improve = confidence_summary.get("how_to_improve_confidence") or []
    assert isinstance(how_to_improve, list)
    assert len(how_to_improve) >= 1
    trust_summary = homeowner_summary.get("trust_summary") or {}
    assert isinstance(trust_summary, dict)
    assert trust_summary.get("confidence_level") == "limited confidence"
    uncertainty_drivers = trust_summary.get("uncertainty_drivers") or []
    assert isinstance(uncertainty_drivers, list)
    assert len(uncertainty_drivers) >= 1
    assert isinstance(trust_summary.get("fallback_drivers") or [], list)
    assert isinstance(trust_summary.get("missing_inputs") or [], list)
    assert isinstance(trust_summary.get("coverage_gaps") or [], list)
    low_diff = trust_summary.get("low_differentiation_explanation")
    assert isinstance(low_diff, dict)
    assert low_diff.get("applies") is True
    assert isinstance(low_diff.get("what_was_measured_directly") or [], list)
    assert isinstance(low_diff.get("what_was_estimated_from_regional_context") or [], list)
    assert isinstance(low_diff.get("what_would_make_this_more_property_specific") or [], list)
    assert isinstance(low_diff.get("why_nearby_properties_may_appear_similar"), str)
    estimated_blob = " ".join(str(item).lower() for item in (low_diff.get("what_was_estimated_from_regional_context") or []))
    assert "building footprint was missing" in estimated_blob
    assert "map point" in estimated_blob or "regional context" in estimated_blob
    improvement_blob = " ".join(str(item).lower() for item in (low_diff.get("what_would_make_this_more_property_specific") or []))
    assert "roof type" in improvement_blob or "vent type" in improvement_blob or "defensible space" in improvement_blob
    assert isinstance(trust_summary.get("nearby_home_comparison_safeguard_triggered"), bool)
    if trust_summary.get("nearby_home_comparison_safeguard_triggered"):
        assert trust_summary.get("parcel_level_comparison_allowed") is False
        assert (
            trust_summary.get("nearby_home_comparison_safeguard_message")
            == "This estimate is not precise enough to compare adjacent homes."
        )
        assert (
            homeowner_summary.get("nearby_home_comparison_safeguard_message")
            == "This estimate is not precise enough to compare adjacent homes."
        )
        assert assessed.get("top_near_structure_risk_drivers") == []
        assert assessed.get("prioritized_vegetation_actions") == []

    confidence_actions = homeowner_summary.get("confidence_improvement_actions") or []
    assert isinstance(confidence_actions, list)
    assert len(confidence_actions) >= 1
    joined_actions = " ".join(str(item).lower() for item in confidence_actions)
    assert "gridmet" in joined_actions or "wf_default_gridmet_dryness" in joined_actions
    assert any(
        "map point" in str(item).lower()
        or "map modal" in str(item).lower()
        or "place the pin" in str(item).lower()
        for item in confidence_actions
    )

    diagnostics = assessed.get("developer_diagnostics") or {}
    assert isinstance((diagnostics.get("fallback_decisions") or []), list)
    assert "preflight" in diagnostics
    improve_your_result = homeowner_summary.get("improve_your_result") or {}
    assert isinstance(improve_your_result, dict)
    suggestions = improve_your_result.get("suggestions") or []
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 1
    joined_suggestions = " ".join(str(item).lower() for item in suggestions)
    assert "roof type" in joined_suggestions or "defensible space" in joined_suggestions
    diagnostic_sources = improve_your_result.get("diagnostic_sources") or {}
    assert isinstance(diagnostic_sources, dict)
    assert isinstance(diagnostic_sources.get("coverage_gaps") or [], list)
    assert isinstance(diagnostic_sources.get("fallback_inputs") or [], list)


def test_missing_factor_omission_reports_weight_and_counts(monkeypatch, tmp_path):
    ctx = _ctx(
        env=54.0,
        wildland=49.0,
        historic=41.0,
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 44.0},
            "ring_5_30_ft": {"vegetation_density": 52.0},
            "ring_30_100_ft": {"vegetation_density": 50.0},
        },
    )
    _setup(monkeypatch, tmp_path, ctx)
    assessed = _run(_payload("Omission Path Rd", {}, confirmed=[]))

    assert assessed["observed_factor_count"] >= 1
    assert assessed["missing_factor_count"] >= 1
    assert assessed["observed_weight_fraction"] < 1.0
    assert assessed["fallback_dominance_ratio"] >= 0.0
    diagnostics = assessed["assessment_diagnostics"]
    fallback_decisions = diagnostics.get("fallback_decisions") or []
    assert any(str(row.get("fallback_type")) == "missing_factor_omitted" for row in fallback_decisions)
    weighted = assessed["weighted_contributions"]
    assert any("effective_weight" in row for row in weighted.values())
    assert any(bool(row.get("omitted_due_to_missing")) for row in weighted.values())


def test_homeowner_trust_summary_maps_internal_tiers_to_user_friendly_labels():
    high = app_main._build_homeowner_trust_summary(
        confidence_tier="high",
        fallback_decisions=[],
        missing_inputs=[],
        preflight={},
    )
    moderate = app_main._build_homeowner_trust_summary(
        confidence_tier="moderate",
        fallback_decisions=[],
        missing_inputs=[],
        preflight={},
    )
    low = app_main._build_homeowner_trust_summary(
        confidence_tier="low",
        fallback_decisions=[],
        missing_inputs=[],
        preflight={},
    )

    assert high.get("confidence_level") == "high confidence"
    assert moderate.get("confidence_level") == "moderate confidence"
    assert low.get("confidence_level") == "limited confidence"

    safeguarded = app_main._build_homeowner_trust_summary(
        confidence_tier="low",
        fallback_decisions=[],
        missing_inputs=[],
        preflight={},
        differentiation_snapshot={
            "differentiation_mode": "mostly_regional",
            "neighborhood_differentiation_confidence": 22.0,
            "notes": [],
        },
    )
    assert safeguarded.get("nearby_home_comparison_safeguard_triggered") is True
    assert safeguarded.get("parcel_level_comparison_allowed") is False
    assert (
        safeguarded.get("nearby_home_comparison_safeguard_message")
        == "This estimate is not precise enough to compare adjacent homes."
    )

    property_specific = app_main._build_homeowner_trust_summary(
        confidence_tier="high",
        fallback_decisions=[],
        missing_inputs=[],
        preflight={
            "feature_coverage_summary": {
                "building_footprint_available": True,
                "parcel_polygon_available": True,
                "near_structure_vegetation_available": True,
                "hazard_severity_available": True,
                "burn_probability_available": True,
                "dryness_available": True,
            },
            "geometry_basis": "footprint",
        },
        differentiation_snapshot={
            "differentiation_mode": "property_specific",
            "neighborhood_differentiation_confidence": 82.0,
            "notes": [],
        },
    )
    assert property_specific.get("low_differentiation_explanation") is None


def test_old_rows_without_provenance_defaults_are_readable(tmp_path):
    store = AssessmentStore(str(tmp_path / "legacy_no_prov.db"))
    legacy_payload = {
        "assessment_id": "legacy-prov-1",
        "address": "Legacy Provenance Ln",
        "coordinates": {"latitude": 1.1, "longitude": 2.2},
        "risk_scores": {"wildfire_risk_score": 44.0, "insurance_readiness_score": 61.0},
        "risk_drivers": {"environmental": 45.0, "structural": 43.0, "access_exposure": 20.0},
        "mitigation_recommendations": [{"action": "clear brush", "related_factor": "fuel"}],
    }
    conn = sqlite3.connect(tmp_path / "legacy_no_prov.db")
    conn.execute(
        "INSERT INTO assessments (assessment_id, created_at, payload_json, model_version) VALUES (?, datetime('now'), ?, ?)",
        ("legacy-prov-1", json.dumps(legacy_payload), LEGACY_MODEL_VERSION),
    )
    conn.commit()
    conn.close()

    loaded = store.get("legacy-prov-1")
    assert loaded is not None
    assert loaded.data_provenance.summary.missing_data_share >= 0.0
    assert isinstance(loaded.site_hazard_input_quality.model_dump(), dict)
    assert isinstance(loaded.home_vulnerability_input_quality.model_dump(), dict)
    assert isinstance(loaded.insurance_readiness_input_quality.model_dump(), dict)
