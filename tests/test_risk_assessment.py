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
        "legacy_weighted_wildfire_risk_score",
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "insurance_readiness_score",
        "wildfire_risk_score_available",
        "site_hazard_score_available",
        "home_ignition_vulnerability_score_available",
        "insurance_readiness_score_available",
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
    assert body["risk_scores"]["insurance_readiness_score"] == body["insurance_readiness_score"]
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
    assert isinstance(ring_context["ring_metrics"], dict)
    assert ring_context["ring_metrics"]["zone_0_5_ft"]["vegetation_density"] == pytest.approx(70.0)
    assert any("point-based annulus" in note.lower() for note in assumptions)
    assert any("point-proxy ring vegetation summaries" in s.lower() for s in sources)


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
    )
    env_names = set(app_main.ENVIRONMENTAL_SUBMODELS)
    struct_names = set(app_main.STRUCTURAL_SUBMODELS)
    weights = app_main.scoring_config.submodel_weights
    weighted = assessed["weighted_contributions"]
    env_weight = sum(weights[name] for name in env_names)
    struct_weight = sum(weights[name] for name in struct_names)
    expected_site = round(
        sum(weighted[name]["contribution"] for name in env_names) / env_weight,
        1,
    )
    expected_home = round(
        sum(weighted[name]["contribution"] for name in struct_names) / struct_weight,
        1,
    )
    assert assessed["site_hazard_score"] == expected_site
    assert assessed["home_ignition_vulnerability_score"] == expected_home
    assert assessed["wildfire_risk_score"] == expected_blended
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
    assert assessed["confidence"]["inferred_fields_count"] >= 1
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
    assert "layer_coverage_audit" in payload
    assert "coverage_summary" in payload
    assert isinstance(payload["layer_coverage_audit"], list)
    assert isinstance(payload["coverage_summary"], dict)
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
    assert set(metrics.keys()) == {
        "ring_0_5_ft",
        "ring_5_30_ft",
        "ring_30_100_ft",
        "ring_100_300_ft",
        "zone_0_5_ft",
        "zone_5_30_ft",
        "zone_30_100_ft",
        "zone_100_300_ft",
    }
    assert metrics["ring_0_5_ft"]["vegetation_density"] == 60.0
    assert metrics["zone_0_5_ft"]["vegetation_density"] == 60.0
    assert metrics["ring_5_30_ft"]["canopy_mean"] == 62.0
    assert "Structure ring vegetation summaries" in sources
    assert assumptions == []


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
