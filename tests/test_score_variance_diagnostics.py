from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import load_scoring_config
from backend.wildfire_data import WildfireContext
from scripts.analyze_score_variance import run_variance_analysis


client = TestClient(app_main.app)


def _fixture_context(payload: dict[str, float | dict]) -> WildfireContext:
    ring_metrics = payload.get("structure_ring_metrics") or {}
    return WildfireContext(
        environmental_index=None,
        slope_index=float(payload.get("slope_index")) if payload.get("slope_index") is not None else None,
        aspect_index=float(payload.get("aspect_index")) if payload.get("aspect_index") is not None else None,
        fuel_index=float(payload.get("fuel_index")) if payload.get("fuel_index") is not None else None,
        moisture_index=float(payload.get("moisture_index")) if payload.get("moisture_index") is not None else None,
        canopy_index=float(payload.get("canopy_index")) if payload.get("canopy_index") is not None else None,
        wildland_distance_index=(
            float(payload.get("wildland_distance_index")) if payload.get("wildland_distance_index") is not None else None
        ),
        historic_fire_index=float(payload.get("historic_fire_index")) if payload.get("historic_fire_index") is not None else None,
        burn_probability_index=(
            float(payload.get("burn_probability_index")) if payload.get("burn_probability_index") is not None else None
        ),
        hazard_severity_index=(
            float(payload.get("hazard_severity_index")) if payload.get("hazard_severity_index") is not None else None
        ),
        access_exposure_index=float(payload.get("access_exposure_index")) if payload.get("access_exposure_index") is not None else None,
        burn_probability=float(payload.get("burn_probability")) if payload.get("burn_probability") is not None else None,
        wildfire_hazard=float(payload.get("wildfire_hazard")) if payload.get("wildfire_hazard") is not None else None,
        slope=float(payload.get("slope")) if payload.get("slope") is not None else None,
        fuel_model=float(payload.get("fuel_model")) if payload.get("fuel_model") is not None else None,
        canopy_cover=float(payload.get("canopy_cover")) if payload.get("canopy_cover") is not None else None,
        historic_fire_distance=(
            float(payload.get("historic_fire_distance")) if payload.get("historic_fire_distance") is not None else None
        ),
        wildland_distance=float(payload.get("wildland_distance")) if payload.get("wildland_distance") is not None else None,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["score-variance-fixture"],
        assumptions=[],
        structure_ring_metrics=ring_metrics,
        property_level_context={
            "footprint_used": bool(ring_metrics),
            "footprint_status": "used" if ring_metrics else "not_found",
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "ring_metrics": ring_metrics,
            "near_structure_vegetation_0_5_pct": (
                float(payload.get("near_structure_vegetation_0_5_pct"))
                if payload.get("near_structure_vegetation_0_5_pct") is not None
                else None
            ),
            "canopy_adjacency_proxy_pct": (
                float(payload.get("canopy_adjacency_proxy_pct"))
                if payload.get("canopy_adjacency_proxy_pct") is not None
                else None
            ),
            "vegetation_continuity_proxy_pct": (
                float(payload.get("vegetation_continuity_proxy_pct"))
                if payload.get("vegetation_continuity_proxy_pct") is not None
                else None
            ),
            "nearest_high_fuel_patch_distance_ft": (
                float(payload.get("nearest_high_fuel_patch_distance_ft"))
                if payload.get("nearest_high_fuel_patch_distance_ft") is not None
                else None
            ),
            "imagery_local_percentiles": payload.get("imagery_local_percentiles") or {},
            "neighboring_structure_metrics": payload.get("neighboring_structure_metrics") or {},
            "feature_sampling": {
                "fuel_model": {
                    "raw_point_value": payload.get("fuel_model"),
                    "sample_count": 64,
                    "blended_index": payload.get("fuel_index"),
                    "scope": "neighborhood_level",
                }
            },
        },
    )


def test_debug_payload_includes_score_variance_sections(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    fixture = json.loads(Path("tests/fixtures/score_variance_scenarios.json").read_text(encoding="utf-8"))["scenarios"][0]
    context = _fixture_context(fixture["context"])

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon, **_kwargs: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "variance_debug.db")))

    geocode_resolution = app_main.GeocodeResolution(
        raw_input=fixture["address"],
        normalized_address=fixture["address"].lower(),
        geocode_status="accepted",
        candidate_count=1,
        selected_candidate=None,
        confidence_score=0.92,
        latitude=46.87,
        longitude=-113.99,
        geocode_source="test",
        geocode_meta={
            "submitted_address": fixture["address"],
            "normalized_address": fixture["address"].lower(),
            "geocode_status": "accepted",
            "resolved_latitude": 46.87,
            "resolved_longitude": -113.99,
        },
    )
    coverage_resolution = app_main.RegionCoverageResolution(
        coverage_available=True,
        resolved_region_id="missoula_pilot",
        reason="covered",
        diagnostics=[],
        coverage={
            "coverage_available": True,
            "resolved_region_id": "missoula_pilot",
            "resolved_region_display_name": "Missoula Pilot",
            "region_check_result": "inside",
            "candidate_regions_containing_point": ["missoula_pilot"],
        },
    )

    monkeypatch.setattr(
        app_main,
        "_resolve_location_for_route",
        lambda **_kwargs: (geocode_resolution, coverage_resolution, 46.87, -113.99),
    )

    response = client.post(
        "/risk/debug",
        json={
            "address": fixture["address"],
            "attributes": fixture["attributes"],
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert "score_variance_diagnostics" in body
    assert "raw_feature_vector" in body
    assert "transformed_feature_vector" in body
    assert "factor_contribution_breakdown" in body
    assert "compression_flags" in body
    raw = body["raw_feature_vector"]
    assert "near_structure_vegetation_0_5_pct" in raw
    assert "canopy_adjacency_proxy_pct" in raw
    assert "vegetation_continuity_proxy_pct" in raw
    assert "nearest_high_fuel_patch_distance_ft" in raw
    assert isinstance(body["score_variance_diagnostics"].get("compression_analysis_summary"), list)


def test_variance_fixture_produces_material_score_spread(tmp_path):
    fixture_path = Path("tests/fixtures/score_variance_scenarios.json")
    csv_out = tmp_path / "variance.csv"
    summary = run_variance_analysis(fixture_path=fixture_path, csv_out=csv_out)

    wildfire_stats = summary["score_stats"]["wildfire_risk_score"]
    assert summary["scenario_count"] >= 6
    assert wildfire_stats["max"] - wildfire_stats["min"] >= 20.0
    assert wildfire_stats["stddev"] >= 6.0
    assert csv_out.exists()


def test_same_site_structure_inputs_shift_score_meaningfully():
    payload = json.loads(Path("tests/fixtures/score_variance_scenarios.json").read_text(encoding="utf-8"))
    by_id = {row["scenario_id"]: row for row in payload["scenarios"]}
    hardened = by_id["same_site_hardened_home"]
    poor = by_id["same_site_poor_hardening"]

    engine = RiskEngine(load_scoring_config())

    ctx_h = _fixture_context(hardened["context"])
    attrs_h = PropertyAttributes.model_validate(hardened["attributes"])
    risk_h = engine.score(attrs_h, lat=0.0, lon=0.0, context=ctx_h)
    site_h = engine.compute_site_hazard_score(risk_h)
    home_h = engine.compute_home_ignition_vulnerability_score(risk_h)
    readiness_h = engine.compute_insurance_readiness(attrs_h, ctx_h, risk_h).insurance_readiness_score
    wildfire_h = engine.compute_blended_wildfire_score(site_h, home_h, readiness_h)

    ctx_p = _fixture_context(poor["context"])
    attrs_p = PropertyAttributes.model_validate(poor["attributes"])
    risk_p = engine.score(attrs_p, lat=0.0, lon=0.0, context=ctx_p)
    site_p = engine.compute_site_hazard_score(risk_p)
    home_p = engine.compute_home_ignition_vulnerability_score(risk_p)
    readiness_p = engine.compute_insurance_readiness(attrs_p, ctx_p, risk_p).insurance_readiness_score
    wildfire_p = engine.compute_blended_wildfire_score(site_p, home_p, readiness_p)

    assert home_p - home_h >= 12.0
    assert wildfire_p - wildfire_h >= 8.0
