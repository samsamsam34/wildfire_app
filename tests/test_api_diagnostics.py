from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.trust_metadata import clear_trust_reference_cache
from backend.wildfire_data import WildfireContext

client = TestClient(app_main.app)


def _context() -> WildfireContext:
    ring_metrics = {
        "zone_0_5_ft": {"vegetation_density": 36.0},
        "zone_5_30_ft": {"vegetation_density": 44.0},
        "zone_30_100_ft": {"vegetation_density": 40.0},
    }
    return WildfireContext(
        environmental_index=55.0,
        slope_index=52.0,
        aspect_index=50.0,
        fuel_index=58.0,
        moisture_index=46.0,
        canopy_index=57.0,
        wildland_distance_index=53.0,
        historic_fire_index=49.0,
        burn_probability_index=56.0,
        hazard_severity_index=59.0,
        burn_probability=0.56,
        wildfire_hazard=3.2,
        slope=18.0,
        fuel_model=62.0,
        canopy_cover=54.0,
        historic_fire_distance=1.5,
        wildland_distance=180.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["test-context"],
        assumptions=[],
        structure_ring_metrics=ring_metrics,
        property_level_context={
            "footprint_used": True,
            "footprint_status": "used",
            "fallback_mode": "footprint",
            "ring_metrics": ring_metrics,
            "parcel_geometry": {
                "type": "Polygon",
                "coordinates": [[[-104.99045, 39.7390], [-104.99010, 39.7390], [-104.99010, 39.7393], [-104.99045, 39.7393], [-104.99045, 39.7390]]],
            },
            "near_structure_vegetation_0_5_pct": 38.0,
            "canopy_adjacency_proxy_pct": 41.0,
            "vegetation_continuity_proxy_pct": 35.0,
            "nearest_high_fuel_patch_distance_ft": 26.0,
            "region_id": "test_region",
        },
    )


def _setup(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(
        app_main.wildfire_data,
        "collect_context",
        lambda _lat, _lon, **_kwargs: _context(),
    )
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_assessments.db")))


def _payload() -> dict[str, object]:
    return {
        "address": "123 Test Diagnostics Way, Test, CO",
        "attributes": {
            "roof_type": "composite",
            "vent_type": "standard",
            "defensible_space_ft": 14,
            "construction_year": 2008,
        },
        "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        "audience": "homeowner",
        "tags": ["diagnostics-test"],
    }


def test_diagnostics_omitted_by_default(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    clear_trust_reference_cache()
    response = client.post("/risk/assess", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert "assessment_id" in body
    assert "diagnostics" not in body
    assert "assessment" not in body


def test_diagnostics_included_when_flag_enabled(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    clear_trust_reference_cache()
    response = client.post("/risk/assess?include_diagnostics=true", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert "assessment" in body
    assert "diagnostics" in body
    diagnostics = body["diagnostics"]
    assert diagnostics["version"] == "ngt_eval_v1"
    assert diagnostics["evaluation_basis"] == "no_ground_truth"
    assert "do not establish real-world predictive accuracy" in diagnostics["caveat"].lower()
    assert "confidence" in diagnostics
    assert "stability" in diagnostics
    assert "mitigation_sensitivity" in diagnostics
    assert "monotonicity" in diagnostics
    assert "benchmark_alignment" in diagnostics
    assert "distribution_context" in diagnostics
    assert "differentiation_mode" in diagnostics
    assert "property_specific_feature_count" in diagnostics
    assert "proxy_feature_count" in diagnostics
    assert "defaulted_feature_count" in diagnostics
    assert "regional_feature_count" in diagnostics
    assert "local_differentiation_score" in diagnostics
    assert "neighborhood_differentiation_confidence" in diagnostics
    assert "vegetation_signal" in diagnostics
    assert "inferred_fields" in diagnostics["confidence"]
    assert "fallback_weight_fraction" in diagnostics["confidence"]
    assert "confidence_reduction_reasons" in diagnostics["confidence"]
    assert "assumption_sensitive" in diagnostics["stability"]
    assert "major_driver" in diagnostics["vegetation_signal"]
    assert "driver_strength" in diagnostics["vegetation_signal"]
    assert "fallback_heavy" in diagnostics["confidence"]


def test_missing_reference_artifacts_degrade_gracefully(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    monkeypatch.delenv("WF_TRUST_REFERENCE_ARTIFACT_DIR", raising=False)
    clear_trust_reference_cache()
    response = client.post("/risk/assess?include_diagnostics=true", json=_payload())
    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["benchmark_alignment"]["available"] is False
    assert diagnostics["benchmark_alignment"]["local_alignment"] == "unknown"
    assert diagnostics["distribution_context"]["relative_risk_percentile"] is None


def test_diagnostics_metrics_are_deterministic_for_fixed_fixture_inputs(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    monkeypatch.delenv("WF_TRUST_REFERENCE_ARTIFACT_DIR", raising=False)
    clear_trust_reference_cache()
    first = client.post("/risk/assess?include_diagnostics=true", json=_payload())
    second = client.post("/risk/assess?include_diagnostics=true", json=_payload())
    assert first.status_code == 200
    assert second.status_code == 200
    d1 = first.json()["diagnostics"]
    d2 = second.json()["diagnostics"]
    assert d1["confidence"]["fallback_heavy"] == d2["confidence"]["fallback_heavy"]
    assert d1["confidence"]["fallback_weight_fraction"] == d2["confidence"]["fallback_weight_fraction"]
    assert d1["stability"]["local_sensitivity_score"] == d2["stability"]["local_sensitivity_score"]
    assert d1["stability"]["tier_flip_risk"] == d2["stability"]["tier_flip_risk"]
    assert d1["mitigation_sensitivity"]["top_interventions"] == d2["mitigation_sensitivity"]["top_interventions"]
    assert d1["vegetation_signal"] == d2["vegetation_signal"]


def test_report_endpoint_supports_opt_in_diagnostics(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    clear_trust_reference_cache()
    assessed = client.post("/risk/assess", json=_payload())
    assert assessed.status_code == 200
    assessment_id = assessed.json()["assessment_id"]
    report_default = client.get(f"/report/{assessment_id}")
    assert report_default.status_code == 200
    assert "diagnostics" not in report_default.json()
    report_diag = client.get(f"/report/{assessment_id}?include_diagnostics=true")
    assert report_diag.status_code == 200
    body = report_diag.json()
    assert "assessment" in body
    assert "diagnostics" in body
    assert "do not establish real-world predictive accuracy" in body["diagnostics"]["caveat"].lower()
    assert "vegetation_signal" in body["diagnostics"]
    assert "confidence_reduction_reasons" in body["diagnostics"]["confidence"]
    assert "assumption_sensitive" in body["diagnostics"]["stability"]


def test_differentiation_mode_highly_local_with_full_geometry(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    clear_trust_reference_cache()
    response = client.post("/risk/assess?include_diagnostics=true", json=_payload())
    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["differentiation_mode"] == "highly_local"
    assert float(diagnostics["local_differentiation_score"] or 0.0) >= 70.0
    assert float(diagnostics["neighborhood_differentiation_confidence"] or 0.0) >= 70.0
    assert int(diagnostics["property_specific_feature_count"] or 0) >= 4


def test_differentiation_mode_mixed_when_building_footprint_missing(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    clear_trust_reference_cache()
    base_context = _context()
    base_context.structure_ring_metrics = {}
    base_context.property_level_context.update(
        {
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": {},
        }
    )
    monkeypatch.setattr(
        app_main.wildfire_data,
        "collect_context",
        lambda _lat, _lon, **_kwargs: base_context,
    )
    response = client.post("/risk/assess?include_diagnostics=true", json=_payload())
    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["differentiation_mode"] in {"mixed", "mostly_regional"}
    assert float(diagnostics["neighborhood_differentiation_confidence"] or 0.0) < 75.0


def test_differentiation_mode_mostly_regional_with_sparse_property_signals(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    clear_trust_reference_cache()
    sparse = _context()
    sparse.hazard_severity_index = None
    sparse.wildfire_hazard = None
    sparse.burn_probability_index = None
    sparse.burn_probability = None
    sparse.structure_ring_metrics = {}
    sparse.environmental_layer_status = {
        "burn_probability": "missing",
        "hazard": "missing",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "ok",
        "fire_history": "ok",
    }
    sparse.property_level_context.update(
        {
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": {},
            "parcel_geometry": None,
            "near_structure_vegetation_0_5_pct": None,
            "canopy_adjacency_proxy_pct": None,
            "vegetation_continuity_proxy_pct": None,
            "nearest_high_fuel_patch_distance_ft": None,
        }
    )
    monkeypatch.setattr(
        app_main.wildfire_data,
        "collect_context",
        lambda _lat, _lon, **_kwargs: sparse,
    )
    sparse_payload = {
        "address": "123 Test Diagnostics Way, Test, CO",
        "attributes": {},
        "confirmed_fields": [],
        "audience": "homeowner",
        "tags": ["diagnostics-test"],
    }
    response = client.post("/risk/assess?include_diagnostics=true", json=sparse_payload)
    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["differentiation_mode"] == "mostly_regional"
    assert float(diagnostics["neighborhood_differentiation_confidence"] or 0.0) <= 40.0
