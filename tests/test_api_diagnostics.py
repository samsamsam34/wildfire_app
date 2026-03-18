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
        property_level_context={
            "footprint_used": True,
            "footprint_status": "used",
            "fallback_mode": "footprint",
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
    assert d1["stability"]["local_sensitivity_score"] == d2["stability"]["local_sensitivity_score"]
    assert d1["stability"]["tier_flip_risk"] == d2["stability"]["tier_flip_risk"]
    assert d1["mitigation_sensitivity"]["top_interventions"] == d2["mitigation_sensitivity"]["top_interventions"]


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
