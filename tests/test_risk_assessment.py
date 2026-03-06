from __future__ import annotations

import json
import sqlite3

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.version import LEGACY_MODEL_VERSION, MODEL_VERSION
from backend.wildfire_data import WildfireContext


client = TestClient(app_main.app)


def _ctx(env: float, wildland: float, historic: float) -> WildfireContext:
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
        data_sources=[
            "Burn probability raster",
            "Wildfire hazard severity raster",
            "Fuel model raster",
            "Historical fire perimeter recurrence",
            "Slope raster",
            "Aspect raster",
        ],
        assumptions=[],
    )


def _setup(monkeypatch, tmp_path, context: WildfireContext) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_assessments.db")))


def _assert_core_contract(body: dict) -> None:
    required = [
        "assessment_id",
        "address",
        "latitude",
        "longitude",
        "wildfire_risk_score",
        "insurance_readiness_score",
        "submodel_scores",
        "weighted_contributions",
        "factor_breakdown",
        "readiness_factors",
        "readiness_blockers",
        "readiness_summary",
        "mitigation_plan",
        "model_version",
    ]
    for key in required:
        assert key in body

    assert 0.0 <= body["wildfire_risk_score"] <= 100.0
    assert 0.0 <= body["insurance_readiness_score"] <= 100.0

    for sm in [
        "ember_exposure",
        "flame_contact_exposure",
        "topography_risk",
        "fuel_proximity_risk",
        "vegetation_intensity_risk",
        "historic_fire_risk",
        "home_hardening_risk",
        "defensible_space_risk",
    ]:
        assert sm in body["submodel_scores"]
        assert "score" in body["submodel_scores"][sm]
        assert "explanation" in body["submodel_scores"][sm]
        assert "key_contributing_inputs" in body["submodel_scores"][sm]

    for rec in body["mitigation_plan"]:
        assert "title" in rec
        assert "impacted_submodels" in rec
        assert isinstance(rec["impacted_submodels"], list)


def _run(payload: dict) -> dict:
    res = client.post("/risk/assess", json=payload)
    assert res.status_code == 200
    return res.json()


def test_low_risk_profile(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=20.0, wildland=20.0, historic=10.0))

    payload = {
        "address": "123 Safe St, Boulder, CO",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 40,
            "construction_year": 2020,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)
    assert body["model_version"] == MODEL_VERSION
    assert body["wildfire_risk_score"] < 35
    assert body["insurance_readiness_score"] > 75


def test_medium_risk_profile(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=55.0, historic=50.0))

    payload = {
        "address": "456 Mid St, Denver, CO",
        "attributes": {
            "defensible_space_ft": 20,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)
    assert 35 <= body["wildfire_risk_score"] <= 75
    assert body["confidence_score"] < 90


def test_high_risk_profile(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=85.0, wildland=90.0, historic=80.0))

    payload = {
        "address": "789 High St, Colorado Springs, CO",
        "attributes": {
            "roof_type": "wood",
            "vent_type": "standard",
            "defensible_space_ft": 3,
            "construction_year": 1990,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)
    assert body["wildfire_risk_score"] > 75
    assert body["readiness_blockers"]
    assert "Combustible roof material" in body["readiness_blockers"]


def test_weak_structure_moderate_environment(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=45.0, wildland=45.0, historic=40.0))

    payload = {
        "address": "100 Structure Risk Ln",
        "attributes": {
            "roof_type": "wood",
            "vent_type": "standard",
            "defensible_space_ft": 8,
            "construction_year": 1985,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)

    assert body["submodel_scores"]["home_hardening_risk"]["score"] >= 65
    assert body["insurance_readiness_score"] < 65
    assert any("home_hardening_risk" in rec["impacted_submodels"] for rec in body["mitigation_plan"])


def test_strong_structure_high_environment(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=88.0, wildland=92.0, historic=85.0))

    payload = {
        "address": "200 Env Pressure Ave",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 45,
            "construction_year": 2022,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)

    assert body["submodel_scores"]["home_hardening_risk"]["score"] < 40
    assert body["submodel_scores"]["fuel_proximity_risk"]["score"] > 70
    assert body["wildfire_risk_score"] > 60


def test_deterministic_outputs_for_fixed_inputs(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=60.0, wildland=65.0, historic=55.0))

    payload = {
        "address": "Deterministic Case Way",
        "attributes": {
            "defensible_space_ft": 20,
            "roof_type": "class a",
            "vent_type": "ember-resistant",
        },
    }

    a = _run(payload)
    b = _run(payload)

    for field in [
        "wildfire_risk_score",
        "insurance_readiness_score",
        "submodel_scores",
        "weighted_contributions",
        "readiness_blockers",
        "readiness_summary",
    ]:
        assert a[field] == b[field]


def test_legacy_row_without_model_version_is_readable(tmp_path):
    db_path = tmp_path / "legacy.db"
    store = AssessmentStore(str(db_path))

    legacy_payload = {
        "assessment_id": "legacy-1",
        "address": "Legacy Address",
        "latitude": 40.0,
        "longitude": -105.0,
        "wildfire_risk_score": 42.0,
        "insurance_readiness_score": 61.0,
        "risk_drivers": {"environmental": 50.0, "structural": 35.0, "access_exposure": 20.0},
        "assumptions_used": ["legacy assumption"],
        "data_sources": ["legacy source"],
        "mitigation_plan": [],
        "explanation": "legacy explanation",
    }

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO assessments (assessment_id, created_at, payload_json, model_version) VALUES (?, datetime('now'), ?, ?)",
            ("legacy-1", json.dumps(legacy_payload), LEGACY_MODEL_VERSION),
        )

    loaded = store.get("legacy-1")
    assert loaded is not None
    assert loaded.model_version == LEGACY_MODEL_VERSION
    assert loaded.factor_breakdown.access_included_in_total is False
