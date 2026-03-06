from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.version import LEGACY_MODEL_VERSION, MODEL_VERSION
from backend.wildfire_data import WildfireContext


client = TestClient(app_main.app)
FIXTURE_DIR = Path(__file__).parent / "fixtures"


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


def _assert_common(response_json: dict) -> None:
    required = [
        "assessment_id",
        "address",
        "latitude",
        "longitude",
        "wildfire_risk_score",
        "insurance_readiness_score",
        "risk_drivers",
        "factor_breakdown",
        "top_risk_drivers",
        "top_protective_factors",
        "explanation_summary",
        "observed_inputs",
        "inferred_inputs",
        "missing_inputs",
        "assumptions_used",
        "confidence_score",
        "low_confidence_flags",
        "data_sources",
        "mitigation_plan",
        "model_version",
        "scoring_notes",
    ]
    for key in required:
        assert key in response_json


def _assert_subset(actual, expected) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for k, v in expected.items():
            assert k in actual
            _assert_subset(actual[k], v)
        return

    if isinstance(expected, list):
        assert actual == expected
        return

    assert actual == expected


def _assert_matches_fixture(body: dict, fixture_name: str) -> None:
    expected = json.loads((FIXTURE_DIR / fixture_name).read_text())
    _assert_subset(body, expected)


def test_low_wildfire_risk_scenario(monkeypatch, tmp_path):
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
    res = client.post("/risk/assess", json=payload)
    assert res.status_code == 200

    body = res.json()
    _assert_common(body)
    assert body["model_version"] == MODEL_VERSION
    assert body["factor_breakdown"]["access_included_in_total"] is False
    _assert_matches_fixture(body, "low_risk_baseline.json")


def test_medium_wildfire_risk_scenario(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=55.0, wildland=55.0, historic=50.0))

    payload = {
        "address": "456 Mid St, Denver, CO",
        "attributes": {
            "defensible_space_ft": 20,
        },
    }
    res = client.post("/risk/assess", json=payload)
    assert res.status_code == 200

    body = res.json()
    _assert_common(body)
    assert body["confidence_score"] < 90.0
    _assert_matches_fixture(body, "medium_risk_baseline.json")


def test_high_wildfire_risk_scenario(monkeypatch, tmp_path):
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
    res = client.post("/risk/assess", json=payload)
    assert res.status_code == 200

    body = res.json()
    _assert_common(body)
    assert "Access risk is provisional" in body["explanation_summary"]
    _assert_matches_fixture(body, "high_risk_baseline.json")


def test_provisional_access_not_weighted(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda addr: (39.7392, -104.9903, "test-geocoder"))

    contexts = {
        "A": _ctx(env=60.0, wildland=5.0, historic=5.0),
        "B": _ctx(env=60.0, wildland=95.0, historic=95.0),
    }

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: contexts["A"])
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_access_a.db")))
    payload = {"address": "Access Scenario A", "attributes": {"defensible_space_ft": 20}}
    score_a = client.post("/risk/assess", json=payload).json()["wildfire_risk_score"]

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: contexts["B"])
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test_access_b.db")))
    payload = {"address": "Access Scenario B", "attributes": {"defensible_space_ft": 20}}
    score_b = client.post("/risk/assess", json=payload).json()["wildfire_risk_score"]

    assert score_a == score_b


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
    assert loaded.scoring_notes
