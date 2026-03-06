from __future__ import annotations

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
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


def _assert_common(response_json: dict) -> None:
    assert "factor_breakdown" in response_json
    assert "confidence" in response_json
    assert "model_version" in response_json
    assert "top_risk_drivers" in response_json
    assert "top_protective_factors" in response_json
    assert "explanation_summary" in response_json
    assert "risk_scores" in response_json
    assert "assumptions" in response_json


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
    assert body["risk_scores"]["wildfire_risk_score"] == 13.5
    assert body["confidence"]["confidence_score"] >= 90.0


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
    assert body["risk_scores"]["wildfire_risk_score"] == 51.1
    assert 45.0 <= body["confidence"]["confidence_score"] <= 80.0


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
    assert body["risk_scores"]["wildfire_risk_score"] == 87.6
    assert body["confidence"]["confidence_score"] >= 80.0
