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
        "submodel_explanations",
        "factor_breakdown",
        "readiness_factors",
        "readiness_blockers",
        "readiness_penalties",
        "readiness_summary",
        "mitigation_plan",
        "model_version",
    ]
    for key in required:
        assert key in body

    assert 0.0 <= body["wildfire_risk_score"] <= 100.0
    assert 0.0 <= body["insurance_readiness_score"] <= 100.0

    for sm in REQUIRED_SUBMODELS:
        assert sm in body["submodel_scores"]
        assert "score" in body["submodel_scores"][sm]
        assert "weighted_contribution" in body["submodel_scores"][sm]
        assert "explanation" in body["submodel_scores"][sm]
        assert "key_inputs" in body["submodel_scores"][sm]
        assert "assumptions" in body["submodel_scores"][sm]

    assert "submodels" in body["factor_breakdown"]
    assert "environmental" in body["factor_breakdown"]
    assert "structural" in body["factor_breakdown"]
    for sm in REQUIRED_SUBMODELS:
        assert sm in body["factor_breakdown"]["submodels"]

    for rec in body["mitigation_plan"]:
        assert "title" in rec
        assert "impacted_submodels" in rec
        assert "impacted_readiness_factors" in rec
        assert isinstance(rec["impacted_submodels"], list)
        assert isinstance(rec["impacted_readiness_factors"], list)


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


def _core_shape_subset(body: dict) -> dict:
    keys = [
        "model_version",
        "wildfire_risk_score",
        "insurance_readiness_score",
        "submodel_scores",
        "weighted_contributions",
        "factor_breakdown",
        "top_risk_drivers",
        "top_protective_factors",
        "explanation_summary",
        "readiness_factors",
        "readiness_blockers",
        "readiness_penalties",
        "readiness_summary",
        "observed_inputs",
        "inferred_inputs",
        "missing_inputs",
        "assumptions_used",
        "confidence_score",
        "low_confidence_flags",
        "mitigation_plan",
        "data_sources",
    ]
    return {k: body[k] for k in keys}


def _assert_calibration_fixture(body: dict, fixture_name: str) -> None:
    expected = json.loads((FIXTURE_DIR / fixture_name).read_text())
    _subset_assert(body, expected)


def test_assess_and_report_core_shape_match(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=35.0, wildland=40.0, historic=20.0))
    payload = {
        "address": "101 Contract Way, Boulder, CO",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 32,
            "construction_year": 2018,
        },
    }

    assessed = _run(payload)
    report_res = client.get(f"/report/{assessed['assessment_id']}")
    assert report_res.status_code == 200
    report = report_res.json()

    _assert_core_contract(report)
    assert _core_shape_subset(assessed) == _core_shape_subset(report)


def test_low_environment_strong_structure(monkeypatch, tmp_path):
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
    _assert_calibration_fixture(body, "step2_calibration_low.json")


def test_moderate_environment_weak_structure(monkeypatch, tmp_path):
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
    _assert_calibration_fixture(body, "step2_calibration_medium.json")


def test_high_environment_strong_structure(monkeypatch, tmp_path):
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
    assert body["submodel_scores"]["structure_vulnerability_risk"]["score"] < 40
    assert body["submodel_scores"]["fuel_proximity_risk"]["score"] > 70
    assert body["wildfire_risk_score"] > 60


def test_high_fuel_proximity_poor_defensible_space(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=70.0, wildland=95.0, historic=60.0))
    payload = {
        "address": "Fuel Edge Case",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 4,
            "construction_year": 2015,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)
    assert "Severely inadequate defensible space" in body["readiness_blockers"]
    assert body["submodel_scores"]["fuel_proximity_risk"]["score"] > 80
    assert any("defensible_space_risk" in rec["impacted_submodels"] for rec in body["mitigation_plan"])


def test_strong_structure_extreme_ember_topography(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=95.0, wildland=75.0, historic=85.0))
    payload = {
        "address": "Topography Ember Case",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 35,
            "construction_year": 2020,
        },
    }
    body = _run(payload)
    _assert_core_contract(body)
    assert body["submodel_scores"]["structure_vulnerability_risk"]["score"] < 45
    assert body["submodel_scores"]["ember_exposure_risk"]["score"] >= 60
    assert body["submodel_scores"]["slope_topography_risk"]["score"] >= 70


def test_high_risk_profile_blockers_and_fixture(monkeypatch, tmp_path):
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
    assert "Combustible roof material" in body["readiness_blockers"]
    _assert_calibration_fixture(body, "step2_calibration_high.json")


def test_provisional_access_not_in_wildfire_score(monkeypatch, tmp_path):
    auth.API_KEYS = set()

    payload = {
        "address": "Access Control",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 20,
            "construction_year": 2015,
        },
    }

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
    assert a["factor_breakdown"]["access_risk"] != b["factor_breakdown"]["access_risk"]


def test_debug_endpoint_returns_intermediate_payload(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=60.0, wildland=65.0, historic=55.0))

    payload = {
        "address": "Debug Case Way",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 20,
        },
    }
    res = client.post("/risk/debug", json=payload)
    assert res.status_code == 200
    body = res.json()

    assert "context_indices" in body
    assert "submodel_scores" in body
    assert "weighted_contributions" in body
    assert "readiness" in body
    assert "penalties" in body["readiness"]
    assert "config" in body


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
        "readiness_penalties",
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
