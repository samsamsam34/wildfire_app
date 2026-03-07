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


def _payload(address: str, attrs: dict, confirmed: list[str] | None = None) -> dict:
    return {
        "address": address,
        "attributes": attrs,
        "confirmed_fields": confirmed or [],
        "audience": "homeowner",
    }


def _assert_core_contract(body: dict) -> None:
    required = [
        "assessment_id",
        "address",
        "model_version",
        "generated_at",
        "wildfire_risk_score",
        "insurance_readiness_score",
        "submodel_scores",
        "weighted_contributions",
        "submodel_explanations",
        "factor_breakdown",
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
        "low_confidence_flags",
        "mitigation_plan",
        "data_sources",
    ]
    for key in required:
        assert key in body

    assert 0.0 <= body["wildfire_risk_score"] <= 100.0
    assert 0.0 <= body["insurance_readiness_score"] <= 100.0

    for sm in REQUIRED_SUBMODELS:
        assert sm in body["submodel_scores"]
        assert sm in body["factor_breakdown"]["submodels"]
        assert "score" in body["submodel_scores"][sm]
        assert "weighted_contribution" in body["submodel_scores"][sm]


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
    assert confirmed["confirmed_inputs"]["roof_type"] == "class a"
    assert confirmed["confidence_score"] > unconfirmed["confidence_score"]


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
    assert "delta" in sim
    assert sim["delta"]["wildfire_risk_score_delta"] <= 0
    assert sim["delta"]["insurance_readiness_score_delta"] >= 0
    assert "roof_type" in sim["changed_inputs"]
    assert len(sim["next_best_actions"]) > 0




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
    assert "wildfire_risk_score_delta" in rows[0]
def test_reassess_from_existing_assessment(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=60.0, wildland=60.0, historic=55.0))

    original = _run(_payload("100 Reassess Ln", {"defensible_space_ft": 15}))

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


def test_report_export_and_view(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=45.0, wildland=40.0, historic=30.0))
    assessed = _run(
        _payload(
            "200 Report View Dr",
            {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 30},
            confirmed=["roof_type", "vent_type", "defensible_space_ft"],
        )
    )

    export_res = client.get(f"/report/{assessed['assessment_id']}/export")
    assert export_res.status_code == 200
    exported = export_res.json()
    assert "property_summary" in exported
    assert "wildfire_risk_summary" in exported
    assert "insurance_readiness_summary" in exported

    view_res = client.get(f"/report/{assessed['assessment_id']}/view")
    assert view_res.status_code == 200
    assert "text/html" in view_res.headers.get("content-type", "")
    assert "WildfireRisk Advisor Report" in view_res.text


def test_assessments_listing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=35.0, wildland=35.0, historic=20.0))
    _run(_payload("One Listing St", {"defensible_space_ft": 20}))
    _run(_payload("Two Listing St", {"defensible_space_ft": 22}))

    res = client.get("/assessments?limit=5")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) >= 2
    assert "assessment_id" in rows[0]
    assert "created_at" in rows[0]
    assert "model_version" in rows[0]


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
    assert a["factor_breakdown"]["access_risk"] != b["factor_breakdown"]["access_risk"]


def test_deterministic_outputs_for_fixed_inputs(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=60.0, wildland=65.0, historic=55.0))

    payload = _payload(
        "Deterministic Case Way",
        {
            "defensible_space_ft": 20,
            "roof_type": "class a",
            "vent_type": "ember-resistant",
        },
        confirmed=["roof_type", "vent_type", "defensible_space_ft"],
    )

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


def test_step2_calibration_regression(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=20.0, wildland=20.0, historic=10.0))
    body = _run(
        _payload(
            "123 Safe St, Boulder, CO",
            {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 40,
                "construction_year": 2020,
            },
            confirmed=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        )
    )
    _assert_calibration_fixture(body, "step2_calibration_low.json")
    assert body["model_version"] == MODEL_VERSION
