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
from backend.building_footprints import BuildingFootprintClient, BuildingFootprintResult, compute_structure_rings
from backend.database import AssessmentStore
from backend.mitigation import build_mitigation_plan
from backend.models import PropertyAttributes
from backend.version import LEGACY_MODEL_VERSION, MODEL_VERSION
from backend.wildfire_data import WildfireContext, WildfireDataClient


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


def _ctx(
    env: float,
    wildland: float,
    historic: float,
    ring_metrics: dict[str, dict[str, float | None]] | None = None,
) -> WildfireContext:
    ring_metrics = ring_metrics or {}
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
        structure_ring_metrics=ring_metrics,
        property_level_context={
            "footprint_used": bool(ring_metrics),
            "ring_metrics": ring_metrics,
        },
    )


def _setup(monkeypatch, tmp_path, context: WildfireContext) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
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
        "submodel_scores",
        "weighted_contributions",
        "submodel_explanations",
        "factor_breakdown",
        "property_findings",
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
        "confidence_tier",
        "use_restriction",
        "low_confidence_flags",
        "mitigation_plan",
        "data_sources",
        "property_level_context",
        "site_hazard_section",
        "home_ignition_vulnerability_section",
        "insurance_readiness_section",
        "review_status",
    ]
    for key in required:
        assert key in body

    assert 0.0 <= body["wildfire_risk_score"] <= 100.0
    assert 0.0 <= body["site_hazard_score"] <= 100.0
    assert 0.0 <= body["home_ignition_vulnerability_score"] <= 100.0
    assert 0.0 <= body["insurance_readiness_score"] <= 100.0
    assert body["confidence_tier"] in {"high", "moderate", "low", "preliminary"}
    assert body["use_restriction"] in {
        "shareable",
        "homeowner_review_recommended",
        "agent_or_inspector_review_recommended",
        "not_for_underwriting_or_binding",
    }

    for sm in REQUIRED_SUBMODELS:
        assert sm in body["submodel_scores"]
        assert sm in body["factor_breakdown"]["submodels"]
        assert "score" in body["submodel_scores"][sm]
        assert "weighted_contribution" in body["submodel_scores"][sm]


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
    assert any("dense vegetation close to the home" == d for d in assessed["top_risk_drivers"])


def test_property_findings_fallback_empty_when_ring_data_missing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ctx(env=45.0, wildland=45.0, historic=40.0, ring_metrics={}))

    assessed = _run(_payload("311 No Ring Data Ln", {"defensible_space_ft": 20}))
    assert "property_findings" in assessed
    assert assessed["property_findings"] == []


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

    expected_blended = round(0.6 * assessed["site_hazard_score"] + 0.4 * assessed["home_ignition_vulnerability_score"], 1)
    assert assessed["wildfire_risk_score"] == expected_blended
    assert assessed["legacy_weighted_wildfire_risk_score"] >= 0
    assert assessed["insurance_readiness_score"] != round(100.0 - assessed["wildfire_risk_score"], 1)


def test_confidence_tier_high_and_shareable_when_inputs_are_strong(monkeypatch, tmp_path):
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 20.0},
        "ring_5_30_ft": {"vegetation_density": 25.0},
        "ring_30_100_ft": {"vegetation_density": 30.0},
    }
    _setup(monkeypatch, tmp_path, _ctx(env=35.0, wildland=35.0, historic=20.0, ring_metrics=ring_metrics))

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
        property_level_context={"footprint_used": False, "footprint_status": "source_unavailable", "ring_metrics": {}},
    )
    _setup(monkeypatch, tmp_path, degraded)

    assessed = _run(_payload("344 Low Confidence Ct", {"defensible_space_ft": 10}))
    assert assessed["confidence_tier"] in {"low", "preliminary"}
    assert assessed["use_restriction"] == "not_for_underwriting_or_binding"


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
    assert "trusted location match is required" in res.text


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
    assert sim["delta"]["wildfire_risk_score_delta"] <= 0
    assert sim["delta"]["insurance_readiness_score_delta"] >= 0
    assert "roof_type" in sim["changed_inputs"]


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

    export_res = client.get(f"/report/{assessed['assessment_id']}/export?audience=insurer")
    assert export_res.status_code == 200
    exported = export_res.json()
    assert exported["audience_mode"] == "insurer"
    assert "audience_focus" in exported

    view_res = client.get(f"/report/{assessed['assessment_id']}/view?audience=agent")
    assert view_res.status_code == 200
    assert "text/html" in view_res.headers.get("content-type", "")
    assert "Audience View agent" in view_res.text


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

    assert set(rings.keys()) == {"ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft"}
    assert not assumptions

    area_0_5 = rings["ring_0_5_ft"].area
    area_5_30 = rings["ring_5_30_ft"].area
    area_30_100 = rings["ring_30_100_ft"].area
    assert area_0_5 > 0
    assert area_5_30 > area_0_5
    assert area_30_100 > area_5_30


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
        lambda _geom: {
            "canopy_mean": 62.0,
            "canopy_max": 81.0,
            "coverage_pct": 58.0,
            "vegetation_density": 70.0,
        },
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda _geom: 50.0)

    context_blob, assumptions, sources = client._compute_structure_ring_metrics(40.0, -105.0)
    metrics = context_blob["ring_metrics"]

    assert context_blob["footprint_used"] is True
    assert context_blob["footprint_status"] == "used"
    assert set(metrics.keys()) == {"ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft"}
    assert metrics["ring_0_5_ft"]["vegetation_density"] == 60.0
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
    assert ctx.property_level_context.get("footprint_status") in {"not_found", "source_unavailable"}
    assert "ring_metrics" in ctx.property_level_context


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
