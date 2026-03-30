from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext


client = TestClient(app_main.app)


def _ctx() -> WildfireContext:
    return WildfireContext(
        environmental_index=54.0,
        slope_index=51.0,
        aspect_index=50.0,
        fuel_index=56.0,
        moisture_index=49.0,
        canopy_index=53.0,
        wildland_distance_index=45.0,
        historic_fire_index=41.0,
        burn_probability_index=52.0,
        hazard_severity_index=55.0,
        burn_probability=52.0,
        wildfire_hazard=55.0,
        slope=34.0,
        fuel_model=61.0,
        canopy_cover=48.0,
        historic_fire_distance=1.5,
        wildland_distance=140.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["burn_probability", "hazard", "fuel", "canopy", "slope", "fire_history"],
        assumptions=[],
        structure_ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 68.0, "coverage_pct": 63.0, "fuel_presence_proxy": 65.0},
            "zone_5_30_ft": {"vegetation_density": 58.0, "coverage_pct": 56.0, "fuel_presence_proxy": 57.0},
            "zone_30_100_ft": {"vegetation_density": 52.0, "coverage_pct": 49.0, "fuel_presence_proxy": 50.0},
        },
        property_level_context={
            "footprint_used": True,
            "footprint_status": "used",
            "fallback_mode": "footprint",
            "ring_metrics": {
                "zone_0_5_ft": {"vegetation_density": 68.0},
                "zone_5_30_ft": {"vegetation_density": 58.0},
            },
            "region_id": "missoula_pilot",
        },
    )


def _setup(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _address: (46.8721, -113.9940, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx())
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "homeowner_improvement.db")))


def _assess(address: str, attrs: dict | None = None, confirmed: list[str] | None = None) -> dict:
    response = client.post(
        "/risk/assess",
        json={
            "address": address,
            "attributes": attrs or {},
            "confirmed_fields": confirmed or [],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    return response.json()


def test_homeowner_improvement_options_detect_missing_key_inputs(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    baseline = _assess("20 Missing Inputs Way, Missoula, MT 59802")

    response = client.get(f"/risk/improve/{baseline['assessment_id']}")
    assert response.status_code == 200
    body = response.json()

    missing = set(body.get("missing_key_inputs") or [])
    assert {"roof_type", "vent_type", "defensible_space_condition"} <= missing
    prompts = [row.get("prompt") for row in (body.get("optional_follow_up_inputs") or [])]
    assert any("roof" in str(prompt).lower() for prompt in prompts)
    assert any("vent" in str(prompt).lower() for prompt in prompts)
    assert any("non-combustible" in str(prompt).lower() for prompt in prompts)


def test_homeowner_improvement_rerun_increases_confidence_and_updates_guidance(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    baseline = _assess("21 Improvement Loop, Missoula, MT 59802")

    response = client.post(
        f"/risk/improve/{baseline['assessment_id']}",
        json={
            "attributes": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
            },
            "defensible_space_condition": "good",
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["updated_assessment_id"] != baseline["assessment_id"]
    assert body["confidence_improved"] is True
    assert body["recommendations_adjusted"] is True
    assert float(body["after_summary"]["confidence_score"]) > float(body["before_summary"]["confidence_score"])
    assert "defensible_space_ft" in (body.get("what_changed") or {})

    before_missing = set((body.get("improve_your_result_before") or {}).get("missing_key_inputs") or [])
    after_missing = set((body.get("improve_your_result_after") or {}).get("missing_key_inputs") or [])
    assert "roof_type" in before_missing
    assert "roof_type" not in after_missing
    assert "vent_type" in before_missing
    assert "vent_type" not in after_missing
