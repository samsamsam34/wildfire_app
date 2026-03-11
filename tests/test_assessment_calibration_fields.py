from __future__ import annotations

import json

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext


client = TestClient(app_main.app)


def _context() -> WildfireContext:
    return WildfireContext(
        environmental_index=60.0,
        slope_index=58.0,
        aspect_index=52.0,
        fuel_index=62.0,
        moisture_index=55.0,
        canopy_index=64.0,
        wildland_distance_index=68.0,
        historic_fire_index=44.0,
        burn_probability_index=66.0,
        hazard_severity_index=64.0,
        burn_probability=0.66,
        wildfire_hazard=3.5,
        slope=24.0,
        fuel_model=110.0,
        canopy_cover=57.0,
        historic_fire_distance=1.6,
        wildland_distance=420.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["test"],
        assumptions=[],
        structure_ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 52.0},
            "ring_5_30_ft": {"vegetation_density": 60.0},
            "ring_30_100_ft": {"vegetation_density": 66.0},
            "ring_100_300_ft": {"vegetation_density": 58.0},
        },
        property_level_context={
            "footprint_used": True,
            "footprint_status": "used",
            "fallback_mode": "footprint",
            "ring_metrics": {
                "ring_0_5_ft": {"vegetation_density": 52.0},
                "ring_5_30_ft": {"vegetation_density": 60.0},
                "ring_30_100_ft": {"vegetation_density": 66.0},
                "ring_100_300_ft": {"vegetation_density": 58.0},
            },
        },
    )


def test_assessment_includes_optional_calibration_fields(monkeypatch, tmp_path):
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (46.87, -113.99, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda *_args, **_kwargs: _context())
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "assessment_calibration.db")))

    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "1.0.0",
                "method": "logistic",
                "parameters": {"intercept": -4.0, "slope": 8.0, "x_scale": 100.0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PUBLIC_CALIBRATION_ARTIFACT", str(artifact_path))

    response = client.post(
        "/risk/assess",
        json={
            "address": "123 Calibration Ln, Missoula, MT",
            "attributes": {
                "roof_type": "class a",
                "vent_type": "ember-resistant",
                "defensible_space_ft": 25,
                "construction_year": 2016,
            },
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["calibration_applied"] is True
    assert body["calibration_method"] == "logistic"
    assert body["calibrated_damage_likelihood"] is not None

