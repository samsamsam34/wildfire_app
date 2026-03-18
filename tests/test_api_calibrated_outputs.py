from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext

client = TestClient(app_main.app)


def _context() -> WildfireContext:
    return WildfireContext(
        environmental_index=57.0,
        slope_index=55.0,
        aspect_index=51.0,
        fuel_index=63.0,
        moisture_index=49.0,
        canopy_index=60.0,
        wildland_distance_index=58.0,
        historic_fire_index=45.0,
        burn_probability_index=62.0,
        hazard_severity_index=64.0,
        burn_probability=0.62,
        wildfire_hazard=3.4,
        slope=20.0,
        fuel_model=84.0,
        canopy_cover=55.0,
        historic_fire_distance=1.8,
        wildland_distance=220.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["test-calibrated-api"],
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
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda *_args, **_kwargs: _context())
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "calibrated_outputs_test.db")))


def _payload() -> dict[str, object]:
    return {
        "address": "123 Calibrated Metadata Way, Test, CO",
        "attributes": {
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 25,
            "construction_year": 2018,
        },
        "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
        "audience": "homeowner",
    }


def test_default_behavior_unchanged_without_opt_in(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    monkeypatch.delenv("WF_PUBLIC_CALIBRATION_ARTIFACT", raising=False)
    response = client.post("/risk/assess", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert "wildfire_risk_score" in body
    assert (
        "calibrated_public_outcome_metadata" not in body
        or body["calibrated_public_outcome_metadata"] is None
    )


def test_calibrated_metadata_returned_when_requested(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "2.0.0",
                "method": "logistic",
                "parameters": {"intercept": -4.0, "slope": 8.0, "x_scale": 100.0},
                "dataset": {"row_count": 400, "adverse_rate": 0.24},
                "limitations": ["Public outcomes are directional only."],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PUBLIC_CALIBRATION_ARTIFACT", str(artifact_path))

    response = client.post("/risk/assess?include_calibrated_outputs=true", json=_payload())
    assert response.status_code == 200
    body = response.json()
    meta = body.get("calibrated_public_outcome_metadata")
    assert isinstance(meta, dict)
    assert meta["requested"] is True
    assert meta["available"] is True
    assert meta["calibrated_public_outcome_probability"] is not None
    assert meta["calibrated_public_outcome_probability"] == body["calibrated_damage_likelihood"]
    caveat = str(meta.get("calibration_caveat") or "").lower()
    assert "public observed wildfire damage outcomes" in caveat
    assert "not be interpreted as carrier underwriting probability" in caveat


def test_request_body_opt_in_supported(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "2.0.0",
                "method": "logistic",
                "parameters": {"intercept": -4.0, "slope": 8.0, "x_scale": 100.0},
                "dataset": {"row_count": 300, "adverse_rate": 0.21},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PUBLIC_CALIBRATION_ARTIFACT", str(artifact_path))
    payload = _payload()
    payload["include_calibrated_outputs"] = True
    response = client.post("/risk/assess", json=payload)
    assert response.status_code == 200
    meta = response.json().get("calibrated_public_outcome_metadata")
    assert isinstance(meta, dict)
    assert meta["requested"] is True


def test_missing_or_incompatible_artifact_degrades_gracefully(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    monkeypatch.delenv("WF_PUBLIC_CALIBRATION_ARTIFACT", raising=False)
    no_artifact = client.post("/risk/assess?include_calibrated_outputs=true", json=_payload())
    assert no_artifact.status_code == 200
    no_art_meta = no_artifact.json().get("calibrated_public_outcome_metadata")
    assert isinstance(no_art_meta, dict)
    assert no_art_meta["available"] is False
    assert no_art_meta["availability_status"] == "unavailable_no_artifact"

    incompatible_path = tmp_path / "incompatible_calibration.json"
    incompatible_path.write_text(
        json.dumps(
            {
                "artifact_version": "2.0.0",
                "method": "unknown_method",
                "parameters": {},
                "dataset": {"row_count": 50},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PUBLIC_CALIBRATION_ARTIFACT", str(incompatible_path))
    incompatible = client.post("/risk/assess?include_calibrated_outputs=true", json=_payload())
    assert incompatible.status_code == 200
    inc_meta = incompatible.json().get("calibrated_public_outcome_metadata")
    assert isinstance(inc_meta, dict)
    assert inc_meta["available"] is False
    assert inc_meta["availability_status"] == "unavailable_incompatible_artifact"


def test_calibrated_metadata_with_diagnostics_wrapper(monkeypatch, tmp_path: Path) -> None:
    _setup(monkeypatch, tmp_path)
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "2.0.0",
                "method": "logistic",
                "parameters": {"intercept": -3.5, "slope": 7.0, "x_scale": 100.0},
                "dataset": {"row_count": 520, "adverse_rate": 0.19},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PUBLIC_CALIBRATION_ARTIFACT", str(artifact_path))
    response = client.post(
        "/risk/assess?include_diagnostics=true&include_calibrated_outputs=true",
        json=_payload(),
    )
    assert response.status_code == 200
    body = response.json()
    assessment = body.get("assessment")
    diagnostics = body.get("diagnostics")
    assert isinstance(assessment, dict)
    assert isinstance(diagnostics, dict)
    meta = assessment.get("calibrated_public_outcome_metadata")
    assert isinstance(meta, dict)
    assert meta["requested"] is True
