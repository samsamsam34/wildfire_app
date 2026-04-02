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
        environmental_index=55.0,
        slope_index=55.0,
        aspect_index=50.0,
        fuel_index=55.0,
        moisture_index=55.0,
        canopy_index=55.0,
        wildland_distance_index=50.0,
        historic_fire_index=40.0,
        burn_probability_index=55.0,
        hazard_severity_index=55.0,
        burn_probability=55.0,
        wildfire_hazard=55.0,
        slope=12.0,
        fuel_model=141.0,
        canopy_cover=28.0,
        historic_fire_distance=1.2,
        wildland_distance=120.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["test-layer"],
        assumptions=[],
        property_level_context={
            "footprint_used": True,
            "footprint_status": "used",
            "fallback_mode": "footprint",
            "structure_match_status": "matched",
            "structure_match_method": "parcel_intersection",
            "candidate_structure_count": 2,
            "parcel_resolution": {
                "status": "matched",
                "confidence": 91.0,
                "source": "County GIS Parcel Fabric",
                "geometry_used": "parcel_polygon",
                "overlap_score": 100.0,
                "candidates_considered": 1,
                "lookup_method": "contains_point",
                "lookup_distance_m": 0.0,
            },
            "footprint_resolution": {
                "selected_source": "openstreetmap_buildings",
                "confidence_score": 0.92,
                "candidates_considered": 2,
                "fallback_used": False,
                "match_status": "matched",
                "match_method": "parcel_intersection",
                "match_distance_m": 0.0,
            },
        },
    )


def test_assessment_response_includes_property_linkage(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    context = _ctx()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon, **_kwargs: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "property_linkage_api.db")))

    response = client.post(
        "/risk/assess",
        json={
            "address": "17 Linkage Test Ln, Missoula, MT 59802",
            "attributes": {"roof_type": "class_a"},
            "confirmed_fields": [],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    linkage = payload.get("property_linkage") or {}
    assert linkage["geocode_confidence"] >= 50.0
    assert linkage["parcel_confidence"] >= 80.0
    assert linkage["footprint_confidence"] >= 80.0
    assert linkage["overall_property_confidence"] >= 60.0
    assert linkage["parcel_status"] == "matched"
    assert linkage["footprint_status"] == "matched"
    assert isinstance((payload.get("property_level_context") or {}).get("property_linkage"), dict)

