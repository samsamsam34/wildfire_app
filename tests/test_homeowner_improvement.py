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
    prioritized = list(body.get("prioritized_missing_key_inputs") or [])
    assert 1 <= len(prioritized) <= 3
    assert set(prioritized).issubset(missing)
    assert isinstance(body.get("remaining_optional_input_count"), int)
    next_q = body.get("highest_value_next_question") or {}
    assert isinstance(next_q, dict)
    assert str(next_q.get("input_key") or "") in set(prioritized)
    prompts = [row.get("prompt") for row in (body.get("optional_follow_up_inputs") or [])]
    assert any("roof" in str(prompt).lower() for prompt in prompts)
    assert any(
        ("vent" in str(prompt).lower())
        or ("building polygon" in str(prompt).lower())
        or ("draw your building outline" in str(prompt).lower())
        for prompt in prompts
    )
    assert any(
        ("non-combustible" in str(prompt).lower()) or ("map pin" in str(prompt).lower())
        for prompt in prompts
    )
    joined_suggestions = " ".join(str(item).lower() for item in (body.get("improve_your_result_suggestions") or []))
    assert "roof type" in joined_suggestions or "vent" in joined_suggestions


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
    concise = body.get("what_changed_summary") or {}
    assert isinstance(concise.get("score_direction"), dict)
    assert isinstance(concise.get("specificity_change"), dict)
    assert isinstance(concise.get("confidence_change"), dict)
    assert isinstance(concise.get("recommendation_changes"), dict)

    before_missing = set((body.get("improve_your_result_before") or {}).get("missing_key_inputs") or [])
    after_missing = set((body.get("improve_your_result_after") or {}).get("missing_key_inputs") or [])
    assert "roof_type" in before_missing
    assert "roof_type" not in after_missing
    assert "vent_type" in before_missing
    assert "vent_type" not in after_missing


def test_homeowner_map_point_correction_can_improve_specificity_and_trust(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _address: (46.8721, -113.9940, "test-geocoder"))

    def _collect(_lat, _lon, **kwargs):
        has_anchor = bool(kwargs.get("property_anchor_point") or kwargs.get("user_selected_point"))
        context = _ctx()
        if has_anchor:
            context.property_level_context.update(
                {
                    "footprint_used": True,
                    "footprint_status": "used",
                    "fallback_mode": "footprint",
                    "geometry_basis": "footprint",
                    "structure_match_status": "matched",
                        "ring_generation_mode": "footprint_aware_rings",
                        "property_anchor_source": "user_selected_point",
                        "property_anchor_quality_score": 0.92,
                        "anchor_quality_score": 0.92,
                        "parcel_id": "parcel-22",
                        "parcel_lookup_method": "contains_point",
                        "naip_feature_source": "prepared_region_naip",
                        "near_structure_vegetation_0_5_pct": 45.0,
                    "near_structure_vegetation_5_30_pct": 51.0,
                    "canopy_adjacency_proxy_pct": 48.0,
                    "vegetation_continuity_proxy_pct": 52.0,
                }
            )
        else:
            context.property_level_context.update(
                {
                    "footprint_used": False,
                    "footprint_status": "not_found",
                    "fallback_mode": "point_based",
                    "geometry_basis": "point",
                    "structure_match_status": "none",
                    "ring_generation_mode": "point_annulus_fallback",
                    "property_anchor_source": "approximate_geocode",
                    "property_anchor_quality_score": 0.31,
                    "anchor_quality_score": 0.31,
                    "naip_feature_source": None,
                    "near_structure_vegetation_0_5_pct": None,
                    "near_structure_vegetation_5_30_pct": None,
                    "canopy_adjacency_proxy_pct": None,
                    "vegetation_continuity_proxy_pct": None,
                }
            )
        return context

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", _collect)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "homeowner_improvement_anchor.db")))

    baseline = _assess("22 Anchor Improvement Way, Missoula, MT 59802", attrs={"roof_type": "class a"}, confirmed=["roof_type"])
    baseline_specificity = (baseline.get("specificity_summary") or {}).get("specificity_tier")
    baseline_trust = ((baseline.get("homeowner_summary") or {}).get("trust_summary") or {})
    assert baseline_trust.get("geometry_specificity_limited") is True
    baseline_confidence_score = float(baseline.get("confidence_score") or 0.0)
    baseline_diff_score = float(
        baseline_trust.get("local_differentiation_score")
        or baseline_trust.get("neighborhood_differentiation_confidence")
        or 0.0
    )

    options_response = client.get(f"/risk/improve/{baseline['assessment_id']}")
    assert options_response.status_code == 200
    options = options_response.json()
    option_flags = set(options.get("geometry_issue_flags") or [])
    assert {"missing_footprint", "low_anchor_confidence", "parcel_mismatch"} <= option_flags
    option_keys = set(options.get("missing_key_inputs") or [])
    assert {"map_point_correction", "select_building_polygon", "draw_structure_manually"} <= option_keys
    option_text = " ".join(str(row).lower() for row in (options.get("improve_your_result_suggestions") or []))
    assert "move pin to your home" in option_text
    assert "confirm building location" in option_text

    response = client.post(
        f"/risk/improve/{baseline['assessment_id']}",
        json={
            "attributes": {},
            "property_anchor_point": {"latitude": 46.87215, "longitude": -113.99385},
            "confirmed_fields": [],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("confidence_improved") is True
    assert body.get("what_changed", {}).get("map_point_correction") is not None
    concise = body.get("what_changed_summary") or {}
    assert (concise.get("specificity_change") or {}).get("changed") in {True, False}
    assert (concise.get("confidence_change") or {}).get("score_direction") in {"up", "down", "unchanged", "unknown"}
    assert isinstance(concise.get("differentiation_change"), dict)
    assert isinstance(concise.get("geometry_change"), dict)

    updated_report = client.get(f"/report/{body['updated_assessment_id']}")
    assert updated_report.status_code == 200
    updated = updated_report.json()
    updated_specificity = (updated.get("specificity_summary") or {}).get("specificity_tier")
    updated_trust = ((updated.get("homeowner_summary") or {}).get("trust_summary") or {})
    updated_confidence_score = float(updated.get("confidence_score") or 0.0)
    updated_diff_score = float(
        updated_trust.get("local_differentiation_score")
        or updated_trust.get("neighborhood_differentiation_confidence")
        or 0.0
    )
    assert baseline_specificity in {"regional_estimate", "address_level", "insufficient_data", "property_specific"}
    assert updated_specificity in {"property_specific", "address_level"}
    assert updated_trust.get("geometry_specificity_limited") is False
    assert updated_confidence_score > baseline_confidence_score
    assert updated_diff_score > baseline_diff_score


def test_homeowner_polygon_geometry_correction_updates_what_changed_and_improves_confidence(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _address: (46.8721, -113.9940, "test-geocoder"))

    def _collect(_lat, _lon, **kwargs):
        context = _ctx()
        has_polygon = isinstance(kwargs.get("selected_structure_geometry"), dict)
        if has_polygon:
            context.property_level_context.update(
                {
                    "footprint_used": True,
                    "footprint_status": "used",
                    "fallback_mode": "footprint",
                    "geometry_basis": "footprint",
                    "structure_match_status": "matched",
                    "ring_generation_mode": "footprint_aware_rings",
                    "property_anchor_source": "user_selected_polygon",
                    "property_anchor_quality_score": 0.94,
                    "anchor_quality_score": 0.94,
                    "parcel_id": "parcel-33",
                    "parcel_lookup_method": "contains_point",
                    "naip_feature_source": "prepared_region_naip",
                    "near_structure_vegetation_0_5_pct": 40.0,
                    "near_structure_vegetation_5_30_pct": 47.0,
                    "canopy_adjacency_proxy_pct": 42.0,
                    "vegetation_continuity_proxy_pct": 46.0,
                    "final_structure_geometry_source": "user_selected_polygon",
                    "selected_structure_geometry": kwargs.get("selected_structure_geometry"),
                    "selected_structure_id": kwargs.get("selected_structure_id") or "manual-1",
                    "structure_match_confidence": 0.95,
                }
            )
        else:
            context.property_level_context.update(
                {
                    "footprint_used": False,
                    "footprint_status": "not_found",
                    "fallback_mode": "point_based",
                    "geometry_basis": "point",
                    "structure_match_status": "none",
                    "ring_generation_mode": "point_annulus_fallback",
                    "property_anchor_source": "approximate_geocode",
                    "property_anchor_quality_score": 0.33,
                    "anchor_quality_score": 0.33,
                    "naip_feature_source": None,
                    "near_structure_vegetation_0_5_pct": None,
                    "near_structure_vegetation_5_30_pct": None,
                    "canopy_adjacency_proxy_pct": None,
                    "vegetation_continuity_proxy_pct": None,
                }
            )
        return context

    monkeypatch.setattr(app_main.wildfire_data, "collect_context", _collect)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "homeowner_improvement_polygon.db")))

    baseline = _assess("33 Polygon Update Way, Missoula, MT 59802")
    baseline_conf = float(baseline.get("confidence_score") or 0.0)
    baseline_diff = float(
        (((baseline.get("homeowner_summary") or {}).get("trust_summary") or {}).get("local_differentiation_score"))
        or (((baseline.get("homeowner_summary") or {}).get("trust_summary") or {}).get("neighborhood_differentiation_confidence"))
        or 0.0
    )

    selected_structure_geometry = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-113.99410, 46.87210],
                    [-113.99400, 46.87210],
                    [-113.99400, 46.87218],
                    [-113.99410, 46.87218],
                    [-113.99410, 46.87210],
                ]
            ],
        },
        "properties": {"structure_id": "manual-1"},
    }
    response = client.post(
        f"/risk/improve/{baseline['assessment_id']}",
        json={
            "selected_structure_id": "manual-1",
            "selected_structure_geometry": selected_structure_geometry,
            "structure_geometry_source": "user_modified",
            "selection_mode": "polygon",
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("confidence_improved") is True
    assert body.get("what_changed", {}).get("geometry_updated") is True
    assert isinstance(body.get("what_changed", {}).get("score_change"), dict)
    assert isinstance(body.get("what_changed", {}).get("specificity_change"), dict)
    assert isinstance(body.get("what_changed", {}).get("confidence_change"), dict)

    updated_report = client.get(f"/report/{body['updated_assessment_id']}")
    assert updated_report.status_code == 200
    updated = updated_report.json()
    updated_conf = float(updated.get("confidence_score") or 0.0)
    updated_diff = float(
        (((updated.get("homeowner_summary") or {}).get("trust_summary") or {}).get("local_differentiation_score"))
        or (((updated.get("homeowner_summary") or {}).get("trust_summary") or {}).get("neighborhood_differentiation_confidence"))
        or 0.0
    )
    assert updated_conf > baseline_conf
    assert updated_diff > baseline_diff
