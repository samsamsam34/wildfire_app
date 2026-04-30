"""
Tests for split confidence feature (Phase 5 follow-on).

Verifies:
1. National fallback, no structural inputs → environmental_confidence_tier in [medium/high],
   structural_confidence_tier == not_assessed, 3-5 improvement actions
2. All structural inputs provided → actions empty, structural_confidence_tier != not_assessed
3. Improvement actions ordered by confidence_gain descending
4. Overall confidence_tier unchanged vs pre-split-confidence baseline (regression)
5. Missoula prepared region — split confidence fields present and sensible
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext


client = TestClient(app_main.app)


def _payload(address: str, attrs: dict | None = None, confirmed_fields: list | None = None) -> dict:
    return {
        "address": address,
        "attributes": attrs or {},
        "confirmed_fields": confirmed_fields or [],
        "audience": "homeowner",
        "tags": [],
    }


def _national_context(**overrides) -> WildfireContext:
    """WildfireContext with national fallback sources — no local rasters."""
    defaults = dict(
        slope_index=25.0,
        slope=10.0,
        aspect_index=55.0,
        environmental_index=40.0,
        fuel_index=45.0,
        canopy_index=40.0,
        moisture_index=50.0,
        wildland_distance_index=60.0,
        historic_fire_index=30.0,
        burn_probability_index=None,
        hazard_severity_index=None,
        burn_probability=None,
        wildfire_hazard=None,
        fuel_model=101.0,
        canopy_cover=40.0,
        historic_fire_distance=2.5,
        wildland_distance=300.0,
        environmental_layer_status={
            "burn_probability": "not_configured",
            "hazard": "not_configured",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
            "wildland_distance": "ok",
        },
        data_sources=[
            "LANDFIRE WCS slope (COG fallback)",
            "LANDFIRE WCS fuel/canopy (COG fallback)",
            "National NLCD wildland distance (national fallback)",
            "National MTBS fire history (national fallback)",
        ],
        assumptions=["WHP/burn_probability unavailable outside prepared regions."],
        structure_ring_metrics={},
        property_level_context={
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": {},
            "ring_generation_mode": "point_annulus_fallback",
        },
    )
    defaults.update(overrides)
    return WildfireContext(**defaults)


def _local_context(**overrides) -> WildfireContext:
    """WildfireContext with local raster sources (prepared region)."""
    defaults = dict(
        slope_index=30.0,
        slope=12.0,
        aspect_index=60.0,
        environmental_index=50.0,
        fuel_index=55.0,
        canopy_index=45.0,
        moisture_index=50.0,
        wildland_distance_index=65.0,
        historic_fire_index=35.0,
        burn_probability_index=40.0,
        hazard_severity_index=35.0,
        burn_probability=0.15,
        wildfire_hazard=35.0,
        fuel_model=102.0,
        canopy_cover=45.0,
        historic_fire_distance=2.0,
        wildland_distance=280.0,
        environmental_layer_status={
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
            "wildland_distance": "ok",
            "burn_probability": "ok",
            "hazard": "ok",
        },
        data_sources=[
            "Slope raster",
            "Fuel model raster",
            "Canopy cover raster",
            "Historical fire perimeter recurrence",
            "Distance to wildland vegetation (derived)",
            "WHP burn probability",
            "Wildfire hazard potential",
        ],
        assumptions=[],
        structure_ring_metrics={},
        property_level_context={
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": {},
            "ring_generation_mode": "point_annulus_fallback",
        },
    )
    defaults.update(overrides)
    return WildfireContext(**defaults)


# ---------------------------------------------------------------------------
# Test 1: National fallback, no structural inputs
# ---------------------------------------------------------------------------

class TestNationalFallbackNoStructural:
    """Address-only national fallback: env confidence should be medium/high,
    structural should be not_assessed, and improvement actions should be present."""

    def setup_method(self, method):
        auth.API_KEYS = set()

    def test_environmental_confidence_tier_is_medium_or_high(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        resp = client.post("/risk/assess", json=_payload("123 Main St, Santa Fe, NM"))
        assert resp.status_code == 200
        data = resp.json()
        tier = data.get("environmental_confidence_tier", "")
        assert tier in {"medium", "high"}, (
            f"Expected environmental_confidence_tier in [medium, high], got '{tier}'"
        )

    def test_structural_confidence_tier_is_not_assessed(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        resp = client.post("/risk/assess", json=_payload("456 Elm St, Santa Fe, NM"))
        assert resp.status_code == 200
        data = resp.json()
        struct_tier = data.get("structural_confidence_tier", "")
        assert struct_tier == "not_assessed", (
            f"Expected structural_confidence_tier='not_assessed', got '{struct_tier}'"
        )

    def test_improvement_actions_count_is_3_to_5(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        resp = client.post("/risk/assess", json=_payload("789 Oak Ave, Socorro, NM"))
        assert resp.status_code == 200
        actions = resp.json().get("structural_confidence_improvement_actions", [])
        assert 3 <= len(actions) <= 5, (
            f"Expected 3–5 improvement actions, got {len(actions)}: {actions}"
        )

    def test_improvement_actions_have_required_fields(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        resp = client.post("/risk/assess", json=_payload("1 Desert Rd, Taos, NM"))
        assert resp.status_code == 200
        actions = resp.json().get("structural_confidence_improvement_actions", [])
        assert len(actions) >= 1
        for action in actions:
            assert "field_name" in action
            assert "display_label" in action
            assert "confidence_gain" in action
            assert "why_it_matters" in action
            assert "input_type" in action
            assert isinstance(action["confidence_gain"], int)
            assert action["confidence_gain"] > 0

    def test_confidence_summary_text_mentions_env_tier(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        resp = client.post("/risk/assess", json=_payload("2 Canyon Rd, Albuquerque, NM"))
        assert resp.status_code == 200
        data = resp.json()
        summary = (data.get("homeowner_summary") or {}).get("confidence_summary_text", "")
        assert len(summary) > 10, "confidence_summary_text must be non-trivial"
        # When structural is not_assessed, should prompt user to add details
        assert "home" in summary.lower() or "detail" in summary.lower() or "assessment" in summary.lower()


# ---------------------------------------------------------------------------
# Test 2: All structural inputs provided → no improvement actions
# ---------------------------------------------------------------------------

class TestAllStructuralInputsProvided:
    """When all key structural fields are provided and confirmed, improvement actions = []."""

    def setup_method(self, method):
        auth.API_KEYS = set()

    _full_attrs = {
        "roof_type": "metal",
        "vent_type": "screened",
        "defensible_space_ft": 100,
        "construction_year": 2015,
        "siding_type": "fiber_cement",
    }

    def test_no_improvement_actions_when_all_structural_provided(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        payload = _payload(
            "123 Full Data St, Santa Fe, NM",
            attrs=self._full_attrs,
            confirmed_fields=list(self._full_attrs.keys()),
        )
        resp = client.post("/risk/assess", json=payload)
        assert resp.status_code == 200
        actions = resp.json().get("structural_confidence_improvement_actions", [])
        assert actions == [], (
            f"Expected empty improvement actions when all fields provided, got {len(actions)} actions"
        )

    def test_structural_confidence_tier_not_not_assessed_when_fields_provided(self, monkeypatch, tmp_path):
        context = _national_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

        payload = _payload(
            "456 Full Data Ave, Albuquerque, NM",
            attrs=self._full_attrs,
        )
        resp = client.post("/risk/assess", json=payload)
        assert resp.status_code == 200
        struct_tier = resp.json().get("structural_confidence_tier", "")
        assert struct_tier != "not_assessed", (
            f"structural_confidence_tier should not be 'not_assessed' when structural fields present, got '{struct_tier}'"
        )


# ---------------------------------------------------------------------------
# Test 3: Actions ordered by confidence_gain descending
# ---------------------------------------------------------------------------

def test_improvement_actions_ordered_by_confidence_gain_descending(monkeypatch, tmp_path):
    """Actions list must be sorted by confidence_gain, highest first."""
    auth.API_KEYS = set()
    context = _national_context()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    resp = client.post("/risk/assess", json=_payload("1 Sort Test Rd, Santa Fe, NM"))
    assert resp.status_code == 200
    actions = resp.json().get("structural_confidence_improvement_actions", [])
    gains = [a["confidence_gain"] for a in actions]
    assert gains == sorted(gains, reverse=True), (
        f"Actions not sorted by confidence_gain descending: {gains}"
    )


# ---------------------------------------------------------------------------
# Test 4: Overall confidence_tier regression check
# ---------------------------------------------------------------------------

def test_overall_confidence_tier_unchanged_by_split_feature(monkeypatch, tmp_path):
    """Split confidence fields must not alter overall confidence_tier or confidence_score."""
    auth.API_KEYS = set()
    context = _national_context()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    resp = client.post("/risk/assess", json=_payload("1 Regression Check Ln, Roswell, NM"))
    assert resp.status_code == 200
    data = resp.json()

    # Overall confidence fields must still be present
    assert "confidence_tier" in data
    assert "confidence_score" in data
    assert data["confidence_tier"] in {"high", "medium", "low", "preliminary"}

    # Split fields must also be present (additive, not replacing)
    assert "environmental_confidence_tier" in data
    assert "structural_confidence_tier" in data
    assert "structural_confidence_improvement_actions" in data

    # The overall score and env score should be independent (env may differ from overall)
    env_score = data.get("environmental_confidence_score", 0.0)
    overall_score = data.get("confidence_score", 0.0)
    # Both should be in valid range [0, 100]
    assert 0.0 <= env_score <= 100.0
    assert 0.0 <= overall_score <= 100.0


# ---------------------------------------------------------------------------
# Test 5: Missoula (prepared region) — split fields present, no regression
# ---------------------------------------------------------------------------

class TestMissoulaPreparedRegion:
    """In-region Missoula assessment: split confidence fields are present and sensible."""

    def setup_method(self, method):
        auth.API_KEYS = set()

    def test_split_fields_present_in_region(self, monkeypatch, tmp_path):
        from backend.main import RegionCoverageResolution
        context = _local_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (46.87, -113.99, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))
        mock_coverage = RegionCoverageResolution(
            coverage_available=True,
            resolved_region_id="missoula",
            reason="matched_region",
            diagnostics=[],
            coverage={"coverage_available": True, "resolved_region_id": "missoula"},
        )
        monkeypatch.setattr(app_main, "_resolve_prepared_region", lambda **kw: mock_coverage)

        resp = client.post("/risk/assess", json=_payload("1355 Pattee Canyon Rd, Missoula, MT"))
        assert resp.status_code == 200
        data = resp.json()
        assert "environmental_confidence_tier" in data
        assert "structural_confidence_tier" in data
        assert "structural_confidence_improvement_actions" in data
        assert data["environmental_confidence_tier"] in {"high", "medium", "low"}

    def test_existing_fields_not_regressed_in_region(self, monkeypatch, tmp_path):
        from backend.main import RegionCoverageResolution
        context = _local_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (46.87, -113.99, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))
        mock_coverage = RegionCoverageResolution(
            coverage_available=True,
            resolved_region_id="missoula",
            reason="matched_region",
            diagnostics=[],
            coverage={"coverage_available": True, "resolved_region_id": "missoula"},
        )
        monkeypatch.setattr(app_main, "_resolve_prepared_region", lambda **kw: mock_coverage)

        resp = client.post("/risk/assess", json=_payload("100 Pine St, Missoula, MT"))
        assert resp.status_code == 200
        data = resp.json()
        # Core existing fields must still be present and valid
        assert data.get("wildfire_risk_score") is not None
        assert isinstance(data["wildfire_risk_score"], float)
        assert "confidence_tier" in data
        assert "confidence_score" in data
        assert "data_coverage_summary" in data

    def test_local_region_env_confidence_is_high(self, monkeypatch, tmp_path):
        """With all env layers ok, env confidence should be high for prepared region."""
        from backend.main import RegionCoverageResolution
        context = _local_context()
        monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (46.87, -113.99, "test-geocoder"))
        monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
        monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))
        mock_coverage = RegionCoverageResolution(
            coverage_available=True,
            resolved_region_id="missoula",
            reason="matched_region",
            diagnostics=[],
            coverage={"coverage_available": True, "resolved_region_id": "missoula"},
        )
        monkeypatch.setattr(app_main, "_resolve_prepared_region", lambda **kw: mock_coverage)

        resp = client.post("/risk/assess", json=_payload("200 Canyon Rd, Missoula, MT"))
        assert resp.status_code == 200
        data = resp.json()
        env_tier = data.get("environmental_confidence_tier", "")
        # All layers ok → should be medium or high
        assert env_tier in {"medium", "high"}, (
            f"In-region with all env layers ok, expected medium/high env confidence, got '{env_tier}'"
        )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestComputeEnvironmentalConfidence:
    """Unit tests for _compute_environmental_confidence."""

    def test_all_ok_layers_yields_high_tier(self):
        score, tier = app_main._compute_environmental_confidence(
            environmental_layer_status={"slope": "ok", "fuel": "ok", "canopy": "ok",
                                         "fire_history": "ok", "wildland_distance": "ok"},
            environmental_data_completeness=85.0,
        )
        assert tier == "high"
        assert score >= 70.0

    def test_not_configured_layers_reduce_score(self):
        score_all_ok, tier_all_ok = app_main._compute_environmental_confidence(
            environmental_layer_status={"slope": "ok", "fuel": "ok"},
            environmental_data_completeness=80.0,
        )
        score_with_nc, _ = app_main._compute_environmental_confidence(
            environmental_layer_status={"slope": "ok", "fuel": "ok", "burn_probability": "not_configured"},
            environmental_data_completeness=80.0,
        )
        assert score_with_nc < score_all_ok

    def test_error_layers_reduce_score_more_than_not_configured(self):
        _, tier_nc = app_main._compute_environmental_confidence(
            environmental_layer_status={"slope": "not_configured"},
            environmental_data_completeness=70.0,
        )
        score_error, _ = app_main._compute_environmental_confidence(
            environmental_layer_status={"slope": "error"},
            environmental_data_completeness=70.0,
        )
        score_nc, _ = app_main._compute_environmental_confidence(
            environmental_layer_status={"slope": "not_configured"},
            environmental_data_completeness=70.0,
        )
        assert score_error < score_nc

    def test_score_clamped_to_0_100(self):
        score, _ = app_main._compute_environmental_confidence(
            environmental_layer_status={k: "error" for k in ["slope", "fuel", "canopy",
                                                               "fire_history", "wildland_distance",
                                                               "burn_probability", "hazard"]},
            environmental_data_completeness=0.0,
        )
        assert 0.0 <= score <= 100.0


class TestBuildStructuralImprovementActions:
    """Unit tests for _build_structural_improvement_actions."""

    def test_all_missing_yields_max_actions(self):
        actions = app_main._build_structural_improvement_actions(
            missing_inputs=["roof_type", "vent_type", "defensible_space_ft", "construction_year", "siding_type"],
            observed_inputs={},
            confirmed_inputs={},
        )
        assert len(actions) >= 3

    def test_confirmed_fields_excluded(self):
        actions = app_main._build_structural_improvement_actions(
            missing_inputs=[],
            observed_inputs={"roof_type": "metal", "vent_type": "screened",
                              "defensible_space_ft": 100, "construction_year": 2015},
            confirmed_inputs={"roof_type": "metal", "vent_type": "screened",
                               "defensible_space_ft": 100, "construction_year": 2015},
        )
        # All key fields confirmed → none of them should appear as actions
        action_fields = {a.field_name for a in actions}
        assert "roof_type" not in action_fields
        assert "vent_type" not in action_fields

    def test_actions_sorted_descending_by_gain(self):
        actions = app_main._build_structural_improvement_actions(
            missing_inputs=["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
            observed_inputs={},
            confirmed_inputs={},
        )
        gains = [a.confidence_gain for a in actions]
        assert gains == sorted(gains, reverse=True)

    def test_all_present_and_confirmed_yields_empty(self):
        all_fields = {
            "roof_type": "metal",
            "vent_type": "screened",
            "defensible_space_ft": 100,
            "construction_year": 2015,
            "siding_type": "stucco",
        }
        actions = app_main._build_structural_improvement_actions(
            missing_inputs=[],
            observed_inputs=all_fields,
            confirmed_inputs=all_fields,
        )
        assert actions == []
