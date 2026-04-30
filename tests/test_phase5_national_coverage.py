"""
Phase 5 tests: national coverage and region gate removal.

Verifies that:
1. Out-of-region US addresses proceed to full assessment (no 4xx rejection)
2. data_coverage_summary is present and correctly populated
3. Addresses outside the US return 422 with a clear error
4. _is_us_coordinate() correctly classifies coordinates
5. _build_data_coverage_summary() produces correct coverage classification
6. get_region_context (region_resolution) behavior for in/out-of-region addresses
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext, WildfireDataClient


client = TestClient(app_main.app)


def _payload(address: str, attrs: dict | None = None) -> dict:
    return {
        "address": address,
        "attributes": attrs or {"construction_year": 1990, "roof_material": "tile"},
        "confirmed_fields": [],
        "audience": "homeowner",
        "tags": [],
    }


def _minimal_context(**overrides) -> WildfireContext:
    """Return a WildfireContext with all national-fallback sources tagged."""
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
        burn_probability_index=None,   # WHP — expected None for out-of-region
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


# ---------------------------------------------------------------------------
# Test 1: _is_us_coordinate correctly classifies coordinates
# ---------------------------------------------------------------------------

class TestIsUsCoordinate:
    """Unit tests for _is_us_coordinate boundary check."""

    def test_conus_center_is_us(self):
        assert app_main._is_us_coordinate(40.0, -105.0) is True

    def test_alaska_is_us(self):
        assert app_main._is_us_coordinate(64.2, -153.0) is True

    def test_hawaii_is_us(self):
        assert app_main._is_us_coordinate(21.3, -157.8) is True

    def test_london_is_not_us(self):
        assert app_main._is_us_coordinate(51.5, -0.1) is False

    def test_mexico_city_is_not_us(self):
        # 19.4°N, -99.1°W — outside US bbox (lat < 15)
        assert app_main._is_us_coordinate(14.0, -90.0) is False

    def test_canada_is_not_us(self):
        # 73°N is above the 72° cutoff
        assert app_main._is_us_coordinate(73.0, -100.0) is False

    def test_provo_ut_is_us(self):
        assert app_main._is_us_coordinate(40.23, -111.66) is True

    def test_missoula_mt_is_us(self):
        assert app_main._is_us_coordinate(46.87, -113.99) is True


# ---------------------------------------------------------------------------
# Test 2: _build_data_coverage_summary classification
# ---------------------------------------------------------------------------

class TestBuildDataCoverageSummary:
    def _build(self, coverage_available, data_sources, env_status):
        return app_main._build_data_coverage_summary(
            coverage_available=coverage_available,
            data_sources=data_sources,
            environmental_layer_status=env_status,
        )

    def test_full_local_coverage(self):
        summary = self._build(
            coverage_available=True,
            data_sources=["Slope raster", "Fuel raster", "Canopy cover raster"],
            env_status={"slope": "ok", "fuel": "ok", "canopy": "ok"},
        )
        assert summary.overall_coverage == "full"
        assert summary.local_data_available is True
        assert summary.layers_from_national_sources == []
        assert "high-resolution local data" in summary.coverage_note

    def test_partial_national_coverage(self):
        summary = self._build(
            coverage_available=False,
            data_sources=[
                "LANDFIRE WCS slope (COG fallback)",
                "LANDFIRE WCS fuel/canopy (COG fallback)",
                "National NLCD wildland distance (national fallback)",
            ],
            env_status={
                "slope": "ok",
                "fuel": "ok",
                "canopy": "ok",
                "wildland_distance": "ok",
                "burn_probability": "not_configured",
            },
        )
        assert summary.overall_coverage in {"partial", "limited"}
        assert summary.local_data_available is False
        assert len(summary.layers_from_national_sources) > 0

    def test_limited_coverage_many_missing(self):
        summary = self._build(
            coverage_available=False,
            data_sources=[],
            env_status={
                "slope": "not_configured",
                "fuel": "not_configured",
                "canopy": "not_configured",
                "fire_history": "not_configured",
                "burn_probability": "not_configured",
            },
        )
        assert summary.overall_coverage == "limited"
        assert summary.local_data_available is False
        assert len(summary.layers_unavailable) >= 4

    def test_coverage_note_not_empty(self):
        summary = self._build(
            coverage_available=False,
            data_sources=["USGS 3DEP slope (national COG fallback)"],
            env_status={"slope": "ok", "fuel": "not_configured"},
        )
        assert len(summary.coverage_note) > 10


# ---------------------------------------------------------------------------
# Test 3: Out-of-region US address → 200 + data_coverage_summary present
# ---------------------------------------------------------------------------

def test_out_of_region_us_address_returns_200(monkeypatch, tmp_path):
    """An out-of-region US address (central NM) gets a full assessment, not 4xx."""
    auth.API_KEYS = set()
    context = _minimal_context()
    # Geocode to central New Mexico (no prepared region)
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    response = client.post("/risk/assess", json=_payload("123 Main St, Santa Fe, NM 87501"))
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:500]}"

    data = response.json()
    assert data.get("wildfire_risk_score") is not None, "wildfire_risk_score must be present"
    assert isinstance(data["wildfire_risk_score"], float)

    dcs = data.get("data_coverage_summary")
    assert dcs is not None, "data_coverage_summary must be present in response"
    assert dcs["local_data_available"] is False
    assert dcs["overall_coverage"] in {"partial", "limited"}
    assert "coverage_note" in dcs


# ---------------------------------------------------------------------------
# Test 4: data_coverage_summary shows national sources for out-of-region
# ---------------------------------------------------------------------------

def test_data_coverage_summary_shows_national_layers(monkeypatch, tmp_path):
    """data_coverage_summary.layers_from_national_sources is non-empty for national fallback."""
    auth.API_KEYS = set()
    context = _minimal_context()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    response = client.post("/risk/assess", json=_payload("456 Elm St, Albuquerque, NM 87102"))
    assert response.status_code == 200

    dcs = response.json().get("data_coverage_summary", {})
    assert len(dcs.get("layers_from_national_sources", [])) > 0, (
        "National layers must be listed in data_coverage_summary"
    )


# ---------------------------------------------------------------------------
# Test 5: Confidence tier is not "high" for all-national fallback
# ---------------------------------------------------------------------------

def test_national_fallback_confidence_not_high(monkeypatch, tmp_path):
    """Out-of-region assessment with national fallback should have confidence < high."""
    auth.API_KEYS = set()
    context = _minimal_context()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    response = client.post("/risk/assess", json=_payload("789 Oak Ave, Socorro, NM 87801"))
    assert response.status_code == 200
    data = response.json()
    confidence_tier = data.get("confidence_tier", "")
    assert confidence_tier != "high", (
        f"National fallback assessment should not have confidence_tier='high', got '{confidence_tier}'"
    )


# ---------------------------------------------------------------------------
# Test 6: Coordinates outside US → 422 with clear error message
# ---------------------------------------------------------------------------

def test_outside_us_coordinates_returns_422(monkeypatch, tmp_path):
    """London coordinates return 422 with 'US addresses' error message."""
    auth.API_KEYS = set()
    # Geocode to London
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (51.5, -0.1, "test-geocoder"))
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    response = client.post("/risk/assess", json=_payload("10 Downing St, London, UK"))
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text[:300]}"

    detail = response.json().get("detail", {})
    error_class = detail.get("error_class", "")
    message = detail.get("message", "")
    assert error_class == "outside_us_coverage"
    assert "US" in message or "us" in message.lower()


# ---------------------------------------------------------------------------
# Test 7: In-region address still has local_data_available = True
# ---------------------------------------------------------------------------

def test_in_region_address_has_local_data_available(monkeypatch, tmp_path):
    """An in-region address produces data_coverage_summary.local_data_available=True."""
    auth.API_KEYS = set()
    # Context with local raster sources (no national markers)
    context_local = _minimal_context(
        data_sources=[
            "Slope raster",
            "Fuel model raster",
            "Canopy cover raster",
            "Historical fire perimeter recurrence",
            "Distance to wildland vegetation (derived)",
        ],
        environmental_layer_status={
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
            "wildland_distance": "ok",
        },
    )
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (46.87, -113.99, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context_local)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    # Patch coverage_available = True for Missoula
    from backend.main import RegionCoverageResolution
    mock_coverage = RegionCoverageResolution(
        coverage_available=True,
        resolved_region_id="missoula",
        reason="matched_region",
        diagnostics=[],
        coverage={"coverage_available": True, "resolved_region_id": "missoula"},
    )
    monkeypatch.setattr(app_main, "_resolve_prepared_region", lambda **kw: mock_coverage)

    response = client.post("/risk/assess", json=_payload("1355 Pattee Canyon Rd, Missoula, MT 59803"))
    assert response.status_code == 200

    dcs = response.json().get("data_coverage_summary", {})
    assert dcs.get("local_data_available") is True
    assert dcs.get("overall_coverage") == "full"


# ---------------------------------------------------------------------------
# Test 8: data_coverage_summary always present (never missing from response)
# ---------------------------------------------------------------------------

def test_data_coverage_summary_always_present(monkeypatch, tmp_path):
    """data_coverage_summary appears in every AssessmentResult, even minimal context."""
    auth.API_KEYS = set()
    context = _minimal_context(
        slope_index=None,
        fuel_index=None,
        canopy_index=None,
        wildland_distance_index=None,
        historic_fire_index=None,
        burn_probability_index=None,
        hazard_severity_index=None,
        data_sources=[],
        environmental_layer_status={
            k: "not_configured"
            for k in ("burn_probability", "hazard", "slope", "fuel", "canopy", "fire_history")
        },
    )
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _: (35.0, -105.0, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "test.db")))

    response = client.post("/risk/assess", json=_payload("1 Unknown Rd, Nowhere, NM"))
    assert response.status_code == 200
    data = response.json()
    assert "data_coverage_summary" in data
    dcs = data["data_coverage_summary"]
    assert "overall_coverage" in dcs
    assert "local_data_available" in dcs
    assert "layers_from_national_sources" in dcs
    assert "layers_unavailable" in dcs
    assert "coverage_note" in dcs


# ---------------------------------------------------------------------------
# Test 9: get_region_context() for out-of-region coordinates
# ---------------------------------------------------------------------------

def test_get_region_context_out_of_region_is_national_fallback():
    """Out-of-region coordinates return RegionContext(is_national_fallback=True)."""
    from backend.wildfire_data import WildfireDataClient
    wdc = WildfireDataClient()
    ctx = wdc.get_region_context(35.0, -105.0)  # central New Mexico — no prepared region
    assert ctx.is_national_fallback is True
    assert ctx.is_prepared_region is False
    assert ctx.region_id == "national_fallback"
    assert ctx.bbox is None
    assert isinstance(ctx.missing_layers, list)
    assert len(ctx.missing_layers) > 0


# ---------------------------------------------------------------------------
# Test 10: get_region_context() for Missoula (prepared region)
# ---------------------------------------------------------------------------

def test_get_region_context_missoula_is_prepared_region():
    """Missoula coordinates return RegionContext(is_prepared_region=True)."""
    from backend.wildfire_data import WildfireDataClient
    wdc = WildfireDataClient()
    ctx = wdc.get_region_context(46.87, -113.99)  # Missoula, MT
    assert ctx.is_prepared_region is True
    assert ctx.is_national_fallback is False
    assert ctx.region_id != "national_fallback"
    assert isinstance(ctx.available_layers, list)
