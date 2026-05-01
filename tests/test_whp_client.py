"""
Tests for backend/whp_client.py — WHPClient proxy formula.

All tests are pure-Python (no network, no raster I/O). Tests skip if the
whp_client module cannot be imported.
"""

from __future__ import annotations

import importlib
import sys
from typing import Optional

import pytest


def _import() -> object:
    try:
        from backend import whp_client
        return whp_client
    except Exception as exc:
        pytest.skip(f"Could not import whp_client: {exc}")


# ---------------------------------------------------------------------------
# Test 1: all 5 components present → output in range, matches hand calculation
# ---------------------------------------------------------------------------

def test_proxy_all_components_produces_correct_value() -> None:
    """
    Hand calculation for:
      fuel_index=75, canopy_index=50, slope_index=60, aspect_degrees=195°, fire_count_30yr=1

    Weights: fuel=0.35, canopy=0.20, slope=0.20, aspect=0.10, fire_hist=0.15

    aspect_score(195°): 135≤195<225 → 1.0
    historical_fire_score(count=1): → 0.50

    component 0-1 scores:
      fuel_c    = 75/100 = 0.75
      canopy_c  = 50/100 = 0.50
      slope_c   = 60/100 = 0.60
      aspect_c  = 1.0
      fire_c    = 0.50

    weighted sum = 0.35*0.75 + 0.20*0.50 + 0.20*0.60 + 0.10*1.0 + 0.15*0.50
                = 0.2625 + 0.10 + 0.12 + 0.10 + 0.075
                = 0.6575

    total weight = 1.0 (all present)
    whp_0_to_1  = 0.6575 / 1.0 = 0.6575
    whp_index   = 65.8 (rounded to 1 decimal)
    """
    mod = _import()
    client = mod.WHPClient()

    features = {
        "fuel_index": 75.0,
        "canopy_index": 50.0,
        "slope_index": 60.0,
        "aspect_degrees": 195.0,
        "fire_count_30yr": 1,
        "burned_within_radius": False,
    }
    result = client.get_whp_index(35.0, -105.0, features)

    assert result is not None
    assert 0.0 <= result <= 100.0
    assert abs(result - 65.8) < 0.5, f"Expected ~65.8, got {result}"


# ---------------------------------------------------------------------------
# Test 2: known values → hand-calculated expected output
# ---------------------------------------------------------------------------

def test_proxy_specific_inputs_fuel80_canopy60_slope70_aspect180_count2() -> None:
    """
    Hand calculation for:
      fuel=80, canopy=60, slope=70, aspect=180° (south-facing), fire_count_30yr=2

    aspect_score(180°): 135≤180<225 → 1.0
    historical_fire_score(count=2): → 0.75

    fuel_c   = 80/100 = 0.80
    canopy_c = 60/100 = 0.60
    slope_c  = 70/100 = 0.70
    aspect_c = 1.0
    fire_c   = 0.75

    weighted = 0.35*0.80 + 0.20*0.60 + 0.20*0.70 + 0.10*1.0 + 0.15*0.75
             = 0.28 + 0.12 + 0.14 + 0.10 + 0.1125
             = 0.7525

    whp_index = 75.3 (rounded to 1 decimal)
    """
    mod = _import()
    client = mod.WHPClient()

    features = {
        "fuel_index": 80.0,
        "canopy_index": 60.0,
        "slope_index": 70.0,
        "aspect_degrees": 180.0,
        "fire_count_30yr": 2,
        "burned_within_radius": False,
    }
    result = client.get_whp_index(40.0, -105.0, features)

    assert result is not None
    assert abs(result - 75.3) < 0.5, f"Expected ~75.3, got {result}"


# ---------------------------------------------------------------------------
# Test 3: fewer than 3 components non-None → returns None
# ---------------------------------------------------------------------------

def test_proxy_insufficient_inputs_returns_none() -> None:
    """Only 2 primary components non-None (fuel + canopy) — minimum is 3."""
    mod = _import()
    client = mod.WHPClient()

    features = {
        "fuel_index": 70.0,
        "canopy_index": 50.0,
        # slope, aspect, fire_count all None
        "slope_index": None,
        "aspect_degrees": None,
        "fire_count_30yr": None,
        "burned_within_radius": None,
    }
    result = client.get_whp_index(35.0, -105.0, features)
    assert result is None


# ---------------------------------------------------------------------------
# Test 4: aspect score mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degrees,expected", [
    (180.0, 1.0),   # south
    (250.0, 0.85),  # southwest
    (300.0, 0.70),  # west
    (100.0, 0.65),  # southeast
    (0.0, 0.40),    # north
    (None, 0.50),   # flat/unknown
])
def test_aspect_score_values(degrees, expected) -> None:
    mod = _import()
    score = mod._aspect_score(degrees)
    assert score == expected, f"aspect_score({degrees}) = {score}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 5: historical fire score mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count,burned,expected", [
    (3, False, 1.0),
    (4, False, 1.0),   # count >= 3 → 1.0
    (1, False, 0.50),
    (0, True, 0.25),   # count=0 but burned
    (0, False, 0.0),
    (None, False, 0.10),
    (None, None, 0.10),
])
def test_historical_fire_score_values(count, burned, expected) -> None:
    mod = _import()
    score = mod._historical_fire_score(count, burned)
    assert score == expected, (
        f"_historical_fire_score({count}, {burned}) = {score}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Test 6: source tag written into features dict
# ---------------------------------------------------------------------------

def test_source_tag_set_in_features_dict() -> None:
    mod = _import()
    client = mod.WHPClient()

    features = {
        "fuel_index": 60.0,
        "canopy_index": 40.0,
        "slope_index": 50.0,
        "aspect_degrees": 200.0,
        "fire_count_30yr": 0,
        "burned_within_radius": False,
    }
    result = client.get_whp_index(35.0, -105.0, features)

    assert result is not None
    assert "whp_index_source" in features, (
        "get_whp_index must set features['whp_index_source']"
    )
    assert features["whp_index_source"] == "whp_proxy"


# ---------------------------------------------------------------------------
# Test 7: disabled client → returns None, no computation
# ---------------------------------------------------------------------------

def test_disabled_client_returns_none() -> None:
    mod = _import()
    client = mod.WHPClient(enabled=False)

    features = {
        "fuel_index": 80.0,
        "canopy_index": 70.0,
        "slope_index": 60.0,
        "aspect_degrees": 180.0,
        "fire_count_30yr": 2,
        "burned_within_radius": True,
    }
    result = client.get_whp_index(35.0, -105.0, features)
    assert result is None
    assert "whp_index_source" not in features, (
        "Disabled client must not write whp_index_source"
    )


# ---------------------------------------------------------------------------
# Test 8: get_whp_components returns expected structure
# ---------------------------------------------------------------------------

def test_get_whp_components_returns_breakdown() -> None:
    mod = _import()
    client = mod.WHPClient()

    features = {
        "fuel_index": 70.0,
        "canopy_index": 55.0,
        "slope_index": 45.0,
        "aspect_degrees": 225.0,
        "fire_count_30yr": 1,
        "burned_within_radius": False,
    }
    components = client.get_whp_components(features)

    assert isinstance(components, dict)
    assert "mode" in components
    assert components["mode"] == "proxy"
    assert "components" in components
    comp = components["components"]
    assert "fuel_model_index" in comp
    assert "canopy_cover_index" in comp
    assert "slope_index" in comp
    assert "aspect_score" in comp
    assert "historical_fire_score" in comp
    # aspect_score derived_score should match 225° → southwest → 0.85
    assert comp["aspect_score"]["derived_score"] == 0.85
    # historical_fire_score for count=1 → 0.50
    assert comp["historical_fire_score"]["derived_score"] == 0.50


# ---------------------------------------------------------------------------
# Test 9: high-risk inputs produce higher WHP than low-risk inputs
# ---------------------------------------------------------------------------

def test_high_risk_greater_than_low_risk() -> None:
    mod = _import()
    client = mod.WHPClient()

    high_risk = {
        "fuel_index": 90.0,
        "canopy_index": 80.0,
        "slope_index": 75.0,
        "aspect_degrees": 200.0,   # south-facing
        "fire_count_30yr": 3,
        "burned_within_radius": True,
    }
    low_risk = {
        "fuel_index": 15.0,
        "canopy_index": 5.0,
        "slope_index": 3.0,
        "aspect_degrees": 10.0,    # north-facing
        "fire_count_30yr": 0,
        "burned_within_radius": False,
    }
    high = client.get_whp_index(40.0, -105.0, high_risk)
    low = client.get_whp_index(40.0, -90.0, low_risk)

    assert high is not None and low is not None
    assert high > low, (
        f"High-risk WHP ({high}) should exceed low-risk WHP ({low})"
    )


# ---------------------------------------------------------------------------
# Test 10: missing slope (3 of 5 remaining) → result still produced
# ---------------------------------------------------------------------------

def test_three_components_sufficient() -> None:
    """3 of 5 primary components is the minimum — should still return a value."""
    mod = _import()
    client = mod.WHPClient()

    features = {
        "fuel_index": 70.0,
        "canopy_index": 50.0,
        "slope_index": None,          # missing
        "aspect_degrees": None,       # missing (counted as absent)
        "fire_count_30yr": 1,
    }
    result = client.get_whp_index(35.0, -105.0, features)
    # fuel, canopy, fire_count = 3 available → meets minimum
    assert result is not None


# ---------------------------------------------------------------------------
# Test 11: aspect score boundary conditions
# ---------------------------------------------------------------------------

def test_aspect_score_boundaries() -> None:
    mod = _import()
    # Exact boundary 135.0 is south-facing
    assert mod._aspect_score(135.0) == 1.0
    # 224.9 still south-facing
    assert mod._aspect_score(224.9) == 1.0
    # 225.0 is southwest
    assert mod._aspect_score(225.0) == 0.85
    # 360.0 wraps to 0 (north)
    assert mod._aspect_score(360.0) == 0.40
    # 359.9 wraps to near-north
    assert mod._aspect_score(359.9) == 0.40
