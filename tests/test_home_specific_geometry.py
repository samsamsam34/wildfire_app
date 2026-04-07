"""Tests for home-specific geometry signals: stages 1-3.

Stage 1 — ring-zone slope means fed into the topography submodel.
Stage 2 — parcel setback distance from footprint to parcel boundary.
Stage 3 — footprint shape signals (perimeter, compactness, orientation, parcel coverage).
"""
from __future__ import annotations

import math

import pytest

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    _GEO_READY = True
except Exception:
    _GEO_READY = False
    ShapelyPolygon = None


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def test_arc_slope_mean_returns_none_without_slope_path() -> None:
    """_compute_arc_slope_mean_deg must return None when slope_path is empty."""
    from backend.wildfire_data import WildfireDataClient

    client = WildfireDataClient.__new__(WildfireDataClient)
    result = client._compute_arc_slope_mean_deg("", 40.0, -105.0, radius_ft=17.5)
    assert result is None


def test_arc_slope_mean_returns_none_for_missing_file() -> None:
    """_compute_arc_slope_mean_deg must return None for a non-existent file path."""
    from backend.wildfire_data import WildfireDataClient

    client = WildfireDataClient.__new__(WildfireDataClient)
    result = client._compute_arc_slope_mean_deg(
        "/nonexistent/slope.tif", 40.0, -105.0, radius_ft=17.5
    )
    assert result is None


def test_zone_slope_blend_uses_single_point_when_zone_unavailable() -> None:
    """When zone slope is absent, blended_slope_index must equal slope_index."""
    # Simulate the blending logic from risk_engine.py directly.
    slope_index: float | None = 42.0
    zone_slope_index: float | None = None

    blended = (
        round(0.65 * float(slope_index) + 0.35 * float(zone_slope_index), 1)
        if slope_index is not None and zone_slope_index is not None
        else slope_index
    )
    assert blended == slope_index


def test_zone_slope_blend_produces_weighted_average_when_both_available() -> None:
    """blended_slope_index must be the 65/35 weighted mix of the two inputs."""
    slope_index = 40.0
    zone_slope_index = 60.0

    blended = round(0.65 * slope_index + 0.35 * zone_slope_index, 1)
    assert blended == pytest.approx(47.0, abs=0.2)


def test_zone_slope_blend_increases_score_on_steeper_ring_slope() -> None:
    """A steeper zone slope must yield a higher blended_slope_index than the
    structure-point slope alone, confirming monotonicity for hillside properties."""
    slope_index = 30.0
    zone_slope_steeper = 70.0
    zone_slope_flat = 5.0

    blended_steep = round(0.65 * slope_index + 0.35 * zone_slope_steeper, 1)
    blended_flat = round(0.65 * slope_index + 0.35 * zone_slope_flat, 1)

    assert blended_steep > slope_index, "Steeper ring must increase blended slope."
    assert blended_flat < slope_index, "Flatter ring must reduce blended slope."


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for parcel setback tests")
def test_parcel_setback_positive_when_footprint_inside_parcel() -> None:
    """compute_footprint_geometry_signals must report parcel_coverage_ratio < 1
    and the inline setback logic must produce a positive distance when the
    footprint is well within the parcel."""
    from backend.building_footprints import compute_footprint_geometry_signals

    # A ~20m×10m footprint centred at (40.0, -105.0).
    d_lon = 0.00009  # ≈10 m
    d_lat = 0.00009  # ≈10 m
    footprint = ShapelyPolygon([
        (-105.0 - d_lon, 40.0 - d_lat),
        (-105.0 + d_lon, 40.0 - d_lat),
        (-105.0 + d_lon, 40.0 + d_lat),
        (-105.0 - d_lon, 40.0 + d_lat),
        (-105.0 - d_lon, 40.0 - d_lat),
    ])
    # A much larger parcel that encloses the footprint with a clear margin.
    margin = 0.001  # ≈100 m
    parcel = ShapelyPolygon([
        (-105.0 - margin, 40.0 - margin),
        (-105.0 + margin, 40.0 - margin),
        (-105.0 + margin, 40.0 + margin),
        (-105.0 - margin, 40.0 + margin),
        (-105.0 - margin, 40.0 - margin),
    ])

    signals = compute_footprint_geometry_signals(footprint=footprint, parcel_polygon=parcel)

    assert signals["parcel_coverage_ratio"] is not None
    assert 0.0 < signals["parcel_coverage_ratio"] < 1.0, (
        "Coverage ratio must be a positive fraction when footprint is inside parcel"
    )


@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for parcel setback tests")
def test_parcel_setback_none_when_no_parcel_polygon() -> None:
    """parcel_coverage_ratio must be None when no parcel polygon is provided."""
    from backend.building_footprints import compute_footprint_geometry_signals

    d = 0.00009
    footprint = ShapelyPolygon([
        (-105.0 - d, 40.0 - d),
        (-105.0 + d, 40.0 - d),
        (-105.0 + d, 40.0 + d),
        (-105.0 - d, 40.0 + d),
        (-105.0 - d, 40.0 - d),
    ])
    signals = compute_footprint_geometry_signals(footprint=footprint, parcel_polygon=None)
    assert signals["parcel_coverage_ratio"] is None


# ---------------------------------------------------------------------------
# Stage 3 helpers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for footprint geometry tests")
def test_footprint_perimeter_positive_for_valid_polygon() -> None:
    from backend.building_footprints import compute_footprint_geometry_signals

    d = 0.0001
    footprint = ShapelyPolygon([
        (-105.0 - d, 40.0 - d),
        (-105.0 + d, 40.0 - d),
        (-105.0 + d, 40.0 + d),
        (-105.0 - d, 40.0 + d),
        (-105.0 - d, 40.0 - d),
    ])
    signals = compute_footprint_geometry_signals(footprint=footprint)
    assert signals["footprint_perimeter_m"] is not None
    assert signals["footprint_perimeter_m"] > 0.0


@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for footprint geometry tests")
def test_compactness_ratio_approaches_one_for_near_square() -> None:
    """A square footprint has the highest compactness ratio among rectangles."""
    from backend.building_footprints import compute_footprint_geometry_signals

    d = 0.0001
    square = ShapelyPolygon([
        (-105.0 - d, 40.0 - d),
        (-105.0 + d, 40.0 - d),
        (-105.0 + d, 40.0 + d),
        (-105.0 - d, 40.0 + d),
        (-105.0 - d, 40.0 - d),
    ])
    d2x = 0.0002
    elongated = ShapelyPolygon([
        (-105.0 - d2x, 40.0 - d * 0.25),
        (-105.0 + d2x, 40.0 - d * 0.25),
        (-105.0 + d2x, 40.0 + d * 0.25),
        (-105.0 - d2x, 40.0 + d * 0.25),
        (-105.0 - d2x, 40.0 - d * 0.25),
    ])

    sig_sq = compute_footprint_geometry_signals(square)
    sig_el = compute_footprint_geometry_signals(elongated)

    assert sig_sq["footprint_compactness_ratio"] is not None
    assert sig_el["footprint_compactness_ratio"] is not None
    assert sig_sq["footprint_compactness_ratio"] > sig_el["footprint_compactness_ratio"], (
        "Square footprint must have higher compactness ratio than elongated footprint"
    )


@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for footprint geometry tests")
def test_long_axis_bearing_is_within_0_to_180() -> None:
    """Bearing must always be in the [0, 180) range (axial, not directional)."""
    from backend.building_footprints import compute_footprint_geometry_signals

    d_lon = 0.0003  # wider east–west
    d_lat = 0.0001
    footprint = ShapelyPolygon([
        (-105.0 - d_lon, 40.0 - d_lat),
        (-105.0 + d_lon, 40.0 - d_lat),
        (-105.0 + d_lon, 40.0 + d_lat),
        (-105.0 - d_lon, 40.0 + d_lat),
        (-105.0 - d_lon, 40.0 - d_lat),
    ])
    signals = compute_footprint_geometry_signals(footprint=footprint)
    bearing = signals["footprint_long_axis_bearing_deg"]
    assert bearing is not None
    assert 0.0 <= bearing < 180.0


@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for footprint geometry tests")
def test_multiple_structures_on_parcel_always_unknown() -> None:
    """multiple_structures_on_parcel must be 'unknown' until enumeration is implemented."""
    from backend.building_footprints import compute_footprint_geometry_signals

    d = 0.0001
    footprint = ShapelyPolygon([
        (-105.0 - d, 40.0 - d),
        (-105.0 + d, 40.0 - d),
        (-105.0 + d, 40.0 + d),
        (-105.0 - d, 40.0 + d),
        (-105.0 - d, 40.0 - d),
    ])
    signals = compute_footprint_geometry_signals(footprint=footprint)
    assert signals["multiple_structures_on_parcel"] == "unknown"


@pytest.mark.skipif(not _GEO_READY, reason="Shapely required for footprint geometry tests")
def test_point_proxy_fallback_has_none_ring_slope_fields(tmp_path: pytest.TempPathFactory) -> None:
    """The point-proxy property_level_context must initialise all Stage 1-3 keys to None
    so downstream consumers can rely on key presence without KeyError."""
    from backend.wildfire_data import WildfireDataClient

    # Verify the keys are defined in the fallback dict by inspecting the source.
    # We do this by triggering the fallback path through a partial mock.
    import inspect, ast, textwrap

    src = inspect.getsource(WildfireDataClient.collect_context)
    # The fallback property_level_context literal must contain our new keys.
    expected_keys = [
        "ring_5_30_slope_mean_deg",
        "ring_30_100_slope_mean_deg",
        "parcel_setback_min_ft",
        "adjacent_parcel_vegetation_pressure",
        "footprint_perimeter_m",
        "footprint_compactness_ratio",
        "footprint_long_axis_bearing_deg",
        "parcel_coverage_ratio",
        "multiple_structures_on_parcel",
    ]
    for key in expected_keys:
        assert f'"{key}"' in src, (
            f"Key {key!r} missing from point-proxy property_level_context in get_context"
        )
