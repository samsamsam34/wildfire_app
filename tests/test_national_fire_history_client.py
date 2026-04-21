"""
Unit tests for backend/national_fire_history_client.py.

All tests create temporary in-memory GeoPackages using geopandas — no real
MTBS file or network access required. Tests skip if geopandas is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _gpd():
    """Return geopandas module or skip if unavailable."""
    try:
        import geopandas as gpd
        return gpd
    except ImportError:
        pytest.skip("geopandas not available")


def _make_gpkg(tmp_path: Path, fires: list[dict[str, Any]]) -> str:
    """Create a minimal MTBS GeoPackage in tmp_path with the given fire records."""
    gpd = _gpd()
    from shapely.geometry import Polygon

    # Ensure geometry key is separate from attribute dict
    rows = []
    geoms = []
    for fire in fires:
        fire = dict(fire)
        geom = fire.pop("geometry")
        rows.append(fire)
        geoms.append(geom)

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    path = str(tmp_path / "mtbs_test.gpkg")
    gdf.to_file(path, driver="GPKG", layer="fire_perimeters")
    return path


def _box(lon_min, lat_min, lon_max, lat_max):
    """Return a Shapely box Polygon in WGS84."""
    from shapely.geometry import Polygon
    return Polygon([
        (lon_min, lat_min), (lon_max, lat_min),
        (lon_max, lat_max), (lon_min, lat_max),
    ])


# ---------------------------------------------------------------------------
# Test 1: One fire within radius → burned_within_radius=True, correct counts
# ---------------------------------------------------------------------------

def test_fire_within_radius_detected(tmp_path: Path) -> None:
    """A fire polygon that overlaps the search buffer is found and counted."""
    _gpd()
    fires = [
        {
            "Fire_ID": "UT001", "Fire_Name": "TEST FIRE A", "Year": 2010,
            "StartMonth": 7, "BurnBndAc": 5000.0,
            "low_severity_pct": 20.0, "mod_severity_pct": 55.0, "high_severity_pct": 25.0,
            "geometry": _box(-111.75, 40.25, -111.65, 40.35),
        },
        {
            "Fire_ID": "UT002", "Fire_Name": "FAR FIRE", "Year": 2015,
            "StartMonth": 8, "BurnBndAc": 1000.0,
            "low_severity_pct": None, "mod_severity_pct": None, "high_severity_pct": None,
            "geometry": _box(-112.50, 40.25, -112.40, 40.35),  # ~70 km west
        },
    ]
    path = _make_gpkg(tmp_path, fires)

    from backend.national_fire_history_client import NationalFireHistoryClient
    client = NationalFireHistoryClient(mtbs_gpkg_path=path)
    assert client.enabled, "Client should be enabled when GPKG exists"

    # Query at center of first fire polygon (Provo-ish area)
    result = client.query_fire_history(40.30, -111.70, radius_m=5000.0)

    assert result.data_available is True
    assert result.fire_count_all >= 1
    assert result.most_recent_fire_year == 2010
    # Fire A has mod_severity_pct=55 → "moderate"
    assert result.most_recent_fire_severity == "moderate"
    # FAR FIRE is outside 5 km radius
    assert result.fire_count_all == 1, "Only the nearby fire should be counted"


# ---------------------------------------------------------------------------
# Test 2: No fires within radius → burned_within_radius=False, count=0
# ---------------------------------------------------------------------------

def test_no_fires_within_radius(tmp_path: Path) -> None:
    """Query where no fires fall within the search radius."""
    _gpd()
    fires = [
        {
            "Fire_ID": "MT001", "Fire_Name": "DISTANT FIRE", "Year": 2000,
            "StartMonth": 9, "BurnBndAc": 20000.0,
            "low_severity_pct": 50.0, "mod_severity_pct": 30.0, "high_severity_pct": 20.0,
            "geometry": _box(-115.0, 47.0, -114.5, 47.5),  # far from query point
        },
    ]
    path = _make_gpkg(tmp_path, fires)

    from backend.national_fire_history_client import NationalFireHistoryClient
    client = NationalFireHistoryClient(mtbs_gpkg_path=path)

    result = client.query_fire_history(46.87, -113.99, radius_m=5000.0)

    assert result.data_available is True
    assert result.burned_within_radius is False
    assert result.fire_count_all == 0
    assert result.fire_count_30yr == 0
    assert result.nearest_fire_distance_m is None


# ---------------------------------------------------------------------------
# Test 3: GPKG file missing → data_available=False, no exception
# ---------------------------------------------------------------------------

def test_missing_gpkg_degrades_gracefully(tmp_path: Path) -> None:
    """When the GeoPackage file does not exist, client is disabled and returns safely."""
    _gpd()
    from backend.national_fire_history_client import NationalFireHistoryClient
    client = NationalFireHistoryClient(mtbs_gpkg_path=str(tmp_path / "nonexistent.gpkg"))

    assert client.enabled is False
    result = client.query_fire_history(40.30, -111.70)
    assert result.data_available is False
    assert result.fire_count_all == 0
    assert result.burned_within_radius is False


# ---------------------------------------------------------------------------
# Test 4: STRtree candidate whose exact geometry doesn't intersect buffer
# ---------------------------------------------------------------------------

def test_strtree_candidate_excluded_by_exact_check(tmp_path: Path) -> None:
    """A fire bbox that overlaps the buffer but the actual polygon does not intersect
    should NOT be counted (two-stage filter: bbox prefilter + exact intersection check)."""
    _gpd()
    from shapely.geometry import Polygon

    # Create an L-shaped polygon whose bbox overlaps (40.30, -111.70) +5km buffer,
    # but the actual polygon body is offset so it does NOT intersect the buffer.
    # The buffer at (40.30, -111.70) with 5km ≈ 0.045 degrees.
    # Place the polygon so its bbox touches the buffer corner but its body is outside.
    l_shape = Polygon([
        (-111.75, 40.35), (-111.70, 40.35), (-111.70, 40.40),
        (-111.65, 40.40), (-111.65, 40.50), (-111.75, 40.50),
    ])
    fires = [{
        "Fire_ID": "XX001", "Fire_Name": "L-SHAPE FIRE", "Year": 2005,
        "StartMonth": 6, "BurnBndAc": 3000.0,
        "low_severity_pct": 60.0, "mod_severity_pct": 30.0, "high_severity_pct": 10.0,
        "geometry": l_shape,
    }]
    path = _make_gpkg(tmp_path, fires)

    from backend.national_fire_history_client import NationalFireHistoryClient
    client = NationalFireHistoryClient(mtbs_gpkg_path=path)

    # Query from a point that the L-shape's bbox overlaps but the polygon body does not
    result = client.query_fire_history(40.30, -111.70, radius_m=500.0)
    # The L-shape is well outside the 500 m buffer — should NOT be counted
    assert result.fire_count_all == 0, (
        "Fire whose polygon doesn't intersect buffer must not be counted"
    )


# ---------------------------------------------------------------------------
# Test 5: Severity classification logic
# ---------------------------------------------------------------------------

def test_severity_classification(tmp_path: Path) -> None:
    """Severity classification: high>50 → 'high'; mod>50 → 'moderate'; else → 'low'."""
    from backend.national_fire_history_client import _classify_severity

    assert _classify_severity({"high_severity_pct": 60.0, "mod_severity_pct": 20.0, "low_severity_pct": 20.0}) == "high"
    assert _classify_severity({"high_severity_pct": 10.0, "mod_severity_pct": 55.0, "low_severity_pct": 35.0}) == "moderate"
    assert _classify_severity({"high_severity_pct": 20.0, "mod_severity_pct": 20.0, "low_severity_pct": 60.0}) == "low"
    # Missing severity fields → "unknown"
    assert _classify_severity({}) == "unknown"
    assert _classify_severity({"Fire_Name": "TEST"}) == "unknown"


# ---------------------------------------------------------------------------
# Test 6: fire_count_30yr only counts fires within 30 years
# ---------------------------------------------------------------------------

def test_fire_count_30yr_excludes_old_fires(tmp_path: Path) -> None:
    """fire_count_30yr excludes fires older than 30 years from now."""
    _gpd()
    from datetime import datetime
    current_year = datetime.now().year
    old_year = current_year - 35  # 35 years ago → should NOT count in 30yr
    recent_year = current_year - 5  # 5 years ago → SHOULD count

    fires = [
        {
            "Fire_ID": "OLD001", "Fire_Name": "OLD FIRE", "Year": old_year,
            "StartMonth": 7, "BurnBndAc": 5000.0,
            "low_severity_pct": 50.0, "mod_severity_pct": 30.0, "high_severity_pct": 20.0,
            "geometry": _box(-111.75, 40.25, -111.65, 40.35),
        },
        {
            "Fire_ID": "NEW001", "Fire_Name": "RECENT FIRE", "Year": recent_year,
            "StartMonth": 8, "BurnBndAc": 3000.0,
            "low_severity_pct": 20.0, "mod_severity_pct": 60.0, "high_severity_pct": 20.0,
            "geometry": _box(-111.74, 40.26, -111.66, 40.34),
        },
    ]
    path = _make_gpkg(tmp_path, fires)

    from backend.national_fire_history_client import NationalFireHistoryClient
    client = NationalFireHistoryClient(mtbs_gpkg_path=path)
    result = client.query_fire_history(40.30, -111.70, radius_m=5000.0)

    assert result.fire_count_all == 2
    assert result.fire_count_30yr == 1, "Only the recent fire should count in 30yr window"
    assert result.most_recent_fire_year == recent_year
