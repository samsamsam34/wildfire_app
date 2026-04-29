"""
Unit tests for backend/national_elevation_client.py.

All rasterio operations and network calls are mocked — no real 3DEP requests
are made. Tests use an in-memory (tmp_path) SQLite cache.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


def _make_client(tmp_path: Path, *, enabled: bool = True):
    from backend.national_elevation_client import NationalElevationClient
    return NationalElevationClient(
        cache_db_path=str(tmp_path / "elev_cache.db"),
        timeout_seconds=10,
        enabled=enabled,
    )


def _make_slope_dem(
    rows: int,
    cols: int,
    *,
    rise_m: float = 10.0,
    run_m: float = 100.0,
    direction: str = "east",
    pixel_size_deg: float = 0.00009,  # ~10 m at mid-latitudes
) -> tuple:
    """
    Build a synthetic elevation array and a matching rasterio-style mock dataset.

    direction: "east"  → elevation increases eastward (west-facing slope)
               "north" → elevation increases northward (south-facing slope,
                         aspect should be ~0° downslope = northward descending)

    Returns (mock_ds, expected_slope_deg, expected_aspect_deg).
    """
    try:
        import numpy as np
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    pixel_m = pixel_size_deg * 111320.0
    # Build a tilted plane: elevation increases in `direction` by rise_m over run_m.
    elev = np.zeros((rows, cols), dtype=np.float64)
    if direction == "east":
        # elevation increases as col increases (westward is lower)
        for c in range(cols):
            elev[:, c] = (c / (cols - 1)) * (rise_m / run_m) * (cols * pixel_m)
        expected_aspect = 90.0   # downslope = eastward → 90°? No.
        # Downslope convention: direction water flows.
        # If elevation rises eastward, water flows westward = aspect 270°.
        expected_aspect = 270.0
    else:  # "north"
        # elevation increases as row decreases (row 0 = north, higher elevation)
        for r in range(rows):
            # row 0 = north (high), row max = south (low)
            elev[r, :] = ((rows - 1 - r) / (rows - 1)) * (rise_m / run_m) * (rows * pixel_m)
        # elevation higher to north → slope descends southward → aspect 180°?
        # No: downslope direction = southward = 180°.
        # But wait: "slope rising toward north" per test 3 spec → aspect should be 0/360
        # because downslope direction is SOUTH = 180°.
        # Re-reading spec: "slope rising toward north → assert aspect within 10° of 0/360"
        # This means they define aspect as the UPSLOPE direction.
        # But wildfire_data.py uses downslope convention: 0° = descends northward.
        # If elevation rises northward (higher in north, lower in south), water flows
        # southward → downslope = 180°.
        # The spec comment says 0/360, which would be the upslope direction.
        # We follow the codebase (downslope) and adjust the test assertion accordingly.
        expected_aspect = 180.0  # downslope southward
    slope_rad = math.atan(rise_m / run_m)
    expected_slope = math.degrees(slope_rad)

    # Build rasterio mock
    west = -111.70
    south = 40.30
    east = west + cols * pixel_size_deg
    north = south + rows * pixel_size_deg
    transform = from_bounds(west, south, east, north, cols, rows)

    mock_ds = MagicMock()
    mock_ds.__enter__ = MagicMock(return_value=mock_ds)
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = transform
    mock_ds.bounds = mock_ds.bounds
    # Provide real bounds
    from rasterio.coords import BoundingBox
    mock_ds.bounds = BoundingBox(west, south, east, north)
    mock_ds.nodata = -9999.0
    mock_ds.read = MagicMock(return_value=elev)
    mock_ds.window_transform = MagicMock(return_value=transform)

    return mock_ds, expected_slope, expected_aspect


# ---------------------------------------------------------------------------
# Test 1: Known slope from synthetic DEM → returns value within 0.5° of expected
# ---------------------------------------------------------------------------

def test_known_slope_from_synthetic_dem(tmp_path: Path) -> None:
    """Synthetic DEM with fixed rise-per-pixel → get_slope_degrees matches expected slope."""
    try:
        import numpy as np
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from rasterio.coords import BoundingBox
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    rows, cols = 21, 21
    pixel_deg = 0.00009  # ~10 m
    lat = 40.30 + (rows / 2) * pixel_deg  # center lat used for cos correction
    west, south = -111.70, 40.30
    east = west + cols * pixel_deg
    north = south + rows * pixel_deg
    transform = from_bounds(west, south, east, north, cols, rows)

    # Compute the actual dx_m the client will use (mirrors client implementation).
    cos_lat = abs(math.cos(math.radians(lat)))
    dx_m = pixel_deg * 111320.0 * cos_lat

    # Set a fixed rise-per-pixel; compute the expected slope from the same dx_m.
    rise_per_pixel_m = 1.0  # 1 m per pixel in x direction
    expected_slope_deg = math.degrees(math.atan(rise_per_pixel_m / dx_m))

    elev = np.zeros((rows, cols), dtype=np.float64)
    for c in range(cols):
        elev[:, c] = c * rise_per_pixel_m

    mock_ds = MagicMock()
    mock_ds.__enter__ = MagicMock(return_value=mock_ds)
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = transform
    mock_ds.bounds = BoundingBox(west, south, east, north)
    mock_ds.nodata = None
    mock_ds.read = MagicMock(return_value=elev)
    mock_ds.window_transform = MagicMock(return_value=transform)

    client = _make_client(tmp_path)
    with patch("rasterio.open", return_value=mock_ds):
        slope = client.get_slope_degrees(lat, -111.70 + (cols / 2) * pixel_deg,
                                         sample_radius_m=100.0)

    assert slope is not None, "Should return a slope value from the synthetic DEM"
    assert abs(slope - expected_slope_deg) < 0.5, (
        f"Expected ~{expected_slope_deg:.2f}°, got {slope:.3f}°"
    )


# ---------------------------------------------------------------------------
# Test 2: Flat terrain → slope=0.0, aspect=None
# ---------------------------------------------------------------------------

def test_flat_terrain_returns_zero_slope_no_aspect(tmp_path: Path) -> None:
    """When all elevation values are identical, slope=0.0 and aspect=None."""
    try:
        import numpy as np
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from rasterio.coords import BoundingBox
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    rows, cols = 11, 11
    pixel_deg = 0.00009
    elev = np.full((rows, cols), 1500.0, dtype=np.float64)

    west, south = -111.70, 40.30
    east = west + cols * pixel_deg
    north = south + rows * pixel_deg
    transform = from_bounds(west, south, east, north, cols, rows)

    mock_ds = MagicMock()
    mock_ds.__enter__ = MagicMock(return_value=mock_ds)
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = transform
    mock_ds.bounds = BoundingBox(west, south, east, north)
    mock_ds.nodata = None
    mock_ds.read = MagicMock(return_value=elev)
    mock_ds.window_transform = MagicMock(return_value=transform)

    # Use center of tile to ensure the point is within bounds
    lat = south + (rows / 2) * pixel_deg
    lon = west + (cols / 2) * pixel_deg

    client = _make_client(tmp_path)
    with patch("rasterio.open", return_value=mock_ds):
        slope, aspect = client.get_slope_and_aspect(lat, lon)

    assert slope == 0.0, f"Flat terrain should return slope=0.0, got {slope}"
    assert aspect is None, f"Flat terrain should return aspect=None, got {aspect}"


# ---------------------------------------------------------------------------
# Test 3: Slope rising toward north → aspect near 180° (downslope southward)
# ---------------------------------------------------------------------------

def test_north_facing_slope_aspect(tmp_path: Path) -> None:
    """Elevation highest in north, lowest in south → downslope direction ≈ 180°."""
    try:
        import numpy as np
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from rasterio.coords import BoundingBox
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    rows, cols = 21, 21
    pixel_deg = 0.00009
    # Row 0 = north (high elevation), last row = south (low elevation)
    elev = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        elev[r, :] = (rows - 1 - r) * 5.0  # 5 m/pixel drop going south

    west, south = -111.70, 40.30
    east = west + cols * pixel_deg
    north = south + rows * pixel_deg
    transform = from_bounds(west, south, east, north, cols, rows)

    mock_ds = MagicMock()
    mock_ds.__enter__ = MagicMock(return_value=mock_ds)
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = transform
    mock_ds.bounds = BoundingBox(west, south, east, north)
    mock_ds.nodata = None
    mock_ds.read = MagicMock(return_value=elev)
    mock_ds.window_transform = MagicMock(return_value=transform)

    # Use center of tile to ensure the point is within bounds
    lat = south + (rows / 2) * pixel_deg
    lon = west + (cols / 2) * pixel_deg

    client = _make_client(tmp_path)
    with patch("rasterio.open", return_value=mock_ds):
        slope, aspect = client.get_slope_and_aspect(lat, lon)

    assert slope is not None and slope > 0.5, "Expected a meaningful slope"
    assert aspect is not None, "Aspect should not be None for a sloped surface"
    # Downslope direction is southward → aspect ~180° (water flows south)
    angular_err = abs(((aspect - 180.0 + 180.0) % 360.0) - 180.0)
    assert angular_err < 10.0, f"Expected aspect ≈180° (downslope south), got {aspect:.1f}°"


# ---------------------------------------------------------------------------
# Test 4: Cache hit → rasterio.open not called second time
# ---------------------------------------------------------------------------

def test_cache_hit_avoids_second_rasterio_open(tmp_path: Path) -> None:
    """Second call for same (lat, lon) hits cache; rasterio.open not called again."""
    try:
        import numpy as np
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from rasterio.coords import BoundingBox
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    rows, cols = 11, 11
    pixel_deg = 0.00009
    elev = np.zeros((rows, cols), dtype=np.float64)
    for c in range(cols):
        elev[:, c] = c * 3.0  # gentle eastward slope

    west, south = -111.70, 40.30
    east = west + cols * pixel_deg
    north = south + rows * pixel_deg
    transform = from_bounds(west, south, east, north, cols, rows)

    mock_ds = MagicMock()
    mock_ds.__enter__ = MagicMock(return_value=mock_ds)
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = transform
    mock_ds.bounds = BoundingBox(west, south, east, north)
    mock_ds.nodata = None
    mock_ds.read = MagicMock(return_value=elev)
    mock_ds.window_transform = MagicMock(return_value=transform)

    lat = south + (rows / 2) * pixel_deg
    lon = west + (cols / 2) * pixel_deg
    client = _make_client(tmp_path)

    with patch("rasterio.open", return_value=mock_ds) as mock_open:
        s1, a1 = client.get_slope_and_aspect(lat, lon)
        s2, a2 = client.get_slope_and_aspect(lat, lon)

    assert s1 == s2, "Both calls should return the same slope"
    assert a1 == a2, "Both calls should return the same aspect"
    assert mock_open.call_count == 1, "Second call should hit cache, not re-open rasterio"


# ---------------------------------------------------------------------------
# Test 5: Rasterio exception → returns (None, None), no exception raised
# ---------------------------------------------------------------------------

def test_rasterio_exception_returns_none(tmp_path: Path) -> None:
    """If rasterio.open raises, get_slope_and_aspect returns (None, None) without raising."""
    client = _make_client(tmp_path)
    with patch("rasterio.open", side_effect=RuntimeError("GDAL error")):
        slope, aspect = client.get_slope_and_aspect(40.30, -111.70)

    assert slope is None
    assert aspect is None


# ---------------------------------------------------------------------------
# Test 6: Network/timeout exception → returns None
# ---------------------------------------------------------------------------

def test_network_error_returns_none(tmp_path: Path) -> None:
    """Any network-level exception returns None without raising."""
    client = _make_client(tmp_path)
    with patch("rasterio.open", side_effect=OSError("Connection timed out")):
        result = client.get_slope_degrees(40.30, -111.70)

    assert result is None


# ---------------------------------------------------------------------------
# Test 7: Stale cache entry → re-fetch triggered
# ---------------------------------------------------------------------------

def test_stale_cache_triggers_refetch(tmp_path: Path) -> None:
    """Cache entries older than 365 days are ignored and trigger a new fetch."""
    try:
        import numpy as np
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from rasterio.coords import BoundingBox
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    from backend.national_elevation_client import _CACHE_TTL_DAYS

    lat, lon = 40.30, -111.70
    client = _make_client(tmp_path)
    key = client._cache_key(lat, lon)

    # Inject a stale entry (TTL + 1 day old)
    stale_at = (
        datetime.now(tz=timezone.utc) - timedelta(days=_CACHE_TTL_DAYS + 1)
    ).isoformat()
    with sqlite3.connect(str(tmp_path / "elev_cache.db")) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO elevation_slope_cache "
            "(cache_key, lat, lon, slope_degrees, aspect_degrees, fetched_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, round(lat, 3), round(lon, 3), 5.0, 225.0, stale_at, "3dep"),
        )

    rows, cols = 11, 11
    pixel_deg = 0.00009
    elev = np.zeros((rows, cols), dtype=np.float64)
    for c in range(cols):
        elev[:, c] = c * 4.0

    west, south = -111.70, 40.30
    east = west + cols * pixel_deg
    north = south + rows * pixel_deg
    transform = from_bounds(west, south, east, north, cols, rows)

    mock_ds = MagicMock()
    mock_ds.__enter__ = MagicMock(return_value=mock_ds)
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = transform
    mock_ds.bounds = BoundingBox(west, south, east, north)
    mock_ds.nodata = None
    mock_ds.read = MagicMock(return_value=elev)
    mock_ds.window_transform = MagicMock(return_value=transform)

    with patch("rasterio.open", return_value=mock_ds) as mock_open:
        client.get_slope_and_aspect(lat, lon)

    assert mock_open.call_count == 1, "Stale entry should trigger a new 3DEP fetch"
