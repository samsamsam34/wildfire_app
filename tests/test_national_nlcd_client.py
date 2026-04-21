"""
Unit tests for backend/national_nlcd_client.py.

All network calls and rasterio operations are mocked — no real NLCD WCS requests
are made. Tests use an in-memory (tmp_path) SQLite cache.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_nlcd_geotiff(
    lat_center: float,
    lon_center: float,
    pixel_classes: list[list[int]],  # 2D array [row][col] of NLCD class codes
) -> bytes:
    """
    Build a minimal GeoTIFF containing the given NLCD class grid.
    Grid origin is (lon_center - pad, lat_center + pad) with 30m/pixel.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    height = len(pixel_classes)
    width = len(pixel_classes[0]) if height > 0 else 0
    # ~30 m/pixel ≈ 0.00027° at 45° lat
    pixel_deg = 0.00027
    half_w = (width / 2) * pixel_deg
    half_h = (height / 2) * pixel_deg

    west = lon_center - half_w
    east = lon_center + half_w
    south = lat_center - half_h
    north = lat_center + half_h

    arr = np.array(pixel_classes, dtype=np.uint8)
    with MemoryFile() as mf:
        with mf.open(
            driver="GTiff",
            count=1,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=from_bounds(west, south, east, north, width, height),
            width=width,
            height=height,
        ) as ds:
            ds.write(arr, 1)
        return mf.read()


def _make_client(tmp_path: Path, *, enabled: bool = True):
    from backend.national_nlcd_client import NationalNLCDClient
    return NationalNLCDClient(
        cache_db_path=str(tmp_path / "nlcd_cache.db"),
        timeout_seconds=10,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# Test 1: Wildland pixels exist nearby → returns approximate distance
# ---------------------------------------------------------------------------

def test_wildland_pixels_nearby_returns_distance(tmp_path: Path) -> None:
    """WCS returns a tile with wildland pixels; client returns ~correct distance."""
    # 5×5 grid: center (2,2) is developed (22), corners are wildland (42=Evergreen Forest)
    grid = [
        [42, 42, 42, 42, 42],
        [42, 22, 22, 22, 42],
        [42, 22, 22, 22, 42],  # query point at center pixel (2,2)
        [42, 22, 22, 22, 42],
        [42, 42, 42, 42, 42],
    ]
    lat, lon = 40.30, -111.70
    geotiff = _make_nlcd_geotiff(lat, lon, grid)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = geotiff

    client = _make_client(tmp_path)
    with patch("requests.get", return_value=fake_response):
        dist = client.get_wildland_distance_m(lat, lon)

    assert dist is not None, "Should return a distance when wildland pixels exist"
    # Nearest wildland pixel is roughly 1 pixel away (~30 m); allow up to 200 m tolerance
    assert dist < 200.0, f"Expected distance < 200 m, got {dist}"


# ---------------------------------------------------------------------------
# Test 2: Property on wildland pixel → returns 0.0
# ---------------------------------------------------------------------------

def test_property_on_wildland_returns_zero(tmp_path: Path) -> None:
    """When the center pixel is a wildland class, distance should be ~0."""
    # All wildland
    grid = [
        [42, 42, 42],
        [42, 42, 42],
        [42, 42, 42],
    ]
    lat, lon = 46.87, -113.99
    geotiff = _make_nlcd_geotiff(lat, lon, grid)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = geotiff

    client = _make_client(tmp_path)
    with patch("requests.get", return_value=fake_response):
        dist = client.get_wildland_distance_m(lat, lon)

    assert dist is not None
    assert dist < 50.0, f"Property on wildland pixel should return ~0 m, got {dist}"


# ---------------------------------------------------------------------------
# Test 3: All pixels non-wildland → returns sample_radius_m
# ---------------------------------------------------------------------------

def test_no_wildland_returns_sample_radius(tmp_path: Path) -> None:
    """When no wildland pixels are in the window, returns sample_radius_m."""
    # All developed (22)
    grid = [[22, 22, 22], [22, 22, 22], [22, 22, 22]]
    lat, lon = 40.30, -111.70
    geotiff = _make_nlcd_geotiff(lat, lon, grid)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = geotiff

    client = _make_client(tmp_path)
    radius_m = 750.0
    with patch("requests.get", return_value=fake_response):
        dist = client.get_wildland_distance_m(lat, lon, sample_radius_m=radius_m)

    assert dist == radius_m, f"No wildland → should return sample_radius_m={radius_m}, got {dist}"


# ---------------------------------------------------------------------------
# Test 4: Network exception → returns None, no exception raised
# ---------------------------------------------------------------------------

def test_network_error_returns_none(tmp_path: Path) -> None:
    """Any network failure returns None without raising."""
    import requests as _req
    client = _make_client(tmp_path)
    with patch("requests.get", side_effect=_req.exceptions.ConnectionError("unreachable")):
        dist = client.get_wildland_distance_m(40.30, -111.70)

    assert dist is None


# ---------------------------------------------------------------------------
# Test 5: Cache hit → requests.get not called second time
# ---------------------------------------------------------------------------

def test_cache_hit_avoids_second_fetch(tmp_path: Path) -> None:
    """Second call for same (lat, lon) hits cache; WCS is not re-fetched."""
    grid = [[41, 41], [41, 41]]
    lat, lon = 40.30, -111.70
    geotiff = _make_nlcd_geotiff(lat, lon, grid)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = geotiff

    client = _make_client(tmp_path)
    with patch("requests.get", return_value=fake_response) as mock_get:
        d1 = client.get_wildland_distance_m(lat, lon)
        d2 = client.get_wildland_distance_m(lat, lon)

    assert d1 is not None
    assert d1 == d2
    assert mock_get.call_count == 1, "Second call should hit cache, not re-fetch"


# ---------------------------------------------------------------------------
# Test 6: disabled client returns None without fetching
# ---------------------------------------------------------------------------

def test_disabled_client_returns_none(tmp_path: Path) -> None:
    """enabled=False → get_wildland_distance_m returns None, no network calls."""
    client = _make_client(tmp_path, enabled=False)
    with patch("requests.get") as mock_get:
        dist = client.get_wildland_distance_m(40.30, -111.70)

    assert dist is None
    assert mock_get.call_count == 0


# ---------------------------------------------------------------------------
# Test 7: WCS returns XML exception report → returns None
# ---------------------------------------------------------------------------

def test_wcs_exception_report_returns_none(tmp_path: Path) -> None:
    """WCS returns an XML error body; client returns None without raising."""
    xml_body = (
        b'<?xml version="1.0"?>'
        b'<ServiceExceptionReport><ServiceException>Coverage not found.</ServiceException>'
        b'</ServiceExceptionReport>'
    )
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "application/xml"}
    fake_response.content = xml_body
    fake_response.text = xml_body.decode()

    client = _make_client(tmp_path)
    with patch("requests.get", return_value=fake_response):
        dist = client.get_wildland_distance_m(40.30, -111.70)

    assert dist is None


# ---------------------------------------------------------------------------
# Test 8: get_cache_stats reflects fetched entries
# ---------------------------------------------------------------------------

def test_cache_stats_after_fetch(tmp_path: Path) -> None:
    """After two successful fetches for different coordinates, cache shows 2 entries."""
    client = _make_client(tmp_path)
    # Inject two entries directly to test get_cache_stats without a live WCS fetch
    client._cache_set(client._cache_key(40.30, -111.70), 40.30, -111.70, 250.0)
    client._cache_set(client._cache_key(40.50, -111.90), 40.50, -111.90, 500.0)

    stats = client.get_cache_stats()
    assert stats.get("wildland_distance", 0) >= 2
