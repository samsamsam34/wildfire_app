"""
Unit tests for backend/landfire_cog_client.py.

All network calls and rasterio file operations are mocked — no real LANDFIRE
requests are made. Tests use an in-memory (tmp_path) SQLite cache.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from backend.landfire_cog_client import LandfireCOGClient, _CACHE_TTL_DAYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path, *, enabled: bool = True) -> LandfireCOGClient:
    return LandfireCOGClient(
        cache_db_path=str(tmp_path / "lf_cache.db"),
        timeout_seconds=10,
        enabled=enabled,
    )


def _make_fake_geotiff_bytes(pixel_value: float) -> bytes:
    """
    Return a minimal GeoTIFF bytes blob that rasterio can open and that
    returns `pixel_value` when sampled. Uses rasterio.io.MemoryFile + numpy.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.io import MemoryFile
        from rasterio.crs import CRS
    except ImportError:
        pytest.skip("rasterio or numpy not available")

    data = np.array([[pixel_value]], dtype=np.float32)
    buf = io.BytesIO()
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            count=1,
            dtype="float32",
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-114.0, 46.8, -113.9, 46.9, 1, 1),
            width=1,
            height=1,
        ) as ds:
            ds.write(data, 1)
        return memfile.read()


# ---------------------------------------------------------------------------
# Test 1: sample_point with mocked WCS → correct dict returned
# ---------------------------------------------------------------------------

def test_sample_point_returns_correct_values(tmp_path: Path) -> None:
    """Mocked WCS fetch returns a GeoTIFF; sample_point extracts the pixel value."""
    client = _make_client(tmp_path)
    fake_bytes = _make_fake_geotiff_bytes(65.0)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = fake_bytes

    with patch("requests.get", return_value=fake_response):
        result = client.sample_point(46.87, -113.99, ["canopy"])

    assert result["canopy"] is not None
    assert abs(result["canopy"] - 65.0) < 1.0


# ---------------------------------------------------------------------------
# Test 2: network timeout → returns None values, no exception
# ---------------------------------------------------------------------------

def test_network_timeout_returns_none(tmp_path: Path) -> None:
    """WCS fetch raises a timeout; sample_point returns None for the layer."""
    client = _make_client(tmp_path)

    import requests as _requests
    with patch("requests.get", side_effect=_requests.exceptions.Timeout("timeout")):
        result = client.sample_point(46.87, -113.99, ["fuel", "slope"])

    assert result["fuel"] is None
    assert result["slope"] is None


# ---------------------------------------------------------------------------
# Test 3: cache hit — fetch called only once across two identical calls
# ---------------------------------------------------------------------------

def test_cache_hit_avoids_second_fetch(tmp_path: Path) -> None:
    """Second call for same (layer, lat, lon) hits cache; WCS not re-fetched."""
    client = _make_client(tmp_path)
    fake_bytes = _make_fake_geotiff_bytes(14.3)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = fake_bytes

    with patch("requests.get", return_value=fake_response) as mock_get:
        r1 = client.sample_point(46.87, -113.99, ["slope"])
        r2 = client.sample_point(46.87, -113.99, ["slope"])

    assert r1["slope"] is not None
    assert r2["slope"] == r1["slope"]
    assert mock_get.call_count == 1, "WCS should be called only once; second call should hit cache"


# ---------------------------------------------------------------------------
# Test 4: stale cache entry → re-fetch triggered
# ---------------------------------------------------------------------------

def test_stale_cache_triggers_refetch(tmp_path: Path) -> None:
    """A cache entry older than TTL is ignored; a fresh WCS fetch is made."""
    client = _make_client(tmp_path)
    layer_id = "canopy"
    lat, lon = 46.87, -113.99

    # Inject a stale cache entry (TTL + 1 day old)
    stale_at = (
        datetime.now(tz=timezone.utc) - timedelta(days=_CACHE_TTL_DAYS + 1)
    ).isoformat()
    cache_key = f"{layer_id}_{round(lat, 3)}_{round(lon, 3)}"
    with sqlite3.connect(str(tmp_path / "lf_cache.db")) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO landfire_pixel_cache "
            "(cache_key, layer_id, lat, lon, pixel_value, fetched_at, coverage_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cache_key, layer_id, round(lat, 3), round(lon, 3), 42.0, stale_at, "landfire_wcs:LF2024_CC_CONUS"),
        )

    fake_bytes = _make_fake_geotiff_bytes(77.0)
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = fake_bytes

    with patch("requests.get", return_value=fake_response) as mock_get:
        result = client.sample_point(lat, lon, [layer_id])

    assert mock_get.call_count == 1, "Stale entry should trigger a new WCS fetch"
    assert result[layer_id] is not None


# ---------------------------------------------------------------------------
# Test 5: 3 layers requested, 2 cached, 1 not — only 1 WCS fetch
# ---------------------------------------------------------------------------

def test_partial_cache_fetches_only_missing_layers(tmp_path: Path) -> None:
    """When 2 of 3 layers are cached, exactly 1 WCS call is made."""
    client = _make_client(tmp_path)
    lat, lon = 46.87, -113.99
    now = datetime.now(tz=timezone.utc).isoformat()

    # Pre-populate cache for "slope" and "aspect"
    with sqlite3.connect(str(tmp_path / "lf_cache.db")) as conn:
        for lid, val in [("slope", 12.5), ("aspect", 225.0)]:
            key = f"{lid}_{round(lat, 3)}_{round(lon, 3)}"
            conn.execute(
                "INSERT OR REPLACE INTO landfire_pixel_cache "
                "(cache_key, layer_id, lat, lon, pixel_value, fetched_at, coverage_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, lid, round(lat, 3), round(lon, 3), val, now, "test"),
            )

    fake_bytes = _make_fake_geotiff_bytes(91.0)
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = fake_bytes

    with patch("requests.get", return_value=fake_response) as mock_get:
        result = client.sample_point(lat, lon, ["slope", "aspect", "fuel"])

    assert mock_get.call_count == 1, "Only the uncached 'fuel' layer should trigger a WCS fetch"
    assert result["slope"] == 12.5
    assert result["aspect"] == 225.0
    assert result["fuel"] is not None


# ---------------------------------------------------------------------------
# Test 6: all layers return None → dict with all None, no exception
# ---------------------------------------------------------------------------

def test_all_layers_fail_returns_all_none(tmp_path: Path) -> None:
    """All WCS fetches fail; result dict has None for every layer, no exception raised."""
    client = _make_client(tmp_path)

    import requests as _requests
    with patch("requests.get", side_effect=_requests.exceptions.ConnectionError("unreachable")):
        result = client.sample_point(46.87, -113.99, ["fuel", "canopy", "slope", "aspect"])

    assert all(v is None for v in result.values())
    assert set(result.keys()) == {"fuel", "canopy", "slope", "aspect"}


# ---------------------------------------------------------------------------
# Test 7: WCS returns XML exception report → returns None, no exception
# ---------------------------------------------------------------------------

def test_wcs_exception_report_returns_none(tmp_path: Path) -> None:
    """WCS returns an XML ServiceExceptionReport; client returns None without raising."""
    client = _make_client(tmp_path)

    xml_body = (
        b'<?xml version="1.0" ?>'
        b'<ServiceExceptionReport version="1.2.0">'
        b'<ServiceException code="InvalidParameterValue">Coverage not found.</ServiceException>'
        b'</ServiceExceptionReport>'
    )
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "application/xml"}
    fake_response.content = xml_body
    fake_response.text = xml_body.decode()

    with patch("requests.get", return_value=fake_response):
        result = client.sample_point(46.87, -113.99, ["fuel"])

    assert result["fuel"] is None


# ---------------------------------------------------------------------------
# Test 8: disabled client returns all None without any network call
# ---------------------------------------------------------------------------

def test_disabled_client_returns_none_without_fetching(tmp_path: Path) -> None:
    """When enabled=False, sample_point returns all None and makes no network call."""
    client = _make_client(tmp_path, enabled=False)

    with patch("requests.get") as mock_get:
        result = client.sample_point(46.87, -113.99, ["fuel", "canopy"])

    assert result["fuel"] is None
    assert result["canopy"] is None
    assert mock_get.call_count == 0


# ---------------------------------------------------------------------------
# Test 9: get_available_layers returns expected layer IDs
# ---------------------------------------------------------------------------

def test_get_available_layers(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    layers = client.get_available_layers()
    for expected in ("fuel", "canopy", "slope", "aspect", "dem"):
        assert expected in layers


# ---------------------------------------------------------------------------
# Test 10: get_cache_stats reflects entries after successful fetches
# ---------------------------------------------------------------------------

def test_cache_stats_reflects_fetched_layers(tmp_path: Path) -> None:
    """After two successful fetches, get_cache_stats reports 1 entry per layer."""
    client = _make_client(tmp_path)
    fake_bytes = _make_fake_geotiff_bytes(50.0)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {"Content-Type": "image/tiff"}
    fake_response.content = fake_bytes

    with patch("requests.get", return_value=fake_response):
        client.sample_point(46.87, -113.99, ["fuel"])
        client.sample_point(46.88, -113.98, ["fuel"])  # different coord → new entry

    stats = client.get_cache_stats()
    assert stats.get("fuel", 0) >= 2
