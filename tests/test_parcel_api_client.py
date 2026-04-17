"""Tests for backend/parcel_api_client.py (RegridParcelClient)."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import urllib.error
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.parcel_api_client import RegridParcelClient, RegridParcelResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geojson_response(
    parcel_id: str = "UT-12345",
    address: str = "3184 FOOTHILL DR",
    state: str = "UT",
    county: str = "Utah County",
    acres: float = 0.42,
) -> dict[str, Any]:
    """Return a minimal Regrid GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-111.655, 40.251],
                            [-111.654, 40.251],
                            [-111.654, 40.252],
                            [-111.655, 40.252],
                            [-111.655, 40.251],
                        ]
                    ],
                },
                "properties": {
                    "fields": {
                        "parcelnumb": parcel_id,
                        "address": address,
                        "owner": "SMITH JOHN",
                        "usedesc": "RESIDENTIAL",
                        "ll_gisacre": acres,
                        "state_abbr": state,
                        "county": county,
                    }
                },
            }
        ],
    }


class _FakeHTTPResponse:
    """Minimal urllib-compatible response wrapper."""

    def __init__(self, payload: Any) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_client(tmp_path) -> RegridParcelClient:
    return RegridParcelClient(
        api_key="test-key",
        cache_db_path=str(tmp_path / "cache.db"),
        timeout_seconds=5,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_successful_parse(tmp_path):
    """Successful API response with valid GeoJSON populates all result fields."""
    payload = _make_geojson_response()
    client = _make_client(tmp_path)

    with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
        result = client.fetch_parcel(40.2515, -111.6548)

    assert result is not None
    assert isinstance(result, RegridParcelResult)
    assert result.parcel_id == "UT-12345"
    assert result.parcel_address == "3184 FOOTHILL DR"
    assert result.state == "UT"
    assert result.county == "Utah County"
    assert result.area_m2 is not None
    assert abs(result.area_m2 - 0.42 * 4046.8564) < 1.0
    assert result.geometry["type"] == "Polygon"
    assert result.source == "regrid_api"
    assert result.cached is False


def test_empty_features_returns_none(tmp_path):
    """Empty features array causes fetch_parcel to return None."""
    payload = {"type": "FeatureCollection", "features": []}
    client = _make_client(tmp_path)

    with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
        result = client.fetch_parcel(40.2515, -111.6548)

    assert result is None


def test_http_401_returns_none(tmp_path):
    """HTTP 401 causes fetch_parcel to return None without raising."""
    client = _make_client(tmp_path)
    err = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs={}, fp=None)

    with patch("urllib.request.urlopen", side_effect=err):
        result = client.fetch_parcel(40.2515, -111.6548)

    assert result is None


def test_http_429_returns_none(tmp_path):
    """HTTP 429 causes fetch_parcel to return None without raising."""
    client = _make_client(tmp_path)
    err = urllib.error.HTTPError(url="", code=429, msg="Too Many Requests", hdrs={}, fp=None)

    with patch("urllib.request.urlopen", side_effect=err):
        result = client.fetch_parcel(40.2515, -111.6548)

    assert result is None


def test_network_timeout_returns_none(tmp_path):
    """Network timeout causes fetch_parcel to return None without raising."""
    client = _make_client(tmp_path)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = client.fetch_parcel(40.2515, -111.6548)

    assert result is None


def test_cache_hit_on_second_call(tmp_path):
    """Second call with identical coordinates reads from cache, makes no HTTP request."""
    payload = _make_geojson_response()
    client = _make_client(tmp_path)

    with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)) as mock_open:
        r1 = client.fetch_parcel(40.2515, -111.6548)
        r2 = client.fetch_parcel(40.2515, -111.6548)

    assert r1 is not None
    assert r2 is not None
    assert r2.cached is True
    # urlopen should have been called only once (first fetch).
    assert mock_open.call_count == 1


def test_stale_cache_triggers_refetch(tmp_path):
    """Cache entry older than 90 days is ignored and a fresh API call is made."""
    client = _make_client(tmp_path)
    cache_key = "40.2515_-111.6548"
    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(days=91)).isoformat()
    stale_payload = _make_geojson_response(parcel_id="OLD-ID")

    # Inject a stale cache entry directly into SQLite.
    client._db.execute(
        "INSERT INTO parcel_cache (cache_key, response_json, fetched_at, lat, lon) VALUES (?, ?, ?, ?, ?)",
        (cache_key, json.dumps(stale_payload), stale_ts, 40.2515, -111.6548),
    )
    client._db.commit()

    fresh_payload = _make_geojson_response(parcel_id="FRESH-ID")

    with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(fresh_payload)) as mock_open:
        result = client.fetch_parcel(40.2515, -111.6548)

    assert result is not None
    assert result.parcel_id == "FRESH-ID"
    assert mock_open.call_count == 1


def test_get_cache_stats_three_entries(tmp_path):
    """get_cache_stats returns total_cached == 3 after 3 distinct cache insertions."""
    client = _make_client(tmp_path)
    ts = datetime.now(tz=timezone.utc).isoformat()
    payload_str = json.dumps(_make_geojson_response())

    for i, (lat, lon) in enumerate([(40.0, -111.0), (41.0, -112.0), (42.0, -113.0)]):
        cache_key = f"{round(lat, 5)}_{round(lon, 5)}"
        client._db.execute(
            "INSERT INTO parcel_cache (cache_key, response_json, fetched_at, lat, lon) VALUES (?, ?, ?, ?, ?)",
            (cache_key, payload_str, ts, lat, lon),
        )
    client._db.commit()

    stats = client.get_cache_stats()
    assert stats["total_cached"] == 3
