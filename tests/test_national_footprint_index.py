"""Tests for NationalFootprintIndex — DuckDB/Overture footprint tile cache.

All DuckDB queries are mocked; no real S3 or network access is performed.
DuckDB is not required to be installed for these tests to run — a fake module
is injected into sys.modules where needed.
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — fake duckdb module and index factory
# ---------------------------------------------------------------------------

def _make_fake_duckdb(con: MagicMock) -> types.ModuleType:
    """Return a minimal fake duckdb module that returns *con* from connect()."""
    mod = types.ModuleType("duckdb")
    mod.connect = MagicMock(return_value=con)
    return mod


def _make_index_with_fake_duckdb(tmp_path: Path, fake_duckdb: types.ModuleType, **kwargs):
    """Construct NationalFootprintIndex with duckdb faked in sys.modules."""
    from backend.national_footprint_index import NationalFootprintIndex

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        index = NationalFootprintIndex(
            cache_db_path=str(tmp_path / "fp_cache.db"),
            **kwargs,
        )
    return index


def _polygon_geojson(lon: float = -105.0, lat: float = 40.0, half: float = 0.0002) -> str:
    coords = [
        [lon - half, lat + half],
        [lon + half, lat + half],
        [lon + half, lat - half],
        [lon - half, lat - half],
        [lon - half, lat + half],
    ]
    return json.dumps({"type": "Polygon", "coordinates": [coords]})


# ---------------------------------------------------------------------------
# 1. Successful parse — 3 rows returned from mocked DuckDB
# ---------------------------------------------------------------------------

def test_successful_parse_returns_features(tmp_path: Path) -> None:
    rows = [
        (_polygon_geojson(-105.001, 40.001), "residential", 6.5),
        (_polygon_geojson(-105.002, 40.002), "garage", None),
        (_polygon_geojson(-105.003, 40.003), None, 3.2),
    ]

    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = rows

    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)

    from backend.national_footprint_index import NationalFootprintIndex

    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    # Patch sys.modules during query so _query_overture's `import duckdb` resolves.
    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        features = index.fetch_footprints_for_bbox(
            min_lat=39.995, min_lon=-105.005, max_lat=40.005, max_lon=-104.995
        )

    assert len(features) == 3
    for feat in features:
        assert "geometry" in feat
        assert feat["properties"]["source"] == "overture"
    assert features[0]["properties"]["building_class"] == "residential"
    assert features[1]["properties"]["building_class"] == "garage"
    assert features[2]["properties"]["building_class"] is None
    assert features[0]["properties"]["height_m"] == pytest.approx(6.5)
    assert features[1]["properties"]["height_m"] is None


# ---------------------------------------------------------------------------
# 2. DuckDB ImportError → index disabled, returns []
# ---------------------------------------------------------------------------

def test_duckdb_import_error_disables_index(tmp_path: Path) -> None:
    from backend.national_footprint_index import NationalFootprintIndex

    with patch.dict(sys.modules, {"duckdb": None}):
        index = NationalFootprintIndex(
            cache_db_path=str(tmp_path / "fp_cache.db"),
            enabled=True,
        )
    assert index.enabled is False
    result = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)
    assert result == []


# ---------------------------------------------------------------------------
# 3. S3 / DuckDB query exception → returns []
# ---------------------------------------------------------------------------

def test_s3_query_exception_returns_empty(tmp_path: Path) -> None:
    fake_con = MagicMock()
    fake_con.execute.side_effect = RuntimeError("S3 connection refused")

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        result = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)

    assert result == []


# ---------------------------------------------------------------------------
# 4. Cache hit on second call — DuckDB connect called only once
# ---------------------------------------------------------------------------

def test_cache_hit_on_second_call(tmp_path: Path) -> None:
    rows = [(_polygon_geojson(), "residential", 5.0)]

    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = rows
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        first = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)
        second = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)

    assert len(first) == 1
    assert len(second) == 1
    # connect() should only be called once (cache hit on second call).
    assert fake_duckdb.connect.call_count == 1


# ---------------------------------------------------------------------------
# 5. Stale cache (181 days old) → triggers re-query
# ---------------------------------------------------------------------------

def test_stale_cache_triggers_requery(tmp_path: Path) -> None:
    rows = [(_polygon_geojson(), "residential", 5.0)]
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = rows
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(days=181)).isoformat()
    stale_features = json.dumps([{
        "geometry": {},
        "properties": {"source": "overture", "building_class": "old", "height_m": None, "area_m2": None},
    }])

    cache_key = "39.99_-105.01_40.01_-104.99"
    index._db.execute(
        "INSERT OR REPLACE INTO footprint_cache"
        " (cache_key, features_json, fetched_at, bbox_str, feature_count)"
        " VALUES (?,?,?,?,?)",
        (cache_key, stale_features, stale_ts, "-105.01,39.99,-104.99,40.01", 1),
    )
    index._db.commit()

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        result = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)

    # Should have re-queried; new rows come from mocked DuckDB.
    assert len(result) == 1
    assert result[0]["properties"]["building_class"] == "residential"
    assert fake_duckdb.connect.called


# ---------------------------------------------------------------------------
# 6. Empty result → not cached
# ---------------------------------------------------------------------------

def test_empty_result_not_cached(tmp_path: Path) -> None:
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = []
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        first = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)
        second = index.fetch_footprints_for_bbox(39.99, -105.01, 40.01, -104.99)

    assert first == []
    assert second == []
    # connect() called twice — empty results are not cached.
    assert fake_duckdb.connect.call_count == 2


# ---------------------------------------------------------------------------
# 7. get_cache_stats — 2 bbox entries × 10 features each
# ---------------------------------------------------------------------------

def test_get_cache_stats(tmp_path: Path) -> None:
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = []
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    features_json = json.dumps([{"geometry": {}, "properties": {}} for _ in range(10)])
    now = datetime.now(tz=timezone.utc).isoformat()

    index._db.execute(
        "INSERT INTO footprint_cache (cache_key, features_json, fetched_at, bbox_str, feature_count)"
        " VALUES (?,?,?,?,?)",
        ("key_a", features_json, now, "bbox_a", 10),
    )
    index._db.execute(
        "INSERT INTO footprint_cache (cache_key, features_json, fetched_at, bbox_str, feature_count)"
        " VALUES (?,?,?,?,?)",
        ("key_b", features_json, now, "bbox_b", 10),
    )
    index._db.commit()

    stats = index.get_cache_stats()
    assert stats["total_cached_bboxes"] == 2
    assert stats["total_cached_features"] == 20
    assert stats["oldest_entry"] is not None
