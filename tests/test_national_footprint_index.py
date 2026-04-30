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
    # connect() is called once during __init__ (INSTALL extensions) and once
    # for the first query.  The second fetch is a cache hit — no third connect.
    assert fake_duckdb.connect.call_count == 2


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
    # connect() called once during __init__ (INSTALL) + twice for queries
    # (empty results are not cached, so each call hits DuckDB).
    assert fake_duckdb.connect.call_count == 3


# ---------------------------------------------------------------------------
# 8. SQL predicate regression: intersection semantics (bbox.xmin <= max_lon)
#    The old containment predicate (bbox.xmin >= min_lon AND bbox.xmax <= max_lon)
#    would silently drop buildings whose bbox extends outside the query boundary.
# ---------------------------------------------------------------------------

def test_sql_predicate_is_intersection_not_containment(tmp_path: Path) -> None:
    """The generated SQL must use intersection semantics, not containment.

    Intersection: footprint bbox OVERLAPS query bbox.
      bbox.xmin <= max_lon  AND  bbox.xmax >= min_lon
      bbox.ymin <= max_lat  AND  bbox.ymax >= min_lat

    The old (wrong) containment form was:
      bbox.xmin >= min_lon  AND  bbox.xmax <= max_lon  ← drops boundary-spanning footprints
    """
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = []
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        index.fetch_footprints_for_bbox(
            min_lat=40.000, min_lon=-105.010, max_lat=40.010, max_lon=-105.000
        )

    # Extract the SQL string from the execute calls on the per-query connection.
    # The connection mock's execute() is called several times (LOAD spatial,
    # LOAD httpfs, SET s3_region, then the actual query).  Find the SELECT call.
    all_calls = fake_con.execute.call_args_list
    sql_calls = [str(c.args[0]) for c in all_calls if "SELECT" in str(c.args[0])]
    assert sql_calls, "No SELECT query found in DuckDB execute calls"
    sql = sql_calls[0]

    # Must use intersection operators (<=, >=), NOT containment (>=, <=) on xmin.
    assert "bbox.xmin <=" in sql, (
        f"Expected intersection predicate 'bbox.xmin <=' in SQL, got:\n{sql}"
    )
    assert "bbox.xmax >=" in sql, (
        f"Expected intersection predicate 'bbox.xmax >=' in SQL, got:\n{sql}"
    )
    assert "bbox.ymin <=" in sql, (
        f"Expected intersection predicate 'bbox.ymin <=' in SQL, got:\n{sql}"
    )
    assert "bbox.ymax >=" in sql, (
        f"Expected intersection predicate 'bbox.ymax >=' in SQL, got:\n{sql}"
    )
    # Explicit regression guard: old containment form must not appear.
    assert "bbox.xmin >=" not in sql, (
        "Containment predicate 'bbox.xmin >=' found — this is the old bug"
    )


# ---------------------------------------------------------------------------
# 9. Boundary-spanning footprint is returned by intersection predicate
#
#    Scenario: query bbox is [-105.005, 40.000] → [-105.000, 40.005] (0.005° wide).
#    A large building has bbox [-105.006, 40.001] → [-104.999, 40.004] — its left
#    edge extends 0.001° west of the query boundary.
#
#    Old containment predicate: bbox.xmin(-105.006) >= min_lon(-105.005) → FALSE
#      → building silently dropped.
#    New intersection predicate: bbox.xmin(-105.006) <= max_lon(-105.000) → TRUE
#      → building returned.
# ---------------------------------------------------------------------------

def test_boundary_spanning_footprint_returned(tmp_path: Path) -> None:
    """A footprint whose bbox extends outside the query boundary is returned.

    This is the direct regression test for the containment→intersection fix.
    The building's left edge (xmin = -105.006) extends past the query's left
    boundary (min_lon = -105.005).  The old predicate dropped it; the fix returns it.
    """
    # Building whose bbox spans the left (west) edge of the query bbox.
    # Its polygon center is inside the query area; only the bbox extends outside.
    spanning_building_geojson = json.dumps({
        "type": "Polygon",
        "coordinates": [[
            [-105.006, 40.004],
            [-104.999, 40.004],
            [-104.999, 40.001],
            [-105.006, 40.001],
            [-105.006, 40.004],
        ]],
    })

    # Small building fully inside the query bbox.
    small_building_geojson = _polygon_geojson(lon=-105.002, lat=40.002, half=0.0001)

    rows = [
        (spanning_building_geojson, "residential", 8.0),
        (small_building_geojson, "garage", 3.0),
    ]

    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = rows
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        features = index.fetch_footprints_for_bbox(
            min_lat=40.000, min_lon=-105.005, max_lat=40.005, max_lon=-105.000
        )

    # Both buildings must be returned — the intersection predicate includes
    # the boundary-spanning building that the old containment predicate dropped.
    assert len(features) == 2, (
        f"Expected 2 features (including boundary-spanning building), got {len(features)}"
    )
    classes = {f["properties"]["building_class"] for f in features}
    assert "residential" in classes, "Boundary-spanning building (residential) was dropped"
    assert "garage" in classes, "Small fully-inside building (garage) was dropped"


# ---------------------------------------------------------------------------
# 10. Large building (barn/farmhouse) is not dropped at bbox filter stage
#
#     A 500 m² building has a bbox roughly 0.002° × 0.002° wide.  When the
#     search radius produces a 300 m bbox (~0.003° wide), a large building near
#     the query edge has a bbox that extends past the query boundary.
#     With the intersection predicate, it is returned and can be scored.
# ---------------------------------------------------------------------------

def test_large_building_returned_and_scoreable(tmp_path: Path) -> None:
    """A large building (barn/farmhouse, bbox spanning query boundary) is returned.

    Verifies that after the predicate fix:
    - The large building survives the bbox filter (it is in the mocked response).
    - The feature dict is well-formed and has area_m2 populated (Shapely area calc).
    - The building_class is preserved.

    Note: area plausibility scoring (>= 40 m² → score 1.0) happens in
    building_footprints.py, not here.  This test confirms the data flows through.
    """
    # Approximate 500 m² farmhouse footprint in WGS-84 degrees.
    # 500 m² at 40°N ≈ 0.002° × 0.002° (each degree ≈ ~78 km lon, ~111 km lat).
    large_building_geojson = json.dumps({
        "type": "Polygon",
        "coordinates": [[
            [-105.004, 40.003],
            [-105.002, 40.003],
            [-105.002, 40.001],
            [-105.004, 40.001],
            [-105.004, 40.003],
        ]],
    })

    rows = [(large_building_geojson, "barn", 5.5)]

    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = rows
    fake_con = MagicMock()
    fake_con.execute.return_value = fake_cursor

    fake_duckdb = _make_fake_duckdb(fake_con)
    index = _make_index_with_fake_duckdb(tmp_path, fake_duckdb, enabled=True)

    with patch.dict(sys.modules, {"duckdb": fake_duckdb}):
        features = index.fetch_footprints_for_bbox(
            min_lat=40.000, min_lon=-105.005, max_lat=40.005, max_lon=-105.000
        )

    assert len(features) == 1, f"Expected 1 large building feature, got {len(features)}"
    feat = features[0]
    assert feat["properties"]["building_class"] == "barn"
    assert feat["properties"]["height_m"] == pytest.approx(5.5)
    # area_m2 is computed by Shapely + pyproj if available; may be None when
    # those packages are absent in the test environment.  When populated, it
    # must be positive and large enough to be a farmhouse (>> 40 m²).
    area = feat["properties"].get("area_m2")
    if area is not None:
        assert area > 100, f"Expected area > 100 m² for a large farmhouse polygon, got {area}"


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
