"""On-demand national building footprint index using Overture Maps + DuckDB.

Queries Overture Maps building footprints from public S3-hosted GeoParquet files
via DuckDB httpfs range requests — no full dataset download required.  Results are
cached in a local SQLite database (180-day TTL) so each unique 600 m × 600 m tile
is fetched at most twice per year.

Architecture
------------
Each call to ``get_footprints_near_point`` converts the requested radius into an
approximate lat/lon bounding box, rounds the bbox to 4 decimal places (~11 m
precision at mid-latitudes) to maximise cache reuse across nearby assessments, then
checks the SQLite cache before issuing a DuckDB query.

DuckDB queries Overture's S3 GeoParquet directly via HTTP range requests (httpfs
extension).  Only the rows whose ``bbox`` column intersects the requested bounding
box are fetched — DuckDB's Parquet predicate pushdown makes this efficient even
against the national dataset.

DuckDB is an optional dependency.  If it is not importable the index disables
itself silently; ``BuildingFootprintClient`` then falls through to its existing
``provider_unavailable`` behavior.

Overture S3 path
----------------
The default path references the ``2026-04-15.0`` release.  Override with the
``WF_OVERTURE_RELEASE`` environment variable to pin a specific release date::

    export WF_OVERTURE_RELEASE=2026-04-15.0

See https://docs.overturemaps.org for current release dates.

Overture schema (buildings/type=building)
-----------------------------------------
Columns used by this module:

* ``geometry``       — WKB-encoded Polygon / MultiPolygon (footprint or roofprint).
* ``class``          — Building use classification string (e.g. ``residential``,
                       ``commercial``, ``barn``).  May be NULL.
* ``height``         — Building height in metres above ground.  May be NULL.
* ``bbox``           — Struct with ``xmin``, ``xmax``, ``ymin``, ``ymax`` used for
                       Parquet predicate pushdown.

Geometry is decoded with ``ST_GeomFromWkb(geometry)`` (DuckDB spatial extension)
and serialised to GeoJSON with ``ST_AsGeoJSON``.  Area is computed by Shapely in
Python after deserialisation to avoid requiring PROJ inside DuckDB.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("wildfire_app.national_footprint_index")

# Default Overture release — update when a newer release is available.
_DEFAULT_OVERTURE_RELEASE = "2026-04-15.0"

_CACHE_TTL_DAYS = 180

# 1 degree of latitude ≈ 111,000 m everywhere (used for bbox conversion).
_DEG_PER_METER_LAT = 1.0 / 111_000.0


class NationalFootprintIndex:
    """On-demand building footprint lookup via Overture Maps GeoParquet on S3.

    Queries are executed through DuckDB's ``httpfs`` extension using HTTP range
    requests against Overture's public S3 bucket — the full dataset is never
    downloaded.  Results are cached locally in SQLite.

    Parameters
    ----------
    cache_db_path:
        Path to the SQLite cache database.  Created on first use.
        Default: ``data/footprint_cache.db``.
    timeout_seconds:
        DuckDB query timeout in seconds.  Default: 15.
    enabled:
        Set to ``False`` to disable all queries (cache reads still work).
    max_buildings_per_query:
        Maximum number of buildings returned per bbox query.  Default: 500.
    """

    def __init__(
        self,
        cache_db_path: str = "data/footprint_cache.db",
        timeout_seconds: int = 15,
        enabled: bool = True,
        max_buildings_per_query: int = 500,
    ) -> None:
        self._cache_db_path = Path(str(cache_db_path or "data/footprint_cache.db"))
        self._timeout = int(timeout_seconds)
        self.enabled = bool(enabled)
        self._max_buildings = int(max_buildings_per_query)
        self._db: sqlite3.Connection | None = None
        self._overture_release = (
            os.environ.get("WF_OVERTURE_RELEASE", _DEFAULT_OVERTURE_RELEASE).strip()
            or _DEFAULT_OVERTURE_RELEASE
        )

        # Probe DuckDB availability once at construction time and pre-install
        # the required extensions.  INSTALL is idempotent but slow (~1 s on a
        # cold DuckDB home dir); running it here means each per-query connection
        # only needs LOAD (fast, per-connection, always required).
        self._duckdb_available = False
        if self.enabled:
            try:
                import duckdb as _duckdb
                self._duckdb_available = True
                # Install extensions once at startup.  Errors are suppressed
                # because the extensions may already be present in the DuckDB
                # extensions directory from a previous run.
                _con = _duckdb.connect()
                try:
                    _con.execute("INSTALL spatial;")
                except Exception:
                    pass
                try:
                    _con.execute("INSTALL httpfs;")
                except Exception:
                    pass
                _con.close()
            except ImportError:
                LOGGER.warning(
                    "national_footprint_index duckdb_not_installed"
                    " — national footprint index unavailable"
                )
                self.enabled = False

        self._init_cache()

    # ------------------------------------------------------------------
    # S3 path
    # ------------------------------------------------------------------

    @property
    def _overture_s3_path(self) -> str:
        """Overture GeoParquet S3 glob path for the configured release."""
        return (
            f"s3://overturemaps-us-west-2/release/{self._overture_release}/"
            "theme=buildings/type=building/*"
        )

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        """Create the SQLite cache database and table if needed."""
        try:
            self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(
                str(self._cache_db_path), check_same_thread=False
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS footprint_cache (
                    cache_key      TEXT PRIMARY KEY,
                    features_json  TEXT NOT NULL,
                    fetched_at     TEXT NOT NULL,
                    bbox_str       TEXT NOT NULL,
                    feature_count  INTEGER NOT NULL
                )
                """
            )
            self._db.commit()
        except (sqlite3.Error, OSError) as exc:
            LOGGER.warning(
                "national_footprint_index cache_init_error path=%s error=%s",
                self._cache_db_path,
                exc,
            )
            self._db = None

    @staticmethod
    def _make_cache_key(
        min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> str:
        """Round-trip-stable cache key for a lat/lon bounding box."""
        return (
            f"{round(min_lat, 4)}_{round(min_lon, 4)}"
            f"_{round(max_lat, 4)}_{round(max_lon, 4)}"
        )

    def _cache_get(self, cache_key: str) -> list[dict[str, Any]] | None:
        """Return cached features if a fresh entry exists, else None."""
        if self._db is None:
            return None
        try:
            cur = self._db.execute(
                "SELECT features_json, fetched_at FROM footprint_cache"
                " WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
        except sqlite3.Error as exc:
            LOGGER.warning("national_footprint_index cache_read_error %s", exc)
            return None

        if row is None:
            return None

        features_json_str, fetched_at_str = row
        try:
            fetched_at = datetime.fromisoformat(fetched_at_str)
        except (ValueError, TypeError):
            return None

        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - fetched_at).days
        if age_days > _CACHE_TTL_DAYS:
            LOGGER.debug(
                "national_footprint_index cache_stale cache_key=%r age_days=%d",
                cache_key,
                age_days,
            )
            return None

        try:
            return json.loads(features_json_str)
        except json.JSONDecodeError:
            return None

    def _cache_put(
        self,
        cache_key: str,
        features: list[dict[str, Any]],
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> None:
        """Insert or replace a cache entry.  Empty result lists are NOT cached."""
        if self._db is None or not features:
            return
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        fetched_at = datetime.now(tz=timezone.utc).isoformat()
        try:
            self._db.execute(
                """
                INSERT OR REPLACE INTO footprint_cache
                    (cache_key, features_json, fetched_at, bbox_str, feature_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    json.dumps(features),
                    fetched_at,
                    bbox_str,
                    len(features),
                ),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            LOGGER.warning("national_footprint_index cache_write_error %s", exc)

    # ------------------------------------------------------------------
    # DuckDB query
    # ------------------------------------------------------------------

    def _row_to_feature(self, row: tuple) -> dict[str, Any] | None:
        """Convert a DuckDB result row to a GeoJSON-style feature dict.

        Parameters
        ----------
        row:
            ``(geometry_json_str, building_class, height)`` — columns in the
            order returned by ``_query_overture``.

        Returns
        -------
        Feature dict or ``None`` if the geometry cannot be parsed.
        """
        geometry_json, building_class, height = row
        try:
            geometry = json.loads(geometry_json) if geometry_json else None
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(geometry, dict):
            return None

        # Compute area in Python via Shapely — avoids requiring PROJ in DuckDB.
        area_m2: float | None = None
        try:
            from shapely.geometry import shape as _shape
            from shapely.ops import transform as _transform
            from pyproj import Transformer as _Transformer

            geom = _shape(geometry)
            if not geom.is_empty:
                to_3857 = _Transformer.from_crs(
                    "EPSG:4326", "EPSG:3857", always_xy=True
                ).transform
                area_m2 = round(float(_transform(to_3857, geom).area), 2)
        except Exception:
            area_m2 = None

        return {
            "geometry": geometry,
            "properties": {
                "source": "overture",
                "building_class": (
                    str(building_class).strip() if building_class else None
                ),
                "height_m": float(height) if height is not None else None,
                "area_m2": area_m2,
            },
        }

    def _query_overture(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> list[dict[str, Any]]:
        """Execute a DuckDB bbox query against the Overture S3 GeoParquet.

        Returns a list of feature dicts on success, or ``[]`` on any failure.
        Never raises.
        """
        try:
            import duckdb
        except ImportError:
            return []

        try:
            # Extensions were installed once at __init__ time; here we only
            # LOAD them into this connection (fast, required per-connection).
            con = duckdb.connect()
            con.execute("LOAD spatial;")
            con.execute("LOAD httpfs;")
            con.execute("SET s3_region='us-west-2';")

            # Intersection predicate: return any footprint whose bbox OVERLAPS
            # the query bbox.  The correct form is:
            #   footprint.left  <= query.right   (bbox.xmin <= max_lon)
            #   footprint.right >= query.left    (bbox.xmax >= min_lon)
            #   footprint.bottom <= query.top    (bbox.ymin <= max_lat)
            #   footprint.top  >= query.bottom   (bbox.ymax >= min_lat)
            # The previous containment predicate (xmin >= min_lon AND xmax <=
            # max_lon) silently dropped buildings whose bbox extended outside
            # the query boundary — common for large structures and buildings
            # near the edge of the search radius.  No bbox buffer margin is
            # needed: exact-geometry matching in building_footprints.py still
            # filters to the precise search radius after this coarse filter.
            query = f"""
                SELECT
                    ST_AsGeoJSON(
                        CASE
                            WHEN typeof(geometry) LIKE 'GEOMETRY%'
                                THEN geometry::GEOMETRY
                            ELSE ST_GeomFromWkb(CAST(geometry AS BLOB))
                        END
                    )                                      AS geometry_json,
                    class                                  AS building_class,
                    height
                FROM read_parquet('{self._overture_s3_path}', hive_partitioning=1)
                WHERE bbox.xmin <= {max_lon}
                  AND bbox.xmax >= {min_lon}
                  AND bbox.ymin <= {max_lat}
                  AND bbox.ymax >= {min_lat}
                LIMIT {self._max_buildings}
            """
            rows = con.execute(query).fetchall()
            con.close()
        except Exception as exc:
            LOGGER.warning(
                "national_footprint_index overture_query_error"
                " bbox=(%.4f,%.4f,%.4f,%.4f) error=%s",
                min_lat, min_lon, max_lat, max_lon,
                exc,
            )
            return []

        features: list[dict[str, Any]] = []
        for row in rows:
            feat = self._row_to_feature(row)
            if feat is not None:
                features.append(feat)

        LOGGER.debug(
            "national_footprint_index overture_query_result"
            " bbox=(%.4f,%.4f,%.4f,%.4f) count=%d",
            min_lat, min_lon, max_lat, max_lon,
            len(features),
        )
        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_footprints_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> list[dict]:
        """Return GeoJSON-style feature dicts for all buildings in the bbox.

        Checks the SQLite cache first.  On a cache miss, queries Overture S3
        and stores the result.  Returns ``[]`` when the index is disabled,
        DuckDB is unavailable, or any query error occurs.

        Parameters
        ----------
        min_lat, min_lon, max_lat, max_lon:
            WGS-84 bounding box corners.

        Returns
        -------
        List of feature dicts, each with ``"geometry"`` (GeoJSON dict) and
        ``"properties"`` (dict with ``area_m2``, ``source``, ``building_class``,
        ``height_m``).
        """
        if not self.enabled:
            return []

        cache_key = self._make_cache_key(min_lat, min_lon, max_lat, max_lon)
        cached = self._cache_get(cache_key)
        if cached is not None:
            LOGGER.debug(
                "national_footprint_index cache_hit cache_key=%r features=%d",
                cache_key,
                len(cached),
            )
            return cached

        features = self._query_overture(min_lat, min_lon, max_lat, max_lon)
        if features:
            self._cache_put(cache_key, features, min_lat, min_lon, max_lat, max_lon)
        return features

    def get_footprints_near_point(
        self,
        lat: float,
        lon: float,
        radius_m: float = 300.0,
    ) -> list[dict]:
        """Return building footprints within ``radius_m`` of ``(lat, lon)``.

        Converts the radius to an approximate lat/lon bounding box, then
        delegates to :meth:`fetch_footprints_for_bbox`.  The bbox corners are
        rounded to 4 decimal places (~11 m) for cache key stability across
        nearby assessments of the same neighbourhood.

        Parameters
        ----------
        lat, lon:
            WGS-84 coordinates of the anchor point.
        radius_m:
            Search radius in metres.  Default: 300 m.

        Returns
        -------
        List of GeoJSON-style feature dicts.
        """
        if not self.enabled:
            return []

        lat_deg = radius_m * _DEG_PER_METER_LAT
        lon_deg = radius_m / max(
            1.0, 111_000.0 * abs(math.cos(math.radians(float(lat))))
        )

        min_lat = round(float(lat) - lat_deg, 4)
        max_lat = round(float(lat) + lat_deg, 4)
        min_lon = round(float(lon) - lon_deg, 4)
        max_lon = round(float(lon) + lon_deg, 4)

        return self.fetch_footprints_for_bbox(min_lat, min_lon, max_lat, max_lon)

    def get_cache_stats(self) -> dict[str, Any]:
        """Return summary statistics about the local footprint cache.

        Returns
        -------
        Dict with keys ``total_cached_bboxes``, ``total_cached_features``,
        ``oldest_entry``.
        """
        if self._db is None:
            return {
                "total_cached_bboxes": 0,
                "total_cached_features": 0,
                "oldest_entry": None,
            }
        try:
            cur = self._db.execute(
                "SELECT COUNT(*), SUM(feature_count), MIN(fetched_at)"
                " FROM footprint_cache"
            )
            row = cur.fetchone()
            return {
                "total_cached_bboxes": int(row[0] or 0),
                "total_cached_features": int(row[1] or 0),
                "oldest_entry": row[2],
            }
        except sqlite3.Error as exc:
            LOGGER.warning("national_footprint_index cache_stats_error %s", exc)
            return {
                "total_cached_bboxes": 0,
                "total_cached_features": 0,
                "oldest_entry": None,
            }
