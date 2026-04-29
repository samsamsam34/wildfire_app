"""
National slope/aspect from USGS 3DEP elevation via COG range requests.

Elevation source: USGS 3D Elevation Program (3DEP)
  URL: https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/USGS_Seamless_DEM_1.vrt
  Resolution: 1/3 arc-second (~10 m). Covers CONUS, Hawaii, and Alaska.
  Access: HTTP range requests (no authentication required). Confirmed accessible
          via HEAD request (200 OK, Accept-Ranges: bytes, Content-Length: 2,123,229)
          on 2026-04-21.
  Format: VRT mosaic of GeoTIFF tiles. rasterio opens this directly via GDAL's
          VSICURL mechanism (HTTP range requests, no full download required).
  Vertical datum: NAVD88 / NGVD29 (meters above sea level).

Why 3DEP in addition to LANDFIRE WCS slope:
  LANDFIRE WCS already provides pre-computed slope (LF2020_SlpD_CONUS, 30 m)
  as a Phase-4 COG fallback via LandfireCOGClient. 3DEP is an additional
  resilience layer:
    1. Higher resolution (10 m vs 30 m).
    2. Independent data source — available when LANDFIRE WCS is down or
       returning an exception report.
    3. Computes slope/aspect from raw elevation on-the-fly — no pre-computation
       lag and always matches the current DEM.
  Priority order for slope in wildfire_data.py:
    1. Local prepared-region slope raster (fastest, highest local fidelity)
    2. Local prepared-region DEM (derived slope)
    3. LANDFIRE WCS COG (LandfireCOGClient, national, pre-computed 30 m)
    4. USGS 3DEP COG (this module, national, computed 10 m)  ← tertiary fallback

Aspect convention: downslope direction (ArcGIS / ESRI standard).
  0° = slope descends northward (north-facing slope)
  90° = slope descends eastward
  180° = slope descends southward (south-facing slope, elevation rises northward)
  225° = southwest-facing (most fire-dangerous in CONUS risk model)
  This matches the convention used in wildfire_data.py _derive_slope_aspect_from_dem.

Cache: SQLite, 365-day TTL. Key precision: 3 decimal places (≈ 111 m grid).
Error handling: Never raises. Returns (None, None) on any failure.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("wildfire_app.national_elevation_client")

# USGS 3DEP 1/3 arc-second seamless VRT (HTTP range-request accessible).
_3DEP_VRT_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/"
    "USGS_Seamless_DEM_1.vrt"
)

# Cache settings — match LandfireCOGClient
_CACHE_TTL_DAYS = 365
_CACHE_COORD_PRECISION = 3

# Elevation validity sentinels (meters)
_ELEV_MIN_VALID = -500.0   # below Dead Sea; anything lower is nodata / ocean fill
_ELEV_MAX_VALID = 9000.0   # above Everest; anything higher is a sentinel/fill value

# If max - min elevation across the window is below this (m), terrain is flat.
_FLAT_THRESHOLD_M = 1.0

# Minimum sample window padding (degrees). 0.003° ≈ 330 m — ensures enough pixels
# for a stable finite-difference gradient regardless of sample_radius_m.
_MIN_PAD_DEG = 0.003


class NationalElevationClient:
    """
    Compute slope and aspect from USGS 3DEP elevation via HTTP range requests.

    Opens the national 3DEP VRT via rasterio (GDAL VSICURL), reads a small
    elevation window around the query point, and computes slope/aspect using
    numpy finite differences. Results are cached in SQLite for 365 days.

    Usage::

        client = NationalElevationClient()
        slope, aspect = client.get_slope_and_aspect(40.30, -111.70)
        # → (12.4, 225.3)

    Never raises. Returns (None, None) on any failure.
    """

    def __init__(
        self,
        cache_db_path: str = "data/elevation_cache.db",
        timeout_seconds: int = 20,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self._cache_db_path = cache_db_path
        if enabled:
            self._init_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_slope_degrees(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float = 200.0,
    ) -> Optional[float]:
        """Return slope in degrees at (lat, lon), or None on failure."""
        slope, _ = self.get_slope_and_aspect(lat, lon, sample_radius_m)
        return slope

    def get_slope_and_aspect(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float = 200.0,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Return (slope_degrees, aspect_degrees) at (lat, lon).

        slope_degrees: 0–90, in degrees.
        aspect_degrees: 0–360 clockwise from north (downslope convention).
                        None when terrain is flat (max–min elevation < 1 m).

        Returns (None, None) on any failure (network error, out-of-bounds, etc.).
        """
        if not self.enabled:
            return None, None

        key = self._cache_key(lat, lon)
        cached = self._cache_get(key)
        if cached is not None:
            LOGGER.debug(
                "national_elevation_client cache_hit lat=%.4f lon=%.4f slope=%s aspect=%s",
                lat, lon, cached[0], cached[1],
            )
            return cached

        slope, aspect = self._compute_slope_aspect(lat, lon, sample_radius_m)
        self._cache_set(key, lat, lon, slope, aspect)
        LOGGER.info(
            "national_elevation_client 3dep_fetch lat=%.4f lon=%.4f slope=%s aspect=%s",
            lat, lon, slope, aspect,
        )
        return slope, aspect

    def get_cache_stats(self) -> dict:
        """Return entry count in the elevation slope cache."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM elevation_slope_cache"
                ).fetchone()
            return {"elevation_slope": row[0] if row else 0}
        except Exception as exc:
            LOGGER.debug("national_elevation_client cache_stats_error %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Slope / aspect computation
    # ------------------------------------------------------------------

    def _compute_slope_aspect(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """Fetch 3DEP window and compute slope/aspect. Never raises."""
        try:
            return self._compute_slope_aspect_inner(lat, lon, sample_radius_m)
        except Exception as exc:
            LOGGER.warning(
                "national_elevation_client compute_error lat=%.4f lon=%.4f error=%s",
                lat, lon, exc,
            )
            return None, None

    def _compute_slope_aspect_inner(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float,
    ) -> tuple[Optional[float], Optional[float]]:
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds as window_from_bounds

        # Bbox padding: enough degrees to cover sample_radius_m on each side.
        lat_pad = max(_MIN_PAD_DEG, sample_radius_m / 111320.0)
        cos_lat = max(0.01, abs(math.cos(math.radians(lat))))
        lon_pad = max(_MIN_PAD_DEG, sample_radius_m / (111320.0 * cos_lat))
        lat_pad = min(lat_pad, 0.05)
        lon_pad = min(lon_pad, 0.05)

        west = lon - lon_pad
        east = lon + lon_pad
        south = lat - lat_pad
        north = lat + lat_pad

        # Set GDAL HTTP timeout for range requests, then restore original value.
        import os
        _old_timeout = os.environ.get("GDAL_HTTP_TIMEOUT")
        os.environ["GDAL_HTTP_TIMEOUT"] = str(self.timeout_seconds)
        try:
            with rasterio.open(_3DEP_VRT_URL) as ds:
                bounds = ds.bounds
                if not (
                    bounds.left <= lon <= bounds.right
                    and bounds.bottom <= lat <= bounds.top
                ):
                    LOGGER.debug(
                        "national_elevation_client outside_coverage lat=%.4f lon=%.4f",
                        lat, lon,
                    )
                    return None, None

                window = window_from_bounds(west, south, east, north, ds.transform)
                elev = ds.read(1, window=window).astype(np.float64)
                nodata = ds.nodata
                window_transform = ds.window_transform(window)
        finally:
            if _old_timeout is None:
                os.environ.pop("GDAL_HTTP_TIMEOUT", None)
            else:
                os.environ["GDAL_HTTP_TIMEOUT"] = _old_timeout

        if elev.size < 4:
            return None, None

        # Mask nodata / out-of-range sentinels.
        if nodata is not None:
            elev = np.where(elev == float(nodata), np.nan, elev)
        elev = np.where(
            (elev < _ELEV_MIN_VALID) | (elev > _ELEV_MAX_VALID), np.nan, elev
        )

        valid = elev[~np.isnan(elev)]
        if valid.size < 4:
            return None, None

        # Flat terrain: skip gradient computation, aspect is undefined.
        elev_range = float(np.nanmax(elev) - np.nanmin(elev))
        if elev_range < _FLAT_THRESHOLD_M:
            return 0.0, None

        # Fill NaN with window mean for a stable finite-difference computation.
        # The center pixel (the property) should be valid; filled periphery
        # pixels only affect the gradient near edges.
        elev_filled = np.where(np.isnan(elev), float(np.nanmean(elev)), elev)

        # Pixel resolution in meters.
        # window_transform.a = pixel width in degrees (positive, W→E)
        # window_transform.e = pixel height in degrees (negative, N→S)
        dx_m = abs(window_transform.a) * 111320.0 * cos_lat
        dy_m = abs(window_transform.e) * 111320.0

        # np.gradient returns [dz/d_row, dz/d_col].
        # Raster rows run N→S: row 0 is north, last row is south.
        # dz_row_axis: positive = elevation increases going south (row index increases).
        # dz_col_axis: positive = elevation increases going east (col index increases).
        dz_row, dz_col = np.gradient(elev_filled, dy_m, dx_m)

        # Flip row gradient to geographic convention: positive = increases going NORTH.
        dz_north = -dz_row   # dz_dy in geographic space
        dz_east = dz_col     # dz_dx in geographic space

        # Slope in degrees (0–90).
        slope_rad = np.arctan(np.sqrt(dz_east**2 + dz_north**2))
        slope_grid = np.degrees(slope_rad)

        # Aspect in degrees, downslope convention (matching wildfire_data.py
        # _derive_slope_aspect_from_dem):  atan2(dz_east, -dz_north).
        # Result: 0° = slope descends northward (north-facing), 90° = east-facing,
        # 180° = south-facing (elevation rises northward), 225° = SW-facing.
        aspect_rad = np.arctan2(dz_east, -dz_north)
        aspect_grid = (np.degrees(aspect_rad) + 360.0) % 360.0

        # Sample the center pixel (property location).
        cy = elev.shape[0] // 2
        cx = elev.shape[1] // 2

        slope_val = float(slope_grid[cy, cx])
        aspect_raw = float(aspect_grid[cy, cx])

        if not math.isfinite(slope_val):
            return None, None
        slope_val = round(max(0.0, min(90.0, slope_val)), 2)

        # Aspect is meaningless if the center pixel is essentially flat.
        if slope_val < 0.5 or not math.isfinite(aspect_raw):
            aspect_val: Optional[float] = None
        else:
            aspect_val = round(aspect_raw, 1)

        return slope_val, aspect_val

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        try:
            Path(self._cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS elevation_slope_cache (
                        cache_key   TEXT PRIMARY KEY,
                        lat         REAL NOT NULL,
                        lon         REAL NOT NULL,
                        slope_degrees  REAL,
                        aspect_degrees REAL,
                        fetched_at  TEXT NOT NULL,
                        source      TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            LOGGER.warning("national_elevation_client cache_init_error %s", exc)

    def _connect(self):
        return sqlite3.connect(self._cache_db_path, timeout=10)

    @staticmethod
    def _cache_key(lat: float, lon: float) -> str:
        return (
            f"slope_{round(lat, _CACHE_COORD_PRECISION)}"
            f"_{round(lon, _CACHE_COORD_PRECISION)}"
        )

    def _cache_get(self, key: str) -> Optional[tuple[Optional[float], Optional[float]]]:
        """Return (slope, aspect) if a fresh cache entry exists, else None."""
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
        ).isoformat()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT slope_degrees, aspect_degrees FROM elevation_slope_cache "
                    "WHERE cache_key = ? AND fetched_at > ?",
                    (key, cutoff),
                ).fetchone()
            if row is None:
                return None
            return (row[0], row[1])  # both may be None (stored as SQL NULL)
        except Exception as exc:
            LOGGER.debug("national_elevation_client cache_read_error %s", exc)
            return None

    def _cache_set(
        self,
        key: str,
        lat: float,
        lon: float,
        slope: Optional[float],
        aspect: Optional[float],
    ) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO elevation_slope_cache
                        (cache_key, lat, lon, slope_degrees, aspect_degrees,
                         fetched_at, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        round(lat, _CACHE_COORD_PRECISION),
                        round(lon, _CACHE_COORD_PRECISION),
                        slope,
                        aspect,
                        now,
                        "3dep",
                    ),
                )
                conn.commit()
        except Exception as exc:
            LOGGER.debug("national_elevation_client cache_write_error %s", exc)
