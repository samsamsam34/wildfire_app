"""
National NLCD wildland distance client.

Fetches a window of NLCD land cover pixels via the MRLC WCS service and
computes the distance from a property point to the nearest wildland-class pixel.

Data source:
    MRLC (Multi-Resolution Land Characteristics Consortium) WCS GeoServer
    Service URL: https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs
    Coverage ID: mrlc_Land-Cover-Native_conus_year_data:Land-Cover-Native_conus_year_data
    Vintage: NLCD 2021 (most recent CONUS release as of April 2025)
    Native resolution: 30 m/pixel, CONUS extent

Wildland classes (NLCD 2021 Land Cover classification):
    41  Deciduous Forest
    42  Evergreen Forest
    43  Mixed Forest
    52  Shrub/Scrub
    71  Grassland/Herbaceous
    (81 Pasture/Hay and 82 Cultivated Crops excluded by default — see wildland_classes)

Authentication: None required (public WCS endpoint).
Response time: ~1–5 s per request for a 1 km² window.
Cache TTL: 365 days. NLCD releases annually at most; cached distances are valid
           for an entire assessment season.

This client is additive fallback only. It is never called when a local prepared-
region raster already yielded a wildland distance. Any failure returns None.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("wildfire_app.national_nlcd_client")

# MRLC WCS endpoint for NLCD annual land cover
_NLCD_WCS_URL = (
    "https://dmsdata.cr.usgs.gov/geoserver/"
    "mrlc_Land-Cover-Native_conus_year_data/wcs"
)
_NLCD_COVERAGE_ID = (
    "mrlc_Land-Cover-Native_conus_year_data:Land-Cover-Native_conus_year_data"
)

# NLCD vintage year to request from the time-enabled coverage
_NLCD_YEAR = "2021"
_NLCD_TIME = f"{_NLCD_YEAR}-01-01T00:00:00Z"

# Wildland NLCD class codes (default — excludes pasture/cropland)
_DEFAULT_WILDLAND_CLASSES: frozenset[int] = frozenset({41, 42, 43, 52, 71})

# Approximate meters per degree at CONUS mid-latitudes
_M_PER_DEG = 111_320.0

# WCS tile dimensions (pixels): 67×67 covers ~2 km at 30 m/pixel
_TILE_PX = 67

# Bbox padding per side in degrees (≈ 1 km at 45° lat)
_BBOX_PAD_DEG = 0.009

# Cache TTL: 365 days (NLCD releases at most once per year)
_CACHE_TTL_DAYS = 365

# Cache key coordinate precision: 3 decimal places ≈ 111 m grid
_CACHE_COORD_PRECISION = 3

# Sentinel stored when no wildland found within search radius
_NO_WILDLAND_SENTINEL = -1.0


class NationalNLCDClient:
    """
    Wildland distance estimator using the MRLC NLCD WCS.

    Fetches a window of NLCD land cover pixels around a property, identifies
    wildland-class pixels (forest, shrub, grassland), and returns the distance
    in meters to the nearest wildland pixel.

    Results are cached in SQLite for 365 days.

    Usage::

        client = NationalNLCDClient()
        dist_m = client.get_wildland_distance_m(40.296, -111.694)
        # → 85.0 (meters to nearest wildland pixel)

    Returns None on any failure. Never raises.
    """

    def __init__(
        self,
        cache_db_path: str = "data/nlcd_cache.db",
        timeout_seconds: int = 20,
        enabled: bool = True,
        wildland_classes: Optional[frozenset[int]] = None,
    ) -> None:
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self._cache_db_path = cache_db_path
        self.wildland_classes = (
            wildland_classes if wildland_classes is not None else _DEFAULT_WILDLAND_CLASSES
        )
        if enabled:
            self._init_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_wildland_distance_m(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float = 1000.0,
    ) -> Optional[float]:
        """
        Return distance in meters from (lat, lon) to the nearest wildland pixel.

        Returns 0.0 if the property point itself is on a wildland pixel.
        Returns sample_radius_m if no wildland found within the search window.
        Returns None on any fetch or processing failure.
        """
        if not self.enabled:
            return None

        cache_key = self._cache_key(lat, lon)
        cached = self._cache_get(cache_key)
        if cached is not None:
            val = cached[0]
            if val is None:
                return None
            return None if val == _NO_WILDLAND_SENTINEL else val

        result = self._fetch_distance(lat, lon, sample_radius_m)
        # Cache None as _NO_WILDLAND_SENTINEL only for "no wildland found" case,
        # not for network failures (where we return None without caching).
        if result is not None:
            self._cache_set(cache_key, lat, lon, result)
        return result

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _fetch_distance(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float,
    ) -> Optional[float]:
        """Fetch NLCD tile and compute wildland distance. Never raises."""
        try:
            return self._fetch_distance_inner(lat, lon, sample_radius_m)
        except Exception as exc:
            LOGGER.warning(
                "national_nlcd_client fetch_error lat=%.4f lon=%.4f error=%s",
                lat, lon, exc,
            )
            return None

    def _fetch_distance_inner(
        self,
        lat: float,
        lon: float,
        sample_radius_m: float,
    ) -> Optional[float]:
        import requests as _requests

        # Bbox: pad by roughly sample_radius_m on each side
        lat_pad = max(_BBOX_PAD_DEG, sample_radius_m / _M_PER_DEG)
        lon_pad = max(_BBOX_PAD_DEG, sample_radius_m / (_M_PER_DEG * max(0.1, math.cos(math.radians(lat)))))
        lat_pad = min(lat_pad, 0.05)
        lon_pad = min(lon_pad, 0.05)

        west = lon - lon_pad
        south = lat - lat_pad
        east = lon + lon_pad
        north = lat + lat_pad

        params = {
            "SERVICE": "WCS",
            "VERSION": "1.0.0",
            "REQUEST": "GetCoverage",
            "COVERAGE": _NLCD_COVERAGE_ID,
            "CRS": "EPSG:4326",
            "BBOX": f"{west},{south},{east},{north}",
            "WIDTH": str(_TILE_PX),
            "HEIGHT": str(_TILE_PX),
            "FORMAT": "GeoTIFF",
            "TIME": _NLCD_TIME,
        }

        # Network errors propagate to _fetch_distance's handler.
        response = _requests.get(_NLCD_WCS_URL, params=params, timeout=self.timeout_seconds)

        if response.status_code != 200:
            LOGGER.warning(
                "national_nlcd_client wcs_http_error status=%s", response.status_code
            )
            return None

        content_type = response.headers.get("Content-Type", "")
        if "xml" in content_type.lower() or "html" in content_type.lower():
            snippet = response.text[:200]
            LOGGER.warning("national_nlcd_client wcs_exception_report snippet=%r", snippet)
            return None

        return self._compute_wildland_distance(response.content, lat, lon, sample_radius_m)

    def _compute_wildland_distance(
        self,
        geotiff_bytes: bytes,
        lat: float,
        lon: float,
        sample_radius_m: float,
    ) -> Optional[float]:
        """Sample GeoTIFF bytes and return distance to nearest wildland pixel."""
        try:
            import numpy as np
            import rasterio.io as rio_io
        except ImportError:
            LOGGER.warning("national_nlcd_client rasterio_unavailable")
            return None

        try:
            with rio_io.MemoryFile(geotiff_bytes) as memfile:
                with memfile.open() as ds:
                    # Ensure point is within tile bounds
                    bounds = ds.bounds
                    if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
                        LOGGER.debug(
                            "national_nlcd_client point_outside_tile lat=%.4f lon=%.4f", lat, lon
                        )
                        return None

                    arr = ds.read(1)
                    nodata = ds.nodata
                    transform = ds.transform
                    height, width = arr.shape

                    # Build coordinate grids for each pixel center (lon, lat)
                    cols = np.arange(width)
                    rows = np.arange(height)
                    # rasterio transform: lon = left + (col+0.5)*pixel_width
                    pixel_lons = transform.c + (cols + 0.5) * transform.a
                    pixel_lats = transform.f + (rows + 0.5) * transform.e
                    lon_grid, lat_grid = np.meshgrid(pixel_lons, pixel_lats)

                    # Mask nodata and non-wildland pixels
                    valid = np.ones(arr.shape, dtype=bool)
                    if nodata is not None:
                        valid &= arr != nodata
                    valid &= arr > 0

                    # Wildland class mask
                    wildland_mask = np.zeros(arr.shape, dtype=bool)
                    for cls in self.wildland_classes:
                        wildland_mask |= arr == cls
                    wildland_mask &= valid

                    # Distance from property point to each wildland pixel center (meters)
                    # Use simple degree-to-meter conversion (accurate to ~0.5% for CONUS)
                    lat_cos = math.cos(math.radians(lat))
                    dlat_m = (lat_grid - lat) * _M_PER_DEG
                    dlon_m = (lon_grid - lon) * _M_PER_DEG * lat_cos
                    dist_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

                    if not np.any(wildland_mask):
                        # No wildland found in window — return sample_radius_m (edge of search)
                        LOGGER.debug(
                            "national_nlcd_client no_wildland_found lat=%.4f lon=%.4f", lat, lon
                        )
                        return float(sample_radius_m)

                    min_dist = float(np.min(dist_m[wildland_mask]))
                    LOGGER.info(
                        "national_nlcd_client wildland_distance lat=%.4f lon=%.4f dist_m=%.1f",
                        lat, lon, min_dist,
                    )
                    return round(min_dist, 1)

        except Exception as exc:
            LOGGER.warning(
                "national_nlcd_client rasterio_error lat=%.4f lon=%.4f error=%s",
                lat, lon, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Cache — identical pattern to LandfireCOGClient
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        try:
            Path(self._cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS nlcd_wildland_cache (
                        cache_key TEXT PRIMARY KEY,
                        lat REAL NOT NULL,
                        lon REAL NOT NULL,
                        wildland_distance_m REAL,
                        fetched_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            LOGGER.warning("national_nlcd_client cache_init_error %s", exc)

    def _connect(self):
        return sqlite3.connect(self._cache_db_path, timeout=10)

    @staticmethod
    def _cache_key(lat: float, lon: float) -> str:
        return (
            f"nlcd_wildland"
            f"_{round(lat, _CACHE_COORD_PRECISION)}"
            f"_{round(lon, _CACHE_COORD_PRECISION)}"
        )

    def _cache_get(self, cache_key: str) -> Optional[tuple]:
        """Return (wildland_distance_m,) if fresh cache entry exists, else None."""
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
        ).isoformat()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT wildland_distance_m FROM nlcd_wildland_cache "
                    "WHERE cache_key = ? AND fetched_at > ?",
                    (cache_key, cutoff),
                ).fetchone()
            return (row[0],) if row is not None else None
        except Exception as exc:
            LOGGER.debug("national_nlcd_client cache_read_error %s", exc)
            return None

    def _cache_set(self, cache_key: str, lat: float, lon: float, distance_m: float) -> None:
        """Persist wildland distance (or sentinel) to cache."""
        now = datetime.now(tz=timezone.utc).isoformat()
        # Store _NO_WILDLAND_SENTINEL when no wildland found (avoids repeat fetches)
        store_val = _NO_WILDLAND_SENTINEL if distance_m is None else distance_m
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO nlcd_wildland_cache
                        (cache_key, lat, lon, wildland_distance_m, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        round(lat, _CACHE_COORD_PRECISION),
                        round(lon, _CACHE_COORD_PRECISION),
                        store_val,
                        now,
                    ),
                )
                conn.commit()
        except Exception as exc:
            LOGGER.debug("national_nlcd_client cache_write_error %s", exc)

    def get_cache_stats(self) -> dict:
        """Return total entries in the wildland distance cache."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM nlcd_wildland_cache"
                ).fetchone()
            return {"wildland_distance": row[0] if row else 0}
        except Exception as exc:
            LOGGER.debug("national_nlcd_client cache_stats_error %s", exc)
            return {}
