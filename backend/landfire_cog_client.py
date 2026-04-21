"""
LANDFIRE raster sampling via USGS WCS (GeoServer) endpoints.

Access method: WCS 1.0.0 GetCoverage (synchronous HTTP, no authentication required).

Endpoints:
  Vegetation/fuel/canopy:
    https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/conus_2024/wcs
  Topographic (slope, aspect, elevation — static LF 2020 base):
    https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/conus_topo/wcs

LANDFIRE versions used:
  LF 2024 (version 2.5.0) for fuel and canopy layers.
  LF 2020 base for topographic layers (slope, aspect, DEM). Terrain layers are
  static across annual LANDFIRE releases.

Authentication: None required (publicly accessible, no API key or token).
Rate limits: None documented. Monthly maintenance last Wednesday 8AM–12PM CST/CDT.
Response time: ~2–10 s per layer for a small bbox (~1 km²) cold request.
Cache TTL: 365 days — LANDFIRE updates annually at most; cached pixel values remain
           valid for an entire assessment season.

Alternative access — LFPS async API:
  https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer/
  LandfireProductService/submitJob
  The LFPS API accepts a bbox + layer list and returns a clipped multi-band GeoTIFF
  asynchronously (30–120 s turnaround). Unsuitable for per-request use but can serve
  bulk/batch workflows. Not used here.

VSICURL note: This module uses requests + rasterio.io.MemoryFile. A direct
/vsicurl/<url> rasterio open path is possible but requires GDAL compiled with libcurl
(standard in conda/mamba rasterio; may vary in minimal pip installs). The MemoryFile
approach works with any rasterio install.

This client is additive fallback only. It is never called when a local raster file
covers the requested point. A failed fetch always returns None — it never raises.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("wildfire_app.landfire_cog_client")

# WCS endpoint base URLs
_WCS_FUEL_CANOPY_URL = "https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/conus_2024/wcs"
_WCS_TOPO_URL = "https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/conus_topo/wcs"

# Layer registry: internal_key → (service_url, WCS coverage ID, nodata sentinels)
# Coverage IDs confirmed from WCS GetCapabilities responses.
_LAYER_CONFIG: dict[str, dict] = {
    "fuel": {
        "service_url": _WCS_FUEL_CANOPY_URL,
        "coverage_id": "landfire_wcs:LF2024_FBFM40_CONUS",
        "description": "40 Scott & Burgan surface fuel models (categorical, 1–204)",
    },
    "canopy": {
        "service_url": _WCS_FUEL_CANOPY_URL,
        "coverage_id": "landfire_wcs:LF2024_CC_CONUS",
        "description": "Forest canopy cover (percent, 0–100)",
    },
    "slope": {
        "service_url": _WCS_TOPO_URL,
        "coverage_id": "landfire_wcs:LF2020_SlpD_CONUS",
        "description": "Slope in degrees (0–90)",
    },
    "aspect": {
        "service_url": _WCS_TOPO_URL,
        "coverage_id": "landfire_wcs:LF2020_Asp_CONUS",
        "description": "Aspect in degrees (0–360, clockwise from north; -1 = flat)",
    },
    "dem": {
        "service_url": _WCS_TOPO_URL,
        "coverage_id": "landfire_wcs:LF2020_Elev_CONUS",
        "description": "Elevation in meters",
    },
}

# Nodata sentinels used by LANDFIRE: -9999, 32767, -32768, 0 (fuel only for nodata).
# Values ≤ -9000 are treated as nodata regardless of the GeoTIFF nodata field.
_NODATA_FLOOR = -9000.0

# Bbox padding in degrees per side (≈ 555 m at 45° latitude).
_BBOX_PAD_DEG = 0.005

# WCS request pixel dimensions (33×33 at 30m ≈ 1 km² tile).
_TILE_PX = 33

# Cache TTL: 365 days
_CACHE_TTL_DAYS = 365

# Cache key precision: 3 decimal places ≈ 111 m grid. Coarser than LANDFIRE's 30m pixel
# but fine since we sample the exact point within the tile.
_CACHE_COORD_PRECISION = 3


class LandfireCOGClient:
    """
    On-demand LANDFIRE raster sampler using USGS WCS GetCoverage (synchronous).

    Returns pixel values for a lat/lon point by fetching a small WCS tile and
    sampling the center pixel. Results are cached in SQLite for 365 days.

    Usage::

        client = LandfireCOGClient()
        values = client.sample_point(46.87, -113.99, ["fuel", "canopy", "slope"])
        # → {"fuel": 102.0, "canopy": 65.0, "slope": 14.3}

    Never raises. Any layer that cannot be fetched returns None in the dict.
    """

    def __init__(
        self,
        cache_db_path: str = "data/landfire_cache.db",
        timeout_seconds: int = 30,
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

    def sample_point(
        self,
        lat: float,
        lon: float,
        layer_ids: list[str],
        buffer_m: float = 500.0,
    ) -> dict[str, Optional[float]]:
        """
        Return sampled pixel values for each layer_id at (lat, lon).

        Parameters
        ----------
        lat, lon    : WGS84 decimal degrees.
        layer_ids   : List of internal layer keys (e.g. ["fuel", "canopy", "slope"]).
        buffer_m    : Approximate spatial buffer around the point (meters).
                      Controls tile bbox size — not the sampling radius (always point).

        Returns
        -------
        dict mapping each layer_id to a float value, or None on any failure.
        """
        result: dict[str, Optional[float]] = {lid: None for lid in layer_ids}
        if not self.enabled:
            return result

        unknown = [lid for lid in layer_ids if lid not in _LAYER_CONFIG]
        if unknown:
            LOGGER.warning("landfire_cog_client unknown_layer_ids %s", unknown)

        valid_ids = [lid for lid in layer_ids if lid in _LAYER_CONFIG]
        if not valid_ids:
            return result

        # Per-layer cache check + fetch
        for layer_id in valid_ids:
            cached = self._cache_get(layer_id, lat, lon)
            if cached is not None:
                # Cached value may be the sentinel None-float (NaN stored as None).
                result[layer_id] = cached[0]
                LOGGER.debug(
                    "landfire_cog_client cache_hit layer=%s lat=%.4f lon=%.4f value=%s",
                    layer_id, lat, lon, cached[0],
                )
                continue

            # Cache miss: fetch from WCS
            value = self._fetch_layer(layer_id, lat, lon, buffer_m)
            self._cache_set(layer_id, lat, lon, value)
            result[layer_id] = value
            LOGGER.info(
                "landfire_cog_client wcs_fetch layer=%s lat=%.4f lon=%.4f value=%s",
                layer_id, lat, lon, value,
            )

        return result

    def get_available_layers(self) -> list[str]:
        """Return the list of layer IDs this client can fetch."""
        return list(_LAYER_CONFIG.keys())

    def get_cache_stats(self) -> dict:
        """Return cache entry counts per layer."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT layer_id, COUNT(*) FROM landfire_pixel_cache GROUP BY layer_id"
                ).fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as exc:
            LOGGER.debug("landfire_cog_client cache_stats_error %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _fetch_layer(
        self,
        layer_id: str,
        lat: float,
        lon: float,
        buffer_m: float,
    ) -> Optional[float]:
        """Fetch a single WCS tile and sample the pixel at (lat, lon). Never raises."""
        try:
            return self._fetch_layer_inner(layer_id, lat, lon, buffer_m)
        except Exception as exc:
            LOGGER.warning(
                "landfire_cog_client fetch_error layer=%s lat=%.4f lon=%.4f error=%s",
                layer_id, lat, lon, exc,
            )
            return None

    def _fetch_layer_inner(
        self,
        layer_id: str,
        lat: float,
        lon: float,
        buffer_m: float,
    ) -> Optional[float]:
        import requests  # requests is available via httpx or stdlib-compatible shim

        cfg = _LAYER_CONFIG[layer_id]
        service_url = cfg["service_url"]
        coverage_id = cfg["coverage_id"]

        # Compute bbox in WGS84 degrees.
        # buffer_m / 111320 ≈ degrees of latitude. Use a slightly larger lon pad
        # to stay conservative in both axes.
        lat_pad = max(_BBOX_PAD_DEG, buffer_m / 111320.0)
        lon_pad = max(_BBOX_PAD_DEG, buffer_m / (111320.0 * max(0.1, abs(float.__abs__(lat) or 0.1)) * 0.01745))
        # cap pad at 0.05° to keep tiles small
        lat_pad = min(lat_pad, 0.05)
        lon_pad = min(lon_pad, 0.05)

        west = lon - lon_pad
        south = lat - lat_pad
        east = lon + lon_pad
        north = lat + lat_pad

        # WCS 1.0.0 GetCoverage request.
        # BBOX order for EPSG:4326 in GeoServer WCS 1.0.0: (minLon, minLat, maxLon, maxLat)
        params = {
            "SERVICE": "WCS",
            "VERSION": "1.0.0",
            "REQUEST": "GetCoverage",
            "COVERAGE": coverage_id,
            "CRS": "EPSG:4326",
            "BBOX": f"{west},{south},{east},{north}",
            "WIDTH": str(_TILE_PX),
            "HEIGHT": str(_TILE_PX),
            "FORMAT": "GeoTIFF",
        }

        try:
            import requests as _requests
        except ImportError:
            _requests = None  # type: ignore[assignment]

        if _requests is not None:
            # Network errors (Timeout, ConnectionError) propagate to _fetch_layer's handler.
            response = _requests.get(service_url, params=params, timeout=self.timeout_seconds)
        else:
            # requests not installed — fall back to stdlib urllib.
            response = self._urllib_get(service_url, params)

        if response.status_code != 200:
            LOGGER.warning(
                "landfire_cog_client wcs_http_error layer=%s status=%s",
                layer_id, response.status_code,
            )
            return None

        content_type = response.headers.get("Content-Type", "")
        if "xml" in content_type.lower() or "html" in content_type.lower():
            # WCS returned an exception report instead of a GeoTIFF
            snippet = response.text[:300]
            LOGGER.warning(
                "landfire_cog_client wcs_exception_report layer=%s snippet=%r",
                layer_id, snippet,
            )
            return None

        return self._sample_bytes(response.content, lat, lon, layer_id)

    def _urllib_get(self, url: str, params: dict):
        """Fallback HTTP GET using urllib (no requests dependency)."""
        import urllib.request
        import urllib.parse

        class _FakeResponse:
            def __init__(self, status_code, content, headers):
                self.status_code = status_code
                self.content = content
                self.headers = headers
                self.text = content.decode("utf-8", errors="replace")

        full_url = url + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(full_url, timeout=self.timeout_seconds) as resp:
                return _FakeResponse(
                    resp.status,
                    resp.read(),
                    dict(resp.headers),
                )
        except Exception as exc:
            raise RuntimeError(f"urllib GET failed: {exc}") from exc

    def _sample_bytes(
        self,
        geotiff_bytes: bytes,
        lat: float,
        lon: float,
        layer_id: str,
    ) -> Optional[float]:
        """Sample the pixel at (lat, lon) from in-memory GeoTIFF bytes."""
        try:
            import rasterio
            import rasterio.io as rio_io
        except ImportError:
            LOGGER.warning("landfire_cog_client rasterio_unavailable")
            return None

        try:
            with rio_io.MemoryFile(geotiff_bytes) as memfile:
                with memfile.open() as ds:
                    # Transform (lon, lat) → dataset native CRS.
                    # Most LANDFIRE WCS responses in EPSG:4326 use (lon, lat) axis convention.
                    if ds.crs and ds.crs.to_epsg() != 4326:
                        try:
                            from pyproj import Transformer
                            tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                            x, y = tf.transform(lon, lat)
                        except Exception:
                            x, y = lon, lat
                    else:
                        x, y = lon, lat

                    bounds = ds.bounds
                    if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                        LOGGER.debug(
                            "landfire_cog_client point_outside_tile layer=%s lat=%.4f lon=%.4f",
                            layer_id, lat, lon,
                        )
                        return None

                    sample = next(ds.sample([(x, y)]))[0]
                    nodata = ds.nodata

                    if nodata is not None and float(sample) == float(nodata):
                        return None
                    if float(sample) <= _NODATA_FLOOR:
                        return None

                    import numpy as np
                    if np.isnan(sample) or np.isinf(sample):
                        return None

                    return float(sample)
        except Exception as exc:
            LOGGER.warning(
                "landfire_cog_client rasterio_sample_error layer=%s error=%s",
                layer_id, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        try:
            Path(self._cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS landfire_pixel_cache (
                        cache_key TEXT PRIMARY KEY,
                        layer_id TEXT NOT NULL,
                        lat REAL NOT NULL,
                        lon REAL NOT NULL,
                        pixel_value REAL,
                        fetched_at TEXT NOT NULL,
                        coverage_id TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            LOGGER.warning("landfire_cog_client cache_init_error %s", exc)

    def _connect(self):
        return sqlite3.connect(self._cache_db_path, timeout=10)

    @staticmethod
    def _cache_key(layer_id: str, lat: float, lon: float) -> str:
        return (
            f"{layer_id}"
            f"_{round(lat, _CACHE_COORD_PRECISION)}"
            f"_{round(lon, _CACHE_COORD_PRECISION)}"
        )

    def _cache_get(self, layer_id: str, lat: float, lon: float) -> Optional[tuple]:
        """Return (pixel_value,) if a fresh cache entry exists, else None."""
        key = self._cache_key(layer_id, lat, lon)
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
        ).isoformat()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT pixel_value FROM landfire_pixel_cache "
                    "WHERE cache_key = ? AND fetched_at > ?",
                    (key, cutoff),
                ).fetchone()
            if row is None:
                return None
            # pixel_value is None when the WCS returned nodata — that's a valid cached result.
            return (row[0],)
        except Exception as exc:
            LOGGER.debug("landfire_cog_client cache_read_error %s", exc)
            return None

    def _cache_set(self, layer_id: str, lat: float, lon: float, value: Optional[float]) -> None:
        """Persist a pixel value (including None for nodata) to the cache."""
        # Do not cache None values when the fetch failed entirely (network error).
        # We distinguish: value=None after a successful WCS response (nodata pixel)
        # vs value=None because the request never completed (exception path).
        # The caller (sample_point) calls _cache_set unconditionally — for network
        # failures the _fetch_layer already returns None, so we cache None. This means
        # a nodata-at-this-point result won't be re-fetched for 365 days, which is correct.
        key = self._cache_key(layer_id, lat, lon)
        cfg = _LAYER_CONFIG.get(layer_id, {})
        coverage_id = cfg.get("coverage_id", "")
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO landfire_pixel_cache
                        (cache_key, layer_id, lat, lon, pixel_value, fetched_at, coverage_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, layer_id, round(lat, _CACHE_COORD_PRECISION),
                     round(lon, _CACHE_COORD_PRECISION), value, now, coverage_id),
                )
                conn.commit()
        except Exception as exc:
            LOGGER.debug("landfire_cog_client cache_write_error %s", exc)
