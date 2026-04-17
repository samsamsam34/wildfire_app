"""On-demand Regrid parcel API client for the wildfire risk platform.

Fetches a parcel polygon and attributes for any US lat/lon from the Regrid
Terrain API v1.  Results are cached in a local SQLite database to avoid
repeated network calls for the same property and to stay within Regrid's
rate limits (free tier: 1,000 queries/month).

Cache strategy
--------------
Cache key: ``f"{round(lat, 5)}_{round(lon, 5)}"`` — ~1 m coordinate
precision prevents near-duplicate fetches for the same property.
TTL: 90 days from ``fetched_at``.  Parcel boundaries change infrequently;
90 days is a conservative balance between data freshness and API cost.

HTTP transport
--------------
Uses only Python stdlib ``urllib`` for HTTP, matching the existing pattern
in ``geocoding.py`` and ``geocoding_census.py``.  No ``requests`` import.

Dependencies
------------
stdlib only (``json``, ``sqlite3``, ``urllib``, ``datetime``, ``logging``,
``dataclasses``) plus ``shapely`` (already in requirements.txt) for
downstream geometry consumption.  This module itself never imports shapely —
geometry is returned as a raw ``dict`` so the caller decides what to do with
it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("wildfire_app.parcel_api_client")

_REGRID_BASE_URL = "https://app.regrid.com/api/v1/parcel/latlon"
_CACHE_TTL_DAYS = 90
_USER_AGENT = "WildfireRiskAdvisor/0.1"

# Regrid "fields" keys we extract.  The API may return more; we ignore extras.
_PARCEL_ID_KEYS = ("parcelnumb", "parcelnumb_no_formatting")
_ADDRESS_KEYS = ("address",)
_OWNER_KEYS = ("owner",)
_USE_KEYS = ("usedesc", "usecode")
_AREA_KEYS = ("ll_gisacre", "gisacre")
_STATE_KEYS = ("state_abbr",)
_COUNTY_KEYS = ("county",)
_ACRES_TO_M2 = 4046.8564


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RegridParcelResult:
    """A parcel record returned by the Regrid API (or served from cache).

    The ``geometry`` field holds a raw GeoJSON geometry dict (Polygon or
    MultiPolygon in WGS84) ready for consumption by ``shapely.geometry.shape``
    or for direct serialisation.

    Attributes
    ----------
    parcel_id:
        Assessor parcel number (APN) or equivalent identifier.
    parcel_address:
        Situs / site address from parcel record.
    owner_name:
        Current owner of record (may be absent from free-tier responses).
    land_use_desc:
        Human-readable land use description or code.
    area_m2:
        Parcel area in square metres, derived from GIS acreage field.
    state:
        Two-letter US state abbreviation.
    county:
        County name.
    geometry:
        Raw GeoJSON geometry dict (``{"type": "Polygon", "coordinates": …}``).
    source:
        Always ``"regrid_api"``.
    cached:
        ``True`` when the result was read from the local SQLite cache.
    fetched_at:
        ISO-8601 UTC timestamp of the original API fetch.
    """

    parcel_id: Optional[str]
    parcel_address: Optional[str]
    owner_name: Optional[str]
    land_use_desc: Optional[str]
    area_m2: Optional[float]
    state: Optional[str]
    county: Optional[str]
    geometry: dict[str, Any]
    source: str
    cached: bool
    fetched_at: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RegridParcelClient:
    """On-demand parcel lookup via the Regrid Terrain API with SQLite caching.

    Parameters
    ----------
    api_key:
        Regrid API token (set ``WF_REGRID_API_KEY`` in the environment).
    cache_db_path:
        Path to the SQLite cache database.  Created on first use.
        Default: ``data/parcel_cache.db``.
    timeout_seconds:
        HTTP request timeout in seconds.  Default: 8.
    enabled:
        Set to ``False`` to disable all API calls (cache reads still work).
        Useful for testing or when the key is known-invalid.
    """

    def __init__(
        self,
        api_key: str,
        cache_db_path: str = "data/parcel_cache.db",
        timeout_seconds: int = 8,
        enabled: bool = True,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._cache_db_path = Path(str(cache_db_path or "data/parcel_cache.db"))
        self._timeout = int(timeout_seconds)
        self.enabled = bool(enabled)
        self._db: sqlite3.Connection | None = None
        self._init_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_parcel(self, lat: float, lon: float) -> Optional[RegridParcelResult]:
        """Return the parcel at ``(lat, lon)``, using cache when fresh.

        Parameters
        ----------
        lat, lon:
            WGS-84 coordinates of the property anchor point.

        Returns
        -------
        :class:`RegridParcelResult` on success, ``None`` when no parcel was
        found or any error occurred.  Never raises.
        """
        cache_key = self._make_cache_key(lat, lon)

        cached = self._cache_get(cache_key)
        if cached is not None:
            LOGGER.debug(
                "parcel_api_client cache_hit cache_key=%r lat=%.5f lon=%.5f",
                cache_key, lat, lon,
            )
            return cached

        if not self.enabled:
            LOGGER.debug(
                "parcel_api_client disabled; skipping API call lat=%.5f lon=%.5f",
                lat, lon,
            )
            return None

        if not self._api_key:
            LOGGER.warning("parcel_api_client no api_key configured; skipping fetch")
            return None

        raw_json = self._call_api(lat, lon)
        if raw_json is None:
            return None

        result = self._parse_response(raw_json, cached=False)
        if result is None:
            return None

        self._cache_put(cache_key, raw_json, lat, lon)
        return result

    def get_cache_stats(self) -> dict[str, Any]:
        """Return summary statistics about the local parcel cache.

        Returns
        -------
        dict with keys ``total_cached``, ``oldest_entry``, ``newest_entry``.
        """
        if self._db is None:
            return {"total_cached": 0, "oldest_entry": None, "newest_entry": None}
        try:
            cur = self._db.execute(
                "SELECT COUNT(*), MIN(fetched_at), MAX(fetched_at) FROM parcel_cache"
            )
            row = cur.fetchone()
            return {
                "total_cached": int(row[0] or 0),
                "oldest_entry": row[1],
                "newest_entry": row[2],
            }
        except sqlite3.Error as exc:
            LOGGER.warning("parcel_api_client cache_stats_error %s", exc)
            return {"total_cached": 0, "oldest_entry": None, "newest_entry": None}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        """Create the SQLite cache database and table if needed."""
        try:
            self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._cache_db_path), check_same_thread=False)
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS parcel_cache (
                    cache_key   TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    fetched_at  TEXT NOT NULL,
                    lat         REAL NOT NULL,
                    lon         REAL NOT NULL
                )
                """
            )
            self._db.commit()
        except (sqlite3.Error, OSError) as exc:
            LOGGER.warning(
                "parcel_api_client cache_init_error path=%s error=%s",
                self._cache_db_path,
                exc,
            )
            self._db = None

    @staticmethod
    def _make_cache_key(lat: float, lon: float) -> str:
        return f"{round(float(lat), 5)}_{round(float(lon), 5)}"

    def _cache_get(self, cache_key: str) -> Optional[RegridParcelResult]:
        """Return a cached result if it exists and is within the TTL window."""
        if self._db is None:
            return None
        try:
            cur = self._db.execute(
                "SELECT response_json, fetched_at FROM parcel_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
        except sqlite3.Error as exc:
            LOGGER.warning("parcel_api_client cache_read_error %s", exc)
            return None

        if row is None:
            return None

        response_json_str, fetched_at_str = row
        try:
            fetched_at = datetime.fromisoformat(fetched_at_str)
        except (ValueError, TypeError):
            return None

        # Ensure tz-aware comparison.
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - fetched_at).days
        if age_days > _CACHE_TTL_DAYS:
            LOGGER.debug("parcel_api_client cache_stale cache_key=%r age_days=%d", cache_key, age_days)
            return None

        try:
            raw_json = json.loads(response_json_str)
        except json.JSONDecodeError:
            return None

        result = self._parse_response(raw_json, cached=True)
        if result is not None:
            # Restore the original fetch timestamp.
            object.__setattr__(result, "fetched_at", fetched_at_str) if hasattr(result, "__setattr__") else None
            result = RegridParcelResult(
                parcel_id=result.parcel_id,
                parcel_address=result.parcel_address,
                owner_name=result.owner_name,
                land_use_desc=result.land_use_desc,
                area_m2=result.area_m2,
                state=result.state,
                county=result.county,
                geometry=result.geometry,
                source=result.source,
                cached=True,
                fetched_at=fetched_at_str,
            )
        return result

    def _cache_put(self, cache_key: str, raw_json: dict[str, Any], lat: float, lon: float) -> None:
        """Insert or replace a cache entry."""
        if self._db is None:
            return
        fetched_at = datetime.now(tz=timezone.utc).isoformat()
        try:
            self._db.execute(
                """
                INSERT OR REPLACE INTO parcel_cache
                    (cache_key, response_json, fetched_at, lat, lon)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, json.dumps(raw_json), fetched_at, float(lat), float(lon)),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            LOGGER.warning("parcel_api_client cache_write_error %s", exc)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _call_api(self, lat: float, lon: float) -> Optional[dict[str, Any]]:
        """Call the Regrid API and return the parsed JSON, or None on failure."""
        params = urllib.parse.urlencode(
            {
                "lat": lat,
                "lon": lon,
                "token": self._api_key,
                "return_geometry": "true",
                "return_field_labels": "false",
            }
        )
        url = f"{_REGRID_BASE_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                LOGGER.error(
                    "parcel_api_client regrid_auth_error status=%d "
                    "lat=%.5f lon=%.5f — API key invalid or unauthorized",
                    exc.code, lat, lon,
                )
            elif exc.code == 429:
                LOGGER.warning(
                    "parcel_api_client regrid_rate_limit_hit lat=%.5f lon=%.5f",
                    lat, lon,
                )
            elif exc.code == 404:
                LOGGER.debug(
                    "parcel_api_client no_parcel_found lat=%.5f lon=%.5f",
                    lat, lon,
                )
            else:
                LOGGER.warning(
                    "parcel_api_client http_error status=%d lat=%.5f lon=%.5f",
                    exc.code, lat, lon,
                )
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            LOGGER.warning(
                "parcel_api_client network_error lat=%.5f lon=%.5f error=%s",
                lat, lon, exc,
            )
            return None
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "parcel_api_client json_parse_error lat=%.5f lon=%.5f error=%s",
                lat, lon, exc,
            )
            return None

        return payload

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(
        payload: dict[str, Any],
        *,
        cached: bool,
    ) -> Optional[RegridParcelResult]:
        """Parse a Regrid GeoJSON FeatureCollection into a :class:`RegridParcelResult`.

        The Regrid response shape::

            {
              "type": "FeatureCollection",
              "features": [
                {
                  "type": "Feature",
                  "geometry": { "type": "Polygon", "coordinates": [...] },
                  "properties": {
                    "headline": { ... },
                    "fields": {
                      "parcelnumb": "XX-XXXXX-XXX",
                      "address": "123 MAIN ST",
                      "owner": "SMITH JOHN",
                      "usedesc": "RESIDENTIAL",
                      "ll_gisacre": 0.42,
                      "state_abbr": "UT",
                      "county": "Utah County",
                      ...
                    }
                  }
                }
              ]
            }
        """
        if not isinstance(payload, dict):
            return None

        features = payload.get("features")
        if not isinstance(features, list) or len(features) == 0:
            LOGGER.debug("parcel_api_client empty_features_array")
            return None

        feature = features[0]
        if not isinstance(feature, dict):
            return None

        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in (
            "Polygon", "MultiPolygon"
        ):
            LOGGER.debug(
                "parcel_api_client invalid_geometry type=%s",
                (geometry or {}).get("type"),
            )
            return None

        props = feature.get("properties") or {}
        fields: dict[str, Any] = {}
        # Regrid wraps attributes inside a "fields" sub-dict.
        if isinstance(props.get("fields"), dict):
            fields = props["fields"]
        else:
            # Fallback: some responses flatten fields into properties directly.
            fields = props

        def _first(*keys: str) -> Optional[str]:
            for key in keys:
                val = fields.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
            return None

        parcel_id = _first(*_PARCEL_ID_KEYS)
        parcel_address = _first(*_ADDRESS_KEYS)
        owner_name = _first(*_OWNER_KEYS)
        land_use_desc = _first(*_USE_KEYS)
        state = _first(*_STATE_KEYS)
        county = _first(*_COUNTY_KEYS)

        area_m2: Optional[float] = None
        for key in _AREA_KEYS:
            raw_acres = fields.get(key)
            if raw_acres is not None:
                try:
                    area_m2 = float(raw_acres) * _ACRES_TO_M2
                    break
                except (TypeError, ValueError):
                    continue

        fetched_at = datetime.now(tz=timezone.utc).isoformat()

        LOGGER.debug(
            "parcel_api_client parsed parcel_id=%r state=%r county=%r area_m2=%.0f",
            parcel_id,
            state,
            county,
            area_m2 or 0.0,
        )

        return RegridParcelResult(
            parcel_id=parcel_id,
            parcel_address=parcel_address,
            owner_name=owner_name,
            land_use_desc=land_use_desc,
            area_m2=area_m2,
            state=state,
            county=county,
            geometry=geometry,
            source="regrid_api",
            cached=cached,
            fetched_at=fetched_at,
        )
