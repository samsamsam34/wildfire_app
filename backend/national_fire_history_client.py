"""
National MTBS fire history client.

Loads the MTBS national fire perimeters GeoPackage (data/national/mtbs_perimeters.gpkg)
into memory at startup and exposes a fast spatial query via Shapely STRtree.

Memory footprint: ~100–400 MB depending on whether the dev (3-feature) or full
national (80,000+ feature) GeoPackage is loaded. The full GPKG is expected to
occupy ~200–400 MB of RSS when held as a GeoDataFrame in memory. This is
acceptable for a persistent server process. Do NOT use this client in lambda/
serverless contexts without pre-warming.

Download the full national GPKG with:
    python scripts/download_national_mtbs.py

This client is additive fallback only — it is never called when a local prepared-
region fire perimeter file already produced a result. Any failure returns a
FireHistoryResult with data_available=False; it never raises.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("wildfire_app.national_fire_history_client")

# Fields expected in the MTBS GeoPackage (created by download_national_mtbs.py)
_YEAR_FIELD = "Year"
_NAME_FIELD = "Fire_Name"
_ID_FIELD = "Fire_ID"
_AREA_FIELD = "BurnBndAc"
_HIGH_SEV_FIELD = "high_severity_pct"
_MOD_SEV_FIELD = "mod_severity_pct"
_LOW_SEV_FIELD = "low_severity_pct"
_MONTH_FIELD = "StartMonth"

# Approximate meters-per-degree at mid-latitudes (used for degree↔meter conversions)
_M_PER_DEG_LAT = 111_320.0


@dataclass
class FireHistoryResult:
    """Result of a spatial fire history query."""
    burned_within_radius: bool
    most_recent_fire_year: Optional[int]
    most_recent_fire_severity: Optional[str]   # "low" | "moderate" | "high" | "unknown"
    fire_count_30yr: int
    fire_count_all: int
    nearest_fire_distance_m: Optional[float]
    fires_within_radius: list[dict] = field(default_factory=list)
    data_available: bool = True
    radius_m: float = 5000.0


def _unavailable(radius_m: float = 5000.0) -> FireHistoryResult:
    return FireHistoryResult(
        burned_within_radius=False,
        most_recent_fire_year=None,
        most_recent_fire_severity=None,
        fire_count_30yr=0,
        fire_count_all=0,
        nearest_fire_distance_m=None,
        fires_within_radius=[],
        data_available=False,
        radius_m=radius_m,
    )


def _classify_severity(row_dict: dict[str, Any]) -> str:
    """Map MTBS severity percentage fields to a severity label."""
    try:
        high = float(row_dict.get(_HIGH_SEV_FIELD) or 0)
        mod = float(row_dict.get(_MOD_SEV_FIELD) or 0)
        if high > 50:
            return "high"
        elif mod > 50:
            return "moderate"
        # If any severity field is present but neither threshold met → low
        if any(
            row_dict.get(f) is not None
            for f in (_HIGH_SEV_FIELD, _MOD_SEV_FIELD, _LOW_SEV_FIELD)
        ):
            return "low"
        return "unknown"
    except (TypeError, ValueError):
        return "unknown"


def _dist_deg_to_m(dist_deg: float, lat: float) -> float:
    """Convert a degree-distance (from Shapely WGS84) to approximate meters."""
    # For pure latitudinal distances shapely returns degrees of arc.
    # At mid-latitudes the average of lat and lon degree lengths is used.
    lon_scale = math.cos(math.radians(abs(lat)))
    return dist_deg * _M_PER_DEG_LAT * ((1.0 + lon_scale) / 2.0)


class NationalFireHistoryClient:
    """
    Spatial query client for MTBS national fire perimeters.

    Loads the GeoPackage at data/national/mtbs_perimeters.gpkg into memory and
    builds a Shapely STRtree for fast bounding-box prefilter + exact intersection
    checks at assessment time.

    Usage::

        client = NationalFireHistoryClient()
        result = client.query_fire_history(40.296, -111.694, radius_m=5000.0)
        # → FireHistoryResult(burned_within_radius=True, fire_count_30yr=1, ...)

    Never raises. Returns FireHistoryResult(data_available=False) on any error.
    """

    def __init__(
        self,
        mtbs_gpkg_path: str = "data/national/mtbs_perimeters.gpkg",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._mtbs_gpkg_path = mtbs_gpkg_path
        self._gdf = None
        self._sindex = None

        if not enabled:
            return

        try:
            import geopandas as gpd  # noqa: F401
        except ImportError:
            LOGGER.warning(
                "national_fire_history_client geopandas_unavailable; client disabled"
            )
            self.enabled = False
            return

        gpkg_path = Path(mtbs_gpkg_path)
        if not gpkg_path.exists():
            LOGGER.warning(
                "national_fire_history_client mtbs_gpkg_not_found path=%s; "
                "run scripts/download_national_mtbs.py to download the dataset",
                mtbs_gpkg_path,
            )
            self.enabled = False
            return

        try:
            import geopandas as gpd
            from shapely.strtree import STRtree

            self._gdf = gpd.read_file(str(gpkg_path), layer="fire_perimeters")
            self._sindex = STRtree(self._gdf.geometry.values)
            LOGGER.info(
                "national_fire_history_client loaded path=%s features=%d",
                mtbs_gpkg_path,
                len(self._gdf),
            )
        except Exception as exc:
            LOGGER.warning(
                "national_fire_history_client load_error path=%s error=%s",
                mtbs_gpkg_path,
                exc,
            )
            self._gdf = None
            self._sindex = None
            self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_fire_history(
        self,
        lat: float,
        lon: float,
        radius_m: float = 5000.0,
    ) -> FireHistoryResult:
        """
        Return fire history summary for the given point within radius_m.

        Parameters
        ----------
        lat, lon   : WGS84 decimal degrees.
        radius_m   : Search radius in meters (default 5 km).

        Returns
        -------
        FireHistoryResult — never raises.
        """
        if not self.enabled or self._gdf is None or self._sindex is None:
            return _unavailable(radius_m)

        try:
            return self._query_inner(lat, lon, radius_m)
        except Exception as exc:
            LOGGER.warning(
                "national_fire_history_client query_error lat=%.4f lon=%.4f error=%s",
                lat, lon, exc,
            )
            return _unavailable(radius_m)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _query_inner(
        self,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> FireHistoryResult:
        from shapely.geometry import Point

        pt = Point(lon, lat)

        # Convert radius to approximate degrees for the STRtree prefilter buffer.
        radius_deg = radius_m / _M_PER_DEG_LAT
        buffer = pt.buffer(radius_deg)

        # Stage 1: bbox prefilter via STRtree
        candidate_idxs = self._sindex.query(buffer)

        now_year = datetime.now(tz=timezone.utc).year
        threshold_30yr = now_year - 30

        fires: list[dict] = []
        nearest_dist_m: float | None = None

        for idx in candidate_idxs:
            geom = self._gdf.geometry.iloc[idx]

            # Stage 2: exact intersection check (eliminates false positives from
            # bbox-only match where the perimeter bbox overlaps but the actual polygon
            # doesn't reach within the search radius buffer)
            if not geom.intersects(buffer):
                continue

            row = self._gdf.iloc[idx]
            row_dict = row.to_dict()

            # Distance from property point to nearest edge of this perimeter (degrees)
            try:
                dist_deg = float(pt.distance(geom))
                dist_m = _dist_deg_to_m(dist_deg, lat)
            except Exception:
                dist_m = 0.0  # contained within perimeter

            if nearest_dist_m is None or dist_m < nearest_dist_m:
                nearest_dist_m = dist_m

            year_raw = row_dict.get(_YEAR_FIELD)
            try:
                year = int(year_raw) if year_raw is not None else None
            except (TypeError, ValueError):
                year = None

            fires.append({
                "year": year,
                "name": str(row_dict.get(_NAME_FIELD) or "Unknown"),
                "fire_id": str(row_dict.get(_ID_FIELD) or ""),
                "severity": _classify_severity(row_dict),
                "distance_m": round(dist_m, 1),
                "area_acres": (
                    round(float(row_dict[_AREA_FIELD]), 1)
                    if row_dict.get(_AREA_FIELD) is not None
                    else None
                ),
            })

        fire_count_all = len(fires)
        fire_count_30yr = sum(
            1 for f in fires if f["year"] is not None and f["year"] >= threshold_30yr
        )

        if not fires:
            return FireHistoryResult(
                burned_within_radius=False,
                most_recent_fire_year=None,
                most_recent_fire_severity=None,
                fire_count_30yr=0,
                fire_count_all=0,
                nearest_fire_distance_m=None,
                fires_within_radius=[],
                data_available=True,
                radius_m=radius_m,
            )

        # Most recent fire
        fires_with_year = [f for f in fires if f["year"] is not None]
        if fires_with_year:
            most_recent = max(fires_with_year, key=lambda f: f["year"])
            most_recent_year = most_recent["year"]
            most_recent_severity = most_recent["severity"]
        else:
            most_recent_year = None
            most_recent_severity = None

        # burned_within_radius: True if any perimeter intersects the buffer
        # (distance == 0 means the point is inside the perimeter; positive means edge nearby)
        burned = nearest_dist_m is not None and nearest_dist_m < 1.0

        return FireHistoryResult(
            burned_within_radius=burned,
            most_recent_fire_year=most_recent_year,
            most_recent_fire_severity=most_recent_severity,
            fire_count_30yr=fire_count_30yr,
            fire_count_all=fire_count_all,
            nearest_fire_distance_m=round(nearest_dist_m, 1) if nearest_dist_m is not None else None,
            fires_within_radius=fires,
            data_available=True,
            radius_m=radius_m,
        )
