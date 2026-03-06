from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List


@dataclass
class WildfireContext:
    environmental_index: float
    slope_index: float
    drought_index: float
    historic_fire_index: float
    data_sources: List[str]
    assumptions: List[str]


class WildfireDataClient:
    def _fetch_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "WildfireRiskAdvisor/0.1"})
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))

    def _elevation_meters(self, lat: float, lon: float) -> float | None:
        params = urllib.parse.urlencode({"x": lon, "y": lat, "units": "Meters", "wkid": 4326, "includeDate": "false"})
        url = f"https://epqs.nationalmap.gov/v1/json?{params}"
        payload = self._fetch_json(url)
        value = payload.get("value")
        return float(value) if value is not None else None

    def _recent_precip_mm(self, lat: float, lon: float) -> float | None:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=29)
        params = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "UTC",
            }
        )
        url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
        payload = self._fetch_json(url)
        values = payload.get("daily", {}).get("precipitation_sum", [])
        if not values:
            return None
        return float(sum(v for v in values if v is not None))

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.asin(math.sqrt(a))

    def _historic_fire_index(self, lat: float, lon: float) -> float | None:
        url = "https://eonet.gsfc.nasa.gov/api/v3/events?status=all&category=wildfires&days=365&limit=100"
        payload = self._fetch_json(url)
        events = payload.get("events", [])
        if not events:
            return None

        closest_km = None
        for event in events:
            for geom in event.get("geometry", []):
                coords = geom.get("coordinates")
                if not isinstance(coords, list) or len(coords) < 2:
                    continue
                ev_lon, ev_lat = float(coords[0]), float(coords[1])
                dist = self._haversine_km(lat, lon, ev_lat, ev_lon)
                closest_km = dist if closest_km is None else min(closest_km, dist)

        if closest_km is None:
            return None

        return max(0.0, min(100.0, 100.0 - closest_km))

    def collect_context(self, lat: float, lon: float) -> WildfireContext:
        assumptions: List[str] = []
        sources: List[str] = []

        elevation_m = None
        try:
            elevation_m = self._elevation_meters(lat, lon)
            if elevation_m is not None:
                sources.append("USGS EPQS elevation")
        except Exception:
            assumptions.append("Elevation lookup unavailable; using terrain fallback.")

        recent_precip_mm = None
        try:
            recent_precip_mm = self._recent_precip_mm(lat, lon)
            if recent_precip_mm is not None:
                sources.append("Open-Meteo archive precipitation")
        except Exception:
            assumptions.append("Recent precipitation unavailable; using drought fallback.")

        historic_fire = None
        try:
            historic_fire = self._historic_fire_index(lat, lon)
            if historic_fire is not None:
                sources.append("NASA EONET wildfire events")
        except Exception:
            assumptions.append("Historic wildfire events unavailable; using distance fallback.")

        if elevation_m is None:
            slope_index = 45.0
        else:
            # Elevation is not slope, but acts as a first-pass terrain severity proxy.
            slope_index = max(10.0, min(90.0, 20.0 + elevation_m / 60.0))

        if recent_precip_mm is None:
            drought_index = 55.0
        else:
            drought_index = max(10.0, min(95.0, 90.0 - recent_precip_mm * 0.9))

        if historic_fire is None:
            historic_fire_index = max(10.0, min(90.0, 35.0 + abs(lat + lon) % 40.0))
        else:
            historic_fire_index = historic_fire

        environmental = round(
            0.35 * slope_index + 0.35 * drought_index + 0.30 * historic_fire_index,
            1,
        )

        return WildfireContext(
            environmental_index=environmental,
            slope_index=round(slope_index, 1),
            drought_index=round(drought_index, 1),
            historic_fire_index=round(historic_fire_index, 1),
            data_sources=sources,
            assumptions=assumptions,
        )
