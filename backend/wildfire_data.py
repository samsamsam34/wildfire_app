from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List

try:
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Point, shape
except Exception:  # pragma: no cover - graceful runtime fallback when geo deps are missing
    np = None
    rasterio = None
    Transformer = None
    Point = None
    shape = None


@dataclass
class WildfireContext:
    environmental_index: float
    slope_index: float
    aspect_index: float
    fuel_index: float
    canopy_index: float
    wildland_distance_index: float
    historic_fire_index: float
    burn_probability_index: float
    hazard_severity_index: float
    data_sources: List[str]
    assumptions: List[str]


class WildfireDataClient:
    """Layer-backed wildfire context provider.

    Expected dataset environment variables (GeoTIFF/GeoJSON):
    - WF_LAYER_BURN_PROB_TIF
    - WF_LAYER_HAZARD_SEVERITY_TIF
    - WF_LAYER_SLOPE_TIF (optional when DEM is provided)
    - WF_LAYER_ASPECT_TIF (optional when DEM is provided)
    - WF_LAYER_DEM_TIF
    - WF_LAYER_FUEL_TIF
    - WF_LAYER_CANOPY_TIF
    - WF_LAYER_FIRE_PERIMETERS_GEOJSON
    """

    def __init__(self) -> None:
        self.paths = {
            "burn_prob": os.getenv("WF_LAYER_BURN_PROB_TIF", ""),
            "hazard": os.getenv("WF_LAYER_HAZARD_SEVERITY_TIF", ""),
            "slope": os.getenv("WF_LAYER_SLOPE_TIF", ""),
            "aspect": os.getenv("WF_LAYER_ASPECT_TIF", ""),
            "dem": os.getenv("WF_LAYER_DEM_TIF", ""),
            "fuel": os.getenv("WF_LAYER_FUEL_TIF", ""),
            "canopy": os.getenv("WF_LAYER_CANOPY_TIF", ""),
            "perimeters": os.getenv("WF_LAYER_FIRE_PERIMETERS_GEOJSON", ""),
        }

    @staticmethod
    def _to_index(value: float, src_min: float, src_max: float) -> float:
        if src_max <= src_min:
            return 50.0
        v = max(src_min, min(src_max, value))
        return round(100.0 * (v - src_min) / (src_max - src_min), 1)

    @staticmethod
    def _meters_to_lat_deg(meters: float) -> float:
        return meters / 111_320.0

    @staticmethod
    def _meters_to_lon_deg(meters: float, lat: float) -> float:
        denom = 111_320.0 * max(0.1, math.cos(math.radians(lat)))
        return meters / denom

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6_371_000.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.asin(math.sqrt(a))

    @staticmethod
    def _file_exists(path: str) -> bool:
        return bool(path) and Path(path).exists()

    @lru_cache(maxsize=16)
    def _open_raster(self, path: str):
        return rasterio.open(path)

    @lru_cache(maxsize=8)
    def _load_perimeters(self, path: str) -> List[Any]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", []) if isinstance(payload, dict) else []
        geoms = []
        for feat in features:
            geom = feat.get("geometry") if isinstance(feat, dict) else None
            if geom:
                geoms.append(shape(geom))
        return geoms

    def _to_dataset_crs(self, ds, lon: float, lat: float) -> tuple[float, float]:
        if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return lon, lat
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return transformer.transform(lon, lat)

    def _sample_raster_point(self, path: str, lat: float, lon: float) -> float | None:
        if not (rasterio and self._file_exists(path)):
            return None
        ds = self._open_raster(path)
        x, y = self._to_dataset_crs(ds, lon, lat)
        try:
            sample = next(ds.sample([(x, y)]))[0]
            nodata = ds.nodata
            if nodata is not None and float(sample) == float(nodata):
                return None
            return float(sample)
        except Exception:
            return None

    def _sample_circle(self, path: str, lat: float, lon: float, radius_m: float, step_m: float = 30.0) -> List[float]:
        if not (rasterio and self._file_exists(path)):
            return []

        values: List[float] = []
        rings = max(1, int(radius_m / step_m))
        for ring in range(1, rings + 1):
            r = ring * step_m
            points = max(8, int(2 * math.pi * r / step_m))
            for i in range(points):
                theta = 2.0 * math.pi * i / points
                d_lat = self._meters_to_lat_deg(r * math.sin(theta))
                d_lon = self._meters_to_lon_deg(r * math.cos(theta), lat)
                sample = self._sample_raster_point(path, lat + d_lat, lon + d_lon)
                if sample is not None:
                    values.append(sample)
        return values

    def _derive_slope_aspect_from_dem(self, dem_path: str, lat: float, lon: float, cell_m: float = 30.0) -> tuple[float | None, float | None]:
        center = self._sample_raster_point(dem_path, lat, lon)
        if center is None:
            return None, None

        d_lat = self._meters_to_lat_deg(cell_m)
        d_lon = self._meters_to_lon_deg(cell_m, lat)

        n = self._sample_raster_point(dem_path, lat + d_lat, lon)
        s = self._sample_raster_point(dem_path, lat - d_lat, lon)
        e = self._sample_raster_point(dem_path, lat, lon + d_lon)
        w = self._sample_raster_point(dem_path, lat, lon - d_lon)

        if None in {n, s, e, w}:
            return None, None

        dzdx = (e - w) / (2 * cell_m)
        dzdy = (n - s) / (2 * cell_m)
        slope_rad = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy))
        slope_deg = math.degrees(slope_rad)

        aspect_rad = math.atan2(dzdx, -dzdy)
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0

        return slope_deg, aspect_deg

    def _fuel_combustibility_index(self, fuel_values: Iterable[float]) -> float:
        vals = [v for v in fuel_values if v is not None]
        if not vals:
            return 50.0

        # Proxy mapping for Scott/Burgan-style coded rasters.
        # Higher code ranges often indicate shrub/timber-heavy burn behavior.
        weighted = []
        for raw in vals:
            code = int(round(raw))
            if 1 <= code <= 3:
                weighted.append(25.0)
            elif 4 <= code <= 9:
                weighted.append(45.0)
            elif 10 <= code <= 13:
                weighted.append(65.0)
            elif 101 <= code <= 109:
                weighted.append(75.0)
            elif 121 <= code <= 124:
                weighted.append(85.0)
            elif 141 <= code <= 149:
                weighted.append(70.0)
            else:
                weighted.append(50.0)
        return round(sum(weighted) / len(weighted), 1)

    def _wildland_distance_index(self, lat: float, lon: float, fuel_path: str, canopy_path: str) -> float:
        if not (rasterio and (self._file_exists(fuel_path) or self._file_exists(canopy_path))):
            return 50.0

        for radius in (30, 60, 120, 250, 500, 1000, 2000):
            samples = max(8, int(2 * math.pi * radius / 30))
            for i in range(samples):
                theta = 2.0 * math.pi * i / samples
                d_lat = self._meters_to_lat_deg(radius * math.sin(theta))
                d_lon = self._meters_to_lon_deg(radius * math.cos(theta), lat)
                p_lat, p_lon = lat + d_lat, lon + d_lon
                fuel = self._sample_raster_point(fuel_path, p_lat, p_lon) if self._file_exists(fuel_path) else None
                canopy = self._sample_raster_point(canopy_path, p_lat, p_lon) if self._file_exists(canopy_path) else None

                is_wildland = False
                if fuel is not None:
                    fuel_code = int(round(fuel))
                    is_wildland = (1 <= fuel_code <= 13) or (101 <= fuel_code <= 149)
                if canopy is not None and canopy >= 20:
                    is_wildland = True

                if is_wildland:
                    # Closer wildland vegetation = higher risk.
                    return round(max(0.0, min(100.0, 100.0 - (radius / 2000.0) * 100.0)), 1)

        return 0.0

    def _historical_recurrence_index(self, lat: float, lon: float, perimeter_path: str) -> float:
        if not (shape and Point and self._file_exists(perimeter_path)):
            return 40.0

        try:
            geoms = self._load_perimeters(perimeter_path)
        except Exception:
            return 40.0

        if not geoms:
            return 20.0

        pt = Point(lon, lat)
        near_1km = 0
        near_5km = 0

        lat_1 = self._meters_to_lat_deg(1000)
        lon_1 = self._meters_to_lon_deg(1000, lat)
        lat_5 = self._meters_to_lat_deg(5000)
        lon_5 = self._meters_to_lon_deg(5000, lat)

        for g in geoms:
            minx, miny, maxx, maxy = g.bounds
            if not (maxx < lon - lon_5 or minx > lon + lon_5 or maxy < lat - lat_5 or miny > lat + lat_5):
                near_5km += 1
            if not (maxx < lon - lon_1 or minx > lon + lon_1 or maxy < lat - lat_1 or miny > lat + lat_1):
                near_1km += 1

            if g.contains(pt):
                near_1km = max(near_1km, 2)
                near_5km = max(near_5km, 3)

        score = near_1km * 22.0 + near_5km * 6.0
        return round(max(0.0, min(100.0, score)), 1)

    def collect_context(self, lat: float, lon: float) -> WildfireContext:
        assumptions: List[str] = []
        sources: List[str] = []

        if not (rasterio and np is not None and Transformer is not None):
            assumptions.append("Geospatial stack unavailable; install rasterio/numpy/pyproj/shapely.")
            return WildfireContext(
                environmental_index=55.0,
                slope_index=50.0,
                aspect_index=50.0,
                fuel_index=50.0,
                canopy_index=50.0,
                wildland_distance_index=50.0,
                historic_fire_index=40.0,
                burn_probability_index=55.0,
                hazard_severity_index=55.0,
                data_sources=sources,
                assumptions=assumptions,
            )

        burn_prob = self._sample_raster_point(self.paths["burn_prob"], lat, lon)
        if burn_prob is None:
            assumptions.append("Burn probability layer unavailable at property location.")
            burn_probability_index = 55.0
        else:
            burn_probability_index = self._to_index(burn_prob, 0.0, 1.0 if burn_prob <= 1.0 else 100.0)
            sources.append("Burn probability raster")

        hazard = self._sample_raster_point(self.paths["hazard"], lat, lon)
        if hazard is None:
            assumptions.append("Wildfire hazard severity layer unavailable at property location.")
            hazard_severity_index = 55.0
        else:
            hazard_severity_index = self._to_index(hazard, 0.0, 5.0 if hazard <= 5.0 else 100.0)
            sources.append("Wildfire hazard severity raster")

        slope = self._sample_raster_point(self.paths["slope"], lat, lon)
        aspect = self._sample_raster_point(self.paths["aspect"], lat, lon)
        if slope is None or aspect is None:
            dem_path = self.paths["dem"]
            if self._file_exists(dem_path):
                derived_slope, derived_aspect = self._derive_slope_aspect_from_dem(dem_path, lat, lon)
                if slope is None and derived_slope is not None:
                    slope = derived_slope
                if aspect is None and derived_aspect is not None:
                    aspect = derived_aspect
                if derived_slope is not None and derived_aspect is not None:
                    sources.append("DEM-derived slope/aspect")
                else:
                    assumptions.append("Could not derive slope/aspect from DEM at property location.")
            else:
                assumptions.append("Slope/aspect rasters missing and DEM not configured.")

        if slope is None:
            slope_index = 50.0
        else:
            slope_index = self._to_index(slope, 0.0, 45.0)
            if "DEM-derived slope/aspect" not in sources:
                sources.append("Slope raster")

        if aspect is None:
            aspect_index = 50.0
        else:
            # Southerly/westerly exposures are typically drier in many western US regimes.
            aspect = float(aspect) % 360.0
            if 180.0 <= aspect <= 315.0:
                aspect_index = 75.0
            elif 135.0 <= aspect < 180.0 or 315.0 < aspect <= 360.0:
                aspect_index = 60.0
            else:
                aspect_index = 40.0
            if "DEM-derived slope/aspect" not in sources:
                sources.append("Aspect raster")

        fuel_samples = self._sample_circle(self.paths["fuel"], lat, lon, radius_m=100.0)
        if fuel_samples:
            fuel_index = self._fuel_combustibility_index(fuel_samples)
            sources.append("Fuel model raster")
        else:
            fuel_index = 50.0
            assumptions.append("Fuel model unavailable within 100m neighborhood.")

        canopy_samples = self._sample_circle(self.paths["canopy"], lat, lon, radius_m=100.0)
        if canopy_samples:
            canopy_mean = sum(canopy_samples) / len(canopy_samples)
            canopy_index = self._to_index(canopy_mean, 0.0, 100.0)
            sources.append("Canopy density raster")
        else:
            canopy_index = 45.0
            assumptions.append("Canopy density unavailable within 100m neighborhood.")

        wildland_distance_index = self._wildland_distance_index(lat, lon, self.paths["fuel"], self.paths["canopy"])
        if self._file_exists(self.paths["fuel"]) or self._file_exists(self.paths["canopy"]):
            sources.append("Distance to wildland vegetation (derived)")
        else:
            assumptions.append("Fuel/canopy rasters missing; wildland distance defaulted.")

        historic_fire_index = self._historical_recurrence_index(lat, lon, self.paths["perimeters"])
        if self._file_exists(self.paths["perimeters"]):
            sources.append("Historical fire perimeter recurrence")
        else:
            assumptions.append("Historical perimeter layer missing; recurrence defaulted.")

        environmental = round(
            0.22 * burn_probability_index
            + 0.18 * hazard_severity_index
            + 0.14 * slope_index
            + 0.06 * aspect_index
            + 0.12 * fuel_index
            + 0.10 * canopy_index
            + 0.10 * wildland_distance_index
            + 0.08 * historic_fire_index,
            1,
        )

        return WildfireContext(
            environmental_index=environmental,
            slope_index=round(slope_index, 1),
            aspect_index=round(aspect_index, 1),
            fuel_index=round(fuel_index, 1),
            canopy_index=round(canopy_index, 1),
            wildland_distance_index=round(wildland_distance_index, 1),
            historic_fire_index=round(historic_fire_index, 1),
            burn_probability_index=round(burn_probability_index, 1),
            hazard_severity_index=round(hazard_severity_index, 1),
            data_sources=sorted(set(sources)),
            assumptions=assumptions,
        )
