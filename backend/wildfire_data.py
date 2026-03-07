from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.building_footprints import BuildingFootprintClient, compute_structure_rings

try:
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Point, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - graceful runtime fallback when geo deps are missing
    np = None
    rasterio = None
    Transformer = None
    Point = None
    mapping = None
    shape = None
    shapely_transform = None


@dataclass
class WildfireContext:
    environmental_index: Optional[float]
    slope_index: Optional[float]
    aspect_index: Optional[float]
    fuel_index: Optional[float]
    moisture_index: Optional[float]
    canopy_index: Optional[float]
    wildland_distance_index: Optional[float]
    historic_fire_index: Optional[float]
    burn_probability_index: Optional[float]
    hazard_severity_index: Optional[float]
    burn_probability: Optional[float] = None
    wildfire_hazard: Optional[float] = None
    slope: Optional[float] = None
    fuel_model: Optional[float] = None
    canopy_cover: Optional[float] = None
    historic_fire_distance: Optional[float] = None
    wildland_distance: Optional[float] = None
    environmental_layer_status: Dict[str, str] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    structure_ring_metrics: Dict[str, Dict[str, float | None]] = field(default_factory=dict)
    property_level_context: Dict[str, Any] = field(default_factory=dict)


def compute_environmental_data_completeness(context: WildfireContext) -> float:
    status = context.environmental_layer_status or {}
    if not status:
        return 0.0

    tracked_layers = [
        "burn_probability",
        "hazard",
        "slope",
        "fuel",
        "canopy",
        "fire_history",
    ]
    available = sum(1 for layer in tracked_layers if status.get(layer) == "ok")
    return round((available / float(len(tracked_layers))) * 100.0, 1)


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
    - WF_LAYER_MOISTURE_TIF (optional; higher values should represent drier fuels)
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
            "moisture": os.getenv("WF_LAYER_MOISTURE_TIF", ""),
            "perimeters": os.getenv("WF_LAYER_FIRE_PERIMETERS_GEOJSON", ""),
        }
        self.footprints = BuildingFootprintClient()

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

    def _to_geom_crs(self, geom: Any, ds) -> Any:
        if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return geom
        if not (Transformer and shapely_transform):
            return geom
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return shapely_transform(transformer.transform, geom)

    def _polygon_raster_values(self, path: str, polygon: Any) -> list[float]:
        if not (rasterio and np is not None and mapping and self._file_exists(path)):
            return []

        ds = self._open_raster(path)
        try:
            geom_ds = self._to_geom_crs(polygon, ds)
            window = rasterio.features.geometry_window(ds, [mapping(geom_ds)])
            arr = ds.read(1, window=window, masked=True)
            inside_mask = rasterio.features.geometry_mask(
                [mapping(geom_ds)],
                transform=ds.window_transform(window),
                out_shape=arr.shape,
                invert=True,
                all_touched=True,
            )
            vals = arr[inside_mask]
            if hasattr(vals, "compressed"):
                vals = vals.compressed()
            return [float(v) for v in vals.tolist()]
        except Exception:
            return []

    def _summarize_ring_canopy(self, ring_geometry: Any) -> dict[str, float] | None:
        canopy_vals = self._polygon_raster_values(self.paths["canopy"], ring_geometry)
        if not canopy_vals:
            return None

        canopy_mean = float(sum(canopy_vals) / len(canopy_vals))
        canopy_max = float(max(canopy_vals))
        coverage_pct = float(sum(1 for v in canopy_vals if v >= 20.0) / len(canopy_vals) * 100.0)
        vegetation_density = float(self._to_index(canopy_mean, 0.0, 100.0))

        return {
            "canopy_mean": round(canopy_mean, 1),
            "canopy_max": round(canopy_max, 1),
            "coverage_pct": round(coverage_pct, 1),
            "vegetation_density": round(vegetation_density, 1),
        }

    def _summarize_ring_fuel_presence(self, ring_geometry: Any) -> float | None:
        fuel_vals = self._polygon_raster_values(self.paths["fuel"], ring_geometry)
        if not fuel_vals:
            return None

        wildland_cells = 0
        for raw in fuel_vals:
            code = int(round(raw))
            if (1 <= code <= 13) or (101 <= code <= 149):
                wildland_cells += 1

        return round((wildland_cells / len(fuel_vals)) * 100.0, 1)

    def _compute_structure_ring_metrics(
        self,
        lat: float,
        lon: float,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        assumptions: list[str] = []
        sources: list[str] = []

        try:
            result = self.footprints.get_building_footprint(lat, lon)
        except Exception as exc:  # pragma: no cover - defensive guard for malformed sources
            assumptions.append(f"Building footprint lookup failed: {exc}")
            assumptions.append("Building footprint analysis unavailable; using point-based vegetation context.")
            return {
                "footprint_used": False,
                "footprint_found": False,
                "footprint_status": "error",
                "footprint_source": None,
                "footprint_confidence": 0.0,
                "fallback_mode": "point_based",
                "ring_metrics": None,
            }, assumptions, sources

        assumptions.extend(result.assumptions)

        if not result.found or result.footprint is None:
            assumptions.append("Building footprint analysis unavailable; using point-based vegetation context.")
            status = "not_found"
            assumptions_blob = " ".join(result.assumptions).lower()
            if "not configured" in assumptions_blob or "missing" in assumptions_blob:
                status = "provider_unavailable"
            return {
                "footprint_used": False,
                "footprint_found": False,
                "footprint_status": status,
                "footprint_source": result.source,
                "footprint_confidence": result.confidence,
                "fallback_mode": "point_based",
                "ring_metrics": None,
            }, assumptions, sources

        sources.append("Building footprint source")
        rings, ring_assumptions = compute_structure_rings(result.footprint)
        assumptions.extend(ring_assumptions)

        ring_metrics: dict[str, dict[str, float | None]] = {}
        zone_aliases = {
            "ring_0_5_ft": "zone_0_5_ft",
            "ring_5_30_ft": "zone_5_30_ft",
            "ring_30_100_ft": "zone_30_100_ft",
        }
        for ring_key in ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft"):
            ring_geom = rings.get(ring_key)
            if ring_geom is None:
                metrics = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": None,
                    "coverage_pct": None,
                    "fuel_presence_proxy": None,
                }
                ring_metrics[ring_key] = metrics
                ring_metrics[zone_aliases[ring_key]] = dict(metrics)
                continue

            canopy_stats = self._summarize_ring_canopy(ring_geom)
            fuel_presence = self._summarize_ring_fuel_presence(ring_geom)

            if canopy_stats is None:
                metrics = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": fuel_presence,
                    "coverage_pct": fuel_presence,
                    "fuel_presence_proxy": fuel_presence,
                }
            else:
                vegetation_density = canopy_stats["vegetation_density"]
                if fuel_presence is not None:
                    vegetation_density = round((vegetation_density + fuel_presence) / 2.0, 1)
                metrics = {
                    "canopy_mean": canopy_stats["canopy_mean"],
                    "canopy_max": canopy_stats["canopy_max"],
                    "vegetation_density": vegetation_density,
                    "coverage_pct": canopy_stats["coverage_pct"],
                    "fuel_presence_proxy": fuel_presence,
                }
            ring_metrics[ring_key] = metrics
            ring_metrics[zone_aliases[ring_key]] = dict(metrics)

        if ring_metrics:
            sources.append("Structure ring vegetation summaries")

        return {
            "footprint_used": bool(ring_metrics),
            "footprint_found": result.found,
            "footprint_status": "used" if ring_metrics else "error",
            "footprint_source": result.source,
            "footprint_confidence": result.confidence,
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "ring_metrics": ring_metrics,
            "footprint_centroid": {
                "latitude": result.centroid[0] if result.centroid else None,
                "longitude": result.centroid[1] if result.centroid else None,
            },
        }, assumptions, sources

    def _sample_raster_point_raw(self, path: str, lat: float, lon: float) -> float | None:
        if not (rasterio and self._file_exists(path)):
            return None
        ds = self._open_raster(path)
        x, y = self._to_dataset_crs(ds, lon, lat)
        sample = next(ds.sample([(x, y)]))[0]
        nodata = ds.nodata
        if nodata is not None and float(sample) == float(nodata):
            return None
        return float(sample)

    def _sample_raster_point(self, path: str, lat: float, lon: float) -> float | None:
        try:
            return self._sample_raster_point_raw(path, lat, lon)
        except Exception:
            return None

    def _sample_layer_value(self, path: str, lat: float, lon: float) -> Tuple[float | None, str]:
        if not self._file_exists(path):
            return None, "missing"
        try:
            value = self._sample_raster_point_raw(path, lat, lon)
        except Exception:
            return None, "error"
        if value is None:
            return None, "missing"
        return value, "ok"

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

    def _wildland_distance_metrics(
        self,
        lat: float,
        lon: float,
        fuel_path: str,
        canopy_path: str,
    ) -> Tuple[float | None, float | None]:
        if not (rasterio and (self._file_exists(fuel_path) or self._file_exists(canopy_path))):
            return None, None

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
                    index = round(max(0.0, min(100.0, 100.0 - (radius / 2000.0) * 100.0)), 1)
                    return float(radius), index

        # Layer exists and no contiguous wildland found within search radius.
        return 2000.0, 0.0

    def _historical_fire_metrics(self, lat: float, lon: float, perimeter_path: str) -> Tuple[float | None, float | None, str]:
        if not (shape and Point and self._file_exists(perimeter_path)):
            return None, None, "missing"

        try:
            geoms = self._load_perimeters(perimeter_path)
        except Exception:
            return None, None, "error"

        if not geoms:
            return None, 0.0, "ok"

        pt = Point(lon, lat)
        near_1km = 0
        near_5km = 0
        nearest_km: float | None = None

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

            try:
                # Shapely distance here is in degrees for EPSG:4326; convert approximately to km.
                km = float(pt.distance(g)) * 111.32
                nearest_km = km if nearest_km is None else min(nearest_km, km)
            except Exception:
                pass

        score = near_1km * 22.0 + near_5km * 6.0
        return nearest_km, round(max(0.0, min(100.0, score)), 1), "ok"

    def collect_context(self, lat: float, lon: float) -> WildfireContext:
        assumptions: List[str] = []
        sources: List[str] = []
        environmental_layer_status: dict[str, str] = {
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "missing",
            "fuel": "missing",
            "canopy": "missing",
            "fire_history": "missing",
        }
        property_level_context: dict[str, Any] = {
            "footprint_used": False,
            "footprint_found": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": None,
        }
        structure_ring_metrics: dict[str, dict[str, float | None]] = {}

        if not (rasterio and np is not None and Transformer is not None):
            assumptions.append("Geospatial stack unavailable; install rasterio/numpy/pyproj/shapely.")
            return WildfireContext(
                environmental_index=None,
                slope_index=None,
                aspect_index=None,
                fuel_index=None,
                moisture_index=None,
                canopy_index=None,
                wildland_distance_index=None,
                historic_fire_index=None,
                burn_probability_index=None,
                hazard_severity_index=None,
                burn_probability=None,
                wildfire_hazard=None,
                slope=None,
                fuel_model=None,
                canopy_cover=None,
                historic_fire_distance=None,
                wildland_distance=None,
                environmental_layer_status=environmental_layer_status,
                data_sources=sources,
                assumptions=assumptions,
                structure_ring_metrics=structure_ring_metrics,
                property_level_context=property_level_context,
            )

        ring_context, ring_assumptions, ring_sources = self._compute_structure_ring_metrics(lat, lon)
        property_level_context = ring_context
        structure_ring_metrics = ring_context.get("ring_metrics", {}) or {}
        assumptions.extend(ring_assumptions)
        sources.extend(ring_sources)

        burn_prob, burn_status = self._sample_layer_value(self.paths["burn_prob"], lat, lon)
        environmental_layer_status["burn_probability"] = burn_status
        if burn_prob is None:
            assumptions.append("Burn probability layer unavailable at property location.")
            burn_probability_index = None
        else:
            burn_probability_index = self._to_index(burn_prob, 0.0, 1.0 if burn_prob <= 1.0 else 100.0)
            sources.append("Burn probability raster")

        hazard, hazard_status = self._sample_layer_value(self.paths["hazard"], lat, lon)
        environmental_layer_status["hazard"] = hazard_status
        if hazard is None:
            assumptions.append("Wildfire hazard severity layer unavailable at property location.")
            hazard_severity_index = None
        else:
            hazard_severity_index = self._to_index(hazard, 0.0, 5.0 if hazard <= 5.0 else 100.0)
            sources.append("Wildfire hazard severity raster")

        slope, slope_status = self._sample_layer_value(self.paths["slope"], lat, lon)
        aspect, aspect_status = self._sample_layer_value(self.paths["aspect"], lat, lon)
        if slope is None or aspect is None:
            dem_path = self.paths["dem"]
            if self._file_exists(dem_path):
                derived_slope, derived_aspect = self._derive_slope_aspect_from_dem(dem_path, lat, lon)
                if slope is None and derived_slope is not None:
                    slope = derived_slope
                    slope_status = "ok"
                if aspect is None and derived_aspect is not None:
                    aspect = derived_aspect
                    aspect_status = "ok"
                if derived_slope is not None and derived_aspect is not None:
                    sources.append("DEM-derived slope/aspect")
                else:
                    assumptions.append("Could not derive slope/aspect from DEM at property location.")
            else:
                assumptions.append("Slope/aspect rasters missing and DEM not configured.")

        environmental_layer_status["slope"] = slope_status
        slope_index = None if slope is None else self._to_index(slope, 0.0, 45.0)
        if slope is not None and "DEM-derived slope/aspect" not in sources:
            sources.append("Slope raster")

        if aspect is None:
            aspect_index = None
        else:
            a = float(aspect) % 360.0
            if 180.0 <= a <= 315.0:
                aspect_index = 75.0
            elif 135.0 <= a < 180.0 or 315.0 < a <= 360.0:
                aspect_index = 60.0
            else:
                aspect_index = 40.0
            if "DEM-derived slope/aspect" not in sources:
                sources.append("Aspect raster")

        fuel_path = self.paths["fuel"]
        fuel_samples = self._sample_circle(fuel_path, lat, lon, radius_m=100.0) if self._file_exists(fuel_path) else []
        if fuel_samples:
            fuel_index = self._fuel_combustibility_index(fuel_samples)
            fuel_model = round(sum(fuel_samples) / len(fuel_samples), 2)
            environmental_layer_status["fuel"] = "ok"
            sources.append("Fuel model raster")
        else:
            fuel_index = None
            fuel_model = None
            environmental_layer_status["fuel"] = "missing" if not self._file_exists(fuel_path) else "error"
            assumptions.append("Fuel model unavailable within 100m neighborhood.")

        canopy_path = self.paths["canopy"]
        canopy_samples = self._sample_circle(canopy_path, lat, lon, radius_m=100.0) if self._file_exists(canopy_path) else []
        if canopy_samples:
            canopy_mean = sum(canopy_samples) / len(canopy_samples)
            canopy_index = self._to_index(canopy_mean, 0.0, 100.0)
            canopy_cover = round(canopy_mean, 2)
            environmental_layer_status["canopy"] = "ok"
            sources.append("Canopy density raster")
        else:
            canopy_index = None
            canopy_cover = None
            environmental_layer_status["canopy"] = "missing" if not self._file_exists(canopy_path) else "error"
            assumptions.append("Canopy density unavailable within 100m neighborhood.")

        moisture, _moisture_status = self._sample_layer_value(self.paths["moisture"], lat, lon)
        if moisture is None:
            moisture_index = None
            assumptions.append("Moisture/fuel dryness raster missing; dryness could not be directly measured.")
        else:
            moisture_index = self._to_index(moisture, 0.0, 100.0)
            sources.append("Moisture/fuel dryness raster")

        wildland_distance, wildland_distance_index = self._wildland_distance_metrics(lat, lon, fuel_path, canopy_path)
        if wildland_distance is not None:
            sources.append("Distance to wildland vegetation (derived)")
        else:
            assumptions.append("Fuel/canopy rasters missing; wildland distance unavailable.")

        historic_fire_distance, historic_fire_index, fire_history_status = self._historical_fire_metrics(
            lat, lon, self.paths["perimeters"]
        )
        environmental_layer_status["fire_history"] = fire_history_status
        if fire_history_status == "ok":
            sources.append("Historical fire perimeter recurrence")
        else:
            assumptions.append("Historical perimeter layer unavailable; recurrence unavailable.")

        weighted_terms = [
            (0.20, burn_probability_index),
            (0.16, hazard_severity_index),
            (0.12, slope_index),
            (0.06, aspect_index),
            (0.12, fuel_index),
            (0.10, moisture_index),
            (0.10, canopy_index),
            (0.08, wildland_distance_index),
            (0.06, historic_fire_index),
        ]
        available_terms = [(w, v) for (w, v) in weighted_terms if v is not None]
        environmental = None
        if available_terms:
            numerator = sum(w * float(v) for w, v in available_terms)
            denominator = sum(w for w, _ in available_terms)
            environmental = round(numerator / denominator, 1)

        return WildfireContext(
            environmental_index=environmental,
            slope_index=None if slope_index is None else round(slope_index, 1),
            aspect_index=None if aspect_index is None else round(aspect_index, 1),
            fuel_index=None if fuel_index is None else round(fuel_index, 1),
            moisture_index=None if moisture_index is None else round(moisture_index, 1),
            canopy_index=None if canopy_index is None else round(canopy_index, 1),
            wildland_distance_index=None if wildland_distance_index is None else round(wildland_distance_index, 1),
            historic_fire_index=None if historic_fire_index is None else round(historic_fire_index, 1),
            burn_probability_index=None if burn_probability_index is None else round(burn_probability_index, 1),
            hazard_severity_index=None if hazard_severity_index is None else round(hazard_severity_index, 1),
            burn_probability=burn_prob,
            wildfire_hazard=hazard,
            slope=slope,
            fuel_model=fuel_model,
            canopy_cover=canopy_cover,
            historic_fire_distance=None if historic_fire_distance is None else round(historic_fire_distance, 2),
            wildland_distance=wildland_distance,
            environmental_layer_status=environmental_layer_status,
            data_sources=sorted(set(sources)),
            assumptions=assumptions,
            structure_ring_metrics=structure_ring_metrics,
            property_level_context=property_level_context,
        )
