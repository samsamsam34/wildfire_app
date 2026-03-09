from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - optional geo dependency fallback
    rasterio = None
    Transformer = None
    LineString = None
    MultiLineString = None
    Point = None
    mapping = None
    shape = None
    shapely_transform = None


def _file_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _meters_to_lat_deg(meters: float) -> float:
    return meters / 111_320.0


def _meters_to_lon_deg(meters: float, lat: float) -> float:
    denom = 111_320.0 * max(0.1, math.cos(math.radians(lat)))
    return meters / denom


def _to_index(value: float, src_min: float, src_max: float) -> float:
    if src_max <= src_min:
        return 50.0
    v = max(src_min, min(src_max, value))
    return round(100.0 * (v - src_min) / (src_max - src_min), 1)


@lru_cache(maxsize=16)
def _open_raster(path: str):
    if rasterio is None:
        raise RuntimeError("rasterio is required")
    return rasterio.open(path)


def _to_dataset_crs(ds, lon: float, lat: float) -> tuple[float, float]:
    if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        return lon, lat
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    return transformer.transform(lon, lat)


def _sample_raster_point(path: str, lat: float, lon: float) -> float | None:
    if not (rasterio and _file_exists(path)):
        return None
    try:
        ds = _open_raster(path)
        x, y = _to_dataset_crs(ds, lon, lat)
        sample = next(ds.sample([(x, y)]))[0]
        nodata = ds.nodata
        if nodata is not None and float(sample) == float(nodata):
            return None
        return float(sample)
    except Exception:
        return None


def _sample_raster_circle(path: str, lat: float, lon: float, radius_m: float, step_m: float = 120.0) -> List[float]:
    if not (rasterio and _file_exists(path)):
        return []
    values: List[float] = []
    rings = max(1, int(radius_m / step_m))
    for ring in range(1, rings + 1):
        r = ring * step_m
        points = max(8, int(2 * math.pi * r / step_m))
        for i in range(points):
            theta = 2.0 * math.pi * i / points
            d_lat = _meters_to_lat_deg(r * math.sin(theta))
            d_lon = _meters_to_lon_deg(r * math.cos(theta), lat)
            sample = _sample_raster_point(path, lat + d_lat, lon + d_lon)
            if sample is not None:
                values.append(sample)
    return values


@dataclass
class WHPObservation:
    status: str
    raw_value: float | None = None
    hazard_class: str | None = None
    burn_probability_index: float | None = None
    hazard_severity_index: float | None = None
    source_dataset: str = "USFS Wildfire Hazard Potential (WHP)"
    notes: List[str] = field(default_factory=list)


@dataclass
class GridMETDrynessObservation:
    status: str
    raw_value: float | None = None
    dryness_index: float | None = None
    source_dataset: str = "gridMET dryness proxy"
    rolling_window_days: int = 14
    notes: List[str] = field(default_factory=list)


@dataclass
class MTBSSummary:
    status: str
    nearest_perimeter_km: float | None = None
    intersects_prior_burn: bool = False
    nearby_high_severity: bool = False
    fire_history_index: float | None = None
    source_dataset: str = "MTBS"
    notes: List[str] = field(default_factory=list)


@dataclass
class OSMAccessSummary:
    status: str
    distance_to_nearest_road_m: float | None = None
    road_segments_within_300m: int = 0
    intersections_within_300m: int = 0
    dead_end_indicator: bool = False
    access_exposure_index: float | None = None
    source_dataset: str = "OpenStreetMap road network"
    notes: List[str] = field(default_factory=list)


class WHPAdapter:
    def sample(self, *, lat: float, lon: float, whp_path: str | None) -> WHPObservation:
        if not _file_exists(whp_path):
            return WHPObservation(status="missing", notes=["WHP raster source unavailable."])

        raw = _sample_raster_point(str(whp_path), lat, lon)
        if raw is None:
            return WHPObservation(status="missing", notes=["WHP value unavailable at property location."])

        value = float(raw)
        # WHP common encodings: 1..5 classes or 0..100 normalized values.
        if value <= 5.0:
            hazard_index = _to_index(value, 1.0, 5.0)
        else:
            hazard_index = _to_index(value, 0.0, 100.0)

        burn_index = round(max(0.0, min(100.0, hazard_index * 0.9 + 5.0)), 1)
        cls = "very_high" if hazard_index >= 80 else "high" if hazard_index >= 60 else "moderate" if hazard_index >= 35 else "low"
        return WHPObservation(
            status="ok",
            raw_value=round(value, 3),
            hazard_class=cls,
            burn_probability_index=burn_index,
            hazard_severity_index=hazard_index,
            notes=["WHP sampled at geocoded property location."],
        )


class GridMETAdapter:
    def sample_dryness(
        self,
        *,
        lat: float,
        lon: float,
        dryness_raster_path: str | None,
        rolling_window_days: int = 14,
    ) -> GridMETDrynessObservation:
        if not _file_exists(dryness_raster_path):
            return GridMETDrynessObservation(status="missing", notes=["gridMET dryness source unavailable."])

        values = _sample_raster_circle(str(dryness_raster_path), lat, lon, radius_m=1000.0, step_m=150.0)
        if not values:
            return GridMETDrynessObservation(status="missing", notes=["No gridMET dryness samples available near property."])

        avg = sum(values) / len(values)
        dryness_index = _to_index(avg, 0.0, 100.0 if max(values) > 5 else 1.0)
        return GridMETDrynessObservation(
            status="ok",
            raw_value=round(avg, 3),
            dryness_index=dryness_index,
            rolling_window_days=rolling_window_days,
            notes=[f"Derived from local gridMET proxy within {rolling_window_days}-day rolling context."],
        )


@lru_cache(maxsize=8)
def _load_vector_geometries(path: str, allowed_types: tuple[str, ...]) -> List[Any]:
    if not _file_exists(path):
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    features = payload.get("features", []) if isinstance(payload, dict) else []
    geoms: List[Any] = []
    for feature in features:
        geom_blob = feature.get("geometry") if isinstance(feature, dict) else None
        if not geom_blob:
            continue
        try:
            geom = shape(geom_blob)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if geom.geom_type in allowed_types:
            geoms.append(geom)
    return geoms


def _to_3857(geom: Any) -> Any:
    if not (Transformer and shapely_transform):
        return geom
    tx = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return shapely_transform(tx.transform, geom)


class MTBSAdapter:
    def summarize(
        self,
        *,
        lat: float,
        lon: float,
        perimeter_path: str | None,
        burn_severity_path: str | None = None,
    ) -> MTBSSummary:
        if not _file_exists(perimeter_path) and not _file_exists(burn_severity_path):
            return MTBSSummary(status="missing", notes=["MTBS perimeter/severity sources unavailable."])

        pt = Point(lon, lat) if Point else None
        nearest_km: float | None = None
        intersects = False
        nearby_perimeters = 0

        geoms = _load_vector_geometries(str(perimeter_path), ("Polygon", "MultiPolygon")) if _file_exists(perimeter_path) else []
        if pt and geoms:
            lat_5 = _meters_to_lat_deg(5000)
            lon_5 = _meters_to_lon_deg(5000, lat)
            for geom in geoms:
                minx, miny, maxx, maxy = geom.bounds
                if not (maxx < lon - lon_5 or minx > lon + lon_5 or maxy < lat - lat_5 or miny > lat + lat_5):
                    nearby_perimeters += 1
                try:
                    if geom.contains(pt):
                        intersects = True
                    km = float(pt.distance(geom)) * 111.32
                    nearest_km = km if nearest_km is None else min(nearest_km, km)
                except Exception:
                    pass

        high_severity = False
        if _file_exists(burn_severity_path):
            severity_vals = _sample_raster_circle(str(burn_severity_path), lat, lon, radius_m=5000.0, step_m=250.0)
            if severity_vals:
                # MTBS dNBR thematic-style thresholds vary by source; 3+ often signals moderate/high class bins.
                high_share = sum(1 for v in severity_vals if v >= 3.0) / len(severity_vals)
                high_severity = high_share >= 0.15

        if nearest_km is None and not high_severity and not intersects and not geoms:
            return MTBSSummary(status="missing", notes=["MTBS features unavailable at property location."])

        score = 0.0
        if intersects:
            score += 45.0
        if nearest_km is not None:
            score += max(0.0, 40.0 - min(nearest_km, 20.0) * 2.0)
        score += min(20.0, nearby_perimeters * 4.0)
        if high_severity:
            score += 15.0
        score = max(0.0, min(100.0, score))

        return MTBSSummary(
            status="ok",
            nearest_perimeter_km=None if nearest_km is None else round(nearest_km, 2),
            intersects_prior_burn=intersects,
            nearby_high_severity=high_severity,
            fire_history_index=round(score, 1),
            notes=["MTBS perimeter and burn-severity context sampled for local exposure."],
        )


def _iter_lines(geom: Any) -> Iterable[Any]:
    if LineString is not None and isinstance(geom, LineString):
        yield geom
    elif MultiLineString is not None and isinstance(geom, MultiLineString):
        for line in geom.geoms:
            yield line


class OSMRoadAdapter:
    def summarize(self, *, lat: float, lon: float, roads_path: str | None) -> OSMAccessSummary:
        if not _file_exists(roads_path):
            return OSMAccessSummary(status="missing", notes=["OSM road source unavailable."])
        if not (Point and Transformer and shapely_transform):
            return OSMAccessSummary(status="error", notes=["Road-network analysis unavailable; geospatial dependencies missing."])

        line_geoms = _load_vector_geometries(str(roads_path), ("LineString", "MultiLineString"))
        if not line_geoms:
            return OSMAccessSummary(status="missing", notes=["No road geometries available for OSM source."])

        pt = Point(lon, lat)
        pt_m = _to_3857(pt)
        buffer_300m = pt_m.buffer(300.0)

        nearest_m: float | None = None
        near_segments = 0
        endpoint_counts: Dict[tuple[int, int], int] = {}

        for geom in line_geoms:
            for line in _iter_lines(geom):
                line_m = _to_3857(line)
                dist = float(pt_m.distance(line_m))
                nearest_m = dist if nearest_m is None else min(nearest_m, dist)
                if line_m.intersects(buffer_300m):
                    near_segments += 1
                    coords = list(line_m.coords)
                    if coords:
                        start = (round(coords[0][0]), round(coords[0][1]))
                        end = (round(coords[-1][0]), round(coords[-1][1]))
                        endpoint_counts[start] = endpoint_counts.get(start, 0) + 1
                        endpoint_counts[end] = endpoint_counts.get(end, 0) + 1

        intersections = sum(1 for v in endpoint_counts.values() if v >= 3)
        dead_end = near_segments <= 2 and intersections == 0

        if nearest_m is None:
            return OSMAccessSummary(status="missing", notes=["No roads found near geocoded property point."])

        score = min(70.0, (nearest_m / 800.0) * 70.0)
        if dead_end:
            score += 20.0
        if intersections <= 1:
            score += 10.0
        elif intersections <= 3:
            score += 5.0

        return OSMAccessSummary(
            status="ok",
            distance_to_nearest_road_m=round(nearest_m, 1),
            road_segments_within_300m=near_segments,
            intersections_within_300m=intersections,
            dead_end_indicator=dead_end,
            access_exposure_index=round(max(0.0, min(100.0, score)), 1),
            notes=["Road-network access metrics derived from OSM vector context."],
        )
