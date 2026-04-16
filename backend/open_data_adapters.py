from __future__ import annotations

import json
import logging
import math
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import rasterio
    from pyproj import CRS, Transformer
    from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - optional geo dependency fallback
    rasterio = None
    CRS = None
    Transformer = None
    LineString = None
    MultiLineString = None
    Point = None
    mapping = None
    shape = None
    shapely_transform = None

logger = logging.getLogger(__name__)


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


def _axis_info(ds) -> list[Any]:
    crs = getattr(ds, "crs", None)
    if crs is None:
        return []
    axis = getattr(crs, "axis_info", None)
    if axis is None and CRS is not None:
        try:
            axis = CRS.from_user_input(crs).axis_info
        except Exception:
            axis = None
    if axis is None:
        return []
    return list(axis)


def _axis_info_strings(ds) -> list[str]:
    axis_strings: list[str] = []
    for axis in _axis_info(ds):
        name = str(getattr(axis, "name", "") or "").strip()
        direction = str(getattr(axis, "direction", "") or "").strip()
        unit = str(getattr(axis, "unit_name", "") or "").strip()
        label = f"{name}:{direction}" if (name or direction) else str(axis)
        if unit:
            label = f"{label} ({unit})"
        axis_strings.append(label)
    return axis_strings


def _is_lat_first_axis(ds) -> bool:
    axis = _axis_info(ds)
    if len(axis) < 2:
        return False
    first = str(getattr(axis[0], "direction", "") or "").strip().lower()
    second = str(getattr(axis[1], "direction", "") or "").strip().lower()
    return first in {"north", "south"} and second in {"east", "west"}


def _point_within_bounds(ds, x: float, y: float) -> bool:
    try:
        left, bottom, right, top = ds.bounds
    except Exception:
        return True
    return left <= float(x) <= right and bottom <= float(y) <= top


def _resolve_sample_coords(ds, x: float, y: float) -> tuple[tuple[float, float], bool]:
    default_coords = (float(x), float(y))
    if not (bool(getattr(getattr(ds, "crs", None), "is_geographic", False)) and _is_lat_first_axis(ds)):
        return default_coords, False

    swapped_coords = (float(y), float(x))
    default_in_bounds = _point_within_bounds(ds, default_coords[0], default_coords[1])
    swapped_in_bounds = _point_within_bounds(ds, swapped_coords[0], swapped_coords[1])
    if swapped_in_bounds and not default_in_bounds:
        return swapped_coords, True
    return default_coords, False


def _sample_raster_point_detailed(path: str, lat: float, lon: float) -> dict[str, Any]:
    if not (rasterio and _file_exists(path)):
        return {"status": "missing_file", "reason": "Raster source unavailable.", "value": None}
    try:
        ds = _open_raster(path)
        x, y = _to_dataset_crs(ds, lon, lat)
        sample_coords, swapped = _resolve_sample_coords(ds, x, y)
        axis_info = _axis_info_strings(ds)
        within_bounds = _point_within_bounds(ds, sample_coords[0], sample_coords[1])
        if not within_bounds:
            return {
                "status": "no_data",
                "reason": "Sample coordinate outside raster bounds.",
                "value": None,
                "axis_info": axis_info,
                "sample_coords": sample_coords,
                "within_bounds": False,
                "coords_swapped": swapped,
            }

        sample = next(ds.sample([sample_coords]))[0]
        nodata = ds.nodata
        if nodata is not None and float(sample) == float(nodata):
            return {
                "status": "no_data",
                "reason": "Raster sample returned nodata.",
                "value": None,
                "axis_info": axis_info,
                "sample_coords": sample_coords,
                "within_bounds": True,
                "coords_swapped": swapped,
            }
        return {
            "status": "ok",
            "reason": None,
            "value": float(sample),
            "axis_info": axis_info,
            "sample_coords": sample_coords,
            "within_bounds": True,
            "coords_swapped": swapped,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "value": None,
            "axis_info": [],
            "sample_coords": None,
            "within_bounds": False,
            "coords_swapped": False,
        }


def _sample_raster_point(path: str, lat: float, lon: float) -> float | None:
    detailed = _sample_raster_point_detailed(path, lat, lon)
    if str(detailed.get("status") or "") != "ok":
        return None
    value = detailed.get("value")
    if value is None:
        return None
    return float(value)


def _sample_raster_circle_detailed(path: str, lat: float, lon: float, radius_m: float, step_m: float = 120.0) -> dict[str, Any]:
    if not (rasterio and _file_exists(path)):
        return {"status": "missing_file", "reason": "Raster source unavailable.", "values": []}
    try:
        ds = _open_raster(path)
        axis_info = _axis_info_strings(ds)
        values: list[float] = []
        errors: list[str] = []
        any_within_bounds = False
        representative_coords: tuple[float, float] | None = None
        swapped_flag = False
        rings = max(1, int(radius_m / step_m))

        for ring in range(1, rings + 1):
            r = ring * step_m
            points = max(8, int(2 * math.pi * r / step_m))
            for i in range(points):
                theta = 2.0 * math.pi * i / points
                d_lat = _meters_to_lat_deg(r * math.sin(theta))
                d_lon = _meters_to_lon_deg(r * math.cos(theta), lat)
                x, y = _to_dataset_crs(ds, lon + d_lon, lat + d_lat)
                sample_coords, swapped = _resolve_sample_coords(ds, x, y)
                if representative_coords is None:
                    representative_coords = sample_coords
                    swapped_flag = swapped
                if not _point_within_bounds(ds, sample_coords[0], sample_coords[1]):
                    continue
                any_within_bounds = True
                try:
                    sample = next(ds.sample([sample_coords]))[0]
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                nodata = ds.nodata
                if nodata is not None and float(sample) == float(nodata):
                    continue
                values.append(float(sample))

        if values:
            return {
                "status": "ok",
                "reason": None,
                "values": values,
                "axis_info": axis_info,
                "sample_coords": representative_coords,
                "within_bounds": any_within_bounds,
                "coords_swapped": swapped_flag,
            }
        if errors:
            return {
                "status": "error",
                "reason": "; ".join(errors[:3]),
                "values": [],
                "axis_info": axis_info,
                "sample_coords": representative_coords,
                "within_bounds": any_within_bounds,
                "coords_swapped": swapped_flag,
            }
        return {
            "status": "no_data",
            "reason": (
                "No valid raster samples within search radius."
                if any_within_bounds
                else "All sample coordinates were outside raster bounds."
            ),
            "values": [],
            "axis_info": axis_info,
            "sample_coords": representative_coords,
            "within_bounds": any_within_bounds,
            "coords_swapped": swapped_flag,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "values": [],
            "axis_info": [],
            "sample_coords": None,
            "within_bounds": False,
            "coords_swapped": False,
        }


def _sample_raster_circle(path: str, lat: float, lon: float, radius_m: float, step_m: float = 120.0) -> List[float]:
    detailed = _sample_raster_circle_detailed(path, lat, lon, radius_m, step_m=step_m)
    values = detailed.get("values")
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


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
        try:
            if not _file_exists(whp_path):
                return WHPObservation(status="missing", notes=["WHP raster source unavailable."])

            sample_result = _sample_raster_point_detailed(str(whp_path), lat, lon)
            logger.debug(
                "WHPAdapter axis order: %s, using coords: %s",
                sample_result.get("axis_info"),
                sample_result.get("sample_coords"),
            )
            status = str(sample_result.get("status") or "")
            if status == "error":
                reason = str(sample_result.get("reason") or "Unknown WHP sampling error.")
                return WHPObservation(status="error", notes=[f"WHP sampling failed: {reason}"])

            raw = sample_result.get("value")
            if raw is None:
                reason = str(sample_result.get("reason") or "WHP value unavailable at property location.")
                return WHPObservation(status="missing", notes=[reason])

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
        except Exception as exc:
            logger.warning("WHPAdapter.sample failed: %s\n%s", exc, traceback.format_exc())
            return WHPObservation(status="error", notes=[str(exc)])


class GridMETAdapter:
    def sample_dryness(
        self,
        *,
        lat: float,
        lon: float,
        dryness_raster_path: str | None,
        rolling_window_days: int = 14,
    ) -> GridMETDrynessObservation:
        try:
            if not _file_exists(dryness_raster_path):
                return GridMETDrynessObservation(status="missing", notes=["gridMET dryness source unavailable."])

            sample_result = _sample_raster_circle_detailed(
                str(dryness_raster_path),
                lat,
                lon,
                radius_m=1000.0,
                step_m=150.0,
            )
            logger.debug(
                "GridMETAdapter axis order: %s, using coords: %s",
                sample_result.get("axis_info"),
                sample_result.get("sample_coords"),
            )
            status = str(sample_result.get("status") or "")
            if status == "error":
                reason = str(sample_result.get("reason") or "Unknown gridMET sampling error.")
                return GridMETDrynessObservation(status="error", notes=[f"gridMET sampling failed: {reason}"])

            values = sample_result.get("values")
            if not isinstance(values, list) or not values:
                reason = str(sample_result.get("reason") or "No gridMET dryness samples available near property.")
                return GridMETDrynessObservation(status="missing", notes=[reason])

            avg = sum(float(v) for v in values) / len(values)
            dryness_index = _to_index(avg, 0.0, 100.0 if max(float(v) for v in values) > 5 else 1.0)
            return GridMETDrynessObservation(
                status="ok",
                raw_value=round(avg, 3),
                dryness_index=dryness_index,
                rolling_window_days=rolling_window_days,
                notes=[f"Derived from local gridMET proxy within {rolling_window_days}-day rolling context."],
            )
        except Exception as exc:
            logger.warning("GridMETAdapter.sample_dryness failed: %s\n%s", exc, traceback.format_exc())
            return GridMETDrynessObservation(status="error", notes=[str(exc)])


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
