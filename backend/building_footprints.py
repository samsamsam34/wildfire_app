from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from pyproj import Transformer
    from shapely.geometry import MultiPolygon, Point, Polygon, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - graceful fallback when geo deps are unavailable
    Transformer = None
    Point = None
    Polygon = None
    MultiPolygon = None
    shape = None
    shapely_transform = None


FEET_TO_METERS = 0.3048
RING_KEYS = ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft")


@dataclass
class BuildingFootprintResult:
    found: bool
    footprint: Any | None = None
    centroid: tuple[float, float] | None = None
    source: str | None = None
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)


class BuildingFootprintClient:
    def __init__(self, path: str | None = None, max_search_m: float = 120.0) -> None:
        self.path = path or os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON", "")
        self.max_search_m = max_search_m

    @staticmethod
    def _geo_ready() -> bool:
        return bool(Transformer and Point and shape and shapely_transform)

    @staticmethod
    def _file_exists(path: str) -> bool:
        return bool(path) and Path(path).exists()

    @lru_cache(maxsize=4)
    def _load_features(self, path: str) -> list[Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", []) if isinstance(data, dict) else []
        geoms = []
        for feature in features:
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not geometry:
                continue
            try:
                geom = shape(geometry)
            except Exception:
                continue
            if geom.is_empty:
                continue
            if geom.geom_type in {"Polygon", "MultiPolygon"}:
                geoms.append(geom)
        return geoms

    @staticmethod
    def _primary_polygon(geom: Any) -> Any:
        if MultiPolygon is not None and isinstance(geom, MultiPolygon):
            parts = [g for g in geom.geoms if not g.is_empty]
            if parts:
                return max(parts, key=lambda g: g.area)
        return geom

    def get_building_footprint(self, lat: float, lon: float) -> BuildingFootprintResult:
        assumptions: list[str] = []
        if not self._geo_ready():
            assumptions.append("Building footprint analysis unavailable; geospatial dependencies missing.")
            return BuildingFootprintResult(found=False, assumptions=assumptions)

        if not self._file_exists(self.path):
            assumptions.append("Building footprint source is not configured or missing.")
            return BuildingFootprintResult(found=False, assumptions=assumptions)

        geoms = self._load_features(self.path)
        if not geoms:
            assumptions.append("No building footprints available in configured source.")
            return BuildingFootprintResult(found=False, source=self.path, assumptions=assumptions)

        point_wgs84 = Point(lon, lat)
        containing = [self._primary_polygon(g) for g in geoms if g.contains(point_wgs84)]
        if containing:
            geom = max(containing, key=lambda g: g.area)
            c = geom.centroid
            return BuildingFootprintResult(
                found=True,
                footprint=geom,
                centroid=(float(c.y), float(c.x)),
                source=self.path,
                confidence=0.95,
                assumptions=assumptions,
            )

        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        point_m = shapely_transform(to_3857, point_wgs84)

        nearest_geom = None
        nearest_distance_m = None
        for geom in geoms:
            candidate = self._primary_polygon(geom)
            distance = shapely_transform(to_3857, candidate).distance(point_m)
            if nearest_distance_m is None or distance < nearest_distance_m:
                nearest_distance_m = float(distance)
                nearest_geom = candidate

        if nearest_geom is None or nearest_distance_m is None or nearest_distance_m > self.max_search_m:
            assumptions.append("No nearby building footprint found for this location.")
            return BuildingFootprintResult(found=False, source=self.path, assumptions=assumptions)

        confidence = max(0.35, min(0.9, 0.9 - (nearest_distance_m / max(self.max_search_m, 1.0)) * 0.5))
        c = nearest_geom.centroid
        return BuildingFootprintResult(
            found=True,
            footprint=nearest_geom,
            centroid=(float(c.y), float(c.x)),
            source=self.path,
            confidence=round(confidence, 2),
            assumptions=assumptions,
        )


def compute_structure_rings(footprint: Any) -> tuple[dict[str, Any], list[str]]:
    assumptions: list[str] = []
    if not (Transformer and shapely_transform):
        assumptions.append("Cannot compute structure rings; geospatial dependencies missing.")
        return {}, assumptions

    if footprint is None or footprint.is_empty:
        assumptions.append("Cannot compute structure rings; footprint geometry is missing.")
        return {}, assumptions

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    footprint_m = shapely_transform(to_3857, footprint)
    b5_m = footprint_m.buffer(5.0 * FEET_TO_METERS)
    b30_m = footprint_m.buffer(30.0 * FEET_TO_METERS)
    b100_m = footprint_m.buffer(100.0 * FEET_TO_METERS)

    rings_m = {
        "ring_0_5_ft": b5_m.difference(footprint_m),
        "ring_5_30_ft": b30_m.difference(b5_m),
        "ring_30_100_ft": b100_m.difference(b30_m),
    }

    rings_wgs84: dict[str, Any] = {}
    for key, ring in rings_m.items():
        if ring.is_empty:
            continue
        rings_wgs84[key] = shapely_transform(to_wgs84, ring)

    if len(rings_wgs84) != len(RING_KEYS):
        assumptions.append("Some structure rings could not be generated from footprint geometry.")

    return rings_wgs84, assumptions
