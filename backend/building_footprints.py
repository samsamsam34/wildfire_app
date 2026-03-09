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
RING_KEYS = ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "ring_100_300_ft")


@dataclass
class BuildingFootprintResult:
    found: bool
    footprint: Any | None = None
    centroid: tuple[float, float] | None = None
    source: str | None = None
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)


class BuildingFootprintClient:
    def __init__(
        self,
        path: str | None = None,
        max_search_m: float = 120.0,
        extra_paths: list[str] | None = None,
    ) -> None:
        configured = [
            path or "",
            os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON", ""),
            os.getenv("WF_LAYER_FEMA_STRUCTURES_GEOJSON", ""),
        ]
        if extra_paths:
            configured.extend(extra_paths)

        unique_paths: list[str] = []
        for candidate in configured:
            if candidate and candidate not in unique_paths:
                unique_paths.append(candidate)

        self.path = unique_paths[0] if unique_paths else ""
        self.paths = unique_paths
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

    @staticmethod
    def _geom_area_m2(geom: Any) -> float:
        if not (Transformer and shapely_transform):
            return 0.0
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        geom_m = shapely_transform(to_3857, geom)
        return float(max(0.0, geom_m.area))

    @staticmethod
    def _residential_area_score(area_m2: float) -> float:
        # Prefer residential-sized structures while allowing larger outbuildings.
        if area_m2 <= 0:
            return 0.0
        target = 190.0
        spread = 320.0
        score = 1.0 - min(1.0, abs(area_m2 - target) / spread)
        return max(0.0, score)

    def _all_source_features(self) -> list[tuple[Any, str]]:
        features: list[tuple[Any, str]] = []
        for source_path in self.paths:
            if not self._file_exists(source_path):
                continue
            for geom in self._load_features(source_path):
                features.append((self._primary_polygon(geom), source_path))
        return features

    def get_building_footprint(self, lat: float, lon: float) -> BuildingFootprintResult:
        assumptions: list[str] = []
        if not self._geo_ready():
            assumptions.append("Building footprint analysis unavailable; geospatial dependencies missing.")
            return BuildingFootprintResult(found=False, assumptions=assumptions)

        if not self.paths:
            assumptions.append("Building footprint source is not configured or missing.")
            return BuildingFootprintResult(found=False, assumptions=assumptions)

        features = self._all_source_features()
        if not features:
            assumptions.append("No building footprints available in configured source(s).")
            return BuildingFootprintResult(found=False, source=self.path, assumptions=assumptions)

        point_wgs84 = Point(lon, lat)
        containing = [(geom, src) for geom, src in features if geom.contains(point_wgs84)]
        if containing:
            # Deterministically pick best residential-sized containing polygon.
            def _contain_score(item: tuple[Any, str]) -> float:
                geom, _src = item
                area_m2 = self._geom_area_m2(geom)
                return self._residential_area_score(area_m2) + min(1.0, area_m2 / 10_000.0) * 0.05

            geom, source = max(containing, key=_contain_score)
            c = geom.centroid
            return BuildingFootprintResult(
                found=True,
                footprint=geom,
                centroid=(float(c.y), float(c.x)),
                source=source,
                confidence=0.97,
                assumptions=assumptions,
            )

        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        point_m = shapely_transform(to_3857, point_wgs84)

        nearest_geom = None
        nearest_source = None
        nearest_distance_m = None
        best_score = None
        for candidate, source in features:
            candidate_m = shapely_transform(to_3857, candidate)
            distance = candidate_m.distance(point_m)
            area_score = self._residential_area_score(self._geom_area_m2(candidate))
            score = (max(0.0, 1.0 - (distance / max(self.max_search_m, 1.0))) * 0.8) + (area_score * 0.2)
            if nearest_distance_m is None or distance < nearest_distance_m:
                nearest_distance_m = float(distance)
            if best_score is None or score > best_score:
                best_score = score
                nearest_geom = candidate
                nearest_source = source

        if nearest_geom is None or nearest_distance_m is None or nearest_distance_m > self.max_search_m:
            assumptions.append("No nearby building footprint found for this location.")
            return BuildingFootprintResult(found=False, source=self.path, assumptions=assumptions)

        confidence = max(0.35, min(0.9, 0.9 - (nearest_distance_m / max(self.max_search_m, 1.0)) * 0.5))
        c = nearest_geom.centroid
        return BuildingFootprintResult(
            found=True,
            footprint=nearest_geom,
            centroid=(float(c.y), float(c.x)),
            source=nearest_source or self.path,
            confidence=round(confidence, 2),
            assumptions=assumptions,
        )

    def get_neighbor_structure_metrics(
        self,
        *,
        lat: float,
        lon: float,
        subject_footprint: Any | None = None,
        source_path: str | None = None,
        radius_m: float = 300.0,
    ) -> dict[str, float | int | None]:
        if not self._geo_ready():
            return {
                "nearby_structure_count_100_ft": None,
                "nearby_structure_count_300_ft": None,
                "nearest_structure_distance_ft": None,
            }

        path_to_use = source_path or self.path
        if not self._file_exists(path_to_use):
            return {
                "nearby_structure_count_100_ft": None,
                "nearby_structure_count_300_ft": None,
                "nearest_structure_distance_ft": None,
            }

        try:
            geoms = [self._primary_polygon(g) for g in self._load_features(path_to_use)]
        except Exception:
            geoms = []
        if not geoms:
            return {
                "nearby_structure_count_100_ft": None,
                "nearby_structure_count_300_ft": None,
                "nearest_structure_distance_ft": None,
            }

        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        point = Point(lon, lat)
        point_m = shapely_transform(to_3857, point)
        subject_m = shapely_transform(to_3857, subject_footprint) if subject_footprint is not None else None

        nearby_100 = 0
        nearby_300 = 0
        nearest_m: float | None = None
        r100 = 100.0 * FEET_TO_METERS
        r300 = min(radius_m, 300.0 * FEET_TO_METERS)

        for geom in geoms:
            geom_m = shapely_transform(to_3857, geom)
            if subject_m is not None and geom_m.distance(subject_m) < 0.5:
                continue

            d = float(point_m.distance(geom_m))
            nearest_m = d if nearest_m is None else min(nearest_m, d)
            if d <= r100:
                nearby_100 += 1
            if d <= r300:
                nearby_300 += 1

        nearest_ft = None if nearest_m is None else round(nearest_m / FEET_TO_METERS, 1)
        return {
            "nearby_structure_count_100_ft": nearby_100,
            "nearby_structure_count_300_ft": nearby_300,
            "nearest_structure_distance_ft": nearest_ft,
        }


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
    b300_m = footprint_m.buffer(300.0 * FEET_TO_METERS)

    rings_m = {
        "ring_0_5_ft": b5_m.difference(footprint_m),
        "ring_5_30_ft": b30_m.difference(b5_m),
        "ring_30_100_ft": b100_m.difference(b30_m),
        "ring_100_300_ft": b300_m.difference(b100_m),
    }

    rings_wgs84: dict[str, Any] = {}
    for key, ring in rings_m.items():
        if ring.is_empty:
            continue
        rings_wgs84[key] = shapely_transform(to_wgs84, ring)

    if len(rings_wgs84) != len(RING_KEYS):
        assumptions.append("Some structure rings could not be generated from footprint geometry.")

    return rings_wgs84, assumptions
