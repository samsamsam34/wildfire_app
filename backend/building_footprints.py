from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from backend.national_footprint_index import NationalFootprintIndex

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

LOGGER = logging.getLogger("wildfire_app.building_footprints")

FEET_TO_METERS = 0.3048
RING_KEYS = ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "ring_100_300_ft")


@dataclass
class BuildingFootprintResult:
    found: bool
    footprint: Any | None = None
    centroid: tuple[float, float] | None = None
    source: str | None = None
    confidence: float = 0.0
    match_status: str = "none"
    match_method: str | None = None
    matched_structure_id: str | None = None
    match_distance_m: float | None = None
    candidate_count: int = 0
    candidate_summaries: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    feature_properties: dict[str, Any] | None = None
    on_parcel_structure_count: int | None = None


class BuildingFootprintClient:
    def __init__(
        self,
        path: str | None = None,
        max_search_m: float = 120.0,
        extra_paths: list[str] | None = None,
        national_index: "NationalFootprintIndex | None" = None,
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
        self.max_match_distance_m = self._env_float(
            "WF_STRUCTURE_MATCH_MAX_DISTANCE_M",
            max(5.0, min(float(max_search_m), 35.0)),
            min_value=1.0,
        )
        self.ambiguity_gap_m = self._env_float("WF_STRUCTURE_MATCH_AMBIGUITY_GAP_M", 6.0, min_value=0.1)
        self.max_candidate_summaries = int(
            max(
                1,
                min(
                    8,
                    round(
                        self._env_float(
                            "WF_STRUCTURE_MATCH_MAX_CANDIDATE_SUMMARIES",
                            4.0,
                            min_value=1.0,
                        )
                    ),
                ),
            )
        )
        self._national_index = national_index

    @staticmethod
    def _env_float(name: str, default: float, *, min_value: float | None = None) -> float:
        raw = str(os.getenv(name, str(default))).strip()
        try:
            value = float(raw)
        except ValueError:
            value = float(default)
        if min_value is not None:
            value = max(min_value, value)
        return value

    @staticmethod
    def _geo_ready() -> bool:
        return bool(Transformer and Point and shape and shapely_transform)

    @staticmethod
    def _file_exists(path: str) -> bool:
        return bool(path) and Path(path).exists()

    @lru_cache(maxsize=4)
    def _load_features(self, path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", []) if isinstance(data, dict) else []
        rows: list[dict[str, Any]] = []
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
                props = dict(feature.get("properties") or {}) if isinstance(feature, dict) else {}
                rows.append(
                    {
                        "geometry": geom,
                        "properties": props,
                        "structure_id": self._extract_structure_id(props),
                    }
                )
        return rows

    @staticmethod
    def _extract_structure_id(props: dict[str, Any]) -> str | None:
        for key in (
            "structure_id",
            "building_id",
            "id",
            "objectid",
            "OBJECTID",
            "globalid",
            "GlobalID",
            "fid",
            "FID",
        ):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

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
    def _area_plausibility_score(area_m2: float, parcel_available: bool) -> float:
        """Return a plausibility score (0–1) for a footprint based on its area.

        When a parcel polygon is available, area-based scoring is disabled —
        parcel intersection is a stronger signal and applying an area bias would
        penalise large-but-legitimate structures (farmhouses, barns).

        When no parcel is available a loose filter is applied:
        * ``< 15 m²``: score 0.3  — too small to be a primary residence.
        * ``15–40 m²``: score 0.7  — small but possible (cottage, ADU).
        * ``≥ 40 m²``:  score 1.0  — no penalty for any larger structure.

        Parameters
        ----------
        area_m2:
            Footprint area in square metres.
        parcel_available:
            ``True`` if a parcel polygon was matched for this assessment.
        """
        if parcel_available:
            return 1.0
        if area_m2 < 15:
            return 0.3
        if area_m2 < 40:
            return 0.7
        return 1.0

    # Keep the old name as an alias so external callers (e.g. tests that
    # directly invoke _residential_area_score) don't break immediately.
    # Deprecated — use _area_plausibility_score instead.
    def _residential_area_score(self, area_m2: float) -> float:  # pragma: no cover
        return self._area_plausibility_score(area_m2, parcel_available=False)

    def _all_source_features(self) -> list[dict[str, Any]]:
        features: list[dict[str, Any]] = []
        for source_path in self.paths:
            if not self._file_exists(source_path):
                continue
            try:
                source_rank = self.paths.index(source_path)
            except ValueError:
                source_rank = 999
            for row in self._load_features(source_path):
                geom = self._primary_polygon(row["geometry"])
                features.append(
                    {
                        "geometry": geom,
                        "source": source_path,
                        "source_rank": source_rank,
                        "properties": dict(row.get("properties") or {}),
                        "structure_id": row.get("structure_id"),
                    }
                )
        return features

    def _national_index_features(self, lat: float, lon: float) -> list[dict[str, Any]]:
        """Fetch features from the national index and convert to internal format.

        Returns a list in the same format as ``_all_source_features`` so the
        matching logic can process them identically to local features.  Returns
        an empty list when the national index is unavailable or returns nothing.
        """
        if self._national_index is None or not self._national_index.enabled:
            return []
        if shape is None:
            return []

        try:
            raw_features = self._national_index.get_footprints_near_point(
                lat, lon, radius_m=300.0
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "building_footprints national_index_error lat=%.5f lon=%.5f error=%s",
                lat, lon, exc,
            )
            return []

        if not raw_features:
            return []

        result: list[dict[str, Any]] = []
        for feat in raw_features:
            geometry_raw = feat.get("geometry")
            if not isinstance(geometry_raw, dict):
                continue
            try:
                geom = self._primary_polygon(shape(geometry_raw))
            except Exception:
                continue
            if geom.is_empty:
                continue
            props = dict(feat.get("properties") or {})
            result.append(
                {
                    "geometry": geom,
                    "source": "national_index_overture",
                    "source_rank": 999,
                    "properties": props,
                    "structure_id": None,
                }
            )
        return result

    def _candidate_summary(
        self,
        *,
        row: dict[str, Any],
        point_wgs84: Any,
        to_3857: Any,
    ) -> dict[str, Any]:
        geom = row["geometry"]
        source = str(row.get("source") or "")
        geom_m = shapely_transform(to_3857, geom)
        point_m = shapely_transform(to_3857, point_wgs84)
        distance_m = float(max(0.0, geom_m.distance(point_m)))
        centroid_distance_m = float(max(0.0, geom_m.centroid.distance(point_m)))
        area_m2 = self._geom_area_m2(geom)
        area_score = self._area_plausibility_score(area_m2, parcel_available=False)
        contains_point = bool(getattr(geom, "covers", lambda _p: False)(point_wgs84))
        centroid = geom.centroid
        return {
            "structure_id": row.get("structure_id"),
            "source": source,
            "distance_m": round(distance_m, 2),
            "centroid_distance_m": round(centroid_distance_m, 2),
            "area_m2": round(area_m2, 2),
            "area_score": round(area_score, 3),
            "candidate_score": round(float(row.get("score") or 0.0), 4),
            "contains_point": contains_point,
            "centroid_latitude": round(float(centroid.y), 7),
            "centroid_longitude": round(float(centroid.x), 7),
        }

    @staticmethod
    def _inside_polygon(geom: Any, point_wgs84: Any) -> bool:
        try:
            return bool(getattr(geom, "covers", lambda _p: False)(point_wgs84))
        except Exception:
            return False

    def _count_structures_on_parcel(
        self,
        all_features: list[dict[str, Any]],
        parcel_polygon: Any,
    ) -> int:
        """Return the number of feature geometries that intersect the parcel."""
        count = 0
        for feat in all_features:
            geom = feat.get("geometry") or feat.get("geom")
            if geom is None:
                continue
            try:
                if bool(geom.intersects(parcel_polygon)):
                    count += 1
            except Exception:
                continue
        return count

    def _multiple_structures_label(self, count: int) -> str:
        """Convert an on-parcel structure count to a classification label."""
        if count == 1:
            return "single_structure"
        if count <= 3:
            return "multiple_structures"
        return "complex_property"

    def _try_national_index_match(
        self,
        lat: float,
        lon: float,
        parcel_polygon: Any | None = None,
        anchor_precision: str | None = None,
    ) -> BuildingFootprintResult | None:
        """Attempt to match a building footprint from the national Overture index.

        Runs point-in-polygon then nearest-within-tolerance matching against
        features fetched from the national index.  Returns a
        :class:`BuildingFootprintResult` with confidence 0.88 (point-in-polygon
        match) or 0.82 (nearest match) — slightly lower than local-file matches
        (0.97 / 0.92) to reflect the absence of address-point cross-validation.

        Returns ``None`` when the national index is unavailable, returns no
        features, or no candidate falls within the match tolerance.
        """
        if self._national_index is None or not self._national_index.enabled:
            return None
        if not self._geo_ready():
            return None

        nat_features = self._national_index_features(lat, lon)
        if not nat_features:
            return None

        point_wgs84 = Point(lon, lat)
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        parcel_overlap_available = (
            parcel_polygon is not None
            and not getattr(parcel_polygon, "is_empty", True)
        )
        assumptions = ["Building footprint matched from national Overture index."]

        search_features = list(nat_features)
        if parcel_overlap_available:
            parcel_intersecting = [
                row
                for row in nat_features
                if bool(
                    getattr(row["geometry"], "intersects", lambda _p: False)(parcel_polygon)
                )
            ]
            if parcel_intersecting:
                search_features = parcel_intersecting
                assumptions.append(
                    "Structure matching prioritized footprints intersecting the matched parcel."
                )

        # Phase 1: point-in-polygon.
        containing = [
            row
            for row in search_features
            if self._inside_polygon(row["geometry"], point_wgs84)
        ]
        if containing:
            top_row = containing[0]
            geom = top_row["geometry"]
            c = geom.centroid

            on_parcel_count: int | None = None
            if parcel_overlap_available:
                on_parcel_count = self._count_structures_on_parcel(
                    nat_features, parcel_polygon
                )

            LOGGER.info(
                "building_footprints national_index_match"
                " method=point_in_footprint lat=%.5f lon=%.5f confidence=0.88",
                lat, lon,
            )
            return BuildingFootprintResult(
                found=True,
                footprint=geom,
                centroid=(float(c.y), float(c.x)),
                source="national_index_overture",
                confidence=0.88,
                match_status="matched",
                match_method="point_in_footprint",
                matched_structure_id=None,
                match_distance_m=0.0,
                candidate_count=len(containing),
                candidate_summaries=[],
                assumptions=assumptions,
                feature_properties=dict(top_row.get("properties") or {}),
                on_parcel_structure_count=on_parcel_count,
            )

        # Phase 2: nearest within tolerance.
        point_m = shapely_transform(to_3857, point_wgs84)
        best_row: dict[str, Any] | None = None
        best_dist = float("inf")
        for row in search_features:
            try:
                geom_m = shapely_transform(to_3857, row["geometry"])
                dist = float(max(0.0, geom_m.distance(point_m)))
            except Exception:
                continue
            if dist < best_dist:
                best_dist = dist
                best_row = row

        if best_row is None or best_dist > self.max_match_distance_m:
            return None

        nearest_geom = best_row["geometry"]
        c = nearest_geom.centroid

        on_parcel_count = None
        if parcel_overlap_available:
            on_parcel_count = self._count_structures_on_parcel(
                nat_features, parcel_polygon
            )

        LOGGER.info(
            "building_footprints national_index_match"
            " method=nearest lat=%.5f lon=%.5f dist_m=%.1f confidence=0.82",
            lat, lon, best_dist,
        )
        return BuildingFootprintResult(
            found=True,
            footprint=nearest_geom,
            centroid=(float(c.y), float(c.x)),
            source="national_index_overture",
            confidence=0.82,
            match_status="matched",
            match_method="nearest_building_fallback",
            matched_structure_id=None,
            match_distance_m=round(best_dist, 2),
            candidate_count=1,
            candidate_summaries=[],
            assumptions=assumptions,
            feature_properties=dict(best_row.get("properties") or {}),
            on_parcel_structure_count=on_parcel_count,
        )

    def get_building_footprint(
        self,
        lat: float,
        lon: float,
        *,
        parcel_polygon: Any | None = None,
        anchor_precision: str | None = None,
    ) -> BuildingFootprintResult:
        assumptions: list[str] = []
        configured_paths = [p for p in self.paths if p]
        if not configured_paths or not any(self._file_exists(p) for p in configured_paths):
            assumptions.append("Building footprint source is not configured or missing.")
            # Try national index before giving up.
            nat_result = self._try_national_index_match(
                lat, lon, parcel_polygon=parcel_polygon, anchor_precision=anchor_precision
            )
            if nat_result is not None:
                nat_result.assumptions = assumptions + nat_result.assumptions
                return nat_result
            return BuildingFootprintResult(
                found=False, match_status="provider_unavailable", assumptions=assumptions
            )

        if not self._geo_ready():
            assumptions.append("Building footprint analysis unavailable; geospatial dependencies missing.")
            return BuildingFootprintResult(found=False, match_status="provider_unavailable", assumptions=assumptions)

        features = self._all_source_features()
        if not features:
            assumptions.append("No building footprints available in configured source(s).")
            # Try national index before giving up.
            nat_result = self._try_national_index_match(
                lat, lon, parcel_polygon=parcel_polygon, anchor_precision=anchor_precision
            )
            if nat_result is not None:
                nat_result.assumptions = assumptions + nat_result.assumptions
                return nat_result
            return BuildingFootprintResult(
                found=False,
                source=self.path,
                match_status="provider_unavailable",
                assumptions=assumptions,
            )

        point_wgs84 = Point(lon, lat)
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

        parcel_overlap_available = parcel_polygon is not None and not getattr(parcel_polygon, "is_empty", True)
        search_features = list(features)
        if parcel_overlap_available:
            parcel_intersecting = [
                row
                for row in features
                if bool(getattr(row["geometry"], "intersects", lambda _p: False)(parcel_polygon))
            ]
            if not parcel_intersecting:
                assumptions.append(
                    "Matched parcel contained no intersecting building footprints; using point proxy instead of off-parcel nearest fallback."
                )
                return BuildingFootprintResult(
                    found=False,
                    source=self.path,
                    match_status="none",
                    match_method="parcel_intersection",
                    candidate_count=0,
                    assumptions=assumptions,
                )
            search_features = parcel_intersecting
            assumptions.append("Structure matching prioritized footprints intersecting the matched parcel.")

        containing = [row for row in search_features if self._inside_polygon(row["geometry"], point_wgs84)]
        if containing:
            # Deterministically pick best containing polygon.
            def _contain_score(item: dict[str, Any]) -> tuple[float, float]:
                geom = item["geometry"]
                parcel_intersection_area_m2 = 0.0
                if parcel_overlap_available:
                    try:
                        parcel_intersection_area_m2 = self._geom_area_m2(geom.intersection(parcel_polygon))
                    except Exception:
                        parcel_intersection_area_m2 = 0.0
                area_m2 = self._geom_area_m2(geom)
                area_score = (
                    self._area_plausibility_score(area_m2, parcel_available=parcel_overlap_available)
                    + min(1.0, area_m2 / 10_000.0) * 0.05
                )
                return parcel_intersection_area_m2, area_score

            ranked = sorted(
                containing,
                key=lambda row: (
                    -_contain_score(row)[0],
                    -_contain_score(row)[1],
                    int(row.get("source_rank", 999)),
                ),
            )
            top_row = ranked[0]
            geom = top_row["geometry"]
            source = str(top_row.get("source") or self.path)
            match_method = "parcel_intersection" if parcel_overlap_available else "point_in_footprint"
            base_confidence = 0.97 if match_method == "parcel_intersection" else 0.92
            candidate_summaries = [
                self._candidate_summary(
                    row={
                        **row,
                        "score": _contain_score(row)[1],
                    },
                    point_wgs84=point_wgs84,
                    to_3857=to_3857,
                )
                for row in ranked[: self.max_candidate_summaries]
            ]
            if len(ranked) > 1:
                top_area, top_score = _contain_score(ranked[0])
                second_area, second_score = _contain_score(ranked[1])
                top_geom_m = shapely_transform(to_3857, ranked[0]["geometry"])
                second_geom_m = shapely_transform(to_3857, ranked[1]["geometry"])
                centroid_gap_m = float(max(0.0, top_geom_m.centroid.distance(second_geom_m.centroid)))
                if (
                    abs(top_area - second_area) <= 12.0
                    and (top_score - second_score) < 0.02
                    and centroid_gap_m <= 2.0
                ):
                    assumptions.append("Multiple overlapping structure footprints were equally plausible; using geocoded point fallback.")
                    return BuildingFootprintResult(
                        found=False,
                        source=source,
                        confidence=0.0,
                        match_status="ambiguous",
                        match_method=match_method,
                        matched_structure_id=None,
                        match_distance_m=0.0,
                        candidate_count=len(ranked),
                        candidate_summaries=candidate_summaries,
                        assumptions=assumptions,
                    )

            # Count on-parcel structures when parcel is available.
            on_parcel_count: int | None = None
            if parcel_overlap_available:
                on_parcel_count = self._count_structures_on_parcel(features, parcel_polygon)

            c = geom.centroid
            return BuildingFootprintResult(
                found=True,
                footprint=geom,
                centroid=(float(c.y), float(c.x)),
                source=source,
                confidence=base_confidence,
                match_status="matched",
                match_method=match_method,
                matched_structure_id=top_row.get("structure_id"),
                match_distance_m=0.0,
                candidate_count=len(ranked),
                candidate_summaries=candidate_summaries,
                assumptions=assumptions,
                feature_properties=dict(top_row.get("properties") or {}),
                on_parcel_structure_count=on_parcel_count,
            )

        point_m = shapely_transform(to_3857, point_wgs84)

        candidates: list[dict[str, Any]] = []
        parcel_centroid_m = None
        if parcel_overlap_available:
            try:
                parcel_centroid_m = shapely_transform(to_3857, parcel_polygon).centroid
            except Exception:
                parcel_centroid_m = None
        for candidate in search_features:
            candidate_geom = candidate["geometry"]
            source = str(candidate.get("source") or "")
            candidate_m = shapely_transform(to_3857, candidate_geom)
            distance = float(max(0.0, candidate_m.distance(point_m)))
            if distance > float(self.max_search_m):
                continue
            centroid_distance = float(max(0.0, candidate_m.centroid.distance(point_m)))
            footprint_area_m2 = self._geom_area_m2(candidate_geom)
            area_score = self._area_plausibility_score(
                footprint_area_m2, parcel_available=parcel_overlap_available
            )
            parcel_overlap = False
            parcel_intersection_area_m2 = 0.0
            parcel_centroid_distance_m = centroid_distance
            if parcel_overlap_available:
                try:
                    parcel_overlap = bool(candidate_geom.intersects(parcel_polygon))
                    if parcel_overlap:
                        parcel_intersection = candidate_geom.intersection(parcel_polygon)
                        parcel_intersection_area_m2 = self._geom_area_m2(parcel_intersection)
                    if parcel_centroid_m is not None:
                        parcel_centroid_distance_m = float(
                            max(0.0, candidate_m.centroid.distance(parcel_centroid_m))
                        )
                except Exception:
                    parcel_overlap = False
            score = (
                max(0.0, 1.0 - (distance / max(self.max_match_distance_m, 1.0))) * 0.75
                + max(0.0, 1.0 - (centroid_distance / max(self.max_search_m, 1.0))) * 0.15
                + area_score * 0.10
                + (0.12 if parcel_overlap else 0.0)
                + max(0.0, 0.06 - (0.015 * float(candidate.get("source_rank", 0))))
            )
            candidates.append(
                {
                    "geom": candidate_geom,
                    "source": source,
                    "source_rank": int(candidate.get("source_rank", 999)),
                    "structure_id": candidate.get("structure_id"),
                    "properties": dict(candidate.get("properties") or {}),
                    "distance_m": distance,
                    "centroid_distance_m": centroid_distance,
                    "area_score": area_score,
                    "footprint_area_m2": footprint_area_m2,
                    "parcel_overlap": parcel_overlap,
                    "parcel_intersection_area_m2": parcel_intersection_area_m2,
                    "parcel_centroid_distance_m": parcel_centroid_distance_m,
                    "score": score,
                }
            )

        if not candidates:
            assumptions.append("No nearby building footprint found for this location.")
            return BuildingFootprintResult(
                found=False,
                source=self.path,
                match_status="none",
                match_method="parcel_intersection" if parcel_overlap_available else "nearest_building_fallback",
                candidate_count=0,
                assumptions=assumptions,
            )

        match_method = "nearest_building_fallback"
        if parcel_overlap_available:
            parcel_candidates = [row for row in candidates if row.get("parcel_overlap")]
            if not parcel_candidates:
                assumptions.append(
                    "Matched parcel had no intersecting footprint candidates within search distance; using point proxy."
                )
                return BuildingFootprintResult(
                    found=False,
                    source=self.path,
                    match_status="none",
                    match_method="parcel_intersection",
                    candidate_count=len(candidates),
                    assumptions=assumptions,
                )
            candidates = parcel_candidates
            assumptions.append("Structure matching prioritized candidates intersecting the matched parcel.")
            match_method = "parcel_intersection"

        if match_method == "parcel_intersection":
            candidates.sort(
                key=lambda row: (
                    -float(row.get("parcel_intersection_area_m2") or 0.0),
                    float(row.get("parcel_centroid_distance_m") or 0.0),
                    -float(row.get("footprint_area_m2") or 0.0),
                    float(row.get("distance_m") or 0.0),
                    int(row.get("source_rank", 999)),
                )
            )
        else:
            candidates.sort(
                key=lambda row: (
                    row["distance_m"],
                    row["centroid_distance_m"],
                    int(row.get("source_rank", 999)),
                    -row["score"],
                )
            )
        top = candidates[0]
        top_distance = float(top["distance_m"])
        normalized_precision = str(anchor_precision or "unknown").strip().lower()
        if match_method == "parcel_intersection":
            effective_max_match_distance_m = max(self.max_match_distance_m * 2.5, 60.0)
        elif normalized_precision == "interpolated":
            effective_max_match_distance_m = min(
                float(self.max_search_m),
                max(self.max_match_distance_m + 8.0, 22.0),
            )
        elif normalized_precision == "approximate":
            effective_max_match_distance_m = min(
                float(self.max_search_m),
                max(self.max_match_distance_m + 10.0, 24.0),
            )
        elif normalized_precision == "unknown":
            effective_max_match_distance_m = min(
                float(self.max_search_m),
                max(self.max_match_distance_m + 6.0, 20.0),
            )
        else:
            effective_max_match_distance_m = self.max_match_distance_m
        if match_method != "parcel_intersection" and effective_max_match_distance_m > self.max_match_distance_m:
            assumptions.append(
                f"Expanded structure-match distance to {effective_max_match_distance_m:.1f} m for low-precision anchor handling."
            )
        candidate_summaries = [
            self._candidate_summary(
                row={
                    "geometry": row["geom"],
                    "source": row["source"],
                    "structure_id": row.get("structure_id"),
                },
                point_wgs84=point_wgs84,
                to_3857=to_3857,
            )
            for row in candidates[: self.max_candidate_summaries]
        ]

        if match_method != "parcel_intersection" and top_distance > effective_max_match_distance_m:
            assumptions.append(
                "Nearest structure footprint is too far from the geocoded point; using geocoded point fallback."
            )
            return BuildingFootprintResult(
                found=False,
                source=str(top.get("source") or self.path),
                confidence=0.0,
                match_status="none",
                match_method=match_method,
                matched_structure_id=None,
                match_distance_m=round(top_distance, 2),
                candidate_count=len(candidates),
                candidate_summaries=candidate_summaries,
                assumptions=assumptions,
            )

        if len(candidates) > 1:
            if match_method == "parcel_intersection":
                top_area = float(top.get("parcel_intersection_area_m2") or 0.0)
                second = candidates[1]
                second_area = float(second.get("parcel_intersection_area_m2") or 0.0)
                top_centroid_d = float(top.get("parcel_centroid_distance_m") or 0.0)
                second_centroid_d = float(second.get("parcel_centroid_distance_m") or 0.0)
                if abs(top_area - second_area) <= 15.0 and abs(top_centroid_d - second_centroid_d) <= 4.0:
                    assumptions.append(
                        "Multiple parcel-intersecting structures were similarly plausible; using geocoded point fallback."
                    )
                    return BuildingFootprintResult(
                        found=False,
                        source=str(top.get("source") or self.path),
                        confidence=0.0,
                        match_status="ambiguous",
                        match_method=match_method,
                        matched_structure_id=None,
                        match_distance_m=round(top_distance, 2),
                        candidate_count=len(candidates),
                        candidate_summaries=candidate_summaries,
                        assumptions=assumptions,
                    )
            else:
                second_distance = float(candidates[1]["distance_m"])
                score_gap = float(top.get("score") or 0.0) - float(candidates[1].get("score") or 0.0)
                area_gap = abs(float(top.get("area_score") or 0.0) - float(candidates[1].get("area_score") or 0.0))
                if (second_distance - top_distance) <= float(self.ambiguity_gap_m) and score_gap < 0.08 and area_gap < 0.18:
                    assumptions.append(
                        "Multiple nearby structures were similarly plausible; using geocoded point fallback."
                    )
                    return BuildingFootprintResult(
                        found=False,
                        source=str(top.get("source") or self.path),
                        confidence=0.0,
                        match_status="ambiguous",
                        match_method=match_method,
                        matched_structure_id=None,
                        match_distance_m=round(top_distance, 2),
                        candidate_count=len(candidates),
                        candidate_summaries=candidate_summaries,
                        assumptions=assumptions,
                    )

        distance_component = max(0.0, 1.0 - (top_distance / max(effective_max_match_distance_m, 1.0)))
        if match_method == "parcel_intersection":
            confidence = max(0.65, min(0.96, 0.62 + (distance_component * 0.22) + (float(top["area_score"]) * 0.12)))
        else:
            confidence = max(0.4, min(0.78, 0.4 + (distance_component * 0.28) + (float(top["area_score"]) * 0.1)))
            if normalized_precision in {"interpolated", "approximate"}:
                confidence = min(confidence, 0.68)
            if normalized_precision == "unknown":
                confidence = min(confidence, 0.64)

        # Count on-parcel structures when parcel is available.
        on_parcel_count = None
        if parcel_overlap_available:
            on_parcel_count = self._count_structures_on_parcel(features, parcel_polygon)

        nearest_geom = top["geom"]
        nearest_source = top["source"]
        c = nearest_geom.centroid
        return BuildingFootprintResult(
            found=True,
            footprint=nearest_geom,
            centroid=(float(c.y), float(c.x)),
            source=nearest_source or self.path,
            confidence=round(confidence, 2),
            match_status="matched",
            match_method=match_method,
            matched_structure_id=top.get("structure_id"),
            match_distance_m=round(top_distance, 2),
            candidate_count=len(candidates),
            candidate_summaries=candidate_summaries,
            assumptions=assumptions,
            feature_properties=dict(top.get("properties") or {}),
            on_parcel_structure_count=on_parcel_count,
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
            geoms = [self._primary_polygon(row["geometry"]) for row in self._load_features(path_to_use)]
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
        try:
            subject_m = shapely_transform(to_3857, subject_footprint) if subject_footprint is not None else None
        except Exception:
            subject_m = None

        nearby_100 = 0
        nearby_300 = 0
        nearest_m: float | None = None
        r100 = 100.0 * FEET_TO_METERS
        r300 = min(radius_m, 300.0 * FEET_TO_METERS)

        for geom in geoms:
            try:
                geom_m = shapely_transform(to_3857, geom)
            except Exception:
                continue
            if subject_m is not None and geom_m.distance(subject_m) < 0.5:
                continue

            # When a subject footprint is available, use footprint-edge proximity
            # so nearby homes with different placement get distinct metrics.
            if subject_m is not None:
                d = float(subject_m.distance(geom_m))
            else:
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


def compute_structure_rings(
    footprint: Any,
    parcel_polygon: Any | None = None,
) -> tuple[dict[str, Any], list[str]]:
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

    rings_m: dict[str, Any] = {
        "ring_0_5_ft": b5_m.difference(footprint_m),
        "ring_5_30_ft": b30_m.difference(b5_m),
        "ring_30_100_ft": b100_m.difference(b30_m),
        "ring_100_300_ft": b300_m.difference(b100_m),
    }

    if parcel_polygon is not None:
        try:
            parcel_m = shapely_transform(to_3857, parcel_polygon)
            clipped: dict[str, Any] = {}
            for key, ring in rings_m.items():
                if ring.is_empty:
                    continue
                intersection = ring.intersection(parcel_m)
                if intersection is None or intersection.is_empty:
                    assumptions.append(f"Ring {key} lies entirely outside the parcel boundary after clipping.")
                    continue
                clipped[key] = intersection
            rings_m = clipped
        except Exception:
            assumptions.append("Parcel clipping failed; using unclipped ring geometries.")

    rings_wgs84: dict[str, Any] = {}
    for key, ring in rings_m.items():
        if ring.is_empty:
            continue
        rings_wgs84[key] = shapely_transform(to_wgs84, ring)

    if parcel_polygon is None and len(rings_wgs84) != len(RING_KEYS):
        assumptions.append("Some structure rings could not be generated from footprint geometry.")

    return rings_wgs84, assumptions


def compute_footprint_geometry_signals(
    footprint: Any,
    parcel_polygon: Any | None = None,
    all_footprints: list[Any] | None = None,
) -> dict[str, float | str | None]:
    """Derive geometry signals from the structure footprint and optionally the parcel polygon.

    Returns a dict with the following keys (all None when geometry is unavailable):
      footprint_perimeter_m          -- exterior perimeter in metres (EPSG:3857)
      footprint_compactness_ratio    -- 4π·area / perimeter² (1.0 = circle, lower = irregular)
      footprint_long_axis_bearing_deg -- bearing (0–360°, clockwise from north) of the longer
                                         bounding-box axis; 0° means the structure runs N–S
      parcel_coverage_ratio          -- footprint area / parcel area (0–1); None when no parcel
      multiple_structures_on_parcel  -- classification of on-parcel structure count when a parcel
                                         and footprint list are available; ``"unknown"`` otherwise.

    Parameters
    ----------
    footprint:
        The primary structure geometry (Shapely).
    parcel_polygon:
        Matched parcel boundary, or ``None``.
    all_footprints:
        Optional list of Shapely geometries for all nearby structures.  When
        provided together with ``parcel_polygon``, populates the
        ``multiple_structures_on_parcel`` field with a data-derived value
        (``"single_structure"``, ``"multiple_structures"``, or
        ``"complex_property"``).
    """
    if not (Transformer and shapely_transform):
        return {
            "footprint_perimeter_m": None,
            "footprint_compactness_ratio": None,
            "footprint_long_axis_bearing_deg": None,
            "parcel_coverage_ratio": None,
            "multiple_structures_on_parcel": "unknown",
        }

    import math as _math

    try:
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        fp_m = shapely_transform(to_3857, footprint)

        area_m2 = float(fp_m.area)
        perimeter_m = float(fp_m.exterior.length)
        footprint_perimeter_m = round(perimeter_m, 2) if perimeter_m > 0 else None
        footprint_compactness_ratio = (
            round(min(1.0, (4.0 * _math.pi * area_m2) / (perimeter_m * perimeter_m)), 4)
            if perimeter_m > 0 and area_m2 > 0
            else None
        )

        # Long-axis bearing from the minimum rotated rectangle (MRR).
        footprint_long_axis_bearing_deg = None
        try:
            mrr = fp_m.minimum_rotated_rectangle
            if mrr is not None and not getattr(mrr, "is_empty", True):
                coords = list(mrr.exterior.coords)
                if len(coords) >= 4:
                    s1_dx = coords[1][0] - coords[0][0]
                    s1_dy = coords[1][1] - coords[0][1]
                    s2_dx = coords[2][0] - coords[1][0]
                    s2_dy = coords[2][1] - coords[1][1]
                    len1 = _math.hypot(s1_dx, s1_dy)
                    len2 = _math.hypot(s2_dx, s2_dy)
                    axis_dx, axis_dy = (s1_dx, s1_dy) if len1 >= len2 else (s2_dx, s2_dy)
                    if _math.hypot(axis_dx, axis_dy) > 0:
                        bearing = _math.degrees(_math.atan2(axis_dx, axis_dy)) % 180.0
                        footprint_long_axis_bearing_deg = round(bearing, 1)
        except Exception:
            footprint_long_axis_bearing_deg = None

        parcel_coverage_ratio: float | None = None
        if parcel_polygon is not None:
            try:
                parcel_m = shapely_transform(to_3857, parcel_polygon)
                parcel_area = float(parcel_m.area)
                if parcel_area > 0:
                    parcel_coverage_ratio = round(
                        min(1.0, area_m2 / parcel_area), 4
                    )
            except Exception:
                parcel_coverage_ratio = None

    except Exception:
        return {
            "footprint_perimeter_m": None,
            "footprint_compactness_ratio": None,
            "footprint_long_axis_bearing_deg": None,
            "parcel_coverage_ratio": None,
            "multiple_structures_on_parcel": "unknown",
        }

    # Determine multiple_structures_on_parcel when both parcel and footprint
    # list are available.
    multiple_structures_on_parcel: str = "unknown"
    if parcel_polygon is not None and all_footprints is not None:
        try:
            count = 0
            for fp in all_footprints:
                if fp is None:
                    continue
                if bool(parcel_polygon.intersects(fp)):
                    count += 1
            if count == 1:
                multiple_structures_on_parcel = "single_structure"
            elif count <= 3:
                multiple_structures_on_parcel = "multiple_structures"
            elif count > 3:
                multiple_structures_on_parcel = "complex_property"
        except Exception:
            multiple_structures_on_parcel = "unknown"

    return {
        "footprint_perimeter_m": footprint_perimeter_m,
        "footprint_compactness_ratio": footprint_compactness_ratio,
        "footprint_long_axis_bearing_deg": footprint_long_axis_bearing_deg,
        "parcel_coverage_ratio": parcel_coverage_ratio,
        "multiple_structures_on_parcel": multiple_structures_on_parcel,
    }
