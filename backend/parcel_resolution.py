from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from pyproj import Transformer
    from shapely.geometry import Point, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - optional geospatial runtime deps
    Transformer = None
    Point = None
    mapping = None
    shape = None
    shapely_transform = None


DEFAULT_SOURCE_PRIORITY = ("county_gis", "open_parcel", "prepared_region")


@dataclass
class ParcelResolutionResult:
    status: str = "not_found"
    confidence: float = 0.0  # 0-100
    source: str | None = None
    geometry_used: str = "none"
    overlap_score: float = 0.0  # 0-100
    parcel_id: str | None = None
    parcel_polygon: Any | None = None
    parcel_feature: dict[str, Any] | None = None
    parcel_lookup_method: str | None = None
    parcel_lookup_distance_m: float | None = None
    source_name: str | None = None
    source_vintage: str | None = None
    candidate_count: int = 0
    candidate_summaries: list[dict[str, Any]] = field(default_factory=list)
    approximation_geometry_geojson: dict[str, Any] | None = None
    diagnostics: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        summary = {
            "status": str(self.status or "not_found"),
            "confidence": round(max(0.0, min(100.0, float(self.confidence))), 1),
            "source": self.source,
            "geometry_used": str(self.geometry_used or "none"),
            "overlap_score": round(max(0.0, min(100.0, float(self.overlap_score))), 1),
            "candidates_considered": max(0, int(self.candidate_count)),
            "lookup_method": self.parcel_lookup_method,
            "lookup_distance_m": (
                round(float(self.parcel_lookup_distance_m), 2)
                if self.parcel_lookup_distance_m is not None
                else None
            ),
        }
        if self.approximation_geometry_geojson is not None:
            summary["bounding_geometry"] = self.approximation_geometry_geojson
        if self.candidate_summaries:
            summary["candidate_summaries"] = self.candidate_summaries[:5]
        return summary


class ParcelResolutionClient:
    def __init__(
        self,
        *,
        parcel_paths: list[str] | None = None,
        max_lookup_distance_m: float = 30.0,
    ) -> None:
        self.parcel_paths = self._normalize_paths(parcel_paths or [])
        self.max_lookup_distance_m = self._env_float(
            "WF_PARCEL_LOOKUP_MAX_DISTANCE_M",
            float(max_lookup_distance_m),
            min_value=1.0,
        )
        self.max_candidate_summaries = int(
            max(
                1,
                min(
                    8,
                    round(
                        self._env_float(
                            "WF_PARCEL_MATCH_MAX_CANDIDATE_SUMMARIES",
                            4.0,
                            min_value=1.0,
                        )
                    ),
                ),
            )
        )
        self.fallback_half_size_m = self._env_float(
            "WF_PARCEL_BOUNDING_APPROX_HALF_SIZE_M",
            25.0,
            min_value=5.0,
        )
        self.source_priority = self._resolve_source_priority()

    @staticmethod
    def _geo_ready() -> bool:
        return bool(Transformer and Point and shape and mapping and shapely_transform)

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
    def _normalize_paths(paths: list[str]) -> list[str]:
        ordered: list[str] = []
        for candidate in paths:
            raw = str(candidate or "").strip()
            if not raw:
                continue
            try:
                normalized = str(Path(raw).expanduser().resolve())
            except Exception:
                normalized = raw
            if normalized not in ordered:
                ordered.append(normalized)
        return ordered

    @staticmethod
    def _resolve_source_priority() -> tuple[str, ...]:
        raw = str(
            os.getenv(
                "WF_PARCEL_SOURCE_PRIORITY",
                ",".join(DEFAULT_SOURCE_PRIORITY),
            )
        ).strip()
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
        ordered: list[str] = []
        for token in tokens:
            if token in {"county_gis", "open_parcel", "prepared_region"} and token not in ordered:
                ordered.append(token)
        for token in DEFAULT_SOURCE_PRIORITY:
            if token not in ordered:
                ordered.append(token)
        return tuple(ordered)

    def _source_rank(self, source_class: str) -> int:
        try:
            return self.source_priority.index(source_class)
        except ValueError:
            return len(self.source_priority) + 1

    @staticmethod
    def _extract_parcel_id(props: dict[str, Any]) -> str | None:
        for key in (
            "parcel_id",
            "parcelid",
            "PARCEL_ID",
            "apn",
            "APN",
            "parcel",
            "parcel_number",
        ):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _extract_source_name(props: dict[str, Any], fallback: str) -> str:
        for key in ("source_name", "source", "dataset", "provider"):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    @staticmethod
    def _extract_source_vintage(props: dict[str, Any]) -> str | None:
        for key in ("source_vintage", "vintage", "year", "dataset_year"):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _classify_source(path: str, props: dict[str, Any]) -> str:
        tokens = " ".join(
            [
                str(path or ""),
                str(props.get("source_name") or ""),
                str(props.get("source_type") or ""),
                str(props.get("dataset") or ""),
                str(props.get("provider") or ""),
            ]
        ).lower()
        if "county" in tokens:
            return "county_gis"
        if any(token in tokens for token in ("open", "osm", "overture")):
            return "open_parcel"
        return "prepared_region"

    @staticmethod
    def _distance_m(geom_a: Any, geom_b: Any) -> float:
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        a_m = shapely_transform(to_3857, geom_a)
        b_m = shapely_transform(to_3857, geom_b)
        return float(max(0.0, a_m.distance(b_m)))

    @staticmethod
    def _area_m2(geom: Any) -> float:
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        g_m = shapely_transform(to_3857, geom)
        return float(max(0.0, g_m.area))

    @staticmethod
    def _centroid_distance_m(point: Any, geom: Any) -> float:
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        p_m = shapely_transform(to_3857, point)
        c_m = shapely_transform(to_3857, geom).centroid
        return float(max(0.0, p_m.distance(c_m)))

    @lru_cache(maxsize=12)
    def _load_geojson_features(self, path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        payload = json.loads(p.read_text(encoding="utf-8"))
        features = payload.get("features", []) if isinstance(payload, dict) else []
        rows: list[dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                continue
            gtype = str(geometry.get("type") or "")
            if gtype not in {"Polygon", "MultiPolygon"}:
                continue
            try:
                geom = shape(geometry)
            except Exception:
                continue
            if geom.is_empty:
                continue
            rows.append(feature)
        return rows

    def _build_bounding_approximation(self, anchor_point: Any) -> dict[str, Any] | None:
        if not self._geo_ready():
            return None
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
        point_m = shapely_transform(to_3857, anchor_point)
        approx_m = point_m.buffer(float(self.fallback_half_size_m), cap_style=3)
        approx_wgs84 = shapely_transform(to_wgs84, approx_m)
        return {
            "type": "Feature",
            "properties": {
                "source": "bounding_approximation",
                "half_size_m": round(float(self.fallback_half_size_m), 2),
            },
            "geometry": mapping(approx_wgs84),
        }

    @staticmethod
    def _is_ambiguous(top: dict[str, Any], second: dict[str, Any], *, containing: bool) -> bool:
        if int(top.get("source_rank", 999)) != int(second.get("source_rank", 999)):
            return False
        if containing:
            area_a = float(top.get("area_m2") or 0.0)
            area_b = float(second.get("area_m2") or 0.0)
            bigger = max(area_a, area_b, 1.0)
            smaller = max(min(area_a, area_b), 1.0)
            area_ratio = bigger / smaller
            centroid_gap = abs(
                float(top.get("centroid_distance_m") or 0.0)
                - float(second.get("centroid_distance_m") or 0.0)
            )
            return area_ratio <= 1.2 and centroid_gap <= 2.0
        distance_gap = abs(
            float(top.get("distance_m") or 0.0) - float(second.get("distance_m") or 0.0)
        )
        overlap_gap = abs(
            float(top.get("overlap_score") or 0.0) - float(second.get("overlap_score") or 0.0)
        )
        return distance_gap <= 2.5 and overlap_gap <= 6.0

    def resolve_for_point(
        self,
        *,
        anchor_point: Any,
        max_lookup_distance_m: float | None = None,
    ) -> ParcelResolutionResult:
        lookup_limit_m = float(
            max_lookup_distance_m
            if max_lookup_distance_m is not None
            else self.max_lookup_distance_m
        )
        if not self._geo_ready():
            return ParcelResolutionResult(
                status="not_found",
                confidence=0.0,
                source=None,
                geometry_used="none",
                overlap_score=0.0,
                parcel_lookup_method="none",
                diagnostics=[
                    "Parcel resolution unavailable; geospatial dependencies missing.",
                ],
            )

        available_paths = [path for path in self.parcel_paths if Path(path).exists()]
        if not available_paths:
            approximation = self._build_bounding_approximation(anchor_point)
            return ParcelResolutionResult(
                status="not_found",
                confidence=12.0,
                source=None,
                geometry_used="bounding_approximation" if approximation is not None else "none",
                overlap_score=0.0,
                parcel_lookup_method="none",
                candidate_count=0,
                approximation_geometry_geojson=approximation,
                diagnostics=[
                    "No parcel source files were available; generated a bounded parcel approximation.",
                ],
            )

        candidates: list[dict[str, Any]] = []
        for path in available_paths:
            for feature in self._load_geojson_features(path):
                geometry_raw = feature.get("geometry")
                if not isinstance(geometry_raw, dict):
                    continue
                try:
                    geom = shape(geometry_raw)
                except Exception:
                    continue
                if geom.is_empty:
                    continue
                props = dict(feature.get("properties") or {})
                contains_anchor = bool(getattr(geom, "covers", lambda _p: False)(anchor_point))
                distance_m = 0.0 if contains_anchor else self._distance_m(anchor_point, geom)
                centroid_distance_m = self._centroid_distance_m(anchor_point, geom)
                source_class = self._classify_source(path, props)
                source_rank = self._source_rank(source_class)
                area_m2 = self._area_m2(geom)
                if contains_anchor:
                    overlap_score = 100.0
                else:
                    overlap_score = max(
                        0.0,
                        100.0
                        * (1.0 - (distance_m / max(1.0, lookup_limit_m))),
                    )
                candidates.append(
                    {
                        "feature": feature,
                        "geom": geom,
                        "props": props,
                        "path": path,
                        "source_class": source_class,
                        "source_rank": source_rank,
                        "source_name": self._extract_source_name(props, Path(path).stem),
                        "source_vintage": self._extract_source_vintage(props),
                        "parcel_id": self._extract_parcel_id(props),
                        "area_m2": area_m2,
                        "contains_anchor": contains_anchor,
                        "distance_m": distance_m,
                        "centroid_distance_m": centroid_distance_m,
                        "overlap_score": overlap_score,
                    }
                )

        if not candidates:
            approximation = self._build_bounding_approximation(anchor_point)
            return ParcelResolutionResult(
                status="not_found",
                confidence=15.0,
                source=None,
                geometry_used="bounding_approximation" if approximation is not None else "none",
                overlap_score=0.0,
                parcel_lookup_method="none",
                candidate_count=0,
                approximation_geometry_geojson=approximation,
                diagnostics=[
                    "Parcel sources were configured but no polygon features were readable.",
                ],
            )

        containing = [row for row in candidates if bool(row.get("contains_anchor"))]
        if containing:
            containing.sort(
                key=lambda row: (
                    int(row.get("source_rank", 999)),
                    float(row.get("area_m2") or 0.0),
                    float(row.get("centroid_distance_m") or 0.0),
                )
            )
            top = containing[0]
            status = "matched"
            method = "contains_point"
            diagnostics: list[str] = []
            confidence = 96.0 - (5.0 * float(top.get("source_rank", 0)))
            if len(containing) > 1 and self._is_ambiguous(top, containing[1], containing=True):
                status = "multiple_candidates"
                method = "multiple_candidates"
                confidence = min(confidence, 68.0)
                diagnostics.append(
                    "Multiple containing parcel polygons were similarly plausible; selected best-ranked candidate."
                )
            confidence = max(30.0, min(100.0, confidence))
            summaries = [
                {
                    "parcel_id": row.get("parcel_id"),
                    "source": row.get("source_name"),
                    "source_class": row.get("source_class"),
                    "distance_m": round(float(row.get("distance_m") or 0.0), 2),
                    "area_m2": round(float(row.get("area_m2") or 0.0), 2),
                    "contains_anchor": bool(row.get("contains_anchor")),
                }
                for row in containing[: self.max_candidate_summaries]
            ]
            return ParcelResolutionResult(
                status=status,
                confidence=confidence,
                source=str(top.get("source_name") or top.get("source_class") or ""),
                geometry_used="parcel_polygon",
                overlap_score=float(top.get("overlap_score") or 100.0),
                parcel_id=top.get("parcel_id"),
                parcel_polygon=top.get("geom"),
                parcel_feature=top.get("feature"),
                parcel_lookup_method=method,
                parcel_lookup_distance_m=0.0,
                source_name=top.get("source_name"),
                source_vintage=top.get("source_vintage"),
                candidate_count=len(containing),
                candidate_summaries=summaries,
                diagnostics=diagnostics,
            )

        within_tolerance = [
            row
            for row in candidates
            if float(row.get("distance_m") or 0.0) <= lookup_limit_m
        ]
        if within_tolerance:
            within_tolerance.sort(
                key=lambda row: (
                    int(row.get("source_rank", 999)),
                    float(row.get("distance_m") or 0.0),
                    -float(row.get("overlap_score") or 0.0),
                    float(row.get("area_m2") or 0.0),
                )
            )
            top = within_tolerance[0]
            status = "matched"
            method = "nearest_within_tolerance"
            diagnostics = []
            overlap_score = float(top.get("overlap_score") or 0.0)
            confidence = (
                45.0
                + (0.42 * overlap_score)
                - (4.0 * float(top.get("source_rank", 0)))
            )
            if len(within_tolerance) > 1 and self._is_ambiguous(top, within_tolerance[1], containing=False):
                status = "multiple_candidates"
                method = "multiple_candidates"
                confidence = min(confidence, 62.0)
                diagnostics.append(
                    "Multiple nearby parcel candidates were similarly plausible; selected best-ranked candidate."
                )
            confidence = max(20.0, min(90.0, confidence))
            summaries = [
                {
                    "parcel_id": row.get("parcel_id"),
                    "source": row.get("source_name"),
                    "source_class": row.get("source_class"),
                    "distance_m": round(float(row.get("distance_m") or 0.0), 2),
                    "area_m2": round(float(row.get("area_m2") or 0.0), 2),
                    "contains_anchor": bool(row.get("contains_anchor")),
                }
                for row in within_tolerance[: self.max_candidate_summaries]
            ]
            return ParcelResolutionResult(
                status=status,
                confidence=confidence,
                source=str(top.get("source_name") or top.get("source_class") or ""),
                geometry_used="parcel_polygon",
                overlap_score=overlap_score,
                parcel_id=top.get("parcel_id"),
                parcel_polygon=top.get("geom"),
                parcel_feature=top.get("feature"),
                parcel_lookup_method=method,
                parcel_lookup_distance_m=float(top.get("distance_m") or 0.0),
                source_name=top.get("source_name"),
                source_vintage=top.get("source_vintage"),
                candidate_count=len(within_tolerance),
                candidate_summaries=summaries,
                diagnostics=diagnostics,
            )

        approximation = self._build_bounding_approximation(anchor_point)
        nearest_distance_m = min(float(row.get("distance_m") or 0.0) for row in candidates)
        return ParcelResolutionResult(
            status="not_found",
            confidence=18.0,
            source=None,
            geometry_used="bounding_approximation" if approximation is not None else "none",
            overlap_score=0.0,
            parcel_id=None,
            parcel_polygon=None,
            parcel_feature=None,
            parcel_lookup_method="none",
            parcel_lookup_distance_m=round(nearest_distance_m, 2),
            source_name=None,
            source_vintage=None,
            candidate_count=len(candidates),
            approximation_geometry_geojson=approximation,
            candidate_summaries=[
                {
                    "parcel_id": row.get("parcel_id"),
                    "source": row.get("source_name"),
                    "source_class": row.get("source_class"),
                    "distance_m": round(float(row.get("distance_m") or 0.0), 2),
                }
                for row in sorted(candidates, key=lambda item: float(item.get("distance_m") or 0.0))[
                    : self.max_candidate_summaries
                ]
            ],
            diagnostics=[
                "No parcel candidates were within lookup tolerance; generated a bounded parcel approximation.",
            ],
        )
