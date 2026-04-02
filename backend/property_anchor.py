from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.parcel_resolution import ParcelResolutionClient, ParcelResolutionResult

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


DEFAULT_PRIORITY = ("address_point", "parcel_centroid", "geocode")


@dataclass
class PropertyAnchorResolution:
    anchor_latitude: float
    anchor_longitude: float
    anchor_source: str
    anchor_precision: str
    geocoded_latitude: float
    geocoded_longitude: float
    geocode_provider: str | None = None
    geocode_precision: str = "unknown"
    geocoded_address: str | None = None
    parcel_id: str | None = None
    parcel_polygon: Any | None = None
    parcel_lookup_method: str | None = None
    parcel_lookup_distance_m: float | None = None
    parcel_geometry_geojson: dict[str, Any] | None = None
    parcel_address_point_geojson: dict[str, Any] | None = None
    parcel_source_name: str | None = None
    parcel_source_vintage: str | None = None
    address_point_source_name: str | None = None
    address_point_source_vintage: str | None = None
    geocode_to_anchor_distance_m: float | None = None
    anchor_selection_method: str | None = None
    anchor_quality: str = "low"
    anchor_quality_score: float = 0.0
    address_point_lookup_distance_m: float | None = None
    source_conflict_flag: bool = False
    diagnostics: list[str] = field(default_factory=list)
    alignment_notes: list[str] = field(default_factory=list)
    parcel_resolution: dict[str, Any] = field(default_factory=dict)
    parcel_bounding_approximation_geojson: dict[str, Any] | None = None
    parcel_properties: dict[str, Any] = field(default_factory=dict)
    address_point_properties: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> dict[str, Any]:
        return {
            "property_anchor_point": {
                "latitude": float(self.anchor_latitude),
                "longitude": float(self.anchor_longitude),
            },
            "property_anchor_source": self.anchor_source,
            "property_anchor_precision": self.anchor_precision,
            "geocoded_address_point": {
                "latitude": float(self.geocoded_latitude),
                "longitude": float(self.geocoded_longitude),
            },
            "geocode_provider": self.geocode_provider,
            "geocoded_address": self.geocoded_address,
            "geocode_precision": self.geocode_precision,
            "parcel_address_point": self.parcel_address_point_geojson,
            "parcel_geometry": self.parcel_geometry_geojson,
            "parcel_id": self.parcel_id,
            "parcel_lookup_method": self.parcel_lookup_method,
            "parcel_lookup_distance_m": self.parcel_lookup_distance_m,
            "parcel_source_name": self.parcel_source_name,
            "parcel_source_vintage": self.parcel_source_vintage,
            "address_point_source_name": self.address_point_source_name,
            "address_point_source_vintage": self.address_point_source_vintage,
            "geocode_to_anchor_distance_m": self.geocode_to_anchor_distance_m,
            "property_anchor_selection_method": self.anchor_selection_method,
            "property_anchor_quality": self.anchor_quality,
            "property_anchor_quality_score": self.anchor_quality_score,
            "address_point_lookup_distance_m": self.address_point_lookup_distance_m,
            "source_conflict_flag": bool(self.source_conflict_flag),
            "alignment_notes": list(self.alignment_notes),
            "parcel_resolution": dict(self.parcel_resolution or {}),
            "parcel_bounding_approximation": self.parcel_bounding_approximation_geojson,
            "parcel_properties": dict(self.parcel_properties or {}),
            "address_point_properties": dict(self.address_point_properties or {}),
        }


class PropertyAnchorResolver:
    def __init__(
        self,
        *,
        address_points_path: str | None = None,
        parcels_path: str | None = None,
        parcels_paths: list[str] | None = None,
        source_priority: tuple[str, ...] | None = None,
    ) -> None:
        self.address_points_path = str(address_points_path or "").strip()
        self.parcels_path = str(parcels_path or "").strip()
        self.parcels_paths = self._resolve_parcel_paths(
            primary_path=self.parcels_path,
            extra_paths=parcels_paths,
        )
        self.max_address_point_distance_m = self._env_float(
            "WF_PROPERTY_ANCHOR_MAX_ADDRESS_POINT_DISTANCE_M",
            45.0,
            min_value=2.0,
        )
        self.max_parcel_lookup_distance_m = self._env_float(
            "WF_PARCEL_LOOKUP_MAX_DISTANCE_M",
            30.0,
            min_value=1.0,
        )
        self.source_priority = source_priority or self._resolve_source_priority()
        self.parcel_resolver = ParcelResolutionClient(
            parcel_paths=self.parcels_paths,
            max_lookup_distance_m=self.max_parcel_lookup_distance_m,
        )

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
    def _file_exists(path: str) -> bool:
        return bool(path) and Path(path).exists()

    @staticmethod
    def _resolve_parcel_paths(*, primary_path: str, extra_paths: list[str] | None) -> list[str]:
        ordered: list[str] = []
        for candidate in [primary_path, *(extra_paths or [])]:
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
        raw = str(os.getenv("WF_PROPERTY_ANCHOR_SOURCE_PRIORITY", ",".join(DEFAULT_PRIORITY))).strip().lower()
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        ordered: list[str] = []
        for token in tokens:
            if token in {"address_point", "parcel_centroid", "geocode"} and token not in ordered:
                ordered.append(token)
        for token in DEFAULT_PRIORITY:
            if token not in ordered:
                ordered.append(token)
        return tuple(ordered)

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
    def _extract_source_name(props: dict[str, Any], fallback: str | None) -> str | None:
        for key in ("source_name", "source", "dataset", "provider"):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    @staticmethod
    def _extract_source_vintage(props: dict[str, Any], env_name: str) -> str | None:
        for key in ("source_vintage", "vintage", "year", "dataset_year"):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        raw_env = str(os.getenv(env_name, "")).strip()
        return raw_env or None

    @staticmethod
    def _to_3857_transform():
        return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    @staticmethod
    def _distance_m(point_a: Any, point_b: Any) -> float:
        to_3857 = PropertyAnchorResolver._to_3857_transform()
        a_m = shapely_transform(to_3857, point_a)
        b_m = shapely_transform(to_3857, point_b)
        return float(max(0.0, a_m.distance(b_m)))

    @staticmethod
    def _coords_from_feature(feature: dict[str, Any]) -> tuple[float, float] | None:
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geom, dict):
            return None
        if geom.get("type") != "Point":
            return None
        coords = geom.get("coordinates")
        if not isinstance(coords, list) or len(coords) < 2:
            return None
        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError):
            return None
        return lat, lon

    @staticmethod
    def _point_feature(lat: float, lon: float, *, label: str, source: str) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {
                "label": label,
                "source": source,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
        }

    @lru_cache(maxsize=8)
    def _load_geojson_features(self, path: str) -> list[dict[str, Any]]:
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        features = payload.get("features", []) if isinstance(payload, dict) else []
        rows: list[dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                continue
            rows.append(feature)
        return rows

    def _best_address_point(
        self,
        *,
        geocode_point: Any,
        max_distance_m: float | None = None,
    ) -> tuple[dict[str, Any] | None, float | None]:
        if not self._geo_ready() or not self._file_exists(self.address_points_path):
            return None, None
        threshold = float(max_distance_m if max_distance_m is not None else self.max_address_point_distance_m)
        best_feature: dict[str, Any] | None = None
        best_distance: float | None = None
        for feature in self._load_geojson_features(self.address_points_path):
            coords = self._coords_from_feature(feature)
            if coords is None:
                continue
            point = Point(coords[1], coords[0])
            distance_m = self._distance_m(geocode_point, point)
            if distance_m > threshold:
                continue
            if best_distance is None or distance_m < best_distance:
                best_feature = feature
                best_distance = distance_m
        return best_feature, best_distance

    @staticmethod
    def _parcel_resolution_rank(result: ParcelResolutionResult) -> tuple[int, int, float]:
        status_rank = {
            "matched": 3,
            "multiple_candidates": 2,
            "not_found": 1,
        }.get(str(result.status or "not_found"), 0)
        method_rank = {
            "contains_point": 3,
            "nearest_within_tolerance": 2,
            "multiple_candidates": 1,
            "none": 0,
        }.get(str(result.parcel_lookup_method or "none"), 0)
        return (status_rank, method_rank, float(result.confidence or 0.0))

    def _resolve_parcel_from_candidates(
        self,
        *,
        candidate_points: list[Any],
        max_lookup_distance_m: float | None = None,
    ) -> ParcelResolutionResult:
        best: ParcelResolutionResult | None = None
        for anchor_point in candidate_points:
            current = self.parcel_resolver.resolve_for_point(
                anchor_point=anchor_point,
                max_lookup_distance_m=max_lookup_distance_m,
            )
            if best is None or self._parcel_resolution_rank(current) > self._parcel_resolution_rank(best):
                best = current
            if current.status == "matched" and str(current.parcel_lookup_method or "") == "contains_point":
                break
        return best or ParcelResolutionResult(
            status="not_found",
            confidence=0.0,
            source=None,
            geometry_used="none",
            overlap_score=0.0,
            parcel_lookup_method="none",
            diagnostics=["Parcel resolution did not return a candidate."],
        )

    @staticmethod
    def _anchor_source_from_geocode_precision(precision: str) -> str:
        p = str(precision or "unknown").strip().lower()
        if p == "rooftop":
            return "rooftop_geocode"
        if p == "parcel_or_address_point":
            return "address_point_geocode"
        if p == "interpolated":
            return "interpolated_geocode"
        if p == "approximate":
            return "approximate_geocode"
        return "geocoded_address_point"

    @staticmethod
    def _distance_limits_for_precision(
        *,
        geocode_precision: str,
        address_default_m: float,
        parcel_default_m: float,
        override_anchor: bool,
    ) -> tuple[float, float]:
        # Lower-precision geocodes are frequently offset from true parcels/addresses;
        # widen lookup tolerances before falling back to weaker anchor modes.
        precision = str(geocode_precision or "unknown").strip().lower()
        address_limit = float(address_default_m)
        parcel_limit = float(parcel_default_m)
        if precision == "interpolated":
            address_limit = max(address_limit + 18.0, address_limit * 1.8)
            parcel_limit = max(parcel_limit + 22.0, parcel_limit * 2.0)
        elif precision in {"approximate", "unknown"}:
            address_limit = max(address_limit + 25.0, address_limit * 2.2)
            parcel_limit = max(parcel_limit + 28.0, parcel_limit * 2.4)
        if override_anchor:
            parcel_limit = max(parcel_limit, parcel_default_m * 2.0)
        return min(address_limit, 160.0), min(parcel_limit, 220.0)

    @staticmethod
    def _anchor_quality_summary(
        *,
        anchor_source: str,
        geocode_precision: str,
        geocode_to_anchor_distance_m: float,
        parcel_present: bool,
        address_point_present: bool,
    ) -> tuple[str, float]:
        source = str(anchor_source or "")
        precision = str(geocode_precision or "unknown").strip().lower()
        score = 0.48
        if source == "authoritative_address_point":
            score = 0.95
        elif source == "parcel_polygon_centroid":
            score = 0.83
        elif source == "rooftop_geocode":
            score = 0.80
        elif source == "address_point_geocode":
            score = 0.76
        elif source == "interpolated_geocode":
            score = 0.62
        elif source == "approximate_geocode":
            score = 0.48
        elif source == "user_selected_point":
            score = 0.84 if parcel_present else 0.74
        if address_point_present and source != "authoritative_address_point":
            score += 0.04
        if parcel_present:
            score += 0.03
        if precision == "approximate":
            score -= 0.08
        elif precision == "unknown":
            score -= 0.05
        if geocode_to_anchor_distance_m > 90.0:
            score -= 0.12
        elif geocode_to_anchor_distance_m > 45.0:
            score -= 0.07
        elif geocode_to_anchor_distance_m > 20.0:
            score -= 0.03
        score = round(max(0.0, min(1.0, score)), 2)
        if score >= 0.82:
            tier = "high"
        elif score >= 0.60:
            tier = "medium"
        else:
            tier = "low"
        return tier, score

    def resolve(
        self,
        *,
        geocoded_lat: float,
        geocoded_lon: float,
        geocode_provider: str | None = None,
        geocode_precision: str | None = None,
        geocoded_address: str | None = None,
        property_anchor_override: tuple[float, float] | None = None,
        property_anchor_override_source: str | None = None,
        property_anchor_override_precision: str | None = None,
    ) -> PropertyAnchorResolution:
        fallback_precision = str(geocode_precision or "unknown")
        override_anchor = None
        if property_anchor_override is not None:
            try:
                override_lat = float(property_anchor_override[0])
                override_lon = float(property_anchor_override[1])
                if -90.0 <= override_lat <= 90.0 and -180.0 <= override_lon <= 180.0:
                    override_anchor = (override_lat, override_lon)
            except (TypeError, ValueError, IndexError):
                override_anchor = None
        if not self._geo_ready():
            return PropertyAnchorResolution(
                anchor_latitude=float(override_anchor[0] if override_anchor is not None else geocoded_lat),
                anchor_longitude=float(override_anchor[1] if override_anchor is not None else geocoded_lon),
                anchor_source=(
                    str(property_anchor_override_source or "user_selected_point")
                    if override_anchor is not None
                    else self._anchor_source_from_geocode_precision(fallback_precision)
                ),
                anchor_precision=(
                    str(property_anchor_override_precision or "user_selected_point")
                    if override_anchor is not None
                    else fallback_precision
                ),
                geocoded_latitude=float(geocoded_lat),
                geocoded_longitude=float(geocoded_lon),
                geocode_provider=geocode_provider,
                geocode_precision=fallback_precision,
                geocoded_address=geocoded_address,
                diagnostics=["Property anchor resolver fell back to geocode point; geospatial dependencies unavailable."],
            )

        geocode_point = Point(float(geocoded_lon), float(geocoded_lat))
        requested_anchor_point = (
            Point(float(override_anchor[1]), float(override_anchor[0]))
            if override_anchor is not None
            else geocode_point
        )
        address_limit_m, parcel_limit_m = self._distance_limits_for_precision(
            geocode_precision=fallback_precision,
            address_default_m=self.max_address_point_distance_m,
            parcel_default_m=self.max_parcel_lookup_distance_m,
            override_anchor=override_anchor is not None,
        )
        address_feature, address_distance_m = self._best_address_point(
            geocode_point=geocode_point,
            max_distance_m=address_limit_m,
        )
        address_point = None
        if address_feature is not None:
            coords = self._coords_from_feature(address_feature)
            if coords is not None:
                address_point = Point(coords[1], coords[0])

        parcel_candidate_points: list[Any] = []
        if override_anchor is not None:
            parcel_candidate_points.append(requested_anchor_point)
        elif address_point is not None:
            parcel_candidate_points.append(address_point)
        parcel_candidate_points.append(geocode_point)
        parcel_resolution = self._resolve_parcel_from_candidates(
            candidate_points=parcel_candidate_points,
            max_lookup_distance_m=parcel_limit_m,
        )
        parcel_feature = parcel_resolution.parcel_feature
        parcel_geom = parcel_resolution.parcel_polygon
        parcel_lookup_method = parcel_resolution.parcel_lookup_method
        parcel_lookup_distance_m = parcel_resolution.parcel_lookup_distance_m

        anchor_lat = float(geocoded_lat)
        anchor_lon = float(geocoded_lon)
        anchor_source = self._anchor_source_from_geocode_precision(fallback_precision)
        anchor_precision = fallback_precision
        anchor_selection_method = "geocode_fallback"
        diagnostics: list[str] = []

        if override_anchor is not None:
            anchor_lat = float(override_anchor[0])
            anchor_lon = float(override_anchor[1])
            anchor_source = str(property_anchor_override_source or "user_selected_point")
            anchor_precision = str(property_anchor_override_precision or "user_selected_point")
            anchor_selection_method = "user_selected_point"
            diagnostics.append("Property anchor uses the user-selected home point.")
        elif address_feature is not None and address_point is not None and "address_point" in self.source_priority:
            anchor_lat = float(address_point.y)
            anchor_lon = float(address_point.x)
            anchor_source = "authoritative_address_point"
            anchor_precision = "parcel_or_address_point"
            anchor_selection_method = "address_point_nearest"
            diagnostics.append("Property anchor uses configured address-point source.")
        elif parcel_geom is not None and "parcel_centroid" in self.source_priority:
            centroid = parcel_geom.centroid
            anchor_lat = float(centroid.y)
            anchor_lon = float(centroid.x)
            anchor_source = "parcel_polygon_centroid"
            anchor_precision = "parcel_or_address_point"
            anchor_selection_method = "parcel_centroid"
            diagnostics.append("Property anchor uses parcel centroid because no address point was selected.")
        else:
            diagnostics.append("Property anchor uses geocode point fallback.")

        if parcel_resolution.status == "multiple_candidates":
            diagnostics.append(
                "Parcel lookup found multiple plausible candidates; selected the best-ranked polygon with reduced confidence."
            )
        elif parcel_geom is not None:
            if parcel_lookup_method == "contains_point":
                diagnostics.append("Parcel lookup matched a containing parcel polygon.")
            elif parcel_lookup_method == "nearest_within_tolerance":
                diagnostics.append(
                    f"Parcel lookup used nearest parcel within {parcel_lookup_distance_m:.1f} m tolerance."
                )
        else:
            diagnostics.append("Parcel lookup did not find a parcel polygon for this property anchor.")
        diagnostics.append(
            "Parcel resolution status: "
            f"{parcel_resolution.status} (confidence {float(parcel_resolution.confidence):.1f}/100)."
        )

        anchor_point = Point(anchor_lon, anchor_lat)
        geocode_to_anchor_distance_m = self._distance_m(geocode_point, anchor_point)
        source_conflict_flag = geocode_to_anchor_distance_m >= 25.0
        anchor_quality, anchor_quality_score = self._anchor_quality_summary(
            anchor_source=anchor_source,
            geocode_precision=fallback_precision,
            geocode_to_anchor_distance_m=geocode_to_anchor_distance_m,
            parcel_present=parcel_geom is not None,
            address_point_present=address_point is not None,
        )
        alignment_notes: list[str] = []
        if geocode_to_anchor_distance_m >= 8.0:
            alignment_notes.append(
                f"Property anchor and geocoded point differ by {geocode_to_anchor_distance_m:.1f} m."
            )
        if source_conflict_flag:
            alignment_notes.append(
                "Location sources conflict materially; structure-specific overlays should be treated with caution."
            )
        diagnostics.append(
            f"Anchor lookup tolerances used: address_point<= {address_limit_m:.1f} m, parcel<= {parcel_limit_m:.1f} m."
        )
        diagnostics.append(
            f"Anchor quality assessed as {anchor_quality} ({anchor_quality_score:.2f})."
        )

        parcel_props = dict(parcel_feature.get("properties") or {}) if isinstance(parcel_feature, dict) else {}
        address_props = dict(address_feature.get("properties") or {}) if isinstance(address_feature, dict) else {}

        parcel_source_name = (
            parcel_resolution.source_name
            or self._extract_source_name(parcel_props, os.getenv("WF_PARCEL_SOURCE_NAME") or None)
        )
        parcel_source_vintage = parcel_resolution.source_vintage or self._extract_source_vintage(
            parcel_props,
            "WF_PARCEL_SOURCE_VINTAGE",
        )
        address_source_name = self._extract_source_name(
            address_props,
            os.getenv("WF_ADDRESS_POINT_SOURCE_NAME") or None,
        )
        address_source_vintage = self._extract_source_vintage(address_props, "WF_ADDRESS_POINT_SOURCE_VINTAGE")

        parcel_geometry_geojson = None
        if parcel_geom is not None:
            parcel_geometry_geojson = {
                "type": "Feature",
                "properties": {
                    "source": "parcel_polygon",
                    "parcel_id": self._extract_parcel_id(parcel_props),
                },
                "geometry": mapping(parcel_geom),
            }

        parcel_address_point_geojson = None
        if address_feature is not None:
            coords = self._coords_from_feature(address_feature)
            if coords is not None:
                parcel_address_point_geojson = self._point_feature(
                    coords[0],
                    coords[1],
                    label="Parcel/address point",
                    source="parcel_address_point",
                )
                parcel_address_point_geojson["properties"].update(
                    {
                        "source_name": address_source_name,
                        "source_vintage": address_source_vintage,
                    }
                )

        return PropertyAnchorResolution(
            anchor_latitude=float(anchor_lat),
            anchor_longitude=float(anchor_lon),
            anchor_source=anchor_source,
            anchor_precision=anchor_precision,
            geocoded_latitude=float(geocoded_lat),
            geocoded_longitude=float(geocoded_lon),
            geocode_provider=geocode_provider,
            geocode_precision=fallback_precision,
            geocoded_address=geocoded_address,
            parcel_id=self._extract_parcel_id(parcel_props),
            parcel_polygon=parcel_geom,
            parcel_lookup_method=parcel_lookup_method,
            parcel_lookup_distance_m=parcel_lookup_distance_m,
            parcel_geometry_geojson=parcel_geometry_geojson,
            parcel_address_point_geojson=parcel_address_point_geojson,
            parcel_source_name=parcel_source_name,
            parcel_source_vintage=parcel_source_vintage,
            address_point_source_name=address_source_name,
            address_point_source_vintage=address_source_vintage,
            geocode_to_anchor_distance_m=round(geocode_to_anchor_distance_m, 2),
            anchor_selection_method=anchor_selection_method,
            anchor_quality=anchor_quality,
            anchor_quality_score=anchor_quality_score,
            address_point_lookup_distance_m=(round(float(address_distance_m), 2) if address_distance_m is not None else None),
            source_conflict_flag=source_conflict_flag,
            diagnostics=diagnostics,
            alignment_notes=alignment_notes,
            parcel_resolution=parcel_resolution.to_summary(),
            parcel_bounding_approximation_geojson=parcel_resolution.approximation_geometry_geojson,
            parcel_properties=parcel_props,
            address_point_properties=address_props,
        )
