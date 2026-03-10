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
    parcel_geometry_geojson: dict[str, Any] | None = None
    parcel_address_point_geojson: dict[str, Any] | None = None
    parcel_source_name: str | None = None
    parcel_source_vintage: str | None = None
    address_point_source_name: str | None = None
    address_point_source_vintage: str | None = None
    geocode_to_anchor_distance_m: float | None = None
    source_conflict_flag: bool = False
    diagnostics: list[str] = field(default_factory=list)
    alignment_notes: list[str] = field(default_factory=list)

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
            "parcel_source_name": self.parcel_source_name,
            "parcel_source_vintage": self.parcel_source_vintage,
            "address_point_source_name": self.address_point_source_name,
            "address_point_source_vintage": self.address_point_source_vintage,
            "geocode_to_anchor_distance_m": self.geocode_to_anchor_distance_m,
            "source_conflict_flag": bool(self.source_conflict_flag),
            "alignment_notes": list(self.alignment_notes),
        }


class PropertyAnchorResolver:
    def __init__(
        self,
        *,
        address_points_path: str | None = None,
        parcels_path: str | None = None,
        source_priority: tuple[str, ...] | None = None,
    ) -> None:
        self.address_points_path = str(address_points_path or "").strip()
        self.parcels_path = str(parcels_path or "").strip()
        self.max_address_point_distance_m = self._env_float(
            "WF_PROPERTY_ANCHOR_MAX_ADDRESS_POINT_DISTANCE_M",
            45.0,
            min_value=2.0,
        )
        self.source_priority = source_priority or self._resolve_source_priority()

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
    ) -> tuple[dict[str, Any] | None, float | None]:
        if not self._geo_ready() or not self._file_exists(self.address_points_path):
            return None, None
        best_feature: dict[str, Any] | None = None
        best_distance: float | None = None
        for feature in self._load_geojson_features(self.address_points_path):
            coords = self._coords_from_feature(feature)
            if coords is None:
                continue
            point = Point(coords[1], coords[0])
            distance_m = self._distance_m(geocode_point, point)
            if distance_m > self.max_address_point_distance_m:
                continue
            if best_distance is None or distance_m < best_distance:
                best_feature = feature
                best_distance = distance_m
        return best_feature, best_distance

    def _best_parcel_for_point(
        self,
        *,
        anchor_point: Any,
    ) -> tuple[dict[str, Any] | None, Any | None]:
        if not self._geo_ready() or not self._file_exists(self.parcels_path):
            return None, None
        containing: list[tuple[dict[str, Any], Any]] = []
        for feature in self._load_geojson_features(self.parcels_path):
            geom_raw = feature.get("geometry")
            if not isinstance(geom_raw, dict):
                continue
            gtype = str(geom_raw.get("type") or "")
            if gtype not in {"Polygon", "MultiPolygon"}:
                continue
            try:
                geom = shape(geom_raw)
            except Exception:
                continue
            if geom.is_empty:
                continue
            if geom.covers(anchor_point):
                containing.append((feature, geom))
        if not containing:
            return None, None
        containing.sort(key=lambda row: float(max(0.0, row[1].area)))
        return containing[0]

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

    def resolve(
        self,
        *,
        geocoded_lat: float,
        geocoded_lon: float,
        geocode_provider: str | None = None,
        geocode_precision: str | None = None,
        geocoded_address: str | None = None,
    ) -> PropertyAnchorResolution:
        fallback_precision = str(geocode_precision or "unknown")
        if not self._geo_ready():
            return PropertyAnchorResolution(
                anchor_latitude=float(geocoded_lat),
                anchor_longitude=float(geocoded_lon),
                anchor_source=self._anchor_source_from_geocode_precision(fallback_precision),
                anchor_precision=fallback_precision,
                geocoded_latitude=float(geocoded_lat),
                geocoded_longitude=float(geocoded_lon),
                geocode_provider=geocode_provider,
                geocode_precision=fallback_precision,
                geocoded_address=geocoded_address,
                diagnostics=["Property anchor resolver fell back to geocode point; geospatial dependencies unavailable."],
            )

        geocode_point = Point(float(geocoded_lon), float(geocoded_lat))
        address_feature, address_distance_m = self._best_address_point(geocode_point=geocode_point)
        address_point = None
        if address_feature is not None:
            coords = self._coords_from_feature(address_feature)
            if coords is not None:
                address_point = Point(coords[1], coords[0])

        parcel_feature = None
        parcel_geom = None
        if address_point is not None:
            parcel_feature, parcel_geom = self._best_parcel_for_point(anchor_point=address_point)
        if parcel_feature is None or parcel_geom is None:
            parcel_feature, parcel_geom = self._best_parcel_for_point(anchor_point=geocode_point)

        anchor_lat = float(geocoded_lat)
        anchor_lon = float(geocoded_lon)
        anchor_source = self._anchor_source_from_geocode_precision(fallback_precision)
        anchor_precision = fallback_precision
        diagnostics: list[str] = []

        if address_feature is not None and address_point is not None and "address_point" in self.source_priority:
            anchor_lat = float(address_point.y)
            anchor_lon = float(address_point.x)
            anchor_source = "authoritative_address_point"
            anchor_precision = "parcel_or_address_point"
            diagnostics.append("Property anchor uses configured address-point source.")
        elif parcel_geom is not None and "parcel_centroid" in self.source_priority:
            centroid = parcel_geom.centroid
            anchor_lat = float(centroid.y)
            anchor_lon = float(centroid.x)
            anchor_source = "parcel_polygon_centroid"
            anchor_precision = "parcel_or_address_point"
            diagnostics.append("Property anchor uses parcel centroid because no address point was selected.")
        else:
            diagnostics.append("Property anchor uses geocode point fallback.")

        anchor_point = Point(anchor_lon, anchor_lat)
        geocode_to_anchor_distance_m = self._distance_m(geocode_point, anchor_point)
        source_conflict_flag = geocode_to_anchor_distance_m >= 25.0
        alignment_notes: list[str] = []
        if geocode_to_anchor_distance_m >= 8.0:
            alignment_notes.append(
                f"Property anchor and geocoded point differ by {geocode_to_anchor_distance_m:.1f} m."
            )
        if source_conflict_flag:
            alignment_notes.append(
                "Location sources conflict materially; structure-specific overlays should be treated with caution."
            )

        parcel_props = dict(parcel_feature.get("properties") or {}) if isinstance(parcel_feature, dict) else {}
        address_props = dict(address_feature.get("properties") or {}) if isinstance(address_feature, dict) else {}

        parcel_source_name = self._extract_source_name(parcel_props, os.getenv("WF_PARCEL_SOURCE_NAME") or None)
        parcel_source_vintage = self._extract_source_vintage(parcel_props, "WF_PARCEL_SOURCE_VINTAGE")
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
            parcel_geometry_geojson=parcel_geometry_geojson,
            parcel_address_point_geojson=parcel_address_point_geojson,
            parcel_source_name=parcel_source_name,
            parcel_source_vintage=parcel_source_vintage,
            address_point_source_name=address_source_name,
            address_point_source_vintage=address_source_vintage,
            geocode_to_anchor_distance_m=round(geocode_to_anchor_distance_m, 2),
            source_conflict_flag=source_conflict_flag,
            diagnostics=diagnostics,
            alignment_notes=alignment_notes,
        )
