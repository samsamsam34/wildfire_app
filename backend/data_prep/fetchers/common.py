from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from shapely.geometry import box, mapping, shape
except Exception:  # pragma: no cover - optional dependency
    box = None
    mapping = None
    shape = None


BoundingBox = dict[str, float]


@dataclass
class ParcelFetchResult:
    source_id: str
    success: bool
    output_path: str | None = None
    record_count: int = 0
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def normalize_bounds(bounds: BoundingBox) -> BoundingBox:
    return {
        "min_lon": float(bounds["min_lon"]),
        "min_lat": float(bounds["min_lat"]),
        "max_lon": float(bounds["max_lon"]),
        "max_lat": float(bounds["max_lat"]),
    }


def _bbox_intersects_bounds(
    *,
    bbox: tuple[float, float, float, float],
    bounds: BoundingBox,
) -> bool:
    minx, miny, maxx, maxy = bbox
    return not (
        maxx < bounds["min_lon"]
        or minx > bounds["max_lon"]
        or maxy < bounds["min_lat"]
        or miny > bounds["max_lat"]
    )


def _coords_bounds(coords: Any) -> tuple[float, float, float, float] | None:
    flat: list[tuple[float, float]] = []

    def _walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                flat.append((float(value[0]), float(value[1])))
                return
            for part in value:
                _walk(part)

    _walk(coords)
    if not flat:
        return None
    minx = min(pt[0] for pt in flat)
    maxx = max(pt[0] for pt in flat)
    miny = min(pt[1] for pt in flat)
    maxy = max(pt[1] for pt in flat)
    return (minx, miny, maxx, maxy)


def clip_and_filter_polygon_features(
    *,
    features: list[dict[str, Any]],
    bounds: BoundingBox,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    aoi = None
    if box is not None:
        aoi = box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])

    for row in features:
        if not isinstance(row, dict):
            continue
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            continue
        geometry_type = str(geometry.get("type") or "")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue

        if aoi is not None and shape is not None and mapping is not None:
            try:
                geom = shape(geometry)
            except Exception:
                continue
            if geom.is_empty or not geom.intersects(aoi):
                continue
            clipped_geom = geom.intersection(aoi)
            if clipped_geom.is_empty:
                continue
            clipped.append(
                {
                    "type": "Feature",
                    "properties": dict(row.get("properties") or {}),
                    "geometry": mapping(clipped_geom),
                }
            )
            continue

        bbox = _coords_bounds(geometry.get("coordinates"))
        if bbox is None or not _bbox_intersects_bounds(bbox=bbox, bounds=bounds):
            continue
        clipped.append(
            {
                "type": "Feature",
                "properties": dict(row.get("properties") or {}),
                "geometry": geometry,
            }
        )
    return clipped


def parcel_id_from_props(props: dict[str, Any]) -> str | None:
    for key in (
        "parcel_id",
        "parcelid",
        "PARCEL_ID",
        "apn",
        "APN",
        "parcel",
        "parcel_number",
        "site_id",
        "globalid",
        "GlobalID",
        "OBJECTID",
        "objectid",
        "id",
    ):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def address_from_props(props: dict[str, Any]) -> str | None:
    for key in (
        "address",
        "full_address",
        "fulladdress",
        "site_address",
        "situs_address",
        "mailing_address",
    ):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    number = str(props.get("house_number") or props.get("addrnum") or "").strip()
    street = str(props.get("street_name") or props.get("street") or "").strip()
    city = str(props.get("city") or props.get("site_city") or "").strip()
    state = str(props.get("state") or props.get("site_state") or "").strip()
    zipcode = str(props.get("zip") or props.get("zipcode") or props.get("site_zip") or "").strip()
    parts = [part for part in [" ".join([number, street]).strip(), city, state, zipcode] if part]
    return ", ".join(parts) if parts else None


def owner_from_props(props: dict[str, Any]) -> str | None:
    for key in (
        "owner",
        "owner_name",
        "owner_full_name",
        "ownername",
        "taxpayer_name",
    ):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    first = str(props.get("owner_first_name") or props.get("owner_first") or "").strip()
    last = str(props.get("owner_last_name") or props.get("owner_last") or "").strip()
    full = " ".join([first, last]).strip()
    return full or None


def normalize_parcel_feature(
    *,
    geometry: dict[str, Any],
    properties: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    parcel_id = parcel_id_from_props(properties)
    address = address_from_props(properties)
    owner_name = owner_from_props(properties)
    normalized_props = {
        "parcel_id": parcel_id,
        "address": address,
        "owner_name": owner_name,
        "source": source,
    }
    return {
        "type": "Feature",
        "properties": normalized_props,
        "geometry": geometry,
    }


def write_parcel_geojson(
    *,
    features: list[dict[str, Any]],
    region_dir: Path,
    source_id: str,
    bounds: BoundingBox,
) -> Path:
    region_dir.mkdir(parents=True, exist_ok=True)
    output_path = region_dir / "parcel_polygons.geojson"
    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source_id": source_id,
            "record_count": len(features),
            "bbox": normalize_bounds(bounds),
        },
        "features": features,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
