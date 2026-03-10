from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from backend.building_footprints import BuildingFootprintClient, compute_structure_rings
from backend.models import AssessmentMapLayer, AssessmentMapPayload, AssessmentResult
from backend.region_registry import load_region_manifest, resolve_region_file

try:
    from pyproj import Transformer
    from shapely.geometry import Point, box, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - optional geo runtime dependencies
    Transformer = None
    Point = None
    box = None
    mapping = None
    shape = None
    shapely_transform = None


FEET_TO_METERS = 0.3048


def _geo_ready() -> bool:
    return bool(Transformer and Point and box and mapping and shape and shapely_transform)


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _to_json_geometry(geom: Any) -> dict[str, Any] | None:
    if not _geo_ready() or geom is None:
        return None
    try:
        return mapping(geom)
    except Exception:
        return None


def _meter_buffer_polygon(lat: float, lon: float, radius_m: float) -> Any | None:
    if not _geo_ready():
        return None
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    try:
        p = Point(float(lon), float(lat))
        p_m = shapely_transform(to_3857, p)
        buffer_m = p_m.buffer(float(radius_m), resolution=32)
        return shapely_transform(to_4326, buffer_m)
    except Exception:
        return None


def _viewport_bbox(lat: float, lon: float, radius_m: float = 2500.0) -> tuple[float, float, float, float]:
    lat_deg = radius_m / 111_320.0
    lon_deg = radius_m / max(1e-6, 111_320.0 * max(0.1, math.cos(math.radians(lat))))
    return lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg


def _load_geojson_features(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    features = payload.get("features")
    if not isinstance(features, list):
        return []
    return [f for f in features if isinstance(f, dict) and isinstance(f.get("geometry"), dict)]


def _filter_features_to_bbox(
    features: list[dict[str, Any]],
    *,
    bbox_bounds: tuple[float, float, float, float],
    max_features: int,
    clip: bool,
) -> list[dict[str, Any]]:
    if not features:
        return []
    if not _geo_ready():
        return features[:max_features]

    minx, miny, maxx, maxy = bbox_bounds
    bbox_geom = box(minx, miny, maxx, maxy)
    kept: list[dict[str, Any]] = []

    for feature in features:
        geom_raw = feature.get("geometry")
        if not isinstance(geom_raw, dict):
            continue
        try:
            geom = shape(geom_raw)
        except Exception:
            continue
        if geom.is_empty or not geom.intersects(bbox_geom):
            continue

        out_geom = geom.intersection(bbox_geom) if clip else geom
        if out_geom.is_empty:
            continue

        out_feature = {
            "type": "Feature",
            "properties": dict(feature.get("properties") or {}),
            "geometry": _to_json_geometry(out_geom),
        }
        if out_feature["geometry"] is None:
            continue
        kept.append(out_feature)
        if len(kept) >= max_features:
            break

    return kept


def _resolve_region_manifest_for_result(result: AssessmentResult, region_data_dir: str | None) -> dict[str, Any] | None:
    context = result.property_level_context if isinstance(result.property_level_context, dict) else {}
    region_id = str((context.get("region_id") or result.resolved_region_id or "")).strip()
    manifest_path = str(context.get("region_manifest_path") or "").strip()

    if manifest_path:
        p = Path(manifest_path)
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                payload["_manifest_path"] = str(p)
                payload["_region_dir"] = str(p.parent)
                return payload

    if region_id:
        return load_region_manifest(region_id, base_dir=region_data_dir)
    return None


def _resolve_vector_paths(
    result: AssessmentResult,
    *,
    wildfire_data: Any,
) -> dict[str, str | None]:
    region_data_dir = getattr(wildfire_data, "region_data_dir", None)
    manifest = _resolve_region_manifest_for_result(result, region_data_dir)

    fire_perimeters: str | None = None
    footprints: str | None = None
    fema_structures: str | None = None

    if manifest:
        fire_perimeters = resolve_region_file(manifest, "fire_perimeters", base_dir=region_data_dir)
        footprints = resolve_region_file(manifest, "building_footprints", base_dir=region_data_dir)

    runtime_paths = getattr(wildfire_data, "paths", {}) or {}
    base_paths = getattr(wildfire_data, "base_paths", {}) or {}

    fire_perimeters = fire_perimeters or runtime_paths.get("perimeters") or base_paths.get("perimeters")
    footprints = footprints or runtime_paths.get("footprints") or base_paths.get("footprints")
    fema_structures = runtime_paths.get("fema_structures") or base_paths.get("fema_structures")

    return {
        "fire_perimeters": fire_perimeters,
        "footprints": footprints,
        "fema_structures": fema_structures,
    }


def _property_marker_feature(result: AssessmentResult) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "label": "Assessed property",
            "assessment_id": result.assessment_id,
            "address": result.address,
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(result.longitude), float(result.latitude)],
        },
    }


def _ring_zone_features_from_geometry(rings: dict[str, Any], *, basis: str) -> list[dict[str, Any]]:
    zone_defs = [
        ("ring_0_5_ft", "0-5 ft", "zone_0_5_ft"),
        ("ring_5_30_ft", "5-30 ft", "zone_5_30_ft"),
        ("ring_30_100_ft", "30-100 ft", "zone_30_100_ft"),
        ("ring_100_300_ft", "100-300 ft", "zone_100_300_ft"),
    ]
    features: list[dict[str, Any]] = []
    for ring_key, label, zone_key in zone_defs:
        geom = rings.get(ring_key)
        geometry = _to_json_geometry(geom)
        if geometry is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "zone_key": zone_key,
                    "distance_band_ft": label,
                    "basis_geometry": basis,
                },
                "geometry": geometry,
            }
        )
    return features


def _build_proxy_rings(lat: float, lon: float) -> list[dict[str, Any]]:
    if not _geo_ready():
        return []
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    center = Point(float(lon), float(lat))
    center_m = shapely_transform(to_3857, center)

    buffers = {
        "0_5": center_m.buffer(5.0 * FEET_TO_METERS, resolution=32),
        "5_30": center_m.buffer(30.0 * FEET_TO_METERS, resolution=32),
        "30_100": center_m.buffer(100.0 * FEET_TO_METERS, resolution=32),
        "100_300": center_m.buffer(300.0 * FEET_TO_METERS, resolution=32),
    }
    rings = {
        "ring_0_5_ft": buffers["0_5"],
        "ring_5_30_ft": buffers["5_30"].difference(buffers["0_5"]),
        "ring_30_100_ft": buffers["30_100"].difference(buffers["5_30"]),
        "ring_100_300_ft": buffers["100_300"].difference(buffers["30_100"]),
    }

    rings_4326 = {key: shapely_transform(to_4326, geom) for key, geom in rings.items()}
    return _ring_zone_features_from_geometry(rings_4326, basis="point_proxy")


def build_assessment_map_payload(
    result: AssessmentResult,
    *,
    wildfire_data: Any,
) -> AssessmentMapPayload:
    lat = float(result.latitude)
    lon = float(result.longitude)

    limitations: list[str] = []
    data: dict[str, dict[str, Any]] = {}
    layers: list[AssessmentMapLayer] = []

    property_point = _feature_collection([_property_marker_feature(result)])
    data["property_point"] = property_point
    layers.append(
        AssessmentMapLayer(
            layer_key="property_point",
            display_name="Property",
            available=True,
            default_visible=True,
            description="Assessed property location.",
            legend_label="Property marker",
            geometry_type="point",
            feature_count=1,
        )
    )

    paths = _resolve_vector_paths(result, wildfire_data=wildfire_data)
    footprint_paths = [p for p in [paths.get("footprints"), paths.get("fema_structures")] if p]

    basis_geometry_type = "point_proxy"
    footprint_features: list[dict[str, Any]] = []
    ring_features: list[dict[str, Any]] = []

    if _geo_ready() and footprint_paths:
        try:
            client = BuildingFootprintClient(path=footprint_paths[0], extra_paths=footprint_paths[1:])
            fp_result = client.get_building_footprint(lat=lat, lon=lon)
        except Exception as exc:
            fp_result = None
            limitations.append(f"Building footprint lookup failed: {exc}")

        if fp_result and fp_result.found and fp_result.footprint is not None:
            basis_geometry_type = "building_footprint"
            geometry = _to_json_geometry(fp_result.footprint)
            if geometry:
                footprint_features = [
                    {
                        "type": "Feature",
                        "properties": {
                            "label": "Primary structure footprint",
                            "source": fp_result.source,
                            "confidence": fp_result.confidence,
                        },
                        "geometry": geometry,
                    }
                ]
            rings, _assumptions = compute_structure_rings(fp_result.footprint)
            ring_features = _ring_zone_features_from_geometry(rings, basis="building_footprint")
        else:
            limitations.append(
                "Structure footprint unavailable; defensible-space zones are shown using point-proxy geometry."
            )
            ring_features = _build_proxy_rings(lat=lat, lon=lon)
    else:
        if not _geo_ready():
            limitations.append("Geospatial dependencies unavailable; map uses point-only rendering.")
        elif not footprint_paths:
            limitations.append("Building footprint source not configured; defensible-space zones use point-proxy geometry.")
        ring_features = _build_proxy_rings(lat=lat, lon=lon)

    if footprint_features:
        data["building_footprint"] = _feature_collection(footprint_features)
    layers.append(
        AssessmentMapLayer(
            layer_key="building_footprint",
            display_name="Building Footprint",
            available=bool(footprint_features),
            default_visible=bool(footprint_features),
            description="Subject structure footprint selected near the geocoded location.",
            legend_label="Structure polygon",
            geometry_type="polygon",
            feature_count=len(footprint_features),
            reason_unavailable=None if footprint_features else "No building footprint was resolved for this property.",
        )
    )

    if ring_features:
        data["defensible_space_rings"] = _feature_collection(ring_features)
    layers.append(
        AssessmentMapLayer(
            layer_key="defensible_space_rings",
            display_name="Defensible Space Zones",
            available=bool(ring_features),
            default_visible=bool(ring_features),
            description="Distance-based rings (0-5 ft, 5-30 ft, 30-100 ft, 100-300 ft) used for near-structure analysis.",
            legend_label="Zone polygons",
            geometry_type="polygon",
            feature_count=len(ring_features),
            reason_unavailable=None if ring_features else "Defensible-space zones could not be generated.",
        )
    )

    map_bbox = _viewport_bbox(lat, lon, radius_m=2500.0)

    fire_features: list[dict[str, Any]] = []
    fire_path = str(paths.get("fire_perimeters") or "")
    if fire_path:
        fire_features = _filter_features_to_bbox(
            _load_geojson_features(fire_path),
            bbox_bounds=map_bbox,
            max_features=60,
            clip=True,
        )
    else:
        limitations.append("Historic fire perimeter overlay is unavailable for this region.")

    if fire_features:
        data["fire_perimeters"] = _feature_collection(fire_features)
    layers.append(
        AssessmentMapLayer(
            layer_key="fire_perimeters",
            display_name="Historic Fire Perimeters",
            available=bool(fire_features),
            default_visible=bool(fire_features),
            description="Past wildfire perimeter polygons near the assessed property.",
            legend_label="Historical burn boundaries",
            geometry_type="polygon",
            feature_count=len(fire_features),
            reason_unavailable=None if fire_features else "No nearby fire perimeter features were available.",
        )
    )

    nearby_structures: list[dict[str, Any]] = []
    if footprint_paths:
        nearby_structures = _filter_features_to_bbox(
            _load_geojson_features(footprint_paths[0]),
            bbox_bounds=_viewport_bbox(lat, lon, radius_m=800.0),
            max_features=250,
            clip=False,
        )

    if nearby_structures:
        data["nearby_structures"] = _feature_collection(nearby_structures)
    layers.append(
        AssessmentMapLayer(
            layer_key="nearby_structures",
            display_name="Nearby Structures",
            available=bool(nearby_structures),
            default_visible=False,
            description="Contextual nearby structure footprints around the assessed property.",
            legend_label="Nearby structures",
            geometry_type="polygon",
            feature_count=len(nearby_structures),
            reason_unavailable=None if nearby_structures else "No nearby structures were available in configured footprint data.",
        )
    )

    # Include defensible-space analysis limitation notes if present.
    for note in list(result.defensible_space_limitations_summary or [])[:4]:
        if note not in limitations:
            limitations.append(str(note))

    return AssessmentMapPayload(
        assessment_id=result.assessment_id,
        center={"latitude": lat, "longitude": lon},
        resolved_region_id=result.resolved_region_id or str((result.property_level_context or {}).get("region_id") or "") or None,
        coverage_available=bool(result.coverage_available),
        basis_geometry_type=basis_geometry_type,
        layers=layers,
        data=data,
        limitations=limitations[:8],
        metadata={
            "region_resolution": _dump_region_resolution(result),
            "model_governance": (result.model_governance.model_dump() if hasattr(result.model_governance, "model_dump") else {}),
        },
    )


def _dump_region_resolution(result: AssessmentResult) -> dict[str, Any]:
    region_resolution = result.region_resolution
    if hasattr(region_resolution, "model_dump"):
        try:
            return region_resolution.model_dump()
        except Exception:
            pass
    return {
        "coverage_available": bool(result.coverage_available),
        "resolved_region_id": result.resolved_region_id,
        "reason": None,
    }
