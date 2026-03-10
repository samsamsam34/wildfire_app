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


def _canonical_wgs84_lon_lat(lon: Any, lat: Any) -> tuple[float, float] | None:
    """Return canonical EPSG:4326 lon/lat ordering and repair obvious axis-swaps."""
    try:
        lon_f = float(lon)
        lat_f = float(lat)
    except (TypeError, ValueError):
        return None

    if -180.0 <= lon_f <= 180.0 and -90.0 <= lat_f <= 90.0:
        return lon_f, lat_f

    # Defensive fallback for data paths that accidentally pass lat/lon.
    if -180.0 <= lat_f <= 180.0 and -90.0 <= lon_f <= 90.0:
        swapped_lon = lat_f
        swapped_lat = lon_f
        if -180.0 <= swapped_lon <= 180.0 and -90.0 <= swapped_lat <= 90.0:
            return swapped_lon, swapped_lat

    return None


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
    footprints_overture: str | None = None
    footprints: str | None = None
    fema_structures: str | None = None
    address_points: str | None = None
    parcels: str | None = None

    if manifest:
        fire_perimeters = resolve_region_file(manifest, "fire_perimeters", base_dir=region_data_dir)
        footprints_overture = (
            resolve_region_file(manifest, "building_footprints_overture", base_dir=region_data_dir)
            or resolve_region_file(manifest, "overture_buildings", base_dir=region_data_dir)
        )
        footprints = resolve_region_file(manifest, "building_footprints", base_dir=region_data_dir)
        address_points = (
            resolve_region_file(manifest, "address_points", base_dir=region_data_dir)
            or resolve_region_file(manifest, "parcel_address_points", base_dir=region_data_dir)
        )
        parcels = (
            resolve_region_file(manifest, "parcel_polygons", base_dir=region_data_dir)
            or resolve_region_file(manifest, "parcels", base_dir=region_data_dir)
        )

    runtime_paths = getattr(wildfire_data, "paths", {}) or {}
    base_paths = getattr(wildfire_data, "base_paths", {}) or {}

    fire_perimeters = fire_perimeters or runtime_paths.get("perimeters") or base_paths.get("perimeters")
    footprints_overture = (
        footprints_overture
        or runtime_paths.get("footprints_overture")
        or base_paths.get("footprints_overture")
    )
    footprints = footprints or runtime_paths.get("footprints") or base_paths.get("footprints")
    fema_structures = runtime_paths.get("fema_structures") or base_paths.get("fema_structures")
    address_points = address_points or runtime_paths.get("address_points") or base_paths.get("address_points")
    parcels = parcels or runtime_paths.get("parcels") or base_paths.get("parcels")

    return {
        "fire_perimeters": fire_perimeters,
        "footprints_overture": footprints_overture,
        "footprints": footprints,
        "fema_structures": fema_structures,
        "address_points": address_points,
        "parcels": parcels,
    }


def _point_feature(
    *,
    lon: float,
    lat: float,
    label: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _canonical_wgs84_lon_lat(lon, lat)
    if normalized is None:
        raise ValueError("Point coordinates are not valid WGS84 lon/lat.")
    norm_lon, norm_lat = normalized
    return {
        "type": "Feature",
        "properties": {
            "label": label,
            **(properties or {}),
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(norm_lon), float(norm_lat)],
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
    geocoded_lon_lat = _canonical_wgs84_lon_lat(lon, lat)
    if geocoded_lon_lat is None:
        geocoded_lon_lat = (float(lon), float(lat))
    geocoded_lon, geocoded_lat = geocoded_lon_lat

    limitations: list[str] = []
    data: dict[str, dict[str, Any]] = {}
    layers: list[AssessmentMapLayer] = []
    display_point_source = str(
        (result.property_level_context or {}).get("display_point_source")
        or result.display_point_source
        or "property_anchor_point"
    )
    geocoding = result.geocoding if hasattr(result, "geocoding") else None
    geocode_provider = str(getattr(geocoding, "geocode_provider", "") or getattr(geocoding, "provider", "") or "") or None
    geocoded_address = str(getattr(geocoding, "geocoded_address", "") or getattr(geocoding, "matched_address", "") or "") or None
    geocode_location_type = str(getattr(geocoding, "geocode_location_type", "") or "") or None
    geocode_precision = str(getattr(geocoding, "geocode_precision", "") or "") or None

    property_ctx = result.property_level_context if isinstance(result.property_level_context, dict) else {}
    property_anchor = (
        property_ctx.get("property_anchor_point")
        if isinstance(property_ctx.get("property_anchor_point"), dict)
        else (result.property_anchor_point if isinstance(result.property_anchor_point, dict) else None)
    )
    anchor_lat = geocoded_lat
    anchor_lon = geocoded_lon
    if property_anchor is not None:
        anchor_lon_lat = _canonical_wgs84_lon_lat(property_anchor.get("longitude"), property_anchor.get("latitude"))
        if anchor_lon_lat is not None:
            anchor_lon, anchor_lat = anchor_lon_lat
    property_anchor_source = str(
        property_ctx.get("property_anchor_source")
        or result.property_anchor_source
        or "geocoded_address_point"
    )
    property_anchor_precision = str(
        property_ctx.get("property_anchor_precision")
        or result.property_anchor_precision
        or geocode_precision
        or "unknown"
    )
    source_conflict_flag = bool(
        property_ctx.get("source_conflict_flag")
        if property_ctx.get("source_conflict_flag") is not None
        else result.source_conflict_flag
    )
    alignment_notes = [
        str(note)
        for note in (
            property_ctx.get("alignment_notes")
            if isinstance(property_ctx.get("alignment_notes"), list)
            else result.alignment_notes
        )
        if str(note).strip()
    ]
    parcel_id = str(property_ctx.get("parcel_id") or result.parcel_id or "") or None
    parcel_lookup_method = str(
        property_ctx.get("parcel_lookup_method") or result.parcel_lookup_method or ""
    ) or None
    parcel_lookup_distance_m = (
        float(property_ctx.get("parcel_lookup_distance_m"))
        if property_ctx.get("parcel_lookup_distance_m") is not None
        else (
            float(result.parcel_lookup_distance_m)
            if result.parcel_lookup_distance_m is not None
            else None
        )
    )
    structure_match_status = str(property_ctx.get("structure_match_status") or result.structure_match_status or "none")
    structure_match_method = (
        str(property_ctx.get("structure_match_method") or result.structure_match_method)
        if (property_ctx.get("structure_match_method") or result.structure_match_method)
        else None
    )
    structure_match_confidence = (
        float(property_ctx.get("structure_match_confidence"))
        if property_ctx.get("structure_match_confidence") is not None
        else (
            float(result.structure_match_confidence)
            if result.structure_match_confidence is not None
            else None
        )
    )
    structure_match_distance_m = (
        float(property_ctx.get("structure_match_distance_m"))
        if property_ctx.get("structure_match_distance_m") is not None
        else (
            float(result.structure_match_distance_m)
            if result.structure_match_distance_m is not None
            else None
        )
    )
    candidate_structure_count = (
        int(property_ctx.get("candidate_structure_count"))
        if property_ctx.get("candidate_structure_count") is not None
        else (
            int(result.candidate_structure_count)
            if result.candidate_structure_count is not None
            else None
        )
    )
    structure_geometry_source = str(property_ctx.get("structure_geometry_source") or "auto_detected").strip().lower()
    if structure_geometry_source not in {"auto_detected", "user_selected", "user_modified"}:
        structure_geometry_source = "auto_detected"
    matched_structure_id = str(
        property_ctx.get("matched_structure_id") or result.matched_structure_id or ""
    ) or None
    structure_match_candidates = property_ctx.get("structure_match_candidates")
    if not isinstance(structure_match_candidates, list):
        structure_match_candidates = []

    parcel_address_point_feature = property_ctx.get("parcel_address_point") if isinstance(property_ctx.get("parcel_address_point"), dict) else None
    parcel_polygon_feature = property_ctx.get("parcel_geometry") if isinstance(property_ctx.get("parcel_geometry"), dict) else None
    parcel_geom_for_matching = None
    if _geo_ready() and parcel_polygon_feature and isinstance(parcel_polygon_feature.get("geometry"), dict):
        try:
            parcel_geom_for_matching = shape(parcel_polygon_feature["geometry"])
        except Exception:
            parcel_geom_for_matching = None

    footprint_source_name = str(property_ctx.get("footprint_source_name") or "") or None
    footprint_source_vintage = str(property_ctx.get("footprint_source_vintage") or "") or None
    parcel_source_name = str(property_ctx.get("parcel_source_name") or "") or None
    parcel_source_vintage = str(property_ctx.get("parcel_source_vintage") or "") or None

    geocoded_address_point = _point_feature(
        lon=geocoded_lon,
        lat=geocoded_lat,
        label="Geocoded address point",
        properties={
            "source": "geocoded_address_point",
            "assessment_id": result.assessment_id,
            "address": result.address,
            "crs": "EPSG:4326",
        },
    )
    data["geocoded_address_point"] = _feature_collection([geocoded_address_point])
    layers.append(
        AssessmentMapLayer(
            layer_key="geocoded_address_point",
            display_name="Geocoded Address Point",
            available=True,
            default_visible=False,
            description="Raw geocoded point from address lookup (WGS84).",
            legend_label="Geocoded address point",
            geometry_type="point",
            feature_count=1,
        )
    )

    property_anchor_point_feature = _point_feature(
        lon=anchor_lon,
        lat=anchor_lat,
        label="Property anchor point",
        properties={
            "source": "property_anchor_point",
            "anchor_source": property_anchor_source,
            "anchor_precision": property_anchor_precision,
            "assessment_id": result.assessment_id,
            "address": result.address,
            "crs": "EPSG:4326",
        },
    )
    data["property_anchor_point"] = _feature_collection([property_anchor_point_feature])
    layers.append(
        AssessmentMapLayer(
            layer_key="property_anchor_point",
            display_name="Property Anchor",
            available=True,
            default_visible=False,
            description="Canonical property anchor used before structure matching (WGS84).",
            legend_label="Property anchor point",
            geometry_type="point",
            feature_count=1,
        )
    )

    if parcel_address_point_feature:
        data["parcel_address_point"] = _feature_collection([parcel_address_point_feature])
    layers.append(
        AssessmentMapLayer(
            layer_key="parcel_address_point",
            display_name="Parcel/Address Point",
            available=bool(parcel_address_point_feature),
            default_visible=False,
            description="Authoritative parcel/address point when configured.",
            legend_label="Parcel address point",
            geometry_type="point",
            feature_count=1 if parcel_address_point_feature else 0,
            reason_unavailable=None if parcel_address_point_feature else "No parcel/address point was available for this property.",
        )
    )

    if parcel_polygon_feature:
        data["parcel_polygon"] = _feature_collection([parcel_polygon_feature])
    layers.append(
        AssessmentMapLayer(
            layer_key="parcel_polygon",
            display_name="Parcel Boundary",
            available=bool(parcel_polygon_feature),
            default_visible=False,
            description="Matched parcel polygon used for optional parcel-aware structure matching.",
            legend_label="Parcel boundary",
            geometry_type="polygon",
            feature_count=1 if parcel_polygon_feature else 0,
            reason_unavailable=None if parcel_polygon_feature else "No parcel polygon was available for this property.",
        )
    )

    paths = _resolve_vector_paths(result, wildfire_data=wildfire_data)
    footprint_paths = [
        p
        for p in [paths.get("footprints_overture"), paths.get("footprints"), paths.get("fema_structures")]
        if p
    ]

    basis_geometry_type = "point_proxy"
    footprint_features: list[dict[str, Any]] = []
    ring_features: list[dict[str, Any]] = []
    matched_structure_centroid: dict[str, Any] | None = None
    max_match_distance_m: float | None = None
    ambiguity_gap_m: float | None = None

    if _geo_ready() and footprint_paths:
        try:
            client = BuildingFootprintClient(path=footprint_paths[0], extra_paths=footprint_paths[1:])
            max_match_distance_m = float(getattr(client, "max_match_distance_m", 0.0) or 0.0) or None
            ambiguity_gap_m = float(getattr(client, "ambiguity_gap_m", 0.0) or 0.0) or None
            fp_result = client.get_building_footprint(
                lat=anchor_lat,
                lon=anchor_lon,
                parcel_polygon=parcel_geom_for_matching,
                anchor_precision=property_anchor_precision,
            )
        except Exception as exc:
            fp_result = None
            limitations.append(f"Building footprint lookup failed: {exc}")

        if fp_result and fp_result.found and fp_result.footprint is not None:
            structure_match_status = str(fp_result.match_status or "matched")
            structure_match_method = fp_result.match_method or "point_in_polygon"
            structure_match_confidence = float(fp_result.confidence or 0.0)
            structure_match_distance_m = (
                float(fp_result.match_distance_m) if fp_result.match_distance_m is not None else 0.0
            )
            candidate_structure_count = int(fp_result.candidate_count or 1)
            structure_match_candidates = list(fp_result.candidate_summaries or [])
            basis_geometry_type = "building_footprint"
            geometry = _to_json_geometry(fp_result.footprint)
            if geometry:
                footprint_features = [
                    {
                        "type": "Feature",
                        "properties": {
                            "label": "Primary structure footprint",
                            "source": fp_result.source,
                            "building_source": Path(str(fp_result.source or "")).stem or None,
                            "confidence": fp_result.confidence,
                            "match_status": structure_match_status,
                            "match_method": structure_match_method,
                            "matched_structure_id": fp_result.matched_structure_id,
                            "match_distance_m": structure_match_distance_m,
                            "crs": "EPSG:4326",
                        },
                        "geometry": geometry,
                    }
                ]
            centroid_lat = None
            centroid_lon = None
            if isinstance(fp_result.centroid, tuple) and len(fp_result.centroid) == 2:
                centroid_lat = fp_result.centroid[0]
                centroid_lon = fp_result.centroid[1]
            elif _geo_ready():
                try:
                    centroid_lat = float(fp_result.footprint.centroid.y)
                    centroid_lon = float(fp_result.footprint.centroid.x)
                except Exception:
                    centroid_lat = None
                    centroid_lon = None

            centroid_lon_lat = _canonical_wgs84_lon_lat(centroid_lon, centroid_lat)
            if centroid_lon_lat is not None:
                centroid_lon_norm, centroid_lat_norm = centroid_lon_lat
                matched_structure_centroid = _point_feature(
                    lon=centroid_lon_norm,
                    lat=centroid_lat_norm,
                    label="Matched structure centroid",
                    properties={
                        "source": "matched_structure_centroid",
                        "building_source": Path(str(fp_result.source or "")).stem or None,
                        "footprint_source": fp_result.source,
                        "confidence": fp_result.confidence,
                        "match_status": structure_match_status,
                        "match_method": structure_match_method,
                        "matched_structure_id": fp_result.matched_structure_id,
                        "match_distance_m": structure_match_distance_m,
                        "crs": "EPSG:4326",
                    },
                )
                data["matched_structure_centroid"] = _feature_collection([matched_structure_centroid])
            rings, _assumptions = compute_structure_rings(fp_result.footprint)
            ring_features = _ring_zone_features_from_geometry(rings, basis="building_footprint")
        else:
            if fp_result:
                structure_match_status = str(fp_result.match_status or "none")
                structure_match_method = fp_result.match_method
                structure_match_confidence = float(fp_result.confidence or 0.0)
                structure_match_distance_m = (
                    float(fp_result.match_distance_m) if fp_result.match_distance_m is not None else None
                )
                candidate_structure_count = int(fp_result.candidate_count or 0)
                structure_match_candidates = list(fp_result.candidate_summaries or [])
            limitations.append(
                "Structure footprint unavailable; defensible-space zones are shown using point-proxy geometry."
            )
            if structure_match_status == "ambiguous":
                limitations.append(
                    "Multiple nearby structures were similarly plausible, so the map marker uses the geocoded address point."
                )
            ring_features = _build_proxy_rings(lat=anchor_lat, lon=anchor_lon)
    else:
        if not _geo_ready():
            limitations.append("Geospatial dependencies unavailable; map uses point-only rendering.")
        elif not footprint_paths:
            limitations.append("Building footprint source not configured; defensible-space zones use point-proxy geometry.")
            structure_match_status = "provider_unavailable"
        ring_features = _build_proxy_rings(lat=anchor_lat, lon=anchor_lon)

    if matched_structure_centroid:
        layers.append(
            AssessmentMapLayer(
                layer_key="matched_structure_centroid",
                display_name="Matched Structure Centroid",
                available=True,
                default_visible=False,
                description="Centroid of the matched structure footprint (WGS84).",
                legend_label="Matched structure centroid",
                geometry_type="point",
                feature_count=1,
            )
        )
    else:
        layers.append(
            AssessmentMapLayer(
                layer_key="matched_structure_centroid",
                display_name="Matched Structure Centroid",
                available=False,
                default_visible=False,
                description="Centroid of the matched structure footprint (WGS84).",
                legend_label="Matched structure centroid",
                geometry_type="point",
                feature_count=0,
                reason_unavailable="No matched structure footprint centroid was available.",
            )
        )

    display_point_source = str(
        property_ctx.get("display_point_source")
        or result.display_point_source
        or display_point_source
        or "property_anchor_point"
    )
    if display_point_source == "matched_structure_centroid" and matched_structure_centroid is not None:
        selected_display_feature = matched_structure_centroid
    else:
        display_point_source = "property_anchor_point"
        selected_display_feature = property_anchor_point_feature

    property_geometry = dict(selected_display_feature.get("geometry") or {})
    assessed_display_point_feature = {
        "type": "Feature",
        "properties": {
            "label": "Assessed property display point",
            "source": display_point_source,
            "assessment_id": result.assessment_id,
            "address": result.address,
            "crs": "EPSG:4326",
        },
        "geometry": property_geometry,
    }

    data["assessed_property_display_point"] = _feature_collection([assessed_display_point_feature])
    layers.append(
        AssessmentMapLayer(
            layer_key="assessed_property_display_point",
            display_name="Assessed Property Display Point",
            available=True,
            default_visible=True,
            description="Main marker used in the homeowner map. Uses matched structure centroid only for high-confidence matches.",
            legend_label="Assessed property",
            geometry_type="point",
            feature_count=1,
        )
    )
    property_point = _feature_collection(
        [
            {
                "type": "Feature",
                "properties": {
                    "label": "Assessed property",
                    "assessment_id": result.assessment_id,
                    "address": result.address,
                    "source": display_point_source,
                    "crs": "EPSG:4326",
                },
                "geometry": property_geometry,
            }
        ]
    )
    data["property_point"] = property_point
    layers.append(
        AssessmentMapLayer(
            layer_key="property_point",
            display_name="Property",
            available=True,
            default_visible=True,
            description=(
                "Backward-compatible alias of assessed property display point."
            ),
            legend_label="Assessed property",
            geometry_type="point",
            feature_count=1,
        )
    )

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
    if footprint_features and structure_geometry_source in {"user_selected", "user_modified"}:
        data["user_selected_structure"] = _feature_collection(footprint_features)
    layers.append(
        AssessmentMapLayer(
            layer_key="user_selected_structure",
            display_name="User-Selected Structure",
            available=bool(footprint_features and structure_geometry_source in {"user_selected", "user_modified"}),
            default_visible=bool(footprint_features and structure_geometry_source in {"user_selected", "user_modified"}),
            description="Structure footprint confirmed by user selection.",
            legend_label="Confirmed structure",
            geometry_type="polygon",
            feature_count=len(footprint_features) if structure_geometry_source in {"user_selected", "user_modified"} else 0,
            reason_unavailable=(
                None
                if (footprint_features and structure_geometry_source in {"user_selected", "user_modified"})
                else "No user-selected structure footprint is active for this assessment."
            ),
        )
    )
    if footprint_features and structure_geometry_source == "auto_detected":
        data["auto_detected_structure"] = _feature_collection(footprint_features)
    layers.append(
        AssessmentMapLayer(
            layer_key="auto_detected_structure",
            display_name="Auto-Detected Structure",
            available=bool(footprint_features and structure_geometry_source == "auto_detected"),
            default_visible=False,
            description="Automatically matched structure footprint before user confirmation.",
            legend_label="Auto-detected structure",
            geometry_type="polygon",
            feature_count=len(footprint_features) if structure_geometry_source == "auto_detected" else 0,
            reason_unavailable=(
                None
                if (footprint_features and structure_geometry_source == "auto_detected")
                else "No auto-detected structure footprint is active for this assessment."
            ),
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

    map_bbox = _viewport_bbox(anchor_lat, anchor_lon, radius_m=2500.0)

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
            bbox_bounds=_viewport_bbox(anchor_lat, anchor_lon, radius_m=800.0),
            max_features=250,
            clip=False,
        )

    if nearby_structures:
        data["nearby_structures"] = _feature_collection(nearby_structures)
        data["selectable_structure_footprints"] = _feature_collection(nearby_structures)
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
    layers.append(
        AssessmentMapLayer(
            layer_key="selectable_structure_footprints",
            display_name="Selectable Structures",
            available=bool(nearby_structures),
            default_visible=False,
            description="Nearby structure footprints available for manual home selection.",
            legend_label="Selectable structure footprints",
            geometry_type="polygon",
            feature_count=len(nearby_structures),
            reason_unavailable=(
                None if nearby_structures else "No selectable structure footprints were available near this location."
            ),
        )
    )

    # Include defensible-space analysis limitation notes if present.
    for note in list(result.defensible_space_limitations_summary or [])[:4]:
        if note not in limitations:
            limitations.append(str(note))
    for note in alignment_notes[:3]:
        if note not in limitations:
            limitations.append(note)

    center_lon, center_lat = anchor_lon, anchor_lat
    if isinstance(property_geometry.get("coordinates"), list) and len(property_geometry["coordinates"]) >= 2:
        maybe_lon = property_geometry["coordinates"][0]
        maybe_lat = property_geometry["coordinates"][1]
        normalized_center = _canonical_wgs84_lon_lat(maybe_lon, maybe_lat)
        if normalized_center is not None:
            center_lon, center_lat = normalized_center

    return AssessmentMapPayload(
        assessment_id=result.assessment_id,
        center={"latitude": float(center_lat), "longitude": float(center_lon)},
        resolved_region_id=result.resolved_region_id or str((result.property_level_context or {}).get("region_id") or "") or None,
        coverage_available=bool(result.coverage_available),
        basis_geometry_type=basis_geometry_type,
        geocode_provider=geocode_provider,
        geocode_source_name=geocode_provider,
        geocoded_address=geocoded_address,
        geocode_location_type=geocode_location_type,
        geocode_precision=geocode_precision,
        property_anchor_point=property_anchor_point_feature,
        property_anchor_source=property_anchor_source,
        property_anchor_precision=property_anchor_precision,
        assessed_property_display_point=assessed_display_point_feature,
        parcel_address_point=parcel_address_point_feature,
        parcel_polygon=parcel_polygon_feature,
        parcel_id=parcel_id,
        parcel_lookup_method=parcel_lookup_method,
        parcel_lookup_distance_m=parcel_lookup_distance_m,
        parcel_source_name=parcel_source_name,
        parcel_source_vintage=parcel_source_vintage,
        footprint_source_name=footprint_source_name,
        footprint_source_vintage=footprint_source_vintage,
        source_conflict_flag=source_conflict_flag,
        alignment_notes=alignment_notes[:6],
        structure_match_status=structure_match_status,
        structure_match_method=structure_match_method,
        matched_structure_id=matched_structure_id,
        structure_match_confidence=structure_match_confidence,
        structure_match_distance_m=structure_match_distance_m,
        candidate_structure_count=candidate_structure_count,
        display_point_source=display_point_source,
        geocoded_address_point=geocoded_address_point,
        matched_structure_centroid=matched_structure_centroid,
        matched_structure_footprint=footprint_features[0] if footprint_features else None,
        layers=layers,
        data=data,
        limitations=limitations[:8],
        metadata={
            "geometry_contract": {
                "crs": "EPSG:4326",
                "coordinate_order": "[longitude, latitude]",
                "display_point_source": display_point_source,
            },
            "geocoding": {
                "provider": geocode_provider,
                "matched_address": geocoded_address,
                "geocode_location_type": geocode_location_type,
                "geocode_precision": geocode_precision,
            },
            "property_anchor": {
                "source": property_anchor_source,
                "precision": property_anchor_precision,
                "parcel_id": parcel_id,
                "parcel_lookup_method": parcel_lookup_method,
                "parcel_lookup_distance_m": parcel_lookup_distance_m,
                "source_conflict_flag": source_conflict_flag,
                "alignment_notes": alignment_notes[:6],
                "parcel_source_name": parcel_source_name,
                "parcel_source_vintage": parcel_source_vintage,
            },
            "structure_match": {
                "status": structure_match_status,
                "method": structure_match_method,
                "geometry_source": structure_geometry_source,
                "matched_structure_id": matched_structure_id,
                "confidence": structure_match_confidence,
                "distance_m": structure_match_distance_m,
                "candidate_count": candidate_structure_count,
                "max_match_distance_m": max_match_distance_m,
                "ambiguity_gap_m": ambiguity_gap_m,
                "candidate_summaries": structure_match_candidates[:6] if structure_match_candidates else [],
                "footprint_source_name": footprint_source_name,
                "footprint_source_vintage": footprint_source_vintage,
            },
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
