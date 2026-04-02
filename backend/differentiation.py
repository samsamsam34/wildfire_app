from __future__ import annotations

from typing import Any

DIRECT_SOURCE_TYPES = {"observed", "footprint_derived", "user_provided"}
PROXY_SOURCE_TYPES = {"public_record_inferred", "heuristic"}
DEFAULTED_SOURCE_TYPES = {"missing"}

PROPERTY_SPECIFIC_TRACKED_FIELDS = {
    "roof_type",
    "vent_type",
    "defensible_space_ft",
    "construction_year",
    "zone_0_5_ft",
    "zone_5_30_ft",
    "zone_30_100_ft",
    "near_structure_vegetation_0_5_pct",
    "canopy_adjacency_proxy_pct",
    "vegetation_continuity_proxy_pct",
    "nearest_high_fuel_patch_distance_ft",
}

REGIONAL_TRACKED_FIELDS = {
    "burn_probability",
    "wildfire_hazard",
    "slope",
    "fuel_model",
    "canopy_cover",
    "historic_fire_distance",
    "wildland_distance",
    "moisture",
}

STRUCTURE_VULNERABILITY_FIELDS = {
    "roof_type",
    "vent_type",
    "defensible_space_ft",
}
NEARBY_HOME_COMPARISON_CONFIDENCE_THRESHOLD = 45.0


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_type(meta: Any) -> str:
    if isinstance(meta, dict):
        return str(meta.get("source_type") or "").strip().lower()
    return str(getattr(meta, "source_type", "") or "").strip().lower()


def should_trigger_nearby_home_comparison_safeguard(
    differentiation_mode: str | None,
    neighborhood_differentiation_confidence: float | None,
) -> bool:
    mode = str(differentiation_mode or "").strip().lower()
    try:
        confidence = float(
            0.0
            if neighborhood_differentiation_confidence is None
            else neighborhood_differentiation_confidence
        )
    except (TypeError, ValueError):
        confidence = 0.0
    return (
        mode == "mostly_regional"
        and confidence <= float(NEARBY_HOME_COMPARISON_CONFIDENCE_THRESHOLD)
    )


def build_differentiation_snapshot(
    *,
    feature_coverage_summary: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    property_level_context: dict[str, Any] | None = None,
    environmental_layer_status: dict[str, Any] | None = None,
    fallback_weight_fraction: float | None = None,
    missing_inputs: list[str] | None = None,
    inferred_inputs: list[str] | None = None,
    input_source_metadata: dict[str, Any] | None = None,
    fallback_decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    coverage = dict(feature_coverage_summary or {})
    preflight = dict(preflight or {})
    property_level_context = dict(property_level_context or {})
    environmental_layer_status = dict(environmental_layer_status or {})
    source_meta = dict(input_source_metadata or {})
    fallback_decisions = list(fallback_decisions or [])

    missing_set = {str(v).strip() for v in list(missing_inputs or []) if str(v).strip()}
    inferred_set = {str(v).strip() for v in list(inferred_inputs or []) if str(v).strip()}

    footprint_available = bool(
        coverage.get("building_footprint_available")
        or property_level_context.get("footprint_used")
    )
    parcel_available = bool(
        coverage.get("parcel_polygon_available")
        or isinstance(property_level_context.get("parcel_geometry"), dict)
    )
    near_structure_available = bool(coverage.get("near_structure_vegetation_available"))
    naip_feature_available = bool(
        str(property_level_context.get("naip_feature_source") or "").strip().lower()
        == "prepared_region_naip"
        or bool(property_level_context.get("naip_feature_artifact_path"))
    )
    hazard_available = bool(coverage.get("hazard_severity_available"))
    burn_available = bool(coverage.get("burn_probability_available"))

    geometry_basis = str(
        preflight.get("geometry_basis")
        or property_level_context.get("fallback_mode")
        or "geocode_point"
    ).strip().lower()
    point_sampling = geometry_basis in {"point_based", "geocode_point", "point", "point_proxy"}
    hazard_or_burn_proxied = not (hazard_available and burn_available)

    property_specific_fields: set[str] = set()
    proxy_fields: set[str] = set()
    defaulted_fields: set[str] = set()
    regional_fields: set[str] = set()

    for field in PROPERTY_SPECIFIC_TRACKED_FIELDS:
        source_type = _source_type(source_meta.get(field))
        if source_type in DIRECT_SOURCE_TYPES:
            property_specific_fields.add(field)
        elif source_type in PROXY_SOURCE_TYPES:
            proxy_fields.add(field)
        elif source_type in DEFAULTED_SOURCE_TYPES:
            defaulted_fields.add(field)
        else:
            # Fall back to runtime context value presence for legacy payloads where
            # source metadata for every field is not guaranteed.
            if property_level_context.get(field) is not None:
                property_specific_fields.add(field)
            else:
                defaulted_fields.add(field)

    for field in REGIONAL_TRACKED_FIELDS:
        source_type = _source_type(source_meta.get(field))
        if source_type in DIRECT_SOURCE_TYPES | PROXY_SOURCE_TYPES:
            regional_fields.add(field)
        elif source_type in DEFAULTED_SOURCE_TYPES:
            defaulted_fields.add(field)
        if source_type in PROXY_SOURCE_TYPES:
            proxy_fields.add(field)

    # Environmental status is a hard guardrail even when field-level metadata is sparse.
    if str(environmental_layer_status.get("hazard") or "").strip().lower() != "ok":
        proxy_fields.add("wildfire_hazard")
    if str(environmental_layer_status.get("burn_probability") or "").strip().lower() != "ok":
        proxy_fields.add("burn_probability")

    structure_default_count = 0
    for field in STRUCTURE_VULNERABILITY_FIELDS:
        source_type = _source_type(source_meta.get(field))
        if field in missing_set or source_type in DEFAULTED_SOURCE_TYPES:
            structure_default_count += 1
        elif field in inferred_set or source_type in PROXY_SOURCE_TYPES:
            structure_default_count += 1

    if any(str((row or {}).get("fallback_type") or "") == "point_based_context" for row in fallback_decisions):
        point_sampling = True
    if any(str((row or {}).get("fallback_type") or "") == "layer_proxy" for row in fallback_decisions):
        hazard_or_burn_proxied = True

    property_specific_feature_count = len(property_specific_fields)
    proxy_feature_count = len(proxy_fields)
    defaulted_feature_count = len(defaulted_fields)
    regional_feature_count = len(regional_fields)

    local_signal_ratio = (
        property_specific_feature_count
        / max(1.0, float(property_specific_feature_count + regional_feature_count))
    )
    score = 20.0
    score += 30.0 if footprint_available else 0.0
    score += 15.0 if parcel_available else 0.0
    score += 15.0 if naip_feature_available else 0.0
    score += 8.0 if near_structure_available else 0.0
    score += 6.0 if (hazard_available and burn_available) else 0.0
    score += max(0.0, min(20.0, local_signal_ratio * 20.0))
    score += min(10.0, property_specific_feature_count * 1.5)

    score -= min(12.0, proxy_feature_count * 1.8)
    score -= min(14.0, defaulted_feature_count * 1.6)
    score -= min(30.0, _to_float(fallback_weight_fraction) * 38.0)

    if point_sampling:
        score -= 16.0
    if not near_structure_available:
        score -= 12.0
    if hazard_or_burn_proxied:
        score -= 12.0
    if structure_default_count >= 3:
        score -= 14.0
    elif structure_default_count >= 1:
        score -= 7.0

    if (
        footprint_available
        and (parcel_available or naip_feature_available)
        and not point_sampling
        and not hazard_or_burn_proxied
        and property_specific_feature_count >= 3
    ):
        score = max(score, 72.0)
    elif (
        footprint_available
        and not point_sampling
        and near_structure_available
        and property_specific_feature_count >= 2
    ):
        score = max(score, 65.0)

    if (not footprint_available) and (not parcel_available):
        score = min(score, 35.0)
    if point_sampling and hazard_or_burn_proxied:
        score = min(score, 32.0)
    if point_sampling and structure_default_count >= 2:
        score = min(score, 28.0)

    differentiation_confidence = round(max(0.0, min(100.0, score)), 1)

    if (
        differentiation_confidence >= 70.0
        and property_specific_feature_count >= 4
        and (footprint_available or parcel_available)
        and not point_sampling
    ):
        mode = "highly_local"
    elif differentiation_confidence >= 40.0:
        mode = "mixed"
    else:
        mode = "mostly_regional"

    if (not footprint_available) and (not parcel_available) and point_sampling:
        mode = "mostly_regional"

    notes: list[str] = []
    if not footprint_available:
        notes.append("Building footprint is missing; structure-specific differentiation is reduced.")
    if not parcel_available:
        notes.append("Parcel geometry is missing; parcel-level differentiation is reduced.")
    if point_sampling:
        notes.append("Near-structure vegetation is being sampled from a point proxy rather than full structure geometry.")
    if not naip_feature_available:
        notes.append("NAIP near-structure imagery features are unavailable; local differentiation relies on fallback vegetation context.")
    if hazard_or_burn_proxied:
        notes.append("Hazard and/or burn context relies on proxy/partial regional layers.")
    if structure_default_count >= 1:
        notes.append("Structure vulnerability still depends on defaulted or inferred home attributes.")

    if mode == "highly_local":
        notes.append("This assessment is primarily driven by property-level observed inputs.")
    elif mode == "mixed":
        notes.append("This assessment blends property-level and regional/proxy inputs.")
    else:
        notes.append("This assessment is mostly regional because property-level evidence is limited.")

    return {
        "differentiation_mode": mode,
        "property_specific_feature_count": property_specific_feature_count,
        "proxy_feature_count": proxy_feature_count,
        "defaulted_feature_count": defaulted_feature_count,
        "regional_feature_count": regional_feature_count,
        "local_vs_regional_feature_ratio": round(max(0.0, min(1.0, local_signal_ratio)), 3),
        "local_differentiation_score": differentiation_confidence,
        "neighborhood_differentiation_confidence": differentiation_confidence,
        "legacy_differentiation_mode": (
            "property_specific" if mode == "highly_local" else mode
        ),
        "notes": notes[:8],
    }
