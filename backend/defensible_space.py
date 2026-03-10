from __future__ import annotations

from typing import Any


ZONE_DEFINITIONS: list[tuple[str, str]] = [
    ("zone_0_5_ft", "0-5 ft"),
    ("zone_5_30_ft", "5-30 ft"),
    ("zone_30_100_ft", "30-100 ft"),
    ("zone_100_300_ft", "100-300 ft"),
]


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _zone_metrics(rings: dict[str, Any], zone_key: str) -> dict[str, Any]:
    ring_alias = zone_key.replace("zone_", "ring_")
    value = rings.get(zone_key)
    if not isinstance(value, dict):
        value = rings.get(ring_alias)
    return value if isinstance(value, dict) else {}


def _risk_level(zone_key: str, vegetation_density: float | None) -> str:
    if vegetation_density is None:
        return "unknown"
    high_thresholds = {
        "zone_0_5_ft": 55.0,
        "zone_5_30_ft": 60.0,
        "zone_30_100_ft": 65.0,
        "zone_100_300_ft": 70.0,
    }
    low_thresholds = {
        "zone_0_5_ft": 30.0,
        "zone_5_30_ft": 35.0,
        "zone_30_100_ft": 40.0,
        "zone_100_300_ft": 45.0,
    }
    if vegetation_density >= high_thresholds.get(zone_key, 65.0):
        return "high"
    if vegetation_density <= low_thresholds.get(zone_key, 35.0):
        return "low"
    return "moderate"


def _coverage_note(layer_coverage_audit: list[dict[str, Any]] | None, layer_key: str) -> str | None:
    if not isinstance(layer_coverage_audit, list):
        return None
    for row in layer_coverage_audit:
        if not isinstance(row, dict):
            continue
        if str(row.get("layer_key") or "") != layer_key:
            continue
        status = str(row.get("coverage_status") or "")
        if status == "observed":
            return None
        reason = str(row.get("failure_reason") or "").strip()
        if reason:
            return f"{layer_key}: {reason}"
        if status:
            return f"{layer_key}: {status}"
    return None


def build_defensible_space_analysis(
    *,
    property_level_context: dict[str, Any],
    layer_coverage_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rings = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    rings = rings if isinstance(rings, dict) else {}

    footprint_used = bool(property_level_context.get("footprint_used")) if isinstance(property_level_context, dict) else False
    fallback_mode = str(property_level_context.get("fallback_mode") or "point_based") if isinstance(property_level_context, dict) else "point_based"
    if footprint_used:
        basis_geometry_type = "building_footprint"
        basis_quality = "observed"
    elif rings:
        basis_geometry_type = "point_proxy"
        basis_quality = "derived_proxy"
    else:
        basis_geometry_type = "point_proxy" if fallback_mode == "point_based" else "unknown"
        basis_quality = "unavailable"

    zones: dict[str, dict[str, Any]] = {}
    observed_count = 0
    proxy_count = 0
    unavailable_count = 0

    for zone_key, label in ZONE_DEFINITIONS:
        metrics = _zone_metrics(rings, zone_key)
        vegetation_density = _to_float(metrics.get("vegetation_density"))
        coverage_pct = _to_float(metrics.get("coverage_pct"))
        fuel_presence = _to_float(metrics.get("fuel_presence_proxy"))
        canopy_mean = _to_float(metrics.get("canopy_mean"))
        canopy_max = _to_float(metrics.get("canopy_max"))
        hazardous_pct = max([v for v in [coverage_pct, fuel_presence] if v is not None], default=None)

        if vegetation_density is None and coverage_pct is None and fuel_presence is None:
            zone_status = "unavailable"
            evidence_status = "missing"
            unavailable_count += 1
        elif footprint_used:
            zone_status = "observed"
            evidence_status = "observed"
            observed_count += 1
        else:
            zone_status = "derived_proxy"
            evidence_status = "inferred"
            proxy_count += 1

        zones[zone_key] = {
            "distance_band_ft": label,
            "vegetation_density": vegetation_density,
            "coverage_pct": coverage_pct,
            "fuel_presence_proxy": fuel_presence,
            "hazardous_vegetation_pct": hazardous_pct,
            "canopy_mean": canopy_mean,
            "canopy_max": canopy_max,
            "risk_level": _risk_level(zone_key, vegetation_density),
            "zone_status": zone_status,
            "evidence_status": evidence_status,
        }

    zone_0_5 = zones["zone_0_5_ft"]
    zone_5_30 = zones["zone_5_30_ft"]
    zone_30_100 = zones["zone_30_100_ft"]
    mitigation_flags = {
        "immediate_zone_clearance_needed": zone_0_5.get("risk_level") == "high",
        "intermediate_zone_thinning_needed": zone_5_30.get("risk_level") == "high",
        "extended_zone_fuel_reduction_needed": zone_30_100.get("risk_level") == "high",
    }

    limitations: list[str] = []
    if not footprint_used:
        if rings:
            limitations.append(
                "Building footprint was unavailable; defensible-space zones were approximated from point-based annulus sampling."
            )
        else:
            limitations.append("Near-structure vegetation zone metrics were unavailable for this property.")
    if unavailable_count > 0:
        limitations.append(f"{unavailable_count} zone(s) did not have enough vegetation/fuel evidence for direct analysis.")
    for key in ("canopy", "fuel", "building_footprints"):
        note = _coverage_note(layer_coverage_audit, key)
        if note:
            limitations.append(note)

    if unavailable_count == len(ZONE_DEFINITIONS):
        analysis_status = "unavailable"
    elif unavailable_count == 0 and footprint_used:
        analysis_status = "complete"
    else:
        analysis_status = "partial"

    if mitigation_flags["immediate_zone_clearance_needed"]:
        summary = "Immediate vegetation/fuel pressure is elevated within 0-5 feet of the structure."
    elif mitigation_flags["intermediate_zone_thinning_needed"]:
        summary = "Vegetation pressure is elevated in the 5-30 foot defensible-space zone."
    elif mitigation_flags["extended_zone_fuel_reduction_needed"]:
        summary = "Vegetation/fuel loading is elevated in the 30-100 foot extended zone."
    elif analysis_status == "unavailable":
        summary = "Near-structure defensible-space vegetation analysis was unavailable."
    else:
        summary = "No severe near-structure vegetation hotspot was detected in the analyzed zones."

    return {
        "basis_geometry_type": basis_geometry_type,
        "basis_quality": basis_quality,
        "zones": zones,
        "nearest_vegetation_distance_ft": _to_float(property_level_context.get("nearest_vegetation_distance_ft")),
        "mitigation_flags": mitigation_flags,
        "data_quality": {
            "analysis_status": analysis_status,
            "observed_zone_count": observed_count,
            "proxy_zone_count": proxy_count,
            "unavailable_zone_count": unavailable_count,
            "limitations": limitations[:6],
            "contributing_layers": [k for k in ("canopy", "fuel", "building_footprints")],
        },
        "summary": summary,
    }


def build_top_near_structure_risk_drivers(defensible_space_analysis: dict[str, Any]) -> list[str]:
    zones = defensible_space_analysis.get("zones") if isinstance(defensible_space_analysis, dict) else None
    if not isinstance(zones, dict):
        return []
    drivers: list[str] = []
    if str((zones.get("zone_0_5_ft") or {}).get("risk_level")) == "high":
        drivers.append("vegetation/shrubs appear too close to the structure in the 0-5 ft zone")
    if str((zones.get("zone_5_30_ft") or {}).get("risk_level")) == "high":
        drivers.append("vegetation density is elevated in the 5-30 ft defensible-space zone")
    if str((zones.get("zone_30_100_ft") or {}).get("risk_level")) == "high":
        drivers.append("extended 30-100 ft zone fuel loading can increase fire spread pressure")
    nearest = _to_float(defensible_space_analysis.get("nearest_vegetation_distance_ft"))
    if nearest is not None and nearest <= 5.0:
        drivers.append("nearest meaningful vegetation appears within 5 feet of the home")
    return drivers[:3]


def build_prioritized_vegetation_actions(defensible_space_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    zones = defensible_space_analysis.get("zones") if isinstance(defensible_space_analysis, dict) else None
    if not isinstance(zones, dict):
        return []
    actions: list[dict[str, Any]] = []

    zone_0_5 = zones.get("zone_0_5_ft") or {}
    if str(zone_0_5.get("risk_level")) == "high":
        actions.append(
            {
                "title": "Create a noncombustible 0-5 ft zone",
                "explanation": "Remove combustible vegetation/mulch touching the structure perimeter.",
                "target_zone": "0-5 ft",
                "why_it_matters": "This is the most critical ignition pathway for direct flame and ember ignition.",
                "impact_category": "high",
                "priority": 1,
                "evidence_status": zone_0_5.get("evidence_status", "unknown"),
            }
        )

    zone_5_30 = zones.get("zone_5_30_ft") or {}
    if str(zone_5_30.get("risk_level")) == "high":
        actions.append(
            {
                "title": "Thin and prune vegetation in the 5-30 ft zone",
                "explanation": "Increase spacing and vertical separation between shrubs, grasses, and tree canopies.",
                "target_zone": "5-30 ft",
                "why_it_matters": "This zone controls flame contact and radiant heat exposure to the home envelope.",
                "impact_category": "high",
                "priority": 2,
                "evidence_status": zone_5_30.get("evidence_status", "unknown"),
            }
        )

    zone_30_100 = zones.get("zone_30_100_ft") or {}
    if str(zone_30_100.get("risk_level")) == "high":
        actions.append(
            {
                "title": "Reduce connected fuels in the 30-100 ft zone",
                "explanation": "Break up continuous vegetation/fuel pathways that can channel fire toward the structure.",
                "target_zone": "30-100 ft",
                "why_it_matters": "Lower fuel continuity reduces incoming fire intensity and ember production potential.",
                "impact_category": "medium",
                "priority": 3,
                "evidence_status": zone_30_100.get("evidence_status", "unknown"),
            }
        )

    actions.sort(key=lambda row: int(row.get("priority", 99)))
    return actions[:3]
