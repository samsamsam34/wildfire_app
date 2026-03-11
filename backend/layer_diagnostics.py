from __future__ import annotations

from pathlib import Path
from typing import Any


CORE_REQUIRED_LAYERS = {"dem", "slope", "fuel", "canopy", "fire_perimeters", "building_footprints"}

LAYER_SPECS: dict[str, dict[str, Any]] = {
    "dem": {"display_name": "Digital Elevation Model", "required_for": ["site_hazard", "home_ignition_vulnerability"]},
    "slope": {"display_name": "Slope", "required_for": ["site_hazard"]},
    "fuel": {"display_name": "Fuel Model", "required_for": ["site_hazard", "home_ignition_vulnerability"]},
    "canopy": {"display_name": "Canopy Cover", "required_for": ["site_hazard", "home_ignition_vulnerability"]},
    "fire_perimeters": {"display_name": "Historical Fire Perimeters", "required_for": ["site_hazard", "insurance_readiness"]},
    "building_footprints_overture": {
        "display_name": "Overture Building Footprints",
        "required_for": ["home_ignition_vulnerability", "insurance_readiness"],
    },
    "building_footprints": {"display_name": "Building Footprints", "required_for": ["home_ignition_vulnerability", "insurance_readiness"]},
    "address_points": {"display_name": "Address/Parcel Points", "required_for": ["home_ignition_vulnerability"]},
    "parcels": {"display_name": "Parcel Polygons", "required_for": ["home_ignition_vulnerability"]},
    "whp": {"display_name": "WHP Hazard Raster", "required_for": ["site_hazard"]},
    "mtbs_severity": {"display_name": "MTBS Severity Raster", "required_for": ["site_hazard"]},
    "gridmet_dryness": {"display_name": "gridMET Dryness", "required_for": ["site_hazard"]},
    "naip_imagery": {"display_name": "NAIP Imagery", "required_for": ["home_ignition_vulnerability"]},
    "naip_structure_features": {
        "display_name": "NAIP Structure Features",
        "required_for": ["home_ignition_vulnerability", "insurance_readiness"],
    },
    "roads": {"display_name": "Road Network", "required_for": ["home_ignition_vulnerability"]},
    "fema_structures": {"display_name": "Optional FEMA Structures", "required_for": ["home_ignition_vulnerability"]},
    "neighbor_structures": {"display_name": "Neighboring Structures Context", "required_for": ["home_ignition_vulnerability"]},
}


OPEN_DATA_KEYS = {
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "roads",
    "fema_structures",
    "building_footprints_overture",
    "naip_imagery",
    "naip_structure_features",
}


def _source_type_for_layer(layer_key: str, path: str, region_context: dict[str, Any]) -> str:
    if layer_key == "slope":
        return "derived"
    if layer_key in OPEN_DATA_KEYS:
        return "open_data"
    region_dir = ""
    manifest_path = str(region_context.get("manifest_path") or "")
    if manifest_path:
        region_dir = str(Path(manifest_path).parent)
    if region_context.get("region_status") == "prepared" and region_dir and str(path).startswith(region_dir):
        return "prepared_region"
    return "runtime_env"


def initialize_layer_audit(runtime_paths: dict[str, str], region_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    key_map = {
        "dem": "dem",
        "slope": "slope",
        "fuel": "fuel",
        "canopy": "canopy",
        "fire_perimeters": "perimeters",
        "building_footprints_overture": "footprints_overture",
        "building_footprints": "footprints",
        "address_points": "address_points",
        "parcels": "parcels",
        "whp": "whp",
        "mtbs_severity": "mtbs_severity",
        "gridmet_dryness": "gridmet_dryness",
        "naip_imagery": "naip_imagery",
        "naip_structure_features": "naip_structure_features",
        "roads": "roads",
        "fema_structures": "fema_structures",
    }
    audit: dict[str, dict[str, Any]] = {}
    for layer_key, runtime_key in key_map.items():
        path = str(runtime_paths.get(runtime_key) or "")
        configured = bool(path)
        present = bool(path and Path(path).exists())
        status = "not_configured"
        failure_reason = None
        if configured and not present:
            status = "missing_file"
            failure_reason = "Configured path does not exist."
        elif configured and present:
            status = "partial"
        spec = LAYER_SPECS[layer_key]
        audit[layer_key] = {
            "layer_key": layer_key,
            "display_name": spec["display_name"],
            "required_for": list(spec["required_for"]),
            "configured": configured,
            "present_in_region": present,
            "sample_attempted": False,
            "sample_succeeded": False,
            "coverage_status": status,
            "source_type": _source_type_for_layer(layer_key, path, region_context),
            "source_path": path or None,
            "raw_value_preview": None,
            "failure_reason": failure_reason,
            "notes": [],
        }
    # Derived neighbor structures row (not file-backed).
    spec = LAYER_SPECS["neighbor_structures"]
    audit["neighbor_structures"] = {
        "layer_key": "neighbor_structures",
        "display_name": spec["display_name"],
        "required_for": list(spec["required_for"]),
        "configured": True,
        "present_in_region": True,
        "sample_attempted": False,
        "sample_succeeded": False,
        "coverage_status": "partial",
        "source_type": "derived",
        "source_path": None,
        "raw_value_preview": None,
        "failure_reason": None,
        "notes": [],
    }
    return audit


def update_layer_audit(
    audit: dict[str, dict[str, Any]],
    layer_key: str,
    *,
    sample_attempted: bool | None = None,
    sample_succeeded: bool | None = None,
    coverage_status: str | None = None,
    raw_value_preview: Any | None = None,
    failure_reason: str | None = None,
    note: str | None = None,
) -> None:
    row = audit.get(layer_key)
    if not row:
        return
    if sample_attempted is not None:
        row["sample_attempted"] = bool(sample_attempted)
    if sample_succeeded is not None:
        row["sample_succeeded"] = bool(sample_succeeded)
    if coverage_status:
        row["coverage_status"] = coverage_status
    if raw_value_preview is not None:
        row["raw_value_preview"] = raw_value_preview
    if failure_reason:
        row["failure_reason"] = failure_reason
    if note:
        notes = row.setdefault("notes", [])
        if note not in notes:
            notes.append(note)


def summarize_layer_audit(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(audit_rows)
    observed_count = sum(1 for row in audit_rows if row.get("coverage_status") == "observed")
    partial_count = sum(1 for row in audit_rows if row.get("coverage_status") == "partial")
    fallback_count = sum(1 for row in audit_rows if row.get("coverage_status") == "fallback_used")
    failed_count = sum(
        1
        for row in audit_rows
        if row.get("coverage_status") in {"missing_file", "outside_extent", "sampling_failed"}
    )
    not_configured_count = sum(1 for row in audit_rows if row.get("coverage_status") == "not_configured")

    critical_missing_layers: list[str] = []
    for row in audit_rows:
        key = str(row.get("layer_key") or "")
        if key not in CORE_REQUIRED_LAYERS:
            continue
        if row.get("coverage_status") not in {"observed", "partial"}:
            critical_missing_layers.append(key)

    recommended_actions: list[str] = []
    for row in audit_rows:
        status = str(row.get("coverage_status") or "")
        key = str(row.get("layer_key") or "")
        if status == "missing_file":
            recommended_actions.append(f"{key} file is missing from runtime paths; verify prepared-region manifest files.")
        elif status == "not_configured":
            recommended_actions.append(f"{key} is not configured; add layer path or treat as optional.")
        elif status == "outside_extent":
            recommended_actions.append(f"{key} does not cover this property location; prepare data for the correct region extent.")
        elif status == "sampling_failed":
            recommended_actions.append(f"{key} sampling failed; validate CRS, raster/vector integrity, and read permissions.")
        elif status == "fallback_used":
            recommended_actions.append(f"{key} used fallback behavior; improve source coverage for higher-confidence scoring.")

    dedup_actions = []
    seen = set()
    for action in recommended_actions:
        if action not in seen:
            dedup_actions.append(action)
            seen.add(action)

    return {
        "total_layers_checked": total,
        "observed_count": observed_count,
        "partial_count": partial_count,
        "fallback_count": fallback_count,
        "failed_count": failed_count,
        "not_configured_count": not_configured_count,
        "critical_missing_layers": sorted(set(critical_missing_layers)),
        "recommended_actions": dedup_actions[:10],
    }
