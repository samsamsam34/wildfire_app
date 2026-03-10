from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from backend.region_registry import list_prepared_regions


def _coerce_bounds(bounds: Any) -> tuple[float, float, float, float] | None:
    if isinstance(bounds, dict):
        keys = ("min_lon", "min_lat", "max_lon", "max_lat")
        if not all(k in bounds for k in keys):
            return None
        try:
            return (
                float(bounds["min_lon"]),
                float(bounds["min_lat"]),
                float(bounds["max_lon"]),
                float(bounds["max_lat"]),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            return tuple(float(v) for v in bounds)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _contains_point(bounds: tuple[float, float, float, float], lat: float, lon: float) -> bool:
    return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]


def _contains_point_with_tolerance(
    bounds: tuple[float, float, float, float],
    lat: float,
    lon: float,
    tolerance_m: float,
) -> bool:
    if tolerance_m <= 0:
        return _contains_point(bounds, lat, lon)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1.0, 111_320.0 * math.cos(math.radians(lat)))
    expand_lat = tolerance_m / meters_per_deg_lat
    expand_lon = tolerance_m / meters_per_deg_lon
    expanded = (
        bounds[0] - expand_lon,
        bounds[1] - expand_lat,
        bounds[2] + expand_lon,
        bounds[3] + expand_lat,
    )
    return _contains_point(expanded, lat, lon)


def _contains_bbox(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _intersects_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _distance_point_to_bounds_m(bounds: tuple[float, float, float, float], lat: float, lon: float) -> float:
    """Approximate point-to-bounds distance in meters for lightweight routing diagnostics."""
    clamped_lon = min(max(lon, bounds[0]), bounds[2])
    clamped_lat = min(max(lat, bounds[1]), bounds[3])
    # Degree-to-meter approximation; sufficient for boundary diagnostics.
    lat_rad = math.radians(clamped_lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1.0, 111_320.0 * math.cos(lat_rad))
    delta_lon_m = (lon - clamped_lon) * meters_per_deg_lon
    delta_lat_m = (lat - clamped_lat) * meters_per_deg_lat
    return float(math.hypot(delta_lon_m, delta_lat_m))


def list_region_coverages(regions_root: str | Path | None = None) -> list[dict[str, Any]]:
    manifests = list_prepared_regions(base_dir=str(regions_root) if regions_root else None)
    coverages: list[dict[str, Any]] = []
    for manifest in manifests:
        bounds = _coerce_bounds(manifest.get("bounds"))
        if bounds is None:
            continue
        coverages.append(
            {
                "region_id": str(manifest.get("region_id") or ""),
                "display_name": str(manifest.get("display_name") or manifest.get("region_id") or ""),
                "bounds": {
                    "min_lon": bounds[0],
                    "min_lat": bounds[1],
                    "max_lon": bounds[2],
                    "max_lat": bounds[3],
                },
                "area_deg2": _bounds_area(bounds),
                "manifest_path": manifest.get("_manifest_path"),
                "manifest": manifest,
            }
        )
    coverages.sort(key=lambda c: (float(c["area_deg2"]), str(c["region_id"])))
    return coverages


def find_region_for_point(lat: float, lon: float, regions_root: str | Path | None = None) -> dict[str, Any]:
    coverages = list_region_coverages(regions_root=regions_root)
    tolerance_raw = str(os.getenv("WF_REGION_EDGE_TOLERANCE_M", "0")).strip()
    try:
        edge_tolerance_m = max(0.0, float(tolerance_raw))
    except ValueError:
        edge_tolerance_m = 0.0
    matches = [
        c
        for c in coverages
        if _contains_point_with_tolerance(
            (
                float(c["bounds"]["min_lon"]),
                float(c["bounds"]["min_lat"]),
                float(c["bounds"]["max_lon"]),
                float(c["bounds"]["max_lat"]),
            ),
            lat,
            lon,
            edge_tolerance_m,
        )
    ]
    if not matches:
        nearest_region: dict[str, Any] | None = None
        nearest_distance_m: float | None = None
        for coverage in coverages:
            cb = (
                float(coverage["bounds"]["min_lon"]),
                float(coverage["bounds"]["min_lat"]),
                float(coverage["bounds"]["max_lon"]),
                float(coverage["bounds"]["max_lat"]),
            )
            candidate_distance = _distance_point_to_bounds_m(cb, lat, lon)
            if nearest_distance_m is None or candidate_distance < nearest_distance_m:
                nearest_distance_m = candidate_distance
                nearest_region = coverage
        boundary_note = (
            f"Nearest prepared region '{nearest_region['region_id']}' is ~{nearest_distance_m:.0f} m away."
            if nearest_region and nearest_distance_m is not None
            else "No prepared regions were found to compute boundary distance."
        )
        return {
            "covered": False,
            "region_id": None,
            "display_name": None,
            "manifest": None,
            "match_reason": "no_covering_region",
            "diagnostics": [
                f"No prepared region bounds contain point ({lat:.6f}, {lon:.6f}).",
                boundary_note,
                "Prepare a new region for this area.",
            ],
            "candidate_count": len(coverages),
            "nearest_region_id": nearest_region["region_id"] if nearest_region else None,
            "nearest_region_display_name": nearest_region["display_name"] if nearest_region else None,
            "region_distance_to_boundary_m": round(float(nearest_distance_m), 2)
            if nearest_distance_m is not None
            else None,
        }
    chosen = sorted(matches, key=lambda c: (float(c["area_deg2"]), str(c["region_id"])))[0]
    strict_match = _contains_point(
        (
            float(chosen["bounds"]["min_lon"]),
            float(chosen["bounds"]["min_lat"]),
            float(chosen["bounds"]["max_lon"]),
            float(chosen["bounds"]["max_lat"]),
        ),
        lat,
        lon,
    )
    return {
        "covered": True,
        "region_id": chosen["region_id"],
        "display_name": chosen["display_name"],
        "manifest": chosen["manifest"],
        "match_reason": "smallest_covering_region" if strict_match else "within_edge_tolerance",
        "diagnostics": []
        if strict_match
        else [f"Point is within {edge_tolerance_m:.1f} m edge tolerance for region bounds."],
        "candidate_count": len(matches),
        "edge_tolerance_m": edge_tolerance_m,
        "region_distance_to_boundary_m": 0.0,
        "nearest_region_id": chosen["region_id"],
    }


def find_region_for_bbox(bounds: dict[str, float], regions_root: str | Path | None = None) -> dict[str, Any]:
    requested = _coerce_bounds(bounds)
    if requested is None:
        return {
            "covered": False,
            "region_id": None,
            "display_name": None,
            "manifest": None,
            "match_reason": "invalid_bbox",
            "diagnostics": ["Requested bbox is invalid."],
            "candidate_count": 0,
        }

    coverages = list_region_coverages(regions_root=regions_root)
    containing = []
    intersecting = []
    for c in coverages:
        cb = (
            float(c["bounds"]["min_lon"]),
            float(c["bounds"]["min_lat"]),
            float(c["bounds"]["max_lon"]),
            float(c["bounds"]["max_lat"]),
        )
        if _contains_bbox(cb, requested):
            containing.append(c)
        elif _intersects_bbox(cb, requested):
            intersecting.append(c)

    if containing:
        chosen = sorted(containing, key=lambda c: (float(c["area_deg2"]), str(c["region_id"])))[0]
        return {
            "covered": True,
            "region_id": chosen["region_id"],
            "display_name": chosen["display_name"],
            "manifest": chosen["manifest"],
            "match_reason": "smallest_bbox_covering_region",
            "diagnostics": [],
            "candidate_count": len(containing),
        }

    diagnostics = [f"No prepared region fully covers bbox {requested}."]
    if intersecting:
        diagnostics.append(f"{len(intersecting)} region(s) intersect the bbox but do not fully cover it.")
    else:
        diagnostics.append("No prepared region intersects the requested bbox.")
    return {
        "covered": False,
        "region_id": None,
        "display_name": None,
        "manifest": None,
        "match_reason": "bbox_not_covered",
        "diagnostics": diagnostics,
        "candidate_count": len(intersecting),
    }
