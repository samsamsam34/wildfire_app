from __future__ import annotations

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


def _contains_bbox(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _intersects_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


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
    matches = [
        c
        for c in coverages
        if _contains_point(
            (
                float(c["bounds"]["min_lon"]),
                float(c["bounds"]["min_lat"]),
                float(c["bounds"]["max_lon"]),
                float(c["bounds"]["max_lat"]),
            ),
            lat,
            lon,
        )
    ]
    if not matches:
        return {
            "covered": False,
            "region_id": None,
            "display_name": None,
            "manifest": None,
            "match_reason": "no_covering_region",
            "diagnostics": [
                f"No prepared region bounds contain point ({lat:.6f}, {lon:.6f}).",
                "Prepare a new region for this area.",
            ],
            "candidate_count": len(coverages),
        }
    chosen = sorted(matches, key=lambda c: (float(c["area_deg2"]), str(c["region_id"])))[0]
    return {
        "covered": True,
        "region_id": chosen["region_id"],
        "display_name": chosen["display_name"],
        "manifest": chosen["manifest"],
        "match_reason": "smallest_covering_region",
        "diagnostics": [],
        "candidate_count": len(matches),
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
