from __future__ import annotations

import json
from pathlib import Path

from backend.data_prep.region_lookup import (
    find_region_for_bbox,
    find_region_for_point,
    list_region_coverages,
)


def _write_manifest(root: Path, region_id: str, bounds: dict[str, float]) -> None:
    region_dir = root / region_id
    region_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "region_id": region_id,
        "display_name": region_id.replace("_", " ").title(),
        "bounds": bounds,
        "files": {},
    }
    (region_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_find_region_for_point_prefers_smallest_covering_region(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "large_region",
        {"min_lon": -112.0, "min_lat": 45.0, "max_lon": -110.0, "max_lat": 47.0},
    )
    _write_manifest(
        tmp_path,
        "small_region",
        {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
    )

    match = find_region_for_point(lat=45.67, lon=-111.04, regions_root=tmp_path)
    assert match["covered"] is True
    assert match["region_id"] == "small_region"
    assert match["match_reason"] == "smallest_covering_region"


def test_find_region_for_point_uncovered_includes_diagnostics(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "bozeman_region",
        {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
    )
    match = find_region_for_point(lat=39.7392, lon=-104.9903, regions_root=tmp_path)
    assert match["covered"] is False
    assert match["region_id"] is None
    assert match["match_reason"] == "no_covering_region"
    assert match["diagnostics"]


def test_find_region_for_bbox_reports_intersection_without_full_coverage(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "partial_region",
        {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -111.0, "max_lat": 45.8},
    )
    query_bbox = {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.8, "max_lat": 45.8}
    match = find_region_for_bbox(query_bbox, regions_root=tmp_path)
    assert match["covered"] is False
    assert match["match_reason"] == "bbox_not_covered"
    assert any("intersect" in msg for msg in match["diagnostics"])


def test_list_region_coverages_returns_area_and_manifest_path(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "r1",
        {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -111.0, "max_lat": 45.7},
    )
    coverages = list_region_coverages(regions_root=tmp_path)
    assert len(coverages) == 1
    row = coverages[0]
    assert row["region_id"] == "r1"
    assert row["area_deg2"] > 0
    assert str(row["manifest_path"]).endswith("manifest.json")


def test_find_region_for_point_allows_small_edge_tolerance(monkeypatch, tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "winthrop_region",
        {"min_lon": -120.20, "min_lat": 48.45, "max_lon": -120.18, "max_lat": 48.48},
    )
    # Slightly outside north boundary.
    lat, lon = 48.48025, -120.19
    strict = find_region_for_point(lat=lat, lon=lon, regions_root=tmp_path)
    assert strict["covered"] is False

    monkeypatch.setenv("WF_REGION_EDGE_TOLERANCE_M", "40")
    tolerant = find_region_for_point(lat=lat, lon=lon, regions_root=tmp_path)
    assert tolerant["covered"] is True
    assert tolerant["region_id"] == "winthrop_region"
    assert tolerant["match_reason"] in {"within_edge_tolerance", "smallest_covering_region"}
    monkeypatch.delenv("WF_REGION_EDGE_TOLERANCE_M", raising=False)
