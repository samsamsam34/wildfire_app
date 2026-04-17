from __future__ import annotations

from pathlib import Path

import pytest

from backend.building_footprints import BuildingFootprintClient
from backend.parcel_resolution import ParcelResolutionClient

try:
    from shapely.geometry import Point as ShapelyPoint
except Exception:  # pragma: no cover - optional geospatial deps
    ShapelyPoint = None


def _geo_ready() -> bool:
    return ShapelyPoint is not None


@pytest.mark.skipif(not _geo_ready(), reason="Parcel/footprint intersection tests require shapely")
def test_parcel_intersection_path_selected_when_anchor_is_inside_parcel_fixture() -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "geometry" / "parcel_intersection"
    parcel_path = fixture_root / "parcel_polygons.geojson"
    footprint_path = fixture_root / "building_footprints.geojson"

    parcel_client = ParcelResolutionClient(parcel_paths=[str(parcel_path)], max_lookup_distance_m=30.0)
    anchor = ShapelyPoint(-113.98992, 46.87062)
    parcel_result = parcel_client.resolve_for_point(anchor_point=anchor)

    assert parcel_result.status == "matched"
    assert parcel_result.parcel_polygon is not None
    assert parcel_result.parcel_id == "MISSOULA-PARCEL-1001"

    footprint_client = BuildingFootprintClient(path=str(footprint_path), max_search_m=120.0)
    result = footprint_client.get_building_footprint(
        lat=46.87062,
        lon=-113.98992,
        parcel_polygon=parcel_result.parcel_polygon,
        anchor_precision="parcel_or_address_point",
    )

    assert result.found is True
    assert result.match_status == "matched"
    assert result.match_method == "parcel_intersection"
    assert result.matched_structure_id == "parcel_home_b"
