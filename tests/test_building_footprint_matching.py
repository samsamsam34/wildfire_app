from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.building_footprints import BuildingFootprintClient

try:
    from shapely.geometry import Point as ShapelyPoint, shape as shapely_shape
except Exception:  # pragma: no cover - optional geospatial deps
    ShapelyPoint = None
    shapely_shape = None


def _geo_ready() -> bool:
    return ShapelyPoint is not None and shapely_shape is not None


def _write_geojson(path: Path, features: list[dict]) -> str:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return str(path)


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_building_match_prefers_point_in_polygon(tmp_path: Path) -> None:
    footprints_path = _write_geojson(
        tmp_path / "footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "subject"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00020, 40.00020],
                        [-104.99980, 40.00020],
                        [-104.99980, 39.99980],
                        [-105.00020, 39.99980],
                        [-105.00020, 40.00020],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": "neighbor"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00060, 40.00010],
                        [-105.00030, 40.00010],
                        [-105.00030, 39.99980],
                        [-105.00060, 39.99980],
                        [-105.00060, 40.00010],
                    ]],
                },
            },
        ],
    )
    client = BuildingFootprintClient(path=footprints_path)
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is True
    assert result.match_status == "matched"
    assert result.match_method == "nearest_building_fallback"
    assert result.matched_structure_id == "subject"
    assert result.confidence < 0.9
    assert result.match_distance_m == 0.0
    assert result.candidate_count >= 1
    assert result.centroid is not None
    # Guard against accidental axis flips in centroid serialization.
    centroid_lat, centroid_lon = result.centroid
    assert 39.0 < centroid_lat < 41.0
    assert -106.0 < centroid_lon < -104.0


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_building_match_rejects_ambiguous_cross_street_candidates(tmp_path: Path) -> None:
    footprints_path = _write_geojson(
        tmp_path / "ambiguous_footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "north_house"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00012, 40.00022],
                        [-104.99990, 40.00022],
                        [-104.99990, 40.00008],
                        [-105.00012, 40.00008],
                        [-105.00012, 40.00022],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": "south_house"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00012, 39.99992],
                        [-104.99990, 39.99992],
                        [-104.99990, 39.99978],
                        [-105.00012, 39.99978],
                        [-105.00012, 39.99992],
                    ]],
                },
            },
        ],
    )
    client = BuildingFootprintClient(path=footprints_path, max_search_m=80.0)
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is False
    assert result.match_status == "ambiguous"
    assert result.match_method in {"nearest_building_fallback", "parcel_intersection"}
    assert result.candidate_count >= 2
    assert any("similarly plausible" in note.lower() for note in result.assumptions)


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_interpolated_like_road_point_does_not_force_distant_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_STRUCTURE_MATCH_MAX_DISTANCE_M", "12")
    footprints_path = _write_geojson(
        tmp_path / "distant_house.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "single_house"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00030, 40.00020],
                        [-105.00010, 40.00020],
                        [-105.00010, 40.00005],
                        [-105.00030, 40.00005],
                        [-105.00030, 40.00020],
                    ]],
                },
            },
        ],
    )
    client = BuildingFootprintClient(path=footprints_path, max_search_m=80.0)
    result = client.get_building_footprint(lat=39.99990, lon=-105.0)

    assert result.found is False
    assert result.match_status == "none"
    assert result.match_method == "nearest_building_fallback"
    assert result.match_distance_m is not None
    assert result.match_distance_m > 0
    assert any("too far" in note.lower() for note in result.assumptions)


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_interpolated_precision_can_match_when_within_relaxed_distance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_STRUCTURE_MATCH_MAX_DISTANCE_M", "12")
    footprints_path = _write_geojson(
        tmp_path / "interpolated_relaxed.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "road_offset_home"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00036, 40.00008],
                        [-105.00019, 40.00008],
                        [-105.00019, 39.99994],
                        [-105.00036, 39.99994],
                        [-105.00036, 40.00008],
                    ]],
                },
            }
        ],
    )
    client = BuildingFootprintClient(path=footprints_path, max_search_m=90.0)

    interpolated = client.get_building_footprint(
        lat=40.00000,
        lon=-105.0,
        anchor_precision="interpolated",
    )
    rooftop = client.get_building_footprint(
        lat=40.00000,
        lon=-105.0,
        anchor_precision="rooftop",
    )

    assert interpolated.found is True
    assert interpolated.match_status == "matched"
    assert any("expanded structure-match distance" in note.lower() for note in interpolated.assumptions)
    assert rooftop.found is False


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_parcel_overlap_bias_prefers_structure_on_parcel(tmp_path: Path) -> None:
    from shapely.geometry import Polygon

    footprints_path = _write_geojson(
        tmp_path / "parcel_biased_footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "on_parcel"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00010, 40.00008],
                        [-104.99995, 40.00008],
                        [-104.99995, 39.99992],
                        [-105.00010, 39.99992],
                        [-105.00010, 40.00008],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": "off_parcel"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00034, 40.00010],
                        [-105.00018, 40.00010],
                        [-105.00018, 39.99994],
                        [-105.00034, 39.99994],
                        [-105.00034, 40.00010],
                    ]],
                },
            },
        ],
    )
    parcel_polygon = Polygon(
        [
            (-105.00013, 40.00012),
            (-104.99990, 40.00012),
            (-104.99990, 39.99988),
            (-105.00013, 39.99988),
            (-105.00013, 40.00012),
        ]
    )
    client = BuildingFootprintClient(path=footprints_path, max_search_m=80.0)
    result = client.get_building_footprint(
        lat=40.0,
        lon=-105.00012,
        parcel_polygon=parcel_polygon,
        anchor_precision="parcel_or_address_point",
    )

    assert result.found is True
    assert result.match_status == "matched"
    assert result.match_method == "parcel_intersection"
    assert result.matched_structure_id == "on_parcel"
    assert result.confidence >= 0.85
    assert any("parcel" in note.lower() for note in result.assumptions)


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_structure_matching_prefers_overture_source_when_available(tmp_path: Path) -> None:
    overture_path = _write_geojson(
        tmp_path / "building_footprints_overture.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "overture-home"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00016, 40.00012],
                        [-104.99992, 40.00012],
                        [-104.99992, 39.99990],
                        [-105.00016, 39.99990],
                        [-105.00016, 40.00012],
                    ]],
                },
            }
        ],
    )
    fallback_path = _write_geojson(
        tmp_path / "building_footprints_fallback.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "fallback-home"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00034, 40.00022],
                        [-105.00014, 40.00022],
                        [-105.00014, 40.00004],
                        [-105.00034, 40.00004],
                        [-105.00034, 40.00022],
                    ]],
                },
            }
        ],
    )
    client = BuildingFootprintClient(path=overture_path, extra_paths=[fallback_path], max_search_m=90.0)
    result = client.get_building_footprint(lat=40.0, lon=-105.0, anchor_precision="parcel_or_address_point")

    assert result.found is True
    assert result.source is not None and "overture" in result.source
    assert result.matched_structure_id == "overture-home"


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_structure_matching_falls_back_when_overture_source_missing(tmp_path: Path) -> None:
    missing_overture = str(tmp_path / "building_footprints_overture.geojson")
    fallback_path = _write_geojson(
        tmp_path / "building_footprints_fallback.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "fallback-home"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00018, 40.00010],
                        [-104.99990, 40.00010],
                        [-104.99990, 39.99988],
                        [-105.00018, 39.99988],
                        [-105.00018, 40.00010],
                    ]],
                },
            }
        ],
    )
    client = BuildingFootprintClient(path=missing_overture, extra_paths=[fallback_path], max_search_m=90.0)
    result = client.get_building_footprint(lat=40.0, lon=-105.0, anchor_precision="parcel_or_address_point")

    assert result.found is True
    assert result.source is not None and "fallback" in result.source
    assert result.matched_structure_id == "fallback-home"
