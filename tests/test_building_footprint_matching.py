from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.building_footprints import BuildingFootprintClient, compute_footprint_geometry_signals
from backend.structure_classifier import classify_structures

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
    assert result.match_method == "point_in_footprint"
    assert result.matched_structure_id == "subject"
    assert result.confidence >= 0.85
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
    assert result.match_method in {"point_in_footprint", "nearest_building_fallback", "parcel_intersection"}
    assert result.candidate_count >= 2
    assert any("similarly plausible" in note.lower() for note in result.assumptions)


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_building_match_uses_composite_score_to_break_small_distance_ties(tmp_path: Path) -> None:
    footprints_path = _write_geojson(
        tmp_path / "close_candidates.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "residential_target"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00018, 40.00010],
                        [-104.99998, 40.00010],
                        [-104.99998, 39.99994],
                        [-105.00018, 39.99994],
                        [-105.00018, 40.00010],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": "tiny_outbuilding"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00002, 40.00006],
                        [-104.99996, 40.00006],
                        [-104.99996, 40.00000],
                        [-105.00002, 40.00000],
                        [-105.00002, 40.00006],
                    ]],
                },
            },
        ],
    )
    client = BuildingFootprintClient(path=footprints_path, max_search_m=70.0)
    result = client.get_building_footprint(lat=40.00002, lon=-105.00008)

    assert result.found is True
    assert result.match_status == "matched"
    assert result.match_method in {"point_in_footprint", "nearest_building_fallback", "parcel_intersection"}
    assert result.matched_structure_id == "residential_target"
    assert result.candidate_count >= 1
    assert not any("similarly plausible" in note.lower() for note in result.assumptions)


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


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_neighbor_metrics_use_subject_footprint_distance_when_available(tmp_path: Path) -> None:
    footprints_path = _write_geojson(
        tmp_path / "neighbor_distance_basis.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "west_neighbor"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00060, 40.00003],
                        [-105.00052, 40.00003],
                        [-105.00052, 39.99997],
                        [-105.00060, 39.99997],
                        [-105.00060, 40.00003],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": "east_neighbor"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00020, 40.00003],
                        [-105.00012, 40.00003],
                        [-105.00012, 39.99997],
                        [-105.00020, 39.99997],
                        [-105.00020, 40.00003],
                    ]],
                },
            },
        ],
    )
    client = BuildingFootprintClient(path=footprints_path)

    subject_close = shapely_shape(
        {
            "type": "Polygon",
            "coordinates": [[
                [-105.00066, 40.00003],
                [-105.00062, 40.00003],
                [-105.00062, 39.99997],
                [-105.00066, 39.99997],
                [-105.00066, 40.00003],
            ]],
        }
    )
    subject_far = shapely_shape(
        {
            "type": "Polygon",
            "coordinates": [[
                [-105.00036, 40.00003],
                [-105.00030, 40.00003],
                [-105.00030, 39.99997],
                [-105.00036, 39.99997],
                [-105.00036, 40.00003],
            ]],
        }
    )

    close_metrics = client.get_neighbor_structure_metrics(
        lat=40.0,
        lon=-105.00040,
        subject_footprint=subject_close,
    )
    far_metrics = client.get_neighbor_structure_metrics(
        lat=40.0,
        lon=-105.00040,
        subject_footprint=subject_far,
    )

    assert close_metrics["nearest_structure_distance_ft"] is not None
    assert far_metrics["nearest_structure_distance_ft"] is not None
    assert float(close_metrics["nearest_structure_distance_ft"]) < float(far_metrics["nearest_structure_distance_ft"])


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_feature_properties_populated_on_point_in_polygon_match(tmp_path: Path) -> None:
    """Matched feature's raw GeoJSON properties are surfaced in BuildingFootprintResult."""
    footprints_path = _write_geojson(
        tmp_path / "footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {
                    "id": "subject",
                    "prep_ring_0_5_veg": 42.5,
                    "prep_ring_5_30_veg": 38.0,
                },
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
            }
        ],
    )
    client = BuildingFootprintClient(path=footprints_path)
    result = client.get_building_footprint(40.0, -105.0)

    assert result.found is True
    assert result.feature_properties is not None
    assert result.feature_properties.get("prep_ring_0_5_veg") == pytest.approx(42.5)
    assert result.feature_properties.get("prep_ring_5_30_veg") == pytest.approx(38.0)


@pytest.mark.skipif(not _geo_ready(), reason="Building footprint matching tests require shapely")
def test_feature_properties_populated_on_nearest_match(tmp_path: Path) -> None:
    """feature_properties is carried through for nearest-building (non-containing) match."""
    # Footprint just north of query point (0-5 m away) so nearest-match succeeds
    footprints_path = _write_geojson(
        tmp_path / "footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "nearby", "prep_ring_30_100_veg": 55.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-105.00020, 40.00040],
                        [-104.99980, 40.00040],
                        [-104.99980, 40.00025],
                        [-105.00020, 40.00025],
                        [-105.00020, 40.00040],
                    ]],
                },
            }
        ],
    )
    client = BuildingFootprintClient(path=footprints_path)
    # Query point is just outside the polygon (south edge gap ~2-3 m)
    result = client.get_building_footprint(40.0002, -105.0)

    assert result.found is True
    assert result.feature_properties is not None
    assert result.feature_properties.get("prep_ring_30_100_veg") == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# National index integration tests
# ---------------------------------------------------------------------------

def _make_polygon_feature(lon: float, lat: float, half: float = 0.0002) -> dict:
    """Return a GeoJSON-style feature dict for a small square polygon."""
    coords = [
        [lon - half, lat + half],
        [lon + half, lat + half],
        [lon + half, lat - half],
        [lon - half, lat - half],
        [lon - half, lat + half],
    ]
    return {
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"source": "overture", "building_class": "residential", "height_m": 5.0, "area_m2": 120.0},
    }


@pytest.mark.skipif(not _geo_ready(), reason="Requires shapely")
def test_national_index_used_when_no_local_files() -> None:
    """When no local files are configured, national index returns a match at confidence ≤ 0.88."""
    mock_index = MagicMock()
    mock_index.enabled = True
    mock_index.get_footprints_near_point.return_value = [
        _make_polygon_feature(-105.0, 40.0)
    ]

    client = BuildingFootprintClient(national_index=mock_index)
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is True
    assert result.source == "national_index_overture"
    assert result.confidence <= 0.88
    assert result.match_status == "matched"


@pytest.mark.skipif(not _geo_ready(), reason="Requires shapely")
def test_national_index_empty_returns_provider_unavailable() -> None:
    """When national index returns [] and no local files exist, result is provider_unavailable."""
    mock_index = MagicMock()
    mock_index.enabled = True
    mock_index.get_footprints_near_point.return_value = []

    client = BuildingFootprintClient(national_index=mock_index)
    result = client.get_building_footprint(lat=40.0, lon=-105.0)

    assert result.found is False
    assert result.match_status == "provider_unavailable"


def test_area_plausibility_score_thresholds() -> None:
    """_area_plausibility_score applies correct thresholds without parcel."""
    # Access as staticmethod via class
    score = BuildingFootprintClient._area_plausibility_score

    # < 15 m² → 0.3
    assert score(12.0, parcel_available=False) == pytest.approx(0.3)
    # 15–40 m² → 0.7
    assert score(30.0, parcel_available=False) == pytest.approx(0.7)
    # ≥ 40 m² → 1.0
    assert score(50.0, parcel_available=False) == pytest.approx(1.0)
    assert score(500.0, parcel_available=False) == pytest.approx(1.0)
    # parcel_available=True → always 1.0 regardless of area
    assert score(12.0, parcel_available=True) == pytest.approx(1.0)


@pytest.mark.skipif(not _geo_ready(), reason="Requires shapely")
def test_multiple_structures_on_parcel(tmp_path: Path) -> None:
    """compute_footprint_geometry_signals returns multiple_structures label from parcel intersection."""
    from shapely.geometry import Polygon

    # Three footprints all within the parcel.
    footprints = [
        shapely_shape({"type": "Polygon", "coordinates": [[
            [-105.0002, 40.0002], [-104.9998, 40.0002], [-104.9998, 39.9998],
            [-105.0002, 39.9998], [-105.0002, 40.0002],
        ]]}),
        shapely_shape({"type": "Polygon", "coordinates": [[
            [-105.0006, 40.0002], [-105.0004, 40.0002], [-105.0004, 39.9998],
            [-105.0006, 39.9998], [-105.0006, 40.0002],
        ]]}),
        shapely_shape({"type": "Polygon", "coordinates": [[
            [-105.001, 40.0002], [-105.0008, 40.0002], [-105.0008, 39.9998],
            [-105.001, 39.9998], [-105.001, 40.0002],
        ]]}),
    ]
    parcel_polygon = Polygon([
        (-105.0012, 40.0004), (-104.9996, 40.0004),
        (-104.9996, 39.9996), (-105.0012, 39.9996),
        (-105.0012, 40.0004),
    ])
    subject_footprint = footprints[0]

    signals = compute_footprint_geometry_signals(
        subject_footprint,
        parcel_polygon=parcel_polygon,
        all_footprints=footprints,
    )

    assert signals["multiple_structures_on_parcel"] in {"multiple_structures", "complex_property"}


@pytest.mark.skipif(not _geo_ready(), reason="Requires shapely")
def test_classify_structures_primary_accessory_neighbor() -> None:
    """classify_structures splits on-parcel vs off-parcel correctly."""
    from shapely.geometry import Polygon

    parcel = Polygon([(-105.001, 40.001), (-104.999, 40.001), (-104.999, 39.999), (-105.001, 39.999), (-105.001, 40.001)])
    # On-parcel: primary + garage-sized structure
    primary = shapely_shape({"type": "Polygon", "coordinates": [[
        [-105.0002, 40.0002], [-104.9998, 40.0002], [-104.9998, 39.9998],
        [-105.0002, 39.9998], [-105.0002, 40.0002],
    ]]})
    garage = shapely_shape({"type": "Polygon", "coordinates": [[
        [-105.0005, 40.0005], [-105.0003, 40.0005], [-105.0003, 40.0003],
        [-105.0005, 40.0003], [-105.0005, 40.0005],
    ]]})
    # Off-parcel neighbor
    neighbor = shapely_shape({"type": "Polygon", "coordinates": [[
        [-105.003, 40.003], [-105.002, 40.003], [-105.002, 40.002],
        [-105.003, 40.002], [-105.003, 40.003],
    ]]})

    result = classify_structures(
        parcel_polygon=parcel,
        all_footprints=[primary, garage, neighbor],
        subject_footprint=primary,
    )

    assert result.classification_basis == "parcel_derived"
    assert len(result.accessory) == 1
    assert len(result.neighbors) == 1
    assert result.on_parcel_count == 2  # primary + garage
    assert result.off_parcel_count == 1


@pytest.mark.skipif(not _geo_ready(), reason="Requires shapely")
def test_classify_structures_no_parcel_heuristic() -> None:
    """Without parcel, all non-primary footprints are labeled neighbors (heuristic basis)."""
    primary = shapely_shape({"type": "Polygon", "coordinates": [[
        [-105.0002, 40.0002], [-104.9998, 40.0002], [-104.9998, 39.9998],
        [-105.0002, 39.9998], [-105.0002, 40.0002],
    ]]})
    shed = shapely_shape({"type": "Polygon", "coordinates": [[
        [-105.0005, 40.0005], [-105.0003, 40.0005], [-105.0003, 40.0003],
        [-105.0005, 40.0003], [-105.0005, 40.0005],
    ]]})

    result = classify_structures(
        parcel_polygon=None,
        all_footprints=[primary, shed],
        subject_footprint=primary,
    )

    assert result.classification_basis == "heuristic"
    assert len(result.accessory) == 0
    assert len(result.neighbors) == 1
    assert result.off_parcel_count == 1
