from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.property_anchor import PropertyAnchorResolver

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


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_property_anchor_prefers_nearby_address_point_over_interpolated_geocode(tmp_path: Path) -> None:
    address_points = _write_geojson(
        tmp_path / "address_points.geojson",
        [
            {
                "type": "Feature",
                "properties": {
                    "address_id": "ap-1",
                    "source_name": "County Address Points",
                    "source_vintage": "2025-06",
                },
                "geometry": {"type": "Point", "coordinates": [-113.99405, 46.87212]},
            }
        ],
    )
    resolver = PropertyAnchorResolver(address_points_path=address_points, parcels_path=None)
    result = resolver.resolve(
        geocoded_lat=46.87230,
        geocoded_lon=-113.99430,
        geocode_provider="Test Geocoder",
        geocode_precision="interpolated",
        geocoded_address="201 W Front St, Missoula, MT",
    )

    assert result.anchor_source == "authoritative_address_point"
    assert result.anchor_precision == "parcel_or_address_point"
    assert result.parcel_address_point_geojson is not None
    assert result.parcel_lookup_method in {"contains_point", "none", None}
    assert result.geocode_to_anchor_distance_m is not None
    assert result.geocode_to_anchor_distance_m > 0


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_property_anchor_uses_parcel_centroid_when_parcel_available_without_address_point(tmp_path: Path) -> None:
    parcels = _write_geojson(
        tmp_path / "parcels.geojson",
        [
            {
                "type": "Feature",
                "properties": {"parcel_id": "123-456"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9944, 46.8723],
                        [-113.9938, 46.8723],
                        [-113.9938, 46.8719],
                        [-113.9944, 46.8719],
                        [-113.9944, 46.8723],
                    ]],
                },
            }
        ],
    )
    resolver = PropertyAnchorResolver(address_points_path=None, parcels_path=parcels)
    result = resolver.resolve(
        geocoded_lat=46.8721,
        geocoded_lon=-113.9941,
        geocode_provider="Test Geocoder",
        geocode_precision="unknown",
        geocoded_address="Example",
    )

    assert result.anchor_source in {"parcel_polygon_centroid", "geocoded_address_point"}
    assert result.parcel_id == "123-456"
    assert result.parcel_lookup_method in {"contains_point", "nearest_within_tolerance"}
    assert result.parcel_geometry_geojson is not None
    if result.anchor_source == "parcel_polygon_centroid":
        assert result.anchor_precision == "parcel_or_address_point"


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_property_anchor_returns_geocode_when_no_auxiliary_sources(tmp_path: Path) -> None:
    resolver = PropertyAnchorResolver(address_points_path=None, parcels_path=None)
    result = resolver.resolve(
        geocoded_lat=44.0582,
        geocoded_lon=-121.3153,
        geocode_provider="Test Geocoder",
        geocode_precision="approximate",
        geocoded_address="Bend, OR",
    )

    assert result.anchor_source in {"approximate_geocode", "geocoded_address_point"}
    assert result.anchor_latitude == pytest.approx(44.0582)
    assert result.anchor_longitude == pytest.approx(-121.3153)


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_property_anchor_uses_nearest_parcel_within_tolerance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_PARCEL_LOOKUP_MAX_DISTANCE_M", "60")
    parcels = _write_geojson(
        tmp_path / "parcels_nearest.geojson",
        [
            {
                "type": "Feature",
                "properties": {"parcel_id": "nearest-1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9946, 46.8724],
                        [-113.9941, 46.8724],
                        [-113.9941, 46.8720],
                        [-113.9946, 46.8720],
                        [-113.9946, 46.8724],
                    ]],
                },
            }
        ],
    )
    resolver = PropertyAnchorResolver(address_points_path=None, parcels_path=parcels)
    result = resolver.resolve(
        geocoded_lat=46.87195,
        geocoded_lon=-113.9940,
        geocode_provider="Test Geocoder",
        geocode_precision="interpolated",
        geocoded_address="Missoula Test",
    )

    assert result.parcel_id == "nearest-1"
    assert result.parcel_lookup_method in {"nearest_within_tolerance", "contains_point"}
    if result.parcel_lookup_method == "nearest_within_tolerance":
        assert result.parcel_lookup_distance_m is not None
        assert result.parcel_lookup_distance_m > 0


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_interpolated_geocode_allows_wider_address_point_tolerance(tmp_path: Path) -> None:
    address_points = _write_geojson(
        tmp_path / "address_points_wide.geojson",
        [
            {
                "type": "Feature",
                "properties": {"address_id": "ap-wide"},
                "geometry": {"type": "Point", "coordinates": [-113.99410, 46.87184]},
            }
        ],
    )
    resolver = PropertyAnchorResolver(address_points_path=address_points, parcels_path=None)

    interpolated = resolver.resolve(
        geocoded_lat=46.87230,
        geocoded_lon=-113.99410,
        geocode_precision="interpolated",
    )
    rooftop = resolver.resolve(
        geocoded_lat=46.87230,
        geocoded_lon=-113.99410,
        geocode_precision="rooftop",
    )

    assert interpolated.anchor_source == "authoritative_address_point"
    assert interpolated.address_point_lookup_distance_m is not None
    assert interpolated.address_point_lookup_distance_m > resolver.max_address_point_distance_m
    assert interpolated.anchor_quality in {"medium", "high"}
    assert rooftop.anchor_source != "authoritative_address_point"


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_low_confidence_rooftop_geocode_widens_address_point_tolerance(tmp_path: Path) -> None:
    address_points = _write_geojson(
        tmp_path / "address_points_low_conf.geojson",
        [
            {
                "type": "Feature",
                "properties": {"address_id": "ap-low-conf"},
                "geometry": {"type": "Point", "coordinates": [-113.99410, 46.87199]},
            }
        ],
    )
    resolver = PropertyAnchorResolver(address_points_path=address_points, parcels_path=None)

    high_conf = resolver.resolve(
        geocoded_lat=46.87230,
        geocoded_lon=-113.99410,
        geocode_precision="rooftop",
        geocode_confidence_score=0.82,
    )
    low_conf = resolver.resolve(
        geocoded_lat=46.87230,
        geocoded_lon=-113.99410,
        geocode_precision="rooftop",
        geocode_confidence_score=0.20,
    )

    assert high_conf.anchor_source != "authoritative_address_point"
    assert low_conf.anchor_source == "authoritative_address_point"
    assert low_conf.geocode_confidence_score == pytest.approx(0.20)
    assert any("expanded anchor lookup tolerances" in note.lower() for note in low_conf.diagnostics)


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_low_confidence_rooftop_does_not_expand_parcel_nearest_tolerance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_PARCEL_LOOKUP_MAX_DISTANCE_M", "30")
    parcels = _write_geojson(
        tmp_path / "parcels_low_conf_rooftop.geojson",
        [
            {
                "type": "Feature",
                "properties": {"parcel_id": "near-but-not-too-near"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.99350, 46.87224],
                        [-113.99334, 46.87224],
                        [-113.99334, 46.87196],
                        [-113.99350, 46.87196],
                        [-113.99350, 46.87224],
                    ]],
                },
            }
        ],
    )
    resolver = PropertyAnchorResolver(parcels_path=parcels)

    rooftop_low_conf = resolver.resolve(
        geocoded_lat=46.87210,
        geocoded_lon=-113.99410,
        geocode_precision="rooftop",
        geocode_confidence_score=0.20,
    )
    interpolated_low_conf = resolver.resolve(
        geocoded_lat=46.87210,
        geocoded_lon=-113.99410,
        geocode_precision="interpolated",
        geocode_confidence_score=0.20,
    )

    assert rooftop_low_conf.parcel_id is None
    assert rooftop_low_conf.parcel_lookup_method in {None, "none"}
    assert interpolated_low_conf.parcel_id == "near-but-not-too-near"
    assert interpolated_low_conf.parcel_lookup_method in {"nearest_within_tolerance", "contains_point"}


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_parcel_resolution_clean_match_reports_matched_status(tmp_path: Path) -> None:
    parcels = _write_geojson(
        tmp_path / "county_parcels.geojson",
        [
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": "county-001",
                    "source_name": "County GIS Parcel Fabric",
                    "source_type": "county_parcel_dataset",
                    "source_vintage": "2025-08",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9944, 46.8723],
                        [-113.9938, 46.8723],
                        [-113.9938, 46.8719],
                        [-113.9944, 46.8719],
                        [-113.9944, 46.8723],
                    ]],
                },
            }
        ],
    )
    resolver = PropertyAnchorResolver(parcels_path=parcels)
    result = resolver.resolve(
        geocoded_lat=46.87210,
        geocoded_lon=-113.99410,
        geocode_precision="rooftop",
    )

    parcel_resolution = result.parcel_resolution
    assert parcel_resolution["status"] == "matched"
    assert parcel_resolution["geometry_used"] == "parcel_polygon"
    assert parcel_resolution["overlap_score"] == pytest.approx(100.0)
    assert parcel_resolution["confidence"] >= 80.0
    assert "county" in str(parcel_resolution.get("source") or "").lower()
    assert result.parcel_geometry_geojson is not None


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_parcel_resolution_handles_multiple_candidates(tmp_path: Path) -> None:
    parcels = _write_geojson(
        tmp_path / "ambiguous_parcels.geojson",
        [
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": "amb-1",
                    "source_name": "County Parcel Layer",
                    "source_type": "county_parcel_dataset",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9945, 46.8723],
                        [-113.9939, 46.8723],
                        [-113.9939, 46.8718],
                        [-113.9945, 46.8718],
                        [-113.9945, 46.8723],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": "amb-2",
                    "source_name": "County Parcel Layer",
                    "source_type": "county_parcel_dataset",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.99448, 46.87228],
                        [-113.99388, 46.87228],
                        [-113.99388, 46.87178],
                        [-113.99448, 46.87178],
                        [-113.99448, 46.87228],
                    ]],
                },
            },
        ],
    )
    resolver = PropertyAnchorResolver(parcels_path=parcels)
    result = resolver.resolve(
        geocoded_lat=46.87205,
        geocoded_lon=-113.99415,
        geocode_precision="interpolated",
    )

    parcel_resolution = result.parcel_resolution
    assert parcel_resolution["status"] == "multiple_candidates"
    assert parcel_resolution["lookup_method"] == "multiple_candidates"
    assert 35.0 <= float(parcel_resolution["confidence"]) <= 80.0
    assert result.parcel_geometry_geojson is not None
    assert result.parcel_lookup_method == "multiple_candidates"


@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_parcel_resolution_not_found_uses_bounded_approximation(tmp_path: Path) -> None:
    parcels = _write_geojson(
        tmp_path / "far_parcels.geojson",
        [
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": "far-1",
                    "source_name": "Open Parcel Dataset",
                    "source_type": "open_parcel_dataset",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-114.1000, 46.9000],
                        [-114.0990, 46.9000],
                        [-114.0990, 46.8990],
                        [-114.1000, 46.8990],
                        [-114.1000, 46.9000],
                    ]],
                },
            }
        ],
    )
    resolver = PropertyAnchorResolver(parcels_path=parcels)
    result = resolver.resolve(
        geocoded_lat=46.8721,
        geocoded_lon=-113.9941,
        geocode_precision="rooftop",
    )

    parcel_resolution = result.parcel_resolution
    assert parcel_resolution["status"] == "not_found"
    assert parcel_resolution["geometry_used"] == "bounding_approximation"
    assert parcel_resolution["lookup_method"] == "none"
    assert 0.0 <= float(parcel_resolution["confidence"]) <= 30.0
    assert isinstance(parcel_resolution.get("bounding_geometry"), dict)
    assert result.parcel_geometry_geojson is None
    assert isinstance(result.parcel_bounding_approximation_geojson, dict)


# ---------------------------------------------------------------------------
# Fix 3: rural expansion factor widens address-point proximity window
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_address_point_rural_expansion_factor_allows_match_at_larger_distance(
    tmp_path: Path, monkeypatch
) -> None:
    """Address point 90 m away from geocode point should be matched when the
    rural expansion factor is > 1 but missed with the default factor of 1."""
    # Place address point ~90 m north of geocode point (≈0.00081 deg latitude)
    address_points = _write_geojson(
        tmp_path / "address_points.geojson",
        [
            {
                "type": "Feature",
                "properties": {"address_id": "ap-rural"},
                "geometry": {"type": "Point", "coordinates": [-113.9941, 46.8729]},
            }
        ],
    )

    # Default resolver (45 m base, interpolated → ~81 m ceiling): should miss the 90 m point
    resolver_default = PropertyAnchorResolver(address_points_path=address_points)
    result_default = resolver_default.resolve(
        geocoded_lat=46.8721,
        geocoded_lon=-113.9941,
        geocode_precision="interpolated",
    )
    assert result_default.anchor_source != "authoritative_address_point", (
        "Default resolver should not match address point that is ~90 m away"
    )

    # Resolver with expansion factor 3: ceiling raises to ~243 m → should match
    monkeypatch.setenv("WF_ADDRESS_POINT_RURAL_EXPANSION_FACTOR", "3.0")
    resolver_expanded = PropertyAnchorResolver(address_points_path=address_points)
    result_expanded = resolver_expanded.resolve(
        geocoded_lat=46.8721,
        geocoded_lon=-113.9941,
        geocode_precision="interpolated",
    )
    assert result_expanded.anchor_source == "authoritative_address_point", (
        "Expanded resolver should match address point at ~90 m for interpolated precision"
    )


# ---------------------------------------------------------------------------
# Fix 4: parcel_layer_available flag is set correctly in resolve output
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _geo_ready(), reason="Property anchor tests require shapely")
def test_address_point_matched_when_address_point_within_expanded_window_and_parcel_present(
    tmp_path: Path, monkeypatch
) -> None:
    """When a parcel is present and address point is nearby, anchor_source should
    be authoritative_address_point (not parcel_polygon_centroid)."""
    parcels = _write_geojson(
        tmp_path / "parcels.geojson",
        [
            {
                "type": "Feature",
                "properties": {"parcel_id": "mt-001"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9945, 46.8725],
                        [-113.9935, 46.8725],
                        [-113.9935, 46.8715],
                        [-113.9945, 46.8715],
                        [-113.9945, 46.8725],
                    ]],
                },
            }
        ],
    )
    address_points = _write_geojson(
        tmp_path / "address_points.geojson",
        [
            {
                "type": "Feature",
                "properties": {"address_id": "ap-1"},
                "geometry": {"type": "Point", "coordinates": [-113.9941, 46.8721]},
            }
        ],
    )
    resolver = PropertyAnchorResolver(
        address_points_path=address_points, parcels_path=parcels
    )
    result = resolver.resolve(
        geocoded_lat=46.8720,
        geocoded_lon=-113.9940,
        geocode_precision="interpolated",
    )
    # Address point is very close — should win over parcel centroid
    assert result.anchor_source == "authoritative_address_point"
    assert result.parcel_polygon is not None  # parcel still matched


# ---------------------------------------------------------------------------
# Fix 3: distance_limits_for_precision applies expansion only on coarse geocodes
# ---------------------------------------------------------------------------

def test_distance_limits_expansion_only_applies_to_coarse_precision() -> None:
    from backend.property_anchor import PropertyAnchorResolver as _R
    # rooftop precision → no expansion even with large factor
    addr_rooftop, _ = _R._distance_limits_for_precision(
        geocode_precision="rooftop",
        geocode_confidence_score=0.95,
        address_default_m=45.0,
        parcel_default_m=30.0,
        override_anchor=False,
        address_rural_expansion_factor=3.0,
    )
    assert addr_rooftop <= 160.0, "Rooftop precision should not be expanded"

    # interpolated precision → expansion applied
    addr_interp, _ = _R._distance_limits_for_precision(
        geocode_precision="interpolated",
        geocode_confidence_score=0.5,
        address_default_m=45.0,
        parcel_default_m=30.0,
        override_anchor=False,
        address_rural_expansion_factor=3.0,
    )
    assert addr_interp > 160.0, "Interpolated precision with factor=3 should exceed default cap"
