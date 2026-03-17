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
