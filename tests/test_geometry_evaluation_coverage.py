from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from backend.building_footprints import BuildingFootprintClient
from backend.geocoding import Geocoder
from backend.property_anchor import PropertyAnchorResolver

try:
    from shapely.geometry import shape as shapely_shape
except Exception:  # pragma: no cover - optional geospatial deps
    shapely_shape = None


class _FakeHTTPResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):  # pragma: no cover - exercised by urllib context manager
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - exercised by urllib context manager
        return False


def _write_geojson(path: Path, features: list[dict[str, Any]]) -> str:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return str(path)


@pytest.mark.skipif(shapely_shape is None, reason="Geometry coverage test requires shapely")
def test_geocode_variants_resolve_to_same_parcel_and_footprint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parcel_path = _write_geojson(
        tmp_path / "parcels.geojson",
        [
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": "parcel-12",
                    "source_name": "Unit Test Parcel Fabric",
                    "source_type": "county_parcel_dataset",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-120.18730, 48.47830],
                        [-120.18680, 48.47830],
                        [-120.18680, 48.47795],
                        [-120.18730, 48.47795],
                        [-120.18730, 48.47830],
                    ]],
                },
            }
        ],
    )
    footprint_path = _write_geojson(
        tmp_path / "footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "home-12"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-120.18717, 48.47820],
                        [-120.18700, 48.47820],
                        [-120.18700, 48.47805],
                        [-120.18717, 48.47805],
                        [-120.18717, 48.47820],
                    ]],
                },
            }
        ],
    )

    payload = [
        {
            "display_name": "12 Twin Lakes Rd, Winthrop, WA 98862, USA",
            "lat": "48.47812",
            "lon": "-120.18708",
            "importance": 0.09,
            "class": "building",
            "type": "house",
            "address": {
                "house_number": "12",
                "road": "Twin Lakes Rd",
                "town": "Winthrop",
                "state": "Washington",
                "postcode": "98862",
                "country_code": "us",
            },
        }
    ]

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        parsed = urllib.parse.urlparse(req.full_url)
        query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0].lower()
        # Simulate a realistic retry: unit-bearing query misses, stripped query hits.
        if "apt 3" in query:
            return _FakeHTTPResponse([])
        if "12 twin lakes rd" in query or "12 twin lakes road" in query:
            return _FakeHTTPResponse(payload)
        return _FakeHTTPResponse([])

    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", _fake_urlopen)

    geocoder = Geocoder(user_agent="test-suite")
    resolver = PropertyAnchorResolver(parcels_path=parcel_path)
    footprints = BuildingFootprintClient(path=footprint_path, max_search_m=120.0)

    variant_a = geocoder.geocode_with_diagnostics("12 Twin Lakes Rd Apt 3, Winthrop, WA 98862")
    variant_b = geocoder.geocode_with_diagnostics("12 Twin Lakes Road, Winthrop, WA 98862")

    anchor_a = resolver.resolve(
        geocoded_lat=variant_a.latitude,
        geocoded_lon=variant_a.longitude,
        geocode_provider=variant_a.provider,
        geocode_precision=variant_a.geocode_precision,
        geocode_confidence_score=variant_a.confidence_score,
        geocoded_address=variant_a.matched_address,
    )
    anchor_b = resolver.resolve(
        geocoded_lat=variant_b.latitude,
        geocoded_lon=variant_b.longitude,
        geocode_provider=variant_b.provider,
        geocode_precision=variant_b.geocode_precision,
        geocode_confidence_score=variant_b.confidence_score,
        geocoded_address=variant_b.matched_address,
    )

    footprint_a = footprints.get_building_footprint(
        lat=anchor_a.anchor_latitude,
        lon=anchor_a.anchor_longitude,
        parcel_polygon=anchor_a.parcel_polygon,
        anchor_precision=anchor_a.anchor_precision,
    )
    footprint_b = footprints.get_building_footprint(
        lat=anchor_b.anchor_latitude,
        lon=anchor_b.anchor_longitude,
        parcel_polygon=anchor_b.parcel_polygon,
        anchor_precision=anchor_b.anchor_precision,
    )

    assert variant_a.latitude == pytest.approx(variant_b.latitude, abs=1e-7)
    assert variant_a.longitude == pytest.approx(variant_b.longitude, abs=1e-7)

    assert anchor_a.parcel_id == "parcel-12"
    assert anchor_b.parcel_id == "parcel-12"
    assert (anchor_a.parcel_resolution or {}).get("status") == "matched"
    assert (anchor_b.parcel_resolution or {}).get("status") == "matched"
    assert float((anchor_a.parcel_resolution or {}).get("confidence") or 0.0) >= 70.0
    assert float((anchor_b.parcel_resolution or {}).get("confidence") or 0.0) >= 70.0

    assert footprint_a.found is True
    assert footprint_b.found is True
    assert footprint_a.matched_structure_id == "home-12"
    assert footprint_b.matched_structure_id == "home-12"
