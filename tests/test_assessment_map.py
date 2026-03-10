from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.assessment_map import _geo_ready
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext

try:
    from shapely.geometry import Point as ShapelyPoint, shape as shapely_shape
except Exception:  # pragma: no cover - optional in constrained environments
    ShapelyPoint = None
    shapely_shape = None


client = TestClient(app_main.app)


def _ctx() -> WildfireContext:
    return WildfireContext(
        environmental_index=48.0,
        slope_index=48.0,
        aspect_index=45.0,
        fuel_index=52.0,
        moisture_index=50.0,
        canopy_index=51.0,
        wildland_distance_index=40.0,
        historic_fire_index=44.0,
        burn_probability_index=46.0,
        hazard_severity_index=49.0,
        burn_probability=46.0,
        wildfire_hazard=49.0,
        slope=48.0,
        fuel_model=52.0,
        canopy_cover=51.0,
        historic_fire_distance=2.1,
        wildland_distance=120.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["fuel", "canopy", "perimeters"],
        assumptions=[],
        structure_ring_metrics={},
        property_level_context={
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": None,
            "region_id": "missoula_pilot",
            "region_status": "prepared",
        },
    )


def _write_geojson(path: Path, features: list[dict]) -> str:
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _setup(monkeypatch, tmp_path: Path, *, footprints_path: str | None, perimeters_path: str | None) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "assessment_map.db")))
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _address: (46.8721, -113.9940, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _ctx())

    base_paths = dict(app_main.wildfire_data.base_paths)
    runtime_paths = dict(base_paths)
    base_paths["footprints"] = footprints_path or ""
    base_paths["fema_structures"] = ""
    base_paths["perimeters"] = perimeters_path or ""
    runtime_paths.update(base_paths)
    app_main.wildfire_data.base_paths = base_paths
    app_main.wildfire_data.paths = runtime_paths


def _assess() -> dict:
    response = client.post(
        "/risk/assess",
        json={
            "address": "201 W Front St, Missoula, MT 59802",
            "attributes": {
                "roof_type": "class_a_asphalt_composition",
                "vent_type": "ember_resistant_vents",
                "defensible_space_ft": 24,
            },
            "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
            "audience": "homeowner",
        },
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.skipif(not _geo_ready(), reason="Map geometry tests require shapely/pyproj")
def test_map_endpoint_returns_point_footprint_rings_and_overlays(monkeypatch, tmp_path: Path):
    footprints = _write_geojson(
        tmp_path / "footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9942, 46.8720],
                        [-113.9938, 46.8720],
                        [-113.9938, 46.8723],
                        [-113.9942, 46.8723],
                        [-113.9942, 46.8720],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": 2},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9930, 46.8720],
                        [-113.9927, 46.8720],
                        [-113.9927, 46.8722],
                        [-113.9930, 46.8722],
                        [-113.9930, 46.8720],
                    ]],
                },
            },
        ],
    )
    perimeters = _write_geojson(
        tmp_path / "perimeters.geojson",
        [
            {
                "type": "Feature",
                "properties": {"name": "Sample Fire"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-114.01, 46.86],
                        [-113.97, 46.86],
                        [-113.97, 46.89],
                        [-114.01, 46.89],
                        [-114.01, 46.86],
                    ]],
                },
            }
        ],
    )

    _setup(monkeypatch, tmp_path, footprints_path=footprints, perimeters_path=perimeters)
    assessed = _assess()

    map_response = client.get(f"/report/{assessed['assessment_id']}/map")
    assert map_response.status_code == 200
    payload = map_response.json()

    assert -90.0 <= payload["center"]["latitude"] <= 90.0
    assert -180.0 <= payload["center"]["longitude"] <= 180.0
    assert payload["basis_geometry_type"] in {"building_footprint", "point_proxy"}
    assert payload["display_point_source"] in {"geocoded_address_point", "matched_structure_centroid"}
    assert payload["geocoded_address_point"]["geometry"]["type"] == "Point"

    assert "property_point" in payload["data"]
    assert payload["data"]["property_point"]["type"] == "FeatureCollection"

    layer_keys = {row["layer_key"] for row in payload["layers"]}
    assert {
        "property_point",
        "geocoded_address_point",
        "matched_structure_centroid",
        "building_footprint",
        "defensible_space_rings",
        "fire_perimeters",
    }.issubset(layer_keys)

    assert payload["data"].get("defensible_space_rings", {}).get("features")

    geocode_coords = payload["geocoded_address_point"]["geometry"]["coordinates"]
    assert geocode_coords[0] == pytest.approx(-113.9940)
    assert geocode_coords[1] == pytest.approx(46.8721)

    property_coords = payload["data"]["property_point"]["features"][0]["geometry"]["coordinates"]
    assert -180.0 <= property_coords[0] <= 180.0
    assert -90.0 <= property_coords[1] <= 90.0
    assert payload["metadata"]["geometry_contract"]["coordinate_order"] == "[longitude, latitude]"

    matched_centroid = payload.get("matched_structure_centroid")
    footprint = payload.get("matched_structure_footprint")
    if matched_centroid and footprint:
        centroid_coords = matched_centroid["geometry"]["coordinates"]
        assert property_coords == centroid_coords
        assert payload["display_point_source"] == "matched_structure_centroid"
        if shapely_shape and ShapelyPoint:
            footprint_geom = shapely_shape(footprint["geometry"])
            centroid_point = ShapelyPoint(centroid_coords[0], centroid_coords[1])
            assert footprint_geom.contains(centroid_point) or footprint_geom.touches(centroid_point)


@pytest.mark.skipif(not _geo_ready(), reason="Map geometry tests require shapely/pyproj")
def test_map_endpoint_point_only_fallback_is_graceful(monkeypatch, tmp_path: Path):
    _setup(monkeypatch, tmp_path, footprints_path=None, perimeters_path=None)
    assessed = _assess()

    map_response = client.get(f"/report/{assessed['assessment_id']}/map")
    assert map_response.status_code == 200
    payload = map_response.json()

    assert payload["data"]["property_point"]["type"] == "FeatureCollection"
    assert payload["basis_geometry_type"] == "point_proxy"
    assert payload["display_point_source"] == "geocoded_address_point"
    assert payload["matched_structure_centroid"] is None
    assert payload["data"].get("defensible_space_rings", {}).get("features")
    assert any("point" in s.lower() or "footprint" in s.lower() for s in payload.get("limitations") or [])


@pytest.mark.skipif(not _geo_ready(), reason="Map geometry tests require shapely/pyproj")
def test_map_endpoint_prefers_structure_centroid_for_property_marker_when_available(monkeypatch, tmp_path: Path):
    footprints = _write_geojson(
        tmp_path / "footprints_shifted.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": 9},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9948, 46.8715],
                        [-113.9942, 46.8715],
                        [-113.9942, 46.8719],
                        [-113.9948, 46.8719],
                        [-113.9948, 46.8715],
                    ]],
                },
            }
        ],
    )

    _setup(monkeypatch, tmp_path, footprints_path=footprints, perimeters_path=None)
    assessed = _assess()
    map_response = client.get(f"/report/{assessed['assessment_id']}/map")
    assert map_response.status_code == 200
    payload = map_response.json()

    assert payload["display_point_source"] == "matched_structure_centroid"
    geocoded_coords = payload["geocoded_address_point"]["geometry"]["coordinates"]
    marker_coords = payload["data"]["property_point"]["features"][0]["geometry"]["coordinates"]
    centroid_coords = payload["matched_structure_centroid"]["geometry"]["coordinates"]

    assert marker_coords == centroid_coords
    # Ensure this regression does not silently keep the geocoded point when structure centroid exists.
    assert marker_coords != geocoded_coords


@pytest.mark.skipif(not _geo_ready(), reason="Map geometry tests require shapely/pyproj")
def test_map_endpoint_marks_overlay_unavailable_without_failure(monkeypatch, tmp_path: Path):
    footprints = _write_geojson(
        tmp_path / "footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-113.9942, 46.8720],
                        [-113.9938, 46.8720],
                        [-113.9938, 46.8723],
                        [-113.9942, 46.8723],
                        [-113.9942, 46.8720],
                    ]],
                },
            }
        ],
    )

    _setup(monkeypatch, tmp_path, footprints_path=footprints, perimeters_path=None)
    assessed = _assess()

    map_response = client.get(f"/report/{assessed['assessment_id']}/map")
    assert map_response.status_code == 200
    payload = map_response.json()

    fire_layer = next(row for row in payload["layers"] if row["layer_key"] == "fire_perimeters")
    assert fire_layer["available"] is False
    assert fire_layer["reason_unavailable"]


@pytest.mark.skipif(not _geo_ready(), reason="Map geometry tests require shapely/pyproj")
def test_map_endpoint_handles_invalid_feature_geometry_without_crashing(monkeypatch, tmp_path: Path):
    footprints = _write_geojson(
        tmp_path / "bad_footprints.geojson",
        [
            {
                "type": "Feature",
                "properties": {"id": "bad"},
                "geometry": {"type": "Polygon", "coordinates": []},
            }
        ],
    )
    perimeters = _write_geojson(
        tmp_path / "perimeters.geojson",
        [
            {
                "type": "Feature",
                "properties": {"name": "sample"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-114.01, 46.86],
                        [-113.97, 46.86],
                        [-113.97, 46.89],
                        [-114.01, 46.89],
                        [-114.01, 46.86],
                    ]],
                },
            }
        ],
    )

    _setup(monkeypatch, tmp_path, footprints_path=footprints, perimeters_path=perimeters)
    assessed = _assess()

    map_response = client.get(f"/report/{assessed['assessment_id']}/map")
    assert map_response.status_code == 200
    payload = map_response.json()
    assert payload["data"]["property_point"]["type"] == "FeatureCollection"
    assert isinstance(payload.get("layers"), list)
