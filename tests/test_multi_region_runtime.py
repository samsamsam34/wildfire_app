from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
import backend.data_prep.prepare_region as prep_region_module
import backend.wildfire_data as wildfire_data_module
from backend.data_prep.prepare_region import prepare_region_layers
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireDataClient


client = TestClient(app_main.app)
FIXTURE_PATH = Path("tests") / "fixtures" / "multi_region_sample_properties.json"


def _headers() -> dict[str, str]:
    return {
        "X-User-Role": "admin",
        "X-Organization-Id": "default_org",
        "X-User-Id": "multi_region_test",
    }


def test_multi_region_regression_fixture_has_required_city_coverage() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    samples = list(payload.get("samples") or [])
    missoula = [row for row in samples if str(row.get("city")).lower() == "missoula"]
    bozeman = [row for row in samples if str(row.get("city")).lower() == "bozeman"]
    uncovered = [row for row in samples if row.get("expected_coverage") is False]

    assert len(missoula) >= 5
    assert len(bozeman) >= 5
    assert len(uncovered) >= 2


def _require_region_prep_deps() -> None:
    if getattr(prep_region_module, "rasterio", None) is None:
        pytest.skip("rasterio is not installed in this environment")
    if getattr(prep_region_module, "shape", None) is None:
        pytest.skip("shapely is not available for region prep tests")


def _require_geo_runtime_deps() -> None:
    if getattr(wildfire_data_module, "rasterio", None) is None:
        pytest.skip("rasterio is not installed in this environment")
    if getattr(wildfire_data_module, "Transformer", None) is None:
        pytest.skip("pyproj is not installed in this environment")


def _make_region_sources(tmp_path: Path) -> dict[str, str]:
    _require_region_prep_deps()
    rasterio_mod = prep_region_module.rasterio
    np_mod = prep_region_module.np
    assert rasterio_mod is not None
    assert np_mod is not None

    files = {
        "dem": tmp_path / "src_dem.tif",
        "slope": tmp_path / "src_slope.tif",
        "fuel": tmp_path / "src_fuel.tif",
        "canopy": tmp_path / "src_canopy.tif",
        "fire_perimeters": tmp_path / "src_fire_perimeters.geojson",
        "building_footprints": tmp_path / "src_building_footprints.geojson",
    }
    files["fire_perimeters"].write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-114.3, 45.4], [-110.8, 45.4], [-110.8, 47.1], [-114.3, 47.1], [-114.3, 45.4]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    files["building_footprints"].write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-114.02, 46.86], [-114.01, 46.86], [-114.01, 46.87], [-114.02, 46.87], [-114.02, 46.86]]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"id": 2},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-111.05, 45.67], [-111.04, 45.67], [-111.04, 45.68], [-111.05, 45.68], [-111.05, 45.67]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    transform = rasterio_mod.transform.from_origin(-114.4, 47.2, 0.005, 0.005)
    for key in ["dem", "slope", "fuel", "canopy"]:
        data = np_mod.full((420, 720), 52.0, dtype="float32")
        with rasterio_mod.open(
            files[key],
            "w",
            driver="GTiff",
            width=720,
            height=420,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as ds:
            ds.write(data, 1)
    return {k: str(v) for k, v in files.items()}


def _configure_runtime(monkeypatch, tmp_path: Path, region_root: Path) -> None:
    auth.API_KEYS = set()
    runtime_client = WildfireDataClient()
    runtime_client.region_data_dir = str(region_root)
    runtime_client.use_prepared_regions = True
    runtime_client.allow_legacy_layer_fallback = False
    runtime_client.base_paths = {k: "" for k in runtime_client.base_paths.keys()}
    runtime_client.paths = dict(runtime_client.base_paths)

    monkeypatch.setattr(app_main, "wildfire_data", runtime_client)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "multi_region_runtime.db")))


def _prepare_missoula_and_bozeman(region_root: Path, sources: dict[str, str]) -> tuple[str, str]:
    missoula_id = "missoula_pilot"
    bozeman_id = "bozeman_pilot"
    prepare_region_layers(
        region_id=missoula_id,
        display_name="Missoula Pilot",
        bounds={"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0},
        layer_sources=sources,
        region_data_dir=region_root,
    )
    prepare_region_layers(
        region_id=bozeman_id,
        display_name="Bozeman Pilot",
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        layer_sources=sources,
        region_data_dir=region_root,
    )
    return missoula_id, bozeman_id


def test_assess_auto_resolves_missoula_and_bozeman_without_cross_region_routing(monkeypatch, tmp_path: Path):
    _require_region_prep_deps()
    _require_geo_runtime_deps()

    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    missoula_id, bozeman_id = _prepare_missoula_and_bozeman(region_root, sources)
    _configure_runtime(monkeypatch, tmp_path, region_root)

    geocode_map = {
        "201 W Front St, Missoula, MT 59802": (46.8721, -113.9940, "test-geocoder"),
        "30 W Main St, Bozeman, MT 59715": (45.6796, -111.0386, "test-geocoder"),
    }
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda address: geocode_map[address])

    missoula_payload = {
        "address": "201 W Front St, Missoula, MT 59802",
        "attributes": {"roof_type": "class a", "vent_type": "ember-resistant", "defensible_space_ft": 24},
        "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        "audience": "homeowner",
    }
    bozeman_payload = {
        "address": "30 W Main St, Bozeman, MT 59715",
        "attributes": {"roof_type": "wood_shake", "vent_type": "standard", "defensible_space_ft": 8},
        "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        "audience": "homeowner",
    }

    missoula_res = client.post("/risk/assess", json=missoula_payload, headers=_headers())
    bozeman_res = client.post("/risk/assess", json=bozeman_payload, headers=_headers())
    assert missoula_res.status_code == 200
    assert bozeman_res.status_code == 200

    missoula_body = missoula_res.json()
    bozeman_body = bozeman_res.json()

    assert missoula_body["region_resolution"]["coverage_available"] is True
    assert bozeman_body["region_resolution"]["coverage_available"] is True
    assert missoula_body["region_resolution"]["resolved_region_id"] == missoula_id
    assert bozeman_body["region_resolution"]["resolved_region_id"] == bozeman_id
    assert missoula_body["coverage_available"] is True
    assert bozeman_body["coverage_available"] is True
    assert missoula_body["resolved_region_id"] == missoula_id
    assert bozeman_body["resolved_region_id"] == bozeman_id
    assert missoula_body["geocoding"]["geocode_status"] == "accepted"
    assert bozeman_body["geocoding"]["geocode_status"] == "accepted"
    assert missoula_body["geocoding"]["resolved_latitude"] == pytest.approx(46.8721)
    assert bozeman_body["geocoding"]["resolved_latitude"] == pytest.approx(45.6796)

    assert missoula_body["property_level_context"]["region_id"] == missoula_id
    assert bozeman_body["property_level_context"]["region_id"] == bozeman_id
    assert str(missoula_body["property_level_context"]["region_manifest_path"]).endswith(f"{missoula_id}/manifest.json")
    assert str(bozeman_body["property_level_context"]["region_manifest_path"]).endswith(f"{bozeman_id}/manifest.json")

    # Guard against cross-region misrouting.
    assert missoula_body["region_resolution"]["resolved_region_id"] != bozeman_body["region_resolution"]["resolved_region_id"]

    assert missoula_body["wildfire_risk_score_available"] is True
    assert missoula_body["insurance_readiness_score_available"] is True
    assert bozeman_body["wildfire_risk_score_available"] is True
    assert bozeman_body["insurance_readiness_score_available"] is True


def test_uncovered_address_returns_graceful_uncovered_location_response(monkeypatch, tmp_path: Path):
    _require_region_prep_deps()
    _require_geo_runtime_deps()

    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    _prepare_missoula_and_bozeman(region_root, sources)
    _configure_runtime(monkeypatch, tmp_path, region_root)

    geocode_map = {
        "62910 O B Riley Rd, Bend, OR 97703": (44.0839, -121.3153, "test-geocoder"),
    }
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda address: geocode_map[address])

    response = client.post(
        "/risk/assess",
        json={
            "address": "62910 O B Riley Rd, Bend, OR 97703",
            "attributes": {},
            "confirmed_fields": [],
            "audience": "homeowner",
        },
        headers=_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["geocoding"]["geocode_status"] == "accepted"
    assert body["geocoding"]["resolved_latitude"] == pytest.approx(44.0839)
    assert body["region_resolution"]["coverage_available"] is False
    assert body["region_resolution"]["resolved_region_id"] is None
    assert body["region_resolution"]["reason"] == "no_prepared_region_for_location"
    assert body["region_resolution"]["recommended_action"]
    assert body["coverage_available"] is False
    assert body["resolved_region_id"] is None
    # National fallback clients (Landfire COG, NLCD, elevation, WHP proxy) may
    # produce a partial assessment even when no prepared region is available.
    # The important invariant is coverage_available=False; assessment_status
    # depends on how much national fallback data is available at runtime.
    assert body["assessment_status"] in {"insufficient_data", "partially_scored"}
    assert body["property_level_context"].get("region_id") is None
    assert body["property_level_context"].get("region_status") in {"region_not_prepared", "legacy_fallback", "invalid_manifest"}


def test_regions_coverage_check_returns_resolved_region_id_for_debugging(monkeypatch, tmp_path: Path):
    _require_region_prep_deps()

    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    missoula_id, bozeman_id = _prepare_missoula_and_bozeman(region_root, sources)
    _configure_runtime(monkeypatch, tmp_path, region_root)

    missoula = client.post(
        "/regions/coverage-check",
        json={"latitude": 46.8721, "longitude": -113.9940},
        headers=_headers(),
    )
    bozeman = client.post(
        "/regions/coverage-check",
        json={"latitude": 45.6796, "longitude": -111.0386},
        headers=_headers(),
    )

    assert missoula.status_code == 200
    assert bozeman.status_code == 200

    missoula_body = missoula.json()
    bozeman_body = bozeman.json()
    assert missoula_body["coverage_available"] is True
    assert bozeman_body["coverage_available"] is True
    assert missoula_body["resolved_region_id"] == missoula_id
    assert bozeman_body["resolved_region_id"] == bozeman_id


def test_assessment_emits_structured_region_resolution_log(monkeypatch, tmp_path: Path, caplog):
    _require_region_prep_deps()
    _require_geo_runtime_deps()

    region_root = tmp_path / "regions"
    sources = _make_region_sources(tmp_path)
    missoula_id, _ = _prepare_missoula_and_bozeman(region_root, sources)
    _configure_runtime(monkeypatch, tmp_path, region_root)
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _address: (46.8721, -113.9940, "test-geocoder"))

    with caplog.at_level(logging.INFO, logger="wildfire_app.assessment"):
        response = client.post(
            "/risk/assess",
            json={
                "address": "201 W Front St, Missoula, MT 59802",
                "attributes": {"roof_type": "class a"},
                "confirmed_fields": ["roof_type"],
                "audience": "homeowner",
            },
            headers=_headers(),
        )
    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records if "assessment_region_resolution" in record.getMessage()]
    assert messages, "Expected assessment_region_resolution log event."
    assert any(f'\"resolved_region_id\": \"{missoula_id}\"' in message for message in messages)
    assert any('\"coverage_available\": true' in message for message in messages)
