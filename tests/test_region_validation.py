from __future__ import annotations

import json
from pathlib import Path

import backend.data_prep.validate_region as region_validator
from backend.data_prep.validate_region import validate_prepared_region
from backend.region_registry import REQUIRED_REGION_FILES


def _write_region_fixture(base_dir: Path, region_id: str = "test_region") -> tuple[Path, Path]:
    region_dir = base_dir / region_id
    region_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "dem": "dem.tif",
        "slope": "slope.tif",
        "fuel": "fuel.tif",
        "canopy": "canopy.tif",
        "fire_perimeters": "fire_perimeters.geojson",
        "building_footprints": "building_footprints.geojson",
    }
    for key, rel in files.items():
        path = region_dir / rel
        if key in {"fire_perimeters", "building_footprints"}:
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [-0.1, 0.9], [-0.1, 0.1]]],
                        },
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_bytes(b"fake-raster")

    manifest = {
        "region_id": region_id,
        "display_name": "Test Region",
        "bounds": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "files": files,
    }
    manifest_path = region_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return region_dir, manifest_path


def test_validate_prepared_region_success(monkeypatch, tmp_path):
    _write_region_fixture(tmp_path)
    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    assert result["ready_for_runtime"] is True
    assert result["validation_status"] == "passed"
    assert result["scoring_readiness"] == "full_scoring"
    assert result["footprint_ring_support"] == "available"
    assert result["required_layers_checked"] == list(REQUIRED_REGION_FILES)
    assert result["property_specific_readiness"]["readiness"] in {
        "address_level_only",
        "limited_regional_ready",
        "property_specific_ready",
    }
    readiness = result["property_specific_readiness"]
    assert set(readiness.keys()) >= {
        "parcel_ready",
        "footprint_ready",
        "naip_ready",
        "structure_enrichment_ready",
        "parcel_footprint_linkage_quality",
        "overall_readiness",
    }
    assert "required_layers_present" in result
    assert "optional_layers_missing" in result


def test_validate_prepared_region_missing_file_fails(monkeypatch, tmp_path):
    region_dir, _ = _write_region_fixture(tmp_path)
    (region_dir / "fuel.tif").unlink()
    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    assert result["ready_for_runtime"] is False
    assert any("fuel" in blocker for blocker in result["blockers"])


def test_validate_prepared_region_manifest_not_found(tmp_path):
    result = validate_prepared_region(region_id="missing_region", base_dir=str(tmp_path))
    assert result["ready_for_runtime"] is False
    assert result["validation_status"] == "failed"
    assert any("Manifest not found" in blocker for blocker in result["blockers"])
    readiness = result.get("property_specific_readiness") or {}
    assert readiness.get("overall_readiness") == "limited_regional"
    assert readiness.get("parcel_ready") is False
    assert readiness.get("footprint_ready") is False


def test_validate_prepared_region_sample_point_outside_fails(monkeypatch, tmp_path):
    _write_region_fixture(tmp_path)
    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(
        region_id="test_region",
        base_dir=str(tmp_path),
        sample_lat=2.0,
        sample_lon=2.0,
    )
    assert result["ready_for_runtime"] is False
    assert result["sample_test"]["status"] == "failed"
    assert any("outside region manifest bounds" in blocker for blocker in result["blockers"])


def test_validate_prepared_region_runtime_mapping_mismatch(monkeypatch, tmp_path):
    _write_region_fixture(tmp_path)
    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    monkeypatch.setattr(
        region_validator,
        "RUNTIME_LAYER_MAP",
        {
            "dem": "dem",
            "slope": "slope",
            "fuel": "fuel",
            "canopy": "canopy",
            "fire_perimeters": "fire_perimeters",
            "building_footprints": "building_footprints",
            "missing_runtime": "not_a_real_layer",
        },
    )
    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    assert result["ready_for_runtime"] is False
    assert any("Runtime layer mapping mismatch" in blocker for blocker in result["blockers"])


def test_validate_prepared_region_manifest_status_update(monkeypatch, tmp_path):
    _, manifest_path = _write_region_fixture(tmp_path)
    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(
        region_id="test_region",
        base_dir=str(tmp_path),
        update_manifest=True,
    )
    assert result["ready_for_runtime"] is True
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "passed"
    assert payload["runtime_compatibility_status"] == "pass"
    assert payload["validation_run_at"]
    assert "catalog" in payload
    assert "validation_summary" in payload["catalog"]
    assert "property_specific_readiness" in payload["catalog"]


def test_validate_prepared_region_property_specific_readiness_improves_with_enrichment(monkeypatch, tmp_path):
    region_dir, manifest_path = _write_region_fixture(tmp_path)
    for layer, filename in {
        "roads": "roads.geojson",
        "parcel_polygons": "parcel_polygons.geojson",
    }.items():
        (region_dir / filename).write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"id": 1},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[-0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [-0.1, 0.9], [-0.1, 0.1]]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("files", {})[layer] = filename
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    assert result["property_specific_readiness"]["readiness"] in {
        "address_level_only",
        "property_specific_ready",
    }
    assert result["property_specific_readiness"]["parcel_ready"] is True
    assert result["property_specific_readiness"]["footprint_ready"] is True
    assert result["property_specific_readiness"]["overall_readiness"] in {
        "address_level",
        "property_specific",
    }
    assert "roads" in result["optional_layers_present"]


def test_validate_prepared_region_property_specific_readiness_reports_property_specific_components(
    monkeypatch, tmp_path
):
    region_dir, manifest_path = _write_region_fixture(tmp_path)
    for layer, filename in {
        "roads": "roads.geojson",
        "whp": "whp.tif",
        "parcel_polygons": "parcel_polygons.geojson",
        "parcel_address_points": "parcel_address_points.geojson",
        "naip_structure_features": "naip_structure_features.json",
    }.items():
        path = region_dir / filename
        if filename.endswith(".tif"):
            path.write_bytes(b"fake-raster")
        else:
            path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"id": 1},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[[-0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [-0.1, 0.9], [-0.1, 0.1]]],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", {}).update(
        {
            "roads": "roads.geojson",
            "whp": "whp.tif",
            "parcel_polygons": "parcel_polygons.geojson",
            "parcel_address_points": "parcel_address_points.geojson",
            "naip_structure_features": "naip_structure_features.json",
        }
    )
    manifest.setdefault("catalog", {})["public_record_fields"] = {"year_built": "construction_year"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    readiness = result["property_specific_readiness"]
    assert readiness["parcel_ready"] is True
    assert readiness["footprint_ready"] is True
    assert readiness["naip_ready"] is True
    assert readiness["structure_enrichment_ready"] is True
    assert readiness["parcel_footprint_linkage_quality"] in {"high", "moderate"}
    assert readiness["overall_readiness"] == "property_specific"


def test_validate_prepared_region_includes_configured_optional_layers(monkeypatch, tmp_path):
    region_dir, manifest_path = _write_region_fixture(tmp_path)
    (region_dir / "whp.tif").write_bytes(b"fake-raster")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", {})["whp"] = "whp.tif"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )

    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    assert result["ready_for_runtime"] is True
    assert "whp" in result["required_layers_checked"]


def test_validate_prepared_region_ring_support_partial_when_sample_not_in_footprint(monkeypatch, tmp_path):
    _write_region_fixture(tmp_path)
    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )

    result = validate_prepared_region(
        region_id="test_region",
        base_dir=str(tmp_path),
        sample_lat=0.95,
        sample_lon=0.95,
    )
    assert result["ready_for_runtime"] is True
    if getattr(region_validator, "box", None) is None or getattr(region_validator, "shape", None) is None:
        assert result["footprint_ring_support"] == "available"
    else:
        assert result["footprint_ring_support"] == "partial"
        assert any("structure rings may fallback to point-based mode" in w.lower() for w in result["warnings"])


def test_validate_vector_layer_rejects_polygon_only_parcel_address_points(tmp_path):
    path = tmp_path / "parcel_address_points.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"SITUS_ADDRESS": "19 E ASPEN LN"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [-0.1, 0.9], [-0.1, 0.1]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    errors, _warnings = region_validator._validate_vector_layer(
        path,
        (0.0, 0.0, 1.0, 1.0),
        layer_key="parcel_address_points",
    )
    assert any("point geometry ratio" in err for err in errors)


def test_validate_prepared_region_fails_when_parcel_layers_are_identical(monkeypatch, tmp_path):
    region_dir, manifest_path = _write_region_fixture(tmp_path)
    duplicate_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"OBJECTID": 1, "SITUS_ADDRESS": "19 E ASPEN LN"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [-0.1, 0.9], [-0.1, 0.1]]],
                },
            }
        ],
    }
    (region_dir / "parcel_polygons.geojson").write_text(json.dumps(duplicate_payload), encoding="utf-8")
    (region_dir / "parcel_address_points.geojson").write_text(json.dumps(duplicate_payload), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", {})["parcel_polygons"] = "parcel_polygons.geojson"
    manifest.setdefault("files", {})["parcel_address_points"] = "parcel_address_points.geojson"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        region_validator,
        "_validate_layer_openable_and_intersects",
        lambda layer_key, layer_path, bounds: ([], []),
    )
    result = validate_prepared_region(region_id="test_region", base_dir=str(tmp_path))
    assert result["ready_for_runtime"] is False
    assert any("byte-identical datasets" in blocker for blocker in result["blockers"])
