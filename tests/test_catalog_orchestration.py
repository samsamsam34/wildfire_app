from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.prepare_region_from_catalog_or_sources as prep_orch
from backend.data_prep.catalog import ingest_catalog_raster, ingest_catalog_vector
from scripts.catalog_coverage import build_catalog_coverage_plan, required_core_layers
from scripts.prepare_region_from_catalog_or_sources import (
    RegionPrepExecutionError,
    _build_cli_error_payload,
    _validate_layer_source_config,
    prepare_region_from_catalog_or_sources,
)

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTER_DEPS = True
except Exception:  # pragma: no cover
    np = None
    rasterio = None
    from_origin = None
    HAS_RASTER_DEPS = False


def _write_raster(path: Path, value: float = 10.0, width: int = 240, height: int = 240) -> None:
    if not HAS_RASTER_DEPS:
        pytest.skip("numpy/rasterio are required for catalog orchestration tests")
    transform = from_origin(-1.0, 2.0, 0.01, 0.01)
    data = np.full((height, width), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(data, 1)


def _write_geojson(path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-0.7, 0.1], [0.8, 0.1], [0.8, 1.1], [-0.7, 1.1], [-0.7, 0.1]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_sources(tmp_path: Path) -> dict[str, Path]:
    sources = {
        "dem": tmp_path / "dem.tif",
        "fuel": tmp_path / "fuel.tif",
        "canopy": tmp_path / "canopy.tif",
        "fire_perimeters": tmp_path / "fire_perimeters.geojson",
        "building_footprints": tmp_path / "building_footprints.geojson",
    }
    _write_raster(sources["dem"], value=320.0)
    _write_raster(sources["fuel"], value=45.0)
    _write_raster(sources["canopy"], value=60.0)
    _write_geojson(sources["fire_perimeters"])
    _write_geojson(sources["building_footprints"])
    return sources


def _local_source_config(sources: dict[str, Path]) -> dict[str, dict[str, str]]:
    cfg: dict[str, dict[str, str]] = {}
    for key in required_core_layers():
        if key in sources:
            cfg[key] = {
                "provider_type": "local_file",
                "source_path": str(sources[key]),
            }
    return cfg


def _write_source_registry(path: Path, sources: dict[str, Path]) -> Path:
    layers: dict[str, dict[str, str]] = {}
    for key in required_core_layers():
        layers[key] = {
            "provider_type": "local_file",
            "source_path": str(sources[key]),
        }
    payload = {
        "version": 1,
        "layers": layers,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ingest_core(
    *,
    sources: dict[str, Path],
    bbox: dict[str, float],
    catalog_root: Path,
) -> None:
    ingest_catalog_raster(layer_name="dem", source_path=str(sources["dem"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_raster(layer_name="fuel", source_path=str(sources["fuel"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_raster(layer_name="canopy", source_path=str(sources["canopy"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_vector(
        layer_name="fire_perimeters",
        source_path=str(sources["fire_perimeters"]),
        catalog_root=catalog_root,
        bounds=bbox,
    )
    ingest_catalog_vector(
        layer_name="building_footprints",
        source_path=str(sources["building_footprints"]),
        catalog_root=catalog_root,
        bounds=bbox,
    )


def test_catalog_coverage_plan_full_partial_none(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"

    full_bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    partial_bbox = {"min_lon": -0.8, "min_lat": 0.0, "max_lon": 0.8, "max_lat": 0.9}
    ingest_catalog_raster(layer_name="dem", source_path=str(sources["dem"]), catalog_root=catalog_root, bounds=full_bbox)

    full_plan = build_catalog_coverage_plan(
        bounds=full_bbox,
        required_layers=["dem"],
        optional_layer_keys=[],
        catalog_root=catalog_root,
    )
    partial_plan = build_catalog_coverage_plan(
        bounds=partial_bbox,
        required_layers=["dem"],
        optional_layer_keys=[],
        catalog_root=catalog_root,
    )
    none_plan = build_catalog_coverage_plan(
        bounds=full_bbox,
        required_layers=["fuel"],
        optional_layer_keys=[],
        catalog_root=catalog_root,
    )

    assert full_plan["layers"]["dem"]["coverage_status"] == "full"
    assert partial_plan["layers"]["dem"]["coverage_status"] in {"partial", "full"}
    assert none_plan["layers"]["fuel"]["coverage_status"] == "none"


def test_overture_buildings_vector_ingestion_supported(tmp_path):
    pytest.importorskip("shapely")
    catalog_root = tmp_path / "catalog"
    overture_geojson = tmp_path / "building_footprints_overture.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "ov-1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-0.5, 0.3], [-0.2, 0.3], [-0.2, 0.6], [-0.5, 0.6], [-0.5, 0.3]]],
                },
            }
        ],
    }
    overture_geojson.write_text(json.dumps(payload), encoding="utf-8")

    metadata = ingest_catalog_vector(
        layer_name="building_footprints_overture",
        source_path=str(overture_geojson),
        catalog_root=catalog_root,
        bounds={"min_lon": -0.7, "min_lat": 0.2, "max_lon": 0.7, "max_lat": 0.9},
    )
    assert metadata["layer_name"] == "building_footprints_overture"
    assert Path(str(metadata["catalog_path"])).exists()


def test_prepare_any_region_plan_only_no_name_error_and_policy_structure(tmp_path):
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0}

    result = prepare_region_from_catalog_or_sources(
        region_id="missoula_plan_only",
        display_name="Missoula Plan Only",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={},
        plan_only=True,
    )
    assert result["mode"] == "plan_only"
    assert "required_layer_policy" in result
    policy = result["required_layer_policy"]
    assert "slope" in policy["derived_core_layers"]
    assert "slope" not in policy["required_core_layers"]
    assert "dem" in policy["required_core_layers"]
    assert "parcel_polygons" in policy["optional_layers"]
    assert "parcel_address_points" in policy["optional_layers"]


def test_required_layer_source_validation_raster_and_vector():
    raster = _validate_layer_source_config(
        layer_key="dem",
        layer_cfg={
            "provider_type": "arcgis_image_service",
            "source_endpoint": "https://example.test/dem/ImageServer",
        },
    )
    vector = _validate_layer_source_config(
        layer_key="fire_perimeters",
        layer_cfg={
            "provider_type": "arcgis_feature_service",
            "source_endpoint": "https://example.test/fire/FeatureServer/0",
        },
    )
    assert raster["config_valid"] is True
    assert raster["config_status"] == "configured"
    assert vector["config_valid"] is True
    assert vector["config_status"] == "configured"


def test_required_layer_source_validation_missing_fields_is_actionable():
    invalid = _validate_layer_source_config(
        layer_key="fuel",
        layer_cfg={"provider_type": "arcgis_image_service"},
    )
    assert invalid["config_valid"] is False
    assert invalid["config_status"] == "missing_source_details"
    assert "source_endpoint|source_url|source_path" in invalid["missing_required_fields"]
    assert "missing required source details" in str(invalid["actionable_error"]).lower()


def test_optional_layer_source_validation_includes_env_override_guidance():
    invalid = _validate_layer_source_config(
        layer_key="gridmet_dryness",
        layer_cfg={"provider_type": "arcgis_image_service"},
    )
    assert invalid["config_valid"] is False
    assert invalid["config_status"] == "missing_source_details"
    msg = str(invalid["actionable_error"])
    assert "WF_DEFAULT_GRIDMET_DRYNESS_ENDPOINT" in msg
    assert "WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL" in msg


def test_feature_service_validation_warns_when_endpoint_may_be_stale():
    stale = _validate_layer_source_config(
        layer_key="building_footprints",
        layer_cfg={
            "provider_type": "arcgis_feature_service",
            "source_endpoint": "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/USA_Structures_View/FeatureServer/0",
        },
    )
    assert stale["config_valid"] is True
    assert "configured endpoint may be stale" in str(stale.get("advisory_warning", "")).lower()


def test_plan_only_reports_required_missing_fields_with_specific_diagnostics(tmp_path):
    bbox = {"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0}
    result = prepare_region_from_catalog_or_sources(
        region_id="missing_cfg",
        display_name="Missing Config",
        bounds=bbox,
        catalog_root=tmp_path / "catalog",
        regions_root=tmp_path / "regions",
        source_config={
            "dem": {"provider_type": "arcgis_image_service"},
            "fuel": {"provider_type": "arcgis_image_service", "source_endpoint": "https://example.test/fuel/ImageServer"},
            "canopy": {"provider_type": "arcgis_image_service", "source_endpoint": "https://example.test/canopy/ImageServer"},
            "fire_perimeters": {"provider_type": "arcgis_feature_service", "source_endpoint": "https://example.test/fire/FeatureServer/0"},
            "building_footprints": {"provider_type": "arcgis_feature_service", "source_endpoint": "https://example.test/buildings/FeatureServer/0"},
        },
        skip_optional_layers=True,
        plan_only=True,
    )
    dem_diag = result["required_layer_diagnostics"]["dem"]
    assert dem_diag["config_status"] == "missing_source_details"
    assert dem_diag["provider_type"] == "arcgis_image_service"
    assert "source_endpoint|source_url|source_path" in dem_diag["missing_required_fields"]
    assert isinstance(dem_diag["blocking_reason"], str) and dem_diag["blocking_reason"]
    assert result["buildable_estimate"]["buildable_with_current_config"] is False
    assert "dem" in result["buildable_estimate"]["required_blockers"]


def test_default_source_registry_has_valid_required_core_layer_entries(tmp_path):
    registry_path = Path("config/source_registry.json")
    assert registry_path.exists()
    bbox = {"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0}
    result = prepare_region_from_catalog_or_sources(
        region_id="registry_validation_only",
        display_name="Registry Validation Only",
        bounds=bbox,
        catalog_root=tmp_path / "catalog",
        regions_root=tmp_path / "regions",
        source_config_path=str(registry_path),
        skip_optional_layers=True,
        plan_only=True,
    )
    for layer_key in required_core_layers():
        diag = result["required_layer_diagnostics"][layer_key]
        assert diag["config_valid"] is True, f"{layer_key}: {diag}"


def test_default_source_registry_optional_layers_have_actionable_status(tmp_path):
    registry_path = Path("config/source_registry.json")
    assert registry_path.exists()
    bbox = {"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0}
    result = prepare_region_from_catalog_or_sources(
        region_id="registry_optional_validation",
        display_name="Registry Optional Validation",
        bounds=bbox,
        catalog_root=tmp_path / "catalog",
        regions_root=tmp_path / "regions",
        source_config_path=str(registry_path),
        skip_optional_layers=False,
        plan_only=True,
    )
    assert result["buildable_estimate"]["buildable_with_current_config"] is True
    for layer_key in ("whp", "mtbs_severity", "roads"):
        diag = result["optional_layer_diagnostics"][layer_key]
        assert diag["config_valid"] is True, f"{layer_key}: {diag}"

    gridmet_diag = result["optional_layer_diagnostics"]["gridmet_dryness"]
    assert gridmet_diag["config_valid"] is False
    assert gridmet_diag["config_status"] == "missing_source_details"
    assert "WF_DEFAULT_GRIDMET_DRYNESS_ENDPOINT" in str(gridmet_diag["actionable_error"])
    assert result["optional_config_warnings"]
    assert any("gridmet_dryness" in str(msg) for msg in result["optional_config_warnings"])


def test_prepare_any_region_full_catalog_coverage_no_acquisition(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    _ingest_core(sources=sources, bbox=bbox, catalog_root=catalog_root)

    result = prepare_region_from_catalog_or_sources(
        region_id="full_covered",
        display_name="Full Covered",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={},
        skip_optional_layers=True,
        overwrite=True,
    )
    assert result["acquired_layers"] == []
    assert (regions_root / "full_covered" / "manifest.json").exists()
    assert result["manifest"]["catalog"]["built_from_catalog"] is True


def test_prepare_any_region_partial_coverage_only_missing_layer_acquired(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    smaller = {"min_lon": -0.3, "min_lat": 0.3, "max_lon": 0.3, "max_lat": 0.7}

    ingest_catalog_raster(layer_name="dem", source_path=str(sources["dem"]), catalog_root=catalog_root, bounds=smaller)
    ingest_catalog_raster(layer_name="fuel", source_path=str(sources["fuel"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_raster(layer_name="canopy", source_path=str(sources["canopy"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_vector(
        layer_name="fire_perimeters",
        source_path=str(sources["fire_perimeters"]),
        catalog_root=catalog_root,
        bounds=bbox,
    )
    ingest_catalog_vector(
        layer_name="building_footprints",
        source_path=str(sources["building_footprints"]),
        catalog_root=catalog_root,
        bounds=bbox,
    )

    result = prepare_region_from_catalog_or_sources(
        region_id="partial_fill",
        display_name="Partial Fill",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={"dem": {"provider_type": "local_file", "source_path": str(sources["dem"])}},
        allow_partial_coverage_fill=True,
        skip_optional_layers=True,
        overwrite=True,
    )
    keys = [item["layer_key"] for item in result["acquired_layers"]]
    assert keys == ["dem"]


def test_prepare_any_region_no_coverage_acquires_required_layers(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}

    result = prepare_region_from_catalog_or_sources(
        region_id="no_coverage_fill",
        display_name="No Coverage Fill",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config=_local_source_config(sources),
        skip_optional_layers=True,
        overwrite=True,
    )
    keys = sorted(item["layer_key"] for item in result["acquired_layers"])
    assert keys == sorted(required_core_layers())
    dem_layer = result["manifest"]["layers"]["dem"]
    assert dem_layer["catalog_fetched_this_run"] is True
    assert dem_layer["catalog_source_origin"] in {
        "newly_ingested_bbox_coverage",
        "newly_ingested_source",
        "full_download_fallback",
    }
    assert dem_layer["catalog_coverage_status"] == "full"


def test_prepare_any_region_required_layer_failure_raises(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    cfg = _local_source_config(sources)
    cfg.pop("canopy")

    with pytest.raises(ValueError) as exc:
        prepare_region_from_catalog_or_sources(
            region_id="required_fail",
            display_name="Required Fail",
            bounds=bbox,
            catalog_root=catalog_root,
            regions_root=regions_root,
            source_config=cfg,
            skip_optional_layers=True,
            overwrite=True,
            require_core_layers=True,
        )
    msg = str(exc.value)
    assert "required core layer blockers" in msg.lower()
    assert "failure_stage=coverage_incomplete_after_ingest" in msg
    assert "nameerror" not in msg.lower()


def test_prepare_any_region_optional_layer_omission_recorded(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    _ingest_core(sources=sources, bbox=bbox, catalog_root=catalog_root)

    result = prepare_region_from_catalog_or_sources(
        region_id="optional_omissions",
        display_name="Optional Omissions",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={},
        skip_optional_layers=False,
        overwrite=True,
    )
    assert "whp" in result["optional_omissions"]
    assert result["manifest"]["catalog"]["optional_omissions"]


def test_prepare_any_region_idempotent_rerun_reuses_catalog(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    cfg = _local_source_config(sources)

    first = prepare_region_from_catalog_or_sources(
        region_id="rerun_region",
        display_name="Rerun Region",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config=cfg,
        skip_optional_layers=True,
        overwrite=True,
    )
    second = prepare_region_from_catalog_or_sources(
        region_id="rerun_region",
        display_name="Rerun Region",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config=cfg,
        skip_optional_layers=True,
        overwrite=True,
    )
    assert len(first["acquired_layers"]) > 0
    assert second["acquired_layers"] == []


def test_prepare_any_region_nearby_reuses_overlapping_catalog_coverage(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    large_bbox = {"min_lon": -0.7, "min_lat": 0.1, "max_lon": 0.7, "max_lat": 0.9}
    nearby_bbox = {"min_lon": -0.5, "min_lat": 0.2, "max_lon": 0.5, "max_lat": 0.8}
    cfg = _local_source_config(sources)

    prepare_region_from_catalog_or_sources(
        region_id="region_a",
        display_name="Region A",
        bounds=large_bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config=cfg,
        skip_optional_layers=True,
        overwrite=True,
    )
    second = prepare_region_from_catalog_or_sources(
        region_id="region_b",
        display_name="Region B",
        bounds=nearby_bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={},
        skip_optional_layers=True,
        overwrite=True,
    )
    assert second["acquired_layers"] == []


def test_prepare_any_region_plan_only_reports_steps_without_writes(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    cfg = _local_source_config(sources)

    result = prepare_region_from_catalog_or_sources(
        region_id="plan_only_region",
        display_name="Plan Only Region",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config=cfg,
        skip_optional_layers=True,
        plan_only=True,
    )
    assert result["mode"] == "plan_only"
    assert result["coverage_plan"]["summary"]["required_missing"]
    assert result["acquisition_steps"]
    assert not (regions_root / "plan_only_region").exists()


def test_plan_only_optional_layer_missing_config_does_not_block_required_buildability(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    cfg = _local_source_config(sources)

    result = prepare_region_from_catalog_or_sources(
        region_id="plan_required_only",
        display_name="Plan Required Only",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config=cfg,
        skip_optional_layers=False,
        plan_only=True,
    )
    assert result["buildable_estimate"]["buildable_with_current_config"] is True
    assert result["optional_layer_diagnostics"]["whp"]["config_status"] == "missing_layer_entry"
    assert result["optional_layer_diagnostics"]["roads"]["config_status"] == "missing_layer_entry"


def test_prepare_any_region_uses_default_source_registry_and_runs_new_area_smoke(monkeypatch, tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    registry_path = _write_source_registry(tmp_path / "source_registry.json", sources)
    monkeypatch.setenv("WF_SOURCE_CONFIG_PATH", str(registry_path))

    plan = prepare_region_from_catalog_or_sources(
        region_id="new_area_smoke",
        display_name="New Area Smoke",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        skip_optional_layers=True,
        plan_only=True,
    )
    assert plan["mode"] == "plan_only"
    assert plan["stage_status"]["coverage_plan"]["status"] == "ok"
    assert "required_layer_policy" in plan
    assert plan["required_layer_policy"]["required_core_layers"] == required_core_layers()
    assert plan["compact_summary"]["mode"] == "plan_only"
    assert plan["source_registry"]["default_source_registry_used"] is True
    assert plan["source_registry"]["source_config_path"] == str(registry_path)
    assert sorted(plan["operator_summary"]["required_layers_missing"]) == sorted(required_core_layers())
    assert sorted(plan["operator_summary"]["layers_requiring_acquisition"]) == sorted(required_core_layers())
    assert plan["buildable_estimate"]["buildable_with_current_config"] is True

    executed = prepare_region_from_catalog_or_sources(
        region_id="new_area_smoke",
        display_name="New Area Smoke",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        skip_optional_layers=True,
        overwrite=True,
        validate=True,
    )
    assert executed["mode"] == "executed"
    assert executed["final_status"] in {"success", "partial"}
    assert executed["source_registry"]["default_source_registry_used"] is True
    assert executed["stage_status"]["region_build"]["status"] == "ok"
    assert executed["compact_summary"]["mode"] == "executed"
    assert sorted(item["layer_key"] for item in executed["acquired_layers"]) == sorted(required_core_layers())
    assert (regions_root / "new_area_smoke" / "manifest.json").exists()
    validation = executed.get("validation")
    assert isinstance(validation, dict)
    assert validation.get("validation_status") in {"passed", "failed"}
    if validation.get("validation_status") == "failed":
        assert validation.get("blockers"), "Validation failures should include structured blockers."


def test_prepare_any_region_already_prepared_without_overwrite_returns_fast_status(tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": -0.6, "min_lat": 0.2, "max_lon": 0.6, "max_lat": 0.8}
    _ingest_core(sources=sources, bbox=bbox, catalog_root=catalog_root)

    first = prepare_region_from_catalog_or_sources(
        region_id="already_prepared",
        display_name="Already Prepared",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={},
        skip_optional_layers=True,
        overwrite=True,
    )
    assert first["final_status"] in {"success", "partial"}

    second = prepare_region_from_catalog_or_sources(
        region_id="already_prepared",
        display_name="Already Prepared",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        source_config={},
        skip_optional_layers=True,
        overwrite=False,
        validate=True,
    )
    assert second["final_status"] == "already_prepared"
    assert second["acquired_layers"] == []
    assert second["stage_status"]["region_build"]["status"] == "skipped"


def test_prepare_any_region_second_region_missoula_smoke(monkeypatch, tmp_path):
    sources = _seed_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    # Missoula-like bbox (naming + coordinates for operator parity).
    bbox = {"min_lon": -114.2, "min_lat": 46.75, "max_lon": -113.8, "max_lat": 47.0}
    registry_path = _write_source_registry(tmp_path / "source_registry_missoula.json", sources)
    monkeypatch.setenv("WF_SOURCE_CONFIG_PATH", str(registry_path))

    plan = prepare_region_from_catalog_or_sources(
        region_id="missoula_pilot",
        display_name="Missoula Pilot",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        skip_optional_layers=True,
        plan_only=True,
    )
    assert plan["mode"] == "plan_only"
    assert plan["buildable_estimate"]["buildable_with_current_config"] is True
    assert sorted(plan["operator_summary"]["required_layers_missing"]) == sorted(required_core_layers())
    assert plan["stage_status"]["coverage_plan"]["status"] == "ok"

    executed = prepare_region_from_catalog_or_sources(
        region_id="missoula_pilot",
        display_name="Missoula Pilot",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        skip_optional_layers=True,
        overwrite=True,
        validate=True,
    )
    assert executed["mode"] == "executed"
    assert executed["final_status"] in {"success", "partial"}
    assert sorted(item["layer_key"] for item in executed["acquired_layers"]) == sorted(required_core_layers())
    assert (regions_root / "missoula_pilot" / "manifest.json").exists()
    validation = executed.get("validation")
    assert isinstance(validation, dict)
    assert validation.get("validation_status") in {"passed", "failed"}
    if validation.get("validation_status") == "failed":
        assert validation.get("blockers"), "Validation failure should include structured blockers."


def test_cli_error_payload_flags_internal_missing_constant():
    payload = _build_cli_error_payload(
        exc=NameError("name 'CATALOG_DERIVED_RASTER_LAYERS' is not defined"),
        region_id="x",
        display_name="X",
        requested_bbox={"min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1},
        mode="plan_only",
    )
    assert payload["issue_type"] == "internal_code_error"
    assert payload["failure_stage"] == "internal_layer_definition_reference"
    assert payload["missing_constant"] == "CATALOG_DERIVED_RASTER_LAYERS"


def _coverage_plan_for_required(required: list[str], status: str) -> dict[str, Any]:
    layers = {
        layer: {
            "coverage_status": status,
            "entries_total": 0 if status == "none" else 1,
            "entries_with_bounds": 0 if status == "none" else 1,
            "entries_covering": [],
            "entries_intersecting": [],
            "entries_missing_bounds": [],
            "notes": [],
        }
        for layer in required
    }
    return {
        "requested_bbox": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "required_layers": required,
        "optional_layers": [],
        "layers": layers,
        "summary": {
            "required_missing": sorted(required) if status == "none" else [],
            "required_partial": [],
            "optional_missing": [],
            "optional_partial": [],
            "buildable_from_catalog": status == "full",
            "fully_covered_from_catalog": status == "full",
            "recommended_actions": [],
        },
    }


def _valid_required_source_config() -> dict[str, dict[str, str]]:
    return {
        "dem": {"provider_type": "arcgis_image_service", "source_endpoint": "https://example.test/dem/ImageServer"},
        "fuel": {"provider_type": "arcgis_image_service", "source_endpoint": "https://example.test/fuel/ImageServer"},
        "canopy": {"provider_type": "arcgis_image_service", "source_endpoint": "https://example.test/canopy/ImageServer"},
        "fire_perimeters": {"provider_type": "arcgis_feature_service", "source_endpoint": "https://example.test/fire/FeatureServer/0"},
        "building_footprints": {"provider_type": "arcgis_feature_service", "source_endpoint": "https://example.test/buildings/FeatureServer/0"},
    }


def test_execution_diagnostics_classifies_invalid_raster_payload(monkeypatch, tmp_path):
    required = required_core_layers()

    def fake_coverage_plan(*args, **kwargs):
        return _coverage_plan_for_required(required, "none")

    def fake_ingest(*args, **kwargs):
        if kwargs["layer_key"] == "canopy":
            raise ValueError("download returned JSON/error content for canopy: Token Required")
        return {
            "item_id": f"{kwargs['layer_key']}-item",
            "catalog_path": str(tmp_path / f"{kwargs['layer_key']}.tif"),
            "provider_type": "arcgis_image_service",
            "acquisition_method": "bbox_export",
            "source_url": "https://example.test/layer",
            "source_endpoint": "https://example.test/layer/ImageServer",
            "bbox_used": "0,0,1,1",
            "cache_hit": False,
            "ingest_diagnostics": {
                "fetch_attempted": True,
                "fetch_succeeded": True,
                "catalog_ingest_succeeded": True,
                "temp_input_path": str(tmp_path / "tmp.tif"),
            },
        }

    monkeypatch.setattr(prep_orch, "build_catalog_coverage_plan", fake_coverage_plan)
    monkeypatch.setattr(prep_orch, "_ingest_layer_for_bbox", fake_ingest)

    with pytest.raises(RegionPrepExecutionError) as exc:
        prepare_region_from_catalog_or_sources(
            region_id="diag_invalid_raster",
            display_name="Diag Invalid Raster",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            catalog_root=tmp_path / "catalog",
            regions_root=tmp_path / "regions",
            source_config=_valid_required_source_config(),
            skip_optional_layers=True,
            overwrite=True,
        )
    details = exc.value.details
    canopy = details["per_layer_execution_diagnostics"]["canopy"]
    assert canopy["failure_reason"] == "invalid_provider_payload"
    assert "provider returned an error payload" in canopy["actionable_error"].lower()


def test_execution_diagnostics_classifies_endpoint_not_found(monkeypatch, tmp_path):
    required = required_core_layers()

    def fake_coverage_plan(*args, **kwargs):
        return _coverage_plan_for_required(required, "none")

    def fake_ingest(*args, **kwargs):
        if kwargs["layer_key"] == "fuel":
            raise ValueError(
                "Failed download after retries for https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF240/US_240FBFM40/ImageServer/exportImage?x=1: "
                "HTTP Error 404: Not Found (http_status=404, content_type=application/json, body_snippet={\"error\":{\"message\":\"service not found\"}})"
            )
        return {
            "item_id": f"{kwargs['layer_key']}-item",
            "catalog_path": str(tmp_path / f"{kwargs['layer_key']}.tif"),
            "provider_type": "arcgis_image_service",
            "acquisition_method": "bbox_export",
            "source_url": "https://example.test/layer",
            "source_endpoint": "https://example.test/layer/ImageServer",
            "bbox_used": "0,0,1,1",
            "cache_hit": False,
            "ingest_diagnostics": {
                "fetch_attempted": True,
                "fetch_succeeded": True,
                "catalog_ingest_succeeded": True,
                "temp_input_path": str(tmp_path / "tmp.tif"),
            },
        }

    monkeypatch.setattr(prep_orch, "build_catalog_coverage_plan", fake_coverage_plan)
    monkeypatch.setattr(prep_orch, "_ingest_layer_for_bbox", fake_ingest)

    with pytest.raises(RegionPrepExecutionError) as exc:
        prepare_region_from_catalog_or_sources(
            region_id="diag_404",
            display_name="Diag 404",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            catalog_root=tmp_path / "catalog",
            regions_root=tmp_path / "regions",
            source_config=_valid_required_source_config(),
            skip_optional_layers=True,
            overwrite=True,
        )
    details = exc.value.details
    fuel = details["per_layer_execution_diagnostics"]["fuel"]
    assert fuel["failure_reason"] == "endpoint_not_found"
    assert fuel["response_status_code"] == 404
    assert "lfps.usgs.gov" in str(fuel["request_url"])
    assert "service not found" in str(fuel["provider_error_snippet"]).lower()


def test_execution_diagnostics_classifies_empty_vector_result(monkeypatch, tmp_path):
    required = required_core_layers()

    def fake_coverage_plan(*args, **kwargs):
        return _coverage_plan_for_required(required, "none")

    def fake_ingest(*args, **kwargs):
        if kwargs["layer_key"] == "fire_perimeters":
            raise ValueError("Vector source returned empty feature set for requested bbox")
        return {
            "item_id": f"{kwargs['layer_key']}-item",
            "catalog_path": str(tmp_path / f"{kwargs['layer_key']}.tif"),
            "provider_type": "arcgis_image_service",
            "acquisition_method": "bbox_export",
            "source_url": "https://example.test/layer",
            "source_endpoint": "https://example.test/layer/ImageServer",
            "bbox_used": "0,0,1,1",
            "cache_hit": False,
            "ingest_diagnostics": {
                "fetch_attempted": True,
                "fetch_succeeded": True,
                "catalog_ingest_succeeded": True,
                "temp_input_path": str(tmp_path / "tmp.any"),
            },
        }

    monkeypatch.setattr(prep_orch, "build_catalog_coverage_plan", fake_coverage_plan)
    monkeypatch.setattr(prep_orch, "_ingest_layer_for_bbox", fake_ingest)

    with pytest.raises(RegionPrepExecutionError) as exc:
        prepare_region_from_catalog_or_sources(
            region_id="diag_empty_vector",
            display_name="Diag Empty Vector",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            catalog_root=tmp_path / "catalog",
            regions_root=tmp_path / "regions",
            source_config=_valid_required_source_config(),
            skip_optional_layers=True,
            overwrite=True,
        )
    details = exc.value.details
    fire = details["per_layer_execution_diagnostics"]["fire_perimeters"]
    assert fire["failure_reason"] == "empty_result"


def test_execution_diagnostics_classifies_invalid_request_url(monkeypatch, tmp_path):
    required = required_core_layers()

    def fake_coverage_plan(*args, **kwargs):
        return _coverage_plan_for_required(required, "none")

    def fake_ingest(*args, **kwargs):
        if kwargs["layer_key"] == "building_footprints":
            raise ValueError("invalid_request_url=endpoint https://example.test/FeatureServer/0")
        return {
            "item_id": f"{kwargs['layer_key']}-item",
            "catalog_path": str(tmp_path / f"{kwargs['layer_key']}.tif"),
            "provider_type": "arcgis_image_service",
            "acquisition_method": "bbox_export",
            "source_url": "https://example.test/layer",
            "source_endpoint": "https://example.test/layer/ImageServer",
            "bbox_used": "0,0,1,1",
            "cache_hit": False,
            "ingest_diagnostics": {
                "fetch_attempted": True,
                "fetch_succeeded": True,
                "catalog_ingest_succeeded": True,
                "temp_input_path": str(tmp_path / "tmp.any"),
            },
        }

    monkeypatch.setattr(prep_orch, "build_catalog_coverage_plan", fake_coverage_plan)
    monkeypatch.setattr(prep_orch, "_ingest_layer_for_bbox", fake_ingest)

    with pytest.raises(RegionPrepExecutionError) as exc:
        prepare_region_from_catalog_or_sources(
            region_id="diag_invalid_request_url",
            display_name="Diag Invalid Request URL",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            catalog_root=tmp_path / "catalog",
            regions_root=tmp_path / "regions",
            source_config=_valid_required_source_config(),
            skip_optional_layers=True,
            overwrite=True,
        )
    details = exc.value.details
    row = details["per_layer_execution_diagnostics"]["building_footprints"]
    assert row["failure_reason"] == "invalid_request_url"


def test_extract_provider_context_uses_actual_query_url_not_diagnostic_summary():
    err = (
        "provider_http_error: ArcGIS feature bbox query failed "
        "(endpoint=https://example.test/FeatureServer/0, "
        "attempted_formats=['geojson', 'json'], "
        "attempted_urls=['https://example.test/FeatureServer/0/query?f=geojson', "
        "'https://example.test/FeatureServer/0/query?f=json'], "
        "last_error=json_query_failed=invalid_request_url=endpoint https://example.test/FeatureServer/0; attempted formats=['geojson','json'])"
    )
    context = prep_orch._extract_provider_error_context(err)
    assert context["request_url"] == "https://example.test/FeatureServer/0/query?f=json"
    assert not str(context["request_url"]).startswith("endpoint ")


def test_extract_provider_context_strips_trailing_colon_from_query_url():
    err = (
        "Failed json request after retries for "
        "https://example.test/FeatureServer/0/query?where=1%3D1&f=json&resultOffset=0&resultRecordCount=2000: "
        "HTTP Error 400: Bad Request"
    )
    context = prep_orch._extract_provider_error_context(err)
    assert context["request_url"] == (
        "https://example.test/FeatureServer/0/query?where=1%3D1&f=json&resultOffset=0&resultRecordCount=2000"
    )
    assert not str(context["request_url"]).endswith(":")


def test_execution_diagnostics_flags_coverage_recording_failure(monkeypatch, tmp_path):
    required = required_core_layers()
    call_state = {"count": 0}

    def fake_coverage_plan(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return _coverage_plan_for_required(required, "none")
        after = _coverage_plan_for_required(required, "full")
        after["layers"]["canopy"]["coverage_status"] = "none"
        after["summary"]["required_missing"] = ["canopy"]
        after["summary"]["buildable_from_catalog"] = False
        after["summary"]["fully_covered_from_catalog"] = False
        return after

    def fake_ingest(*args, **kwargs):
        layer_key = kwargs["layer_key"]
        return {
            "item_id": f"{layer_key}-item",
            "catalog_path": str(tmp_path / f"{layer_key}.tif"),
            "provider_type": "arcgis_image_service",
            "acquisition_method": "bbox_export",
            "source_url": "https://example.test/layer",
            "source_endpoint": "https://example.test/layer/ImageServer",
            "bbox_used": "0,0,1,1",
            "cache_hit": False,
            "ingest_diagnostics": {
                "fetch_attempted": True,
                "fetch_succeeded": True,
                "catalog_ingest_succeeded": True,
                "temp_input_path": str(tmp_path / f"{layer_key}.tmp"),
            },
        }

    monkeypatch.setattr(prep_orch, "build_catalog_coverage_plan", fake_coverage_plan)
    monkeypatch.setattr(prep_orch, "_ingest_layer_for_bbox", fake_ingest)

    with pytest.raises(RegionPrepExecutionError) as exc:
        prepare_region_from_catalog_or_sources(
            region_id="diag_coverage_recheck",
            display_name="Diag Coverage Recheck",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            catalog_root=tmp_path / "catalog",
            regions_root=tmp_path / "regions",
            source_config=_valid_required_source_config(),
            skip_optional_layers=True,
            overwrite=True,
        )
    details = exc.value.details
    assert "canopy" in details["stage_failures"]["coverage_recheck"]
    canopy = details["per_layer_execution_diagnostics"]["canopy"]
    assert canopy["failure_reason"] == "coverage_recording_or_recheck_failure"


def test_successful_ingest_sets_non_none_coverage_status(monkeypatch, tmp_path):
    required = required_core_layers()
    call_state = {"count": 0}

    def fake_coverage_plan(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return _coverage_plan_for_required(required, "none")
        return _coverage_plan_for_required(required, "full")

    def fake_ingest(*args, **kwargs):
        layer_key = kwargs["layer_key"]
        suffix = ".geojson" if layer_key in {"fire_perimeters", "building_footprints"} else ".tif"
        return {
            "item_id": f"{layer_key}-item",
            "catalog_path": str(tmp_path / f"{layer_key}{suffix}"),
            "provider_type": kwargs["layer_cfg"].get("provider_type"),
            "acquisition_method": "bbox_export",
            "source_url": "https://example.test/layer",
            "source_endpoint": "https://example.test/endpoint",
            "bbox_used": "0,0,1,1",
            "cache_hit": False,
            "ingest_diagnostics": {
                "fetch_attempted": True,
                "fetch_succeeded": True,
                "catalog_ingest_succeeded": True,
                "temp_input_path": str(tmp_path / f"{layer_key}.tmp"),
            },
        }

    def fake_build_region(**kwargs):
        layers = {
            "dem": {},
            "slope": {},
            "fuel": {},
            "canopy": {},
            "fire_perimeters": {},
            "building_footprints": {},
        }
        return {"region_id": kwargs["region_id"], "layers": layers, "catalog": {}}

    monkeypatch.setattr(prep_orch, "build_catalog_coverage_plan", fake_coverage_plan)
    monkeypatch.setattr(prep_orch, "_ingest_layer_for_bbox", fake_ingest)
    monkeypatch.setattr(prep_orch, "build_region_from_catalog", fake_build_region)

    result = prepare_region_from_catalog_or_sources(
        region_id="diag_success",
        display_name="Diag Success",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        catalog_root=tmp_path / "catalog",
        regions_root=tmp_path / "regions",
        source_config=_valid_required_source_config(),
        skip_optional_layers=True,
        overwrite=True,
    )
    assert result["final_status"] == "success"
    for layer in required:
        assert result["per_layer_execution_diagnostics"][layer]["coverage_status_after_ingest"] == "full"


def test_cli_error_payload_includes_layer_execution_diagnostics():
    exc = RegionPrepExecutionError(
        "failed",
        details={
            "failed_required_layers": ["fuel"],
            "per_layer_execution_diagnostics": {"fuel": {"failure_reason": "remote_provider_error"}},
            "stage_failures": {"acquisition": ["fuel"], "ingest": [], "coverage_recheck": []},
        },
    )
    payload = _build_cli_error_payload(
        exc=exc,
        region_id="x",
        display_name="X",
        requested_bbox={"min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1},
        mode="executed",
    )
    assert payload["failed_required_layers"] == ["fuel"]
    assert "fuel" in payload["per_layer_execution_diagnostics"]
    assert payload["stage_failures"]["acquisition"] == ["fuel"]
