from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.data_prep.catalog import ingest_catalog_raster, ingest_catalog_vector
from scripts.catalog_coverage import build_catalog_coverage_plan, required_core_layers
from scripts.prepare_region_from_catalog_or_sources import prepare_region_from_catalog_or_sources

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
    assert "required core layer blockers" in str(exc.value).lower()


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
    assert executed["source_registry"]["default_source_registry_used"] is True
    assert sorted(item["layer_key"] for item in executed["acquired_layers"]) == sorted(required_core_layers())
    assert (regions_root / "new_area_smoke" / "manifest.json").exists()
    validation = executed.get("validation")
    assert isinstance(validation, dict)
    assert validation.get("validation_status") in {"passed", "failed"}
    if validation.get("validation_status") == "failed":
        assert validation.get("blockers"), "Validation failures should include structured blockers."
