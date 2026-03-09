from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.data_prep.catalog import (
    build_region_from_catalog,
    ingest_catalog_raster,
    ingest_catalog_vector,
    load_catalog_index,
)
from backend.data_prep import catalog as catalog_mod
from backend.data_prep.sources.acquisition import AcquisitionResult
from backend.data_prep.prepare_region import prepare_region_layers

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
        pytest.skip("numpy/rasterio are required for catalog tests")
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
                    "coordinates": [[[-0.4, 0.2], [0.8, 0.2], [0.8, 0.9], [-0.4, 0.9], [-0.4, 0.2]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_empty_geojson(path: Path) -> None:
    payload = {"type": "FeatureCollection", "features": []}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_catalog_sources(tmp_path: Path) -> dict[str, Path]:
    sources = {
        "dem": tmp_path / "dem.tif",
        "fuel": tmp_path / "fuel.tif",
        "canopy": tmp_path / "canopy.tif",
        "fire_perimeters": tmp_path / "fire.geojson",
        "building_footprints": tmp_path / "buildings.geojson",
    }
    _write_raster(sources["dem"], value=320.0)
    _write_raster(sources["fuel"], value=45.0)
    _write_raster(sources["canopy"], value=60.0)
    _write_geojson(sources["fire_perimeters"])
    _write_geojson(sources["building_footprints"])
    return sources


def test_load_catalog_index_empty(tmp_path):
    index = load_catalog_index(tmp_path / "catalog")
    assert index["layers"] == {}
    assert str(tmp_path / "catalog") in index["catalog_root"]


def test_catalog_raster_ingest_creates_metadata_and_index(tmp_path):
    src = _seed_catalog_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    meta = ingest_catalog_raster(
        layer_name="dem",
        source_path=str(src["dem"]),
        catalog_root=catalog_root,
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.8, "max_lat": 0.8},
    )
    assert Path(meta["catalog_path"]).exists()
    assert meta["layer_name"] == "dem"
    index = load_catalog_index(catalog_root)
    assert "dem" in index["layers"]
    assert index["layers"]["dem"]["entries"]
    assert index["layers"]["dem"]["entries"][0]["catalog_path"] == meta["catalog_path"]


def test_catalog_vector_ingest_reuses_existing_entry(tmp_path):
    src = _seed_catalog_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    first = ingest_catalog_vector(
        layer_name="fire_perimeters",
        source_path=str(src["fire_perimeters"]),
        catalog_root=catalog_root,
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.8, "max_lat": 0.8},
    )
    second = ingest_catalog_vector(
        layer_name="fire_perimeters",
        source_path=str(src["fire_perimeters"]),
        catalog_root=catalog_root,
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.8, "max_lat": 0.8},
    )
    assert first["item_id"] == second["item_id"]
    assert second["cache_hit"] is True
    index = load_catalog_index(catalog_root)
    entries = index["layers"]["fire_perimeters"]["entries"]
    ids = [e["item_id"] for e in entries]
    assert ids.count(first["item_id"]) == 1


def test_catalog_vector_ingest_empty_feature_set_fails_clearly(tmp_path):
    if catalog_mod.shape is None:
        pytest.skip("shapely is required for vector ingest tests")
    catalog_root = tmp_path / "catalog"
    empty = tmp_path / "empty.geojson"
    _write_empty_geojson(empty)
    with pytest.raises(ValueError) as exc:
        ingest_catalog_vector(
            layer_name="fire_perimeters",
            source_path=str(empty),
            catalog_root=catalog_root,
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.8, "max_lat": 0.8},
        )
    assert "empty feature set" in str(exc.value).lower()


def test_resolve_ingest_input_prefers_bbox_for_endpoint_only_sources(monkeypatch, tmp_path):
    source_file = tmp_path / "fire.geojson"
    source_file.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {"type": "Point", "coordinates": [0.1, 0.1]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    call_args: dict[str, object] = {}

    def fake_acquire_layer_from_config(**kwargs):
        call_args["prefer_bbox_downloads"] = kwargs["prefer_bbox_downloads"]
        return AcquisitionResult(
            layer_key="fire_perimeters",
            provider_type="arcgis_feature_service",
            acquisition_method="bbox_export",
            source_endpoint="https://example.test/fire/FeatureServer/0",
            source_url="https://example.test/fire/FeatureServer/0/query?f=geojson",
            local_path=str(source_file),
            bbox_used="0,0,1,1",
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )

    monkeypatch.setattr(catalog_mod, "acquire_layer_from_config", fake_acquire_layer_from_config)

    resolved_path, meta = catalog_mod._resolve_ingest_input(
        layer_name="fire_perimeters",
        layer_type="vector",
        source_path=None,
        source_url=None,
        source_endpoint="https://example.test/fire/FeatureServer/0",
        provider_type="arcgis_feature_service",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=False,
        allow_full_download_fallback=True,
        target_resolution=None,
        timeout_seconds=10.0,
        retries=0,
        backoff_seconds=0.0,
    )

    assert call_args["prefer_bbox_downloads"] is True
    assert resolved_path == source_file
    assert meta["acquisition_method"] == "bbox_export"


def test_build_region_from_catalog_success_and_manifest_provenance(tmp_path):
    src = _seed_catalog_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    regions_root = tmp_path / "regions"
    bbox = {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.7, "max_lat": 0.7}
    ingest_catalog_raster(layer_name="dem", source_path=str(src["dem"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_raster(layer_name="fuel", source_path=str(src["fuel"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_raster(layer_name="canopy", source_path=str(src["canopy"]), catalog_root=catalog_root, bounds=bbox)
    ingest_catalog_vector(
        layer_name="fire_perimeters",
        source_path=str(src["fire_perimeters"]),
        catalog_root=catalog_root,
        bounds=bbox,
    )
    ingest_catalog_vector(
        layer_name="building_footprints",
        source_path=str(src["building_footprints"]),
        catalog_root=catalog_root,
        bounds=bbox,
    )
    manifest = build_region_from_catalog(
        region_id="catalog_region",
        display_name="Catalog Region",
        bounds=bbox,
        catalog_root=catalog_root,
        regions_root=regions_root,
        overwrite=True,
        require_core_layers=True,
        skip_optional_layers=True,
    )
    region_dir = regions_root / "catalog_region"
    assert (region_dir / "dem.tif").exists()
    assert (region_dir / "slope.tif").exists()
    assert manifest["catalog"]["used"] is True
    assert manifest["layers"]["dem"]["built_from_catalog"] is True
    assert manifest["layers"]["fuel"]["catalog_source_path"]


def test_build_region_from_catalog_missing_core_layer_fails(tmp_path):
    src = _seed_catalog_sources(tmp_path)
    catalog_root = tmp_path / "catalog"
    bbox = {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.7, "max_lat": 0.7}
    ingest_catalog_raster(layer_name="dem", source_path=str(src["dem"]), catalog_root=catalog_root, bounds=bbox)
    with pytest.raises(ValueError) as exc:
        build_region_from_catalog(
            region_id="catalog_fail",
            display_name="Catalog Fail",
            bounds=bbox,
            catalog_root=catalog_root,
            regions_root=tmp_path / "regions",
            require_core_layers=True,
        )
    assert "missing required core layers" in str(exc.value).lower()


def test_build_region_from_empty_catalog_fails_fast(tmp_path):
    with pytest.raises(ValueError) as exc:
        build_region_from_catalog(
            region_id="catalog_empty_fail",
            display_name="Catalog Empty Fail",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.7, "max_lat": 0.7},
            catalog_root=tmp_path / "catalog",
            regions_root=tmp_path / "regions",
            require_core_layers=True,
        )
    assert "missing required core layers" in str(exc.value).lower()


def test_prepare_region_backward_compat_without_catalog(tmp_path):
    src = _seed_catalog_sources(tmp_path)
    manifest = prepare_region_layers(
        region_id="legacy_prep_region",
        display_name="Legacy Prep Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.7, "max_lat": 0.7},
        layer_sources={k: str(v) for k, v in src.items()},
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
    )
    assert manifest["final_status"] == "success"
    assert manifest["files"]["dem"] == "dem.tif"
