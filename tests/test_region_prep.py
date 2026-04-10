from __future__ import annotations

import argparse
import io
import json
import urllib.error
import zipfile
from pathlib import Path

import pytest

import backend.data_prep.prepare_region as prep_region
import backend.data_prep.sources.acquisition as source_acq
from backend.data_prep.prepare_region import parse_bbox, prepare_region_layers
from backend.data_prep.sources.adapters import (
    LANDFIRECanopyAdapter,
    LANDFIREFuelAdapter,
    MicrosoftBuildingFootprintAdapter,
    NIFCFirePerimeterAdapter,
    SourceAsset,
    USGS3DEPAdapter,
)

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTER_DEPS = True
except Exception:  # pragma: no cover - optional deps in CI
    np = None
    rasterio = None
    from_origin = None
    HAS_RASTER_DEPS = False


def _write_raster(path: Path, value: float = 10.0, width: int = 300, height: int = 300) -> None:
    if not HAS_RASTER_DEPS:
        pytest.skip("numpy/rasterio are required for raster prep tests")
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
                    "coordinates": [[[-0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [-0.2, 0.8], [-0.2, 0.2]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sources(tmp_path: Path, include_slope: bool = False) -> dict[str, str]:
    sources = {
        "dem": tmp_path / "dem_source.tif",
        "fuel": tmp_path / "fuel_source.tif",
        "canopy": tmp_path / "canopy_source.tif",
        "fire_perimeters": tmp_path / "fire_source.geojson",
        "building_footprints": tmp_path / "footprints_source.geojson",
    }
    _write_raster(sources["dem"], value=300.0)
    _write_raster(sources["fuel"], value=40.0)
    _write_raster(sources["canopy"], value=55.0)
    _write_geojson(sources["fire_perimeters"])
    _write_geojson(sources["building_footprints"])
    if include_slope:
        sources["slope"] = tmp_path / "slope_source.tif"
        _write_raster(sources["slope"], value=15.0)
    return {k: str(v) for k, v in sources.items()}


def _raster_bytes(tmp_path: Path, name: str = "bytes_raster.tif") -> bytes:
    path = tmp_path / name
    _write_raster(path, value=20.0)
    return path.read_bytes()


class _FakeHTTPResponse(io.BytesIO):
    status = 200
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class _FakeHTTPResponseWithHeaders(_FakeHTTPResponse):
    def __init__(self, payload: bytes, *, status: int = 200, content_type: str = "application/octet-stream"):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Type": content_type}


def test_parse_bbox_success_and_failure():
    bbox = parse_bbox("-123.1,37.5,-122.0,38.1")
    assert bbox["min_lon"] == -123.1
    assert bbox["max_lat"] == 38.1

    with pytest.raises(ValueError):
        parse_bbox("-123.1,37.5,-122.0")
    with pytest.raises(ValueError):
        parse_bbox("a,b,c,d")
    with pytest.raises(ValueError):
        parse_bbox("-122.0,38.0,-123.0,37.0")


def test_cli_style_bbox_parser_accepts_four_values():
    from scripts.prepare_region_layers import _parse_bbox_args

    bbox = _parse_bbox_args(["-123.1", "37.5", "-122.0", "38.1"])
    assert bbox["min_lon"] == -123.1
    assert bbox["max_lat"] == 38.1
    with pytest.raises(ValueError):
        _parse_bbox_args(["-122.0", "38.1", "-123.0", "37.5"])


def test_cli_checksum_metadata_builder():
    from scripts.prepare_region_layers import _build_source_metadata_from_args

    args = argparse.Namespace(
        dem_checksum="sha256:abc123",
        slope_checksum=None,
        fuel_checksum="sha256:def456",
        canopy_checksum=None,
        fire_perimeters_checksum=None,
        building_footprints_checksum="sha256:7890",
        burn_probability_checksum=None,
        wildfire_hazard_checksum=None,
        moisture_checksum=None,
        aspect_checksum=None,
    )
    meta = _build_source_metadata_from_args(args)
    assert meta["dem"]["checksum"] == "sha256:abc123"
    assert meta["fuel"]["checksum"] == "sha256:def456"
    assert meta["building_footprints"]["checksum"] == "sha256:7890"
    assert "slope" not in meta


def test_cli_source_config_loader(tmp_path):
    from scripts.prepare_region_layers import _load_source_config

    cfg = {"layers": {"fuel": {"provider_type": "arcgis_image_service", "source_endpoint": "https://example.test/fuel/ImageServer"}}}
    cfg_path = tmp_path / "sources.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = _load_source_config(str(cfg_path))
    assert loaded == cfg


def test_archive_member_selection_rejects_ambiguous_candidates(tmp_path):
    archive = tmp_path / "ambiguous_dem.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tile_a.tif", b"fake")
        zf.writestr("tile_b.tif", b"fake")
    with pytest.raises(ValueError) as exc:
        prep_region._select_archive_member(archive, layer_key="dem", layer_type="raster")
    assert "ambiguous" in str(exc.value).lower()


def test_prepare_region_local_mode_derives_slope(tmp_path):
    manifest = prepare_region_layers(
        region_id="pilot_region",
        display_name="Pilot Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=_sources(tmp_path, include_slope=False),
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
    )
    region_dir = tmp_path / "regions" / "pilot_region"
    assert manifest["preparation_status"] == "prepared"
    assert manifest["final_status"] == "success"
    assert (region_dir / "dem.tif").exists()
    assert (region_dir / "slope.tif").exists()
    assert manifest["layers"]["slope"]["source_type"] == "derived_from_dem"
    assert manifest["slope_derived"] is True


def test_prepare_region_manifest_includes_geometry_source_manifest_with_region_override(tmp_path):
    sources = _sources(tmp_path, include_slope=True)
    parcel_path = tmp_path / "parcel_polygons.geojson"
    parcel_override_path = tmp_path / "parcel_polygons_override.geojson"
    _write_geojson(parcel_path)
    _write_geojson(parcel_override_path)
    sources["parcel_polygons"] = str(parcel_path)
    sources["parcel_polygons_override"] = str(parcel_override_path)

    source_config = {
        "geometry_source_registry": {
            "version": 9,
            "defaults": {
                "source_order": {
                    "parcel_sources": ["parcel_polygons", "nearest_parcel_fallback"],
                    "footprint_sources": ["building_footprints", "fema_structures"],
                },
                "source_definitions": {
                    "parcel_sources": {
                        "parcel_polygons": {"layer_keys": ["parcel_polygons"]},
                        "nearest_parcel_fallback": {
                            "layer_keys": [],
                            "fallback_only": True,
                            "explicit_downgrade": True,
                        },
                    },
                    "footprint_sources": {
                        "building_footprints": {"layer_keys": ["building_footprints"]},
                        "fema_structures": {"layer_keys": ["fema_structures"]},
                    },
                },
            },
            "regions": {
                "geometry_override_region": {
                    "source_order": {
                        "parcel_sources": [
                            "parcel_polygons_override",
                            "parcel_polygons",
                            "nearest_parcel_fallback",
                        ]
                    },
                    "source_definitions": {
                        "parcel_sources": {
                            "parcel_polygons_override": {"layer_keys": ["parcel_polygons_override"]}
                        }
                    },
                }
            },
        }
    }

    manifest = prepare_region_layers(
        region_id="geometry_override_region",
        display_name="Geometry Override Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=sources,
        region_data_dir=tmp_path / "regions",
        source_config=source_config,
        auto_discover=False,
    )

    geometry_manifest = manifest.get("geometry_source_manifest") or {}
    assert geometry_manifest.get("version") == 9
    assert geometry_manifest.get("region_id") == "geometry_override_region"
    source_order = (geometry_manifest.get("default_source_order") or {}).get("parcel_sources") or []
    assert source_order[:2] == ["parcel_polygons_override", "parcel_polygons"]
    assert manifest.get("parcel_sources")[:2] == ["parcel_polygons_override", "parcel_polygons"]
    assert "building_footprints" in list(manifest.get("building_sources") or [])


def test_prepare_region_clips_rasters_to_bbox(tmp_path):
    src = _sources(tmp_path, include_slope=True)
    with rasterio.open(src["dem"]) as ds:
        src_w, src_h = ds.width, ds.height

    prepare_region_layers(
        region_id="clip_region",
        display_name="Clip Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.3, "max_lat": 0.3},
        layer_sources=src,
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
    )
    out_dem = tmp_path / "regions" / "clip_region" / "dem.tif"
    with rasterio.open(out_dem) as ds:
        assert ds.width < src_w
        assert ds.height < src_h


def test_download_retry_timeout_and_metadata(monkeypatch, tmp_path):
    sources = _sources(tmp_path, include_slope=True)
    dem_bytes = _raster_bytes(tmp_path, "download_dem.tif")
    call_state = {"count": 0, "timeouts": []}

    def fake_urlopen(url, timeout=0):
        call_state["count"] += 1
        call_state["timeouts"].append(timeout)
        if call_state["count"] < 3:
            raise TimeoutError("transient timeout")
        return _FakeHTTPResponse(dem_bytes)

    monkeypatch.setattr(prep_region.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prep_region.time, "sleep", lambda _s: None)
    sources.pop("dem")

    manifest = prepare_region_layers(
        region_id="retry_region",
        display_name="Retry Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=sources,
        layer_urls={"dem": "https://example.test/dem.tif"},
        region_data_dir=tmp_path / "regions",
        download_timeout=12.0,
        download_retries=3,
        retry_backoff_seconds=0.01,
        auto_discover=False,
    )
    assert call_state["count"] == 3
    assert all(t == 12.0 for t in call_state["timeouts"])
    dem_meta = manifest["layers"]["dem"]
    assert dem_meta["download_status"] == "ok"
    assert dem_meta["bytes_downloaded"] > 0
    assert dem_meta["retry_count_used"] == 2
    assert dem_meta["timeout_seconds"] == 12.0
    assert dem_meta["source_mode"] == "remote_url"


def test_dry_run_does_not_write_outputs(tmp_path):
    region_root = tmp_path / "regions"
    manifest = prepare_region_layers(
        region_id="dry_run_region",
        display_name="Dry Run Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources={"dem": _sources(tmp_path)["dem"]},
        region_data_dir=region_root,
        dry_run=True,
        auto_discover=False,
    )
    assert manifest["preparation_status"] == "dry_run"
    assert manifest["final_status"] == "dry_run_ready"
    assert manifest["errors"] == []
    assert not (region_root / "dry_run_region").exists()
    assert "dem" in manifest["attempted_layers"]
    assert "fuel" in manifest["skipped_layers"]
    assert "fuel" in manifest["unsupported_auto_discovery_layers"]


def test_dry_run_with_no_layers_is_partial_not_failed(tmp_path):
    manifest = prepare_region_layers(
        region_id="dry_run_none_region",
        display_name="Dry Run None Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        region_data_dir=tmp_path / "regions",
        dry_run=True,
        auto_discover=False,
    )
    assert manifest["preparation_status"] == "dry_run"
    assert manifest["final_status"] == "dry_run_partial"
    assert "dem" in manifest["required_blockers"]
    assert manifest["errors"] == []


def test_zip_archive_extraction_for_raster(tmp_path):
    src = _sources(tmp_path, include_slope=True)
    dem_zip = tmp_path / "dem_bundle.zip"
    dem_src = Path(src["dem"])
    with zipfile.ZipFile(dem_zip, "w") as zf:
        zf.write(dem_src, arcname="nested/dem_source.tif")
    src["dem"] = str(dem_zip)

    manifest = prepare_region_layers(
        region_id="zip_region",
        display_name="Zip Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=src,
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
    )
    dem_meta = manifest["layers"]["dem"]
    assert dem_meta["extraction_performed"] is True
    assert dem_meta["extracted_path"]
    assert manifest["archives_extracted"] is True


def test_bad_html_download_rejected(monkeypatch, tmp_path):
    sources = _sources(tmp_path, include_slope=True)
    sources.pop("dem")

    monkeypatch.setattr(
        prep_region.urllib.request,
        "urlopen",
        lambda _url, timeout=0: _FakeHTTPResponse(b"<html><body>Error 404</body></html>"),
    )

    with pytest.raises(ValueError) as exc:
        prepare_region_layers(
            region_id="html_fail_region",
            display_name="HTML Fail Region",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            layer_sources=sources,
            layer_urls={"dem": "https://example.test/dem.tif"},
            region_data_dir=tmp_path / "regions",
            auto_discover=False,
        )
    assert "html" in str(exc.value).lower()


def test_checksum_verification_success_and_failure(tmp_path):
    src = _sources(tmp_path, include_slope=True)
    dem_checksum = prep_region._sha256(Path(src["dem"]))
    ok_manifest = prepare_region_layers(
        region_id="checksum_ok_region",
        display_name="Checksum OK Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=src,
        source_metadata={"dem": {"checksum": f"sha256:{dem_checksum}"}},
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
    )
    assert ok_manifest["layers"]["dem"]["checksum_status"] == "verified"

    with pytest.raises(ValueError):
        prepare_region_layers(
            region_id="checksum_bad_region",
            display_name="Checksum BAD Region",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            layer_sources=src,
            source_metadata={"dem": {"checksum": "sha256:deadbeef"}},
            region_data_dir=tmp_path / "regions",
            auto_discover=False,
        )


def test_partial_mode_and_temp_cleanup_flags(monkeypatch, tmp_path):
    sources = _sources(tmp_path, include_slope=False)
    sources.pop("fuel")
    monkeypatch.setattr(prep_region.urllib.request, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("fail")))
    monkeypatch.setattr(prep_region.time, "sleep", lambda _s: None)

    partial = prepare_region_layers(
        region_id="partial_region",
        display_name="Partial Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=sources,
        layer_urls={"fuel": "https://example.test/fuel.tif"},
        region_data_dir=tmp_path / "regions",
        allow_partial=True,
        keep_temp_on_failure=True,
        clean_download_cache=True,
        auto_discover=False,
    )
    assert partial["preparation_status"] == "partial"
    assert partial["final_status"] == "partial"
    assert "fuel" in partial["failed_layers"]
    assert "dem" not in partial["required_blockers"]
    assert partial["warnings"]
    assert (tmp_path / "regions" / "partial_region" / "_downloads").exists()


def test_allow_partial_still_fails_when_minimum_layers_missing(monkeypatch, tmp_path):
    sources = _sources(tmp_path, include_slope=True)
    sources.pop("dem")
    monkeypatch.setattr(prep_region.urllib.request, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("fail")))
    monkeypatch.setattr(prep_region.time, "sleep", lambda _s: None)
    manifest = prepare_region_layers(
        region_id="partial_minimum_fail_region",
        display_name="Partial Minimum Fail Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=sources,
        layer_urls={"dem": "https://example.test/dem.tif"},
        region_data_dir=tmp_path / "regions",
        allow_partial=True,
        auto_discover=False,
    )
    assert manifest["final_status"] == "failed"
    assert "dem" in manifest["required_blockers"]


def test_usgs_dem_adapter_resolves_assets(monkeypatch):
    adapter = USGS3DEPAdapter()

    def fake_fetch_json(url: str, timeout: float = 30.0):
        return {
            "items": [
                {"downloadURL": "https://example.test/dem_a.tif", "title": "Tile A", "sourceId": "A1"},
                {"downloadURL": "https://example.test/dem_b.tif", "title": "Tile B", "sourceId": "B1"},
            ]
        }

    import backend.data_prep.sources.adapters as src_adapters

    monkeypatch.setattr(src_adapters, "_fetch_json", fake_fetch_json)
    assets = adapter.resolve_sources({"min_lon": -123.0, "min_lat": 37.0, "max_lon": -122.0, "max_lat": 38.0})
    assert len(assets) == 2
    assert assets[0].layer_key == "dem"
    assert assets[0].dataset_provider.startswith("USGS")


def test_microsoft_buildings_adapter_bbox_resolution(monkeypatch):
    csv_text = "QuadKey,Url\n0230102,https://example.test/q1.geojson\n0230103,https://example.test/q2.geojson\n"
    import backend.data_prep.sources.adapters as src_adapters

    monkeypatch.setenv("WF_MS_BUILDINGS_INDEX_URL", "https://example.test/index.csv")
    monkeypatch.setattr(src_adapters, "_fetch_text", lambda url, timeout=30.0: csv_text)
    adapter = MicrosoftBuildingFootprintAdapter()
    assets = adapter.resolve_sources({"min_lon": -123.0, "min_lat": 37.0, "max_lon": -122.0, "max_lat": 38.0})
    assert isinstance(assets, list)
    for asset in assets:
        assert asset.layer_key == "building_footprints"


def test_nifc_adapter_uses_configured_url(monkeypatch):
    monkeypatch.setenv("WF_NIFC_FIRE_PERIMETERS_URL", "https://example.test/nifc.geojson")
    adapter = NIFCFirePerimeterAdapter()
    assets = adapter.resolve_sources({"min_lon": -123.0, "min_lat": 37.0, "max_lon": -122.0, "max_lat": 38.0})
    assert len(assets) == 1
    assert assets[0].layer_key == "fire_perimeters"
    assert assets[0].url == "https://example.test/nifc.geojson"


def test_landfire_template_adapter_resolution(monkeypatch):
    monkeypatch.setenv(
        "WF_LANDFIRE_FUEL_URL_TEMPLATE",
        "https://example.test/fuel?bbox={min_lon},{min_lat},{max_lon},{max_lat}",
    )
    monkeypatch.setenv(
        "WF_LANDFIRE_CANOPY_URL_TEMPLATE",
        "https://example.test/canopy?bbox={bbox}",
    )
    fuel_assets = LANDFIREFuelAdapter().resolve_sources(
        {"min_lon": -123.0, "min_lat": 37.0, "max_lon": -122.0, "max_lat": 38.0}
    )
    canopy_assets = LANDFIRECanopyAdapter().resolve_sources(
        {"min_lon": -123.0, "min_lat": 37.0, "max_lon": -122.0, "max_lat": 38.0}
    )
    assert len(fuel_assets) == 1
    assert len(canopy_assets) == 1
    assert "bbox=-123.0,37.0,-122.0,38.0" in canopy_assets[0].url
    assert fuel_assets[0].layer_key == "fuel"
    assert canopy_assets[0].layer_key == "canopy"


def test_auto_discovery_prepares_region_from_bbox_only(monkeypatch, tmp_path):
    src = _sources(tmp_path, include_slope=False)
    assets = {
        "dem": [SourceAsset(url=Path(src["dem"]).as_uri(), dataset_name="USGS", dataset_version="1", dataset_provider="USGS", layer_key="dem", layer_type="raster", expected_format="tif", tile_id="dem1")],
        "fuel": [SourceAsset(url=Path(src["fuel"]).as_uri(), dataset_name="LANDFIRE Fuel", dataset_version="1", dataset_provider="LANDFIRE", layer_key="fuel", layer_type="raster", expected_format="tif", tile_id="f1")],
        "canopy": [SourceAsset(url=Path(src["canopy"]).as_uri(), dataset_name="LANDFIRE Canopy", dataset_version="1", dataset_provider="LANDFIRE", layer_key="canopy", layer_type="raster", expected_format="tif", tile_id="c1")],
        "fire_perimeters": [SourceAsset(url=Path(src["fire_perimeters"]).as_uri(), dataset_name="NIFC", dataset_version="1", dataset_provider="NIFC", layer_key="fire_perimeters", layer_type="vector", expected_format="geojson")],
        "building_footprints": [SourceAsset(url=Path(src["building_footprints"]).as_uri(), dataset_name="MS Buildings", dataset_version="1", dataset_provider="Microsoft", layer_key="building_footprints", layer_type="vector", expected_format="geojson", tile_id="q1")],
    }
    monkeypatch.setattr(prep_region, "discover_wildfire_sources", lambda bbox: assets)

    manifest = prepare_region_layers(
        region_id="bbox_only_region",
        display_name="BBox Only Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        region_data_dir=tmp_path / "regions",
        auto_discover=True,
    )
    assert manifest["preparation_status"] == "prepared"
    assert manifest["files"]["dem"] == "dem.tif"
    assert manifest["layers"]["dem"]["discovered_source"] is True
    assert manifest["layers"]["dem"]["dataset_provider"] == "USGS"
    assert manifest["layers"]["dem"]["dataset_source"] == "USGS"
    assert manifest["layers"]["building_footprints"]["tile_ids"] == ["q1"]
    assert manifest["layers"]["dem"]["download_url"]


def test_download_cache_reuse(monkeypatch, tmp_path):
    src = _sources(tmp_path, include_slope=False)
    dem_bytes = Path(src["dem"]).read_bytes()
    call_count = {"n": 0}

    def fake_urlopen(url, timeout=0):
        call_count["n"] += 1
        return _FakeHTTPResponse(dem_bytes)

    monkeypatch.setattr(prep_region.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prep_region.time, "sleep", lambda _s: None)

    base_sources = _sources(tmp_path, include_slope=True)
    base_sources.pop("dem")
    common_cache = tmp_path / "cache"
    manifest_a = prepare_region_layers(
        region_id="cache_a",
        display_name="Cache A",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=base_sources,
        layer_urls={"dem": "https://example.test/cached_dem.tif"},
        region_data_dir=tmp_path / "regions",
        cache_dir=common_cache,
        auto_discover=False,
    )
    manifest_b = prepare_region_layers(
        region_id="cache_b",
        display_name="Cache B",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=base_sources,
        layer_urls={"dem": "https://example.test/cached_dem.tif"},
        region_data_dir=tmp_path / "regions",
        cache_dir=common_cache,
        auto_discover=False,
    )
    assert call_count["n"] == 1
    assert manifest_a["layers"]["dem"]["download_status"] == "ok"
    assert manifest_b["layers"]["dem"]["download_status"] == "cache_hit"


def test_arcgis_bbox_export_url_and_request(monkeypatch, tmp_path):
    captured = {"url": None}
    dem_bytes = _raster_bytes(tmp_path, "bbox_dem.tif")

    def fake_urlopen(url, timeout=0):
        captured["url"] = url
        return _FakeHTTPResponse(dem_bytes)

    monkeypatch.setattr(source_acq.urllib.request, "urlopen", fake_urlopen)
    result = source_acq.acquire_layer_from_config(
        layer_key="fuel",
        layer_type="raster",
        layer_config={
            "provider_type": "arcgis_image_service",
            "source_endpoint": "https://example.test/arcgis/rest/services/LANDFIRE_Fuel/ImageServer",
        },
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=30.0,
        timeout_seconds=20.0,
        retries=1,
        backoff_seconds=0.01,
    )
    assert result is not None
    assert result.acquisition_method == "bbox_export"
    assert result.local_path and Path(result.local_path).exists()
    assert captured["url"] is not None
    assert "exportImage" in str(captured["url"])
    assert "bbox=" in str(captured["url"])
    assert "format=tiff" in str(captured["url"])


def test_arcgis_bbox_export_cache_reuse(monkeypatch, tmp_path):
    call_count = {"n": 0}
    dem_bytes = _raster_bytes(tmp_path, "bbox_cached_dem.tif")

    def fake_urlopen(url, timeout=0):
        call_count["n"] += 1
        return _FakeHTTPResponse(dem_bytes)

    monkeypatch.setattr(source_acq.urllib.request, "urlopen", fake_urlopen)
    kwargs = dict(
        layer_key="canopy",
        layer_type="raster",
        layer_config={
            "provider_type": "arcgis_image_service",
            "source_endpoint": "https://example.test/arcgis/rest/services/LANDFIRE_Canopy/ImageServer",
        },
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=30.0,
        timeout_seconds=20.0,
        retries=1,
        backoff_seconds=0.01,
    )
    first = source_acq.acquire_layer_from_config(**kwargs)
    second = source_acq.acquire_layer_from_config(**kwargs)
    assert first is not None and second is not None
    assert call_count["n"] == 1
    assert first.acquisition_method == "bbox_export"
    assert second.acquisition_method == "cached_bbox_export"
    assert second.cache_hit is True


def test_bbox_export_fallbacks_to_full_download_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        source_acq.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("bbox export failed")),
    )
    result = source_acq.acquire_layer_from_config(
        layer_key="fuel",
        layer_type="raster",
        layer_config={
            "provider_type": "arcgis_image_service",
            "source_endpoint": "https://example.test/arcgis/rest/services/LANDFIRE_Fuel/ImageServer",
            "full_download_url": "https://example.test/fuel_full.zip",
        },
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=30.0,
        timeout_seconds=1.0,
        retries=0,
        backoff_seconds=0.0,
    )
    assert result is not None
    assert result.acquisition_method == "full_download_clip"
    assert result.source_url == "https://example.test/fuel_full.zip"
    assert result.warnings


def test_manifest_records_bbox_acquisition_method(monkeypatch, tmp_path):
    src = _sources(tmp_path, include_slope=False)
    dem_bytes = Path(src["dem"]).read_bytes()
    src.pop("dem")

    def fake_urlopen(url, timeout=0):
        return _FakeHTTPResponse(dem_bytes)

    monkeypatch.setattr(source_acq.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prep_region.urllib.request, "urlopen", fake_urlopen)

    manifest = prepare_region_layers(
        region_id="bbox_manifest_region",
        display_name="BBox Manifest Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=src,
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
        source_config={
            "layers": {
                "dem": {
                    "provider_type": "arcgis_image_service",
                    "source_endpoint": "https://example.test/arcgis/rest/services/DEM/ImageServer",
                }
            }
        },
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=30.0,
    )
    dem_meta = manifest["layers"]["dem"]
    assert dem_meta["acquisition_method"] in {"bbox_export", "cached_bbox_export"}
    assert dem_meta["provider_type"] == "arcgis_image_service"
    assert dem_meta["source_endpoint"] == "https://example.test/arcgis/rest/services/DEM/ImageServer"
    assert dem_meta["bbox_used"]


def test_feature_service_geojson_fallback_to_json(monkeypatch, tmp_path):
    esri_json = {
        "features": [
            {
                "attributes": {"id": 1, "name": "a"},
                "geometry": {"rings": [[[-111.0, 45.6], [-110.9, 45.6], [-110.9, 45.7], [-111.0, 45.7], [-111.0, 45.6]]]},
            }
        ]
    }

    def fake_urlopen(url, timeout=0):
        if "f=geojson" in str(url):
            body = io.BytesIO(b'{"error":{"message":"Format geojson not supported"}}')
            raise urllib.error.HTTPError(str(url), 400, "Bad Request", {"Content-Type": "application/json"}, body)
        if "f=json" in str(url):
            # A real server returns an empty features list once offset exceeds
            # the total feature count.  The adaptive pagination loop relies on
            # this to terminate when the server does not set exceededTransferLimit.
            if "resultOffset=0" in str(url):
                return _FakeHTTPResponseWithHeaders(
                    json.dumps(esri_json).encode("utf-8"),
                    status=200,
                    content_type="application/json",
                )
            return _FakeHTTPResponseWithHeaders(
                json.dumps({"features": []}).encode("utf-8"),
                status=200,
                content_type="application/json",
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(source_acq.urllib.request, "urlopen", fake_urlopen)
    result = source_acq.acquire_layer_from_config(
        layer_key="fire_perimeters",
        layer_type="vector",
        layer_config={
            "provider_type": "arcgis_feature_service",
            "source_endpoint": "https://example.test/FeatureServer/0",
            "supports_geojson_direct": True,
            "query_format": "geojson",
        },
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=None,
        timeout_seconds=10.0,
        retries=0,
        backoff_seconds=0.0,
    )
    assert result is not None
    assert result.acquisition_method == "bbox_export_json_fallback"
    assert any("geojson_query_failed" in w for w in result.warnings)
    assert "geojson_unsupported_fallback_to_json_succeeded" in result.warnings
    out = Path(str(result.local_path))
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("type") == "FeatureCollection"
    assert len(payload.get("features", [])) == 1


def test_feature_service_json_paginates(monkeypatch, tmp_path):
    page0 = {
        "features": [
            {"attributes": {"id": 1}, "geometry": {"x": -111.0, "y": 45.6}},
            {"attributes": {"id": 2}, "geometry": {"x": -110.99, "y": 45.61}},
        ],
        "exceededTransferLimit": True,
    }
    page1 = {
        "features": [
            {"attributes": {"id": 3}, "geometry": {"x": -110.98, "y": 45.62}},
        ],
        "exceededTransferLimit": False,
    }

    def fake_urlopen(url, timeout=0):
        text = str(url)
        if "resultOffset=0" in text and "resultRecordCount=2" in text:
            return _FakeHTTPResponseWithHeaders(
                json.dumps(page0).encode("utf-8"),
                status=200,
                content_type="application/json",
            )
        if "resultOffset=2" in text and "resultRecordCount=2" in text:
            return _FakeHTTPResponseWithHeaders(
                json.dumps(page1).encode("utf-8"),
                status=200,
                content_type="application/json",
            )
        raise AssertionError(f"unexpected url: {text}")

    monkeypatch.setenv("WF_ARCGIS_FEATURE_QUERY_PAGE_SIZE", "2")
    monkeypatch.setattr(source_acq.urllib.request, "urlopen", fake_urlopen)
    result = source_acq.acquire_layer_from_config(
        layer_key="fire_perimeters",
        layer_type="vector",
        layer_config={
            "provider_type": "arcgis_feature_service",
            "source_endpoint": "https://example.test/FeatureServer/0",
            "supports_geojson_direct": False,
            "query_format": "json",
        },
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=None,
        timeout_seconds=10.0,
        retries=0,
        backoff_seconds=0.0,
    )
    assert result is not None
    assert result.acquisition_method == "bbox_export_json_fallback"
    assert "json_exceeded_transfer_limit_detected" in result.warnings
    assert "json_pagination_requests=2" in result.warnings
    out = Path(str(result.local_path))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("type") == "FeatureCollection"
    assert len(payload.get("features", [])) == 3


def test_feature_service_paginates_when_server_caps_below_page_size(monkeypatch, tmp_path):
    # Simulate a server whose maxRecordCount (1) is smaller than our requested
    # page_size (2).  Without adaptive batch-size detection the loop would stop
    # after the first page because current_count (1) < page_size (2).  With the
    # fix it detects effective_batch_size=1 from the first page and continues
    # until a page returns fewer than 1 feature.
    page0 = {
        "features": [{"attributes": {"id": 1}, "geometry": {"x": -111.0, "y": 45.6}}],
        "exceededTransferLimit": False,
    }
    page1 = {
        "features": [{"attributes": {"id": 2}, "geometry": {"x": -110.99, "y": 45.61}}],
        "exceededTransferLimit": False,
    }
    page2 = {
        "features": [],
        "exceededTransferLimit": False,
    }

    call_count = {"n": 0}

    def fake_urlopen(url, timeout=0):
        text = str(url)
        n = call_count["n"]
        call_count["n"] += 1
        if "resultOffset=0" in text:
            return _FakeHTTPResponseWithHeaders(
                json.dumps(page0).encode("utf-8"),
                status=200,
                content_type="application/json",
            )
        if "resultOffset=1" in text:
            return _FakeHTTPResponseWithHeaders(
                json.dumps(page1).encode("utf-8"),
                status=200,
                content_type="application/json",
            )
        if "resultOffset=2" in text:
            return _FakeHTTPResponseWithHeaders(
                json.dumps(page2).encode("utf-8"),
                status=200,
                content_type="application/json",
            )
        raise AssertionError(f"unexpected url at call {n}: {text}")

    monkeypatch.setenv("WF_ARCGIS_FEATURE_QUERY_PAGE_SIZE", "2")
    monkeypatch.setattr(source_acq.urllib.request, "urlopen", fake_urlopen)
    result = source_acq.acquire_layer_from_config(
        layer_key="fire_perimeters",
        layer_type="vector",
        layer_config={
            "provider_type": "arcgis_feature_service",
            "source_endpoint": "https://example.test/FeatureServer/0",
            "supports_geojson_direct": False,
            "query_format": "json",
        },
        bounds={"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        cache_root=tmp_path / "cache",
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
        target_resolution=None,
        timeout_seconds=10.0,
        retries=0,
        backoff_seconds=0.0,
    )
    assert result is not None
    out = Path(str(result.local_path))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("type") == "FeatureCollection"
    # Both features from page 0 and page 1 must be present; the naive check
    # would have stopped after page 0 and returned only 1 feature.
    assert len(payload.get("features", [])) == 2


def test_acquisition_url_sanitize_preserves_valid_https_url():
    url = "https://example.test/FeatureServer/0/query?where=1%3D1&f=json"
    assert source_acq._sanitize_url(url) == url


def test_acquisition_url_sanitize_strips_accidental_trailing_punctuation():
    url = "https://example.test/FeatureServer/0/query?where=1%3D1&f=json:  "
    assert source_acq._sanitize_url(url) == "https://example.test/FeatureServer/0/query?where=1%3D1&f=json"


def test_acquisition_validate_request_url_rejects_malformed_url():
    with pytest.raises(ValueError) as exc:
        source_acq._validate_request_url("endpoint https://example.test/FeatureServer/0")
    assert "invalid_request_url=" in str(exc.value)


def test_skip_optional_layers_flag(tmp_path):
    src = _sources(tmp_path, include_slope=False)
    optional = tmp_path / "burn_probability_source.tif"
    _write_raster(optional, value=60.0)
    src["burn_probability"] = str(optional)

    manifest = prepare_region_layers(
        region_id="skip_optional_region",
        display_name="Skip Optional Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=src,
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
        skip_optional_layers=True,
    )
    assert manifest["final_status"] == "success"
    assert "burn_probability" not in manifest["files"]
    assert any("skip-optional-layers" in w for w in manifest["warnings"])


def test_prepare_pipeline_falls_back_from_bbox_to_full_download(monkeypatch, tmp_path):
    src = _sources(tmp_path, include_slope=False)
    dem_local = src.pop("dem")

    monkeypatch.setattr(
        source_acq.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("bbox export failed")),
    )

    manifest = prepare_region_layers(
        region_id="bbox_fallback_region",
        display_name="BBox Fallback Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=src,
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
        source_config={
            "layers": {
                "dem": {
                    "provider_type": "arcgis_image_service",
                    "source_endpoint": "https://example.test/arcgis/rest/services/DEM/ImageServer",
                    "full_download_url": Path(dem_local).as_uri(),
                }
            }
        },
        prefer_bbox_downloads=True,
        allow_full_download_fallback=True,
    )
    dem_meta = manifest["layers"]["dem"]
    assert dem_meta["acquisition_method"] == "full_download_clip"
    assert any("bbox export failed" in w for w in manifest["warnings"])


def test_require_core_layers_flag_controls_blockers(tmp_path):
    src = _sources(tmp_path, include_slope=False)
    src.pop("fuel")
    src.pop("canopy")
    src.pop("fire_perimeters")
    src.pop("building_footprints")
    manifest = prepare_region_layers(
        region_id="minimum_only_region",
        display_name="Minimum Only Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=src,
        region_data_dir=tmp_path / "regions",
        auto_discover=False,
        require_core_layers=False,
    )
    assert manifest["final_status"] == "success"
    assert manifest["required_blockers"] == []
    assert manifest["minimum_required_missing"] == []
