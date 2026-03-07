from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import backend.data_prep.prepare_region as prep_region
from backend.data_prep.prepare_region import parse_bbox, prepare_region_layers

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin


def _write_raster(path: Path, value: float = 10.0, width: int = 300, height: int = 300) -> None:
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
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


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


def test_prepare_region_local_mode_derives_slope(tmp_path):
    manifest = prepare_region_layers(
        region_id="pilot_region",
        display_name="Pilot Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=_sources(tmp_path, include_slope=False),
        region_data_dir=tmp_path / "regions",
    )
    region_dir = tmp_path / "regions" / "pilot_region"
    assert manifest["preparation_status"] == "prepared"
    assert manifest["final_status"] == "success"
    assert (region_dir / "dem.tif").exists()
    assert (region_dir / "slope.tif").exists()
    assert manifest["layers"]["slope"]["source_type"] == "derived_from_dem"
    assert manifest["slope_derived"] is True


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
    )
    assert manifest["preparation_status"] == "dry_run"
    assert not (region_root / "dry_run_region").exists()


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
        )


def test_partial_mode_and_temp_cleanup_flags(monkeypatch, tmp_path):
    sources = _sources(tmp_path, include_slope=True)
    sources.pop("dem")
    monkeypatch.setattr(prep_region.urllib.request, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("fail")))
    monkeypatch.setattr(prep_region.time, "sleep", lambda _s: None)

    partial = prepare_region_layers(
        region_id="partial_region",
        display_name="Partial Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=sources,
        layer_urls={"dem": "https://example.test/dem.tif"},
        region_data_dir=tmp_path / "regions",
        allow_partial=True,
        keep_temp_on_failure=True,
        clean_download_cache=True,
    )
    assert partial["preparation_status"] == "partial"
    assert partial["final_status"] == "partial"
    assert "dem" in partial["failed_layers"]
    assert partial["warnings"]
    assert (tmp_path / "regions" / "partial_region" / "_downloads").exists()
