from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from backend.data_prep.sources import landfire as landfire_mod
from backend.data_prep.sources.landfire import (
    download_with_resume,
    load_latest_staged_assets,
    stage_landfire_assets,
)
from backend.data_prep.prepare_region import prepare_region_layers
from scripts.prepare_region_layers import _apply_landfire_cache_policy
from scripts.build_landfire_region import _can_reuse_existing_region

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover
    np = None
    rasterio = None
    from_origin = None


class _FakeHTTPResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _write_landfire_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("fuel_model.tif", b"fuel-bytes")
        zf.writestr("canopy_cover.tif", b"canopy-bytes")


def _write_raster(path: Path, value: float = 10.0, width: int = 256, height: int = 256) -> None:
    if rasterio is None or np is None or from_origin is None:
        pytest.skip("rasterio/numpy required")
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


def test_download_resume_and_cache_hit(monkeypatch, tmp_path):
    data = b"abcdefghijklmnopqrstuvwxyz"
    calls = {"count": 0}

    cache_dir = tmp_path / "cache"
    target = landfire_mod.cache_path_for_url("https://example.test/lf.zip", cache_dir=cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    part.write_bytes(data[:10])

    def fake_urlopen(req, timeout=0):
        calls["count"] += 1
        headers = getattr(req, "headers", {})
        range_header = headers.get("Range") or headers.get("range")
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            return _FakeHTTPResponse(data[start:], status=206)
        return _FakeHTTPResponse(data, status=200)

    monkeypatch.setattr(landfire_mod.urllib.request, "urlopen", fake_urlopen)

    first = download_with_resume(
        url="https://example.test/lf.zip",
        cache_dir=cache_dir,
        retries=0,
        timeout_seconds=5.0,
        retry_backoff_seconds=0.01,
        force_redownload=False,
        progress_log=[],
    )
    assert Path(first["archive_path"]).read_bytes() == data
    assert first["resumed"] is True
    assert first["cache_hit_download"] is False

    second = download_with_resume(
        url="https://example.test/lf.zip",
        cache_dir=cache_dir,
        retries=0,
        timeout_seconds=5.0,
        retry_backoff_seconds=0.01,
        force_redownload=False,
        progress_log=[],
    )
    assert second["cache_hit_download"] is True
    assert calls["count"] == 1


def test_stage_assets_and_reuse_extraction(tmp_path):
    archive = tmp_path / "landfire.zip"
    _write_landfire_archive(archive)
    uri = archive.as_uri()
    cache_root = tmp_path / "cache"

    first = stage_landfire_assets(
        fuel_url=uri,
        canopy_url=uri,
        cache_root=cache_root,
        retries=0,
    )
    assert first["layers"]["fuel"]["cache_hit_download"] is False
    assert first["layers"]["fuel"]["cache_hit_extract"] is False
    assert Path(first["layers"]["fuel"]["extracted_path"]).exists()

    second = stage_landfire_assets(
        fuel_url=uri,
        canopy_url=uri,
        cache_root=cache_root,
        retries=0,
    )
    assert second["layers"]["fuel"]["cache_hit_download"] is True
    assert second["layers"]["fuel"]["cache_hit_extract"] is True
    latest = load_latest_staged_assets(cache_root)
    assert isinstance(latest, dict)
    assert "fuel" in (latest.get("layers") or {})


def test_cache_only_landfire_failure_mode(tmp_path):
    with pytest.raises(ValueError):
        _apply_landfire_cache_policy(
            layer_sources={},
            layer_urls={},
            cache_root=tmp_path / "cache",
            cache_only_landfire=True,
            stage_landfire_first=False,
            timeout=5.0,
            retries=0,
            backoff=0.01,
            fuel_checksum=None,
            canopy_checksum=None,
        )


def test_bbox_too_large_safety_limit(tmp_path):
    raster = tmp_path / "fuel.tif"
    _write_raster(raster, value=40.0, width=600, height=600)
    with pytest.raises(ValueError) as exc:
        prepare_region_layers(
            region_id="big_bbox_fail",
            display_name="Big BBox Fail",
            bounds={"min_lon": -1.0, "min_lat": -1.0, "max_lon": 2.0, "max_lat": 2.0},
            layer_sources={"fuel": str(raster)},
            region_data_dir=tmp_path / "regions",
            cache_dir=tmp_path / "cache",
            auto_discover=False,
            landfire_only=True,
            max_expected_cells=1000,
        )
    assert "too large" in str(exc.value).lower()


def test_prepare_rerun_reuses_existing_region_manifest(tmp_path):
    region_dir = tmp_path / "regions" / "reuse_region"
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "fuel.tif").write_bytes(b"fuel")
    (region_dir / "canopy.tif").write_bytes(b"canopy")
    manifest = {
        "files": {
            "fuel": "fuel.tif",
            "canopy": "canopy.tif",
        }
    }
    (region_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    reusable, manifest_path = _can_reuse_existing_region(region_dir, ["fuel", "canopy"])
    assert reusable is True
    assert manifest_path and manifest_path.endswith("manifest.json")
