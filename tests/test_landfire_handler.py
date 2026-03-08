from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest

import backend.data_prep.prepare_region as prep_region
from backend.data_prep.sources.landfire import resolve_landfire_raster, subset_cache_path


def _write_landfire_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("landfire/fuel_model.tif", b"fuel-bytes")
        zf.writestr("landfire/canopy_cover.tif", b"canopy-bytes")
        zf.writestr("README.txt", b"metadata")


def test_landfire_archive_index_and_selective_extraction(tmp_path):
    archive = tmp_path / "landfire.zip"
    _write_landfire_archive(archive)
    cache_dir = tmp_path / "cache"
    bounds = {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8}
    progress: list[str] = []

    first = resolve_landfire_raster(
        layer_key="fuel",
        source_path=archive,
        cache_dir=cache_dir,
        bounds=bounds,
        progress_log=progress,
        warnings=[],
    )
    assert first.is_landfire_archive is True
    assert Path(first.index_path or "").exists()
    assert Path(first.extracted_raster_path or "").exists()
    assert first.extraction_performed is True

    second = resolve_landfire_raster(
        layer_key="fuel",
        source_path=archive,
        cache_dir=cache_dir,
        bounds=bounds,
        progress_log=progress,
        warnings=[],
    )
    assert second.extraction_performed is False
    assert Path(second.extracted_raster_path or "").exists()
    assert any("cached archive index" in msg.lower() for msg in progress)


def test_landfire_ambiguous_candidate_fails(tmp_path):
    archive = tmp_path / "ambiguous_landfire.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a_fuel.tif", b"1")
        zf.writestr("b_fuel.tif", b"2")
    with pytest.raises(ValueError) as exc:
        resolve_landfire_raster(
            layer_key="fuel",
            source_path=archive,
            cache_dir=tmp_path / "cache",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            progress_log=[],
            warnings=[],
        )
    assert "ambiguous" in str(exc.value).lower()


def test_landfire_subset_cache_reuse_in_prepare_pipeline(monkeypatch, tmp_path):
    archive = tmp_path / "landfire.zip"
    _write_landfire_archive(archive)
    calls = {"clip": 0}

    def fake_clip(src: Path, dest: Path, bounds: dict[str, float], **kwargs):
        calls["clip"] += 1
        shutil.copy2(src, dest)
        return {"crs": "EPSG:4326", "resolution": [30.0, 30.0], "bounds": [0.0, 0.0, 1.0, 1.0]}

    monkeypatch.setattr(prep_region, "_clip_raster_to_bbox", fake_clip)
    monkeypatch.setattr(prep_region, "_validate_prepared_layer", lambda *args, **kwargs: None)
    monkeypatch.setattr(prep_region, "_raster_file_metadata", lambda _p: {"crs": "EPSG:4326", "resolution": [30.0, 30.0], "bounds": [0.0, 0.0, 1.0, 1.0]})

    kwargs = {
        "display_name": "Landfire Only",
        "bounds": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "layer_sources": {"fuel": str(archive)},
        "region_data_dir": tmp_path / "regions",
        "cache_dir": tmp_path / "cache",
        "auto_discover": False,
        "landfire_only": True,
    }

    first = prep_region.prepare_region_layers(region_id="lf_a", **kwargs)
    second = prep_region.prepare_region_layers(region_id="lf_b", **kwargs)

    assert first["final_status"] == "success"
    assert second["final_status"] == "success"
    assert calls["clip"] == 1
    assert first["layers"]["fuel"]["subset_reused"] is False
    assert second["layers"]["fuel"]["subset_reused"] is True
    assert any("reusing cached clipped subset" in msg.lower() for msg in second.get("progress_log", []))


def test_landfire_manifest_metadata_fields_present(monkeypatch, tmp_path):
    archive = tmp_path / "landfire.zip"
    _write_landfire_archive(archive)
    monkeypatch.setattr(
        prep_region,
        "_clip_raster_to_bbox",
        lambda src, dest, bounds, **kwargs: (
            shutil.copy2(src, dest),
            {"crs": "EPSG:4326", "resolution": [30.0, 30.0], "bounds": [0.0, 0.0, 1.0, 1.0]},
        )[1],
    )
    monkeypatch.setattr(prep_region, "_validate_prepared_layer", lambda *args, **kwargs: None)
    monkeypatch.setattr(prep_region, "_raster_file_metadata", lambda _p: {"crs": "EPSG:4326", "resolution": [30.0, 30.0], "bounds": [0.0, 0.0, 1.0, 1.0]})

    result = prep_region.prepare_region_layers(
        region_id="lf_meta",
        display_name="Landfire Meta",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources={"fuel": str(archive)},
        region_data_dir=tmp_path / "regions",
        cache_dir=tmp_path / "cache",
        auto_discover=False,
        landfire_only=True,
    )
    fuel = result["layers"]["fuel"]
    assert fuel["landfire_handler_version"]
    assert fuel["archive_index_path"]
    assert fuel["extracted_raster_path"]
    assert fuel["subset_cache_path"]
    assert fuel["landfire_layer_type"] == "fuel"
    assert fuel["clipping_bbox"]["max_lat"] == 1.0


def test_landfire_subset_cache_path_is_deterministic(tmp_path):
    bounds = {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8}
    p1 = subset_cache_path(
        cache_dir=tmp_path / "cache",
        source_fingerprint="abc",
        layer_key="fuel",
        bounds=bounds,
    )
    p2 = subset_cache_path(
        cache_dir=tmp_path / "cache",
        source_fingerprint="abc",
        layer_key="fuel",
        bounds=bounds,
    )
    assert p1 == p2
