from __future__ import annotations

import json
from pathlib import Path

import pytest
np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from backend.data_prep.prepare_region import parse_bbox, prepare_region_layers


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
    assert (region_dir / "dem.tif").exists()
    assert (region_dir / "slope.tif").exists()
    assert manifest["layers"]["slope"]["source_type"] == "derived_from_dem"
    assert manifest["layers"]["dem"]["clipped_to_bbox"] is True
    assert manifest["layers"]["fire_perimeters"]["validation_status"] == "ok"


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


def test_prepare_region_validation_failure_on_missing_required_sources(tmp_path):
    src = _sources(tmp_path, include_slope=False)
    src.pop("fuel")
    with pytest.raises(ValueError) as exc:
        prepare_region_layers(
            region_id="invalid_region",
            display_name="Invalid Region",
            bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
            layer_sources=src,
            region_data_dir=tmp_path / "regions",
        )
    assert "fuel" in str(exc.value).lower()


def test_backward_compatible_local_source_mode_with_explicit_slope(tmp_path):
    manifest = prepare_region_layers(
        region_id="legacy_mode_region",
        display_name="Legacy Mode Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources=_sources(tmp_path, include_slope=True),
        region_data_dir=tmp_path / "regions",
    )
    assert manifest["preparation_status"] == "prepared"
    assert manifest["files"]["slope"] == "slope.tif"
    assert manifest["layers"]["slope"]["source_type"] in {"local_file", "downloaded_url"}


def test_partial_mode_warns_when_sources_missing(tmp_path):
    partial = prepare_region_layers(
        region_id="partial_region",
        display_name="Partial Region",
        bounds={"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        layer_sources={"dem": _sources(tmp_path, include_slope=False)["dem"]},
        region_data_dir=tmp_path / "regions",
        skip_download=True,
        allow_partial=True,
    )
    assert partial["preparation_status"] == "partial"
    assert partial["warnings"]
    assert any("no source provided" in w.lower() for w in partial["warnings"])
    assert partial["errors"]
