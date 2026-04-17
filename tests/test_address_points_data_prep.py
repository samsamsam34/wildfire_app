from __future__ import annotations

import json
from pathlib import Path

from backend.data_prep.address_points import download_and_clip_missoula_address_points
from backend.data_prep.sources.acquisition import AcquisitionResult


def test_download_and_clip_missoula_address_points_writes_manifest_and_clipped_geojson(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_geojson = tmp_path / "source.geojson"
    source_geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"fulladdress": "1355 Pattee Canyon Rd, Missoula, MT 59803"},
                        "geometry": {"type": "Point", "coordinates": [-113.9778, 46.8281]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"fulladdress": "Outside BBox"},
                        "geometry": {"type": "Point", "coordinates": [-112.0, 47.5]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_fetch_bbox(self, **kwargs):  # noqa: ANN001
        return AcquisitionResult(
            layer_key="address_points",
            provider_type="arcgis_feature_service",
            acquisition_method="bbox_export_json_fallback",
            source_endpoint="https://example.test/FeatureServer/0",
            source_url="https://example.test/FeatureServer/0/query?f=json",
            local_path=str(source_geojson),
            bbox_used="-114.20,46.80,-113.85,46.95",
            output_resolution=None,
            cache_hit=False,
            warnings=[],
            bytes_downloaded=source_geojson.stat().st_size,
        )

    monkeypatch.setattr(
        "backend.data_prep.address_points.ArcGISFeatureServiceProvider.fetch_bbox",
        _fake_fetch_bbox,
    )

    result = download_and_clip_missoula_address_points(
        region_id="missoula_pilot",
        regions_root=tmp_path / "regions",
        endpoint="https://example.test/FeatureServer/0",
    )
    assert result["record_count"] == 1
    out_path = Path(result["output_path"])
    assert out_path.exists()
    out_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(out_payload.get("features") or []) == 1

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest.get("files") or {}).get("address_points") == "address_points.geojson"
    availability = manifest.get("address_points_availability") or {}
    assert availability.get("available") is True
    assert availability.get("layer_key") == "address_points"
