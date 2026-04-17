from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.data_prep.fetchers.common import ParcelFetchResult
from backend.data_prep.fetchers.regrid_fetcher import RegridParcelFetcher
from backend.data_prep.region_prep import fetch_parcels_for_region


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_regrid_fetcher_paginates_and_normalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page_payloads = [
        {
            "results": [
                {
                    "id": "p-1",
                    "properties": {
                        "parcel_id": "P-1",
                        "full_address": "101 Main St, Missoula, MT 59802",
                        "owner_name": "Alice Example",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-113.99, 46.87],
                            [-113.9898, 46.87],
                            [-113.9898, 46.8702],
                            [-113.99, 46.8702],
                            [-113.99, 46.87],
                        ]],
                    },
                }
            ],
            "next_cursor": "page-2",
        },
        {
            "results": [
                {
                    "id": "p-2",
                    "properties": {
                        "parcelid": "P-2",
                        "fulladdress": "103 Main St, Missoula, MT 59802",
                        "owner": "Bob Example",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-113.9897, 46.8701],
                            [-113.9895, 46.8701],
                            [-113.9895, 46.8703],
                            [-113.9897, 46.8703],
                            [-113.9897, 46.8701],
                        ]],
                    },
                }
            ],
        },
    ]
    call_index = {"idx": 0}

    def _fake_urlopen(_req, timeout=0):  # noqa: ARG001
        idx = call_index["idx"]
        call_index["idx"] += 1
        payload = page_payloads[idx] if idx < len(page_payloads) else {"results": []}
        return _FakeHTTPResponse(payload)

    monkeypatch.setenv("WF_REGRID_API_KEY", "test-key")
    monkeypatch.setenv("WF_REGRID_PARCELS_ENDPOINT", "https://example.com/regrid/parcels")
    monkeypatch.setattr("backend.data_prep.fetchers.regrid_fetcher.urllib.request.urlopen", _fake_urlopen)

    fetcher = RegridParcelFetcher()
    result = fetcher.fetch(
        bounds={"min_lon": -114.0, "min_lat": 46.86, "max_lon": -113.98, "max_lat": 46.88},
        region_dir=tmp_path / "missoula_pilot",
    )

    assert result.success is True
    assert result.record_count == 2
    assert call_index["idx"] == 2
    assert result.output_path is not None
    output = Path(str(result.output_path))
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    features = payload.get("features") or []
    assert len(features) == 2
    assert {f["properties"]["parcel_id"] for f in features} == {"P-1", "P-2"}
    assert all(f["properties"]["source"] == "regrid" for f in features)


def test_fetch_parcels_chain_falls_through_to_state_gis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = tmp_path / "state_gis_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "states": {
                    "MT": {
                        "parcel_endpoints": [
                            "https://example.com/mt/parcels/FeatureServer/0",
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail_regrid(self, *, bounds, region_dir, page_size=500, max_pages=200):  # noqa: ARG001
        return ParcelFetchResult(source_id="regrid", success=False, message="regrid unavailable")

    def _fail_overture(self, *, bounds, region_dir, state_code):  # noqa: ARG001
        return ParcelFetchResult(source_id="overture", success=False, message="overture unavailable")

    def _state_success(*, endpoint, bounds, region_dir, source_id, timeout_seconds, retries, backoff_seconds):  # noqa: ARG001
        region_dir = Path(region_dir)
        region_dir.mkdir(parents=True, exist_ok=True)
        output_path = region_dir / "parcel_polygons.geojson"
        output_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"parcel_id": "STATE-1", "source": source_id},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [-113.99, 46.87],
                                    [-113.9898, 46.87],
                                    [-113.9898, 46.8702],
                                    [-113.99, 46.8702],
                                    [-113.99, 46.87],
                                ]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return ParcelFetchResult(
            source_id=source_id,
            success=True,
            output_path=str(output_path),
            record_count=1,
        )

    monkeypatch.setattr("backend.data_prep.region_prep.RegridParcelFetcher.fetch", _fail_regrid)
    monkeypatch.setattr("backend.data_prep.fetchers.overture_fetcher.OvertureParcelFetcher.fetch", _fail_overture)
    monkeypatch.setattr("backend.data_prep.region_prep._fetch_arcgis_parcels", _state_success)

    result = fetch_parcels_for_region(
        region_id="missoula_pilot",
        bounds={"min_lon": -114.2, "min_lat": 46.80, "max_lon": -113.85, "max_lat": 46.95},
        region_dir=tmp_path / "regions" / "missoula_pilot",
        parcel_source="auto",
        state_code="MT",
        state_registry_path=registry_path,
    )

    assert result["success"] is True
    assert str(result["parcel_source_used"]).startswith("state_gis:")
    attempts = result["attempts"]
    assert [row["source"] for row in attempts][:3] == ["regrid", "overture", "state_gis:MT"]
    manifest = json.loads(Path(str(result["manifest_path"])).read_text(encoding="utf-8"))
    availability = manifest.get("parcel_polygons_availability") or {}
    assert availability.get("available") is True
    assert availability.get("source", "").startswith("state_gis:")
