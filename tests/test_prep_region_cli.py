from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.prep_region as prep_region_script


def test_prep_region_cli_fetch_parcels_auto_uses_manifest_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    regions_root = tmp_path / "regions"
    region_id = "missoula_pilot"
    region_dir = regions_root / region_id
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "manifest.json").write_text(
        json.dumps(
            {
                "region_id": region_id,
                "bounds": {
                    "min_lon": -114.2,
                    "min_lat": 46.8,
                    "max_lon": -113.85,
                    "max_lat": 46.95,
                },
            }
        ),
        encoding="utf-8",
    )

    captured_kwargs: dict[str, object] = {}

    def _fake_fetch_parcels_for_region(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "region_id": kwargs["region_id"],
            "success": True,
            "parcel_source_requested": kwargs["parcel_source"],
            "parcel_source_used": "regrid",
            "record_count": 1,
            "output_path": str(Path(kwargs["region_dir"]) / "parcel_polygons.geojson"),
            "manifest_path": str(Path(kwargs["region_dir"]) / "manifest.json"),
            "attempts": [{"source": "regrid", "success": True}],
        }

    monkeypatch.setattr(prep_region_script, "fetch_parcels_for_region", _fake_fetch_parcels_for_region)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prep_region.py",
            "--region",
            region_id,
            "--regions-root",
            str(regions_root),
            "--fetch-parcels",
            "--parcel-source",
            "auto",
        ],
    )
    exit_code = prep_region_script.main()
    assert exit_code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["success"] is True
    assert payload["parcel_source_used"] == "regrid"
    assert captured_kwargs["region_id"] == region_id
    assert captured_kwargs["parcel_source"] == "auto"
    assert captured_kwargs["bounds"] == {
        "min_lon": -114.2,
        "min_lat": 46.8,
        "max_lon": -113.85,
        "max_lat": 46.95,
    }
