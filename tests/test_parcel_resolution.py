"""Tests for backend/parcel_resolution.py (ParcelResolutionClient + Regrid fallback + STRtree)."""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.parcel_resolution import ParcelResolutionClient, ParcelResolutionResult

try:
    from shapely.geometry import Point, box, mapping
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GEO_AVAILABLE, reason="shapely not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regrid_result(parcel_id: str = "UT-99999") -> Any:
    """Build a minimal mock RegridParcelResult."""
    result = MagicMock()
    result.parcel_id = parcel_id
    result.parcel_address = "3184 FOOTHILL DR"
    result.owner_name = "SMITH JOHN"
    result.land_use_desc = "RESIDENTIAL"
    result.area_m2 = 1700.0
    result.state = "UT"
    result.county = "Utah County"
    result.geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [-111.656, 40.250],
                [-111.654, 40.250],
                [-111.654, 40.252],
                [-111.656, 40.252],
                [-111.656, 40.250],
            ]
        ],
    }
    result.source = "regrid_api"
    result.cached = False
    result.fetched_at = "2026-01-01T00:00:00+00:00"
    return result


def _make_mock_regrid_client(return_value=None) -> MagicMock:
    """Build a mock RegridParcelClient."""
    client = MagicMock()
    client.enabled = True
    client.fetch_parcel.return_value = return_value
    return client


def _write_parcel_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def _make_polygon_feature(
    min_x: float, min_y: float, max_x: float, max_y: float, parcel_id: str = "test"
) -> dict[str, Any]:
    """GeoJSON Feature for a rectangular parcel polygon."""
    return {
        "type": "Feature",
        "geometry": mapping(box(min_x, min_y, max_x, max_y)),
        "properties": {"parcel_id": parcel_id},
    }


# ---------------------------------------------------------------------------
# Test 9 — Regrid fallback used when no local parcel match
# ---------------------------------------------------------------------------

def test_regrid_fallback_used_when_no_local_match(tmp_path):
    """When point is outside all local parcels, Regrid result is returned with confidence 72."""
    # Write a parcel file containing a single polygon far from the anchor point.
    parcel_file = tmp_path / "parcels.geojson"
    _write_parcel_geojson(
        parcel_file,
        [_make_polygon_feature(-120.0, 35.0, -119.9, 35.1, parcel_id="FARAWAY")],
    )

    regrid_result = _make_regrid_result("UT-99999")
    mock_regrid = _make_mock_regrid_client(return_value=regrid_result)

    client = ParcelResolutionClient(
        parcel_paths=[str(parcel_file)],
        max_lookup_distance_m=30.0,
        regrid_client=mock_regrid,
    )
    anchor = Point(-111.655, 40.251)  # ~3,000 km from the "faraway" parcel

    result = client.resolve_for_point(anchor_point=anchor)

    assert result.status == "matched"
    assert result.confidence == pytest.approx(72.0)
    assert result.parcel_lookup_method == "api_lookup"
    assert result.source == "regrid_api"
    assert result.parcel_id == "UT-99999"
    mock_regrid.fetch_parcel.assert_called_once()


# ---------------------------------------------------------------------------
# Test 10 — Regrid None falls through to bounding-box
# ---------------------------------------------------------------------------

def test_regrid_none_falls_through_to_bounding_box(tmp_path):
    """When Regrid returns None and no local match, bounding-box fallback is used."""
    parcel_file = tmp_path / "parcels.geojson"
    _write_parcel_geojson(
        parcel_file,
        [_make_polygon_feature(-120.0, 35.0, -119.9, 35.1, parcel_id="FARAWAY")],
    )

    mock_regrid = _make_mock_regrid_client(return_value=None)

    client = ParcelResolutionClient(
        parcel_paths=[str(parcel_file)],
        max_lookup_distance_m=30.0,
        regrid_client=mock_regrid,
    )
    anchor = Point(-111.655, 40.251)

    result = client.resolve_for_point(anchor_point=anchor)

    assert result.status == "not_found"
    assert result.confidence < 20.0  # bounding-box path; exact value depends on STRtree pre-filter
    assert result.geometry_used == "bounding_approximation"
    mock_regrid.fetch_parcel.assert_called_once()


# ---------------------------------------------------------------------------
# Test 11 — STRtree correctness and performance
# ---------------------------------------------------------------------------

def test_strtree_correctness_and_performance(tmp_path):
    """STRtree returns the correct parcel for a point and completes in < 100 ms for 1,000 polygons."""
    # Build 1,000 non-overlapping parcel polygons arranged in a 40×25 grid,
    # each 0.001° × 0.001° (~111m × 111m).
    cols, rows = 40, 25
    origin_lon, origin_lat = -111.70, 40.22
    step = 0.001

    # Anchor sits in the centre of cell (col=20, row=12).
    # min_x for col 20 = -111.70 + 20*0.001 = -111.680
    # min_y for row 12 = 40.22  + 12*0.001 = 40.232
    # max_x/max_y shrunk by (1 - 0.9) = 0.0001 gap
    anchor_lon = -111.680 + step * 0.45   # centred horizontally
    anchor_lat = 40.232  + step * 0.45   # centred vertically
    target_col, target_row = 20, 12
    target_parcel_id = f"PARCEL-{target_row * cols + target_col:04d}"

    features = []
    for row in range(rows):
        for col in range(cols):
            min_x = origin_lon + col * step
            min_y = origin_lat + row * step
            max_x = min_x + step * 0.9  # slight gap so polygons don't touch
            max_y = min_y + step * 0.9
            pid = f"PARCEL-{row * cols + col:04d}"
            features.append(_make_polygon_feature(min_x, min_y, max_x, max_y, parcel_id=pid))

    parcel_file = tmp_path / "large_parcels.geojson"
    _write_parcel_geojson(parcel_file, features)

    client = ParcelResolutionClient(
        parcel_paths=[str(parcel_file)],
        max_lookup_distance_m=30.0,
        regrid_client=None,
    )
    anchor = Point(anchor_lon, anchor_lat)

    # Warm the lru_cache so the timed run only measures query + CRS transforms,
    # not file I/O or STRtree construction.
    client.resolve_for_point(anchor_point=anchor)

    t0 = time.monotonic()
    result = client.resolve_for_point(anchor_point=anchor)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert result.status == "matched", f"Expected matched, got {result.status}"
    assert result.parcel_id == target_parcel_id, (
        f"Expected {target_parcel_id!r}, got {result.parcel_id!r}"
    )
    assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f} ms — expected < 100 ms"
