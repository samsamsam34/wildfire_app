"""
Integration tests for LANDFIRE WCS COG fallback wiring in wildfire_data.py.

These tests verify that WildfireDataClient.collect_context() correctly:
  1. Uses the COG client when local fuel raster is absent (test 1)
  2. Does NOT call the COG client for fuel when local raster succeeds (test 2)
  3. Degrades gracefully when _landfire_cog_client is None (test 3)

No real LANDFIRE requests are made — rasterio and COG calls are mocked.
The PropertyAnchorResolver and BuildingFootprintClient are mocked at the
class level to keep the tests focused on the COG fallback path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from backend.property_anchor import PropertyAnchorResolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_anchor(lat: float = 46.87, lon: float = -113.99) -> PropertyAnchorResolution:
    return PropertyAnchorResolution(
        anchor_latitude=lat,
        anchor_longitude=lon,
        anchor_source="geocoded_address_point",
        anchor_precision="interpolated",
        geocoded_latitude=lat,
        geocoded_longitude=lon,
        anchor_quality="low",
        anchor_quality_score=0.3,
    )


def _null_footprint_result() -> MagicMock:
    r = MagicMock()
    r.found = False
    r.match_status = "provider_unavailable"
    r.source = "mock"
    r.confidence = 0.0
    r.match_method = None
    r.matched_structure_id = None
    r.match_distance_m = None
    r.candidate_count = 0
    r.candidate_summaries = []
    r.assumptions = []
    return r


def _run_collect_context_with_mocks(
    tmp_path: Path,
    *,
    cog_client: MagicMock | None,
    fuel_samples: list[float],
    canopy_samples: list[float],
    slope_val: float | None = None,
    file_exists_for: frozenset[str] = frozenset(),
):
    """
    Run collect_context with all heavy subsystems mocked.

    cog_client: mock injected as _landfire_cog_client (or None to disable)
    fuel_samples: list returned by _sample_circle for the fuel layer
    canopy_samples: list returned by _sample_circle for the canopy layer
    slope_val: value returned by _sample_layer_value_detailed for slope
    """
    from backend.wildfire_data import WildfireDataClient

    anchor = _minimal_anchor()
    footprint_result = _null_footprint_result()

    # Mock entire PropertyAnchorResolver class so we skip parcel / regrid logic.
    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = anchor

    # Mock BuildingFootprintClient
    mock_footprints = MagicMock()
    mock_footprints.get_building_footprint.return_value = footprint_result
    mock_footprints.compute_structure_rings.return_value = (None, [], [])
    mock_footprints.get_neighbor_structure_metrics.return_value = None

    def _fake_sample_detail(path, lat, lon):
        """Return (None, "not_configured", "No path") for all layers except slope."""
        if slope_val is not None and "slope" in str(path):
            return (slope_val, "ok", None)
        return (None, "not_configured", "No path")

    def _fake_sample_circle(path, lat, lon, radius_m, step_m=30.0):
        if "fuel" in str(path):
            return fuel_samples
        if "canopy" in str(path):
            return canopy_samples
        return []

    with (
        patch("backend.wildfire_data.PropertyAnchorResolver", return_value=mock_resolver_instance),
        patch("backend.wildfire_data.BuildingFootprintClient", return_value=mock_footprints),
    ):
        client = WildfireDataClient()
        # Override the COG client after construction
        client._landfire_cog_client = cog_client

        with (
            patch.object(client, "_sample_layer_value_detailed", side_effect=_fake_sample_detail),
            patch.object(client, "_sample_circle", side_effect=_fake_sample_circle),
            patch.object(client, "_sample_layer_value", return_value=(None, "not_configured")),
            patch.object(
                client,
                "_file_exists",
                side_effect=lambda path: any(k in str(path) for k in file_exists_for),
            ),
            patch.object(client, "_historical_fire_metrics", return_value=(None, None, "missing")),
            patch.object(client, "_wildland_distance_metrics", return_value=(None, None)),
            # Bypass feature-bundle cache so mocks are not skipped on cache hits
            patch.object(client.feature_bundle_cache, "load", return_value=None),
        ):
            return client.collect_context(46.87, -113.99)


# ---------------------------------------------------------------------------
# Test 1: No local fuel raster → COG fallback is called; fuel_index populated
# ---------------------------------------------------------------------------

def test_cog_fallback_used_when_local_fuel_missing(tmp_path: Path) -> None:
    """When local fuel raster yields no samples, COG client is called for 'fuel'."""
    cog_client = MagicMock()
    # Simulate COG returning a fuel model code of 102 (GR2, medium grass)
    cog_client.sample_point.side_effect = lambda lat, lon, layer_ids, **kw: {
        lid: (102.0 if lid == "fuel" else None) for lid in layer_ids
    }

    ctx = _run_collect_context_with_mocks(
        tmp_path, cog_client=cog_client, fuel_samples=[], canopy_samples=[]
    )

    # COG client must have been called for "fuel"
    called_layer_sets = [
        set(c[0][2]) for c in cog_client.sample_point.call_args_list if len(c[0]) >= 3
    ]
    assert any("fuel" in ls for ls in called_layer_sets), (
        "COG client must be called for 'fuel' when local raster yields no samples"
    )

    # fuel_index should be populated from the COG value
    assert ctx.fuel_index is not None, "fuel_index should be set from COG fallback"

    # Sources should mention COG
    assert any(
        "COG" in s or "LANDFIRE WCS" in s or "landfire" in s.lower()
        for s in ctx.data_sources
    ), "data_sources should reference LANDFIRE WCS COG fallback"

    # The source-tagged assumption should be present
    assert any(
        "COG fallback" in a or "LANDFIRE WCS" in a or "national" in a.lower()
        for a in ctx.assumptions
    ), "assumptions should note the COG fallback"


# ---------------------------------------------------------------------------
# Test 2: Local fuel raster succeeds → COG client NOT called for fuel
# ---------------------------------------------------------------------------

def test_cog_client_not_called_when_local_fuel_succeeds(tmp_path: Path) -> None:
    """When local fuel raster yields samples, COG client is not invoked for 'fuel'."""
    cog_client = MagicMock()
    cog_client.sample_point.return_value = {}

    ctx = _run_collect_context_with_mocks(
        tmp_path,
        cog_client=cog_client,
        fuel_samples=[102.0, 101.0, 98.0],  # local fuel succeeds
        canopy_samples=[],
        file_exists_for=frozenset({"fuel"}),  # _file_exists returns True for fuel paths
    )

    # fuel_index should be populated from local samples (no COG needed)
    assert ctx.fuel_index is not None, "fuel_index should be set from local raster samples"

    # COG client should NOT have been called for fuel
    for c in cog_client.sample_point.call_args_list:
        layer_ids = c[0][2] if len(c[0]) >= 3 else []
        assert "fuel" not in layer_ids, (
            "COG client must not be called for 'fuel' when local raster succeeded"
        )


# ---------------------------------------------------------------------------
# Test 3: COG client is None → assessment completes, fuel/canopy are None
# ---------------------------------------------------------------------------

def test_disabled_cog_client_degrades_gracefully(tmp_path: Path) -> None:
    """When _landfire_cog_client is None, collect_context succeeds with None fuel/canopy."""
    ctx = _run_collect_context_with_mocks(
        tmp_path, cog_client=None, fuel_samples=[], canopy_samples=[]
    )

    assert ctx is not None, "collect_context must return a WildfireContext, not raise"
    assert ctx.fuel_index is None, "fuel_index should be None when local file and COG both absent"
    assert ctx.canopy_index is None, "canopy_index should be None when local file and COG both absent"
