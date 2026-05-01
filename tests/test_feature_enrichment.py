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


# ---------------------------------------------------------------------------
# Helpers for national fire history + NLCD integration tests
# ---------------------------------------------------------------------------

def _run_collect_context_national(
    tmp_path: Path,
    *,
    fire_history_client,
    nlcd_client,
):
    """Run collect_context with national clients injected and all local paths empty."""
    from backend.wildfire_data import WildfireDataClient
    from unittest.mock import MagicMock, patch

    anchor = _minimal_anchor()
    footprint_result = _null_footprint_result()

    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = anchor
    mock_footprints = MagicMock()
    mock_footprints.get_building_footprint.return_value = footprint_result
    mock_footprints.compute_structure_rings.return_value = (None, [], [])
    mock_footprints.get_neighbor_structure_metrics.return_value = None

    with (
        patch("backend.wildfire_data.PropertyAnchorResolver", return_value=mock_resolver_instance),
        patch("backend.wildfire_data.BuildingFootprintClient", return_value=mock_footprints),
    ):
        client = WildfireDataClient()
        client._landfire_cog_client = None
        client._fire_history_client = fire_history_client
        client._nlcd_client = nlcd_client

        with (
            patch.object(client, "_sample_layer_value_detailed",
                         return_value=(None, "not_configured", "No path")),
            patch.object(client, "_sample_circle", return_value=[]),
            patch.object(client, "_sample_layer_value", return_value=(None, "not_configured")),
            patch.object(client, "_file_exists", return_value=False),
            patch.object(client, "_historical_fire_metrics", return_value=(None, None, "missing")),
            patch.object(client, "_wildland_distance_metrics", return_value=(None, None)),
            patch.object(client.feature_bundle_cache, "load", return_value=None),
            # Disable real MTBSAdapter so it doesn't mask the national client
            patch.object(client.mtbs_adapter, "summarize",
                         return_value=MagicMock(status="missing", notes=[], source_dataset=None,
                                                nearest_perimeter_km=None, intersects_prior_burn=False,
                                                nearby_high_severity=False, fire_history_index=None)),
        ):
            return client.collect_context(46.87, -113.99)


# ---------------------------------------------------------------------------
# Test 4: Fire history national fallback populates historic_fire_index
# ---------------------------------------------------------------------------

def test_fire_history_national_fallback_populates_index(tmp_path: Path) -> None:
    """When local MTBS data is absent and fire_history_client returns data,
    historic_fire_index is populated and source is tagged 'national_mtbs'."""
    from unittest.mock import MagicMock
    from backend.national_fire_history_client import FireHistoryResult

    mock_fh = MagicMock()
    mock_fh.query_fire_history.return_value = FireHistoryResult(
        burned_within_radius=True,
        most_recent_fire_year=2018,
        most_recent_fire_severity="moderate",
        fire_count_30yr=2,
        fire_count_all=3,
        nearest_fire_distance_m=0.0,
        fires_within_radius=[{"year": 2018, "name": "TEST", "severity": "moderate",
                               "distance_m": 0.0, "area_acres": 5000.0}],
        data_available=True,
        radius_m=5000.0,
    )

    ctx = _run_collect_context_national(tmp_path, fire_history_client=mock_fh, nlcd_client=None)

    assert ctx is not None
    assert ctx.historic_fire_index is not None, "historic_fire_index should be populated from national MTBS"
    assert any("national" in s.lower() or "MTBS" in s for s in ctx.data_sources), (
        "data_sources should reference national MTBS fallback"
    )


# ---------------------------------------------------------------------------
# Test 5: Wildland distance national fallback populates wildland_distance_index
# ---------------------------------------------------------------------------

def test_wildland_distance_national_fallback_populates_index(tmp_path: Path) -> None:
    """When local rasters are absent and nlcd_client returns a distance,
    wildland_distance_index is populated and tagged with national NLCD source."""
    from unittest.mock import MagicMock

    mock_nlcd = MagicMock()
    mock_nlcd.get_wildland_distance_m.return_value = 350.0  # 350 m to nearest wildland

    ctx = _run_collect_context_national(tmp_path, fire_history_client=None, nlcd_client=mock_nlcd)

    assert ctx is not None
    assert ctx.wildland_distance is not None, "wildland_distance should be populated from NLCD"
    assert ctx.wildland_distance_index is not None
    # wildland_distance_index = 100 - (350/2000)*100 = 82.5
    assert abs(ctx.wildland_distance_index - 82.5) < 2.0, (
        f"Expected ~82.5, got {ctx.wildland_distance_index}"
    )
    assert any("NLCD" in s or "nlcd" in s.lower() or "national" in s.lower()
               for s in ctx.data_sources), (
        "data_sources should reference national NLCD fallback"
    )


# ---------------------------------------------------------------------------
# Test 6: Both national clients disabled → graceful degradation
# ---------------------------------------------------------------------------

def test_both_national_clients_disabled_degrades_gracefully(tmp_path: Path) -> None:
    """When fire_history_client and nlcd_client are both None, assessment completes."""
    ctx = _run_collect_context_national(tmp_path, fire_history_client=None, nlcd_client=None)

    assert ctx is not None, "collect_context must not raise when national clients are absent"
    # Without local rasters or national clients, both should be None
    assert ctx.historic_fire_index is None
    assert ctx.wildland_distance_index is None


# ---------------------------------------------------------------------------
# Helpers for elevation national integration tests
# ---------------------------------------------------------------------------

def _run_collect_context_with_elevation(
    tmp_path: Path,
    *,
    elevation_client,
):
    """Run collect_context with elevation client injected and all local paths empty."""
    from backend.wildfire_data import WildfireDataClient
    from unittest.mock import MagicMock, patch

    anchor = _minimal_anchor()
    footprint_result = _null_footprint_result()

    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = anchor
    mock_footprints = MagicMock()
    mock_footprints.get_building_footprint.return_value = footprint_result
    mock_footprints.compute_structure_rings.return_value = (None, [], [])
    mock_footprints.get_neighbor_structure_metrics.return_value = None

    with (
        patch("backend.wildfire_data.PropertyAnchorResolver", return_value=mock_resolver_instance),
        patch("backend.wildfire_data.BuildingFootprintClient", return_value=mock_footprints),
    ):
        client = WildfireDataClient()
        client._landfire_cog_client = None
        client._fire_history_client = None
        client._nlcd_client = None
        client._elevation_client = elevation_client

        with (
            patch.object(client, "_sample_layer_value_detailed",
                         return_value=(None, "not_configured", "No path")),
            patch.object(client, "_sample_circle", return_value=[]),
            patch.object(client, "_sample_layer_value", return_value=(None, "not_configured")),
            patch.object(client, "_file_exists", return_value=False),
            patch.object(client, "_historical_fire_metrics", return_value=(None, None, "missing")),
            patch.object(client, "_wildland_distance_metrics", return_value=(None, None)),
            patch.object(client.feature_bundle_cache, "load", return_value=None),
            patch.object(client.mtbs_adapter, "summarize",
                         return_value=MagicMock(status="missing", notes=[], source_dataset=None,
                                                nearest_perimeter_km=None, intersects_prior_burn=False,
                                                nearby_high_severity=False, fire_history_index=None)),
        ):
            return client.collect_context(46.87, -113.99)


def _run_collect_context_with_local_slope(
    tmp_path: Path,
    *,
    elevation_client,
    slope_val: float,
):
    """Run collect_context with a local slope value available and elevation client injected."""
    from backend.wildfire_data import WildfireDataClient
    from unittest.mock import MagicMock, patch

    anchor = _minimal_anchor()
    footprint_result = _null_footprint_result()

    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = anchor
    mock_footprints = MagicMock()
    mock_footprints.get_building_footprint.return_value = footprint_result
    mock_footprints.compute_structure_rings.return_value = (None, [], [])
    mock_footprints.get_neighbor_structure_metrics.return_value = None

    def _fake_sample_detail(path, lat, lon):
        if "slope" in str(path):
            return (slope_val, "ok", None)
        return (None, "not_configured", "No path")

    with (
        patch("backend.wildfire_data.PropertyAnchorResolver", return_value=mock_resolver_instance),
        patch("backend.wildfire_data.BuildingFootprintClient", return_value=mock_footprints),
    ):
        client = WildfireDataClient()
        client._landfire_cog_client = None
        client._fire_history_client = None
        client._nlcd_client = None
        client._elevation_client = elevation_client

        with (
            patch.object(client, "_sample_layer_value_detailed", side_effect=_fake_sample_detail),
            patch.object(client, "_sample_circle", return_value=[]),
            patch.object(client, "_sample_layer_value", return_value=(None, "not_configured")),
            # _file_exists returns True for slope path so the local raster path is exercised
            patch.object(client, "_file_exists", side_effect=lambda p: "slope" in str(p)),
            patch.object(client, "_historical_fire_metrics", return_value=(None, None, "missing")),
            patch.object(client, "_wildland_distance_metrics", return_value=(None, None)),
            patch.object(client.feature_bundle_cache, "load", return_value=None),
            patch.object(client.mtbs_adapter, "summarize",
                         return_value=MagicMock(status="missing", notes=[], source_dataset=None,
                                                nearest_perimeter_km=None, intersects_prior_burn=False,
                                                nearby_high_severity=False, fire_history_index=None)),
        ):
            return client.collect_context(46.87, -113.99)


# ---------------------------------------------------------------------------
# Test 7: No local slope, elevation_client returns (15.5, 180.0) → slope_index set
# ---------------------------------------------------------------------------

def test_elevation_fallback_populates_slope_index(tmp_path: Path) -> None:
    """When local slope is absent, 3DEP elevation_client populates slope_index."""
    from unittest.mock import MagicMock

    mock_elev = MagicMock()
    mock_elev.get_slope_and_aspect.return_value = (15.5, 180.0)

    ctx = _run_collect_context_with_elevation(tmp_path, elevation_client=mock_elev)

    assert ctx is not None
    assert ctx.slope_index is not None, "slope_index should be populated from 3DEP fallback"
    # slope_index = _to_index(15.5, 0.0, 45.0) = (15.5/45.0)*100 = 34.4
    expected = round((15.5 / 45.0) * 100.0, 1)
    assert abs(ctx.slope_index - expected) < 1.0, (
        f"Expected slope_index ≈ {expected}, got {ctx.slope_index}"
    )
    assert any(
        "3dep" in s.lower() or "3DEP" in s or "elevation" in s.lower() or "national" in s.lower()
        for s in ctx.data_sources
    ), "data_sources should reference 3DEP national elevation fallback"


# ---------------------------------------------------------------------------
# Test 8: Local slope present → elevation_client NOT called
# ---------------------------------------------------------------------------

def test_elevation_client_not_called_when_local_slope_present(tmp_path: Path) -> None:
    """When local slope raster succeeds, elevation_client.get_slope_and_aspect is not called."""
    from unittest.mock import MagicMock

    mock_elev = MagicMock()
    mock_elev.get_slope_and_aspect.return_value = (99.0, 0.0)  # should never be used

    ctx = _run_collect_context_with_local_slope(
        tmp_path, elevation_client=mock_elev, slope_val=12.0
    )

    assert ctx is not None
    assert ctx.slope_index is not None, "slope_index should be set from local raster"
    assert mock_elev.get_slope_and_aspect.call_count == 0, (
        "elevation_client must not be called when local slope raster succeeds"
    )


# ---------------------------------------------------------------------------
# WHP proxy integration tests (Tests 10–12)
# ---------------------------------------------------------------------------

def _run_collect_context_with_whp_client(
    tmp_path,
    *,
    whp_client,
    fuel_samples: list,
    canopy_samples: list,
    slope_val=None,
    burn_prob_val=None,
):
    """Run collect_context with a mock WHPClient injected and local rasters absent."""
    from backend.wildfire_data import WildfireDataClient
    from unittest.mock import MagicMock, patch

    anchor = _minimal_anchor()
    footprint_result = _null_footprint_result()
    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = anchor
    mock_footprints = MagicMock()
    mock_footprints.get_building_footprint.return_value = footprint_result
    mock_footprints.compute_structure_rings.return_value = (None, [], [])
    mock_footprints.get_neighbor_structure_metrics.return_value = None

    def _fake_sample_detail(path, lat, lon):
        if slope_val is not None and "slope" in str(path):
            return (slope_val, "ok", None)
        if burn_prob_val is not None and "burn" in str(path):
            return (burn_prob_val, "ok", None)
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
        client._landfire_cog_client = None
        client._fire_history_client = None
        client._nlcd_client = None
        client._whp_client = whp_client

        whp_missing = MagicMock(
            status="missing", notes=["No WHP path"], source_dataset=None,
            raw_value=None, burn_probability_index=None, hazard_severity_index=None,
            hazard_class=None,
        )
        with (
            patch.object(client, "_sample_layer_value_detailed", side_effect=_fake_sample_detail),
            patch.object(client, "_sample_circle", side_effect=_fake_sample_circle),
            patch.object(client, "_sample_layer_value", return_value=(None, "not_configured")),
            patch.object(client, "_file_exists", return_value=False),
            patch.object(client, "_historical_fire_metrics", return_value=(None, None, "missing")),
            patch.object(client, "_wildland_distance_metrics", return_value=(None, None)),
            patch.object(client.feature_bundle_cache, "load", return_value=None),
            patch.object(client.whp_adapter, "sample", return_value=whp_missing),
            patch.object(
                client.mtbs_adapter, "summarize",
                return_value=MagicMock(
                    status="missing", notes=[], source_dataset=None,
                    nearest_perimeter_km=None, intersects_prior_burn=False,
                    nearby_high_severity=False, fire_history_index=None,
                ),
            ),
        ):
            return client.collect_context(46.87, -113.99)


def test_whp_proxy_fills_burn_probability_index_when_absent(tmp_path) -> None:
    """
    Test 10: When burn_probability_index is None after all raster sampling,
    and WHPClient returns a value, burn_probability_index is populated.
    """
    from unittest.mock import MagicMock

    mock_whp = MagicMock()

    def _get_whp(lat, lon, features):
        features["whp_index_source"] = "whp_proxy"
        return 65.0

    mock_whp.get_whp_index.side_effect = _get_whp
    mock_whp.enabled = True

    ctx = _run_collect_context_with_whp_client(
        tmp_path,
        whp_client=mock_whp,
        fuel_samples=[102.0, 101.0],
        canopy_samples=[50.0, 55.0],
    )

    assert ctx is not None
    assert ctx.burn_probability_index is not None, (
        "burn_probability_index must be populated from WHP proxy when rasters absent"
    )
    assert abs(ctx.burn_probability_index - 65.0) < 1.0, (
        f"Expected ~65.0, got {ctx.burn_probability_index}"
    )
    assert mock_whp.get_whp_index.call_count == 1, (
        "WHPClient.get_whp_index must be called exactly once"
    )


def test_whp_proxy_not_called_when_burn_probability_already_set(tmp_path) -> None:
    """
    Test 11: When burn_probability_index is populated from the WHPAdapter (local WHP raster),
    WHPClient.get_whp_index must NOT be called.

    The WHPAdapter returns status="ok" with a non-None value, which populates
    burn_probability_index before the proxy check. The proxy guard
    `if burn_probability_index is None` then skips the proxy.
    """
    from backend.wildfire_data import WildfireDataClient
    from unittest.mock import MagicMock, patch

    mock_whp = MagicMock()
    mock_whp.enabled = True

    anchor = _minimal_anchor()
    footprint_result = _null_footprint_result()
    mock_resolver_instance = MagicMock()
    mock_resolver_instance.resolve.return_value = anchor
    mock_footprints = MagicMock()
    mock_footprints.get_building_footprint.return_value = footprint_result
    mock_footprints.compute_structure_rings.return_value = (None, [], [])
    mock_footprints.get_neighbor_structure_metrics.return_value = None

    # WHPAdapter returns a successful observation — burn_prob will be non-None.
    whp_ok = MagicMock(
        status="ok", notes=[], source_dataset="USFS WHP raster",
        raw_value=0.35, burn_probability_index=35.0, hazard_severity_index=40.0,
        hazard_class=3,
    )

    with (
        patch("backend.wildfire_data.PropertyAnchorResolver", return_value=mock_resolver_instance),
        patch("backend.wildfire_data.BuildingFootprintClient", return_value=mock_footprints),
    ):
        client = WildfireDataClient()
        client._landfire_cog_client = None
        client._fire_history_client = None
        client._nlcd_client = None
        client._whp_client = mock_whp

        whp_missing_sample = MagicMock(
            status="missing", notes=["No burn_prob path"], source_dataset=None,
            raw_value=None, burn_probability_index=None, hazard_severity_index=None,
            hazard_class=None,
        )
        with (
            patch.object(client, "_sample_layer_value_detailed",
                         return_value=(None, "not_configured", "No path")),
            patch.object(client, "_sample_circle", return_value=[]),
            patch.object(client, "_sample_layer_value", return_value=(None, "not_configured")),
            patch.object(client, "_file_exists", return_value=False),
            patch.object(client, "_historical_fire_metrics", return_value=(None, None, "missing")),
            patch.object(client, "_wildland_distance_metrics", return_value=(None, None)),
            patch.object(client.feature_bundle_cache, "load", return_value=None),
            # whp_adapter returns OK → burn_prob is set → proxy should NOT be called
            patch.object(client.whp_adapter, "sample", return_value=whp_ok),
            patch.object(
                client.mtbs_adapter, "summarize",
                return_value=MagicMock(
                    status="missing", notes=[], source_dataset=None,
                    nearest_perimeter_km=None, intersects_prior_burn=False,
                    nearby_high_severity=False, fire_history_index=None,
                ),
            ),
        ):
            ctx = client.collect_context(46.87, -113.99)

    assert ctx is not None
    assert ctx.burn_probability_index is not None, (
        "burn_probability_index should be set from WHPAdapter"
    )
    assert mock_whp.get_whp_index.call_count == 0, (
        "WHPClient must not be called when WHPAdapter already returned a value"
    )


def test_whp_proxy_returns_none_gracefully(tmp_path) -> None:
    """
    Test 12: When WHPClient.get_whp_index returns None (insufficient inputs),
    burn_probability_index remains None and no exception is raised.
    """
    from unittest.mock import MagicMock

    mock_whp = MagicMock()
    mock_whp.get_whp_index.return_value = None
    mock_whp.enabled = True

    ctx = _run_collect_context_with_whp_client(
        tmp_path,
        whp_client=mock_whp,
        fuel_samples=[],
        canopy_samples=[],
    )

    assert ctx is not None, "collect_context must not raise when WHP proxy returns None"
    assert ctx.burn_probability_index is None
