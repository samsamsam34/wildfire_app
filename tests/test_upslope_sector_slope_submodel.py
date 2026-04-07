"""Tests for upslope sector fuel risk wired into slope_topography_risk.

Covers the change that added upslope_sector_risk_index (weight 0.10) to the
slope_topography_risk submodel (reducing aspect_index weight from 0.30 to 0.20).

Four cases:
1. High upslope fuel increases slope_topography_risk vs. absent sector data.
2. Low upslope fuel decreases slope_topography_risk vs. absent sector data.
3. Increasing upslope sector risk produces monotonically non-decreasing scores.
4. upslope_sector_risk_index is exposed in key_inputs when sector data is present.
"""
from __future__ import annotations

from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import load_scoring_config
from backend.wildfire_data import WildfireContext


def _attrs() -> PropertyAttributes:
    return PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=30.0,
        construction_year=2015,
    )


def _context(*, extra_plc: dict | None = None) -> WildfireContext:
    """Fully-populated context. No upslope sector data unless extra_plc provides it."""
    plc: dict = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
    }
    if extra_plc:
        plc.update(extra_plc)
    return WildfireContext(
        environmental_index=55.0,
        slope_index=60.0,
        aspect_index=50.0,
        fuel_index=55.0,
        moisture_index=45.0,
        canopy_index=55.0,
        wildland_distance_index=50.0,
        historic_fire_index=30.0,
        burn_probability_index=55.0,
        hazard_severity_index=52.0,
        burn_probability=55.0,
        wildfire_hazard=52.0,
        slope=60.0,
        fuel_model=55.0,
        canopy_cover=55.0,
        historic_fire_distance=1.5,
        wildland_distance=150.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["unit-test"],
        assumptions=[],
        structure_ring_metrics={},
        property_level_context=plc,
    )


def _upslope_plc(sector_risk_score: float) -> dict:
    """Build property_level_context fields that activate upslope sector risk.

    slope_aspect_deg=0 means the slope faces north (downhill north), so the
    upslope bearing is 180° (south). The engine maps this to cardinal "S",
    so we place the sector_risk_score under that key.
    """
    return {
        "slope_aspect_deg": 0.0,
        "vegetation_directional_precision": "footprint_boundary",
        "vegetation_directional_sectors": {
            "S": {"sector_risk_score": sector_risk_score},
        },
    }


def _slope_score(engine: RiskEngine, ctx: WildfireContext) -> float:
    result = engine.score(_attrs(), lat=0.0, lon=0.0, context=ctx)
    return float(result.submodel_scores["slope_topography_risk"].score)


def test_high_upslope_fuel_increases_slope_risk() -> None:
    """Dense fuel in the upslope direction must raise slope_topography_risk above
    the baseline where no directional sector data is available."""
    engine = RiskEngine(load_scoring_config())
    baseline = _slope_score(engine, _context())
    with_high_upslope = _slope_score(engine, _context(extra_plc=_upslope_plc(90.0)))
    assert with_high_upslope > baseline, (
        f"High upslope fuel score ({with_high_upslope:.2f}) must exceed "
        f"no-sector baseline ({baseline:.2f})"
    )


def test_low_upslope_fuel_decreases_slope_risk() -> None:
    """Sparse fuel in the upslope direction must lower slope_topography_risk below
    the baseline where no directional sector data is available.

    When sector data is absent the denominator is 0.90 (slope 0.70 + aspect 0.20),
    so the renormalized score is higher than a score including a near-zero upslope
    term at full weight 1.0.
    """
    engine = RiskEngine(load_scoring_config())
    baseline = _slope_score(engine, _context())
    with_low_upslope = _slope_score(engine, _context(extra_plc=_upslope_plc(5.0)))
    assert with_low_upslope < baseline, (
        f"Low upslope fuel score ({with_low_upslope:.2f}) must be below "
        f"no-sector baseline ({baseline:.2f})"
    )


def test_upslope_risk_is_monotonic() -> None:
    """Increasing the upslope sector risk score must never decrease the
    slope_topography_risk score."""
    engine = RiskEngine(load_scoring_config())
    levels = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    scores = [
        _slope_score(engine, _context(extra_plc=_upslope_plc(level)))
        for level in levels
    ]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"Score must be non-decreasing: upslope={levels[i]} → {scores[i]:.2f}, "
            f"upslope={levels[i + 1]} → {scores[i + 1]:.2f}"
        )


def test_upslope_sector_risk_index_in_key_inputs_when_present() -> None:
    """upslope_sector_risk_index must appear as a non-None value in
    slope_topography_risk key_inputs when aspect and sector data are supplied."""
    engine = RiskEngine(load_scoring_config())
    ctx = _context(extra_plc=_upslope_plc(70.0))
    result = engine.score(_attrs(), lat=0.0, lon=0.0, context=ctx)
    key_inputs = result.submodel_scores["slope_topography_risk"].key_inputs
    assert "upslope_sector_risk_index" in key_inputs, (
        "upslope_sector_risk_index must be a key in slope submodel key_inputs"
    )
    assert key_inputs["upslope_sector_risk_index"] is not None, (
        "upslope_sector_risk_index must be non-None when slope_aspect_deg "
        "and vegetation_directional_sectors are both provided"
    )
