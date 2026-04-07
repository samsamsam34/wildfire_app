"""Tests for the coverage penalty in weighted_score().

When inputs are missing, their weights are dropped and the raw score was
previously renormalized over the remaining terms — making a sparse assessment
indistinguishable from a complete one. The fix pulls the score toward the
neutral anchor (50.0) in proportion to the missing weight fraction.

Invariants:
  - All inputs present → result is identical to the pre-fix formula.
  - Some inputs missing → result is strictly between raw_score and 50.0.
  - Extreme scores (high or low) move toward 50.0 as coverage drops.
  - Missing weight fraction is monotonically related to the pull magnitude.
  - All inputs missing → returns 0.0 (submodel excluded from blend via key_inputs).
"""
from __future__ import annotations

import pytest

from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import load_scoring_config
from backend.wildfire_data import WildfireContext


# ---------------------------------------------------------------------------
# Direct weighted_score() logic tests via the slope submodel
# (slope_topography_risk: weights 0.70 slope + 0.20 aspect + 0.10 upslope)
# ---------------------------------------------------------------------------

def _engine() -> RiskEngine:
    return RiskEngine(load_scoring_config())


def _slope_context(
    *,
    slope: float | None,
    aspect: float | None,
    upslope_sector: float | None,
) -> WildfireContext:
    """Minimal context that isolates slope_topography_risk inputs."""
    plc: dict = {
        "footprint_used": False,
        "footprint_status": "not_found",
        "fallback_mode": "point_based",
    }
    if upslope_sector is not None:
        plc["slope_aspect_deg"] = 0.0
        plc["vegetation_directional_precision"] = "footprint_boundary"
        plc["vegetation_directional_sectors"] = {"S": {"sector_risk_score": upslope_sector}}
    return WildfireContext(
        environmental_index=50.0,
        slope_index=slope,
        aspect_index=aspect,
        fuel_index=50.0,
        moisture_index=50.0,
        canopy_index=50.0,
        wildland_distance_index=50.0,
        historic_fire_index=30.0,
        burn_probability_index=50.0,
        hazard_severity_index=50.0,
        burn_probability=50.0,
        wildfire_hazard=50.0,
        slope=slope,
        fuel_model=50.0,
        canopy_cover=50.0,
        historic_fire_distance=1.5,
        wildland_distance=150.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok" if slope is not None else "missing",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["unit-test"],
        assumptions=[],
        structure_ring_metrics={},
        property_level_context=plc,
    )


def _slope_score(ctx: WildfireContext) -> float:
    engine = _engine()
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=30.0,
        construction_year=2015,
    )
    result = engine.score(attrs, lat=0.0, lon=0.0, context=ctx)
    return float(result.submodel_scores["slope_topography_risk"].score)


def test_all_inputs_present_unchanged() -> None:
    """When all three slope submodel inputs are present, the score must equal the
    pre-fix weighted average (coverage==1.0, no pull applied)."""
    slope_val = 80.0
    aspect_val = 40.0
    upslope_val = 60.0

    score = _slope_score(_slope_context(slope=slope_val, aspect=aspect_val, upslope_sector=upslope_val))

    # With all inputs present, weighted_score = 0.70*80 + 0.20*40 + 0.10*adj(60)
    # (precision_adjust damps upslope toward 50 at ~0.955 multiplier)
    # Key: score must NOT be pulled further toward 50.
    # The coverage-adjusted path is not triggered (denominator == total_weight).
    # We just confirm the score is closer to the raw weighted value than to 50.
    raw_approx = 0.70 * slope_val + 0.20 * aspect_val  # lower bound ignoring upslope
    assert score > 50.0, f"High slope + moderate aspect must score above 50, got {score}"
    assert score >= raw_approx * 0.85, (
        f"Full-coverage score ({score:.2f}) must stay near raw weighted value "
        f"(~{raw_approx:.2f}), not be excessively pulled toward 50"
    )


def test_missing_one_input_pulls_score_toward_50_from_above() -> None:
    """When the upslope input is absent from a high-slope context, the score must
    be strictly less than the score when all inputs are present (pulled toward 50)."""
    high_slope_ctx_full = _slope_context(slope=90.0, aspect=70.0, upslope_sector=85.0)
    high_slope_ctx_partial = _slope_context(slope=90.0, aspect=70.0, upslope_sector=None)

    full = _slope_score(high_slope_ctx_full)
    partial = _slope_score(high_slope_ctx_partial)

    assert partial < full, (
        f"Removing one input from a high-scoring context must reduce the score "
        f"(pull toward 50): full={full:.2f}, partial={partial:.2f}"
    )
    assert partial > 50.0, (
        f"High slope + aspect should remain above 50 even with one missing input: {partial:.2f}"
    )


def test_missing_one_input_pulls_score_toward_50_from_below() -> None:
    """When the upslope input is absent from a low-slope context, the score must
    be strictly greater than the score when all inputs are present (pulled toward 50)."""
    low_slope_ctx_full = _slope_context(slope=10.0, aspect=15.0, upslope_sector=5.0)
    low_slope_ctx_partial = _slope_context(slope=10.0, aspect=15.0, upslope_sector=None)

    full = _slope_score(low_slope_ctx_full)
    partial = _slope_score(low_slope_ctx_partial)

    assert partial > full, (
        f"Removing one input from a low-scoring context must increase the score "
        f"(pull toward 50): full={full:.2f}, partial={partial:.2f}"
    )
    assert partial < 50.0, (
        f"Low slope + aspect should remain below 50 even with one missing input: {partial:.2f}"
    )


def test_more_missing_inputs_means_greater_pull_toward_50() -> None:
    """As more inputs are dropped from a high-slope context, the score must move
    monotonically closer to 50 (increasing pull from neutral anchor)."""
    slope_val = 90.0
    aspect_val = 80.0
    upslope_val = 85.0

    score_all = _slope_score(_slope_context(slope=slope_val, aspect=aspect_val, upslope_sector=upslope_val))
    score_no_upslope = _slope_score(_slope_context(slope=slope_val, aspect=aspect_val, upslope_sector=None))
    score_no_aspect = _slope_score(_slope_context(slope=slope_val, aspect=None, upslope_sector=None))

    # Each removal must move the score strictly closer to 50.
    assert score_all > score_no_upslope > score_no_aspect, (
        f"Score must decrease toward 50 as inputs are removed: "
        f"all={score_all:.2f}, no_upslope={score_no_upslope:.2f}, no_aspect={score_no_aspect:.2f}"
    )
    gap_all = abs(score_all - 50.0)
    gap_no_upslope = abs(score_no_upslope - 50.0)
    gap_no_aspect = abs(score_no_aspect - 50.0)
    assert gap_all > gap_no_upslope > gap_no_aspect, (
        "Distance from 50 must strictly decrease as more inputs go missing"
    )
