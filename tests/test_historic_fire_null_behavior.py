"""Tests for historic_fire_risk null behavior.

When historic_fire_index is None (layer absent), the submodel must signal
"no data" rather than "no historical fire." Specifically:
  - raw_score and clamped_score must be None (absent, not zero)
  - key_inputs["historic_fire_index"] must be None
  - The submodel must be excluded from numeric blending (observed_input_count == 0)
  - A property with a present (non-zero) fire index must outscore one with absent data
"""
from __future__ import annotations

from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import load_scoring_config
from backend.wildfire_data import WildfireContext


def _context(*, historic_fire_index: float | None) -> WildfireContext:
    return WildfireContext(
        environmental_index=55.0,
        slope_index=50.0,
        aspect_index=50.0,
        fuel_index=55.0,
        moisture_index=45.0,
        canopy_index=55.0,
        wildland_distance_index=50.0,
        historic_fire_index=historic_fire_index,
        burn_probability_index=55.0,
        hazard_severity_index=52.0,
        burn_probability=55.0,
        wildfire_hazard=52.0,
        slope=50.0,
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
            "fire_history": "ok" if historic_fire_index is not None else "missing",
        },
        data_sources=["unit-test"],
        assumptions=[],
        structure_ring_metrics={},
        property_level_context={
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
        },
    )


def _attrs() -> PropertyAttributes:
    return PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=30.0,
        construction_year=2015,
    )


def test_absent_historic_index_sets_raw_score_to_none() -> None:
    """raw_score must be None when historic_fire_index is absent — not 0.0."""
    engine = RiskEngine(load_scoring_config())
    result = engine.score(_attrs(), lat=0.0, lon=0.0, context=_context(historic_fire_index=None))
    submodel = result.submodel_scores["historic_fire_risk"]
    assert submodel.raw_score is None, (
        f"raw_score must be None when data is absent; got {submodel.raw_score}"
    )


def test_absent_historic_index_sets_clamped_score_to_none() -> None:
    """clamped_score must be None when historic_fire_index is absent."""
    engine = RiskEngine(load_scoring_config())
    result = engine.score(_attrs(), lat=0.0, lon=0.0, context=_context(historic_fire_index=None))
    submodel = result.submodel_scores["historic_fire_risk"]
    assert submodel.clamped_score is None, (
        f"clamped_score must be None when data is absent; got {submodel.clamped_score}"
    )


def test_absent_historic_index_excluded_from_blend() -> None:
    """historic_fire_risk must be omitted from numeric blending when the index is None.

    The blending loop uses observed_input_count == 0 to set effective_weight = 0.
    This is already triggered by key_inputs['historic_fire_index'] == None, so this
    test confirms the exclusion is intact after the null-handling change.
    """
    engine = RiskEngine(load_scoring_config())
    result = engine.score(_attrs(), lat=0.0, lon=0.0, context=_context(historic_fire_index=None))
    contribution = result.weighted_contributions.get("historic_fire_risk", {})
    assert bool(contribution.get("omitted_due_to_missing")), (
        "historic_fire_risk must be omitted from numeric weighting when index is None"
    )


def test_present_historic_index_has_nonnull_raw_and_clamped_score() -> None:
    """When historic_fire_index is provided, raw_score and clamped_score must both
    be numeric (not None)."""
    engine = RiskEngine(load_scoring_config())
    result = engine.score(_attrs(), lat=0.0, lon=0.0, context=_context(historic_fire_index=60.0))
    submodel = result.submodel_scores["historic_fire_risk"]
    assert submodel.raw_score is not None, "raw_score must be numeric when data is present"
    assert submodel.clamped_score is not None, "clamped_score must be numeric when data is present"
    assert submodel.raw_score == 60.0
