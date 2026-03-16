from __future__ import annotations

from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import load_scoring_config
from backend.wildfire_data import WildfireContext


def _context(*, ring_metrics: dict[str, dict[str, float]] | None = None) -> WildfireContext:
    rm = ring_metrics or {}
    return WildfireContext(
        environmental_index=55.0,
        slope_index=52.0,
        aspect_index=50.0,
        fuel_index=60.0,
        moisture_index=42.0,
        canopy_index=58.0,
        wildland_distance_index=47.0,
        historic_fire_index=33.0,
        burn_probability_index=56.0,
        hazard_severity_index=54.0,
        burn_probability=56.0,
        wildfire_hazard=54.0,
        slope=52.0,
        fuel_model=60.0,
        canopy_cover=58.0,
        historic_fire_distance=1.6,
        wildland_distance=180.0,
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
        structure_ring_metrics=rm,
        property_level_context={
            "footprint_used": bool(rm),
            "footprint_status": "used" if rm else "not_found",
            "fallback_mode": "footprint" if rm else "point_based",
            "ring_metrics": rm,
        },
    )


def test_missing_factor_omission_reduces_observed_weight_fraction():
    engine = RiskEngine(load_scoring_config())
    full_attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=35.0,
        construction_year=2016,
    )
    full_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 25.0},
            "ring_5_30_ft": {"vegetation_density": 34.0},
            "ring_30_100_ft": {"vegetation_density": 45.0},
        }
    )
    full = engine.score(full_attrs, lat=0.0, lon=0.0, context=full_context)

    sparse_attrs = PropertyAttributes()
    sparse_context = _context(ring_metrics={})
    sparse_context.moisture_index = None
    sparse_context.burn_probability_index = None
    sparse_context.hazard_severity_index = None
    sparse = engine.score(sparse_attrs, lat=0.0, lon=0.0, context=sparse_context)

    assert full.observed_weight_fraction > sparse.observed_weight_fraction
    assert sparse.missing_factor_count >= 1
    assert sparse.uncertainty_penalty > 0.0
    assert any(bool(row.get("omitted_due_to_missing")) for row in sparse.weighted_contributions.values())
