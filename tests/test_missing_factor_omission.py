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


def test_point_geometry_downweights_structure_specific_components():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=25.0,
        construction_year=2012,
    )
    footprint_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 32.0},
            "ring_5_30_ft": {"vegetation_density": 44.0},
            "ring_30_100_ft": {"vegetation_density": 51.0},
        }
    )
    point_context = _context(ring_metrics={})

    footprint_risk = engine.score(attrs, lat=0.0, lon=0.0, context=footprint_context)
    point_risk = engine.score(attrs, lat=0.0, lon=0.0, context=point_context)

    assert footprint_risk.geometry_basis == "footprint"
    assert point_risk.geometry_basis == "point"
    assert (
        point_risk.weighted_contributions["defensible_space_risk"]["effective_weight"]
        < footprint_risk.weighted_contributions["defensible_space_risk"]["effective_weight"]
    )
    assert bool(point_risk.weighted_contributions["defensible_space_risk"]["omitted_due_to_missing"]) is True
    assert str(point_risk.weighted_contributions["defensible_space_risk"].get("factor_evidence_status")) == "suppressed"
    point_structure_effective = sum(
        float(point_risk.weighted_contributions[name]["effective_weight"])
        for name in ["ember_exposure_risk", "structure_vulnerability_risk", "defensible_space_risk"]
    )
    footprint_structure_effective = sum(
        float(footprint_risk.weighted_contributions[name]["effective_weight"])
        for name in ["ember_exposure_risk", "structure_vulnerability_risk", "defensible_space_risk"]
    )
    assert point_structure_effective < footprint_structure_effective


def test_weighted_contributions_include_basis_component_and_support_metadata():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes()
    context = _context(ring_metrics={})
    context.moisture_index = None
    context.burn_probability_index = None
    context.hazard_severity_index = None

    risk = engine.score(attrs, lat=0.0, lon=0.0, context=context)
    rows = list(risk.weighted_contributions.values())
    assert rows
    assert all("basis" in row for row in rows)
    assert all("factor_evidence_status" in row for row in rows)
    assert all("component" in row for row in rows)
    assert all("support_level" in row for row in rows)
    assert any(str(row.get("basis")) in {"fallback", "missing"} for row in rows)
    assert any(str(row.get("factor_evidence_status")) in {"fallback", "suppressed"} for row in rows)


def test_blended_score_supports_adaptive_weighting_with_risk_context():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=20.0)
    point_context = _context(ring_metrics={})
    risk = engine.score(attrs, lat=0.0, lon=0.0, context=point_context)
    site = engine.compute_site_hazard_score(risk)
    home = engine.compute_home_ignition_vulnerability_score(risk)
    readiness = 60.0

    baseline = engine.compute_blended_wildfire_score(site, home, readiness)
    adaptive = engine.compute_blended_wildfire_score(site, home, readiness, risk=risk)

    assert adaptive != baseline
