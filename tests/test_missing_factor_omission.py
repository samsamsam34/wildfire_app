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


def test_structure_proxy_features_raise_vulnerability_when_direct_structure_age_missing():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=30.0,
        construction_year=None,
    )
    low_proxy_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 24.0},
            "ring_5_30_ft": {"vegetation_density": 31.0},
            "ring_30_100_ft": {"vegetation_density": 42.0},
        }
    )
    low_proxy_context.property_level_context.update(
        {
            "neighboring_structure_metrics": {
                "nearby_structure_count_100_ft": 0.0,
                "nearby_structure_count_300_ft": 3.0,
                "nearest_structure_distance_ft": 180.0,
            },
            "building_age_proxy_year": 2016.0,
            "building_age_material_proxy_risk": 28.0,
        }
    )
    high_proxy_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 24.0},
            "ring_5_30_ft": {"vegetation_density": 31.0},
            "ring_30_100_ft": {"vegetation_density": 42.0},
        }
    )
    high_proxy_context.property_level_context.update(
        {
            "neighboring_structure_metrics": {
                "nearby_structure_count_100_ft": 5.0,
                "nearby_structure_count_300_ft": 18.0,
                "nearest_structure_distance_ft": 12.0,
            },
            "building_age_proxy_year": 1965.0,
            "building_age_material_proxy_risk": 76.0,
        }
    )

    low_proxy = engine.score(attrs, lat=0.0, lon=0.0, context=low_proxy_context)
    high_proxy = engine.score(attrs, lat=0.0, lon=0.0, context=high_proxy_context)

    low_struct = low_proxy.submodel_scores["structure_vulnerability_risk"].score
    high_struct = high_proxy.submodel_scores["structure_vulnerability_risk"].score
    assert high_struct > low_struct
    assert (high_struct - low_struct) >= 5.0


def test_structure_density_and_clustering_proxies_increase_structure_vulnerability() -> None:
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(
        roof_type=None,
        vent_type=None,
        defensible_space_ft=24.0,
        construction_year=None,
    )
    low_proxy_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 28.0},
            "ring_5_30_ft": {"vegetation_density": 37.0},
            "ring_30_100_ft": {"vegetation_density": 46.0},
        }
    )
    low_proxy_context.property_level_context.update(
        {
            "structure_density": 18.0,
            "clustering_index": 16.0,
            "distance_to_nearest_structure_ft": 240.0,
            "neighboring_structure_metrics": {
                "nearby_structure_count_100_ft": 0.0,
                "nearby_structure_count_300_ft": 2.0,
                "nearest_structure_distance_ft": 240.0,
                "distance_to_nearest_structure_ft": 240.0,
            },
            "building_age_proxy_year": 2015.0,
            "building_age_material_proxy_risk": 30.0,
        }
    )
    high_proxy_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 28.0},
            "ring_5_30_ft": {"vegetation_density": 37.0},
            "ring_30_100_ft": {"vegetation_density": 46.0},
        }
    )
    high_proxy_context.property_level_context.update(
        {
            "structure_density": 84.0,
            "clustering_index": 79.0,
            "distance_to_nearest_structure_ft": 14.0,
            "neighboring_structure_metrics": {
                "nearby_structure_count_100_ft": 6.0,
                "nearby_structure_count_300_ft": 21.0,
                "nearest_structure_distance_ft": 14.0,
                "distance_to_nearest_structure_ft": 14.0,
            },
            "building_age_proxy_year": 1962.0,
            "building_age_material_proxy_risk": 82.0,
        }
    )

    low_proxy = engine.score(attrs, lat=0.0, lon=0.0, context=low_proxy_context)
    high_proxy = engine.score(attrs, lat=0.0, lon=0.0, context=high_proxy_context)

    low_struct = low_proxy.submodel_scores["structure_vulnerability_risk"].score
    high_struct = high_proxy.submodel_scores["structure_vulnerability_risk"].score
    assert high_struct > low_struct
    # Coverage penalty in weighted_score() correctly damps spread when direct
    # structure attributes are absent (roof, vent, window, year all missing).
    # Separation narrows from ~8 pts (old renorm) to ~3 pts (correct behavior).
    assert (high_struct - low_struct) >= 2.0

    high_inputs = high_proxy.submodel_scores["structure_vulnerability_risk"].key_inputs
    assert high_inputs.get("structure_density_proxy_index") is not None
    assert high_inputs.get("clustering_index") is not None
    assert high_inputs.get("distance_to_nearest_structure_ft") is not None


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
    baseline_weights = engine.resolve_blend_weights(insurance_readiness_score=readiness, risk=None)
    adaptive_weights = engine.resolve_blend_weights(insurance_readiness_score=readiness, risk=risk)

    assert adaptive_weights != baseline_weights
    assert 0.0 <= baseline <= 100.0
    assert 0.0 <= adaptive <= 100.0


def test_blended_score_compounds_high_hazard_with_high_near_structure_pressure():
    engine = RiskEngine(load_scoring_config())
    high_hazard_high_near = engine.compute_blended_wildfire_score(
        site_hazard_score=82.0,
        home_ignition_vulnerability_score=84.0,
        insurance_readiness_score=40.0,
    )
    high_hazard_low_near = engine.compute_blended_wildfire_score(
        site_hazard_score=82.0,
        home_ignition_vulnerability_score=42.0,
        insurance_readiness_score=40.0,
    )
    assert high_hazard_high_near > high_hazard_low_near + 3.0


def test_blended_score_materially_changes_with_structure_vulnerability_at_fixed_hazard():
    engine = RiskEngine(load_scoring_config())
    low_structure_vulnerability = engine.compute_blended_wildfire_score(
        site_hazard_score=62.0,
        home_ignition_vulnerability_score=28.0,
        insurance_readiness_score=62.0,
    )
    high_structure_vulnerability = engine.compute_blended_wildfire_score(
        site_hazard_score=62.0,
        home_ignition_vulnerability_score=74.0,
        insurance_readiness_score=62.0,
    )
    assert high_structure_vulnerability > low_structure_vulnerability + 8.0


def test_blended_score_penalizes_low_vulnerability_less_in_moderate_hazard_band():
    engine = RiskEngine(load_scoring_config())
    lower_vulnerability = engine.compute_blended_wildfire_score(
        site_hazard_score=54.0,
        home_ignition_vulnerability_score=38.0,
        insurance_readiness_score=60.0,
    )
    moderate_vulnerability = engine.compute_blended_wildfire_score(
        site_hazard_score=54.0,
        home_ignition_vulnerability_score=48.0,
        insurance_readiness_score=60.0,
    )
    assert moderate_vulnerability > lower_vulnerability + 2.0


def test_blended_score_gives_credit_for_strong_readiness_in_low_hazard_conditions():
    engine = RiskEngine(load_scoring_config())
    strong_readiness = engine.compute_blended_wildfire_score(
        site_hazard_score=34.0,
        home_ignition_vulnerability_score=36.0,
        insurance_readiness_score=90.0,
    )
    weak_readiness = engine.compute_blended_wildfire_score(
        site_hazard_score=34.0,
        home_ignition_vulnerability_score=36.0,
        insurance_readiness_score=42.0,
    )
    assert strong_readiness + 4.0 < weak_readiness


def test_blended_score_amplifies_hazard_x_vegetation_interaction():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=18.0)

    low_vegetation_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 24.0},
            "ring_5_30_ft": {"vegetation_density": 32.0},
            "ring_30_100_ft": {"vegetation_density": 40.0},
        }
    )
    high_vegetation_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 78.0},
            "ring_5_30_ft": {"vegetation_density": 86.0},
            "ring_30_100_ft": {"vegetation_density": 82.0},
        }
    )

    low_veg_risk = engine.score(attrs, lat=0.0, lon=0.0, context=low_vegetation_context)
    high_veg_risk = engine.score(attrs, lat=0.0, lon=0.0, context=high_vegetation_context)

    low_veg_blended = engine.compute_blended_wildfire_score(
        site_hazard_score=78.0,
        home_ignition_vulnerability_score=66.0,
        insurance_readiness_score=55.0,
        risk=low_veg_risk,
    )
    high_veg_blended = engine.compute_blended_wildfire_score(
        site_hazard_score=78.0,
        home_ignition_vulnerability_score=66.0,
        insurance_readiness_score=55.0,
        risk=high_veg_risk,
    )
    assert high_veg_blended > low_veg_blended + 1.0


def test_blended_score_amplifies_hazard_x_slope_and_hazard_x_fuel_proximity():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=20.0)

    mild_context = _context(ring_metrics={"ring_5_30_ft": {"vegetation_density": 45.0}})
    mild_context.slope_index = 38.0
    mild_context.wildland_distance_index = 30.0
    steep_close_fuel_context = _context(ring_metrics={"ring_5_30_ft": {"vegetation_density": 45.0}})
    steep_close_fuel_context.slope_index = 88.0
    steep_close_fuel_context.wildland_distance_index = 86.0

    mild_risk = engine.score(attrs, lat=0.0, lon=0.0, context=mild_context)
    steep_close_fuel_risk = engine.score(attrs, lat=0.0, lon=0.0, context=steep_close_fuel_context)

    mild_blended = engine.compute_blended_wildfire_score(
        site_hazard_score=76.0,
        home_ignition_vulnerability_score=62.0,
        insurance_readiness_score=58.0,
        risk=mild_risk,
    )
    steep_close_fuel_blended = engine.compute_blended_wildfire_score(
        site_hazard_score=76.0,
        home_ignition_vulnerability_score=62.0,
        insurance_readiness_score=58.0,
        risk=steep_close_fuel_risk,
    )
    assert steep_close_fuel_blended > mild_blended + 0.5


def test_blended_score_dampens_when_hardening_is_strong():
    engine = RiskEngine(load_scoring_config())
    strong_attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=45.0,
        construction_year=2019,
    )
    weak_attrs = PropertyAttributes(
        roof_type="wood",
        vent_type="standard",
        defensible_space_ft=5.0,
        construction_year=1975,
    )
    shared_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 48.0},
            "ring_5_30_ft": {"vegetation_density": 62.0},
        }
    )

    strong_risk = engine.score(strong_attrs, lat=0.0, lon=0.0, context=shared_context)
    weak_risk = engine.score(weak_attrs, lat=0.0, lon=0.0, context=shared_context)

    strong_blended = engine.compute_blended_wildfire_score(
        site_hazard_score=67.0,
        home_ignition_vulnerability_score=63.0,
        insurance_readiness_score=86.0,
        risk=strong_risk,
    )
    weak_blended = engine.compute_blended_wildfire_score(
        site_hazard_score=67.0,
        home_ignition_vulnerability_score=63.0,
        insurance_readiness_score=86.0,
        risk=weak_risk,
    )
    assert strong_blended + 1.0 < weak_blended


def test_home_vulnerability_weights_0_5_ft_more_than_5_30_ft():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=22.0)

    base_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 28.0},
            "ring_5_30_ft": {"vegetation_density": 34.0},
            "ring_30_100_ft": {"vegetation_density": 45.0},
        }
    )
    high_0_5_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 82.0},
            "ring_5_30_ft": {"vegetation_density": 34.0},
            "ring_30_100_ft": {"vegetation_density": 45.0},
        }
    )
    high_5_30_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 28.0},
            "ring_5_30_ft": {"vegetation_density": 82.0},
            "ring_30_100_ft": {"vegetation_density": 45.0},
        }
    )

    base_home = engine.compute_home_ignition_vulnerability_score(engine.score(attrs, lat=0.0, lon=0.0, context=base_context))
    high_0_5_home = engine.compute_home_ignition_vulnerability_score(
        engine.score(attrs, lat=0.0, lon=0.0, context=high_0_5_context)
    )
    high_5_30_home = engine.compute_home_ignition_vulnerability_score(
        engine.score(attrs, lat=0.0, lon=0.0, context=high_5_30_context)
    )

    delta_0_5 = high_0_5_home - base_home
    delta_5_30 = high_5_30_home - base_home
    assert delta_0_5 > delta_5_30 + 1.5


def test_home_vulnerability_sharp_increase_when_vegetation_is_very_close():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=22.0)
    ring_metrics = {
        "ring_0_5_ft": {"vegetation_density": 55.0},
        "ring_5_30_ft": {"vegetation_density": 60.0},
        "ring_30_100_ft": {"vegetation_density": 48.0},
    }
    far_context = _context(ring_metrics=ring_metrics)
    near_context = _context(ring_metrics=ring_metrics)
    far_context.property_level_context["nearest_vegetation_distance_ft"] = 12.0
    near_context.property_level_context["nearest_vegetation_distance_ft"] = 1.5

    far_home = engine.compute_home_ignition_vulnerability_score(engine.score(attrs, lat=0.0, lon=0.0, context=far_context))
    near_home = engine.compute_home_ignition_vulnerability_score(engine.score(attrs, lat=0.0, lon=0.0, context=near_context))
    assert near_home > far_home + 4.0


def test_home_vulnerability_clearing_has_diminishing_returns():
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(roof_type="class a", vent_type="ember-resistant", defensible_space_ft=25.0)

    high_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 88.0},
            "ring_5_30_ft": {"vegetation_density": 82.0},
            "ring_30_100_ft": {"vegetation_density": 58.0},
        }
    )
    medium_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 44.0},
            "ring_5_30_ft": {"vegetation_density": 54.0},
            "ring_30_100_ft": {"vegetation_density": 58.0},
        }
    )
    low_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 12.0},
            "ring_5_30_ft": {"vegetation_density": 22.0},
            "ring_30_100_ft": {"vegetation_density": 58.0},
        }
    )
    high_context.property_level_context["nearest_vegetation_distance_ft"] = 2.0
    medium_context.property_level_context["nearest_vegetation_distance_ft"] = 8.0
    low_context.property_level_context["nearest_vegetation_distance_ft"] = 24.0

    high_home = engine.compute_home_ignition_vulnerability_score(engine.score(attrs, lat=0.0, lon=0.0, context=high_context))
    medium_home = engine.compute_home_ignition_vulnerability_score(engine.score(attrs, lat=0.0, lon=0.0, context=medium_context))
    low_home = engine.compute_home_ignition_vulnerability_score(engine.score(attrs, lat=0.0, lon=0.0, context=low_context))

    first_clear_delta = high_home - medium_home
    extra_clear_delta = medium_home - low_home
    assert high_home > medium_home > low_home
    assert first_clear_delta > extra_clear_delta + 1.0


def test_extreme_0_5_ft_vegetation_materially_increases_near_structure_submodels() -> None:
    engine = RiskEngine(load_scoring_config())
    attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=24.0,
        construction_year=2008,
    )
    low_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 18.0},
            "ring_5_30_ft": {"vegetation_density": 44.0},
            "ring_30_100_ft": {"vegetation_density": 52.0},
        }
    )
    high_context = _context(
        ring_metrics={
            "ring_0_5_ft": {"vegetation_density": 88.0},
            "ring_5_30_ft": {"vegetation_density": 44.0},
            "ring_30_100_ft": {"vegetation_density": 52.0},
        }
    )
    low_context.property_level_context.update(
        {
            "near_structure_vegetation_0_5_pct": 18.0,
            "near_structure_connectivity_index": 42.0,
        }
    )
    high_context.property_level_context.update(
        {
            "near_structure_vegetation_0_5_pct": 88.0,
            "near_structure_connectivity_index": 42.0,
        }
    )

    low_risk = engine.score(attrs, lat=0.0, lon=0.0, context=low_context)
    high_risk = engine.score(attrs, lat=0.0, lon=0.0, context=high_context)

    low_flame = low_risk.submodel_scores["flame_contact_risk"].score
    high_flame = high_risk.submodel_scores["flame_contact_risk"].score
    low_defensible = low_risk.submodel_scores["defensible_space_risk"].score
    high_defensible = high_risk.submodel_scores["defensible_space_risk"].score
    low_veg = low_risk.submodel_scores["vegetation_intensity_risk"].score
    high_veg = high_risk.submodel_scores["vegetation_intensity_risk"].score

    assert high_flame > low_flame
    assert high_defensible > low_defensible
    assert high_veg > low_veg
    assert (high_flame - low_flame) >= 8.0
    assert (high_defensible - low_defensible) >= 8.0
    assert (high_veg - low_veg) >= 10.0
