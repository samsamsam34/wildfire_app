from __future__ import annotations

from backend.main import _build_confidence
from backend.models import AssumptionsBlock


def _tier_rank(tier: str) -> int:
    return {
        "preliminary": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
    }.get(str(tier or "preliminary"), 0)


def _assumptions(*, missing_inputs: list[str] | None = None) -> AssumptionsBlock:
    return AssumptionsBlock(
        confirmed_inputs={
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 25,
            "construction_year": 2014,
        },
        observed_inputs={
            "address": "123 Confidence Test Ln, Test, MT",
            "roof_type": "class a",
            "vent_type": "ember-resistant",
            "defensible_space_ft": 25,
            "construction_year": 2014,
        },
        inferred_inputs={},
        missing_inputs=list(missing_inputs or []),
        assumptions_used=[],
    )


def _property_context(*, footprint_used: bool = True, ring_metrics: bool = True) -> dict[str, object]:
    return {
        "footprint_used": footprint_used,
        "fallback_mode": "footprint" if footprint_used else "point_based",
        "ring_metrics": (
            {
                "zone_0_5_ft": {"vegetation_density": 22.0},
                "zone_5_30_ft": {"vegetation_density": 35.0},
            }
            if ring_metrics
            else {}
        ),
    }


def _layer_status() -> dict[str, str]:
    return {
        "burn_probability": "ok",
        "hazard": "ok",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "ok",
        "fire_history": "ok",
    }


def _preflight(
    *,
    fallback_weight_fraction: float,
    observed_weight_fraction: float,
    observed_feature_count: int,
    inferred_feature_count: int,
    fallback_feature_count: int,
    geometry_quality_score: float = 0.86,
    regional_context_coverage_score: float = 85.0,
    regional_enrichment_consumption_score: float = 85.0,
    property_specificity_score: float = 82.0,
    assessment_output_state: str = "property_specific_assessment",
) -> dict[str, object]:
    return {
        "assessment_output_state": assessment_output_state,
        "feature_coverage_percent": 82.0,
        "missing_core_layer_count": 0,
        "major_environmental_missing_count": 0,
        "geometry_basis": "footprint" if geometry_quality_score >= 0.65 else "geocode_point",
        "fallback_weight_fraction": fallback_weight_fraction,
        "region_property_specific_readiness": "property_specific_ready",
        "region_required_missing_count": 0,
        "region_optional_missing_count": 1,
        "region_enrichment_missing_count": 1,
        "observed_feature_count": observed_feature_count,
        "inferred_feature_count": inferred_feature_count,
        "fallback_feature_count": fallback_feature_count,
        "geometry_quality_score": geometry_quality_score,
        "regional_context_coverage_score": regional_context_coverage_score,
        "regional_enrichment_consumption_score": regional_enrichment_consumption_score,
        "property_specificity_score": property_specificity_score,
        "observed_weight_fraction": observed_weight_fraction,
    }


def _confidence(
    *,
    assumptions: AssumptionsBlock,
    preflight: dict[str, object],
    observed_weight_fraction: float,
    fallback_dominance_ratio: float,
    geocode_verified: bool = True,
) -> tuple[float, str]:
    block = _build_confidence(
        assumptions,
        environmental_data_completeness=100.0,
        geocode_verified=geocode_verified,
        property_level_context=_property_context(footprint_used=True, ring_metrics=True),
        environmental_layer_status=_layer_status(),
        preflight=preflight,
        assessment_output_state=str(preflight.get("assessment_output_state") or "property_specific_assessment"),
        observed_weight_fraction=observed_weight_fraction,
        fallback_dominance_ratio=fallback_dominance_ratio,
    )
    return float(block.confidence_score), str(block.confidence_tier)


def test_more_missing_critical_fields_reduces_confidence() -> None:
    baseline_preflight = _preflight(
        fallback_weight_fraction=0.18,
        observed_weight_fraction=0.78,
        observed_feature_count=9,
        inferred_feature_count=1,
        fallback_feature_count=1,
    )
    missing_preflight = dict(baseline_preflight)
    score_base, tier_base = _confidence(
        assumptions=_assumptions(),
        preflight=baseline_preflight,
        observed_weight_fraction=0.78,
        fallback_dominance_ratio=0.18,
    )
    score_missing, tier_missing = _confidence(
        assumptions=_assumptions(
            missing_inputs=[
                "roof_type",
                "vent_type",
                "defensible_space_ft",
                "burn_probability_layer",
            ]
        ),
        preflight=missing_preflight,
        observed_weight_fraction=0.78,
        fallback_dominance_ratio=0.18,
    )
    assert score_missing < score_base
    assert _tier_rank(tier_missing) <= _tier_rank(tier_base)


def test_confidence_decreases_as_fallback_weight_increases() -> None:
    assumptions = _assumptions()
    low_score, low_tier = _confidence(
        assumptions=assumptions,
        preflight=_preflight(
            fallback_weight_fraction=0.10,
            observed_weight_fraction=0.82,
            observed_feature_count=10,
            inferred_feature_count=1,
            fallback_feature_count=1,
        ),
        observed_weight_fraction=0.82,
        fallback_dominance_ratio=0.10,
    )
    mid_score, mid_tier = _confidence(
        assumptions=assumptions,
        preflight=_preflight(
            fallback_weight_fraction=0.42,
            observed_weight_fraction=0.55,
            observed_feature_count=6,
            inferred_feature_count=3,
            fallback_feature_count=4,
        ),
        observed_weight_fraction=0.55,
        fallback_dominance_ratio=0.42,
    )
    high_score, high_tier = _confidence(
        assumptions=assumptions,
        preflight=_preflight(
            fallback_weight_fraction=0.72,
            observed_weight_fraction=0.24,
            observed_feature_count=2,
            inferred_feature_count=5,
            fallback_feature_count=8,
            geometry_quality_score=0.44,
            regional_context_coverage_score=48.0,
            regional_enrichment_consumption_score=45.0,
            property_specificity_score=38.0,
            assessment_output_state="limited_regional_estimate",
        ),
        observed_weight_fraction=0.24,
        fallback_dominance_ratio=0.72,
    )
    assert low_score > mid_score > high_score
    assert _tier_rank(low_tier) >= _tier_rank(mid_tier) >= _tier_rank(high_tier)


def test_direct_observation_improves_confidence() -> None:
    assumptions = _assumptions()
    weak_score, weak_tier = _confidence(
        assumptions=assumptions,
        preflight=_preflight(
            fallback_weight_fraction=0.38,
            observed_weight_fraction=0.32,
            observed_feature_count=3,
            inferred_feature_count=4,
            fallback_feature_count=5,
            geometry_quality_score=0.52,
            regional_context_coverage_score=58.0,
            regional_enrichment_consumption_score=56.0,
            property_specificity_score=48.0,
            assessment_output_state="address_level_estimate",
        ),
        observed_weight_fraction=0.32,
        fallback_dominance_ratio=0.38,
    )
    strong_score, strong_tier = _confidence(
        assumptions=assumptions,
        preflight=_preflight(
            fallback_weight_fraction=0.14,
            observed_weight_fraction=0.84,
            observed_feature_count=10,
            inferred_feature_count=1,
            fallback_feature_count=1,
            geometry_quality_score=0.90,
            regional_context_coverage_score=88.0,
            regional_enrichment_consumption_score=87.0,
            property_specificity_score=86.0,
            assessment_output_state="property_specific_assessment",
        ),
        observed_weight_fraction=0.84,
        fallback_dominance_ratio=0.14,
    )
    assert strong_score > weak_score
    assert _tier_rank(strong_tier) >= _tier_rank(weak_tier)


def test_low_observed_weight_fraction_cannot_be_high_confidence() -> None:
    # Only one structural fact confirmed, so the effective_observed boost (requires >=3
    # confirmed core facts) does not apply.  fallback_weight_fraction is intentionally
    # kept below the existing 0.45 gate to isolate the new observed-weight gate.
    assumptions = AssumptionsBlock(
        confirmed_inputs={"roof_type": "class a"},
        observed_inputs={
            "address": "456 Fallback Lane, Test, MT",
            "roof_type": "class a",
            "vent_type": "standard",
            "defensible_space_ft": 15,
            "construction_year": 2005,
        },
        inferred_inputs={},
        missing_inputs=[],
        assumptions_used=[],
    )
    _, tier = _confidence(
        assumptions=assumptions,
        preflight=_preflight(
            fallback_weight_fraction=0.38,
            observed_weight_fraction=0.48,
            observed_feature_count=5,
            inferred_feature_count=3,
            fallback_feature_count=4,
            geometry_quality_score=0.88,
            regional_context_coverage_score=92.0,
            regional_enrichment_consumption_score=90.0,
            property_specificity_score=72.0,
        ),
        observed_weight_fraction=0.48,
        fallback_dominance_ratio=0.38,
    )
    assert tier != "high", (
        f"Properties with <55% observed scoring weight must not reach 'high' confidence; got {tier!r}"
    )


def test_fallback_heavy_records_cannot_be_high_confidence() -> None:
    score, tier = _confidence(
        assumptions=_assumptions(),
        preflight=_preflight(
            fallback_weight_fraction=0.66,
            observed_weight_fraction=0.30,
            observed_feature_count=2,
            inferred_feature_count=4,
            fallback_feature_count=7,
            geometry_quality_score=0.48,
            regional_context_coverage_score=50.0,
            regional_enrichment_consumption_score=49.0,
            property_specificity_score=40.0,
            assessment_output_state="limited_regional_estimate",
        ),
        observed_weight_fraction=0.30,
        fallback_dominance_ratio=0.66,
    )
    assert score <= 45.0
    assert tier in {"preliminary", "low"}
