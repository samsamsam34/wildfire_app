from __future__ import annotations

from backend.main import (
    _build_feature_coverage_preflight,
    _build_property_confidence_summary,
    _property_data_confidence_level,
)
from backend.models import AddressRequest, LayerCoverageSummary, PropertyAttributes
from backend.wildfire_data import WildfireContext


def _context() -> WildfireContext:
    return WildfireContext(
        environmental_index=None,
        slope_index=45.0,
        aspect_index=40.0,
        fuel_index=52.0,
        moisture_index=38.0,
        canopy_index=60.0,
        wildland_distance_index=44.0,
        historic_fire_index=41.0,
        burn_probability_index=55.0,
        hazard_severity_index=49.0,
    )


def test_preflight_uses_feature_bundle_metrics_for_specificity_caps():
    preflight = _build_feature_coverage_preflight(
        context=_context(),
        property_level_context={
            "footprint_used": True,
            "parcel_geometry": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
            "ring_metrics": {"ring_0_5_ft": {"vegetation_density": 55.0}},
            "feature_bundle_summary": {
                "coverage_metrics": {
                    "observed_weight_fraction": 0.31,
                    "fallback_dominance_ratio": 0.79,
                    "structure_geometry_quality_score": 0.58,
                    "environmental_layer_coverage_score": 64.0,
                    "regional_enrichment_consumption_score": 58.0,
                    "property_specificity_score": 52.0,
                    "observed_feature_count": 4,
                    "fallback_feature_count": 7,
                    "missing_feature_count": 2,
                }
            },
            "region_property_specific_readiness": "property_specific_ready",
        },
        coverage_summary=LayerCoverageSummary(),
    )

    assert preflight["assessment_specificity_tier"] in {"address_level", "regional_estimate"}
    assert preflight["limited_assessment_flag"] is True
    assert preflight["fallback_dominance_ratio"] >= 0.7
    assert preflight["structure_geometry_quality_score"] <= 0.6
    assert preflight["geometry_quality_score"] == preflight["structure_geometry_quality_score"]
    assert "regional_context_coverage_score" in preflight
    assert "regional_enrichment_consumption_score" in preflight


def test_preflight_disallows_property_specific_tier_for_proxy_near_structure_geometry():
    preflight = _build_feature_coverage_preflight(
        context=_context(),
        property_level_context={
            "footprint_used": True,
            "fallback_mode": "point_based",
            "parcel_geometry": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
            "ring_metrics": {"ring_0_5_ft": {"vegetation_density": 55.0}},
            "near_structure_vegetation_0_5_pct": 61.0,
            "canopy_adjacency_proxy_pct": 48.0,
            "vegetation_continuity_proxy_pct": 44.0,
            "near_structure_features": {
                "data_quality_tier": "point_proxy",
                "claim_strength": "coarse_directional",
                "supports_property_specific_claims": False,
            },
            "feature_bundle_summary": {
                "coverage_metrics": {
                    "observed_weight_fraction": 0.82,
                    "fallback_dominance_ratio": 0.12,
                    "structure_geometry_quality_score": 0.88,
                    "environmental_layer_coverage_score": 92.0,
                    "regional_enrichment_consumption_score": 91.0,
                    "property_specificity_score": 89.0,
                    "observed_feature_count": 12,
                    "fallback_feature_count": 1,
                    "missing_feature_count": 0,
                }
            },
            "region_property_specific_readiness": "property_specific_ready",
        },
        coverage_summary=LayerCoverageSummary(),
    )

    assert preflight["near_structure_data_quality_tier"] == "point_proxy"
    assert preflight["near_structure_supports_property_specific_claims"] is False
    assert preflight["assessment_specificity_tier"] in {"address_level", "regional_estimate"}
    assert preflight["limited_assessment_flag"] is True


def test_property_confidence_summary_high_with_strong_property_data():
    preflight = _build_feature_coverage_preflight(
        context=_context(),
        property_level_context={
            "footprint_used": True,
            "fallback_mode": "footprint",
            "parcel_id": "parcel-1",
            "parcel_geometry": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}},
            "parcel_resolution": {"status": "matched", "confidence": 93.0},
            "footprint_resolution": {"match_status": "matched", "confidence_score": 0.94},
            "ring_metrics": {"ring_0_5_ft": {"vegetation_density": 55.0}},
            "structure_attributes": {
                "area": {"sqft": 2400.0},
                "density_context": {"index": 71.0},
                "estimated_age_proxy": {"proxy_year": 1992.0},
                "shape_complexity": {"index": 38.0},
            },
            "region_property_specific_readiness": "property_specific_ready",
        },
        coverage_summary=LayerCoverageSummary(),
        payload=AddressRequest(
            address="10 High Confidence Way, Missoula, MT 59802",
            attributes=PropertyAttributes(
                roof_type="class a",
                vent_type="ember-resistant",
                window_type="dual-pane tempered",
                defensible_space_ft=40.0,
                construction_year=1998,
            ),
            confirmed_fields=["roof_type", "vent_type", "defensible_space_ft"],
        ),
    )
    summary = preflight.get("property_confidence_summary") or {}
    assert float(preflight.get("property_data_confidence") or 0.0) >= 75.0
    assert summary.get("level") in {"verified_property_specific", "strong_property_specific"}
    assert isinstance(summary.get("user_action_recommended"), str)
    assert isinstance(summary.get("key_reasons"), list)


def test_property_confidence_summary_low_when_fallback_heavy():
    preflight = _build_feature_coverage_preflight(
        context=_context(),
        property_level_context={
            "footprint_used": False,
            "fallback_mode": "point_based",
            "parcel_resolution": {"status": "not_found", "confidence": 0.0},
            "footprint_resolution": {"match_status": "none", "confidence_score": 0.0},
            "ring_metrics": {},
            "structure_attributes": {
                "area": {"sqft": None},
                "density_context": {"index": None},
                "estimated_age_proxy": None,
                "shape_complexity": {"index": None},
            },
            "feature_bundle_summary": {
                "coverage_metrics": {
                    "fallback_dominance_ratio": 0.84,
                    "observed_feature_count": 1,
                    "fallback_feature_count": 9,
                    "missing_feature_count": 4,
                    "observed_weight_fraction": 0.21,
                }
            },
            "region_property_specific_readiness": "limited_regional_ready",
        },
        coverage_summary=LayerCoverageSummary(),
        payload=AddressRequest(
            address="11 Fallback Heavy Rd, Missoula, MT 59802",
            attributes=PropertyAttributes(),
            confirmed_fields=[],
        ),
    )
    summary = preflight.get("property_confidence_summary") or {}
    assert preflight.get("property_data_confidence") is not None
    assert float(preflight.get("property_data_confidence")) < 50.0
    assert summary.get("level") in {"regional_estimate_with_anchor", "insufficient_property_identification"}
    assert "move pin" in str(summary.get("user_action_recommended") or "").lower()
    assert str(preflight.get("assessment_specificity_tier")) in {"regional_estimate", "address_level"}


def test_property_confidence_level_ladder_threshold_mapping():
    assert _property_data_confidence_level(94.0) == "verified_property_specific"
    assert _property_data_confidence_level(78.0) == "strong_property_specific"
    assert _property_data_confidence_level(61.0) == "address_level"
    assert _property_data_confidence_level(40.0) == "regional_estimate_with_anchor"
    assert _property_data_confidence_level(18.0) == "insufficient_property_identification"


def test_property_confidence_summary_supports_all_ladder_levels():
    def _score_for(ctx: dict, fallback_fraction: float, fallback_ratio: float) -> dict:
        return _build_property_confidence_summary(
                payload=AddressRequest(
                address="12345 Test Address Rd",
                attributes=PropertyAttributes(
                    roof_type="class a",
                    vent_type="ember-resistant",
                    defensible_space_ft=30.0,
                    construction_year=1994,
                ),
                confirmed_fields=["roof_type", "vent_type", "defensible_space_ft"],
                selected_structure_id=ctx.get("selected_structure_id"),
            ),
            property_level_context=ctx,
            fallback_evidence_fraction=fallback_fraction,
            fallback_dominance_ratio=fallback_ratio,
        )

    verified = _score_for(
        {
            "footprint_used": True,
            "fallback_mode": "footprint",
            "naip_feature_source": "prepared_region_naip",
            "ring_metrics": {"ring_0_5_ft": {"vegetation_density": 48.0}, "ring_5_30_ft": {"vegetation_density": 52.0}},
            "structure_attributes": {
                "area": {"sqft": 2200.0},
                "density_context": {"index": 66.0},
                "estimated_age_proxy": {"proxy_year": 1998.0},
                "shape_complexity": {"index": 30.0},
                "year_built": 1996,
                "building_area_sqft": 2250.0,
                "land_use_class": "single_family_residential",
                "roof_material_public_record": "class a",
            },
            "parcel_resolution": {"status": "matched", "confidence": 94.0},
            "footprint_resolution": {"match_status": "matched", "confidence_score": 0.95},
            "property_linkage": {"anchor_confidence": 93.0, "mismatch_flags": []},
            "selected_structure_id": "home-1",
        },
        0.08,
        0.10,
    )
    assert verified["level"] == "verified_property_specific"

    strong = _score_for(
        {
            "footprint_used": True,
            "fallback_mode": "footprint",
            "naip_feature_source": "prepared_region_naip",
            "ring_metrics": {"ring_0_5_ft": {"vegetation_density": 46.0}, "ring_5_30_ft": {"vegetation_density": 49.0}},
            "structure_attributes": {
                "area": {"sqft": 2000.0},
                "density_context": {"index": 58.0},
                "estimated_age_proxy": {"proxy_year": 1990.0},
                "shape_complexity": {"index": 26.0},
                "year_built": 1989,
                "building_area_sqft": 2010.0,
                "land_use_class": "single_family_residential",
            },
            "parcel_resolution": {"status": "matched", "confidence": 86.0},
            "footprint_resolution": {"match_status": "matched", "confidence_score": 0.85},
            "property_linkage": {"anchor_confidence": 84.0, "mismatch_flags": []},
        },
        0.16,
        0.18,
    )
    assert strong["level"] in {"strong_property_specific", "verified_property_specific"}

    address = _score_for(
        {
            "footprint_used": False,
            "fallback_mode": "point_based",
            "naip_feature_source": "regional_fallback",
            "ring_metrics": {"ring_5_30_ft": {"vegetation_density": 42.0}},
            "structure_attributes": {
                "area": {"sqft": None},
                "density_context": {"index": 44.0},
                "estimated_age_proxy": {"proxy_year": 1985.0},
                "shape_complexity": {"index": None},
            },
            "parcel_resolution": {"status": "matched", "confidence": 70.0},
            "footprint_resolution": {"match_status": "ambiguous", "confidence_score": 0.45},
            "property_linkage": {"anchor_confidence": 76.0, "mismatch_flags": ["multiple_footprints_on_parcel"]},
        },
        0.35,
        0.40,
    )
    assert address["level"] in {"address_level", "regional_estimate_with_anchor"}

    regional = _score_for(
        {
            "footprint_used": False,
            "fallback_mode": "point_based",
            "naip_feature_source": "point_proxy",
            "ring_metrics": {},
            "structure_attributes": {
                "area": {"sqft": None},
                "density_context": {"index": None},
                "estimated_age_proxy": None,
                "shape_complexity": {"index": None},
            },
            "parcel_resolution": {"status": "matched", "confidence": 58.0},
            "footprint_resolution": {"match_status": "none", "confidence_score": 0.0},
            "property_linkage": {"anchor_confidence": 68.0, "mismatch_flags": ["no_confident_structure_match"]},
        },
        0.55,
        0.62,
    )
    assert regional["level"] in {"regional_estimate_with_anchor", "insufficient_property_identification"}

    insufficient = _score_for(
        {
            "footprint_used": False,
            "fallback_mode": "point_based",
            "naip_feature_source": "point_proxy",
            "ring_metrics": {},
            "structure_attributes": {
                "area": {"sqft": None},
                "density_context": {"index": None},
                "estimated_age_proxy": None,
                "shape_complexity": {"index": None},
            },
            "parcel_resolution": {"status": "not_found", "confidence": 8.0},
            "footprint_resolution": {"match_status": "none", "confidence_score": 0.0},
            "property_linkage": {
                "anchor_confidence": 28.0,
                "mismatch_flags": ["no_confident_structure_match", "footprint_parcel_misalignment"],
            },
        },
        0.85,
        0.92,
    )
    assert insufficient["level"] == "insufficient_property_identification"
