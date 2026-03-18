from __future__ import annotations

from backend.main import _build_feature_coverage_preflight
from backend.models import LayerCoverageSummary
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
