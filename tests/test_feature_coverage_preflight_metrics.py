from __future__ import annotations

from backend.main import _build_feature_coverage_preflight
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
    assert summary.get("level") == "high"


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
    assert summary.get("level") == "low"
    assert str(preflight.get("assessment_specificity_tier")) in {"regional_estimate", "address_level"}
