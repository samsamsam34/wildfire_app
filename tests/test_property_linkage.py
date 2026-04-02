from __future__ import annotations

from backend.property_linkage import build_property_linkage_summary


def test_property_linkage_flags_multiple_structures_on_matched_parcel() -> None:
    geocode_meta = {
        "geocode_status": "accepted",
        "geocode_precision": "rooftop",
        "confidence_score": 0.92,
    }
    context = {
        "parcel_resolution": {
            "status": "matched",
            "confidence": 94.0,
        },
        "footprint_resolution": {
            "match_status": "matched",
            "match_method": "parcel_intersection",
            "confidence_score": 0.91,
            "candidates_considered": 3,
        },
        "structure_match_status": "matched",
        "structure_match_method": "parcel_intersection",
        "candidate_structure_count": 3,
    }

    summary = build_property_linkage_summary(
        geocode_meta=geocode_meta,
        property_level_context=context,
    )
    assert summary["multiple_footprints_on_parcel"] is True
    assert summary["parcel_status"] == "matched"
    assert summary["footprint_status"] == "matched"
    assert summary["overall_property_confidence"] < 92.0


def test_property_linkage_detects_footprint_parcel_misalignment() -> None:
    geocode_meta = {
        "geocode_status": "accepted",
        "geocode_precision": "parcel_or_address_point",
        "confidence_score": 0.88,
    }
    context = {
        "parcel_resolution": {
            "status": "matched",
            "confidence": 90.0,
        },
        "footprint_resolution": {
            "match_status": "matched",
            "match_method": "nearest_building_fallback",
            "confidence_score": 0.79,
            "candidates_considered": 1,
        },
        "structure_match_status": "matched",
        "structure_match_method": "nearest_building_fallback",
        "candidate_structure_count": 1,
    }

    summary = build_property_linkage_summary(
        geocode_meta=geocode_meta,
        property_level_context=context,
    )
    assert summary["footprint_outside_parcel"] is True
    assert summary["overall_property_confidence"] < 80.0
    assert any("align" in note.lower() for note in summary["selection_notes"])


def test_property_linkage_handles_missing_parcel_with_present_footprint() -> None:
    geocode_meta = {
        "geocode_status": "accepted",
        "geocode_precision": "interpolated",
        "confidence_score": 0.64,
    }
    context = {
        "parcel_resolution": {
            "status": "not_found",
            "confidence": 12.0,
        },
        "footprint_resolution": {
            "match_status": "matched",
            "match_method": "point_in_footprint",
            "confidence_score": 0.86,
            "candidates_considered": 1,
        },
        "structure_match_status": "matched",
        "structure_match_method": "point_in_footprint",
        "candidate_structure_count": 1,
    }

    summary = build_property_linkage_summary(
        geocode_meta=geocode_meta,
        property_level_context=context,
    )
    assert summary["parcel_status"] == "not_found"
    assert summary["footprint_status"] == "matched"
    assert summary["overall_property_confidence"] > 50.0
    assert summary["overall_property_confidence"] < summary["footprint_confidence"]
    assert any("parcel match is missing" in note.lower() for note in summary["selection_notes"])

