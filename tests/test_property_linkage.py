from __future__ import annotations

from backend.property_linkage import build_property_linkage_summary


def test_property_linkage_clean_match_prefers_containing_parcel_and_confident_structure() -> None:
    geocode_meta = {
        "geocode_status": "accepted",
        "geocode_precision": "rooftop",
        "confidence_score": 0.95,
    }
    context = {
        "property_anchor_source": "authoritative_address_point",
        "property_anchor_quality_score": 0.94,
        "parcel_resolution": {
            "status": "matched",
            "confidence": 95.0,
            "source": "county_parcels_2025",
            "lookup_method": "contains_point",
            "candidates_considered": 1,
        },
        "parcel_lookup_method": "contains_point",
        "footprint_resolution": {
            "match_status": "matched",
            "match_method": "parcel_intersection",
            "confidence_score": 0.93,
            "selected_source": "building_footprints",
            "candidates_considered": 1,
            "match_distance_m": 0.7,
        },
        "selected_structure_id": "structure-1",
        "structure_match_status": "matched",
        "structure_match_method": "parcel_intersection",
        "candidate_structure_count": 1,
    }

    summary = build_property_linkage_summary(
        geocode_meta=geocode_meta,
        property_level_context=context,
    )
    assert summary["anchor_status"] == "resolved"
    assert summary["parcel_status"] == "matched"
    assert summary["footprint_status"] == "matched"
    assert summary["selected_structure_id"] == "structure-1"
    assert summary["overall_property_confidence"] >= 85.0
    assert summary["mismatch_flags"] == []


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
    assert "multiple_footprints_on_parcel" in summary["mismatch_flags"]
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
    assert "footprint_parcel_misalignment" in summary["mismatch_flags"]
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
    assert "footprint_without_parcel_match" in summary["mismatch_flags"]
    assert any("parcel match is missing" in note.lower() for note in summary["selection_notes"])


def test_property_linkage_no_confident_match_when_footprint_is_ambiguous() -> None:
    geocode_meta = {
        "geocode_status": "accepted",
        "geocode_precision": "interpolated",
        "confidence_score": 0.63,
    }
    context = {
        "property_anchor_source": "interpolated_geocode",
        "property_anchor_quality_score": 0.58,
        "parcel_resolution": {
            "status": "multiple_candidates",
            "confidence": 59.0,
            "source": "open_parcel",
            "lookup_method": "multiple_candidates",
            "candidates_considered": 3,
        },
        "footprint_resolution": {
            "match_status": "ambiguous",
            "match_method": "nearest_building_fallback",
            "confidence_score": 0.41,
            "candidates_considered": 4,
        },
        "candidate_structure_count": 4,
        "selected_structure_id": None,
    }
    summary = build_property_linkage_summary(
        geocode_meta=geocode_meta,
        property_level_context=context,
    )
    assert summary["anchor_status"] == "approximate"
    assert summary["parcel_status"] == "multiple_candidates"
    assert summary["footprint_status"] == "ambiguous"
    assert "no_confident_structure_match" in summary["mismatch_flags"]
    assert summary["overall_property_confidence"] <= 62.0
