from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend import auth
from backend import main as app_main
from backend.geocoding import GeocodingError


client = TestClient(app_main.app)


def _covered_lookup(lat: float, lon: float, regions_root: str | None = None) -> dict[str, Any]:
    if abs(lat - 48.4772) < 0.01 and abs(lon + 120.1864) < 0.01:
        return {
            "covered": True,
            "region_id": "winthrop_pilot",
            "display_name": "Winthrop Pilot",
            "diagnostics": [],
            "containing_region_ids": ["winthrop_pilot", "winthrop_large"],
        }
    return {
        "covered": False,
        "diagnostics": ["No prepared region bounds contain point."],
        "nearest_region_id": "winthrop_pilot",
        "region_distance_to_boundary_m": 640.0,
    }


def test_primary_geocoder_with_extra_tokens_still_auto_resolves(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (48.4772, -120.1864, "test-primary"),
    )
    monkeypatch.setattr(
        app_main.geocoder,
        "last_result",
        {
            "geocode_status": "accepted",
            "provider": "test-primary",
            "matched_address": "6 Pineview Road, Winthrop, Okanogan County, Washington 98862, United States",
            "confidence_score": 0.44,
            "candidate_count": 1,
            "geocode_precision": "interpolated",
            "raw_response_preview": {"candidate_count": 1},
        },
    )
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _covered_lookup)

    response = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert response.status_code == 200
    body = response.json()
    assert body["coverage_available"] is True
    assert body["resolution_method"] == "primary_geocoder"
    assert body["final_acceptance_decision"] is True
    top = body["resolver_candidates"][0]
    assert top["source_stage"] == "primary_geocoder"
    assert top["confidence_tier"] in {"high", "medium"}
    assert top["auto_eligible"] is True


def test_county_address_points_override_wrong_geocoder_candidate(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (48.1200, -121.9000, "test-primary"),
    )
    monkeypatch.setattr(
        app_main.geocoder,
        "last_result",
        {
            "geocode_status": "accepted",
            "provider": "test-primary",
            "matched_address": "6 Pineview Rd, Somewhere Else, WA 98862",
            "confidence_score": 0.15,
            "candidate_count": 1,
            "geocode_precision": "interpolated",
            "raw_response_preview": {"candidate_count": 1},
        },
    )
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _address: {
            "matched": True,
            "confidence": "high",
            "match_method": "exact_normalized_address",
            "candidate_count": 1,
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "best_match": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
                "source": "okanogan_county_addressing",
                "source_type": "county_address_dataset",
                "match_type": "exact_normalized_address",
                "confidence_tier": "high",
                "match_score": 0.99,
                "feature_properties": {"address_id": "ok-101"},
            },
            "top_candidates": [],
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_statewide_parcel_coordinates",
        lambda _address: {"matched": False, "candidate_count": 0, "top_candidates": [], "failure_reason": "no_match"},
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_fallback_coordinates",
        lambda _address: {"matched": False, "candidate_count": 0, "top_candidates": [], "failure_reason": "no_match"},
    )
    monkeypatch.setattr(app_main, "lookup_region_for_point", _covered_lookup)

    response = client.post("/risk/geocode-debug", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolution_status"] == "resolved_high_confidence"
    assert body["resolution_method"] == "county_address_points"
    assert body["resolved_latitude"] == 48.4772
    assert body["resolved_longitude"] == -120.1864
    assert body["region_resolution"]["coverage_available"] is True


def test_authoritative_in_region_medium_candidate_beats_outside_geocoder(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (48.1200, -121.9000, "test-primary"),
    )
    monkeypatch.setattr(
        app_main.geocoder,
        "last_result",
        {
            "geocode_status": "accepted",
            "provider": "test-primary",
            "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
            "geocode_trust_tier": "high",
            "confidence_score": 0.62,
            "candidate_count": 1,
            "geocode_precision": "interpolated",
            "raw_response_preview": {"candidate_count": 1},
        },
    )
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _address: {
            "matched": True,
            "confidence": "medium",
            "match_method": "address_component_match",
            "candidate_count": 1,
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "best_match": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "matched_address": "6 Pineview Road, Winthrop, WA 98862",
                "source": "okanogan_county_addressing",
                "source_type": "county_address_dataset",
                "match_type": "address_component_match",
                "confidence_tier": "medium",
                "match_score": 0.90,
                "feature_properties": {"address_id": "ok-202"},
            },
            "top_candidates": [],
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_statewide_parcel_coordinates",
        lambda _address: {"matched": False, "candidate_count": 0, "top_candidates": [], "failure_reason": "no_match"},
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_fallback_coordinates",
        lambda _address: {"matched": False, "candidate_count": 0, "top_candidates": [], "failure_reason": "no_match"},
    )
    monkeypatch.setattr(app_main, "lookup_region_for_point", _covered_lookup)

    response = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert response.status_code == 200
    body = response.json()
    assert body["coverage_available"] is True
    assert body["resolved_region_id"] == "winthrop_pilot"
    assert body["resolution_method"] == "county_address_points"
    final_candidate = body.get("final_candidate_selected") or {}
    assert final_candidate.get("source_stage") == "county_address_points"
    assert final_candidate.get("in_region_result") == "inside_prepared_region"
    rejected = [c for c in (body.get("resolver_candidates") or []) if c.get("source_stage") == "primary_geocoder"]
    assert rejected
    assert rejected[0].get("rejection_reason") in {
        "not_selected_higher_rank_candidate_available",
        "in_region_preference_override",
    }


def test_secondary_geocoder_selected_when_local_sources_missing(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )

    class _Secondary:
        provider_name = "Secondary Test Geocoder"
        last_result = {
            "geocode_status": "accepted",
            "provider": "Secondary Test Geocoder",
            "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
            "confidence_score": 0.26,
            "candidate_count": 1,
            "geocode_precision": "parcel_or_address_point",
            "raw_response_preview": {"candidate_count": 1},
        }

        def geocode(self, _address: str):
            return (48.4772, -120.1864, "Secondary Test Geocoder")

    monkeypatch.setattr(app_main, "secondary_geocoder", _Secondary())
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "true")
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_SEARCH_URL", "https://example.test/geocode")
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _addr: {"matched": False, "candidate_count": 0, "top_candidates": []})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0, "top_candidates": []})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0, "top_candidates": []})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _covered_lookup)

    response = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolution_method"] == "secondary_geocoder"
    assert body["coverage_available"] is True


def test_conflicting_high_confidence_candidates_prefer_authoritative_source(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setenv("WF_RESOLVER_CONFLICT_DISTANCE_M", "200")
    monkeypatch.setenv("WF_RESOLVER_CONFLICT_SCORE_MARGIN", "500")
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")

    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (48.4772, -120.1864, "test-primary"))
    monkeypatch.setattr(
        app_main.geocoder,
        "last_result",
        {
            "geocode_status": "accepted",
            "provider": "test-primary",
            "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
            "confidence_score": 0.4,
            "candidate_count": 1,
            "geocode_precision": "parcel_or_address_point",
            "raw_response_preview": {"candidate_count": 1},
        },
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _address: {
            "matched": True,
            "confidence": "high",
            "match_method": "exact_normalized_address",
            "candidate_count": 1,
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "best_match": {
                "latitude": 48.6000,
                "longitude": -120.1000,
                "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
                "source": "okanogan_county_addressing",
                "source_type": "county_address_dataset",
                "match_type": "exact_normalized_address",
                "confidence_tier": "high",
                "match_score": 0.98,
            },
            "top_candidates": [],
        },
    )
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "winthrop_large",
            "display_name": "Winthrop Large",
            "diagnostics": [],
        },
    )

    response = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolution_method"] == "county_address_points"
    assert body["coverage_available"] is True
    assert body["needs_user_confirmation"] is False


def test_user_selected_point_can_resolve_when_no_safe_candidate(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="Unknown",
                normalized_address="Unknown",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _covered_lookup)

    result = app_main._resolve_trusted_geocode(
        address_input="Unresolved Address, Winthrop, WA",
        purpose="assessment",
        route_name="test",
        property_anchor_point={"latitude": 48.4772, "longitude": -120.1864},
    )
    assert result.geocode_status == "accepted"
    assert result.geocode_source == "user_selected_point"
    assert result.geocode_meta.get("resolution_method") == "user_selected_point"


def test_no_match_returns_unresolved_no_safe_candidate(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="999 Invalid", 
                normalized_address="999 Invalid",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})

    response = client.post("/regions/coverage-check", json={"address": "999 Invalid"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["resolution_status"] == "unresolved"
    assert detail["rejection_category"] == "no_geocode_candidates"


def test_debug_and_coverage_routes_share_same_final_coordinates(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="6 Pineview Rd, Winthrop, WA 98862",
                normalized_address="6 Pineview Rd, Winthrop, WA 98862",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _address: {
            "matched": True,
            "confidence": "high",
            "match_method": "exact_normalized_address",
            "candidate_count": 1,
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "best_match": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
                "source": "okanogan_county_addressing",
                "source_type": "county_address_dataset",
                "match_type": "exact_normalized_address",
                "confidence_tier": "high",
                "match_score": 0.99,
            },
            "top_candidates": [],
        },
    )
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _covered_lookup)

    coverage = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    debug = client.post("/risk/geocode-debug", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert coverage.status_code == 200
    assert debug.status_code == 200
    cov_body = coverage.json()
    dbg_body = debug.json()
    final_cov = cov_body["final_coordinates_used"]
    final_dbg = dbg_body["final_coordinates_used"]
    assert final_cov == final_dbg
    assert cov_body["resolution_method"] == dbg_body["resolution_method"]
    assert cov_body["final_candidate_selected"]["latitude"] == final_cov["latitude"]
    assert cov_body["final_candidate_selected"]["longitude"] == final_cov["longitude"]


def test_batch_of_valid_addresses_not_all_forced_to_manual_confirmation(monkeypatch):
    auth.API_KEYS = set()
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")
    address_map = {
        "6 Pineview Rd, Winthrop, WA 98862": (48.4772, -120.1864, "6 Pineview Rd, Winthrop, WA 98862, USA"),
        "500 108th Ave NE, Bellevue, WA 98004": (47.6101, -122.2015, "500 108th Ave NE, Bellevue, WA 98004, USA"),
        "100 Main St, Bozeman, MT 59715": (45.6792, -111.0376, "100 Main St, Bozeman, MT 59715, USA"),
    }

    class _DynamicGeocoder:
        provider_name = "test-primary"

        def __init__(self) -> None:
            self.last_result = {}

        def geocode(self, address: str):
            lat, lon, matched = address_map[address]
            self.last_result = {
                "geocode_status": "accepted",
                "provider": "test-primary",
                "matched_address": matched,
                "confidence_score": 0.38,
                "candidate_count": 1,
                "geocode_precision": "interpolated",
                "raw_response_preview": {"candidate_count": 1},
            }
            return lat, lon, "test-primary"

    monkeypatch.setattr(app_main, "geocoder", _DynamicGeocoder())
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _addr: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": True,
            "region_id": "test_region",
            "display_name": "Test Region",
            "diagnostics": [],
            "containing_region_ids": ["test_region"],
        },
    )

    outcomes = []
    for addr in address_map:
        res = client.post("/regions/coverage-check", json={"address": addr})
        assert res.status_code == 200
        body = res.json()
        outcomes.append(bool(body.get("final_acceptance_decision")))
        assert body.get("error_class") == "ready_for_assessment"
    assert any(outcomes) is True
