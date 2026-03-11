from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend import auth
from backend import main as app_main
from backend.geocoding import GeocodingError


client = TestClient(app_main.app)


def _coverage_lookup(lat: float, lon: float, regions_root: str | None = None) -> dict[str, Any]:
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
        "region_distance_to_boundary_m": 812.0,
    }


def _set_primary_geocoder_candidate(monkeypatch, *, lat: float, lon: float, address: str) -> None:
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (lat, lon, "test-primary"))
    monkeypatch.setattr(
        app_main.geocoder,
        "last_result",
        {
            "geocode_status": "accepted",
            "provider": "test-primary",
            "matched_address": address,
            "confidence_score": 0.42,
            "candidate_count": 1,
            "geocode_precision": "parcel_or_address_point",
            "raw_response_preview": {"candidate_count": 1},
        },
    )


def _set_primary_geocoder_no_match(monkeypatch) -> None:
    monkeypatch.setattr(
        app_main.geocoder,
        "geocode",
        lambda _addr: (_ for _ in ()).throw(
            GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address="unknown",
                normalized_address="unknown",
                rejection_reason="provider returned no candidates",
            )
        ),
    )
    monkeypatch.setattr(app_main.geocoder, "last_result", {})


def test_valid_in_region_address_resolves_ready_for_assessment(monkeypatch) -> None:
    auth.API_KEYS = set()
    _set_primary_geocoder_candidate(
        monkeypatch,
        lat=48.4772,
        lon=-120.1864,
        address="6 Pineview Rd, Winthrop, WA 98862",
    )
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _coverage_lookup)
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")

    resp = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["coverage_available"] is True
    assert body["resolved_region_id"] == "winthrop_pilot"
    assert body["error_class"] == "ready_for_assessment"
    assert body["address_exists"] is True


def test_valid_address_outside_prepared_region_returns_outside_prepared_region(monkeypatch) -> None:
    auth.API_KEYS = set()
    _set_primary_geocoder_candidate(
        monkeypatch,
        lat=47.6101,
        lon=-122.2015,
        address="500 108th Ave NE, Bellevue, WA 98004",
    )
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _coverage_lookup)
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")

    resp = client.post("/regions/coverage-check", json={"address": "500 108th Ave NE, Bellevue, WA 98004"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["coverage_available"] is False
    assert body["error_class"] == "outside_prepared_region"
    assert body["address_exists"] is True
    assert body["final_coordinates_used"] is not None
    assert body["final_status"] == "outside_prepared_region"
    assert body["final_candidate_selected"] is not None


def test_geocoder_miss_can_resolve_via_statewide_parcel_dataset(monkeypatch) -> None:
    auth.API_KEYS = set()
    _set_primary_geocoder_no_match(monkeypatch)
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(
        app_main,
        "_resolve_statewide_parcel_coordinates",
        lambda _a: {
            "matched": True,
            "confidence": "high",
            "candidate_count": 1,
            "match_method": "exact_normalized_address",
            "normalized_address": "6 pineview rd winthrop wa 98862",
            "best_match": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "matched_address": "6 Pineview Rd, Winthrop, WA 98862",
                "source_type": "statewide_parcel_dataset",
                "match_type": "exact_normalized_address",
                "confidence_tier": "high",
                "match_score": 0.98,
                "feature_properties": {"parcel_id": "wa-parcel-1001"},
            },
        },
    )
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _coverage_lookup)
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")

    resp = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolution_method"] == "statewide_parcel_lookup"
    assert body["coverage_available"] is True
    assert body["error_class"] == "ready_for_assessment"


def test_address_exists_but_coordinates_not_safe_returns_address_unresolved(monkeypatch) -> None:
    auth.API_KEYS = set()
    _set_primary_geocoder_no_match(monkeypatch)
    monkeypatch.setattr(
        app_main,
        "_resolve_local_authoritative_coordinates",
        lambda _a: {
            "matched": False,
            "candidate_count": 1,
            "best_candidate": {
                "latitude": 48.4772,
                "longitude": -120.1864,
                "matched_address": "Pineview Rd, Winthrop, WA",
                "source": "okanogan_county_addressing",
                "source_type": "county_address_dataset",
                "match_type": "street_only_match",
                "confidence_tier": "low",
                "match_score": 0.72,
                "feature_properties": {"address_id": "ok-street-only"},
            },
            "top_candidates": [],
            "failure_reason": "street_only_match",
        },
    )
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _coverage_lookup)
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")

    resp = client.post("/regions/coverage-check", json={"address": "6 Pineview Rd, Winthrop, WA 98862"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error_class"] == "address_unresolved"
    assert detail["resolution_status"] == "candidates_found_but_not_safe_enough"
    assert detail["address_exists"] is True
    assert detail["final_coordinates_used"] is None


def test_invalid_address_returns_address_not_found_not_unsupported_location(monkeypatch) -> None:
    auth.API_KEYS = set()
    _set_primary_geocoder_no_match(monkeypatch)
    monkeypatch.setattr(app_main, "_resolve_local_authoritative_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_statewide_parcel_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "_resolve_local_fallback_coordinates", lambda _a: {"matched": False, "candidate_count": 0})
    monkeypatch.setattr(app_main, "lookup_region_for_point", _coverage_lookup)
    monkeypatch.setenv("WF_GEOCODE_SECONDARY_ENABLED", "false")

    resp = client.post("/regions/coverage-check", json={"address": "not a real address 123456"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error_class"] == "address_not_found"
    assert detail["address_exists"] is False
    assert detail.get("reason") != "no_prepared_region_for_location"
