from __future__ import annotations

import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.models import UnderwritingRuleset


@pytest.fixture()
def rate_limit_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    auth.API_KEYS = {}

    monkeypatch.setenv("WF_RATE_LIMIT_ASSESS", "10/minute")
    monkeypatch.setenv("WF_RATE_LIMIT_ASSESS_DAILY", "100/day")
    monkeypatch.setenv("WF_RATE_LIMIT_SIMULATE", "20/minute")
    monkeypatch.delenv("WF_RATE_LIMIT_BYPASS_KEYS", raising=False)

    app_main.limiter.limiter.storage.reset()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "rate_limit.db")))

    ruleset = UnderwritingRuleset(
        ruleset_id="default",
        ruleset_name="Default",
        ruleset_version="1.0.0",
        ruleset_description="",
        config={},
    )

    monkeypatch.setattr(
        app_main,
        "_resolve_write_context",
        lambda ctx, request_org, ruleset_id, role_detail: ("default_org", ruleset),
    )

    geocode_resolution = app_main.GeocodeResolution(
        raw_input="123 Main St",
        normalized_address="123 Main St",
        geocode_status="accepted",
        candidate_count=1,
        selected_candidate=None,
        confidence_score=0.95,
        latitude=39.7392,
        longitude=-104.9903,
        geocode_source="test-geocoder",
        geocode_meta={
            "geocode_status": "accepted",
            "normalized_address": "123 Main St",
            "resolved_latitude": 39.7392,
            "resolved_longitude": -104.9903,
        },
    )
    coverage_resolution = app_main.RegionCoverageResolution(
        coverage_available=True,
        resolved_region_id="test_region",
        reason="inside_prepared_region",
        diagnostics=[],
        coverage={
            "coverage_available": True,
            "resolved_region_id": "test_region",
            "reason": "inside_prepared_region",
            "diagnostics": [],
        },
    )

    monkeypatch.setattr(
        app_main,
        "_resolve_location_for_route",
        lambda **kwargs: (geocode_resolution, coverage_resolution, 39.7392, -104.9903),
    )

    def _fail_assessment(*args, **kwargs):
        raise HTTPException(
            status_code=400,
            detail={
                "error_class": "assessment_failed",
                "message": "Synthetic assessment failure for limiter test.",
            },
        )

    monkeypatch.setattr(app_main, "_compute_assessment", _fail_assessment)

    with TestClient(app_main.app) as client:
        yield client


def _assess_payload() -> dict:
    return {
        "address": "123 Main St, Denver, CO 80202",
        "attributes": {},
        "confirmed_fields": [],
        "audience": "homeowner",
        "tags": [],
    }


def test_assess_rate_limited_after_ten_requests(rate_limit_client: TestClient):
    responses = [
        rate_limit_client.post("/risk/assess", json=_assess_payload())
        for _ in range(11)
    ]
    statuses = [resp.status_code for resp in responses]
    assert all(code != 429 for code in statuses[:10])
    assert statuses[10] == 429, responses[10].json()


def test_rate_limit_429_uses_api_error_envelope(rate_limit_client: TestClient):
    for _ in range(10):
        rate_limit_client.post("/risk/assess", json=_assess_payload())

    response = rate_limit_client.post("/risk/assess", json=_assess_payload())
    assert response.status_code == 429
    body = response.json()
    assert body.get("error") is True
    assert body.get("error_class") == "rate_limit_exceeded"
    assert isinstance(body.get("message"), str) and body.get("message")


def test_rate_limit_bypass_key_skips_limits(rate_limit_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WF_RATE_LIMIT_BYPASS_KEYS", "internal-key")

    statuses = [
        rate_limit_client.post(
            "/risk/assess",
            json=_assess_payload(),
            headers={"X-API-Key": "internal-key"},
        ).status_code
        for _ in range(12)
    ]

    assert all(code != 429 for code in statuses)


def test_rate_limit_resets_after_window_expiry(rate_limit_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WF_RATE_LIMIT_ASSESS", "2/second")
    app_main.limiter.limiter.storage.reset()

    first = rate_limit_client.post("/risk/assess", json=_assess_payload())
    second = rate_limit_client.post("/risk/assess", json=_assess_payload())
    third = rate_limit_client.post("/risk/assess", json=_assess_payload())

    assert first.status_code != 429
    assert second.status_code != 429
    assert third.status_code == 429

    time.sleep(1.1)
    after_reset = rate_limit_client.post("/risk/assess", json=_assess_payload())
    assert after_reset.status_code != 429
