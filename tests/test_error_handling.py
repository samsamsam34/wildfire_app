from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.models import UnderwritingRuleset


def _ensure_test_routes() -> None:
    route_paths = {getattr(route, "path", None) for route in app_main.app.router.routes}

    if "/_test/error/unhandled" not in route_paths:
        @app_main.app.get("/_test/error/unhandled")
        def _test_unhandled_error() -> dict[str, Any]:
            raise RuntimeError("synthetic boom")

    if "/_test/error/http404" not in route_paths:
        @app_main.app.get("/_test/error/http404")
        def _test_http_404() -> dict[str, Any]:
            raise HTTPException(status_code=404, detail="not found test path")

    if "/_test/error/http422" not in route_paths:
        @app_main.app.get("/_test/error/http422")
        def _test_http_422() -> dict[str, Any]:
            raise HTTPException(status_code=422, detail="validation failed")


@pytest.fixture()
def non_raising_client(monkeypatch: pytest.MonkeyPatch):
    _ensure_test_routes()
    app_main.limiter.limiter.storage.reset()
    monkeypatch.setenv("WF_RATE_LIMIT_ASSESS", "1000/minute")
    monkeypatch.setenv("WF_RATE_LIMIT_ASSESS_DAILY", "5000/day")
    monkeypatch.setenv("WF_RATE_LIMIT_SIMULATE", "1000/minute")
    with TestClient(app_main.app, raise_server_exceptions=False) as client:
        yield client


def test_unhandled_exception_returns_api_error_without_trace(non_raising_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WF_DEBUG_ERRORS", raising=False)
    response = non_raising_client.get("/_test/error/unhandled")
    assert response.status_code == 500
    body = response.json()
    assert body.get("error") is True
    assert body.get("error_class") == "internal_error"
    assert "unexpected error" in str(body.get("message", "")).lower()
    assert "detail" not in body
    assert "Traceback" not in response.text


def test_unhandled_exception_includes_detail_when_debug_enabled(non_raising_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WF_DEBUG_ERRORS", "true")
    response = non_raising_client.get("/_test/error/unhandled")
    assert response.status_code == 500
    body = response.json()
    assert body.get("error_class") == "internal_error"
    assert "detail" in body
    assert "synthetic boom" in str(body.get("detail"))


def test_unhandled_exception_omits_detail_when_debug_disabled(non_raising_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WF_DEBUG_ERRORS", "false")
    response = non_raising_client.get("/_test/error/unhandled")
    assert response.status_code == 500
    body = response.json()
    assert body.get("error_class") == "internal_error"
    assert "detail" not in body


def test_http_exception_404_uses_not_found_error_class(non_raising_client: TestClient):
    response = non_raising_client.get("/_test/error/http404")
    assert response.status_code == 404
    body = response.json()
    assert body.get("error") is True
    assert body.get("error_class") == "not_found"


def test_http_exception_422_uses_validation_error_class(non_raising_client: TestClient):
    response = non_raising_client.get("/_test/error/http422")
    assert response.status_code == 422
    body = response.json()
    assert body.get("error") is True
    assert body.get("error_class") == "validation_error"


def test_outside_us_address_uses_outside_us_error_class(monkeypatch: pytest.MonkeyPatch):
    auth.API_KEYS = set()
    app_main.limiter.limiter.storage.reset()

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
        raw_input="Paris",
        normalized_address="Paris",
        geocode_status="accepted",
        candidate_count=1,
        selected_candidate=None,
        confidence_score=0.95,
        latitude=48.8566,
        longitude=2.3522,
        geocode_source="test-geocoder",
        geocode_meta={
            "geocode_status": "accepted",
            "normalized_address": "Paris",
            "resolved_latitude": 48.8566,
            "resolved_longitude": 2.3522,
        },
    )
    coverage_resolution = app_main.RegionCoverageResolution(
        coverage_available=False,
        resolved_region_id=None,
        reason="outside_prepared_regions",
        diagnostics=[],
        coverage={
            "coverage_available": False,
            "resolved_region_id": None,
            "reason": "outside_prepared_regions",
            "diagnostics": [],
        },
    )

    monkeypatch.setattr(
        app_main,
        "_resolve_location_for_route",
        lambda **kwargs: (geocode_resolution, coverage_resolution, 48.8566, 2.3522),
    )

    with TestClient(app_main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/risk/assess",
            json={
                "address": "Paris, France",
                "attributes": {},
                "confirmed_fields": [],
                "audience": "homeowner",
                "tags": [],
            },
        )

    assert response.status_code == 422
    body = response.json()
    assert body.get("error") is True
    assert body.get("error_class") == "outside_us_coverage"
    assert "us addresses" in str(body.get("message", "")).lower()


def test_cors_allows_default_localhost_origin(non_raising_client: TestClient):
    response = non_raising_client.options(
        "/risk/assess",
        headers={
            "Origin": "http://localhost:4173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-API-Key",
        },
    )
    assert response.status_code in {200, 204}
    assert response.headers.get("access-control-allow-origin") == "http://localhost:4173"


def test_cors_blocks_unapproved_origin(non_raising_client: TestClient):
    response = non_raising_client.options(
        "/risk/assess",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-API-Key",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "https://evil.com"
