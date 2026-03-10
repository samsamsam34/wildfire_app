from __future__ import annotations

import json
import urllib.parse
from typing import Any

import pytest

from backend.geocoding import Geocoder, GeocodingError


class _FakeHTTPResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):  # pragma: no cover - exercised by urlopen context usage
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - exercised by urlopen context usage
        return False


def test_geocoder_accepts_precise_low_importance_candidate_with_override(monkeypatch):
    payload = [
        {
            "display_name": "62910 O B Riley Rd, Bend, OR 97703, USA",
            "lat": "44.0839",
            "lon": "-121.3153",
            "importance": 0.01,
            "class": "place",
            "type": "house",
            "address": {
                "house_number": "62910",
                "road": "O B Riley Rd",
                "city": "Bend",
                "state": "Oregon",
                "postcode": "97703",
                "country_code": "us",
            },
        }
    ]

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.2")
    monkeypatch.setenv("WF_GEOCODE_ALLOW_PRECISE_LOW_IMPORTANCE", "true")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", lambda *args, **kwargs: _FakeHTTPResponse(payload))

    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("62910 O B Riley Rd, Bend, OR 97703")
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(44.0839)
    assert result.longitude == pytest.approx(-121.3153)
    assert (result.raw_response_preview or {}).get("trust_filter_rule") == "low_importance_but_precise_address_override"


def test_geocoder_rejects_low_importance_without_precise_address(monkeypatch):
    payload = [
        {
            "display_name": "Bend, OR, USA",
            "lat": "44.0582",
            "lon": "-121.3153",
            "importance": 0.01,
            "class": "boundary",
            "type": "administrative",
            "address": {
                "city": "Bend",
                "state": "Oregon",
                "country_code": "us",
            },
        }
    ]

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.2")
    monkeypatch.setenv("WF_GEOCODE_ALLOW_PRECISE_LOW_IMPORTANCE", "true")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", lambda *args, **kwargs: _FakeHTTPResponse(payload))

    geocoder = Geocoder(user_agent="test-suite")
    with pytest.raises(GeocodingError) as exc:
        geocoder.geocode_with_diagnostics("Bend Oregon")

    assert exc.value.status == "low_confidence"
    assert "importance=" in (exc.value.rejection_reason or "")
    preview = exc.value.raw_response_preview or {}
    assert preview.get("trust_filter_rule") == "min_importance_threshold"


def test_geocoder_accepts_low_importance_when_postcode_and_street_context_present(monkeypatch):
    payload = [
        {
            "display_name": "NW Portland Ave, Bend, OR 97703, USA",
            "lat": "44.0719",
            "lon": "-121.3157",
            "importance": 0.01,
            "class": "highway",
            "type": "residential",
            "address": {
                "road": "NW Portland Ave",
                "city": "Bend",
                "state": "Oregon",
                "postcode": "97703",
                "country_code": "us",
            },
        }
    ]

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.2")
    monkeypatch.setenv("WF_GEOCODE_ALLOW_PRECISE_LOW_IMPORTANCE", "true")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", lambda *args, **kwargs: _FakeHTTPResponse(payload))

    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("NW Portland Ave, Bend, OR 97703")
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(44.0719)
    assert result.longitude == pytest.approx(-121.3157)
    assert (result.raw_response_preview or {}).get("trust_filter_rule") == "low_importance_but_precise_address_override"


def test_geocoder_retries_with_unit_stripped_query_variant(monkeypatch):
    first_payload: list[dict[str, Any]] = []
    second_payload = [
        {
            "display_name": "12 Twin Lakes Rd, Winthrop, WA 98862, USA",
            "lat": "48.4782",
            "lon": "-120.1870",
            "importance": 0.08,
            "class": "building",
            "type": "house",
            "address": {
                "house_number": "12",
                "road": "Twin Lakes Rd",
                "town": "Winthrop",
                "state": "Washington",
                "postcode": "98862",
                "country_code": "us",
            },
        }
    ]

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        parsed = urllib.parse.urlparse(req.full_url)
        q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
        if "Apt 3" in q or "apt 3" in q:
            return _FakeHTTPResponse(first_payload)
        return _FakeHTTPResponse(second_payload)

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.02")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", _fake_urlopen)
    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("12 Twin Lakes Rd Apt 3, Winthrop, WA 98862")
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(48.4782)
    preview = result.raw_response_preview or {}
    assert preview.get("selected_query")
    assert any((attempt or {}).get("candidate_count") == 0 for attempt in preview.get("query_attempts", []))
