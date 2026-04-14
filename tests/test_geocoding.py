from __future__ import annotations

import json
import urllib.parse
from typing import Any

import pytest

from backend.geocoding import Geocoder, GeocodingError, geocode_from_address_points


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


def test_geocoder_retries_with_house_number_stripped_variant(monkeypatch):
    first_payload: list[dict[str, Any]] = []
    second_payload = [
        {
            "display_name": "Pineview Rd, Winthrop, WA 98862, USA",
            "lat": "48.4772",
            "lon": "-120.1864",
            "importance": 0.04,
            "class": "highway",
            "type": "residential",
            "address": {
                "road": "Pineview Rd",
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
        if q.startswith("6 "):
            return _FakeHTTPResponse(first_payload)
        if q.startswith("Pineview Rd"):
            return _FakeHTTPResponse(second_payload)
        return _FakeHTTPResponse(first_payload)

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.0")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", _fake_urlopen)
    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("6 Pineview Rd, Winthrop, WA 98862")
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(48.4772)
    preview = result.raw_response_preview or {}
    attempts = preview.get("query_attempts") or []
    assert attempts
    assert any((attempt or {}).get("query", "").startswith("6 Pineview Rd") for attempt in attempts)
    assert any((attempt or {}).get("query", "").startswith("Pineview Rd") for attempt in attempts)


def test_geocoder_strips_suite_and_hash_noise(monkeypatch):
    empty_payload: list[dict[str, Any]] = []
    clean_payload = [
        {
            "display_name": "12 Twin Lakes Rd, Winthrop, WA 98862, USA",
            "lat": "48.4782",
            "lon": "-120.1870",
            "importance": 0.07,
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
        q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0].lower()
        if "ste 200" in q or "#12" in q:
            return _FakeHTTPResponse(empty_payload)
        if "twin lakes rd" in q:
            return _FakeHTTPResponse(clean_payload)
        return _FakeHTTPResponse(empty_payload)

    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", _fake_urlopen)
    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("12 Twin Lakes Rd Ste 200 #12, Winthrop, WA 98862")
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(48.4782)
    preview = result.raw_response_preview or {}
    attempts = preview.get("query_attempts") or []
    assert any("ste 200" in str((attempt or {}).get("query", "")).lower() for attempt in attempts)
    assert any("twin lakes rd, winthrop, wa 98862" in str((attempt or {}).get("query", "")).lower() for attempt in attempts)


def test_geocoder_prefers_better_candidate_from_later_query_variant(monkeypatch):
    weak_payload = [
        {
            "display_name": "12 Twin Lakes Rd, Other Town, WA 98862, USA",
            "lat": "48.3100",
            "lon": "-120.4300",
            "importance": 0.35,
            "class": "place",
            "type": "residential",
            "address": {
                "house_number": "12",
                "road": "Twin Lakes Rd",
                "town": "Other Town",
                "state": "Washington",
                "postcode": "98862",
                "country_code": "us",
            },
        }
    ]
    strong_payload = [
        {
            "display_name": "12 Twin Lakes Rd, Winthrop, WA 98862, USA",
            "lat": "48.4782",
            "lon": "-120.1870",
            "importance": 0.05,
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
        q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0].lower()
        if "apt 3" in q:
            return _FakeHTTPResponse(weak_payload)
        if "twin lakes rd, winthrop, wa 98862" in q:
            return _FakeHTTPResponse(strong_payload)
        return _FakeHTTPResponse([])

    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", _fake_urlopen)
    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("12 Twin Lakes Rd Apt 3, Winthrop, WA 98862")
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(48.4782)
    assert result.matched_address == "12 Twin Lakes Rd, Winthrop, WA 98862, USA"
    preview = result.raw_response_preview or {}
    assert str(preview.get("selected_query") or "").lower() != "12 twin lakes rd apt 3, winthrop, wa 98862"


def test_geocoder_labels_boundary_locality_as_approximate(monkeypatch):
    payload = [
        {
            "display_name": "Winthrop, WA, USA",
            "lat": "48.4785",
            "lon": "-120.1865",
            "importance": 0.63,
            "class": "boundary",
            "type": "administrative",
            "address": {
                "town": "Winthrop",
                "state": "Washington",
                "postcode": "98862",
                "country_code": "us",
            },
        }
    ]

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.0")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", lambda *args, **kwargs: _FakeHTTPResponse(payload))
    geocoder = Geocoder(user_agent="test-suite")
    result = geocoder.geocode_with_diagnostics("Winthrop, WA 98862")
    assert result.geocode_status == "accepted"
    assert result.geocode_precision == "approximate"


def test_geocoder_rejects_conflicting_house_number_even_if_candidate_is_precise(monkeypatch):
    payload = [
        {
            "display_name": "14 Twin Lakes Rd, Winthrop, WA 98862, USA",
            "lat": "48.4782",
            "lon": "-120.1870",
            "importance": 0.41,
            "class": "building",
            "type": "house",
            "address": {
                "house_number": "14",
                "road": "Twin Lakes Rd",
                "town": "Winthrop",
                "state": "Washington",
                "postcode": "98862",
                "country_code": "us",
            },
        }
    ]

    monkeypatch.setenv("WF_GEOCODE_MIN_IMPORTANCE", "0.0")
    monkeypatch.setattr("backend.geocoding.urllib.request.urlopen", lambda *args, **kwargs: _FakeHTTPResponse(payload))
    geocoder = Geocoder(user_agent="test-suite")
    with pytest.raises(GeocodingError) as exc:
        geocoder.geocode_with_diagnostics("12 Twin Lakes Rd, Winthrop, WA 98862")

    assert exc.value.status == "low_confidence"
    assert "house_number_mismatch" in str(exc.value.rejection_reason or "")
    preview = exc.value.raw_response_preview or {}
    assert preview.get("trust_filter_rule") == "house_number_mismatch"


# ---------------------------------------------------------------------------
# Fix 2: address-point string-match geocoder
# ---------------------------------------------------------------------------

def _write_address_points(path, features):
    import json
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def test_geocode_from_address_points_returns_rooftop_match(tmp_path):
    ap = tmp_path / "address_points.geojson"
    _write_address_points(ap, [
        {
            "type": "Feature",
            "properties": {
                "STREETNUMBER": "1355",
                "STREETNAME": "Pattee Canyon Rd",
                "fulladdress": "1355 Pattee Canyon Rd",
            },
            "geometry": {"type": "Point", "coordinates": [-113.9778, 46.8281]},
        }
    ])
    result = geocode_from_address_points("1355 Pattee Canyon Rd, Missoula, MT", str(ap))
    assert result is not None
    assert result.geocode_precision == "rooftop"
    assert result.source == "parcel_address_point"
    assert abs(result.latitude - 46.8281) < 0.001
    assert abs(result.longitude - (-113.9778)) < 0.001
    assert result.confidence_score is not None and result.confidence_score >= 0.6


def test_geocode_from_address_points_rejects_wrong_number(tmp_path):
    ap = tmp_path / "address_points.geojson"
    _write_address_points(ap, [
        {
            "type": "Feature",
            "properties": {
                "STREETNUMBER": "1400",
                "fulladdress": "1400 Pattee Canyon Rd",
            },
            "geometry": {"type": "Point", "coordinates": [-113.9780, 46.8290]},
        }
    ])
    result = geocode_from_address_points("1355 Pattee Canyon Rd, Missoula, MT", str(ap))
    assert result is None, "Different house number should not match"


def test_geocode_from_address_points_returns_none_for_missing_file(tmp_path):
    result = geocode_from_address_points("1355 Pattee Canyon Rd", str(tmp_path / "nonexistent.geojson"))
    assert result is None


def test_geocode_from_address_points_returns_none_when_no_number_in_query(tmp_path):
    ap = tmp_path / "address_points.geojson"
    _write_address_points(ap, [
        {
            "type": "Feature",
            "properties": {"STREETNUMBER": "100", "fulladdress": "100 Main St"},
            "geometry": {"type": "Point", "coordinates": [-113.0, 46.0]},
        }
    ])
    result = geocode_from_address_points("Main St, Missoula, MT", str(ap))
    assert result is None, "Address without house number should return None"


def test_geocode_from_address_points_uses_fulladdress_number_fallback(tmp_path):
    """When STREETNUMBER field is absent, house number should be parsed from fulladdress."""
    ap = tmp_path / "address_points.geojson"
    _write_address_points(ap, [
        {
            "type": "Feature",
            "properties": {"fulladdress": "1355 Pattee Canyon Rd"},
            "geometry": {"type": "Point", "coordinates": [-113.9778, 46.8281]},
        }
    ])
    result = geocode_from_address_points("1355 Pattee Canyon Rd, Missoula, MT", str(ap))
    assert result is not None
    assert result.geocode_precision == "rooftop"
