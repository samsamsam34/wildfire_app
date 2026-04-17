from __future__ import annotations

import json
import urllib.parse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.geocoding import Geocoder, GeocodingError, GeocodeResult, geocode_from_address_points
from backend.geocoding_census import CensusGeocoder
from backend.geocoding_fallback_chain import GeocodeFallbackChain
from backend.geocoding_zip_validator import ZipCodeValidator, ZipValidationResult, haversine_km


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


# ---------------------------------------------------------------------------
# CensusGeocoder tests
# ---------------------------------------------------------------------------

# Minimal Census API response shape for a successful match.
_CENSUS_SUCCESS_PAYLOAD = {
    "result": {
        "input": {"address": {"address": "4600 Silver Hill Rd, Suitland, MD"}},
        "addressMatches": [
            {
                "tigerLine": {"side": "L", "tigerLineId": "76355984"},
                "coordinates": {"x": -76.92743, "y": 38.845985},
                "addressComponents": {
                    "zip": "20746",
                    "streetName": "SILVER HILL",
                    "preType": "",
                    "city": "SUITLAND",
                    "preDirection": "",
                    "suffixDirection": "",
                    "fromAddress": "4500",
                    "state": "MD",
                    "suffixType": "RD",
                    "toAddress": "4798",
                    "suffixQualifier": "",
                    "preQualifier": "",
                },
                "matchedAddress": "4600 SILVER HILL RD, SUITLAND, MD, 20746",
            }
        ],
    }
}

_CENSUS_NO_MATCH_PAYLOAD = {
    "result": {
        "input": {"address": {"address": "1 Nowhere Blvd, Fakeville, XX 00000"}},
        "addressMatches": [],
    }
}


class _FakeCensusHTTPResponse:
    """Minimal urllib response stub for Census API calls."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_census_geocoder_success_returns_correct_precision_and_coordinates(monkeypatch):
    """Valid Census response → parcel_or_address_point precision and correct lat/lon."""
    monkeypatch.setattr(
        "backend.geocoding_census.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeCensusHTTPResponse(_CENSUS_SUCCESS_PAYLOAD),
    )
    geocoder = CensusGeocoder()
    result = geocoder.geocode("4600 Silver Hill Rd, Suitland, MD 20746")

    assert result is not None
    assert result.geocode_status == "accepted"
    assert result.latitude == pytest.approx(38.845985)
    assert result.longitude == pytest.approx(-76.92743)
    assert result.geocode_precision == "parcel_or_address_point"
    assert result.provider == "census_tiger"
    assert result.matched_address == "4600 SILVER HILL RD, SUITLAND, MD, 20746"
    assert result.candidate_count == 1
    # last_result must be populated for the fallback chain to read metadata.
    assert geocoder.last_result is not None
    assert geocoder.last_result["geocode_precision"] == "parcel_or_address_point"


def test_census_geocoder_empty_address_matches_returns_none(monkeypatch):
    """Empty addressMatches array → geocode() returns None without raising."""
    monkeypatch.setattr(
        "backend.geocoding_census.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeCensusHTTPResponse(_CENSUS_NO_MATCH_PAYLOAD),
    )
    geocoder = CensusGeocoder()
    result = geocoder.geocode("1 Nowhere Blvd, Fakeville, XX 00000")

    assert result is None
    assert geocoder.last_result is None


def test_census_geocoder_network_timeout_returns_none(monkeypatch):
    """Network timeout → geocode() returns None without raising."""

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "backend.geocoding_census.urllib.request.urlopen",
        _raise_timeout,
    )
    geocoder = CensusGeocoder(timeout_seconds=1.0)
    result = geocoder.geocode("123 Main St, Missoula, MT 59801")

    assert result is None
    assert geocoder.last_result is None


def test_census_geocoder_interpolated_precision_when_no_city(monkeypatch):
    """tigerLineId present but city missing → interpolated precision."""
    payload = {
        "result": {
            "addressMatches": [
                {
                    "tigerLine": {"side": "R", "tigerLineId": "99999"},
                    "coordinates": {"x": -113.99, "y": 46.87},
                    "addressComponents": {
                        "zip": "59801",
                        "streetName": "PATTEE CANYON",
                        "preType": "",
                        "city": "",        # empty city → incomplete components
                        "state": "MT",
                        "suffixType": "RD",
                    },
                    "matchedAddress": "1355 PATTEE CANYON RD, MT, 59801",
                }
            ]
        }
    }
    monkeypatch.setattr(
        "backend.geocoding_census.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeCensusHTTPResponse(payload),
    )
    result = CensusGeocoder().geocode("1355 Pattee Canyon Rd, Missoula, MT")
    assert result is not None
    assert result.geocode_precision == "interpolated"


# ---------------------------------------------------------------------------
# GeocodeFallbackChain tests
# ---------------------------------------------------------------------------

def _make_nominatim_result() -> GeocodeResult:
    return GeocodeResult(
        latitude=48.4782,
        longitude=-120.1870,
        source="OpenStreetMap Nominatim",
        geocode_status="accepted",
        submitted_address="12 Twin Lakes Rd, Winthrop, WA",
        normalized_address="12 twin lakes rd winthrop wa",
        provider="OpenStreetMap Nominatim",
        geocode_precision="parcel_or_address_point",
    )


def _mock_census_none() -> MagicMock:
    m = MagicMock(spec=CensusGeocoder)
    m.provider_name = "census_tiger"
    m.geocode.return_value = None
    m.last_result = None
    return m


def _mock_nominatim_tuple(result: GeocodeResult) -> MagicMock:
    m = MagicMock()
    m.provider_name = "OpenStreetMap Nominatim"
    m.geocode.return_value = (result.latitude, result.longitude, result.source)
    m.last_result = {
        "geocode_status": "accepted",
        "geocode_precision": result.geocode_precision,
        "provider": result.provider,
        "submitted_address": result.submitted_address,
        "normalized_address": result.normalized_address,
        "matched_address": result.matched_address,
        "confidence_score": None,
        "candidate_count": 1,
        "geocode_location_type": None,
        "raw_response_preview": None,
        "rejection_reason": None,
    }
    return m


def test_fallback_chain_advances_to_nominatim_when_census_returns_none():
    """Census returns None → chain tries Nominatim and succeeds."""
    nom_result = _make_nominatim_result()
    census_mock = _mock_census_none()
    nom_mock = _mock_nominatim_tuple(nom_result)

    chain = GeocodeFallbackChain([census_mock, nom_mock])
    lat, lon, source = chain.geocode("12 Twin Lakes Rd, Winthrop, WA")

    assert census_mock.geocode.called
    assert nom_mock.geocode.called
    assert lat == pytest.approx(48.4782)
    assert lon == pytest.approx(-120.1870)
    assert chain.last_result is not None
    assert chain.last_result.get("provider_used") == "OpenStreetMap Nominatim"


def test_fallback_chain_stops_at_census_when_census_succeeds(monkeypatch):
    """Census returns a valid result → Nominatim must NOT be called."""
    census_result = GeocodeResult(
        latitude=38.845985,
        longitude=-76.92743,
        source="census_tiger",
        geocode_status="accepted",
        submitted_address="4600 Silver Hill Rd, Suitland, MD",
        normalized_address="4600 silver hill rd suitland md",
        provider="census_tiger",
        geocode_precision="parcel_or_address_point",
    )
    census_mock = MagicMock(spec=CensusGeocoder)
    census_mock.provider_name = "census_tiger"
    census_mock.geocode.return_value = census_result
    census_mock.last_result = {
        "geocode_status": "accepted",
        "geocode_precision": "parcel_or_address_point",
        "provider": "census_tiger",
        "submitted_address": census_result.submitted_address,
        "normalized_address": census_result.normalized_address,
        "matched_address": census_result.matched_address,
        "confidence_score": None,
        "candidate_count": 1,
        "geocode_location_type": None,
        "raw_response_preview": None,
        "rejection_reason": None,
    }

    nom_mock = MagicMock()
    nom_mock.provider_name = "OpenStreetMap Nominatim"

    chain = GeocodeFallbackChain([census_mock, nom_mock])
    lat, lon, source = chain.geocode("4600 Silver Hill Rd, Suitland, MD")

    assert census_mock.geocode.called
    assert not nom_mock.geocode.called
    assert lat == pytest.approx(38.845985)
    assert chain.last_result is not None
    assert chain.last_result.get("provider_used") == "census_tiger"


def test_fallback_chain_raises_geocoding_error_when_all_providers_fail():
    """All providers return None or raise → chain raises GeocodingError."""
    census_mock = _mock_census_none()

    nom_mock = MagicMock()
    nom_mock.provider_name = "OpenStreetMap Nominatim"
    nom_mock.geocode.side_effect = GeocodingError(
        status="no_match",
        message="No geocoding result found.",
        submitted_address="1 Nowhere Blvd, XX",
        normalized_address="1 nowhere blvd xx",
        provider="OpenStreetMap Nominatim",
        rejection_reason="provider returned no candidates",
    )

    chain = GeocodeFallbackChain([census_mock, nom_mock])

    with pytest.raises(GeocodingError) as exc_info:
        chain.geocode("1 Nowhere Blvd, Fakeville, XX 00000")

    assert exc_info.value.status == "no_match"
    assert exc_info.value.provider == "geocode_fallback_chain"
    assert census_mock.geocode.called
    assert nom_mock.geocode.called


# ---------------------------------------------------------------------------
# ZipCodeValidator tests
# ---------------------------------------------------------------------------

# Synthetic centroid dict injected directly so no network or file I/O occurs.
# 98862 → Winthrop WA area centroid (approx)
# 59803 → Missoula MT area centroid (approx)
# 84604 → Provo UT area centroid (approx)
_SYNTHETIC_CENTROIDS: dict[str, tuple[float, float]] = {
    "98862": (48.4785, -120.1865),   # Winthrop, WA
    "59803": (46.8302, -113.9803),   # Missoula, MT
    "84604": (40.2764, -111.6355),   # Provo, UT
}


def _make_validator_with_centroids(
    centroids: dict[str, tuple[float, float]] | None = None,
) -> ZipCodeValidator:
    """Return a ZipCodeValidator with a synthetic centroid dict (no network/file I/O)."""
    v = ZipCodeValidator.__new__(ZipCodeValidator)
    v._cache_path = None  # type: ignore[assignment]
    v._download_url = ""
    v._timeout = 5.0
    v._load_failed = centroids is None
    v._centroids = centroids
    return v


def _make_result(lat: float, lon: float, *, matched_address: str = "", raw: dict | None = None) -> GeocodeResult:
    return GeocodeResult(
        latitude=lat,
        longitude=lon,
        source="test",
        geocode_status="accepted",
        submitted_address="",
        normalized_address="",
        provider="test",
        matched_address=matched_address or None,
        raw_response_preview=raw,
    )


# --- haversine sanity check ---

def test_haversine_km_same_point_is_zero():
    assert haversine_km(48.4785, -120.1865, 48.4785, -120.1865) == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_winthrop_to_dutchess_county():
    """WA 98862 centroid to Dutchess County NY should be ~3,660 km."""
    dist = haversine_km(48.4785, -120.1865, 41.5089, -73.9335)
    assert 3_500 < dist < 3_800


# --- ZipCodeValidator.validate() ---

def test_zip_validator_passes_when_point_near_zip_centroid():
    """Geocoded point within 20 km of ZIP centroid → PASSES."""
    v = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)
    # Missoula address; returned point within 1 km of 59803 centroid.
    result = _make_result(46.830, -113.981)
    vr = v.validate(result, "1355 Pattee Canyon Rd, Missoula, MT 59803", max_distance_km=20.0)
    assert vr.passed is True
    assert vr.input_zip == "59803"
    assert vr.distance_km is not None
    assert vr.distance_km < 20.0


def test_zip_validator_fails_winthrop_wa_returned_as_dutchess_county_ny():
    """Canonical failure: WA 98862 input, NY coordinates returned → FAILS."""
    v = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)
    # Nominatim returned Dutchess County NY coordinates for a WA address.
    result = _make_result(41.5089, -73.9335)
    vr = v.validate(result, "6 Pineview Rd, Winthrop, WA 98862", max_distance_km=20.0)
    assert vr.passed is False
    assert vr.input_zip == "98862"
    assert vr.distance_km is not None
    assert vr.distance_km > 3_000
    assert "98862" in (vr.reason or "")
    assert "km" in (vr.reason or "")


def test_zip_validator_passes_when_state_matches_and_no_zip():
    """No ZIP in input, geocoder returns matching state → PASSES."""
    v = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)
    result = _make_result(
        48.47, -120.19,
        raw={
            "top_candidate": {
                "address": {"state": "Washington", "postcode": None}
            }
        },
    )
    vr = v.validate(result, "6 Pineview Rd, Winthrop, WA", max_distance_km=20.0)
    assert vr.passed is True
    assert vr.input_zip is None
    assert vr.input_state == "WA"


def test_zip_validator_fails_when_state_mismatches_and_no_zip():
    """No ZIP in input, geocoder returns mismatched state → FAILS."""
    v = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)
    result = _make_result(
        41.508, -73.934,
        raw={
            "top_candidate": {
                "address": {"state": "New York", "postcode": None}
            }
        },
    )
    vr = v.validate(result, "6 Pineview Rd, Winthrop, WA", max_distance_km=20.0)
    assert vr.passed is False
    assert vr.input_state == "WA"
    assert "NY" in (vr.reason or "") or "New York" in (vr.reason or "")


def test_zip_validator_passes_when_no_zip_and_no_state_in_input():
    """Input with no ZIP and no recognisable state abbreviation → PASSES (cannot validate)."""
    v = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)
    result = _make_result(41.508, -73.934)
    vr = v.validate(result, "6 Pineview Road", max_distance_km=20.0)
    assert vr.passed is True
    assert vr.input_zip is None
    assert vr.input_state is None
    assert "cannot validate" in (vr.reason or "")


def test_zip_validator_passes_when_zip_not_in_centroid_dataset():
    """ZIP present in input but not in the centroid dict → PASSES with a note."""
    v = _make_validator_with_centroids({"99999": (48.0, -122.0)})  # only 99999 loaded
    result = _make_result(41.508, -73.934)
    vr = v.validate(result, "100 Main St, Somewhere, WA 00001", max_distance_km=20.0)
    assert vr.passed is True
    assert vr.input_zip == "00001"
    assert "not_in_centroid_dataset" in (vr.reason or "")


def test_zip_validator_disabled_when_load_failed():
    """Centroid data load failure → validator disables; all calls return PASSES."""
    v = _make_validator_with_centroids(None)  # _load_failed=True, _centroids=None
    result = _make_result(41.508, -73.934)  # obviously wrong coordinates for WA
    vr = v.validate(result, "6 Pineview Rd, Winthrop, WA 98862", max_distance_km=20.0)
    assert vr.passed is True
    assert "unavailable" in (vr.reason or "")


# --- Chain integration tests with ZIP validation ---

def _make_geocode_result_obj(lat: float, lon: float, zip_centroid_close: bool) -> GeocodeResult:
    """Helper: build a GeocodeResult with a matched_address hinting at WA or NY."""
    state_str = "Washington" if zip_centroid_close else "New York"
    return GeocodeResult(
        latitude=lat,
        longitude=lon,
        source="test_provider",
        geocode_status="accepted",
        submitted_address="6 Pineview Rd, Winthrop, WA 98862",
        normalized_address="6 pineview rd winthrop wa 98862",
        provider="test_provider",
        matched_address=f"6 Pineview Rd, {state_str}",
        raw_response_preview={
            "top_candidate": {
                "address": {"state": state_str, "postcode": "98862" if zip_centroid_close else "12508"}
            }
        },
    )


def test_chain_with_zip_validation_discards_failing_provider_advances_to_next():
    """First provider fails ZIP validation → chain tries second provider and returns it."""
    validator = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)

    # Provider 1: returns NY coordinates for a WA address → validation fails.
    bad_result = _make_geocode_result_obj(41.5089, -73.9335, zip_centroid_close=False)
    p1 = MagicMock()
    p1.provider_name = "bad_provider"
    p1.geocode.return_value = bad_result
    p1.last_result = {
        "geocode_status": "accepted", "geocode_precision": "interpolated",
        "provider": "bad_provider", "matched_address": bad_result.matched_address,
        "submitted_address": bad_result.submitted_address,
        "normalized_address": bad_result.normalized_address,
        "confidence_score": None, "candidate_count": 1,
        "geocode_location_type": None, "raw_response_preview": bad_result.raw_response_preview,
        "rejection_reason": None,
    }

    # Provider 2: returns WA coordinates → validation passes.
    good_result = _make_geocode_result_obj(48.4772, -120.1864, zip_centroid_close=True)
    p2 = MagicMock()
    p2.provider_name = "good_provider"
    p2.geocode.return_value = good_result
    p2.last_result = {
        "geocode_status": "accepted", "geocode_precision": "parcel_or_address_point",
        "provider": "good_provider", "matched_address": good_result.matched_address,
        "submitted_address": good_result.submitted_address,
        "normalized_address": good_result.normalized_address,
        "confidence_score": None, "candidate_count": 1,
        "geocode_location_type": None, "raw_response_preview": good_result.raw_response_preview,
        "rejection_reason": None,
    }

    chain = GeocodeFallbackChain(
        [p1, p2],
        zip_validator=validator,
        zip_validation_enabled=True,
        zip_max_distance_km=20.0,
    )
    lat, lon, source = chain.geocode("6 Pineview Rd, Winthrop, WA 98862")

    assert p1.geocode.called
    assert p2.geocode.called
    assert lat == pytest.approx(48.4772)
    assert lon == pytest.approx(-120.1864)
    assert chain.last_result is not None
    assert chain.last_result.get("provider_used") == "good_provider"
    assert chain.last_result.get("zip_validation_passed") is True


def test_chain_with_zip_validation_all_fail_returns_last_result_not_none():
    """All providers fail ZIP validation → chain returns last result with zip_validation_passed=False."""
    validator = _make_validator_with_centroids(_SYNTHETIC_CENTROIDS)

    bad_result = _make_geocode_result_obj(41.5089, -73.9335, zip_centroid_close=False)
    bad_meta = {
        "geocode_status": "accepted", "geocode_precision": "interpolated",
        "provider": "bad_provider", "matched_address": bad_result.matched_address,
        "submitted_address": bad_result.submitted_address,
        "normalized_address": bad_result.normalized_address,
        "confidence_score": None, "candidate_count": 1,
        "geocode_location_type": None, "raw_response_preview": bad_result.raw_response_preview,
        "rejection_reason": None,
    }
    p1 = MagicMock()
    p1.provider_name = "bad_provider"
    p1.geocode.return_value = bad_result
    p1.last_result = bad_meta

    chain = GeocodeFallbackChain(
        [p1],
        zip_validator=validator,
        zip_validation_enabled=True,
        zip_max_distance_km=20.0,
    )
    lat, lon, source = chain.geocode("6 Pineview Rd, Winthrop, WA 98862")

    # Must return coordinates, not raise.
    assert lat == pytest.approx(41.5089)
    assert lon == pytest.approx(-73.9335)
    assert chain.last_result is not None
    assert chain.last_result.get("zip_validation_passed") is False
    assert chain.last_result.get("zip_validation_reason") is not None
