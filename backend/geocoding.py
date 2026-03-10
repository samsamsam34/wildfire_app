from __future__ import annotations

import os
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Tuple


def normalize_address(address: str) -> str:
    return " ".join(str(address or "").strip().split())


@dataclass
class GeocodeResult:
    latitude: float
    longitude: float
    source: str
    geocode_status: str = "accepted"
    submitted_address: str = ""
    normalized_address: str = ""
    provider: str = "OpenStreetMap Nominatim"
    matched_address: str | None = None
    confidence_score: float | None = None
    candidate_count: int = 0
    geocode_location_type: str | None = None
    geocode_precision: str = "unknown"
    raw_response_preview: dict[str, Any] | None = None


class GeocodingError(Exception):
    def __init__(
        self,
        *,
        status: str,
        message: str,
        submitted_address: str,
        normalized_address: str,
        provider: str = "OpenStreetMap Nominatim",
        rejection_reason: str | None = None,
        raw_response_preview: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.submitted_address = submitted_address
        self.normalized_address = normalized_address
        self.provider = provider
        self.rejection_reason = rejection_reason or message
        self.raw_response_preview = raw_response_preview


class Geocoder:
    def __init__(self, user_agent: str = "WildfireRiskAdvisor/0.1") -> None:
        self.user_agent = user_agent
        timeout_raw = os.getenv("WF_GEOCODE_TIMEOUT_SECONDS", "8")
        runtime_env = str(os.getenv("WF_ENV") or os.getenv("APP_ENV") or "").strip().lower()
        is_dev_mode = (
            runtime_env in {"dev", "development", "local", "test"}
            or bool(os.getenv("PYTEST_CURRENT_TEST"))
            or str(os.getenv("WF_DEBUG_MODE") or "").strip().lower() in {"1", "true", "yes", "on"}
        )
        default_min_importance = "0.0" if is_dev_mode else "0.02"
        min_importance_raw = os.getenv("WF_GEOCODE_MIN_IMPORTANCE", default_min_importance)
        ambiguity_delta_raw = os.getenv("WF_GEOCODE_AMBIGUITY_DELTA", "0.0")
        max_candidates_raw = os.getenv("WF_GEOCODE_MAX_CANDIDATES", "5")
        allow_precise_low_importance_raw = os.getenv("WF_GEOCODE_ALLOW_PRECISE_LOW_IMPORTANCE", "true")

        try:
            self.timeout_seconds = max(1.0, float(timeout_raw))
        except ValueError:
            self.timeout_seconds = 8.0
        try:
            self.min_importance = max(0.0, min(1.0, float(min_importance_raw)))
        except ValueError:
            self.min_importance = 0.0
        try:
            self.ambiguity_delta = max(0.0, float(ambiguity_delta_raw))
        except ValueError:
            self.ambiguity_delta = 0.0
        try:
            self.max_candidates = max(1, min(10, int(max_candidates_raw)))
        except ValueError:
            self.max_candidates = 5
        self.allow_precise_low_importance = allow_precise_low_importance_raw.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        self.last_result: dict[str, Any] | None = None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _preview_candidate(candidate: Any) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        address = candidate.get("address") if isinstance(candidate.get("address"), dict) else {}
        return {
            "display_name": candidate.get("display_name"),
            "lat": candidate.get("lat"),
            "lon": candidate.get("lon"),
            "importance": candidate.get("importance"),
            "class": candidate.get("class"),
            "type": candidate.get("type"),
            "address": {
                "house_number": address.get("house_number"),
                "road": address.get("road"),
                "city": address.get("city") or address.get("town") or address.get("village") or address.get("hamlet"),
                "state": address.get("state"),
                "postcode": address.get("postcode"),
                "country_code": address.get("country_code"),
            },
        }

    @staticmethod
    def _candidate_has_precise_address(candidate: Any) -> bool:
        if not isinstance(candidate, dict):
            return False
        address = candidate.get("address")
        if not isinstance(address, dict):
            return False
        house = str(address.get("house_number") or "").strip()
        road = str(address.get("road") or "").strip()
        locality = str(
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
            or ""
        ).strip()
        state = str(address.get("state") or "").strip()
        postcode = str(address.get("postcode") or "").strip()
        # Treat house+road+locality as strongest precision, but also accept road/locality/postcode
        # because some valid provider matches omit house_number while still returning precise coordinates.
        has_precise_street_context = bool(road and locality and (house or postcode))
        has_regional_anchor = bool(state or postcode)
        return has_precise_street_context and has_regional_anchor

    def _derive_precision_tier(self, candidate: Any, importance: float | None) -> tuple[str, str | None]:
        if not isinstance(candidate, dict):
            return "unknown", None
        cand_class = str(candidate.get("class") or "").strip().lower()
        cand_type = str(candidate.get("type") or "").strip().lower()
        location_type = f"{cand_class}:{cand_type}" if cand_class or cand_type else None
        precise = self._candidate_has_precise_address(candidate)

        if cand_type in {"house", "building"} or cand_class in {"building"}:
            return "rooftop", location_type
        if precise:
            return "parcel_or_address_point", location_type
        if cand_class in {"highway", "place"} or cand_type in {"residential", "road", "street"}:
            return "interpolated", location_type
        if importance is not None and importance < max(0.01, self.min_importance or 0.0):
            return "approximate", location_type
        return "unknown", location_type

    def geocode_with_diagnostics(self, address: str) -> GeocodeResult:
        submitted_address = str(address or "")
        normalized_address = normalize_address(submitted_address)
        provider = "OpenStreetMap Nominatim"

        if len(normalized_address) < 5:
            raise GeocodingError(
                status="parser_error",
                message="Address input is too short for geocoding.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
            )

        query = urllib.parse.urlencode(
            {
                "q": normalized_address,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": self.max_candidates,
            }
        )
        url = f"https://nominatim.openstreetmap.org/search?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:350]
            except Exception:
                body = ""
            raise GeocodingError(
                status="provider_error",
                message=f"Geocoding provider returned HTTP {exc.code}.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
                rejection_reason=body or f"HTTP {exc.code}",
                raw_response_preview={"http_status": exc.code, "body_snippet": body},
            ) from exc
        except urllib.error.URLError as exc:
            raise GeocodingError(
                status="provider_error",
                message="Geocoding provider is unavailable.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
                rejection_reason=str(exc.reason) if getattr(exc, "reason", None) else str(exc),
            ) from exc
        except TimeoutError as exc:
            raise GeocodingError(
                status="provider_error",
                message="Geocoding request timed out.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
            ) from exc
        except json.JSONDecodeError as exc:
            raise GeocodingError(
                status="parser_error",
                message="Geocoding provider returned malformed JSON.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
            ) from exc

        if not isinstance(payload, list):
            raise GeocodingError(
                status="parser_error",
                message="Geocoding payload had an unexpected shape.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
                raw_response_preview={"payload_type": type(payload).__name__},
            )

        if not payload:
            raise GeocodingError(
                status="no_match",
                message="No geocoding result found.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
            )

        first = payload[0]
        lat = self._to_float(first.get("lat") if isinstance(first, dict) else None)
        lon = self._to_float(first.get("lon") if isinstance(first, dict) else None)
        if lat is None or lon is None:
            raise GeocodingError(
                status="missing_coordinates",
                message="Geocoding response did not include valid coordinates.",
                submitted_address=submitted_address,
                normalized_address=normalized_address,
                provider=provider,
                raw_response_preview={"top_candidate": self._preview_candidate(first)},
            )

        importance = self._to_float(first.get("importance") if isinstance(first, dict) else None)
        trust_filter_rejected = False
        trust_filter_rule = None
        if self.min_importance > 0.0 and (importance is None or importance < self.min_importance):
            precise_override = self.allow_precise_low_importance and self._candidate_has_precise_address(first)
            if precise_override:
                trust_filter_rule = "low_importance_but_precise_address_override"
            else:
                trust_filter_rejected = True
                trust_filter_rule = "min_importance_threshold"
                raise GeocodingError(
                    status="low_confidence",
                    message="Best geocoding match was below the confidence threshold.",
                    submitted_address=submitted_address,
                    normalized_address=normalized_address,
                    provider=provider,
                    rejection_reason=f"importance={importance} threshold={self.min_importance}",
                    raw_response_preview={
                        "top_candidate": self._preview_candidate(first),
                        "candidate_count": len(payload),
                        "trust_filter_rule": trust_filter_rule,
                    },
                )

        if self.ambiguity_delta > 0.0 and len(payload) > 1 and isinstance(first, dict):
            second = payload[1]
            top_importance = importance or 0.0
            second_importance = self._to_float(second.get("importance") if isinstance(second, dict) else None) or 0.0
            if second_importance >= (top_importance - self.ambiguity_delta):
                if (first.get("display_name") or "") != (second.get("display_name") if isinstance(second, dict) else ""):
                    raise GeocodingError(
                        status="ambiguous_match",
                        message="Geocoding returned multiple similarly ranked matches.",
                        submitted_address=submitted_address,
                        normalized_address=normalized_address,
                        provider=provider,
                        rejection_reason=(
                            f"top_importance={top_importance}, second_importance={second_importance}, "
                            f"ambiguity_delta={self.ambiguity_delta}"
                        ),
                        raw_response_preview={
                            "top_candidate": self._preview_candidate(first),
                            "second_candidate": self._preview_candidate(second),
                            "candidate_count": len(payload),
                            "trust_filter_rule": "ambiguity_delta_threshold",
                        },
                    )

        matched_address = first.get("display_name") if isinstance(first, dict) else None
        geocode_precision, geocode_location_type = self._derive_precision_tier(first, importance)
        candidate_previews = [p for p in (self._preview_candidate(c) for c in payload[: self.max_candidates]) if p]
        return GeocodeResult(
            latitude=float(lat),
            longitude=float(lon),
            source=provider,
            geocode_status="accepted",
            submitted_address=submitted_address,
            normalized_address=normalized_address,
            provider=provider,
            matched_address=matched_address,
            confidence_score=importance,
            candidate_count=len(payload),
            geocode_location_type=geocode_location_type,
            geocode_precision=geocode_precision,
            raw_response_preview={
                "top_candidate": self._preview_candidate(first),
                "parsed_candidates": candidate_previews,
                "candidate_count": len(payload),
                "trust_filter_rule": trust_filter_rule,
                "trust_filter_rejected": trust_filter_rejected,
            },
        )

    def geocode(self, address: str) -> Tuple[float, float, str]:
        self.last_result = None
        try:
            result = self.geocode_with_diagnostics(address)
        except GeocodingError as exc:
            self.last_result = {
                "geocode_status": exc.status,
                "submitted_address": exc.submitted_address,
                "normalized_address": exc.normalized_address,
                "provider": exc.provider,
                "matched_address": None,
                "confidence_score": None,
                "candidate_count": 0,
                "geocode_location_type": None,
                "geocode_precision": "unknown",
                "raw_response_preview": exc.raw_response_preview,
                "rejection_reason": exc.rejection_reason,
            }
            raise
        self.last_result = {
            "geocode_status": result.geocode_status,
            "submitted_address": result.submitted_address,
            "normalized_address": result.normalized_address,
            "provider": result.provider,
            "matched_address": result.matched_address,
            "confidence_score": result.confidence_score,
            "candidate_count": result.candidate_count,
            "geocode_location_type": result.geocode_location_type,
            "geocode_precision": result.geocode_precision,
            "raw_response_preview": result.raw_response_preview,
            "rejection_reason": None,
        }
        return result.latitude, result.longitude, result.source
