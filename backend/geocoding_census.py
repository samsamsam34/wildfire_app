"""US Census TIGER geocoder for the wildfire risk platform.

Wraps the Census Geocoding Services REST API (``onelineaddress`` endpoint,
``Public_AR_Current`` benchmark).  Returns a
:class:`~backend.geocoding.GeocodeResult` on success, or ``None`` when the
address is not found or the request fails, so
:class:`~backend.geocoding_fallback_chain.GeocodeFallbackChain` can advance
to the next provider without raising.

Census precision notes
----------------------
The ``Public_AR_Current`` benchmark interpolates coordinates along TIGER/Line
road centerline segments — it is not rooftop-level.  The precision tiers used
here are conservative:

* ``parcel_or_address_point`` — ``tigerLineId`` present **and** all key address
  components (zip, street, city, state) returned.  Best Census can offer.
* ``interpolated`` — ``tigerLineId`` present but components incomplete.
* ``approximate`` — No ``tigerLineId`` (rare; fallback match).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from backend.geocoding import GeocodeResult, normalize_address

LOGGER = logging.getLogger("wildfire_app.geocoding_census")

_CENSUS_GEOCODE_URL = (
    "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
)
_CENSUS_BENCHMARK = "Public_AR_Current"
_DEFAULT_TIMEOUT_SECONDS: float = 5.0
_USER_AGENT = "WildfireRiskAdvisor/0.1"


class CensusGeocoder:
    """Geocoder backed by the US Census Bureau TIGER/Line REST API.

    Returns ``None`` instead of raising on no-match or network failure so the
    :class:`~backend.geocoding_fallback_chain.GeocodeFallbackChain` can
    silently advance to the next provider.

    Attributes
    ----------
    provider_name:
        Fixed string ``"census_tiger"`` used for logging and audit trails.
    last_result:
        Populated after every :meth:`geocode` call (success or failure) with
        a metadata dict compatible with the shape read by
        ``_geocode_address_or_raise`` in ``main.py``.
    """

    provider_name: str = "census_tiger"

    def __init__(self, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def geocode(self, address: str) -> Optional[GeocodeResult]:
        """Geocode *address* via the Census API.

        Parameters
        ----------
        address:
            Free-form US address string (e.g. ``"4600 Silver Hill Rd, Suitland, MD"``).

        Returns
        -------
        :class:`~backend.geocoding.GeocodeResult` on success, ``None`` if the
        address was not found or any network / parse error occurred.
        """
        self.last_result = None
        submitted = str(address or "").strip()

        result = self._call_api(submitted)

        if result is None:
            return None

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
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, submitted_address: str) -> GeocodeResult | None:
        """Fetch the Census API and return a parsed result, or ``None``."""
        params = urllib.parse.urlencode(
            {
                "address": submitted_address,
                "benchmark": _CENSUS_BENCHMARK,
                "format": "json",
            }
        )
        url = f"{_CENSUS_GEOCODE_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            LOGGER.warning(
                "census_geocoder http_error address=%r status=%d",
                submitted_address,
                exc.code,
            )
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            LOGGER.warning(
                "census_geocoder network_error address=%r reason=%s",
                submitted_address,
                exc,
            )
            return None
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "census_geocoder malformed_json address=%r error=%s",
                submitted_address,
                exc,
            )
            return None

        return self._parse_response(submitted_address, payload)

    def _parse_response(
        self,
        submitted_address: str,
        payload: Any,
    ) -> GeocodeResult | None:
        """Parse a Census API JSON payload into a :class:`GeocodeResult`."""
        try:
            matches = payload["result"]["addressMatches"]
        except (KeyError, TypeError):
            LOGGER.warning(
                "census_geocoder unexpected_shape address=%r payload_type=%s",
                submitted_address,
                type(payload).__name__,
            )
            return None

        if not isinstance(matches, list) or len(matches) == 0:
            LOGGER.debug("census_geocoder no_match address=%r", submitted_address)
            return None

        best = matches[0]
        if not isinstance(best, dict):
            return None

        try:
            coords = best["coordinates"]
            lon = float(coords["x"])
            lat = float(coords["y"])
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "census_geocoder missing_coordinates address=%r error=%s",
                submitted_address,
                exc,
            )
            return None

        matched_address: str | None = str(best.get("matchedAddress") or "").strip() or None
        components: dict[str, Any] = best.get("addressComponents") or {}
        tiger_line: dict[str, Any] = best.get("tigerLine") or {}
        tiger_line_id: str = str(tiger_line.get("tigerLineId") or "").strip()

        has_complete_components = bool(
            components.get("zip")
            and components.get("streetName")
            and components.get("city")
            and components.get("state")
        )

        if tiger_line_id and has_complete_components:
            geocode_precision = "parcel_or_address_point"
        elif tiger_line_id:
            geocode_precision = "interpolated"
        else:
            geocode_precision = "approximate"

        normalized_address = normalize_address(submitted_address)

        LOGGER.debug(
            "census_geocoder matched address=%r matched=%r precision=%s lat=%.6f lon=%.6f",
            submitted_address,
            matched_address,
            geocode_precision,
            lat,
            lon,
        )

        return GeocodeResult(
            latitude=lat,
            longitude=lon,
            source="census_tiger",
            geocode_status="accepted",
            submitted_address=submitted_address,
            normalized_address=normalized_address,
            provider=self.provider_name,
            matched_address=matched_address,
            confidence_score=None,
            candidate_count=len(matches),
            geocode_location_type=f"census:{geocode_precision}",
            geocode_precision=geocode_precision,
            raw_response_preview={
                "matched_address": matched_address,
                "tiger_line_id": tiger_line_id or None,
                "address_components": components,
                "candidate_count": len(matches),
                "provider": self.provider_name,
            },
        )
