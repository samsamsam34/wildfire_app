"""Ordered geocoding fallback chain for the wildfire risk platform.

Tries each configured geocoder in sequence and returns the first successful
result.  The chain's public interface — ``.geocode()``, ``.last_result``, and
``.provider_name`` — matches :class:`~backend.geocoding.Geocoder` exactly, so
it can be passed directly to ``_geocode_address_or_raise`` in ``main.py``
without any changes at the call site.

Default provider order (from ``config/geocoding_config.yaml``):

1. **Census TIGER** (free, national, no API key required)
2. **Nominatim** (current Geocoder instance, passed in at construction)
3. **Google Maps** (stub only; enabled when ``WF_GOOGLE_GEOCODE_API_KEY`` is
   set *and* ``enabled: true`` in the config file)

Each provider's ``geocode()`` method may return either:

* A :class:`~backend.geocoding.GeocodeResult` (Census-style) — success.
* A ``(lat, lon, source)`` tuple (Nominatim-style) — success.
* ``None`` — no match; advance to the next provider.
* Raise :class:`~backend.geocoding.GeocodingError` — failed; log and advance.

ZIP validation
--------------
When ``validation.zip_validation_enabled`` is ``true`` in
``config/geocoding_config.yaml``, each network geocoder result is validated
against the ZIP code or state in the input address before being accepted.  A
result that fails validation is discarded and the next provider is tried.  If
every provider's result fails validation, the last failed result is returned
anyway with ``zip_validation_passed=False`` in :attr:`last_result` so the
caller can apply its own confidence penalty rather than receiving nothing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from backend.geocoding import GeocodeResult, GeocodingError, normalize_address
from backend.geocoding_census import CensusGeocoder
from backend.geocoding_zip_validator import ZipCodeValidator

LOGGER = logging.getLogger("wildfire_app.geocoding_fallback_chain")

_DEFAULT_CONFIG_PATH = Path("config") / "geocoding_config.yaml"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _parse_yaml_like(text: str) -> dict[str, Any]:
    """Parse YAML with a fallback to JSON, matching the pattern in scoring_config.py."""
    try:  # pragma: no cover - exercised when PyYAML is installed
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _load_geocoding_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``config/geocoding_config.yaml`` (or the path from the env var)."""
    config_path = path or Path(
        os.getenv("WF_GEOCODING_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))
    )
    if not config_path.exists():
        return {}
    try:
        return _parse_yaml_like(config_path.read_text(encoding="utf-8"))
    except OSError:
        return {}


# ---------------------------------------------------------------------------
# Google stub
# ---------------------------------------------------------------------------

class _GoogleGeocoderStub:
    """Placeholder for a future Google Maps Geocoding API integration.

    Always returns ``None`` so the chain passes through to any later provider.
    Logging at INFO level makes it visible that the stub is active.
    """

    provider_name: str = "google_maps"

    def __init__(self, *, api_key: str, timeout_seconds: float = 5.0) -> None:
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.last_result: dict[str, Any] | None = None
        LOGGER.info(
            "google_geocoder_stub enabled but not yet implemented; "
            "will always pass through to next provider"
        )

    def geocode(self, address: str) -> Optional[GeocodeResult]:  # noqa: ARG002
        LOGGER.debug("google_geocoder_stub pass_through address=%r", address)
        return None


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class GeocodeFallbackChain:
    """Ordered geocoding fallback chain with optional ZIP validation.

    Tries each provider in sequence and returns the first successful result
    that passes geographic validation.  Implements the same ``geocode()`` /
    ``last_result`` / ``provider_name`` protocol as
    :class:`~backend.geocoding.Geocoder` so it can replace it in
    ``_geocode_address_or_raise`` without any changes at the call site.

    Parameters
    ----------
    providers:
        Ordered list of geocoder instances.  Each must expose a
        ``geocode(address: str)`` method and a ``provider_name`` attribute.
    zip_validator:
        :class:`~backend.geocoding_zip_validator.ZipCodeValidator` instance.
        If ``None``, a default instance is created but validation is only
        active when ``zip_validation_enabled=True``.
    zip_validation_enabled:
        Whether to run ZIP/state validation after each provider result.
        Default ``False``; set ``True`` via :meth:`from_config`.
    zip_max_distance_km:
        Distance threshold passed to the validator.  Default 20 km.

    Construction
    ------------
    Use :meth:`from_config` for production use; pass providers directly in
    tests.

    Example
    -------
    >>> chain = GeocodeFallbackChain.from_config(nominatim_instance=geocoder)
    >>> lat, lon, source = chain.geocode("123 Main St, Missoula, MT 59801")
    """

    provider_name: str = "geocode_fallback_chain"

    def __init__(
        self,
        providers: list[Any],
        *,
        zip_validator: ZipCodeValidator | None = None,
        zip_validation_enabled: bool = False,
        zip_max_distance_km: float = 20.0,
    ) -> None:
        if not providers:
            raise ValueError("GeocodeFallbackChain requires at least one provider.")
        self._providers = list(providers)
        self._zip_validator = zip_validator or ZipCodeValidator()
        self._zip_validation_enabled = zip_validation_enabled
        self._zip_max_distance_km = zip_max_distance_km
        self.last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public interface — matches Geocoder.geocode() / last_result protocol
    # ------------------------------------------------------------------

    def geocode(self, address: str) -> Tuple[float, float, str]:
        """Geocode *address*, trying each provider in order.

        Providers are tried in sequence.  When ZIP validation is enabled, each
        result is checked against the ZIP/state in *address* before being
        accepted.  A failing result causes the chain to advance to the next
        provider.

        If every provider's result fails validation, the last validated-but-failed
        result is still returned so downstream code has coordinates to work with;
        ``self.last_result`` will contain ``zip_validation_passed=False`` and
        ``zip_validation_reason`` so the caller can apply a confidence penalty.

        Returns ``(latitude, longitude, source)`` on success.
        Raises :class:`~backend.geocoding.GeocodingError` if every provider
        fails to return any result at all (not just validation failures).

        ``self.last_result`` is populated with the winning provider's metadata
        dict, augmented with ``provider_used``, ``zip_validation_passed``, and
        optionally ``zip_validation_reason``.
        """
        self.last_result = None
        submitted = str(address or "").strip()
        last_exc: GeocodingError | None = None

        # Holds the most recent result that returned coordinates but failed
        # validation, so we can return it as a last resort.
        last_failed_coords: tuple[float, float, str] | None = None
        last_failed_meta: dict[str, Any] | None = None
        last_failed_reason: str | None = None

        for provider in self._providers:
            provider_label = str(
                getattr(provider, "provider_name", type(provider).__name__)
            )

            try:
                raw = provider.geocode(submitted)
            except GeocodingError as exc:
                LOGGER.warning(
                    "geocode_fallback provider=%r status=%r address=%r reason=%r",
                    provider_label,
                    exc.status,
                    submitted,
                    exc.rejection_reason,
                )
                last_exc = exc
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "geocode_fallback provider=%r unexpected_error address=%r error=%s",
                    provider_label,
                    submitted,
                    exc,
                )
                continue

            if raw is None:
                LOGGER.debug(
                    "geocode_fallback provider=%r no_match address=%r",
                    provider_label,
                    submitted,
                )
                continue

            # Normalise: Census returns GeocodeResult; Nominatim returns tuple.
            if isinstance(raw, GeocodeResult):
                lat: float = raw.latitude
                lon: float = raw.longitude
                source: str = raw.source
            elif isinstance(raw, tuple) and len(raw) == 3:
                lat, lon, source = raw
            else:
                LOGGER.warning(
                    "geocode_fallback provider=%r unexpected_return_type address=%r type=%s",
                    provider_label,
                    submitted,
                    type(raw).__name__,
                )
                continue

            provider_last = getattr(provider, "last_result", None)

            # Build the metadata dict for this candidate result.
            if isinstance(provider_last, dict):
                candidate_meta = dict(provider_last)
            else:
                candidate_meta = {
                    "geocode_status": "accepted",
                    "submitted_address": submitted,
                    "normalized_address": normalize_address(submitted),
                    "provider": provider_label,
                    "matched_address": None,
                    "confidence_score": None,
                    "candidate_count": 1,
                    "geocode_location_type": None,
                    "geocode_precision": "unknown",
                    "raw_response_preview": None,
                    "rejection_reason": None,
                }
            candidate_meta["provider_used"] = provider_label

            # --- ZIP validation ---
            if self._zip_validation_enabled:
                # Build a GeocodeResult for the validator when the provider
                # returned a raw tuple (Nominatim) rather than a GeocodeResult.
                if isinstance(raw, GeocodeResult):
                    result_for_validation = raw
                else:
                    result_for_validation = GeocodeResult(
                        latitude=lat,
                        longitude=lon,
                        source=source,
                        geocode_status="accepted",
                        submitted_address=submitted,
                        normalized_address=normalize_address(submitted),
                        provider=provider_label,
                        matched_address=candidate_meta.get("matched_address"),
                        confidence_score=candidate_meta.get("confidence_score"),
                        geocode_precision=str(
                            candidate_meta.get("geocode_precision") or "unknown"
                        ),
                        raw_response_preview=candidate_meta.get("raw_response_preview"),
                    )

                try:
                    validation = self._zip_validator.validate(
                        result_for_validation,
                        submitted,
                        max_distance_km=self._zip_max_distance_km,
                    )
                except Exception as exc:  # pragma: no cover - never raise from validate
                    LOGGER.warning(
                        "geocode_fallback zip_validator_error provider=%r error=%s",
                        provider_label,
                        exc,
                    )
                    validation = None  # treat as passed on unexpected error

                if validation is not None and not validation.passed:
                    LOGGER.warning(
                        "geocode_fallback zip_validation_failed provider=%r "
                        "address=%r reason=%r lat=%.4f lon=%.4f",
                        provider_label,
                        submitted,
                        validation.reason,
                        lat,
                        lon,
                    )
                    # Track last failed result so we can return it if all fail.
                    last_failed_coords = (lat, lon, source)
                    last_failed_meta = dict(candidate_meta)
                    last_failed_meta["zip_validation_passed"] = False
                    last_failed_meta["zip_validation_reason"] = validation.reason
                    last_failed_reason = validation.reason
                    continue  # try next provider

                # Validation passed (or disabled).
                if validation is not None:
                    candidate_meta["zip_validation_passed"] = True
                    if validation.reason:
                        candidate_meta["zip_validation_reason"] = validation.reason
                    if validation.distance_km is not None:
                        candidate_meta["zip_validation_distance_km"] = validation.distance_km

            LOGGER.info(
                "geocode_fallback success provider=%r address=%r "
                "precision=%s lat=%.6f lon=%.6f",
                provider_label,
                submitted,
                candidate_meta.get("geocode_precision", "unknown"),
                lat,
                lon,
            )

            self.last_result = candidate_meta
            return (lat, lon, source)

        # --- All providers exhausted ---

        # If we have a result that returned coordinates but failed validation,
        # return it with zip_validation_passed=False rather than raising.
        if last_failed_coords is not None and last_failed_meta is not None:
            LOGGER.warning(
                "geocode_fallback all_providers_failed_zip_validation address=%r "
                "returning last failed result with zip_validation_passed=False reason=%r",
                submitted,
                last_failed_reason,
            )
            self.last_result = last_failed_meta
            return last_failed_coords

        # No provider returned any coordinates at all.
        if last_exc is not None:
            raise GeocodingError(
                status=last_exc.status,
                message=last_exc.message,
                submitted_address=last_exc.submitted_address,
                normalized_address=last_exc.normalized_address,
                provider="geocode_fallback_chain",
                rejection_reason=last_exc.rejection_reason,
                raw_response_preview=last_exc.raw_response_preview,
            )

        raise GeocodingError(
            status="no_match",
            message="No geocoding provider matched the address.",
            submitted_address=submitted,
            normalized_address=normalize_address(submitted),
            provider="geocode_fallback_chain",
            rejection_reason="all_providers_returned_no_match",
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        nominatim_instance: Any,
        config_path: Path | None = None,
    ) -> "GeocodeFallbackChain":
        """Build a chain from ``config/geocoding_config.yaml`` and env vars.

        Always includes Census TIGER and the supplied Nominatim instance
        (unless explicitly disabled in the config file).  Adds the Google stub
        if ``WF_GOOGLE_GEOCODE_API_KEY`` is set **and** ``enabled: true`` in
        the config's ``provider_chain`` entry for ``google``.

        ZIP validation settings are read from ``config.validation``.

        Parameters
        ----------
        nominatim_instance:
            An already-constructed :class:`~backend.geocoding.Geocoder`
            instance to use as the Nominatim provider.
        config_path:
            Override the default config file path (useful in tests).
        """
        config = _load_geocoding_config(config_path)
        chain_config: list[Any] = config.get("provider_chain") or []
        validation_cfg: dict[str, Any] = config.get("validation") or {}

        provider_settings: dict[str, dict[str, Any]] = {}
        for entry in chain_config:
            if isinstance(entry, dict) and entry.get("name"):
                provider_settings[str(entry["name"])] = entry

        providers: list[Any] = []

        # --- Census TIGER ---
        census_cfg = provider_settings.get("census_tiger", {})
        if census_cfg.get("enabled", True) is not False:
            census_timeout = float(census_cfg.get("timeout_seconds", 5.0))
            providers.append(CensusGeocoder(timeout_seconds=census_timeout))
            LOGGER.debug(
                "geocode_fallback_chain adding census_tiger timeout=%.1fs",
                census_timeout,
            )

        # --- Nominatim (passed in) ---
        nominatim_cfg = provider_settings.get("nominatim", {})
        if nominatim_cfg.get("enabled", True) is not False:
            providers.append(nominatim_instance)
            LOGGER.debug("geocode_fallback_chain adding nominatim")

        # --- Google (stub; only when API key present AND enabled in config) ---
        google_api_key = str(os.getenv("WF_GOOGLE_GEOCODE_API_KEY") or "").strip()
        google_cfg = provider_settings.get("google", {})
        if google_api_key and google_cfg.get("enabled", False):
            google_timeout = float(google_cfg.get("timeout_seconds", 5.0))
            providers.append(
                _GoogleGeocoderStub(
                    api_key=google_api_key,
                    timeout_seconds=google_timeout,
                )
            )
            LOGGER.info(
                "geocode_fallback_chain adding google_maps_stub "
                "(WF_GOOGLE_GEOCODE_API_KEY is set)"
            )

        if not providers:
            LOGGER.warning(
                "geocode_fallback_chain no providers configured; "
                "falling back to nominatim only"
            )
            providers.append(nominatim_instance)

        # --- ZIP validation ---
        zip_validation_enabled = bool(validation_cfg.get("zip_validation_enabled", False))
        zip_max_distance_km = float(validation_cfg.get("zip_max_distance_km", 20.0))
        zip_validator = ZipCodeValidator()

        LOGGER.info(
            "geocode_fallback_chain initialized providers=%s zip_validation=%s",
            [str(getattr(p, "provider_name", type(p).__name__)) for p in providers],
            zip_validation_enabled,
        )
        return cls(
            providers,
            zip_validator=zip_validator,
            zip_validation_enabled=zip_validation_enabled,
            zip_max_distance_km=zip_max_distance_km,
        )
