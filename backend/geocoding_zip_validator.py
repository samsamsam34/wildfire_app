"""ZIP code geographic validation for the wildfire risk platform.

After a network geocoder returns coordinates, this module validates that the
result is geographically consistent with the state or ZIP code present in the
input address.  The canonical failure case it catches: Nominatim matches
"6 Pineview Rd, Winthrop, WA 98862" to a road in Dutchess County, New York
because street name and house number match — the returned point is ~3,500 km
from the WA 98862 ZIP centroid.

Data source
-----------
US ZIP code centroids from the Census Bureau ZCTA Gazetteer (free, updated
annually).  Downloaded once to ``data/zip_centroids.csv`` and read from disk
on all subsequent calls.  If the download fails the validator disables itself
gracefully — no assessment is blocked due to missing ZIP data.

Dependencies
------------
Zero external dependencies.  Uses only Python stdlib (``math``, ``csv``,
``io``, ``zipfile``, ``urllib``, ``pathlib``) and the existing
:class:`~backend.geocoding.GeocodeResult` type.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import re
import zipfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.geocoding import GeocodeResult

LOGGER = logging.getLogger("wildfire_app.geocoding_zip_validator")

# ---------------------------------------------------------------------------
# Census Gazetteer download target
# ---------------------------------------------------------------------------

_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_zcta_national.zip"
)
_CACHE_PATH = Path("data") / "zip_centroids.csv"
_DOWNLOAD_TIMEOUT_SECONDS: float = 20.0

# ---------------------------------------------------------------------------
# US state name → 2-letter abbreviation lookup
# Used to normalise Nominatim full-name state strings ("New York" → "NY").
# ---------------------------------------------------------------------------

_STATE_NAME_TO_ABBR: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "puerto rico": "PR",
    "guam": "GU", "virgin islands": "VI", "american samoa": "AS",
    "northern mariana islands": "MP",
}

# Valid US state/territory abbreviations (used to filter false positives).
_VALID_STATE_ABBRS: frozenset[str] = frozenset(_STATE_NAME_TO_ABBR.values())

# Regex to extract a 5-digit ZIP from an address string.
# Anchored to word boundaries; optionally followed by a +4 extension.
_ZIP_RE = re.compile(r'\b(\d{5})(?:-\d{4})?\b')

# Regex to find a 2-letter all-caps state abbreviation immediately before a ZIP
# or at the end of the address string.
_STATE_BEFORE_ZIP_RE = re.compile(r'\b([A-Z]{2})\s+\d{5}\b')
_STATE_AT_END_RE = re.compile(r'\b([A-Z]{2})\s*$')


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two WGS-84 points.

    Uses the haversine formula.  Pure stdlib — no external dependencies.

    Parameters
    ----------
    lat1, lon1:
        Latitude and longitude of the first point (decimal degrees).
    lat2, lon2:
        Latitude and longitude of the second point (decimal degrees).
    """
    R = 6_371.0  # Earth mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ZipValidationResult:
    """Outcome of a single ZIP/state geographic validation check.

    Attributes
    ----------
    passed:
        ``True`` when the geocoded point is consistent with the input ZIP/state.
    reason:
        Human-readable explanation when ``passed=False``, or an informational
        note (e.g. "ZIP not in centroid dataset") when passing with caveats.
    input_zip:
        5-digit ZIP extracted from the input address, or ``None``.
    input_state:
        2-letter state abbreviation extracted from the input address, or ``None``.
    distance_km:
        Haversine distance from the geocoded point to the ZIP centroid when a
        ZIP-based check was performed; ``None`` otherwise.
    max_distance_km:
        The distance threshold used for this check.
    """

    passed: bool
    reason: Optional[str]
    input_zip: Optional[str]
    input_state: Optional[str]
    distance_km: Optional[float]
    max_distance_km: float


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ZipCodeValidator:
    """Validates geocoded coordinates against the input address's ZIP or state.

    Instantiate once and reuse — the centroid dict is loaded lazily on the
    first :meth:`validate` call and cached in memory thereafter.

    If the centroid data cannot be loaded (network failure, file permission
    error, etc.) the validator disables itself: all subsequent :meth:`validate`
    calls return ``passed=True`` with an explanatory ``reason``.

    Parameters
    ----------
    cache_path:
        Where to store / read the downloaded centroid CSV.
        Defaults to ``data/zip_centroids.csv``.
    download_url:
        Override for the Gazetteer download URL (useful in tests).
    timeout_seconds:
        Network timeout for the download request.
    """

    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        download_url: str = _GAZETTEER_URL,
        timeout_seconds: float = _DOWNLOAD_TIMEOUT_SECONDS,
    ) -> None:
        self._cache_path = cache_path or _CACHE_PATH
        self._download_url = download_url
        self._timeout = timeout_seconds
        # None = not yet attempted; dict = loaded; empty dict = load failed
        self._centroids: dict[str, tuple[float, float]] | None = None
        self._load_failed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        result: GeocodeResult,
        input_address: str,
        max_distance_km: float = 20.0,
    ) -> ZipValidationResult:
        """Validate that *result* is geographically consistent with *input_address*.

        Parameters
        ----------
        result:
            The :class:`~backend.geocoding.GeocodeResult` to validate.
        input_address:
            The original address string submitted to the geocoder.
        max_distance_km:
            Maximum allowed distance (km) between the geocoded point and the
            ZIP centroid.  Default 20 km.

        Returns
        -------
        :class:`ZipValidationResult` — always returns a value, never raises.
        """
        input_zip = _extract_zip(input_address)
        input_state = _extract_state(input_address)

        # --- ZIP-based validation (primary path) ---
        if input_zip is not None:
            centroids = self._get_centroids()
            if centroids is None:
                # Load failed; disable gracefully.
                return ZipValidationResult(
                    passed=True,
                    reason="zip_validation_unavailable: centroid data could not be loaded",
                    input_zip=input_zip,
                    input_state=input_state,
                    distance_km=None,
                    max_distance_km=max_distance_km,
                )

            centroid = centroids.get(input_zip)
            if centroid is None:
                return ZipValidationResult(
                    passed=True,
                    reason=f"zip_{input_zip}_not_in_centroid_dataset: validation skipped",
                    input_zip=input_zip,
                    input_state=input_state,
                    distance_km=None,
                    max_distance_km=max_distance_km,
                )

            centroid_lat, centroid_lon = centroid
            dist_km = haversine_km(
                result.latitude, result.longitude, centroid_lat, centroid_lon
            )

            if dist_km > max_distance_km:
                return ZipValidationResult(
                    passed=False,
                    reason=(
                        f"geocoded point is {dist_km:.1f} km from ZIP {input_zip} centroid "
                        f"({centroid_lat:.4f},{centroid_lon:.4f}); "
                        f"threshold is {max_distance_km:.0f} km"
                    ),
                    input_zip=input_zip,
                    input_state=input_state,
                    distance_km=round(dist_km, 2),
                    max_distance_km=max_distance_km,
                )

            return ZipValidationResult(
                passed=True,
                reason=None,
                input_zip=input_zip,
                input_state=input_state,
                distance_km=round(dist_km, 2),
                max_distance_km=max_distance_km,
            )

        # --- State-only validation (fallback when no ZIP in input) ---
        if input_state is not None:
            result_state = _extract_state_from_result(result)
            if result_state is not None and result_state.upper() != input_state.upper():
                return ZipValidationResult(
                    passed=False,
                    reason=(
                        f"result state '{result_state}' does not match "
                        f"input state '{input_state}'"
                    ),
                    input_zip=None,
                    input_state=input_state,
                    distance_km=None,
                    max_distance_km=max_distance_km,
                )
            return ZipValidationResult(
                passed=True,
                reason=None,
                input_zip=None,
                input_state=input_state,
                distance_km=None,
                max_distance_km=max_distance_km,
            )

        # --- Cannot validate ---
        return ZipValidationResult(
            passed=True,
            reason="no_zip_or_state_in_input: cannot validate",
            input_zip=None,
            input_state=None,
            distance_km=None,
            max_distance_km=max_distance_km,
        )

    # ------------------------------------------------------------------
    # Centroid data loading
    # ------------------------------------------------------------------

    def _get_centroids(self) -> dict[str, tuple[float, float]] | None:
        """Return the centroid dict, loading from disk or downloading as needed.

        Returns ``None`` if loading has previously failed.
        """
        if self._load_failed:
            return None
        if self._centroids is not None:
            return self._centroids

        # Try loading from cached CSV first.
        if self._cache_path.exists():
            centroids = self._load_from_csv(self._cache_path)
            if centroids is not None:
                self._centroids = centroids
                LOGGER.info(
                    "zip_validator loaded %d centroids from cache %s",
                    len(centroids),
                    self._cache_path,
                )
                return self._centroids

        # Download and parse the Gazetteer ZIP.
        centroids = self._download_and_parse()
        if centroids is None:
            self._load_failed = True
            LOGGER.warning(
                "zip_validator failed to load centroid data; "
                "ZIP validation will be skipped for all subsequent calls"
            )
            return None

        self._centroids = centroids
        self._write_csv_cache(centroids)
        LOGGER.info(
            "zip_validator downloaded and cached %d centroids to %s",
            len(centroids),
            self._cache_path,
        )
        return self._centroids

    def _download_and_parse(self) -> dict[str, tuple[float, float]] | None:
        """Download the Census Gazetteer ZIP and return the parsed centroid dict."""
        LOGGER.info(
            "zip_validator downloading centroid data from %s (timeout=%.0fs)",
            self._download_url,
            self._timeout,
        )
        try:
            req = urllib.request.Request(
                self._download_url,
                headers={"User-Agent": "WildfireRiskAdvisor/0.1"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw_bytes = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            LOGGER.warning("zip_validator download_error %s", exc)
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                # The Gazetteer ZIP contains a single .txt (tab-separated) file.
                txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
                if not txt_names:
                    LOGGER.warning("zip_validator no .txt file found in downloaded ZIP")
                    return None
                with zf.open(txt_names[0]) as f:
                    content = f.read().decode("utf-8", errors="replace")
        except (zipfile.BadZipFile, KeyError, Exception) as exc:
            LOGGER.warning("zip_validator zip_parse_error %s", exc)
            return None

        return _parse_gazetteer_text(content)

    @staticmethod
    def _load_from_csv(path: Path) -> dict[str, tuple[float, float]] | None:
        """Load a previously cached centroid CSV (GEOID,lat,lon)."""
        try:
            centroids: dict[str, tuple[float, float]] = {}
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    zip_code = str(row.get("GEOID") or "").strip()
                    if len(zip_code) != 5 or not zip_code.isdigit():
                        continue
                    try:
                        lat = float(row["INTPTLAT"])
                        lon = float(row["INTPTLONG"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    centroids[zip_code] = (lat, lon)
            return centroids if centroids else None
        except OSError as exc:
            LOGGER.warning("zip_validator csv_read_error path=%s error=%s", path, exc)
            return None

    def _write_csv_cache(self, centroids: dict[str, tuple[float, float]]) -> None:
        """Write the centroid dict to the cache CSV; silently skip on failure."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["GEOID", "INTPTLAT", "INTPTLONG"])
                for zip_code, (lat, lon) in sorted(centroids.items()):
                    writer.writerow([zip_code, lat, lon])
        except OSError as exc:
            LOGGER.warning(
                "zip_validator csv_write_error path=%s error=%s",
                self._cache_path,
                exc,
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_gazetteer_text(content: str) -> dict[str, tuple[float, float]] | None:
    """Parse the tab-separated Census Gazetteer text into a centroid dict."""
    try:
        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
        centroids: dict[str, tuple[float, float]] = {}
        for row in reader:
            # Strip whitespace from keys — Census files sometimes have leading spaces.
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            zip_code = str(row.get("GEOID") or "").strip()
            if len(zip_code) != 5 or not zip_code.isdigit():
                continue
            try:
                lat = float(row["INTPTLAT"])
                lon = float(row["INTPTLONG"])
            except (KeyError, TypeError, ValueError):
                continue
            centroids[zip_code] = (lat, lon)
        return centroids if centroids else None
    except Exception as exc:
        LOGGER.warning("zip_validator gazetteer_parse_error %s", exc)
        return None


def _extract_zip(address: str) -> str | None:
    """Extract the first 5-digit US ZIP code from *address*.

    Searches from the end of the string backwards so that the ZIP in a standard
    US address format (``..., City, ST 99999``) is found before any ZIP-like
    number that might appear in a street address (e.g. ``12345 Highway 1``).

    Returns the 5-digit ZIP string, or ``None`` if none found.
    """
    # Find all matches and prefer the last one (closest to end of string).
    matches = _ZIP_RE.findall(address)
    if not matches:
        return None
    # The last match in a standard US address is the postal code.
    return matches[-1]


def _extract_state(address: str) -> str | None:
    """Extract a US state abbreviation from *address*.

    Looks for a 2-letter uppercase token immediately before a 5-digit ZIP
    (``..., WA 98862``), or at the very end of the string if no ZIP is present
    (``..., WA``).  Only returns the token if it is a known US state/territory
    abbreviation.

    Returns the 2-letter abbreviation string (uppercased), or ``None``.
    """
    # Prefer state-before-ZIP pattern.
    m = _STATE_BEFORE_ZIP_RE.search(address)
    if m:
        candidate = m.group(1).upper()
        if candidate in _VALID_STATE_ABBRS:
            return candidate

    # Fall back to state-at-end-of-string pattern.
    m = _STATE_AT_END_RE.search(address)
    if m:
        candidate = m.group(1).upper()
        if candidate in _VALID_STATE_ABBRS:
            return candidate

    return None


def _extract_state_from_result(result: GeocodeResult) -> str | None:
    """Extract a 2-letter state abbreviation from a :class:`GeocodeResult`.

    Checks, in order:

    1. ``raw_response_preview["address_components"]["state"]`` — Census style
       (already a 2-letter abbreviation).
    2. ``raw_response_preview["top_candidate"]["address"]["state"]`` — Nominatim
       style (may be a full state name like "New York").
    3. ``matched_address`` — parse any trailing 2-letter token before a ZIP.

    Returns the 2-letter abbreviation string (uppercased), or ``None``.
    """
    preview = result.raw_response_preview
    if isinstance(preview, dict):
        # --- Census: address_components.state ---
        components = preview.get("address_components")
        if isinstance(components, dict):
            raw = str(components.get("state") or "").strip().upper()
            if raw in _VALID_STATE_ABBRS:
                return raw

        # --- Nominatim: top_candidate.address.state ---
        top = preview.get("top_candidate")
        if isinstance(top, dict):
            addr = top.get("address")
            if isinstance(addr, dict):
                raw = str(addr.get("state") or "").strip()
                if raw:
                    # Try direct abbreviation first.
                    if raw.upper() in _VALID_STATE_ABBRS:
                        return raw.upper()
                    # Try full-name lookup.
                    abbr = _STATE_NAME_TO_ABBR.get(raw.lower())
                    if abbr:
                        return abbr

    # --- Fallback: parse matched_address ---
    if result.matched_address:
        candidate = _extract_state(result.matched_address)
        if candidate:
            return candidate

    return None
