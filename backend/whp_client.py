"""
Wildfire Hazard Potential (WHP) proxy client.

USFS ArcGIS API Research (2026-04-30)
======================================
Three USFS ArcGIS endpoints were tested:

  1. https://apps.fs.usda.gov/arcgis/rest/services/RDW_Wildfire?f=json
     HTTP 403 — Access forbidden. Services catalog not publicly accessible.

  2. https://apps.fs.usda.gov/arcgis/rest/services/RDW_Wildfire/WhpFinal/MapServer?f=json
     HTTP 403 — Access forbidden. WHP MapServer does not support public identify queries.

  3. https://apps.fs.usda.gov/arcgis/rest/services/RDW_Wildfire/WhpFinal/MapServer/identify?...
     HTTP 403 — Access forbidden. Point-identify operation blocked for external callers.

Conclusion: no USFS ArcGIS endpoint is publicly accessible for per-point WHP lookup.
Approach: proxy formula (source="whp_proxy") using already-computed enrichment features.

Proxy Weights and Rationale
============================
The proxy approximates WHP from features that are independently available for every
property via LANDFIRE WCS COG (national coverage) and the MTBS fire history GPKG.
Weights are approximate — tuned to match relative factor importances in the USFS WHP
methodology (see Dillon et al. 2015, https://doi.org/10.3390/f6010153).

  fuel_model_index  0.35  — dominant WHP driver (surface fuel load + type)
  canopy_cover_index 0.20 — canopy fire propagation potential
  slope_index        0.20 — rate of spread increases exponentially with slope
  aspect_score       0.10 — SW/S aspects have highest fire weather exposure
  historical_fire_score 0.15 — recurrence signal (MTBS 30-year window)

Output range: 0–100 (same scale as burn_probability_index in WildfireContext).

TODO: validate proxy output against actual USFS WHP raster values at a sample of
known-risk properties (Missoula, Winthrop, and 10+ WUI sites in high-risk zones)
once ground-truth comparison tooling is available.
"""

from __future__ import annotations

import logging
from typing import Optional

LOGGER = logging.getLogger("wildfire_app.whp_client")

# Proxy formula weights — must sum to 1.0.
_FUEL_WEIGHT = 0.35
_CANOPY_WEIGHT = 0.20
_SLOPE_WEIGHT = 0.20
_ASPECT_WEIGHT = 0.10
_FIRE_HIST_WEIGHT = 0.15

# Minimum number of non-None primary components required to produce a result.
_MIN_COMPONENTS = 3

# Source tag written into enriched_features dict for provenance tracking.
_SOURCE_TAG = "whp_proxy"


def _aspect_score(aspect_deg: Optional[float]) -> float:
    """
    Map raw aspect degrees to a 0.0–1.0 fire-exposure score.

    South-facing slopes (135–225°) receive the highest score because they receive
    more direct solar radiation, dry out faster, and have higher afternoon fire weather
    exposure. Southwest and west follow. North/east are coolest and wettest.
    """
    if aspect_deg is None:
        return 0.50  # flat terrain or unknown — neutral

    a = float(aspect_deg) % 360.0

    if 135.0 <= a < 225.0:
        return 1.0   # south-facing
    elif 225.0 <= a < 270.0:
        return 0.85  # southwest
    elif 270.0 <= a < 315.0:
        return 0.70  # west
    elif 90.0 <= a < 135.0:
        return 0.65  # southeast
    else:
        return 0.40  # east, north, northwest


def _historical_fire_score(
    fire_count_30yr: Optional[int],
    burned_within_radius: Optional[bool],
) -> float:
    """
    Map MTBS 30-year fire history to a 0.0–1.0 recurrence score.

    A property inside a prior burn perimeter is a strong predictor of future fire
    proximity; repeated burns compound the signal.
    """
    if fire_count_30yr is None:
        return 0.10  # no data — neutral (not zero, so absence doesn't mask risk)

    count = int(fire_count_30yr)
    if count >= 3:
        return 1.0
    elif count == 2:
        return 0.75
    elif count == 1:
        return 0.50
    elif burned_within_radius:
        return 0.25  # count=0 but inside a perimeter (edge case — partial overlap)
    else:
        return 0.0


class WHPClient:
    """
    Wildfire Hazard Potential proxy client.

    Computes a WHP index (0–100) from enrichment features that are already available
    for every assessment regardless of region coverage. The output is written to
    burn_probability_index in WildfireContext via the wildfire_data.py pipeline.

    The proxy is only invoked when no local WHP/burn-probability raster was found for
    the property. Local raster data retains full priority.

    No network calls are made — this client reads from the enriched_features dict
    that is assembled from LANDFIRE WCS COG, slope/aspect rasters, and MTBS GPKG
    earlier in the same pipeline pass.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._mode = "proxy"
        if enabled:
            LOGGER.info("WHPClient initialized: proxy mode, enabled=True")
        else:
            LOGGER.info("WHPClient initialized: disabled")

    def get_whp_index(
        self,
        lat: float,
        lon: float,
        enriched_features: dict,
    ) -> Optional[float]:
        """
        Return WHP index (0–100) derived from already-computed enrichment features.

        Sets enriched_features["whp_index_source"] = "whp_proxy" on success.
        Returns None if fewer than _MIN_COMPONENTS primary components are non-None.
        Never raises.
        """
        if not self.enabled:
            return None

        try:
            fuel = enriched_features.get("fuel_index")
            canopy = enriched_features.get("canopy_index")
            slope = enriched_features.get("slope_index")
            aspect_deg = enriched_features.get("aspect_degrees")
            fire_count = enriched_features.get("fire_count_30yr")
            burned = enriched_features.get("burned_within_radius")

            # Determine component availability
            # Primary components: fuel, canopy, slope, aspect, fire_history
            # (aspect and fire_history can produce non-None even from defaults,
            #  so we count them as available only when we have actual data)
            primary_available = sum([
                fuel is not None,
                canopy is not None,
                slope is not None,
                aspect_deg is not None,       # None → flat terrain default used
                fire_count is not None,        # None → neutral default used
            ])

            if primary_available < _MIN_COMPONENTS:
                LOGGER.debug(
                    "whp_client proxy insufficient inputs lat=%.4f lon=%.4f "
                    "available=%d min=%d",
                    lat, lon, primary_available, _MIN_COMPONENTS,
                )
                return None

            # Compute component scores.
            # Inputs are already normalized to 0–100 by wildfire_data._to_index().
            # Divide by 100 to bring to 0–1 for the weighted sum, then multiply back.
            fuel_c = float(fuel) / 100.0 if fuel is not None else None
            canopy_c = float(canopy) / 100.0 if canopy is not None else None
            slope_c = float(slope) / 100.0 if slope is not None else None
            aspect_c = _aspect_score(aspect_deg)
            fire_c = _historical_fire_score(fire_count, burned)

            # Weighted combination with partial-weight renormalization for missing inputs
            terms = [
                (_FUEL_WEIGHT, fuel_c),
                (_CANOPY_WEIGHT, canopy_c),
                (_SLOPE_WEIGHT, slope_c),
                (_ASPECT_WEIGHT, aspect_c),
                (_FIRE_HIST_WEIGHT, fire_c),
            ]
            numerator = sum(w * v for w, v in terms if v is not None)
            denominator = sum(w for w, v in terms if v is not None)

            if denominator <= 0.0:
                return None

            whp_0_to_1 = numerator / denominator
            # Scale to 0–100 to match burn_probability_index convention.
            whp_index = round(max(0.0, min(100.0, whp_0_to_1 * 100.0)), 1)

            enriched_features["whp_index_source"] = _SOURCE_TAG
            LOGGER.debug(
                "whp_client proxy lat=%.4f lon=%.4f whp=%.1f "
                "fuel=%.1f canopy=%.1f slope=%.1f aspect=%.2f fire=%.2f",
                lat, lon, whp_index,
                (fuel_c or 0) * 100,
                (canopy_c or 0) * 100,
                (slope_c or 0) * 100,
                aspect_c * 100,
                fire_c * 100,
            )
            return whp_index

        except Exception as exc:
            LOGGER.warning(
                "whp_client proxy_error lat=%.4f lon=%.4f error=%s",
                lat, lon, exc,
            )
            return None

    def get_whp_components(self, enriched_features: dict) -> dict:
        """
        Return component breakdown for debugging and API transparency.

        For the proxy approach, this shows each input score and its weight.
        """
        if not self.enabled:
            return {}

        fuel = enriched_features.get("fuel_index")
        canopy = enriched_features.get("canopy_index")
        slope = enriched_features.get("slope_index")
        aspect_deg = enriched_features.get("aspect_degrees")
        fire_count = enriched_features.get("fire_count_30yr")
        burned = enriched_features.get("burned_within_radius")

        return {
            "mode": "proxy",
            "components": {
                "fuel_model_index": {"input_value": fuel, "weight": _FUEL_WEIGHT},
                "canopy_cover_index": {"input_value": canopy, "weight": _CANOPY_WEIGHT},
                "slope_index": {"input_value": slope, "weight": _SLOPE_WEIGHT},
                "aspect_score": {
                    "input_degrees": aspect_deg,
                    "derived_score": _aspect_score(aspect_deg),
                    "weight": _ASPECT_WEIGHT,
                },
                "historical_fire_score": {
                    "fire_count_30yr": fire_count,
                    "burned_within_radius": burned,
                    "derived_score": _historical_fire_score(fire_count, burned),
                    "weight": _FIRE_HIST_WEIGHT,
                },
            },
            "weights_sum": (
                _FUEL_WEIGHT + _CANOPY_WEIGHT + _SLOPE_WEIGHT
                + _ASPECT_WEIGHT + _FIRE_HIST_WEIGHT
            ),
        }
