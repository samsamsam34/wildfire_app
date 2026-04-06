"""Geocode resolver tuning parameters.

All values are read from environment variables at call time so they can be
overridden per-deployment without a code change.  The previous pattern was 18
separate try/except blocks inline in the resolve function; this module
centralises them in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float_env(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ResolverConfig:
    # Conflict detection
    conflict_distance_m: float
    conflict_score_margin: float
    conflict_min_authority_gap: float
    conflict_dominant_score_margin: float

    # Scoring boosts
    in_region_boost: float
    authoritative_source_bonus: float

    # Winner selection
    clear_winner_min_margin: float
    clear_winner_min_score: float
    in_region_preference_margin: float

    # Tolerance multipliers for precision types
    centroid_tolerance_multiplier: float
    interpolated_tolerance_multiplier: float
    allow_interpolated_auto: bool

    # Token-similarity quality gates
    min_geocoder_token_similarity: float
    min_geocoder_token_coverage: float
    min_auto_candidate_score: float

    # Emergency in-region guardrail
    emergency_in_region_guardrail: bool
    emergency_min_score: float
    emergency_min_margin: float

    @classmethod
    def from_env(cls) -> "ResolverConfig":
        return cls(
            conflict_distance_m=_float_env("WF_RESOLVER_CONFLICT_DISTANCE_M", 1500.0, minimum=50.0),
            conflict_score_margin=_float_env("WF_RESOLVER_CONFLICT_SCORE_MARGIN", 18.0, minimum=1.0),
            conflict_min_authority_gap=_float_env("WF_RESOLVER_CONFLICT_MIN_AUTHORITY_GAP", 8.0, minimum=0.0),
            conflict_dominant_score_margin=_float_env("WF_RESOLVER_CONFLICT_DOMINANT_SCORE_MARGIN", 20.0, minimum=0.0),
            in_region_boost=_float_env("WF_RESOLVER_IN_REGION_BOOST", 35.0, minimum=0.0),
            authoritative_source_bonus=_float_env("WF_RESOLVER_AUTHORITATIVE_SOURCE_BONUS", 18.0, minimum=0.0),
            clear_winner_min_margin=_float_env("WF_RESOLVER_CLEAR_WINNER_MIN_MARGIN", 12.0, minimum=0.0),
            clear_winner_min_score=_float_env("WF_RESOLVER_CLEAR_WINNER_MIN_SCORE", 230.0, minimum=0.0),
            in_region_preference_margin=_float_env("WF_RESOLVER_IN_REGION_PREFERENCE_MARGIN", 18.0, minimum=0.0),
            centroid_tolerance_multiplier=_float_env("WF_RESOLVER_CENTROID_TOLERANCE_MULTIPLIER", 2.5, minimum=1.0),
            interpolated_tolerance_multiplier=_float_env(
                "WF_RESOLVER_INTERPOLATED_TOLERANCE_MULTIPLIER", 1.8, minimum=1.0
            ),
            allow_interpolated_auto=_bool_env("WF_RESOLVER_ALLOW_INTERPOLATED_AUTO", True),
            min_geocoder_token_similarity=_float_env(
                "WF_RESOLVER_MIN_GEOCODER_TOKEN_SIMILARITY", 0.55, minimum=0.0, maximum=1.0
            ),
            min_geocoder_token_coverage=_float_env(
                "WF_RESOLVER_MIN_GEOCODER_TOKEN_COVERAGE", 0.72, minimum=0.0, maximum=1.0
            ),
            min_auto_candidate_score=_float_env("WF_RESOLVER_MIN_AUTO_CANDIDATE_SCORE", 150.0, minimum=0.0),
            emergency_in_region_guardrail=_bool_env("WF_RESOLVER_EMERGENCY_IN_REGION_MEDIUM_AUTORESOLVE", True),
            emergency_min_score=_float_env("WF_RESOLVER_EMERGENCY_MIN_SCORE", 155.0, minimum=0.0),
            emergency_min_margin=_float_env("WF_RESOLVER_EMERGENCY_MIN_MARGIN", 8.0, minimum=0.0),
        )
