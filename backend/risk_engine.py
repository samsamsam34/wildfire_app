from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from backend.models import PropertyAttributes, RiskDrivers
from backend.normalization import normalize_property_attributes
from backend.scoring_config import ScoringConfig, load_scoring_config
from backend.wildfire_data import WildfireContext

# MVP insurer-oriented heuristic model for transparency and calibration workflows.
# Not a carrier-approved underwriting model.
ENVIRONMENT_SUBMODELS = {
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "slope_topography_risk",
    "ember_exposure_risk",
    "flame_contact_risk",
    "historic_fire_risk",
}

STRUCTURE_SUBMODELS = {
    "structure_vulnerability_risk",
    "defensible_space_risk",
}

REGIONAL_CONTEXT_SUBMODELS = {
    "slope_topography_risk",
    "historic_fire_risk",
}

PROPERTY_SURROUNDINGS_SUBMODELS = {
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "flame_contact_risk",
}

STRUCTURE_SPECIFIC_SUBMODELS = {
    "ember_exposure_risk",
    "structure_vulnerability_risk",
    "defensible_space_risk",
}

GEOMETRY_SENSITIVE_SUBMODELS = {
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "flame_contact_risk",
    "ember_exposure_risk",
    "defensible_space_risk",
}


@dataclass
class SubmodelResult:
    score: float
    explanation: str
    key_inputs: Dict[str, object]
    assumptions: List[str] = field(default_factory=list)
    raw_score: float | None = None
    clamped_score: float | None = None


@dataclass
class ReadinessRuleResult:
    insurance_readiness_score: float
    readiness_factors: List[dict]
    readiness_blockers: List[str]
    readiness_penalties: Dict[str, float]
    readiness_summary: str


@dataclass
class RiskComputation:
    total_score: float
    drivers: RiskDrivers
    assumptions: List[str]
    submodel_scores: Dict[str, SubmodelResult]
    weighted_contributions: Dict[str, dict]
    observed_factor_count: int = 0
    missing_factor_count: int = 0
    fallback_factor_count: int = 0
    observed_weight_fraction: float = 0.0
    fallback_dominance_ratio: float = 0.0
    fallback_weight_fraction: float = 0.0
    uncertainty_penalty: float = 0.0
    regional_context_score: float = 0.0
    property_surroundings_score: float = 0.0
    structure_specific_score: float = 0.0
    component_weight_fractions: Dict[str, float] = field(default_factory=dict)
    geometry_basis: str = "point"
    observed_feature_count: int = 0
    inferred_feature_count: int = 0
    fallback_feature_count: int = 0
    geometry_quality_score: float = 0.0
    regional_context_coverage_score: float = 0.0
    property_specificity_score: float = 0.0
    structure_data_completeness: float = 0.0
    structure_assumption_mode: str = "unknown"
    structure_score_confidence: float = 0.0
    access_provisional: bool = True
    access_note: str = (
        "Access exposure is derived separately from wildfire total scoring and may be limited by available road-network data."
    )


class RiskEngine:
    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or load_scoring_config()

    def geocode_stub(self, address: str) -> Tuple[float, float]:
        digest = hashlib.md5(address.strip().lower().encode("utf-8")).hexdigest()
        lat_seed = int(digest[:8], 16) / 0xFFFFFFFF
        lon_seed = int(digest[8:16], 16) / 0xFFFFFFFF
        latitude = 32.5 + (lat_seed * 9.5)
        longitude = -124.3 + (lon_seed * 10.0)
        return round(latitude, 6), round(longitude, 6)

    def _build_submodels(self, attrs: PropertyAttributes, context: WildfireContext) -> Dict[str, SubmodelResult]:
        attrs = normalize_property_attributes(attrs)
        submodels: Dict[str, SubmodelResult] = {}

        def clamp_score(value: float | None) -> float:
            if value is None:
                return 0.0
            return round(max(0.0, min(100.0, float(value))), 2)

        roof = (attrs.roof_type or "unknown").lower()
        vent = (attrs.vent_type or "unknown").lower()
        defensible_ft = attrs.defensible_space_ft
        ring_metrics = context.structure_ring_metrics or {}
        property_level_context = (
            context.property_level_context
            if isinstance(context.property_level_context, dict)
            else {}
        )
        context_assumptions: List[str] = []

        def ctx_metric(value: float | None, note: str) -> float | None:
            if value is None:
                context_assumptions.append(note)
                return None
            return float(value)

        def weighted_score(components: List[tuple[float, float | None, str]], assumptions: List[str]) -> float:
            available: List[tuple[float, float]] = []
            for weight, value, note in components:
                if value is None:
                    if note:
                        assumptions.append(note)
                    continue
                available.append((weight, float(value)))
            if not available:
                return 0.0
            numerator = sum(weight * value for weight, value in available)
            denominator = sum(weight for weight, _ in available)
            if denominator <= 0:
                return 0.0
            return numerator / denominator

        def ring_density(ring_key: str) -> float | None:
            return ring_metric(ring_key, "vegetation_density")

        def ring_metric(ring_key: str, metric_key: str) -> float | None:
            alias = ring_key.replace("ring_", "zone_")
            metrics = ring_metrics.get(ring_key) or ring_metrics.get(alias) or {}
            value = metrics.get(metric_key) if isinstance(metrics, dict) else None
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        ring_0_5_density = ring_density("ring_0_5_ft")
        ring_5_30_density = ring_density("ring_5_30_ft")
        ring_30_100_density = ring_density("ring_30_100_ft")
        ring_100_300_density = ring_density("ring_100_300_ft")
        ring_0_5_local_percentile = ring_metric("ring_0_5_ft", "imagery_vegetation_cover_local_percentile")
        ring_5_30_local_percentile = ring_metric("ring_5_30_ft", "imagery_vegetation_cover_local_percentile")
        ring_5_30_continuity = ring_metric("ring_5_30_ft", "imagery_vegetation_continuity_pct")
        ring_30_100_continuity = ring_metric("ring_30_100_ft", "imagery_vegetation_continuity_pct")
        ring_0_5_canopy_proxy = ring_metric("ring_0_5_ft", "imagery_canopy_proxy_pct")
        ring_5_30_canopy_proxy = ring_metric("ring_5_30_ft", "imagery_canopy_proxy_pct")
        nearest_vegetation_distance_ft = None
        try:
            raw_distance = (context.property_level_context or {}).get("nearest_vegetation_distance_ft")
            if raw_distance is not None:
                nearest_vegetation_distance_ft = float(raw_distance)
        except (TypeError, ValueError, AttributeError):
            nearest_vegetation_distance_ft = None

        def _to_float(value: object) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _clamp_percent(value: float | None) -> float | None:
            if value is None:
                return None
            return max(0.0, min(100.0, float(value)))

        def _nonlinear_close_in_vegetation_penalty(value: float | None) -> float | None:
            """Apply a stronger monotonic penalty for 0-5 ft vegetation loading."""
            clamped = _clamp_percent(value)
            if clamped is None:
                return None
            if clamped <= 20.0:
                transformed = clamped * 0.60
            elif clamped <= 40.0:
                transformed = 12.0 + ((clamped - 20.0) * 1.35)
            elif clamped <= 65.0:
                transformed = 39.0 + ((clamped - 40.0) * 1.55)
            else:
                transformed = 77.75 + ((clamped - 65.0) * 0.90)
            return round(max(0.0, min(100.0, transformed)), 1)

        near_structure_vegetation_0_5_pct = _to_float(
            property_level_context.get("near_structure_vegetation_0_5_pct")
        )
        near_structure_vegetation_5_30_pct = _to_float(
            property_level_context.get("near_structure_vegetation_5_30_pct")
        )
        vegetation_edge_directional_concentration_pct = _to_float(
            property_level_context.get("vegetation_edge_directional_concentration_pct")
        )
        canopy_dense_fuel_asymmetry_pct = _to_float(
            property_level_context.get("canopy_dense_fuel_asymmetry_pct")
        )
        nearest_continuous_vegetation_distance_ft = _to_float(
            property_level_context.get("nearest_continuous_vegetation_distance_ft")
        )
        vegetation_directional_precision = str(
            property_level_context.get("vegetation_directional_precision") or ""
        ).strip().lower()
        vegetation_directional_precision_score = _to_float(
            property_level_context.get("vegetation_directional_precision_score")
        )
        if vegetation_directional_precision_score is None:
            if vegetation_directional_precision == "footprint_boundary":
                vegetation_directional_precision_score = 0.90
            elif vegetation_directional_precision == "point_proxy":
                vegetation_directional_precision_score = 0.45
            else:
                vegetation_directional_precision_score = 0.65
        vegetation_directional_precision_score = max(
            0.0,
            min(1.0, float(vegetation_directional_precision_score)),
        )
        vegetation_precision_multiplier = max(
            0.45,
            min(1.0, 0.55 + (0.45 * vegetation_directional_precision_score)),
        )
        if vegetation_directional_precision == "point_proxy":
            context_assumptions.append(
                "Directional near-structure vegetation features used point-proxy sampling with lower precision."
            )
        if vegetation_directional_precision_score < 0.60:
            context_assumptions.append(
                "Near-structure directional vegetation evidence had low geometry precision; vegetation extremes were damped toward conservative values."
            )

        def _precision_adjust(value: float | None) -> float | None:
            if value is None:
                return None
            anchor = 50.0
            return round(
                (float(value) * vegetation_precision_multiplier)
                + (anchor * (1.0 - vegetation_precision_multiplier)),
                2,
            )
        near_structure_connectivity_index = _to_float(
            property_level_context.get("near_structure_connectivity_index")
        )
        canopy_adjacency_proxy_pct = _to_float(
            property_level_context.get("canopy_adjacency_proxy_pct")
        )
        vegetation_continuity_proxy_pct = _to_float(
            property_level_context.get("vegetation_continuity_proxy_pct")
        )
        nearest_high_fuel_patch_distance_ft = _to_float(
            property_level_context.get("nearest_high_fuel_patch_distance_ft")
        )
        nearest_high_fuel_patch_index = (
            round(max(0.0, min(100.0, 100.0 - (nearest_high_fuel_patch_distance_ft / 300.0) * 100.0)), 1)
            if nearest_high_fuel_patch_distance_ft is not None
            else None
        )
        nearest_continuous_vegetation_index = (
            round(
                max(0.0, min(100.0, 100.0 - ((nearest_continuous_vegetation_distance_ft / 120.0) * 100.0))),
                1,
            )
            if nearest_continuous_vegetation_distance_ft is not None
            else None
        )
        directional_concentration_index = _precision_adjust(
            _clamp_percent(vegetation_edge_directional_concentration_pct)
        )
        canopy_dense_fuel_asymmetry_index = _precision_adjust(
            _clamp_percent(canopy_dense_fuel_asymmetry_pct)
        )

        neighboring_structures = property_level_context.get("neighboring_structure_metrics") or {}
        nearby_structure_count_100 = _to_float((neighboring_structures or {}).get("nearby_structure_count_100_ft"))
        nearby_structure_count_300 = _to_float((neighboring_structures or {}).get("nearby_structure_count_300_ft"))
        nearest_structure_distance_ft = _to_float(
            (neighboring_structures or {}).get("nearest_structure_distance_ft")
        )
        if nearest_structure_distance_ft is None:
            nearest_structure_distance_ft = _to_float(
                (neighboring_structures or {}).get("distance_to_nearest_structure_ft")
            )
        if nearest_structure_distance_ft is None:
            nearest_structure_distance_ft = _to_float(
                property_level_context.get("nearest_structure_distance_ft")
            )
        distance_to_nearest_structure_ft = _to_float(
            property_level_context.get("distance_to_nearest_structure_ft")
        )
        if nearest_structure_distance_ft is None and distance_to_nearest_structure_ft is not None:
            nearest_structure_distance_ft = distance_to_nearest_structure_ft
        if distance_to_nearest_structure_ft is None and nearest_structure_distance_ft is not None:
            distance_to_nearest_structure_ft = nearest_structure_distance_ft
        structure_to_structure_exposure_index = None
        if nearby_structure_count_100 is not None or nearby_structure_count_300 is not None:
            c100 = nearby_structure_count_100 or 0.0
            c300 = nearby_structure_count_300 or 0.0
            structure_to_structure_exposure_index = round(min(100.0, (c100 * 14.0) + (c300 * 2.5)), 1)
        nearest_structure_proximity_index = None
        if nearest_structure_distance_ft is not None:
            nearest_structure_proximity_index = round(
                max(0.0, min(100.0, 100.0 - ((nearest_structure_distance_ft / 300.0) * 100.0))),
                1,
            )
        nearest_structure_isolation_index = None
        if nearest_structure_distance_ft is not None:
            nearest_structure_isolation_index = round(
                max(0.0, min(100.0, (nearest_structure_distance_ft / 300.0) * 100.0)),
                1,
            )
        local_structure_density_index = None
        if nearby_structure_count_100 is not None or nearby_structure_count_300 is not None:
            c100 = nearby_structure_count_100 or 0.0
            c300 = nearby_structure_count_300 or 0.0
            # Normalize counts into an interpretable 0-100 proxy that can vary
            # even when explicit structure attributes are missing.
            local_structure_density_index = round(
                min(
                    100.0,
                    ((min(c100, 8.0) / 8.0) * 70.0) + ((min(c300, 24.0) / 24.0) * 30.0),
                ),
                1,
            )
        structure_density_proxy_index = _to_float(
            property_level_context.get("structure_density")
        )
        if structure_density_proxy_index is None:
            structure_density_proxy_index = _to_float(
                property_level_context.get("structure_density_proxy")
            )
        if structure_density_proxy_index is None:
            structure_density_proxy_index = local_structure_density_index
        if structure_density_proxy_index is not None:
            structure_density_proxy_index = round(max(0.0, min(100.0, float(structure_density_proxy_index))), 1)

        clustering_index = _to_float(property_level_context.get("clustering_index"))
        if clustering_index is None and structure_density_proxy_index is not None:
            prox = nearest_structure_proximity_index if nearest_structure_proximity_index is not None else 0.0
            clustering_index = round(
                max(0.0, min(100.0, (0.70 * float(structure_density_proxy_index)) + (0.30 * float(prox)))),
                1,
            )

        if near_structure_connectivity_index is None:
            continuity_base = _clamp_percent(ring_5_30_continuity)
            if continuity_base is None:
                continuity_base = _clamp_percent(vegetation_continuity_proxy_pct)
            density_bridge_terms = [v for v in (_clamp_percent(ring_0_5_density), _clamp_percent(ring_5_30_density)) if v is not None]
            density_bridge = (sum(density_bridge_terms) / len(density_bridge_terms)) if density_bridge_terms else None
            canopy_bridge = _clamp_percent(canopy_adjacency_proxy_pct)
            if canopy_bridge is None:
                canopy_bridge = _clamp_percent(ring_0_5_canopy_proxy)
            if continuity_base is not None and density_bridge is not None:
                near_structure_connectivity_index = round(
                    max(
                        0.0,
                        min(
                            100.0,
                            (0.55 * continuity_base)
                            + (0.30 * density_bridge)
                            + (0.15 * (canopy_bridge if canopy_bridge is not None else density_bridge)),
                        ),
                    ),
                    1,
                )
            elif continuity_base is not None:
                near_structure_connectivity_index = round(
                    max(
                        0.0,
                        min(
                            100.0,
                            (0.78 * continuity_base)
                            + (0.22 * (canopy_bridge if canopy_bridge is not None else continuity_base)),
                        ),
                    ),
                    1,
                )
            elif density_bridge is not None:
                near_structure_connectivity_index = round(
                    max(
                        0.0,
                        min(
                            100.0,
                            (0.72 * density_bridge)
                            + (0.28 * (canopy_bridge if canopy_bridge is not None else density_bridge)),
                        ),
                    ),
                    1,
                )

        available_ring_densities = [
            density
            for density in [ring_0_5_density, ring_5_30_density, ring_30_100_density, ring_100_300_density]
            if density is not None
        ]
        ring_density_average = (
            round(sum(available_ring_densities) / len(available_ring_densities), 1)
            if available_ring_densities
            else None
        )
        burn_probability_index = ctx_metric(
            context.burn_probability_index,
            "Burn probability context missing; scoring used conservative fallback.",
        )
        hazard_severity_index = ctx_metric(
            context.hazard_severity_index,
            "Hazard severity context missing; scoring used conservative fallback.",
        )
        fuel_index = ctx_metric(
            context.fuel_index,
            "Fuel model context missing; scoring used conservative fallback.",
        )
        wildland_distance_index = ctx_metric(
            context.wildland_distance_index,
            "Wildland distance context missing; scoring used conservative fallback.",
        )
        canopy_index = ctx_metric(
            context.canopy_index,
            "Canopy context missing; scoring used conservative fallback.",
        )
        moisture_index = ctx_metric(
            context.moisture_index,
            "Moisture context missing; scoring used conservative fallback.",
        )
        slope_index = ctx_metric(
            context.slope_index,
            "Slope context missing; scoring used conservative fallback.",
        )
        aspect_index = ctx_metric(
            context.aspect_index,
            "Aspect context missing; scoring used conservative fallback.",
        )
        historic_fire_index = ctx_metric(
            context.historic_fire_index,
            "Historic fire context missing; scoring used conservative fallback.",
        )

        roof_ignition = None
        if roof in {"wood", "untreated wood shake"}:
            roof_ignition = 75.0
        elif roof in {"class a", "metal", "tile", "composite"}:
            roof_ignition = 25.0

        vent_ignition = None
        if attrs.vent_type is not None:
            vent_ignition = 20.0 if "ember" in vent else 65.0

        ember_assumptions: List[str] = []
        if attrs.roof_type is None:
            ember_assumptions.append("Roof type missing; ember model excludes roof material contribution.")
        if attrs.vent_type is None:
            ember_assumptions.append("Vent type missing; ember model excludes vent contribution.")

        ember_score = weighted_score(
            [
                (0.31, burn_probability_index, "Burn probability unavailable for ember model."),
                (0.21, hazard_severity_index, "Hazard severity unavailable for ember model."),
                (0.18, vent_ignition, "Vent type unavailable for ember model."),
                (0.15, roof_ignition, "Roof type unavailable for ember model."),
                (0.15, structure_to_structure_exposure_index, "Neighbor structure density unavailable for ember model."),
            ],
            ember_assumptions,
        )
        ember_clamped = clamp_score(ember_score)
        submodels["ember_exposure_risk"] = SubmodelResult(
            score=ember_clamped,
            explanation="Ember exposure reflects ember storm likelihood and structure ember vulnerability.",
            key_inputs={
                "burn_probability": burn_probability_index,
                "hazard_severity": hazard_severity_index,
                "roof_ignition_proxy": roof_ignition,
                "vent_ignition_proxy": vent_ignition,
                "structure_to_structure_exposure_index": structure_to_structure_exposure_index,
            },
            assumptions=ember_assumptions + context_assumptions,
            raw_score=round(float(ember_score), 4),
            clamped_score=ember_clamped,
        )

        flame_assumptions: List[str] = []
        if attrs.defensible_space_ft is None:
            flame_assumptions.append("Defensible space missing; flame-contact model excludes defensible-space contribution.")
        if ring_0_5_density is None or ring_5_30_density is None:
            flame_assumptions.append("Structure-ring vegetation metrics unavailable; flame-contact model used point-based vegetation context.")

        close_in_veg_pressure_raw = (
            near_structure_vegetation_0_5_pct
            if near_structure_vegetation_0_5_pct is not None
            else (ring_0_5_density if ring_0_5_density is not None else canopy_index)
        )
        close_in_veg_pressure = _precision_adjust(close_in_veg_pressure_raw)
        zone1_veg_pressure_raw = (
            near_structure_vegetation_5_30_pct
            if near_structure_vegetation_5_30_pct is not None
            else (ring_5_30_density if ring_5_30_density is not None else fuel_index)
        )
        zone1_veg_pressure = _precision_adjust(zone1_veg_pressure_raw)
        near_structure_continuity = (
            near_structure_connectivity_index
            if near_structure_connectivity_index is not None
            else (
                ring_5_30_continuity
                if ring_5_30_continuity is not None
                else vegetation_continuity_proxy_pct
            )
        )
        near_structure_continuity = _precision_adjust(near_structure_continuity)
        close_in_zone_nonlinear_penalty = _nonlinear_close_in_vegetation_penalty(close_in_veg_pressure)
        defensible_component = (
            max(0.0, min(100.0, 100.0 - defensible_ft * 2.2))
            if attrs.defensible_space_ft is not None
            else None
        )
        near_ring_component = None
        near_terms: list[tuple[float, float]] = []
        if close_in_zone_nonlinear_penalty is not None:
            near_terms.append((0.44, close_in_zone_nonlinear_penalty))
        if zone1_veg_pressure is not None:
            near_terms.append((0.17, zone1_veg_pressure))
        if near_structure_continuity is not None:
            near_terms.append((0.14, near_structure_continuity))
        if canopy_adjacency_proxy_pct is not None:
            near_terms.append((0.07, _precision_adjust(canopy_adjacency_proxy_pct) or canopy_adjacency_proxy_pct))
        elif ring_0_5_canopy_proxy is not None:
            near_terms.append((0.07, _precision_adjust(ring_0_5_canopy_proxy) or ring_0_5_canopy_proxy))
        if directional_concentration_index is not None:
            near_terms.append((0.08, directional_concentration_index))
        if canopy_dense_fuel_asymmetry_index is not None:
            near_terms.append((0.05, canopy_dense_fuel_asymmetry_index))
        if nearest_continuous_vegetation_index is not None:
            near_terms.append((0.05, nearest_continuous_vegetation_index))
        if near_terms:
            num = sum(weight * value for weight, value in near_terms)
            den = sum(weight for weight, _ in near_terms)
            near_ring_component = num / den if den > 0 else None

        flame_score = weighted_score(
            [
                (0.14, fuel_index, "Fuel model unavailable for flame-contact model."),
                (0.10, wildland_distance_index, "Wildland distance unavailable for flame-contact model."),
                (0.12, defensible_component, "Defensible space unavailable for flame-contact model."),
                (0.56, near_ring_component, "Near-structure vegetation unavailable for flame-contact model."),
                (0.08, nearest_high_fuel_patch_index, "Nearest high-fuel patch proxy unavailable for flame-contact model."),
            ],
            flame_assumptions,
        )
        flame_clamped = clamp_score(flame_score)
        submodels["flame_contact_risk"] = SubmodelResult(
            score=flame_clamped,
            explanation="Flame-contact risk reflects near-structure fuels and vegetation continuity.",
            key_inputs={
                "fuel_index": fuel_index,
                "wildland_distance_index": wildland_distance_index,
                "defensible_space_ft": defensible_ft,
                "ring_0_5_ft_vegetation_density": ring_0_5_density,
                "ring_5_30_ft_vegetation_density": ring_5_30_density,
                "ring_30_100_ft_vegetation_density": ring_30_100_density,
                "near_structure_vegetation_0_5_pct": near_structure_vegetation_0_5_pct,
                "near_structure_vegetation_5_30_pct": near_structure_vegetation_5_30_pct,
                "ring_0_5_nonlinear_penalty_index": close_in_zone_nonlinear_penalty,
                "canopy_adjacency_proxy_pct": canopy_adjacency_proxy_pct,
                "vegetation_edge_directional_concentration_pct": vegetation_edge_directional_concentration_pct,
                "canopy_dense_fuel_asymmetry_pct": canopy_dense_fuel_asymmetry_pct,
                "near_structure_connectivity_index": near_structure_connectivity_index,
                "vegetation_continuity_proxy_pct": near_structure_continuity,
                "nearest_high_fuel_patch_index": nearest_high_fuel_patch_index,
                "nearest_vegetation_distance_ft": nearest_vegetation_distance_ft,
                "nearest_continuous_vegetation_distance_ft": nearest_continuous_vegetation_distance_ft,
                "nearest_continuous_vegetation_index": nearest_continuous_vegetation_index,
                "vegetation_directional_precision": vegetation_directional_precision,
                "vegetation_directional_precision_score": vegetation_directional_precision_score,
            },
            assumptions=flame_assumptions + context_assumptions,
            raw_score=round(float(flame_score), 4),
            clamped_score=flame_clamped,
        )

        slope_assumptions: List[str] = []
        slope_score = weighted_score(
            [
                (0.70, slope_index, "Slope input unavailable for topography model."),
                (0.30, aspect_index, "Aspect input unavailable for topography model."),
            ],
            slope_assumptions,
        )
        slope_clamped = clamp_score(slope_score)
        submodels["slope_topography_risk"] = SubmodelResult(
            score=slope_clamped,
            explanation="Slope/topography risk captures terrain-driven spread amplification.",
            key_inputs={"slope_index": slope_index, "aspect_index": aspect_index},
            assumptions=slope_assumptions + list(context_assumptions),
            raw_score=round(float(slope_score), 4),
            clamped_score=slope_clamped,
        )

        fuel_proximity_assumptions: List[str] = []
        if ring_30_100_density is None and ring_100_300_density is None:
            fuel_proximity_assumptions.append(
                "Structure-ring vegetation metrics unavailable for 30-300 ft; fuel proximity used point-based distance index."
            )

        outer_ring_pressure_values = [d for d in [ring_30_100_density, ring_100_300_density] if d is not None]
        outer_ring_pressure = (
            sum(outer_ring_pressure_values) / len(outer_ring_pressure_values)
            if outer_ring_pressure_values
            else canopy_index
        )
        outer_continuity_values = [
            v for v in [ring_30_100_continuity, vegetation_continuity_proxy_pct] if v is not None
        ]
        outer_continuity_pressure = (
            sum(outer_continuity_values) / len(outer_continuity_values)
            if outer_continuity_values
            else None
        )
        fuel_proximity_score = weighted_score(
            [
                (0.45, wildland_distance_index, "Wildland distance unavailable for fuel proximity model."),
                (0.22, outer_ring_pressure, "30-300 ft ring/cover unavailable for fuel proximity model."),
                (0.12, ring_0_5_density, "Immediate ring density unavailable for fuel-proximity refinement."),
                (0.11, outer_continuity_pressure, "Outer-ring vegetation continuity unavailable for fuel proximity model."),
                (0.10, nearest_high_fuel_patch_index, "Nearest high-fuel patch proxy unavailable for fuel proximity model."),
            ],
            fuel_proximity_assumptions,
        )
        fuel_proximity_clamped = clamp_score(fuel_proximity_score)
        submodels["fuel_proximity_risk"] = SubmodelResult(
            score=fuel_proximity_clamped,
            explanation="Fuel proximity risk reflects distance to contiguous wildland vegetation.",
            key_inputs={
                "wildland_distance_index": wildland_distance_index,
                "ring_30_100_ft_vegetation_density": ring_30_100_density,
                "ring_100_300_ft_vegetation_density": ring_100_300_density,
                "outer_ring_continuity_proxy": outer_continuity_pressure,
                "nearest_high_fuel_patch_index": nearest_high_fuel_patch_index,
            },
            assumptions=fuel_proximity_assumptions + context_assumptions,
            raw_score=round(float(fuel_proximity_score), 4),
            clamped_score=fuel_proximity_clamped,
        )

        vegetation_assumptions: List[str] = []
        if ring_density_average is None:
            vegetation_assumptions.append(
                "Structure-ring vegetation metrics unavailable; vegetation intensity used 100m point-neighborhood context."
            )
        near_structure_specific_available = any(
            value is not None
            for value in (
                ring_0_5_density,
                ring_5_30_density,
                near_structure_vegetation_0_5_pct,
                near_structure_vegetation_5_30_pct,
                near_structure_connectivity_index,
                directional_concentration_index,
                canopy_dense_fuel_asymmetry_index,
                nearest_continuous_vegetation_index,
            )
        )
        if near_structure_specific_available:
            vegetation_assumptions.append(
                "Near-structure vegetation signals were prioritized over coarse regional canopy/fuel averages."
            )
        structure_ring_veg = ring_density_average if ring_density_average is not None else canopy_index
        continuity_pressure = near_structure_continuity
        local_percentile_pressure = (
            ring_0_5_local_percentile
            if ring_0_5_local_percentile is not None
            else ring_5_30_local_percentile
        )
        local_percentile_pressure = _precision_adjust(local_percentile_pressure)
        if near_structure_specific_available:
            vegetation_score = weighted_score(
                [
                    (0.34, close_in_zone_nonlinear_penalty, "Immediate 0-5 ft vegetation unavailable for vegetation intensity model."),
                    (0.22, zone1_veg_pressure, "5-30 ft vegetation unavailable for vegetation intensity model."),
                    (0.14, continuity_pressure, "Near-structure connectivity unavailable for vegetation intensity model."),
                    (0.08, local_percentile_pressure, "Local vegetation percentile unavailable for vegetation intensity model."),
                    (0.08, directional_concentration_index, "Directional vegetation concentration unavailable for vegetation intensity model."),
                    (0.08, canopy_dense_fuel_asymmetry_index, "Canopy/dense-fuel asymmetry unavailable for vegetation intensity model."),
                    (0.06, nearest_continuous_vegetation_index, "Nearest continuous vegetation distance unavailable for vegetation intensity model."),
                    (0.04, moisture_index, "Moisture input unavailable for vegetation intensity model."),
                ],
                vegetation_assumptions,
            )
        else:
            vegetation_score = weighted_score(
                [
                    (0.20, fuel_index, "Fuel model unavailable for vegetation intensity model."),
                    (0.16, canopy_index, "Canopy cover unavailable for vegetation intensity model."),
                    (0.16, moisture_index, "Moisture input unavailable for vegetation intensity model."),
                    (0.28, structure_ring_veg, "Ring vegetation unavailable for vegetation intensity model."),
                    (0.12, continuity_pressure, "Vegetation continuity proxy unavailable for vegetation intensity model."),
                    (0.08, local_percentile_pressure, "Local vegetation percentile unavailable for vegetation intensity model."),
                ],
                vegetation_assumptions,
            )
        vegetation_clamped = clamp_score(vegetation_score)
        submodels["vegetation_intensity_risk"] = SubmodelResult(
            score=vegetation_clamped,
            explanation="Vegetation intensity risk captures fuel loading, canopy continuity, and dryness.",
            key_inputs={
                "fuel_index": fuel_index,
                "canopy_index": canopy_index,
                "moisture_index": moisture_index,
                "ring_vegetation_density_avg": ring_density_average,
                "ring_0_5_ft_vegetation_density": ring_0_5_density,
                "ring_5_30_ft_vegetation_density": ring_5_30_density,
                "ring_0_5_nonlinear_penalty_index": close_in_zone_nonlinear_penalty,
                "near_structure_connectivity_index": near_structure_connectivity_index,
                "near_structure_vegetation_5_30_pct": near_structure_vegetation_5_30_pct,
                "vegetation_continuity_proxy_pct": continuity_pressure,
                "ring_0_5_vegetation_local_percentile": local_percentile_pressure,
                "vegetation_edge_directional_concentration_pct": vegetation_edge_directional_concentration_pct,
                "canopy_dense_fuel_asymmetry_pct": canopy_dense_fuel_asymmetry_pct,
                "nearest_continuous_vegetation_distance_ft": nearest_continuous_vegetation_distance_ft,
                "nearest_continuous_vegetation_index": nearest_continuous_vegetation_index,
                "vegetation_directional_precision": vegetation_directional_precision,
                "vegetation_directional_precision_score": vegetation_directional_precision_score,
            },
            assumptions=vegetation_assumptions + context_assumptions,
            raw_score=round(float(vegetation_score), 4),
            clamped_score=vegetation_clamped,
        )

        historic_assumptions: List[str] = []
        if historic_fire_index is None:
            historic_assumptions.append("Historic fire recurrence unavailable for this location.")
        historic_raw = 0.0 if historic_fire_index is None else float(historic_fire_index)
        historic_clamped = clamp_score(historic_raw)
        submodels["historic_fire_risk"] = SubmodelResult(
            score=historic_clamped,
            explanation="Historic fire risk reflects nearby fire recurrence and perimeter history.",
            key_inputs={"historic_fire_index": historic_fire_index},
            assumptions=historic_assumptions + list(context_assumptions),
            raw_score=round(historic_raw, 4),
            clamped_score=historic_clamped,
        )

        structure_assumptions: List[str] = []
        construction_risk = None
        building_age_proxy_year = _to_float((context.property_level_context or {}).get("building_age_proxy_year"))
        building_age_material_proxy_risk = _to_float(
            (context.property_level_context or {}).get("building_age_material_proxy_risk")
        )
        window = (attrs.window_type or "unknown").lower()
        window_ignition = None
        if attrs.window_type is not None:
            if "tempered" in window:
                window_ignition = 26.0
            elif "dual" in window or "double" in window:
                window_ignition = 34.0
            else:
                window_ignition = 58.0

        def _construction_risk_from_year(year_value: int | float | None) -> float | None:
            if year_value is None:
                return None
            year_int = int(max(1900, min(2028, round(float(year_value)))))
            if year_int >= 2015:
                return 30.0
            if year_int >= 2008:
                return 42.0
            # Continuous aging risk so older stock does not collapse into one bucket.
            years_pre_2008 = max(0, 2008 - year_int)
            return min(75.0, 50.0 + (years_pre_2008 * 0.55))

        if attrs.construction_year is None:
            if building_age_material_proxy_risk is not None:
                construction_risk = max(0.0, min(100.0, float(building_age_material_proxy_risk)))
                structure_assumptions.append(
                    "Construction year missing; used proxy age/material risk estimate from local structure context."
                )
            elif building_age_proxy_year is not None:
                construction_risk = _construction_risk_from_year(building_age_proxy_year)
                if construction_risk is not None:
                    structure_assumptions.append(
                        "Construction year missing; using neighborhood age proxy year for material-era vulnerability."
                    )
            elif (
                structure_density_proxy_index is not None
                or nearest_structure_proximity_index is not None
                or clustering_index is not None
            ):
                proxy_terms: list[tuple[float, float]] = []
                if structure_density_proxy_index is not None:
                    proxy_terms.append((0.52, 42.0 + (structure_density_proxy_index * 0.34)))
                if nearest_structure_isolation_index is not None:
                    proxy_terms.append((0.30, 36.0 + (nearest_structure_isolation_index * 0.30)))
                if clustering_index is not None:
                    proxy_terms.append((0.18, 40.0 + (clustering_index * 0.24)))
                if proxy_terms:
                    proxy_num = sum(weight * value for weight, value in proxy_terms)
                    proxy_den = sum(weight for weight, _ in proxy_terms)
                    construction_risk = max(0.0, min(100.0, proxy_num / max(proxy_den, 1e-6)))
                    structure_assumptions.append(
                        "Construction year missing; applied neighborhood building-pattern proxy for age/material risk."
                    )
            else:
                structure_assumptions.append(
                    "Construction year missing; structure vulnerability model excludes age contribution."
                )
        else:
            construction_risk = _construction_risk_from_year(attrs.construction_year)

        structure_observed_fields: list[str] = []
        structure_inferred_fields: list[str] = []
        structure_defaulted_fields: list[str] = []
        if attrs.roof_type is not None:
            structure_observed_fields.append("roof_type")
        else:
            structure_defaulted_fields.append("roof_type")
        if attrs.vent_type is not None:
            structure_observed_fields.append("vent_type")
        else:
            structure_defaulted_fields.append("vent_type")
        if attrs.window_type is not None:
            structure_observed_fields.append("window_type")
        else:
            structure_defaulted_fields.append("window_type")
        if attrs.construction_year is not None:
            structure_observed_fields.append("construction_year")
        elif construction_risk is not None:
            structure_inferred_fields.append("construction_year")
        else:
            structure_defaulted_fields.append("construction_year")

        structure_completeness_units = float(len(structure_observed_fields)) + (0.5 * float(len(structure_inferred_fields)))
        structure_data_completeness = max(0.0, min(100.0, (structure_completeness_units / 4.0) * 100.0))
        if len(structure_observed_fields) >= 3 and not structure_defaulted_fields:
            structure_assumption_mode = "observed"
        elif len(structure_observed_fields) == 0 and len(structure_inferred_fields) == 0:
            structure_assumption_mode = "default_assumed"
        else:
            structure_assumption_mode = "mixed"
        structure_geometry_confidence = _to_float(property_level_context.get("structure_geometry_confidence"))
        if structure_geometry_confidence is None:
            structure_geometry_confidence = 0.55
        structure_geometry_confidence = max(0.0, min(1.0, float(structure_geometry_confidence)))
        structure_score_confidence = (0.85 * structure_data_completeness) + (15.0 * structure_geometry_confidence)
        if structure_assumption_mode == "default_assumed":
            structure_score_confidence = min(structure_score_confidence, 32.0)
            structure_assumptions.append(
                "Structure details are mostly unknown; structure vulnerability used conservative neutral assumptions."
            )
        elif len(structure_observed_fields) <= 1 and len(structure_defaulted_fields) >= 2:
            structure_score_confidence = min(structure_score_confidence, 56.0)
            structure_assumptions.append(
                "Structure details are sparse; structure vulnerability confidence is reduced."
            )
        if attrs.roof_type is None and attrs.vent_type is None:
            structure_score_confidence = min(structure_score_confidence, 52.0)
            structure_assumptions.append(
                "Roof and vent details are missing; ember hardening interpretation is provisional."
            )
        structure_score_confidence = max(5.0, min(100.0, structure_score_confidence))

        structure_score = weighted_score(
            [
                (0.30, roof_ignition, "Roof type unavailable for structure vulnerability model."),
                (0.23, vent_ignition, "Vent type unavailable for structure vulnerability model."),
                (0.20, construction_risk, "Construction year and building-age proxy unavailable for structure vulnerability model."),
                (0.10, window_ignition, "Window type unavailable for structure vulnerability model."),
                (0.08, structure_density_proxy_index, "Structure-density proxy unavailable for structure vulnerability model."),
                (0.05, nearest_structure_isolation_index, "Nearest-structure isolation proxy unavailable for structure vulnerability model."),
                (0.04, clustering_index, "Structure clustering proxy unavailable for structure vulnerability model."),
            ],
            structure_assumptions,
        )
        # Low-confidence structure evidence should not yield highly separated structure
        # vulnerability scores; damp toward a neutral anchor when details are mostly assumed.
        neutral_anchor = 50.0
        confidence_multiplier = max(0.22, min(1.0, float(structure_score_confidence) / 100.0))
        structure_score = neutral_anchor + ((float(structure_score) - neutral_anchor) * confidence_multiplier)
        structure_clamped = clamp_score(structure_score)
        submodels["structure_vulnerability_risk"] = SubmodelResult(
            score=structure_clamped,
            explanation="Structure vulnerability risk reflects hardening quality against ember and radiant heat intrusion.",
            key_inputs={
                "roof_ignition_proxy": roof_ignition,
                "vent_ignition_proxy": vent_ignition,
                "construction_risk_proxy": construction_risk,
                "building_age_proxy_year": building_age_proxy_year,
                "building_age_material_proxy_risk": building_age_material_proxy_risk,
                "window_ignition_proxy": window_ignition,
                "structure_density_proxy_index": structure_density_proxy_index,
                "distance_to_nearest_structure_ft": distance_to_nearest_structure_ft,
                "clustering_index": clustering_index,
                "local_structure_density_index": local_structure_density_index,
                "nearest_structure_distance_ft": nearest_structure_distance_ft,
                "nearest_structure_proximity_index": nearest_structure_proximity_index,
                "nearest_structure_isolation_index": nearest_structure_isolation_index,
                "structure_data_completeness": round(structure_data_completeness, 1),
                "structure_assumption_mode": structure_assumption_mode,
                "structure_score_confidence": round(structure_score_confidence, 1),
                "structure_observed_fields": list(structure_observed_fields),
                "structure_inferred_fields": list(structure_inferred_fields),
                "structure_defaulted_fields": list(structure_defaulted_fields),
            },
            assumptions=structure_assumptions,
            raw_score=round(float(structure_score), 4),
            clamped_score=structure_clamped,
        )

        defensible_assumptions: List[str] = []
        if attrs.defensible_space_ft is None:
            defensible_assumptions.append("Defensible space missing; defensible-space model excludes clearance distance contribution.")
        if ring_0_5_density is None and ring_5_30_density is None:
            defensible_assumptions.append(
                "Structure-ring vegetation metrics unavailable; defensible-space pressure used fuel index proxy."
            )

        if close_in_zone_nonlinear_penalty is not None:
            immediate_zone_pressure = close_in_zone_nonlinear_penalty
        elif ring_0_5_density is not None:
            immediate_zone_pressure = _nonlinear_close_in_vegetation_penalty(ring_0_5_density)
        elif ring_5_30_density is not None:
            immediate_zone_pressure = round(ring_5_30_density * 0.92, 1)
        else:
            immediate_zone_pressure = fuel_index

        intermediate_zone_pressure = (
            zone1_veg_pressure
            if zone1_veg_pressure is not None
            else (
                ring_5_30_density
                if ring_5_30_density is not None
                else (ring_30_100_density if ring_30_100_density is not None else fuel_index)
            )
        )
        canopy_close_pressure = near_structure_continuity
        directional_near_pressure = directional_concentration_index
        asymmetry_near_pressure = canopy_dense_fuel_asymmetry_index
        nearest_continuous_pressure = nearest_continuous_vegetation_index
        if immediate_zone_pressure is not None:
            immediate_zone_pressure = _precision_adjust(immediate_zone_pressure)
        defensible_clearance_component = (
            max(0.0, min(100.0, 95.0 - defensible_ft * 2.6))
            if attrs.defensible_space_ft is not None
            else None
        )
        defensible_score = weighted_score(
            [
                (0.31, defensible_clearance_component, "Defensible space value unavailable for defensible-space model."),
                (0.34, immediate_zone_pressure, "Immediate-zone pressure unavailable for defensible-space model."),
                (0.14, intermediate_zone_pressure, "5-30 ft vegetation pressure unavailable for defensible-space model."),
                (0.09, canopy_close_pressure, "Near-structure connectivity proxy unavailable for defensible-space model."),
                (0.05, directional_near_pressure, "Directional vegetation concentration unavailable for defensible-space model."),
                (0.04, asymmetry_near_pressure, "Canopy/dense-fuel asymmetry unavailable for defensible-space model."),
                (0.03, nearest_continuous_pressure, "Nearest continuous vegetation distance unavailable for defensible-space model."),
            ],
            defensible_assumptions,
        )
        defensible_clamped = clamp_score(defensible_score)
        submodels["defensible_space_risk"] = SubmodelResult(
            score=defensible_clamped,
            explanation="Defensible space risk reflects clearance sufficiency under local fuel pressure.",
            key_inputs={
                "defensible_space_ft": defensible_ft,
                "fuel_index": fuel_index,
                "ring_0_5_ft_vegetation_density": ring_0_5_density,
                "ring_5_30_ft_vegetation_density": ring_5_30_density,
                "ring_30_100_ft_vegetation_density": ring_30_100_density,
                "near_structure_vegetation_0_5_pct": near_structure_vegetation_0_5_pct,
                "near_structure_vegetation_5_30_pct": near_structure_vegetation_5_30_pct,
                "immediate_zone_pressure": immediate_zone_pressure,
                "ring_0_5_nonlinear_penalty_index": close_in_zone_nonlinear_penalty,
                "intermediate_zone_pressure": intermediate_zone_pressure,
                "canopy_adjacency_proxy_pct": canopy_close_pressure,
                "near_structure_connectivity_index": near_structure_connectivity_index,
                "vegetation_edge_directional_concentration_pct": vegetation_edge_directional_concentration_pct,
                "canopy_dense_fuel_asymmetry_pct": canopy_dense_fuel_asymmetry_pct,
                "nearest_continuous_vegetation_distance_ft": nearest_continuous_vegetation_distance_ft,
                "nearest_continuous_vegetation_index": nearest_continuous_vegetation_index,
                "nearest_high_fuel_patch_distance_ft": nearest_high_fuel_patch_distance_ft,
                "nearest_vegetation_distance_ft": nearest_vegetation_distance_ft,
                "vegetation_directional_precision": vegetation_directional_precision,
                "vegetation_directional_precision_score": vegetation_directional_precision_score,
            },
            assumptions=defensible_assumptions + context_assumptions,
            raw_score=round(float(defensible_score), 4),
            clamped_score=defensible_clamped,
        )

        return submodels

    def score(self, attrs: PropertyAttributes, lat: float, lon: float, context: WildfireContext) -> RiskComputation:
        submodels = self._build_submodels(attrs, context)

        weighted_contributions: Dict[str, dict] = {}
        assumptions: List[str] = list(context.assumptions)
        observed_factor_count = 0
        missing_factor_count = 0
        fallback_factor_count = 0
        fallback_effective_weight = 0.0
        total_base_weight = 0.0
        total_effective_weight = 0.0
        property_level_context = (
            context.property_level_context
            if isinstance(context.property_level_context, dict)
            else {}
        )
        footprint_used = bool(property_level_context.get("footprint_used"))
        parcel_available = bool(
            property_level_context.get("parcel_id")
            or property_level_context.get("parcel_polygon")
            or property_level_context.get("parcel_geometry")
        )
        geometry_basis = "footprint" if footprint_used else ("parcel" if parcel_available else "point")
        feature_bundle_summary = (
            property_level_context.get("feature_bundle_summary")
            if isinstance(property_level_context.get("feature_bundle_summary"), dict)
            else {}
        )
        bundle_metrics = (
            feature_bundle_summary.get("coverage_metrics")
            if isinstance(feature_bundle_summary.get("coverage_metrics"), dict)
            else {}
        )
        observed_feature_count = int(bundle_metrics.get("observed_feature_count") or 0)
        inferred_feature_count = int(bundle_metrics.get("inferred_feature_count") or 0)
        fallback_feature_count = int(bundle_metrics.get("fallback_feature_count") or 0)
        geometry_quality_score = float(
            bundle_metrics.get("structure_geometry_quality_score")
            if bundle_metrics.get("structure_geometry_quality_score") is not None
            else (0.92 if geometry_basis == "footprint" else (0.74 if geometry_basis == "parcel" else 0.46))
        )
        structure_geometry_confidence = float(
            bundle_metrics.get("structure_geometry_confidence")
            if bundle_metrics.get("structure_geometry_confidence") is not None
            else (
                property_level_context.get("structure_geometry_confidence")
                if property_level_context.get("structure_geometry_confidence") is not None
                else geometry_quality_score
            )
        )
        anchor_quality_score = float(
            bundle_metrics.get("anchor_quality_score")
            if bundle_metrics.get("anchor_quality_score") is not None
            else (
                property_level_context.get("anchor_quality_score")
                if property_level_context.get("anchor_quality_score") is not None
                else (
                    property_level_context.get("property_anchor_quality_score")
                    or (0.90 if geometry_basis == "footprint" else (0.72 if geometry_basis == "parcel" else 0.52))
                )
            )
        )
        env_status = (
            context.environmental_layer_status
            if isinstance(getattr(context, "environmental_layer_status", None), dict)
            else {}
        )
        ok_env_layers = sum(
            1
            for status in env_status.values()
            if str(status).strip().lower() in {"ok", "ok_nearby"}
        )
        env_layer_total = max(1, len(env_status))
        env_coverage_default = (ok_env_layers / float(env_layer_total)) * 100.0
        regional_context_coverage_score = float(
            bundle_metrics.get("environmental_layer_coverage_score")
            if bundle_metrics.get("environmental_layer_coverage_score") is not None
            else env_coverage_default
        )
        regional_enrichment_consumption_score = float(
            bundle_metrics.get("regional_enrichment_consumption_score")
            if bundle_metrics.get("regional_enrichment_consumption_score") is not None
            else regional_context_coverage_score
        )
        property_specificity_score = float(
            bundle_metrics.get("property_specificity_score")
            if bundle_metrics.get("property_specificity_score") is not None
            else (85.0 if geometry_basis == "footprint" else (62.0 if geometry_basis == "parcel" else 42.0))
        )
        if bundle_metrics.get("fallback_dominance_ratio") is not None:
            feature_fallback_ratio = float(bundle_metrics.get("fallback_dominance_ratio") or 0.0)
        else:
            total_feature_count = max(1, observed_feature_count + inferred_feature_count + fallback_feature_count)
            feature_fallback_ratio = float(fallback_feature_count) / float(total_feature_count)
        structure_model_inputs = (
            submodels.get("structure_vulnerability_risk").key_inputs
            if isinstance(submodels.get("structure_vulnerability_risk"), SubmodelResult)
            else {}
        )
        try:
            structure_data_completeness = float((structure_model_inputs or {}).get("structure_data_completeness") or 0.0)
        except (TypeError, ValueError):
            structure_data_completeness = 0.0
        structure_assumption_mode = str((structure_model_inputs or {}).get("structure_assumption_mode") or "unknown")
        try:
            structure_score_confidence = float((structure_model_inputs or {}).get("structure_score_confidence") or 0.0)
        except (TypeError, ValueError):
            structure_score_confidence = 0.0
        ring_metrics = (
            property_level_context.get("ring_metrics")
            if isinstance(property_level_context.get("ring_metrics"), dict)
            else {}
        )
        ring_has_direct_geometry = bool(footprint_used and ring_metrics)
        ring_metric_rows = [row for row in ring_metrics.values() if isinstance(row, dict)]
        near_structure_observed = bool(
            ring_has_direct_geometry
            or any(
                (row.get("vegetation_density") is not None)
                for row in ring_metric_rows
            )
            or
            any(
                property_level_context.get(key) is not None
                for key in (
                    "near_structure_vegetation_0_5_pct",
                    "near_structure_vegetation_5_30_pct",
                    "canopy_adjacency_proxy_pct",
                    "vegetation_continuity_proxy_pct",
                    "vegetation_edge_directional_concentration_pct",
                    "canopy_dense_fuel_asymmetry_pct",
                    "nearest_continuous_vegetation_distance_ft",
                    "nearest_high_fuel_patch_distance_ft",
                )
            )
            or any(row.get("vegetation_density") is not None for row in ring_metric_rows)
            or any(row.get("imagery_vegetation_continuity_pct") is not None for row in ring_metric_rows)
            or any(row.get("imagery_canopy_proxy_pct") is not None for row in ring_metric_rows)
        )
        burn_missing = context.burn_probability_index is None
        hazard_missing = context.hazard_severity_index is None
        dryness_missing = context.moisture_index is None
        major_env_missing_count = sum(1 for flag in (burn_missing, hazard_missing, dryness_missing) if flag)

        def _component_name(submodel: str) -> str:
            if submodel in REGIONAL_CONTEXT_SUBMODELS:
                return "regional_context"
            if submodel in PROPERTY_SURROUNDINGS_SUBMODELS:
                return "property_surroundings"
            if submodel in STRUCTURE_SPECIFIC_SUBMODELS:
                return "structure_specific"
            return "unknown"

        def _availability_multiplier(
            submodel: str,
            observed_fraction: float,
            assumptions_text: str,
            *,
            has_fallback_tokens: bool,
        ) -> float:
            multiplier = 1.0
            has_low_quality_assumption = any(
                token in assumptions_text
                for token in ("fallback", "proxy", "missing", "unavailable", "point-based")
            )
            if has_low_quality_assumption:
                # Keep evidence penalties monotonic without collapsing viable runs.
                multiplier *= 0.88
            if geometry_basis == "point":
                if submodel == "defensible_space_risk":
                    multiplier *= 0.05
                elif submodel == "ember_exposure_risk":
                    multiplier *= 0.18
                elif submodel == "flame_contact_risk":
                    multiplier *= 0.18
                elif submodel == "fuel_proximity_risk":
                    multiplier *= 0.48
                elif submodel == "vegetation_intensity_risk":
                    multiplier *= 0.42
            elif geometry_basis == "parcel":
                if submodel == "defensible_space_risk":
                    multiplier *= 0.38
                elif submodel == "ember_exposure_risk":
                    multiplier *= 0.60
            if not footprint_used and submodel in {
                "vegetation_intensity_risk",
                "fuel_proximity_risk",
                "flame_contact_risk",
                "defensible_space_risk",
            }:
                multiplier *= 0.50
            if not parcel_available and submodel == "defensible_space_risk":
                multiplier *= 0.64
            if not ring_has_direct_geometry and submodel in {"defensible_space_risk", "flame_contact_risk"}:
                multiplier *= 0.52
            if not near_structure_observed and submodel in {"vegetation_intensity_risk", "fuel_proximity_risk", "flame_contact_risk"}:
                multiplier *= 0.56
            if structure_geometry_confidence < 0.60 and submodel in GEOMETRY_SENSITIVE_SUBMODELS:
                multiplier *= 0.68
            if structure_geometry_confidence < 0.45 and submodel in GEOMETRY_SENSITIVE_SUBMODELS:
                multiplier *= 0.52
            if anchor_quality_score < 0.60 and submodel in GEOMETRY_SENSITIVE_SUBMODELS:
                multiplier *= 0.72
            if dryness_missing and submodel in {"vegetation_intensity_risk", "flame_contact_risk"}:
                multiplier *= 0.62
            if burn_missing and submodel == "ember_exposure_risk":
                multiplier *= 0.60
            if hazard_missing and submodel == "ember_exposure_risk":
                multiplier *= 0.64
            if major_env_missing_count >= 2 and submodel in PROPERTY_SURROUNDINGS_SUBMODELS:
                multiplier *= 0.74
            if major_env_missing_count >= 2 and submodel in REGIONAL_CONTEXT_SUBMODELS:
                multiplier *= 0.60
            if regional_context_coverage_score < 50.0 and submodel in REGIONAL_CONTEXT_SUBMODELS:
                multiplier *= 0.62
            if regional_context_coverage_score < 60.0 and submodel in REGIONAL_CONTEXT_SUBMODELS:
                multiplier *= 0.76
            if regional_enrichment_consumption_score < 45.0 and submodel in REGIONAL_CONTEXT_SUBMODELS:
                multiplier *= 0.55
            elif regional_enrichment_consumption_score < 60.0 and submodel in REGIONAL_CONTEXT_SUBMODELS:
                multiplier *= 0.72
            if geometry_quality_score < 0.62 and submodel in STRUCTURE_SPECIFIC_SUBMODELS:
                multiplier *= 0.62
            if property_specificity_score < 55.0 and submodel in GEOMETRY_SENSITIVE_SUBMODELS:
                multiplier *= 0.70
            if feature_fallback_ratio >= 0.50:
                multiplier *= 0.86
            if observed_fraction < 0.45:
                multiplier *= 0.75
            if has_fallback_tokens:
                multiplier *= 0.94
            if footprint_used and ring_has_direct_geometry and submodel in GEOMETRY_SENSITIVE_SUBMODELS:
                # When trustworthy footprint/ring evidence exists, avoid collapsing
                # geometry-sensitive factors to near-zero availability.
                minimum = 0.45 if submodel in {"defensible_space_risk", "flame_contact_risk"} else 0.50
                multiplier = max(multiplier, minimum)
            return max(0.0, min(1.0, multiplier))

        for name, result in submodels.items():
            weight = self.config.submodel_weights[name]
            total_base_weight += weight
            key_inputs = result.key_inputs if isinstance(result.key_inputs, dict) else {}
            expected_input_count = max(1, len(key_inputs))
            observed_input_count = sum(1 for value in key_inputs.values() if value is not None)
            observed_fraction = max(
                0.0,
                min(1.0, float(observed_input_count) / float(expected_input_count)),
            )
            if footprint_used and ring_has_direct_geometry and name in GEOMETRY_SENSITIVE_SUBMODELS:
                observed_fraction = max(observed_fraction, 0.65)
            assumptions_text = " ".join(result.assumptions).lower()
            has_fallback_tokens = any(
                tok in assumptions_text for tok in ("fallback", "proxy", "missing", "unavailable", "point-based")
            )
            if name == "structure_vulnerability_risk":
                observed_fraction = min(
                    observed_fraction,
                    max(0.12, min(1.0, structure_score_confidence / 100.0)),
                )
            availability_multiplier = _availability_multiplier(
                name,
                observed_fraction,
                assumptions_text,
                has_fallback_tokens=has_fallback_tokens,
            )
            if name == "structure_vulnerability_risk":
                availability_multiplier *= max(0.20, min(1.0, structure_score_confidence / 100.0))
                if structure_assumption_mode == "default_assumed":
                    availability_multiplier *= 0.55
                elif structure_assumption_mode == "mixed":
                    availability_multiplier *= 0.82
                availability_multiplier = max(0.0, min(1.0, availability_multiplier))
            omitted_due_to_missing = observed_input_count <= 0
            effective_weight = 0.0 if omitted_due_to_missing else (weight * observed_fraction * availability_multiplier)
            suppressed_by_evidence = False
            if not omitted_due_to_missing and name in GEOMETRY_SENSITIVE_SUBMODELS:
                weak_geometry_context = (
                    geometry_basis == "point"
                    and (not ring_has_direct_geometry or geometry_quality_score < 0.55)
                    and (has_fallback_tokens or observed_fraction < 0.80)
                )
                if weak_geometry_context and name in {"defensible_space_risk", "flame_contact_risk"}:
                    suppressed_by_evidence = True
                elif weak_geometry_context and name == "ember_exposure_risk" and structure_geometry_confidence < 0.45:
                    suppressed_by_evidence = True
                elif weak_geometry_context and name in {"vegetation_intensity_risk", "fuel_proximity_risk"}:
                    effective_weight *= 0.45
                elif weak_geometry_context and name == "ember_exposure_risk":
                    effective_weight *= 0.38
            if (
                not omitted_due_to_missing
                and observed_fraction < 0.25
                and availability_multiplier < 0.35
            ):
                omitted_due_to_missing = True
                effective_weight = 0.0
            if suppressed_by_evidence:
                omitted_due_to_missing = True
                effective_weight = 0.0
            if omitted_due_to_missing:
                missing_factor_count += 1
                assumptions.append(
                    f"{name.replace('_', ' ')} omitted from numeric weighting because required evidence was missing."
                )
                if suppressed_by_evidence:
                    assumptions.append(
                        f"{name.replace('_', ' ')} was suppressed because structure geometry evidence was too weak for reliable property-level weighting."
                    )
                basis = "missing"
                support_level = "low"
            else:
                observed_factor_count += 1
                if observed_fraction < 0.60 or has_fallback_tokens or availability_multiplier < 0.65:
                    fallback_factor_count += 1
                if observed_fraction >= 0.80 and availability_multiplier >= 0.85 and not has_fallback_tokens:
                    basis = "observed"
                    support_level = "high"
                elif observed_fraction >= 0.45 and availability_multiplier >= 0.25:
                    basis = "inferred"
                    support_level = "medium"
                else:
                    basis = "fallback"
                    support_level = "low"
            total_effective_weight += effective_weight
            if basis == "fallback":
                fallback_effective_weight += effective_weight
            weighted_contributions[name] = {
                "weight": 0.0,
                "score": round(result.score, 2),
                "contribution": 0.0,
                "base_weight": round(weight, 6),
                "effective_weight": round(effective_weight, 6),
                "observed_fraction": round(observed_fraction, 4),
                "availability_multiplier": round(availability_multiplier, 4),
                "omitted_due_to_missing": omitted_due_to_missing,
                "basis": basis,
                "factor_evidence_status": (
                    "suppressed"
                    if omitted_due_to_missing
                    else ("fallback" if basis == "fallback" else ("inferred" if basis == "inferred" else "observed"))
                ),
                "support_level": support_level,
                "component": _component_name(name),
            }
            assumptions.extend(result.assumptions)

        total = 0.0
        for name, row in weighted_contributions.items():
            effective_weight = float(row.get("effective_weight") or 0.0)
            if total_effective_weight > 0.0 and effective_weight > 0.0:
                normalized_weight = effective_weight / total_effective_weight
                contribution = normalized_weight * float(row.get("score") or 0.0)
            else:
                normalized_weight = 0.0
                contribution = 0.0
            row["weight"] = round(normalized_weight, 6)
            row["contribution"] = round(contribution, 2)
            total += contribution

        def _group_driver(group: set[str]) -> float:
            group_weight = sum(float(weighted_contributions.get(n, {}).get("weight", 0.0)) for n in group)
            if group_weight <= 0.0:
                return 0.0
            group_contribution = sum(float(weighted_contributions.get(n, {}).get("contribution", 0.0)) for n in group)
            return round(max(0.0, min(100.0, group_contribution / group_weight)), 1)

        regional_context_score = _group_driver(REGIONAL_CONTEXT_SUBMODELS)
        property_surroundings_score = _group_driver(PROPERTY_SURROUNDINGS_SUBMODELS)
        structure_specific_score = _group_driver(STRUCTURE_SPECIFIC_SUBMODELS)
        environmental_driver = round(
            max(0.0, min(100.0, (regional_context_score * 0.72) + (property_surroundings_score * 0.28))),
            1,
        )
        if sum(float(weighted_contributions.get(n, {}).get("weight", 0.0)) for n in STRUCTURE_SPECIFIC_SUBMODELS) <= 0.05:
            structural_driver = round(max(0.0, min(100.0, property_surroundings_score)), 1)
        else:
            structural_driver = round(
                max(0.0, min(100.0, (structure_specific_score * 0.64) + (property_surroundings_score * 0.36))),
                1,
            )

        observed_weight_fraction = (
            (total_effective_weight / total_base_weight)
            if total_base_weight > 0.0
            else 0.0
        )
        fallback_dominance_ratio = (
            (float(fallback_factor_count) / float(observed_factor_count))
            if observed_factor_count > 0
            else (1.0 if missing_factor_count > 0 else 0.0)
        )
        fallback_weight_fraction = (
            (fallback_effective_weight / total_effective_weight)
            if total_effective_weight > 0.0
            else (1.0 if fallback_factor_count > 0 else 0.0)
        )
        if fallback_feature_count > 0:
            fallback_feature_fraction = float(fallback_feature_count) / float(
                max(1, observed_feature_count + inferred_feature_count + fallback_feature_count)
            )
            fallback_weight_fraction = max(fallback_weight_fraction, fallback_feature_fraction)
        uncertainty_penalty = min(
            25.0,
            max(0.0, (1.0 - observed_weight_fraction) * 20.0)
            + (missing_factor_count * 1.5)
            + (fallback_factor_count * 0.5)
            + (fallback_weight_fraction * 4.0),
        )
        component_weight_fractions = {
            "regional_context": round(
                sum(
                    float(weighted_contributions.get(n, {}).get("weight", 0.0))
                    for n in REGIONAL_CONTEXT_SUBMODELS
                ),
                4,
            ),
            "property_surroundings": round(
                sum(
                    float(weighted_contributions.get(n, {}).get("weight", 0.0))
                    for n in PROPERTY_SURROUNDINGS_SUBMODELS
                ),
                4,
            ),
            "structure_specific": round(
                sum(
                    float(weighted_contributions.get(n, {}).get("weight", 0.0))
                    for n in STRUCTURE_SPECIFIC_SUBMODELS
                ),
                4,
            ),
        }

        access_exposure = context.access_exposure_index
        access_provisional = True
        access_note = (
            "Access exposure is derived from OSM road-network context and remains separate from weighted wildfire scoring."
        )
        if access_exposure is None:
            access_exposure = 0.0
            access_note = (
                "Access exposure was not computed because road-network evidence was unavailable; metric is advisory only."
            )
            assumptions.append("Access exposure unavailable; road-network context missing for this property.")
        else:
            access_provisional = False
            assumptions.append("Access exposure derived from road-network context (not included in wildfire total score).")

        return RiskComputation(
            total_score=round(max(0.0, min(100.0, total)), 1),
            drivers=RiskDrivers(environmental=environmental_driver, structural=structural_driver, access_exposure=access_exposure),
            assumptions=sorted(set(assumptions)),
            submodel_scores=submodels,
            weighted_contributions=weighted_contributions,
            observed_factor_count=observed_factor_count,
            missing_factor_count=missing_factor_count,
            fallback_factor_count=fallback_factor_count,
            observed_weight_fraction=round(observed_weight_fraction, 4),
            fallback_dominance_ratio=round(fallback_dominance_ratio, 4),
            fallback_weight_fraction=round(fallback_weight_fraction, 4),
            uncertainty_penalty=round(uncertainty_penalty, 2),
            regional_context_score=regional_context_score,
            property_surroundings_score=property_surroundings_score,
            structure_specific_score=structure_specific_score,
            component_weight_fractions=component_weight_fractions,
            geometry_basis=geometry_basis,
            observed_feature_count=observed_feature_count,
            inferred_feature_count=inferred_feature_count,
            fallback_feature_count=fallback_feature_count,
            geometry_quality_score=round(max(0.0, min(1.0, geometry_quality_score)), 3),
            regional_context_coverage_score=round(max(0.0, min(100.0, regional_context_coverage_score)), 1),
            property_specificity_score=round(max(0.0, min(100.0, property_specificity_score)), 1),
            structure_data_completeness=round(max(0.0, min(100.0, structure_data_completeness)), 1),
            structure_assumption_mode=structure_assumption_mode,
            structure_score_confidence=round(max(0.0, min(100.0, structure_score_confidence)), 1),
            access_provisional=access_provisional,
            access_note=access_note,
        )

    def _group_weight(self, group: set[str]) -> float:
        return sum(self.config.submodel_weights.get(name, 0.0) for name in group)

    def _group_score(self, risk: RiskComputation, group: set[str]) -> float:
        weight = sum(
            float((risk.weighted_contributions.get(name) or {}).get("weight", 0.0))
            for name in group
        )
        if weight <= 0:
            return 0.0

        contribution_sum = sum(
            risk.weighted_contributions.get(name, {}).get("contribution", 0.0)
            for name in group
        )
        return round(max(0.0, min(100.0, contribution_sum / weight)), 1)

    def compute_site_hazard_score(self, risk: RiskComputation) -> float:
        return self._group_score(risk, ENVIRONMENT_SUBMODELS)

    def compute_home_ignition_vulnerability_score(self, risk: RiskComputation) -> float:
        base = self._group_score(risk, STRUCTURE_SUBMODELS)

        # Keep structure vulnerability distinct from regional hazard by grounding
        # the base in structure/near-structure submodels before ring penalties.
        near_weights: dict[str, float] = {
            "defensible_space_risk": 0.34,
            "flame_contact_risk": 0.28,
            "ember_exposure_risk": 0.18,
            "fuel_proximity_risk": 0.12,
            "vegetation_intensity_risk": 0.08,
        }
        near_num = 0.0
        near_den = 0.0
        for factor_key, factor_weight in near_weights.items():
            factor_result = risk.submodel_scores.get(factor_key)
            if factor_result is None:
                continue
            effective_weight = float((risk.weighted_contributions.get(factor_key) or {}).get("effective_weight") or 0.0)
            if effective_weight <= 0.0:
                continue
            near_num += float(factor_result.score) * float(factor_weight)
            near_den += float(factor_weight)
        if near_den > 0.0:
            near_structure_core = near_num / near_den
            if base > 0.0:
                base = (base * 0.66) + (near_structure_core * 0.34)
            else:
                base = near_structure_core

        def _to_float(value: object) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        defensible_inputs = risk.submodel_scores.get("defensible_space_risk", SubmodelResult(0.0, "", {})).key_inputs
        fuel_inputs = risk.submodel_scores.get("fuel_proximity_risk", SubmodelResult(0.0, "", {})).key_inputs
        flame_inputs = risk.submodel_scores.get("flame_contact_risk", SubmodelResult(0.0, "", {})).key_inputs

        zone_0_5 = _to_float(defensible_inputs.get("ring_0_5_ft_vegetation_density"))
        zone_5_30 = _to_float(defensible_inputs.get("ring_5_30_ft_vegetation_density"))
        zone_30_100 = _to_float(fuel_inputs.get("ring_30_100_ft_vegetation_density"))
        zone_100_300 = _to_float(fuel_inputs.get("ring_100_300_ft_vegetation_density"))
        nearest_vegetation_distance_ft = _to_float(
            defensible_inputs.get("nearest_vegetation_distance_ft") or flame_inputs.get("nearest_vegetation_distance_ft")
        )

        ring_cfg = self.config.vulnerability_ring_penalties or {}

        def _penalty(
            zone_key: str,
            value: float | None,
            default_threshold: float,
            default_slope: float,
            *,
            default_nonlinear_boost: float = 0.0,
        ) -> float:
            if value is None:
                return 0.0
            zone_params = ring_cfg.get(zone_key) if isinstance(ring_cfg.get(zone_key), dict) else {}
            try:
                threshold = float(zone_params.get("threshold", default_threshold))
            except (TypeError, ValueError, AttributeError):
                threshold = default_threshold
            try:
                slope = float(zone_params.get("slope", default_slope))
            except (TypeError, ValueError, AttributeError):
                slope = default_slope
            try:
                nonlinear_boost = float(zone_params.get("nonlinear_boost", default_nonlinear_boost))
            except (TypeError, ValueError, AttributeError):
                nonlinear_boost = default_nonlinear_boost
            excess = max(0.0, value - threshold)
            if excess <= 0.0:
                return 0.0
            baseline = excess * slope
            nonlinear_scale = 1.0 + (max(0.0, nonlinear_boost) * min(1.25, excess / 28.0))
            return baseline * nonlinear_scale

        ring_penalty = 0.0
        ring_penalty += _penalty("zone_0_5_ft", zone_0_5, 42.0, 0.46, default_nonlinear_boost=0.60)
        ring_penalty += _penalty("zone_5_30_ft", zone_5_30, 52.0, 0.27, default_nonlinear_boost=0.35)
        ring_penalty += _penalty("zone_30_100_ft", zone_30_100, 63.0, 0.10)
        ring_penalty += _penalty("zone_100_300_ft", zone_100_300, 70.0, 0.05)

        # Piecewise surcharges keep the near-home vegetation effect monotonic
        # and meaningful without allowing unbounded score jumps.
        if zone_0_5 is not None:
            if zone_0_5 >= 85.0:
                ring_penalty += 9.0
            elif zone_0_5 >= 75.0:
                ring_penalty += 6.5
            elif zone_0_5 >= 65.0:
                ring_penalty += 4.5
            elif zone_0_5 >= 50.0:
                ring_penalty += 2.5
        if zone_5_30 is not None:
            if zone_5_30 >= 88.0:
                ring_penalty += 4.5
            elif zone_5_30 >= 76.0:
                ring_penalty += 3.0
            elif zone_5_30 >= 70.0:
                ring_penalty += 2.0

        distance_cfg = ring_cfg.get("nearest_vegetation_distance_ft") if isinstance(ring_cfg.get("nearest_vegetation_distance_ft"), dict) else {}
        try:
            critical_max_ft = float(distance_cfg.get("critical_max_ft", 5.0))
        except (TypeError, ValueError, AttributeError):
            critical_max_ft = 5.0
        try:
            watch_max_ft = float(distance_cfg.get("watch_max_ft", 15.0))
        except (TypeError, ValueError, AttributeError):
            watch_max_ft = 15.0
        try:
            critical_penalty = float(distance_cfg.get("critical_penalty", 8.0))
        except (TypeError, ValueError, AttributeError):
            critical_penalty = 8.0
        try:
            watch_penalty = float(distance_cfg.get("watch_penalty", 4.0))
        except (TypeError, ValueError, AttributeError):
            watch_penalty = 4.0
        try:
            ultra_critical_max_ft = float(distance_cfg.get("ultra_critical_max_ft", 2.0))
        except (TypeError, ValueError, AttributeError):
            ultra_critical_max_ft = 2.0
        try:
            ultra_critical_penalty = float(distance_cfg.get("ultra_critical_penalty", 13.0))
        except (TypeError, ValueError, AttributeError):
            ultra_critical_penalty = 13.0
        if nearest_vegetation_distance_ft is not None:
            if nearest_vegetation_distance_ft <= ultra_critical_max_ft:
                ring_penalty += ultra_critical_penalty
            elif nearest_vegetation_distance_ft <= critical_max_ft:
                ring_penalty += critical_penalty
            elif nearest_vegetation_distance_ft <= watch_max_ft:
                ring_penalty += watch_penalty

        clearance_credit = 0.0
        if zone_0_5 is not None and zone_0_5 < 40.0:
            margin = max(0.0, 40.0 - zone_0_5)
            clearance_credit += 4.2 * (1.0 - math.exp(-margin / 11.0))
        if zone_5_30 is not None and zone_5_30 < 50.0:
            margin = max(0.0, 50.0 - zone_5_30)
            clearance_credit += 2.3 * (1.0 - math.exp(-margin / 16.0))
        if nearest_vegetation_distance_ft is not None and nearest_vegetation_distance_ft > watch_max_ft:
            extra_clearance = max(0.0, nearest_vegetation_distance_ft - watch_max_ft)
            clearance_credit += 1.9 * (1.0 - math.exp(-extra_clearance / 20.0))
        clearance_credit = min(clearance_credit, 6.0)

        return round(max(0.0, min(100.0, base + ring_penalty - clearance_credit)), 1)

    def compute_blended_wildfire_score(
        self,
        site_hazard_score: float,
        home_ignition_vulnerability_score: float,
        insurance_readiness_score: float | None = None,
        risk: RiskComputation | None = None,
    ) -> float:
        env_weight, struct_weight, readiness_weight = self.resolve_blend_weights(
            insurance_readiness_score=insurance_readiness_score,
            risk=risk,
        )
        denom = env_weight + struct_weight + max(0.0, readiness_weight)
        if denom <= 0:
            return 0.0

        readiness_risk_equivalent = (
            max(0.0, min(100.0, 100.0 - float(insurance_readiness_score)))
            if insurance_readiness_score is not None
            else 0.0
        )
        base_blended = (
            site_hazard_score * env_weight
            + home_ignition_vulnerability_score * struct_weight
            + readiness_risk_equivalent * max(0.0, readiness_weight)
        ) / denom

        hazard_norm = max(0.0, min(1.0, float(site_hazard_score) / 100.0))
        vulnerability_norm = max(0.0, min(1.0, float(home_ignition_vulnerability_score) / 100.0))
        harmonic_core = (
            (2.0 * hazard_norm * vulnerability_norm) / (hazard_norm + vulnerability_norm)
            if (hazard_norm + vulnerability_norm) > 0.0
            else 0.0
        )
        product_core = hazard_norm * vulnerability_norm
        hazard_vulnerability_core = 100.0 * ((0.55 * harmonic_core) + (0.45 * product_core))
        interaction_blended = (base_blended * 0.58) + (hazard_vulnerability_core * 0.42)

        near_structure_signal = float(home_ignition_vulnerability_score)
        vegetation_signal = float(home_ignition_vulnerability_score)
        slope_signal = float(site_hazard_score)
        fuel_signal = float(site_hazard_score)
        structure_vulnerability_signal = float(home_ignition_vulnerability_score)
        structure_proxy_signal = float(home_ignition_vulnerability_score)
        interaction_gate = 1.0
        if risk is not None:
            def _to_float_local(value: object) -> float | None:
                try:
                    if value is None:
                        return None
                    return float(value)
                except (TypeError, ValueError):
                    return None

            near_weights: dict[str, float] = {
                "defensible_space_risk": 0.34,
                "flame_contact_risk": 0.30,
                "vegetation_intensity_risk": 0.22,
                "fuel_proximity_risk": 0.14,
            }
            near_num = 0.0
            near_den = 0.0
            for factor_key, factor_weight in near_weights.items():
                result = risk.submodel_scores.get(factor_key)
                if result is None:
                    continue
                effective_weight = float((risk.weighted_contributions.get(factor_key) or {}).get("effective_weight") or 0.0)
                if effective_weight <= 0.0:
                    continue
                near_num += float(result.score) * float(factor_weight)
                near_den += float(factor_weight)
            if near_den > 0.0:
                near_structure_signal = near_num / near_den
            vegetation_signal = float(
                (risk.submodel_scores.get("vegetation_intensity_risk") or SubmodelResult(near_structure_signal, "", {})).score
            )
            slope_signal = float(
                (risk.submodel_scores.get("slope_topography_risk") or SubmodelResult(site_hazard_score, "", {})).score
            )
            fuel_signal = float(
                (risk.submodel_scores.get("fuel_proximity_risk") or SubmodelResult(site_hazard_score, "", {})).score
            )
            structure_vulnerability_signal = float(
                (risk.submodel_scores.get("structure_vulnerability_risk") or SubmodelResult(home_ignition_vulnerability_score, "", {})).score
            )
            structure_inputs = (
                risk.submodel_scores.get("structure_vulnerability_risk", SubmodelResult(structure_vulnerability_signal, "", {})).key_inputs
                if isinstance(risk.submodel_scores.get("structure_vulnerability_risk"), SubmodelResult)
                else {}
            )
            structure_density_proxy = _to_float_local((structure_inputs or {}).get("structure_density_proxy_index"))
            clustering_proxy = _to_float_local((structure_inputs or {}).get("clustering_index"))
            nearest_structure_proximity = _to_float_local((structure_inputs or {}).get("nearest_structure_proximity_index"))
            nearest_structure_isolation = _to_float_local((structure_inputs or {}).get("nearest_structure_isolation_index"))
            proxy_terms: list[tuple[float, float]] = []
            if structure_density_proxy is not None:
                proxy_terms.append((0.28, float(structure_density_proxy)))
            if clustering_proxy is not None:
                proxy_terms.append((0.18, float(clustering_proxy)))
            if nearest_structure_proximity is not None:
                proxy_terms.append((0.12, float(nearest_structure_proximity)))
            if nearest_structure_isolation is not None:
                proxy_terms.append((0.42, float(nearest_structure_isolation)))
            if proxy_terms:
                p_num = sum(weight * value for weight, value in proxy_terms)
                p_den = sum(weight for weight, _ in proxy_terms)
                if p_den > 0.0:
                    structure_proxy_signal = p_num / p_den
            else:
                structure_proxy_signal = structure_vulnerability_signal
            geometry_gate = 0.55 + (0.45 * max(0.0, min(1.0, float(risk.geometry_quality_score))))
            evidence_gate = 0.50 + (0.50 * max(0.0, min(1.0, float(risk.observed_weight_fraction))))
            fallback_gate = 1.0 - min(0.35, max(0.0, float(risk.fallback_weight_fraction)) * 0.50)
            interaction_gate = max(0.35, min(1.0, geometry_gate * evidence_gate * fallback_gate))

        # Explainable interactions for stronger separation (deterministic, bounded):
        # 1) high hazard x high vegetation intensity
        # 2) high hazard x steep slope
        # 3) high vulnerability x high vegetation intensity
        # 4) high hazard x close fuel proximity
        hazard_tail = max(0.0, float(site_hazard_score) - 55.0)
        near_tail = max(0.0, float(near_structure_signal) - 45.0)
        veg_tail = max(0.0, vegetation_signal - 55.0)
        slope_tail = max(0.0, slope_signal - 55.0)
        fuel_tail = max(0.0, fuel_signal - 55.0)
        vulnerability_tail = max(0.0, structure_vulnerability_signal - 50.0)
        structure_proxy_tail = max(0.0, structure_proxy_signal - 48.0)

        overlap_high = max(0.0, min(float(site_hazard_score), float(near_structure_signal)) - 50.0)
        high_risk_compound = overlap_high * 0.14 * interaction_gate
        hazard_near_compound = ((hazard_tail * near_tail) / 100.0) * 0.18 * interaction_gate
        hazard_vegetation_compound = ((hazard_tail * veg_tail) / 100.0) * 0.20 * interaction_gate
        hazard_slope_compound = ((max(0.0, float(site_hazard_score) - 58.0) * slope_tail) / 100.0) * 0.11 * interaction_gate
        vulnerability_vegetation_compound = ((vulnerability_tail * veg_tail) / 100.0) * 0.13 * interaction_gate
        hazard_fuel_compound = ((hazard_tail * fuel_tail) / 100.0) * 0.14 * interaction_gate
        hazard_structure_proxy_compound = ((hazard_tail * structure_proxy_tail) / 100.0) * 0.10 * interaction_gate
        readiness_drag = 0.0
        low_risk_credit = 0.0
        hardening_dampen = 0.0
        if insurance_readiness_score is not None:
            readiness_value = max(0.0, min(100.0, float(insurance_readiness_score)))
            if readiness_value < 50.0:
                severity_gate = 0.70 + (0.30 * max(float(site_hazard_score), float(near_structure_signal)) / 100.0)
                readiness_drag = (50.0 - readiness_value) * 0.11 * severity_gate
            low_overlap = max(0.0, 50.0 - max(float(site_hazard_score), float(near_structure_signal)))
            if readiness_value >= 72.0 and low_overlap > 0.0:
                low_risk_credit = ((readiness_value - 72.0) * 0.06) + (low_overlap * 0.08)
            if readiness_value < 45.0 and near_structure_signal > 70.0:
                readiness_drag += (float(near_structure_signal) - 70.0) * 0.08

            # Strong hardening dampens wildfire score only partially in severe hazard;
            # this avoids over-crediting hardening when landscape hazard is extreme.
            hardening_strength = max(0.0, 55.0 - structure_vulnerability_signal)
            readiness_strength = max(0.0, readiness_value - 70.0)
            if hardening_strength > 0.0 or readiness_strength > 0.0:
                severity = max(float(site_hazard_score), vegetation_signal, fuel_signal)
                severity_dampen_gate = max(0.40, 1.0 - max(0.0, severity - 55.0) * 0.01)
                hardening_dampen = (
                    ((hardening_strength * 0.11) + (readiness_strength * 0.09))
                    * severity_dampen_gate
                    * interaction_gate
                )
                hardening_dampen = min(hardening_dampen, 9.0)

        low_vulnerability_dampen = 0.0
        if home_ignition_vulnerability_score < 48.0:
            vulnerability_gap = max(0.0, 48.0 - float(home_ignition_vulnerability_score))
            hazard_gate = 1.0 if float(site_hazard_score) < 70.0 else 0.55
            low_vulnerability_dampen = vulnerability_gap * 0.14 * hazard_gate * interaction_gate

        interaction_adjusted = (
            interaction_blended
            + high_risk_compound
            + hazard_near_compound
            + hazard_vegetation_compound
            + hazard_slope_compound
            + vulnerability_vegetation_compound
            + hazard_fuel_compound
            + hazard_structure_proxy_compound
            + readiness_drag
            - low_risk_credit
            - hardening_dampen
            - low_vulnerability_dampen
        )
        contrast_adjusted = 50.0 + ((interaction_adjusted - 50.0) * 1.12)
        return round(max(0.0, min(100.0, contrast_adjusted)), 1)

    def resolve_blend_weights(
        self,
        *,
        insurance_readiness_score: float | None = None,
        risk: RiskComputation | None = None,
    ) -> tuple[float, float, float]:
        blend_weights = self.config.risk_blending_weights or {}
        try:
            env_weight = float(blend_weights.get("environmental", 0.0))
            struct_weight = float(blend_weights.get("structural", 0.0))
            readiness_weight = float(blend_weights.get("readiness", 0.0))
        except (TypeError, ValueError):
            env_weight = 0.0
            struct_weight = 0.0
            readiness_weight = 0.0
        if env_weight <= 0 or struct_weight <= 0:
            env_weight = self._group_weight(ENVIRONMENT_SUBMODELS)
            struct_weight = self._group_weight(STRUCTURE_SUBMODELS)
        if risk is not None:
            component_weights = risk.component_weight_fractions or {}
            regional_share = float(component_weights.get("regional_context", 0.0))
            surroundings_share = float(component_weights.get("property_surroundings", 0.0))
            structure_share = float(component_weights.get("structure_specific", 0.0))
            env_multiplier = 0.6 + min(0.7, (regional_share * 0.9) + (surroundings_share * 0.25))
            struct_multiplier = 0.35 + min(0.8, (structure_share * 0.9) + (surroundings_share * 0.35))
            if risk.geometry_basis == "point":
                struct_multiplier *= 0.55
            elif risk.geometry_basis == "parcel":
                struct_multiplier *= 0.80
            if risk.geometry_quality_score < 0.62:
                struct_multiplier *= 0.72
            env_weight *= env_multiplier
            struct_weight *= struct_multiplier
            if risk.geometry_basis == "point":
                readiness_weight *= 0.45
            elif risk.geometry_basis == "parcel":
                readiness_weight *= 0.75
            if risk.geometry_quality_score < 0.62:
                readiness_weight *= 0.55
            if risk.fallback_weight_fraction >= 0.55 or risk.observed_weight_fraction < 0.45:
                readiness_weight *= 0.35
            if risk.property_specificity_score < 55.0:
                readiness_weight *= 0.60
            if risk.geometry_basis == "point" and risk.fallback_factor_count > max(1, risk.observed_factor_count):
                readiness_weight *= 0.55
            total_core = env_weight + struct_weight
            if total_core > 0.0:
                if risk.geometry_basis == "point":
                    min_struct_share = 0.33
                elif risk.geometry_basis == "parcel":
                    min_struct_share = 0.37
                else:
                    min_struct_share = 0.42
                current_struct_share = struct_weight / total_core
                if current_struct_share < min_struct_share:
                    target_struct_weight = (env_weight * min_struct_share) / max(1e-6, (1.0 - min_struct_share))
                    struct_weight = max(struct_weight, target_struct_weight)
        if insurance_readiness_score is None:
            readiness_weight = 0.0
        return env_weight, struct_weight, max(0.0, readiness_weight)

    def compute_insurance_readiness(self, attrs: PropertyAttributes, context: WildfireContext, risk: RiskComputation) -> ReadinessRuleResult:
        attrs = normalize_property_attributes(attrs)
        p = self.config.readiness_penalties
        b = self.config.readiness_bonuses
        thresholds = self.config.readiness_thresholds or {}

        def _threshold(key: str, default: float) -> float:
            try:
                return float(thresholds.get(key, default))
            except (TypeError, ValueError):
                return default

        factors: List[dict] = []
        blockers: List[str] = []
        penalties_applied: Dict[str, float] = {}
        penalty = 0.0
        bonus = 0.0

        def add_penalty(name: str, value: float) -> None:
            nonlocal penalty
            penalty += value
            penalties_applied[name] = round(penalties_applied.get(name, 0.0) + value, 1)

        structure_inputs = (
            risk.submodel_scores.get("structure_vulnerability_risk", SubmodelResult(0.0, "", {})).key_inputs
            if isinstance(risk.submodel_scores.get("structure_vulnerability_risk"), SubmodelResult)
            else {}
        )
        try:
            structure_data_completeness = float((structure_inputs or {}).get("structure_data_completeness") or 0.0)
        except (TypeError, ValueError):
            structure_data_completeness = 0.0
        structure_assumption_mode = str((structure_inputs or {}).get("structure_assumption_mode") or "unknown")
        try:
            structure_score_confidence = float((structure_inputs or {}).get("structure_score_confidence") or 0.0)
        except (TypeError, ValueError):
            structure_score_confidence = 0.0

        structure_evidence_weight = sum(
            float((risk.weighted_contributions.get(name) or {}).get("weight", 0.0))
            for name in STRUCTURE_SPECIFIC_SUBMODELS
        )
        weak_structure_evidence = (
            (risk.geometry_basis == "point" and structure_evidence_weight < 0.22)
            or (risk.geometry_quality_score < 0.62 and structure_evidence_weight < 0.28)
            or structure_score_confidence < 55.0
        )
        if weak_structure_evidence:
            evidence_penalty = max(4.0, float(p.get("defensible_watch", 8.0)) * 0.45)
            factors.append(
                {
                    "name": "structure_evidence_quality",
                    "status": "watch",
                    "score_impact": -round(evidence_penalty, 1),
                    "detail": "Structure geometry evidence is weak; readiness is provisional and should not be treated as parcel-precise.",
                }
            )
            add_penalty("structure_evidence_quality", evidence_penalty)
            blockers.append("Structure geometry evidence is weak")
        if structure_score_confidence < 45.0:
            uncertainty_penalty = max(3.0, float(p.get("roof_watch", 6.0)) * 0.50)
            factors.append(
                {
                    "name": "structure_data_completeness",
                    "status": "watch",
                    "score_impact": -round(uncertainty_penalty, 1),
                    "detail": "Most structure hardening inputs are unknown; readiness remains a cautious estimate.",
                }
            )
            add_penalty("structure_data_completeness", uncertainty_penalty)
            blockers.append("Structure details mostly unknown")

        roof = (attrs.roof_type or "unknown").lower()
        if roof in {"wood", "untreated wood shake"}:
            factors.append({"name": "roof_material", "status": "fail", "score_impact": -p["roof_fail"], "detail": "Combustible roof material is a major insurer concern."})
            blockers.append("Combustible roof material")
            add_penalty("roof_material", p["roof_fail"])
        elif roof in {"class a", "metal", "tile", "composite"}:
            factors.append({"name": "roof_material", "status": "pass", "score_impact": b["roof_pass"], "detail": "Fire-rated roof supports insurability."})
            bonus += b["roof_pass"]
        else:
            factors.append({"name": "roof_material", "status": "watch", "score_impact": -p["roof_watch"], "detail": "Roof material unknown; treated as moderate risk."})
            add_penalty("roof_material", p["roof_watch"])

        vent = (attrs.vent_type or "unknown").lower()
        if "ember" in vent:
            factors.append({"name": "vent_quality", "status": "pass", "score_impact": b["vent_pass"], "detail": "Ember-resistant vents reduce intrusion risk."})
            bonus += b["vent_pass"]
        elif attrs.vent_type is None:
            factors.append({"name": "vent_quality", "status": "watch", "score_impact": -p["vent_watch"], "detail": "Vent details missing; assumed standard vents."})
            add_penalty("vent_quality", p["vent_watch"])
        else:
            factors.append({"name": "vent_quality", "status": "fail", "score_impact": -p["vent_fail"], "detail": "Non-ember-resistant vents increase ignition vulnerability."})
            add_penalty("vent_quality", p["vent_fail"])
            blockers.append("Non-ember-resistant venting")

        defensible_ft = attrs.defensible_space_ft
        if defensible_ft is None:
            factors.append(
                {
                    "name": "defensible_space",
                    "status": "watch",
                    "score_impact": -p["defensible_watch"],
                    "detail": "Defensible space details missing; readiness cannot confirm 30 ft clearance.",
                }
            )
            add_penalty("defensible_space", p["defensible_watch"])
        elif defensible_ft >= _threshold("defensible_space_pass_ft", 30.0):
            factors.append({"name": "defensible_space", "status": "pass", "score_impact": b["defensible_pass"], "detail": "Defensible space at/above 30 ft supports carrier criteria."})
            bonus += b["defensible_pass"]
        elif defensible_ft >= _threshold("defensible_space_watch_min_ft", 5.0):
            factors.append({"name": "defensible_space", "status": "watch", "score_impact": -p["defensible_watch"], "detail": "Defensible space below 30 ft may trigger underwriting concerns."})
            add_penalty("defensible_space", p["defensible_watch"])
            blockers.append("Defensible space below 30 ft")
        else:
            factors.append({"name": "defensible_space", "status": "fail", "score_impact": -p["defensible_fail"], "detail": "Minimal defensible space is a frequent non-renewal trigger."})
            add_penalty("defensible_space", p["defensible_fail"])
            blockers.append("Severely inadequate defensible space")

        structure_score = risk.submodel_scores["structure_vulnerability_risk"].score
        if structure_score_confidence < 60.0 and structure_assumption_mode in {"default_assumed", "mixed"}:
            cautious_penalty = 2.5 if structure_score_confidence >= 45.0 else 4.0
            factors.append(
                {
                    "name": "structure_vulnerability",
                    "status": "watch",
                    "score_impact": -cautious_penalty,
                    "detail": (
                        "Structure hardening score is driven by partial/default assumptions "
                        "and is treated conservatively."
                    ),
                }
            )
            add_penalty("structure_vulnerability", cautious_penalty)
        elif structure_score >= _threshold("structure_vulnerability_fail_score", 75.0):
            factors.append({"name": "structure_vulnerability", "status": "fail", "score_impact": -8.0, "detail": "Overall structure hardening signal is poor."})
            add_penalty("structure_vulnerability", 8.0)
        elif structure_score >= _threshold("structure_vulnerability_watch_score", 60.0):
            factors.append({"name": "structure_vulnerability", "status": "watch", "score_impact": -4.0, "detail": "Structure hardening is moderate and should be improved."})
            add_penalty("structure_vulnerability", 4.0)
        else:
            factors.append({"name": "structure_vulnerability", "status": "pass", "score_impact": 1.0, "detail": "Structure hardening signal is comparatively strong."})
            bonus += 1.0

        fuel_score = risk.submodel_scores["fuel_proximity_risk"].score
        defensible_inputs = risk.submodel_scores.get("defensible_space_risk", SubmodelResult(0.0, "", {})).key_inputs
        zone_0_5_density = defensible_inputs.get("ring_0_5_ft_vegetation_density")
        zone_5_30_density = defensible_inputs.get("ring_5_30_ft_vegetation_density")
        try:
            zone_0_5_density = float(zone_0_5_density) if zone_0_5_density is not None else None
        except (TypeError, ValueError):
            zone_0_5_density = None
        try:
            zone_5_30_density = float(zone_5_30_density) if zone_5_30_density is not None else None
        except (TypeError, ValueError):
            zone_5_30_density = None

        def _penalty_value(key: str, default: float) -> float:
            try:
                return float(p.get(key, default))
            except (TypeError, ValueError):
                return default

        immediate_zone_fail_penalty = _penalty_value("immediate_zone_0_5_fail", 8.5)
        immediate_zone_watch_penalty = _penalty_value("immediate_zone_0_5_watch", 4.0)
        intermediate_zone_fail_penalty = _penalty_value("intermediate_zone_5_30_fail", 5.0)
        intermediate_zone_watch_penalty = _penalty_value("intermediate_zone_5_30_watch", 2.5)

        if zone_0_5_density is not None and zone_0_5_density >= _threshold("zone_0_5_fail_density", 55.0):
            factors.append(
                {
                    "name": "immediate_zone_0_5_ft",
                    "status": "fail",
                    "score_impact": -immediate_zone_fail_penalty,
                    "detail": "Dense vegetation in the 0-5 ft zone is a direct home-ignition concern.",
                }
            )
            add_penalty("immediate_zone_0_5_ft", immediate_zone_fail_penalty)
            blockers.append("Dense vegetation within 5 ft of structure")
        elif zone_0_5_density is not None and zone_0_5_density >= _threshold("zone_0_5_watch_density", 40.0):
            factors.append(
                {
                    "name": "immediate_zone_0_5_ft",
                    "status": "watch",
                    "score_impact": -immediate_zone_watch_penalty,
                    "detail": "Some combustible vegetation is present in the 0-5 ft zone.",
                }
            )
            add_penalty("immediate_zone_0_5_ft", immediate_zone_watch_penalty)

        if zone_5_30_density is not None and zone_5_30_density >= _threshold("zone_5_30_fail_density", 68.0):
            factors.append(
                {
                    "name": "intermediate_zone_5_30_ft",
                    "status": "fail",
                    "score_impact": -intermediate_zone_fail_penalty,
                    "detail": "Vegetation pressure in the 5-30 ft zone may sustain flame spread to the structure.",
                }
            )
            add_penalty("intermediate_zone_5_30_ft", intermediate_zone_fail_penalty)
        elif zone_5_30_density is not None and zone_5_30_density >= _threshold("zone_5_30_watch_density", 52.0):
            factors.append(
                {
                    "name": "intermediate_zone_5_30_ft",
                    "status": "watch",
                    "score_impact": -intermediate_zone_watch_penalty,
                    "detail": "Moderate vegetation pressure in the 5-30 ft zone should be mitigated.",
                }
            )
            add_penalty("intermediate_zone_5_30_ft", intermediate_zone_watch_penalty)

        if fuel_score >= _threshold("adjacent_fuel_fail_score", 75.0):
            factors.append({"name": "adjacent_fuel_pressure", "status": "fail", "score_impact": -p["fuel_fail"], "detail": "Very high adjacent fuel pressure reduces readiness."})
            add_penalty("adjacent_fuel_pressure", p["fuel_fail"])
            blockers.append("Very high adjacent fuel proximity")
        elif fuel_score >= _threshold("adjacent_fuel_watch_score", 55.0):
            factors.append({"name": "adjacent_fuel_pressure", "status": "watch", "score_impact": -p["fuel_watch"], "detail": "Moderate-high adjacent fuel pressure warrants mitigation."})
            add_penalty("adjacent_fuel_pressure", p["fuel_watch"])
        else:
            factors.append({"name": "adjacent_fuel_pressure", "status": "pass", "score_impact": b["fuel_pass"], "detail": "Low adjacent fuel pressure supports readiness."})
            bonus += b["fuel_pass"]

        vegetation_score = risk.submodel_scores["vegetation_intensity_risk"].score
        if vegetation_score >= _threshold("vegetation_intensity_fail_score", 75.0):
            factors.append({"name": "vegetation_intensity", "status": "fail", "score_impact": -p["vegetation_fail"], "detail": "High near-structure vegetation intensity elevates spread potential."})
            add_penalty("vegetation_intensity", p["vegetation_fail"])
            blockers.append("High vegetation intensity near structure")
        elif vegetation_score >= _threshold("vegetation_intensity_watch_score", 55.0):
            factors.append({"name": "vegetation_intensity", "status": "watch", "score_impact": -p["vegetation_watch"], "detail": "Moderate vegetation intensity should be reduced in home ignition zones."})
            add_penalty("vegetation_intensity", p["vegetation_watch"])
        else:
            factors.append({"name": "vegetation_intensity", "status": "pass", "score_impact": b["vegetation_pass"], "detail": "Vegetation intensity is comparatively manageable near structure."})
            bonus += b["vegetation_pass"]

        ember_score = risk.submodel_scores["ember_exposure_risk"].score
        if ember_score >= _threshold("ember_exposure_fail_score", 80.0):
            factors.append({"name": "severe_ember_exposure", "status": "fail", "score_impact": -10.0, "detail": "Extreme ember exposure can materially reduce availability."})
            add_penalty("severe_ember_exposure", 10.0)
            blockers.append("Severe ember exposure")
        elif ember_score >= _threshold("ember_exposure_watch_score", 65.0):
            factors.append({"name": "severe_ember_exposure", "status": "watch", "score_impact": -5.0, "detail": "Elevated ember exposure should be mitigated."})
            add_penalty("severe_ember_exposure", 5.0)

        severe_env_inputs = [
            ember_score,
            risk.submodel_scores["historic_fire_risk"].score,
        ]
        if context.hazard_severity_index is not None:
            severe_env_inputs.append(float(context.hazard_severity_index))
        severe_env = max(severe_env_inputs) if severe_env_inputs else 0.0
        if severe_env >= _threshold("severe_environment_fail_score", 85.0):
            factors.append({"name": "severe_environmental_hazard", "status": "fail", "score_impact": -p["severe_env_fail"], "detail": "Severe environmental hazard conditions reduce availability."})
            add_penalty("severe_environmental_hazard", p["severe_env_fail"])
            blockers.append("Severe environmental hazard conditions")
        elif severe_env >= _threshold("severe_environment_watch_score", 70.0):
            factors.append({"name": "severe_environmental_hazard", "status": "watch", "score_impact": -p["severe_env_watch"], "detail": "Elevated environmental hazard conditions present."})
            add_penalty("severe_environmental_hazard", p["severe_env_watch"])
        else:
            factors.append({"name": "severe_environmental_hazard", "status": "pass", "score_impact": b["severe_env_pass"], "detail": "Environmental hazard signal is within moderate limits."})
            bonus += b["severe_env_pass"]

        readiness = max(0.0, min(100.0, round(100.0 - penalty + bonus, 1)))

        if blockers:
            summary = f"Readiness reduced by {len(blockers)} blocker(s): {', '.join(sorted(set(blockers))[:3])}."
        else:
            summary = "No major insurer-style blockers detected; maintain mitigation posture and documentation."

        return ReadinessRuleResult(
            insurance_readiness_score=readiness,
            readiness_factors=factors,
            readiness_blockers=sorted(set(blockers)),
            readiness_penalties=penalties_applied,
            readiness_summary=summary,
        )
