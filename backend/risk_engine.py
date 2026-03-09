from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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


@dataclass
class SubmodelResult:
    score: float
    explanation: str
    key_inputs: Dict[str, object]
    assumptions: List[str] = field(default_factory=list)


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

        roof = (attrs.roof_type or "unknown").lower()
        vent = (attrs.vent_type or "unknown").lower()
        defensible_ft = attrs.defensible_space_ft
        ring_metrics = context.structure_ring_metrics or {}
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
            return round(numerator / denominator, 1)

        def ring_density(ring_key: str) -> float | None:
            alias = ring_key.replace("ring_", "zone_")
            metrics = ring_metrics.get(ring_key) or ring_metrics.get(alias) or {}
            value = metrics.get("vegetation_density") if isinstance(metrics, dict) else None
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        ring_0_5_density = ring_density("ring_0_5_ft")
        ring_5_30_density = ring_density("ring_5_30_ft")
        ring_30_100_density = ring_density("ring_30_100_ft")

        available_ring_densities = [
            density
            for density in [ring_0_5_density, ring_5_30_density, ring_30_100_density]
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
                (0.35, burn_probability_index, "Burn probability unavailable for ember model."),
                (0.25, hazard_severity_index, "Hazard severity unavailable for ember model."),
                (0.20, vent_ignition, "Vent type unavailable for ember model."),
                (0.20, roof_ignition, "Roof type unavailable for ember model."),
            ],
            ember_assumptions,
        )
        submodels["ember_exposure_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, ember_score)),
            explanation="Ember exposure reflects ember storm likelihood and structure ember vulnerability.",
            key_inputs={
                "burn_probability": burn_probability_index,
                "hazard_severity": hazard_severity_index,
                "roof_ignition_proxy": roof_ignition,
                "vent_ignition_proxy": vent_ignition,
            },
            assumptions=ember_assumptions + context_assumptions,
        )

        flame_assumptions: List[str] = []
        if attrs.defensible_space_ft is None:
            flame_assumptions.append("Defensible space missing; flame-contact model excludes defensible-space contribution.")
        if ring_0_5_density is None or ring_5_30_density is None:
            flame_assumptions.append("Structure-ring vegetation metrics unavailable; flame-contact model used point-based vegetation context.")

        close_in_veg_pressure = ring_0_5_density if ring_0_5_density is not None else canopy_index
        zone1_veg_pressure = ring_5_30_density if ring_5_30_density is not None else fuel_index
        defensible_component = (
            max(0.0, min(100.0, 100.0 - defensible_ft * 2.2))
            if attrs.defensible_space_ft is not None
            else None
        )
        near_ring_component = None
        if close_in_veg_pressure is not None and zone1_veg_pressure is not None:
            near_ring_component = 0.55 * close_in_veg_pressure + 0.45 * zone1_veg_pressure
        elif close_in_veg_pressure is not None:
            near_ring_component = close_in_veg_pressure
        elif zone1_veg_pressure is not None:
            near_ring_component = zone1_veg_pressure

        flame_score = weighted_score(
            [
                (0.30, fuel_index, "Fuel model unavailable for flame-contact model."),
                (0.25, wildland_distance_index, "Wildland distance unavailable for flame-contact model."),
                (0.20, defensible_component, "Defensible space unavailable for flame-contact model."),
                (0.25, near_ring_component, "Near-structure vegetation unavailable for flame-contact model."),
            ],
            flame_assumptions,
        )
        submodels["flame_contact_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, flame_score)),
            explanation="Flame-contact risk reflects near-structure fuels and vegetation continuity.",
            key_inputs={
                "fuel_index": fuel_index,
                "wildland_distance_index": wildland_distance_index,
                "defensible_space_ft": defensible_ft,
                "ring_0_5_ft_vegetation_density": ring_0_5_density,
                "ring_5_30_ft_vegetation_density": ring_5_30_density,
            },
            assumptions=flame_assumptions + context_assumptions,
        )

        slope_assumptions: List[str] = []
        slope_score = weighted_score(
            [
                (0.70, slope_index, "Slope input unavailable for topography model."),
                (0.30, aspect_index, "Aspect input unavailable for topography model."),
            ],
            slope_assumptions,
        )
        submodels["slope_topography_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, slope_score)),
            explanation="Slope/topography risk captures terrain-driven spread amplification.",
            key_inputs={"slope_index": slope_index, "aspect_index": aspect_index},
            assumptions=slope_assumptions + list(context_assumptions),
        )

        fuel_proximity_assumptions: List[str] = []
        if ring_30_100_density is None:
            fuel_proximity_assumptions.append(
                "Structure-ring 30-100 ft vegetation metrics unavailable; fuel proximity used point-based distance index."
            )

        outer_ring_pressure = ring_30_100_density if ring_30_100_density is not None else canopy_index
        fuel_proximity_score = weighted_score(
            [
                (0.75, wildland_distance_index, "Wildland distance unavailable for fuel proximity model."),
                (0.25, outer_ring_pressure, "30-100 ft ring/cover unavailable for fuel proximity model."),
            ],
            fuel_proximity_assumptions,
        )
        submodels["fuel_proximity_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, fuel_proximity_score)),
            explanation="Fuel proximity risk reflects distance to contiguous wildland vegetation.",
            key_inputs={
                "wildland_distance_index": wildland_distance_index,
                "ring_30_100_ft_vegetation_density": ring_30_100_density,
            },
            assumptions=fuel_proximity_assumptions + context_assumptions,
        )

        vegetation_assumptions: List[str] = []
        if ring_density_average is None:
            vegetation_assumptions.append(
                "Structure-ring vegetation metrics unavailable; vegetation intensity used 100m point-neighborhood context."
            )
        structure_ring_veg = ring_density_average if ring_density_average is not None else canopy_index
        vegetation_score = weighted_score(
            [
                (0.35, fuel_index, "Fuel model unavailable for vegetation intensity model."),
                (0.20, canopy_index, "Canopy cover unavailable for vegetation intensity model."),
                (0.20, moisture_index, "Moisture input unavailable for vegetation intensity model."),
                (0.25, structure_ring_veg, "Ring vegetation unavailable for vegetation intensity model."),
            ],
            vegetation_assumptions,
        )
        submodels["vegetation_intensity_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, vegetation_score)),
            explanation="Vegetation intensity risk captures fuel loading, canopy continuity, and dryness.",
            key_inputs={
                "fuel_index": fuel_index,
                "canopy_index": canopy_index,
                "moisture_index": moisture_index,
                "ring_vegetation_density_avg": ring_density_average,
            },
            assumptions=vegetation_assumptions + context_assumptions,
        )

        historic_assumptions: List[str] = []
        if historic_fire_index is None:
            historic_assumptions.append("Historic fire recurrence unavailable for this location.")
        submodels["historic_fire_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, 0.0 if historic_fire_index is None else round(historic_fire_index, 1))),
            explanation="Historic fire risk reflects nearby fire recurrence and perimeter history.",
            key_inputs={"historic_fire_index": historic_fire_index},
            assumptions=historic_assumptions + list(context_assumptions),
        )

        structure_assumptions: List[str] = []
        construction_risk = None
        if attrs.construction_year is None:
            structure_assumptions.append("Construction year missing; structure vulnerability model excludes age contribution.")
        elif attrs.construction_year >= 2015:
            construction_risk = 30.0
        elif attrs.construction_year >= 2008:
            construction_risk = 42.0
        else:
            construction_risk = 55.0

        structure_score = weighted_score(
            [
                (0.45, roof_ignition, "Roof type unavailable for structure vulnerability model."),
                (0.35, vent_ignition, "Vent type unavailable for structure vulnerability model."),
                (0.20, construction_risk, "Construction year unavailable for structure vulnerability model."),
            ],
            structure_assumptions,
        )
        submodels["structure_vulnerability_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, structure_score)),
            explanation="Structure vulnerability risk reflects hardening quality against ember and radiant heat intrusion.",
            key_inputs={
                "roof_ignition_proxy": roof_ignition,
                "vent_ignition_proxy": vent_ignition,
                "construction_risk_proxy": construction_risk,
            },
            assumptions=structure_assumptions,
        )

        defensible_assumptions: List[str] = []
        if attrs.defensible_space_ft is None:
            defensible_assumptions.append("Defensible space missing; defensible-space model excludes clearance distance contribution.")
        if ring_0_5_density is None and ring_5_30_density is None:
            defensible_assumptions.append(
                "Structure-ring vegetation metrics unavailable; defensible-space pressure used fuel index proxy."
            )

        zone_pressure_values = [d for d in [ring_0_5_density, ring_5_30_density] if d is not None]
        zone_pressure = (
            round(sum(zone_pressure_values) / len(zone_pressure_values), 1)
            if zone_pressure_values
            else fuel_index
        )
        defensible_clearance_component = (
            max(0.0, min(100.0, 95.0 - defensible_ft * 2.6))
            if attrs.defensible_space_ft is not None
            else None
        )
        defensible_score = weighted_score(
            [
                (0.55, defensible_clearance_component, "Defensible space value unavailable for defensible-space model."),
                (0.25, fuel_index, "Fuel model unavailable for defensible-space model."),
                (0.20, zone_pressure, "Ring vegetation unavailable for defensible-space model."),
            ],
            defensible_assumptions,
        )
        submodels["defensible_space_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, defensible_score)),
            explanation="Defensible space risk reflects clearance sufficiency under local fuel pressure.",
            key_inputs={
                "defensible_space_ft": defensible_ft,
                "fuel_index": fuel_index,
                "ring_0_5_ft_vegetation_density": ring_0_5_density,
                "ring_5_30_ft_vegetation_density": ring_5_30_density,
            },
            assumptions=defensible_assumptions + context_assumptions,
        )

        return submodels

    def score(self, attrs: PropertyAttributes, lat: float, lon: float, context: WildfireContext) -> RiskComputation:
        submodels = self._build_submodels(attrs, context)

        weighted_contributions: Dict[str, dict] = {}
        total = 0.0
        assumptions: List[str] = list(context.assumptions)

        for name, result in submodels.items():
            weight = self.config.submodel_weights[name]
            contribution = round(weight * result.score, 2)
            weighted_contributions[name] = {"weight": weight, "score": round(result.score, 1), "contribution": contribution}
            total += contribution
            assumptions.extend(result.assumptions)

        env_weight = sum(self.config.submodel_weights[n] for n in ENVIRONMENT_SUBMODELS)
        struct_weight = sum(self.config.submodel_weights[n] for n in STRUCTURE_SUBMODELS)

        environmental_driver = round(sum(weighted_contributions[n]["contribution"] for n in ENVIRONMENT_SUBMODELS) / env_weight, 1)
        structural_driver = round(sum(weighted_contributions[n]["contribution"] for n in STRUCTURE_SUBMODELS) / struct_weight, 1)

        # Provisional placeholder only: no parcel driveway/egress model yet in MVP.
        access_exposure = round(max(0.0, min(100.0, abs(lat * 2.1 + lon * 1.3) % 100.0)), 1)
        assumptions.append(
            "Access exposure remains provisional (synthetic placeholder) and is not a weighted submodel in v1.5.0."
        )

        return RiskComputation(
            total_score=round(max(0.0, min(100.0, total)), 1),
            drivers=RiskDrivers(environmental=environmental_driver, structural=structural_driver, access_exposure=access_exposure),
            assumptions=sorted(set(assumptions)),
            submodel_scores=submodels,
            weighted_contributions=weighted_contributions,
        )

    def _group_weight(self, group: set[str]) -> float:
        return sum(self.config.submodel_weights.get(name, 0.0) for name in group)

    def _group_score(self, risk: RiskComputation, group: set[str]) -> float:
        weight = self._group_weight(group)
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

        def _to_float(value: object) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        defensible_inputs = risk.submodel_scores.get("defensible_space_risk", SubmodelResult(0.0, "", {})).key_inputs
        fuel_inputs = risk.submodel_scores.get("fuel_proximity_risk", SubmodelResult(0.0, "", {})).key_inputs

        zone_0_5 = _to_float(defensible_inputs.get("ring_0_5_ft_vegetation_density"))
        zone_5_30 = _to_float(defensible_inputs.get("ring_5_30_ft_vegetation_density"))
        zone_30_100 = _to_float(fuel_inputs.get("ring_30_100_ft_vegetation_density"))

        ring_penalty = 0.0
        if zone_0_5 is not None:
            ring_penalty += max(0.0, (zone_0_5 - 55.0) * 0.18)
        if zone_5_30 is not None:
            ring_penalty += max(0.0, (zone_5_30 - 60.0) * 0.12)
        if zone_30_100 is not None:
            ring_penalty += max(0.0, (zone_30_100 - 65.0) * 0.08)

        return round(max(0.0, min(100.0, base + ring_penalty)), 1)

    def compute_blended_wildfire_score(self, site_hazard_score: float, home_ignition_vulnerability_score: float) -> float:
        env_weight = self._group_weight(ENVIRONMENT_SUBMODELS)
        struct_weight = self._group_weight(STRUCTURE_SUBMODELS)
        denom = env_weight + struct_weight
        if denom <= 0:
            return 0.0

        blended = (site_hazard_score * env_weight + home_ignition_vulnerability_score * struct_weight) / denom
        return round(max(0.0, min(100.0, blended)), 1)

    def compute_insurance_readiness(self, attrs: PropertyAttributes, context: WildfireContext, risk: RiskComputation) -> ReadinessRuleResult:
        attrs = normalize_property_attributes(attrs)
        p = self.config.readiness_penalties
        b = self.config.readiness_bonuses

        factors: List[dict] = []
        blockers: List[str] = []
        penalties_applied: Dict[str, float] = {}
        penalty = 0.0
        bonus = 0.0

        def add_penalty(name: str, value: float) -> None:
            nonlocal penalty
            penalty += value
            penalties_applied[name] = round(penalties_applied.get(name, 0.0) + value, 1)

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
        elif defensible_ft >= 30:
            factors.append({"name": "defensible_space", "status": "pass", "score_impact": b["defensible_pass"], "detail": "Defensible space at/above 30 ft supports carrier criteria."})
            bonus += b["defensible_pass"]
        elif defensible_ft >= 5:
            factors.append({"name": "defensible_space", "status": "watch", "score_impact": -p["defensible_watch"], "detail": "Defensible space below 30 ft may trigger underwriting concerns."})
            add_penalty("defensible_space", p["defensible_watch"])
            blockers.append("Defensible space below 30 ft")
        else:
            factors.append({"name": "defensible_space", "status": "fail", "score_impact": -p["defensible_fail"], "detail": "Minimal defensible space is a frequent non-renewal trigger."})
            add_penalty("defensible_space", p["defensible_fail"])
            blockers.append("Severely inadequate defensible space")

        structure_score = risk.submodel_scores["structure_vulnerability_risk"].score
        if structure_score >= 75:
            factors.append({"name": "structure_vulnerability", "status": "fail", "score_impact": -8.0, "detail": "Overall structure hardening signal is poor."})
            add_penalty("structure_vulnerability", 8.0)
        elif structure_score >= 60:
            factors.append({"name": "structure_vulnerability", "status": "watch", "score_impact": -4.0, "detail": "Structure hardening is moderate and should be improved."})
            add_penalty("structure_vulnerability", 4.0)
        else:
            factors.append({"name": "structure_vulnerability", "status": "pass", "score_impact": 1.0, "detail": "Structure hardening signal is comparatively strong."})
            bonus += 1.0

        fuel_score = risk.submodel_scores["fuel_proximity_risk"].score
        if fuel_score >= 75:
            factors.append({"name": "adjacent_fuel_pressure", "status": "fail", "score_impact": -p["fuel_fail"], "detail": "Very high adjacent fuel pressure reduces readiness."})
            add_penalty("adjacent_fuel_pressure", p["fuel_fail"])
            blockers.append("Very high adjacent fuel proximity")
        elif fuel_score >= 55:
            factors.append({"name": "adjacent_fuel_pressure", "status": "watch", "score_impact": -p["fuel_watch"], "detail": "Moderate-high adjacent fuel pressure warrants mitigation."})
            add_penalty("adjacent_fuel_pressure", p["fuel_watch"])
        else:
            factors.append({"name": "adjacent_fuel_pressure", "status": "pass", "score_impact": b["fuel_pass"], "detail": "Low adjacent fuel pressure supports readiness."})
            bonus += b["fuel_pass"]

        vegetation_score = risk.submodel_scores["vegetation_intensity_risk"].score
        if vegetation_score >= 75:
            factors.append({"name": "vegetation_intensity", "status": "fail", "score_impact": -p["vegetation_fail"], "detail": "High near-structure vegetation intensity elevates spread potential."})
            add_penalty("vegetation_intensity", p["vegetation_fail"])
            blockers.append("High vegetation intensity near structure")
        elif vegetation_score >= 55:
            factors.append({"name": "vegetation_intensity", "status": "watch", "score_impact": -p["vegetation_watch"], "detail": "Moderate vegetation intensity should be reduced in home ignition zones."})
            add_penalty("vegetation_intensity", p["vegetation_watch"])
        else:
            factors.append({"name": "vegetation_intensity", "status": "pass", "score_impact": b["vegetation_pass"], "detail": "Vegetation intensity is comparatively manageable near structure."})
            bonus += b["vegetation_pass"]

        ember_score = risk.submodel_scores["ember_exposure_risk"].score
        if ember_score >= 80:
            factors.append({"name": "severe_ember_exposure", "status": "fail", "score_impact": -10.0, "detail": "Extreme ember exposure can materially reduce availability."})
            add_penalty("severe_ember_exposure", 10.0)
            blockers.append("Severe ember exposure")
        elif ember_score >= 65:
            factors.append({"name": "severe_ember_exposure", "status": "watch", "score_impact": -5.0, "detail": "Elevated ember exposure should be mitigated."})
            add_penalty("severe_ember_exposure", 5.0)

        severe_env_inputs = [
            ember_score,
            risk.submodel_scores["historic_fire_risk"].score,
        ]
        if context.hazard_severity_index is not None:
            severe_env_inputs.append(float(context.hazard_severity_index))
        severe_env = max(severe_env_inputs) if severe_env_inputs else 0.0
        if severe_env >= 85:
            factors.append({"name": "severe_environmental_hazard", "status": "fail", "score_impact": -p["severe_env_fail"], "detail": "Severe environmental hazard conditions reduce availability."})
            add_penalty("severe_environmental_hazard", p["severe_env_fail"])
            blockers.append("Severe environmental hazard conditions")
        elif severe_env >= 70:
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
