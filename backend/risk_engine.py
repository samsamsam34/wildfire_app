from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from backend.models import PropertyAttributes, RiskDrivers
from backend.wildfire_data import WildfireContext


# MVP insurer-oriented heuristic weights. These are not underwriting-approved models.
SUBMODEL_WEIGHTS: Dict[str, float] = {
    "ember_exposure": 0.15,
    "flame_contact_exposure": 0.14,
    "topography_risk": 0.12,
    "fuel_proximity_risk": 0.14,
    "vegetation_intensity_risk": 0.12,
    "historic_fire_risk": 0.11,
    "home_hardening_risk": 0.12,
    "defensible_space_risk": 0.10,
}

ENVIRONMENT_SUBMODELS = {
    "ember_exposure",
    "flame_contact_exposure",
    "topography_risk",
    "fuel_proximity_risk",
    "vegetation_intensity_risk",
    "historic_fire_risk",
}

STRUCTURE_SUBMODELS = {
    "home_hardening_risk",
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
    readiness_summary: str


@dataclass
class RiskComputation:
    total_score: float
    drivers: RiskDrivers
    assumptions: List[str]
    submodel_scores: Dict[str, SubmodelResult]
    weighted_contributions: Dict[str, dict]


class RiskEngine:
    def geocode_stub(self, address: str) -> Tuple[float, float]:
        digest = hashlib.md5(address.strip().lower().encode("utf-8")).hexdigest()
        lat_seed = int(digest[:8], 16) / 0xFFFFFFFF
        lon_seed = int(digest[8:16], 16) / 0xFFFFFFFF
        latitude = 32.5 + (lat_seed * 9.5)
        longitude = -124.3 + (lon_seed * 10.0)
        return round(latitude, 6), round(longitude, 6)

    def _build_submodels(self, attrs: PropertyAttributes, context: WildfireContext) -> Dict[str, SubmodelResult]:
        submodels: Dict[str, SubmodelResult] = {}

        roof = (attrs.roof_type or "unknown").lower()
        vent = (attrs.vent_type or "unknown").lower()
        defensible_ft = attrs.defensible_space_ft if attrs.defensible_space_ft is not None else 15.0

        ember_assumptions: List[str] = []
        roof_ignition = 75.0 if roof in {"wood", "untreated wood shake"} else 25.0 if roof in {"class a", "metal", "tile", "composite"} else 50.0
        if attrs.roof_type is None:
            ember_assumptions.append("Roof type missing; ember vulnerability assumed moderate.")
        vent_ignition = 20.0 if "ember" in vent else 65.0
        if attrs.vent_type is None:
            ember_assumptions.append("Vent type missing; ember entry vulnerability assumed moderate-high.")
        ember_score = round(
            0.35 * context.burn_probability_index
            + 0.25 * context.hazard_severity_index
            + 0.20 * vent_ignition
            + 0.20 * roof_ignition,
            1,
        )
        submodels["ember_exposure"] = SubmodelResult(
            score=max(0.0, min(100.0, ember_score)),
            explanation="Ember exposure reflects ember storm likelihood and structure ember vulnerability.",
            key_inputs={
                "burn_probability": context.burn_probability_index,
                "hazard_severity": context.hazard_severity_index,
                "roof_ignition_proxy": roof_ignition,
                "vent_ignition_proxy": vent_ignition,
            },
            assumptions=ember_assumptions,
        )

        flame_score = round(
            0.40 * context.fuel_index
            + 0.35 * context.wildland_distance_index
            + 0.25 * max(0.0, min(100.0, 100.0 - defensible_ft * 2.2)),
            1,
        )
        flame_assumptions = []
        if attrs.defensible_space_ft is None:
            flame_assumptions.append("Defensible space missing; assumed 15 ft for flame-contact model.")
        submodels["flame_contact_exposure"] = SubmodelResult(
            score=max(0.0, min(100.0, flame_score)),
            explanation="Flame-contact exposure reflects near-structure fuels and vegetation continuity.",
            key_inputs={
                "fuel_index": context.fuel_index,
                "wildland_distance_index": context.wildland_distance_index,
                "defensible_space_ft": defensible_ft,
            },
            assumptions=flame_assumptions,
        )

        topo_score = round(0.70 * context.slope_index + 0.30 * context.aspect_index, 1)
        submodels["topography_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, topo_score)),
            explanation="Topography risk captures slope-driven spread potential and aspect dryness effects.",
            key_inputs={
                "slope_index": context.slope_index,
                "aspect_index": context.aspect_index,
            },
        )

        submodels["fuel_proximity_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, round(context.wildland_distance_index, 1))),
            explanation="Fuel proximity risk reflects distance to contiguous wildland vegetation.",
            key_inputs={"wildland_distance_index": context.wildland_distance_index},
        )

        veg_score = round(0.45 * context.fuel_index + 0.30 * context.canopy_index + 0.25 * context.moisture_index, 1)
        submodels["vegetation_intensity_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, veg_score)),
            explanation="Vegetation intensity risk captures fuel loading, canopy continuity, and dryness.",
            key_inputs={
                "fuel_index": context.fuel_index,
                "canopy_index": context.canopy_index,
                "moisture_index": context.moisture_index,
            },
        )

        submodels["historic_fire_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, round(context.historic_fire_index, 1))),
            explanation="Historic fire risk reflects nearby fire recurrence and perimeter history.",
            key_inputs={"historic_fire_index": context.historic_fire_index},
        )

        hardening_assumptions: List[str] = []
        construction_risk = 55.0
        if attrs.construction_year is None:
            hardening_assumptions.append("Construction year missing; hardening baseline assumed pre-modern code profile.")
        elif attrs.construction_year >= 2015:
            construction_risk = 30.0
        elif attrs.construction_year >= 2008:
            construction_risk = 42.0

        home_hardening_score = round(
            0.45 * roof_ignition + 0.35 * vent_ignition + 0.20 * construction_risk,
            1,
        )
        submodels["home_hardening_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, home_hardening_score)),
            explanation="Home hardening risk reflects ember-resistant construction readiness.",
            key_inputs={
                "roof_ignition_proxy": roof_ignition,
                "vent_ignition_proxy": vent_ignition,
                "construction_risk_proxy": construction_risk,
            },
            assumptions=hardening_assumptions,
        )

        defensible_risk = round(
            max(0.0, min(100.0, 95.0 - defensible_ft * 2.6)) * 0.70 + 0.30 * context.fuel_index,
            1,
        )
        defensible_assumptions: List[str] = []
        if attrs.defensible_space_ft is None:
            defensible_assumptions.append("Defensible space missing; assumed 15 ft for defensible-space model.")

        submodels["defensible_space_risk"] = SubmodelResult(
            score=max(0.0, min(100.0, defensible_risk)),
            explanation="Defensible space risk reflects clearance sufficiency under local fuel pressure.",
            key_inputs={
                "defensible_space_ft": defensible_ft,
                "fuel_index": context.fuel_index,
            },
            assumptions=defensible_assumptions,
        )

        return submodels

    def score(
        self,
        attrs: PropertyAttributes,
        lat: float,
        lon: float,
        context: WildfireContext,
    ) -> RiskComputation:
        _ = (lat, lon)
        submodels = self._build_submodels(attrs, context)

        weighted_contributions: Dict[str, dict] = {}
        total = 0.0
        assumptions: List[str] = list(context.assumptions)

        for name, result in submodels.items():
            weight = SUBMODEL_WEIGHTS[name]
            contribution = round(weight * result.score, 2)
            weighted_contributions[name] = {
                "weight": weight,
                "score": round(result.score, 1),
                "contribution": contribution,
            }
            total += contribution
            assumptions.extend(result.assumptions)

        env_weight = sum(SUBMODEL_WEIGHTS[n] for n in ENVIRONMENT_SUBMODELS)
        struct_weight = sum(SUBMODEL_WEIGHTS[n] for n in STRUCTURE_SUBMODELS)

        environmental_driver = round(
            sum(weighted_contributions[n]["contribution"] for n in ENVIRONMENT_SUBMODELS) / env_weight,
            1,
        )
        structural_driver = round(
            sum(weighted_contributions[n]["contribution"] for n in STRUCTURE_SUBMODELS) / struct_weight,
            1,
        )

        access_exposure = round(0.65 * context.wildland_distance_index + 0.35 * context.historic_fire_index, 1)
        assumptions.append("Access exposure remains provisional and is not a weighted submodel in v1.2.0.")

        return RiskComputation(
            total_score=round(max(0.0, min(100.0, total)), 1),
            drivers=RiskDrivers(
                environmental=environmental_driver,
                structural=structural_driver,
                access_exposure=access_exposure,
            ),
            assumptions=sorted(set(assumptions)),
            submodel_scores=submodels,
            weighted_contributions=weighted_contributions,
        )

    def compute_insurance_readiness(
        self,
        attrs: PropertyAttributes,
        context: WildfireContext,
        risk: RiskComputation,
    ) -> ReadinessRuleResult:
        factors: List[dict] = []
        blockers: List[str] = []
        penalty = 0.0
        bonus = 0.0

        roof = (attrs.roof_type or "unknown").lower()
        if roof in {"wood", "untreated wood shake"}:
            factors.append({"name": "roof_material", "status": "fail", "score_impact": -26.0, "detail": "Combustible roof material is a major insurer concern."})
            blockers.append("Combustible roof material")
            penalty += 26.0
        elif roof in {"class a", "metal", "tile", "composite"}:
            factors.append({"name": "roof_material", "status": "pass", "score_impact": 4.0, "detail": "Fire-rated roof supports insurability."})
            bonus += 4.0
        else:
            factors.append({"name": "roof_material", "status": "watch", "score_impact": -8.0, "detail": "Roof material unknown; treated as moderate risk."})
            penalty += 8.0

        vent = (attrs.vent_type or "unknown").lower()
        if "ember" in vent:
            factors.append({"name": "vent_quality", "status": "pass", "score_impact": 3.0, "detail": "Ember-resistant vents reduce intrusion risk."})
            bonus += 3.0
        elif attrs.vent_type is None:
            factors.append({"name": "vent_quality", "status": "watch", "score_impact": -7.0, "detail": "Vent details missing; assumed standard vents."})
            penalty += 7.0
        else:
            factors.append({"name": "vent_quality", "status": "fail", "score_impact": -12.0, "detail": "Non-ember-resistant vents increase ignition vulnerability."})
            penalty += 12.0
            blockers.append("Non-ember-resistant venting")

        defensible_ft = attrs.defensible_space_ft if attrs.defensible_space_ft is not None else 15.0
        if defensible_ft >= 30:
            factors.append({"name": "defensible_space", "status": "pass", "score_impact": 5.0, "detail": "Defensible space at/above 30 ft supports carrier criteria."})
            bonus += 5.0
        elif defensible_ft >= 5:
            factors.append({"name": "defensible_space", "status": "watch", "score_impact": -14.0, "detail": "Defensible space below 30 ft may trigger underwriting concerns."})
            penalty += 14.0
            blockers.append("Defensible space below 30 ft")
        else:
            factors.append({"name": "defensible_space", "status": "fail", "score_impact": -25.0, "detail": "Minimal defensible space is a frequent non-renewal trigger."})
            penalty += 25.0
            blockers.append("Severely inadequate defensible space")

        if risk.submodel_scores["fuel_proximity_risk"].score >= 75:
            factors.append({"name": "adjacent_fuel_pressure", "status": "fail", "score_impact": -15.0, "detail": "Very high adjacent fuel pressure reduces readiness."})
            penalty += 15.0
            blockers.append("Very high adjacent fuel proximity")
        elif risk.submodel_scores["fuel_proximity_risk"].score >= 55:
            factors.append({"name": "adjacent_fuel_pressure", "status": "watch", "score_impact": -8.0, "detail": "Moderate-high adjacent fuel pressure warrants mitigation."})
            penalty += 8.0
        else:
            factors.append({"name": "adjacent_fuel_pressure", "status": "pass", "score_impact": 2.0, "detail": "Low adjacent fuel pressure supports readiness."})
            bonus += 2.0

        if risk.submodel_scores["vegetation_intensity_risk"].score >= 75:
            factors.append({"name": "vegetation_intensity", "status": "fail", "score_impact": -14.0, "detail": "High vegetation intensity can limit insurer appetite."})
            penalty += 14.0
            blockers.append("High vegetation intensity near structure")
        elif risk.submodel_scores["vegetation_intensity_risk"].score >= 60:
            factors.append({"name": "vegetation_intensity", "status": "watch", "score_impact": -7.0, "detail": "Vegetation intensity is elevated."})
            penalty += 7.0
        else:
            factors.append({"name": "vegetation_intensity", "status": "pass", "score_impact": 2.0, "detail": "Vegetation intensity is relatively controlled."})
            bonus += 2.0

        severe_env = max(
            risk.submodel_scores["ember_exposure"].score,
            risk.submodel_scores["historic_fire_risk"].score,
            context.hazard_severity_index,
        )
        if severe_env >= 85:
            factors.append({"name": "severe_environmental_hazard", "status": "fail", "score_impact": -12.0, "detail": "Severe environmental hazard conditions reduce availability."})
            penalty += 12.0
            blockers.append("Severe environmental hazard conditions")
        elif severe_env >= 70:
            factors.append({"name": "severe_environmental_hazard", "status": "watch", "score_impact": -6.0, "detail": "Elevated environmental hazard conditions present."})
            penalty += 6.0
        else:
            factors.append({"name": "severe_environmental_hazard", "status": "pass", "score_impact": 1.0, "detail": "Environmental hazard signal is within moderate limits."})
            bonus += 1.0

        readiness = max(0.0, min(100.0, round(100.0 - penalty + bonus, 1)))

        if blockers:
            summary = f"Readiness reduced by {len(blockers)} blocker(s): {', '.join(sorted(set(blockers))[:3])}."
        else:
            summary = "No major insurer-style blockers detected; maintain mitigation posture and documentation."

        return ReadinessRuleResult(
            insurance_readiness_score=readiness,
            readiness_factors=factors,
            readiness_blockers=sorted(set(blockers)),
            readiness_summary=summary,
        )
