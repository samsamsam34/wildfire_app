from __future__ import annotations

from typing import Dict, List

from backend.models import MitigationAction, PropertyAttributes
from backend.wildfire_data import WildfireContext


def _band_from_score(score: float, low: float = 45.0, high: float = 70.0) -> str:
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def build_mitigation_plan(
    attrs: PropertyAttributes,
    context: WildfireContext,
    submodel_scores: Dict[str, float],
    readiness_blockers: List[str],
) -> List[MitigationAction]:
    recommendations: List[MitigationAction] = []
    ring_metrics = context.structure_ring_metrics or {}

    def ring_density(ring_key: str) -> float | None:
        metrics = ring_metrics.get(ring_key, {})
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

    def add_rec(
        title: str,
        reason: str,
        impacted_submodels: List[str],
        impacted_readiness_factors: List[str],
        risk_band: str,
        readiness_band: str,
        priority: int,
        insurer_relevance: str,
        related_factor: str,
    ) -> None:
        recommendations.append(
            MitigationAction(
                title=title,
                reason=reason,
                impacted_submodels=impacted_submodels,
                impacted_readiness_factors=impacted_readiness_factors,
                estimated_risk_reduction_band=risk_band,
                estimated_readiness_improvement_band=readiness_band,
                priority=priority,
                insurer_relevance=insurer_relevance,
                action=title,
                related_factor=related_factor,
                impact_statement=reason,
                effort=risk_band,
            )
        )

    if "Combustible roof material" in readiness_blockers or submodel_scores.get("structure_vulnerability_risk", 0) >= 70:
        add_rec(
            title="Upgrade roof to Class A fire-rated assembly",
            reason="Combustible roof vulnerability materially increases ember-driven loss severity.",
            impacted_submodels=["structure_vulnerability_risk", "ember_exposure_risk"],
            impacted_readiness_factors=["roof_material", "structure_vulnerability"],
            risk_band="high",
            readiness_band="high",
            priority=1,
            insurer_relevance="required",
            related_factor="structure_vulnerability_risk",
        )

    if (
        "Non-ember-resistant venting" in readiness_blockers
        or submodel_scores.get("ember_exposure_risk", 0) >= 65
        or attrs.vent_type is None
    ):
        add_rec(
            title="Install ember-resistant vents and seal vulnerable openings",
            reason="Reduces ember entry pathways that frequently drive structure ignition.",
            impacted_submodels=["ember_exposure_risk", "structure_vulnerability_risk"],
            impacted_readiness_factors=["vent_quality", "severe_ember_exposure"],
            risk_band=_band_from_score(submodel_scores.get("ember_exposure_risk", 0)),
            readiness_band="high" if "Non-ember-resistant venting" in readiness_blockers else "medium",
            priority=2,
            insurer_relevance="required" if "Non-ember-resistant venting" in readiness_blockers else "recommended",
            related_factor="ember_exposure_risk",
        )

    if (
        "Defensible space below 30 ft" in readiness_blockers
        or "Severely inadequate defensible space" in readiness_blockers
        or submodel_scores.get("defensible_space_risk", 0) >= 60
    ):
        add_rec(
            title="Increase defensible space to at least 30 feet",
            reason="Improves fire separation between structure and contiguous fuels.",
            impacted_submodels=["defensible_space_risk", "flame_contact_risk", "fuel_proximity_risk"],
            impacted_readiness_factors=["defensible_space", "adjacent_fuel_pressure"],
            risk_band="high" if submodel_scores.get("defensible_space_risk", 0) >= 75 else "medium",
            readiness_band="high",
            priority=1,
            insurer_relevance="required",
            related_factor="defensible_space_risk",
        )

    if ring_0_5_density is not None and ring_0_5_density >= 55:
        add_rec(
            title="Create a noncombustible 0-5 ft zone around the structure",
            reason="Dense close-in vegetation in the immediate 0-5 ft ring materially increases direct ignition potential.",
            impacted_submodels=["flame_contact_risk", "defensible_space_risk", "vegetation_intensity_risk"],
            impacted_readiness_factors=["defensible_space", "vegetation_intensity"],
            risk_band="high",
            readiness_band="medium",
            priority=1,
            insurer_relevance="required",
            related_factor="defensible_space_risk",
        )

    if ring_5_30_density is not None and ring_5_30_density >= 60:
        add_rec(
            title="Thin and prune vegetation in the 5-30 ft zone",
            reason="Elevated vegetation density in the 5-30 ft ring can sustain flame spread toward the home.",
            impacted_submodels=["defensible_space_risk", "flame_contact_risk", "vegetation_intensity_risk"],
            impacted_readiness_factors=["defensible_space", "adjacent_fuel_pressure", "vegetation_intensity"],
            risk_band="high" if ring_5_30_density >= 75 else "medium",
            readiness_band="medium",
            priority=2,
            insurer_relevance="recommended",
            related_factor="flame_contact_risk",
        )

    if ring_30_100_density is not None and ring_30_100_density >= 65:
        add_rec(
            title="Reduce fuels in the 30-100 ft zone",
            reason="High vegetation loading in the 30-100 ft ring can intensify incoming fire pressure and ember production.",
            impacted_submodels=["fuel_proximity_risk", "vegetation_intensity_risk", "flame_contact_risk"],
            impacted_readiness_factors=["adjacent_fuel_pressure", "vegetation_intensity"],
            risk_band="medium",
            readiness_band="medium",
            priority=3,
            insurer_relevance="recommended",
            related_factor="fuel_proximity_risk",
        )

    if (
        "High vegetation intensity near structure" in readiness_blockers
        or submodel_scores.get("vegetation_intensity_risk", 0) >= 65
        or ((context.canopy_index if context.canopy_index is not None else 0.0) >= 65)
    ):
        add_rec(
            title="Thin dense vegetation and break canopy continuity near the home",
            reason="Lowers flame intensity and spotting potential in the home ignition zone.",
            impacted_submodels=["vegetation_intensity_risk", "flame_contact_risk", "fuel_proximity_risk"],
            impacted_readiness_factors=["vegetation_intensity", "adjacent_fuel_pressure"],
            risk_band=_band_from_score(submodel_scores.get("vegetation_intensity_risk", 0), low=50, high=75),
            readiness_band="medium",
            priority=3,
            insurer_relevance="recommended",
            related_factor="vegetation_intensity_risk",
        )

    if (
        "Severe environmental hazard conditions" in readiness_blockers
        or "Severe ember exposure" in readiness_blockers
        or submodel_scores.get("historic_fire_risk", 0) >= 70
        or submodel_scores.get("slope_topography_risk", 0) >= 70
    ):
        add_rec(
            title="Adopt annual pre-season hardening inspection and mitigation verification",
            reason="Recurring hazard pressure requires routine hardening validation for insurer confidence.",
            impacted_submodels=["historic_fire_risk", "slope_topography_risk", "ember_exposure_risk"],
            impacted_readiness_factors=["severe_environmental_hazard", "severe_ember_exposure"],
            risk_band="medium",
            readiness_band="medium",
            priority=4,
            insurer_relevance="recommended",
            related_factor="historic_fire_risk",
        )

    if not recommendations:
        add_rec(
            title="Maintain current wildfire hardening and annual vegetation management",
            reason="No severe blockers detected; maintain controls and documentation for insurers.",
            impacted_submodels=["structure_vulnerability_risk", "defensible_space_risk"],
            impacted_readiness_factors=["roof_material", "defensible_space"],
            risk_band="low",
            readiness_band="low",
            priority=5,
            insurer_relevance="nice_to_have",
            related_factor="maintenance",
        )

    recommendations.sort(key=lambda r: r.priority)
    return recommendations[:6]
