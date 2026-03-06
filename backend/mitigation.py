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

    def add_rec(
        title: str,
        reason: str,
        impacted: List[str],
        risk_band: str,
        readiness_band: str,
        priority: int,
        related_factor: str,
    ) -> None:
        recommendations.append(
            MitigationAction(
                title=title,
                reason=reason,
                impacted_submodels=impacted,
                estimated_risk_reduction_band=risk_band,
                estimated_readiness_improvement_band=readiness_band,
                priority=priority,
                action=title,
                related_factor=related_factor,
                impact_statement=reason,
                effort=risk_band,
                insurer_relevance="required" if readiness_band == "high" else "recommended",
            )
        )

    if "Combustible roof material" in readiness_blockers or submodel_scores.get("home_hardening_risk", 0) >= 70:
        add_rec(
            title="Upgrade roof to Class A fire-rated assembly",
            reason="Combustible roof vulnerability materially increases ember-driven loss severity.",
            impacted=["home_hardening_risk", "ember_exposure"],
            risk_band="high",
            readiness_band="high",
            priority=1,
            related_factor="home_hardening_risk",
        )

    if (
        "Non-ember-resistant venting" in readiness_blockers
        or submodel_scores.get("ember_exposure", 0) >= 65
        or attrs.vent_type is None
    ):
        add_rec(
            title="Install ember-resistant vents and seal vulnerable openings",
            reason="Reduces ember entry pathways that frequently drive structure ignition.",
            impacted=["ember_exposure", "home_hardening_risk"],
            risk_band=_band_from_score(submodel_scores.get("ember_exposure", 0)),
            readiness_band="high" if "Non-ember-resistant venting" in readiness_blockers else "medium",
            priority=2,
            related_factor="ember_exposure",
        )

    if (
        "Defensible space below 30 ft" in readiness_blockers
        or "Severely inadequate defensible space" in readiness_blockers
        or submodel_scores.get("defensible_space_risk", 0) >= 60
    ):
        add_rec(
            title="Increase defensible space to at least 30 feet",
            reason="Improves fire separation between structure and contiguous fuels.",
            impacted=["defensible_space_risk", "flame_contact_exposure", "fuel_proximity_risk"],
            risk_band="high" if submodel_scores.get("defensible_space_risk", 0) >= 75 else "medium",
            readiness_band="high",
            priority=1,
            related_factor="defensible_space_risk",
        )

    if (
        "High vegetation intensity near structure" in readiness_blockers
        or submodel_scores.get("vegetation_intensity_risk", 0) >= 65
        or context.canopy_index >= 65
    ):
        add_rec(
            title="Thin dense vegetation and break canopy continuity near the home",
            reason="Lowers flame intensity and spotting potential in the home ignition zone.",
            impacted=["vegetation_intensity_risk", "flame_contact_exposure", "fuel_proximity_risk"],
            risk_band=_band_from_score(submodel_scores.get("vegetation_intensity_risk", 0), low=50, high=75),
            readiness_band="medium",
            priority=3,
            related_factor="vegetation_intensity_risk",
        )

    if (
        "Severe environmental hazard conditions" in readiness_blockers
        or submodel_scores.get("historic_fire_risk", 0) >= 70
        or submodel_scores.get("topography_risk", 0) >= 70
    ):
        add_rec(
            title="Adopt annual pre-season hardening inspection and mitigation verification",
            reason="Recurring hazard pressure requires routine hardening validation for insurer confidence.",
            impacted=["historic_fire_risk", "topography_risk", "ember_exposure"],
            risk_band="medium",
            readiness_band="medium",
            priority=4,
            related_factor="historic_fire_risk",
        )

    if not recommendations:
        add_rec(
            title="Maintain current wildfire hardening and annual vegetation management",
            reason="No severe blockers detected; maintain controls and documentation for insurers.",
            impacted=["home_hardening_risk", "defensible_space_risk"],
            risk_band="low",
            readiness_band="low",
            priority=5,
            related_factor="maintenance",
        )

    recommendations.sort(key=lambda r: r.priority)
    return recommendations[:6]
