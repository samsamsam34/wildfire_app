from __future__ import annotations

from typing import List

from backend.models import MitigationAction, PropertyAttributes
from backend.wildfire_data import WildfireContext


def build_mitigation_plan(attrs: PropertyAttributes, risk_score: float, context: WildfireContext) -> List[MitigationAction]:
    actions: List[MitigationAction] = []

    if attrs.defensible_space_ft is None or attrs.defensible_space_ft < 30 or context.wildland_distance_index >= 65:
        impact = 8.0 + max(0.0, (context.wildland_distance_index - 60.0) * 0.12)
        actions.append(
            MitigationAction(
                action="Increase defensible space to at least 30 feet",
                related_factor="fuel_proximity",
                impact_statement="Reduces parcel-adjacent fuel proximity and short-range ember/flame exposure.",
                estimated_risk_reduction=round(min(16.0, impact), 1),
                effort="medium",
                insurer_relevance="required",
                reason="Near-structure fuels materially increase ignition probability.",
            )
        )

    if context.canopy_index >= 65:
        impact = 5.0 + (context.canopy_index - 60.0) * 0.08
        actions.append(
            MitigationAction(
                action="Thin/limb canopy in the immediate home ignition zone",
                related_factor="canopy_density",
                impact_statement="Lowers canopy continuity and ember retention potential around structures.",
                estimated_risk_reduction=round(min(11.0, impact), 1),
                effort="medium",
                insurer_relevance="recommended",
                reason="Dense canopy increases flame spread potential and radiant heat intensity.",
            )
        )

    if attrs.vent_type is None or "ember" not in (attrs.vent_type or "").lower():
        actions.append(
            MitigationAction(
                action="Install ember-resistant attic/crawlspace vents",
                related_factor="historical_fire_recurrence",
                impact_statement="Reduces ember intrusion risk during recurring fire-weather events.",
                estimated_risk_reduction=8.0,
                effort="medium",
                insurer_relevance="recommended",
                reason="Legacy vents are a frequent ember entry pathway in wildfire losses.",
            )
        )

    if attrs.roof_type is None or attrs.roof_type.lower() in {"wood", "untreated wood shake"}:
        actions.append(
            MitigationAction(
                action="Upgrade to a Class A fire-rated roof",
                related_factor="burn_probability",
                impact_statement="Improves roof survivability under high burn-probability ember exposure.",
                estimated_risk_reduction=15.0,
                effort="high",
                insurer_relevance="required",
                reason="Combustible roof coverings are high-severity loss multipliers during ember storms.",
            )
        )

    if context.moisture_index >= 65:
        actions.append(
            MitigationAction(
                action="Increase seasonal vegetation moisture management (irrigation/mulch zoning)",
                related_factor="drought_moisture",
                impact_statement="Reduces fine-fuel dryness and ignition susceptibility around the parcel.",
                estimated_risk_reduction=6.0,
                effort="low",
                insurer_relevance="nice_to_have",
                reason="Dry fuel conditions elevate ember ignition likelihood.",
            )
        )

    if context.historic_fire_index >= 70 or risk_score >= 70:
        actions.append(
            MitigationAction(
                action="Prepare annual perimeter hardening and pre-season risk inspection",
                related_factor="historical_fire_recurrence",
                impact_statement="Targets repeated local fire pressure with recurring property hardening actions.",
                estimated_risk_reduction=7.0,
                effort="low",
                insurer_relevance="recommended",
                reason="Repeated nearby fire activity increases expected annual exposure.",
            )
        )

    actions.sort(key=lambda a: (a.estimated_risk_reduction, a.insurer_relevance == "required"), reverse=True)
    return actions[:6]
