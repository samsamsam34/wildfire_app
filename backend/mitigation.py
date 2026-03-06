from __future__ import annotations

from typing import List

from backend.models import MitigationAction, PropertyAttributes


def build_mitigation_plan(attrs: PropertyAttributes, risk_score: float) -> List[MitigationAction]:
    actions: List[MitigationAction] = []

    if attrs.defensible_space_ft is None or attrs.defensible_space_ft < 30:
        actions.append(
            MitigationAction(
                action="Increase defensible space to at least 30 feet",
                estimated_risk_reduction=12.0,
                effort="medium",
                insurer_relevance="required",
                reason="Near-structure fuels materially increase ember ignition and flame contact risk.",
            )
        )

    if attrs.vent_type is None or "ember" not in (attrs.vent_type or "").lower():
        actions.append(
            MitigationAction(
                action="Install ember-resistant attic/crawlspace vents",
                estimated_risk_reduction=8.0,
                effort="medium",
                insurer_relevance="recommended",
                reason="Embers commonly enter homes through non-screened or legacy vent openings.",
            )
        )

    if attrs.roof_type is None or attrs.roof_type.lower() in {"wood", "untreated wood shake"}:
        actions.append(
            MitigationAction(
                action="Upgrade to a Class A fire-rated roof",
                estimated_risk_reduction=15.0,
                effort="high",
                insurer_relevance="required",
                reason="Combustible roof coverings are high-severity loss multipliers during ember storms.",
            )
        )

    if risk_score >= 65:
        actions.append(
            MitigationAction(
                action="Create a yearly vegetation maintenance schedule",
                estimated_risk_reduction=6.0,
                effort="low",
                insurer_relevance="recommended",
                reason="Annual upkeep preserves mitigation gains and aligns with carrier inspection criteria.",
            )
        )

    actions.sort(key=lambda a: (a.estimated_risk_reduction, a.insurer_relevance == "required"), reverse=True)
    return actions[:5]
