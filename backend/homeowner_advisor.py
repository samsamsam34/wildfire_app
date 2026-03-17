from __future__ import annotations

from typing import Any, Dict, Iterable, List

from backend.models import (
    HomeownerConfidenceSummary,
    HomeownerPrioritizedAction,
    HomeownerRiskDriver,
    MitigationAction,
    SubmodelScore,
    WeightedContribution,
)


_FACTOR_PRESENTATION: Dict[str, Dict[str, str]] = {
    "vegetation_intensity_risk": {
        "factor": "vegetation_proximity",
        "title": "Dense vegetation near the home",
        "explanation": "Dense vegetation near the structure can increase ember and flame exposure.",
    },
    "fuel_proximity_risk": {
        "factor": "nearby_fuel_load",
        "title": "Nearby fuels can carry fire toward the home",
        "explanation": "Nearby fuels can sustain fire spread and raise heat and ember pressure at the property.",
    },
    "slope_topography_risk": {
        "factor": "slope_and_terrain",
        "title": "Slope and terrain can amplify fire behavior",
        "explanation": "Steeper or exposed terrain can increase fire spread speed and intensity toward structures.",
    },
    "ember_exposure_risk": {
        "factor": "ember_exposure",
        "title": "High ember exposure potential",
        "explanation": "Wind-blown embers can ignite vulnerable parts of the home even when flames are not nearby.",
    },
    "flame_contact_risk": {
        "factor": "flame_contact_potential",
        "title": "Direct flame-contact risk near the home",
        "explanation": "Vegetation and combustibles close to the structure can increase direct ignition potential.",
    },
    "historic_fire_risk": {
        "factor": "historical_fire_exposure",
        "title": "Nearby fire history indicates recurring pressure",
        "explanation": "Past wildfire activity near the property suggests recurring landscape fire pressure.",
    },
    "structure_vulnerability_risk": {
        "factor": "home_hardening_gaps",
        "title": "Home hardening gaps increase vulnerability",
        "explanation": "Roof, vents, and other structure details can meaningfully change ignition susceptibility.",
    },
    "defensible_space_risk": {
        "factor": "defensible_space",
        "title": "Defensible space is limited",
        "explanation": "Limited defensible space can allow fire and embers to reach the structure more easily.",
    },
}

_FRIENDLY_FIELD_LABELS: Dict[str, str] = {
    "roof_type": "roof material",
    "vent_type": "vent protection",
    "defensible_space_ft": "defensible space distance",
    "construction_year": "construction year",
    "siding_type": "siding material",
    "window_type": "window type",
    "deck_attachment": "deck/fence attachment detail",
}


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _impact_from_share(share_pct: float) -> str:
    if share_pct >= 35.0:
        return "high"
    if share_pct >= 18.0:
        return "medium"
    return "low"


def _impact_rank(level: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(level).lower(), 2)


def _effort_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(level).lower(), 1)


def _friendly_field(field_name: str) -> str:
    return _FRIENDLY_FIELD_LABELS.get(field_name, field_name.replace("_", " "))


def build_ranked_risk_drivers(
    *,
    submodel_scores: Dict[str, SubmodelScore],
    weighted_contributions: Dict[str, WeightedContribution],
    limit: int = 3,
) -> tuple[List[str], List[HomeownerRiskDriver]]:
    rows: List[Dict[str, Any]] = []
    for submodel_key, presentation in _FACTOR_PRESENTATION.items():
        score_obj = submodel_scores.get(submodel_key)
        contribution_obj = weighted_contributions.get(submodel_key)
        evidence_status = str(
            getattr(contribution_obj, "factor_evidence_status", None)
            or getattr(contribution_obj, "basis", "")
            or ""
        ).strip().lower()
        omitted_due_to_missing = bool(getattr(contribution_obj, "omitted_due_to_missing", False))
        support_level = str(getattr(contribution_obj, "support_level", "") or "").strip().lower()
        if evidence_status == "suppressed" or omitted_due_to_missing:
            continue
        contribution = abs(_to_float(getattr(contribution_obj, "contribution", None)) or 0.0)
        if contribution <= 0.0 and score_obj is not None:
            contribution = max(0.0, float(score_obj.score) / 100.0)
        if score_obj is None and contribution <= 0.0:
            continue
        low_evidence = evidence_status in {"fallback", "missing"} or support_level == "low"
        explanation = str(presentation["explanation"])
        if low_evidence:
            explanation = f"{explanation} (Lower-evidence factor; treated as directional guidance.)"
        rows.append(
            {
                "submodel_key": submodel_key,
                "factor": presentation["factor"],
                "title": presentation["title"],
                "explanation": explanation,
                "score": float(score_obj.score) if score_obj is not None else 0.0,
                "contribution": contribution,
                "low_evidence": low_evidence,
            }
        )

    if not rows:
        return [], []

    rows.sort(
        key=lambda row: (
            bool(row["low_evidence"]),
            -float(row["contribution"]),
            -float(row["score"]),
        )
    )
    total_contribution = sum(float(row["contribution"]) for row in rows)
    if total_contribution <= 0.0:
        total_contribution = float(len(rows))

    detailed: List[HomeownerRiskDriver] = []
    plain_titles: List[str] = []
    for row in rows[: max(1, limit)]:
        share_pct = (float(row["contribution"]) / total_contribution) * 100.0
        detailed.append(
            HomeownerRiskDriver(
                factor=str(row["factor"]),
                impact=_impact_from_share(share_pct),  # type: ignore[arg-type]
                explanation=str(row["explanation"]),
                relative_contribution_pct=round(share_pct, 1),
            )
        )
        plain_titles.append(str(row["title"]))

    return plain_titles[:limit], detailed[:limit]


def _estimate_effort_level(action: MitigationAction) -> str:
    if action.effort in {"low", "medium", "high"}:
        return str(action.effort)
    impact = str(action.estimated_risk_reduction_band or "low")
    if impact == "high" and int(action.priority or 5) <= 2:
        return "medium"
    if impact == "high":
        return "high"
    if impact == "low":
        return "low"
    return "medium"


def _estimate_timeline(impact_level: str, effort_level: str) -> str:
    if impact_level == "high" and effort_level == "low":
        return "now"
    if impact_level in {"high", "medium"} and effort_level in {"low", "medium"}:
        return "this_season"
    return "later"


def prioritize_mitigation_actions(
    mitigation_plan: Iterable[MitigationAction],
    *,
    limit: int = 5,
) -> List[HomeownerPrioritizedAction]:
    rows: List[HomeownerPrioritizedAction] = []
    for item in mitigation_plan:
        if not isinstance(item, MitigationAction):
            continue
        action = str(item.title or item.action or "").strip()
        if not action:
            continue
        explanation = str(item.reason or item.impact_statement or "").strip()
        impact_level = str(item.estimated_risk_reduction_band or "low")
        if impact_level not in {"high", "medium", "low"}:
            impact_level = "low"
        effort_level = _estimate_effort_level(item)
        cost_band = {"low": "low", "medium": "medium", "high": "high"}.get(effort_level, "medium")
        rows.append(
            HomeownerPrioritizedAction(
                action=action,
                explanation=explanation,
                impact_level=impact_level,  # type: ignore[arg-type]
                effort_level=effort_level,  # type: ignore[arg-type]
                estimated_cost_band=cost_band,  # type: ignore[arg-type]
                timeline=_estimate_timeline(impact_level, effort_level),  # type: ignore[arg-type]
                priority=int(item.priority or 5),
            )
        )

    rows.sort(
        key=lambda row: (
            _impact_rank(row.impact_level),
            _effort_rank(row.effort_level),
            int(row.priority or 5),
            row.action.lower(),
        )
    )
    return rows[: max(1, limit)]


def build_confidence_summary(
    *,
    confidence_tier: str,
    observed_inputs: Dict[str, object],
    inferred_inputs: Dict[str, object],
    missing_inputs: List[str],
    assumptions_used: List[str],
) -> HomeownerConfidenceSummary:
    observed = [_friendly_field(field) for field, value in observed_inputs.items() if value is not None]
    inferred = [_friendly_field(field) for field, value in inferred_inputs.items() if value is not None]
    missing = [_friendly_field(field) for field in missing_inputs]

    fallback_assumptions = [
        note
        for note in assumptions_used
        if any(token in str(note).lower() for token in ("fallback", "missing", "unavailable", "estimated", "inferred"))
    ][:6]

    accuracy_improvements: List[str] = []
    improvement_templates = {
        "roof_type": "Providing roof material information could improve structural vulnerability estimates.",
        "vent_type": "Providing vent protection details could improve ember exposure estimates.",
        "defensible_space_ft": "Providing defensible space distance could improve near-home ignition estimates.",
        "construction_year": "Providing construction year could improve structure hardening estimates.",
        "siding_type": "Providing siding material could improve structure susceptibility estimates.",
        "window_type": "Providing window type could improve ember intrusion estimates.",
    }
    for field in missing_inputs:
        if field in improvement_templates:
            accuracy_improvements.append(improvement_templates[field])
    if not accuracy_improvements and missing:
        accuracy_improvements.append("Providing additional home details could improve estimate precision.")

    normalized_confidence = str(confidence_tier or "preliminary").strip().lower()
    if normalized_confidence not in {"high", "moderate", "low", "preliminary"}:
        normalized_confidence = "preliminary"

    return HomeownerConfidenceSummary(
        confidence=normalized_confidence,  # type: ignore[arg-type]
        observed_data=observed[:12],
        estimated_data=inferred[:12],
        missing_data=missing[:12],
        fallback_assumptions=fallback_assumptions,
        accuracy_improvements=accuracy_improvements[:6],
    )


def build_simulator_explanations(
    *,
    baseline: Dict[str, Any],
    simulated: Dict[str, Any],
) -> Dict[str, Any]:
    current_score = _to_float(baseline.get("wildfire_risk_score"))
    simulated_score = _to_float(simulated.get("wildfire_risk_score"))
    risk_reduction = None
    if current_score is not None and simulated_score is not None:
        risk_reduction = round(current_score - simulated_score, 1)

    baseline_drivers = [
        row.get("factor")
        for row in (baseline.get("top_risk_drivers_detailed") or [])
        if isinstance(row, dict) and row.get("factor")
    ]
    simulated_drivers = [
        row.get("factor")
        for row in (simulated.get("top_risk_drivers_detailed") or [])
        if isinstance(row, dict) and row.get("factor")
    ]
    removed_drivers = [factor for factor in baseline_drivers if factor not in set(simulated_drivers)]

    baseline_actions = [
        row.get("action")
        for row in (baseline.get("prioritized_mitigation_actions") or [])
        if isinstance(row, dict) and row.get("action")
    ]
    simulated_actions = [
        row.get("action")
        for row in (simulated.get("prioritized_mitigation_actions") or [])
        if isinstance(row, dict) and row.get("action")
    ]
    resolved_actions = [action for action in baseline_actions if action not in set(simulated_actions)]

    if risk_reduction is None:
        plain = "Simulation completed, but risk reduction could not be quantified from available scores."
    elif risk_reduction > 0:
        plain = f"The simulated changes reduce overall wildfire risk by about {risk_reduction:.1f} points."
    elif risk_reduction < 0:
        plain = f"The simulated changes increase overall wildfire risk by about {abs(risk_reduction):.1f} points."
    else:
        plain = "The simulated changes keep overall wildfire risk about the same."

    return {
        "current_risk_score": current_score,
        "simulated_risk_score": simulated_score,
        "estimated_risk_reduction": risk_reduction,
        "updated_top_risk_drivers": simulated.get("top_risk_drivers_detailed") or [],
        "risk_drivers_removed": removed_drivers,
        "mitigation_actions_resolved": resolved_actions,
        "plain_language_change_summary": plain,
    }
