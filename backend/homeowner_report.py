from __future__ import annotations

from datetime import timezone
import re
import textwrap
from typing import Any, Literal

from backend.models import (
    AssessmentResult,
    HomeownerReport,
    HomeownerReportAction,
    HomeownerPrioritizedAction,
    MitigationAction,
    NearStructureAction,
)


def _dump_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return value
    return value


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_band(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score >= 70:
        return "high"
    if score >= 40:
        return "moderate"
    return "lower"


def _home_hardening_band(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    return "limited"


def _plain_driver(text: str) -> str:
    mapping = {
        "dense vegetation close to the home": "Dense vegetation is very close to the structure.",
        "limited defensible space within 30 feet": "Defensible space appears limited within 30 feet of the home.",
        "elevated vegetation and fuels within 100 feet": "Vegetation and fuels are elevated within 100 feet of the structure.",
        "high ember exposure": "Ember exposure appears elevated for this location.",
        "high flame-contact exposure": "Nearby fuels may increase direct flame-contact risk.",
        "close proximity to wildland fuels": "Wildland fuels are close enough to increase fire pressure.",
        "slope/topography amplification": "Terrain may increase fire spread pressure toward the home.",
        "dense and dry vegetation": "Vegetation appears dense and dry near the property.",
        "insufficient defensible space": "Defensible space appears insufficient.",
        "high structure vulnerability": "Home hardening details indicate elevated structure vulnerability.",
    }
    normalized = str(text or "").strip().lower()
    if not normalized:
        return ""
    return mapping.get(normalized, str(text or "").strip())


def _de_jargonize(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    replacements = {
        "fuel model": "vegetation and dry brush conditions",
        "canopy": "tree cover",
        "ember exposure": "wind-blown ember exposure",
        "flame-contact": "direct flame contact",
    }
    lowered = normalized.lower()
    for src, dst in replacements.items():
        if src in lowered:
            lowered = lowered.replace(src, dst)
    if not lowered:
        return ""
    return lowered[0].upper() + lowered[1:]


def _select_tone_level(result: AssessmentResult) -> str:
    tier = str(result.confidence_tier or "").lower()
    fallback_share = _to_float(getattr(result, "fallback_weight_fraction", None))
    if fallback_share is None:
        fallback_share = _to_float(getattr(result, "fallback_dominance_ratio", None))
    fallback_share = float(fallback_share or 0.0)

    missing_count = len(list(result.confidence_summary.missing_data or []))
    fallback_assumption_count = len(list(result.confidence_summary.fallback_assumptions or []))
    fallback_decision_count = len(list((result.assessment_diagnostics.fallback_decisions or [])))
    fallback_count = fallback_assumption_count + fallback_decision_count

    if tier in {"low", "preliminary"}:
        return "advisory"
    if fallback_share >= 0.50 or missing_count >= 4 or fallback_count >= 5:
        return "advisory"
    if tier == "high" and fallback_share <= 0.20 and missing_count == 0 and fallback_count == 0:
        return "direct"
    if tier in {"moderate", "medium"} and (fallback_share >= 0.35 or missing_count >= 2 or fallback_count >= 3):
        return "advisory"
    return "slightly_hedged"


def _action_tone_level(base_tone: str, evidence_status: str | None) -> str:
    evidence = str(evidence_status or "").lower()
    if evidence not in {"inferred", "missing", "unknown"}:
        return base_tone
    if base_tone == "direct":
        return "slightly_hedged"
    return "advisory"


def _headline_risk_summary(result: AssessmentResult, risk_score: float | None, *, tone_level: str) -> str:
    risk_band = _risk_band(risk_score)
    risk_label = "unknown" if risk_band == "unavailable" else ("low" if risk_band == "lower" else risk_band)
    if risk_label == "unknown":
        return "We could not determine a reliable wildfire risk level from current data."
    if tone_level == "advisory":
        return (
            f"Your home may have {risk_label} wildfire risk, but this estimate is uncertain because "
            "some property details were estimated or missing."
        )
    if tone_level == "slightly_hedged":
        return f"Your home appears to have {risk_label} wildfire risk based on available property and regional evidence."
    return f"Your home has {risk_label} wildfire risk."


def _risk_driver_from_text(driver_text: str, *, tone_level: str) -> str:
    text = str(driver_text or "").strip().lower()
    if not text:
        base = "This condition near your home"
    elif any(token in text for token in ("vegetation", "fuel", "tree cover", "brush", "wildland")):
        base = "Vegetation near your home"
    elif any(token in text for token in ("slope", "terrain", "topography")):
        base = "Terrain near your home"
    elif "defensible space" in text:
        base = "Defensible space near your home"
    elif "ember" in text:
        base = "Wind-blown ember exposure"
    elif any(token in text for token in ("hardening", "vulnerability", "structure")):
        base = "Home hardening details"
    else:
        base = "This condition near your home"

    if tone_level == "direct":
        return f"{base} is a major risk factor."
    if tone_level == "slightly_hedged":
        return f"{base} appears to increase risk."
    return f"{base} may be contributing to risk, but some details are estimated."


def _summarize_top_risk_drivers(key_risk_drivers: list[str], *, tone_level: str) -> list[str]:
    cleaned = [_de_jargonize(_plain_driver(row)) for row in key_risk_drivers if str(row).strip()]
    cleaned = [row for row in cleaned if row]
    unique = list(dict.fromkeys([_risk_driver_from_text(row, tone_level=tone_level) for row in cleaned if row]))
    if unique:
        return unique[:4]
    return ["We could not confirm property-specific risk drivers from current data."]


def _estimated_benefit_phrase(impact_level: str, tone_level: str) -> str:
    impact = str(impact_level or "low").lower()
    if tone_level == "advisory":
        if impact == "high":
            phrase = "May help reduce wildfire risk meaningfully when completed and maintained."
        elif impact == "medium":
            phrase = "May help reduce wildfire risk over time."
        else:
            phrase = "May provide incremental wildfire protection."
        return phrase + " Benefit estimate is directional because some inputs were estimated."
    if tone_level == "slightly_hedged":
        if impact == "high":
            return "Appears likely to reduce wildfire risk meaningfully when completed and maintained."
        if impact == "medium":
            return "Appears likely to reduce wildfire risk over time."
        return "May provide incremental wildfire protection."
    if impact == "high":
        return "Likely to reduce wildfire risk meaningfully when completed and maintained."
    if impact == "medium":
        return "Likely to reduce wildfire risk over time."
    return "May provide incremental wildfire protection."


def _expected_effect_from_impact(impact_level: str | None) -> str:
    impact = str(impact_level or "low").lower()
    if impact == "high":
        return "significant"
    if impact == "medium":
        return "moderate"
    return "small"


def _infer_what_it_reduces(
    *,
    action_text: str,
    context_text: str = "",
    target_zone: str | None = None,
    impacted_submodels: list[str] | None = None,
) -> str:
    haystack = " ".join(
        [
            str(action_text or "").lower(),
            str(context_text or "").lower(),
            str(target_zone or "").lower(),
            " ".join(str(item or "").lower() for item in list(impacted_submodels or [])),
        ]
    )
    if any(token in haystack for token in ("0-5", "ember", "debris", "gutter", "vent", "roofline", "roof")):
        return "ember ignition and direct flame exposure right next to your home"
    if any(token in haystack for token in ("5-30", "30 feet", "defensible", "thinning", "prune", "pruning")):
        return "radiant heat and flame spread pressure around the home"
    if any(token in haystack for token in ("fuel", "brush", "vegetation", "canopy", "tree")):
        return "the chance that nearby vegetation carries fire toward your home"
    return "the chance that wildfire reaches and ignites your home"


def _build_why_this_matters_sentence(*, action_text: str, what_it_reduces: str, tone_level: str) -> str:
    action = str(action_text or "").strip().rstrip(".")
    if not action:
        action = "This action"
    if tone_level == "advisory":
        return f"{action} may help reduce {what_it_reduces}, but some details are estimated."
    if tone_level == "slightly_hedged":
        return f"{action} appears to help reduce {what_it_reduces}."
    return f"{action} helps reduce {what_it_reduces}."


def _enrich_prioritized_action(
    action: HomeownerPrioritizedAction,
    *,
    tone_level: str,
) -> HomeownerPrioritizedAction:
    what_it_reduces = str(action.what_it_reduces or "").strip() or _infer_what_it_reduces(
        action_text=str(action.action or ""),
        context_text=str(action.explanation or ""),
    )
    action_tone = _action_tone_level(base_tone=tone_level, evidence_status=None)
    why_this_matters = str(action.why_this_matters or "").strip() or _build_why_this_matters_sentence(
        action_text=str(action.action or "This action"),
        what_it_reduces=what_it_reduces,
        tone_level=action_tone,
    )
    expected_effect = str(action.expected_effect or "").strip().lower()
    if expected_effect not in {"small", "moderate", "significant"}:
        expected_effect = _expected_effect_from_impact(str(action.impact_level or "low"))
    data_confidence = str(action.data_confidence or "unknown").strip().lower()
    if data_confidence not in {"high", "medium", "low", "unknown"}:
        data_confidence = "unknown"
    if data_confidence == "unknown":
        if tone_level == "direct":
            data_confidence = "high"
        elif tone_level == "slightly_hedged":
            data_confidence = "medium"
        else:
            data_confidence = "low"
    return action.model_copy(
        update={
            "why_this_matters": why_this_matters,
            "what_it_reduces": what_it_reduces,
            "expected_effect": expected_effect,
            "data_confidence": data_confidence,
        }
    )


def _parse_proximity_score(action_text: str, what_it_reduces: str) -> float:
    text = f"{action_text} {what_it_reduces}".lower()
    if any(token in text for token in ("0-5", "0 to 5", "within 5", "5 feet", "5 ft")):
        return 1.0
    if any(token in text for token in ("5-30", "5 to 30", "within 30", "30 feet", "30 ft")):
        return 0.82
    if any(token in text for token in ("30-100", "30 to 100", "within 100", "100 feet", "100 ft")):
        return 0.55

    distance_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:ft|feet)", text)
    if distance_matches:
        try:
            nearest = min(float(match) for match in distance_matches)
        except Exception:
            nearest = 9999.0
        if nearest <= 10:
            return 0.95
        if nearest <= 30:
            return 0.80
        if nearest <= 60:
            return 0.50
        if nearest <= 100:
            return 0.40
        return 0.30
    return 0.60


def _rank_risk_contribution(impact_level: str) -> float:
    value = str(impact_level or "low").lower()
    if value == "high":
        return 1.0
    if value == "medium":
        return 0.72
    return 0.45


def _rank_feasibility(effort_level: str) -> float:
    value = str(effort_level or "medium").lower()
    if value == "low":
        return 1.0
    if value == "medium":
        return 0.75
    if value == "high":
        return 0.50
    return 0.65


def _rank_data_confidence(data_confidence: str) -> float:
    value = str(data_confidence or "unknown").lower()
    if value == "high":
        return 1.0
    if value == "medium":
        return 0.70
    if value == "low":
        return 0.35
    return 0.55


def _rank_mitigation_actions(prioritized_actions: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranked: list[dict[str, object]] = []
    for index, row in enumerate(prioritized_actions):
        action = str(row.get("action") or "").strip()
        if not action:
            continue
        impact_level = str(row.get("impact_level") or "low")
        effort_level = str(row.get("effort_level") or "medium")
        data_confidence = str(row.get("data_confidence") or "unknown")
        what_it_reduces = str(row.get("what_it_reduces") or "")

        risk_contribution_score = _rank_risk_contribution(impact_level)
        proximity_score = _parse_proximity_score(action, what_it_reduces)
        feasibility_score = _rank_feasibility(effort_level)
        data_confidence_score = _rank_data_confidence(data_confidence)

        # Explainable weighted blend emphasizing risk/proximity while ensuring
        # low-confidence recommendations are not ranked above high-confidence peers.
        prioritization_score = (
            (0.30 * risk_contribution_score)
            + (0.25 * proximity_score)
            + (0.15 * feasibility_score)
            + (0.30 * data_confidence_score)
        )

        row_with_rank = dict(row)
        row_with_rank.update(
            {
                "risk_contribution_score": round(risk_contribution_score, 4),
                "proximity_score": round(proximity_score, 4),
                "feasibility_score": round(feasibility_score, 4),
                "data_confidence_score": round(data_confidence_score, 4),
                "prioritization_score": round(prioritization_score, 4),
                "ranking_basis": {
                    "risk_contribution_weight": 0.30,
                    "proximity_weight": 0.25,
                    "feasibility_weight": 0.15,
                    "data_confidence_weight": 0.30,
                },
                "rank_order_source_index": index,
            }
        )
        ranked.append(row_with_rank)

    ranked.sort(
        key=lambda row: (
            -float(row.get("prioritization_score") or 0.0),
            -float(row.get("proximity_score") or 0.0),
            -float(row.get("data_confidence_score") or 0.0),
            str(row.get("action") or "").lower(),
        )
    )
    ranked = ranked[:5]
    most_impactful: list[dict[str, object]] = []
    for index, row in enumerate(ranked):
        row["most_impactful"] = index < 2
        if index < 2:
            most_impactful.append(row)
    return ranked, most_impactful


def _summarize_prioritized_actions(
    prioritized_actions: list[HomeownerPrioritizedAction],
    *,
    tone_level: str,
) -> list[dict[str, object]]:
    impact_rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(
        list(prioritized_actions or []),
        key=lambda row: (impact_rank.get(str(row.impact_level), 3), int(row.priority or 99), str(row.action or "").lower()),
    )
    rows: list[dict[str, object]] = []
    for row in ordered[:5]:
        action = str(row.action or "").strip()
        if not action:
            continue
        effort = str(row.effort_level or "medium")
        if effort not in {"low", "medium", "high"}:
            effort = "medium"
        what_it_reduces = str(row.what_it_reduces or "").strip() or _infer_what_it_reduces(
            action_text=action,
            context_text=str(row.explanation or ""),
        )
        action_tone = _action_tone_level(base_tone=tone_level, evidence_status=None)
        why_this_matters = str(row.why_this_matters or "").strip() or _build_why_this_matters_sentence(
            action_text=action,
            what_it_reduces=what_it_reduces,
            tone_level=action_tone,
        )
        expected_effect = str(row.expected_effect or "").strip().lower()
        if expected_effect not in {"small", "moderate", "significant"}:
            expected_effect = _expected_effect_from_impact(str(row.impact_level or "low"))
        rows.append(
            {
                "action": action,
                "effort_level": effort,
                "impact_level": str(row.impact_level or "low"),
                "data_confidence": str(row.data_confidence or "unknown"),
                "expected_effect": expected_effect,
                "what_it_reduces": what_it_reduces,
                "why_this_matters": why_this_matters,
                "estimated_benefit": _estimated_benefit_phrase(str(row.impact_level or "low"), action_tone),
                "why_it_matters": str(row.explanation or "").strip(),
            }
        )
    return rows


def _limitations_notice(result: AssessmentResult, combined_limitations: list[str]) -> str:
    if combined_limitations:
        prefix = "Some results are estimated because: "
        return prefix + "; ".join(combined_limitations[:2])
    if result.confidence_tier in {"low", "preliminary"}:
        return "Some key property details were estimated or missing, so use this report as planning guidance."
    if result.confidence_tier == "moderate":
        return "Most major inputs are available, but some details were estimated."
    return "Most high-impact inputs were observed directly for this property."


def _confidence_summary(result: AssessmentResult) -> str:
    tier = str(result.confidence_tier or "preliminary")
    restriction = str(result.use_restriction or "review_required")
    if tier == "high":
        return "High confidence: most high-impact inputs were observed directly from prepared data and confirmed facts."
    if tier == "moderate":
        return "Moderate confidence: major location context is available, with some inferred inputs and assumptions."
    if tier == "low":
        return "Low confidence: significant assumptions or fallback data were required for one or more score components."
    return (
        "Preliminary confidence: this result is useful for homeowner planning, but additional data is needed before high-stakes decisions."
    ) + f" Use restriction: {restriction}."


def _zone_findings(defensible_space_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    zones = defensible_space_analysis.get("zones") if isinstance(defensible_space_analysis, dict) else None
    if not isinstance(zones, dict):
        return []

    zone_order = ["zone_0_5_ft", "zone_5_30_ft", "zone_30_100_ft", "zone_100_300_ft"]
    findings: list[dict[str, Any]] = []
    for key in zone_order:
        zone = zones.get(key)
        if not isinstance(zone, dict):
            continue
        findings.append(
            {
                "zone_key": key,
                "distance_band_ft": zone.get("distance_band_ft") or key.replace("zone_", "").replace("_", " "),
                "risk_level": zone.get("risk_level") or "unknown",
                "hazardous_vegetation_pct": _to_float(zone.get("hazardous_vegetation_pct")),
                "vegetation_density": _to_float(zone.get("vegetation_density")),
                "evidence_status": zone.get("evidence_status") or "unknown",
                "zone_status": zone.get("zone_status") or "unavailable",
            }
        )
    return findings


def _mitigation_actions(result: AssessmentResult, *, tone_level: str) -> list[HomeownerReportAction]:
    actions: list[HomeownerReportAction] = []
    seen_titles: set[str] = set()

    for action in result.prioritized_vegetation_actions:
        if not isinstance(action, NearStructureAction):
            continue
        title = str(action.title or "").strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        action_tone = _action_tone_level(base_tone=tone_level, evidence_status=action.evidence_status)
        what_it_reduces = _infer_what_it_reduces(
            action_text=title,
            context_text=str(action.why_it_matters or action.explanation or ""),
            target_zone=str(action.target_zone or ""),
        )
        why_this_matters = _build_why_this_matters_sentence(
            action_text=title,
            what_it_reduces=what_it_reduces,
            tone_level=action_tone,
        )
        actions.append(
            HomeownerReportAction(
                title=title,
                priority=int(action.priority or 5),
                target_zone=(str(action.target_zone).strip() or None),
                why_it_matters=str(action.why_it_matters or "").strip() or why_this_matters,
                why_this_matters=why_this_matters,
                what_it_reduces=what_it_reduces,
                expected_effect=_expected_effect_from_impact(str(action.impact_category or "low")),  # type: ignore[arg-type]
                expected_impact_category=action.impact_category,
                evidence_status=action.evidence_status,
                explanation=str(action.explanation or "").strip(),
            )
        )

    for mitigation in sorted(result.mitigation_plan, key=lambda row: int(row.priority or 99)):
        if not isinstance(mitigation, MitigationAction):
            continue
        title = str(mitigation.title or "").strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        action_tone = _action_tone_level(base_tone=tone_level, evidence_status="observed")
        what_it_reduces = _infer_what_it_reduces(
            action_text=title,
            context_text=str(mitigation.reason or ""),
            impacted_submodels=list(mitigation.impacted_submodels or []),
        )
        why_this_matters = _build_why_this_matters_sentence(
            action_text=title,
            what_it_reduces=what_it_reduces,
            tone_level=action_tone,
        )
        actions.append(
            HomeownerReportAction(
                title=title,
                priority=int(mitigation.priority or 5),
                target_zone=None,
                why_it_matters=str(mitigation.reason or "").strip() or why_this_matters,
                why_this_matters=why_this_matters,
                what_it_reduces=what_it_reduces,
                expected_effect=_expected_effect_from_impact(str(mitigation.estimated_risk_reduction_band or "low")),
                expected_impact_category=mitigation.estimated_risk_reduction_band,
                evidence_status="observed",
                explanation=str(mitigation.reason or "").strip(),
            )
        )

    actions.sort(key=lambda row: (int(row.priority or 99), row.title.lower()))
    return actions[:8]


def _prioritized_actions(result: AssessmentResult) -> list[HomeownerPrioritizedAction]:
    tone_level = _select_tone_level(result)
    base: list[HomeownerPrioritizedAction] = []
    if result.prioritized_mitigation_actions:
        base = list(result.prioritized_mitigation_actions)[:5]
    else:
        fallback: list[HomeownerPrioritizedAction] = []
        for action in result.mitigation_plan[:5]:
            impact = str(action.estimated_risk_reduction_band or "low")
            effort = str(action.effort or "medium")
            if effort not in {"low", "medium", "high"}:
                effort = "medium"
            if impact not in {"low", "medium", "high"}:
                impact = "low"
            timeline = "now" if impact == "high" and effort == "low" else ("this_season" if impact in {"high", "medium"} else "later")
            fallback.append(
                HomeownerPrioritizedAction(
                    action=str(action.title or action.action or "Mitigation action"),
                    explanation=str(action.reason or action.impact_statement or ""),
                    impact_level=impact,  # type: ignore[arg-type]
                    effort_level=effort,  # type: ignore[arg-type]
                    estimated_cost_band=effort,  # type: ignore[arg-type]
                    timeline=timeline,  # type: ignore[arg-type]
                    priority=int(action.priority or 5),
                )
            )
        base = fallback

    return [
        _enrich_prioritized_action(row, tone_level=tone_level)
        for row in base
    ][:5]


def build_homeowner_report(
    result: AssessmentResult,
    *,
    include_professional_debug_metadata: bool = False,
) -> HomeownerReport:
    resolved_region_id = result.resolved_region_id or str(result.property_level_context.get("region_id") or "") or None
    report_generated_at = result.generated_at.astimezone(timezone.utc).isoformat()
    home_hardening_score = (
        result.home_hardening_readiness
        if result.home_hardening_readiness is not None
        else result.insurance_readiness_score
    )
    home_hardening_available = bool(
        result.home_hardening_readiness_score_available
        or result.insurance_readiness_score_available
    )

    key_risk_drivers = [_plain_driver(row) for row in list(result.top_risk_drivers or [])]
    key_risk_drivers = [row for row in key_risk_drivers if row][:6]
    prioritized_actions = _prioritized_actions(result)

    defensible_space_analysis = result.defensible_space_analysis if isinstance(result.defensible_space_analysis, dict) else {}
    zone_findings = _zone_findings(defensible_space_analysis)
    ds_limitations = list(result.defensible_space_limitations_summary or [])

    assessment_limitations = list(result.assessment_limitations_summary or [])
    low_confidence_flags = list(result.low_confidence_flags or [])
    grouped_limitations = [
        str(row.get("summary") or "").strip()
        for row in list(result.assessment_limitations or [])
        if isinstance(row, dict) and str(row.get("summary") or "").strip()
    ]
    combined_limitations = list(
        dict.fromkeys(grouped_limitations + assessment_limitations + ds_limitations + low_confidence_flags)
    )[:6]
    homeowner_confidence = (
        (result.homeowner_summary or {}).get("confidence_summary")
        if isinstance(result.homeowner_summary, dict)
        else {}
    )
    homeowner_confidence = homeowner_confidence if isinstance(homeowner_confidence, dict) else {}
    confidence_headline = str(homeowner_confidence.get("headline") or "").strip() or _confidence_summary(result)
    confidence_limitations = [
        str(item).strip()
        for item in list(homeowner_confidence.get("why_confidence_is_limited") or [])
        if str(item).strip()
    ]
    if confidence_limitations:
        combined_limitations = list(dict.fromkeys(confidence_limitations + combined_limitations))[:6]

    confidence_and_limitations = {
        "confidence_score": result.confidence_score,
        "confidence_tier": result.confidence_tier,
        "use_restriction": result.use_restriction,
        "confidence_statement": confidence_headline,
        "observed_data": list(result.confidence_summary.observed_data or []),
        "estimated_data": list(result.confidence_summary.estimated_data or []),
        "missing_data": list(result.confidence_summary.missing_data or []),
        "accuracy_improvements": list(result.confidence_summary.accuracy_improvements or []),
        "limitations": combined_limitations,
        "fallback_decisions": [
            _dump_value(row) for row in list((result.assessment_diagnostics.fallback_decisions or []))[:8]
        ],
        "decision_support_disclaimer": (
            "This report is decision-support guidance based on prepared geospatial data and provided inputs; "
            "it is not a guarantee of insurability or wildfire safety."
        ),
    }

    professional_debug_metadata: dict[str, Any] | None = None
    if include_professional_debug_metadata:
        professional_debug_metadata = {
            "layer_coverage_audit": [_dump_value(row) for row in result.layer_coverage_audit],
            "coverage_summary": _dump_value(result.coverage_summary),
            "score_evidence_ledger": _dump_value(result.score_evidence_ledger),
            "evidence_quality_summary": _dump_value(result.evidence_quality_summary),
            "assessment_diagnostics": _dump_value(result.assessment_diagnostics),
        }

    tone_level = _select_tone_level(result)

    all_mitigation_actions = _mitigation_actions(result, tone_level=tone_level)
    top_recommended_actions = all_mitigation_actions[:3]
    top_risk_drivers = _summarize_top_risk_drivers(key_risk_drivers, tone_level=tone_level)
    prioritized_actions_summary = _summarize_prioritized_actions(
        prioritized_actions,
        tone_level=tone_level,
    )
    ranked_actions, most_impactful_actions = _rank_mitigation_actions(prioritized_actions_summary)
    what_to_do_first = ranked_actions[0] if ranked_actions else (prioritized_actions_summary[0] if prioritized_actions_summary else {})
    headline_risk_summary = _headline_risk_summary(
        result,
        result.overall_wildfire_risk if result.overall_wildfire_risk is not None else result.wildfire_risk_score,
        tone_level=tone_level,
    )
    limitations_notice = _limitations_notice(result, combined_limitations)

    return HomeownerReport(
        assessment_id=result.assessment_id,
        generated_at=report_generated_at,
        headline_risk_summary=headline_risk_summary,
        top_risk_drivers=top_risk_drivers,
        prioritized_actions=prioritized_actions_summary,
        ranked_actions=ranked_actions,
        most_impactful_actions=most_impactful_actions,
        what_to_do_first=what_to_do_first,
        limitations_notice=limitations_notice,
        report_header={
            "title": "WildfireRisk Advisor Home Hardening Report",
            "subtitle": "Property-specific wildfire risk and practical hardening guidance",
            "assessment_generated_at": result.generated_at.astimezone(timezone.utc).isoformat(),
            "report_generated_at": report_generated_at,
            "report_audience": "homeowner",
        },
        property_summary={
            "address": result.address,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "resolved_region_id": resolved_region_id,
            "coverage_available": result.coverage_available,
            "region_resolution_reason": result.region_resolution.reason,
        },
        score_summary={
            "overall_wildfire_risk": (
                result.overall_wildfire_risk
                if result.overall_wildfire_risk is not None
                else result.wildfire_risk_score
            ),
            "wildfire_risk_score": result.wildfire_risk_score,
            "wildfire_risk_band": _risk_band(result.wildfire_risk_score),
            "wildfire_risk_score_available": result.wildfire_risk_score_available,
            "home_hardening_readiness": home_hardening_score,
            "home_hardening_readiness_band": _home_hardening_band(home_hardening_score),
            "home_hardening_readiness_score_available": home_hardening_available,
            "calibrated_damage_likelihood": result.calibrated_damage_likelihood,
            "empirical_damage_likelihood_proxy": result.empirical_damage_likelihood_proxy,
            "empirical_loss_likelihood_proxy": result.empirical_loss_likelihood_proxy,
            "calibration_applied": result.calibration_applied,
            "calibration_status": result.calibration_status,
            "insurance_readiness_score": result.insurance_readiness_score,
            "insurance_readiness_band": _home_hardening_band(result.insurance_readiness_score),
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "legacy_insurance_readiness_note": "Insurance-facing readiness is retained for compatibility and future-facing workflows.",
            "confidence_score": result.confidence_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
        },
        key_risk_drivers=key_risk_drivers,
        top_risk_drivers_detailed=list(result.top_risk_drivers_detailed or []),
        defensible_space_summary={
            "summary": defensible_space_analysis.get("summary")
            or "Defensible-space analysis was unavailable for this property.",
            "basis_geometry_type": defensible_space_analysis.get("basis_geometry_type") or "unknown",
            "basis_quality": defensible_space_analysis.get("basis_quality") or "unknown",
            "zone_findings": zone_findings,
            "top_near_structure_risk_drivers": list(result.top_near_structure_risk_drivers or []),
            "limitations": ds_limitations,
            "analysis_status": ((defensible_space_analysis.get("data_quality") or {}).get("analysis_status") or "unknown"),
        },
        top_recommended_actions=top_recommended_actions,
        prioritized_mitigation_actions=prioritized_actions,
        mitigation_plan=all_mitigation_actions,
        home_hardening_readiness_summary={
            "home_hardening_readiness": home_hardening_score,
            "home_hardening_readiness_score_available": home_hardening_available,
            "summary": result.readiness_summary,
            "blockers": list(result.readiness_blockers or []),
            "factors": [_dump_value(row) for row in list(result.readiness_factors or [])[:8]],
            "top_recommended_actions": [row.model_dump() for row in top_recommended_actions],
        },
        insurance_readiness_summary={
            "insurance_readiness_score": result.insurance_readiness_score,
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "readiness_summary": result.readiness_summary,
            "readiness_blockers": list(result.readiness_blockers or []),
            "readiness_factors": [_dump_value(row) for row in list(result.readiness_factors or [])[:8]],
            "status": "optional_future_facing",
            "note": "Insurance readiness outputs are optional references and not the primary homeowner outcome.",
        },
        confidence_summary=result.confidence_summary,
        confidence_and_limitations=confidence_and_limitations,
        metadata={
            "model_version": result.model_version,
            "product_version": result.product_version,
            "api_version": result.api_version,
            "model_governance": _dump_value(result.model_governance),
            "region_data_version": result.region_data_version,
            "data_bundle_version": result.data_bundle_version,
            "calibration_version": result.calibration_version,
            "calibration_method": result.calibration_method,
            "calibration_limitations": list(result.calibration_limitations or []),
            "calibration_scope_warning": result.calibration_scope_warning,
            "ruleset": {
                "ruleset_id": result.ruleset_id,
                "ruleset_name": result.ruleset_name,
                "ruleset_version": result.ruleset_version,
            },
        },
        professional_debug_metadata=professional_debug_metadata,
    )


def _export_confidence_summary(report: HomeownerReport) -> dict[str, object]:
    score_summary = report.score_summary or {}
    confidence_and_limitations = report.confidence_and_limitations or {}
    tier = str(
        score_summary.get("confidence_tier")
        or confidence_and_limitations.get("confidence_tier")
        or "preliminary"
    ).strip().lower()
    score = _to_float(score_summary.get("confidence_score"))

    if tier == "high":
        summary = "High confidence: key property and location inputs were observed directly."
    elif tier in {"moderate", "medium"}:
        summary = "Moderate confidence: most major inputs are available, with some estimated details."
    else:
        summary = "Lower confidence: several details were estimated or missing, so this report is advisory."

    limitations = [
        str(row).strip()
        for row in list(confidence_and_limitations.get("limitations") or [])
        if str(row).strip()
    ][:3]
    return {
        "confidence_tier": tier,
        "confidence_score": score,
        "summary": summary,
        "limitations": limitations,
    }


def export_homeowner_report(
    result: AssessmentResult,
    *,
    output_format: Literal["structured", "pdf"] = "structured",
) -> dict[str, object] | bytes:
    report = build_homeowner_report(result, include_professional_debug_metadata=False)
    if output_format == "pdf":
        return render_homeowner_report_pdf(report)

    property_summary = report.property_summary or {}
    ranked_actions = [dict(row) for row in list(report.ranked_actions or [])][:5]
    if not ranked_actions:
        ranked_actions = [dict(row) for row in list(report.prioritized_actions or [])][:5]
    most_impactful = [dict(row) for row in list(report.most_impactful_actions or [])][:2]
    if not most_impactful:
        most_impactful = ranked_actions[:2]
    what_to_do_first = dict(report.what_to_do_first or (ranked_actions[0] if ranked_actions else {}))

    limitations_notice = str(report.limitations_notice or "").strip()
    confidence_summary = _export_confidence_summary(report)
    if confidence_summary.get("confidence_tier") in {"low", "preliminary"} and limitations_notice:
        if "estimated" not in limitations_notice.lower() and "missing" not in limitations_notice.lower():
            limitations_notice = limitations_notice + " Some details were estimated or missing."

    return {
        "assessment_id": report.assessment_id,
        "generated_at": report.generated_at,
        "address": str(property_summary.get("address") or ""),
        "headline_risk_summary": str(report.headline_risk_summary or ""),
        "top_risk_drivers": list(report.top_risk_drivers or [])[:4],
        "prioritized_actions": ranked_actions[:5],
        "most_impactful_actions": most_impactful,
        "what_to_do_first": what_to_do_first,
        "confidence_summary": confidence_summary,
        "limitations_notice": limitations_notice,
        "disclaimer": (
            "This report supports homeowner planning and conversations with contractors, agents, "
            "or insurers. It is not a guarantee of wildfire outcomes or insurability."
        ),
    }


def _wrap_text_line(text: str, *, width: int = 96, prefix: str = "") -> list[str]:
    normalized = " ".join(str(text or "").replace("\n", " ").split())
    if not normalized:
        return []
    wrapped = textwrap.wrap(
        normalized,
        width=max(16, width - len(prefix)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return []
    lines = [f"{prefix}{wrapped[0]}"]
    continuation = " " * len(prefix)
    for chunk in wrapped[1:]:
        lines.append(f"{continuation}{chunk}")
    return lines


def _escape_pdf_text(text: str) -> str:
    cleaned = str(text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    encoded = cleaned.encode("latin-1", errors="replace")
    return encoded.decode("latin-1")


def _build_report_lines(report: HomeownerReport) -> list[str]:
    lines: list[str] = []

    header = report.report_header or {}
    property_summary = report.property_summary or {}
    score_summary = report.score_summary or {}
    ds_summary = report.defensible_space_summary or {}
    readiness = report.home_hardening_readiness_summary or {}
    insurance_reference = report.insurance_readiness_summary or {}
    confidence = report.confidence_and_limitations or {}
    metadata = report.metadata or {}

    lines.extend(_wrap_text_line(str(header.get("title") or "WildfireRisk Advisor Home Hardening Report")))
    lines.extend(_wrap_text_line(str(header.get("subtitle") or "")))
    lines.append("")

    lines.extend(_wrap_text_line(f"Assessment ID: {report.assessment_id}"))
    lines.extend(_wrap_text_line(f"Address: {property_summary.get('address') or 'Unknown'}"))
    lines.extend(_wrap_text_line(f"Region: {property_summary.get('resolved_region_id') or 'unknown'}"))
    lines.extend(_wrap_text_line(f"Assessment Date: {header.get('assessment_generated_at') or report.generated_at}"))
    lines.append("")

    lines.extend(_wrap_text_line("Score Summary"))
    lines.extend(_wrap_text_line(
        f"Overall Wildfire Risk: {score_summary.get('overall_wildfire_risk', score_summary.get('wildfire_risk_score'))} "
        f"({score_summary.get('wildfire_risk_band', 'unavailable')})"
    ))
    lines.extend(_wrap_text_line(
        f"Home Hardening Readiness: {score_summary.get('home_hardening_readiness')} "
        f"({score_summary.get('home_hardening_readiness_band', 'unavailable')})"
    ))
    lines.extend(_wrap_text_line(
        f"Confidence: {score_summary.get('confidence_score')} ({score_summary.get('confidence_tier', 'unknown')})"
    ))
    lines.append("")

    if report.headline_risk_summary:
        lines.extend(_wrap_text_line("Homeowner Summary"))
        lines.extend(_wrap_text_line(report.headline_risk_summary, prefix="- "))
        if report.what_to_do_first:
            lines.extend(
                _wrap_text_line(
                    f"What to do first: {report.what_to_do_first.get('action', 'No primary action available')}",
                    prefix="- ",
                )
            )
        if report.limitations_notice:
            lines.extend(_wrap_text_line(f"Limitations: {report.limitations_notice}", prefix="- "))
        lines.append("")

    lines.extend(_wrap_text_line("Top Risk Drivers"))
    if report.key_risk_drivers:
        for driver in report.key_risk_drivers[:6]:
            lines.extend(_wrap_text_line(driver, prefix="- "))
    else:
        lines.extend(_wrap_text_line("No major risk drivers were identified in this run.", prefix="- "))
    lines.append("")

    lines.extend(_wrap_text_line("Defensible-Space Findings"))
    lines.extend(_wrap_text_line(str(ds_summary.get("summary") or "No defensible-space summary available."), prefix="- "))
    zone_findings = ds_summary.get("zone_findings") if isinstance(ds_summary.get("zone_findings"), list) else []
    for row in zone_findings[:4]:
        if not isinstance(row, dict):
            continue
        lines.extend(
            _wrap_text_line(
                f"{row.get('distance_band_ft', row.get('zone_key', 'zone'))}: "
                f"risk={row.get('risk_level', 'unknown')}, "
                f"vegetation_density={row.get('vegetation_density')}, "
                f"evidence={row.get('evidence_status', 'unknown')}",
                prefix="- ",
            )
        )
    for limitation in list(ds_summary.get("limitations") or [])[:3]:
        lines.extend(_wrap_text_line(f"Limitation: {limitation}", prefix="- "))
    lines.append("")

    lines.extend(_wrap_text_line("Prioritized Mitigation Actions"))
    for action in report.mitigation_plan[:8]:
        lines.extend(
            _wrap_text_line(
                f"Priority {action.priority}: {action.title}"
                + (f" [{action.target_zone}]" if action.target_zone else ""),
                prefix="- ",
            )
        )
        if action.explanation:
            lines.extend(_wrap_text_line(action.explanation, prefix="  "))
    if not report.mitigation_plan:
        lines.extend(_wrap_text_line("No prioritized actions were generated.", prefix="- "))
    lines.append("")

    lines.extend(_wrap_text_line("Home Hardening Checklist"))
    if report.prioritized_mitigation_actions:
        for row in report.prioritized_mitigation_actions[:5]:
            lines.extend(
                _wrap_text_line(
                    f"{row.action} (impact={row.impact_level}, effort={row.effort_level}, cost={row.estimated_cost_band}, timeline={row.timeline})",
                    prefix="- ",
                )
            )
    else:
        lines.extend(_wrap_text_line("No checklist items available.", prefix="- "))
    lines.append("")

    lines.extend(_wrap_text_line("Mitigation Simulator Examples"))
    lines.extend(
        _wrap_text_line(
            "Use the simulator in the app to compare current risk with upgrade scenarios and see which top drivers improve.",
            prefix="- ",
        )
    )
    lines.append("")

    lines.extend(_wrap_text_line("Home Hardening Readiness"))
    lines.extend(_wrap_text_line(str(readiness.get("summary") or "No home hardening summary available."), prefix="- "))
    lines.extend(_wrap_text_line("Top Recommended Actions", prefix="- "))
    for action in report.top_recommended_actions[:3]:
        lines.extend(_wrap_text_line(action.title, prefix="  - "))
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    if blockers:
        lines.extend(_wrap_text_line("Key blockers:", prefix="- "))
        for blocker in blockers[:5]:
            lines.extend(_wrap_text_line(str(blocker), prefix="  - "))
    if insurance_reference:
        lines.extend(
            _wrap_text_line(
                "Optional insurance reference score: "
                + str(insurance_reference.get("insurance_readiness_score", "unavailable")),
                prefix="- ",
            )
        )
    lines.append("")

    lines.extend(_wrap_text_line("Confidence and Limitations"))
    lines.extend(_wrap_text_line(str(confidence.get("confidence_statement") or "Confidence summary unavailable."), prefix="- "))
    for limitation in list(confidence.get("limitations") or [])[:6]:
        lines.extend(_wrap_text_line(str(limitation), prefix="- "))
    lines.extend(
        _wrap_text_line(
            str(
                confidence.get("decision_support_disclaimer")
                or "This report is decision-support guidance and not a guarantee of insurability or wildfire safety."
            ),
            prefix="- ",
        )
    )
    lines.append("")

    governance = metadata.get("model_governance") if isinstance(metadata.get("model_governance"), dict) else {}
    lines.extend(_wrap_text_line("Metadata"))
    lines.extend(_wrap_text_line(f"Model Version: {metadata.get('model_version')}", prefix="- "))
    lines.extend(_wrap_text_line(f"Scoring Model Version: {governance.get('scoring_model_version')}", prefix="- "))
    lines.extend(_wrap_text_line(f"Ruleset Version: {governance.get('ruleset_version')}", prefix="- "))
    lines.extend(_wrap_text_line(f"Region Data Version: {metadata.get('region_data_version')}", prefix="- "))

    return lines


def _paginate_lines(lines: list[str], *, max_lines_per_page: int = 46) -> list[list[str]]:
    pages: list[list[str]] = []
    page: list[str] = []
    for line in lines:
        page.append(line)
        if len(page) >= max_lines_per_page:
            pages.append(page)
            page = []
    if page:
        pages.append(page)
    if not pages:
        pages.append([""])
    return pages


def _build_pdf_content_stream(page_lines: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
    first = True
    for line in page_lines:
        escaped = _escape_pdf_text(line)
        if first:
            commands.append(f"({escaped}) Tj")
            first = False
        else:
            commands.append("T*")
            commands.append(f"({escaped}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def _serialize_pdf(objects: dict[int, bytes]) -> bytes:
    max_id = max(objects.keys()) if objects else 0
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (max_id + 1)

    for obj_id in range(1, max_id + 1):
        payload = objects.get(obj_id)
        if payload is None:
            continue
        offsets[obj_id] = len(out)
        out.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        out.extend(payload)
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, max_id + 1):
        out.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))

    out.extend(b"trailer\n")
    out.extend(f"<< /Size {max_id + 1} /Root 1 0 R >>\n".encode("ascii"))
    out.extend(b"startxref\n")
    out.extend(f"{xref_offset}\n".encode("ascii"))
    out.extend(b"%%EOF")
    return bytes(out)


def render_homeowner_report_pdf(report: HomeownerReport) -> bytes:
    lines = _build_report_lines(report)
    pages = _paginate_lines(lines)

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    page_ids: list[int] = []
    for idx, page_lines in enumerate(pages):
        content_id = 4 + idx * 2
        page_id = 5 + idx * 2
        page_ids.append(page_id)

        content_stream = _build_pdf_content_stream(page_lines)
        objects[content_id] = (
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
            + content_stream
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")

    return _serialize_pdf(objects)
