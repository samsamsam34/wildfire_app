from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import json
import os
import re
import textwrap
from typing import Any, Literal

from backend.insurability import derive_insurability_status
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
        return "We could not produce a reliable wildfire risk estimate from the available data."
    if tone_level == "advisory":
        return (
            f"Your property appears to have {risk_label} wildfire risk, but some details were estimated, "
            "so treat this as a screening assessment."
        )
    if tone_level == "slightly_hedged":
        return f"Your property appears to have {risk_label} wildfire risk based on available property and area conditions."
    return f"Your property shows {risk_label} wildfire risk in this assessment."


def _specificity_summary(result: AssessmentResult, homeowner_trust_summary: dict[str, Any]) -> dict[str, Any]:
    raw = _dump_value(getattr(result, "specificity_summary", None))
    if isinstance(raw, dict):
        tier = str(raw.get("specificity_tier") or "").strip().lower()
        headline = str(raw.get("headline") or "").strip()
        what_this_means = str(raw.get("what_this_means") or "").strip()
        comparison_allowed = bool(raw.get("comparison_allowed"))
    else:
        tier = ""
        headline = ""
        what_this_means = ""
        comparison_allowed = False

    mode = str(getattr(result, "assessment_mode", "") or "").strip().lower()
    if mode == "insufficient_data":
        tier = "insufficient_data"
    elif tier not in {"property_specific", "address_level", "regional_estimate"}:
        tier = str(getattr(result, "assessment_specificity_tier", "regional_estimate") or "regional_estimate").strip().lower()
        if tier not in {"property_specific", "address_level", "regional_estimate"}:
            tier = "regional_estimate"

    if not headline:
        if tier == "property_specific":
            headline = "Property-specific estimate"
        elif tier == "address_level":
            headline = "Address-level estimate"
        elif tier == "regional_estimate":
            headline = "Regional estimate"
        else:
            headline = "Insufficient data for property estimate"

    if not what_this_means:
        if tier == "property_specific":
            what_this_means = (
                "This result uses home-specific geometry and nearby conditions and can usually distinguish nearby homes."
            )
        elif tier == "address_level":
            what_this_means = (
                "This result uses address-level and nearby context, but some home-specific details were estimated."
            )
        elif tier == "regional_estimate":
            what_this_means = (
                "This result relies more on shared neighborhood and regional conditions, so nearby homes may appear similar."
            )
        else:
            what_this_means = (
                "There was not enough reliable data for a property-level estimate. Nearby homes may appear similar because this run uses limited regional context."
            )

    safeguard_triggered = bool(homeowner_trust_summary.get("nearby_home_comparison_safeguard_triggered"))
    if tier in {"regional_estimate", "insufficient_data"} or safeguard_triggered:
        comparison_allowed = False

    return {
        "specificity_tier": tier,
        "headline": headline,
        "what_this_means": what_this_means,
        "comparison_allowed": bool(comparison_allowed),
    }


def _apply_specificity_tone_guardrail(base_tone: str, specificity_summary: dict[str, Any]) -> str:
    tier = str(specificity_summary.get("specificity_tier") or "").strip().lower()
    comparison_allowed = bool(specificity_summary.get("comparison_allowed"))
    if tier in {"regional_estimate", "insufficient_data"}:
        return "advisory"
    if tier == "property_specific" and not comparison_allowed:
        return "advisory" if base_tone == "advisory" else "slightly_hedged"
    if tier == "address_level" and base_tone == "direct":
        return "slightly_hedged"
    return base_tone


def _with_low_specificity_limitation(limitations_notice: str, specificity_summary: dict[str, Any]) -> str:
    tier = str(specificity_summary.get("specificity_tier") or "").strip().lower()
    comparison_allowed = bool(specificity_summary.get("comparison_allowed"))
    if tier not in {"regional_estimate", "insufficient_data"} and comparison_allowed:
        return limitations_notice
    caution = "Nearby homes may appear similar because this estimate relies more on shared neighborhood and regional conditions."
    base = str(limitations_notice or "").strip()
    if caution.lower() in base.lower():
        return base
    if not base:
        return caution
    return f"{base.rstrip('.')}." + f" {caution}"


def _simplify_homeowner_action(action_row: dict[str, object]) -> dict[str, object]:
    simple: dict[str, object] = {
        "action": str(action_row.get("action") or "").strip(),
        "effort_level": str(action_row.get("effort_level") or "medium"),
        "expected_effect": str(action_row.get("expected_effect") or "moderate"),
        "why_this_matters": str(action_row.get("why_this_matters") or "").strip(),
    }
    optional = {
        "impact_level": str(action_row.get("impact_level") or "").strip(),
        "what_it_reduces": str(action_row.get("what_it_reduces") or "").strip(),
        "estimated_benefit": str(action_row.get("estimated_benefit") or "").strip(),
    }
    for key, value in optional.items():
        if value:
            simple[key] = value
    return simple


def _clean_homeowner_line(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("_", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    friendly_map = {
        "defensible_space_ft": "defensible space clearance distance",
        "roof_type": "roof material/type",
        "vent_type": "vent protection type",
        "window_type": "window protection/type",
        "construction_year": "home construction year",
        "siding_type": "siding material/type",
        "structure_geometry": "building footprint/location geometry",
        "parcel_geometry": "parcel boundary geometry",
    }
    lowered = normalized.lower()
    if lowered in friendly_map:
        normalized = friendly_map[lowered]
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]


def _dedupe_nonempty_lines(values: list[Any], *, limit: int = 3) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for row in values:
        line = _clean_homeowner_line(row)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
        if len(cleaned) >= max(1, int(limit)):
            break
    return cleaned


def _build_homeowner_limitations_snapshot(
    *,
    result: AssessmentResult,
    homeowner_improve_your_result: dict[str, Any],
) -> dict[str, object]:
    observed = _dedupe_nonempty_lines(
        list(result.confidence_summary.observed_data or []) + list(result.what_was_observed or []),
        limit=3,
    )
    estimated = _dedupe_nonempty_lines(
        list(result.confidence_summary.estimated_data or [])
        + list(result.confidence_summary.fallback_assumptions or [])
        + list(result.what_was_estimated or []),
        limit=3,
    )
    missing = _dedupe_nonempty_lines(
        list(result.confidence_summary.missing_data or []) + list(result.what_was_missing or []),
        limit=3,
    )

    optional_follow_up_inputs = (
        homeowner_improve_your_result.get("optional_follow_up_inputs")
        if isinstance(homeowner_improve_your_result.get("optional_follow_up_inputs"), list)
        else []
    )
    follow_up_lines: list[str] = []
    for row in optional_follow_up_inputs:
        if not isinstance(row, dict):
            continue
        label = _clean_homeowner_line(row.get("label") or row.get("input_key") or "")
        prompt = str(row.get("prompt") or "").strip()
        if label and prompt:
            follow_up_lines.append(f"{label}: {prompt}")
        elif label:
            follow_up_lines.append(label)
    inputs_to_improve = _dedupe_nonempty_lines(
        follow_up_lines
        + list(result.confidence_summary.accuracy_improvements or [])
        + list(result.recommended_data_improvements or []),
        limit=3,
    )

    observed_count = len(list(result.confidence_summary.observed_data or []))
    estimated_count = len(list(result.confidence_summary.estimated_data or []))
    missing_count = len(list(result.confidence_summary.missing_data or []))
    headline = (
        "Measured vs estimated snapshot: "
        f"{observed_count} observed, {estimated_count} estimated, {missing_count} missing inputs."
    )

    return {
        "headline": headline,
        "directly_observed": observed or ["No major property-specific details were directly observed yet."],
        "estimated_or_inferred": estimated or ["No major estimated assumptions were highlighted."],
        "missing_or_unknown": missing or ["No major missing fields were highlighted."],
        "inputs_to_improve": inputs_to_improve or [
            "Confirm roof type, vent protection, and defensible-space details to improve confidence."
        ],
    }


def _build_first_screen_payload(
    *,
    specificity_summary: dict[str, Any],
    property_confidence_summary: dict[str, Any],
    overall_wildfire_risk: dict[str, object],
    headline_risk_summary: str,
    top_risk_drivers: list[str],
    ranked_actions: list[dict[str, object]],
    what_to_do_first: dict[str, object],
    limitations_notice: str,
) -> dict[str, object]:
    limitation_line = " ".join(str(limitations_notice or "").replace("\n", " ").split()).strip()
    if len(limitation_line) > 240:
        limitation_line = limitation_line[:237].rstrip() + "..."
    top_actions = [_simplify_homeowner_action(row) for row in list(ranked_actions or [])[:3]]
    return {
        "overall_wildfire_risk": dict(overall_wildfire_risk or {}),
        "specificity_summary": dict(specificity_summary),
        "property_confidence_summary": dict(property_confidence_summary or {}),
        "top_risk_drivers": [str(row).strip() for row in list(top_risk_drivers or [])[:3] if str(row).strip()],
        "top_actions": top_actions,
        "what_to_do_first": _simplify_homeowner_action(dict(what_to_do_first or {})) if what_to_do_first else {},
        "limitations_note": limitation_line,
        # Backward-compatible alias for existing consumers.
        "headline_risk_summary": str(headline_risk_summary or "").strip(),
    }


def _insurance_outlook_status(
    *,
    wildfire_risk_score: float | None,
    home_hardening_readiness: float | None,
    confidence_tier: str | None,
) -> str:
    risk = _to_float(wildfire_risk_score)
    readiness = _to_float(home_hardening_readiness)
    tier = str(confidence_tier or "").strip().lower()

    if risk is not None and (risk >= 70.0 or (risk >= 60.0 and (readiness is not None and readiness < 45.0))):
        return "High Risk of Insurance Issues"
    if readiness is not None and readiness < 30.0:
        return "High Risk of Insurance Issues"
    if (risk is not None and risk >= 45.0) or (readiness is not None and readiness < 60.0):
        return "At Risk"
    if tier in {"low", "preliminary"} and risk is not None and risk >= 40.0:
        return "At Risk"
    return "Likely Insurable"


def _status_one_sentence_summary(
    *,
    status_label: str,
    headline_risk_summary: str,
    tone_level: str,
) -> str:
    headline = str(headline_risk_summary or "").strip().rstrip(".")
    if status_label == "Likely Insurable":
        base = (
            "Screening status suggests comparatively fewer wildfire-related insurance risk signals based on observable factors"
        )
    elif status_label == "High Risk of Insurance Issues":
        base = (
            "Screening status suggests elevated wildfire-related insurance friction risk due to strong wildfire exposure and limited protective conditions"
        )
    else:
        if status_label:
            base = "Screening status suggests meaningful wildfire-related insurance friction risk due to wildfire exposure and remaining hardening gaps"
        elif headline:
            return f"{headline}."
        else:
            return "This report provides wildfire-risk screening guidance and practical next steps."
    if tone_level == "advisory":
        return f"{base}, and some details were estimated, so treat this as a screening assessment."
    return f"{base}."


def _brief_confidence_limitations(
    *,
    confidence_tier: str | None,
    limitations: list[str],
) -> str:
    tier = str(confidence_tier or "preliminary").strip().lower()
    if tier == "high":
        prefix = "Most key inputs were directly observed."
    elif tier in {"moderate", "medium"}:
        prefix = "Some key inputs were directly observed, and some were estimated."
    else:
        prefix = "Some key inputs were estimated or missing, so treat this as a screening assessment."
    first_limitation = str((limitations or [""])[0] or "").strip()
    if not first_limitation:
        return prefix
    return f"{prefix} Main limitation: {first_limitation}"


def _before_after_snapshot(
    *,
    simulation_overview: dict[str, Any] | None,
) -> dict[str, object] | None:
    if not isinstance(simulation_overview, dict) or not simulation_overview:
        return None
    embedded = (
        simulation_overview.get("homeowner_before_after_summary")
        if isinstance(simulation_overview.get("homeowner_before_after_summary"), dict)
        else {}
    )
    embedded = dict(embedded) if isinstance(embedded, dict) else {}
    scenario_name = str(simulation_overview.get("scenario_name") or "").strip() or "what-if"
    wildfire_delta = _to_float(
        embedded.get("wildfire_risk_score_delta")
        if embedded.get("wildfire_risk_score_delta") is not None
        else simulation_overview.get("wildfire_risk_score_delta")
    )
    hardening_delta = _to_float(
        (
            embedded.get("home_hardening_readiness_delta")
            if embedded.get("home_hardening_readiness_delta") is not None
            else (
                simulation_overview.get("home_hardening_readiness_delta")
                if simulation_overview.get("home_hardening_readiness_delta") is not None
                else simulation_overview.get("insurance_readiness_score_delta")
            )
        )
    )
    current_status = str(embedded.get("current_insurability_status") or "").strip()
    projected_status = str(embedded.get("projected_insurability_status") or "").strip()
    status_shift = str(embedded.get("status_shift") or "").strip()
    action_rows = [
        row
        for row in list(embedded.get("top_actions_driving_change") or [])
        if isinstance(row, dict) and str(row.get("action") or "").strip()
    ][:3]
    if wildfire_delta is None and hardening_delta is None and not embedded:
        return None
    parts: list[str] = []
    if current_status and projected_status:
        if current_status != projected_status:
            parts.append(f"status shifted from {current_status} to {projected_status}")
        else:
            parts.append(f"status remained {projected_status}")
    if wildfire_delta is not None:
        direction = (
            "decreased"
            if wildfire_delta < 0
            else ("increased" if wildfire_delta > 0 else "did not change")
        )
        parts.append(f"wildfire risk score {direction} ({wildfire_delta:+.1f})")
    if hardening_delta is not None:
        direction = (
            "increased"
            if hardening_delta > 0
            else ("decreased" if hardening_delta < 0 else "did not change")
        )
        parts.append(f"home hardening readiness score {direction} ({hardening_delta:+.1f})")
    if action_rows:
        lead_action = str(action_rows[0].get("action") or "").strip()
        if lead_action:
            parts.append(f"top driver: {lead_action}")
    summary = str(embedded.get("summary") or "").strip() or (f"What-if scenario ({scenario_name}): " + (
        "; ".join(parts) if parts else "no measurable score change was recorded."
    ))
    return {
        "available": True,
        "scenario_name": scenario_name,
        "wildfire_risk_score_delta": wildfire_delta,
        "home_hardening_readiness_delta": hardening_delta,
        "current_insurability_status": current_status,
        "projected_insurability_status": projected_status,
        "status_shift": status_shift,
        "top_actions_driving_change": action_rows,
        "summary": summary,
        "technical_summary": {
            "wildfire_risk_score_before": _to_float(embedded.get("wildfire_risk_score_before")),
            "wildfire_risk_score_after": _to_float(embedded.get("wildfire_risk_score_after")),
            "home_hardening_readiness_before": _to_float(embedded.get("home_hardening_readiness_before")),
            "home_hardening_readiness_after": _to_float(embedded.get("home_hardening_readiness_after")),
            "confidence_tier_before": str(embedded.get("confidence_tier_before") or "").strip(),
            "confidence_tier_after": str(embedded.get("confidence_tier_after") or "").strip(),
        },
    }


def _build_homeowner_focus_summary(
    *,
    insurability_status: str,
    insurability_status_reasons: list[str],
    insurability_status_methodology_note: str,
    confidence_tier: str | None,
    headline_risk_summary: str,
    top_risk_drivers: list[str],
    ranked_actions: list[dict[str, object]],
    limitations: list[str],
    limitations_snapshot: dict[str, object],
    tone_level: str,
    simulation_overview: dict[str, Any] | None = None,
) -> dict[str, object]:
    status_label = str(insurability_status or "").strip()
    top_actions = [_simplify_homeowner_action(row) for row in list(ranked_actions or [])[:3]]
    return {
        "insurability_status": status_label,
        "insurability_status_reasons": [str(v).strip() for v in list(insurability_status_reasons or []) if str(v).strip()][:3],
        "insurability_status_methodology_note": str(insurability_status_methodology_note or "").strip(),
        # Backward-compatible alias retained for existing consumers.
        "status_label": status_label,
        "question_answer": (
            f"{status_label}: heuristic screening status based on observable wildfire risk factors and property conditions, with practical next steps."
        ),
        "one_sentence_summary": _status_one_sentence_summary(
            status_label=status_label,
            headline_risk_summary=headline_risk_summary,
            tone_level=tone_level,
        ),
        "top_risk_drivers": [str(row).strip() for row in list(top_risk_drivers or [])[:3] if str(row).strip()],
        "top_recommended_actions": top_actions,
        "before_after_summary": _before_after_snapshot(simulation_overview=simulation_overview),
        "limitations_snapshot": dict(limitations_snapshot or {}),
        "confidence_limitations_summary": _brief_confidence_limitations(
            confidence_tier=confidence_tier,
            limitations=limitations,
        ),
        "advanced_details_hint": (
            "Technical subscores, diagnostics, evidence ledgers, and calibration metadata are available in internal details."
        ),
    }


def _build_internal_calibration_debug_block(
    result: AssessmentResult,
    *,
    calibration_metadata_requested: bool,
    optional_calibration: dict[str, object] | None,
    calibration_prob: float | None,
    empirical_damage_prob: float | None,
    empirical_loss_prob: float | None,
) -> dict[str, object]:
    return {
        "subscores": {
            "wildfire_risk_score": result.wildfire_risk_score,
            "overall_wildfire_risk": result.overall_wildfire_risk,
            "site_hazard_score": result.site_hazard_score,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "home_hardening_readiness": result.home_hardening_readiness,
            "insurance_readiness_score": result.insurance_readiness_score,
            "score_summaries": _dump_value(result.score_summaries),
            "site_hazard_section": _dump_value(result.site_hazard_section),
            "home_ignition_vulnerability_section": _dump_value(result.home_ignition_vulnerability_section),
            "insurance_readiness_section": _dump_value(result.insurance_readiness_section),
            "weighted_contributions": {
                str(key): _dump_value(value)
                for key, value in dict(result.weighted_contributions or {}).items()
            },
            "factor_breakdown": _dump_value(result.factor_breakdown),
        },
        "diagnostics": {
            "assessment_status": result.assessment_status,
            "assessment_blockers": list(result.assessment_blockers or []),
            "assessment_limitations_summary": list(result.assessment_limitations_summary or []),
            "assessment_diagnostics": _dump_value(result.assessment_diagnostics),
            "coverage_summary": _dump_value(result.coverage_summary),
            "layer_coverage_audit": [_dump_value(row) for row in list(result.layer_coverage_audit or [])],
            "specificity_summary": _dump_value(result.specificity_summary),
            "confidence_summary": _dump_value(result.confidence_summary),
        },
        "evidence_ledgers": {
            "score_evidence_ledger": _dump_value(result.score_evidence_ledger),
            "evidence_quality_summary": _dump_value(result.evidence_quality_summary),
        },
        "calibration_fields": {
            "calibration_metadata_requested": bool(calibration_metadata_requested),
            "calibration_applied": result.calibration_applied,
            "calibration_status": result.calibration_status,
            "calibration_version": result.calibration_version,
            "calibration_method": result.calibration_method,
            "calibration_limitations": list(result.calibration_limitations or []),
            "calibration_scope_warning": result.calibration_scope_warning,
            "optional_public_outcome_calibration": optional_calibration,
            "calibrated_damage_likelihood": calibration_prob,
            "empirical_damage_likelihood_proxy": empirical_damage_prob,
            "empirical_loss_likelihood_proxy": empirical_loss_prob,
        },
        "compatibility_outputs": {
            "insurance_readiness_summary": {
                "insurance_readiness_score": result.insurance_readiness_score,
                "insurance_readiness_score_available": result.insurance_readiness_score_available,
                "readiness_summary": result.readiness_summary,
                "readiness_blockers": list(result.readiness_blockers or []),
                "readiness_factors": [_dump_value(row) for row in list(result.readiness_factors or [])[:8]],
            },
            "legacy_weighted_wildfire_risk_score": result.legacy_weighted_wildfire_risk_score,
            "model_governance": _dump_value(result.model_governance),
            "ruleset": {
                "ruleset_id": result.ruleset_id,
                "ruleset_name": result.ruleset_name,
                "ruleset_version": result.ruleset_version,
            },
        },
    }


def _optional_public_outcome_calibration_metadata(result: AssessmentResult) -> dict[str, object] | None:
    status = str(result.calibration_status or "disabled").strip().lower()
    has_payload = bool(
        result.calibration_applied
        or result.calibrated_damage_likelihood is not None
        or result.empirical_damage_likelihood_proxy is not None
        or status not in {"", "disabled", "disabled_no_artifact"}
    )
    if not has_payload:
        return None
    return {
        "available": bool(result.calibration_applied and result.calibrated_damage_likelihood is not None),
        "status": status,
        "calibrated_public_outcome_probability": (
            float(result.calibrated_damage_likelihood)
            if result.calibrated_damage_likelihood is not None
            else None
        ),
        "summary": (
            "Optional public-outcome calibration metadata is available as additive context only. "
            "Deterministic risk drivers and prioritized actions remain the primary homeowner guidance."
        ),
        "caveat": (
            "This optional value is based on public observed wildfire outcomes and should not be interpreted "
            "as insurer underwriting or claims probability."
        ),
    }


def _risk_driver_from_text(driver_text: str, *, tone_level: str) -> str:
    text = str(driver_text or "").strip().lower()
    if not text:
        base = "Nearby conditions around the home"
    elif any(token in text for token in ("vegetation", "fuel", "tree cover", "brush", "wildland")):
        base = "Vegetation close to the home"
    elif any(token in text for token in ("slope", "terrain", "topography")):
        base = "Terrain near the home"
    elif "defensible space" in text:
        base = "Defensible space in the closest zones"
    elif "ember" in text:
        base = "Wind-blown ember exposure"
    elif any(token in text for token in ("hardening", "vulnerability", "structure")):
        base = "Home hardening conditions"
    else:
        base = "Nearby conditions around the home"

    if tone_level == "direct":
        return f"{base} is increasing wildfire exposure."
    if tone_level == "slightly_hedged":
        return f"{base} appears to be increasing wildfire exposure."
    return f"{base} may be increasing wildfire exposure."


def _summarize_top_risk_drivers(
    key_risk_drivers: list[str],
    *,
    tone_level: str,
    limit: int = 3,
) -> list[str]:
    cleaned = [_de_jargonize(_plain_driver(row)) for row in key_risk_drivers if str(row).strip()]
    cleaned = [row for row in cleaned if row]
    unique = list(dict.fromkeys([_risk_driver_from_text(row, tone_level=tone_level) for row in cleaned if row]))
    if unique:
        return unique[: max(1, int(limit))]
    return ["Key property-level risk drivers were not clearly observed, so this result leans on broader area conditions."]


def _estimated_benefit_phrase(impact_level: str, tone_level: str) -> str:
    impact = str(impact_level or "low").lower()
    if tone_level == "advisory":
        if impact == "high":
            phrase = "Directional estimate: could provide meaningful risk reduction."
        elif impact == "medium":
            phrase = "Directional estimate: could provide moderate risk reduction."
        else:
            phrase = "Directional estimate: likely incremental risk reduction."
        return phrase
    if tone_level == "slightly_hedged":
        if impact == "high":
            return "Expected to provide meaningful risk reduction."
        if impact == "medium":
            return "Expected to provide moderate risk reduction."
        return "Expected to provide incremental risk reduction."
    if impact == "high":
        return "Likely meaningful risk reduction if completed and maintained."
    if impact == "medium":
        return "Likely moderate risk reduction if completed and maintained."
    return "Likely incremental risk reduction."


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
        return f"{action} could reduce {what_it_reduces}."
    if tone_level == "slightly_hedged":
        return f"{action} can reduce {what_it_reduces}."
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
    limit: int = 3,
) -> list[dict[str, object]]:
    impact_rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(
        list(prioritized_actions or []),
        key=lambda row: (impact_rank.get(str(row.impact_level), 3), int(row.priority or 99), str(row.action or "").lower()),
    )
    rows: list[dict[str, object]] = []
    for row in ordered[: max(1, int(limit))]:
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
        prefix = "Screening note: some results were estimated because "
        return prefix + "; ".join(combined_limitations[:2])
    if result.confidence_tier in {"low", "preliminary"}:
        return "Some key property details were estimated or missing, so use this as a screening assessment."
    if result.confidence_tier == "moderate":
        return "Most major inputs were available, with some estimated details."
    return "Most key inputs were directly observed for this property."


def _confidence_summary(result: AssessmentResult) -> str:
    tier = str(result.confidence_tier or "preliminary")
    restriction = str(result.use_restriction or "review_required")
    if tier == "high":
        return "High confidence: most key inputs were directly observed from prepared data and confirmed property details."
    if tier == "moderate":
        return "Moderate confidence: major location context is available, with some estimated property details."
    if tier == "low":
        return "Lower confidence: significant assumptions or fallback data were required for one or more score components."
    return (
        "Preliminary confidence: this result is useful for screening and planning, but more data is needed before high-stakes decisions."
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
    elif result.top_recommended_actions:
        # Compose from existing top recommendations when prioritized action rows
        # are unavailable in legacy payloads.
        base = [
            HomeownerPrioritizedAction(
                action=str(action or "").strip(),
                impact_level="medium",
                effort_level="medium",
                estimated_cost_band="medium",
                timeline="this_season",
                priority=index + 1,
            )
            for index, action in enumerate(list(result.top_recommended_actions or [])[:5])
            if str(action or "").strip()
        ]
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
    include_optional_calibration_metadata: bool = False,
    simulation_overview: dict[str, Any] | None = None,
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
    homeowner_confidence = (
        (result.homeowner_summary or {}).get("confidence_summary")
        if isinstance(result.homeowner_summary, dict)
        else {}
    )
    homeowner_confidence = homeowner_confidence if isinstance(homeowner_confidence, dict) else {}
    homeowner_trust_summary = (
        (result.homeowner_summary or {}).get("trust_summary")
        if isinstance(result.homeowner_summary, dict)
        else {}
    )
    homeowner_trust_summary = homeowner_trust_summary if isinstance(homeowner_trust_summary, dict) else {}
    property_confidence_summary = (
        homeowner_trust_summary.get("property_confidence_summary")
        if isinstance(homeowner_trust_summary.get("property_confidence_summary"), dict)
        else {}
    )
    if not isinstance(property_confidence_summary, dict):
        property_confidence_summary = {}
    if not property_confidence_summary:
        if hasattr(result.property_confidence_summary, "model_dump"):
            property_confidence_summary = dict(result.property_confidence_summary.model_dump())
        elif isinstance(result.property_confidence_summary, dict):
            property_confidence_summary = dict(result.property_confidence_summary)
    specificity_summary = _specificity_summary(result, homeowner_trust_summary)
    tone_level = _apply_specificity_tone_guardrail(_select_tone_level(result), specificity_summary)
    nearby_home_comparison_safeguard_triggered = bool(
        homeowner_trust_summary.get("nearby_home_comparison_safeguard_triggered")
    )
    nearby_home_comparison_safeguard_message = str(
        homeowner_trust_summary.get("nearby_home_comparison_safeguard_message")
        or "This estimate is not precise enough to compare adjacent homes."
    ).strip()

    raw_driver_candidates = list(result.top_risk_drivers or [])
    if not raw_driver_candidates and result.top_risk_drivers_detailed:
        raw_driver_candidates = [
            str(row.explanation or row.factor or "").strip()
            for row in list(result.top_risk_drivers_detailed or [])
            if str(row.explanation or row.factor or "").strip()
        ]
    if not raw_driver_candidates and str(result.explanation_summary or "").strip():
        raw_driver_candidates = [str(result.explanation_summary).strip()]

    key_risk_drivers = [_plain_driver(row) for row in raw_driver_candidates]
    key_risk_drivers = [row for row in key_risk_drivers if row][:6]
    if nearby_home_comparison_safeguard_triggered:
        key_risk_drivers = [
            row
            for row in key_risk_drivers
            if not any(
                token in str(row).lower()
                for token in ("near-structure", "0-5 ft", "5-30 ft", "defensible space")
            )
        ]
        key_risk_drivers = [nearby_home_comparison_safeguard_message] + key_risk_drivers
        key_risk_drivers = list(dict.fromkeys(key_risk_drivers))[:6]
    prioritized_actions = _prioritized_actions(result)

    defensible_space_analysis = result.defensible_space_analysis if isinstance(result.defensible_space_analysis, dict) else {}
    zone_findings = _zone_findings(defensible_space_analysis)
    if nearby_home_comparison_safeguard_triggered:
        zone_findings = []
    ds_limitations = list(result.defensible_space_limitations_summary or [])
    if nearby_home_comparison_safeguard_triggered:
        ds_limitations = list(
            dict.fromkeys([nearby_home_comparison_safeguard_message] + ds_limitations)
        )[:6]

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
    homeowner_improve_your_result = (
        (result.homeowner_summary or {}).get("improve_your_result")
        if isinstance(result.homeowner_summary, dict)
        else {}
    )
    homeowner_improve_your_result = homeowner_improve_your_result if isinstance(homeowner_improve_your_result, dict) else {}
    limitations_snapshot = _build_homeowner_limitations_snapshot(
        result=result,
        homeowner_improve_your_result=homeowner_improve_your_result,
    )
    confidence_headline = str(homeowner_confidence.get("headline") or "").strip() or _confidence_summary(result)
    confidence_limitations = [
        str(item).strip()
        for item in list(homeowner_confidence.get("why_confidence_is_limited") or [])
        if str(item).strip()
    ]
    if confidence_limitations:
        combined_limitations = list(dict.fromkeys(confidence_limitations + combined_limitations))[:6]

    confidence_and_limitations: dict[str, object] = {
        "confidence_score": result.confidence_score,
        "confidence_tier": result.confidence_tier,
        "use_restriction": result.use_restriction,
        "confidence_statement": confidence_headline,
        "trust_summary": homeowner_trust_summary,
        "improve_your_result": homeowner_improve_your_result,
        "observed_data": list(result.confidence_summary.observed_data or []),
        "estimated_data": list(result.confidence_summary.estimated_data or []),
        "missing_data": list(result.confidence_summary.missing_data or []),
        "accuracy_improvements": list(result.confidence_summary.accuracy_improvements or []),
        "limitations": combined_limitations,
        "decision_support_disclaimer": (
            "This report is decision-support guidance based on prepared geospatial data and provided inputs; "
            "it is not a prediction or guarantee of insurer underwriting approval, insurability, or wildfire safety."
        ),
        "property_confidence_summary": property_confidence_summary,
    }
    if include_professional_debug_metadata:
        confidence_and_limitations["fallback_decisions"] = [
            _dump_value(row) for row in list((result.assessment_diagnostics.fallback_decisions or []))[:8]
        ]

    professional_debug_metadata: dict[str, Any] | None = None
    if include_professional_debug_metadata:
        professional_debug_metadata = {
            "layer_coverage_audit": [_dump_value(row) for row in result.layer_coverage_audit],
            "coverage_summary": _dump_value(result.coverage_summary),
            "score_evidence_ledger": _dump_value(result.score_evidence_ledger),
            "evidence_quality_summary": _dump_value(result.evidence_quality_summary),
            "assessment_diagnostics": _dump_value(result.assessment_diagnostics),
            "public_outcome_governance_note": {
                "summary": (
                    "Public-outcome validation/calibration is internal governance metadata and is not "
                    "foregrounded in homeowner guidance."
                ),
                "docs": [
                    "docs/public_outcome_validation.md",
                    "docs/public_outcome_calibration.md",
                ],
            },
        }

    all_mitigation_actions = _mitigation_actions(result, tone_level=tone_level)
    top_recommended_actions = all_mitigation_actions[:3]
    top_risk_drivers = _summarize_top_risk_drivers(key_risk_drivers, tone_level=tone_level)
    prioritized_actions_summary = _summarize_prioritized_actions(
        prioritized_actions,
        tone_level=tone_level,
        limit=3,
    )
    ranked_actions, most_impactful_actions = _rank_mitigation_actions(prioritized_actions_summary)
    what_to_do_first = ranked_actions[0] if ranked_actions else (prioritized_actions_summary[0] if prioritized_actions_summary else {})
    headline_risk_summary = _headline_risk_summary(
        result,
        result.overall_wildfire_risk if result.overall_wildfire_risk is not None else result.wildfire_risk_score,
        tone_level=tone_level,
    )
    overall_risk_score = (
        result.overall_wildfire_risk
        if result.overall_wildfire_risk is not None
        else result.wildfire_risk_score
    )
    overall_wildfire_risk = {
        "label": "Overall wildfire risk",
        "risk_band": _risk_band(overall_risk_score),
        "score": overall_risk_score,
        "headline": headline_risk_summary,
    }
    limitations_notice = _with_low_specificity_limitation(
        _limitations_notice(result, combined_limitations),
        specificity_summary,
    )
    first_screen = _build_first_screen_payload(
        specificity_summary=specificity_summary,
        property_confidence_summary=property_confidence_summary,
        overall_wildfire_risk=overall_wildfire_risk,
        headline_risk_summary=headline_risk_summary,
        top_risk_drivers=top_risk_drivers,
        ranked_actions=ranked_actions,
        what_to_do_first=what_to_do_first,
        limitations_notice=limitations_notice,
    )
    calibration_metadata_requested = bool(
        include_professional_debug_metadata or include_optional_calibration_metadata
    )
    optional_calibration = (
        _optional_public_outcome_calibration_metadata(result)
        if calibration_metadata_requested
        else None
    )
    calibration_prob = (
        result.calibrated_damage_likelihood if calibration_metadata_requested else None
    )
    empirical_damage_prob = (
        result.empirical_damage_likelihood_proxy if calibration_metadata_requested else None
    )
    empirical_loss_prob = (
        result.empirical_loss_likelihood_proxy if calibration_metadata_requested else None
    )
    try:
        homeowner_explanations = generate_homeowner_explanations(result)
    except Exception:
        homeowner_explanations = _template_homeowner_explanations(result)

    fallback_insurability = derive_insurability_status(
        wildfire_risk_score=overall_risk_score,
        home_hardening_readiness=home_hardening_score,
        confidence_tier=result.confidence_tier,
        assessment_specificity_tier=result.assessment_specificity_tier,
        top_near_structure_risk_drivers=result.top_near_structure_risk_drivers,
        top_risk_drivers=result.top_risk_drivers,
        defensible_space_analysis=result.defensible_space_analysis,
        defensible_space_limitations_summary=result.defensible_space_limitations_summary,
        readiness_blockers=result.readiness_blockers,
        scoring_status=result.scoring_status,
    )
    insurability_status = str(
        result.insurability_status or fallback_insurability.insurability_status
    ).strip()
    insurability_status_reasons = [
        str(v).strip()
        for v in (
            list(result.insurability_status_reasons or [])
            if list(result.insurability_status_reasons or [])
            else list(fallback_insurability.insurability_status_reasons)
        )
        if str(v).strip()
    ][:3]
    insurability_status_methodology_note = str(
        result.insurability_status_methodology_note
        or fallback_insurability.insurability_status_methodology_note
    ).strip()

    homeowner_focus_summary = _build_homeowner_focus_summary(
        insurability_status=insurability_status,
        insurability_status_reasons=insurability_status_reasons,
        insurability_status_methodology_note=insurability_status_methodology_note,
        confidence_tier=result.confidence_tier,
        headline_risk_summary=headline_risk_summary,
        top_risk_drivers=top_risk_drivers,
        ranked_actions=ranked_actions,
        limitations=combined_limitations,
        limitations_snapshot=limitations_snapshot,
        tone_level=tone_level,
        simulation_overview=simulation_overview,
    )
    internal_calibration_debug = _build_internal_calibration_debug_block(
        result,
        calibration_metadata_requested=calibration_metadata_requested,
        optional_calibration=optional_calibration,
        calibration_prob=calibration_prob,
        empirical_damage_prob=empirical_damage_prob,
        empirical_loss_prob=empirical_loss_prob,
    )
    advanced_details: dict[str, object] = {
        "default_visibility": "collapsed",
        "calibration_and_diagnostics": internal_calibration_debug,
        "sections": {
            "calibration_and_diagnostics": "Technical subscores, diagnostics, evidence ledgers, compatibility outputs, and calibration metadata.",
            "professional_debug_metadata": "Additional internal diagnostics and audit metadata when explicitly requested.",
        },
        "legacy_aliases": {
            "internal_calibration_debug": "advanced_details.calibration_and_diagnostics",
        },
        "professional_debug_metadata_available": bool(professional_debug_metadata),
        "professional_debug_metadata": (
            professional_debug_metadata if professional_debug_metadata is not None else {}
        ),
        "note": (
            "Advanced details are retained for model-calibration and internal review workflows; "
            "homeowner-first guidance appears in summary sections above."
        ),
    }

    return HomeownerReport(
        assessment_id=result.assessment_id,
        generated_at=report_generated_at,
        insurability_status=insurability_status,
        insurability_status_reasons=insurability_status_reasons,
        insurability_status_methodology_note=insurability_status_methodology_note,
        homeowner_focus_summary=homeowner_focus_summary,
        internal_calibration_debug=internal_calibration_debug,
        advanced_details=advanced_details,
        first_screen=first_screen,
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
            "insurance_readiness_score": result.insurance_readiness_score,
            "insurance_readiness_band": _home_hardening_band(result.insurance_readiness_score),
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "legacy_insurance_readiness_note": (
                "Insurance-readiness compatibility output is retained as a heuristic screening reference; "
                "it is not an insurer underwriting decision."
            ),
            "confidence_score": result.confidence_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
            "public_outcome_calibration_note": (
                "Optional/additive metadata only (secondary/internal); deterministic risk drivers, actions, specificity, "
                "and limitations are the primary homeowner guidance."
                if calibration_metadata_requested and optional_calibration is not None
                else (
                    "No optional public-outcome calibration metadata is currently applied."
                    if calibration_metadata_requested
                    else (
                        "Optional public-outcome calibration metadata is hidden in homeowner view by default; "
                        "request include_optional_calibration_metadata=true for internal review context."
                    )
                )
            ),
            # Compatibility fields retained as secondary metadata; not used in homeowner first-screen guidance.
            "calibration_applied": result.calibration_applied,
            "calibration_status": (
                result.calibration_status
                if calibration_metadata_requested
                else "hidden_in_homeowner_view"
            ),
            "calibrated_damage_likelihood": calibration_prob,
            "empirical_damage_likelihood_proxy": empirical_damage_prob,
            "empirical_loss_likelihood_proxy": empirical_loss_prob,
        },
        key_risk_drivers=key_risk_drivers,
        top_risk_drivers_detailed=(
            list(result.top_risk_drivers_detailed or [])
            if include_professional_debug_metadata
            else []
        ),
        defensible_space_summary={
            "summary": (
                nearby_home_comparison_safeguard_message
                if nearby_home_comparison_safeguard_triggered
                else (
                    defensible_space_analysis.get("summary")
                    or "Defensible-space analysis was unavailable for this property."
                )
            ),
            "basis_geometry_type": defensible_space_analysis.get("basis_geometry_type") or "unknown",
            "basis_quality": defensible_space_analysis.get("basis_quality") or "unknown",
            "zone_findings": zone_findings,
            "top_near_structure_risk_drivers": (
                []
                if nearby_home_comparison_safeguard_triggered
                else list(result.top_near_structure_risk_drivers or [])
            ),
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
            "note": (
                "Insurance readiness outputs are optional heuristic references for screening context, "
                "not insurer-specific underwriting decisions."
            ),
        },
        confidence_summary=result.confidence_summary,
        confidence_and_limitations=confidence_and_limitations,
        specificity_summary=specificity_summary,
        metadata={
            "model_version": result.model_version,
            "product_version": result.product_version,
            "api_version": result.api_version,
            "homeowner_explanations": homeowner_explanations,
            "model_governance": _dump_value(result.model_governance),
            "region_data_version": result.region_data_version,
            "data_bundle_version": result.data_bundle_version,
            "calibration_version": (
                result.calibration_version if calibration_metadata_requested else None
            ),
            "calibration_method": (
                result.calibration_method if calibration_metadata_requested else None
            ),
            "calibration_limitations": (
                list(result.calibration_limitations or []) if calibration_metadata_requested else []
            ),
            "calibration_scope_warning": (
                result.calibration_scope_warning if calibration_metadata_requested else None
            ),
            "optional_public_outcome_calibration": optional_calibration,
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
        summary = "High confidence: key property and location inputs were directly observed."
    elif tier in {"moderate", "medium"}:
        summary = "Moderate confidence: key inputs were mostly available, with some estimated details."
    else:
        summary = "Lower confidence: several details were estimated or missing, so treat this as a screening assessment."

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
    include_optional_calibration_metadata: bool = False,
) -> dict[str, object] | bytes:
    report = build_homeowner_report(
        result,
        include_professional_debug_metadata=False,
        include_optional_calibration_metadata=include_optional_calibration_metadata,
    )
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
    trust_summary = report.confidence_and_limitations.get("trust_summary")
    trust_summary = trust_summary if isinstance(trust_summary, dict) else {}
    improve_your_result = report.confidence_and_limitations.get("improve_your_result")
    improve_your_result = improve_your_result if isinstance(improve_your_result, dict) else {}
    optional_calibration = (
        (report.metadata or {}).get("optional_public_outcome_calibration")
        if isinstance((report.metadata or {}).get("optional_public_outcome_calibration"), dict)
        else None
    )
    first_screen = report.first_screen if isinstance(report.first_screen, dict) and report.first_screen else _build_first_screen_payload(
        specificity_summary=(_dump_value(report.specificity_summary) if report.specificity_summary else {}),
        property_confidence_summary=(
            trust_summary.get("property_confidence_summary")
            if isinstance(trust_summary.get("property_confidence_summary"), dict)
            else {}
        ),
        overall_wildfire_risk={
            "label": "Overall wildfire risk",
            "risk_band": _risk_band(_to_float((report.score_summary or {}).get("overall_wildfire_risk"))),
            "score": _to_float((report.score_summary or {}).get("overall_wildfire_risk")),
            "headline": str(report.headline_risk_summary or ""),
        },
        headline_risk_summary=str(report.headline_risk_summary or ""),
        top_risk_drivers=list(report.top_risk_drivers or [])[:3],
        ranked_actions=ranked_actions[:3],
        what_to_do_first=what_to_do_first,
        limitations_notice=limitations_notice,
    )

    return {
        "assessment_id": report.assessment_id,
        "generated_at": report.generated_at,
        "address": str(property_summary.get("address") or ""),
        "homeowner_focus_summary": (
            dict(report.homeowner_focus_summary)
            if isinstance(report.homeowner_focus_summary, dict)
            else {}
        ),
        "first_screen": first_screen,
        "headline_risk_summary": str(report.headline_risk_summary or ""),
        "top_risk_drivers": list(report.top_risk_drivers or [])[:3],
        "prioritized_actions": ranked_actions[:3],
        "most_impactful_actions": most_impactful,
        "what_to_do_first": what_to_do_first,
        "confidence_summary": confidence_summary,
        "specificity_summary": (_dump_value(report.specificity_summary) if report.specificity_summary else {}),
        "property_confidence_summary": (
            trust_summary.get("property_confidence_summary")
            if isinstance(trust_summary.get("property_confidence_summary"), dict)
            else {}
        ),
        "trust_summary": trust_summary,
        "improve_your_result": improve_your_result,
        "limitations_notice": limitations_notice,
        "optional_public_outcome_calibration": (
            optional_calibration if include_optional_calibration_metadata else None
        ),
        "disclaimer": (
            "This report supports homeowner planning and conversations with contractors, agents, "
            "or insurers. It is a heuristic screening assessment and not a prediction or guarantee of "
            "insurer underwriting approval, wildfire outcomes, or insurability."
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


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_line(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [str(p).strip() for p in parts if str(p).strip()]


def _sentence_key(text: Any) -> str:
    normalized = _normalize_line(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _dedupe_sentences(sentences: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        key = _sentence_key(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def _align_action_explanation(action_name: str, explanation: str) -> str:
    action = _normalize_line(action_name).rstrip(".")
    text = _normalize_line(explanation)
    if not action or not text:
        return text
    action_tokens = [token for token in _explanation_lookup_key(action).split() if len(token) >= 4][:3]
    explanation_key = _explanation_lookup_key(text)
    if action_tokens and not any(token in explanation_key for token in action_tokens):
        if len(text) > 1:
            text = text[0].lower() + text[1:]
        else:
            text = text.lower()
        return f"{action}: {text}"
    return text


def _sanitize_explanation_text(
    text: Any,
    *,
    fallback: str,
    max_sentences: int = 2,
    max_chars: int = 240,
) -> str:
    candidate = _normalize_line(text)
    if not candidate:
        candidate = _normalize_line(fallback)
    candidate = candidate.replace("…", "...").replace("–", "-")
    candidate = re.sub(r"[^\x20-\x7E]", " ", candidate)
    candidate = re.sub(r"\b(?:re|st|te|ti|ri|ly)\?(?=\s|$)", "", candidate, flags=re.I)
    candidate = re.sub(r"\b[a-z]{1,2}\?(?=\s|$)", "", candidate, flags=re.I)
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    # Keep phrasing directional; avoid hard percentages in homeowner narrative.
    candidate = re.sub(r"\b\d+(?:\.\d+)?\s*%", "a meaningful amount", candidate, flags=re.I)
    candidate = re.sub(
        r"\b\d+(?:\.\d+)?\s*percent(?:age)?\b",
        "a meaningful amount",
        candidate,
        flags=re.I,
    )
    candidate = re.sub(
        r"\bmay be contributing to risk,?\s*but some details (?:were|are) estimated\b",
        "may increase wildfire exposure",
        candidate,
        flags=re.I,
    )
    candidate = re.sub(r"\bthis may help reduce risk\b", "This could reduce wildfire exposure", candidate, flags=re.I)
    candidate = re.sub(r"\bthis can help reduce risk\b", "This can reduce wildfire exposure", candidate, flags=re.I)
    candidate = re.sub(r"\bthis helps reduce risk\b", "This reduces wildfire exposure", candidate, flags=re.I)
    candidate = re.sub(r"\bmay help reduce\b", "could reduce", candidate, flags=re.I)
    candidate = re.sub(r"\bappears to help reduce\b", "can reduce", candidate, flags=re.I)
    jargon_map = {
        "topography": "terrain",
        "submodel": "risk factor",
        "mitigation": "risk reduction action",
        "ignition vulnerability": "home ignition risk",
    }
    lowered = candidate.lower()
    for src, dst in jargon_map.items():
        if src in lowered:
            lowered = lowered.replace(src, dst)
    if lowered:
        candidate = lowered[0].upper() + lowered[1:]
    sentences = _dedupe_sentences(_split_sentences(candidate))[: max(1, int(max_sentences))]
    merged = " ".join(sentences).strip()
    if not merged:
        merged = _normalize_line(fallback)
    if len(merged) > max_chars:
        merged = merged[: max_chars - 1].rstrip() + "…"
    if merged and merged[-1] not in ".!?":
        merged += "."
    return merged


def _sanitize_explanation_list(
    values: Any,
    *,
    fallback: list[str],
    limit: int = 3,
    max_sentences: int = 2,
    max_chars: int = 240,
) -> list[str]:
    raw_items = list(values) if isinstance(values, list) else []
    out: list[str] = []
    seen: set[str] = set()
    desired = max(1, int(limit))
    attempts = 0
    max_attempts = max(desired * 6, len(raw_items) + len(fallback) + 6)
    while len(out) < desired and attempts < max_attempts:
        raw = raw_items[attempts] if attempts < len(raw_items) else ""
        if fallback:
            fallback_text = fallback[min(attempts, len(fallback) - 1)]
        else:
            fallback_text = "Use this guidance as directional planning support."
        cleaned = _sanitize_explanation_text(
            raw,
            fallback=fallback_text,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )
        key = _sentence_key(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            out.append(cleaned)
        attempts += 1

    if len(out) < desired:
        for fallback_text in fallback:
            cleaned = _sanitize_explanation_text(
                "",
                fallback=fallback_text,
                max_sentences=max_sentences,
                max_chars=max_chars,
            )
            key = _sentence_key(cleaned)
            if cleaned and key and key not in seen:
                seen.add(key)
                out.append(cleaned)
            if len(out) >= desired:
                break
    return out[:desired]


def _extract_assessment_actions(assessment: AssessmentResult, *, limit: int = 3) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for row in list(assessment.prioritized_mitigation_actions or [])[: max(1, int(limit))]:
        action = _normalize_line(getattr(row, "action", ""))
        if not action:
            continue
        explanation = _normalize_line(
            getattr(row, "why_this_matters", "")
            or getattr(row, "explanation", "")
        )
        rows.append({"action": action, "explanation": explanation})

    if not rows:
        for action in list(assessment.top_recommended_actions or [])[: max(1, int(limit))]:
            text = _normalize_line(action)
            if text:
                rows.append({"action": text, "explanation": ""})

    if not rows:
        for row in list(assessment.mitigation_plan or [])[: max(1, int(limit))]:
            action = _normalize_line(getattr(row, "title", "") or getattr(row, "action", ""))
            if not action:
                continue
            explanation = _normalize_line(getattr(row, "reason", "") or getattr(row, "impact_statement", ""))
            rows.append({"action": action, "explanation": explanation})

    return rows[: max(1, int(limit))]


def _explanation_lookup_key(value: Any) -> str:
    normalized = _normalize_line(value).lower()
    if not normalized:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _build_headline_summary_template(risk_band: str, tone_profile: str) -> str:
    band = str(risk_band or "Unavailable").strip().lower()
    if tone_profile == "direct":
        return f"Verdict: this property shows {band} wildfire risk in this assessment."
    if tone_profile == "balanced":
        return f"Verdict: this property appears {band} risk based on available property and area data."
    return f"Verdict: this property may have {band} wildfire risk, and some details were estimated."


def _build_driver_explanation_template(driver_text: str, tone_profile: str) -> str:
    base = _normalize_line(_plain_driver(driver_text) or driver_text).rstrip(".")
    if not base:
        base = "Nearby conditions"
    if tone_profile == "direct":
        return f"{base} is one of the main reasons this property scores higher for wildfire exposure."
    if tone_profile == "balanced":
        return f"{base} is an important reason this property scores higher for wildfire exposure."
    return f"{base} may be increasing wildfire exposure."


def _build_action_explanation_template(action: str, why: str, tone_profile: str) -> str:
    action_text = _normalize_line(action).rstrip(".")
    if not action_text:
        action_text = "This action"
    reason = _normalize_line(why).rstrip(".")
    if tone_profile == "direct":
        sentence = f"{action_text} helps lower ignition pressure around the home."
    elif tone_profile == "balanced":
        sentence = f"{action_text} can lower ignition pressure around the home."
    else:
        sentence = f"{action_text} could lower ignition pressure around the home."
    if reason:
        sentence = f"{sentence.rstrip('.')} by addressing {reason}."
    return sentence


def _build_limitation_summary_template(tone_profile: str) -> str:
    if tone_profile == "direct":
        return "Most key inputs were directly observed for this report."
    if tone_profile == "balanced":
        return "Some key inputs were directly observed, and some were estimated."
    return "Several details were estimated or missing, so treat this as a screening assessment."


def _extract_action_explanation_candidates(values: Any) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in list(values) if isinstance(values, list) else []:
        if isinstance(row, dict):
            candidates.append(
                {
                    "action": _normalize_line(row.get("action") or row.get("title") or row.get("label") or ""),
                    "explanation": _normalize_line(
                        row.get("explanation")
                        or row.get("text")
                        or row.get("summary")
                        or row.get("reason")
                        or ""
                    ),
                }
            )
        else:
            candidates.append({"action": "", "explanation": _normalize_line(row)})
    return candidates


def _template_homeowner_explanations(assessment: AssessmentResult) -> dict[str, object]:
    confidence_tier = str(assessment.confidence_tier or "preliminary").strip().lower()
    missing_count = len(list(assessment.confidence_summary.missing_data or []))
    estimated_count = len(list(assessment.confidence_summary.estimated_data or []))
    fallback_count = len(list(assessment.confidence_summary.fallback_assumptions or []))
    fallback_count += len(list((assessment.assessment_diagnostics.fallback_decisions or [])))
    tone_profile = _pdf_tone_profile(
        confidence_tier,
        missing_count=missing_count,
        fallback_count=fallback_count,
        estimated_count=estimated_count,
    )

    risk_score = (
        assessment.overall_wildfire_risk
        if assessment.overall_wildfire_risk is not None
        else assessment.wildfire_risk_score
    )
    risk_band = _section_band_label(_risk_band(_to_float(risk_score)))
    headline = _build_headline_summary_template(risk_band, tone_profile)

    driver_rows = [str(v).strip() for v in list(assessment.top_risk_drivers or []) if str(v).strip()]
    if not driver_rows and assessment.top_risk_drivers_detailed:
        driver_rows = [
            _normalize_line(getattr(row, "explanation", "") or getattr(row, "factor", ""))
            for row in list(assessment.top_risk_drivers_detailed or [])
            if _normalize_line(getattr(row, "explanation", "") or getattr(row, "factor", ""))
        ]
    if not driver_rows:
        driver_rows = ["Nearby vegetation and terrain conditions are increasing wildfire exposure."]
    driver_rows = driver_rows[:3]

    driver_explanations: list[str] = []
    for row in driver_rows:
        driver_explanations.append(_build_driver_explanation_template(row, tone_profile))

    action_rows = _extract_assessment_actions(assessment, limit=3)
    action_explanations: list[str] = []
    action_explanations_by_action: dict[str, str] = {}
    for row in action_rows:
        action = row.get("action") or "This action"
        why = _normalize_line(row.get("explanation") or "")
        sentence = _build_action_explanation_template(action, why, tone_profile)
        action_explanations.append(sentence)
        action_key = _explanation_lookup_key(action)
        if action_key:
            action_explanations_by_action[action_key] = sentence
    if not action_explanations:
        action_explanations = ["Prioritized actions could help reduce wildfire risk and improve home hardening readiness."]

    confidence_line = _build_limitation_summary_template(tone_profile)

    return {
        "source": "template",
        "headline_summary": headline,
        "risk_driver_explanations": driver_explanations[:3],
        "recommended_action_explanations": action_explanations[:3],
        "recommended_action_explanations_by_action": action_explanations_by_action,
        "confidence_limitations_explanation": confidence_line,
    }


def _generate_homeowner_explanations_with_llm(
    payload: dict[str, Any],
    *,
    llm_client: Any | None = None,
) -> dict[str, Any] | None:
    client = llm_client
    if client is None:
        api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None
        try:
            client = OpenAI(api_key=api_key)
        except Exception:
            return None

    model = str(os.getenv("WF_HOMEOWNER_EXPLANATION_MODEL") or "gpt-4o-mini").strip()
    if not model:
        model = "gpt-4o-mini"

    system_prompt = (
        "You write homeowner-facing wildfire explanations. Return strict JSON with keys: "
        "headline_summary (string), "
        "risk_driver_explanations (array of up to 3 objects with keys driver and explanation), "
        "recommended_action_explanations (array of up to 3 objects with keys action and explanation), "
        "confidence_limitations_explanation (string). "
        "Each explanation must be a single short sentence in plain language, no technical jargon, no percentages, "
        "and no repeated phrases such as 'may help reduce risk' across multiple items. "
        "Tie each action explanation to its matching action text."
    )
    user_prompt = (
        "Generate concise homeowner explanations from this assessment summary JSON.\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )

    text_output = ""
    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                temperature=0.2,
                max_output_tokens=700,
                input=[
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                ],
            )
            text_output = str(getattr(response, "output_text", "") or "")
        elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choices = getattr(response, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                text_output = str(getattr(message, "content", "") or "")
    except Exception:
        return None

    if not text_output:
        return None
    match = re.search(r"\{.*\}", text_output, flags=re.S)
    payload_text = match.group(0) if match else text_output.strip()
    try:
        parsed = json.loads(payload_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def generate_homeowner_explanations(
    assessment: AssessmentResult,
    *,
    llm_client: Any | None = None,
) -> dict[str, object]:
    template = _template_homeowner_explanations(assessment)
    top_actions = _extract_assessment_actions(assessment, limit=3)

    llm_payload = {
        "address": assessment.address,
        "confidence_tier": str(assessment.confidence_tier or "preliminary"),
        "missing_data": list(assessment.confidence_summary.missing_data or [])[:6],
        "fallback_assumptions": list(assessment.confidence_summary.fallback_assumptions or [])[:6],
        "risk_band": _section_band_label(
            _risk_band(
                _to_float(
                    assessment.overall_wildfire_risk
                    if assessment.overall_wildfire_risk is not None
                    else assessment.wildfire_risk_score
                )
            )
        ),
        "top_risk_drivers": list(assessment.top_risk_drivers or [])[:3],
        "top_actions": top_actions,
    }

    llm_generated = _generate_homeowner_explanations_with_llm(llm_payload, llm_client=llm_client)
    source = "template"
    llm_provided_action_map = False
    merged = dict(template)
    if isinstance(llm_generated, dict):
        source = "llm"
        llm_provided_action_map = "recommended_action_explanations_by_action" in llm_generated
        for key in (
            "headline_summary",
            "risk_driver_explanations",
            "recommended_action_explanations",
            "recommended_action_explanations_by_action",
            "confidence_limitations_explanation",
        ):
            if key in llm_generated:
                merged[key] = llm_generated.get(key)

    headline = _sanitize_explanation_text(
        merged.get("headline_summary"),
        fallback=str(template.get("headline_summary") or ""),
        max_sentences=1,
        max_chars=220,
    )

    raw_driver_values = merged.get("risk_driver_explanations")
    if isinstance(raw_driver_values, list):
        driver_inputs = [
            row.get("explanation") if isinstance(row, dict) else row
            for row in raw_driver_values
        ]
    else:
        driver_inputs = raw_driver_values
    driver_explanations = _sanitize_explanation_list(
        driver_inputs,
        fallback=[str(v) for v in list(template.get("risk_driver_explanations") or [])],
        limit=3,
        max_sentences=1,
        max_chars=220,
    )

    fallback_action_explanations = [str(v) for v in list(template.get("recommended_action_explanations") or [])]
    raw_action_candidates = _extract_action_explanation_candidates(merged.get("recommended_action_explanations"))
    action_explanations = _sanitize_explanation_list(
        [row.get("explanation") or "" for row in raw_action_candidates],
        fallback=fallback_action_explanations,
        limit=3,
        max_sentences=1,
        max_chars=220,
    )
    action_explanations_by_action: dict[str, str] = {}

    for idx, explanation_text in enumerate(action_explanations):
        action_hint = _normalize_line(raw_action_candidates[idx].get("action") or "") if idx < len(raw_action_candidates) else ""
        aligned_explanation = _align_action_explanation(action_hint, explanation_text) if action_hint else explanation_text
        action_explanations[idx] = aligned_explanation
        action_key = _explanation_lookup_key(action_hint)
        if action_key:
            action_explanations_by_action[action_key] = aligned_explanation

    raw_action_map = merged.get("recommended_action_explanations_by_action")
    if isinstance(raw_action_map, dict) and (source == "template" or llm_provided_action_map):
        for action_name, explanation in raw_action_map.items():
            action_key = _explanation_lookup_key(action_name)
            if not action_key:
                continue
            sanitized = _sanitize_explanation_text(
                explanation,
                fallback=action_explanations_by_action.get(action_key, ""),
                max_sentences=1,
                max_chars=220,
            )
            if sanitized:
                action_explanations_by_action[action_key] = _align_action_explanation(action_name, sanitized)

    confidence_line = _sanitize_explanation_text(
        merged.get("confidence_limitations_explanation"),
        fallback=str(template.get("confidence_limitations_explanation") or ""),
        max_sentences=1,
        max_chars=220,
    )

    return {
        "source": source,
        "headline_summary": headline,
        "risk_driver_explanations": driver_explanations,
        "recommended_action_explanations": action_explanations,
        "recommended_action_explanations_by_action": action_explanations_by_action,
        "confidence_limitations_explanation": confidence_line,
    }


@dataclass(frozen=True)
class _PdfEntry:
    text: str = ""
    style: str = "body"


_PDF_TEXT_STYLES: dict[str, dict[str, float | str]] = {
    "product_name": {"font": "F2", "size": 10.8, "leading": 14.0, "indent": 52.0, "width": 88.0, "gray": 0.22},
    "title": {"font": "F2", "size": 22.0, "leading": 30.0, "indent": 52.0, "width": 68.0, "gray": 0.0},
    "meta": {"font": "F1", "size": 10.2, "leading": 14.6, "indent": 52.0, "width": 96.0, "gray": 0.0},
    "section": {"font": "F2", "size": 14.2, "leading": 22.0, "indent": 52.0, "width": 88.0, "gray": 0.0},
    "summary_card_header": {
        "font": "F2",
        "size": 11.0,
        "leading": 16.0,
        "indent": 60.0,
        "width": 82.0,
        "gray": 0.0,
        "box_fill": 0.95,
        "box_stroke": 0.80,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 20.0,
    },
    "summary_card_value": {
        "font": "F2",
        "size": 18.0,
        "leading": 23.0,
        "indent": 60.0,
        "width": 82.0,
        "gray": 0.0,
        "box_fill": 0.95,
        "box_stroke": 0.80,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 24.0,
    },
    "summary_card_body": {
        "font": "F1",
        "size": 10.4,
        "leading": 14.8,
        "indent": 60.0,
        "width": 82.0,
        "gray": 0.0,
        "box_fill": 0.95,
        "box_stroke": 0.80,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 18.0,
    },
    "evidence_box_header": {
        "font": "F2",
        "size": 10.8,
        "leading": 15.0,
        "indent": 60.0,
        "width": 82.0,
        "gray": 0.0,
        "box_fill": 0.97,
        "box_stroke": 0.84,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 18.0,
    },
    "evidence_box_body": {
        "font": "F1",
        "size": 10.0,
        "leading": 14.0,
        "indent": 60.0,
        "width": 82.0,
        "gray": 0.0,
        "box_fill": 0.97,
        "box_stroke": 0.84,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 17.0,
    },
    "map_canvas": {
        "font": "F1",
        "size": 9.0,
        "leading": 154.0,
        "indent": 52.0,
        "width": 90.0,
        "gray": 0.0,
        "panel_left": 52.0,
        "panel_right": 560.0,
        "panel_height": 146.0,
    },
    "risk_label": {"font": "F2", "size": 11.0, "leading": 15.0, "indent": 52.0, "width": 90.0, "gray": 0.08},
    "risk_level": {"font": "F2", "size": 21.5, "leading": 25.0, "indent": 52.0, "width": 84.0, "gray": 0.0},
    "body": {"font": "F1", "size": 10.4, "leading": 14.6, "indent": 52.0, "width": 94.0, "gray": 0.0},
    "bullet": {"font": "F1", "size": 10.4, "leading": 14.6, "indent": 64.0, "width": 86.0, "gray": 0.0},
    "callout_label": {
        "font": "F2",
        "size": 11.0,
        "leading": 16.0,
        "indent": 58.0,
        "width": 86.0,
        "gray": 0.0,
        "box_fill": 0.96,
        "box_stroke": 0.82,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 20.0,
    },
    "callout_action": {
        "font": "F2",
        "size": 13.5,
        "leading": 19.0,
        "indent": 60.0,
        "width": 80.0,
        "gray": 0.0,
        "box_fill": 0.96,
        "box_stroke": 0.82,
        "box_left": 52.0,
        "box_right": 560.0,
        "box_height": 24.0,
    },
    "technical_header": {"font": "F2", "size": 11.0, "leading": 16.2, "indent": 52.0, "width": 88.0, "gray": 0.22},
    "technical_body": {"font": "F1", "size": 9.2, "leading": 12.8, "indent": 60.0, "width": 84.0, "gray": 0.34},
    # Reusable style aliases for consistency with report styling nomenclature.
    "TitleStyle": {"font": "F2", "size": 22.0, "leading": 30.0, "indent": 52.0, "width": 68.0, "gray": 0.0},
    "SectionHeaderStyle": {"font": "F2", "size": 14.2, "leading": 22.0, "indent": 52.0, "width": 88.0, "gray": 0.0},
    "BodyStyle": {"font": "F1", "size": 10.4, "leading": 14.6, "indent": 52.0, "width": 94.0, "gray": 0.0},
    "HighlightStyle": {"font": "F2", "size": 13.5, "leading": 19.0, "indent": 60.0, "width": 80.0, "gray": 0.0},
}

_PDF_SPACER_HEIGHTS: dict[str, float] = {
    "spacer_sm": 6.0,
    "spacer_md": 9.0,
    "spacer_lg": 13.0,
    "page_break": 0.0,
}

_PDF_PAGE_LEFT = 52.0
_PDF_PAGE_RIGHT = 560.0
_PDF_START_Y = 770.0
_PDF_BOTTOM_Y = 50.0


def _normalize_line(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip()


def _section_band_label(raw_band: str | None) -> str:
    band = str(raw_band or "").strip().lower()
    mapping = {
        "high": "High",
        "moderate": "Moderate",
        "lower": "Low",
        "low": "Low",
        "unavailable": "Unavailable",
    }
    return mapping.get(band, "Unavailable")


def _safe_effort_label(value: Any) -> str:
    raw = str(value or "medium").strip().lower()
    if raw not in {"low", "medium", "high"}:
        raw = "medium"
    return raw.capitalize()


def _driver_fallback_explanation(driver_text: str) -> str:
    lowered = str(driver_text or "").lower()
    if "vegetation" in lowered or "fuel" in lowered:
        return "Nearby vegetation and fuels can increase flame and ember pressure near the home."
    if "slope" in lowered or "topograph" in lowered:
        return "Steeper terrain can accelerate fire spread and increase exposure."
    if "ember" in lowered:
        return "Wind-blown embers can ignite vulnerable areas around the structure."
    if "flame" in lowered:
        return "Direct flame contact risk is elevated in the immediate structure zone."
    if "historic" in lowered or "fire history" in lowered:
        return "Past wildfire activity nearby indicates repeated local fire potential."
    if "defensible" in lowered:
        return "Limited defensible space can increase structure ignition pathways."
    return "This factor contributes to wildfire exposure at this property."


def _driver_explanation(driver_text: str, detailed_rows: list[dict[str, Any]]) -> str:
    driver_norm = _normalize_line(driver_text).lower()
    if not driver_norm:
        return _driver_fallback_explanation(driver_text)

    for row in detailed_rows:
        factor = _normalize_line(row.get("factor") or "").replace("_", " ").lower()
        explanation = _normalize_line(row.get("explanation") or "")
        if not explanation:
            continue
        if factor and (factor in driver_norm or any(token in driver_norm for token in factor.split() if len(token) > 4)):
            return explanation

    return _driver_fallback_explanation(driver_text)


def _select_recommended_actions(report: HomeownerReport, *, limit: int = 5) -> list[dict[str, Any]]:
    ranked = [row for row in list(report.ranked_actions or []) if isinstance(row, dict)]
    prioritized = [row for row in list(report.prioritized_actions or []) if isinstance(row, dict)]

    candidates = ranked or prioritized
    deduped: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for row in candidates:
        name = _normalize_line(row.get("action") or row.get("title") or "")
        if not name:
            continue
        key = name.lower()
        if key in seen_actions:
            continue
        seen_actions.add(key)
        deduped.append(dict(row))
        if len(deduped) >= limit:
            break
    return deduped


def _improvement_prefix(confidence_tier: str) -> str:
    tier = str(confidence_tier or "").strip().lower()
    if tier in {"direct", "high"}:
        return "is likely to"
    if tier in {"balanced", "moderate", "medium"}:
        return "can help"
    return "could help"


def _pdf_tone_profile(
    confidence_tier: str,
    *,
    missing_count: int,
    fallback_count: int,
    estimated_count: int,
) -> str:
    tier = str(confidence_tier or "").strip().lower()
    if tier == "high" and missing_count == 0 and fallback_count == 0 and estimated_count <= 2:
        return "direct"
    if tier in {"low", "preliminary"} or missing_count >= 2 or fallback_count >= 1:
        return "cautious"
    return "balanced"


def _summary_tone_line(tone_profile: str) -> str:
    if tone_profile == "direct":
        return "This summary is based mostly on directly observed property details."
    if tone_profile == "balanced":
        return "This summary combines observed property details with some estimated inputs."
    return "This summary uses several estimated or missing details and should be treated as screening guidance."


def _driver_line_with_tone(driver: str, tone_profile: str) -> str:
    text = _normalize_line(driver)
    if tone_profile != "cautious":
        return text
    lowered = text.lower()
    if any(token in lowered for token in ("may ", "appears", "estimated")):
        return text
    return f"{text} (this may increase wildfire exposure)."


def _driver_explanation_prefix(tone_profile: str) -> str:
    if tone_profile == "direct":
        return "Why it matters"
    if tone_profile == "balanced":
        return "Why it likely matters"
    return "Potential impact"


def _action_rationale_prefix(tone_profile: str) -> str:
    if tone_profile == "direct":
        return "This reduces wildfire exposure"
    if tone_profile == "balanced":
        return "This can reduce wildfire exposure"
    return "This could reduce wildfire exposure"


def _limitations_tone_line(tone_profile: str) -> str:
    if tone_profile == "direct":
        return "Confidence is stronger because key property details were directly observed."
    if tone_profile == "balanced":
        return "Confidence is moderate because some details were estimated."
    return "Confidence is lower because several details were estimated or missing."


def _how_this_could_improve_lines(
    recommended_actions: list[dict[str, Any]],
    *,
    confidence_tier: str,
) -> list[str]:
    lines: list[str] = []
    phrase = _improvement_prefix(confidence_tier)
    for row in list(recommended_actions or [])[:3]:
        action_name = _normalize_line(row.get("action") or row.get("title") or "")
        if not action_name:
            continue
        direction_line = (
            f"{action_name} {phrase} lower wildfire exposure and address wildfire-risk factors insurers often review."
        )
        lines.append(direction_line)
        why = _normalize_line(
            row.get("why_this_matters")
            or row.get("explanation")
            or row.get("why_it_matters")
            or ""
        )
        if why:
            lines.append(f"Impact rationale: {why}")
    return lines


def _property_context_lines(report: HomeownerReport) -> list[str]:
    context_lines: list[str] = []
    defensible_space = report.defensible_space_summary if isinstance(report.defensible_space_summary, dict) else {}
    ds_summary = _normalize_line(defensible_space.get("summary") or "")
    if ds_summary:
        context_lines.append(f"Vegetation: {ds_summary}")
    else:
        context_lines.append("Vegetation: No detailed vegetation summary was available for this run.")

    detailed_rows = [
        _dump_value(row) if not isinstance(row, dict) else row
        for row in list(report.top_risk_drivers_detailed or [])
    ]
    detailed_rows = [row for row in detailed_rows if isinstance(row, dict)]

    slope_line = ""
    history_line = ""
    for row in detailed_rows:
        factor = _normalize_line(row.get("factor") or "").lower()
        explanation = _normalize_line(row.get("explanation") or "")
        if not slope_line and ("slope" in factor or "topograph" in factor):
            slope_line = explanation or "Slope and terrain influence local fire spread conditions."
        if not history_line and ("historic" in factor or "fire_history" in factor):
            history_line = explanation or "Historic fire activity contributes to regional context."
        if slope_line and history_line:
            break

    if not slope_line:
        slope_line = "Slope and terrain were considered in this assessment's hazard context."
    if not history_line:
        history_line = "Historic fire context did not emerge as a primary differentiator in this run."

    context_lines.append(f"Slope: {slope_line}")
    context_lines.append(f"Fire history: {history_line}")
    return context_lines


def _friendly_evidence_label(value: Any) -> str:
    raw = _normalize_line(value)
    if not raw:
        return ""
    normalized_key = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    mapping = {
        "roof_type": "Roof material details",
        "roof_material": "Roof material details",
        "siding_type": "Siding material details",
        "vent_type": "Vent protection details",
        "defensible_space_ft": "Defensible-space distance",
        "structure_geometry": "Structure geometry",
        "parcel_geometry": "Parcel boundary geometry",
        "building_footprint": "Building footprint geometry",
        "near_structure_fuels": "Near-structure vegetation and fuels",
        "address_point": "Verified address point",
    }
    if normalized_key in mapping:
        return mapping[normalized_key]
    if re.search(r"\b(fallback|diagnostic|proxy|decision)\b", raw, flags=re.I):
        return ""
    cleaned = re.sub(r"[_\-]+", " ", raw).strip()
    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else ""
    return cleaned


def _select_evidence_items(values: Any, *, limit: int = 5) -> list[str]:
    rows = list(values or []) if isinstance(values, list) else []
    cleaned: list[tuple[int, str]] = []
    seen: set[str] = set()
    keywords = (
        "roof",
        "vent",
        "siding",
        "defensible",
        "vegetation",
        "structure",
        "building",
        "parcel",
        "address",
        "slope",
        "fuel",
        "fire history",
    )
    for idx, row in enumerate(rows):
        label = _friendly_evidence_label(row)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        priority = 10
        lowered = key
        for p_idx, token in enumerate(keywords):
            if token in lowered:
                priority = p_idx
                break
        cleaned.append((priority * 100 + idx, label))
    cleaned.sort(key=lambda item: item[0])
    return [row[1] for row in cleaned[: max(1, int(limit))]]


def _add_wrapped(entries: list[_PdfEntry], text: Any, *, style: str, prefix: str = "") -> None:
    normalized = _normalize_line(text)
    if not normalized:
        return
    style_meta = _PDF_TEXT_STYLES.get(style) or _PDF_TEXT_STYLES["body"]
    width = int(style_meta.get("width", 94.0))
    for line in _wrap_text_line(normalized, width=width, prefix=prefix):
        entries.append(_PdfEntry(line, style))


def _build_report_entries(report: HomeownerReport) -> list[_PdfEntry]:
    entries: list[_PdfEntry] = []
    header = report.report_header or {}
    property_summary = report.property_summary or {}
    score_summary = report.score_summary or {}
    metadata = report.metadata or {}
    confidence = report.confidence_and_limitations or {}
    homeowner_focus = (
        report.homeowner_focus_summary
        if isinstance(report.homeowner_focus_summary, dict)
        else {}
    )
    first_screen = report.first_screen if isinstance(report.first_screen, dict) else {}
    specificity_summary = _dump_value(report.specificity_summary)
    specificity_summary = specificity_summary if isinstance(specificity_summary, dict) else {}
    explanation_block = (
        metadata.get("homeowner_explanations")
        if isinstance(metadata.get("homeowner_explanations"), dict)
        else {}
    )
    explanation_block = explanation_block if isinstance(explanation_block, dict) else {}

    fs_specificity = (
        first_screen.get("specificity_summary")
        if isinstance(first_screen.get("specificity_summary"), dict)
        else specificity_summary
    )
    fs_specificity = fs_specificity if isinstance(fs_specificity, dict) else {}
    fs_overall = (
        first_screen.get("overall_wildfire_risk")
        if isinstance(first_screen.get("overall_wildfire_risk"), dict)
        else {}
    )
    fs_overall = fs_overall if isinstance(fs_overall, dict) else {}

    fs_headline = _normalize_line(
        fs_overall.get("headline")
        or first_screen.get("headline_risk_summary")
        or homeowner_focus.get("one_sentence_summary")
        or report.headline_risk_summary
        or ""
    )
    focus_status_label = _normalize_line(
        homeowner_focus.get("insurability_status")
        or report.insurability_status
        or homeowner_focus.get("status_label")
        or ""
    )
    focus_status_reasons = [
        _normalize_line(v)
        for v in list(
            homeowner_focus.get("insurability_status_reasons")
            if isinstance(homeowner_focus.get("insurability_status_reasons"), list)
            else report.insurability_status_reasons
        )
        if _normalize_line(v)
    ][:3]
    focus_status_note = _normalize_line(
        homeowner_focus.get("insurability_status_methodology_note")
        or report.insurability_status_methodology_note
        or ""
    )
    focus_confidence_summary = _normalize_line(
        homeowner_focus.get("confidence_limitations_summary") or ""
    )
    focus_before_after_obj = (
        homeowner_focus.get("before_after_summary")
        if isinstance(homeowner_focus.get("before_after_summary"), dict)
        else {}
    )
    focus_before_after_obj = focus_before_after_obj if isinstance(focus_before_after_obj, dict) else {}
    focus_before_after_summary = _normalize_line(
        focus_before_after_obj.get("summary")
    )
    focus_before_after_status_before = _normalize_line(
        focus_before_after_obj.get("current_insurability_status") or ""
    )
    focus_before_after_status_after = _normalize_line(
        focus_before_after_obj.get("projected_insurability_status") or ""
    )
    focus_before_after_actions = [
        row
        for row in list(focus_before_after_obj.get("top_actions_driving_change") or [])
        if isinstance(row, dict) and _normalize_line(row.get("action") or "")
    ][:3]
    limitations_snapshot = (
        homeowner_focus.get("limitations_snapshot")
        if isinstance(homeowner_focus.get("limitations_snapshot"), dict)
        else {}
    )
    limitations_snapshot = limitations_snapshot if isinstance(limitations_snapshot, dict) else {}
    snapshot_observed = _dedupe_nonempty_lines(list(limitations_snapshot.get("directly_observed") or []), limit=3)
    snapshot_estimated = _dedupe_nonempty_lines(
        list(limitations_snapshot.get("estimated_or_inferred") or [])
        + list(limitations_snapshot.get("missing_or_unknown") or []),
        limit=3,
    )
    snapshot_improve = _dedupe_nonempty_lines(list(limitations_snapshot.get("inputs_to_improve") or []), limit=3)
    snapshot_headline = _normalize_line(limitations_snapshot.get("headline") or "")
    risk_band_label = _section_band_label(
        fs_overall.get("risk_band") or score_summary.get("wildfire_risk_band")
    )
    risk_score = _to_float(
        fs_overall.get("score")
        if fs_overall.get("score") is not None
        else score_summary.get("overall_wildfire_risk", score_summary.get("wildfire_risk_score"))
    )
    risk_score_suffix = f" ({risk_score:.1f}/100)" if risk_score is not None else ""

    readiness_available = bool(score_summary.get("home_hardening_readiness_score_available"))
    readiness_score = _to_float(score_summary.get("home_hardening_readiness"))
    readiness_band = _section_band_label(score_summary.get("home_hardening_readiness_band"))
    if readiness_available and readiness_score is not None:
        readiness_line = f"{readiness_score:.1f}/100 ({readiness_band})"
    else:
        readiness_line = "Not available"

    top_drivers = first_screen.get("top_risk_drivers") if isinstance(first_screen.get("top_risk_drivers"), list) else list(report.top_risk_drivers or [])
    top_drivers = [str(v).strip() for v in top_drivers if str(v).strip()][:3]
    detailed_rows = [
        _dump_value(row) if not isinstance(row, dict) else row
        for row in list(report.top_risk_drivers_detailed or [])
    ]
    detailed_rows = [row for row in detailed_rows if isinstance(row, dict)]

    first_action = (
        first_screen.get("what_to_do_first")
        if isinstance(first_screen.get("what_to_do_first"), dict)
        else (dict(report.what_to_do_first or {}) if isinstance(report.what_to_do_first, dict) else {})
    )
    first_action_text = _normalize_line(first_action.get("action") or "Review the top recommended action.")
    first_action_why = _normalize_line(
        first_action.get("why_this_matters")
        or first_action.get("explanation")
        or first_action.get("why_it_matters")
        or ""
    )

    recommended_actions = _select_recommended_actions(report, limit=5)

    confidence_tier = str(score_summary.get("confidence_tier") or confidence.get("confidence_tier") or "unknown").strip().lower()
    confidence_score = _to_float(score_summary.get("confidence_score") or confidence.get("confidence_score"))
    confidence_score_text = f"{confidence_score:.1f}/100" if confidence_score is not None else "n/a"
    observed_count = len(list(confidence.get("observed_data") or []))
    estimated_count = len(list(confidence.get("estimated_data") or []))
    missing_count = len(list(confidence.get("missing_data") or []))
    confidence_summary = _dump_value(report.confidence_summary)
    confidence_summary = confidence_summary if isinstance(confidence_summary, dict) else {}
    fallback_assumptions_count = len(
        [v for v in list(confidence_summary.get("fallback_assumptions") or []) if _normalize_line(v)]
    )
    fallback_decisions_count = len(
        [v for v in list(confidence.get("fallback_decisions") or []) if v]
    ) if isinstance(confidence.get("fallback_decisions"), list) else 0
    fallback_count = fallback_assumptions_count + fallback_decisions_count
    tone_profile = _pdf_tone_profile(
        confidence_tier,
        missing_count=missing_count,
        fallback_count=fallback_count,
        estimated_count=estimated_count,
    )
    specificity_headline = _normalize_line(fs_specificity.get("headline") or "Regional estimate")
    specificity_tier = _normalize_line(fs_specificity.get("specificity_tier") or "regional_estimate")
    specificity_meaning = _normalize_line(fs_specificity.get("what_this_means") or "")
    comparison_allowed = bool(fs_specificity.get("comparison_allowed"))

    limitations: list[str] = []
    limitations.extend([_normalize_line(v) for v in list(confidence.get("limitations") or []) if _normalize_line(v)])
    notice = _normalize_line(first_screen.get("limitations_note") or report.limitations_notice or "")
    if notice:
        limitations.append(notice)
    deduped_limitations: list[str] = []
    seen_limitations: set[str] = set()
    for row in limitations:
        key = row.lower()
        if key in seen_limitations:
            continue
        seen_limitations.add(key)
        deduped_limitations.append(row)
    deduped_limitations = deduped_limitations[:4]

    explanation_headline = _normalize_line(
        explanation_block.get("headline_summary")
        or fs_headline
        or "No summary sentence was available."
    )
    raw_driver_explanations = list(explanation_block.get("risk_driver_explanations") or [])
    explanation_drivers = [
        _normalize_line(row.get("explanation") if isinstance(row, dict) else row)
        for row in raw_driver_explanations
        if _normalize_line(row.get("explanation") if isinstance(row, dict) else row)
    ][:3]
    action_explanation_rows = _extract_action_explanation_candidates(
        explanation_block.get("recommended_action_explanations")
    )
    explanation_actions_by_action: dict[str, str] = {}
    for row in action_explanation_rows[:3]:
        key = _explanation_lookup_key(row.get("action") or "")
        text = _normalize_line(row.get("explanation") or "")
        if key and text:
            explanation_actions_by_action[key] = text
    raw_action_map = explanation_block.get("recommended_action_explanations_by_action")
    if isinstance(raw_action_map, dict):
        for action_name, explanation in raw_action_map.items():
            key = _explanation_lookup_key(action_name)
            text = _normalize_line(explanation)
            if key and text:
                explanation_actions_by_action[key] = text
    explanation_confidence = _normalize_line(
        explanation_block.get("confidence_limitations_explanation") or ""
    )

    confidence_level_label = confidence_tier.capitalize() if confidence_tier else "Unknown"
    defensible_space_summary = report.defensible_space_summary if isinstance(report.defensible_space_summary, dict) else {}
    basis_geometry_type = _normalize_line(defensible_space_summary.get("basis_geometry_type") or "unknown").lower()
    analysis_status = _normalize_line(defensible_space_summary.get("analysis_status") or "unknown").lower()
    zone_findings = defensible_space_summary.get("zone_findings")
    zone_findings = zone_findings if isinstance(zone_findings, list) else []

    property_address = _normalize_line(property_summary.get("address") or "Unknown address")
    property_lat = _to_float(property_summary.get("latitude"))
    property_lon = _to_float(property_summary.get("longitude"))
    resolved_region_id = _normalize_line(property_summary.get("resolved_region_id") or "")

    observed_items = _select_evidence_items(confidence.get("observed_data"), limit=5)
    missing_or_estimated_items = _select_evidence_items(
        list(confidence.get("estimated_data") or []) + list(confidence.get("missing_data") or []),
        limit=5,
    )
    observed_joined = " ".join(observed_items).lower()
    has_building_footprint = (
        ("footprint" in basis_geometry_type)
        or ("structure" in basis_geometry_type)
        or ("building footprint" in observed_joined)
        or ("structure geometry" in observed_joined)
    )
    has_parcel_outline = (
        ("parcel" in basis_geometry_type)
        or ("parcel" in observed_joined)
        or ("parcel boundary geometry" in observed_joined)
    )
    has_rings = bool(zone_findings) or any("zone_" in str(k) for k in list((defensible_space_summary.get("zone_findings") or [])))
    wildfire_overlay_signal = " ".join(top_drivers).lower()
    has_wildfire_overlay = any(
        token in wildfire_overlay_signal
        for token in ("historic fire", "fire history", "wildland", "fuel", "vegetation", "ember")
    )
    geometry_is_approximate = (
        specificity_tier in {"regional_estimate", "insufficient_data"}
        or basis_geometry_type in {"point_proxy", "unknown", "none"}
        or analysis_status in {"partial", "unavailable"}
    )
    map_config = {
        "has_building_footprint": has_building_footprint,
        "has_parcel_outline": has_parcel_outline,
        "has_rings": has_rings,
        "has_wildfire_overlay": has_wildfire_overlay,
        "geometry_is_approximate": geometry_is_approximate,
    }
    internal_debug = report.internal_calibration_debug if isinstance(report.internal_calibration_debug, dict) else {}
    debug_subscores = internal_debug.get("subscores") if isinstance(internal_debug.get("subscores"), dict) else {}

    def _fmt_score(value: Any) -> str:
        num = _to_float(value)
        return f"{num:.1f}" if num is not None else "n/a"

    # 1) Header / Title
    _add_wrapped(entries, "WildfireRisk Advisor", style="product_name")
    _add_wrapped(entries, "Wildfire Risk Report", style="TitleStyle")
    _add_wrapped(entries, f"Property Address: {property_address}", style="meta")
    if property_lat is not None and property_lon is not None:
        _add_wrapped(entries, f"Location context: {property_lat:.5f}, {property_lon:.5f}", style="meta")
    if resolved_region_id:
        _add_wrapped(entries, f"Prepared region context: {resolved_region_id}", style="meta")
    _add_wrapped(entries, f"Date generated: {_normalize_line(header.get('assessment_generated_at') or report.generated_at)}", style="meta")
    entries.append(_PdfEntry(style="spacer_lg"))

    # 2) PAGE 1: Decision-first homeowner snapshot
    _add_wrapped(entries, "Homeowner Decision Snapshot", style="SectionHeaderStyle")
    _add_wrapped(entries, "Insurance screening status (heuristic)", style="summary_card_header")
    _add_wrapped(entries, focus_status_label or "Status unavailable", style="summary_card_value")
    _add_wrapped(entries, f"One-sentence summary: {explanation_headline}", style="summary_card_body")
    _add_wrapped(entries, f"Wildfire risk level: {risk_band_label}{risk_score_suffix}", style="summary_card_body")
    if focus_status_reasons:
        _add_wrapped(entries, "Why this screening status:", style="summary_card_body")
        for reason in focus_status_reasons[:3]:
            _add_wrapped(entries, reason, style="summary_card_body", prefix="- ")
    if focus_status_note:
        _add_wrapped(entries, focus_status_note, style="summary_card_body")
    entries.append(_PdfEntry(style="spacer_md"))

    # 3) PAGE 1: Top risk drivers
    _add_wrapped(entries, "Top 3 Risk Drivers", style="SectionHeaderStyle")
    if top_drivers:
        for idx, driver in enumerate(top_drivers[:3]):
            driver_text = _normalize_line(_plain_driver(driver) or driver)
            _add_wrapped(entries, driver_text, style="bullet", prefix="- ")
            driver_explanation = (
                explanation_drivers[idx]
                if idx < len(explanation_drivers) and explanation_drivers[idx]
                else _build_driver_explanation_template(driver, tone_profile)
            )
            _add_wrapped(entries, driver_explanation, style="body", prefix="  ")
    else:
        _add_wrapped(entries, "Top risk drivers were not available for this assessment.", style="body")
    entries.append(_PdfEntry(style="spacer_md"))

    # 4) PAGE 1: Top recommended actions
    _add_wrapped(entries, "Top 3 Recommended Actions", style="SectionHeaderStyle")
    if recommended_actions:
        for row in recommended_actions[:3]:
            action_name = _normalize_line(row.get("action") or row.get("title") or "Recommended action")
            why = _normalize_line(row.get("why_this_matters") or row.get("explanation") or row.get("why_it_matters") or "")
            action_key = _explanation_lookup_key(action_name)
            mapped_explanation = explanation_actions_by_action.get(action_key, "")
            _add_wrapped(entries, action_name, style="bullet", prefix="- ")
            if mapped_explanation:
                _add_wrapped(entries, mapped_explanation, style="body", prefix="  ")
            elif why:
                _add_wrapped(entries, _build_action_explanation_template(action_name, why, tone_profile), style="body", prefix="  ")
    else:
        _add_wrapped(entries, "No prioritized actions were generated for this run.", style="body")
    entries.append(_PdfEntry(style="spacer_md"))

    # 5) PAGE 1: Before vs after callout (if simulation/improvement exists)
    _add_wrapped(entries, "Before vs After Snapshot", style="SectionHeaderStyle")
    if focus_before_after_summary or focus_before_after_status_before or focus_before_after_status_after:
        if focus_before_after_status_before and focus_before_after_status_after:
            _add_wrapped(
                entries,
                f"Screening status shift: {focus_before_after_status_before} -> {focus_before_after_status_after}",
                style="callout_label",
            )
        _add_wrapped(
            entries,
            focus_before_after_summary or "No score change summary was available for the latest simulation/improvement run.",
            style="callout_action",
        )
        if focus_before_after_actions:
            _add_wrapped(entries, "Actions driving change:", style="body")
            for row in focus_before_after_actions:
                action_label = _normalize_line(row.get("action") or "")
                action_why = _normalize_line(row.get("why_this_matters") or "")
                if action_label:
                    _add_wrapped(
                        entries,
                        f"{action_label}{': ' + action_why if action_why else ''}",
                        style="bullet",
                        prefix="- ",
                    )
    else:
        _add_wrapped(entries, "No simulation or improvement run is available yet for before/after comparison.", style="body")
    entries.append(_PdfEntry(style="spacer_md"))

    # 6) PAGE 1: Short confidence/limitations note
    _add_wrapped(entries, "Confidence Note", style="SectionHeaderStyle")
    _add_wrapped(
        entries,
        f"Confidence level: {confidence_level_label} ({confidence_score_text}).",
        style="body",
    )
    _add_wrapped(
        entries,
        focus_confidence_summary or explanation_confidence or _limitations_tone_line(tone_profile),
        style="body",
    )
    if deduped_limitations:
        _add_wrapped(entries, deduped_limitations[0], style="body", prefix="- ")
    entries.append(_PdfEntry(style="page_break"))

    # 7) PAGE 2+: Risk breakdown / subscores
    _add_wrapped(entries, "Risk Breakdown and Subscores", style="SectionHeaderStyle")
    _add_wrapped(entries, f"Overall wildfire risk: {_fmt_score(score_summary.get('overall_wildfire_risk'))}/100 ({risk_band_label}).", style="body")
    _add_wrapped(entries, f"Wildfire risk score: {_fmt_score(score_summary.get('wildfire_risk_score'))}/100.", style="body")
    _add_wrapped(entries, f"Home hardening readiness: {_fmt_score(score_summary.get('home_hardening_readiness'))}/100 ({readiness_band}).", style="body")
    _add_wrapped(entries, f"Insurance readiness (compatibility heuristic): {_fmt_score(score_summary.get('insurance_readiness_score'))}/100.", style="body")
    _add_wrapped(entries, f"Confidence score: {_fmt_score(score_summary.get('confidence_score'))}/100 ({confidence_level_label}).", style="body")
    if debug_subscores:
        _add_wrapped(entries, "Subscores", style="technical_header")
        _add_wrapped(entries, f"- Site hazard score: {_fmt_score(debug_subscores.get('site_hazard_score'))}/100", style="technical_body")
        _add_wrapped(entries, f"- Home ignition vulnerability score: {_fmt_score(debug_subscores.get('home_ignition_vulnerability_score'))}/100", style="technical_body")
        _add_wrapped(entries, f"- Home hardening readiness score: {_fmt_score(debug_subscores.get('home_hardening_readiness'))}/100", style="technical_body")
        _add_wrapped(entries, f"- Insurance readiness score (compatibility heuristic): {_fmt_score(debug_subscores.get('insurance_readiness_score'))}/100", style="technical_body")
    entries.append(_PdfEntry(style="spacer_md"))

    # 8) PAGE 2+: Property context and map
    _add_wrapped(entries, "Property Context and Map", style="SectionHeaderStyle")
    for line in _property_context_lines(report):
        _add_wrapped(entries, line, style="body", prefix="- ")
    _add_wrapped(entries, "Local Map View", style="SectionHeaderStyle")
    _add_wrapped(entries, "Map centered on this report location:", style="body")
    _add_wrapped(entries, property_address, style="body")
    entries.append(_PdfEntry(text=json.dumps(map_config, ensure_ascii=True), style="map_canvas"))
    _add_wrapped(
        entries,
        "Ring legend: 0-5 ft immediate zone, 5-30 ft extended zone, 30-100 ft surrounding zone.",
        style="body",
    )
    if geometry_is_approximate:
        _add_wrapped(
            entries,
            "Map note: geometry is approximate, so the footprint and rings represent best-available location context.",
            style="body",
        )
    else:
        _add_wrapped(
            entries,
            "Map note: geometry is anchored to property-level footprint and parcel context where available.",
            style="body",
        )
    entries.append(_PdfEntry(style="spacer_md"))

    # 9) PAGE 2+: Mitigation details (expanded)
    _add_wrapped(entries, "Mitigation Details", style="SectionHeaderStyle")
    _add_wrapped(entries, "Most Important Next Step", style="callout_label")
    _add_wrapped(entries, first_action_text, style="callout_action")
    if first_action_why:
        _add_wrapped(entries, first_action_why, style="body", prefix="  ")
    for row in recommended_actions[:5]:
        action_name = _normalize_line(row.get("action") or row.get("title") or "Recommended action")
        why = _normalize_line(row.get("why_this_matters") or row.get("explanation") or row.get("why_it_matters") or "")
        action_key = _explanation_lookup_key(action_name)
        mapped_explanation = explanation_actions_by_action.get(action_key, "")
        effort = _safe_effort_label(row.get("effort_level") or row.get("estimated_cost_band"))
        _add_wrapped(entries, action_name, style="bullet", prefix="- ")
        if mapped_explanation:
            _add_wrapped(entries, mapped_explanation, style="body", prefix="  ")
        elif why:
            _add_wrapped(entries, _build_action_explanation_template(action_name, why, tone_profile), style="body", prefix="  ")
        _add_wrapped(entries, f"Effort level: {effort}", style="body", prefix="  ")
    improvement_lines = _how_this_could_improve_lines(
        recommended_actions,
        confidence_tier=tone_profile,
    )
    if improvement_lines:
        _add_wrapped(entries, "If You Complete These Actions", style="SectionHeaderStyle")
        for line in improvement_lines:
            if line.startswith("Impact rationale:"):
                _add_wrapped(entries, line, style="body", prefix="  ")
            else:
                _add_wrapped(entries, line, style="bullet", prefix="- ")
    entries.append(_PdfEntry(style="spacer_md"))

    # 10) PAGE 2+: Detailed confidence and limitations
    _add_wrapped(entries, "Confidence and Limitations", style="SectionHeaderStyle")
    _add_wrapped(entries, f"Data completeness: observed {observed_count}, estimated {estimated_count}, missing {missing_count}.", style="body")
    _add_wrapped(entries, f"Fallback usage: {fallback_count} fallback assumptions or decisions.", style="body")
    _add_wrapped(entries, explanation_confidence or _limitations_tone_line(tone_profile), style="body")
    _add_wrapped(entries, f"Specificity: {specificity_headline} (tier: {specificity_tier}).", style="body")
    _add_wrapped(entries, f"Confidence tier: {confidence_level_label} ({confidence_score_text}).", style="body")
    if specificity_meaning:
        _add_wrapped(entries, specificity_meaning, style="body", prefix="  ")
    if not comparison_allowed:
        _add_wrapped(entries, "Nearby-home comparisons should be treated cautiously for this estimate.", style="body", prefix="  ")
    if missing_count > 0:
        _add_wrapped(entries, "Limited-data case: some fields were estimated or missing, so this result is advisory.", style="body")
    if deduped_limitations:
        for limitation in deduped_limitations:
            _add_wrapped(entries, limitation, style="bullet", prefix="- ")
    if snapshot_headline:
        _add_wrapped(entries, snapshot_headline, style="body")
    _add_wrapped(
        entries,
        "Directly observed: " + ("; ".join(snapshot_observed) if snapshot_observed else "Limited direct observations."),
        style="body",
    )
    _add_wrapped(
        entries,
        "Estimated or inferred: " + ("; ".join(snapshot_estimated) if snapshot_estimated else "No major estimated signals were listed."),
        style="body",
    )
    _add_wrapped(
        entries,
        "To improve this result: " + ("; ".join(snapshot_improve) if snapshot_improve else "Add more confirmed home details."),
        style="body",
    )
    _add_wrapped(entries, "Observed for this report", style="evidence_box_header")
    if observed_items:
        for item in observed_items:
            _add_wrapped(entries, item, style="evidence_box_body", prefix="- ")
    else:
        _add_wrapped(entries, "No major observed details were listed.", style="evidence_box_body", prefix="- ")
    _add_wrapped(entries, "Missing or estimated", style="evidence_box_header")
    if missing_or_estimated_items:
        for item in missing_or_estimated_items:
            _add_wrapped(entries, item, style="evidence_box_body", prefix="- ")
    else:
        _add_wrapped(entries, "No major missing or estimated details were identified.", style="evidence_box_body", prefix="- ")
    if specificity_tier in {"regional_estimate", "insufficient_data"}:
        _add_wrapped(
            entries,
            "Why this may be broader: this report relied more on regional conditions because some property details were unavailable.",
            style="evidence_box_body",
        )
    entries.append(_PdfEntry(style="spacer_md"))

    # 13) Advanced Details (de-emphasized)
    _add_wrapped(entries, "Advanced Details (Optional / Technical Details)", style="SectionHeaderStyle")
    _add_wrapped(
        entries,
        "This section is included for advanced users and internal calibration review.",
        style="technical_body",
    )
    _add_wrapped(entries, "Factor breakdown", style="technical_header")
    if detailed_rows:
        for row in detailed_rows[:8]:
            factor = _normalize_line(row.get("factor") or "factor").replace("_", " ")
            impact = _normalize_line(row.get("impact") or "unknown")
            contribution = _to_float(row.get("relative_contribution_pct"))
            explanation = _normalize_line(row.get("explanation") or "")
            contribution_part = f", contribution={contribution:.1f}%" if contribution is not None else ""
            _add_wrapped(entries, f"- {factor} (impact={impact}{contribution_part})", style="technical_body")
            if explanation:
                _add_wrapped(entries, f"  {explanation}", style="technical_body")
    else:
        _add_wrapped(entries, "- No detailed factor breakdown was available.", style="technical_body")

    _add_wrapped(entries, "Diagnostics", style="technical_header")
    _add_wrapped(entries, f"- Assessment ID: {report.assessment_id}", style="technical_body")
    _add_wrapped(entries, f"- Region: {_normalize_line(property_summary.get('resolved_region_id') or 'unknown')}", style="technical_body")
    _add_wrapped(entries, f"- Confidence tier: {confidence_tier}", style="technical_body")
    _add_wrapped(entries, f"- Risk band: {risk_band_label.lower()}", style="technical_body")

    _add_wrapped(entries, "Evidence summaries", style="technical_header")
    observed_preview = ", ".join([_normalize_line(v) for v in list(confidence.get("observed_data") or [])[:4] if _normalize_line(v)])
    estimated_preview = ", ".join([_normalize_line(v) for v in list(confidence.get("estimated_data") or [])[:4] if _normalize_line(v)])
    missing_preview = ", ".join([_normalize_line(v) for v in list(confidence.get("missing_data") or [])[:4] if _normalize_line(v)])
    _add_wrapped(entries, f"- Observed evidence: {observed_preview or 'none listed'}", style="technical_body")
    _add_wrapped(entries, f"- Estimated evidence: {estimated_preview or 'none listed'}", style="technical_body")
    _add_wrapped(entries, f"- Missing evidence: {missing_preview or 'none listed'}", style="technical_body")
    if report.professional_debug_metadata:
        _add_wrapped(entries, "- Additional internal diagnostics are available in the raw JSON export.", style="technical_body")

    return entries


def _entry_height(entry: _PdfEntry) -> float:
    if entry.style in _PDF_SPACER_HEIGHTS:
        return _PDF_SPACER_HEIGHTS[entry.style]
    style = _PDF_TEXT_STYLES.get(entry.style) or _PDF_TEXT_STYLES["body"]
    leading = float(style.get("leading", 14.0))
    box_height = float(style.get("box_height", 0.0))
    if box_height > 0.0:
        return max(leading, box_height + 2.0)
    return leading


def _paginate_entries(entries: list[_PdfEntry], *, start_y: float = _PDF_START_Y, bottom_y: float = _PDF_BOTTOM_Y) -> list[list[_PdfEntry]]:
    pages: list[list[_PdfEntry]] = []
    page: list[_PdfEntry] = []
    remaining = max(80.0, start_y - bottom_y)

    for entry in entries:
        if entry.style == "page_break":
            if page:
                pages.append(page)
                page = []
            remaining = max(80.0, start_y - bottom_y)
            continue
        height = _entry_height(entry)
        if page and height > remaining:
            pages.append(page)
            page = []
            remaining = max(80.0, start_y - bottom_y)
        page.append(entry)
        remaining -= height

    if page:
        pages.append(page)
    if not pages:
        pages.append([_PdfEntry("Wildfire Risk Report", "title")])
    return pages


def _circle_path_commands(cx: float, cy: float, radius: float) -> list[str]:
    r = max(0.5, float(radius))
    k = 0.5522847498
    ox = r * k
    oy = r * k
    return [
        f"{cx + r:.2f} {cy:.2f} m",
        f"{cx + r:.2f} {cy + oy:.2f} {cx + ox:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c",
        f"{cx - ox:.2f} {cy + r:.2f} {cx - r:.2f} {cy + oy:.2f} {cx - r:.2f} {cy:.2f} c",
        f"{cx - r:.2f} {cy - oy:.2f} {cx - ox:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c",
        f"{cx + ox:.2f} {cy - r:.2f} {cx + r:.2f} {cy - oy:.2f} {cx + r:.2f} {cy:.2f} c",
    ]


def _draw_local_map_panel_commands(
    config: dict[str, Any],
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> list[str]:
    commands: list[str] = []
    width = max(80.0, right - left)
    height = max(64.0, top - bottom)
    center_x = left + (width * 0.42)
    center_y = bottom + (height * 0.5)

    has_building = bool(config.get("has_building_footprint"))
    has_parcel = bool(config.get("has_parcel_outline"))
    has_rings = bool(config.get("has_rings"))
    has_wildfire_overlay = bool(config.get("has_wildfire_overlay"))
    approximate = bool(config.get("geometry_is_approximate"))

    commands.extend(
        [
            "q",
            "0.98 g",
            "0.82 G",
            "0.8 w",
            f"{left:.2f} {bottom:.2f} {width:.2f} {height:.2f} re B",
            "Q",
        ]
    )

    # Subtle terrain-like guide lines.
    for idx in range(1, 4):
        y = bottom + (height * (idx / 4.0))
        commands.extend(
            [
                "q",
                "0.92 G",
                "0.4 w",
                f"{left + 10.0:.2f} {y:.2f} m {right - 10.0:.2f} {y + (3.0 if idx % 2 == 0 else -2.0):.2f} l S",
                "Q",
            ]
        )

    if has_wildfire_overlay:
        commands.extend(
            [
                "q",
                "0.90 g",
                "0.75 G",
                "0.6 w",
                f"{left + (width * 0.66):.2f} {bottom + (height * 0.62):.2f} m",
                f"{left + (width * 0.86):.2f} {bottom + (height * 0.74):.2f} l",
                f"{left + (width * 0.90):.2f} {bottom + (height * 0.52):.2f} l",
                f"{left + (width * 0.72):.2f} {bottom + (height * 0.40):.2f} l",
                "h b",
                "Q",
            ]
        )

    if has_parcel:
        parcel_left = center_x - (width * 0.18)
        parcel_bottom = center_y - (height * 0.16)
        parcel_w = width * 0.36
        parcel_h = height * 0.32
        commands.extend(
            [
                "q",
                "0.70 G",
                "1.0 w",
                f"{parcel_left:.2f} {parcel_bottom:.2f} {parcel_w:.2f} {parcel_h:.2f} re S",
                "Q",
            ]
        )

    if has_rings:
        for radius, gray in ((12.0, 0.45), (24.0, 0.55), (40.0, 0.65)):
            commands.extend(["q", f"{gray:.2f} G", "0.9 w", *_circle_path_commands(center_x, center_y, radius), "S", "Q"])
    else:
        commands.extend(["q", "0.60 G", "0.8 w", *_circle_path_commands(center_x, center_y, 28.0), "S", "Q"])

    if has_building:
        commands.extend(
            [
                "q",
                "0.80 g",
                "0.30 G",
                "0.9 w",
                f"{center_x - 7.0:.2f} {center_y - 5.0:.2f} 14.0 10.0 re B",
                "Q",
            ]
        )
    else:
        commands.extend(
            [
                "q",
                "0.30 G",
                "1.1 w",
                f"{center_x - 5.0:.2f} {center_y - 5.0:.2f} m {center_x + 5.0:.2f} {center_y + 5.0:.2f} l S",
                f"{center_x - 5.0:.2f} {center_y + 5.0:.2f} m {center_x + 5.0:.2f} {center_y - 5.0:.2f} l S",
                "Q",
            ]
        )

    # Home marker
    commands.extend(
        [
            "q",
            "0.18 g",
            "0.18 G",
            *_circle_path_commands(center_x, center_y, 2.8),
            "B",
            "Q",
        ]
    )

    # Approximate-geometry indicator tag bar at bottom edge of panel.
    if approximate:
        commands.extend(
            [
                "q",
                "0.93 g",
                "0.78 G",
                "0.6 w",
                f"{left + 8.0:.2f} {bottom + 6.0:.2f} {width - 16.0:.2f} 12.0 re B",
                "Q",
            ]
        )
    return commands


def _build_pdf_content_stream(page_entries: list[_PdfEntry]) -> bytes:
    commands: list[str] = []
    y = _PDF_START_Y
    section_counter = 0

    for entry in page_entries:
        if entry.style in _PDF_SPACER_HEIGHTS:
            y -= _PDF_SPACER_HEIGHTS[entry.style]
            continue

        style = _PDF_TEXT_STYLES.get(entry.style) or _PDF_TEXT_STYLES["body"]
        font = str(style.get("font", "F1"))
        size = float(style.get("size", 10.4))
        indent = float(style.get("indent", _PDF_PAGE_LEFT))
        gray = float(style.get("gray", 0.0))
        line_height = _entry_height(entry)

        y -= line_height
        if y < (_PDF_BOTTOM_Y - 6.0):
            break

        if entry.style == "map_canvas":
            panel_style = _PDF_TEXT_STYLES.get("map_canvas") or {}
            panel_left = float(panel_style.get("panel_left", _PDF_PAGE_LEFT))
            panel_right = float(panel_style.get("panel_right", _PDF_PAGE_RIGHT))
            panel_height = float(panel_style.get("panel_height", max(80.0, line_height - 6.0)))
            panel_top = y + line_height - 4.0
            panel_bottom = panel_top - panel_height
            try:
                map_config = json.loads(str(entry.text or "{}"))
            except Exception:
                map_config = {}
            if not isinstance(map_config, dict):
                map_config = {}
            commands.extend(
                _draw_local_map_panel_commands(
                    map_config,
                    left=panel_left,
                    right=panel_right,
                    bottom=panel_bottom,
                    top=panel_top,
                )
            )
            continue

        if entry.style in {"section", "SectionHeaderStyle"}:
            if section_counter > 0:
                separator_y = y + line_height - 7.0
                commands.extend(
                    [
                        "q",
                        "0.84 G",
                        "0.8 w",
                        f"{_PDF_PAGE_LEFT:.2f} {separator_y:.2f} m {_PDF_PAGE_RIGHT:.2f} {separator_y:.2f} l S",
                        "Q",
                    ]
                )
            section_counter += 1

        box_fill = style.get("box_fill")
        if box_fill is not None:
            fill_gray = float(box_fill)
            stroke_gray = float(style.get("box_stroke", 0.82))
            box_left = float(style.get("box_left", _PDF_PAGE_LEFT))
            box_right = float(style.get("box_right", _PDF_PAGE_RIGHT))
            box_width = max(12.0, box_right - box_left)
            box_height = float(style.get("box_height", max(16.0, line_height)))
            box_top = y + (line_height * 0.80)
            box_bottom = box_top - box_height
            commands.extend(
                [
                    "q",
                    f"{fill_gray:.2f} g",
                    f"{stroke_gray:.2f} G",
                    "0.8 w",
                    f"{box_left:.2f} {box_bottom:.2f} {box_width:.2f} {box_height:.2f} re B",
                    "Q",
                ]
            )

        escaped = _escape_pdf_text(entry.text)
        commands.extend(
            [
                "BT",
                f"/{font} {size:.2f} Tf",
                f"{gray:.2f} g",
                f"1 0 0 1 {indent:.2f} {y:.2f} Tm",
                f"({escaped}) Tj",
                "ET",
            ]
        )

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
    entries = _build_report_entries(report)
    pages = _paginate_entries(entries)

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    }

    page_ids: list[int] = []
    for idx, page_entries in enumerate(pages):
        content_id = 5 + idx * 2
        page_id = 6 + idx * 2
        page_ids.append(page_id)

        content_stream = _build_pdf_content_stream(page_entries)
        objects[content_id] = (
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
            + content_stream
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")

    return _serialize_pdf(objects)
