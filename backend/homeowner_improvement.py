from __future__ import annotations

from typing import Any, Iterable

from backend.models import (
    AssessmentResult,
    HomeownerFollowUpInput,
    HomeownerImprovementOptions,
    PropertyAttributes,
)


_IMPROVEMENT_INPUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "input_key": "map_point_correction",
        "assessment_field": "user_selected_point",
        "label": "Move pin to your home",
        "prompt": "Move map pin to your home and confirm building location for better property-specific detail.",
        "input_type": "map_point",
        "options": [],
        "suggestion": "Move pin to your home and confirm building location to improve structure-level specificity.",
        "base_lift": 100,
    },
    {
        "input_key": "roof_type",
        "assessment_field": "roof_type",
        "label": "Roof type",
        "prompt": "What best describes your roof material?",
        "input_type": "select",
        "options": [
            "class a",
            "metal",
            "tile",
            "asphalt composition",
            "wood/combustible",
        ],
        "suggestion": "Add your roof type to tighten structure ignition estimates.",
        "base_lift": 95,
    },
    {
        "input_key": "vent_type",
        "assessment_field": "vent_type",
        "label": "Vent type",
        "prompt": "Do your vents have ember-resistant protection?",
        "input_type": "select",
        "options": [
            "ember-resistant",
            "screened (1/8 inch or finer)",
            "standard",
            "unknown",
        ],
        "suggestion": "Add vent protection details to improve ember intrusion estimates.",
        "base_lift": 90,
    },
    {
        "input_key": "window_type",
        "assessment_field": "window_type",
        "label": "Window type",
        "prompt": "What best describes your windows?",
        "input_type": "select",
        "options": [
            "dual-pane tempered",
            "dual-pane standard",
            "single-pane",
            "unknown",
        ],
        "suggestion": "Adding window type can improve structure-vulnerability detail.",
        "base_lift": 74,
    },
    {
        "input_key": "defensible_space_condition",
        "assessment_field": "defensible_space_ft",
        "label": "Defensible space condition",
        "prompt": "About how many feet around the home are kept mostly non-combustible?",
        "input_type": "number",
        "unit": "feet",
        "suggestion": "Add defensible space condition to improve near-structure ignition estimates.",
        "base_lift": 88,
    },
)

_DEFENSIBLE_SPACE_CONDITION_TO_FT: dict[str, float] = {
    "poor": 5.0,
    "limited": 12.0,
    "moderate": 25.0,
    "good": 40.0,
    "excellent": 60.0,
}


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def defensible_space_ft_from_condition(condition: str | None) -> float | None:
    normalized = str(condition or "").strip().lower()
    if not normalized:
        return None
    return _DEFENSIBLE_SPACE_CONDITION_TO_FT.get(normalized)


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not bool(value.strip())
    return False


def _collect_missing_input_fields(result: AssessmentResult) -> set[str]:
    missing: set[str] = set()
    for row in result.missing_inputs or []:
        text = str(row).strip()
        if text:
            missing.add(text)
    diagnostics = result.assessment_diagnostics
    for row in diagnostics.critical_inputs_missing or []:
        text = str(row).strip()
        if text:
            missing.add(text)
    for row in diagnostics.inferred_inputs or []:
        text = str(row).strip()
        if text:
            missing.add(text)
    for row in diagnostics.fallback_decisions or []:
        if not isinstance(row, dict):
            continue
        missing_input = str(row.get("missing_input") or "").strip()
        if missing_input:
            missing.add(missing_input)
    coverage_summary = result.coverage_summary
    for layer in list(getattr(coverage_summary, "critical_missing_layers", []) or []):
        token = str(layer).strip()
        if not token:
            continue
        if token.endswith("_layer"):
            missing.add(token)
        else:
            missing.add(f"{token}_layer")
    ledger = result.score_evidence_ledger
    for family in (
        list(ledger.site_hazard_score or []),
        list(ledger.home_ignition_vulnerability_score or []),
        list(ledger.insurance_readiness_score or []),
        list(ledger.wildfire_risk_score or []),
    ):
        for factor in family:
            status = str(getattr(factor, "evidence_status", "")).strip().lower()
            if status not in {"missing", "inferred", "fallback"}:
                continue
            source_field = str(getattr(factor, "source_field", "") or "").strip()
            if not source_field:
                continue
            source_tokens = (
                source_field.replace("+", ",")
                .replace(";", ",")
                .split(",")
            )
            for raw_token in source_tokens:
                token = str(raw_token).strip().lower()
                if not token:
                    continue
                if token in {
                    "site_hazard_score",
                    "home_ignition_vulnerability_score",
                    "insurance_readiness_score",
                    "derived_interaction",
                    "building_footprint",
                    "building_footprint_layer",
                }:
                    continue
                missing.add(token)
    return missing


_DIAGNOSTIC_GAP_SUGGESTIONS: dict[str, str] = {
    "roof_type": "Adding your roof type can improve accuracy.",
    "vent_type": "Adding your vent type can improve ember-exposure accuracy.",
    "defensible_space_ft": "Adding defensible space condition can improve near-structure accuracy.",
    "building_footprint": "Adding building footprint coverage can improve structure-level accuracy.",
    "building_footprint_layer": "Adding building footprint coverage can improve structure-level accuracy.",
    "burn_probability_layer": "Adding burn probability coverage can improve regional hazard accuracy.",
    "hazard_layer": "Adding hazard-severity coverage can improve regional hazard accuracy.",
    "gridmet_dryness_layer": "Adding dryness coverage can improve climate-stress accuracy.",
    "roads_layer": "Adding road-network coverage can improve access and evacuation context.",
    "parcel_polygons_layer": "Adding parcel coverage can improve property boundary accuracy.",
    "parcel_address_points_layer": "Adding parcel-address points can improve property matching accuracy.",
}


def _friendly_gap_name(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return "data input"
    if text.endswith("_layer"):
        text = text[:-6]
    return text.replace("_", " ")


def _build_diagnostic_gap_suggestions(result: AssessmentResult, missing_fields: set[str]) -> list[str]:
    suggestions: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        candidate = str(text or "").strip()
        if not candidate:
            return
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        suggestions.append(candidate)

    for field in sorted(missing_fields):
        if field in _DIAGNOSTIC_GAP_SUGGESTIONS:
            _add(_DIAGNOSTIC_GAP_SUGGESTIONS[field])
        elif field.endswith("_layer"):
            _add(f"Adding {_friendly_gap_name(field)} data can improve accuracy.")

    for row in list(getattr(result.coverage_summary, "recommended_actions", []) or []):
        text = str(row).strip()
        if not text:
            continue
        if text.endswith("."):
            _add(text)
        else:
            _add(text + ".")

    for row in result.assessment_diagnostics.fallback_decisions or []:
        if not isinstance(row, dict):
            continue
        note = str(row.get("note") or "").strip()
        if note:
            if note.endswith("."):
                _add(note)
            else:
                _add(note + ".")

    return suggestions[:10]


def build_improve_your_result_block(result: AssessmentResult) -> dict[str, Any]:
    options = build_homeowner_improvement_options(result)
    missing_fields = _collect_missing_input_fields(result)
    coverage_summary = result.coverage_summary

    diagnostic_sources = {
        "evidence_ledger_missing_or_inferred": sorted(
            {
                str(field)
                for field in missing_fields
                if str(field).strip()
            }
        )[:12],
        "coverage_gaps": [
            str(layer)
            for layer in list(getattr(coverage_summary, "critical_missing_layers", []) or [])
            if str(layer).strip()
        ][:8],
        "fallback_inputs": sorted(
            {
                str((row or {}).get("missing_input") or "").strip()
                for row in list(result.assessment_diagnostics.fallback_decisions or [])
                if isinstance(row, dict) and str((row or {}).get("missing_input") or "").strip()
            }
        )[:12],
    }

    suggestions = list(options.improve_your_result_suggestions or [])
    for row in _build_diagnostic_gap_suggestions(result, missing_fields):
        if row not in suggestions:
            suggestions.append(row)

    return {
        "missing_key_inputs": list(options.missing_key_inputs or []),
        "prioritized_missing_key_inputs": list(options.prioritized_missing_key_inputs or []),
        "geometry_issue_flags": list(options.geometry_issue_flags or []),
        "highest_value_next_question": (
            options.highest_value_next_question.model_dump()
            if options.highest_value_next_question
            else None
        ),
        "remaining_optional_input_count": int(options.remaining_optional_input_count or 0),
        "suggestions": suggestions[:6],
        "optional_follow_up_inputs": [row.model_dump() for row in list(options.optional_follow_up_inputs or [])],
        "diagnostic_sources": diagnostic_sources,
    }


def build_homeowner_improvement_options(result: AssessmentResult) -> HomeownerImprovementOptions:
    facts = result.property_facts if isinstance(result.property_facts, dict) else {}
    missing_fields = _collect_missing_input_fields(result)
    strict_missing_fields: set[str] = set(str(row).strip() for row in (result.missing_inputs or []) if str(row).strip())
    diagnostics = result.assessment_diagnostics
    for row in diagnostics.critical_inputs_missing or []:
        token = str(row).strip()
        if token:
            strict_missing_fields.add(token)
    for row in diagnostics.fallback_decisions or []:
        if not isinstance(row, dict):
            continue
        token = str(row.get("missing_input") or "").strip()
        if token:
            strict_missing_fields.add(token)

    geometry_resolution = (
        result.geometry_resolution.model_dump()
        if hasattr(result.geometry_resolution, "model_dump")
        else (
            dict(result.geometry_resolution)
            if isinstance(result.geometry_resolution, dict)
            else {}
        )
    )
    ring_mode = str(geometry_resolution.get("ring_generation_mode") or "").strip().lower()
    footprint_status = str(geometry_resolution.get("footprint_match_status") or "").strip().lower()
    try:
        anchor_quality = float(geometry_resolution.get("anchor_quality_score") or 0.0)
    except (TypeError, ValueError):
        anchor_quality = 0.0
    naip_status = str(geometry_resolution.get("naip_structure_feature_status") or "").strip().lower()
    parcel_match_status = str(geometry_resolution.get("parcel_match_status") or "").strip().lower()
    source_conflict_flag = bool(getattr(result, "source_conflict_flag", False))
    footprint_missing = footprint_status in {"none", "ambiguous", "provider_unavailable", "error"}
    low_anchor_confidence = anchor_quality < 0.70
    parcel_mismatch_or_unresolved = parcel_match_status in {"not_found", "provider_unavailable"} or source_conflict_flag
    geometry_limited = bool(
        ring_mode == "point_annulus_fallback"
        or footprint_missing
        or low_anchor_confidence
        or naip_status in {"missing", "provider_unavailable", "present_but_not_consumed", "fallback_or_proxy"}
        or parcel_mismatch_or_unresolved
    )
    geometry_issue_flags: list[str] = []
    if footprint_missing:
        geometry_issue_flags.append("missing_footprint")
    if low_anchor_confidence:
        geometry_issue_flags.append("low_anchor_confidence")
    if parcel_mismatch_or_unresolved:
        geometry_issue_flags.append("parcel_mismatch")
    if ring_mode == "point_annulus_fallback":
        geometry_issue_flags.append("point_fallback_rings")

    candidate_rows: list[dict[str, Any]] = []
    for spec in _IMPROVEMENT_INPUT_SPECS:
        field = str(spec["assessment_field"])
        input_key = str(spec["input_key"])
        if input_key == "map_point_correction":
            is_missing = geometry_limited
        else:
            is_missing = field in strict_missing_fields or _is_missing_value(facts.get(field))
        if not is_missing:
            continue
        score = float(spec.get("base_lift") or 50)
        if field in strict_missing_fields:
            score += 10.0
        if field in {"roof_type", "vent_type", "window_type", "defensible_space_ft"} and field in missing_fields:
            score += 5.0
        if input_key == "map_point_correction":
            if ring_mode == "point_annulus_fallback":
                score += 15.0
            if anchor_quality < 0.50:
                score += 8.0
            if footprint_status in {"none", "ambiguous"}:
                score += 8.0
        candidate_rows.append(
            {
                "input_key": input_key,
                "lift_score": round(score, 2),
                "suggestion": str(spec["suggestion"]),
                "follow_up": HomeownerFollowUpInput(
                    input_key=input_key,
                    assessment_field=field,
                    label=str(spec["label"]),
                    prompt=str(spec["prompt"]),
                    input_type=str(spec["input_type"]),  # type: ignore[arg-type]
                    options=[str(item) for item in (spec.get("options") or [])],
                    unit=(str(spec["unit"]) if spec.get("unit") else None),
                ),
            }
        )
    candidate_rows.sort(key=lambda row: (-float(row.get("lift_score") or 0.0), str(row.get("input_key") or "")))

    missing_key_inputs = [str(row.get("input_key") or "") for row in candidate_rows if str(row.get("input_key") or "")]
    prioritized_rows = candidate_rows[:3]
    prioritized_missing_key_inputs = [
        str(row.get("input_key") or "")
        for row in prioritized_rows
        if str(row.get("input_key") or "")
    ]
    follow_ups = [row["follow_up"] for row in prioritized_rows if isinstance(row.get("follow_up"), HomeownerFollowUpInput)]

    suggestions: list[str] = []
    if geometry_limited:
        suggestions.append("Move pin to your home.")
        suggestions.append("Confirm building location.")
    for row in prioritized_rows:
        suggestion = str(row.get("suggestion") or "").strip()
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)

    if not suggestions:
        suggestions.append(
            "Key property details are already provided. Re-run the assessment after major home or vegetation changes."
        )
    else:
        for text in _build_diagnostic_gap_suggestions(result, missing_fields):
            if text not in suggestions:
                suggestions.append(text)
    return HomeownerImprovementOptions(
        assessment_id=result.assessment_id,
        missing_key_inputs=missing_key_inputs,
        prioritized_missing_key_inputs=prioritized_missing_key_inputs,
        highest_value_next_question=(follow_ups[0] if follow_ups else None),
        remaining_optional_input_count=max(0, len(candidate_rows) - len(follow_ups)),
        geometry_issue_flags=geometry_issue_flags,
        improve_your_result_suggestions=suggestions[:6],
        optional_follow_up_inputs=follow_ups,
    )


def summarize_assessment_for_improvement(result: AssessmentResult) -> dict[str, Any]:
    return {
        "assessment_id": result.assessment_id,
        "wildfire_risk_score": result.wildfire_risk_score,
        "site_hazard_score": result.site_hazard_score,
        "insurance_readiness_score": result.insurance_readiness_score,
        "confidence_score": result.confidence_score,
        "confidence_tier": result.confidence_tier,
        "top_recommended_actions": list(result.top_recommended_actions or [])[:3],
        "top_risk_drivers": list(result.top_risk_drivers or [])[:3],
    }


def build_improvement_change_set(
    before: AssessmentResult,
    after: AssessmentResult,
    *,
    changed_fields_hint: Iterable[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    tracked_facts = set(changed_fields_hint or [])
    if not tracked_facts:
        tracked_facts = {
            "roof_type",
            "vent_type",
            "defensible_space_ft",
            "construction_year",
            "siding_type",
            "window_type",
            "vegetation_condition",
        }
    before_facts = before.property_facts if isinstance(before.property_facts, dict) else {}
    after_facts = after.property_facts if isinstance(after.property_facts, dict) else {}

    changed: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for field in sorted(tracked_facts):
        if before_facts.get(field) == after_facts.get(field):
            continue
        changed[field] = {
            "before": before_facts.get(field),
            "after": after_facts.get(field),
        }
    if changed:
        notes.append("Property details were updated and the assessment was re-run.")

    for score_key in ("wildfire_risk_score", "insurance_readiness_score", "confidence_score"):
        before_value = getattr(before, score_key, None)
        after_value = getattr(after, score_key, None)
        if before_value == after_value:
            continue
        changed[score_key] = {"before": before_value, "after": after_value}

    if before.confidence_tier != after.confidence_tier:
        changed["confidence_tier"] = {"before": before.confidence_tier, "after": after.confidence_tier}
        notes.append("Confidence tier changed after adding property details.")

    before_actions = list(before.top_recommended_actions or [])[:3]
    after_actions = list(after.top_recommended_actions or [])[:3]
    if before_actions != after_actions:
        changed["top_recommended_actions"] = {"before": before_actions, "after": after_actions}
        notes.append("Top recommendations changed based on the updated inputs.")

    before_specificity = (
        before.specificity_summary.model_dump()
        if hasattr(before.specificity_summary, "model_dump")
        else (dict(before.specificity_summary) if isinstance(before.specificity_summary, dict) else {})
    )
    after_specificity = (
        after.specificity_summary.model_dump()
        if hasattr(after.specificity_summary, "model_dump")
        else (dict(after.specificity_summary) if isinstance(after.specificity_summary, dict) else {})
    )
    before_tier = str(before_specificity.get("specificity_tier") or "regional_estimate")
    after_tier = str(after_specificity.get("specificity_tier") or "regional_estimate")

    def _geometry_dict(result: AssessmentResult) -> dict[str, Any]:
        raw = (
            result.geometry_resolution.model_dump()
            if hasattr(result.geometry_resolution, "model_dump")
            else (dict(result.geometry_resolution) if isinstance(result.geometry_resolution, dict) else {})
        )
        return raw if isinstance(raw, dict) else {}

    def _delta_direction(before_value: float | None, after_value: float | None) -> str:
        if before_value is None or after_value is None:
            return "unknown"
        if float(after_value) > float(before_value):
            return "up"
        if float(after_value) < float(before_value):
            return "down"
        return "unchanged"

    before_conf = (
        ((before.homeowner_summary or {}).get("trust_summary") or {})
        if isinstance(before.homeowner_summary, dict)
        else {}
    )
    after_conf = (
        ((after.homeowner_summary or {}).get("trust_summary") or {})
        if isinstance(after.homeowner_summary, dict)
        else {}
    )
    before_diff_score = _safe_float(
        before_conf.get("local_differentiation_score")
    ) or _safe_float(before_conf.get("neighborhood_differentiation_confidence"))
    after_diff_score = _safe_float(
        after_conf.get("local_differentiation_score")
    ) or _safe_float(after_conf.get("neighborhood_differentiation_confidence"))
    before_geom = _geometry_dict(before)
    after_geom = _geometry_dict(after)
    summary = {
        "score_direction": {
            "metric": "wildfire_risk_score",
            "direction": _delta_direction(before.wildfire_risk_score, after.wildfire_risk_score),
            "before": before.wildfire_risk_score,
            "after": after.wildfire_risk_score,
        },
        "specificity_change": {
            "before_tier": before_tier,
            "after_tier": after_tier,
            "changed": before_tier != after_tier,
            "comparison_allowed_before": bool(before_specificity.get("comparison_allowed")),
            "comparison_allowed_after": bool(after_specificity.get("comparison_allowed")),
        },
        "confidence_change": {
            "score_direction": _delta_direction(before.confidence_score, after.confidence_score),
            "before_score": before.confidence_score,
            "after_score": after.confidence_score,
            "before_level": str(before_conf.get("confidence_level") or ""),
            "after_level": str(after_conf.get("confidence_level") or ""),
        },
        "differentiation_change": {
            "score_direction": _delta_direction(before_diff_score, after_diff_score),
            "before_score": before_diff_score,
            "after_score": after_diff_score,
            "before_mode": str(before_conf.get("differentiation_mode") or ""),
            "after_mode": str(after_conf.get("differentiation_mode") or ""),
            "changed": (
                (before_diff_score is not None and after_diff_score is not None and float(before_diff_score) != float(after_diff_score))
                or str(before_conf.get("differentiation_mode") or "") != str(after_conf.get("differentiation_mode") or "")
            ),
        },
        "geometry_change": {
            "anchor_quality_direction": _delta_direction(
                _safe_float(before_geom.get("anchor_quality_score")),
                _safe_float(after_geom.get("anchor_quality_score")),
            ),
            "before_footprint_match_status": str(before_geom.get("footprint_match_status") or ""),
            "after_footprint_match_status": str(after_geom.get("footprint_match_status") or ""),
            "before_parcel_match_status": str(before_geom.get("parcel_match_status") or ""),
            "after_parcel_match_status": str(after_geom.get("parcel_match_status") or ""),
        },
        "recommendation_changes": {
            "changed": before_actions != after_actions,
            "before_top_actions": before_actions,
            "after_top_actions": after_actions,
        },
    }
    changed["score_direction"] = summary["score_direction"]
    changed["specificity_change"] = summary["specificity_change"]
    changed["differentiation_change"] = summary["differentiation_change"]
    changed["geometry_change"] = summary["geometry_change"]
    changed["confidence_change"] = summary["confidence_change"]
    changed["recommendation_changes"] = summary["recommendation_changes"]
    return changed, notes, summary
