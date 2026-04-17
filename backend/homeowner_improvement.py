from __future__ import annotations

from typing import Any, Iterable

from backend.insurability import derive_insurability_status
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
        "input_key": "confirm_selected_parcel",
        "assessment_field": "selected_parcel_id",
        "label": "Confirm parcel boundary",
        "prompt": "Confirm your parcel boundary on the map so parcel-relative features stay on your property.",
        "input_type": "map_polygon",
        "options": [],
        "suggestion": "Confirming your parcel boundary can improve property anchoring and specificity.",
        "base_lift": 97,
    },
    {
        "input_key": "confirm_selected_footprint",
        "assessment_field": "selected_structure_id",
        "label": "Confirm building footprint",
        "prompt": "Confirm the selected building footprint to keep defensible-space rings tied to your home.",
        "input_type": "map_polygon",
        "options": [],
        "suggestion": "Confirming your building footprint can improve structure-level feature precision.",
        "base_lift": 96,
    },
    {
        "input_key": "select_building_polygon",
        "assessment_field": "selected_structure_geometry",
        "label": "Select your building",
        "prompt": "Select your building polygon on the map to anchor defensible-space rings to the actual structure.",
        "input_type": "map_polygon",
        "options": [],
        "suggestion": "Selecting your building polygon can materially improve property-specific differentiation.",
        "base_lift": 98,
    },
    {
        "input_key": "draw_structure_manually",
        "assessment_field": "selected_structure_geometry",
        "label": "Draw your building outline",
        "prompt": "If map footprints are missing or wrong, draw your building outline manually.",
        "input_type": "map_polygon",
        "options": [],
        "suggestion": "Drawing your structure manually can improve specificity when footprint data is unavailable.",
        "base_lift": 92,
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

_STRUCTURE_ATTRIBUTE_FIELDS: tuple[str, ...] = (
    "roof_type",
    "vent_type",
    "window_type",
    "defensible_space_ft",
    "construction_year",
    "siding_type",
)

_STATUS_SEVERITY: dict[str, int] = {
    "Likely Insurable": 0,
    "At Risk": 1,
    "High Risk of Insurance Issues": 2,
}

_CHANGED_INPUT_ACTIONS: dict[str, tuple[str, str]] = {
    "defensible_space_ft": (
        "Reduce vegetation in the closest defensible-space zones.",
        "This lowers ember and flame pressure close to the home.",
    ),
    "defensible_space_condition": (
        "Improve defensible space around the structure.",
        "This helps reduce ignition pathways near the home.",
    ),
    "roof_type": (
        "Upgrade to a Class A fire-resistant roof.",
        "The roof is a major ignition pathway during ember storms.",
    ),
    "vent_type": (
        "Install ember-resistant vents.",
        "This helps block ember intrusion into vulnerable openings.",
    ),
    "window_type": (
        "Upgrade to more fire-resistant windows.",
        "Stronger windows reduce breakage risk from heat and embers.",
    ),
    "siding_type": (
        "Use ignition-resistant siding in vulnerable areas.",
        "More resistant exterior materials reduce flame-contact vulnerability.",
    ),
    "construction_year": (
        "Prioritize upgrades for older vulnerable components.",
        "Older construction details can increase ember and flame exposure.",
    ),
    "map_point_correction": (
        "Confirm the map pin and building location.",
        "Better location accuracy improves property-specific scoring and action targeting.",
    ),
    "confirm_selected_parcel": (
        "Confirm parcel boundaries.",
        "Better parcel context improves property-specific inputs and recommendations.",
    ),
    "confirm_selected_footprint": (
        "Confirm building footprint selection.",
        "Footprint-anchored analysis improves near-structure risk and action precision.",
    ),
    "selected_structure_geometry": (
        "Provide or refine building geometry.",
        "Structure-anchored rings improve near-home risk interpretation.",
    ),
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


def _structure_attribute_gaps(
    *,
    facts: dict[str, Any],
    strict_missing_fields: set[str],
    missing_fields: set[str],
) -> list[str]:
    gaps: list[str] = []
    for field in _STRUCTURE_ATTRIBUTE_FIELDS:
        if field in strict_missing_fields or field in missing_fields or _is_missing_value(facts.get(field)):
            gaps.append(field)
    return sorted(set(gaps))


def _geometry_uncertainty_summary(
    geometry_resolution: dict[str, Any],
    geometry_issue_flags: list[str],
) -> dict[str, Any]:
    return {
        "issue_flags": list(geometry_issue_flags),
        "anchor_source": str(geometry_resolution.get("anchor_source") or ""),
        "anchor_quality_score": _safe_float(geometry_resolution.get("anchor_quality_score")),
        "parcel_match_status": str(geometry_resolution.get("parcel_match_status") or ""),
        "footprint_match_status": str(geometry_resolution.get("footprint_match_status") or ""),
        "ring_generation_mode": str(geometry_resolution.get("ring_generation_mode") or ""),
        "naip_structure_feature_status": str(geometry_resolution.get("naip_structure_feature_status") or ""),
        "property_mismatch_flag": bool(geometry_resolution.get("property_mismatch_flag")),
        "mismatch_reason": (
            str(geometry_resolution.get("mismatch_reason") or "").strip() or None
        ),
    }


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
        "structure_attribute_gaps": list(options.structure_attribute_gaps or []),
        "geometry_uncertainty": dict(options.geometry_uncertainty or {}),
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
        "missing_property_fields": list(diagnostic_sources["structure_attribute_gaps"]),
        "structure_attribute_gaps": list(diagnostic_sources["structure_attribute_gaps"]),
        "geometry_uncertainty": dict(diagnostic_sources["geometry_uncertainty"]),
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
    property_mismatch_flag = bool(geometry_resolution.get("property_mismatch_flag"))
    mismatch_reason = str(geometry_resolution.get("mismatch_reason") or "").strip()
    property_confidence_summary = (
        result.property_confidence_summary.model_dump()
        if hasattr(result.property_confidence_summary, "model_dump")
        else (
            dict(result.property_confidence_summary)
            if isinstance(result.property_confidence_summary, dict)
            else {}
        )
    )
    property_confidence_level = str(property_confidence_summary.get("level") or "").strip().lower()
    source_conflict_flag = bool(getattr(result, "source_conflict_flag", False))
    footprint_missing = footprint_status in {"none", "ambiguous", "provider_unavailable", "error"}
    low_anchor_confidence = anchor_quality < 0.70
    parcel_mismatch_or_unresolved = parcel_match_status in {"not_found", "provider_unavailable"} or source_conflict_flag
    geometry_limited = bool(
        ring_mode in {"point_annulus_fallback", "point_annulus_parcel_clipped"}
        or footprint_missing
        or low_anchor_confidence
        or naip_status in {"missing", "provider_unavailable", "present_but_not_consumed", "fallback_or_proxy"}
        or parcel_mismatch_or_unresolved
        or property_mismatch_flag
        or property_confidence_level in {"regional_estimate_with_anchor", "insufficient_property_identification", "low"}
    )
    geometry_issue_flags: list[str] = []
    if footprint_missing:
        geometry_issue_flags.append("missing_footprint")
    if low_anchor_confidence:
        geometry_issue_flags.append("low_anchor_confidence")
    if parcel_mismatch_or_unresolved:
        geometry_issue_flags.append("parcel_mismatch")
    if ring_mode in {"point_annulus_fallback", "point_annulus_parcel_clipped"}:
        geometry_issue_flags.append("point_fallback_rings")
    if property_mismatch_flag:
        geometry_issue_flags.append("property_mismatch")
    if property_confidence_level in {"regional_estimate_with_anchor", "insufficient_property_identification", "low"}:
        geometry_issue_flags.append("low_property_confidence")
    structure_attribute_gaps = _structure_attribute_gaps(
        facts=facts,
        strict_missing_fields=strict_missing_fields,
        missing_fields=missing_fields,
    )
    geometry_uncertainty = _geometry_uncertainty_summary(
        geometry_resolution if isinstance(geometry_resolution, dict) else {},
        geometry_issue_flags,
    )

    candidate_rows: list[dict[str, Any]] = []
    geometry_input_keys = {
        "map_point_correction",
        "confirm_selected_parcel",
        "confirm_selected_footprint",
        "select_building_polygon",
        "draw_structure_manually",
    }
    for spec in _IMPROVEMENT_INPUT_SPECS:
        field = str(spec["assessment_field"])
        input_key = str(spec["input_key"])
        if input_key in geometry_input_keys:
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
        if input_key in geometry_input_keys:
            if ring_mode in {"point_annulus_fallback", "point_annulus_parcel_clipped"}:
                score += 15.0
            if anchor_quality < 0.50:
                score += 8.0
            if footprint_status in {"none", "ambiguous"}:
                score += 8.0
            if property_mismatch_flag:
                score += 12.0
            if input_key == "confirm_selected_parcel" and parcel_match_status not in {"matched", "multiple_candidates"}:
                score -= 6.0
            if input_key == "confirm_selected_footprint" and footprint_status in {"none", "provider_unavailable", "error"}:
                score -= 6.0
            if input_key == "draw_structure_manually" and footprint_status not in {"none", "provider_unavailable", "error"}:
                score -= 10.0
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
    prioritized_rows = candidate_rows[:1]
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
        suggestions.append("Select your building polygon or draw your home outline when map footprints are wrong.")
    if property_mismatch_flag and mismatch_reason:
        message = f"We may be analyzing the wrong property: {mismatch_reason}"
        if message not in suggestions:
            suggestions.insert(0, message)
    for row in prioritized_rows:
        suggestion = str(row.get("suggestion") or "").strip()
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
    if geometry_limited:
        for row in candidate_rows:
            input_key = str(row.get("input_key") or "")
            if input_key in geometry_input_keys:
                continue
            suggestion = str(row.get("suggestion") or "").strip()
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
                break

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
        missing_property_fields=structure_attribute_gaps,
        structure_attribute_gaps=structure_attribute_gaps,
        geometry_uncertainty=geometry_uncertainty,
        improve_your_result_suggestions=suggestions[:6],
        optional_follow_up_inputs=follow_ups,
    )


def summarize_assessment_for_improvement(result: AssessmentResult) -> dict[str, Any]:
    property_confidence_summary = (
        result.property_confidence_summary.model_dump()
        if hasattr(result.property_confidence_summary, "model_dump")
        else (
            dict(result.property_confidence_summary)
            if isinstance(result.property_confidence_summary, dict)
            else {}
        )
    )
    insurability_snapshot = _insurability_snapshot(result)
    return {
        "assessment_id": result.assessment_id,
        "wildfire_risk_score": result.wildfire_risk_score,
        "site_hazard_score": result.site_hazard_score,
        "insurance_readiness_score": result.insurance_readiness_score,
        "home_hardening_readiness": result.home_hardening_readiness,
        "confidence_score": result.confidence_score,
        "confidence_tier": result.confidence_tier,
        "property_data_confidence": result.property_data_confidence,
        "property_confidence_level": str(property_confidence_summary.get("level") or ""),
        "property_confidence_key_gaps": list(
            property_confidence_summary.get("key_reasons")
            or property_confidence_summary.get("key_gaps")
            or []
        )[:4],
        "insurability_status": insurability_snapshot.get("insurability_status"),
        "insurability_status_reasons": list(insurability_snapshot.get("insurability_status_reasons") or [])[:3],
        "insurability_status_methodology_note": str(
            insurability_snapshot.get("insurability_status_methodology_note") or ""
        ),
        "top_recommended_actions": list(result.top_recommended_actions or [])[:3],
        "top_risk_drivers": list(result.top_risk_drivers or [])[:3],
    }


def _insurability_snapshot(result: AssessmentResult) -> dict[str, Any]:
    risk_score = (
        result.overall_wildfire_risk
        if result.overall_wildfire_risk is not None
        else result.wildfire_risk_score
    )
    hardening = (
        result.home_hardening_readiness
        if result.home_hardening_readiness is not None
        else result.insurance_readiness_score
    )
    fallback = derive_insurability_status(
        wildfire_risk_score=risk_score,
        home_hardening_readiness=hardening,
        confidence_tier=result.confidence_tier,
        assessment_specificity_tier=result.assessment_specificity_tier,
        top_near_structure_risk_drivers=result.top_near_structure_risk_drivers,
        top_risk_drivers=result.top_risk_drivers,
        defensible_space_analysis=result.defensible_space_analysis,
        defensible_space_limitations_summary=result.defensible_space_limitations_summary,
        readiness_blockers=result.readiness_blockers,
        scoring_status=result.scoring_status,
    )
    status = str(result.insurability_status or fallback.insurability_status).strip()
    reasons = [
        str(v).strip()
        for v in (
            list(result.insurability_status_reasons or [])
            if list(result.insurability_status_reasons or [])
            else list(fallback.insurability_status_reasons)
        )
        if str(v).strip()
    ][:3]
    methodology_note = str(
        result.insurability_status_methodology_note
        or fallback.insurability_status_methodology_note
    ).strip()
    return {
        "insurability_status": status,
        "insurability_status_reasons": reasons,
        "insurability_status_methodology_note": methodology_note,
    }


def _status_shift(before_status: str, after_status: str) -> str:
    before_rank = _STATUS_SEVERITY.get(str(before_status or "").strip(), 1)
    after_rank = _STATUS_SEVERITY.get(str(after_status or "").strip(), 1)
    if after_rank < before_rank:
        return "improved"
    if after_rank > before_rank:
        return "worsened"
    return "unchanged"


def _delta(before_value: float | None, after_value: float | None) -> float | None:
    if before_value is None or after_value is None:
        return None
    return round(float(after_value) - float(before_value), 1)


def _action_drivers_from_changed_inputs(changed_input_keys: Iterable[str] | None) -> list[dict[str, str]]:
    keys = [str(v).strip().lower() for v in list(changed_input_keys or []) if str(v).strip()]
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in keys:
        action_pair = _CHANGED_INPUT_ACTIONS.get(key)
        if not action_pair:
            continue
        action, why = action_pair
        dedupe_key = action.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(
            {
                "action": action,
                "why_this_matters": why,
                "source": "changed_input_mapping",
                "input_key": key,
            }
        )
        if len(rows) >= 3:
            break
    return rows


def _action_drivers_from_assessment(result: AssessmentResult, *, limit: int = 3) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in list(result.prioritized_mitigation_actions or []):
        action = str(getattr(row, "action", "") or "").strip()
        if not action:
            continue
        dedupe_key = action.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        why = str(
            getattr(row, "why_this_matters", "")
            or getattr(row, "what_it_reduces", "")
            or getattr(row, "explanation", "")
            or ""
        ).strip()
        rows.append(
            {
                "action": action,
                "why_this_matters": (
                    why
                    if why
                    else "This action targets one of the leading wildfire risk factors for this property."
                ),
                "source": "post_change_prioritized_actions",
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    if rows:
        return rows
    fallback_actions = [str(v).strip() for v in list(result.top_recommended_actions or []) if str(v).strip()]
    for action in fallback_actions[: max(1, int(limit))]:
        rows.append(
            {
                "action": action,
                "why_this_matters": "This is a high-priority risk-reduction step for this property.",
                "source": "post_change_top_recommended_actions",
            }
        )
    return rows


def build_homeowner_before_after_summary(
    *,
    before: AssessmentResult,
    after: AssessmentResult,
    scenario_name: str | None = None,
    changed_input_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    before_risk = _safe_float(before.wildfire_risk_score)
    after_risk = _safe_float(after.wildfire_risk_score)
    before_hardening = _safe_float(
        before.home_hardening_readiness
        if before.home_hardening_readiness is not None
        else before.insurance_readiness_score
    )
    after_hardening = _safe_float(
        after.home_hardening_readiness
        if after.home_hardening_readiness is not None
        else after.insurance_readiness_score
    )
    risk_delta = _delta(before_risk, after_risk)
    hardening_delta = _delta(before_hardening, after_hardening)

    before_status = _insurability_snapshot(before)
    after_status = _insurability_snapshot(after)
    current_status = str(before_status.get("insurability_status") or "").strip()
    projected_status = str(after_status.get("insurability_status") or "").strip()
    status_shift = _status_shift(current_status, projected_status)

    action_drivers = _action_drivers_from_changed_inputs(changed_input_keys)
    if len(action_drivers) < 3:
        for row in _action_drivers_from_assessment(after, limit=3):
            action_key = str(row.get("action") or "").strip().lower()
            if not action_key:
                continue
            if any(str(existing.get("action") or "").strip().lower() == action_key for existing in action_drivers):
                continue
            action_drivers.append(row)
            if len(action_drivers) >= 3:
                break

    if status_shift == "improved":
        shift_phrase = f"Status improved from {current_status} to {projected_status}"
    elif status_shift == "worsened":
        shift_phrase = f"Status shifted from {current_status} to {projected_status}"
    else:
        shift_phrase = f"Status remained {projected_status or current_status or 'unchanged'}"

    risk_phrase = ""
    if risk_delta is not None:
        if risk_delta < 0:
            risk_phrase = f"wildfire risk score decreased by {abs(risk_delta):.1f} points"
        elif risk_delta > 0:
            risk_phrase = f"wildfire risk score increased by {risk_delta:.1f} points"
        else:
            risk_phrase = "wildfire risk score stayed flat"
    readiness_phrase = ""
    if hardening_delta is not None:
        if hardening_delta > 0:
            readiness_phrase = f"home hardening readiness improved by {hardening_delta:.1f} points"
        elif hardening_delta < 0:
            readiness_phrase = f"home hardening readiness declined by {abs(hardening_delta):.1f} points"
        else:
            readiness_phrase = "home hardening readiness stayed flat"
    parts = [shift_phrase]
    if risk_phrase:
        parts.append(risk_phrase)
    if readiness_phrase:
        parts.append(readiness_phrase)
    if action_drivers:
        parts.append(f"Top driver: {str(action_drivers[0].get('action') or '').strip()}")

    return {
        "available": True,
        "scenario_name": str(scenario_name or "what_if").strip() or "what_if",
        "current_insurability_status": current_status,
        "projected_insurability_status": projected_status,
        "status_shift": status_shift,
        "current_insurability_status_reasons": list(before_status.get("insurability_status_reasons") or [])[:3],
        "projected_insurability_status_reasons": list(after_status.get("insurability_status_reasons") or [])[:3],
        "insurability_status_methodology_note": str(
            after_status.get("insurability_status_methodology_note")
            or before_status.get("insurability_status_methodology_note")
            or ""
        ),
        "wildfire_risk_score_before": before_risk,
        "wildfire_risk_score_after": after_risk,
        "wildfire_risk_score_delta": risk_delta,
        "home_hardening_readiness_before": before_hardening,
        "home_hardening_readiness_after": after_hardening,
        "home_hardening_readiness_delta": hardening_delta,
        "confidence_tier_before": str(before.confidence_tier or ""),
        "confidence_tier_after": str(after.confidence_tier or ""),
        "top_actions_driving_change": action_drivers[:3],
        "summary": ". ".join([str(v).rstrip(".") for v in parts if str(v).strip()]) + ".",
    }


def build_improvement_change_set(
    before: AssessmentResult,
    after: AssessmentResult,
    *,
    changed_fields_hint: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
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

    changed: dict[str, Any] = {}
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


def build_improvement_why_it_matters(
    before: AssessmentResult,
    after: AssessmentResult,
    summary: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    confidence_change = summary.get("confidence_change") if isinstance(summary, dict) else {}
    if isinstance(confidence_change, dict):
        direction = str(confidence_change.get("score_direction") or "")
        if direction == "up":
            notes.append(
                "Confidence improved because the update reduced missing or inferred property details."
            )
        elif direction == "down":
            notes.append(
                "Confidence decreased because the updated inputs introduced more uncertainty."
            )
    specificity_change = summary.get("specificity_change") if isinstance(summary, dict) else {}
    if isinstance(specificity_change, dict) and bool(specificity_change.get("changed")):
        after_tier = str(specificity_change.get("after_tier") or "")
        if after_tier == "property_specific":
            notes.append(
                "Specificity improved to property-specific, so nearby-home differentiation is more reliable."
            )
        elif after_tier == "address_level":
            notes.append(
                "Specificity improved to address-level, reducing reliance on broad regional assumptions."
            )
    recommendation_changes = summary.get("recommendation_changes") if isinstance(summary, dict) else {}
    if isinstance(recommendation_changes, dict) and bool(recommendation_changes.get("changed")):
        notes.append(
            "Recommendations updated because the model now has better property evidence to rank actions."
        )

    before_prop_conf = float(before.property_data_confidence or 0.0)
    after_prop_conf = float(after.property_data_confidence or 0.0)
    if after_prop_conf > before_prop_conf and not any("Confidence improved" in n for n in notes):
        notes.append(
            "Property data confidence improved, which makes this assessment more property-specific."
        )

    score_direction = summary.get("score_direction") if isinstance(summary, dict) else {}
    if isinstance(score_direction, dict):
        direction = str(score_direction.get("direction") or "")
        if direction == "up":
            notes.append(
                "Risk increased after updates, indicating the newly provided details reveal higher exposure."
            )
        elif direction == "down":
            notes.append(
                "Risk decreased after updates, indicating the newly provided details support lower exposure."
            )

    if not notes:
        notes.append(
            "The update keeps results aligned with current evidence and clarifies how missing details affect this estimate."
        )
    return list(dict.fromkeys(notes))[:4]
