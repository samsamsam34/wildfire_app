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
    },
    {
        "input_key": "defensible_space_condition",
        "assessment_field": "defensible_space_ft",
        "label": "Defensible space condition",
        "prompt": "About how many feet around the home are kept mostly non-combustible?",
        "input_type": "number",
        "unit": "feet",
        "suggestion": "Add defensible space condition to improve near-structure ignition estimates.",
    },
)

_DEFENSIBLE_SPACE_CONDITION_TO_FT: dict[str, float] = {
    "poor": 5.0,
    "limited": 12.0,
    "moderate": 25.0,
    "good": 40.0,
    "excellent": 60.0,
}


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
        "suggestions": suggestions[:10],
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

    missing_key_inputs: list[str] = []
    suggestions: list[str] = []
    follow_ups: list[HomeownerFollowUpInput] = []
    for spec in _IMPROVEMENT_INPUT_SPECS:
        field = str(spec["assessment_field"])
        is_missing = field in strict_missing_fields or _is_missing_value(facts.get(field))
        if not is_missing:
            continue
        missing_key_inputs.append(str(spec["input_key"]))
        suggestions.append(str(spec["suggestion"]))
        follow_ups.append(
            HomeownerFollowUpInput(
                input_key=str(spec["input_key"]),
                assessment_field=field,
                label=str(spec["label"]),
                prompt=str(spec["prompt"]),
                input_type=str(spec["input_type"]),  # type: ignore[arg-type]
                options=[str(item) for item in (spec.get("options") or [])],
                unit=(str(spec["unit"]) if spec.get("unit") else None),
            )
        )

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
        improve_your_result_suggestions=suggestions[:10],
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
) -> tuple[dict[str, dict[str, Any]], list[str]]:
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

    return changed, notes
