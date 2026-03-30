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
    return missing


def build_homeowner_improvement_options(result: AssessmentResult) -> HomeownerImprovementOptions:
    facts = result.property_facts if isinstance(result.property_facts, dict) else {}
    missing_fields = _collect_missing_input_fields(result)
    missing_key_inputs: list[str] = []
    suggestions: list[str] = []
    follow_ups: list[HomeownerFollowUpInput] = []
    for spec in _IMPROVEMENT_INPUT_SPECS:
        field = str(spec["assessment_field"])
        is_missing = field in missing_fields or _is_missing_value(facts.get(field))
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
    return HomeownerImprovementOptions(
        assessment_id=result.assessment_id,
        missing_key_inputs=missing_key_inputs,
        improve_your_result_suggestions=suggestions,
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
