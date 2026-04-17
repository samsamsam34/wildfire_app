from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


INSURABILITY_STATUS_METHODOLOGY_NOTE = (
    "This insurability status is a rule-based interpretation of observable wildfire risk factors "
    "and property conditions from available data, not a guarantee of underwriting outcome."
)


_HIGH_RISK_STATUS = "High Risk of Insurance Issues"
_AT_RISK_STATUS = "At Risk"
_LIKELY_INSURABLE_STATUS = "Likely Insurable"
_ALLOWED_STATUSES = {
    _LIKELY_INSURABLE_STATUS,
    _AT_RISK_STATUS,
    _HIGH_RISK_STATUS,
}


@dataclass(frozen=True)
class InsurabilityStatusResult:
    insurability_status: str
    insurability_status_reasons: list[str]
    insurability_status_methodology_note: str = INSURABILITY_STATUS_METHODOLOGY_NOTE


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_clean(lines: Sequence[str], *, limit: int = 3) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = str(raw or "").strip()
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


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in tokens)


def derive_insurability_status(
    *,
    wildfire_risk_score: float | None,
    home_hardening_readiness: float | None,
    confidence_tier: str | None,
    assessment_specificity_tier: str | None,
    top_near_structure_risk_drivers: Sequence[str] | None = None,
    top_risk_drivers: Sequence[str] | None = None,
    defensible_space_analysis: dict[str, Any] | None = None,
    defensible_space_limitations_summary: Sequence[str] | None = None,
    readiness_blockers: Sequence[str] | None = None,
    scoring_status: str | None = None,
) -> InsurabilityStatusResult:
    # TODO(calibration): Replace fixed thresholds with empirical calibration tables
    # once enough observed outcomes are available by region/carrier profile.
    # TODO(calibration): Refit the point weights below to learned coefficients
    # from future underwriting-outcome validation data.
    risk = _to_float(wildfire_risk_score)
    readiness = _to_float(home_hardening_readiness)
    confidence = str(confidence_tier or "").strip().lower()
    specificity = str(assessment_specificity_tier or "").strip().lower()
    scoring = str(scoring_status or "").strip().lower()

    points = 0
    reasons: list[str] = []

    if risk is not None:
        if risk >= 70.0:
            points += 3
            reasons.append("Overall wildfire risk is in a high range for this property.")
        elif risk >= 55.0:
            points += 2
            reasons.append("Overall wildfire risk is elevated.")
        elif risk >= 40.0:
            points += 1
            reasons.append("Overall wildfire risk is moderate.")

    if readiness is not None:
        if readiness < 40.0:
            points += 3
            reasons.append("Home hardening readiness is low, which can create insurance friction.")
        elif readiness < 55.0:
            points += 2
            reasons.append("Home hardening readiness has notable gaps.")
        elif readiness < 70.0:
            points += 1
            reasons.append("Home hardening readiness could be improved.")

    near_structure_rows = [str(v).strip() for v in list(top_near_structure_risk_drivers or []) if str(v).strip()]
    near_structure_text = " ".join(near_structure_rows).lower()
    if near_structure_rows:
        points += 1
        reasons.append("Near-structure conditions are contributing to wildfire exposure.")
    if _contains_any(
        near_structure_text,
        (
            "0-5 ft",
            "immediate zone",
            "defensible space",
            "dense vegetation",
            "ember",
            "fuel",
            "wildland",
        ),
    ):
        points += 1
        reasons.append("Defensible-space or vegetation proximity signals indicate higher exposure near the home.")

    all_driver_rows = [str(v).strip() for v in list(top_risk_drivers or []) if str(v).strip()]
    if not near_structure_rows and _contains_any(
        " ".join(all_driver_rows).lower(),
        ("defensible space", "vegetation", "ember", "fuel", "wildland", "structure vulnerability"),
    ):
        points += 1
        reasons.append("Top wildfire drivers include near-home fuel or structure-vulnerability factors.")

    ds_summary_text = str((defensible_space_analysis or {}).get("summary") or "").strip().lower()
    ds_limit_rows = [str(v).strip() for v in list(defensible_space_limitations_summary or []) if str(v).strip()]
    if _contains_any(
        ds_summary_text + " " + " ".join(ds_limit_rows).lower(),
        ("limited", "insufficient", "unavailable", "missing", "partial", "point fallback", "fallback"),
    ):
        points += 1
        reasons.append("Defensible-space evidence is limited, so this status is conservative.")

    blocker_rows = [str(v).strip() for v in list(readiness_blockers or []) if str(v).strip()]
    if blocker_rows:
        points += 1
        reasons.append("Readiness blockers are present and should be addressed.")

    if scoring in {"insufficient_data_to_score", "insufficient_data"}:
        points += 2
        reasons.append("Some key property details are missing, so risk is treated conservatively.")

    if confidence in {"low", "preliminary"} or specificity in {"regional_estimate", "insufficient_data"}:
        points += 1
        reasons.append("Confidence/specificity is limited, so this should be treated as planning guidance.")

    high_risk_gate = (
        (risk is not None and risk >= 75.0)
        or (readiness is not None and readiness < 35.0)
        or (
            risk is not None
            and readiness is not None
            and risk >= 65.0
            and readiness < 50.0
        )
        or points >= 6
    )
    at_risk_gate = (
        points >= 3
        or (risk is not None and risk >= 45.0)
        or (readiness is not None and readiness < 70.0)
    )

    if high_risk_gate:
        status = _HIGH_RISK_STATUS
    elif at_risk_gate:
        status = _AT_RISK_STATUS
    else:
        status = _LIKELY_INSURABLE_STATUS

    final_reasons = _dedupe_clean(reasons, limit=3)
    if not final_reasons:
        if status == _LIKELY_INSURABLE_STATUS:
            final_reasons = ["Current wildfire and home-hardening signals appear comparatively favorable."]
        elif status == _AT_RISK_STATUS:
            final_reasons = ["Available evidence suggests meaningful wildfire-related insurance risk factors."]
        else:
            final_reasons = ["Available evidence indicates elevated wildfire-related insurance risk factors."]

    if status not in _ALLOWED_STATUSES:
        status = _AT_RISK_STATUS

    return InsurabilityStatusResult(
        insurability_status=status,
        insurability_status_reasons=final_reasons,
    )

