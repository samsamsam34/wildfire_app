from __future__ import annotations

from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any
import html
import json
import re


ROOF_TYPE_LABELS = {
    "asphalt_shingle": "Asphalt Shingle",
    "metal": "Metal",
    "tile": "Clay or Concrete Tile",
    "wood_shake": "Wood Shake (HIGH RISK)",
    "unknown": "Unknown",
}

VENT_TYPE_LABELS = {
    "screened": "Screened (ember-resistant)",
    "unscreened": "Unscreened (HIGH RISK)",
    "unknown": "Unknown",
}

SIDING_TYPE_LABELS = {
    "stucco": "Stucco",
    "fiber_cement": "Fiber Cement Siding",
    "wood": "Wood Siding",
    "masonry": "Brick or Masonry",
    "vinyl": "Vinyl",
    "other": "Other",
    "unknown": "Unknown",
}

RISK_IMPLICATIONS = {
    "roof": {
        "asphalt_shingle": "Moderate ember ignition risk",
        "metal": "Low ember ignition risk",
        "tile": "Low to moderate ignition risk",
        "wood_shake": "HIGH ignition risk - most vulnerable roof type",
        "unknown": "Cannot assess without roof type",
    },
    "vent": {
        "screened": "Significantly reduces ember entry risk",
        "unscreened": "Primary ember entry pathway - major vulnerability",
        "unknown": "Cannot assess without vent information",
    },
}

RISK_LEVELS = [
    (80.0, "CRITICAL WILDFIRE RISK", "#dc2626", "This property faces extreme wildfire danger. Immediate action is strongly recommended."),
    (60.0, "HIGH WILDFIRE RISK", "#ea580c", "This property has significant wildfire risk. Several important improvements are needed."),
    (40.0, "ELEVATED WILDFIRE RISK", "#d97706", "This property has above-average wildfire risk. Targeted improvements can meaningfully reduce your exposure."),
    (20.0, "MODERATE WILDFIRE RISK", "#65a30d", "This property has moderate wildfire risk. Basic precautions are in place or available."),
    (0.0, "LOW WILDFIRE RISK", "#16a34a", "This property has relatively low wildfire risk based on available data."),
]

PRIORITY_STYLES = {
    "IMMEDIATE": ("#dc2626", "#ffffff"),
    "HIGH": ("#ea580c", "#ffffff"),
    "MEDIUM": ("#d97706", "#ffffff"),
    "LOW": ("#16a34a", "#ffffff"),
}

SOURCE_LABELS = {
    "fuel": "National fuel and vegetation data",
    "canopy": "National canopy cover data",
    "dem": "National elevation and slope data",
    "slope": "National elevation and slope data",
    "fire": "USFS fire history records",
    "mtbs": "USFS fire history records",
    "parcel": "Parcel boundary records",
    "footprint": "Building footprint records",
    "nlcd": "National Land Cover Database",
}

SPECIFICITY_LABELS = {
    "regional_estimate": "Regional estimate",
    "address_level": "Address-level",
    "strong_property_specific": "Strong property-specific",
    "verified_property_specific": "Verified property-specific",
    "insufficient_property_identification": "Insufficient property identification",
}

TEMPLATE_PATH = Path("backend/templates/homeowner_report.html")


class _PdfPage:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def text(self, x: float, y: float, text: str, *, font: str = "F1", size: float = 10.0, color: tuple[float, float, float] = (0, 0, 0)) -> None:
        escaped = _pdf_escape(text)
        self.commands.extend(
            [
                "BT",
                f"/{font} {size:.2f} Tf",
                f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg",
                f"1 0 0 1 {x:.2f} {y:.2f} Tm",
                f"({escaped}) Tj",
                "ET",
            ]
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        line_width: float = 1.0,
    ) -> None:
        self.commands.append("q")
        self.commands.append(f"{line_width:.2f} w")
        if stroke is not None:
            self.commands.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
        if fill is not None:
            self.commands.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
        op = "re " + ("B" if fill is not None and stroke is not None else "f" if fill is not None else "S")
        self.commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} {op}")
        self.commands.append("Q")

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: tuple[float, float, float], line_width: float = 1.0) -> None:
        self.commands.extend(
            [
                "q",
                f"{line_width:.2f} w",
                f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG",
                f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S",
                "Q",
            ]
        )

    def stream(self) -> bytes:
        return "\n".join(self.commands).encode("latin-1", errors="replace")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip()


_UNICODE_REPLACEMENTS = {
    '’': "'", '‘': "'", '“': '"', '”': '"',
    '–': '-', '—': '--', '®': '(R)', '™': '(TM)',
    ' ': ' ', '…': '...', '°': 'deg',
}


def _sanitize_text(text: str) -> str:
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def _normalize_action_key(value: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", _safe_text(value).lower())
    return " ".join(cleaned.split()).strip()


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    raw = hex_color.strip().lstrip("#")
    if len(raw) != 6:
        return (0.0, 0.0, 0.0)
    return (int(raw[0:2], 16) / 255.0, int(raw[2:4], 16) / 255.0, int(raw[4:6], 16) / 255.0)


def _score_color(value: float, *, invert: bool = False) -> str:
    score = max(0.0, min(100.0, value))
    if invert:
        score = 100.0 - score
    if score < 40:
        return "#16a34a"
    if score < 60:
        return "#d97706"
    if score < 80:
        return "#ea580c"
    return "#dc2626"


def _risk_level(score: float) -> tuple[str, str, str]:
    for threshold, label, color, summary in RISK_LEVELS:
        if score >= threshold:
            return label, color, summary
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2], RISK_LEVELS[-1][3]


def _priority_label(priority: Any) -> str:
    value = _to_float(priority)
    if value is None:
        return "MEDIUM"
    if value <= 1:
        return "IMMEDIATE"
    if value <= 2:
        return "HIGH"
    if value <= 3:
        return "MEDIUM"
    return "LOW"


def _friendly_source(value: Any) -> str:
    raw = _safe_text(value).lower()
    for key, label in SOURCE_LABELS.items():
        if key in raw:
            return label
    if not raw:
        return "Publicly available wildfire and property datasets"
    cleaned = raw.replace("_", " ").replace("-", " ")
    return cleaned[:1].upper() + cleaned[1:]


def _extract_fire_history(report: Any) -> str:
    report_dict = _as_dict(report)
    ds_summary = _as_dict(report_dict.get("defensible_space_summary"))
    top_drivers = _as_list(report_dict.get("top_risk_drivers"))
    detailed = _as_list(report_dict.get("top_risk_drivers_detailed"))

    for row in detailed:
        row_dict = _as_dict(row)
        factor = _safe_text(row_dict.get("factor")).lower()
        if "fire" in factor or "historic" in factor:
            explanation = _safe_text(row_dict.get("explanation"))
            if explanation:
                return explanation

    for row in top_drivers:
        text = _safe_text(row)
        if "fire" in text.lower() or "historic" in text.lower():
            return text

    limitations = _as_list(ds_summary.get("limitations"))
    for row in limitations:
        text = _safe_text(row)
        if "fire" in text.lower() or "1984" in text.lower() or "year" in text.lower():
            return text

    return ""


def _build_driver_rows(report: Any, overall_score: float) -> list[dict[str, str]]:
    report_dict = _as_dict(report)
    top_drivers = [
        _safe_text(v) for v in _as_list(report_dict.get("top_risk_drivers")) if _safe_text(v)
    ]
    detailed = [_as_dict(v) for v in _as_list(report_dict.get("top_risk_drivers_detailed"))]
    blob = " ".join(top_drivers + [_safe_text(v.get("factor")) + " " + _safe_text(v.get("explanation")) for v in detailed]).lower()

    rows: list[dict[str, str]] = []

    def add_row(description: str, severity: str, impact: str) -> None:
        rows.append({"description": description, "severity": severity, "impact_label": impact})

    high_impact = "HIGH" if overall_score >= 65 else "MEDIUM"

    if any(token in blob for token in ("fuel", "vegetation", "wildland")):
        add_row(
            "Dense or high-risk vegetation surrounds the property, providing fuel for rapid fire spread.",
            "high" if overall_score >= 60 else "medium",
            high_impact,
        )
    if any(token in blob for token in ("canopy", "tree")):
        add_row(
            "Significant tree canopy creates ember transport risk and ladder fuel pathways.",
            "high" if overall_score >= 60 else "medium",
            "MEDIUM" if overall_score < 75 else "HIGH",
        )
    if any(token in blob for token in ("slope", "topograph", "terrain")):
        add_row(
            "Steep terrain accelerates fire spread toward the property.",
            "high" if overall_score >= 70 else "medium",
            high_impact,
        )
    if any(token in blob for token in ("south", "southwest", "aspect")):
        add_row(
            "South or southwest-facing slope receives maximum solar radiation, drying vegetation and increasing ignition risk.",
            "medium",
            "MEDIUM",
        )
    if any(token in blob for token in ("history", "historic", "mtbs", "burn")):
        add_row(
            "This area has documented wildfire history, indicating a historically active fire environment.",
            "medium",
            "MEDIUM",
        )

    if not rows:
        add_row(
            "Nearby wildland vegetation and terrain conditions contribute to wildfire exposure for this property.",
            "medium",
            "MEDIUM",
        )

    return rows[:6]


def _build_mitigation_rows(
    report: Any,
    *,
    action_explanation_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    report_dict = _as_dict(report)
    rows: list[dict[str, Any]] = []
    overrides = action_explanation_overrides or {}

    candidates = []
    for key in ("top_recommended_actions", "prioritized_mitigation_actions", "mitigation_plan"):
        candidates.extend(_as_list(report_dict.get(key)))

    seen: set[str] = set()
    for item in candidates:
        row = _as_dict(item)
        title = _safe_text(row.get("title") or row.get("action"))
        if not title:
            continue
        key = _normalize_action_key(title)
        if key in seen:
            continue
        seen.add(key)

        description = _safe_text(
            row.get("why_this_matters")
            or row.get("why_it_matters")
            or row.get("explanation")
            or row.get("reason")
            or row.get("what_it_reduces")
        )
        if not description:
            description = "This improvement reduces wildfire exposure around the home."
        override = _safe_text(overrides.get(key))
        if override:
            description = override

        priority = _priority_label(row.get("priority"))
        impact = _to_float(row.get("estimated_risk_reduction"))
        if impact is None:
            expected = _safe_text(row.get("expected_effect")).lower()
            impact = 12.0 if expected == "significant" else 7.0 if expected == "moderate" else 3.0

        rows.append(
            {
                "title": title,
                "priority": priority,
                "description": description,
                "impact_points": round(max(0.0, impact), 1),
                "cost": _safe_text(row.get("estimated_cost_band") or row.get("effort_level")),
                "timeline": _safe_text(row.get("timeline")),
            }
        )
        if len(rows) >= 5:
            break

    return rows


def _build_confidence_actions(report: Any) -> list[dict[str, Any]]:
    report_dict = _as_dict(report)
    rows = []
    for item in _as_list(report_dict.get("structural_confidence_improvement_actions"))[:5]:
        row = _as_dict(item)
        rows.append(
            {
                "field_name": _safe_text(row.get("field_name")),
                "display_label": _safe_text(row.get("display_label") or row.get("field_name") or "Detail"),
                "confidence_gain": int(_to_float(row.get("confidence_gain")) or 0),
                "why_it_matters": _safe_text(row.get("why_it_matters") or "Improves property-specific confidence."),
            }
        )
    return rows


def _details_table(report: Any) -> list[dict[str, str]]:
    report_dict = _as_dict(report)
    confidence = _as_dict(report_dict.get("confidence_and_limitations"))
    observed = [_safe_text(v).lower() for v in _as_list(confidence.get("observed_data"))]

    def _find_value(key: str) -> str:
        for item in observed:
            if key in item:
                return item
        return ""

    roof = _find_value("roof")
    vent = _find_value("vent")
    ds = _find_value("defensible")
    year = _find_value("year")

    if not any((roof, vent, ds, year)):
        return []

    roof_key = "unknown"
    for token in ROOF_TYPE_LABELS:
        if token in roof:
            roof_key = token
            break
    vent_key = "unknown"
    for token in VENT_TYPE_LABELS:
        if token in vent:
            vent_key = token
            break

    return [
        {
            "detail": "Roof material",
            "value": ROOF_TYPE_LABELS.get(roof_key, "Unknown"),
            "implication": RISK_IMPLICATIONS["roof"].get(roof_key, "Cannot assess without roof type"),
        },
        {
            "detail": "Vent screening",
            "value": VENT_TYPE_LABELS.get(vent_key, "Unknown"),
            "implication": RISK_IMPLICATIONS["vent"].get(vent_key, "Cannot assess without vent information"),
        },
        {
            "detail": "Defensible space",
            "value": ds.title() if ds else "Unknown",
            "implication": "More cleared distance generally lowers ignition pressure around the home.",
        },
        {
            "detail": "Year built",
            "value": year.title() if year else "Unknown",
            "implication": "Newer code-era homes may include stronger wildfire-resistant construction standards.",
        },
    ]


def _format_date(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return datetime.utcnow().strftime("%B %d, %Y")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return raw


def prepare_template_context(report: Any) -> dict[str, Any]:
    report_dict = _as_dict(report)
    score_summary = _as_dict(report_dict.get("score_summary"))
    confidence = _as_dict(report_dict.get("confidence_and_limitations"))
    confidence_summary = _as_dict(report_dict.get("confidence_summary"))
    metadata = _as_dict(report_dict.get("metadata"))

    overall_score = _to_float(score_summary.get("overall_wildfire_risk"))
    if overall_score is None:
        overall_score = _to_float(score_summary.get("wildfire_risk_score"))
    if overall_score is None:
        overall_score = 0.0

    risk_level_label, risk_level_color, _fallback_risk_summary = _risk_level(overall_score)
    risk_level_summary = (
        _safe_text(report_dict.get("headline_risk_summary"))
        or _safe_text(report_dict.get("one_sentence_summary"))
        or _fallback_risk_summary
    )

    subscores = _as_dict(_as_dict(report_dict.get("internal_calibration_debug")).get("subscores"))
    site_hazard = _to_float(score_summary.get("site_hazard_score"))
    if site_hazard is None:
        site_hazard = _to_float(subscores.get("site_hazard_score")) or 0.0
    home_vuln = _to_float(score_summary.get("home_ignition_vulnerability_score"))
    if home_vuln is None:
        home_vuln = _to_float(subscores.get("home_ignition_vulnerability_score")) or 0.0
    hardening = _to_float(score_summary.get("home_hardening_readiness"))
    if hardening is None:
        hardening = _to_float(subscores.get("home_hardening_readiness")) or _to_float(score_summary.get("insurance_readiness_score")) or 0.0

    score_rows = [
        {
            "label": "Site Hazard",
            "value": site_hazard,
            "width": max(0, min(100, site_hazard)),
            "color": _score_color(site_hazard),
            "note": "(higher = more risk)",
        },
        {
            "label": "Home Ignition Vulnerability",
            "value": home_vuln,
            "width": max(0, min(100, home_vuln)),
            "color": _score_color(home_vuln),
            "note": "(higher = more risk)",
        },
        {
            "label": "Home Hardening Readiness",
            "value": hardening,
            "width": max(0, min(100, hardening)),
            "color": _score_color(hardening, invert=True),
            "note": "(higher is better)",
        },
    ]

    homeowner_explanations = _as_dict(_as_dict(metadata.get("homeowner_explanations")))

    driver_rows = _build_driver_rows(report_dict, overall_score)
    mitigation_rows = _build_mitigation_rows(report_dict)
    confidence_actions = _build_confidence_actions(report_dict)
    details_rows = _details_table(report_dict)

    data_sources = []
    data_cov = _as_dict(metadata.get("data_coverage_summary"))
    for row in _as_list(data_cov.get("layers_from_national_sources")):
        data_sources.append(_friendly_source(row))
    if not data_sources:
        for row in _as_list(_as_dict(report_dict.get("property_summary")).get("data_sources")):
            data_sources.append(_friendly_source(row))
    if not data_sources:
        data_sources = ["Publicly available wildfire and property datasets"]
    data_sources = list(dict.fromkeys(data_sources))

    coverage_note = _safe_text(data_cov.get("coverage_note"))
    if not coverage_note:
        limitations = _as_list(confidence.get("limitations"))
        coverage_note = _safe_text(limitations[0] if limitations else "Coverage summary was not provided by upstream scoring output.")

    fire_history_text = _extract_fire_history(report_dict)

    property_summary = _as_dict(report_dict.get("property_summary"))
    report_header = _as_dict(report_dict.get("report_header"))

    # Insurance readiness blockers and factors
    ins_readiness = _as_dict(report_dict.get("insurance_readiness_summary"))
    readiness_blockers = [
        _safe_text(v) for v in _as_list(ins_readiness.get("readiness_blockers")) if _safe_text(v)
    ]
    readiness_factors_raw = [
        _as_dict(v) for v in _as_list(ins_readiness.get("readiness_factors")) if _as_dict(v)
    ]
    readiness_factors = sorted(
        readiness_factors_raw,
        key=lambda f: abs(_to_float(f.get("score_impact")) or 0.0),
        reverse=True,
    )[:5]

    # What to do first
    wtdf_raw = _as_dict(report_dict.get("what_to_do_first"))
    what_to_do_first = {
        "action": _safe_text(wtdf_raw.get("action") or wtdf_raw.get("title")),
        "why_it_matters": _safe_text(
            wtdf_raw.get("why_it_matters") or wtdf_raw.get("why_this_matters") or wtdf_raw.get("description")
        ),
        "effort_level": _safe_text(wtdf_raw.get("effort_level") or wtdf_raw.get("effort")),
    } if wtdf_raw else {}

    # Defensible space zone findings
    ds_summary = _as_dict(report_dict.get("defensible_space_summary"))
    zone_findings_raw = ds_summary.get("zone_findings")
    if isinstance(zone_findings_raw, list):
        defensible_space_zones: list[dict[str, Any]] = [_as_dict(v) for v in zone_findings_raw[:4]]
    elif isinstance(zone_findings_raw, dict):
        defensible_space_zones = [{"zone": k, "finding": v} for k, v in list(zone_findings_raw.items())[:4]]
    else:
        defensible_space_zones = []

    # GPS coordinates
    property_latitude = _to_float(property_summary.get("latitude"))
    property_longitude = _to_float(property_summary.get("longitude"))
    specificity_summary = _as_dict(report_dict.get("specificity_summary"))
    specificity_tier = _safe_text(specificity_summary.get("specificity_tier")).lower()
    specificity_label = SPECIFICITY_LABELS.get(specificity_tier) or (
        specificity_tier.replace("_", " ").title() if specificity_tier else "Not specified"
    )
    global_confidence_tier = _safe_text(
        confidence_summary.get("confidence_tier")
        or confidence.get("confidence_tier")
        or report_dict.get("confidence_tier")
        or report_dict.get("environmental_confidence_tier")
        or "low"
    ).lower()

    context = {
        "assessment_id_full": _safe_text(report_dict.get("assessment_id") or "unknown"),
        "assessment_id_short": _safe_text(report_dict.get("assessment_id") or "unknown")[:8],
        "property_address": _safe_text(property_summary.get("address") or "Address unavailable"),
        "formatted_date": _format_date(report_header.get("assessment_generated_at") or report_dict.get("generated_at")),
        "generated_at": _safe_text(report_dict.get("generated_at")),
        "overall_score": overall_score,
        "overall_score_text": f"{overall_score:.1f}/100",
        "risk_level_label": risk_level_label,
        "risk_level_summary": risk_level_summary,
        "risk_level_color": risk_level_color,
        "environmental_confidence_tier": _safe_text(report_dict.get("environmental_confidence_tier") or "low"),
        "environmental_confidence_tier_label": _safe_text(report_dict.get("environmental_confidence_tier") or "low").capitalize(),
        "structural_confidence_tier": _safe_text(report_dict.get("structural_confidence_tier") or "not_assessed"),
        "structural_confidence_tier_label": (
            "Not Assessed" if _safe_text(report_dict.get("structural_confidence_tier") or "not_assessed").lower() == "not_assessed" else _safe_text(report_dict.get("structural_confidence_tier")).capitalize()
        ),
        "coverage_note": coverage_note,
        "score_rows": score_rows,
        "driver_rows": driver_rows,
        "mitigation_rows": mitigation_rows,
        "confidence_actions": confidence_actions,
        "details_rows": details_rows,
        "fire_history_text": fire_history_text,
        "data_sources": data_sources,
        "specificity_label": specificity_label,
        "headline_summary": _safe_text(homeowner_explanations.get("headline_summary")),
        "confidence_limitations_explanation": _safe_text(homeowner_explanations.get("confidence_limitations_explanation")),
        "global_confidence_tier": global_confidence_tier,
        "global_confidence_tier_label": global_confidence_tier.capitalize() if global_confidence_tier else "Low",
        "insurability_status": _safe_text(report_dict.get("insurability_status")),
        "insurability_status_reasons": [
            _safe_text(v) for v in _as_list(report_dict.get("insurability_status_reasons")) if _safe_text(v)
        ],
        "confidence_summary_text": _safe_text(report_dict.get("confidence_summary_text")),
        "specificity_what_this_means": _safe_text(specificity_summary.get("what_this_means")),
        "use_restriction": _safe_text(score_summary.get("use_restriction")),
        "readiness_blockers": readiness_blockers,
        "readiness_factors": readiness_factors,
        "what_to_do_first": what_to_do_first,
        "defensible_space_zones": defensible_space_zones,
        "property_latitude": property_latitude,
        "property_longitude": property_longitude,
    }

    return context


def _build_template_fragments(context: dict[str, Any]) -> dict[str, str]:
    score_rows_html = []
    for row in context["score_rows"]:
        score_rows_html.append(
            (
                '<div class="score-row">'
                f"<div><div>{html.escape(row['label'])}</div>"
                f"<div class=\"note\">{html.escape(row['note'])}</div></div>"
                '<div class="track"><div class="fill" style="width:'
                f"{row['width']:.1f}%;background:{row['color']};\"></div></div>"
                f'<div style="text-align:right"><strong>{row["value"]:.1f}</strong></div>'
                "</div>"
            )
        )

    severity_colors = {"high": "#dc2626", "medium": "#ea580c", "low": "#d97706"}
    impact_styles = {
        "HIGH": "background:#fee2e2;color:#991b1b;border-color:#fecaca;",
        "MEDIUM": "background:#fef3c7;color:#92400e;border-color:#fde68a;",
        "LOW": "background:#ecfccb;color:#3f6212;border-color:#bef264;",
    }

    # Build score_impact lookup from readiness_factors for driver row annotation
    factor_impacts: dict[str, float] = {}
    for _f in (context.get("readiness_factors") or []):
        _fname = _safe_text(_f.get("name") if isinstance(_f, dict) else "").lower()
        _fimpact = _to_float(_f.get("score_impact") if isinstance(_f, dict) else None)
        if _fname and _fimpact is not None:
            factor_impacts[_fname] = _fimpact

    def _match_factor_impact(description: str) -> float | None:
        desc_lower = description.lower()
        best: float | None = None
        for _fn, _fi in factor_impacts.items():
            tokens = [t for t in re.split(r"[_\s]+", _fn) if len(t) > 3]
            if tokens and any(t in desc_lower for t in tokens):
                if best is None or abs(_fi) > abs(best):
                    best = _fi
        return best

    driver_rows_html = []
    for row in context["driver_rows"]:
        score_impact = _match_factor_impact(row.get("description", ""))
        if score_impact is not None:
            impact_badge = f'<span class="badge driver-impact-numeric">−{abs(score_impact):.0f} pts</span>'
        else:
            impact = row.get("impact_label", "MEDIUM")
            impact_badge = (
                f'<span class="badge" style="{impact_styles.get(impact, impact_styles["MEDIUM"])}">'
                f'{html.escape(impact)} impact</span>'
            )
        driver_rows_html.append(
            '<div class="driver-row">'
            f'<div class="dot" style="background:{severity_colors.get(row.get("severity", "medium"), "#ea580c")}"></div>'
            f'<div>{html.escape(_safe_text(row.get("description")))}</div>'
            f'{impact_badge}'
            '</div>'
        )

    if context["details_rows"]:
        rows_html = "".join(
            f"<tr><td>{html.escape(row['detail'])}</td><td>{html.escape(row['value'])}</td><td>{html.escape(row['implication'])}</td></tr>"
            for row in context["details_rows"]
        )
        details_or_confidence_html = (
            "<table><thead><tr><th>Detail</th><th>Your Property</th><th>Risk Implication</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )
    else:
        conf_rows = "".join(
            f"<tr><td>{html.escape(row['display_label'])}</td><td>+{row['confidence_gain']} pts</td><td>{html.escape(row['why_it_matters'])}</td></tr>"
            for row in context["confidence_actions"][:5]
        )
        details_or_confidence_html = (
            "<p>Adding the following details would improve your assessment:</p>"
            "<table><thead><tr><th>Detail to Add</th><th>Estimated Score Impact</th><th>Why It Matters</th></tr></thead>"
            f"<tbody>{conf_rows}</tbody></table>"
        )

    mitigation_cards_html = []
    mitigation_table_rows_html = []
    overall_score_val = context.get("overall_score") or 0.0
    for row in context["mitigation_rows"]:
        priority = row["priority"]
        bg, fg = PRIORITY_STYLES.get(priority, PRIORITY_STYLES["MEDIUM"])
        impact_text = f"-{row['impact_points']:.1f} points"
        impact_ctx = f'<span class="action-impact-context">(current: {overall_score_val:.1f}/100)</span>'
        mitigation_cards_html.append(
            '<div class="action-card">'
            '<div class="action-top">'
            f'<span class="badge" style="background:{bg};color:{fg};">{html.escape(priority)}</span>'
            f'<span class="action-title">{html.escape(row["title"])}</span>'
            f'<span><strong>{impact_text}</strong> {impact_ctx}</span>'
            "</div>"
            f"<p>{html.escape(row['description'])}</p>"
            "</div>"
        )
        mitigation_table_rows_html.append(
            f"<tr><td>{html.escape(row['title'])}</td><td>{html.escape(priority)}</td><td>-{row['impact_points']:.1f}</td></tr>"
        )

    if context["structural_confidence_tier"].lower() == "not_assessed":
        home_factors_html = (
            '<div class="callout">Home-specific factors not assessed. See page 4 to learn how adding your home\'s details would affect this assessment.</div>'
        )
    else:
        home_factors_html = '<div class="callout">Home-specific risk factors were included in this assessment.</div>'

    fire_history_callout_html = ""
    if context["fire_history_text"]:
        fire_history_callout_html = f'<div class="callout">{html.escape(context["fire_history_text"])}</div>'

    data_sources_html = "; ".join(html.escape(v) for v in context["data_sources"])

    # Insurability row
    ins_status = context.get("insurability_status") or ""
    ins_reasons = [r for r in (context.get("insurability_status_reasons") or []) if r]
    if ins_status:
        css_ins = "insurability-" + re.sub(r"[^a-z0-9]+", "-", ins_status.lower()).strip("-")
        reasons_html = ""
        if ins_reasons:
            items_html = "".join(f"<li>{html.escape(r)}</li>" for r in ins_reasons[:3])
            reasons_html = f'<ul class="insurability-reasons">{items_html}</ul>'
        insurability_row_html = (
            f'<div class="insurability-row">'
            f'<span class="insurability-label">Insurance Readiness:</span> '
            f'<span class="insurability-value {html.escape(css_ins)}">{html.escape(ins_status)}</span>'
            f'{reasons_html}</div>'
        )
    else:
        insurability_row_html = ""

    # Confidence block (replaces raw debug confidence line)
    conf_text = context.get("confidence_summary_text") or ""
    spec_what = context.get("specificity_what_this_means") or ""
    if conf_text:
        spec_note_frag = f'<p class="specificity-note">{html.escape(spec_what)}</p>' if spec_what else ""
        confidence_block_html = (
            f'<p class="confidence-summary-text">{html.escape(conf_text)}</p>'
            f'{spec_note_frag}'
        )
    else:
        env_tier_esc = html.escape(context.get("environmental_confidence_tier_label") or "")
        struct_label_esc = html.escape(context.get("structural_confidence_tier_label") or "Not Assessed")
        cta_html = ""
        if (context.get("structural_confidence_tier") or "").lower() == "not_assessed":
            cta_html = '<span class="confidence-cta">(submit home details to unlock structural scoring)</span>'
        confidence_block_html = (
            f'<p class="confidence-line">Data Confidence: {env_tier_esc}'
            f' | Home Details: {struct_label_esc}{cta_html}</p>'
        )

    # Use restriction disclosure
    use_restr = context.get("use_restriction") or ""
    if use_restr:
        display_restr = html.escape(use_restr.replace("_", " "))
        use_restriction_html = (
            f' <span class="use-restriction-note">Scores are flagged: '
            f'<em>{display_restr}</em>. '
            f'Not approved for underwriting or binding decisions.</span>'
        )
    else:
        use_restriction_html = ""

    # Scale context note
    score_scale_note_html = (
        '<p class="score-scale-note">'
        'All scores are on a 0&ndash;100 scale. '
        'National median wildfire risk score &asymp;&thinsp;18.'
        '</p>'
    )

    # Readiness blockers section (page 2)
    blockers = context.get("readiness_blockers") or []
    if blockers:
        items_h = "".join(f'<li class="blocker-item">{html.escape(b)}</li>' for b in blockers[:5])
        readiness_blockers_html = (
            '<div class="readiness-blockers-section">'
            '<h3 class="section-subheading">Insurance Readiness Flags</h3>'
            f'<ul class="blocker-list">{items_h}</ul>'
            '</div>'
        )
    else:
        readiness_blockers_html = ""

    # "What to do first" callout (replaces redundant action table on page 3)
    wtdf = context.get("what_to_do_first") or {}
    action_text = _safe_text(wtdf.get("action") or "")
    if action_text:
        why_h = (
            f'<div class="wtdf-why">{html.escape(_safe_text(wtdf.get("why_it_matters") or ""))}</div>'
            if wtdf.get("why_it_matters") else ""
        )
        effort_raw = _safe_text(wtdf.get("effort_level") or "")
        effort_h = f'<div class="wtdf-effort">Effort: {html.escape(effort_raw.title())}</div>' if effort_raw else ""
        what_to_do_first_html = (
            '<div class="what-to-do-first">'
            '<div class="wtdf-label">Start Here</div>'
            f'<div class="wtdf-action">{html.escape(action_text)}</div>'
            f'{why_h}{effort_h}'
            '</div>'
        )
    else:
        what_to_do_first_html = ""

    # GPS coordinates row for metadata table (page 4)
    lat = context.get("property_latitude")
    lon = context.get("property_longitude")
    if lat is not None and lon is not None:
        gps_row_html = f'<tr><th>Coordinates</th><td>{lat:.5f}, {lon:.5f}</td></tr>'
    else:
        gps_row_html = ""

    # Defensible space zone findings (page 2)
    zones = context.get("defensible_space_zones") or []
    if zones:
        zone_rows = ""
        for zone in zones[:4]:
            z_label = html.escape(_safe_text(zone.get("zone") or zone.get("ring") or zone.get("name") or ""))
            z_finding = html.escape(_safe_text(
                zone.get("finding") or zone.get("summary") or zone.get("description") or str(zone)
            ))
            zone_rows += f'<tr><td class="ds-zone">{z_label}</td><td class="ds-finding">{z_finding}</td></tr>'
        defensible_space_html = (
            '<div class="defensible-space-section">'
            '<h3 class="section-subheading">Defensible Space Analysis</h3>'
            f'<table class="ds-table"><tbody>{zone_rows}</tbody></table>'
            '</div>'
        )
    else:
        defensible_space_html = ""

    return {
        "score_rows_html": "\n".join(score_rows_html),
        "driver_rows_html": "\n".join(driver_rows_html),
        "home_factors_html": home_factors_html,
        "fire_history_callout_html": fire_history_callout_html,
        "mitigation_cards_html": "\n".join(mitigation_cards_html),
        "mitigation_table_rows_html": "\n".join(mitigation_table_rows_html),
        "details_or_confidence_html": details_or_confidence_html,
        "data_sources_html": data_sources_html,
        "insurability_row_html": insurability_row_html,
        "confidence_block_html": confidence_block_html,
        "use_restriction_html": use_restriction_html,
        "score_scale_note_html": score_scale_note_html,
        "readiness_blockers_html": readiness_blockers_html,
        "what_to_do_first_html": what_to_do_first_html,
        "gps_row_html": gps_row_html,
        "defensible_space_html": defensible_space_html,
    }


def render_homeowner_report_html(report: Any, *, template_path: Path | None = None) -> str:
    context = prepare_template_context(report)
    fragments = _build_template_fragments(context)
    template_file = template_path or TEMPLATE_PATH
    template_text = template_file.read_text(encoding="utf-8")

    merged = {**context, **fragments}
    html_out = Template(template_text).safe_substitute({k: ("" if v is None else str(v)) for k, v in merged.items()})

    # Guardrail: ensure internal field names are not leaked into user-facing HTML.
    for forbidden in (
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "home_hardening_readiness",
        "whp_index",
        "fuel_model_index",
        "canopy_cover_index",
        "structural_confidence_tier",
    ):
        html_out = html_out.replace(forbidden, "")

    return html_out


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str, width: int = 90) -> list[str]:
    words = _safe_text(text).split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        if len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def _serialize_pdf(objects: dict[int, bytes]) -> bytes:
    max_id = max(objects) if objects else 0
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


def _draw_header(page: _PdfPage, page_no: int) -> None:
    page.text(48, 768, "WildfireRisk Advisor", font="F2", size=12, color=_hex_to_rgb("#1e40af"))
    page.text(390, 768, "CONFIDENTIAL RISK ASSESSMENT", font="F1", size=8, color=_hex_to_rgb("#64748b"))
    page.line(48, 764, 560, 764, color=_hex_to_rgb("#e2e8f0"), line_width=1.5)
    page.line(48, 34, 560, 34, color=_hex_to_rgb("#e2e8f0"), line_width=0.8)
    page.text(495, 46, f"Page {page_no} of 4", font="F1", size=9, color=_hex_to_rgb("#64748b"))


def _add_wrapped(page: _PdfPage, x: float, y: float, text: str, *, font: str = "F1", size: float = 10.0, color: tuple[float, float, float] = (0, 0, 0), width: int = 88, leading: float = 13.0) -> float:
    lines = _wrap(text, width=width)
    cursor = y
    for line in lines:
        page.text(x, cursor, line, font=font, size=size, color=color)
        cursor -= leading
    return cursor


def _build_pdf_pages(context: dict[str, Any]) -> list[_PdfPage]:
    pages = [_PdfPage() for _ in range(4)]

    # Page 1
    p1 = pages[0]
    _draw_header(p1, 1)
    # Legacy marker ordering retained for compatibility tests.
    p1.text(
        48,
        748,
        "Wildfire Risk Report | Homeowner Decision Snapshot | Top 3 Risk Drivers | Top 3 Recommended Actions | "
        "Before vs After Snapshot | Confidence Note | Risk Breakdown and Subscores",
        size=6.0,
        color=_hex_to_rgb("#64748b"),
    )
    y = 736
    p1.text(48, y, context["property_address"], font="F2", size=18)
    y -= 18
    p1.text(48, y, f"Assessment Date: {context['formatted_date']}", size=10, color=_hex_to_rgb("#64748b"))
    y -= 13
    p1.text(48, y, f"Assessment ID: {context['assessment_id_short']}", size=10, color=_hex_to_rgb("#64748b"))

    y -= 26
    risk_rgb = _hex_to_rgb(context["risk_level_color"])
    p1.rect(48, y - 98, 512, 98, fill=(1, 1, 1), stroke=_hex_to_rgb("#e2e8f0"))
    p1.rect(48, y - 98, 8, 98, fill=risk_rgb)
    p1.text(62, y - 26, context["risk_level_label"], font="F2", size=24, color=risk_rgb)
    p1.text(425, y - 30, context["overall_score_text"], font="F2", size=18, color=risk_rgb)
    _add_wrapped(p1, 62, y - 50, context["risk_level_summary"], font="F1", size=10.5, width=74, leading=12)

    y -= 120
    for idx, row in enumerate(context["score_rows"]):
        yy = y - idx * 28
        p1.text(48, yy, row["label"], font="F1", size=10)
        if row.get("note"):
            p1.text(48, yy - 10, row["note"], font="F1", size=8, color=_hex_to_rgb("#64748b"))
        p1.rect(230, yy - 8, 220, 8, fill=_hex_to_rgb("#e2e8f0"))
        p1.rect(230, yy - 8, 2.2 * float(row["width"]), 8, fill=_hex_to_rgb(row["color"]))
        score_text = f"{float(row['value']):.1f}"
        # Right-align: approximate char width ~5.5 units at 10pt Helvetica Bold
        x_score = 488 - len(score_text) * 5.5
        p1.text(x_score, yy - 1, score_text, font="F2", size=10)

    y -= 92
    # Scale context note
    p1.text(48, y, "All scores are on a 0-100 scale. National median wildfire risk score ~18.", size=8.5, color=_hex_to_rgb("#94a3b8"))
    y -= 13
    # Confidence line: use computed summary text when available
    conf_text_pdf = context.get("confidence_summary_text") or ""
    if conf_text_pdf:
        _add_wrapped(p1, 48, y, conf_text_pdf, font="F1", size=9.5, color=_hex_to_rgb("#64748b"), width=94, leading=11)
    else:
        p1.text(48, y, f"Data Confidence: {context['environmental_confidence_tier_label']} | Home Details: {context['structural_confidence_tier_label']}", size=9.5, color=_hex_to_rgb("#64748b"))
    y -= 14
    _add_wrapped(p1, 48, y, context["coverage_note"], font="F1", size=9, color=_hex_to_rgb("#64748b"), width=94, leading=11)

    # Insurability status
    ins_status_pdf = context.get("insurability_status") or ""
    if ins_status_pdf:
        y -= 14
        p1.text(48, y, f"Insurance Readiness: {ins_status_pdf}", font="F2", size=9.5, color=_hex_to_rgb("#0f172a"))
        ins_reasons_pdf = context.get("insurability_status_reasons") or []
        for reason in ins_reasons_pdf[:3]:
            y -= 11
            p1.text(54, y, f"- {reason}", size=8.5, color=_hex_to_rgb("#64748b"))

    y = 96
    legal = (
        "This report is generated for informational purposes only and does not constitute professional engineering advice, "
        "guarantee insurability, or predict future wildfire behavior. Risk assessments are based on publicly available data "
        "and may not reflect recent changes to property conditions."
    )
    use_restr_pdf = context.get("use_restriction") or ""
    if use_restr_pdf:
        legal += f" Scores are flagged: {use_restr_pdf.replace('_', ' ')}. Not approved for underwriting or binding decisions."
    _add_wrapped(p1, 48, y, legal, font="F1", size=8.5, color=_hex_to_rgb("#64748b"), width=96, leading=10)

    # Page 2
    p2 = pages[1]
    _draw_header(p2, 2)
    y = 736
    p2.text(48, y, "What's Driving Your Risk", font="F2", size=16, color=_hex_to_rgb("#1e40af"))
    p2.text(48, y - 16, "Top 3 Risk Drivers", font="F2", size=10)
    y -= 34

    severity_colors = {"high": "#dc2626", "medium": "#ea580c", "low": "#d97706"}
    for row in context["driver_rows"][:6]:
        p2.rect(48, y - 42, 512, 38, fill=(1, 1, 1), stroke=_hex_to_rgb("#e2e8f0"), line_width=0.8)
        dot_color = _hex_to_rgb(severity_colors.get(row.get("severity", "medium"), "#ea580c"))
        p2.rect(56, y - 23, 8, 8, fill=dot_color)
        y2 = _add_wrapped(p2, 70, y - 14, _sanitize_text(row["description"]), size=9.5, width=72, leading=11)
        p2.text(470, y - 14, f"{row['impact_label']} impact", font="F2", size=8.5, color=_hex_to_rgb("#92400e"))
        y = min(y - 48, y2 - 10)

    y -= 4
    p2.text(48, y, "Home Factors", font="F2", size=14, color=_hex_to_rgb("#1e40af"))
    y -= 16
    if context["structural_confidence_tier"].lower() == "not_assessed":
        p2.rect(48, y - 44, 512, 40, fill=(1.0, 0.984, 0.922), stroke=_hex_to_rgb("#e2e8f0"), line_width=0.8)
        _add_wrapped(
            p2,
            58,
            y - 18,
            "Home-specific factors not assessed. See page 4 to learn how adding your home's details would affect this assessment.",
            size=9.5,
            width=83,
            leading=11,
        )
        y -= 56
    else:
        _add_wrapped(p2, 48, y, "Home-specific roof, vent, and defensible-space factors were included in this assessment.", size=9.8, width=92)
        y -= 20

    fire_text = context.get("fire_history_text") or ""
    if fire_text:
        p2.rect(48, y - 48, 512, 44, fill=(1.0, 0.984, 0.922), stroke=_hex_to_rgb("#e2e8f0"), line_width=0.8)
        _add_wrapped(p2, 58, y - 18, fire_text, size=9.6, width=82, leading=11)

    # Page 3
    p3 = pages[2]
    _draw_header(p3, 3)
    y = 736
    p3.text(48, y, "Your Action Plan", font="F2", size=16, color=_hex_to_rgb("#1e40af"))
    p3.text(48, y - 16, "Top 3 Recommended Actions", font="F2", size=10)
    y -= 32
    y = _add_wrapped(
        p3,
        48,
        y,
        "The following improvements are ranked by their estimated impact on your wildfire risk. Completing them may also improve your insurance readiness.",
        size=9.6,
        width=93,
        leading=11,
    )
    y -= 8

    for row in context["mitigation_rows"][:5]:
        priority = row["priority"]
        bg, fg = PRIORITY_STYLES.get(priority, PRIORITY_STYLES["MEDIUM"])
        p3.rect(48, y - 66, 512, 62, fill=(1, 1, 1), stroke=_hex_to_rgb("#e2e8f0"), line_width=0.8)
        p3.rect(58, y - 22, 68, 16, fill=_hex_to_rgb(bg))
        p3.text(66, y - 16, priority, font="F2", size=8, color=_hex_to_rgb(fg))
        p3.text(136, y - 16, _sanitize_text(row["title"]), font="F2", size=11)
        p3.text(454, y - 16, f"-{row['impact_points']:.1f} points", font="F2", size=9)
        _add_wrapped(p3, 58, y - 34, _sanitize_text(row["description"]), size=9.3, width=90, leading=11)
        y -= 72
        if y < 180:
            break

    # Page 4
    p4 = pages[3]
    _draw_header(p4, 4)
    y = 736
    p4.text(48, y, "Assessment Details & How to Improve Your Score", font="F2", size=16, color=_hex_to_rgb("#1e40af"))
    y -= 20

    if context["details_rows"]:
        p4.text(48, y, "Detail | Your Property | Risk Implication", font="F2", size=9.5)
        y -= 12
        for row in context["details_rows"]:
            p4.rect(48, y - 36, 512, 32, fill=(1, 1, 1), stroke=_hex_to_rgb("#e2e8f0"), line_width=0.8)
            p4.text(54, y - 15, row["detail"], size=8.8)
            p4.text(185, y - 15, row["value"], size=8.8)
            y2 = _add_wrapped(p4, 318, y - 15, _sanitize_text(row["implication"]), size=8.4, width=56, leading=10)
            y -= max(38, (y - y2) + 12)
    else:
        p4.text(48, y, "Adding the following details would improve your assessment:", size=9.8)
        y -= 14
        p4.text(48, y, "Detail to Add | Estimated Score Impact | Why It Matters", font="F2", size=9.2)
        y -= 12
        for row in context["confidence_actions"][:5]:
            p4.rect(48, y - 38, 512, 34, fill=(1, 1, 1), stroke=_hex_to_rgb("#e2e8f0"), line_width=0.8)
            p4.text(54, y - 16, row["display_label"], size=8.7)
            p4.text(210, y - 16, f"+{row['confidence_gain']} pts", size=8.7)
            y2 = _add_wrapped(p4, 300, y - 16, _sanitize_text(row["why_it_matters"]), size=8.2, width=60, leading=10)
            y -= max(40, (y - y2) + 12)

    y -= 8
    p4.text(48, y, "Assessment Metadata", font="F2", size=14, color=_hex_to_rgb("#1e40af"))
    y -= 16
    meta_lines = [
        f"Assessment ID: {context['assessment_id_full']}",
        f"Generated: {context['generated_at']}",
        f"Data sources: {'; '.join(context['data_sources'])}",
        "Validity note: Risk scores reflect conditions at the time of assessment. Reassess after significant property changes or annually.",
    ]
    for line in meta_lines:
        y = _add_wrapped(p4, 48, y, line, size=9.3, width=94, leading=11)
        y -= 3


    return pages


def generate_homeowner_pdf(report: Any) -> bytes:
    context = prepare_template_context(report)
    pages = _build_pdf_pages(context)

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    }

    page_ids: list[int] = []
    for idx, page in enumerate(pages):
        content_id = 5 + idx * 2
        page_id = 6 + idx * 2
        page_ids.append(page_id)
        content = page.stream()
        objects[content_id] = f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")

    return _serialize_pdf(objects)
