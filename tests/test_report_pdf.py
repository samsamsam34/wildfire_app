from __future__ import annotations

from backend.report_pdf import generate_homeowner_pdf, prepare_template_context, render_homeowner_report_html


def _sample_report(
    *,
    score: float = 75.0,
    structural_confidence_tier: str = "moderate",
    include_details: bool = True,
    include_mitigations: bool = True,
) -> dict[str, object]:
    observed_data = []
    if include_details:
        observed_data = [
            "roof: wood_shake",
            "vent: unscreened",
            "defensible_space: 20 feet",
            "year built: 1985",
        ]

    top_recommended_actions = []
    prioritized_mitigation_actions = []
    mitigation_plan = []
    if include_mitigations:
        top_recommended_actions = [
            {
                "title": "Clear needles and leaves from roof and gutters",
                "priority": 1,
                "estimated_risk_reduction": 9,
                "why_this_matters": "Reduces ember ignition pathways on the home.",
                "estimated_cost_band": "Low",
                "timeline": "1-2 weekends",
            },
            {
                "title": "Create 30 feet of defensible space",
                "priority": 2,
                "estimated_risk_reduction": 12,
                "why_this_matters": "Lowers flame-contact and radiant heat exposure.",
                "estimated_cost_band": "Medium",
                "timeline": "2-4 weeks",
            },
            {
                "title": "Install ember-resistant vent screening",
                "priority": 3,
                "estimated_risk_reduction": 7,
                "why_this_matters": "Helps prevent ember intrusion into attic/crawlspace areas.",
                "estimated_cost_band": "Medium",
                "timeline": "1-2 days",
            },
        ]
        prioritized_mitigation_actions = list(top_recommended_actions)
        mitigation_plan = list(top_recommended_actions)

    return {
        "assessment_id": "12345678-90ab-cdef-1234-567890abcdef",
        "generated_at": "2026-04-30T12:00:00Z",
        "report_header": {"assessment_generated_at": "2026-04-30T12:00:00Z"},
        "property_summary": {"address": "1355 Pattee Canyon Rd, Missoula, MT 59803"},
        "score_summary": {
            "overall_wildfire_risk": score,
            "wildfire_risk_score": score,
            "site_hazard_score": 72.0,
            "home_ignition_vulnerability_score": 68.0,
            "home_hardening_readiness": 44.0,
        },
        "internal_calibration_debug": {
            "subscores": {
                "site_hazard_score": 72.0,
                "home_ignition_vulnerability_score": 68.0,
                "home_hardening_readiness": 44.0,
            }
        },
        "top_risk_drivers": [
            "Dense fuel and vegetation near the home",
            "Steep slope/topography amplification",
            "Historic fire history in this area",
        ],
        "top_risk_drivers_detailed": [
            {"factor": "fuel_proximity_risk", "explanation": "Fuel continuity is high near the parcel."},
            {"factor": "slope_topography_risk", "explanation": "Slope is steep in the 5-30ft and 30-100ft zones."},
        ],
        "defensible_space_summary": {
            "limitations": [
                "This area has burned 2 times since 1984. Most recently in 2017.",
            ]
        },
        "environmental_confidence_tier": "moderate",
        "structural_confidence_tier": structural_confidence_tier,
        "confidence_and_limitations": {"observed_data": observed_data, "limitations": ["Regional vegetation data used."]},
        "metadata": {
            "data_coverage_summary": {
                "coverage_note": "Local parcel geometry was available; national layers used for fuels and slope.",
                "layers_from_national_sources": ["fuel", "canopy", "dem", "mtbs"],
            }
        },
        "top_recommended_actions": top_recommended_actions,
        "prioritized_mitigation_actions": prioritized_mitigation_actions,
        "mitigation_plan": mitigation_plan,
        "structural_confidence_improvement_actions": [
            {
                "field_name": "roof_type",
                "display_label": "Roof material",
                "confidence_gain": 8,
                "why_it_matters": "Roof class strongly affects ember ignition potential.",
            },
            {
                "field_name": "vent_type",
                "display_label": "Vent screening",
                "confidence_gain": 7,
                "why_it_matters": "Vents are a common ember entry path in structure losses.",
            },
            {
                "field_name": "defensible_space_ft",
                "display_label": "Vegetation cleared around home",
                "confidence_gain": 6,
                "why_it_matters": "Clearance distance controls flame-contact pressure near structures.",
            },
            {
                "field_name": "construction_year",
                "display_label": "Year built",
                "confidence_gain": 4,
                "why_it_matters": "Code-era construction can materially shift vulnerability estimates.",
            },
        ],
    }


def test_pdf_bytes_validity() -> None:
    pdf_bytes = generate_homeowner_pdf(_sample_report())
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000


def test_risk_level_label_thresholds_in_html() -> None:
    html_high = render_homeowner_report_html(_sample_report(score=75))
    html_moderate = render_homeowner_report_html(_sample_report(score=25))
    html_critical = render_homeowner_report_html(_sample_report(score=85))
    assert "HIGH WILDFIRE RISK" in html_high
    assert "MODERATE WILDFIRE RISK" in html_moderate
    assert "CRITICAL WILDFIRE RISK" in html_critical


def test_risk_level_color_for_high_score() -> None:
    html_out = render_homeowner_report_html(_sample_report(score=75))
    assert "#ea580c" in html_out


def test_plain_english_roof_translation() -> None:
    html_out = render_homeowner_report_html(_sample_report(include_details=True))
    assert "Wood Shake" in html_out
    assert "wood_shake" not in html_out


def test_internal_field_name_leak_guard() -> None:
    html_out = render_homeowner_report_html(_sample_report())
    for forbidden in (
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "home_hardening_readiness",
        "whp_index",
        "fuel_model_index",
        "canopy_cover_index",
        "structural_confidence_tier",
    ):
        assert forbidden not in html_out


def test_missing_optional_fields_render_confidence_improvement_section() -> None:
    html_out = render_homeowner_report_html(
        _sample_report(
            include_details=False,
            include_mitigations=False,
            structural_confidence_tier="not_assessed",
        )
    )
    assert "Adding the following details would improve your assessment:" in html_out
    assert "Detail to Add" in html_out


def test_mitigation_actions_show_priority_and_impact() -> None:
    html_out = render_homeowner_report_html(_sample_report(include_mitigations=True))
    assert "IMMEDIATE" in html_out
    assert "HIGH" in html_out
    assert "MEDIUM" in html_out
    assert "-9.0 points" in html_out
    assert "-12.0 points" in html_out
    assert "-7.0 points" in html_out


def test_driver_rows_include_fuel_and_slope_descriptions() -> None:
    context = prepare_template_context(_sample_report())
    descriptions = [row["description"] for row in context["driver_rows"]]
    assert any("vegetation surrounds the property" in text.lower() for text in descriptions)
    assert any("steep terrain accelerates fire spread" in text.lower() for text in descriptions)


def test_confidence_improvement_actions_use_display_labels() -> None:
    html_out = render_homeowner_report_html(_sample_report(include_details=False))
    assert "Roof material" in html_out
    assert "Vent screening" in html_out
    assert "Vegetation cleared around home" in html_out
    assert "roof_type" not in html_out
    assert "vent_type" not in html_out


def test_pdf_file_size_ceiling() -> None:
    pdf_bytes = generate_homeowner_pdf(_sample_report())
    assert len(pdf_bytes) < 600_000
