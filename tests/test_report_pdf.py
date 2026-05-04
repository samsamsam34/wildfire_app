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


# --- Pass 1 improvement tests ---


def _sample_report_with_new_fields(**overrides: object) -> dict[str, object]:
    base = _sample_report()
    base["insurability_status"] = "At Risk"
    base["insurability_status_reasons"] = [
        "Defensible space is insufficient",
        "Adjacent fuel pressure is very high",
    ]
    base["headline_risk_summary"] = "This property has elevated risk due to dense adjacent fuels."
    base["confidence_summary_text"] = "Environmental data is high confidence. Add home details for a complete assessment."
    base["specificity_summary"] = {"what_this_means": "Nearby homes may appear similar based on shared regional data.", "specificity_tier": "regional_estimate"}
    base["score_summary"] = dict(base.get("score_summary", {}))  # type: ignore[arg-type]
    base["score_summary"]["use_restriction"] = "not_for_underwriting_or_binding"  # type: ignore[index]
    base.update(overrides)
    return base


def test_insurability_status_appears_in_html() -> None:
    html_out = render_homeowner_report_html(_sample_report_with_new_fields())
    assert "At Risk" in html_out
    assert "Insurance Readiness" in html_out


def test_insurability_reasons_appear_in_html() -> None:
    html_out = render_homeowner_report_html(_sample_report_with_new_fields())
    assert "Defensible space is insufficient" in html_out
    assert "Adjacent fuel pressure is very high" in html_out


def test_headline_risk_summary_overrides_hardcoded_string() -> None:
    html_out = render_homeowner_report_html(_sample_report_with_new_fields())
    assert "This property has elevated risk due to dense adjacent fuels." in html_out
    assert "Several important improvements are needed" not in html_out


def test_confidence_summary_text_replaces_debug_line() -> None:
    html_out = render_homeowner_report_html(_sample_report_with_new_fields())
    assert "Environmental data is high confidence" in html_out
    assert "Data Confidence:" not in html_out


def test_use_restriction_appears_in_legal_footer() -> None:
    html_out = render_homeowner_report_html(_sample_report_with_new_fields())
    assert "not approved for underwriting or binding decisions" in html_out.lower()
    assert "use-restriction-note" in html_out


def test_score_label_home_ignition_vulnerability() -> None:
    html_out = render_homeowner_report_html(_sample_report())
    assert "Home Ignition Vulnerability" in html_out
    assert "Home Fire Vulnerability" not in html_out


def test_score_label_site_hazard() -> None:
    html_out = render_homeowner_report_html(_sample_report())
    assert "Site Hazard" in html_out
    assert "Site &amp; Landscape Hazard" not in html_out
    assert "Site & Landscape Hazard" not in html_out


def test_score_direction_indicators_on_all_bars() -> None:
    html_out = render_homeowner_report_html(_sample_report())
    assert html_out.count("higher = more risk") >= 2
    assert "higher is better" in html_out


def test_score_scale_note_present() -> None:
    html_out = render_homeowner_report_html(_sample_report())
    assert "0–100" in html_out or "0–100" in html_out
    assert "median" in html_out.lower()


def test_legacy_invisible_markers_removed_from_pdf() -> None:
    pdf_bytes = generate_homeowner_pdf(_sample_report())
    pdf_text = pdf_bytes.decode("latin-1", errors="replace")
    assert "layout-marker" not in pdf_text
    assert "compatibility_markers" not in pdf_text
    assert "Wildfire risk level:" not in pdf_text


# --- Pass 2 improvement tests ---


def _sample_report_pass2(**overrides: object) -> dict[str, object]:
    base = _sample_report()
    base["insurance_readiness_summary"] = {
        "readiness_blockers": [
            "Inspection required before underwriting decision",
            "Very high adjacent fuel proximity",
        ],
        "readiness_factors": [
            {"name": "adjacent_fuel_pressure", "status": "fail", "score_impact": -15.0, "detail": "Fuel very close."},
            {"name": "defensible_space_distance", "status": "watch", "score_impact": -8.0, "detail": "Space insufficient."},
        ],
    }
    base["what_to_do_first"] = {
        "action": "Clear debris within 30 feet of all structures",
        "why_it_matters": "Defensible space is the highest-impact single action.",
        "effort_level": "medium",
    }
    base["property_summary"] = {
        "address": "1355 Pattee Canyon Rd, Missoula, MT 59803",
        "latitude": 46.83000,
        "longitude": -113.98000,
    }
    base["defensible_space_summary"] = {
        "zone_findings": [
            {"zone": "0-5 ft", "finding": "No combustible materials observed."},
            {"zone": "5-30 ft", "finding": "Dense brush present, needs clearing."},
        ],
    }
    base.update(overrides)
    return base


def test_readiness_blockers_appear_in_html() -> None:
    html_out = render_homeowner_report_html(_sample_report_pass2())
    assert "Inspection required before underwriting decision" in html_out
    assert "Very high adjacent fuel proximity" in html_out
    assert "Insurance Readiness Flags" in html_out


def test_what_to_do_first_appears_in_html() -> None:
    html_out = render_homeowner_report_html(_sample_report_pass2())
    assert "Clear debris within 30 feet of all structures" in html_out
    assert "Start Here" in html_out
    assert "Defensible space is the highest-impact single action." in html_out


def test_score_impact_shown_on_driver_rows() -> None:
    html_out = render_homeowner_report_html(_sample_report_pass2())
    # Numeric pts badge should appear for at least one driver row
    assert "pts" in html_out


def test_action_baseline_context_shown() -> None:
    html_out = render_homeowner_report_html(_sample_report_pass2())
    assert "(current:" in html_out


def test_gps_coordinates_in_metadata() -> None:
    html_out = render_homeowner_report_html(_sample_report_pass2())
    assert "46.83000" in html_out
    assert "-113.98000" in html_out


def test_defensible_space_zones_appear() -> None:
    html_out = render_homeowner_report_html(_sample_report_pass2())
    assert "0-5 ft" in html_out
    assert "Dense brush present" in html_out


def test_redundant_action_table_removed() -> None:
    html_out = render_homeowner_report_html(_sample_report())
    assert "Est. Risk Reduction" not in html_out
