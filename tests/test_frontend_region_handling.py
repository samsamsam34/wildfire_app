from __future__ import annotations

from pathlib import Path


def _frontend_html() -> str:
    return Path("frontend/public/index.html").read_text(encoding="utf-8")


def test_frontend_has_results_and_map_sections() -> None:
    html = _frontend_html()
    assert "What you can do to reduce your risk" in html
    assert "Property Map" in html
    assert "defensible_space_rings" in html


def test_frontend_uses_expected_assessment_and_report_endpoints() -> None:
    html = _frontend_html()
    assert 'apiFetch("/risk/assess"' in html
    assert '/risk/reassess/${assessment.assessment_id}' in html
    assert '/report/${assessmentId}/homeowner' in html
    assert '/report/${assessmentId}/map' in html
    assert '/report/${assessmentId}/homeowner/pdf' in html


def test_frontend_includes_expected_home_details_api_fields() -> None:
    html = _frontend_html()
    for field_name in [
        "roof_type",
        "vent_type",
        "defensible_space_ft",
        "construction_year",
        "siding_type",
    ]:
        assert field_name in html
