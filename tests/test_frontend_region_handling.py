from __future__ import annotations

from pathlib import Path


def _frontend_html() -> str:
    path = Path("frontend/public/index.html")
    return path.read_text(encoding="utf-8")


def test_frontend_has_uncovered_location_renderer() -> None:
    html = _frontend_html()
    assert "function renderUncoveredLocationState(detail)" in html
    assert "Location not yet prepared" in html
    assert "no_prepared_region_for_location" in html


def test_frontend_handles_structured_region_not_ready_errors() -> None:
    html = _frontend_html()
    assert "detail.region_not_ready === true" in html
    assert "detail.coverage_available === false" in html
    assert "fetchCoverageForAddress(" in html
    assert "renderUncoveredLocationState(enriched);" in html


def test_frontend_does_not_require_manual_region_selection() -> None:
    html = _frontend_html()
    assert "/risk/assess" in html
    assert "fetchCoverageForAddress(" in html
    assert "resolved_region_id" in html
    assert 'id="region_id"' not in html
    assert "outside the currently prepared region set" in html


def test_frontend_region_debug_metadata_is_dev_mode_only() -> None:
    html = _frontend_html()
    assert "window.WILDFIRE_DEBUG_MODE" in html
    assert 'id="regionDebugInline"' in html
    assert "Region: unsupported_location" in html
    assert "Region: ${regionText}" in html
    assert 'id="regionResolutionText"' in html
    assert 'id="geocodeDebugText"' in html
    assert "Geocode: status=" in html
    assert 'style="display:none;"' in html
