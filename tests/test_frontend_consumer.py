from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        self.tags.append(tag)


HTML_PATH = Path("frontend/public/index.html")
MAIN_PATH = Path("backend/main.py")


def _frontend_html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _main_py() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def _script_block(html: str) -> str:
    match = re.search(r"<script type=\"module\">(.*?)</script>", html, flags=re.DOTALL)
    assert match, "Missing module script block"
    return match.group(1)


def test_index_html_has_no_obvious_syntax_breakage() -> None:
    html = _frontend_html()
    parser = _TagCollector()
    parser.feed(html)
    assert "html" in parser.tags
    assert "head" in parser.tags
    assert "body" in parser.tags

    script = _script_block(html)
    assert "createRoot(" in script
    assert script.count("{") == script.count("}")

    node = shutil.which("node")
    if node:
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as tmp:
            tmp.write(script)
            tmp_path = tmp.name
        result = subprocess.run([node, "--check", tmp_path], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_api_base_uses_window_api_base() -> None:
    html = _frontend_html()
    assert "window.API_BASE" in html


def test_required_cdns_present() -> None:
    html = _frontend_html()
    assert "https://esm.sh/react@18" in html
    assert "https://esm.sh/react-dom@18/client" in html
    assert "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" in html
    assert "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" in html
    assert "https://cdn.tailwindcss.com" in html


def test_esri_satellite_tile_url_present() -> None:
    html = _frontend_html()
    assert "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" in html


def test_reassess_endpoint_matches_main_route() -> None:
    html = _frontend_html()
    main_py = _main_py()
    assert "@app.post(\"/risk/reassess/{assessment_id}\"" in main_py
    assert "/risk/reassess/${assessment.assessment_id}" in html


def test_pdf_endpoint_matches_main_route() -> None:
    html = _frontend_html()
    main_py = _main_py()
    assert "@app.get(\"/report/{assessment_id}/homeowner/pdf\"" in main_py
    assert "/report/${assessmentId}/homeowner/pdf" in html


def test_risk_level_thresholds_and_labels_present() -> None:
    html = _frontend_html()
    for token in ["< 20", "< 40", "< 60", "< 80", "LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]:
        assert token in html


def test_confidence_improvement_actions_referenced() -> None:
    html = _frontend_html()
    assert "structural_confidence_improvement_actions" in html


def test_mobile_viewport_meta_present() -> None:
    html = _frontend_html()
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0" />' in html


def test_home_details_api_fields_match_models_contract() -> None:
    html = _frontend_html()
    # These are the PropertyAttributes keys in backend/models.py.
    for field_name in [
        "roof_type",
        "vent_type",
        "defensible_space_ft",
        "construction_year",
        "siding_type",
    ]:
        assert field_name in html


def test_overlay_z_index_above_leaflet() -> None:
    html = _frontend_html()
    # .overlay must sit above all Leaflet panes (max z-index 700) and above
    # the leaflet-container stacking context (z-index 400).
    assert "z-index: 1000" in html


def test_details_panel_z_index_above_overlay() -> None:
    html = _frontend_html()
    assert "z-index: 1001" in html


def test_close_button_has_aria_label() -> None:
    script = _script_block(_frontend_html())
    assert "Close home details panel" in script


def test_details_panel_has_dialog_role() -> None:
    script = _script_block(_frontend_html())
    assert 'role: "dialog"' in script
    assert '"aria-modal": "true"' in script


def test_rate_limit_error_has_specific_message() -> None:
    script = _script_block(_frontend_html())
    assert "rate_limit_exceeded" in script
    assert "wait a minute" in script.lower()


def test_network_error_has_specific_message() -> None:
    script = _script_block(_frontend_html())
    assert "network_error" in script
    assert "connection" in script.lower()


def test_load_result_extras_uses_allsettled() -> None:
    script = _script_block(_frontend_html())
    assert "Promise.allSettled" in script
    # The old Promise.all-based dual-fetch should be gone.
    assert script.count("apiFetch(`/report/${assessmentId}/homeowner`)") == 1


def test_reposition_pin_ui_present() -> None:
    script = _script_block(_frontend_html())
    assert "Reposition pin" in script
    assert "repositionMode" in script
    # Backend AddressRequest model field is property_anchor_point (not requested_property_anchor_point)
    assert "property_anchor_point" in script


def test_score_range_context_present() -> None:
    script = _script_block(_frontend_html())
    assert "0\u2013100" in script or "0-100" in script
    assert "national median" in script.lower()


def test_toast_feedback_on_reassess() -> None:
    script = _script_block(_frontend_html())
    assert "setToast" in script
    assert "Assessment updated" in script
