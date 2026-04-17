from __future__ import annotations

from pathlib import Path


def _frontend_html() -> str:
    return Path("frontend/public/index.html").read_text(encoding="utf-8")


def test_frontend_avoids_unsafe_dom_html_injection_apis() -> None:
    html = _frontend_html()
    assert ".innerHTML" not in html
    assert "insertAdjacentHTML" not in html
    assert "outerHTML" not in html
    assert "document.write(" not in html


def test_frontend_includes_safe_rendering_helpers() -> None:
    html = _frontend_html()
    assert "function clearNode(" in html
    assert "function setListItems(" in html
    assert "function setPillSpans(" in html
    assert "function setSubmodelTableRows(" in html
