from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.version import (
    API_VERSION,
    FACTOR_SCHEMA_VERSION,
    PRODUCT_VERSION,
    build_model_governance,
    compare_model_governance,
    release_note_template,
)


def test_compare_model_governance_direct_match():
    left = build_model_governance(ruleset_version="1.0.0", region_data_version="region-a")
    right = build_model_governance(ruleset_version="1.0.0", region_data_version="region-a")
    compared = compare_model_governance(left, right)
    assert compared["directly_comparable"] is True
    assert compared["comparison_label"] == "directly_comparable"
    assert compared["differences"] == {}


def test_compare_model_governance_non_direct_when_scoring_differs():
    left = build_model_governance(ruleset_version="1.0.0", scoring_model_version="1.5.0")
    right = build_model_governance(ruleset_version="1.0.0", scoring_model_version="1.6.0")
    compared = compare_model_governance(left, right)
    assert compared["directly_comparable"] is False
    assert compared["comparison_label"] == "not_directly_comparable"
    assert "scoring_model_version" in compared["differences"]


def test_compare_model_governance_review_required_for_region_data_change():
    left = build_model_governance(ruleset_version="1.0.0", region_data_version="region-a")
    right = build_model_governance(ruleset_version="1.0.0", region_data_version="region-b")
    compared = compare_model_governance(left, right)
    assert compared["directly_comparable"] is False
    assert compared["comparison_label"] == "comparable_with_review"
    assert "region_data_version" in compared["review_differences"]


def test_print_model_versions_script_outputs_expected_fields():
    script = Path("scripts") / "print_model_versions.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--compact"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["product_version"] == PRODUCT_VERSION
    assert payload["api_version"] == API_VERSION
    assert payload["factor_schema_version"] == FACTOR_SCHEMA_VERSION


def test_check_version_consistency_script_passes():
    script = Path("scripts") / "check_version_consistency.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "passed" in proc.stdout.lower()


def test_release_note_template_contains_required_sections():
    rendered = release_note_template("0.10.1", "2026-03-09")
    assert "## [0.10.1] - 2026-03-09" in rendered
    assert "### Version changes" in rendered
    assert "### Reason" in rendered
    assert "### Expected effect on outputs" in rendered
    assert "### Migration/interpretation notes" in rendered
    assert "### Historical comparison validity" in rendered


def test_print_release_note_template_script_outputs_heading():
    script = Path("scripts") / "print_release_note_template.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--version", "0.10.1", "--date", "2026-03-09"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "## [0.10.1] - 2026-03-09" in proc.stdout
