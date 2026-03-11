from __future__ import annotations

from pathlib import Path

from scripts.analyze_open_model_score_spread import run_open_model_spread


def test_open_model_score_spread_fixture_has_material_variance(tmp_path):
    fixture = Path("tests/fixtures/score_variance_scenarios.json")
    csv_out = tmp_path / "spread.csv"
    summary = run_open_model_spread(fixture_path=fixture, csv_out=csv_out)

    wildfire = (summary.get("score_stats") or {}).get("wildfire_risk_score") or {}
    assert int(summary.get("scenario_count") or 0) >= 6
    assert float(wildfire.get("max") or 0.0) - float(wildfire.get("min") or 0.0) >= 20.0
    assert float(wildfire.get("stddev") or 0.0) >= 6.0
    assert "property_signal_stats" in summary
    assert "near_structure_vegetation_0_5_pct" in (summary.get("property_signal_stats") or {})
    assert csv_out.exists()

