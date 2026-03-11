from __future__ import annotations

import json

from backend.calibration import apply_public_calibration, resolve_public_calibration


def test_apply_public_calibration_logistic(tmp_path):
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "1.0.0",
                "method": "logistic",
                "parameters": {"intercept": -4.0, "slope": 8.0},
                "dataset": {"source_name": "fixture"},
            }
        ),
        encoding="utf-8",
    )

    low = apply_public_calibration(raw_wildfire_score=20.0, artifact_path=str(artifact_path))
    high = apply_public_calibration(raw_wildfire_score=80.0, artifact_path=str(artifact_path))

    assert low is not None
    assert high is not None
    assert low["calibration_applied"] is True
    assert high["calibration_applied"] is True
    assert high["calibrated_damage_likelihood"] > low["calibrated_damage_likelihood"]


def test_apply_public_calibration_piecewise(tmp_path):
    artifact_path = tmp_path / "piecewise.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "1.0.0",
                "method": "piecewise_linear",
                "points": [[0.0, 0.01], [50.0, 0.2], [100.0, 0.9]],
            }
        ),
        encoding="utf-8",
    )

    out = apply_public_calibration(raw_wildfire_score=75.0, artifact_path=str(artifact_path))
    assert out is not None
    assert out["calibration_applied"] is True
    assert 0.20 < out["calibrated_damage_likelihood"] < 0.90


def test_apply_public_calibration_missing_artifact_returns_none():
    assert apply_public_calibration(raw_wildfire_score=60.0, artifact_path="") is None


def test_resolve_public_calibration_out_of_scope(tmp_path):
    artifact_path = tmp_path / "calibration_scope.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": "1.0.0",
                "method": "logistic",
                "parameters": {"intercept": -4.0, "slope": 8.0},
                "scope": {"region_ids": ["california_pilot"]},
                "limitations": ["Public outcomes are directional only."],
            }
        ),
        encoding="utf-8",
    )
    payload = resolve_public_calibration(
        raw_wildfire_score=70.0,
        artifact_path=str(artifact_path),
        resolved_region_id="missoula_pilot",
    )
    assert payload["calibration_enabled"] is True
    assert payload["calibration_applied"] is False
    assert payload["calibration_status"] == "out_of_scope"
    assert payload["scope_warning"] is not None
