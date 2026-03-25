from __future__ import annotations

import json
from pathlib import Path

from backend.calibration import resolve_public_calibration
from scripts.fit_public_outcome_calibration import (
    fit_calibration,
    run_public_outcome_calibration,
)


def _write_dataset(path: Path) -> None:
    rows = []
    values = [
        (95.0, 1, "high", "high", "high"),
        (90.0, 1, "high", "high", "high"),
        (84.0, 1, "moderate", "moderate", "moderate"),
        (79.0, 1, "moderate", "moderate", "moderate"),
        (73.0, 1, "low", "low", "low"),
        (66.0, 0, "moderate", "moderate", "moderate"),
        (59.0, 0, "moderate", "moderate", "moderate"),
        (52.0, 0, "high", "high", "high"),
        (45.0, 0, "low", "low", "low"),
        (39.0, 0, "low", "low", "low"),
        (33.0, 0, "low", "low", "low"),
        (27.0, 0, "high", "high", "high"),
    ]
    for idx, (score, label, confidence_tier, evidence_tier, join_tier) in enumerate(values, start=1):
        rows.append(
            {
                "event_id": "evt-a" if idx <= 6 else "evt-b",
                "record_id": f"r{idx}",
                "outcome_label": "destroyed" if label == 1 else "no_damage",
                "structure_loss_or_major_damage": label,
                "scores": {
                    "wildfire_risk_score": score,
                    "site_hazard_score": max(0.0, score - 4.0),
                    "home_ignition_vulnerability_score": max(0.0, score - 2.0),
                    "insurance_readiness_score": max(0.0, 100.0 - score),
                },
                "confidence_tier": confidence_tier,
                "evidence_quality_tier": evidence_tier,
                "join_confidence_tier": join_tier,
                "fallback_default_flags": {
                    "fallback_factor_count": 0 if evidence_tier == "high" else (2 if evidence_tier == "low" else 1),
                    "missing_factor_count": 0 if evidence_tier == "high" else (3 if evidence_tier == "low" else 1),
                    "inferred_factor_count": 0 if evidence_tier == "high" else 1,
                    "coverage_failed_count": 1 if evidence_tier == "low" else 0,
                    "coverage_fallback_count": 1 if evidence_tier != "high" else 0,
                    "fallback_weight_fraction": 0.05 if evidence_tier == "high" else (0.7 if evidence_tier == "low" else 0.35),
                },
            }
        )
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def test_run_public_outcome_calibration_writes_bundle(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation_dataset.json"
    _write_dataset(dataset)
    result = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=tmp_path / "calibration_runs",
        run_id="calibration_bundle",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    run_dir = Path(result["run_dir"])
    assert result["fitted"] is True
    assert (run_dir / "calibration_model.json").exists()
    assert (run_dir / "calibration_config.json").exists()
    assert (run_dir / "pre_vs_post_metrics.json").exists()
    assert (run_dir / "calibration_curve.json").exists()
    assert (run_dir / "comparison_to_previous.json").exists()
    assert (run_dir / "comparison_to_previous.md").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "manifest.json").exists()


def test_fit_calibration_exports_artifact_and_runtime_loader_applies(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation_dataset.json"
    _write_dataset(dataset)
    artifact_path = tmp_path / "public_calibration.json"
    artifact = fit_calibration(dataset_path=dataset, output_path=artifact_path)
    assert artifact_path.exists()
    assert artifact.get("method") in {"logistic", "piecewise_linear", "bin_rate_table"}
    payload = resolve_public_calibration(raw_wildfire_score=80.0, artifact_path=str(artifact_path))
    assert payload["calibration_enabled"] is True
    assert payload["calibration_applied"] is True
    assert payload["calibrated_damage_likelihood"] is not None


def test_run_public_outcome_calibration_supports_binned_method(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation_dataset.json"
    _write_dataset(dataset)
    result = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=tmp_path / "calibration_runs",
        run_id="binned_method",
        method="binned",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        min_rows_for_binned=10,
        min_unique_scores_for_binned=5,
        overwrite=True,
    )
    artifact = json.loads((Path(result["run_dir"]) / "calibration_model.json").read_text(encoding="utf-8"))
    assert artifact.get("method") in {"bin_rate_table", None}


def test_calibration_guardrail_skips_when_candidate_worsens(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "evaluation_dataset.json"
    _write_dataset(dataset)

    def _fake_choose_method(**_: object):
        return (
            {
                "method": "logistic",
                "parameters": {"intercept": 0.0, "slope": 0.0, "x_scale": 100.0},
                "selected_key": "logistic",
            },
            [],
            {
                "raw_validation_metrics": {"brier_probability": 0.10},
                "candidate_methods": {
                    "logistic": {
                        "method": "logistic",
                        "validation_metrics": {"brier_probability": 0.30},
                        "brier_delta_vs_raw": 0.20,
                    }
                },
                "selection_reason": "forced_bad_candidate",
            },
        )

    monkeypatch.setattr(
        "scripts.fit_public_outcome_calibration._choose_method",
        _fake_choose_method,
    )
    result = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=tmp_path / "calibration_runs",
        run_id="guardrail_skip",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    assert result["fitted"] is False
    artifact = json.loads((Path(result["run_dir"]) / "calibration_model.json").read_text(encoding="utf-8"))
    assert artifact.get("method") is None
    guardrail = artifact.get("guardrail_decision") if isinstance(artifact.get("guardrail_decision"), dict) else {}
    assert guardrail.get("applied") is False
    assert guardrail.get("reason") == "brier_worsened_vs_raw"


def test_pre_post_metrics_are_reported(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation_dataset.json"
    _write_dataset(dataset)
    result = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=tmp_path / "calibration_runs",
        run_id="pre_post_metrics",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    metrics = json.loads((Path(result["run_dir"]) / "pre_vs_post_metrics.json").read_text(encoding="utf-8"))
    assert "pre" in metrics
    assert "post" in metrics
    assert "delta" in metrics
    assert metrics["pre"]["brier_probability"] is not None
    assert metrics["post"]["brier_probability"] is not None


def test_calibration_deterministic_with_fixed_run_id(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation_dataset.json"
    _write_dataset(dataset)
    root = tmp_path / "calibration_runs"

    first = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=root,
        run_id="deterministic_calibration",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    second = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=root,
        run_id="deterministic_calibration",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    assert first["manifest_path"] == second["manifest_path"]
    assert Path(first["manifest_path"]).read_text(encoding="utf-8") == Path(second["manifest_path"]).read_text(encoding="utf-8")
