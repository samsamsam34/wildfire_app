#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.event_backtesting import run_event_backtest  # noqa: E402
from backend.public_outcome_validation import evaluate_public_outcome_dataset_file  # noqa: E402

DEFAULT_EVAL_ROOT = Path("benchmark/public_outcomes/evaluation_dataset")
DEFAULT_OUTPUT_ROOT = Path("benchmark/public_outcomes/synthetic_sensitivity")

OUTCOME_TO_RANK = {
    "unknown": 0,
    "no_damage": 1,
    "no_known_damage": 1,
    "minor_damage": 2,
    "major_damage": 3,
    "destroyed": 4,
}

SCENARIO_PROFILES = (
    "baseline_observed",
    "vegetation_up",
    "slope_up",
    "fuel_near",
    "combined_high",
    "mitigation_low",
)


def _now_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_latest_eval_dataset(root: Path) -> Path:
    if not root.exists():
        raise ValueError(f"Evaluation dataset root does not exist: {root}")
    runs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name, reverse=True)
    for run in runs:
        candidate = run / "evaluation_dataset.jsonl"
        if candidate.exists():
            return candidate
    raise ValueError(f"No evaluation_dataset.jsonl found under {root}")


def _norm_outcome_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if not text:
        return "unknown"
    if text in OUTCOME_TO_RANK:
        return text
    if "destroy" in text:
        return "destroyed"
    if "major" in text or "severe" in text:
        return "major_damage"
    if "minor" in text or "affected" in text:
        return "minor_damage"
    if text in {"none", "undamaged", "no known damage"}:
        return "no_damage"
    return "unknown"


def _rng_offset(base_id: str, profile: str, lo: float, hi: float) -> float:
    seed_text = f"{base_id}|{profile}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    span = float(hi - lo)
    if span <= 0.0:
        return float(lo)
    raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    frac = raw / float((2**64) - 1)
    return float(lo + (span * frac))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _set_percent(raw: dict[str, Any], key: str, delta: float) -> None:
    current = _safe_float(raw.get(key))
    if current is None:
        return
    raw[key] = round(_clamp(float(current) + float(delta), 0.0, 100.0), 3)


def _set_distance(raw: dict[str, Any], key: str, delta: float, lo: float = 0.0, hi: float = 5000.0) -> None:
    current = _safe_float(raw.get(key))
    if current is None:
        return
    raw[key] = round(_clamp(float(current) + float(delta), lo, hi), 3)


def _set_numeric(raw: dict[str, Any], key: str, delta: float, lo: float, hi: float, digits: int = 3) -> None:
    current = _safe_float(raw.get(key))
    if current is None:
        return
    raw[key] = round(_clamp(float(current) + float(delta), lo, hi), digits)


def _sync_raw_transformed(raw: dict[str, Any], transformed: dict[str, Any]) -> None:
    burn = _safe_float(raw.get("burn_probability"))
    if burn is not None:
        transformed["burn_probability_index"] = round(_clamp(burn * 100.0, 0.0, 100.0), 3)
    burn_idx = _safe_float(transformed.get("burn_probability_index"))
    if burn_idx is not None:
        raw["burn_probability"] = round(_clamp(burn_idx / 100.0, 0.0, 1.0), 6)

    slope = _safe_float(raw.get("slope"))
    if slope is not None:
        transformed["slope_index"] = round(_clamp((slope / 45.0) * 100.0, 0.0, 100.0), 3)
    slope_idx = _safe_float(transformed.get("slope_index"))
    if slope_idx is not None:
        raw["slope"] = round(_clamp((slope_idx / 100.0) * 45.0, 0.0, 45.0), 3)

    fuel_model = _safe_float(raw.get("fuel_model"))
    if fuel_model is not None:
        transformed["fuel_index"] = round(_clamp(fuel_model, 0.0, 100.0), 3)
    fuel_idx = _safe_float(transformed.get("fuel_index"))
    if fuel_idx is not None:
        raw["fuel_model"] = round(_clamp(fuel_idx, 0.0, 100.0), 3)

    canopy_cover = _safe_float(raw.get("canopy_cover"))
    if canopy_cover is not None:
        transformed["canopy_index"] = round(_clamp(canopy_cover, 0.0, 100.0), 3)
    canopy_idx = _safe_float(transformed.get("canopy_index"))
    if canopy_idx is not None:
        raw["canopy_cover"] = round(_clamp(canopy_idx, 0.0, 100.0), 3)

    wildland_distance_m = _safe_float(raw.get("wildland_distance_m"))
    if wildland_distance_m is not None:
        transformed["wildland_distance_index"] = round(_clamp(100.0 - (wildland_distance_m / 20.0), 0.0, 100.0), 3)
    wildland_idx = _safe_float(transformed.get("wildland_distance_index"))
    if wildland_idx is not None:
        raw["wildland_distance_m"] = round(_clamp((100.0 - wildland_idx) * 20.0, 0.0, 2000.0), 3)


def apply_synthetic_profile(
    *,
    base_id: str,
    profile: str,
    raw_features: dict[str, Any],
    transformed_features: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = dict(raw_features)
    transformed = dict(transformed_features)

    if profile == "baseline_observed":
        _sync_raw_transformed(raw, transformed)
        return raw, transformed

    jitter = _rng_offset(base_id, profile, -1.0, 1.0)
    if profile in {"vegetation_up", "combined_high"}:
        _set_percent(raw, "ring_0_5_ft_vegetation_density", 8.0 + jitter)
        _set_percent(raw, "ring_5_30_ft_vegetation_density", 7.0 + jitter)
        _set_percent(raw, "near_structure_vegetation_0_5_pct", 9.0 + jitter)
        _set_percent(raw, "canopy_adjacency_proxy_pct", 6.0 + jitter)
        _set_percent(raw, "vegetation_continuity_proxy_pct", 6.0 + jitter)
        _set_distance(raw, "nearest_high_fuel_patch_distance_ft", -90.0)
    if profile in {"mitigation_low"}:
        _set_percent(raw, "ring_0_5_ft_vegetation_density", -10.0 + jitter)
        _set_percent(raw, "ring_5_30_ft_vegetation_density", -8.0 + jitter)
        _set_percent(raw, "near_structure_vegetation_0_5_pct", -11.0 + jitter)
        _set_percent(raw, "canopy_adjacency_proxy_pct", -8.0 + jitter)
        _set_percent(raw, "vegetation_continuity_proxy_pct", -8.0 + jitter)
        _set_distance(raw, "nearest_high_fuel_patch_distance_ft", +140.0)

    if profile in {"slope_up", "combined_high"}:
        _set_numeric(raw, "slope", 4.0 + jitter, 0.0, 45.0)
        _set_numeric(raw, "burn_probability", 0.07 + (jitter * 0.002), 0.0, 1.0, digits=6)
    if profile in {"mitigation_low"}:
        _set_numeric(raw, "slope", -4.0 + jitter, 0.0, 45.0)
        _set_numeric(raw, "burn_probability", -0.08 + (jitter * 0.002), 0.0, 1.0, digits=6)

    if profile in {"fuel_near", "combined_high"}:
        _set_numeric(raw, "fuel_model", 8.0 + jitter, 0.0, 100.0)
        _set_distance(raw, "wildland_distance_m", -140.0)
    if profile in {"mitigation_low"}:
        _set_numeric(raw, "fuel_model", -9.0 + jitter, 0.0, 100.0)
        _set_distance(raw, "wildland_distance_m", +180.0)

    _sync_raw_transformed(raw, transformed)
    return raw, transformed


def _scenario_outcome(profile: str, observed_label: str) -> tuple[str, int]:
    if profile == "mitigation_low":
        return "no_damage", OUTCOME_TO_RANK["no_damage"]
    if profile in {"vegetation_up", "slope_up", "fuel_near", "combined_high"}:
        return "major_damage", OUTCOME_TO_RANK["major_damage"]
    label = _norm_outcome_label(observed_label)
    return label, OUTCOME_TO_RANK.get(label, 0)


def _build_synthetic_event_records(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in base_rows:
        feature = row.get("feature") if isinstance(row.get("feature"), dict) else {}
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
        raw = snapshot.get("raw_feature_vector") if isinstance(snapshot.get("raw_feature_vector"), dict) else {}
        transformed = (
            snapshot.get("transformed_feature_vector")
            if isinstance(snapshot.get("transformed_feature_vector"), dict)
            else {}
        )
        if not raw and not transformed:
            continue
        lat = _safe_float(feature.get("latitude"))
        lon = _safe_float(feature.get("longitude"))
        if lat is None or lon is None:
            continue
        base_id = str(row.get("property_event_id") or feature.get("record_id") or "")
        if not base_id:
            continue
        event_id = str(event.get("event_id") or "synthetic_event")
        event_name = str(event.get("event_name") or event_id)
        event_date = str(event.get("event_date") or "2021-01-01")
        address_text = str(feature.get("address_text") or f"Synthetic {base_id}")
        observed_label = _norm_outcome_label(outcome.get("damage_label"))

        for profile in SCENARIO_PROFILES:
            raw_variant, transformed_variant = apply_synthetic_profile(
                base_id=base_id,
                profile=profile,
                raw_features=raw,
                transformed_features=transformed,
            )
            scenario_label, scenario_rank = _scenario_outcome(profile, observed_label)
            record_id = f"{base_id}__syn__{profile}"
            records.append(
                {
                    "event_id": f"{event_id}__synthetic",
                    "event_name": f"{event_name} Synthetic",
                    "event_date": event_date,
                    "source_name": "synthetic_sensitivity",
                    "record_id": record_id,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "address_text": address_text,
                    "outcome_label": scenario_label,
                    "outcome_rank": scenario_rank,
                    "label_confidence": 0.95,
                    "source_metadata": {
                        "synthetic_variation": True,
                        "synthetic_profile": profile,
                        "base_property_event_id": base_id,
                        "observed_outcome_label": observed_label,
                    },
                    "input_payload": {
                        "attributes": {},
                        "confirmed_fields": [],
                        "audience": "insurer",
                        "tags": ["synthetic-sensitivity"],
                    },
                    "raw_feature_vector": raw_variant,
                    "transformed_feature_vector": transformed_variant,
                }
            )
    return records


def _extract_score(record: dict[str, Any]) -> float | None:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    return _safe_float(scores.get("wildfire_risk_score"))


def _summarize_response(scored_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_base: dict[str, dict[str, float]] = {}
    for record in scored_records:
        record_id = str(record.get("record_id") or "")
        if "__syn__" not in record_id:
            continue
        base_id, profile = record_id.split("__syn__", 1)
        score = _extract_score(record)
        if score is None:
            continue
        by_base.setdefault(base_id, {})[profile] = float(score)

    checks = {
        "vegetation_up_ge_baseline": {"pass": 0, "fail": 0},
        "slope_up_ge_baseline": {"pass": 0, "fail": 0},
        "fuel_near_ge_baseline": {"pass": 0, "fail": 0},
        "combined_high_ge_baseline": {"pass": 0, "fail": 0},
        "mitigation_low_le_baseline": {"pass": 0, "fail": 0},
    }
    for profile_scores in by_base.values():
        baseline = _safe_float(profile_scores.get("baseline_observed"))
        if baseline is None:
            continue
        for profile, check_key, direction in (
            ("vegetation_up", "vegetation_up_ge_baseline", "ge"),
            ("slope_up", "slope_up_ge_baseline", "ge"),
            ("fuel_near", "fuel_near_ge_baseline", "ge"),
            ("combined_high", "combined_high_ge_baseline", "ge"),
            ("mitigation_low", "mitigation_low_le_baseline", "le"),
        ):
            candidate = _safe_float(profile_scores.get(profile))
            if candidate is None:
                continue
            if direction == "ge":
                if float(candidate) >= float(baseline):
                    checks[check_key]["pass"] += 1
                else:
                    checks[check_key]["fail"] += 1
            else:
                if float(candidate) <= float(baseline):
                    checks[check_key]["pass"] += 1
                else:
                    checks[check_key]["fail"] += 1

    check_rates = {
        key: (
            round(payload["pass"] / float(payload["pass"] + payload["fail"]), 4)
            if (payload["pass"] + payload["fail"]) > 0
            else None
        )
        for key, payload in checks.items()
    }
    return {
        "base_property_count": len(by_base),
        "directionality_checks": checks,
        "directionality_pass_rates": check_rates,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_auc_brier(report: dict[str, Any]) -> tuple[float | None, float | None]:
    disc = report.get("discrimination_metrics") if isinstance(report.get("discrimination_metrics"), dict) else {}
    brier = report.get("brier_scores") if isinstance(report.get("brier_scores"), dict) else {}
    return _safe_float(disc.get("wildfire_risk_score_auc")), _safe_float(brier.get("wildfire_probability_proxy"))


def run_synthetic_sensitivity(
    *,
    evaluation_dataset: Path,
    output_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_name = run_id or f"synthetic_sensitivity_{_now_id()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _read_jsonl(evaluation_dataset)
    synthetic_records = _build_synthetic_event_records(base_rows)
    synthetic_dataset = {
        "event_backtest_version": "1.0.0",
        "dataset_id": f"synthetic_sensitivity::{run_name}",
        "dataset_name": "Synthetic Sensitivity Evaluation",
        "source_name": "synthetic_sensitivity",
        "metadata": {
            "synthetic_variation": True,
            "caveat": (
                "Synthetic scenarios are for sensitivity testing only. "
                "They are not real wildfire outcomes and must not be interpreted as claims-grade validation."
            ),
            "scenario_profiles": list(SCENARIO_PROFILES),
            "base_row_count": len(base_rows),
            "synthetic_record_count": len(synthetic_records),
            "source_evaluation_dataset": str(evaluation_dataset),
        },
        "records": synthetic_records,
    }
    synthetic_dataset_path = run_dir / "synthetic_event_dataset.json"
    _write_json(synthetic_dataset_path, synthetic_dataset)

    backtest_dir = run_dir / "synthetic_backtest"
    backtest_artifact = run_event_backtest(
        dataset_paths=[synthetic_dataset_path],
        output_dir=backtest_dir,
        use_runtime_context_when_no_overrides=False,
    )
    backtest_path = Path(str(backtest_artifact.get("artifact_path") or ""))
    if not backtest_path.exists():
        raise ValueError("Synthetic backtest artifact was not created.")
    scored_payload = json.loads(backtest_path.read_text(encoding="utf-8"))
    scored_records = scored_payload.get("records") if isinstance(scored_payload.get("records"), list) else []

    baseline_report, _ = evaluate_public_outcome_dataset_file(dataset_path=evaluation_dataset)
    synthetic_report, _ = evaluate_public_outcome_dataset_file(dataset_path=backtest_path)

    baseline_auc, baseline_brier = _resolve_auc_brier(baseline_report)
    synthetic_auc, synthetic_brier = _resolve_auc_brier(synthetic_report)
    comparison = {
        "baseline_dataset": str(evaluation_dataset),
        "synthetic_dataset": str(backtest_path),
        "baseline_auc": baseline_auc,
        "synthetic_auc": synthetic_auc,
        "auc_delta_synthetic_minus_baseline": (
            round(float(synthetic_auc) - float(baseline_auc), 6)
            if baseline_auc is not None and synthetic_auc is not None
            else None
        ),
        "baseline_brier": baseline_brier,
        "synthetic_brier": synthetic_brier,
        "brier_delta_synthetic_minus_baseline": (
            round(float(synthetic_brier) - float(baseline_brier), 6)
            if baseline_brier is not None and synthetic_brier is not None
            else None
        ),
        "synthetic_auc_above_0_60": bool(synthetic_auc is not None and float(synthetic_auc) > 0.60),
    }
    sensitivity = _summarize_response(scored_records)

    _write_json(run_dir / "synthetic_validation_metrics.json", synthetic_report)
    _write_json(run_dir / "baseline_validation_metrics.json", baseline_report)
    _write_json(run_dir / "comparison_to_baseline.json", comparison)
    _write_json(run_dir / "sensitivity_response.json", sensitivity)

    summary_lines = [
        "# Synthetic Sensitivity Evaluation",
        "",
        "Synthetic scenario expansion was used to test model sensitivity. These are not real-world outcomes.",
        "",
        f"- Run ID: `{run_name}`",
        f"- Base rows: `{len(base_rows)}`",
        f"- Synthetic records: `{len(synthetic_records)}`",
        f"- Baseline AUC: `{baseline_auc}`",
        f"- Synthetic AUC: `{synthetic_auc}`",
        f"- AUC delta (synthetic - baseline): `{comparison.get('auc_delta_synthetic_minus_baseline')}`",
        f"- Baseline Brier: `{baseline_brier}`",
        f"- Synthetic Brier: `{synthetic_brier}`",
        f"- Brier delta (synthetic - baseline): `{comparison.get('brier_delta_synthetic_minus_baseline')}`",
        "",
        "## Directionality Checks",
    ]
    for key, payload in (sensitivity.get("directionality_checks") or {}).items():
        rate = (sensitivity.get("directionality_pass_rates") or {}).get(key)
        summary_lines.append(f"- `{key}`: pass={payload.get('pass')} fail={payload.get('fail')} pass_rate={rate}")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "run_id": run_name,
        "run_dir": str(run_dir),
        "synthetic_event_dataset_path": str(synthetic_dataset_path),
        "synthetic_backtest_path": str(backtest_path),
        "synthetic_validation_metrics_path": str(run_dir / "synthetic_validation_metrics.json"),
        "baseline_validation_metrics_path": str(run_dir / "baseline_validation_metrics.json"),
        "comparison_path": str(run_dir / "comparison_to_baseline.json"),
        "sensitivity_response_path": str(run_dir / "sensitivity_response.json"),
        "summary_path": str(summary_path),
        "comparison": comparison,
        "sensitivity": sensitivity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic feature variations per property and evaluate sensitivity/discrimination impact."
    )
    parser.add_argument("--evaluation-dataset", default=None, help="Path to baseline evaluation_dataset.jsonl")
    parser.add_argument("--evaluation-dataset-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    dataset = (
        Path(args.evaluation_dataset).expanduser()
        if args.evaluation_dataset
        else _resolve_latest_eval_dataset(Path(args.evaluation_dataset_root).expanduser())
    )
    result = run_synthetic_sensitivity(
        evaluation_dataset=dataset,
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
