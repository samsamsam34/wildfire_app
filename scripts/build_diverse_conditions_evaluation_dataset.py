#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.event_backtesting import run_event_backtest  # noqa: E402
from backend.public_outcome_validation import evaluate_public_outcome_dataset_file  # noqa: E402
from scripts.run_synthetic_sensitivity_evaluation import (  # noqa: E402
    SCENARIO_PROFILES,
    apply_synthetic_profile,
)

DEFAULT_BASE_DATASET_ROOT = Path("benchmark/public_outcomes/evaluation_dataset")
DEFAULT_OUTPUT_ROOT = Path("benchmark/public_outcomes/evaluation_dataset")
DEFAULT_VALIDATION_OUTPUT_ROOT = Path("benchmark/public_outcomes/validation")


@dataclass(frozen=True)
class BaseRowSignals:
    row: dict[str, Any]
    property_event_id: str
    hazard_signal: float | None
    vegetation_signal: float | None
    terrain_signal: float | None


def _now_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_pct(value: float | None, *, lo: float = 0.0, hi: float = 100.0) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if hi <= lo:
        return None
    return max(0.0, min(100.0, ((float(value) - lo) / float(hi - lo)) * 100.0))


def _avg(values: list[float | None]) -> float | None:
    usable = [float(v) for v in values if isinstance(v, (int, float))]
    if not usable:
        return None
    return sum(usable) / float(len(usable))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_latest_dataset(root: Path) -> Path:
    if not root.exists():
        raise ValueError(f"Evaluation dataset root does not exist: {root}")
    run_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name, reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / "evaluation_dataset.jsonl"
        if candidate.exists():
            return candidate
    raise ValueError(f"No evaluation_dataset.jsonl found under {root}")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q_clamped = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    pos = q_clamped * float(len(ordered) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    fraction = pos - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _bin_tertile(value: float | None, low_cut: float, high_cut: float) -> str:
    if value is None:
        return "unknown"
    if value < low_cut:
        return "low"
    if value > high_cut:
        return "high"
    return "medium"


def _row_feature_vectors(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
    raw = snapshot.get("raw_feature_vector") if isinstance(snapshot.get("raw_feature_vector"), dict) else {}
    transformed = (
        snapshot.get("transformed_feature_vector")
        if isinstance(snapshot.get("transformed_feature_vector"), dict)
        else {}
    )
    return dict(raw), dict(transformed)


def _extract_signals(row: dict[str, Any]) -> BaseRowSignals | None:
    raw, transformed = _row_feature_vectors(row)
    if not raw and not transformed:
        return None
    feature = row.get("feature") if isinstance(row.get("feature"), dict) else {}
    lat = _safe_float(feature.get("latitude"))
    lon = _safe_float(feature.get("longitude"))
    if lat is None or lon is None:
        return None
    property_event_id = str(
        row.get("property_event_id") or feature.get("record_id") or row.get("row_id") or ""
    ).strip()
    if not property_event_id:
        return None

    burn_idx = _safe_float(transformed.get("burn_probability_index"))
    if burn_idx is None:
        burn = _safe_float(raw.get("burn_probability"))
        burn_idx = burn * 100.0 if burn is not None else None
    fuel_idx = _safe_float(transformed.get("fuel_index"))
    if fuel_idx is None:
        fuel_idx = _safe_float(raw.get("fuel_model"))
    wildland_idx = _safe_float(transformed.get("wildland_distance_index"))
    if wildland_idx is None:
        wildland_m = _safe_float(raw.get("wildland_distance_m"))
        wildland_idx = _norm_pct((2000.0 - wildland_m) if wildland_m is not None else None, lo=0.0, hi=2000.0)
    hazard_signal = _avg([burn_idx, fuel_idx, wildland_idx])

    veg_signal = _avg(
        [
            _safe_float(raw.get("near_structure_vegetation_0_5_pct")),
            _safe_float(raw.get("ring_0_5_ft_vegetation_density")),
            _safe_float(raw.get("ring_5_30_ft_vegetation_density")),
            _safe_float(raw.get("canopy_adjacency_proxy_pct")),
            _safe_float(raw.get("vegetation_continuity_proxy_pct")),
        ]
    )

    slope_idx = _safe_float(transformed.get("slope_index"))
    if slope_idx is None:
        slope = _safe_float(raw.get("slope"))
        slope_idx = _norm_pct(slope, lo=0.0, hi=45.0)

    return BaseRowSignals(
        row=row,
        property_event_id=property_event_id,
        hazard_signal=hazard_signal,
        vegetation_signal=veg_signal,
        terrain_signal=slope_idx,
    )


def _select_diverse_rows(
    *,
    base_rows: list[dict[str, Any]],
    max_per_cell: int,
    min_selected_rows: int,
) -> tuple[list[BaseRowSignals], dict[str, Any]]:
    signals: list[BaseRowSignals] = []
    for row in base_rows:
        sig = _extract_signals(row)
        if sig is not None:
            signals.append(sig)
    if not signals:
        raise ValueError("No rows with usable coordinates and feature vectors were found in the base dataset.")

    hazard_values = [float(sig.hazard_signal) for sig in signals if sig.hazard_signal is not None]
    vegetation_values = [float(sig.vegetation_signal) for sig in signals if sig.vegetation_signal is not None]
    terrain_values = [float(sig.terrain_signal) for sig in signals if sig.terrain_signal is not None]
    if not hazard_values or not vegetation_values or not terrain_values:
        raise ValueError("Cannot stratify rows; hazard/vegetation/terrain signals are missing.")

    h_low, h_high = _quantile(hazard_values, 1.0 / 3.0), _quantile(hazard_values, 2.0 / 3.0)
    v_low, v_high = _quantile(vegetation_values, 1.0 / 3.0), _quantile(vegetation_values, 2.0 / 3.0)
    t_low, t_high = _quantile(terrain_values, 1.0 / 3.0), _quantile(terrain_values, 2.0 / 3.0)

    cells: dict[tuple[str, str, str], list[BaseRowSignals]] = {}
    for sig in signals:
        key = (
            _bin_tertile(sig.hazard_signal, h_low, h_high),
            _bin_tertile(sig.vegetation_signal, v_low, v_high),
            _bin_tertile(sig.terrain_signal, t_low, t_high),
        )
        cells.setdefault(key, []).append(sig)

    for cell_rows in cells.values():
        cell_rows.sort(
            key=lambda sig: (
                sig.property_event_id,
                sig.row.get("event", {}).get("event_id") if isinstance(sig.row.get("event"), dict) else "",
            )
        )

    selected: list[BaseRowSignals] = []
    selected_ids: set[str] = set()
    for key in sorted(cells.keys()):
        for sig in cells[key][: max(1, int(max_per_cell))]:
            if sig.property_event_id in selected_ids:
                continue
            selected.append(sig)
            selected_ids.add(sig.property_event_id)

    if len(selected) < max(1, int(min_selected_rows)):
        remaining = [sig for sig in signals if sig.property_event_id not in selected_ids]
        remaining.sort(
            key=lambda sig: (
                sig.property_event_id,
                abs(float(sig.hazard_signal or 50.0) - 50.0)
                + abs(float(sig.vegetation_signal or 50.0) - 50.0)
                + abs(float(sig.terrain_signal or 50.0) - 50.0),
            )
        )
        for sig in remaining:
            selected.append(sig)
            selected_ids.add(sig.property_event_id)
            if len(selected) >= int(min_selected_rows):
                break

    coverage = {
        "hazard_low_cut": round(h_low, 4),
        "hazard_high_cut": round(h_high, 4),
        "vegetation_low_cut": round(v_low, 4),
        "vegetation_high_cut": round(v_high, 4),
        "terrain_low_cut": round(t_low, 4),
        "terrain_high_cut": round(t_high, 4),
        "cell_count_total": len(cells),
        "cell_counts": {
            f"{key[0]}|{key[1]}|{key[2]}": len(cell_rows) for key, cell_rows in sorted(cells.items())
        },
        "selected_property_count": len(selected),
    }
    return selected, coverage


def _norm_outcome_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"destroyed", "major_damage", "minor_damage", "no_damage", "no_known_damage"}:
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


def _scenario_outcome(profile: str, observed_label: str) -> tuple[str, int]:
    ranks = {"unknown": 0, "no_damage": 1, "no_known_damage": 1, "minor_damage": 2, "major_damage": 3, "destroyed": 4}
    if profile == "mitigation_low":
        return "no_damage", 1
    if profile in {"vegetation_up", "slope_up", "fuel_near", "combined_high"}:
        return "major_damage", 3
    label = _norm_outcome_label(observed_label)
    return label, ranks.get(label, 0)


def _build_synthetic_records(selected: list[BaseRowSignals]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sig in selected:
        row = sig.row
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        feature = row.get("feature") if isinstance(row.get("feature"), dict) else {}
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        raw, transformed = _row_feature_vectors(row)
        lat = _safe_float(feature.get("latitude"))
        lon = _safe_float(feature.get("longitude"))
        if lat is None or lon is None:
            continue
        if not raw and not transformed:
            continue
        event_id = str(event.get("event_id") or "synthetic_event")
        event_name = str(event.get("event_name") or event_id)
        event_date = str(event.get("event_date") or "2021-01-01")
        observed_label = _norm_outcome_label(outcome.get("damage_label"))
        address_text = str(feature.get("address_text") or f"Synthetic {sig.property_event_id}")
        for profile in SCENARIO_PROFILES:
            raw_variant, transformed_variant = apply_synthetic_profile(
                base_id=sig.property_event_id,
                profile=profile,
                raw_features=raw,
                transformed_features=transformed,
            )
            scenario_label, scenario_rank = _scenario_outcome(profile, observed_label)
            record_id = f"{sig.property_event_id}__diverse__{profile}"
            records.append(
                {
                    "event_id": f"{event_id}__diverse",
                    "event_name": f"{event_name} Diverse",
                    "event_date": event_date,
                    "source_name": "synthetic_diverse_conditions",
                    "record_id": record_id,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "address_text": address_text,
                    "outcome_label": scenario_label,
                    "outcome_rank": scenario_rank,
                    "label_confidence": 0.95,
                    "source_metadata": {
                        "synthetic_variation": True,
                        "diverse_condition_pack": True,
                        "synthetic_profile": profile,
                        "base_property_event_id": sig.property_event_id,
                        "observed_outcome_label": observed_label,
                        "base_signal_bins": {
                            "hazard": _bin_tertile(sig.hazard_signal, 33.3333, 66.6667),
                            "vegetation": _bin_tertile(sig.vegetation_signal, 33.3333, 66.6667),
                            "terrain": _bin_tertile(sig.terrain_signal, 33.3333, 66.6667),
                        },
                    },
                    "input_payload": {
                        "attributes": {},
                        "confirmed_fields": [],
                        "audience": "insurer",
                        "tags": ["synthetic-diverse-conditions"],
                    },
                    "raw_feature_vector": raw_variant,
                    "transformed_feature_vector": transformed_variant,
                }
            )
    return records


def _feature_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _raw_vector(row: dict[str, Any]) -> dict[str, Any]:
        direct = row.get("raw_feature_vector") if isinstance(row.get("raw_feature_vector"), dict) else {}
        if direct:
            return direct
        snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
        nested = snapshot.get("raw_feature_vector") if isinstance(snapshot.get("raw_feature_vector"), dict) else {}
        return nested

    keys = [
        ("slope", lambda r: _safe_float(_raw_vector(r).get("slope"))),
        ("fuel_model", lambda r: _safe_float(_raw_vector(r).get("fuel_model"))),
        ("ring_0_5_ft_vegetation_density", lambda r: _safe_float(_raw_vector(r).get("ring_0_5_ft_vegetation_density"))),
        ("ring_5_30_ft_vegetation_density", lambda r: _safe_float(_raw_vector(r).get("ring_5_30_ft_vegetation_density"))),
        ("burn_probability", lambda r: _safe_float(_raw_vector(r).get("burn_probability"))),
        ("nearby_structure_count_100_ft", lambda r: _safe_float(_raw_vector(r).get("nearby_structure_count_100_ft"))),
        ("nearby_structure_count_300_ft", lambda r: _safe_float(_raw_vector(r).get("nearby_structure_count_300_ft"))),
        ("nearest_structure_distance_ft", lambda r: _safe_float(_raw_vector(r).get("nearest_structure_distance_ft"))),
        ("building_age_proxy_year", lambda r: _safe_float(_raw_vector(r).get("building_age_proxy_year"))),
        ("building_age_material_proxy_risk", lambda r: _safe_float(_raw_vector(r).get("building_age_material_proxy_risk"))),
    ]
    out: dict[str, Any] = {}
    for key, extractor in keys:
        values = [extractor(row) for row in rows]
        usable = [float(v) for v in values if isinstance(v, (int, float))]
        if not usable:
            out[key] = {"count": 0, "min": None, "max": None, "mean": None, "stddev": None}
            continue
        mean = sum(usable) / float(len(usable))
        variance = sum((v - mean) ** 2 for v in usable) / float(len(usable))
        out[key] = {
            "count": len(usable),
            "min": round(min(usable), 6),
            "max": round(max(usable), 6),
            "mean": round(mean, 6),
            "stddev": round(math.sqrt(variance), 6),
        }
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def build_diverse_conditions_dataset(
    *,
    base_dataset_path: Path,
    output_root: Path,
    run_id: str | None,
    max_per_cell: int,
    min_selected_rows: int,
) -> dict[str, Any]:
    run_name = run_id or f"diverse_conditions_eval_{_now_id()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _load_jsonl_rows(base_dataset_path)
    selected, coverage = _select_diverse_rows(
        base_rows=base_rows,
        max_per_cell=max_per_cell,
        min_selected_rows=min_selected_rows,
    )
    synthetic_records = _build_synthetic_records(selected)
    if not synthetic_records:
        raise ValueError("Failed to build synthetic records; selected rows did not produce valid scenarios.")

    synthetic_event_dataset = {
        "event_backtest_version": "1.0.0",
        "dataset_id": f"diverse_conditions::{run_name}",
        "dataset_name": "Diverse Conditions Synthetic Evaluation Dataset",
        "source_name": "synthetic_diverse_conditions",
        "metadata": {
            "synthetic_variation": True,
            "diverse_condition_pack": True,
            "base_evaluation_dataset": str(base_dataset_path),
            "selected_property_count": len(selected),
            "scenario_profiles": list(SCENARIO_PROFILES),
            "caveat": (
                "Synthetic diverse-condition records are for model sensitivity and discrimination stress-testing only. "
                "They are not real public wildfire outcomes."
            ),
        },
        "records": synthetic_records,
    }
    synthetic_event_path = run_dir / "synthetic_event_dataset.json"
    _write_json(synthetic_event_path, synthetic_event_dataset)

    backtest_dir = run_dir / "scored_backtest"
    backtest_artifact = run_event_backtest(
        dataset_paths=[synthetic_event_path],
        output_dir=backtest_dir,
        use_runtime_context_when_no_overrides=False,
    )
    scored_path = Path(str(backtest_artifact.get("artifact_path") or ""))
    if not scored_path.exists():
        raise ValueError("Scored synthetic backtest artifact was not produced.")

    report, prepared_rows = evaluate_public_outcome_dataset_file(dataset_path=scored_path)
    evaluation_dataset_path = run_dir / "evaluation_dataset.jsonl"
    _write_jsonl(evaluation_dataset_path, prepared_rows)

    base_dist = _feature_distribution([sig.row for sig in selected])
    diverse_dist = _feature_distribution(prepared_rows)
    auc = (
        ((report.get("discrimination_metrics") or {}).get("wildfire_risk_score_auc"))
        if isinstance(report.get("discrimination_metrics"), dict)
        else None
    )
    manifest = {
        "run_id": run_name,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "base_dataset_path": str(base_dataset_path),
        "selected_property_count": len(selected),
        "synthetic_record_count": len(synthetic_records),
        "scenario_profile_count": len(SCENARIO_PROFILES),
        "coverage_selection": coverage,
        "feature_distributions": {
            "selected_base_rows": base_dist,
            "diverse_evaluation_rows": diverse_dist,
        },
        "summary": {
            "prepared_rows": len(prepared_rows),
            "wildfire_risk_score_auc": auc,
        },
        "caveat": (
            "This bundle intentionally increases condition diversity for stress-testing model sensitivity. "
            "It is not a replacement for real public-outcome validation."
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "quick_validation_report.json", report)

    return {
        "run_id": run_name,
        "run_dir": str(run_dir),
        "base_dataset_path": str(base_dataset_path),
        "evaluation_dataset_path": str(evaluation_dataset_path),
        "synthetic_event_dataset_path": str(synthetic_event_path),
        "scored_backtest_path": str(scored_path),
        "manifest_path": str(run_dir / "manifest.json"),
        "quick_validation_report_path": str(run_dir / "quick_validation_report.json"),
        "prepared_row_count": len(prepared_rows),
        "wildfire_risk_score_auc": auc,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deliberately diverse synthetic evaluation dataset across hazard/vegetation/terrain conditions."
    )
    parser.add_argument("--base-evaluation-dataset", default=None, help="Path to baseline evaluation_dataset.jsonl")
    parser.add_argument("--base-evaluation-root", default=str(DEFAULT_BASE_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-per-cell", type=int, default=2)
    parser.add_argument("--min-selected-rows", type=int, default=18)
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run full public outcome validation on the generated diverse evaluation dataset.",
    )
    parser.add_argument(
        "--validation-output-root",
        default=str(DEFAULT_VALIDATION_OUTPUT_ROOT),
        help="Output root for full validation runs when --run-validation is set.",
    )
    args = parser.parse_args(argv)

    base_dataset_path = (
        Path(args.base_evaluation_dataset).expanduser()
        if args.base_evaluation_dataset
        else _resolve_latest_dataset(Path(args.base_evaluation_root).expanduser())
    )
    result = build_diverse_conditions_dataset(
        base_dataset_path=base_dataset_path,
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        max_per_cell=max(1, int(args.max_per_cell)),
        min_selected_rows=max(1, int(args.min_selected_rows)),
    )

    if bool(args.run_validation):
        validation_run_id = f"{result['run_id']}_validation"
        cmd = [
            sys.executable,
            "scripts/run_public_outcome_validation.py",
            "--evaluation-dataset",
            str(result["evaluation_dataset_path"]),
            "--run-id",
            validation_run_id,
            "--output-root",
            str(Path(args.validation_output_root).expanduser()),
        ]
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
        result["validation_run_id"] = validation_run_id
        result["validation_stdout"] = completed.stdout.strip()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
