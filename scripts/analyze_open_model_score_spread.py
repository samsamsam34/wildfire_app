#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_score_variance import run_variance_analysis


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "stddev": 0.0}
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "stddev": round(statistics.pstdev(values), 3),
    }


def _collect_property_signal_stats(fixture_payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = fixture_payload.get("scenarios")
    if not isinstance(scenarios, list):
        return {}
    signal_keys = [
        "near_structure_vegetation_0_5_pct",
        "canopy_adjacency_proxy_pct",
        "vegetation_continuity_proxy_pct",
        "nearest_high_fuel_patch_distance_ft",
    ]
    out: dict[str, Any] = {}
    for key in signal_keys:
        values = [
            _safe_float((row.get("context") or {}).get(key))
            for row in scenarios
            if isinstance(row, dict)
        ]
        nums = [float(v) for v in values if v is not None]
        out[key] = _summarize(nums)
    return out


def run_open_model_spread(
    *,
    fixture_path: Path,
    csv_out: Path | None = None,
) -> dict[str, Any]:
    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    summary = run_variance_analysis(fixture_path=fixture_path, csv_out=csv_out)
    property_signal_stats = _collect_property_signal_stats(fixture_payload)

    warnings: list[str] = []
    wildfire_stats = summary.get("score_stats", {}).get("wildfire_risk_score", {})
    if float(wildfire_stats.get("stddev") or 0.0) < 6.0:
        warnings.append("Wildfire score standard deviation is low; investigate feature compression.")
    if float(wildfire_stats.get("max") or 0.0) - float(wildfire_stats.get("min") or 0.0) < 18.0:
        warnings.append("Wildfire score range is narrow; property-level discrimination may be weak.")

    out = {
        "scenario_count": summary.get("scenario_count"),
        "score_stats": summary.get("score_stats"),
        "fallback_frequency": summary.get("fallback_frequency"),
        "factor_contribution_variance_ranked": summary.get("factor_contribution_variance_ranked"),
        "property_signal_stats": property_signal_stats,
        "clustered_output_warning": summary.get("clustered_output_warning"),
        "open_model_warnings": warnings,
        "rows": summary.get("rows"),
    }
    return out


def _write_detail_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary.get("rows")
    if not isinstance(rows, list):
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scenario_id",
                "address",
                "site_hazard_score",
                "home_ignition_vulnerability_score",
                "insurance_readiness_score",
                "wildfire_risk_score",
                "top_contributors",
                "top_contributions",
            ],
        )
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow(
                {
                    **row,
                    "top_contributors": ";".join(row.get("top_contributors") or []),
                    "top_contributions": ";".join(str(v) for v in (row.get("top_contributions") or [])),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze score spread for the open-data wildfire model.")
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/score_variance_scenarios.json",
        help="Path to scenario fixture JSON.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    fixture_path = Path(args.fixture).expanduser()
    csv_out = Path(args.csv_out).expanduser() if args.csv_out else None
    summary = run_open_model_spread(
        fixture_path=fixture_path,
        csv_out=csv_out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_out:
        _write_detail_csv(summary, Path(args.csv_out).expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
