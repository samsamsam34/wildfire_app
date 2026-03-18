from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.benchmarking import compare_benchmark_artifacts, load_artifact, run_benchmark_suite


DEFAULT_PACK = Path("benchmark") / "scenario_pack_confidence_v2.json"
DEFAULT_OUTPUT_DIR = Path("benchmark") / "results"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric_values(snapshots: list[dict[str, Any]], path: list[str]) -> list[float]:
    values: list[float] = []
    for row in snapshots:
        current: Any = row
        for segment in path:
            if isinstance(current, dict):
                current = current.get(segment)
            else:
                current = None
                break
        if _is_number(current):
            values.append(float(current))
    return values


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "stddev": None}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "stddev": round(statistics.pstdev(values), 3),
    }


def _extract_snapshots(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in artifact.get("scenario_results", []):
        if not isinstance(item, dict):
            continue
        snap = item.get("snapshot")
        if isinstance(snap, dict):
            rows.append(snap)
    return rows


def _summary_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    snapshots = _extract_snapshots(artifact)
    wildfire_scores = _metric_values(snapshots, ["scores", "wildfire_risk_score"])
    confidence_scores = _metric_values(snapshots, ["confidence", "confidence_score"])
    fallback_weight_fraction = _metric_values(snapshots, ["evidence_metrics", "fallback_weight_fraction"])
    observed_feature_count = _metric_values(snapshots, ["evidence_metrics", "observed_feature_count"])
    suppressed_factor_count = _metric_values(snapshots, ["evidence_metrics", "suppressed_factor_count"])

    risk_spread = None
    if wildfire_scores:
        risk_spread = round(max(wildfire_scores) - min(wildfire_scores), 3)

    return {
        "artifact_path": artifact.get("artifact_path"),
        "summary": artifact.get("summary", {}),
        "distribution": {
            "wildfire_risk_score": _distribution(wildfire_scores),
            "confidence_score": _distribution(confidence_scores),
            "fallback_weight_fraction": _distribution(fallback_weight_fraction),
            "observed_feature_count": _distribution(observed_feature_count),
            "suppressed_factor_count": _distribution(suppressed_factor_count),
        },
        "spread_checks": {
            "wildfire_risk_score_spread": risk_spread,
            "score_clustering_flag": bool(risk_spread is not None and risk_spread < 10.0),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the confidence/accuracy benchmark pack and summarize score spread, fallback pressure, "
            "and suppressed low-evidence factors."
        )
    )
    parser.add_argument("--pack", default=str(DEFAULT_PACK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--compare-to", default=None, help="Optional prior benchmark artifact for drift comparison.")
    args = parser.parse_args(argv)

    artifact = run_benchmark_suite(
        pack_path=Path(args.pack).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
    )
    payload = _summary_payload(artifact)

    if args.compare_to:
        baseline = load_artifact(Path(args.compare_to).expanduser())
        payload["drift"] = compare_benchmark_artifacts(
            baseline_artifact=baseline,
            current_artifact=artifact,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool((artifact.get("summary") or {}).get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
