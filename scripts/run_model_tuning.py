#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.event_backtesting import DEFAULT_DATASET_PATH
from backend.model_tuning import DEFAULT_TUNING_RESULTS_DIR, run_model_tuning


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic model tuning experiments from event backtest datasets and emit JSON/markdown artifacts."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Path to event dataset file (CSV/JSON/GeoJSON). Can be repeated.",
    )
    parser.add_argument(
        "--scoring-parameters",
        default="config/scoring_parameters.yaml",
        help="Path to base scoring parameters file.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_TUNING_RESULTS_DIR))
    parser.add_argument("--ruleset-id", default=None)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Exit non-zero when best candidate does not improve objective score over baseline.",
    )
    args = parser.parse_args(argv)

    datasets = args.dataset if args.dataset else [str(DEFAULT_DATASET_PATH)]
    artifact = run_model_tuning(
        dataset_paths=datasets,
        scoring_parameters_path=Path(args.scoring_parameters).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        ruleset_id=args.ruleset_id,
        max_candidates=max(1, int(args.max_candidates)),
    )

    summary = artifact.get("summary", {})
    best = artifact.get("best_experiment") or {}
    print(
        json.dumps(
            {
                "artifact_path": artifact.get("artifact_path"),
                "markdown_summary_path": artifact.get("markdown_summary_path"),
                "candidate_count": summary.get("candidate_count"),
                "passing_guardrail_count": summary.get("passing_guardrail_count"),
                "baseline_objective_score": summary.get("baseline_objective_score"),
                "best_objective_score": summary.get("best_objective_score"),
                "best_parameter_set_id": best.get("parameter_set_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if int(summary.get("passing_guardrail_count") or 0) <= 0:
        return 2

    if args.require_improvement:
        baseline = float(summary.get("baseline_objective_score") or 0.0)
        best_score = float(summary.get("best_objective_score") or 0.0)
        if best_score <= baseline:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
