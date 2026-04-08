#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.public_outcome_validation import (
    DEFAULT_THRESHOLDS,
    evaluate_public_outcome_dataset_file,
    write_evaluation_rows_csv,
)


def evaluate_dataset(
    *,
    dataset_path: Path,
    output_json: Path,
    output_csv: Path | None = None,
    thresholds: list[float] | None = None,
    bins: int = 10,
    allow_label_derived_target: bool = False,
    allow_surrogate_wildfire_score: bool = False,
) -> dict[str, Any]:
    report, rows = evaluate_public_outcome_dataset_file(
        dataset_path=dataset_path,
        thresholds=thresholds or list(DEFAULT_THRESHOLDS),
        bins=bins,
        allow_label_derived_target=bool(allow_label_derived_target),
        allow_surrogate_wildfire_score=bool(allow_surrogate_wildfire_score),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if output_csv is not None:
        write_evaluation_rows_csv(rows=rows, output_csv=output_csv)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate current wildfire model discrimination/calibration against public structure-damage outcomes."
    )
    parser.add_argument(
        "--dataset",
        default="benchmark/calibration/public_outcome_calibration_dataset.json",
        help="Calibration dataset path from build_calibration_dataset.py.",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark/calibration/public_outcome_evaluation.json",
        help="Evaluation report output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional per-record summary CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        default="30,40,50,60,70,80",
        help="Comma-separated wildfire-risk thresholds for PR/confusion summaries.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins for calibration tables.",
    )
    parser.add_argument(
        "--allow-label-derived-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to deriving target from outcome label when structure_loss_or_major_damage is missing.",
    )
    parser.add_argument(
        "--allow-surrogate-wildfire-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to surrogate wildfire score when wildfire_risk_score is missing.",
    )
    args = parser.parse_args()

    thresholds = [float(token) for token in str(args.thresholds).split(",") if token.strip()]
    report = evaluate_dataset(
        dataset_path=Path(args.dataset).expanduser(),
        output_json=Path(args.output_json).expanduser(),
        output_csv=(Path(args.output_csv).expanduser() if args.output_csv else None),
        thresholds=thresholds,
        bins=max(2, int(args.bins)),
        allow_label_derived_target=bool(args.allow_label_derived_target),
        allow_surrogate_wildfire_score=bool(args.allow_surrogate_wildfire_score),
    )

    print(
        json.dumps(
            {
                "dataset": str(Path(args.dataset).expanduser()),
                "output_json": str(Path(args.output_json).expanduser()),
                "row_count_labeled": report.get("row_count_labeled"),
                "wildfire_risk_score_auc": (
                    (report.get("discrimination_metrics") or {}).get("wildfire_risk_score_auc")
                ),
                "wildfire_probability_proxy_brier": (
                    (report.get("brier_scores") or {}).get("wildfire_probability_proxy")
                ),
                "default_threshold_70_precision": (
                    (report.get("default_threshold_70") or {}).get("precision")
                ),
                "default_threshold_70_recall": (
                    (report.get("default_threshold_70") or {}).get("recall")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
