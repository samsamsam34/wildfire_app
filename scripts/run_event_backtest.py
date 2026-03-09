#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.event_backtesting import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_DIR,
    run_event_backtest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run event-based wildfire backtesting against labeled outcomes and emit JSON + markdown artifacts."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Path to event dataset file (CSV/JSON/GeoJSON). Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--ruleset-id", default=None, help="Optional ruleset override for all records.")
    parser.add_argument(
        "--reuse-existing-assessments",
        action="store_true",
        help="Reuse record.assessment_id payloads when present.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=1,
        help="Fail when fewer than this many records are processed.",
    )
    args = parser.parse_args(argv)

    datasets = args.dataset if args.dataset else [str(DEFAULT_DATASET_PATH)]
    artifact = run_event_backtest(
        dataset_paths=datasets,
        output_dir=Path(args.output_dir).expanduser(),
        ruleset_id=args.ruleset_id,
        reuse_existing_assessments=bool(args.reuse_existing_assessments),
    )

    summary = artifact.get("summary", {})
    print(
        json.dumps(
            {
                "artifact_path": artifact.get("artifact_path"),
                "markdown_summary_path": artifact.get("markdown_summary_path"),
                "record_count": summary.get("record_count"),
                "event_count": summary.get("event_count"),
                "high_evidence_count": summary.get("high_evidence_count"),
                "fallback_heavy_count": summary.get("fallback_heavy_count"),
                "false_low_count": artifact.get("analysis", {}).get("false_low_count"),
                "false_high_count": artifact.get("analysis", {}).get("false_high_count"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if int(summary.get("record_count") or 0) < max(1, int(args.min_records)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
