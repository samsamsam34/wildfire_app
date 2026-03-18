#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.evaluation.no_ground_truth import (
    DEFAULT_FIXTURE_PATH,
    DEFAULT_OUTPUT_ROOT,
    run_no_ground_truth_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-ground-truth model evaluation (coherence, monotonicity, sensitivity, stability, "
            "distribution, alignment, and confidence diagnostics)."
        )
    )
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="", help="Optional fixed run ID for deterministic output naming.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory when run-id exists.")
    parser.add_argument("--verbose", action="store_true", help="Enable info-level logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.INFO if args.verbose else logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run_no_ground_truth_evaluation(
        fixture_path=Path(args.fixture).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        seed=args.seed,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

