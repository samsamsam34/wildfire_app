from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.benchmarking import (
    DEFAULT_PACK_PATH,
    DEFAULT_RESULTS_DIR,
    compare_benchmark_artifacts,
    load_artifact,
    run_benchmark_suite,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the wildfire benchmark scenario suite, emit artifact JSON, and optionally compare "
            "against a prior run for drift detection."
        )
    )
    parser.add_argument("--pack", default=str(DEFAULT_PACK_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--compare-to", default=None, help="Optional previous benchmark artifact path.")
    parser.add_argument("--score-drift-threshold", type=float, default=8.0)
    parser.add_argument("--confidence-drift-threshold", type=float, default=8.0)
    parser.add_argument("--contribution-drift-threshold", type=float, default=10.0)
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit nonzero when drift is classified as material.",
    )
    args = parser.parse_args(argv)

    artifact = run_benchmark_suite(
        pack_path=Path(args.pack).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
    )
    summary = artifact.get("summary", {})
    print(
        json.dumps(
            {
                "artifact_path": artifact.get("artifact_path"),
                "scenario_count": summary.get("scenario_count"),
                "scenario_failures": summary.get("scenario_failures"),
                "assertion_failures": summary.get("assertion_failures"),
                "passed": summary.get("passed"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    drift = None
    if args.compare_to:
        baseline = load_artifact(Path(args.compare_to).expanduser())
        drift = compare_benchmark_artifacts(
            baseline_artifact=baseline,
            current_artifact=artifact,
            score_drift_threshold=float(args.score_drift_threshold),
            confidence_drift_threshold=float(args.confidence_drift_threshold),
            contribution_drift_threshold=float(args.contribution_drift_threshold),
        )
        print(
            json.dumps(
                {
                    "drift_summary": drift.get("summary"),
                    "material_drift_count": drift.get("material_drift_count"),
                    "material_drift_scenarios": drift.get("material_drift_scenarios"),
                },
                indent=2,
                sort_keys=True,
            )
        )

    success = bool(summary.get("passed"))
    if not success:
        return 1
    if args.fail_on_drift and drift and int(drift.get("material_drift_count", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
