from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.benchmarking import (
    DEFAULT_NEARBY_PACK_PATH,
    DEFAULT_PACK_PATH,
    DEFAULT_RESULTS_DIR,
    compare_benchmark_artifacts,
    evaluate_nearby_release_gate,
    load_artifact,
    run_benchmark_suite,
)


def _write_nearby_summary_artifacts(
    *,
    artifact: dict[str, object],
    nearby_release_gate: dict[str, object],
) -> dict[str, str] | None:
    nearby = (
        artifact.get("nearby_differentiation_performance")
        if isinstance(artifact.get("nearby_differentiation_performance"), dict)
        else {}
    )
    if not bool(nearby.get("available")):
        return None
    artifact_path = Path(str(artifact.get("artifact_path") or "")).expanduser()
    if not str(artifact_path):
        return None
    pair_count = int(((nearby.get("separation_analysis") or {}).get("pair_count") or 0))
    separation_success_rate = nearby.get("separation_success_rate")
    abstention_success_rate = nearby.get("abstention_success_rate_when_data_weak")
    false_similarity_case_count = int(nearby.get("false_similarity_case_count") or 0)
    false_similarity_cases = (
        nearby.get("false_similarity_cases")
        if isinstance(nearby.get("false_similarity_cases"), list)
        else []
    )
    payload = {
        "artifact_path": str(artifact_path),
        "generated_at": artifact.get("generated_at"),
        "nearby_differentiation_release_gate": nearby_release_gate,
        "summary": {
            "pair_count": pair_count,
            "separation_success_rate": separation_success_rate,
            "abstention_success_rate_when_data_weak": abstention_success_rate,
            "false_similarity_case_count": false_similarity_case_count,
        },
        "false_similarity_cases": false_similarity_cases,
    }
    summary_json_path = artifact_path.with_name(f"{artifact_path.stem}_nearby_release_gate.json")
    summary_md_path = artifact_path.with_name(f"{artifact_path.stem}_nearby_release_gate.md")
    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Nearby-Home Differentiation Release Gate",
        "",
        f"- Source benchmark artifact: `{artifact_path}`",
        f"- Pair count: `{pair_count}`",
        f"- Separation success rate: `{separation_success_rate}`",
        f"- Honest-abstention success rate (weak data): `{abstention_success_rate}`",
        f"- False similarity cases (collapsed without warning): `{false_similarity_case_count}`",
        f"- Release gate passed: `{nearby_release_gate.get('passed')}`",
    ]
    reasons = nearby_release_gate.get("reasons") if isinstance(nearby_release_gate.get("reasons"), list) else []
    if reasons:
        lines.append("")
        lines.append("## Gate Reasons")
        for reason in reasons:
            lines.append(f"- {reason}")
    if false_similarity_cases:
        lines.append("")
        lines.append("## False Similarity Cases")
        for row in false_similarity_cases[:25]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                + f"{row.get('assertion_id')} | {row.get('left')} vs {row.get('right')} | "
                + f"{row.get('metric')} | delta={row.get('absolute_delta')} threshold={row.get('collapse_threshold')}"
            )
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(summary_json_path), "markdown": str(summary_md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the wildfire benchmark scenario suite, emit artifact JSON, and optionally compare "
            "against a prior run for drift detection."
        )
    )
    parser.add_argument("--pack", default=str(DEFAULT_PACK_PATH))
    parser.add_argument(
        "--nearby-suite",
        action="store_true",
        help=(
            "Run the dedicated nearby-home differentiation suite "
            f"({DEFAULT_NEARBY_PACK_PATH})"
        ),
    )
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
    parser.add_argument(
        "--enforce-nearby-release-gate",
        action="store_true",
        help=(
            "Fail nonzero when nearby-home scenarios collapse into similar outputs "
            "without low-specificity warnings."
        ),
    )
    args = parser.parse_args(argv)

    pack_path = Path(args.pack).expanduser()
    if args.nearby_suite:
        pack_path = DEFAULT_NEARBY_PACK_PATH

    artifact = run_benchmark_suite(
        pack_path=pack_path,
        output_dir=Path(args.output_dir).expanduser(),
    )
    summary = artifact.get("summary", {})
    nearby = (
        artifact.get("nearby_differentiation_performance")
        if isinstance(artifact.get("nearby_differentiation_performance"), dict)
        else {}
    )
    print(
        json.dumps(
            {
                "artifact_path": artifact.get("artifact_path"),
                "scenario_count": summary.get("scenario_count"),
                "scenario_failures": summary.get("scenario_failures"),
                "assertion_failures": summary.get("assertion_failures"),
                "passed": summary.get("passed"),
                "nearby_differentiation_performance": {
                    "available": nearby.get("available"),
                    "scenario_count": nearby.get("scenario_count"),
                    "assertion_fail_count": nearby.get("assertion_fail_count"),
                    "local_subscore_assertion_fail_count": (nearby.get("local_subscore_assertions") or {}).get("failed"),
                    "confidence_caution_assertion_fail_count": (nearby.get("confidence_caution_assertions") or {}).get("failed"),
                    "separation_achieved_count": (nearby.get("separation_analysis") or {}).get("separation_achieved_count"),
                    "collapsed_toward_similarity_count": (nearby.get("separation_analysis") or {}).get("collapsed_toward_similarity_count"),
                    "collapsed_correctly_flagged_low_specificity_count": (
                        (nearby.get("separation_analysis") or {}).get("collapsed_correctly_flagged_low_specificity_count")
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    nearby_release_gate = evaluate_nearby_release_gate(
        artifact,
        require_available=bool(args.enforce_nearby_release_gate or args.nearby_suite),
    )
    print(
        json.dumps(
            {"nearby_release_gate": nearby_release_gate},
            indent=2,
            sort_keys=True,
        )
    )
    nearby_summary_artifacts = _write_nearby_summary_artifacts(
        artifact=artifact,
        nearby_release_gate=nearby_release_gate,
    )
    if nearby_summary_artifacts:
        print(
            json.dumps(
                {"nearby_release_gate_artifacts": nearby_summary_artifacts},
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
    if bool(args.enforce_nearby_release_gate or args.nearby_suite) and not bool(
        nearby_release_gate.get("passed")
    ):
        return 3
    if args.fail_on_drift and drift and int(drift.get("material_drift_count", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
