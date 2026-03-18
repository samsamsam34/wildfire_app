#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.evaluation.no_ground_truth import (
    DEFAULT_FIXTURE_PATH,
    DEFAULT_OUTPUT_ROOT,
    run_no_ground_truth_evaluation,
)
from backend.no_ground_truth_paths import resolve_no_ground_truth_artifact_root

REQUIRED_ARTIFACTS = [
    "evaluation_manifest.json",
    "summary.md",
    "monotonicity_results.json",
    "counterfactual_results.json",
    "stability_results.json",
    "distribution_results.json",
    "confidence_diagnostics.json",
]
OPTIONAL_ARTIFACTS = [
    "benchmark_alignment_results.json",
]


def _minimal_fixture_payload() -> dict[str, Any]:
    return {
        "no_ground_truth_eval_version": "1.0.0",
        "description": "Auto-generated minimal no-ground-truth fixture fallback.",
        "seed": 17,
        "scenarios": [
            {
                "scenario_id": "minimal_baseline",
                "description": "Minimal fallback scenario to keep diagnostics pipeline runnable.",
                "region": "synthetic_fallback",
                "segments": ["fallback_fixture"],
                "external_signals": {"fire_regime_index": 1.0, "hazard_zone_class": "moderate"},
                "input_payload": {
                    "address": "1 Minimal Fixture Way, Synthetic, ST",
                    "attributes": {
                        "roof_type": "class a",
                        "vent_type": "ember-resistant",
                        "defensible_space_ft": 20,
                        "construction_year": 2010,
                    },
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
                    "audience": "homeowner",
                    "tags": ["no-gt", "fallback-fixture"],
                },
                "location": {"lat": 39.7392, "lon": -104.9903, "geocode_source": "no-gt-fallback"},
                "context": {
                    "burn_probability_index": 45.0,
                    "hazard_severity_index": 42.0,
                    "slope_index": 36.0,
                    "fuel_index": 39.0,
                    "moisture_index": 44.0,
                    "canopy_index": 35.0,
                    "historic_fire_index": 31.0,
                },
            }
        ],
        "monotonicity_rules": [],
        "counterfactual_groups": [],
        "stability_tests": [],
        "benchmark_alignment_rules": [],
    }


def _write_fallback_fixture(output_root: Path) -> Path:
    fixture_dir = output_root / "_autogen_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "minimal_no_ground_truth_fixture.json"
    fixture_path.write_text(
        json.dumps(_minimal_fixture_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return fixture_path


def _normalize(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line, file=sys.stderr)


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
    requested_fixture_path = Path(args.fixture).expanduser()
    output_root = Path(args.output_root).expanduser()
    dashboard_artifact_root = resolve_no_ground_truth_artifact_root()
    default_fixture_path = Path(DEFAULT_FIXTURE_PATH).expanduser()
    using_default_fixture = _normalize(requested_fixture_path) == _normalize(default_fixture_path)
    fixture_path = requested_fixture_path
    auto_fixture_used = False
    if not fixture_path.exists() and using_default_fixture:
        fixture_path = _write_fallback_fixture(output_root)
        auto_fixture_used = True

    writer_root = _normalize(output_root)
    loader_root = _normalize(dashboard_artifact_root)
    _print_lines(
        [
            f"[no-gt-eval] repo_root: {REPO_ROOT}",
            f"[no-gt-eval] fixture_path: {fixture_path}",
            f"[no-gt-eval] artifact_root (writer): {output_root}",
            f"[no-gt-eval] artifact_root (dashboard loader): {dashboard_artifact_root}",
        ]
    )
    if writer_root != loader_root:
        _print_lines(
            [
                "[no-gt-eval] WARNING: writer artifact root and dashboard artifact root differ.",
                "[no-gt-eval] This can cause the dashboard to report 'No run directories found'.",
                "[no-gt-eval] Align with --output-root or set WF_NO_GROUND_TRUTH_EVAL_DIR.",
            ]
        )
    if auto_fixture_used:
        _print_lines(
            [
                "[no-gt-eval] WARNING: default fixture was missing.",
                f"[no-gt-eval] Generated fallback fixture: {fixture_path}",
            ]
        )

    try:
        result = run_no_ground_truth_evaluation(
            fixture_path=fixture_path,
            output_root=output_root,
            run_id=(args.run_id or None),
            seed=args.seed,
            overwrite=bool(args.overwrite),
        )
    except FileNotFoundError as err:
        _print_lines(
            [
                f"[no-gt-eval] ERROR (missing fixture data): {err}",
                "[no-gt-eval] Ensure the fixture file exists or re-run without a custom fixture path.",
            ]
        )
        return 2
    except ModuleNotFoundError as err:
        _print_lines(
            [
                f"[no-gt-eval] ERROR (import/module path): {err}",
                "[no-gt-eval] Run from repository root so local modules resolve correctly.",
            ]
        )
        return 2
    except Exception as err:  # pragma: no cover - defensive failure categorization
        _print_lines(
            [
                f"[no-gt-eval] ERROR (unhandled evaluation exception): {err}",
                "[no-gt-eval] Check fixture validity and evaluation module stack trace for details.",
            ]
        )
        raise

    run_dir = Path(result.get("run_dir") or output_root)
    _print_lines(
        [
            f"[no-gt-eval] run_id: {result.get('run_id')}",
            f"[no-gt-eval] run_dir: {run_dir}",
        ]
    )
    missing_required = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    for filename in REQUIRED_ARTIFACTS:
        path = run_dir / filename
        if path.exists():
            _print_lines([f"[no-gt-eval] wrote: {path}"])
    missing_optional = [name for name in OPTIONAL_ARTIFACTS if not (run_dir / name).exists()]
    for filename in OPTIONAL_ARTIFACTS:
        path = run_dir / filename
        if path.exists():
            _print_lines([f"[no-gt-eval] wrote optional: {path}"])
    if missing_optional:
        _print_lines(
            [f"[no-gt-eval] optional section skipped: {name}" for name in missing_optional]
        )
    if missing_required:
        _print_lines(
            [
                "[no-gt-eval] ERROR (artifact path mismatch or write failure):",
                *[f"[no-gt-eval] missing required artifact: {name}" for name in missing_required],
            ]
        )
        return 3

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
