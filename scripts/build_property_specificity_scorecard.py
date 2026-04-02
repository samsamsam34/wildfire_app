#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmark" / "results"
DEFAULT_MULTI_REGION_ROOT = REPO_ROOT / "benchmark" / "multi_region_runtime"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "property_specificity_scorecard"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _list_sorted(pattern: str) -> list[Path]:
    paths = sorted(REPO_ROOT.glob(pattern))
    return [path for path in paths if path.is_file()]


def _resolve_current_and_baseline(
    *,
    explicit_current: str | None,
    explicit_baseline: str | None,
    candidates: list[Path],
) -> tuple[Path, Path | None]:
    resolved_candidates = [path.resolve() for path in candidates]
    if explicit_current:
        current = Path(explicit_current).expanduser().resolve()
        if not current.exists():
            raise FileNotFoundError(f"Current artifact not found: {current}")
        if current not in resolved_candidates:
            resolved_candidates.append(current)
            resolved_candidates = sorted(resolved_candidates)
    else:
        if not resolved_candidates:
            raise FileNotFoundError("No artifacts found for current selection.")
        current = resolved_candidates[-1]

    if explicit_baseline:
        baseline = Path(explicit_baseline).expanduser().resolve()
        if not baseline.exists():
            raise FileNotFoundError(f"Baseline artifact not found: {baseline}")
        return current, baseline

    baseline = None
    if current in resolved_candidates:
        idx = resolved_candidates.index(current)
        if idx > 0:
            baseline = resolved_candidates[idx - 1]
    return current, baseline


def _safe_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((float(numerator) / float(denominator)) * 100.0, 2)


def _normalize_specificity_tier(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value in {"property_specific", "address_level", "regional_estimate", "insufficient_data"}:
        return value
    if value in {"limited_regional_estimate", "limited_regional_ready"}:
        return "regional_estimate"
    return "regional_estimate"


def _extract_specificity_tier(snapshot: dict[str, Any]) -> str:
    specificity = snapshot.get("specificity")
    if isinstance(specificity, dict):
        tier = specificity.get("specificity_tier")
        if tier:
            return _normalize_specificity_tier(tier)
    coverage = ((snapshot.get("debug_excerpt") or {}).get("coverage") or {})
    if isinstance(coverage, dict):
        tier = coverage.get("assessment_specificity_tier")
        if tier:
            return _normalize_specificity_tier(tier)
    return _normalize_specificity_tier(snapshot.get("assessment_specificity_tier"))


def _has_footprint_match(snapshot: dict[str, Any]) -> bool:
    if bool(snapshot.get("footprint_used")):
        return True
    geometry = snapshot.get("geometry")
    if not isinstance(geometry, dict):
        return False
    source = str(
        geometry.get("geometry_source")
        or geometry.get("final_structure_geometry_source")
        or ""
    ).strip().lower()
    ring_mode = str(geometry.get("ring_generation_mode") or "").strip().lower()
    return "footprint" in source or "footprint" in ring_mode


def _nearby_differentiation_pass_pct(benchmark_payload: dict[str, Any]) -> float | None:
    nearby = benchmark_payload.get("nearby_differentiation_performance")
    if not isinstance(nearby, dict) or not bool(nearby.get("available")):
        return None
    local = nearby.get("local_subscore_assertions")
    if isinstance(local, dict):
        total = int(local.get("count") or 0)
        passed = int(local.get("passed") or 0)
        pct = _safe_pct(passed, total)
        if pct is not None:
            return pct
    total = int(nearby.get("assertion_count") or 0)
    passed = int(nearby.get("assertion_pass_count") or 0)
    return _safe_pct(passed, total)


def _extract_benchmark_metrics(benchmark_payload: dict[str, Any]) -> dict[str, Any]:
    scenario_results = benchmark_payload.get("scenario_results")
    if not isinstance(scenario_results, list):
        scenario_results = []

    specificity_counts = {
        "property_specific": 0,
        "address_level": 0,
        "regional_estimate": 0,
        "insufficient_data": 0,
    }
    footprint_match_count = 0
    scenario_count = 0

    for row in scenario_results:
        if not isinstance(row, dict):
            continue
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        scenario_count += 1
        tier = _extract_specificity_tier(snapshot)
        specificity_counts[tier] = int(specificity_counts.get(tier, 0)) + 1
        if _has_footprint_match(snapshot):
            footprint_match_count += 1

    nearby_pass_pct = _nearby_differentiation_pass_pct(benchmark_payload)

    return {
        "scenario_count": scenario_count,
        "specificity_counts": specificity_counts,
        "percent_property_specific": _safe_pct(specificity_counts["property_specific"], scenario_count),
        "percent_address_level": _safe_pct(specificity_counts["address_level"], scenario_count),
        "percent_regional_estimate": _safe_pct(specificity_counts["regional_estimate"], scenario_count),
        "percent_insufficient_data": _safe_pct(specificity_counts["insufficient_data"], scenario_count),
        "footprint_match_count": footprint_match_count,
        "percent_with_footprint_match": _safe_pct(footprint_match_count, scenario_count),
        "percent_nearby_home_differentiation_pass": nearby_pass_pct,
    }


def _is_naip_available_status(status: str) -> bool:
    positive = {"observed", "loaded", "ok", "available", "present", "configured"}
    negative = {
        "",
        "missing",
        "not_configured",
        "error",
        "failed",
        "fallback_used",
        "fallback",
        "outside_extent",
        "sampling_failed",
        "unavailable",
        "unknown",
    }
    if status in positive:
        return True
    if status in negative:
        return False
    return False


def _extract_multi_region_naip_metrics(runtime_payload: dict[str, Any]) -> dict[str, Any]:
    samples = runtime_payload.get("samples")
    if not isinstance(samples, list):
        samples = []

    sample_count = 0
    naip_available_count = 0
    availability_basis = "naip_structure_features_status"

    for row in samples:
        if not isinstance(row, dict):
            continue
        sample_count += 1
        property_context = row.get("property_level_context")
        if not isinstance(property_context, dict):
            continue

        enrichment_status = property_context.get("enrichment_source_status")
        if isinstance(enrichment_status, dict):
            naip_status = enrichment_status.get("naip_structure_features")
            if isinstance(naip_status, dict):
                status = str(naip_status.get("status") or "").strip().lower()
                has_status = _is_naip_available_status(status)
                has_reference = bool(
                    str(naip_status.get("path") or "").strip()
                    or str(naip_status.get("runtime_key") or "").strip()
                    or str(naip_status.get("source") or "").strip()
                )
                if has_status or (has_reference and status not in {"missing", "not_configured", "error", "failed"}):
                    naip_available_count += 1
                continue

        # Fallback to broader near-structure vegetation availability when NAIP-specific status is absent.
        coverage = row.get("debug_excerpt")
        if isinstance(coverage, dict):
            cov = coverage.get("coverage")
            if isinstance(cov, dict):
                feature_cov = cov.get("feature_coverage_summary")
                if isinstance(feature_cov, dict):
                    availability_basis = "near_structure_vegetation_available_proxy"
                    if bool(feature_cov.get("near_structure_vegetation_available")):
                        naip_available_count += 1

    return {
        "sample_count": sample_count,
        "naip_structure_features_available_count": naip_available_count,
        "percent_with_naip_structure_features": _safe_pct(naip_available_count, sample_count),
        "availability_basis": availability_basis,
    }


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 2)


def _build_scorecard(
    *,
    benchmark_current_path: Path,
    benchmark_baseline_path: Path | None,
    multi_region_current_path: Path,
    multi_region_baseline_path: Path | None,
) -> dict[str, Any]:
    benchmark_current = _load_json(benchmark_current_path)
    benchmark_baseline = _load_json(benchmark_baseline_path) if benchmark_baseline_path else None
    multi_current = _load_json(multi_region_current_path)
    multi_baseline = _load_json(multi_region_baseline_path) if multi_region_baseline_path else None

    current_benchmark_metrics = _extract_benchmark_metrics(benchmark_current)
    baseline_benchmark_metrics = (
        _extract_benchmark_metrics(benchmark_baseline) if isinstance(benchmark_baseline, dict) else None
    )
    current_naip_metrics = _extract_multi_region_naip_metrics(multi_current)
    baseline_naip_metrics = (
        _extract_multi_region_naip_metrics(multi_baseline) if isinstance(multi_baseline, dict) else None
    )

    current = {
        "percent_property_specific": current_benchmark_metrics["percent_property_specific"],
        "percent_address_level": current_benchmark_metrics["percent_address_level"],
        "percent_regional_estimate": current_benchmark_metrics["percent_regional_estimate"],
        "percent_with_footprint_match": current_benchmark_metrics["percent_with_footprint_match"],
        "percent_with_naip_structure_features": current_naip_metrics["percent_with_naip_structure_features"],
        "percent_nearby_home_differentiation_pass": current_benchmark_metrics[
            "percent_nearby_home_differentiation_pass"
        ],
        "denominators": {
            "benchmark_scenarios": current_benchmark_metrics["scenario_count"],
            "multi_region_samples": current_naip_metrics["sample_count"],
        },
        "counts": {
            "specificity": current_benchmark_metrics["specificity_counts"],
            "footprint_match_count": current_benchmark_metrics["footprint_match_count"],
            "naip_structure_features_available_count": current_naip_metrics[
                "naip_structure_features_available_count"
            ],
        },
        "naip_availability_basis": current_naip_metrics["availability_basis"],
    }

    baseline: dict[str, Any] | None = None
    if baseline_benchmark_metrics is not None or baseline_naip_metrics is not None:
        baseline = {
            "percent_property_specific": (
                baseline_benchmark_metrics["percent_property_specific"]
                if baseline_benchmark_metrics
                else None
            ),
            "percent_address_level": (
                baseline_benchmark_metrics["percent_address_level"]
                if baseline_benchmark_metrics
                else None
            ),
            "percent_regional_estimate": (
                baseline_benchmark_metrics["percent_regional_estimate"]
                if baseline_benchmark_metrics
                else None
            ),
            "percent_with_footprint_match": (
                baseline_benchmark_metrics["percent_with_footprint_match"]
                if baseline_benchmark_metrics
                else None
            ),
            "percent_with_naip_structure_features": (
                baseline_naip_metrics["percent_with_naip_structure_features"]
                if baseline_naip_metrics
                else None
            ),
            "percent_nearby_home_differentiation_pass": (
                baseline_benchmark_metrics["percent_nearby_home_differentiation_pass"]
                if baseline_benchmark_metrics
                else None
            ),
            "denominators": {
                "benchmark_scenarios": (
                    baseline_benchmark_metrics["scenario_count"] if baseline_benchmark_metrics else None
                ),
                "multi_region_samples": (
                    baseline_naip_metrics["sample_count"] if baseline_naip_metrics else None
                ),
            },
        }

    delta_vs_baseline = {
        "percent_property_specific": _delta(
            current["percent_property_specific"],
            baseline.get("percent_property_specific") if baseline else None,
        ),
        "percent_address_level": _delta(
            current["percent_address_level"],
            baseline.get("percent_address_level") if baseline else None,
        ),
        "percent_regional_estimate": _delta(
            current["percent_regional_estimate"],
            baseline.get("percent_regional_estimate") if baseline else None,
        ),
        "percent_with_footprint_match": _delta(
            current["percent_with_footprint_match"],
            baseline.get("percent_with_footprint_match") if baseline else None,
        ),
        "percent_with_naip_structure_features": _delta(
            current["percent_with_naip_structure_features"],
            baseline.get("percent_with_naip_structure_features") if baseline else None,
        ),
        "percent_nearby_home_differentiation_pass": _delta(
            current["percent_nearby_home_differentiation_pass"],
            baseline.get("percent_nearby_home_differentiation_pass") if baseline else None,
        ),
    }

    notes: list[str] = []
    if current["percent_nearby_home_differentiation_pass"] is None:
        notes.append(
            "Nearby-home differentiation pass rate is unavailable because the selected benchmark artifact "
            "does not contain nearby_differentiation_performance metrics."
        )
    if current["percent_with_naip_structure_features"] is None:
        notes.append(
            "NAIP structure-feature coverage is unavailable in selected multi-region artifact samples."
        )
    if baseline is None:
        notes.append("No baseline artifact available for comparison.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "benchmark_current_artifact": str(benchmark_current_path),
            "benchmark_baseline_artifact": str(benchmark_baseline_path) if benchmark_baseline_path else None,
            "multi_region_current_artifact": str(multi_region_current_path),
            "multi_region_baseline_artifact": str(multi_region_baseline_path) if multi_region_baseline_path else None,
        },
        "current": current,
        "baseline": baseline,
        "delta_vs_baseline": delta_vs_baseline,
        "notes": notes,
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _build_summary_markdown(scorecard: dict[str, Any]) -> str:
    current = scorecard.get("current") if isinstance(scorecard.get("current"), dict) else {}
    baseline = scorecard.get("baseline") if isinstance(scorecard.get("baseline"), dict) else {}
    delta = scorecard.get("delta_vs_baseline") if isinstance(scorecard.get("delta_vs_baseline"), dict) else {}
    sources = scorecard.get("sources") if isinstance(scorecard.get("sources"), dict) else {}
    notes = scorecard.get("notes") if isinstance(scorecard.get("notes"), list) else []

    lines = [
        "# Property-Specificity Scorecard",
        "",
        f"- Generated: `{scorecard.get('generated_at')}`",
        f"- Benchmark artifact: `{sources.get('benchmark_current_artifact')}`",
        f"- Multi-region artifact: `{sources.get('multi_region_current_artifact')}`",
        "",
        "## Current",
        "",
        f"- Percent property-specific: `{_fmt_pct(current.get('percent_property_specific'))}`",
        f"- Percent address-level: `{_fmt_pct(current.get('percent_address_level'))}`",
        f"- Percent regional-estimate: `{_fmt_pct(current.get('percent_regional_estimate'))}`",
        f"- Percent with footprint match: `{_fmt_pct(current.get('percent_with_footprint_match'))}`",
        f"- Percent with NAIP structure features: `{_fmt_pct(current.get('percent_with_naip_structure_features'))}`",
        f"- Percent nearby-home differentiation pass: `{_fmt_pct(current.get('percent_nearby_home_differentiation_pass'))}`",
        "",
        "## Baseline Comparison",
        "",
        f"- Baseline benchmark artifact: `{sources.get('benchmark_baseline_artifact') or 'none'}`",
        f"- Baseline multi-region artifact: `{sources.get('multi_region_baseline_artifact') or 'none'}`",
        f"- Property-specific delta: `{_fmt_pct(delta.get('percent_property_specific'))}`",
        f"- Address-level delta: `{_fmt_pct(delta.get('percent_address_level'))}`",
        f"- Regional-estimate delta: `{_fmt_pct(delta.get('percent_regional_estimate'))}`",
        f"- Footprint-match delta: `{_fmt_pct(delta.get('percent_with_footprint_match'))}`",
        f"- NAIP-structure-feature delta: `{_fmt_pct(delta.get('percent_with_naip_structure_features'))}`",
        f"- Nearby-home differentiation-pass delta: `{_fmt_pct(delta.get('percent_nearby_home_differentiation_pass'))}`",
    ]
    if notes:
        lines.extend(["", "## Notes"])
        for note in notes:
            lines.append(f"- {str(note)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an internal property-specificity scorecard from benchmark and runtime artifacts."
        )
    )
    parser.add_argument("--benchmark-artifact", default="", help="Optional benchmark_run_*.json path.")
    parser.add_argument("--baseline-benchmark-artifact", default="", help="Optional baseline benchmark artifact path.")
    parser.add_argument("--multi-region-artifact", default="", help="Optional multi-region runtime artifact path.")
    parser.add_argument("--baseline-multi-region-artifact", default="", help="Optional baseline multi-region artifact path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="", help="Optional run id for output directory naming.")
    args = parser.parse_args(argv)

    benchmark_candidates = _list_sorted("benchmark/results/benchmark_run_*.json")
    multi_region_candidates = _list_sorted("benchmark/multi_region_runtime/*.json")
    if not benchmark_candidates:
        raise FileNotFoundError("No benchmark artifacts found under benchmark/results.")
    if not multi_region_candidates:
        raise FileNotFoundError("No multi-region runtime artifacts found under benchmark/multi_region_runtime.")

    benchmark_current, benchmark_baseline = _resolve_current_and_baseline(
        explicit_current=(args.benchmark_artifact or None),
        explicit_baseline=(args.baseline_benchmark_artifact or None),
        candidates=benchmark_candidates,
    )
    multi_current, multi_baseline = _resolve_current_and_baseline(
        explicit_current=(args.multi_region_artifact or None),
        explicit_baseline=(args.baseline_multi_region_artifact or None),
        candidates=multi_region_candidates,
    )

    scorecard = _build_scorecard(
        benchmark_current_path=benchmark_current,
        benchmark_baseline_path=benchmark_baseline,
        multi_region_current_path=multi_current,
        multi_region_baseline_path=multi_baseline,
    )

    run_id = (args.run_id or _timestamp()).strip()
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "property_specificity_scorecard.json"
    md_path = run_dir / "summary.md"
    json_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_summary_markdown(scorecard), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "json_path": str(json_path),
                "summary_path": str(md_path),
                "sources": scorecard.get("sources"),
                "current": scorecard.get("current"),
                "delta_vs_baseline": scorecard.get("delta_vs_baseline"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
