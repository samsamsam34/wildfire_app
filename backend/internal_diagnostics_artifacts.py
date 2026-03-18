from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.no_ground_truth_paths import resolve_no_ground_truth_artifact_root

DEFAULT_ARTIFACT_ROOT = resolve_no_ground_truth_artifact_root()
SECTION_FILES = {
    "monotonicity": "monotonicity_results.json",
    "counterfactual": "counterfactual_results.json",
    "stability": "stability_results.json",
    "distribution": "distribution_results.json",
    "benchmark_alignment": "benchmark_alignment_results.json",
    "confidence_diagnostics": "confidence_diagnostics.json",
    "comparison_to_previous": "comparison_to_previous.json",
}
COMPARABLE_SECTION_KEYS = (
    "monotonicity",
    "counterfactual",
    "stability",
    "distribution",
    "benchmark_alignment",
    "confidence_diagnostics",
)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _extract_recommendations_from_markdown(text: str | None) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    lines = text.splitlines()
    out: list[str] = []
    in_recommendation = False
    for line in lines:
        raw = line.strip()
        if raw.lower().startswith("## recommendation"):
            in_recommendation = True
            continue
        if in_recommendation and raw.startswith("## "):
            break
        if in_recommendation and raw.startswith("- "):
            out.append(raw[2:].strip())
    return out


def _bundle_section_payloads(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = bundle.get("sections") if isinstance(bundle.get("sections"), dict) else {}
    payloads: dict[str, dict[str, Any]] = {}
    for section_key in COMPARABLE_SECTION_KEYS:
        row = sections.get(section_key) if isinstance(sections.get(section_key), dict) else {}
        payload = row.get("payload")
        payloads[section_key] = payload if isinstance(payload, dict) else {}
    return payloads


def resolve_artifact_root(path_hint: str | Path | None = None) -> Path:
    return resolve_no_ground_truth_artifact_root(path_hint)


def list_no_ground_truth_runs(
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_artifact_root(artifact_root)
    root_exists = root.exists()
    is_dir = root.is_dir()
    if not root_exists or not is_dir:
        return {
            "available": False,
            "artifact_root": str(root),
            "artifact_root_exists": bool(root_exists),
            "artifact_root_is_dir": bool(is_dir),
            "run_directory_count": 0,
            "runs": [],
            "message": (
                "No offline evaluation artifacts found yet. "
                f"Checked artifact root: {root} (exists={root_exists}, is_dir={is_dir}). "
                "Run `python scripts/run_no_ground_truth_evaluation.py` first."
            ),
        }

    runs: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / "evaluation_manifest.json"
        manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None
        generated_at = (
            str(manifest.get("generated_at")) if isinstance(manifest, dict) and manifest.get("generated_at") else None
        )
        if generated_at is None:
            generated_at = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat()
        runs.append(
            {
                "run_id": entry.name,
                "path": str(entry),
                "generated_at": generated_at,
                "has_manifest": bool(manifest),
                "status_summary": (
                    manifest.get("status_summary") if isinstance(manifest, dict) else {}
                ),
            }
        )

    runs.sort(key=lambda row: str(row.get("generated_at") or ""), reverse=True)
    run_count = len(runs)
    return {
        "available": bool(run_count),
        "artifact_root": str(root),
        "artifact_root_exists": True,
        "artifact_root_is_dir": True,
        "run_directory_count": run_count,
        "runs": runs,
        "latest_run_id": (runs[0]["run_id"] if run_count else None),
        "message": (
            None
            if run_count
            else (
                f"No run directories found in artifact root: {root}. "
                f"Directory exists={root_exists}, run_directories_found={run_count}. "
                "Run `python scripts/run_no_ground_truth_evaluation.py` to generate diagnostics artifacts."
            )
        ),
    }


def load_no_ground_truth_run_bundle(
    *,
    run_id: str | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    listing = list_no_ground_truth_runs(artifact_root=artifact_root)
    if not listing.get("available"):
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": None,
            "run_path": None,
            "manifest": None,
            "summary_markdown": None,
            "sections": {
                key: {"available": False, "payload": None, "message": "Artifact run not available."}
                for key in SECTION_FILES
            },
            "message": listing.get("message"),
        }

    available_runs = {str(row["run_id"]): row for row in listing.get("runs") or []}
    selected_run_id = str(run_id) if run_id and str(run_id) in available_runs else str(listing.get("latest_run_id"))
    run_path = Path(available_runs[selected_run_id]["path"])

    manifest_path = run_path / "evaluation_manifest.json"
    manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None

    summary_path = run_path / "summary.md"
    summary_markdown = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None

    sections: dict[str, dict[str, Any]] = {}
    for section_key, filename in SECTION_FILES.items():
        section_path = run_path / filename
        if not section_path.exists():
            sections[section_key] = {
                "available": False,
                "payload": None,
                "path": str(section_path),
                "message": f"{filename} not found for this run.",
            }
            continue
        payload = _safe_load_json(section_path)
        if payload is None:
            sections[section_key] = {
                "available": False,
                "payload": None,
                "path": str(section_path),
                "message": f"{filename} could not be parsed.",
            }
            continue
        sections[section_key] = {
            "available": True,
            "payload": payload,
            "path": str(section_path),
            "message": None,
        }

    return {
        "available": True,
        "artifact_root": listing.get("artifact_root"),
        "run_id": selected_run_id,
        "run_path": str(run_path),
        "manifest": manifest,
        "summary_markdown": summary_markdown,
        "sections": sections,
        "available_runs": listing.get("runs") or [],
    }


def compare_no_ground_truth_runs(
    *,
    current_run_id: str | None = None,
    baseline_run_id: str | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    from backend.evaluation.no_ground_truth import build_no_ground_truth_run_comparison

    listing = list_no_ground_truth_runs(artifact_root=artifact_root)
    if not bool(listing.get("available")):
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": None,
            "baseline_run_id": None,
            "reason": "no_runs_available",
            "message": listing.get("message"),
        }

    runs = listing.get("runs") if isinstance(listing.get("runs"), list) else []
    ordered_ids = [str(row.get("run_id")) for row in runs if isinstance(row, dict) and row.get("run_id")]
    if not ordered_ids:
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": None,
            "baseline_run_id": None,
            "reason": "no_runs_available",
            "message": "No run directories were found in the diagnostics artifact root.",
        }

    selected_current = str(current_run_id or listing.get("latest_run_id") or ordered_ids[0])
    if selected_current not in ordered_ids:
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": selected_current,
            "baseline_run_id": None,
            "reason": "current_run_not_found",
            "message": f"Current run '{selected_current}' was not found.",
            "available_run_ids": ordered_ids,
        }

    selected_baseline = str(baseline_run_id) if baseline_run_id else ""
    if not selected_baseline:
        current_idx = ordered_ids.index(selected_current)
        selected_baseline = ordered_ids[current_idx + 1] if current_idx + 1 < len(ordered_ids) else ""
    elif selected_baseline not in ordered_ids:
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": selected_current,
            "baseline_run_id": selected_baseline,
            "reason": "baseline_run_not_found",
            "message": f"Baseline run '{selected_baseline}' was not found.",
            "available_run_ids": ordered_ids,
        }

    if not selected_baseline:
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": selected_current,
            "baseline_run_id": None,
            "reason": "no_previous_run_available",
            "message": (
                "No previous run was found for before/after comparison. "
                "Run the evaluation again to compare latest vs previous."
            ),
            "available_run_ids": ordered_ids,
        }

    current_bundle = load_no_ground_truth_run_bundle(
        run_id=selected_current,
        artifact_root=artifact_root,
    )
    baseline_bundle = load_no_ground_truth_run_bundle(
        run_id=selected_baseline,
        artifact_root=artifact_root,
    )
    if not bool(current_bundle.get("available")) or not bool(baseline_bundle.get("available")):
        return {
            "available": False,
            "artifact_root": listing.get("artifact_root"),
            "run_id": selected_current,
            "baseline_run_id": selected_baseline,
            "reason": "bundle_load_failed",
            "message": "Unable to load one or both runs for comparison.",
        }

    current_manifest = current_bundle.get("manifest") if isinstance(current_bundle.get("manifest"), dict) else {}
    baseline_manifest = baseline_bundle.get("manifest") if isinstance(baseline_bundle.get("manifest"), dict) else {}
    comparison = build_no_ground_truth_run_comparison(
        current_run_id=selected_current,
        current_manifest=current_manifest,
        current_sections=_bundle_section_payloads(current_bundle),
        baseline_run_id=selected_baseline,
        baseline_manifest=baseline_manifest,
        baseline_sections=_bundle_section_payloads(baseline_bundle),
    )
    comparison["artifact_root"] = listing.get("artifact_root")
    comparison["comparison_mode"] = (
        "explicit_runs" if baseline_run_id else "latest_vs_previous"
    )
    comparison["available_run_ids"] = ordered_ids
    return comparison


def build_no_ground_truth_health_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    if not bool(bundle.get("available")):
        return {
            "available": False,
            "run_id": None,
            "version": "ngt_eval_v1",
            "generated_at": None,
            "caveat": (
                "These diagnostics measure model coherence, stability, evidence quality, and external alignment. "
                "They do not establish real-world predictive accuracy."
            ),
            "message": bundle.get("message")
            or (
                "No offline evaluation artifacts found yet. "
                "Run `python scripts/run_no_ground_truth_evaluation.py` first."
            ),
        }

    sections = bundle.get("sections") if isinstance(bundle.get("sections"), dict) else {}
    manifest = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}

    def _section_payload(name: str) -> dict[str, Any]:
        row = sections.get(name) if isinstance(sections.get(name), dict) else {}
        payload = row.get("payload")
        return payload if isinstance(payload, dict) else {}

    mono = _section_payload("monotonicity")
    counter = _section_payload("counterfactual")
    stability = _section_payload("stability")
    distribution = _section_payload("distribution")
    alignment = _section_payload("benchmark_alignment")
    confidence = _section_payload("confidence_diagnostics")
    comparison = _section_payload("comparison_to_previous")
    summary_markdown = str(bundle.get("summary_markdown") or "")

    monotonicity_rows = mono.get("rows") if isinstance(mono.get("rows"), list) else []
    top_violations = [
        {
            "rule_id": str(row.get("rule_id") or ""),
            "detail": str(row.get("detail") or ""),
        }
        for row in monotonicity_rows
        if isinstance(row, dict) and not bool(row.get("passed"))
    ][:8]

    intervention_table = (
        counter.get("top_interventions_by_median_impact")
        if isinstance(counter.get("top_interventions_by_median_impact"), list)
        else []
    )
    top_interventions = [row for row in intervention_table if isinstance(row, dict)][:8]
    backwards_flags = (
        counter.get("flagged_interventions")
        if isinstance(counter.get("flagged_interventions"), list)
        else []
    )

    stability_tests = stability.get("tests") if isinstance(stability.get("tests"), list) else []
    unstable_tests = [
        row
        for row in stability_tests
        if isinstance(row, dict) and str(row.get("test_id") or "").strip()
        and (
            (float(row.get("max_abs_score_swing") or 0.0) >= 12.0)
            or (float(row.get("confidence_tier_change_rate") or 0.0) >= 0.35)
        )
    ]
    top_unstable = sorted(
        unstable_tests,
        key=lambda row: float(row.get("max_abs_score_swing") or 0.0),
        reverse=True,
    )[:8]
    stability_swing_values = [
        float(row.get("mean_abs_score_swing") or 0.0)
        for row in stability_tests
        if isinstance(row, dict) and row.get("mean_abs_score_swing") is not None
    ]
    if not stability_swing_values:
        stability_swing_values = [
            float(row.get("max_abs_score_swing") or 0.0)
            for row in stability_tests
            if isinstance(row, dict) and row.get("max_abs_score_swing") is not None
        ]
    unstable_factor_map: dict[str, list[float]] = {}
    for test in stability_tests:
        if not isinstance(test, dict):
            continue
        rows = test.get("rows") if isinstance(test.get("rows"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("variant_type") or "unknown").strip() or "unknown"
            swing = abs(float(row.get("wildfire_risk_score_delta") or 0.0))
            unstable_factor_map.setdefault(key, []).append(swing)
    top_unstable_factors = sorted(
        [
            {
                "factor": key,
                "mean_abs_swing": _mean(values),
                "max_abs_swing": max(values) if values else 0.0,
                "sample_count": len(values),
            }
            for key, values in unstable_factor_map.items()
            if values
        ],
        key=lambda row: float(row.get("mean_abs_swing") or 0.0),
        reverse=True,
    )[:8]

    alignment_rows = alignment.get("rows") if isinstance(alignment.get("rows"), list) else []
    disagreement_count = 0
    signals_used: list[str] = []
    for row in alignment_rows:
        if not isinstance(row, dict):
            continue
        signal = str(row.get("signal_key") or "").strip()
        if signal:
            signals_used.append(signal)
        disagreements = row.get("disagreement_cases")
        if isinstance(disagreements, list):
            disagreement_count += len(disagreements)
    spearman_values = [
        float(row.get("spearman_rank_correlation"))
        for row in alignment_rows
        if isinstance(row, dict) and row.get("spearman_rank_correlation") is not None
    ]
    agreement_values = [
        float(row.get("bucket_agreement_ratio"))
        for row in alignment_rows
        if isinstance(row, dict) and row.get("bucket_agreement_ratio") is not None
    ]

    warnings: list[str] = []
    for name in ("monotonicity", "counterfactual", "stability", "distribution", "benchmark_alignment", "confidence_diagnostics"):
        row = sections.get(name) if isinstance(sections.get(name), dict) else {}
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        section_warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
        for warning in section_warnings[:5]:
            warnings.append(str(warning))

    recommended_from_summary = _extract_recommendations_from_markdown(summary_markdown)

    return {
        "available": True,
        "run_id": bundle.get("run_id"),
        "version": str(manifest.get("schema_version") or "ngt_eval_v1"),
        "generated_at": manifest.get("generated_at") or (
            bundle.get("available_runs")[0]["generated_at"]
            if isinstance(bundle.get("available_runs"), list) and bundle.get("available_runs")
            else None
        ),
        "caveat": (
            "These diagnostics measure coherence, stability, evidence quality, and external alignment. "
            "They do not establish real-world predictive accuracy or insurer approval."
        ),
        "monotonicity": {
            "status": mono.get("status"),
            "checks_run": mono.get("rule_count"),
            "passed_count": mono.get("passed_count"),
            "failed_count": mono.get("failed_count"),
            "top_violations": top_violations,
        },
        "mitigation_sensitivity": {
            "status": counter.get("status"),
            "top_interventions": top_interventions,
            "backwards_or_zero_impact_flags": list(backwards_flags)[:12],
        },
        "stability": {
            "status": stability.get("status"),
            "test_count": stability.get("test_count"),
            "average_score_swing": _mean(stability_swing_values),
            "median_score_swing": _median(stability_swing_values),
            "unstable_scenario_count": len(unstable_tests),
            "top_unstable_tests": top_unstable,
            "top_unstable_factors": top_unstable_factors,
            "warnings": (stability.get("warnings") if isinstance(stability.get("warnings"), list) else [])[:10],
        },
        "distribution": {
            "status": distribution.get("status"),
            "score_spread": ((distribution.get("overall") or {}).get("wildfire_risk_score") if isinstance(distribution.get("overall"), dict) else {}),
            "confidence_tier_counts": distribution.get("confidence_tier_counts") or {},
            "fallback_group_counts": distribution.get("fallback_group_counts") or {},
            "warnings": (distribution.get("warnings") if isinstance(distribution.get("warnings"), list) else [])[:10],
        },
        "benchmark_alignment": {
            "status": alignment.get("status"),
            "available": bool((sections.get("benchmark_alignment") or {}).get("available")),
            "signals_used": sorted(set(signals_used)),
            "rule_count": alignment.get("rule_count"),
            "average_spearman_rank_correlation": _mean(spearman_values),
            "average_bucket_agreement_ratio": _mean(agreement_values),
            "disagreement_review_count": disagreement_count,
            "warnings": (alignment.get("warnings") if isinstance(alignment.get("warnings"), list) else [])[:10],
            "caveat": "Benchmark alignment is a sanity check only and not ground-truth validation.",
        },
        "confidence_diagnostics": {
            "status": confidence.get("status"),
            "record_count": confidence.get("record_count"),
            "confidence_tier_distribution": confidence.get("confidence_tier_distribution") or {},
            "fallback_group_distribution": confidence.get("fallback_group_distribution") or {},
            "warnings": (confidence.get("warnings") if isinstance(confidence.get("warnings"), list) else [])[:10],
        },
        "comparison_to_previous": {
            "available": bool(comparison.get("available")),
            "baseline_run_id": comparison.get("baseline_run_id"),
            "monotonicity_failed_count_delta": ((comparison.get("monotonicity") or {}).get("failed_count_delta")),
            "confidence_warning_count_delta": ((comparison.get("confidence_diagnostics") or {}).get("warning_count_delta")),
            "largest_bucket_fraction_delta": ((comparison.get("distribution") or {}).get("largest_bucket_fraction_delta")),
            "overall_direction_signals": comparison.get("overall_direction_signals") or [],
            "likely_change_drivers": comparison.get("likely_change_drivers") or [],
        },
        "recommended_next_actions": sorted(set(recommended_from_summary + warnings))[:16],
    }
