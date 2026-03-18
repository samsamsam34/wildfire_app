from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACT_ROOT = Path("benchmark") / "no_ground_truth_evaluation"
SECTION_FILES = {
    "monotonicity": "monotonicity_results.json",
    "counterfactual": "counterfactual_results.json",
    "stability": "stability_results.json",
    "distribution": "distribution_results.json",
    "benchmark_alignment": "benchmark_alignment_results.json",
    "confidence_diagnostics": "confidence_diagnostics.json",
}


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def resolve_artifact_root(path_hint: str | Path | None = None) -> Path:
    hint = str(path_hint or os.getenv("WF_NO_GROUND_TRUTH_EVAL_DIR") or "").strip()
    if hint:
        return Path(hint).expanduser()
    return DEFAULT_ARTIFACT_ROOT


def list_no_ground_truth_runs(
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_artifact_root(artifact_root)
    if not root.exists() or not root.is_dir():
        return {
            "available": False,
            "artifact_root": str(root),
            "runs": [],
            "message": (
                "No offline evaluation artifacts found yet. "
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
    return {
        "available": bool(runs),
        "artifact_root": str(root),
        "runs": runs,
        "latest_run_id": (runs[0]["run_id"] if runs else None),
        "message": None if runs else "No run directories found in artifact root.",
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

    warnings: list[str] = []
    for name in ("monotonicity", "counterfactual", "stability", "distribution", "benchmark_alignment", "confidence_diagnostics"):
        row = sections.get(name) if isinstance(sections.get(name), dict) else {}
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        section_warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
        for warning in section_warnings[:5]:
            warnings.append(str(warning))

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
            "unstable_scenario_count": len(unstable_tests),
            "top_unstable_tests": top_unstable,
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
        "recommended_next_actions": sorted(set(warnings))[:12],
    }

