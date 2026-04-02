#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.benchmarking import load_artifact, run_benchmark_suite


DEFAULT_PACK = Path("benchmark") / "scenario_pack_property_data_differentiation_v1.json"
DEFAULT_OUTPUT_DIR = Path("benchmark") / "results"

STAGE_ORDER = [
    "geocode_only",
    "parcel_matched",
    "footprint_matched",
    "footprint_naip",
    "footprint_naip_enriched",
]

STAGE_LABELS = {
    "geocode_only": "geocode-only",
    "parcel_matched": "parcel matched",
    "footprint_matched": "footprint matched",
    "footprint_naip": "footprint + NAIP",
    "footprint_naip_enriched": "footprint + NAIP + public-record/user structure attributes",
}

VARIANT_ORDER = ["dense", "clear"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _sorted_unique_strings(values: list[Any]) -> list[str]:
    cleaned = []
    for value in values:
        text = str(value or "").strip()
        if text:
            cleaned.append(text)
    return sorted(set(cleaned))


def _ordered_unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _extract_stage_key(snapshot: dict[str, Any], scenario_id: str) -> str | None:
    tags = snapshot.get("profile_tags")
    if isinstance(tags, list):
        for tag in tags:
            text = str(tag or "").strip().lower()
            if text.startswith("stage:"):
                return text.split(":", 1)[1]
    sid = str(scenario_id or "").strip().lower()
    if sid.startswith("stage_"):
        body = sid[len("stage_") :]
        for suffix in ("_dense", "_clear"):
            if body.endswith(suffix):
                return body[: -len(suffix)]
    return None


def _extract_variant(snapshot: dict[str, Any], scenario_id: str) -> str:
    tags = snapshot.get("profile_tags")
    if isinstance(tags, list):
        for tag in tags:
            text = str(tag or "").strip().lower()
            if text in {"dense", "clear"}:
                return text
    sid = str(scenario_id or "").strip().lower()
    if sid.endswith("_dense"):
        return "dense"
    if sid.endswith("_clear"):
        return "clear"
    return "unknown"


def _specificity_tier(snapshot: dict[str, Any]) -> str:
    specificity = snapshot.get("specificity")
    if isinstance(specificity, dict):
        value = str(specificity.get("specificity_tier") or "").strip()
        if value:
            return value
    return "regional_estimate"


def _specificity_rank(tier: str) -> int:
    normalized = str(tier or "").strip().lower()
    mapping = {
        "insufficient_data": 0,
        "regional_estimate": 1,
        "address_level": 2,
        "property_specific": 3,
    }
    return int(mapping.get(normalized, 1))


def _comparison_allowed(snapshot: dict[str, Any]) -> bool:
    specificity = snapshot.get("specificity")
    if isinstance(specificity, dict) and "comparison_allowed" in specificity:
        return bool(specificity.get("comparison_allowed"))
    return False


def _property_confidence_score(snapshot: dict[str, Any]) -> float | None:
    summary = snapshot.get("property_confidence_summary")
    if isinstance(summary, dict):
        score = _safe_float(summary.get("score"))
        if score is not None:
            return score
    return None


def _property_confidence_level(snapshot: dict[str, Any]) -> str:
    summary = snapshot.get("property_confidence_summary")
    if isinstance(summary, dict):
        value = str(summary.get("level") or "").strip()
        if value:
            return value
    return ""


def _local_differentiation_score(snapshot: dict[str, Any]) -> float | None:
    differentiation = snapshot.get("differentiation")
    if isinstance(differentiation, dict):
        return _safe_float(differentiation.get("local_differentiation_score"))
    return None


def _confidence_language(snapshot: dict[str, Any]) -> str:
    confidence = snapshot.get("confidence")
    if isinstance(confidence, dict):
        return str(confidence.get("confidence_language") or "").strip()
    return ""


def _confidence_language_level(confidence_language: str) -> int:
    text = str(confidence_language or "").strip().lower()
    if "high confidence" in text:
        return 3
    if "moderate confidence" in text:
        return 2
    if (
        "limited confidence" in text
        or "low confidence" in text
        or "confidence unavailable" in text
        or "preliminary" in text
    ):
        return 1
    return 0


def _list_strings(snapshot: dict[str, Any], key: str) -> list[str]:
    raw = snapshot.get(key)
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _jaccard(a: list[str], b: list[str]) -> float | None:
    set_a = {v for v in a if v}
    set_b = {v for v in b if v}
    union = set_a | set_b
    if not union:
        return None
    return float(len(set_a & set_b)) / float(len(union))


def _is_generic_specificity_driver(driver: str) -> bool:
    text = str(driver or "").strip().lower()
    if not text:
        return False
    generic_tokens = (
        "not precise enough to compare adjacent homes",
        "nearby homes may appear similar",
        "relies mostly on regional context",
    )
    return any(token in text for token in generic_tokens)


def _drivers_for_stability(drivers: list[str]) -> list[str]:
    filtered = [row for row in drivers if not _is_generic_specificity_driver(row)]
    return filtered if filtered else list(drivers)


def _honest_abstention_quality(snapshot: dict[str, Any]) -> float:
    differentiation = snapshot.get("differentiation")
    if not isinstance(differentiation, dict):
        differentiation = {}
    safeguard_triggered = bool(differentiation.get("nearby_home_comparison_safeguard_triggered"))
    safeguard_message = str(differentiation.get("nearby_home_comparison_safeguard_message") or "").strip()
    local_diff = _safe_float(differentiation.get("local_differentiation_score")) or 0.0
    tier = _specificity_tier(snapshot).strip().lower()
    disallow_compare = not _comparison_allowed(snapshot)

    score = 0.0
    if disallow_compare:
        score += 40.0
    if safeguard_triggered:
        score += 35.0
    if safeguard_message:
        score += 15.0
    if tier in {"regional_estimate", "insufficient_data"} and local_diff <= 40.0:
        score += 10.0
    return round(min(score, 100.0), 1)


def _snapshot_summary(snapshot: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "property_confidence_summary": (
            dict(snapshot.get("property_confidence_summary"))
            if isinstance(snapshot.get("property_confidence_summary"), dict)
            else {}
        ),
        "assessment_specificity_tier": _specificity_tier(snapshot),
        "comparison_allowed": _comparison_allowed(snapshot),
        "local_differentiation_score": _local_differentiation_score(snapshot),
        "confidence_language": _confidence_language(snapshot),
        "top_risk_drivers": _list_strings(snapshot, "top_risk_drivers")[:3],
        "top_recommended_actions": _list_strings(snapshot, "top_recommended_actions")[:3],
    }


def _transition_summary(before: dict[str, Any], after: dict[str, Any], *, variant: str) -> dict[str, Any]:
    before_local = _local_differentiation_score(before)
    after_local = _local_differentiation_score(after)
    local_delta = None
    if before_local is not None and after_local is not None:
        local_delta = round(after_local - before_local, 1)

    before_prop_score = _property_confidence_score(before)
    after_prop_score = _property_confidence_score(after)
    prop_delta = None
    if before_prop_score is not None and after_prop_score is not None:
        prop_delta = round(after_prop_score - before_prop_score, 1)

    before_tier = _specificity_tier(before)
    after_tier = _specificity_tier(after)
    tier_delta = _specificity_rank(after_tier) - _specificity_rank(before_tier)

    before_drivers = _list_strings(before, "top_risk_drivers")[:3]
    after_drivers = _list_strings(after, "top_risk_drivers")[:3]
    before_actions = _list_strings(before, "top_recommended_actions")[:3]
    after_actions = _list_strings(after, "top_recommended_actions")[:3]
    before_drivers_for_stability = _drivers_for_stability(before_drivers)
    after_drivers_for_stability = _drivers_for_stability(after_drivers)
    before_driver_set = set(before_drivers_for_stability)
    after_driver_set = set(after_drivers_for_stability)
    before_action_set = set(before_actions)
    after_action_set = set(after_actions)

    driver_jaccard = _jaccard(before_drivers_for_stability, after_drivers_for_stability)
    driver_overlap_count = len(before_driver_set & after_driver_set)
    recommendation_added = sorted(after_action_set - before_action_set)
    recommendation_removed = sorted(before_action_set - after_action_set)
    recommendation_unchanged = sorted(before_action_set & after_action_set)

    before_conf_language = _confidence_language(before)
    after_conf_language = _confidence_language(after)
    before_conf_level = _confidence_language_level(before_conf_language)
    after_conf_level = _confidence_language_level(after_conf_language)

    before_abstention_quality = _honest_abstention_quality(before)
    after_abstention_quality = _honest_abstention_quality(after)
    abstention_delta = round(after_abstention_quality - before_abstention_quality, 1)

    trust_wording_regression = (
        (prop_delta is not None and prop_delta >= 5.0)
        and (tier_delta >= 0)
        and (after_conf_level < before_conf_level)
    )

    return {
        "variant": variant,
        "local_differentiation_delta": local_delta,
        "property_confidence_delta": prop_delta,
        "specificity_transition": {
            "before": before_tier,
            "after": after_tier,
            "rank_delta": tier_delta,
        },
        "comparison_allowed_transition": {
            "before": _comparison_allowed(before),
            "after": _comparison_allowed(after),
        },
        "top_risk_driver_stability": {
            "primary_driver_stable": bool(
                before_drivers_for_stability
                and after_drivers_for_stability
                and before_drivers_for_stability[0] == after_drivers_for_stability[0]
            ),
            "overlap_count": driver_overlap_count,
            "jaccard_similarity": (
                round(driver_jaccard, 3) if driver_jaccard is not None else None
            ),
            "before": before_drivers,
            "after": after_drivers,
            "before_for_stability": before_drivers_for_stability,
            "after_for_stability": after_drivers_for_stability,
        },
        "recommendation_changes": {
            "added": recommendation_added,
            "removed": recommendation_removed,
            "unchanged": recommendation_unchanged,
            "before": before_actions,
            "after": after_actions,
        },
        "confidence_language_transition": {
            "before": before_conf_language,
            "after": after_conf_language,
            "before_level": before_conf_level,
            "after_level": after_conf_level,
        },
        "honest_abstention_quality": {
            "before": before_abstention_quality,
            "after": after_abstention_quality,
            "delta": abstention_delta,
        },
        "improvement_signals": {
            "differentiation_improved": bool(local_delta is not None and local_delta > 0.0),
            "specificity_improved": bool(tier_delta > 0),
            "property_confidence_improved": bool(prop_delta is not None and prop_delta > 0.0),
            "honest_abstention_quality_improved": bool(abstention_delta > 0.0),
        },
        "trust_wording_regression": trust_wording_regression,
    }


def _aggregate_stage_metrics(variant_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    property_conf_scores = [
        score for score in (_property_confidence_score(row) for row in variant_rows.values()) if score is not None
    ]
    local_scores = [
        score for score in (_local_differentiation_score(row) for row in variant_rows.values()) if score is not None
    ]
    specificity_counts: dict[str, int] = {}
    compare_allowed_count = 0
    for row in variant_rows.values():
        tier = _specificity_tier(row)
        specificity_counts[tier] = int(specificity_counts.get(tier, 0)) + 1
        if _comparison_allowed(row):
            compare_allowed_count += 1
    scenario_count = len(variant_rows)
    return {
        "scenario_count": scenario_count,
        "average_property_confidence_score": (
            round(statistics.fmean(property_conf_scores), 1) if property_conf_scores else None
        ),
        "average_local_differentiation_score": (
            round(statistics.fmean(local_scores), 1) if local_scores else None
        ),
        "specificity_counts": dict(sorted(specificity_counts.items())),
        "comparison_allowed_rate": (
            round(compare_allowed_count / scenario_count, 3) if scenario_count > 0 else None
        ),
    }


def build_property_data_stage_report(artifact: dict[str, Any]) -> dict[str, Any]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    scenario_rows = artifact.get("scenario_results")
    if not isinstance(scenario_rows, list):
        scenario_rows = []

    ignored_scenarios: list[str] = []
    for row in scenario_rows:
        if not isinstance(row, dict):
            continue
        scenario_id = str(row.get("scenario_id") or "").strip()
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        stage_key = _extract_stage_key(snapshot, scenario_id)
        if not stage_key:
            ignored_scenarios.append(scenario_id)
            continue
        variant = _extract_variant(snapshot, scenario_id)
        matrix.setdefault(stage_key, {})[variant] = snapshot

    stage_summaries: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    for idx, stage_key in enumerate(STAGE_ORDER):
        variants = matrix.get(stage_key, {})
        per_variant = {}
        for variant in VARIANT_ORDER:
            snapshot = variants.get(variant)
            if snapshot is None:
                continue
            scenario_id = str(snapshot.get("scenario_id") or "")
            if not scenario_id:
                for row in scenario_rows:
                    if (
                        isinstance(row, dict)
                        and row.get("snapshot") is snapshot
                        and isinstance(row.get("scenario_id"), str)
                    ):
                        scenario_id = str(row.get("scenario_id") or "")
                        break
            per_variant[variant] = _snapshot_summary(snapshot, scenario_id=scenario_id)
        stage_summaries.append(
            {
                "stage_key": stage_key,
                "stage_label": STAGE_LABELS.get(stage_key, stage_key),
                "variant_summaries": per_variant,
                "aggregate": _aggregate_stage_metrics(variants),
            }
        )

        if idx == 0:
            continue
        prev_key = STAGE_ORDER[idx - 1]
        prev_variants = matrix.get(prev_key, {})
        for variant in VARIANT_ORDER:
            before = prev_variants.get(variant)
            after = variants.get(variant)
            if isinstance(before, dict) and isinstance(after, dict):
                transition = _transition_summary(before, after, variant=variant)
                transitions.append(
                    {
                        "from_stage": prev_key,
                        "to_stage": stage_key,
                        "from_stage_label": STAGE_LABELS.get(prev_key, prev_key),
                        "to_stage_label": STAGE_LABELS.get(stage_key, stage_key),
                        **transition,
                    }
                )

    chain_evaluations: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        snapshots_in_chain = [
            matrix.get(stage_key, {}).get(variant)
            for stage_key in STAGE_ORDER
            if isinstance(matrix.get(stage_key, {}).get(variant), dict)
        ]
        if len(snapshots_in_chain) < 2:
            chain_evaluations[variant] = {
                "evaluated": False,
                "reason": "insufficient_stage_snapshots",
            }
            continue
        first = snapshots_in_chain[0]
        last = snapshots_in_chain[-1]
        first_local = _local_differentiation_score(first)
        last_local = _local_differentiation_score(last)
        local_delta = None
        if first_local is not None and last_local is not None:
            local_delta = round(last_local - first_local, 1)
        first_prop = _property_confidence_score(first)
        last_prop = _property_confidence_score(last)
        prop_delta = None
        if first_prop is not None and last_prop is not None:
            prop_delta = round(last_prop - first_prop, 1)
        spec_delta = _specificity_rank(_specificity_tier(last)) - _specificity_rank(
            _specificity_tier(first)
        )
        abstention_delta = round(
            _honest_abstention_quality(last) - _honest_abstention_quality(first),
            1,
        )
        variant_transitions = [
            row for row in transitions if str(row.get("variant")) == variant
        ]
        trust_wording_regression = any(
            bool(row.get("trust_wording_regression")) for row in variant_transitions
        )
        improved_or_honest = (
            bool(local_delta is not None and local_delta > 0.0)
            or bool(abstention_delta > 0.0)
            or bool(spec_delta > 0)
            or bool(prop_delta is not None and prop_delta > 0.0)
        )
        chain_evaluations[variant] = {
            "evaluated": True,
            "local_differentiation_delta": local_delta,
            "property_confidence_delta": prop_delta,
            "specificity_rank_delta": spec_delta,
            "honest_abstention_quality_delta": abstention_delta,
            "trust_wording_regression": trust_wording_regression,
            "passes_expectation": improved_or_honest and not trust_wording_regression,
        }

    trust_wording_regressions = [
        row
        for row in transitions
        if bool(row.get("trust_wording_regression"))
    ]
    improved_transition_count = sum(
        1
        for row in transitions
        if any(
            bool((row.get("improvement_signals") or {}).get(signal))
            for signal in (
                "differentiation_improved",
                "specificity_improved",
                "property_confidence_improved",
                "honest_abstention_quality_improved",
            )
        )
    )
    evaluated_chains = [row for row in chain_evaluations.values() if bool(row.get("evaluated"))]
    chain_pass_count = sum(
        1 for row in evaluated_chains if bool(row.get("passes_expectation"))
    )
    notes: list[str] = []
    if trust_wording_regressions:
        notes.append(
            f"Detected {len(trust_wording_regressions)} stage transition(s) where confidence wording became more cautious despite better property-data confidence."
        )
    if evaluated_chains and chain_pass_count < len(evaluated_chains):
        notes.append(
            "One or more stage chains did not show differentiation improvement or stronger honest-abstention quality."
        )

    stage_quality = {
        "transition_count": len(transitions),
        "improved_or_honest_transition_count": improved_transition_count,
        "trust_wording_regression_count": len(trust_wording_regressions),
        "variant_chain_pass_count": chain_pass_count,
        "variant_chain_count": len(evaluated_chains),
        "passed": (len(evaluated_chains) > 0)
        and (chain_pass_count == len(evaluated_chains))
        and (len(trust_wording_regressions) == 0),
        "notes": notes,
    }

    return {
        "generated_at": _now_iso(),
        "benchmark_artifact_path": str(artifact.get("artifact_path") or ""),
        "benchmark_pack_version": str(artifact.get("benchmark_pack_version") or ""),
        "stage_order": [
            {"stage_key": key, "stage_label": STAGE_LABELS.get(key, key)}
            for key in STAGE_ORDER
        ],
        "stage_summaries": stage_summaries,
        "stage_transitions": transitions,
        "variant_chain_evaluation": chain_evaluations,
        "stage_quality": stage_quality,
        "ignored_scenarios": _sorted_unique_strings(ignored_scenarios),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Property-Data Stage Differentiation Benchmark",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Source benchmark artifact: `{report.get('benchmark_artifact_path')}`",
        f"- Stage quality passed: `{(report.get('stage_quality') or {}).get('passed')}`",
        "",
        "## Stage Summaries",
    ]

    stage_summaries = report.get("stage_summaries")
    if isinstance(stage_summaries, list):
        for stage in stage_summaries:
            if not isinstance(stage, dict):
                continue
            lines.append(
                f"- **{stage.get('stage_label')}** (`{stage.get('stage_key')}`)"
            )
            aggregate = stage.get("aggregate") if isinstance(stage.get("aggregate"), dict) else {}
            lines.append(
                "  "
                + f"avg_property_confidence={aggregate.get('average_property_confidence_score')} | "
                + f"avg_local_differentiation={aggregate.get('average_local_differentiation_score')} | "
                + f"specificity_counts={aggregate.get('specificity_counts')}"
            )
            variant_summaries = (
                stage.get("variant_summaries")
                if isinstance(stage.get("variant_summaries"), dict)
                else {}
            )
            for variant in VARIANT_ORDER:
                row = variant_summaries.get(variant)
                if not isinstance(row, dict):
                    continue
                prop_conf = row.get("property_confidence_summary")
                if not isinstance(prop_conf, dict):
                    prop_conf = {}
                lines.append(
                    "  "
                    + f"- {variant}: specificity={row.get('assessment_specificity_tier')} "
                    + f"compare_allowed={row.get('comparison_allowed')} "
                    + f"local_diff={row.get('local_differentiation_score')} "
                    + f"property_conf={prop_conf.get('score')} ({prop_conf.get('level')})"
                )
                lines.append("    " + f"top_drivers={row.get('top_risk_drivers')}")
                lines.append("    " + f"top_actions={row.get('top_recommended_actions')}")

    transitions = report.get("stage_transitions")
    if isinstance(transitions, list) and transitions:
        lines.append("")
        lines.append("## Stage Transitions")
        for row in transitions:
            if not isinstance(row, dict):
                continue
            stability = (
                row.get("top_risk_driver_stability")
                if isinstance(row.get("top_risk_driver_stability"), dict)
                else {}
            )
            rec_changes = (
                row.get("recommendation_changes")
                if isinstance(row.get("recommendation_changes"), dict)
                else {}
            )
            lines.append(
                "- "
                + f"{row.get('from_stage_label')} -> {row.get('to_stage_label')} ({row.get('variant')}): "
                + f"local_diff_delta={row.get('local_differentiation_delta')}, "
                + f"specificity_delta={((row.get('specificity_transition') or {}).get('rank_delta'))}, "
                + f"property_conf_delta={row.get('property_confidence_delta')}, "
                + f"driver_jaccard={stability.get('jaccard_similarity')}, "
                + f"primary_driver_stable={stability.get('primary_driver_stable')}, "
                + f"actions_added={rec_changes.get('added')}, actions_removed={rec_changes.get('removed')}, "
                + f"trust_wording_regression={row.get('trust_wording_regression')}"
            )

    stage_quality = report.get("stage_quality")
    if isinstance(stage_quality, dict):
        lines.append("")
        lines.append("## Stage Quality Gate")
        lines.append(f"- Transition count: `{stage_quality.get('transition_count')}`")
        lines.append(
            f"- Improved-or-honest transitions: `{stage_quality.get('improved_or_honest_transition_count')}`"
        )
        lines.append(
            f"- Trust wording regressions: `{stage_quality.get('trust_wording_regression_count')}`"
        )
        lines.append(
            f"- Variant chains passing expectation: `{stage_quality.get('variant_chain_pass_count')} / {stage_quality.get('variant_chain_count')}`"
        )
        notes = stage_quality.get("notes")
        if isinstance(notes, list) and notes:
            lines.append("- Notes:")
            for note in notes:
                lines.append(f"  - {note}")

    return "\n".join(lines).rstrip() + "\n"


def _write_sidecar_artifacts(report: dict[str, Any], *, fallback_dir: Path) -> dict[str, str]:
    source_artifact = str(report.get("benchmark_artifact_path") or "").strip()
    source_path = Path(source_artifact).expanduser() if source_artifact else None
    if source_path and source_path.suffix.lower() == ".json":
        base = source_path.with_suffix("")
        out_json = base.with_name(f"{base.name}_property_data_stages.json")
        out_md = base.with_name(f"{base.name}_property_data_stages.md")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        out_json = fallback_dir / f"property_data_stage_benchmark_{stamp}.json"
        out_md = fallback_dir / f"property_data_stage_benchmark_{stamp}.md"

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    return {"json": str(out_json), "markdown": str(out_md)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the property-data stage benchmark (geocode-only -> parcel -> footprint -> "
            "footprint+NAIP -> footprint+NAIP+enrichment) and emit before/after differentiation artifacts."
        )
    )
    parser.add_argument("--pack", default=str(DEFAULT_PACK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--benchmark-artifact",
        default=None,
        help="Optional existing benchmark artifact. If provided, skip benchmark execution and summarize this artifact.",
    )
    parser.add_argument(
        "--fail-on-stage-regression",
        action="store_true",
        help="Exit nonzero when stage quality checks fail.",
    )
    parser.add_argument(
        "--fail-on-benchmark-failures",
        action="store_true",
        help="Exit nonzero when scenario-level expectations in the underlying benchmark artifact fail.",
    )
    args = parser.parse_args(argv)

    if args.benchmark_artifact:
        artifact = load_artifact(Path(args.benchmark_artifact).expanduser())
        artifact["artifact_path"] = str(Path(args.benchmark_artifact).expanduser())
    else:
        artifact = run_benchmark_suite(
            pack_path=Path(args.pack).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
        )

    report = build_property_data_stage_report(artifact)
    sidecars = _write_sidecar_artifacts(
        report,
        fallback_dir=Path(args.output_dir).expanduser(),
    )

    print(
        json.dumps(
            {
                "benchmark_artifact_path": artifact.get("artifact_path"),
                "benchmark_summary_passed": bool((artifact.get("summary") or {}).get("passed", True)),
                "summary_artifacts": sidecars,
                "stage_quality": report.get("stage_quality"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    benchmark_passed = bool((artifact.get("summary") or {}).get("passed", True))
    stage_passed = bool((report.get("stage_quality") or {}).get("passed"))
    if args.fail_on_benchmark_failures and not benchmark_passed:
        return 1
    if args.fail_on_stage_regression and not stage_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
