from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.models import (
    AssessmentResult,
    TrustDiagnostics,
    TrustDiagnosticsBenchmarkAlignment,
    TrustDiagnosticsConfidence,
    TrustDiagnosticsDistributionContext,
    TrustDiagnosticsDistributionSegment,
    TrustDiagnosticsInterventionImpact,
    TrustDiagnosticsMitigationSensitivity,
    TrustDiagnosticsMonotonicity,
    TrustDiagnosticsStability,
)

TRUST_DIAGNOSTIC_CAVEAT = (
    "These diagnostics measure model coherence, stability, evidence quality, and external alignment. "
    "They do not establish real-world predictive accuracy or insurer approval."
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_band(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 33.0:
        return "low"
    if score < 66.0:
        return "moderate"
    return "high"


def _evidence_tier_for_result(result: AssessmentResult) -> str:
    if float(result.fallback_weight_fraction or 0.0) >= 0.45:
        return "low"
    if float(result.observed_weight_fraction or 0.0) >= 0.7:
        return "high"
    return "moderate"


def _percentile(value: float, samples: list[float]) -> float | None:
    if not samples:
        return None
    below_or_equal = sum(1 for row in samples if row <= value)
    return round((below_or_equal / float(len(samples))) * 100.0, 1)


def _resolve_reference_root(path_hint: str | Path | None = None) -> Path | None:
    raw = str(path_hint or os.getenv("WF_TRUST_REFERENCE_ARTIFACT_DIR") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    return None


@lru_cache(maxsize=8)
def _load_reference_artifacts_cached(path_hint: str) -> dict[str, Any]:
    root = _resolve_reference_root(path_hint)
    if root is None:
        return {
            "available": False,
            "reason": "reference_artifact_dir_not_configured",
            "distribution_scores": [],
            "alignment_rows": [],
        }

    distribution_path = root / "distribution_results.json"
    alignment_path = root / "benchmark_alignment_results.json"
    manifest_path = root / "evaluation_manifest.json"
    distribution_scores: list[float] = []
    alignment_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if distribution_path.exists():
        try:
            payload = json.loads(distribution_path.read_text(encoding="utf-8"))
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                score = _safe_float(row.get("risk_score"))
                if score is not None:
                    distribution_scores.append(score)
        except Exception:
            warnings.append("Failed to parse distribution_results.json")
    else:
        warnings.append("distribution_results.json not found")

    if alignment_path.exists():
        try:
            payload = json.loads(alignment_path.read_text(encoding="utf-8"))
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            alignment_rows = [row for row in rows if isinstance(row, dict)]
        except Exception:
            warnings.append("Failed to parse benchmark_alignment_results.json")
    else:
        warnings.append("benchmark_alignment_results.json not found")

    manifest_meta: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            warnings.append("Failed to parse evaluation_manifest.json")

    return {
        "available": bool(distribution_scores or alignment_rows),
        "reason": None if bool(distribution_scores or alignment_rows) else "reference_artifacts_missing_or_invalid",
        "root": str(root),
        "distribution_scores": distribution_scores,
        "alignment_rows": alignment_rows,
        "manifest_meta": manifest_meta,
        "warnings": warnings,
    }


def clear_trust_reference_cache() -> None:
    _load_reference_artifacts_cached.cache_clear()


def load_trust_reference_artifacts(path_hint: str | Path | None = None) -> dict[str, Any]:
    cache_key = str(path_hint or os.getenv("WF_TRUST_REFERENCE_ARTIFACT_DIR") or "")
    return _load_reference_artifacts_cached(cache_key)


def build_trust_diagnostics(
    *,
    result: AssessmentResult,
    stability_samples: list[dict[str, Any]],
    mitigation_samples: list[dict[str, Any]],
    reference_artifacts: dict[str, Any],
) -> TrustDiagnostics:
    evidence_completeness = _safe_float(result.feature_coverage_percent)
    if evidence_completeness is None or evidence_completeness <= 0.0:
        evidence_completeness = round(float(result.observed_weight_fraction or 0.0) * 100.0, 1)

    fallback_heavy = bool(
        float(result.fallback_weight_fraction or 0.0) >= 0.45
        or int(result.fallback_feature_count or 0) > int(result.observed_feature_count or 0)
    )
    confidence_notes: list[str] = []
    if fallback_heavy:
        confidence_notes.append("Fallback assumptions influence a large share of the score.")
    if result.assessment_diagnostics.trust_tier_blockers:
        confidence_notes.append("Trust-tier blockers are present in this assessment.")
    if result.assessment_diagnostics.critical_inputs_missing:
        confidence_notes.append("Critical property inputs are missing or inferred.")

    confidence = TrustDiagnosticsConfidence(
        tier=result.confidence_tier,
        score=float(result.confidence_score or 0.0),
        evidence_completeness=float(evidence_completeness or 0.0),
        fallback_heavy=fallback_heavy,
        missing_critical_fields=list(result.assessment_diagnostics.critical_inputs_missing or []),
        notes=confidence_notes,
    )

    jitter_swings = [
        abs(float(row.get("risk_delta") or 0.0))
        for row in stability_samples
        if str(row.get("sample_type") or "") == "geocode_jitter"
    ]
    fallback_swings = [
        abs(float(row.get("risk_delta") or 0.0))
        for row in stability_samples
        if str(row.get("sample_type") or "") == "fallback_assumption"
    ]
    tier_flips = sum(1 for row in stability_samples if bool(row.get("tier_changed")))
    band_flips = sum(1 for row in stability_samples if bool(row.get("band_changed")))
    max_jitter = max(jitter_swings) if jitter_swings else 0.0
    max_fallback = max(fallback_swings) if fallback_swings else 0.0
    max_combined = max(max_jitter, max_fallback)
    local_sensitivity_score = max(0.0, min(100.0, 100.0 - (max_combined * 6.0)))
    if max_combined >= 12.0 or tier_flips >= 2:
        rating = "unstable"
    elif max_combined >= 6.0 or tier_flips >= 1 or band_flips >= 1:
        rating = "moderate"
    else:
        rating = "stable"
    if tier_flips >= 2:
        tier_flip_risk = "high"
    elif tier_flips == 1 or band_flips >= 1:
        tier_flip_risk = "medium"
    else:
        tier_flip_risk = "low"
    stability_notes: list[str] = []
    if max_jitter > 0.0:
        stability_notes.append(f"Small location perturbations changed wildfire risk by up to {max_jitter:.1f}.")
    if max_fallback > 0.0:
        stability_notes.append(f"Fallback-assumption perturbations changed wildfire risk by up to {max_fallback:.1f}.")
    stability = TrustDiagnosticsStability(
        rating=rating,
        local_sensitivity_score=round(local_sensitivity_score, 1),
        geocode_jitter_swing=round(max_jitter, 1),
        fallback_assumption_swing=round(max_fallback, 1),
        tier_flip_risk=tier_flip_risk,
        notes=stability_notes,
    )

    interventions: list[TrustDiagnosticsInterventionImpact] = []
    backwards_or_zero: list[str] = []
    monotonic_checks: list[str] = []
    monotonic_violations: list[str] = []
    for row in mitigation_samples:
        name = str(row.get("name") or "intervention")
        expected_direction = str(row.get("expected_direction") or "down")
        risk_delta = float(row.get("risk_delta") or 0.0)
        readiness_delta = float(row.get("readiness_delta") or 0.0)
        direction_ok = (
            risk_delta <= 0.0 if expected_direction == "down" else risk_delta >= 0.0
        )
        if expected_direction == "down" and risk_delta >= 0.0:
            backwards_or_zero.append(name)
        monotonic_checks.append(f"{name}:{expected_direction}")
        if not direction_ok:
            monotonic_violations.append(name)
        interventions.append(
            TrustDiagnosticsInterventionImpact(
                name=name,
                estimated_risk_delta=round(risk_delta, 2),
                estimated_readiness_delta=round(readiness_delta, 2),
                directionally_expected=direction_ok,
                notes=list(row.get("notes") or []),
            )
        )
    interventions.sort(key=lambda item: item.estimated_risk_delta)
    mitigation = TrustDiagnosticsMitigationSensitivity(
        top_interventions=interventions[:5],
        backwards_or_zero_impact_flags=sorted(set(backwards_or_zero)),
    )

    if monotonic_violations:
        monotonic_status = "fail"
    elif backwards_or_zero:
        monotonic_status = "warn"
    else:
        monotonic_status = "pass"
    monotonicity = TrustDiagnosticsMonotonicity(
        checks_run=monotonic_checks,
        violations=sorted(set(monotonic_violations)),
        status=monotonic_status,
    )

    alignment_available = bool(reference_artifacts.get("available"))
    alignment_rows = reference_artifacts.get("alignment_rows") if isinstance(reference_artifacts.get("alignment_rows"), list) else []
    signals_used: list[str] = []
    correlations: list[float] = []
    for row in alignment_rows:
        if not isinstance(row, dict):
            continue
        signal = str(row.get("signal_key") or "").strip()
        if signal:
            signals_used.append(signal)
        corr = _safe_float(row.get("spearman_rank_correlation"))
        if corr is not None:
            correlations.append(corr)
    avg_corr = (sum(correlations) / float(len(correlations))) if correlations else None
    if not alignment_available or avg_corr is None:
        local_alignment = "unknown"
    elif avg_corr >= 0.5:
        local_alignment = "high"
    elif avg_corr >= 0.25:
        local_alignment = "moderate"
    else:
        local_alignment = "low"
    alignment_notes = [
        "Benchmark alignment is a sanity check only and not ground-truth validation."
    ]
    if reference_artifacts.get("warnings"):
        alignment_notes.extend(list(reference_artifacts.get("warnings") or []))
    benchmark_alignment = TrustDiagnosticsBenchmarkAlignment(
        available=alignment_available,
        signals_used=sorted(set(signals_used)),
        local_alignment=local_alignment,
        notes=alignment_notes,
    )

    reference_scores = (
        reference_artifacts.get("distribution_scores")
        if isinstance(reference_artifacts.get("distribution_scores"), list)
        else []
    )
    percentile = (
        _percentile(float(result.wildfire_risk_score or 0.0), [float(v) for v in reference_scores])
        if reference_scores and result.wildfire_risk_score is not None
        else None
    )
    distribution_notes: list[str] = []
    if percentile is None:
        distribution_notes.append("Reference distribution baseline unavailable; percentile is unknown.")
    segment = TrustDiagnosticsDistributionSegment(
        region=result.resolved_region_id or ((result.property_level_context or {}).get("region_id") if isinstance(result.property_level_context, dict) else None),
        settlement_pattern=(
            str((result.property_level_context or {}).get("settlement_pattern"))
            if isinstance(result.property_level_context, dict)
            and (result.property_level_context or {}).get("settlement_pattern") is not None
            else None
        ),
        evidence_tier=_evidence_tier_for_result(result),
    )
    distribution_context = TrustDiagnosticsDistributionContext(
        relative_risk_percentile=percentile,
        segment=segment,
        notes=distribution_notes,
    )

    explanations: list[str] = []
    if confidence.notes:
        explanations.append(
            f"Confidence is {confidence.tier} because " + confidence.notes[0][0].lower() + confidence.notes[0][1:]
        )
    if stability.rating == "unstable":
        explanations.append("This result is sensitive to small location or assumption changes.")
    elif stability.rating == "moderate":
        explanations.append("This result shows moderate sensitivity to small perturbations.")
    if mitigation.top_interventions:
        top = mitigation.top_interventions[0]
        explanations.append(
            f"Highest estimated mitigation leverage: {top.name} (risk delta {top.estimated_risk_delta:+.1f})."
        )

    return TrustDiagnostics(
        generated_at=result.generated_at,
        caveat=TRUST_DIAGNOSTIC_CAVEAT,
        confidence=confidence,
        stability=stability,
        mitigation_sensitivity=mitigation,
        monotonicity=monotonicity,
        benchmark_alignment=benchmark_alignment,
        distribution_context=distribution_context,
        explanations=explanations,
    )

