from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.differentiation import (
    build_differentiation_snapshot,
    should_trigger_nearby_home_comparison_safeguard,
)
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
    TrustDiagnosticsVegetationSignal,
)
from backend.scoring_config import load_scoring_config

TRUST_DIAGNOSTIC_CAVEAT = (
    "These diagnostics measure model coherence, stability, evidence quality, and external alignment. "
    "They do not establish real-world predictive accuracy or insurer approval."
)
_SCORING_CONFIG = load_scoring_config()
_RISK_BUCKET_THRESHOLDS = _SCORING_CONFIG.risk_bucket_thresholds or {}
_TRUST_STABILITY_PARAMS = _SCORING_CONFIG.trust_stability_params or {}


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
    try:
        low_max = float(_RISK_BUCKET_THRESHOLDS.get("low_max", 40.0))
    except (TypeError, ValueError):
        low_max = 40.0
    try:
        medium_max = float(_RISK_BUCKET_THRESHOLDS.get("medium_max", 60.0))
    except (TypeError, ValueError):
        medium_max = 60.0
    if score < low_max:
        return "low"
    if score < medium_max:
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


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(row) for row in value if isinstance(row, str) and str(row).strip()]


def _build_vegetation_signal(result: AssessmentResult) -> TrustDiagnosticsVegetationSignal:
    vegetation_submodels = [
        "vegetation_intensity_risk",
        "fuel_proximity_risk",
        "flame_contact_risk",
        "defensible_space_risk",
    ]
    weighted = (
        result.weighted_contributions
        if isinstance(result.weighted_contributions, dict)
        else {}
    )
    total_contribution = 0.0
    vegetation_contribution = 0.0
    for key, row in weighted.items():
        contribution_value: Any = None
        if isinstance(row, dict):
            contribution_value = row.get("contribution")
        elif hasattr(row, "contribution"):
            contribution_value = getattr(row, "contribution", None)
        contribution = _safe_float(contribution_value)
        if contribution is None:
            continue
        abs_contribution = abs(contribution)
        total_contribution += abs_contribution
        if str(key) in vegetation_submodels:
            vegetation_contribution += abs_contribution
    contribution_share = (
        round((vegetation_contribution / total_contribution), 4)
        if total_contribution > 0.0
        else None
    )
    top_driver_text = [str(row) for row in (result.top_risk_drivers or []) if str(row).strip()]
    near_structure_drivers = [
        str(row) for row in (result.top_near_structure_risk_drivers or []) if str(row).strip()
    ]
    driver_text_pool = " ".join(top_driver_text + near_structure_drivers).lower()
    keywords = (
        "vegetation",
        "defensible",
        "fuel",
        "0-5",
        "5-30",
        "canopy",
        "near-structure",
    )
    vegetation_driver_mentions = [
        row
        for row in (top_driver_text + near_structure_drivers)
        if any(token in str(row).lower() for token in keywords)
    ]
    keyword_hit = any(token in driver_text_pool for token in keywords)
    major_driver = bool(
        (contribution_share is not None and contribution_share >= 0.30)
        or (near_structure_drivers and keyword_hit)
        or len(vegetation_driver_mentions) >= 2
    )
    if contribution_share is None:
        strength = "unknown"
    elif contribution_share >= 0.40:
        strength = "high"
    elif contribution_share >= 0.25:
        strength = "moderate"
    elif contribution_share > 0.0:
        strength = "low"
    else:
        strength = "unknown"

    defensible = (
        result.defensible_space_analysis
        if isinstance(result.defensible_space_analysis, dict)
        else {}
    )
    near_structure_summary = (
        str(defensible.get("summary"))
        if defensible.get("summary") is not None
        else None
    )
    data_quality = defensible.get("data_quality") if isinstance(defensible.get("data_quality"), dict) else {}
    analysis_status = str(data_quality.get("analysis_status") or "").strip().lower()
    notes: list[str] = []
    if near_structure_summary:
        notes.append(near_structure_summary)
    if analysis_status in {"partial", "unavailable"}:
        notes.append("Near-structure vegetation evidence is incomplete; some vegetation effects use proxy inputs.")
    if major_driver and contribution_share is not None:
        notes.append(
            f"Vegetation-linked submodels contribute about {contribution_share * 100.0:.1f}% of modeled weighted risk pressure."
        )
    related_submodels: list[str] = []
    for key in vegetation_submodels:
        if key not in weighted:
            continue
        row = weighted[key]
        contribution_value: Any = None
        if isinstance(row, dict):
            contribution_value = row.get("contribution")
        elif hasattr(row, "contribution"):
            contribution_value = getattr(row, "contribution", None)
        if _safe_float(contribution_value) is not None:
            related_submodels.append(key)
    return TrustDiagnosticsVegetationSignal(
        major_driver=major_driver,
        driver_strength=strength,
        contribution_share=contribution_share,
        related_submodels=related_submodels,
        related_risk_drivers=vegetation_driver_mentions[:8],
        near_structure_summary=near_structure_summary,
        notes=notes[:8],
    )


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
    # WHP / burn_probability_index confidence tiers:
    #
    # The burn_probability_index feeds the ember exposure submodel at 0.31 weight
    # (risk_engine.py). Three source tiers produce different confidence impacts:
    #
    #   Tier 1 — local WHP/burn-prob raster (source not in hazard_context["whp_index_source"]):
    #     No penalty. Direct measurement from prepared region raster.
    #
    #   Tier 2 — proxy formula (hazard_context["whp_index_source"] == "whp_proxy"):
    #     Small penalty applied via the "proxy" token in the assumptions list.
    #     risk_engine._availability_multiplier() applies multiplier *= 0.88 for ember
    #     exposure when assumptions contain "proxy". This is roughly -4 to -6 confidence
    #     points depending on submodel weight. The proxy appends:
    #       "Wildfire Hazard Potential derived from proxy formula; direct measurement
    #        unavailable at property location."
    #     to the assumptions list in wildfire_data.py immediately after computing it.
    #
    #   Tier 3 — missing (burn_probability_index is None, burn_missing=True):
    #     Full penalty: multiplier *= 0.60 for ember_exposure_risk submodel
    #     (risk_engine.py line ~1399). This is the largest single-layer confidence hit.
    #     No change needed here — this path is unchanged.
    #
    # Action: the proxy assumption text ("proxy formula") triggers the existing
    # "proxy" token in _availability_multiplier's has_low_quality_assumption check.
    # No additional code changes to risk_engine.py are required.
    evidence_completeness = _safe_float(result.feature_coverage_percent)
    if evidence_completeness is None or evidence_completeness <= 0.0:
        evidence_completeness = round(float(result.observed_weight_fraction or 0.0) * 100.0, 1)

    fallback_weight_fraction = float(result.fallback_weight_fraction or 0.0)
    _fallback_heavy_threshold = float(_TRUST_STABILITY_PARAMS.get("fallback_heavy_fraction", 0.45))
    fallback_heavy = bool(
        float(result.fallback_weight_fraction or 0.0) >= _fallback_heavy_threshold
        or int(result.fallback_feature_count or 0) > int(result.observed_feature_count or 0)
    )
    inferred_fields = sorted(
        set(
            list(result.assessment_diagnostics.inferred_inputs or [])
            + list(result.assessment_diagnostics.heuristic_inputs or [])
        )
    )
    confidence_reduction_reasons = sorted(
        {
            *[str(row) for row in (result.assessment_diagnostics.confidence_downgrade_reasons or []) if str(row).strip()],
            *[str(row) for row in (result.assessment_diagnostics.trust_tier_blockers or []) if str(row).strip()],
            *[str(row) for row in (result.low_confidence_flags or []) if str(row).strip()],
        }
    )
    confidence_notes: list[str] = []
    if fallback_heavy:
        confidence_notes.append("Fallback assumptions influence a large share of the score.")
    if result.assessment_diagnostics.trust_tier_blockers:
        confidence_notes.append("Trust-tier blockers are present in this assessment.")
    if result.assessment_diagnostics.critical_inputs_missing:
        confidence_notes.append("Critical property inputs are missing or inferred.")
    if inferred_fields:
        confidence_notes.append("Some property inputs were inferred or proxy-derived instead of directly observed.")

    confidence = TrustDiagnosticsConfidence(
        tier=result.confidence_tier,
        score=float(result.confidence_score or 0.0),
        evidence_completeness=float(evidence_completeness or 0.0),
        fallback_heavy=fallback_heavy,
        fallback_weight_fraction=round(fallback_weight_fraction, 4),
        observed_feature_count=int(result.observed_feature_count or 0),
        inferred_feature_count=int(result.inferred_feature_count or 0),
        fallback_feature_count=int(result.fallback_feature_count or 0),
        missing_feature_count=int(result.missing_feature_count or 0),
        missing_critical_fields=list(result.assessment_diagnostics.critical_inputs_missing or []),
        missing_critical_field_count=len(result.assessment_diagnostics.critical_inputs_missing or []),
        inferred_fields=inferred_fields,
        inferred_field_count=len(inferred_fields),
        confidence_reduction_reasons=confidence_reduction_reasons[:10],
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
    _unstable_thresh = float(_TRUST_STABILITY_PARAMS.get("unstable_swing_threshold", 12.0))
    _moderate_thresh = float(_TRUST_STABILITY_PARAMS.get("moderate_swing_threshold", 6.0))
    _sensitivity_mult = float(_TRUST_STABILITY_PARAMS.get("sensitivity_score_multiplier", 6.0))
    _assumption_sensitive_swing = float(_TRUST_STABILITY_PARAMS.get("assumption_sensitive_swing", 4.0))
    local_sensitivity_score = max(0.0, min(100.0, 100.0 - (max_combined * _sensitivity_mult)))
    if max_combined >= _unstable_thresh or tier_flips >= 2:
        rating = "unstable"
    elif max_combined >= _moderate_thresh or tier_flips >= 1 or band_flips >= 1:
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
        assumption_sensitive=bool(max_fallback >= _assumption_sensitive_swing or tier_flips >= 1),
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
    preflight = (
        (result.developer_diagnostics or {}).get("preflight")
        if isinstance(result.developer_diagnostics, dict)
        else {}
    )
    preflight = preflight if isinstance(preflight, dict) else {}
    differentiation = build_differentiation_snapshot(
        feature_coverage_summary=dict(result.feature_coverage_summary or {}),
        preflight=preflight,
        property_level_context=(
            dict(result.property_level_context)
            if isinstance(result.property_level_context, dict)
            else {}
        ),
        environmental_layer_status=dict(result.environmental_layer_status or {}),
        fallback_weight_fraction=float(result.fallback_weight_fraction or 0.0),
        missing_inputs=list(result.missing_inputs or []),
        inferred_inputs=list(result.assessment_diagnostics.inferred_inputs or []),
        input_source_metadata=dict(result.input_source_metadata or {}),
        fallback_decisions=list(result.assessment_diagnostics.fallback_decisions or []),
    )
    vegetation_signal = _build_vegetation_signal(result)

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
    if vegetation_signal.major_driver:
        explanations.append("Near-structure vegetation appears to be a major contributor to this property risk profile.")
    elif vegetation_signal.driver_strength in {"moderate", "low"}:
        explanations.append("Near-structure vegetation contributes to risk, but it is not the dominant modeled driver.")
    if confidence.confidence_reduction_reasons:
        explanations.append(
            "Confidence reductions are driven by: "
            + ", ".join(confidence.confidence_reduction_reasons[:3])
            + "."
        )
    differentiation_mode = str(differentiation.get("differentiation_mode") or "")
    differentiation_confidence = float(
        differentiation.get("local_differentiation_score")
        or differentiation.get("neighborhood_differentiation_confidence")
        or 0.0
    )
    if should_trigger_nearby_home_comparison_safeguard(differentiation_mode, differentiation_confidence):
        explanations.append("This estimate is not precise enough to compare adjacent homes.")
    elif differentiation_mode == "mostly_regional":
        explanations.append("Property-level differentiation is limited; this estimate is mostly regional.")

    return TrustDiagnostics(
        generated_at=result.generated_at,
        caveat=TRUST_DIAGNOSTIC_CAVEAT,
        confidence=confidence,
        stability=stability,
        mitigation_sensitivity=mitigation,
        monotonicity=monotonicity,
        benchmark_alignment=benchmark_alignment,
        distribution_context=distribution_context,
        differentiation_mode=str(differentiation.get("differentiation_mode") or "mostly_regional"),
        property_specific_feature_count=int(differentiation.get("property_specific_feature_count") or 0),
        proxy_feature_count=int(differentiation.get("proxy_feature_count") or 0),
        defaulted_feature_count=int(differentiation.get("defaulted_feature_count") or 0),
        regional_feature_count=int(differentiation.get("regional_feature_count") or 0),
        local_differentiation_score=float(
            differentiation.get("local_differentiation_score")
            or differentiation.get("neighborhood_differentiation_confidence")
            or 0.0
        ),
        neighborhood_differentiation_confidence=float(
            differentiation.get("neighborhood_differentiation_confidence") or 0.0
        ),
        differentiation_notes=[str(row) for row in list(differentiation.get("notes") or []) if str(row).strip()],
        vegetation_signal=vegetation_signal,
        explanations=explanations,
    )
