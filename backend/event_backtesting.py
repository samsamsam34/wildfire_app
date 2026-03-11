from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.benchmarking import build_wildfire_context, patched_runtime_inputs
from backend.models import AddressRequest, AssessmentResult, UnderwritingRuleset
from backend.scoring_config import load_scoring_config
from backend.version import (
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    FACTOR_SCHEMA_VERSION,
    MODEL_VERSION,
    build_model_governance,
)

OUTCOME_LABELS = {
    "destroyed",
    "major_damage",
    "minor_damage",
    "no_known_damage",
    "unknown",
}
OUTCOME_RANK_DEFAULTS = {
    "unknown": 0,
    "no_known_damage": 1,
    "minor_damage": 2,
    "major_damage": 3,
    "destroyed": 4,
}
DEFAULT_DATASET_PATH = Path("benchmark") / "event_backtest_sample_v1.json"
DEFAULT_RESULTS_DIR = Path("benchmark") / "event_backtest_results"
DEFAULT_SCORING_CONFIG = load_scoring_config()


@dataclass
class EventBacktestRecord:
    event_id: str
    event_name: str
    event_date: str
    source_name: str
    record_id: str
    latitude: float
    longitude: float
    address_text: str | None = None
    geometry: dict[str, Any] | None = None
    outcome_label: str = "unknown"
    outcome_rank: int = 0
    label_confidence: float | None = None
    notes: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    input_payload: dict[str, Any] = field(default_factory=dict)
    context_overrides: dict[str, Any] = field(default_factory=dict)
    geocode_source: str = "event-backtest"
    organization_id: str = "default_org"
    ruleset_id: str = "default"
    assessment_id: str | None = None


@dataclass
class EventBacktestDataset:
    dataset_id: str
    dataset_name: str
    source_name: str
    event_id: str | None
    event_name: str | None
    event_date: str | None
    records: list[EventBacktestRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_outcome_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    if label in OUTCOME_LABELS:
        return label
    return "unknown"


def _normalize_outcome_rank(label: str, rank: Any) -> int:
    if rank is not None and str(rank).strip() != "":
        try:
            return int(float(rank))
        except Exception:
            pass
    return OUTCOME_RANK_DEFAULTS.get(label, 0)


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _normalize_record(
    raw: dict[str, Any],
    *,
    dataset_defaults: dict[str, Any],
    fallback_index: int,
) -> EventBacktestRecord:
    event_id = str(raw.get("event_id") or dataset_defaults.get("event_id") or "unknown_event")
    event_name = str(raw.get("event_name") or dataset_defaults.get("event_name") or event_id)
    event_date = str(raw.get("event_date") or dataset_defaults.get("event_date") or "")
    source_name = str(raw.get("source_name") or dataset_defaults.get("source_name") or "unknown_source")
    record_id = str(raw.get("record_id") or f"{event_id}_{fallback_index:04d}")
    latitude = _to_float(raw.get("latitude"))
    longitude = _to_float(raw.get("longitude"))
    if latitude is None or longitude is None:
        raise ValueError(f"Record {record_id} missing latitude/longitude.")

    outcome_label = _normalize_outcome_label(raw.get("outcome_label"))
    outcome_rank = _normalize_outcome_rank(outcome_label, raw.get("outcome_rank"))
    label_confidence = _to_float(raw.get("label_confidence"))
    source_metadata = _parse_jsonish(raw.get("source_metadata"), {})
    input_payload = _parse_jsonish(raw.get("input_payload"), {})
    context_overrides = _parse_jsonish(raw.get("context_overrides"), {})
    geometry = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else None
    return EventBacktestRecord(
        event_id=event_id,
        event_name=event_name,
        event_date=event_date,
        source_name=source_name,
        record_id=record_id,
        latitude=float(latitude),
        longitude=float(longitude),
        address_text=(str(raw.get("address_text")).strip() if raw.get("address_text") else None),
        geometry=geometry,
        outcome_label=outcome_label,
        outcome_rank=outcome_rank,
        label_confidence=label_confidence,
        notes=(str(raw.get("notes")).strip() if raw.get("notes") else None),
        source_metadata=source_metadata if isinstance(source_metadata, dict) else {},
        input_payload=input_payload if isinstance(input_payload, dict) else {},
        context_overrides=context_overrides if isinstance(context_overrides, dict) else {},
        geocode_source=str(raw.get("geocode_source") or "event-backtest"),
        organization_id=str(raw.get("organization_id") or dataset_defaults.get("organization_id") or "default_org"),
        ruleset_id=str(raw.get("ruleset_id") or dataset_defaults.get("ruleset_id") or "default"),
        assessment_id=(str(raw.get("assessment_id")) if raw.get("assessment_id") else None),
    )


def _load_from_json(path: Path) -> EventBacktestDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"records": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset JSON must be object/list: {path}")
    records_raw = payload.get("records")
    if not isinstance(records_raw, list):
        raise ValueError(f"Dataset JSON missing 'records' list: {path}")
    defaults = {
        "event_id": payload.get("event_id"),
        "event_name": payload.get("event_name"),
        "event_date": payload.get("event_date"),
        "source_name": payload.get("source_name"),
        "organization_id": payload.get("organization_id"),
        "ruleset_id": payload.get("ruleset_id"),
    }
    records: list[EventBacktestRecord] = []
    for i, raw in enumerate(records_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset record must be object at index {i}: {path}")
        records.append(_normalize_record(raw, dataset_defaults=defaults, fallback_index=i + 1))
    return EventBacktestDataset(
        dataset_id=str(payload.get("dataset_id") or path.stem),
        dataset_name=str(payload.get("dataset_name") or path.stem),
        source_name=str(payload.get("source_name") or "unknown_source"),
        event_id=(str(payload.get("event_id")) if payload.get("event_id") else None),
        event_name=(str(payload.get("event_name")) if payload.get("event_name") else None),
        event_date=(str(payload.get("event_date")) if payload.get("event_date") else None),
        records=records,
        metadata={
            "input_path": str(path),
            "dataset_format": "json",
            "event_backtest_version": payload.get("event_backtest_version"),
        },
    )


def _load_from_csv(path: Path) -> EventBacktestDataset:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    defaults = {
        "event_id": rows[0].get("event_id") if rows else path.stem,
        "event_name": rows[0].get("event_name") if rows else path.stem,
        "event_date": rows[0].get("event_date") if rows else "",
        "source_name": rows[0].get("source_name") if rows else "unknown_source",
        "organization_id": rows[0].get("organization_id") if rows else "default_org",
        "ruleset_id": rows[0].get("ruleset_id") if rows else "default",
    }
    records = [
        _normalize_record(row, dataset_defaults=defaults, fallback_index=i + 1)
        for i, row in enumerate(rows)
    ]
    return EventBacktestDataset(
        dataset_id=path.stem,
        dataset_name=path.stem,
        source_name=str(defaults["source_name"] or "unknown_source"),
        event_id=str(defaults["event_id"] or ""),
        event_name=str(defaults["event_name"] or ""),
        event_date=str(defaults["event_date"] or ""),
        records=records,
        metadata={"input_path": str(path), "dataset_format": "csv"},
    )


def _load_from_geojson(path: Path) -> EventBacktestDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"GeoJSON must be object: {path}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON missing features list: {path}")
    defaults = {
        "event_id": payload.get("event_id") or payload.get("name") or path.stem,
        "event_name": payload.get("event_name") or payload.get("name") or path.stem,
        "event_date": payload.get("event_date") or "",
        "source_name": payload.get("source_name") or "unknown_source",
        "organization_id": payload.get("organization_id") or "default_org",
        "ruleset_id": payload.get("ruleset_id") or "default",
    }
    rows: list[dict[str, Any]] = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
        geom = feat.get("geometry") if isinstance(feat.get("geometry"), dict) else None
        lat = props.get("latitude")
        lon = props.get("longitude")
        if geom and isinstance(geom.get("coordinates"), list):
            coords = geom.get("coordinates")
            if geom.get("type") == "Point" and len(coords) >= 2:
                lon = coords[0]
                lat = coords[1]
        row = dict(props)
        row["latitude"] = lat
        row["longitude"] = lon
        row["geometry"] = geom
        row.setdefault("record_id", props.get("record_id") or f"{path.stem}_{i+1:04d}")
        rows.append(row)
    records = [
        _normalize_record(row, dataset_defaults=defaults, fallback_index=i + 1)
        for i, row in enumerate(rows)
    ]
    return EventBacktestDataset(
        dataset_id=path.stem,
        dataset_name=str(payload.get("dataset_name") or path.stem),
        source_name=str(defaults["source_name"] or "unknown_source"),
        event_id=str(defaults["event_id"] or ""),
        event_name=str(defaults["event_name"] or ""),
        event_date=str(defaults["event_date"] or ""),
        records=records,
        metadata={"input_path": str(path), "dataset_format": "geojson"},
    )


def load_event_backtest_dataset(path: str | Path) -> EventBacktestDataset:
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"Backtest dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _load_from_csv(p)
    if suffix in {".geojson", ".json"}:
        if suffix == ".geojson":
            return _load_from_geojson(p)
        # For .json, support either JSON records or GeoJSON feature collection.
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
            return _load_from_geojson(p)
        return _load_from_json(p)
    raise ValueError(f"Unsupported dataset format for {p}. Use CSV, JSON, or GeoJSON.")


def _risk_bucket(score: float | None) -> str:
    thresholds = DEFAULT_SCORING_CONFIG.risk_bucket_thresholds
    try:
        low_max = float(thresholds.get("low_max", 33.0))
    except (TypeError, ValueError):
        low_max = 33.0
    try:
        medium_max = float(thresholds.get("medium_max", 66.0))
    except (TypeError, ValueError):
        medium_max = 66.0

    if score is None:
        return "unscored"
    if score < low_max:
        return "low"
    if score < medium_max:
        return "medium"
    return "high"


def _adverse_outcome(rank: int) -> bool:
    return int(rank) >= 3


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman_rank_correlation(pairs: list[tuple[float | None, float | None]]) -> float | None:
    filtered = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
    if len(filtered) < 3:
        return None
    xs = [p[0] for p in filtered]
    ys = [p[1] for p in filtered]
    rx = _rank_values(xs)
    ry = _rank_values(ys)
    return _pearson(rx, ry)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    seq = sorted(values)
    n = len(seq)
    mid = n // 2
    if n % 2:
        return seq[mid]
    return (seq[mid - 1] + seq[mid]) / 2.0


def _score_distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, list[float]]] = {}
    fields = [
        "wildfire_risk_score",
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "insurance_readiness_score",
    ]
    for rec in records:
        label = str(rec["outcome_label"])
        by_label.setdefault(label, {k: [] for k in fields})
        for field in fields:
            val = rec.get("scores", {}).get(field)
            if isinstance(val, (int, float)):
                by_label[label][field].append(float(val))
    out: dict[str, Any] = {}
    for label, bucket in by_label.items():
        out[label] = {}
        for field, values in bucket.items():
            out[label][field] = {
                "count": len(values),
                "mean": (sum(values) / len(values)) if values else None,
                "median": _median(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    return out


def _bucket_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"low": [], "medium": [], "high": [], "unscored": []}
    for rec in records:
        score = rec.get("scores", {}).get("wildfire_risk_score")
        bucket = _risk_bucket(score if isinstance(score, (int, float)) else None)
        buckets[bucket].append(rec)
    out: dict[str, Any] = {}
    for bucket, rows in buckets.items():
        adverse = sum(1 for r in rows if _adverse_outcome(int(r.get("outcome_rank", 0))))
        out[bucket] = {
            "count": len(rows),
            "adverse_count": adverse,
            "adverse_rate": (adverse / len(rows)) if rows else None,
        }
    return out


def _evidence_group(record: dict[str, Any]) -> str:
    evidence = record.get("evidence_quality_summary") or {}
    coverage = record.get("coverage_summary") or {}
    observed = int(evidence.get("observed_factor_count") or 0)
    inferred = int(evidence.get("inferred_factor_count") or 0)
    missing = int(evidence.get("missing_factor_count") or 0)
    fallback = int(evidence.get("fallback_factor_count") or 0)
    failed_layers = int(coverage.get("failed_count") or 0)
    if failed_layers > 0 or fallback > observed or (missing + fallback) >= max(2, observed):
        return "fallback_heavy"
    if fallback == 0 and missing <= 1 and observed >= max(3, inferred + missing):
        return "high_evidence"
    return "mixed_evidence"


def _confidence_stratification(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"high_evidence": [], "mixed_evidence": [], "fallback_heavy": []}
    for rec in records:
        grp = _evidence_group(rec)
        rec["evidence_group"] = grp
        groups[grp].append(rec)
    out: dict[str, Any] = {}
    for name, rows in groups.items():
        pairs = [
            (
                (r.get("scores", {}) or {}).get("wildfire_risk_score"),
                r.get("outcome_rank"),
            )
            for r in rows
        ]
        conf_vals = [
            float((r.get("confidence") or {}).get("confidence_score"))
            for r in rows
            if isinstance((r.get("confidence") or {}).get("confidence_score"), (int, float))
        ]
        adverse = sum(1 for r in rows if _adverse_outcome(int(r.get("outcome_rank", 0))))
        out[name] = {
            "count": len(rows),
            "wildfire_outcome_spearman": spearman_rank_correlation(pairs),
            "mean_confidence": (sum(conf_vals) / len(conf_vals)) if conf_vals else None,
            "adverse_rate": (adverse / len(rows)) if rows else None,
        }
    return out


def _false_low_false_high(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = DEFAULT_SCORING_CONFIG.error_analysis_thresholds
    false_low_max = float(thresholds.get("false_low_max_score", 40.0))
    false_high_min = float(thresholds.get("false_high_min_score", 70.0))
    adverse_min = int(float(thresholds.get("adverse_outcome_min_rank", 3.0)))
    non_adverse_max = int(float(thresholds.get("non_adverse_outcome_max_rank", 1.0)))

    false_low: list[dict[str, Any]] = []
    false_high: list[dict[str, Any]] = []
    for rec in records:
        score = (rec.get("scores") or {}).get("wildfire_risk_score")
        if not isinstance(score, (int, float)):
            continue
        rank = int(rec.get("outcome_rank", 0))
        if rank >= adverse_min and score < false_low_max:
            false_low.append(rec)
        if rank <= non_adverse_max and score >= false_high_min:
            false_high.append(rec)
    false_low.sort(key=lambda r: (r.get("scores", {}).get("wildfire_risk_score") or 0))
    false_high.sort(key=lambda r: (r.get("scores", {}).get("wildfire_risk_score") or 0), reverse=True)
    return false_low, false_high


def _recommendations(analysis: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    corr = analysis.get("rank_correlation", {}).get("wildfire_vs_outcome")
    if isinstance(corr, (int, float)) and corr < 0.2:
        recs.append("Wildfire risk ordering is weak vs outcome rank; review score-band thresholds and key factor weights.")
    bucket_high = (analysis.get("bucket_analysis", {}).get("high") or {}).get("adverse_rate")
    bucket_med = (analysis.get("bucket_analysis", {}).get("medium") or {}).get("adverse_rate")
    if isinstance(bucket_high, (int, float)) and isinstance(bucket_med, (int, float)) and bucket_high <= bucket_med:
        recs.append("High-risk bucket is not separating adverse outcomes from medium bucket; review high-risk threshold and penalties.")
    strat = analysis.get("confidence_stratification", {})
    fallback = (strat.get("fallback_heavy") or {}).get("count")
    if isinstance(fallback, int) and records:
        if fallback / max(1, len(records)) >= 0.30:
            recs.append("Fallback-heavy records exceed 30%; prioritize layer coverage improvements before weight tuning.")
    if (analysis.get("false_low_count") or 0) > 0:
        recs.append("False-low adverse records detected; review defensible-space and near-structure vegetation penalties.")
    if (analysis.get("false_high_count") or 0) > 0:
        recs.append("False-high undamaged records detected; review over-penalization in severe vegetation/hazard contexts.")
    if not recs:
        recs.append("No major directional issues detected in this sample; continue monitoring with larger event datasets.")
    return recs[:6]


def _resolve_ruleset(ruleset_id: str | None) -> UnderwritingRuleset:
    import backend.main as app_main  # lazy import

    try:
        return app_main._get_ruleset_or_default(ruleset_id or "default")
    except Exception:
        return UnderwritingRuleset(
            ruleset_id=ruleset_id or "default",
            ruleset_name="Default Carrier Profile",
            ruleset_version="1.0.0",
            ruleset_description="Fallback ruleset for event backtest harness.",
            config={},
        )


def _record_snapshot(
    record: EventBacktestRecord,
    result: AssessmentResult,
    *,
    dataset_id: str,
    debug_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug_payload = debug_payload or {}
    governance = (
        result.model_governance.model_dump()
        if result.model_governance
        else build_model_governance(
            ruleset_version=result.ruleset_version,
            benchmark_pack_version=BENCHMARK_PACK_VERSION,
            region_data_version=(result.property_level_context or {}).get("region_manifest_path"),
        )
    )
    return {
        "dataset_id": dataset_id,
        "event_id": record.event_id,
        "event_name": record.event_name,
        "event_date": record.event_date,
        "source_name": record.source_name,
        "record_id": record.record_id,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "address_text": record.address_text,
        "outcome_label": record.outcome_label,
        "outcome_rank": record.outcome_rank,
        "label_confidence": record.label_confidence,
        "scores": {
            "wildfire_risk_score": result.wildfire_risk_score,
            "site_hazard_score": result.site_hazard_score,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "insurance_readiness_score": result.insurance_readiness_score,
            "wildfire_risk_score_available": result.wildfire_risk_score_available,
            "site_hazard_score_available": result.site_hazard_score_available,
            "home_ignition_vulnerability_score_available": result.home_ignition_vulnerability_score_available,
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "risk_bucket": _risk_bucket(result.wildfire_risk_score),
        },
        "confidence": {
            "confidence_score": result.confidence_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
        },
        "evidence_quality_summary": result.evidence_quality_summary.model_dump(),
        "coverage_summary": result.coverage_summary.model_dump(),
        "assessment_status": result.assessment_status,
        "assessment_blockers": list(result.assessment_blockers or []),
        "readiness_blockers": list(result.readiness_blockers or []),
        "top_risk_drivers": list(result.top_risk_drivers or []),
        "score_evidence_ledger_summary": {
            "wildfire_risk_score": {
                row.factor_key: float(row.contribution)
                for row in result.score_evidence_ledger.wildfire_risk_score
            },
            "site_hazard_score": {
                row.factor_key: float(row.contribution)
                for row in result.score_evidence_ledger.site_hazard_score
            },
        },
        "scoring_notes": list(result.scoring_notes or []),
        "raw_feature_vector": (
            debug_payload.get("raw_feature_vector")
            if isinstance(debug_payload.get("raw_feature_vector"), dict)
            else {}
        ),
        "transformed_feature_vector": (
            debug_payload.get("transformed_feature_vector")
            if isinstance(debug_payload.get("transformed_feature_vector"), dict)
            else {}
        ),
        "factor_contribution_breakdown": (
            debug_payload.get("factor_contribution_breakdown")
            if isinstance(debug_payload.get("factor_contribution_breakdown"), dict)
            else {}
        ),
        "compression_flags": (
            debug_payload.get("compression_flags")
            if isinstance(debug_payload.get("compression_flags"), list)
            else []
        ),
        "score_variance_diagnostics": (
            debug_payload.get("score_variance_diagnostics")
            if isinstance(debug_payload.get("score_variance_diagnostics"), dict)
            else {}
        ),
        "calibration": (
            debug_payload.get("calibration")
            if isinstance(debug_payload.get("calibration"), dict)
            else {}
        ),
        "property_level_context": (
            debug_payload.get("property_level_context")
            if isinstance(debug_payload.get("property_level_context"), dict)
            else {}
        ),
        "model_governance": governance,
    }


def _run_record_assessment(
    record: EventBacktestRecord,
    *,
    ruleset_id: str | None = None,
    reuse_existing_assessments: bool = False,
) -> dict[str, Any]:
    import backend.main as app_main  # lazy import

    if reuse_existing_assessments and record.assessment_id:
        existing = app_main.store.get_assessment(record.assessment_id)
        if existing is not None:
            return _record_snapshot(record, existing, dataset_id="")

    payload_dict = dict(record.input_payload or {})
    payload_dict.setdefault(
        "address",
        record.address_text or f"{record.event_name} record {record.record_id}",
    )
    payload_dict.setdefault("attributes", {})
    payload_dict.setdefault("confirmed_fields", [])
    payload_dict.setdefault("audience", "insurer")
    payload_dict.setdefault("tags", ["event-backtest"])
    payload_dict["ruleset_id"] = str(payload_dict.get("ruleset_id") or ruleset_id or record.ruleset_id or "default")
    payload = AddressRequest.model_validate(payload_dict)
    resolved_ruleset = _resolve_ruleset(payload.ruleset_id)
    context = build_wildfire_context(record.context_overrides or {})
    with patched_runtime_inputs(
        latitude=record.latitude,
        longitude=record.longitude,
        geocode_source=record.geocode_source,
        context=context,
    ):
        result, debug_payload = app_main._run_assessment(
            payload,
            organization_id=record.organization_id,
            ruleset=resolved_ruleset,
        )
    return _record_snapshot(record, result, dataset_id="", debug_payload=debug_payload)


def _build_markdown_summary(artifact: dict[str, Any]) -> str:
    summary = artifact.get("summary", {})
    analysis = artifact.get("analysis", {})
    top_false_low = artifact.get("false_low_examples", [])[:5]
    top_false_high = artifact.get("false_high_examples", [])[:5]
    recs = analysis.get("recommendations", [])
    lines = [
        "# Event Backtest Summary",
        "",
        f"- Generated at: `{artifact.get('generated_at')}`",
        f"- Records assessed: `{summary.get('record_count')}`",
        f"- Events covered: `{summary.get('event_count')}`",
        f"- Datasets: `{', '.join(summary.get('dataset_ids', []))}`",
        "",
        "## Key Metrics",
        f"- Spearman (wildfire risk vs outcome rank): `{analysis.get('rank_correlation', {}).get('wildfire_vs_outcome')}`",
        f"- Spearman (site hazard vs outcome rank): `{analysis.get('rank_correlation', {}).get('site_hazard_vs_outcome')}`",
        f"- Spearman (insurance readiness vs adverse outcome rank): `{analysis.get('rank_correlation', {}).get('readiness_vs_outcome')}`",
        "",
        "## Evidence Coverage Notes",
        "- Backtest output separates `high_evidence`, `mixed_evidence`, and `fallback_heavy` groups.",
        "- Treat fallback-heavy and out-of-coverage records as directional QA, not threshold-tuning anchors.",
        "",
        "## False-Low Review",
    ]
    if not top_false_low:
        lines.append("- None in this run.")
    else:
        for row in top_false_low:
            lines.append(
                f"- `{row.get('event_id')}/{row.get('record_id')}` outcome={row.get('outcome_label')} "
                f"risk={row.get('scores', {}).get('wildfire_risk_score')} "
                f"confidence={row.get('confidence', {}).get('confidence_tier')}"
            )
    lines.extend(["", "## False-High Review"])
    if not top_false_high:
        lines.append("- None in this run.")
    else:
        for row in top_false_high:
            lines.append(
                f"- `{row.get('event_id')}/{row.get('record_id')}` outcome={row.get('outcome_label')} "
                f"risk={row.get('scores', {}).get('wildfire_risk_score')} "
                f"confidence={row.get('confidence', {}).get('confidence_tier')}"
            )
    lines.extend(["", "## Recommended Review Items"])
    for rec in recs or ["- No recommendation generated."]:
        lines.append(f"- {rec}" if not str(rec).startswith("- ") else str(rec))
    lines.extend(
        [
            "",
            "## Interpretation Caveats",
            "- Public event labels are proxy outcomes, not insurer claims truth.",
            "- Use this as directional empirical validation, not actuarial pricing validation.",
            "- Missing evidence must not be interpreted as low risk.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_event_backtest(
    *,
    dataset_paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    ruleset_id: str | None = None,
    reuse_existing_assessments: bool = False,
) -> dict[str, Any]:
    paths = [Path(p).expanduser() for p in (dataset_paths or [DEFAULT_DATASET_PATH])]
    datasets = [load_event_backtest_dataset(p) for p in paths]
    records: list[dict[str, Any]] = []
    for ds in datasets:
        for rec in ds.records:
            row = _run_record_assessment(
                rec,
                ruleset_id=ruleset_id,
                reuse_existing_assessments=reuse_existing_assessments,
            )
            row["dataset_id"] = ds.dataset_id
            records.append(row)

    pairs_risk = [(r.get("scores", {}).get("wildfire_risk_score"), r.get("outcome_rank")) for r in records]
    pairs_site = [(r.get("scores", {}).get("site_hazard_score"), r.get("outcome_rank")) for r in records]
    pairs_readiness = [(-(r.get("scores", {}).get("insurance_readiness_score")) if isinstance((r.get("scores", {}) or {}).get("insurance_readiness_score"), (int, float)) else None, r.get("outcome_rank")) for r in records]

    false_low, false_high = _false_low_false_high(records)
    analysis = {
        "rank_correlation": {
            "wildfire_vs_outcome": spearman_rank_correlation(pairs_risk),
            "site_hazard_vs_outcome": spearman_rank_correlation(pairs_site),
            "readiness_vs_outcome": spearman_rank_correlation(pairs_readiness),
        },
        "score_distributions_by_outcome": _score_distributions(records),
        "bucket_analysis": _bucket_analysis(records),
        "confidence_stratification": _confidence_stratification(records),
        "false_low_count": len(false_low),
        "false_high_count": len(false_high),
    }
    analysis["recommendations"] = _recommendations(analysis, records)

    product_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("product_version") or "unknown")
            for r in records
        }
    )
    api_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("api_version") or "unknown")
            for r in records
        }
    )
    scoring_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("scoring_model_version") or MODEL_VERSION)
            for r in records
        }
    )
    ruleset_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("ruleset_version") or "unknown")
            for r in records
        }
    )
    factor_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("factor_schema_version") or FACTOR_SCHEMA_VERSION)
            for r in records
        }
    )
    benchmark_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("benchmark_pack_version") or BENCHMARK_PACK_VERSION)
            for r in records
        }
    )
    calibration_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("calibration_version") or CALIBRATION_VERSION)
            for r in records
        }
    )
    region_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("region_data_version") or "unknown")
            for r in records
        }
    )
    data_bundle_versions = sorted(
        {
            str((r.get("model_governance") or {}).get("data_bundle_version") or "unknown")
            for r in records
        }
    )

    governance = build_model_governance(
        ruleset_version=(ruleset_versions[0] if len(ruleset_versions) == 1 else "mixed"),
        benchmark_pack_version=(benchmark_versions[0] if len(benchmark_versions) == 1 else BENCHMARK_PACK_VERSION),
        region_data_version=(region_versions[0] if len(region_versions) == 1 else None),
        data_bundle_version=(data_bundle_versions[0] if len(data_bundle_versions) == 1 else None),
        scoring_model_version=(scoring_versions[0] if len(scoring_versions) == 1 else MODEL_VERSION),
    )

    artifact = {
        "generated_at": _now_iso(),
        "event_backtest_version": "1.0.0",
        "dataset_count": len(datasets),
        "datasets": [
            {
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "source_name": ds.source_name,
                "event_id": ds.event_id,
                "event_name": ds.event_name,
                "event_date": ds.event_date,
                "record_count": len(ds.records),
                "metadata": ds.metadata,
            }
            for ds in datasets
        ],
        "records": records,
        "analysis": analysis,
        "false_low_examples": false_low[:25],
        "false_high_examples": false_high[:25],
        "summary": {
            "record_count": len(records),
            "event_count": len({(r.get("event_id"), r.get("event_name")) for r in records}),
            "dataset_ids": [ds.dataset_id for ds in datasets],
            "adverse_record_count": sum(1 for r in records if _adverse_outcome(int(r.get("outcome_rank", 0)))),
            "high_evidence_count": sum(1 for r in records if r.get("evidence_group") == "high_evidence"),
            "fallback_heavy_count": sum(1 for r in records if r.get("evidence_group") == "fallback_heavy"),
            "out_of_coverage_count": sum(
                1
                for r in records
                if int((r.get("coverage_summary") or {}).get("failed_count") or 0) > 0
            ),
        },
        "model_governance": governance,
        "governance": {
            "product_versions": product_versions,
            "api_versions": api_versions,
            "scoring_model_versions": scoring_versions,
            "ruleset_versions": ruleset_versions,
            "factor_schema_versions": factor_versions,
            "benchmark_pack_versions": benchmark_versions,
            "calibration_versions": calibration_versions,
            "region_data_versions": region_versions,
            "data_bundle_versions": data_bundle_versions,
        },
    }

    out_dir = Path(output_dir or DEFAULT_RESULTS_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"event_backtest_{stamp}.json"
    md_path = out_dir / f"event_backtest_{stamp}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    md_summary = _build_markdown_summary(artifact)
    md_path.write_text(md_summary, encoding="utf-8")
    artifact["artifact_path"] = str(json_path)
    artifact["markdown_summary_path"] = str(md_path)
    return artifact
