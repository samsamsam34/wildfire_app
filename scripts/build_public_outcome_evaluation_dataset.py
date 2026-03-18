#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "1.0.0"
LEAKAGE_TOKENS = (
    "outcome",
    "damage",
    "destroyed",
    "structure_loss_or_major_damage",
    "adverse_outcome",
    "label",
)


@dataclass(frozen=True)
class OutcomeRecord:
    payload: dict[str, Any]
    event_id: str
    event_name_norm: str
    event_year: int | None
    source_record_id: str
    record_id: str
    parcel_id: str
    address_norm: str
    latitude: float | None
    longitude: float | None


@dataclass(frozen=True)
class FeatureRecord:
    payload: dict[str, Any]
    artifact_path: str
    event_id: str
    event_name_norm: str
    event_year: int | None
    source_record_id: str
    record_id: str
    parcel_id: str
    address_norm: str
    latitude: float | None
    longitude: float | None


def _timestamp_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _deterministic_generated_at(run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return _safe_int(text)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_address(value: Any) -> str:
    return _normalize_text(value)


def _haversine_m(
    lat1: float | None,
    lon1: float | None,
    lat2: float | None,
    lon2: float | None,
) -> float | None:
    if None in (lat1, lon1, lat2, lon2):
        return None
    r = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _iter_outcome_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("records")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _iter_feature_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("records")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _as_outcome_record(row: dict[str, Any]) -> OutcomeRecord:
    event_id = str(row.get("event_id") or "").strip()
    event_name = str(row.get("event_name") or "").strip()
    event_date = row.get("event_date")
    event_year = _parse_year(row.get("event_year")) or _parse_year(event_date)
    source_record_id = str(row.get("source_record_id") or "").strip()
    record_id = str(row.get("record_id") or "").strip()
    parcel_id = str(row.get("parcel_identifier") or row.get("parcel_id") or "").strip()
    address_norm = _normalize_address(row.get("address_text") or row.get("address") or "")
    return OutcomeRecord(
        payload=row,
        event_id=event_id,
        event_name_norm=_normalize_text(event_name),
        event_year=event_year,
        source_record_id=source_record_id,
        record_id=record_id,
        parcel_id=parcel_id,
        address_norm=address_norm,
        latitude=_safe_float(row.get("latitude")),
        longitude=_safe_float(row.get("longitude")),
    )


def _as_feature_record(row: dict[str, Any], artifact_path: str) -> FeatureRecord:
    event_id = str(row.get("event_id") or "").strip()
    event_name = str(row.get("event_name") or "").strip()
    event_date = row.get("event_date")
    event_year = _parse_year(row.get("event_year")) or _parse_year(event_date)
    source_record_id = str(row.get("source_record_id") or "").strip()
    record_id = str(row.get("record_id") or "").strip()
    parcel_id = str(row.get("parcel_identifier") or row.get("parcel_id") or "").strip()
    address_norm = _normalize_address(row.get("address_text") or row.get("address") or "")
    return FeatureRecord(
        payload=row,
        artifact_path=artifact_path,
        event_id=event_id,
        event_name_norm=_normalize_text(event_name),
        event_year=event_year,
        source_record_id=source_record_id,
        record_id=record_id,
        parcel_id=parcel_id,
        address_norm=address_norm,
        latitude=_safe_float(row.get("latitude")),
        longitude=_safe_float(row.get("longitude")),
    )


def _derive_severity_and_binary(outcome: dict[str, Any]) -> tuple[str, int | None]:
    severity = str(outcome.get("damage_severity_class") or "").strip().lower()
    if not severity:
        label = str(outcome.get("damage_label") or "").strip().lower()
        if label in {"no_damage", "none"}:
            severity = "none"
        elif label == "minor_damage":
            severity = "minor"
        elif label == "major_damage":
            severity = "major"
        elif label == "destroyed":
            severity = "destroyed"
        else:
            severity = "unknown"
    binary_raw = outcome.get("structure_loss_or_major_damage")
    if binary_raw in (0, 1):
        binary = int(binary_raw)
    elif severity in {"major", "destroyed"}:
        binary = 1
    elif severity in {"none", "minor"}:
        binary = 0
    else:
        binary = None
    return severity, binary


def _event_year_consistent(feature: FeatureRecord, outcome: OutcomeRecord) -> bool:
    if feature.event_year is None or outcome.event_year is None:
        return True
    return abs(int(feature.event_year) - int(outcome.event_year)) <= 1


def _join_confidence_for_distance(distance_m: float | None, max_distance_m: float) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= 0.0:
        return 0.75
    ratio = min(1.0, distance_m / max_distance_m)
    return max(0.3, 0.75 - (0.35 * ratio))


def _pick_nearest(feature: FeatureRecord, candidates: list[OutcomeRecord]) -> tuple[OutcomeRecord | None, float | None]:
    best: OutcomeRecord | None = None
    best_distance: float | None = None
    for row in candidates:
        d = _haversine_m(feature.latitude, feature.longitude, row.latitude, row.longitude)
        if d is None:
            continue
        if best is None or (best_distance is not None and d < best_distance) or best_distance is None:
            best = row
            best_distance = d
    return best, best_distance


def _assess_join_tier(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.7:
        return "moderate"
    return "low"


def _detect_leakage_flags(feature_row: dict[str, Any], event_date: Any) -> list[str]:
    flags: list[str] = []
    for vector_key in ("raw_feature_vector", "transformed_feature_vector", "factor_contribution_breakdown"):
        payload = feature_row.get(vector_key)
        if not isinstance(payload, dict):
            continue
        lowered_keys = {str(key).lower() for key in payload.keys()}
        for token in LEAKAGE_TOKENS:
            if any(token in key for key in lowered_keys):
                flags.append(f"potential_outcome_leakage_token_in_{vector_key}")
                break

    event_year = _parse_year(event_date)
    for key in ("assessment_generated_at", "assessment_timestamp", "score_generated_at", "generated_at"):
        value = feature_row.get(key)
        if value is None:
            continue
        stamp_year = _parse_year(value)
        if event_year is not None and stamp_year is not None and stamp_year > event_year:
            flags.append("post_event_assessment_timestamp_possible")
            break
    return sorted(set(flags))


def _build_indexes(outcomes: list[OutcomeRecord]) -> dict[str, Any]:
    by_source_record: dict[str, list[OutcomeRecord]] = {}
    by_event_record: dict[str, list[OutcomeRecord]] = {}
    by_parcel_event: dict[str, list[OutcomeRecord]] = {}
    by_address_event: dict[str, list[OutcomeRecord]] = {}
    by_event: dict[str, list[OutcomeRecord]] = {}
    by_event_name_year: dict[str, list[OutcomeRecord]] = {}
    for row in outcomes:
        if row.source_record_id:
            by_source_record.setdefault(row.source_record_id, []).append(row)
        if row.event_id and row.record_id:
            by_event_record.setdefault(f"{row.event_id}|{row.record_id}", []).append(row)
        if row.event_id and row.parcel_id:
            by_parcel_event.setdefault(f"{row.event_id}|{row.parcel_id}", []).append(row)
        if row.event_id and row.address_norm:
            by_address_event.setdefault(f"{row.event_id}|{row.address_norm}", []).append(row)
        if row.event_id:
            by_event.setdefault(row.event_id, []).append(row)
        if row.event_name_norm and row.event_year is not None:
            by_event_name_year.setdefault(f"{row.event_name_norm}|{row.event_year}", []).append(row)
    return {
        "by_source_record": by_source_record,
        "by_event_record": by_event_record,
        "by_parcel_event": by_parcel_event,
        "by_address_event": by_address_event,
        "by_event": by_event,
        "by_event_name_year": by_event_name_year,
        "all": outcomes,
    }


def _choose_outcome(
    feature: FeatureRecord,
    indexes: dict[str, Any],
    *,
    max_distance_m: float,
) -> tuple[OutcomeRecord | None, dict[str, Any]]:
    # 1) Exact parcel + event.
    if feature.event_id and feature.parcel_id:
        rows = indexes["by_parcel_event"].get(f"{feature.event_id}|{feature.parcel_id}", [])
        if rows:
            return rows[0], {
                "join_method": "exact_parcel_event",
                "join_confidence_score": 0.98,
                "join_confidence_tier": "high",
                "join_distance_m": 0.0,
            }

    # 2) Exact source record id.
    if feature.source_record_id:
        rows = indexes["by_source_record"].get(feature.source_record_id, [])
        if rows:
            chosen = rows[0]
            score = 0.97 if _event_year_consistent(feature, chosen) else 0.82
            return chosen, {
                "join_method": "exact_source_record_id",
                "join_confidence_score": score,
                "join_confidence_tier": _assess_join_tier(score),
                "join_distance_m": _haversine_m(feature.latitude, feature.longitude, chosen.latitude, chosen.longitude),
            }

    # 3) Exact event+record id.
    if feature.event_id and feature.record_id:
        rows = indexes["by_event_record"].get(f"{feature.event_id}|{feature.record_id}", [])
        if rows:
            chosen = rows[0]
            return chosen, {
                "join_method": "exact_event_record_id",
                "join_confidence_score": 0.96,
                "join_confidence_tier": "high",
                "join_distance_m": _haversine_m(feature.latitude, feature.longitude, chosen.latitude, chosen.longitude),
            }

    # 4) Event+address.
    if feature.event_id and feature.address_norm:
        rows = indexes["by_address_event"].get(f"{feature.event_id}|{feature.address_norm}", [])
        if rows:
            chosen = rows[0]
            return chosen, {
                "join_method": "exact_event_address",
                "join_confidence_score": 0.90,
                "join_confidence_tier": "high",
                "join_distance_m": _haversine_m(feature.latitude, feature.longitude, chosen.latitude, chosen.longitude),
            }

    # 5) Nearest within event.
    if feature.event_id:
        candidates = indexes["by_event"].get(feature.event_id, [])
        chosen, distance_m = _pick_nearest(feature, candidates)
        if chosen is not None and distance_m is not None and distance_m <= max_distance_m:
            score = _join_confidence_for_distance(distance_m, max_distance_m)
            return chosen, {
                "join_method": "nearest_event_coordinates",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _assess_join_tier(score),
                "join_distance_m": round(distance_m, 2),
            }

    # 6) Nearest by event name/year.
    if feature.event_name_norm and feature.event_year is not None:
        candidates = indexes["by_event_name_year"].get(f"{feature.event_name_norm}|{feature.event_year}", [])
        chosen, distance_m = _pick_nearest(feature, candidates)
        if chosen is not None and distance_m is not None and distance_m <= max_distance_m:
            score = max(0.55, _join_confidence_for_distance(distance_m, max_distance_m) - 0.05)
            return chosen, {
                "join_method": "nearest_event_name_year_coordinates",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _assess_join_tier(score),
                "join_distance_m": round(distance_m, 2),
            }

    # 7) Global nearest as low confidence fallback.
    chosen, distance_m = _pick_nearest(feature, indexes["all"])
    if chosen is not None and distance_m is not None and distance_m <= max_distance_m:
        score = max(0.30, _join_confidence_for_distance(distance_m, max_distance_m) - 0.20)
        return chosen, {
            "join_method": "nearest_global_coordinates",
            "join_confidence_score": round(score, 4),
            "join_confidence_tier": _assess_join_tier(score),
            "join_distance_m": round(distance_m, 2),
        }
    return None, {
        "join_method": "unmatched",
        "join_confidence_score": 0.0,
        "join_confidence_tier": "low",
        "join_distance_m": None,
    }


def _load_outcomes(path: Path) -> list[OutcomeRecord]:
    payload = _load_json(path)
    rows = _iter_outcome_rows(payload)
    return [_as_outcome_record(row) for row in rows]


def _load_feature_records(paths: list[Path]) -> tuple[list[FeatureRecord], list[dict[str, Any]]]:
    records: list[FeatureRecord] = []
    missing_artifacts: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            missing_artifacts.append({"feature_artifact_path": str(p), "reason": "missing_feature_artifact"})
            continue
        payload = _load_json(p)
        rows = _iter_feature_rows(payload)
        for row in rows:
            records.append(_as_feature_record(row, str(p)))
    return records, missing_artifacts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "property_event_id",
        "event_id",
        "event_name",
        "event_date",
        "record_id",
        "source_record_id",
        "join_method",
        "join_confidence_score",
        "join_confidence_tier",
        "join_distance_m",
        "wildfire_risk_score",
        "site_hazard_score",
        "home_ignition_vulnerability_score",
        "insurance_readiness_score",
        "confidence_tier",
        "confidence_score",
        "evidence_quality_tier",
        "damage_label",
        "damage_severity_class",
        "structure_loss_or_major_damage",
        "adverse_outcome_binary",
        "leakage_flags",
        "caveat_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "property_event_id": row.get("property_event_id"),
                "event_id": row.get("event", {}).get("event_id"),
                "event_name": row.get("event", {}).get("event_name"),
                "event_date": row.get("event", {}).get("event_date"),
                "record_id": row.get("feature", {}).get("record_id"),
                "source_record_id": row.get("feature", {}).get("source_record_id"),
                "join_method": row.get("join_metadata", {}).get("join_method"),
                "join_confidence_score": row.get("join_metadata", {}).get("join_confidence_score"),
                "join_confidence_tier": row.get("join_metadata", {}).get("join_confidence_tier"),
                "join_distance_m": row.get("join_metadata", {}).get("join_distance_m"),
                "wildfire_risk_score": row.get("scores", {}).get("wildfire_risk_score"),
                "site_hazard_score": row.get("scores", {}).get("site_hazard_score"),
                "home_ignition_vulnerability_score": row.get("scores", {}).get("home_ignition_vulnerability_score"),
                "insurance_readiness_score": row.get("scores", {}).get("insurance_readiness_score"),
                "confidence_tier": row.get("confidence", {}).get("confidence_tier"),
                "confidence_score": row.get("confidence", {}).get("confidence_score"),
                "evidence_quality_tier": row.get("evidence", {}).get("evidence_quality_tier"),
                "damage_label": row.get("outcome", {}).get("damage_label"),
                "damage_severity_class": row.get("outcome", {}).get("damage_severity_class"),
                "structure_loss_or_major_damage": row.get("outcome", {}).get("structure_loss_or_major_damage"),
                "adverse_outcome_binary": row.get("outcome", {}).get("adverse_outcome_binary"),
                "leakage_flags": ";".join(row.get("leakage_flags") or []),
                "caveat_flags": ";".join(row.get("caveat_flags") or []),
            }
            writer.writerow(out)


def _build_markdown_summary(
    *,
    run_id: str,
    generated_at: str,
    quality: dict[str, Any],
) -> str:
    lines = [
        "# Public Outcome Evaluation Dataset",
        "",
        "- Joined labeled dataset for directional validation/calibration against public observed outcomes.",
        "- Not insurer claims validation and not a production-model behavior change.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Outcomes loaded: `{quality.get('total_outcomes_loaded')}`",
        f"- Feature rows loaded: `{quality.get('total_feature_rows_loaded')}`",
        f"- Joined rows: `{quality.get('total_joined_records')}`",
        f"- Join rate: `{quality.get('join_rate')}`",
        "",
        "## Join Quality",
        f"- Join method counts: `{quality.get('join_method_counts')}`",
        f"- Join confidence tier counts: `{quality.get('join_confidence_tier_counts')}`",
        f"- Low-confidence joins: `{quality.get('low_confidence_join_count')}`",
        "",
        "## Coverage",
        f"- By event joined counts: `{quality.get('by_event_join_counts')}`",
        f"- By label joined counts: `{quality.get('by_label_join_counts')}`",
        f"- Excluded rows: `{quality.get('excluded_row_count')}`",
        "",
        "## Leakage Guardrails",
        f"- Leakage warning count: `{len(quality.get('leakage_warnings') or [])}`",
    ]
    for warning in (quality.get("leakage_warnings") or [])[:10]:
        lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Caveats",
            "- Public outcomes are heterogeneous and incomplete.",
            "- Low-confidence joins should be reviewed before calibration fitting.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_public_outcome_evaluation_dataset(
    *,
    outcomes_path: Path,
    feature_artifacts: list[Path],
    output_root: Path = Path("benchmark/public_outcomes/evaluation_dataset"),
    run_id: str | None = None,
    max_distance_m: float = 120.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_token = str(run_id or _timestamp_id())
    generated_at = _deterministic_generated_at(run_id)
    run_dir = Path(output_root).expanduser() / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    outcomes = _load_outcomes(Path(outcomes_path).expanduser())
    if not outcomes:
        raise ValueError("No normalized outcome rows available.")
    feature_rows, missing_feature_artifacts = _load_feature_records(feature_artifacts)
    if not feature_rows:
        raise ValueError("No feature rows were loaded from provided feature artifacts.")

    indexes = _build_indexes(outcomes)
    joined_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = list(missing_feature_artifacts)
    leakage_warnings: list[str] = []
    join_method_counts: dict[str, int] = {}
    join_tier_counts: dict[str, int] = {}
    by_event_counts: dict[str, int] = {}
    by_label_counts: dict[str, int] = {}
    low_confidence_join_count = 0

    for feature in sorted(feature_rows, key=lambda row: (row.artifact_path, row.event_id, row.record_id)):
        matched, join_meta = _choose_outcome(feature, indexes, max_distance_m=float(max_distance_m))
        method = str(join_meta.get("join_method") or "unmatched")
        if matched is None:
            excluded_rows.append(
                {
                    "feature_artifact_path": feature.artifact_path,
                    "record_id": feature.record_id,
                    "event_id": feature.event_id,
                    "reason": "no_outcome_match_within_constraints",
                }
            )
            continue

        join_method_counts[method] = join_method_counts.get(method, 0) + 1
        tier = str(join_meta.get("join_confidence_tier") or "low")
        join_tier_counts[tier] = join_tier_counts.get(tier, 0) + 1
        if tier == "low":
            low_confidence_join_count += 1

        severity, binary = _derive_severity_and_binary(matched.payload)
        leak_flags = _detect_leakage_flags(feature.payload, feature.payload.get("event_date"))
        if leak_flags:
            leakage_warnings.append(
                f"{feature.event_id or 'unknown_event'}/{feature.record_id or 'unknown_record'}: {','.join(leak_flags)}"
            )

        caveat_flags: list[str] = []
        if not _event_year_consistent(feature, matched):
            caveat_flags.append("event_year_mismatch")
        distance_m = _safe_float(join_meta.get("join_distance_m"))
        if distance_m is not None and distance_m > float(max_distance_m) * 0.6:
            caveat_flags.append("high_join_distance")
        if tier == "low":
            caveat_flags.append("low_confidence_join")
        if leak_flags:
            caveat_flags.append("leakage_warning_present")

        feature_scores = feature.payload.get("scores") if isinstance(feature.payload.get("scores"), dict) else {}
        wildfire_risk = _safe_float(feature_scores.get("wildfire_risk_score") if feature_scores else feature.payload.get("wildfire_risk_score"))
        site_hazard = _safe_float(feature_scores.get("site_hazard_score") if feature_scores else feature.payload.get("site_hazard_score"))
        home_vuln = _safe_float(
            feature_scores.get("home_ignition_vulnerability_score")
            if feature_scores
            else feature.payload.get("home_ignition_vulnerability_score")
        )
        readiness = _safe_float(feature_scores.get("insurance_readiness_score") if feature_scores else feature.payload.get("insurance_readiness_score"))

        confidence_payload = feature.payload.get("confidence") if isinstance(feature.payload.get("confidence"), dict) else {}
        evidence_summary = feature.payload.get("evidence_quality_summary") if isinstance(feature.payload.get("evidence_quality_summary"), dict) else {}
        coverage_summary = feature.payload.get("coverage_summary") if isinstance(feature.payload.get("coverage_summary"), dict) else {}

        joined = {
            "property_event_id": f"{feature.event_id or 'unknown_event'}::{feature.record_id or feature.source_record_id or 'unknown_record'}",
            "event": {
                "event_id": feature.payload.get("event_id") or matched.payload.get("event_id"),
                "event_name": feature.payload.get("event_name") or matched.payload.get("event_name"),
                "event_date": feature.payload.get("event_date") or matched.payload.get("event_date"),
                "event_year": feature.event_year or matched.event_year,
            },
            "feature": {
                "record_id": feature.payload.get("record_id"),
                "source_record_id": feature.payload.get("source_record_id"),
                "source_name": feature.payload.get("source_name"),
                "feature_artifact_path": feature.artifact_path,
                "address_text": feature.payload.get("address_text"),
                "latitude": feature.latitude,
                "longitude": feature.longitude,
                "parcel_identifier": feature.payload.get("parcel_identifier") or feature.payload.get("parcel_id"),
            },
            "outcome": {
                "record_id": matched.payload.get("record_id"),
                "source_record_id": matched.payload.get("source_record_id"),
                "source_name": matched.payload.get("source_name"),
                "source_path": matched.payload.get("source_path"),
                "address_text": matched.payload.get("address_text"),
                "latitude": matched.latitude,
                "longitude": matched.longitude,
                "damage_label": matched.payload.get("damage_label"),
                "damage_severity_class": severity,
                "structure_loss_or_major_damage": binary,
                "adverse_outcome_binary": (bool(binary) if binary in (0, 1) else None),
                "source_native_label": matched.payload.get("source_native_label") or matched.payload.get("raw_damage_label"),
            },
            "scores": {
                "wildfire_risk_score": wildfire_risk,
                "site_hazard_score": site_hazard,
                "home_ignition_vulnerability_score": home_vuln,
                "insurance_readiness_score": readiness,
            },
            "confidence": {
                "confidence_tier": confidence_payload.get("confidence_tier") or feature.payload.get("confidence_tier"),
                "confidence_score": _safe_float(confidence_payload.get("confidence_score") or feature.payload.get("confidence_score")),
                "use_restriction": confidence_payload.get("use_restriction") or feature.payload.get("use_restriction"),
            },
            "evidence": {
                "evidence_quality_tier": evidence_summary.get("evidence_tier"),
                "evidence_quality_summary": evidence_summary,
                "coverage_summary": coverage_summary,
            },
            "feature_snapshot": {
                "raw_feature_vector": (
                    feature.payload.get("raw_feature_vector")
                    if isinstance(feature.payload.get("raw_feature_vector"), dict)
                    else {}
                ),
                "transformed_feature_vector": (
                    feature.payload.get("transformed_feature_vector")
                    if isinstance(feature.payload.get("transformed_feature_vector"), dict)
                    else {}
                ),
                "factor_contribution_breakdown": (
                    feature.payload.get("factor_contribution_breakdown")
                    if isinstance(feature.payload.get("factor_contribution_breakdown"), dict)
                    else {}
                ),
                "compression_flags": (
                    feature.payload.get("compression_flags")
                    if isinstance(feature.payload.get("compression_flags"), list)
                    else []
                ),
            },
            "join_metadata": {
                **join_meta,
                "max_distance_m": float(max_distance_m),
                "event_year_consistent": _event_year_consistent(feature, matched),
            },
            "provenance": {
                "model_governance": (
                    feature.payload.get("model_governance")
                    if isinstance(feature.payload.get("model_governance"), dict)
                    else {}
                ),
                "outcome_provenance_notes": list(matched.payload.get("provenance_notes") or matched.payload.get("source_quality_flags") or []),
            },
            "leakage_flags": leak_flags,
            "caveat_flags": sorted(set(caveat_flags)),
        }
        joined_rows.append(joined)

        event_key = str(joined["event"]["event_id"] or "unknown_event")
        by_event_counts[event_key] = by_event_counts.get(event_key, 0) + 1
        label_key = str(joined["outcome"]["damage_label"] or "unknown")
        by_label_counts[label_key] = by_label_counts.get(label_key, 0) + 1

    joined_rows.sort(key=lambda row: str(row.get("property_event_id") or ""))
    join_rate = round(len(joined_rows) / float(len(feature_rows)), 4) if feature_rows else 0.0

    join_quality = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "total_outcomes_loaded": len(outcomes),
        "total_feature_rows_loaded": len(feature_rows),
        "total_joined_records": len(joined_rows),
        "join_rate": join_rate,
        "join_method_counts": dict(sorted(join_method_counts.items())),
        "join_confidence_tier_counts": dict(sorted(join_tier_counts.items())),
        "low_confidence_join_count": low_confidence_join_count,
        "by_event_join_counts": dict(sorted(by_event_counts.items())),
        "by_label_join_counts": dict(sorted(by_label_counts.items())),
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows[:200],
        "leakage_warnings": sorted(set(leakage_warnings)),
    }

    jsonl_path = run_dir / "evaluation_dataset.jsonl"
    _write_jsonl(jsonl_path, joined_rows)
    csv_path = run_dir / "evaluation_dataset.csv"
    _write_csv(csv_path, joined_rows)
    join_report_path = run_dir / "join_quality_report.json"
    join_report_path.write_text(json.dumps(join_quality, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = run_dir / "summary.md"
    summary_path.write_text(
        _build_markdown_summary(run_id=run_token, generated_at=generated_at, quality=join_quality),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_token,
        "generated_at": generated_at,
        "inputs": {
            "normalized_outcomes_path": str(Path(outcomes_path).expanduser()),
            "feature_artifacts": [str(Path(path).expanduser()) for path in feature_artifacts],
            "max_distance_m": float(max_distance_m),
        },
        "artifacts": {
            "evaluation_dataset_jsonl": str(jsonl_path),
            "evaluation_dataset_csv": str(csv_path),
            "join_quality_report_json": str(join_report_path),
            "summary_markdown": str(summary_path),
        },
        "summary": {
            "outcomes_loaded": len(outcomes),
            "feature_rows_loaded": len(feature_rows),
            "joined_records": len(joined_rows),
            "join_rate": join_rate,
            "excluded_rows": len(excluded_rows),
            "low_confidence_join_count": low_confidence_join_count,
            "leakage_warning_count": len(set(leakage_warnings)),
        },
        "caveat": (
            "Public observed outcomes are directional validation signals and not equivalent to insurer claims truth."
        ),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_id": run_token,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "join_quality_report_path": str(join_report_path),
        "joined_record_count": len(joined_rows),
        "excluded_row_count": len(excluded_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a joined labeled public-outcome evaluation dataset by matching historical model "
            "feature artifacts with normalized public observed wildfire outcomes."
        )
    )
    parser.add_argument(
        "--outcomes",
        required=True,
        help="Path to normalized public outcomes JSON (records array expected).",
    )
    parser.add_argument(
        "--feature-artifact",
        action="append",
        default=[],
        help="Event-backtest feature artifact JSON path. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/public_outcomes/evaluation_dataset",
        help="Root output directory for timestamped run bundles.",
    )
    parser.add_argument("--run-id", default="", help="Optional fixed run id for deterministic output naming.")
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=120.0,
        help="Maximum nearest-neighbor join distance in meters for geospatial fallback joins.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory.")
    args = parser.parse_args()

    feature_artifacts = [Path(token).expanduser() for token in (args.feature_artifact or []) if str(token).strip()]
    if not feature_artifacts:
        raise ValueError("At least one --feature-artifact path is required.")
    result = build_public_outcome_evaluation_dataset(
        outcomes_path=Path(args.outcomes).expanduser(),
        feature_artifacts=feature_artifacts,
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        max_distance_m=float(args.max_distance_m),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
