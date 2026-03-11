#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_outcome_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"no_damage", "minor_damage", "major_damage", "destroyed"}:
        return text
    if text in {"no_known_damage", "undamaged", "none"}:
        return "no_damage"
    if text in {"minor", "affected"}:
        return "minor_damage"
    if text in {"major", "severe"}:
        return "major_damage"
    if "destroy" in text:
        return "destroyed"
    return "unknown"


def _binary_target(label: str) -> int | None:
    if label in {"major_damage", "destroyed"}:
        return 1
    if label in {"no_damage", "minor_damage"}:
        return 0
    return None


def _coord_key(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    return f"{round(lat, 5):.5f},{round(lon, 5):.5f}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _iter_feature_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("records")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _build_outcome_index(outcome_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    def _set(key: str | None, row: dict[str, Any]) -> None:
        if not key:
            return
        index.setdefault(key, row)

    for row in outcome_records:
        event_id = str(row.get("event_id") or "").strip()
        source_record_id = str(row.get("source_record_id") or "").strip()
        record_id = str(row.get("record_id") or "").strip()
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        coord = _coord_key(lat, lon)

        _set(f"src:{source_record_id}" if source_record_id else None, row)
        _set(f"rec:{record_id}" if record_id else None, row)
        _set(f"event_rec:{event_id}|{record_id}" if event_id and record_id else None, row)
        _set(f"event_src:{event_id}|{source_record_id}" if event_id and source_record_id else None, row)
        _set(f"event_coord:{event_id}|{coord}" if event_id and coord else None, row)
        _set(f"coord:{coord}" if coord else None, row)
    return index


def _find_outcome_match(feature_row: dict[str, Any], outcome_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    event_id = str(feature_row.get("event_id") or "").strip()
    record_id = str(feature_row.get("record_id") or "").strip()
    source_record_id = str(feature_row.get("source_record_id") or "").strip()
    lat = _to_float(feature_row.get("latitude"))
    lon = _to_float(feature_row.get("longitude"))
    coord = _coord_key(lat, lon)

    keys = [
        f"event_src:{event_id}|{source_record_id}" if event_id and source_record_id else None,
        f"event_rec:{event_id}|{record_id}" if event_id and record_id else None,
        f"src:{source_record_id}" if source_record_id else None,
        f"rec:{record_id}" if record_id else None,
        f"event_coord:{event_id}|{coord}" if event_id and coord else None,
        f"coord:{coord}" if coord else None,
    ]
    for key in keys:
        if key and key in outcome_index:
            return outcome_index[key]
    return None


def _extract_scores(row: dict[str, Any]) -> dict[str, float | None]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return {
        "wildfire_risk_score": _to_float(scores.get("wildfire_risk_score") if scores else row.get("wildfire_risk_score")),
        "site_hazard_score": _to_float(scores.get("site_hazard_score") if scores else row.get("site_hazard_score")),
        "home_ignition_vulnerability_score": _to_float(
            scores.get("home_ignition_vulnerability_score")
            if scores
            else row.get("home_ignition_vulnerability_score")
        ),
        "insurance_readiness_score": _to_float(
            scores.get("insurance_readiness_score") if scores else row.get("insurance_readiness_score")
        ),
        "calibrated_damage_likelihood": _to_float(row.get("calibrated_damage_likelihood")),
    }


def build_calibration_dataset(
    *,
    outcome_path: Path | None,
    feature_artifacts: list[Path],
    output_path: Path,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    outcome_records: list[dict[str, Any]] = []
    if outcome_path is not None:
        outcome_payload = _load_json(outcome_path)
        raw = outcome_payload.get("records")
        if isinstance(raw, list):
            outcome_records = [row for row in raw if isinstance(row, dict)]
    outcome_index = _build_outcome_index(outcome_records)

    rows: list[dict[str, Any]] = []
    matched_count = 0
    unmatched_with_label = 0
    skipped_no_label = 0

    for artifact_path in feature_artifacts:
        artifact = _load_json(artifact_path)
        for feature_row in _iter_feature_rows(artifact):
            matched = _find_outcome_match(feature_row, outcome_index) if outcome_index else None
            if matched:
                matched_count += 1

            label_raw = (
                (matched or {}).get("damage_label")
                or feature_row.get("outcome_label")
                or feature_row.get("label")
            )
            label = _normalize_outcome_label(label_raw)
            target = (
                (matched or {}).get("structure_loss_or_major_damage")
                if matched and (matched or {}).get("structure_loss_or_major_damage") is not None
                else _binary_target(label)
            )
            if label == "unknown" and target is None:
                skipped_no_label += 1
                continue
            if not matched:
                unmatched_with_label += 1

            evidence = feature_row.get("evidence_quality_summary")
            evidence = evidence if isinstance(evidence, dict) else {}
            coverage = feature_row.get("coverage_summary")
            coverage = coverage if isinstance(coverage, dict) else {}

            row = {
                "event_id": feature_row.get("event_id") or (matched or {}).get("event_id"),
                "event_name": feature_row.get("event_name") or (matched or {}).get("event_name"),
                "event_date": feature_row.get("event_date") or (matched or {}).get("event_date"),
                "record_id": feature_row.get("record_id") or (matched or {}).get("record_id"),
                "source_record_id": (matched or {}).get("source_record_id"),
                "address_text": feature_row.get("address_text") or (matched or {}).get("address_text"),
                "latitude": _to_float(feature_row.get("latitude") or (matched or {}).get("latitude")),
                "longitude": _to_float(feature_row.get("longitude") or (matched or {}).get("longitude")),
                "outcome_label": label,
                "outcome_rank": feature_row.get("outcome_rank"),
                "structure_loss_or_major_damage": target,
                "label_confidence": (matched or {}).get("label_confidence") or feature_row.get("label_confidence"),
                "source_quality_flags": (matched or {}).get("source_quality_flags") or [],
                "scores": _extract_scores(feature_row),
                "raw_feature_vector": (
                    feature_row.get("raw_feature_vector")
                    if isinstance(feature_row.get("raw_feature_vector"), dict)
                    else {}
                ),
                "transformed_feature_vector": (
                    feature_row.get("transformed_feature_vector")
                    if isinstance(feature_row.get("transformed_feature_vector"), dict)
                    else {}
                ),
                "factor_contribution_breakdown": (
                    feature_row.get("factor_contribution_breakdown")
                    if isinstance(feature_row.get("factor_contribution_breakdown"), dict)
                    else {}
                ),
                "compression_flags": feature_row.get("compression_flags")
                if isinstance(feature_row.get("compression_flags"), list)
                else [],
                "fallback_default_flags": {
                    "fallback_factor_count": int(evidence.get("fallback_factor_count") or 0),
                    "missing_factor_count": int(evidence.get("missing_factor_count") or 0),
                    "inferred_factor_count": int(evidence.get("inferred_factor_count") or 0),
                    "coverage_failed_count": int(coverage.get("failed_count") or 0),
                    "coverage_fallback_count": int(coverage.get("fallback_count") or 0),
                },
                "feature_artifact_path": str(artifact_path),
                "model_governance": feature_row.get("model_governance")
                if isinstance(feature_row.get("model_governance"), dict)
                else {},
            }
            rows.append(row)

    payload = {
        "schema_version": "1.1.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "outcome_source_path": str(outcome_path) if outcome_path else None,
        "feature_artifacts": [str(p) for p in feature_artifacts],
        "row_count": len(rows),
        "matched_outcome_count": matched_count,
        "unmatched_feature_rows_with_label": unmatched_with_label,
        "skipped_rows_without_usable_label": skipped_no_label,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "event_id",
                    "event_name",
                    "event_date",
                    "record_id",
                    "source_record_id",
                    "address_text",
                    "latitude",
                    "longitude",
                    "outcome_label",
                    "structure_loss_or_major_damage",
                    "wildfire_risk_score",
                    "site_hazard_score",
                    "home_ignition_vulnerability_score",
                    "insurance_readiness_score",
                    "calibrated_damage_likelihood",
                    "fallback_factor_count",
                    "missing_factor_count",
                    "inferred_factor_count",
                    "coverage_failed_count",
                    "coverage_fallback_count",
                ],
            )
            writer.writeheader()
            for row in rows:
                scores = row.get("scores") or {}
                fallback = row.get("fallback_default_flags") or {}
                writer.writerow(
                    {
                        "event_id": row.get("event_id"),
                        "event_name": row.get("event_name"),
                        "event_date": row.get("event_date"),
                        "record_id": row.get("record_id"),
                        "source_record_id": row.get("source_record_id"),
                        "address_text": row.get("address_text"),
                        "latitude": row.get("latitude"),
                        "longitude": row.get("longitude"),
                        "outcome_label": row.get("outcome_label"),
                        "structure_loss_or_major_damage": row.get("structure_loss_or_major_damage"),
                        "wildfire_risk_score": scores.get("wildfire_risk_score"),
                        "site_hazard_score": scores.get("site_hazard_score"),
                        "home_ignition_vulnerability_score": scores.get("home_ignition_vulnerability_score"),
                        "insurance_readiness_score": scores.get("insurance_readiness_score"),
                        "calibrated_damage_likelihood": scores.get("calibrated_damage_likelihood"),
                        "fallback_factor_count": fallback.get("fallback_factor_count"),
                        "missing_factor_count": fallback.get("missing_factor_count"),
                        "inferred_factor_count": fallback.get("inferred_factor_count"),
                        "coverage_failed_count": fallback.get("coverage_failed_count"),
                        "coverage_fallback_count": fallback.get("coverage_fallback_count"),
                    }
                )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build calibration/evaluation dataset by joining public structure outcomes to model feature artifacts."
    )
    parser.add_argument(
        "--outcomes",
        default="",
        help="Normalized outcomes JSON from ingest_public_structure_damage.py (optional if feature rows already include labels).",
    )
    parser.add_argument(
        "--feature-artifact",
        action="append",
        required=True,
        help="Event-backtest artifact JSON with scored records. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        default="benchmark/calibration/public_outcome_calibration_dataset.json",
        help="Output dataset JSON path.",
    )
    parser.add_argument("--output-csv", default="", help="Optional flattened output CSV path.")
    args = parser.parse_args()

    payload = build_calibration_dataset(
        outcome_path=Path(args.outcomes).expanduser() if args.outcomes else None,
        feature_artifacts=[Path(p).expanduser() for p in args.feature_artifact],
        output_path=Path(args.output).expanduser(),
        output_csv=(Path(args.output_csv).expanduser() if args.output_csv else None),
    )

    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "row_count": payload.get("row_count"),
                "matched_outcome_count": payload.get("matched_outcome_count"),
                "unmatched_feature_rows_with_label": payload.get("unmatched_feature_rows_with_label"),
                "skipped_rows_without_usable_label": payload.get("skipped_rows_without_usable_label"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
