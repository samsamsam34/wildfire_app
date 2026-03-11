#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("records")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _extract_score(row: dict[str, Any]) -> float | None:
    if isinstance(row.get("scores"), dict):
        score = _to_float((row.get("scores") or {}).get("wildfire_risk_score"))
        if score is not None:
            return score
    if isinstance(row.get("snapshot"), dict):
        score = _to_float(((row.get("snapshot") or {}).get("risk_scores") or {}).get("wildfire_risk_score"))
        if score is not None:
            return score
    return _to_float(row.get("wildfire_risk_score"))


def build_dataset(*, input_path: Path, output_path: Path, adverse_min_rank: int) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input payload must be a JSON object.")

    records = _extract_rows(payload)
    dataset_rows: list[dict[str, Any]] = []
    for row in records:
        score = _extract_score(row)
        rank = _to_int(row.get("outcome_rank"))
        if score is None or rank is None:
            continue
        dataset_rows.append(
            {
                "event_id": row.get("event_id"),
                "record_id": row.get("record_id"),
                "outcome_label": row.get("outcome_label"),
                "outcome_rank": rank,
                "wildfire_risk_score": round(score, 4),
                "adverse": 1 if int(rank) >= int(adverse_min_rank) else 0,
            }
        )

    output_payload = {
        "schema_version": "1.0.0",
        "input_path": str(input_path),
        "source_name": payload.get("dataset_name") or payload.get("source_name") or "public_outcomes",
        "adverse_min_rank": adverse_min_rank,
        "row_count": len(dataset_rows),
        "rows": dataset_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a calibration training dataset from public outcome backtest artifacts."
    )
    parser.add_argument("--input", required=True, help="Path to event backtest results JSON.")
    parser.add_argument(
        "--output",
        default="benchmark/calibration/public_outcome_calibration_dataset.json",
        help="Output calibration dataset path.",
    )
    parser.add_argument("--adverse-min-rank", type=int, default=3, help="Outcome rank threshold for adverse label.")
    args = parser.parse_args()

    payload = build_dataset(
        input_path=Path(args.input).expanduser(),
        output_path=Path(args.output).expanduser(),
        adverse_min_rank=int(args.adverse_min_rank),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "row_count": payload.get("row_count"),
                "source_name": payload.get("source_name"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

