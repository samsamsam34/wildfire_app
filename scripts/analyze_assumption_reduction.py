#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _coerce_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return [row for row in payload["rows"] if isinstance(row, dict)]
        if isinstance(payload.get("assessments"), list):
            return [row for row in payload["assessments"] if isinstance(row, dict)]
        return [payload]
    return []


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "stddev": 0.0}
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "stddev": round(statistics.pstdev(values), 3),
    }


def run(input_path: Path, csv_out: Path | None = None) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = _coerce_records(payload)
    if not records:
        raise ValueError(f"No assessment-like records found in {input_path}")

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        feature_bundle = record.get("feature_bundle_summary")
        if not isinstance(feature_bundle, dict):
            feature_bundle = ((record.get("property_level_context") or {}).get("feature_bundle_summary"))
        feature_bundle = feature_bundle if isinstance(feature_bundle, dict) else {}
        coverage_metrics = feature_bundle.get("coverage_metrics") if isinstance(feature_bundle.get("coverage_metrics"), dict) else {}
        geometry = feature_bundle.get("geometry_provenance") if isinstance(feature_bundle.get("geometry_provenance"), dict) else {}

        fallback_dominance_ratio = float(
            record.get("fallback_dominance_ratio")
            if record.get("fallback_dominance_ratio") is not None
            else coverage_metrics.get("fallback_dominance_ratio") or 0.0
        )
        observed_weight_fraction = float(
            record.get("observed_weight_fraction")
            if record.get("observed_weight_fraction") is not None
            else coverage_metrics.get("observed_weight_fraction") or 0.0
        )
        rows.append(
            {
                "row_id": str(record.get("assessment_id") or record.get("address") or f"row_{idx + 1}"),
                "assessment_mode": str(record.get("assessment_mode") or "unknown"),
                "assessment_output_state": str(record.get("assessment_output_state") or "unknown"),
                "feature_coverage_percent": float(record.get("feature_coverage_percent") or 0.0),
                "fallback_dominance_ratio": fallback_dominance_ratio,
                "observed_weight_fraction": observed_weight_fraction,
                "structure_geometry_quality_score": float(coverage_metrics.get("structure_geometry_quality_score") or 0.0),
                "environmental_layer_coverage_score": float(coverage_metrics.get("environmental_layer_coverage_score") or 0.0),
                "property_specificity_score": float(coverage_metrics.get("property_specificity_score") or 0.0),
                "observed_feature_count": int(coverage_metrics.get("observed_feature_count") or 0),
                "fallback_feature_count": int(coverage_metrics.get("fallback_feature_count") or 0),
                "geometry_basis": str(geometry.get("geometry_basis") or "unknown"),
                "anchor_quality": str(geometry.get("property_anchor_quality") or "unknown"),
                "structure_selection_method": str(geometry.get("structure_selection_method") or "unknown"),
            }
        )

    summary = {
        "record_count": len(rows),
        "feature_coverage_percent": _summary([r["feature_coverage_percent"] for r in rows]),
        "fallback_dominance_ratio": _summary([r["fallback_dominance_ratio"] for r in rows]),
        "observed_weight_fraction": _summary([r["observed_weight_fraction"] for r in rows]),
        "structure_geometry_quality_score": _summary([r["structure_geometry_quality_score"] for r in rows]),
        "environmental_layer_coverage_score": _summary([r["environmental_layer_coverage_score"] for r in rows]),
        "property_specificity_score": _summary([r["property_specificity_score"] for r in rows]),
        "fallback_heavy_count": sum(1 for r in rows if r["fallback_dominance_ratio"] >= 0.7),
        "property_specific_count": sum(1 for r in rows if r["assessment_output_state"] == "property_specific_assessment"),
    }

    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze assumption/fallback reduction from assessment payloads.")
    parser.add_argument("--input", required=True, help="Path to JSON payload (single assessment, rows, or assessments list).")
    parser.add_argument("--csv-out", default=None, help="Optional CSV export path.")
    args = parser.parse_args()

    report = run(
        input_path=Path(args.input),
        csv_out=Path(args.csv_out) if args.csv_out else None,
    )
    print(json.dumps(report["summary"], indent=2))
    if args.csv_out:
        print(f"CSV written to: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
