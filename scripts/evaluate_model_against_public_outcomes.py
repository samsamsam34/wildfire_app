#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_label(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return 1 if float(value) >= 0.5 else 0
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = _mean(values) or 0.0
    return math.sqrt(sum((v - mu) ** 2 for v in values) / float(len(values)))


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def _roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) != len(y_score) or len(y_true) < 3:
        return None
    pos = sum(1 for y in y_true if y == 1)
    neg = sum(1 for y in y_true if y == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = _rank(y_score)
    pos_rank_sum = sum(rank for rank, y in zip(ranks, y_true) if y == 1)
    auc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return max(0.0, min(1.0, auc))


def _confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        elif actual == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _precision_recall(conf: dict[str, int]) -> dict[str, float | None]:
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    prec = tp / float(tp + fp) if (tp + fp) > 0 else None
    rec = tp / float(tp + fn) if (tp + fn) > 0 else None
    return {"precision": prec, "recall": rec}


def _distribution_by_class(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = _to_float(((row.get("scores") or {}).get(score_key)))
        if score is None:
            continue
        label = str(row.get("outcome_label") or "unknown")
        grouped[label].append(score)

    out: dict[str, Any] = {}
    for label, values in grouped.items():
        out[label] = {
            "count": len(values),
            "mean": _mean(values),
            "stddev": _stddev(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return out


def _calibration_table(y_true: list[int], score: list[float], bins: int = 10) -> list[dict[str, Any]]:
    if len(y_true) != len(score) or not y_true:
        return []
    ranked = sorted(zip(score, y_true), key=lambda x: x[0])
    n = len(ranked)
    table: list[dict[str, Any]] = []
    for idx in range(bins):
        start = int((idx / bins) * n)
        end = int(((idx + 1) / bins) * n)
        if end <= start:
            continue
        chunk = ranked[start:end]
        probs = [max(0.0, min(1.0, s / 100.0)) for s, _ in chunk]
        events = [int(y) for _, y in chunk]
        table.append(
            {
                "bin": idx + 1,
                "count": len(chunk),
                "mean_predicted": _mean(probs),
                "observed_rate": _mean([float(y) for y in events]),
                "score_min": min(s for s, _ in chunk),
                "score_max": max(s for s, _ in chunk),
            }
        )
    return table


def _factor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Average factor contribution by confusion class.
    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = str(row.get("_confusion_class") or "unknown")
        factors = row.get("factor_contribution_breakdown")
        if not isinstance(factors, dict):
            continue
        for factor, detail in factors.items():
            if not isinstance(detail, dict):
                continue
            contrib = _to_float(detail.get("contribution"))
            if contrib is None:
                continue
            by_class[label][factor].append(contrib)

    out: dict[str, Any] = {}
    for cls, factor_map in by_class.items():
        ranked = sorted(
            (
                {
                    "factor": factor,
                    "mean_contribution": _mean(values),
                    "stddev_contribution": _stddev(values),
                    "count": len(values),
                }
                for factor, values in factor_map.items()
            ),
            key=lambda row: abs(float(row.get("mean_contribution") or 0.0)),
            reverse=True,
        )
        out[cls] = ranked[:12]
    return out


def evaluate_dataset(
    *,
    dataset_path: Path,
    output_json: Path,
    output_csv: Path | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Calibration dataset must contain a rows array.")
    rows = [row for row in rows if isinstance(row, dict)]

    base_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    risk_score: list[float] = []
    site_score: list[float] = []
    home_score: list[float] = []
    readiness_risk_proxy: list[float] = []
    calibrated_proxy: list[float] = []

    for row in rows:
        label = _to_label(row.get("structure_loss_or_major_damage"))
        if label is None:
            continue
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        wr = _to_float(scores.get("wildfire_risk_score"))
        if wr is None:
            continue

        y_true.append(label)
        risk_score.append(wr)
        site_score.append(_to_float(scores.get("site_hazard_score")) or 0.0)
        home_score.append(_to_float(scores.get("home_ignition_vulnerability_score")) or 0.0)
        readiness = _to_float(scores.get("insurance_readiness_score"))
        readiness_risk_proxy.append((100.0 - readiness) if readiness is not None else 0.0)
        cp = _to_float(scores.get("calibrated_damage_likelihood"))
        calibrated_proxy.append((cp * 100.0) if cp is not None else -1.0)
        base_rows.append(dict(row))

    if len(base_rows) < 5:
        raise ValueError("Need at least 5 labeled rows with wildfire_risk_score to evaluate.")

    ths = thresholds or [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    threshold_metrics: dict[str, Any] = {}
    for th in ths:
        preds = [1 if s >= th else 0 for s in risk_score]
        conf = _confusion(y_true, preds)
        threshold_metrics[str(int(th))] = {
            "threshold": th,
            "confusion_matrix": conf,
            **_precision_recall(conf),
        }

    default_preds = [1 if s >= 70.0 else 0 for s in risk_score]
    default_conf = _confusion(y_true, default_preds)
    labels = ["tp", "fp", "fn", "tn"]
    for row, pred, actual in zip(base_rows, default_preds, y_true):
        if actual == 1 and pred == 1:
            row["_confusion_class"] = "tp"
        elif actual == 0 and pred == 1:
            row["_confusion_class"] = "fp"
        elif actual == 1 and pred == 0:
            row["_confusion_class"] = "fn"
        else:
            row["_confusion_class"] = "tn"

    fallback_stats = {
        "rows_with_any_fallback_flag": sum(
            1
            for row in base_rows
            if int(((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0)) > 0
            or int(((row.get("fallback_default_flags") or {}).get("coverage_fallback_count") or 0)) > 0
        ),
        "mean_fallback_factor_count": _mean(
            [float((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0) for row in base_rows]
        ),
        "mean_missing_factor_count": _mean(
            [float((row.get("fallback_default_flags") or {}).get("missing_factor_count") or 0) for row in base_rows]
        ),
        "mean_fallback_count_by_confusion_class": {
            cls: _mean(
                [
                    float((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0)
                    for row in base_rows
                    if row.get("_confusion_class") == cls
                ]
            )
            for cls in labels
        },
    }

    report = {
        "schema_version": "1.0.0",
        "dataset_path": str(dataset_path),
        "row_count_labeled": len(base_rows),
        "positive_rate": _mean([float(v) for v in y_true]),
        "discrimination_metrics": {
            "wildfire_risk_score_auc": _roc_auc(y_true, risk_score),
            "site_hazard_score_auc": _roc_auc(y_true, site_score),
            "home_ignition_vulnerability_score_auc": _roc_auc(y_true, home_score),
            "readiness_risk_proxy_auc": _roc_auc(y_true, readiness_risk_proxy),
            "calibrated_damage_likelihood_auc": (
                _roc_auc(
                    [y for y, p in zip(y_true, calibrated_proxy) if p >= 0.0],
                    [p for p in calibrated_proxy if p >= 0.0],
                )
                if any(p >= 0.0 for p in calibrated_proxy)
                else None
            ),
        },
        "threshold_metrics_wildfire_risk_score": threshold_metrics,
        "default_threshold_70": {
            "confusion_matrix": default_conf,
            **_precision_recall(default_conf),
        },
        "score_distributions_by_outcome": {
            "wildfire_risk_score": _distribution_by_class(base_rows, "wildfire_risk_score"),
            "site_hazard_score": _distribution_by_class(base_rows, "site_hazard_score"),
            "home_ignition_vulnerability_score": _distribution_by_class(
                base_rows, "home_ignition_vulnerability_score"
            ),
            "insurance_readiness_score": _distribution_by_class(base_rows, "insurance_readiness_score"),
        },
        "calibration_table_wildfire_risk_score": _calibration_table(y_true, risk_score, bins=10),
        "fallback_diagnostics": fallback_stats,
        "factor_contribution_summary_by_confusion_class": _factor_summary(base_rows),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "record_id",
                    "event_id",
                    "outcome_label",
                    "structure_loss_or_major_damage",
                    "wildfire_risk_score",
                    "site_hazard_score",
                    "home_ignition_vulnerability_score",
                    "insurance_readiness_score",
                    "calibrated_damage_likelihood",
                    "confusion_class_default_threshold_70",
                    "fallback_factor_count",
                    "missing_factor_count",
                ],
            )
            writer.writeheader()
            for row in base_rows:
                scores = row.get("scores") or {}
                fallback = row.get("fallback_default_flags") or {}
                writer.writerow(
                    {
                        "record_id": row.get("record_id"),
                        "event_id": row.get("event_id"),
                        "outcome_label": row.get("outcome_label"),
                        "structure_loss_or_major_damage": row.get("structure_loss_or_major_damage"),
                        "wildfire_risk_score": scores.get("wildfire_risk_score"),
                        "site_hazard_score": scores.get("site_hazard_score"),
                        "home_ignition_vulnerability_score": scores.get(
                            "home_ignition_vulnerability_score"
                        ),
                        "insurance_readiness_score": scores.get("insurance_readiness_score"),
                        "calibrated_damage_likelihood": scores.get("calibrated_damage_likelihood"),
                        "confusion_class_default_threshold_70": row.get("_confusion_class"),
                        "fallback_factor_count": fallback.get("fallback_factor_count"),
                        "missing_factor_count": fallback.get("missing_factor_count"),
                    }
                )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate current wildfire model discrimination/calibration against public structure-damage outcomes."
    )
    parser.add_argument(
        "--dataset",
        default="benchmark/calibration/public_outcome_calibration_dataset.json",
        help="Calibration dataset path from build_calibration_dataset.py.",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark/calibration/public_outcome_evaluation.json",
        help="Evaluation report output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional per-record summary CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        default="30,40,50,60,70,80",
        help="Comma-separated wildfire-risk thresholds for PR/confusion summaries.",
    )
    args = parser.parse_args()

    thresholds = [float(tok) for tok in str(args.thresholds).split(",") if tok.strip()]
    report = evaluate_dataset(
        dataset_path=Path(args.dataset).expanduser(),
        output_json=Path(args.output_json).expanduser(),
        output_csv=(Path(args.output_csv).expanduser() if args.output_csv else None),
        thresholds=thresholds,
    )

    print(
        json.dumps(
            {
                "dataset": str(Path(args.dataset).expanduser()),
                "output_json": str(Path(args.output_json).expanduser()),
                "row_count_labeled": report.get("row_count_labeled"),
                "wildfire_risk_score_auc": (
                    (report.get("discrimination_metrics") or {}).get("wildfire_risk_score_auc")
                ),
                "calibrated_damage_likelihood_auc": (
                    (report.get("discrimination_metrics") or {}).get(
                        "calibrated_damage_likelihood_auc"
                    )
                ),
                "default_threshold_70_precision": (
                    (report.get("default_threshold_70") or {}).get("precision")
                ),
                "default_threshold_70_recall": (
                    (report.get("default_threshold_70") or {}).get("recall")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
