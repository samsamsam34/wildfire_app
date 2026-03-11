#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _log_loss(y: list[int], p: list[float]) -> float:
    if not y:
        return 0.0
    eps = 1e-12
    total = 0.0
    for yi, pi in zip(y, p):
        pi = min(1.0 - eps, max(eps, pi))
        total += -(yi * math.log(pi) + (1 - yi) * math.log(1.0 - pi))
    return total / float(len(y))


def _brier(y: list[int], p: list[float]) -> float:
    if not y:
        return 0.0
    total = 0.0
    for yi, pi in zip(y, p):
        total += (float(yi) - float(pi)) ** 2
    return total / float(len(y))


def _fit_logistic(scores: list[float], labels: list[int], *, epochs: int = 2500, lr: float = 0.35) -> tuple[float, float]:
    # Fit on normalized score x in [0,1] for stable training.
    if not scores:
        return -2.0, 3.0
    x = [max(0.0, min(1.0, s / 100.0)) for s in scores]
    y = [int(v) for v in labels]
    b0 = 0.0
    b1 = 1.0
    n = float(len(x))

    for _ in range(epochs):
        grad0 = 0.0
        grad1 = 0.0
        for xi, yi in zip(x, y):
            pred = _sigmoid(b0 + (b1 * xi))
            err = pred - float(yi)
            grad0 += err
            grad1 += err * xi
        b0 -= lr * (grad0 / n)
        b1 -= lr * (grad1 / n)
    return b0, b1


def _piecewise_points(scores: list[float], labels: list[int], bins: int = 10) -> list[list[float]]:
    if not scores:
        return []
    pairs = sorted(zip(scores, labels), key=lambda row: row[0])
    n = len(pairs)
    out: list[list[float]] = []
    for idx in range(bins):
        start = int((idx / bins) * n)
        end = int(((idx + 1) / bins) * n)
        if end <= start:
            continue
        chunk = pairs[start:end]
        score_mean = sum(v for v, _ in chunk) / float(len(chunk))
        rate = sum(lbl for _, lbl in chunk) / float(len(chunk))
        out.append([round(score_mean, 4), round(rate, 6)])
    if len(out) >= 2:
        return out
    # fallback two-point mapping
    low = min(scores)
    high = max(scores)
    base_rate = sum(labels) / float(len(labels))
    return [[round(low, 4), round(base_rate, 6)], [round(high, 4), round(base_rate, 6)]]


def fit_calibration(*, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Calibration dataset must contain a rows array.")

    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Support both legacy flat rows and newer nested calibration dataset rows.
        row_scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        score = _to_float(row.get("wildfire_risk_score"))
        if score is None:
            score = _to_float(row_scores.get("wildfire_risk_score"))
        label = _to_float(row.get("adverse"))
        if label is None:
            label = _to_float(row.get("structure_loss_or_major_damage"))
        if score is None or label is None:
            continue
        scores.append(float(score))
        labels.append(1 if float(label) >= 0.5 else 0)
    if len(scores) < 10:
        raise ValueError("Need at least 10 calibration rows with wildfire_risk_score and adverse labels.")

    intercept, slope = _fit_logistic(scores, labels)
    probs = [_sigmoid(intercept + (slope * max(0.0, min(1.0, s / 100.0)))) for s in scores]
    artifact = {
        "artifact_version": "1.0.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "method": "logistic",
        "parameters": {
            "intercept": round(intercept, 8),
            "slope": round(slope, 8),
            "x_scale": 100.0,
        },
        "dataset": {
            "source_name": payload.get("source_name"),
            "input_path": str(dataset_path),
            "row_count": len(scores),
            "adverse_rate": round(sum(labels) / float(len(labels)), 6),
            "adverse_min_rank": payload.get("adverse_min_rank"),
        },
        "metrics": {
            "log_loss": round(_log_loss(labels, probs), 6),
            "brier_score": round(_brier(labels, probs), 6),
        },
        "piecewise_reference": {
            "method": "piecewise_linear",
            "points": _piecewise_points(scores, labels, bins=10),
        },
        "notes": [
            "Public-outcome calibration is optional and additive.",
            "This calibration does not replace deterministic rules/factor scoring.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a transparent public-outcome wildfire calibration artifact.")
    parser.add_argument(
        "--dataset",
        default="benchmark/calibration/public_outcome_calibration_dataset.json",
        help="Calibration dataset JSON path.",
    )
    parser.add_argument(
        "--output",
        default="config/public_outcome_calibration.json",
        help="Calibration artifact output path.",
    )
    args = parser.parse_args()

    artifact = fit_calibration(
        dataset_path=Path(args.dataset).expanduser(),
        output_path=Path(args.output).expanduser(),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "method": artifact.get("method"),
                "row_count": ((artifact.get("dataset") or {}).get("row_count")),
                "log_loss": ((artifact.get("metrics") or {}).get("log_loss")),
                "brier_score": ((artifact.get("metrics") or {}).get("brier_score")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
