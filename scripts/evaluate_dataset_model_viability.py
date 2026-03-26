#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_ROOT = Path("benchmark/public_outcomes/evaluation_dataset")
DEFAULT_OUTPUT_ROOT = Path("benchmark/public_outcomes/model_viability")


def _now_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _resolve_dataset_path(dataset: str | None, dataset_root: Path, dataset_run_id: str | None) -> Path:
    if dataset:
        path = Path(dataset).expanduser()
        if path.exists():
            return path
        raise ValueError(f"Dataset path does not exist: {path}")
    if dataset_run_id:
        candidate = dataset_root / dataset_run_id / "evaluation_dataset.jsonl"
        if candidate.exists():
            return candidate
        raise ValueError(f"Dataset run not found: {candidate}")
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    runs = sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run in runs:
        candidate = run / "evaluation_dataset.jsonl"
        if candidate.exists():
            return candidate
    raise ValueError(f"No evaluation_dataset.jsonl found under {dataset_root}")


def _load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("rows")
            if isinstance(items, list):
                return [row for row in items if isinstance(row, dict)]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported dataset format (expected .jsonl/.json): {path}")


def _target(row: dict[str, Any]) -> int | None:
    outcome = row.get("outcome")
    if not isinstance(outcome, dict):
        return None
    value = outcome.get("structure_loss_or_major_damage")
    return int(value) if value in (0, 1) else None


def _dedupe_independent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, tuple[tuple[float, float, str, str], dict[str, Any]]] = {}
    for row in rows:
        y = _target(row)
        if y is None:
            continue
        property_event_id = str(row.get("property_event_id") or "").strip()
        if not property_event_id:
            continue
        join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
        confidence = _safe_float(join_meta.get("join_confidence_score")) or 0.0
        distance = _safe_float(join_meta.get("distance_m"))
        distance_key = -(distance if distance is not None else 1e9)
        event_id = str((row.get("event") or {}).get("event_id") or "")
        record_id = str((row.get("outcome") or {}).get("record_id") or "")
        score = (float(confidence), float(distance_key), event_id, record_id)
        prev = best.get(property_event_id)
        if prev is None or score > prev[0]:
            best[property_event_id] = (score, row)
    deduped = [payload[1] for payload in best.values()]
    deduped.sort(key=lambda row: str(row.get("property_event_id") or ""))
    return deduped


def _build_feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], np.ndarray]:
    feature_maps: list[dict[str, float]] = []
    all_keys: set[str] = set()
    y: list[int] = []
    for row in rows:
        target = _target(row)
        if target is None:
            continue
        snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
        raw = snapshot.get("raw_feature_vector") if isinstance(snapshot.get("raw_feature_vector"), dict) else {}
        transformed = (
            snapshot.get("transformed_feature_vector")
            if isinstance(snapshot.get("transformed_feature_vector"), dict)
            else {}
        )
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        features: dict[str, float] = {}
        for key, value in {**raw, **transformed}.items():
            numeric = _safe_float(value)
            if numeric is not None:
                features[str(key)] = float(numeric)
        for key in (
            "site_hazard_score",
            "home_ignition_vulnerability_score",
            "insurance_readiness_score",
        ):
            numeric = _safe_float(scores.get(key))
            if numeric is not None:
                features[f"score_{key}"] = float(numeric)
        feature_maps.append(features)
        all_keys.update(features.keys())
        y.append(int(target))
    keys = sorted(all_keys)
    X = np.array([[row.get(key, np.nan) for key in keys] for row in feature_maps], dtype=float)
    y_np = np.array(y, dtype=int)
    keep_cols: list[int] = []
    for idx in range(X.shape[1]):
        column = X[:, idx]
        valid = column[np.isfinite(column)]
        if valid.size < 3:
            continue
        if float(np.std(valid)) < 1e-9:
            continue
        keep_cols.append(idx)
    X = X[:, keep_cols] if keep_cols else np.empty((len(y_np), 0), dtype=float)
    kept_keys = [keys[idx] for idx in keep_cols]
    return X, kept_keys, y_np


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    auc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(max(0.0, min(1.0, auc)))


def _stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = random.Random(seed)
    pos = [idx for idx, val in enumerate(y.tolist()) if int(val) == 1]
    neg = [idx for idx, val in enumerate(y.tolist()) if int(val) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for idx, value in enumerate(pos):
        folds[idx % n_splits].append(int(value))
    for idx, value in enumerate(neg):
        folds[idx % n_splits].append(int(value))
    all_idx = set(range(len(y)))
    return [
        (
            np.array(sorted(all_idx.difference(fold)), dtype=int),
            np.array(sorted(fold), dtype=int),
        )
        for fold in folds
    ]


def _median_impute(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X_train.shape[1] == 0:
        return X_train, X_test
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xt = np.where(np.isnan(X_train), med, X_train)
    Xv = np.where(np.isnan(X_test), med, X_test)
    return Xt, Xv


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X_train.shape[1] == 0:
        return X_train, X_test
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.08,
    iterations: int = 3000,
    l2: float = 0.03,
) -> tuple[np.ndarray, float]:
    n, p = X.shape
    w = np.zeros(p, dtype=float)
    b = 0.0
    y_float = y.astype(float)
    for _ in range(iterations):
        z = np.clip(X @ w + b, -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        err = p1 - y_float
        grad_w = (X.T @ err) / n + (l2 * w)
        grad_b = float(np.mean(err))
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return w, b


def _predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    if X.shape[1] == 0:
        return np.full(X.shape[0], 0.5, dtype=float)
    z = np.clip(X @ weights + bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class _TreeNode:
    __slots__ = ("feature_idx", "threshold", "left", "right", "probability", "is_leaf")

    def __init__(
        self,
        *,
        feature_idx: int | None = None,
        threshold: float | None = None,
        left: "_TreeNode | None" = None,
        right: "_TreeNode | None" = None,
        probability: float = 0.5,
        is_leaf: bool = True,
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.probability = probability
        self.is_leaf = is_leaf


def _gini_from_counts(positive_count: float, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    p = positive_count / float(total_count)
    return 1.0 - (p * p + (1.0 - p) * (1.0 - p))


def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    depth: int = 0,
    max_depth: int = 3,
    min_leaf: int = 2,
) -> _TreeNode:
    n = int(len(y))
    positive = float(np.sum(y))
    probability = positive / max(n, 1)
    if depth >= max_depth or n < (2 * min_leaf) or positive in {0.0, float(n)}:
        return _TreeNode(probability=probability, is_leaf=True)
    parent_gini = _gini_from_counts(positive, n)
    best: tuple[float, int, float, np.ndarray] | None = None
    for feature_idx in range(X.shape[1]):
        column = X[:, feature_idx]
        values = np.unique(column[np.isfinite(column)])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left_mask = column <= float(threshold)
            left_n = int(np.sum(left_mask))
            right_n = n - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            left_positive = float(np.sum(y[left_mask]))
            right_positive = float(np.sum(y[~left_mask]))
            weighted = (left_n / n) * _gini_from_counts(left_positive, left_n) + (
                right_n / n
            ) * _gini_from_counts(right_positive, right_n)
            gain = parent_gini - weighted
            if (
                best is None
                or gain > best[0]
                or (
                    abs(gain - best[0]) < 1e-12
                    and (feature_idx < best[1] or (feature_idx == best[1] and float(threshold) < best[2]))
                )
            ):
                best = (gain, feature_idx, float(threshold), left_mask)
    if best is None or best[0] <= 1e-12:
        return _TreeNode(probability=probability, is_leaf=True)
    _, feature_idx, threshold, left_mask = best
    left_node = _build_tree(X[left_mask], y[left_mask], depth=depth + 1, max_depth=max_depth, min_leaf=min_leaf)
    right_node = _build_tree(
        X[~left_mask],
        y[~left_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
    )
    return _TreeNode(
        feature_idx=feature_idx,
        threshold=threshold,
        left=left_node,
        right=right_node,
        probability=probability,
        is_leaf=False,
    )


def _predict_tree_one(node: _TreeNode, row: np.ndarray) -> float:
    cur = node
    while not cur.is_leaf:
        value = row[cur.feature_idx] if cur.feature_idx is not None else np.nan
        if not np.isfinite(value):
            return float(cur.probability)
        if value <= float(cur.threshold):
            cur = cur.left if cur.left is not None else cur
        else:
            cur = cur.right if cur.right is not None else cur
        if cur.left is cur and cur.right is cur:
            break
    return float(cur.probability)


def _predict_tree(node: _TreeNode, X: np.ndarray) -> np.ndarray:
    return np.array([_predict_tree_one(node, row) for row in X], dtype=float)


def _bootstrap_auc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    random_state: np.random.Generator,
    samples: int,
) -> tuple[float | None, float | None, float | None]:
    vals: list[float] = []
    n = len(y)
    for _ in range(samples):
        idx = random_state.integers(0, n, size=n)
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            continue
        auc = _auc_roc(yb, scores[idx])
        if auc is not None:
            vals.append(float(auc))
    if not vals:
        return None, None, None
    arr = np.array(vals, dtype=float)
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 50.0)),
        float(np.percentile(arr, 97.5)),
    )


def _run_models(
    *,
    rows: list[dict[str, Any]],
    cv_splits: int,
    seed: int,
    random_baseline_draws: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    X, feature_keys, y = _build_feature_matrix(rows)
    if len(y) < 6 or int(np.sum(y == 1)) < 2 or int(np.sum(y == 0)) < 2:
        raise ValueError("Not enough independent labeled rows for viability modeling.")
    scores_full = np.array(
        [_safe_float((row.get("scores") or {}).get("wildfire_risk_score")) or 0.0 for row in rows],
        dtype=float,
    )
    folds = _stratified_kfold_indices(y, n_splits=max(2, cv_splits), seed=seed)
    pred_logit = np.zeros(len(y), dtype=float)
    pred_tree = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in folds:
        X_train, X_test = _median_impute(X[train_idx], X[test_idx])
        X_train, X_test = _standardize(X_train, X_test)
        weights, bias = _fit_logistic(X_train, y[train_idx])
        pred_logit[test_idx] = _predict_logistic(X_test, weights, bias)

        X_train_tree, X_test_tree = _median_impute(X[train_idx], X[test_idx])
        tree = _build_tree(X_train_tree, y[train_idx], max_depth=3, min_leaf=2)
        pred_tree[test_idx] = _predict_tree(tree, X_test_tree)

    rng = np.random.default_rng(seed)
    random_auc_samples: list[float] = []
    for _ in range(max(200, random_baseline_draws)):
        random_probs = rng.random(len(y))
        auc = _auc_roc(y, random_probs)
        if auc is not None:
            random_auc_samples.append(float(auc))
    random_arr = np.array(random_auc_samples, dtype=float)

    auc_full = _auc_roc(y, scores_full)
    auc_logit = _auc_roc(y, pred_logit)
    auc_tree = _auc_roc(y, pred_tree)
    ci_rng = np.random.default_rng(seed + 1)
    ci_full = _bootstrap_auc_ci(y, scores_full, random_state=ci_rng, samples=max(300, bootstrap_samples))
    ci_logit = _bootstrap_auc_ci(y, pred_logit, random_state=ci_rng, samples=max(300, bootstrap_samples))
    ci_tree = _bootstrap_auc_ci(y, pred_tree, random_state=ci_rng, samples=max(300, bootstrap_samples))
    metrics = {
        "full_model_auc": auc_full,
        "logistic_regression_auc_oof": auc_logit,
        "decision_tree_depth3_auc_oof": auc_tree,
        "random_baseline_auc_mean": float(np.mean(random_arr)) if random_arr.size else None,
        "random_baseline_auc_ci95": (
            [float(np.percentile(random_arr, 2.5)), float(np.percentile(random_arr, 97.5))]
            if random_arr.size
            else [None, None]
        ),
        "full_model_auc_ci95_bootstrap": [ci_full[0], ci_full[2]],
        "logistic_regression_auc_ci95_bootstrap": [ci_logit[0], ci_logit[2]],
        "decision_tree_auc_ci95_bootstrap": [ci_tree[0], ci_tree[2]],
    }
    best = max(
        [
            ("full_model", auc_full if auc_full is not None else -1.0),
            ("logistic_regression", auc_logit if auc_logit is not None else -1.0),
            ("decision_tree_depth3", auc_tree if auc_tree is not None else -1.0),
        ],
        key=lambda item: float(item[1]),
    )
    n_rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    best_auc = float(best[1]) if best[1] >= 0.0 else None
    random_mean = metrics["random_baseline_auc_mean"]
    if n_rows < 30:
        verdict = "insufficient_sample"
        usable = False
        reason = "Too few independent labeled rows for stable predictive conclusions."
    elif best_auc is None:
        verdict = "insufficient_label_diversity"
        usable = False
        reason = "Class imbalance or missing labels prevented viable AUC computation."
    elif best_auc < 0.55:
        verdict = "no_clear_signal"
        usable = False
        reason = "Best model AUC is close to random baseline."
    elif n_rows < 100:
        verdict = "limited_signal_small_sample"
        usable = False
        reason = "Some directional signal exists, but sample size is too small for robust model development."
    elif random_mean is not None and best_auc < (float(random_mean) + 0.05):
        verdict = "weak_signal_vs_random"
        usable = False
        reason = "Best model does not beat random baseline by a robust margin."
    else:
        verdict = "usable_directional_signal"
        usable = True
        reason = "Dataset shows directional discrimination beyond random under independent-row evaluation."
    return {
        "sample": {
            "independent_labeled_rows": n_rows,
            "positive_rows": positives,
            "negative_rows": negatives,
            "feature_count_after_variance_filter": int(X.shape[1]),
            "feature_keys": feature_keys,
        },
        "metrics": metrics,
        "best_model": {"name": best[0], "auc": best_auc},
        "verdict": {
            "dataset_usable_for_predictive_modeling": usable,
            "classification": verdict,
            "reason": reason,
            "caveat": (
                "This is directional viability screening against public observed outcomes. "
                "It is not insurer claims-ground-truth validation."
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary(path: Path, report: dict[str, Any], *, run_id: str, dataset_path: Path) -> None:
    sample = report.get("sample") if isinstance(report.get("sample"), dict) else {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    verdict = report.get("verdict") if isinstance(report.get("verdict"), dict) else {}
    lines = [
        "# Dataset Predictive Viability",
        "",
        f"- Run ID: `{run_id}`",
        f"- Dataset: `{dataset_path}`",
        f"- Independent labeled rows: `{sample.get('independent_labeled_rows')}`",
        f"- Positives / negatives: `{sample.get('positive_rows')}` / `{sample.get('negative_rows')}`",
        f"- Feature count after variance filtering: `{sample.get('feature_count_after_variance_filter')}`",
        "",
        "## AUC Comparison",
        "",
        f"- Full model AUC: `{metrics.get('full_model_auc')}`",
        f"- Logistic regression AUC (5-fold OOF): `{metrics.get('logistic_regression_auc_oof')}`",
        f"- Shallow decision tree AUC (5-fold OOF): `{metrics.get('decision_tree_depth3_auc_oof')}`",
        f"- Random baseline AUC mean: `{metrics.get('random_baseline_auc_mean')}`",
        f"- Random baseline AUC 95% range: `{metrics.get('random_baseline_auc_ci95')}`",
        "",
        "## Verdict",
        "",
        f"- Usable for predictive modeling now: `{verdict.get('dataset_usable_for_predictive_modeling')}`",
        f"- Classification: `{verdict.get('classification')}`",
        f"- Reason: {verdict.get('reason')}",
        f"- Caveat: {verdict.get('caveat')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether the current public-outcome dataset has enough signal for predictive modeling."
    )
    parser.add_argument("--evaluation-dataset", default=None, help="Path to evaluation_dataset.jsonl/json.")
    parser.add_argument(
        "--evaluation-dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Root containing evaluation dataset runs.",
    )
    parser.add_argument(
        "--evaluation-dataset-run-id",
        default=None,
        help="Optional dataset run id under --evaluation-dataset-root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for viability artifacts.",
    )
    parser.add_argument("--run-id", default=None, help="Optional deterministic run id.")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-baseline-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_id = str(args.run_id or f"model_viability_{_now_id()}")
    dataset_root = Path(args.evaluation_dataset_root).expanduser()
    dataset_path = _resolve_dataset_path(
        dataset=args.evaluation_dataset,
        dataset_root=dataset_root,
        dataset_run_id=args.evaluation_dataset_run_id,
    )
    run_dir = Path(args.output_root).expanduser() / run_id
    if run_dir.exists() and not bool(args.overwrite):
        raise ValueError(f"Run directory already exists: {run_dir} (use --overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_dataset_rows(dataset_path)
    independent_rows = _dedupe_independent_rows(raw_rows)
    report = _run_models(
        rows=independent_rows,
        cv_splits=max(2, int(args.cv_splits)),
        seed=int(args.seed),
        random_baseline_draws=max(200, int(args.random_baseline_draws)),
        bootstrap_samples=max(300, int(args.bootstrap_samples)),
    )
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "raw_row_count": len(raw_rows),
        "independent_row_count": len(independent_rows),
        "seed": int(args.seed),
        "cv_splits": int(args.cv_splits),
        "random_baseline_draws": int(args.random_baseline_draws),
        "bootstrap_samples": int(args.bootstrap_samples),
        "caveat": (
            "Public-outcome viability checks are directional and do not establish insurer-grade predictive validity."
        ),
    }
    _write_json(run_dir / "model_viability_report.json", report)
    _write_json(run_dir / "manifest.json", manifest)
    _write_summary(run_dir / "summary.md", report, run_id=run_id, dataset_path=dataset_path)

    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "report_path": str(run_dir / "model_viability_report.json"),
        "summary_path": str(run_dir / "summary.md"),
        "manifest_path": str(run_dir / "manifest.json"),
        "verdict": report.get("verdict"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
