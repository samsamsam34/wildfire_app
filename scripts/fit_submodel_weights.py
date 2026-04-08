#!/usr/bin/env python3
"""Fit submodel contribution weights from public outcome evaluation data.

Reads pre-scored rows from the evaluation dataset, extracts each submodel's
clamped score, and fits a regularised logistic regression against the binary
structure-loss outcome label.  Fitted coefficients are normalised into a
proposed weight vector and compared against the current weights in
scoring_parameters.yaml.

Exit codes
----------
0  Success (weights updated when AUC gate is met, or --dry-run requested).
1  Stop condition triggered (gate not met, high variance, or data error).

Usage
-----
    python scripts/fit_submodel_weights.py
    python scripts/fit_submodel_weights.py --dry-run
    python scripts/fit_submodel_weights.py --dataset path/to/eval.jsonl --folds 5
    python scripts/fit_submodel_weights.py --allow-row-level-cv
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVAL_PATH = (
    REPO_ROOT
    / "benchmark/public_outcomes/evaluation_dataset"
    / "all_diag_eval_ds_20260325T1909"
    / "evaluation_dataset.jsonl"
)
SCORING_CONFIG_PATH = REPO_ROOT / "config" / "scoring_parameters.yaml"

SUBMODELS: List[str] = [
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "slope_topography_risk",
    "ember_exposure_risk",
    "flame_contact_risk",
    "historic_fire_risk",
    "structure_vulnerability_risk",
    "defensible_space_risk",
]

AUC_IMPROVEMENT_GATE = 0.03   # minimum AUC gain to update weights
CV_VARIANCE_GATE = 0.08       # maximum fold-AUC std before flagging
MIN_SUBMODEL_WEIGHT = 0.02    # floor to preserve every submodel's presence
LOGREG_C = 1.0                # inverse L2 regularisation strength
LOGREG_LR = 0.05              # gradient-descent learning rate
LOGREG_ITERS = 4000           # gradient-descent iterations


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[dict]:
    """Return usable rows that carry a binary outcome label."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    usable = [
        r for r in rows
        if r.get("evaluation", {}).get("row_usable")
        and r.get("outcome", {}).get("structure_loss_or_major_damage") is not None
    ]
    return usable


def extract_features(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) from usable evaluation rows.

    X shape: (n_rows, n_submodels), values normalised to [0, 1].
    y shape: (n_rows,), binary 0/1.
    Rows where any submodel score is absent are dropped.
    """
    X, y, _groups = extract_features_with_groups(rows)
    return X, y


def _extract_year(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    m = re.search(r"(19|20)\d{2}", text)
    if not m:
        return "unknown"
    return m.group(0)


def _row_group_key(row: dict, row_idx: int) -> str:
    event_id = (
        str(row.get("property_event_id") or "").strip()
        or str(row.get("event_id") or "").strip()
        or str((row.get("event") or {}).get("event_id") if isinstance(row.get("event"), dict) else "").strip()
    )
    region_id = (
        str(row.get("region_id") or "").strip()
        or str(row.get("resolved_region_id") or "").strip()
        or str((row.get("feature_snapshot") or {}).get("region_id") if isinstance(row.get("feature_snapshot"), dict) else "").strip()
        or str((row.get("property_level_context") or {}).get("region_id") if isinstance(row.get("property_level_context"), dict) else "").strip()
        or str((row.get("model_governance") or {}).get("region_data_version") if isinstance(row.get("model_governance"), dict) else "").strip()
    )
    event_date = (
        row.get("event_date")
        or ((row.get("event") or {}).get("event_date") if isinstance(row.get("event"), dict) else None)
        or ((row.get("outcome") or {}).get("event_date") if isinstance(row.get("outcome"), dict) else None)
    )
    year = _extract_year(event_date)
    if not event_id:
        fallback_id = str(row.get("record_id") or "").strip() or str(row.get("source_record_id") or "").strip()
        event_id = fallback_id or f"row_{row_idx}"
    return f"event={event_id}|region={region_id or 'unknown'}|year={year}"


def extract_features_with_groups(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, groups) from usable evaluation rows.

    groups represent event/region/year holdout buckets used to avoid
    optimistic leakage across rows from the same wildfire context.
    """
    X_list, y_list, groups = [], [], []
    for row in rows:
        fcd = row.get("feature_snapshot", {}).get("factor_contribution_breakdown", {})
        scores = [fcd.get(s, {}).get("clamped_submodel_score") for s in SUBMODELS]
        if any(v is None for v in scores):
            continue
        X_list.append([float(v) / 100.0 for v in scores])
        y_list.append(int(row["outcome"]["structure_loss_or_major_damage"]))
        groups.append(_row_group_key(row, len(groups)))
    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64),
        np.array(groups, dtype=object),
    )


# ---------------------------------------------------------------------------
# Logistic regression (pure numpy, gradient descent with L2)
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = LOGREG_C,
    lr: float = LOGREG_LR,
    n_iter: int = LOGREG_ITERS,
) -> Tuple[np.ndarray, float]:
    """Fit logistic regression; return (weights, intercept).

    Optimises binary cross-entropy + L2 penalty (lambda = 1/C) using
    full-batch gradient descent.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    lam = 1.0 / max(C, 1e-9)

    for _ in range(n_iter):
        pred = sigmoid(X @ w + b)
        err = pred - y
        grad_w = (X.T @ err) / n + lam * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC via the trapezoidal rule."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0.0
    auc = 0.0
    prev_fp = 0.0
    for lab in labels_sorted:
        if lab:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return float(auc / (n_pos * n_neg))


def stratified_kfold_indices(y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) for stratified k-fold."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(k):
        val_pos = pos_idx[fold::k]
        val_neg = neg_idx[fold::k]
        val = np.concatenate([val_pos, val_neg])
        train = np.setdiff1d(np.arange(len(y)), val)
        folds.append((train, val))
    return folds


def grouped_stratified_kfold_indices(
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return grouped folds while keeping label balance as close as possible."""
    if len(y) != len(groups):
        raise ValueError("y and groups must have equal length")
    unique_groups = [g for g in sorted(set(str(v) for v in groups.tolist())) if g]
    if len(unique_groups) < 2:
        return stratified_kfold_indices(y, k)
    k_eff = max(2, min(int(k), len(unique_groups)))

    members: dict[str, np.ndarray] = {}
    for group in unique_groups:
        members[group] = np.where(groups.astype(str) == group)[0]
    group_order = sorted(unique_groups, key=lambda g: (-len(members[g]), g))

    total_pos = float(y.sum())
    total_neg = float(len(y) - total_pos)
    target_pos = total_pos / float(k_eff)
    target_neg = total_neg / float(k_eff)

    fold_indices: List[List[int]] = [[] for _ in range(k_eff)]
    fold_pos = [0.0 for _ in range(k_eff)]
    fold_neg = [0.0 for _ in range(k_eff)]

    for group in group_order:
        idx = members[group]
        g_pos = float(y[idx].sum())
        g_neg = float(len(idx) - g_pos)
        best_fold = 0
        best_score = float("inf")
        for fold in range(k_eff):
            score = (
                abs((fold_pos[fold] + g_pos) - target_pos)
                + abs((fold_neg[fold] + g_neg) - target_neg)
                + (0.01 * len(fold_indices[fold]))
            )
            if score < best_score:
                best_score = score
                best_fold = fold
        fold_indices[best_fold].extend(idx.tolist())
        fold_pos[best_fold] += g_pos
        fold_neg[best_fold] += g_neg

    all_idx = np.arange(len(y))
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for vals in fold_indices:
        val = np.array(sorted(set(vals)), dtype=int)
        if val.size == 0:
            continue
        train = np.setdiff1d(all_idx, val)
        folds.append((train, val))
    if len(folds) < 2:
        return stratified_kfold_indices(y, k)
    return folds


def cross_val_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
    C: float = LOGREG_C,
    groups: np.ndarray | None = None,
    grouped_holdout: bool = True,
) -> Tuple[float, float, List[float]]:
    """Return (mean_auc, std_auc, per_fold_aucs)."""
    if grouped_holdout and groups is not None and len(set(str(v) for v in groups.tolist())) >= 2:
        folds = grouped_stratified_kfold_indices(y, groups, k)
    else:
        folds = stratified_kfold_indices(y, k)
    fold_aucs = []
    for train_idx, val_idx in folds:
        w, b = fit_logreg(X[train_idx], y[train_idx], C=C)
        val_scores = sigmoid(X[val_idx] @ w + b)
        fold_aucs.append(roc_auc(val_scores, y[val_idx]))
    mean = float(np.mean(fold_aucs))
    std = float(np.std(fold_aucs))
    return mean, std, fold_aucs


def direct_composite_auc(X: np.ndarray, y: np.ndarray, weights: dict) -> float:
    """AUC of a direct weighted-sum composite using the given weight dict.

    X columns correspond to SUBMODELS (normalised to [0, 1]).
    Submodels absent from *weights* receive zero weight.
    """
    w = np.array([weights.get(s, 0.0) for s in SUBMODELS], dtype=np.float64)
    total = w.sum()
    if total <= 0:
        return 0.5
    scores = X @ (w / total)
    return roc_auc(scores, y)


# ---------------------------------------------------------------------------
# Weight derivation
# ---------------------------------------------------------------------------

def normalize_weights(
    coefficients: np.ndarray,
    *,
    min_weight: float = MIN_SUBMODEL_WEIGHT,
) -> dict:
    """Map logistic regression coefficients to a valid weight vector.

    Negative coefficients are floored at min_weight so every submodel retains
    a minimum contribution.  Weights are normalised to sum to 1.0 and rounded
    to 2 decimal places.
    """
    w = np.array(coefficients, dtype=np.float64)
    w = np.maximum(w, 0.0)           # negative coef → no predictive signal
    total = w.sum()
    if total <= 0:
        w = np.ones(len(SUBMODELS))  # fallback: equal weights
        total = w.sum()
    w = w / total
    w = np.maximum(w, min_weight)    # apply per-submodel floor
    w = w / w.sum()                  # re-normalise after flooring
    # Round to 2dp and fix rounding residual on the highest-weight submodel.
    rounded = [round(float(v), 2) for v in w]
    residual = round(1.0 - sum(rounded), 2)
    if residual != 0.0:
        top = int(np.argmax(rounded))
        rounded[top] = round(rounded[top] + residual, 2)
    return {s: rounded[i] for i, s in enumerate(SUBMODELS)}


# ---------------------------------------------------------------------------
# Config read / write
# ---------------------------------------------------------------------------

def load_current_weights(config_path: Path) -> dict:
    """Parse submodel_weights from scoring_parameters.yaml (no PyYAML needed)."""
    text = config_path.read_text(encoding="utf-8")
    in_block = False
    weights: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        # Match both bare YAML key and JSON-quoted key:
        #   submodel_weights:      (YAML)
        #   "submodel_weights": {  (JSON-style)
        if stripped.lstrip('"\'').startswith("submodel_weights") and ":" in stripped:
            in_block = True
            continue
        if in_block:
            if stripped.startswith('"') or stripped.startswith("'") or (stripped and stripped[0].isalpha() and ":" in stripped):
                parts = stripped.split(":")
                if len(parts) == 2:
                    key = parts[0].strip().strip('"').strip("'")
                    try:
                        val = float(parts[1].strip().rstrip(","))
                        weights[key] = val
                    except ValueError:
                        pass
            if stripped.startswith("{") or (stripped and not stripped.startswith('"') and not stripped.startswith("'") and ":" not in stripped and in_block and weights):
                break
    return weights


def update_weights_in_config(config_path: Path, new_weights: dict) -> None:
    """Rewrite the submodel_weights block in scoring_parameters.yaml in-place."""
    text = config_path.read_text(encoding="utf-8")
    # Detect format: JSON-style {...} or YAML-style key: value lines
    lines = text.splitlines()
    out: List[str] = []
    in_block = False
    replaced = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not in_block and "submodel_weights" in line and ":" in line:
            in_block = True
            out.append(line)
            i += 1
            # Determine indent of first key line
            while i < len(lines) and not lines[i].strip():
                out.append(lines[i])
                i += 1
            if i < len(lines):
                indent = len(lines[i]) - len(lines[i].lstrip())
                prefix = " " * indent
                # Detect whether the original block uses trailing commas
                # (JSON-style) by checking the first existing weight line.
                first_line = lines[i].strip()
                use_commas = first_line.endswith(",")
                # Emit new weights, matching original comma style.
                items = list(new_weights.items())
                for idx, (submodel, weight) in enumerate(items):
                    is_last = idx == len(items) - 1
                    suffix = "" if (is_last and not use_commas) or (not use_commas) else ","
                    if use_commas and not is_last:
                        suffix = ","
                    elif use_commas and is_last:
                        suffix = ""
                    out.append(f'{prefix}"{submodel}": {weight}{suffix}')
                # Skip old weight lines
                while i < len(lines):
                    s = lines[i].strip()
                    if not s or s.startswith('"') or s.startswith("'") or (s[0].isalpha() and ":" in s and not any(c in s for c in ["{", "}"])):
                        # Check it's a weight-like line
                        parts = s.split(":")
                        if len(parts) == 2:
                            try:
                                float(parts[1].strip().rstrip(","))
                                i += 1
                                continue
                            except ValueError:
                                pass
                    break
            replaced = True
            continue
        out.append(line)
        i += 1
    if not replaced:
        raise ValueError("Could not locate submodel_weights block in config")
    config_path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=LOGREG_C, dest="C")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report proposed weights but do not write config.")
    parser.add_argument(
        "--allow-row-level-cv",
        action="store_true",
        help=(
            "Use legacy row-level stratified CV. Default is grouped holdout "
            "by event/region/year for stricter leakage control."
        ),
    )
    parser.add_argument(
        "--compare-weights",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON dict of a reference weight vector to compare against the "
            "current config weights via direct_composite_auc.  Exits after "
            "printing the comparison; does not fit or update weights."
        ),
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    rows = load_dataset(args.dataset)
    if not rows:
        print("ERROR: no usable rows found.", file=sys.stderr)
        return 1

    X, y, groups = extract_features_with_groups(rows)
    n, d = X.shape
    if n < 50:
        print(f"ERROR: only {n} rows with complete submodel scores (minimum 50).", file=sys.stderr)
        return 1

    group_count = len(set(str(v) for v in groups.tolist()))
    print(
        f"Usable rows: {n}  |  positive rate: {y.mean():.3f}  |  "
        f"submodels: {d}  |  holdout groups: {group_count}"
    )

    # ── Load current weights (needed for baseline comparison) ─────────────
    current = load_current_weights(SCORING_CONFIG_PATH)

    # ── --compare-weights: validate a reference vector vs current ─────────
    if args.compare_weights is not None:
        try:
            ref_weights = json.loads(args.compare_weights)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --compare-weights is not valid JSON: {exc}", file=sys.stderr)
            return 1
        ref_auc = direct_composite_auc(X, y, ref_weights)
        cur_auc = direct_composite_auc(X, y, current)
        print(f"\nDirect-blend AUC comparison:")
        print(f"  Reference weights : {ref_auc:.4f}")
        print(f"  Current weights   : {cur_auc:.4f}")
        print(f"  Delta (cur - ref) : {cur_auc - ref_auc:+.4f}")
        if cur_auc < ref_auc - 1e-4:
            print("  VERDICT: current weights are WORSE — consider reverting.")
        elif cur_auc > ref_auc + 1e-4:
            print("  VERDICT: current weights are BETTER — change confirmed.")
        else:
            print("  VERDICT: negligible difference.")
        return 0

    # ── Baseline AUC: two views ───────────────────────────────────────────
    # 1. Stale archived composite score from the dataset (scored with old weights
    #    at eval-dataset creation time; provided as historical context only).
    archived_scores = np.array([
        r["scores"].get("wildfire_risk_score", 50.0)
        for r in rows
        if r.get("evaluation", {}).get("row_usable")
        and r["outcome"].get("structure_loss_or_major_damage") is not None
        and all(
            r.get("feature_snapshot", {}).get("factor_contribution_breakdown", {}).get(s, {}).get("clamped_submodel_score") is not None
            for s in SUBMODELS
        )
    ], dtype=np.float64)
    archived_auc = roc_auc(archived_scores, y)
    # 2. Direct composite AUC using current config weights on the same submodel
    #    features extracted above — apples-to-apples baseline for gate comparison.
    old_direct_auc = direct_composite_auc(X, y, current)
    baseline_auc = old_direct_auc
    print(f"\nBaseline AUC (archived wildfire_risk_score): {archived_auc:.4f}")
    print(f"Baseline AUC (current weights, direct blend): {old_direct_auc:.4f}")

    # ── Cross-validated AUC with logistic regression ──────────────────────
    holdout_mode = "row_level" if args.allow_row_level_cv else "grouped_event_region_year"
    print(f"\nFitting logistic regression ({args.folds}-fold CV, C={args.C}, holdout={holdout_mode}) ...")
    mean_auc, std_auc, fold_aucs = cross_val_auc(
        X,
        y,
        k=args.folds,
        C=args.C,
        groups=groups,
        grouped_holdout=(not args.allow_row_level_cv),
    )
    print(f"CV AUC: {mean_auc:.4f} ± {std_auc:.4f}  (folds: {[round(a,4) for a in fold_aucs]})")

    # ── Stop conditions ───────────────────────────────────────────────────
    auc_improvement = mean_auc - baseline_auc
    print(f"\nAUC improvement over baseline: {auc_improvement:+.4f}")

    high_variance = std_auc > CV_VARIANCE_GATE
    if high_variance:
        print(f"WARNING: CV std {std_auc:.4f} exceeds variance gate {CV_VARIANCE_GATE}. "
              "Dataset may be too small for reliable fitting.")

    if auc_improvement < AUC_IMPROVEMENT_GATE:
        print(f"\nSTOP: AUC improvement {auc_improvement:+.4f} is below gate "
              f"{AUC_IMPROVEMENT_GATE:.2f}. Weights will NOT be updated.")
        _print_weight_table(load_current_weights(SCORING_CONFIG_PATH), {})
        return 1

    # ── Fit on full dataset for final coefficients ────────────────────────
    w_full, _ = fit_logreg(X, y, C=args.C)
    print(f"\nFull-dataset coefficients:\n  " +
          "\n  ".join(f"{SUBMODELS[i]:35s}: {w_full[i]:+.4f}" for i in range(d)))

    proposed = normalize_weights(w_full)
    new_direct_auc = direct_composite_auc(X, y, proposed)
    print(f"\nDirect-blend AUC — old weights: {old_direct_auc:.4f}  →  new weights: {new_direct_auc:.4f}  "
          f"({new_direct_auc - old_direct_auc:+.4f})")

    _print_weight_table(current, proposed)

    if args.dry_run:
        print("\n--dry-run: config NOT updated.")
        return 0

    update_weights_in_config(SCORING_CONFIG_PATH, proposed)
    print(f"\nUpdated submodel_weights in {SCORING_CONFIG_PATH}")
    return 0


def _print_weight_table(current: dict, proposed: dict) -> None:
    print(f"\n{'Submodel':<35} {'Current':>9} {'Proposed':>9} {'Delta':>8}")
    print("-" * 65)
    for s in SUBMODELS:
        cur = current.get(s, float("nan"))
        prop = proposed.get(s, float("nan"))
        delta = prop - cur if math.isfinite(cur) and math.isfinite(prop) else float("nan")
        delta_str = f"{delta:+.2f}" if math.isfinite(delta) else "—"
        prop_str = f"{prop:.2f}" if math.isfinite(prop) else "—"
        print(f"{s:<35} {cur:>9.2f} {prop_str:>9} {delta_str:>8}")


if __name__ == "__main__":
    sys.exit(main())
