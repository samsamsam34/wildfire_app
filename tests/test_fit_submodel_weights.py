"""Unit tests for scripts/fit_submodel_weights.py.

Covers the pure-Python/numpy utility functions without touching the filesystem
or requiring the evaluation dataset to be present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_submodel_weights import (
    SUBMODELS,
    MIN_SUBMODEL_WEIGHT,
    direct_composite_auc,
    extract_features,
    load_current_weights,
    normalize_weights,
    roc_auc,
    sigmoid,
    stratified_kfold_indices,
)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid_zero():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_large_positive():
    assert sigmoid(np.array([500.0]))[0] == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative():
    assert sigmoid(np.array([-500.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_monotone():
    z = np.linspace(-10, 10, 50)
    s = sigmoid(z)
    assert np.all(np.diff(s) > 0), "sigmoid must be strictly increasing"


# ---------------------------------------------------------------------------
# roc_auc
# ---------------------------------------------------------------------------

def test_roc_auc_perfect():
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_random():
    """AUC of a random (uniform) predictor should be near 0.5."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(size=500)
    labels = rng.integers(0, 2, size=500).astype(float)
    auc = roc_auc(scores, labels)
    assert 0.4 <= auc <= 0.6, f"Random predictor AUC should be ~0.5, got {auc:.4f}"


def test_roc_auc_all_positive():
    """All-positive label set returns 0.5 (undefined, safe fallback)."""
    scores = np.array([0.9, 0.8])
    labels = np.array([1.0, 1.0])
    assert roc_auc(scores, labels) == 0.5


def test_roc_auc_worse_than_random():
    """Inverted predictor gives ~0.0 AUC."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------

def test_normalize_weights_sums_to_one():
    coef = np.array([1.0, 2.0, 3.0, 0.5, 1.5, 0.1, 2.5, 1.0])
    w = normalize_weights(coef)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


def test_normalize_weights_covers_all_submodels():
    coef = np.ones(len(SUBMODELS))
    w = normalize_weights(coef)
    assert set(w.keys()) == set(SUBMODELS)


def test_normalize_weights_floor_applied():
    """Every submodel must have weight >= MIN_SUBMODEL_WEIGHT."""
    # One large coefficient dominates; others would be near zero without floor.
    coef = np.zeros(len(SUBMODELS))
    coef[0] = 100.0
    w = normalize_weights(coef)
    assert all(v >= MIN_SUBMODEL_WEIGHT - 1e-9 for v in w.values()), (
        f"All weights must be >= {MIN_SUBMODEL_WEIGHT}; got {w}"
    )


def test_normalize_weights_all_negative_sums_to_one():
    """All-negative coefficients fall back to equal pre-rounding weights;
    post-rounding the sum must still be exactly 1.0."""
    coef = np.full(len(SUBMODELS), -1.0)
    w = normalize_weights(coef)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    # Floor must be applied to every submodel
    assert all(v >= MIN_SUBMODEL_WEIGHT - 1e-9 for v in w.values())


def test_normalize_weights_rounding_residual_absorbed():
    """Output weights must sum exactly to 1.0 after rounding."""
    rng = np.random.default_rng(7)
    coef = rng.uniform(0.1, 2.0, size=len(SUBMODELS))
    w = normalize_weights(coef)
    total = round(sum(w.values()), 10)
    assert total == pytest.approx(1.0, abs=1e-9), f"Weights sum to {total}, not 1.0"


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

def _make_row(scores: list[float | None], label: int) -> dict:
    fcd = {}
    for submodel, score in zip(SUBMODELS, scores):
        if score is not None:
            fcd[submodel] = {"clamped_submodel_score": score}
    return {
        "evaluation": {"row_usable": True},
        "outcome": {"structure_loss_or_major_damage": label},
        "feature_snapshot": {"factor_contribution_breakdown": fcd},
        "scores": {"wildfire_risk_score": 50.0},
    }


def test_extract_features_shape():
    rows = [_make_row([float(i)] * len(SUBMODELS), i % 2) for i in range(10)]
    X, y = extract_features(rows)
    assert X.shape == (10, len(SUBMODELS))
    assert y.shape == (10,)


def test_extract_features_normalised_to_unit():
    """All feature values must be in [0, 1] after normalisation."""
    rows = [_make_row([0.0] * len(SUBMODELS), 0),
            _make_row([100.0] * len(SUBMODELS), 1)]
    X, y = extract_features(rows)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_extract_features_drops_incomplete_rows():
    """Rows with any missing submodel score must be excluded."""
    scores_complete = [50.0] * len(SUBMODELS)
    scores_partial = scores_complete[:]
    scores_partial[0] = None  # one missing
    rows = [_make_row(scores_complete, 1), _make_row(scores_partial, 0)]
    X, y = extract_features(rows)
    assert X.shape[0] == 1, "Row with missing submodel score must be dropped"


def test_extract_features_labels_binary():
    rows = [_make_row([50.0] * len(SUBMODELS), 0),
            _make_row([50.0] * len(SUBMODELS), 1)]
    X, y = extract_features(rows)
    assert set(y.tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# stratified_kfold_indices
# ---------------------------------------------------------------------------

def test_kfold_covers_all_indices():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    folds = stratified_kfold_indices(y, k=5)
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val.tolist()) == list(range(len(y))), (
        "All indices must appear exactly once across validation folds"
    )


def test_kfold_train_val_disjoint():
    y = np.array([0, 1] * 10, dtype=float)
    for train, val in stratified_kfold_indices(y, k=5):
        assert len(np.intersect1d(train, val)) == 0, "Train and val must be disjoint"


def test_kfold_each_fold_has_both_classes():
    y = np.array([0, 1] * 10, dtype=float)
    for train, val in stratified_kfold_indices(y, k=5):
        assert 1 in y[val] and 0 in y[val], "Each val fold must contain both classes"


# ---------------------------------------------------------------------------
# load_current_weights — JSON-quoted key handling
# ---------------------------------------------------------------------------

def test_load_current_weights_reads_json_quoted_key(tmp_path):
    """load_current_weights must parse a JSON-style config where the block key
    is surrounded by double quotes, e.g. "submodel_weights": { ... }"""
    cfg = tmp_path / "scoring_parameters.yaml"
    lines = [
        '{',
        '  "submodel_weights": {',
    ]
    for i, s in enumerate(SUBMODELS):
        comma = "," if i < len(SUBMODELS) - 1 else ""
        lines.append(f'    "{s}": 0.{i + 10}{comma}')
    lines += ['  },', '  "other_key": 1', '}']
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    weights = load_current_weights(cfg)
    assert set(weights.keys()) == set(SUBMODELS), (
        f"All submodels must be parsed; got keys: {list(weights.keys())}"
    )
    for s in SUBMODELS:
        assert weights[s] > 0, f"Weight for {s} must be positive"


def test_load_current_weights_reads_bare_yaml_key(tmp_path):
    """load_current_weights must also handle bare YAML-style keys."""
    cfg = tmp_path / "scoring_parameters.yaml"
    lines = ["submodel_weights:"]
    for s in SUBMODELS:
        lines.append(f'  "{s}": 0.12')
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    weights = load_current_weights(cfg)
    assert set(weights.keys()) == set(SUBMODELS)


# ---------------------------------------------------------------------------
# direct_composite_auc
# ---------------------------------------------------------------------------

def test_direct_composite_auc_perfect_predictor():
    """A weight dict that puts all mass on the one perfectly discriminating
    feature must return AUC == 1.0."""
    n = 20
    X = np.zeros((n, len(SUBMODELS)))
    y = np.array([i % 2 for i in range(n)], dtype=float)
    # Make the first submodel a perfect predictor
    X[:, 0] = y
    weights = {SUBMODELS[0]: 1.0}
    assert direct_composite_auc(X, y, weights) == pytest.approx(1.0)


def test_direct_composite_auc_empty_weights_returns_half():
    """Zero or absent weights must return 0.5 (undefined, safe fallback)."""
    X = np.ones((10, len(SUBMODELS))) * 0.5
    y = np.array([0, 1] * 5, dtype=float)
    assert direct_composite_auc(X, y, {}) == pytest.approx(0.5)


def test_direct_composite_auc_better_weights_lift_auc():
    """Fitted weights should produce higher AUC than uniform weights on a
    dataset where one submodel is strongly predictive."""
    rng = np.random.default_rng(99)
    n = 100
    y = rng.integers(0, 2, size=n).astype(float)
    X = rng.uniform(0, 0.4, size=(n, len(SUBMODELS)))
    # Make the last submodel a near-perfect predictor
    X[:, -1] = y * 0.9 + rng.uniform(0, 0.05, size=n)

    uniform_weights = {s: 1.0 / len(SUBMODELS) for s in SUBMODELS}
    focused_weights = {SUBMODELS[-1]: 1.0}

    auc_uniform = direct_composite_auc(X, y, uniform_weights)
    auc_focused = direct_composite_auc(X, y, focused_weights)
    assert auc_focused > auc_uniform, (
        f"Weights focused on discriminating submodel must beat uniform weights: "
        f"focused={auc_focused:.4f}, uniform={auc_uniform:.4f}"
    )


# ---------------------------------------------------------------------------
# --compare-weights: weight comparison verdict logic
# ---------------------------------------------------------------------------

def test_compare_weights_better_current_gives_better_verdict():
    """direct_composite_auc delta is positive when current weights outperform
    the reference — the comparison arithmetic is correct."""
    rng = np.random.default_rng(17)
    n = 200
    y = rng.integers(0, 2, size=n).astype(float)
    # All submodels carry moderate noise; submodel 0 has a weak signal.
    X = rng.uniform(0.3, 0.7, size=(n, len(SUBMODELS)))
    X[:, 0] = 0.5 + (y - 0.5) * 0.25 + rng.uniform(-0.1, 0.1, size=n)

    # "Current" weights: heavy on the weak-signal submodel
    current = {SUBMODELS[0]: 0.8, **{s: 0.2 / (len(SUBMODELS) - 1) for s in SUBMODELS[1:]}}
    # "Reference" weights: uniform (dilutes the signal with noise)
    reference = {s: 1.0 / len(SUBMODELS) for s in SUBMODELS}

    cur_auc = direct_composite_auc(X, y, current)
    ref_auc = direct_composite_auc(X, y, reference)
    assert cur_auc > ref_auc, (
        f"Current weights focused on discriminating submodel must beat reference: "
        f"cur={cur_auc:.4f}, ref={ref_auc:.4f}"
    )


def test_compare_weights_worse_current_gives_negative_delta():
    """When the reference weights are better, the delta (cur - ref) is negative."""
    rng = np.random.default_rng(31)
    n = 200
    y = rng.integers(0, 2, size=n).astype(float)
    X = rng.uniform(0.3, 0.7, size=(n, len(SUBMODELS)))
    X[:, 0] = 0.5 + (y - 0.5) * 0.25 + rng.uniform(-0.1, 0.1, size=n)

    # Reference is focused; current is uniform (worse)
    reference = {SUBMODELS[0]: 0.8, **{s: 0.2 / (len(SUBMODELS) - 1) for s in SUBMODELS[1:]}}
    current = {s: 1.0 / len(SUBMODELS) for s in SUBMODELS}

    cur_auc = direct_composite_auc(X, y, current)
    ref_auc = direct_composite_auc(X, y, reference)
    assert cur_auc < ref_auc, (
        f"Current uniform weights must score below focused reference: "
        f"cur={cur_auc:.4f}, ref={ref_auc:.4f}"
    )
