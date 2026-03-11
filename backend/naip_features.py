from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


NAIP_FEATURES_FILENAME = "naip_structure_features.json"
RING_KEYS = ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "ring_100_300_ft")


def structure_feature_key(
    *,
    structure_id: str | None,
    centroid_lat: float | None,
    centroid_lon: float | None,
) -> str | None:
    if structure_id and str(structure_id).strip():
        return f"structure_id:{str(structure_id).strip()}"
    if centroid_lat is None or centroid_lon is None:
        return None
    return f"centroid:{float(centroid_lat):.6f},{float(centroid_lon):.6f}"


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 4)
    rank = max(0.0, min(1.0, float(q) / 100.0)) * (len(values) - 1)
    lo = int(rank)
    hi = min(len(values) - 1, lo + 1)
    frac = rank - lo
    return round((values[lo] * (1.0 - frac)) + (values[hi] * frac), 4)


def build_quantiles(values: list[float], *, percentiles: list[float] | None = None) -> dict[str, float]:
    if not values:
        return {}
    requested = percentiles or [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
    sorted_values = sorted(float(v) for v in values)
    out: dict[str, float] = {}
    for p in requested:
        qv = _safe_quantile(sorted_values, float(p))
        if qv is not None:
            out[str(int(p) if float(p).is_integer() else p)] = qv
    return out


def percentile_from_quantiles(value: float | None, quantiles: dict[str, Any] | None) -> float | None:
    if value is None or not isinstance(quantiles, dict):
        return None
    q_pairs: list[tuple[float, float]] = []
    for k, v in quantiles.items():
        p = _coerce_float(k)
        qv = _coerce_float(v)
        if p is None or qv is None:
            continue
        q_pairs.append((p, qv))
    if not q_pairs:
        return None
    q_pairs.sort(key=lambda item: item[0])
    v = float(value)
    if v <= q_pairs[0][1]:
        return round(q_pairs[0][0], 1)
    if v >= q_pairs[-1][1]:
        return round(q_pairs[-1][0], 1)

    for i in range(1, len(q_pairs)):
        prev_p, prev_v = q_pairs[i - 1]
        next_p, next_v = q_pairs[i]
        if prev_v <= v <= next_v:
            if next_v == prev_v:
                return round((prev_p + next_p) / 2.0, 1)
            ratio = (v - prev_v) / (next_v - prev_v)
            return round(prev_p + (ratio * (next_p - prev_p)), 1)
    return None


@lru_cache(maxsize=32)
def load_naip_feature_artifact(path: str) -> dict[str, Any]:
    feature_path = Path(path)
    if not feature_path.exists():
        return {}
    try:
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def resolve_naip_feature_path(
    *,
    region_manifest_path: str | None,
    runtime_path: str | None = None,
) -> str | None:
    candidates: list[Path] = []
    if runtime_path:
        candidates.append(Path(runtime_path))
    if region_manifest_path:
        manifest = Path(region_manifest_path)
        candidates.append(manifest.parent / NAIP_FEATURES_FILENAME)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None

