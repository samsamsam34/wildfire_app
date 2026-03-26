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

SCHEMA_VERSION = "1.4.0"
WGS84_COORD_DECIMALS = 6
DISTANCE_TIEBREAK_DECIMALS = 6
DEFAULT_HIGH_CONFIDENCE_DISTANCE_M = 30.0
HIGH_CONFIDENCE_DISTANCE_NEAR_MISS_MARGIN_M = 10.0
HIGH_CONFIDENCE_SCORE_THRESHOLD = 0.9
HIGH_CONFIDENCE_SCORE_NEAR_MISS_MARGIN = 0.05
FALLBACK_HEAVY_WEIGHT_THRESHOLD = 0.65
FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD = 0.45
FALLBACK_HEAVY_FACTOR_RATIO_THRESHOLD = 0.60
FALLBACK_HEAVY_MISSING_RATIO_THRESHOLD = 0.50
FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD = 2
LEAKAGE_TOKENS = (
    "outcome",
    "damage",
    "destroyed",
    "structure_loss_or_major_damage",
    "adverse_outcome",
    "label",
)

STRUCTURE_PROXY_FEATURE_KEYS: tuple[str, ...] = (
    "nearby_structure_count_100_ft",
    "nearby_structure_count_300_ft",
    "nearest_structure_distance_ft",
    "distance_to_nearest_structure_ft",
    "structure_density",
    "structure_density_proxy",
    "clustering_index",
    "local_structure_clustering_index",
    "building_age_proxy_year",
    "building_age_material_proxy_risk",
)

NEAR_STRUCTURE_VEGETATION_FEATURE_KEYS: tuple[str, ...] = (
    "near_structure_vegetation_0_5_pct",
    "ring_0_5_ft_vegetation_density",
    "ring_5_30_ft_vegetation_density",
    "ring_0_5_ft_vegetation_density_proxy_blend",
    "ring_5_30_ft_vegetation_density_proxy_blend",
    "canopy_adjacency_proxy_pct",
    "vegetation_continuity_proxy_pct",
    "near_structure_connectivity_index",
    "near_structure_connectivity_index_proxy_blend",
    "nearest_high_fuel_patch_distance_ft",
    "nearest_high_fuel_patch_distance_ft_proxy_blend",
    "defensible_space_proxy_score",
    "defensible_space_proxy_score_blend",
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


@dataclass(frozen=True)
class JoinConfig:
    exact_match_distance_m: float = 3.0
    near_match_distance_m: float = 30.0
    max_distance_m: float = 120.0
    global_max_distance_m: float = 1000.0
    buffer_match_radius_m: float = 80.0
    high_confidence_distance_m: float = DEFAULT_HIGH_CONFIDENCE_DISTANCE_M
    medium_confidence_distance_m: float = 100.0
    event_year_tolerance_years: int = 1
    enable_global_nearest_fallback: bool = True
    allow_duplicate_outcome_matches: bool = False
    address_token_overlap_min: float = 0.75


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


def _normalize_wgs84_coordinate(value: Any, *, kind: str) -> float | None:
    coord = _safe_float(value)
    if coord is None:
        return None
    limit = 90.0 if kind == "lat" else 180.0
    if coord < -limit or coord > limit:
        return None
    return round(float(coord), WGS84_COORD_DECIMALS)


def _mercator_to_wgs84(*, x: float, y: float) -> tuple[float | None, float | None]:
    max_extent = 20037508.342789244
    if abs(x) > max_extent or abs(y) > max_extent:
        return None, None
    lon = (x / max_extent) * 180.0
    lat = (y / max_extent) * 180.0
    lat = (180.0 / math.pi) * (2.0 * math.atan(math.exp(lat * math.pi / 180.0)) - (math.pi / 2.0))
    return _normalize_wgs84_coordinate(lat, kind="lat"), _normalize_wgs84_coordinate(lon, kind="lon")


def _extract_normalized_point(row: dict[str, Any]) -> tuple[float | None, float | None, str]:
    lat_raw = row.get("latitude")
    lon_raw = row.get("longitude")
    lat = _normalize_wgs84_coordinate(lat_raw, kind="lat")
    lon = _normalize_wgs84_coordinate(lon_raw, kind="lon")
    if lat is not None and lon is not None:
        return lat, lon, "wgs84_latlon"

    # Common alternative keys.
    alt_lat = row.get("lat")
    alt_lon = row.get("lon")
    if alt_lon is None:
        alt_lon = row.get("lng")
    lat = _normalize_wgs84_coordinate(alt_lat, kind="lat")
    lon = _normalize_wgs84_coordinate(alt_lon, kind="lon")
    if lat is not None and lon is not None:
        return lat, lon, "wgs84_alt_latlon"

    # Try swapping if lat/lon appear reversed.
    swapped_lat = _normalize_wgs84_coordinate(lon_raw, kind="lat")
    swapped_lon = _normalize_wgs84_coordinate(lat_raw, kind="lon")
    if swapped_lat is not None and swapped_lon is not None:
        return swapped_lat, swapped_lon, "wgs84_swapped_latlon"

    # Attempt Web Mercator conversion from longitude=x and latitude=y.
    x = _safe_float(lon_raw)
    y = _safe_float(lat_raw)
    if x is not None and y is not None and (abs(x) > 180.0 or abs(y) > 90.0):
        conv_lat, conv_lon = _mercator_to_wgs84(x=x, y=y)
        if conv_lat is not None and conv_lon is not None:
            return conv_lat, conv_lon, "web_mercator_from_latlon_fields"

    # Attempt conversion from explicit x/y projected columns.
    x = _safe_float(row.get("x"))
    y = _safe_float(row.get("y"))
    if x is not None and y is not None:
        conv_lat, conv_lon = _mercator_to_wgs84(x=x, y=y)
        if conv_lat is not None and conv_lon is not None:
            return conv_lat, conv_lon, "web_mercator_xy"

    return None, None, "missing_or_invalid_coordinates"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_bool_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


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


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(str(a or "").split())
    tokens_b = set(str(b or "").split())
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union <= 0:
        return 0.0
    return inter / float(union)


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


def _extract_rows_with_filter_stats(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = payload.get("records")
    if rows is None:
        return [], {"raw_row_count": 0, "valid_dict_row_count": 0, "invalid_non_dict_row_count": 0, "invalid_records_container_count": 0}
    if not isinstance(rows, list):
        return [], {"raw_row_count": 0, "valid_dict_row_count": 0, "invalid_non_dict_row_count": 0, "invalid_records_container_count": 1}
    valid_rows = [row for row in rows if isinstance(row, dict)]
    invalid_non_dict = max(0, len(rows) - len(valid_rows))
    return valid_rows, {
        "raw_row_count": len(rows),
        "valid_dict_row_count": len(valid_rows),
        "invalid_non_dict_row_count": invalid_non_dict,
        "invalid_records_container_count": 0,
    }


def _as_outcome_record(row: dict[str, Any]) -> OutcomeRecord:
    event_id = str(row.get("event_id") or "").strip()
    event_name = str(row.get("event_name") or "").strip()
    event_date = row.get("event_date")
    event_year = _parse_year(row.get("event_year")) or _parse_year(event_date)
    source_record_id = str(row.get("source_record_id") or "").strip()
    record_id = str(row.get("record_id") or "").strip()
    parcel_id = str(row.get("parcel_identifier") or row.get("parcel_id") or "").strip()
    address_norm = _normalize_address(row.get("address_text") or row.get("address") or "")
    latitude, longitude, coord_mode = _extract_normalized_point(row)
    row["_coordinate_normalization_mode"] = coord_mode
    return OutcomeRecord(
        payload=row,
        event_id=event_id,
        event_name_norm=_normalize_text(event_name),
        event_year=event_year,
        source_record_id=source_record_id,
        record_id=record_id,
        parcel_id=parcel_id,
        address_norm=address_norm,
        latitude=latitude,
        longitude=longitude,
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
    latitude, longitude, coord_mode = _extract_normalized_point(row)
    row["_coordinate_normalization_mode"] = coord_mode
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
        latitude=latitude,
        longitude=longitude,
    )


def _stable_outcome_sort_key(row: OutcomeRecord) -> tuple[Any, ...]:
    return (
        str(row.event_id or ""),
        int(row.event_year or 0),
        str(row.record_id or ""),
        str(row.source_record_id or ""),
        str(row.parcel_id or ""),
        str(row.address_norm or ""),
        float(row.latitude) if row.latitude is not None else float("inf"),
        float(row.longitude) if row.longitude is not None else float("inf"),
    )


def _stable_feature_sort_key(row: FeatureRecord) -> tuple[Any, ...]:
    return (
        str(row.artifact_path or ""),
        str(row.event_id or ""),
        int(row.event_year or 0),
        str(row.record_id or ""),
        str(row.source_record_id or ""),
        str(row.parcel_id or ""),
        str(row.address_norm or ""),
        float(row.latitude) if row.latitude is not None else float("inf"),
        float(row.longitude) if row.longitude is not None else float("inf"),
    )


def _joined_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    event_payload = row.get("event") if isinstance(row.get("event"), dict) else {}
    feature_payload = row.get("feature") if isinstance(row.get("feature"), dict) else {}
    outcome_payload = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    join_payload = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
    distance = _safe_float(join_payload.get("join_distance_m"))
    score = _safe_float(join_payload.get("join_confidence_score"))
    return (
        str(row.get("property_event_id") or ""),
        str(event_payload.get("event_id") or ""),
        str(feature_payload.get("record_id") or ""),
        str(feature_payload.get("source_record_id") or ""),
        str(feature_payload.get("feature_artifact_path") or ""),
        str(outcome_payload.get("record_id") or ""),
        str(outcome_payload.get("source_record_id") or ""),
        str(join_payload.get("join_method") or ""),
        float(distance) if distance is not None else float("inf"),
        -float(score) if score is not None else float("inf"),
    )


def _join_confidence_tier_rank(value: str) -> int:
    tier = str(value or "").strip().lower()
    if tier == "high":
        return 3
    if tier in {"moderate", "medium"}:
        return 2
    if tier == "low":
        return 1
    return 0


def _dedupe_joined_rows_by_property_event_id(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], [], {
            "total_rows_before_property_event_dedupe": 0,
            "unique_property_event_id_count": 0,
            "duplicate_property_event_rows_removed_count": 0,
            "property_event_ids_with_duplicates_count": 0,
            "duplication_factor": 1.0,
            "duplicate_property_event_ids_examples": [],
            "duplicate_rows_removed_by_join_confidence_tier": {},
            "duplicate_rows_removed_by_join_method": {},
        }

    def _selection_rank(row: dict[str, Any]) -> tuple[Any, ...]:
        join_payload = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
        evaluation_payload = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
        feature_payload = row.get("feature") if isinstance(row.get("feature"), dict) else {}
        outcome_payload = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        score = _safe_float(join_payload.get("join_confidence_score"))
        distance = _safe_float(join_payload.get("join_distance_m"))
        tier_rank = _join_confidence_tier_rank(str(join_payload.get("join_confidence_tier") or ""))
        soft_flags = evaluation_payload.get("soft_filter_flags")
        soft_flag_count = len(soft_flags) if isinstance(soft_flags, list) else 0
        fallback_heavy = 1 if bool(evaluation_payload.get("fallback_heavy")) else 0
        return (
            -float(score) if score is not None else 1.0,
            -int(tier_rank),
            float(distance) if distance is not None else float("inf"),
            int(soft_flag_count),
            int(fallback_heavy),
            str(join_payload.get("join_method") or ""),
            str(feature_payload.get("feature_artifact_path") or ""),
            str(feature_payload.get("record_id") or ""),
            str(outcome_payload.get("record_id") or ""),
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=_joined_row_sort_key):
        property_event_id = str(row.get("property_event_id") or "").strip()
        if not property_event_id:
            event_payload = row.get("event") if isinstance(row.get("event"), dict) else {}
            feature_payload = row.get("feature") if isinstance(row.get("feature"), dict) else {}
            property_event_id = (
                f"unknown::{event_payload.get('event_id') or 'unknown_event'}::"
                f"{feature_payload.get('record_id') or feature_payload.get('source_record_id') or 'unknown_feature'}::"
                f"{feature_payload.get('feature_artifact_path') or 'unknown_artifact'}"
            )
        grouped.setdefault(property_event_id, []).append(row)

    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    duplicate_tier_counts: dict[str, int] = {}
    duplicate_method_counts: dict[str, int] = {}
    duplicate_ids: list[str] = []
    for property_event_id in sorted(grouped.keys()):
        candidates = grouped[property_event_id]
        if len(candidates) <= 1:
            kept_rows.extend(candidates)
            continue
        duplicate_ids.append(property_event_id)
        ranked = sorted(candidates, key=_selection_rank)
        best = ranked[0]
        best_eval = best.get("evaluation") if isinstance(best.get("evaluation"), dict) else {}
        best_eval = dict(best_eval)
        best_eval["property_event_dedupe_applied"] = True
        best_eval["property_event_duplicate_count_collapsed"] = max(0, len(candidates) - 1)
        best["evaluation"] = best_eval
        kept_rows.append(best)
        for removed in ranked[1:]:
            removed_meta = removed.get("join_metadata") if isinstance(removed.get("join_metadata"), dict) else {}
            removed_tier = str(removed_meta.get("join_confidence_tier") or "unknown")
            removed_method = str(removed_meta.get("join_method") or "unknown")
            duplicate_tier_counts[removed_tier] = duplicate_tier_counts.get(removed_tier, 0) + 1
            duplicate_method_counts[removed_method] = duplicate_method_counts.get(removed_method, 0) + 1
            removed_rows.append(
                {
                    "feature_artifact_path": ((removed.get("feature") or {}).get("feature_artifact_path")),
                    "record_id": ((removed.get("feature") or {}).get("record_id")),
                    "event_id": ((removed.get("event") or {}).get("event_id")),
                    "reason": "duplicate_property_event_id_collapsed",
                    "property_event_id": property_event_id,
                    "retained_feature_record_id": ((best.get("feature") or {}).get("record_id")),
                    "retained_outcome_record_id": ((best.get("outcome") or {}).get("record_id")),
                    "removed_outcome_record_id": ((removed.get("outcome") or {}).get("record_id")),
                    "removed_join_confidence_tier": removed_tier,
                    "removed_join_distance_m": _safe_float(removed_meta.get("join_distance_m")),
                    "removed_join_confidence_score": _safe_float(removed_meta.get("join_confidence_score")),
                    "removed_join_method": removed_method,
                    "join_pass": str(removed_meta.get("join_pass") or ""),
                }
            )

    kept_rows = sorted(kept_rows, key=_joined_row_sort_key)
    before_count = len(rows)
    after_count = len(kept_rows)
    duplication_factor = round(float(before_count) / float(after_count), 4) if after_count > 0 else None
    stats = {
        "total_rows_before_property_event_dedupe": before_count,
        "unique_property_event_id_count": after_count,
        "duplicate_property_event_rows_removed_count": len(removed_rows),
        "property_event_ids_with_duplicates_count": len(duplicate_ids),
        "duplication_factor": duplication_factor,
        "duplicate_property_event_ids_examples": duplicate_ids[:20],
        "duplicate_rows_removed_by_join_confidence_tier": dict(sorted(duplicate_tier_counts.items())),
        "duplicate_rows_removed_by_join_method": dict(sorted(duplicate_method_counts.items())),
    }
    return kept_rows, removed_rows, stats


def _distance_sort_key(distance_m: float | None, outcome: OutcomeRecord) -> tuple[Any, ...]:
    distance_token = round(float(distance_m or 0.0), DISTANCE_TIEBREAK_DECIMALS)
    return (distance_token, *_stable_outcome_sort_key(outcome))


def _feature_identity_key(row: FeatureRecord) -> tuple[str, str, str, str, str]:
    return (
        str(row.artifact_path or ""),
        str(row.event_id or ""),
        str(row.record_id or ""),
        str(row.source_record_id or ""),
        str(row.address_norm or ""),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _build_feature_spatial_context(feature_rows: list[FeatureRecord]) -> dict[tuple[str, str, str, str, str], dict[str, float | None]]:
    by_event: dict[str, list[FeatureRecord]] = {}
    all_rows_with_coords: list[FeatureRecord] = []
    for row in feature_rows:
        if row.latitude is None or row.longitude is None:
            continue
        event_key = str(row.event_id or "")
        by_event.setdefault(event_key, []).append(row)
        all_rows_with_coords.append(row)

    for event_key in list(by_event.keys()):
        by_event[event_key] = sorted(by_event[event_key], key=_stable_feature_sort_key)
    all_rows_with_coords = sorted(all_rows_with_coords, key=_stable_feature_sort_key)

    output: dict[tuple[str, str, str, str, str], dict[str, float | None]] = {}
    for row in all_rows_with_coords:
        identity = _feature_identity_key(row)
        event_candidates = by_event.get(str(row.event_id or ""), [])
        candidates = [candidate for candidate in event_candidates if _feature_identity_key(candidate) != identity]
        if not candidates:
            candidates = [candidate for candidate in all_rows_with_coords if _feature_identity_key(candidate) != identity]
        distances_m: list[float] = []
        for candidate in candidates:
            d = _haversine_m(row.latitude, row.longitude, candidate.latitude, candidate.longitude)
            if d is not None:
                distances_m.append(float(d))
        distances_m.sort()
        nearest_m = distances_m[0] if distances_m else None
        nearby_100 = sum(1 for d in distances_m if d <= 30.48)  # 100 ft
        nearby_300 = sum(1 for d in distances_m if d <= 91.44)  # 300 ft
        nearest_ft = float(nearest_m * 3.28084) if nearest_m is not None else None
        nearest_proximity_index = (
            _clamp(100.0 - ((float(nearest_ft) / 300.0) * 100.0), 0.0, 100.0)
            if nearest_ft is not None
            else None
        )
        structure_density_index = _clamp(
            ((min(float(nearby_100), 8.0) / 8.0) * 70.0)
            + ((min(float(nearby_300), 24.0) / 24.0) * 30.0),
            0.0,
            100.0,
        )
        clustering_index = (
            _clamp((0.70 * structure_density_index) + (0.30 * float(nearest_proximity_index)), 0.0, 100.0)
            if nearest_proximity_index is not None
            else structure_density_index
        )
        output[identity] = {
            "nearby_structure_count_100_ft": float(nearby_100),
            "nearby_structure_count_300_ft": float(nearby_300),
            "nearest_structure_distance_ft": (round(float(nearest_ft), 3) if nearest_ft is not None else None),
            "distance_to_nearest_structure_ft": (round(float(nearest_ft), 3) if nearest_ft is not None else None),
            "structure_density": round(float(structure_density_index), 3),
            "structure_density_proxy": round(float(structure_density_index), 3),
            "clustering_index": round(float(clustering_index), 3),
            "local_structure_clustering_index": round(float(clustering_index), 3),
            "spatial_neighbor_count": float(len(distances_m)),
        }
    return output


def _extract_proxy_build_year(
    *,
    feature: FeatureRecord,
    raw_feature_vector: dict[str, Any],
    transformed_feature_vector: dict[str, Any],
) -> float | None:
    input_payload = feature.payload.get("input_payload") if isinstance(feature.payload.get("input_payload"), dict) else {}
    attributes = input_payload.get("attributes") if isinstance(input_payload.get("attributes"), dict) else {}
    candidate_values: list[Any] = [
        raw_feature_vector.get("building_age_proxy_year"),
        transformed_feature_vector.get("building_age_proxy_year"),
        feature.payload.get("building_age_proxy_year"),
        feature.payload.get("year_built"),
        feature.payload.get("built_year"),
        feature.payload.get("construction_year"),
        feature.payload.get("structure_year_built"),
        attributes.get("year_built"),
        attributes.get("built_year"),
        attributes.get("construction_year"),
    ]
    for value in candidate_values:
        numeric = _safe_float(value)
        if numeric is None:
            continue
        if 1800.0 <= float(numeric) <= 2100.0:
            return round(float(numeric), 1)
    return None


def _enrich_feature_vectors_with_property_proxies(
    *,
    feature: FeatureRecord,
    raw_feature_vector: dict[str, Any],
    transformed_feature_vector: dict[str, Any],
    spatial_context: dict[str, float | None] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = dict(raw_feature_vector or {})
    transformed = dict(transformed_feature_vector or {})
    property_context = (
        feature.payload.get("property_level_context")
        if isinstance(feature.payload.get("property_level_context"), dict)
        else {}
    )
    source_by_feature: dict[str, str] = {}

    def _observed(key: str) -> float | None:
        value = _coalesce_float(
            raw.get(key),
            transformed.get(key),
            property_context.get(key),
            feature.payload.get(key),
        )
        if value is not None:
            source_by_feature[key] = "observed"
        return value

    def _set_value(
        key: str,
        value: float | None,
        *,
        source: str,
        mirrored_transformed: bool = True,
    ) -> float | None:
        if value is None:
            source_by_feature.setdefault(key, "missing")
            return None
        numeric = round(float(value), 4)
        raw[key] = numeric
        if mirrored_transformed:
            if _safe_float(transformed.get(key)) is None:
                transformed[key] = numeric
        source_by_feature[key] = source
        return numeric

    def _fuel_to_percent(value: float | None) -> float | None:
        if value is None:
            return None
        return _clamp((float(value) / 40.0) * 100.0, 0.0, 100.0)

    spatial = dict(spatial_context or {})

    for key in (
        "nearby_structure_count_100_ft",
        "nearby_structure_count_300_ft",
        "nearest_structure_distance_ft",
        "distance_to_nearest_structure_ft",
        "structure_density",
        "structure_density_proxy",
        "clustering_index",
        "local_structure_clustering_index",
    ):
        observed = _observed(key)
        if observed is not None:
            _set_value(key, observed, source="observed")
            continue
        spatial_value = _safe_float(spatial.get(key))
        _set_value(key, spatial_value, source=("spatial_proxy" if spatial_value is not None else "missing"))

    build_year = _extract_proxy_build_year(
        feature=feature,
        raw_feature_vector=raw,
        transformed_feature_vector=transformed,
    )
    if build_year is not None:
        _set_value("building_age_proxy_year", build_year, source=("observed" if _observed("building_age_proxy_year") is not None else "derived_proxy"), mirrored_transformed=False)
    else:
        source_by_feature.setdefault("building_age_proxy_year", "missing")

    material_proxy = _observed("building_age_material_proxy_risk")
    if material_proxy is None and build_year is not None:
        event_year = feature.event_year or _parse_year(feature.payload.get("event_date")) or 2026
        building_age = max(0.0, float(event_year) - float(build_year))
        material_proxy = _clamp((building_age / 90.0) * 100.0, 0.0, 100.0)
        _set_value("building_age_material_proxy_risk", material_proxy, source="derived_proxy")
    elif material_proxy is not None:
        _set_value("building_age_material_proxy_risk", material_proxy, source="observed")
    else:
        source_by_feature.setdefault("building_age_material_proxy_risk", "missing")

    canopy_pct = _coalesce_float(raw.get("canopy_cover"), transformed.get("canopy_index"), property_context.get("canopy_cover"))
    fuel_pct = _coalesce_float(
        _fuel_to_percent(_coalesce_float(raw.get("fuel_model"), transformed.get("fuel_index"))),
        transformed.get("fuel_index"),
    )
    structure_density = _coalesce_float(raw.get("structure_density"), transformed.get("structure_density"))
    clustering = _coalesce_float(raw.get("clustering_index"), transformed.get("clustering_index"))
    nearest_structure_ft = _coalesce_float(raw.get("distance_to_nearest_structure_ft"), raw.get("nearest_structure_distance_ft"))
    proximity_index = (
        _clamp(100.0 - ((float(nearest_structure_ft) / 300.0) * 100.0), 0.0, 100.0)
        if nearest_structure_ft is not None
        else None
    )

    ring_0_5_estimate = _clamp(
        (
            (0.42 * float(canopy_pct if canopy_pct is not None else 45.0))
            + (0.26 * float(fuel_pct if fuel_pct is not None else 45.0))
            + (0.20 * float(proximity_index if proximity_index is not None else (structure_density or 45.0)))
            + (0.12 * float(clustering if clustering is not None else (structure_density or 45.0)))
        ),
        0.0,
        100.0,
    )
    ring_0_5 = _observed("ring_0_5_ft_vegetation_density")
    if ring_0_5 is None:
        ring_0_5 = ring_0_5_estimate
        _set_value("ring_0_5_ft_vegetation_density", ring_0_5, source="derived_proxy")
    else:
        _set_value("ring_0_5_ft_vegetation_density", ring_0_5, source="observed")
    _set_value(
        "ring_0_5_ft_vegetation_density_proxy_blend",
        _clamp((0.70 * float(ring_0_5)) + (0.30 * float(ring_0_5_estimate)), 0.0, 100.0),
        source="derived_proxy",
        mirrored_transformed=False,
    )

    near_0_5 = _observed("near_structure_vegetation_0_5_pct")
    if near_0_5 is None:
        _set_value("near_structure_vegetation_0_5_pct", ring_0_5, source="derived_proxy")
    else:
        _set_value("near_structure_vegetation_0_5_pct", near_0_5, source="observed")

    ring_5_30_estimate = _clamp(
        max(
            float(ring_0_5 or 0.0) + 4.0,
            (0.48 * float(canopy_pct if canopy_pct is not None else 45.0))
            + (0.22 * float(fuel_pct if fuel_pct is not None else 45.0))
            + (0.18 * float(structure_density if structure_density is not None else 45.0))
            + (0.12 * float(clustering if clustering is not None else 45.0)),
        ),
        0.0,
        100.0,
    )
    ring_5_30 = _observed("ring_5_30_ft_vegetation_density")
    if ring_5_30 is None:
        ring_5_30 = ring_5_30_estimate
        _set_value("ring_5_30_ft_vegetation_density", ring_5_30, source="derived_proxy")
    else:
        _set_value("ring_5_30_ft_vegetation_density", ring_5_30, source="observed")
    _set_value(
        "ring_5_30_ft_vegetation_density_proxy_blend",
        _clamp((0.70 * float(ring_5_30)) + (0.30 * float(ring_5_30_estimate)), 0.0, 100.0),
        source="derived_proxy",
        mirrored_transformed=False,
    )

    canopy_adj = _observed("canopy_adjacency_proxy_pct")
    if canopy_adj is None:
        canopy_adj = _clamp(
            (0.55 * float(ring_0_5 or 0.0))
            + (0.45 * float(structure_density if structure_density is not None else 45.0)),
            0.0,
            100.0,
        )
        _set_value("canopy_adjacency_proxy_pct", canopy_adj, source="derived_proxy")
    else:
        _set_value("canopy_adjacency_proxy_pct", canopy_adj, source="observed")

    continuity = _observed("vegetation_continuity_proxy_pct")
    if continuity is None:
        continuity = _clamp(
            (0.40 * float(ring_5_30 or 0.0))
            + (0.35 * float(ring_0_5 or 0.0))
            + (0.25 * float(canopy_adj or 0.0)),
            0.0,
            100.0,
        )
        _set_value("vegetation_continuity_proxy_pct", continuity, source="derived_proxy")
    else:
        _set_value("vegetation_continuity_proxy_pct", continuity, source="observed")

    connectivity = _observed("near_structure_connectivity_index")
    if connectivity is None:
        connectivity = _clamp(
            (0.45 * float(ring_0_5 or 0.0))
            + (0.35 * float(ring_5_30 or 0.0))
            + (0.20 * float(continuity or 0.0)),
            0.0,
            100.0,
        )
        _set_value("near_structure_connectivity_index", connectivity, source="derived_proxy")
    else:
        _set_value("near_structure_connectivity_index", connectivity, source="observed")
    connectivity_proxy_blend = _clamp(
        (0.50 * float(connectivity or 0.0))
        + (0.30 * float(ring_5_30_estimate))
        + (0.20 * float(ring_0_5_estimate)),
        0.0,
        100.0,
    )
    _set_value(
        "near_structure_connectivity_index_proxy_blend",
        connectivity_proxy_blend,
        source="derived_proxy",
        mirrored_transformed=False,
    )

    high_fuel_dist = _observed("nearest_high_fuel_patch_distance_ft")
    if high_fuel_dist is None:
        high_fuel_dist = _clamp(
            25.0 + ((100.0 - float(max(connectivity or 0.0, ring_5_30 or 0.0))) * 7.5),
            0.0,
            1000.0,
        )
        _set_value("nearest_high_fuel_patch_distance_ft", high_fuel_dist, source="derived_proxy")
    else:
        _set_value("nearest_high_fuel_patch_distance_ft", high_fuel_dist, source="observed")
    high_fuel_dist_proxy_blend = _clamp(
        25.0 + ((100.0 - float(max(connectivity_proxy_blend, ring_5_30_estimate))) * 7.5),
        0.0,
        1000.0,
    )
    _set_value(
        "nearest_high_fuel_patch_distance_ft_proxy_blend",
        high_fuel_dist_proxy_blend,
        source="derived_proxy",
        mirrored_transformed=False,
    )

    defensible_proxy = _observed("defensible_space_proxy_score")
    if defensible_proxy is None:
        defensible_proxy = _clamp(100.0 - ((0.72 * float(ring_0_5 or 0.0)) + (0.28 * float(ring_5_30 or 0.0))), 0.0, 100.0)
        _set_value("defensible_space_proxy_score", defensible_proxy, source="derived_proxy")
    else:
        _set_value("defensible_space_proxy_score", defensible_proxy, source="observed")
    defensible_proxy_blend = _clamp(
        100.0 - ((0.72 * float(ring_0_5_estimate)) + (0.28 * float(ring_5_30_estimate))),
        0.0,
        100.0,
    )
    _set_value(
        "defensible_space_proxy_score_blend",
        defensible_proxy_blend,
        source="derived_proxy",
        mirrored_transformed=False,
    )

    observed_fields = sorted([key for key, source in source_by_feature.items() if source == "observed"])
    inferred_fields = sorted([key for key, source in source_by_feature.items() if source in {"spatial_proxy", "derived_proxy"}])
    missing_fields = sorted([key for key, source in source_by_feature.items() if source == "missing"])
    feature_observation_summary = {
        "source_by_feature": dict(sorted(source_by_feature.items())),
        "observed_fields": observed_fields,
        "inferred_fields": inferred_fields,
        "missing_fields": missing_fields,
        "observed_count": len(observed_fields),
        "inferred_count": len(inferred_fields),
        "missing_count": len(missing_fields),
    }
    return raw, transformed, feature_observation_summary


def _sorted_outcomes(rows: list[OutcomeRecord]) -> list[OutcomeRecord]:
    return sorted(rows, key=_stable_outcome_sort_key)


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


def _event_year_consistent_with_tolerance(
    feature: FeatureRecord,
    outcome: OutcomeRecord,
    tolerance_years: int,
) -> bool:
    if feature.event_year is None or outcome.event_year is None:
        return True
    return abs(int(feature.event_year) - int(outcome.event_year)) <= max(0, int(tolerance_years))


def _outcome_identity_key(row: OutcomeRecord) -> str:
    source_name = str(row.payload.get("source_name") or "").strip()
    if source_name and row.source_record_id:
        return f"source::{source_name}::{row.source_record_id}"
    if row.event_id and row.record_id:
        return f"event_record::{row.event_id}::{row.record_id}"
    if row.event_id and row.latitude is not None and row.longitude is not None:
        return f"event_coord::{row.event_id}::{round(float(row.latitude), 5)}::{round(float(row.longitude), 5)}"
    return f"fallback::{row.event_id}::{row.record_id}::{row.source_record_id}"


def _join_confidence_for_distance(distance_m: float | None, max_distance_m: float) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= 0.0:
        return 0.75
    ratio = min(1.0, distance_m / max_distance_m)
    return max(0.3, 0.75 - (0.35 * ratio))


def _distance_match_tier(
    *,
    distance_m: float | None,
    exact_match_distance_m: float,
    near_match_distance_m: float,
    extended_match_distance_m: float,
) -> str:
    if distance_m is None:
        return "none"
    if distance_m <= max(0.0, float(exact_match_distance_m)):
        return "exact"
    if distance_m <= max(0.0, float(near_match_distance_m)):
        return "near"
    if distance_m <= max(0.0, float(extended_match_distance_m)):
        return "extended"
    return "outside"


def _spatial_cell_key(lat: float, lon: float, *, cell_size_deg: float = 0.01) -> tuple[int, int]:
    return (int(math.floor(float(lat) / cell_size_deg)), int(math.floor(float(lon) / cell_size_deg)))


def _build_spatial_index(rows: list[OutcomeRecord], *, cell_size_deg: float = 0.01) -> dict[tuple[int, int], list[OutcomeRecord]]:
    index: dict[tuple[int, int], list[OutcomeRecord]] = {}
    for row in rows:
        if row.latitude is None or row.longitude is None:
            continue
        key = _spatial_cell_key(row.latitude, row.longitude, cell_size_deg=cell_size_deg)
        index.setdefault(key, []).append(row)
    for key, values in list(index.items()):
        index[key] = _sorted_outcomes(values)
    return index


def _spatial_candidates_within_radius(
    *,
    feature: FeatureRecord,
    radius_m: float,
    fallback_candidates: list[OutcomeRecord],
    spatial_index: dict[tuple[int, int], list[OutcomeRecord]] | None = None,
    cell_size_deg: float = 0.01,
) -> list[OutcomeRecord]:
    if not spatial_index or feature.latitude is None or feature.longitude is None or radius_m <= 0:
        return fallback_candidates
    lat = float(feature.latitude)
    lon = float(feature.longitude)
    lat_deg = float(radius_m) / 111_320.0
    lon_scale = max(0.2, math.cos(math.radians(lat)))
    lon_deg = float(radius_m) / (111_320.0 * lon_scale)
    lat_steps = max(1, int(math.ceil(lat_deg / cell_size_deg)))
    lon_steps = max(1, int(math.ceil(lon_deg / cell_size_deg)))
    center_i, center_j = _spatial_cell_key(lat, lon, cell_size_deg=cell_size_deg)
    out: list[OutcomeRecord] = []
    seen: set[str] = set()
    for di in range(-lat_steps, lat_steps + 1):
        for dj in range(-lon_steps, lon_steps + 1):
            cell_rows = spatial_index.get((center_i + di, center_j + dj), [])
            for row in cell_rows:
                row_key = _outcome_identity_key(row)
                if row_key in seen:
                    continue
                seen.add(row_key)
                out.append(row)
    return out if out else fallback_candidates


def _filter_unused_outcomes(
    rows: list[OutcomeRecord],
    *,
    excluded_outcome_keys: set[str] | None,
) -> list[OutcomeRecord]:
    if not excluded_outcome_keys:
        return _sorted_outcomes(rows)
    return _sorted_outcomes([row for row in rows if _outcome_identity_key(row) not in excluded_outcome_keys])


def _pick_nearest_within_radius(
    feature: FeatureRecord,
    candidates: list[OutcomeRecord],
    *,
    radius_m: float,
    spatial_index: dict[tuple[int, int], list[OutcomeRecord]] | None = None,
) -> tuple[OutcomeRecord | None, float | None, int]:
    if radius_m <= 0:
        return None, None, 0
    search_candidates = _spatial_candidates_within_radius(
        feature=feature,
        radius_m=radius_m,
        fallback_candidates=candidates,
        spatial_index=spatial_index,
    )
    within: list[tuple[OutcomeRecord, float]] = []
    for row in search_candidates:
        d = _haversine_m(feature.latitude, feature.longitude, row.latitude, row.longitude)
        if d is None:
            continue
        if d <= radius_m:
            within.append((row, d))
    if not within:
        return None, None, 0
    within.sort(key=lambda item: _distance_sort_key(item[1], item[0]))
    chosen, distance_m = within[0]
    return chosen, distance_m, len(within)


def _pick_nearest(feature: FeatureRecord, candidates: list[OutcomeRecord]) -> tuple[OutcomeRecord | None, float | None]:
    if not candidates:
        return None, None
    best: OutcomeRecord | None = None
    best_distance: float | None = None
    best_key: tuple[Any, ...] | None = None
    for row in candidates:
        d = _haversine_m(feature.latitude, feature.longitude, row.latitude, row.longitude)
        if d is None:
            continue
        candidate_key = _distance_sort_key(d, row)
        if best is None or best_key is None or candidate_key < best_key:
            best = row
            best_distance = d
            best_key = candidate_key
    return best, best_distance


def _assess_join_tier(score: float) -> str:
    if score >= HIGH_CONFIDENCE_SCORE_THRESHOLD:
        return "high"
    if score >= 0.7:
        return "moderate"
    return "low"


def _tier_rank(tier: str) -> int:
    normalized = str(tier or "").strip().lower()
    if normalized == "high":
        return 2
    if normalized == "moderate":
        return 1
    return 0


def _tier_name(rank: int) -> str:
    if rank >= 2:
        return "high"
    if rank >= 1:
        return "moderate"
    return "low"


def _distance_join_tier(
    *,
    distance_m: float | None,
    high_confidence_distance_m: float,
    medium_confidence_distance_m: float,
) -> str:
    if distance_m is None:
        return "low"
    if distance_m <= max(0.0, float(high_confidence_distance_m)):
        return "high"
    if distance_m <= max(float(high_confidence_distance_m), float(medium_confidence_distance_m)):
        return "moderate"
    return "low"


def _join_tier_from_score_and_distance(
    *,
    score: float,
    distance_m: float | None,
    high_confidence_distance_m: float,
    medium_confidence_distance_m: float,
    max_allowed_tier: str = "high",
    prefer_score_for_high: bool = False,
) -> str:
    by_score = _assess_join_tier(float(score))
    by_distance = _distance_join_tier(
        distance_m=distance_m,
        high_confidence_distance_m=high_confidence_distance_m,
        medium_confidence_distance_m=medium_confidence_distance_m,
    )
    if distance_m is not None and prefer_score_for_high:
        resolved_rank = max(_tier_rank(by_score), _tier_rank(by_distance))
    else:
        # Default path keeps distance-led tiers where distance is known.
        resolved_rank = _tier_rank(by_distance) if distance_m is not None else _tier_rank(by_score)
    capped_rank = min(resolved_rank, _tier_rank(max_allowed_tier))
    return _tier_name(capped_rank)


def _join_max_allowed_tier_for_method(method: str) -> str:
    normalized = str(method or "").strip().lower()
    if normalized in {"nearest_global_coordinates", "unmatched"}:
        return "low"
    if normalized in {"nearest_event_name_coordinates_tolerant_year", "approx_global_address_token_overlap"}:
        return "moderate"
    return "high"


def _build_join_confidence_debug(
    *,
    score: float,
    distance_m: float | None,
    high_confidence_distance_m: float,
    medium_confidence_distance_m: float,
    max_allowed_tier: str,
    resolved_tier: str,
) -> dict[str, Any]:
    by_score = _assess_join_tier(float(score))
    by_distance = _distance_join_tier(
        distance_m=distance_m,
        high_confidence_distance_m=high_confidence_distance_m,
        medium_confidence_distance_m=medium_confidence_distance_m,
    )
    reasons: list[str] = []
    if str(resolved_tier or "").strip().lower() != "high":
        if distance_m is None:
            reasons.append("distance_missing")
        else:
            if float(distance_m) > float(high_confidence_distance_m):
                if float(distance_m) <= float(medium_confidence_distance_m):
                    reasons.append("distance_between_high_and_medium_threshold")
                else:
                    reasons.append("distance_above_medium_threshold")
        if float(score) < float(HIGH_CONFIDENCE_SCORE_THRESHOLD):
            reasons.append("score_below_high_threshold")
        if _tier_rank(max_allowed_tier) < _tier_rank("high"):
            reasons.append("max_allowed_tier_cap")
    near_high_distance = (
        distance_m is not None
        and float(distance_m) > float(high_confidence_distance_m)
        and float(distance_m) <= float(high_confidence_distance_m + HIGH_CONFIDENCE_DISTANCE_NEAR_MISS_MARGIN_M)
    )
    just_below_high_score = (
        float(score) < float(HIGH_CONFIDENCE_SCORE_THRESHOLD)
        and float(score) >= float(HIGH_CONFIDENCE_SCORE_THRESHOLD - HIGH_CONFIDENCE_SCORE_NEAR_MISS_MARGIN)
    )
    return {
        "score": round(float(score), 4),
        "distance_m": (round(float(distance_m), 3) if distance_m is not None else None),
        "high_confidence_distance_threshold_m": float(high_confidence_distance_m),
        "medium_confidence_distance_threshold_m": float(medium_confidence_distance_m),
        "high_confidence_score_threshold": float(HIGH_CONFIDENCE_SCORE_THRESHOLD),
        "by_score_tier": by_score,
        "by_distance_tier": by_distance,
        "max_allowed_tier": str(max_allowed_tier),
        "resolved_tier": str(resolved_tier),
        "near_high_distance_threshold": bool(near_high_distance),
        "just_below_high_score_threshold": bool(just_below_high_score),
        "non_high_reason_codes": sorted(set(reasons)),
    }


def _derive_row_confidence_tier(
    *,
    join_confidence_tier: str,
    model_confidence_tier: str,
    evidence_quality_tier: str,
) -> str:
    join_tier = str(join_confidence_tier or "").strip().lower()
    model_tier = str(model_confidence_tier or "").strip().lower()
    evidence_tier = str(evidence_quality_tier or "").strip().lower()
    if join_tier == "low":
        return "low-confidence"
    # Row confidence is join-centric: strong geospatial/id/address match should remain high-confidence
    # even when model/evidence tiers are provisional.
    if join_tier == "high":
        return "high-confidence"
    if model_tier in {"low", "preliminary"} or evidence_tier in {"low", "preliminary"}:
        return "low-confidence"
    return "medium-confidence"


def _derive_fallback_usage(
    *,
    evidence_summary: dict[str, Any] | None,
    coverage_summary: dict[str, Any] | None,
    evidence_tier: str,
) -> dict[str, Any]:
    evidence_summary = evidence_summary if isinstance(evidence_summary, dict) else {}
    coverage_summary = coverage_summary if isinstance(coverage_summary, dict) else {}

    observed_factor_count = int(_safe_float(evidence_summary.get("observed_factor_count")) or 0)
    inferred_factor_count = int(_safe_float(evidence_summary.get("inferred_factor_count")) or 0)
    fallback_factor_count = int(_safe_float(evidence_summary.get("fallback_factor_count")) or 0)
    missing_factor_count = int(_safe_float(evidence_summary.get("missing_factor_count")) or 0)
    coverage_failed_count = int(_safe_float(coverage_summary.get("failed_count")) or 0)
    coverage_fallback_count = int(_safe_float(coverage_summary.get("fallback_count")) or 0)
    fallback_weight_fraction = float(_safe_float(evidence_summary.get("fallback_weight_fraction")) or 0.0)

    total_factor_count = max(
        1,
        observed_factor_count + inferred_factor_count + fallback_factor_count + missing_factor_count,
    )
    fallback_factor_ratio = float(fallback_factor_count) / float(total_factor_count)
    missing_factor_ratio = float(missing_factor_count) / float(total_factor_count)
    evidence_tier_norm = str(evidence_tier or "").strip().lower()

    fallback_heavy_reasons: list[str] = []
    if evidence_tier_norm in {"low", "preliminary", "fallback_heavy"}:
        fallback_heavy_reasons.append("low_evidence_tier")
    if coverage_failed_count > 0:
        fallback_heavy_reasons.append("coverage_failed")
    if fallback_weight_fraction >= FALLBACK_HEAVY_WEIGHT_THRESHOLD:
        fallback_heavy_reasons.append("high_fallback_weight_fraction")
    if (
        fallback_factor_count >= 2
        and fallback_factor_ratio >= FALLBACK_HEAVY_FACTOR_RATIO_THRESHOLD
    ):
        fallback_heavy_reasons.append("high_fallback_factor_ratio")
    if (
        coverage_fallback_count >= FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD
        and fallback_weight_fraction >= FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD
    ):
        fallback_heavy_reasons.append("multi_layer_fallback_with_elevated_weight")
    if (
        missing_factor_count >= 3
        and missing_factor_ratio >= FALLBACK_HEAVY_MISSING_RATIO_THRESHOLD
    ):
        fallback_heavy_reasons.append("high_missing_factor_ratio")

    fallback_heavy = bool(fallback_heavy_reasons)
    if fallback_heavy:
        classification = "fallback_heavy"
    elif (
        (
            fallback_factor_count == 0
            and missing_factor_count <= 1
            and coverage_failed_count == 0
            and fallback_weight_fraction < 0.35
            and observed_factor_count >= max(3, inferred_factor_count + 1)
        )
        or (
            evidence_tier_norm == "high"
            and fallback_factor_count <= 1
            and missing_factor_count <= 1
            and coverage_failed_count == 0
            and fallback_weight_fraction < FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD
        )
    ):
        classification = "high_evidence"
    else:
        classification = "mixed_evidence"

    return {
        "classification": classification,
        "fallback_heavy": fallback_heavy,
        "fallback_heavy_reasons": sorted(set(fallback_heavy_reasons)),
        "observed_factor_count": observed_factor_count,
        "inferred_factor_count": inferred_factor_count,
        "fallback_factor_count": fallback_factor_count,
        "missing_factor_count": missing_factor_count,
        "coverage_failed_count": coverage_failed_count,
        "coverage_fallback_count": coverage_fallback_count,
        "fallback_weight_fraction": round(fallback_weight_fraction, 4),
        "fallback_factor_ratio": round(fallback_factor_ratio, 4),
        "missing_factor_ratio": round(missing_factor_ratio, 4),
    }


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
    by_event_name: dict[str, list[OutcomeRecord]] = {}
    by_event_name_year: dict[str, list[OutcomeRecord]] = {}
    spatial_by_event: dict[str, dict[tuple[int, int], list[OutcomeRecord]]] = {}
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
        if row.event_name_norm:
            by_event_name.setdefault(row.event_name_norm, []).append(row)
        if row.event_name_norm and row.event_year is not None:
            by_event_name_year.setdefault(f"{row.event_name_norm}|{row.event_year}", []).append(row)
    for event_id, event_rows in by_event.items():
        spatial_by_event[event_id] = _build_spatial_index(event_rows)
    for mapping in (
        by_source_record,
        by_event_record,
        by_parcel_event,
        by_address_event,
        by_event,
        by_event_name,
        by_event_name_year,
    ):
        for key, values in list(mapping.items()):
            mapping[key] = _sorted_outcomes(values)
    return {
        "by_source_record": by_source_record,
        "by_event_record": by_event_record,
        "by_parcel_event": by_parcel_event,
        "by_address_event": by_address_event,
        "by_event": by_event,
        "by_event_name": by_event_name,
        "by_event_name_year": by_event_name_year,
        "spatial_by_event": spatial_by_event,
        "spatial_all": _build_spatial_index(outcomes),
        "all": outcomes,
    }


def _choose_outcome(
    feature: FeatureRecord,
    indexes: dict[str, Any],
    *,
    join_config: JoinConfig,
    excluded_outcome_keys: set[str] | None = None,
) -> tuple[OutcomeRecord | None, dict[str, Any]]:
    join_config_near = max(float(join_config.exact_match_distance_m), float(join_config.near_match_distance_m))
    join_config_extended = max(join_config_near, float(join_config.max_distance_m))
    event_spatial_index = (
        indexes["spatial_by_event"].get(feature.event_id)
        if feature.event_id and isinstance(indexes.get("spatial_by_event"), dict)
        else None
    )
    global_spatial_index = indexes.get("spatial_all") if isinstance(indexes.get("spatial_all"), dict) else None

    # 1) Exact parcel + event.
    if feature.event_id and feature.parcel_id:
        rows = _filter_unused_outcomes(
            indexes["by_parcel_event"].get(f"{feature.event_id}|{feature.parcel_id}", []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        if rows:
            return rows[0], {
                "join_method": "exact_parcel_event",
                "join_confidence_score": 0.98,
                "join_confidence_tier": "high",
                "join_distance_m": 0.0,
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(rows),
                "match_tier": "exact",
            }

    # 2) Exact source record id.
    if feature.source_record_id:
        rows = _filter_unused_outcomes(
            indexes["by_source_record"].get(feature.source_record_id, []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        if rows:
            chosen = rows[0]
            score = 0.97 if _event_year_consistent_with_tolerance(feature, chosen, join_config.event_year_tolerance_years) else 0.82
            distance_m = _haversine_m(feature.latitude, feature.longitude, chosen.latitude, chosen.longitude)
            return chosen, {
                "join_method": "exact_source_record_id",
                "join_confidence_score": score,
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="high",
                    prefer_score_for_high=True,
                ),
                "join_distance_m": distance_m,
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(rows),
                "match_tier": "exact",
            }

    # 3) Exact event+record id.
    if feature.event_id and feature.record_id:
        rows = _filter_unused_outcomes(
            indexes["by_event_record"].get(f"{feature.event_id}|{feature.record_id}", []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        if rows:
            chosen = rows[0]
            return chosen, {
                "join_method": "exact_event_record_id",
                "join_confidence_score": 0.96,
                "join_confidence_tier": "high",
                "join_distance_m": _haversine_m(feature.latitude, feature.longitude, chosen.latitude, chosen.longitude),
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(rows),
                "match_tier": "exact",
            }

    # 4) Event+address.
    if feature.event_id and feature.address_norm:
        rows = _filter_unused_outcomes(
            indexes["by_address_event"].get(f"{feature.event_id}|{feature.address_norm}", []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        if rows:
            chosen = rows[0]
            return chosen, {
                "join_method": "exact_event_address",
                "join_confidence_score": 0.90,
                "join_confidence_tier": "high",
                "join_distance_m": _haversine_m(feature.latitude, feature.longitude, chosen.latitude, chosen.longitude),
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(rows),
                "match_tier": "exact",
            }

    # 5) Approximate event+address token overlap fallback.
    if feature.event_id and feature.address_norm:
        candidates = _filter_unused_outcomes(
            indexes["by_event"].get(feature.event_id, []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        best: OutcomeRecord | None = None
        best_overlap = 0.0
        best_key: tuple[Any, ...] | None = None
        for row in candidates:
            overlap = _token_overlap_ratio(feature.address_norm, row.address_norm)
            candidate_key = _stable_outcome_sort_key(row)
            if overlap > best_overlap or (math.isclose(overlap, best_overlap) and (best_key is None or candidate_key < best_key)):
                best_overlap = overlap
                best = row
                best_key = candidate_key
        if best is not None and best_overlap >= max(0.35, min(1.0, float(join_config.address_token_overlap_min))):
            score = min(0.9, max(0.55, 0.55 + 0.35 * best_overlap))
            distance_m = _haversine_m(feature.latitude, feature.longitude, best.latitude, best.longitude)
            return best, {
                "join_method": "approx_event_address_token_overlap",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="high",
                ),
                "join_distance_m": distance_m,
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(candidates),
                "address_overlap_ratio": round(best_overlap, 4),
                "match_tier": "near",
            }

    # 6) Exact event coordinates within strict radius.
    if feature.event_id:
        candidates = _filter_unused_outcomes(
            indexes["by_event"].get(feature.event_id, []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        chosen, distance_m, candidate_count = _pick_nearest_within_radius(
            feature,
            candidates,
            radius_m=float(join_config.exact_match_distance_m),
            spatial_index=event_spatial_index,
        )
        if chosen is not None and distance_m is not None:
            return chosen, {
                "join_method": "exact_event_coordinates",
                "join_confidence_score": 0.94,
                "join_confidence_tier": "high",
                "join_distance_m": round(distance_m, 2),
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(candidates),
                "match_tier": "exact",
                "radius_candidate_count": candidate_count,
            }

    # 7) Buffered event coordinates.
    if feature.event_id:
        candidates = _filter_unused_outcomes(
            indexes["by_event"].get(feature.event_id, []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        buffer_radius = min(float(join_config.buffer_match_radius_m), join_config_extended)
        chosen, distance_m, candidate_count = _pick_nearest_within_radius(
            feature,
            candidates,
            radius_m=buffer_radius,
            spatial_index=event_spatial_index,
        )
        if (
            chosen is not None
            and distance_m is not None
            and distance_m > float(join_config.exact_match_distance_m)
        ):
            score = max(0.68, _join_confidence_for_distance(distance_m, join_config.max_distance_m))
            match_tier = _distance_match_tier(
                distance_m=distance_m,
                exact_match_distance_m=join_config.exact_match_distance_m,
                near_match_distance_m=join_config_near,
                extended_match_distance_m=join_config_extended,
            )
            return chosen, {
                "join_method": "buffered_event_coordinates",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="high",
                ),
                "join_distance_m": round(distance_m, 2),
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(candidates),
                "match_tier": match_tier if match_tier != "outside" else "near",
                "buffer_radius_m": round(buffer_radius, 2),
                "buffer_candidate_count": candidate_count,
            }

    # 8) Nearest within event: near and extended tiers.
    if feature.event_id:
        candidates = _filter_unused_outcomes(
            indexes["by_event"].get(feature.event_id, []),
            excluded_outcome_keys=excluded_outcome_keys,
        )
        search_candidates = _spatial_candidates_within_radius(
            feature=feature,
            radius_m=join_config_extended,
            fallback_candidates=candidates,
            spatial_index=event_spatial_index,
        )
        chosen, distance_m = _pick_nearest(feature, search_candidates)
        if chosen is not None and distance_m is not None and distance_m <= join_config_extended:
            score = _join_confidence_for_distance(distance_m, join_config.max_distance_m)
            distance_tier = _distance_match_tier(
                distance_m=distance_m,
                exact_match_distance_m=join_config.exact_match_distance_m,
                near_match_distance_m=join_config_near,
                extended_match_distance_m=join_config_extended,
            )
            if distance_tier == "exact":
                method = "exact_event_coordinates"
                score = max(score, 0.93)
            elif distance_tier == "near":
                method = "near_event_coordinates"
                score = max(score, 0.74)
            else:
                method = "extended_event_coordinates"
                score = max(0.45, score - 0.08)
            return chosen, {
                "join_method": method,
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="high",
                ),
                "join_distance_m": round(distance_m, 2),
                "join_confidence_max_allowed_tier": "high",
                "diagnostic_candidate_pool_count": len(search_candidates),
                "match_tier": distance_tier,
            }

    # 9) Nearest by event name/year within tolerance.
    if feature.event_name_norm and feature.event_year is not None:
        candidates = [
            row
            for row in indexes["by_event_name"].get(feature.event_name_norm, [])
            if (
                _event_year_consistent_with_tolerance(feature, row, join_config.event_year_tolerance_years)
                and (not excluded_outcome_keys or _outcome_identity_key(row) not in excluded_outcome_keys)
            )
        ]
        chosen, distance_m = _pick_nearest(feature, candidates)
        if chosen is not None and distance_m is not None and distance_m <= join_config_extended:
            score = max(0.55, _join_confidence_for_distance(distance_m, join_config.max_distance_m) - 0.05)
            distance_tier = _distance_match_tier(
                distance_m=distance_m,
                exact_match_distance_m=join_config.exact_match_distance_m,
                near_match_distance_m=join_config_near,
                extended_match_distance_m=join_config_extended,
            )
            if distance_tier == "near":
                score = max(score, 0.7)
            elif distance_tier == "extended":
                score = max(0.4, score - 0.05)
            return chosen, {
                "join_method": "nearest_event_name_coordinates_tolerant_year",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="moderate",
                ),
                "join_distance_m": round(distance_m, 2),
                "join_confidence_max_allowed_tier": "moderate",
                "diagnostic_candidate_pool_count": len(candidates),
                "match_tier": distance_tier if distance_tier != "outside" else "extended",
            }

    # 10) Approximate global address overlap fallback.
    if feature.address_norm:
        best: OutcomeRecord | None = None
        best_overlap = 0.0
        best_key: tuple[Any, ...] | None = None
        for row in _filter_unused_outcomes(indexes["all"], excluded_outcome_keys=excluded_outcome_keys):
            overlap = _token_overlap_ratio(feature.address_norm, row.address_norm)
            candidate_key = _stable_outcome_sort_key(row)
            if overlap > best_overlap or (math.isclose(overlap, best_overlap) and (best_key is None or candidate_key < best_key)):
                best_overlap = overlap
                best = row
                best_key = candidate_key
        overlap_threshold = max(0.45, min(1.0, float(join_config.address_token_overlap_min)))
        if best is not None and best_overlap >= overlap_threshold:
            score = min(0.72, max(0.38, 0.35 + 0.45 * best_overlap))
            distance_m = _haversine_m(feature.latitude, feature.longitude, best.latitude, best.longitude)
            return best, {
                "join_method": "approx_global_address_token_overlap",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="moderate",
                ),
                "join_distance_m": distance_m,
                "join_confidence_max_allowed_tier": "moderate",
                "diagnostic_candidate_pool_count": len(indexes["all"]),
                "address_overlap_ratio": round(best_overlap, 4),
                "match_tier": "fallback",
            }

    # 11) Global nearest as low confidence fallback (optional).
    if join_config.enable_global_nearest_fallback:
        candidates = _filter_unused_outcomes(indexes["all"], excluded_outcome_keys=excluded_outcome_keys)
        search_candidates = _spatial_candidates_within_radius(
            feature=feature,
            radius_m=float(join_config.global_max_distance_m),
            fallback_candidates=candidates,
            spatial_index=global_spatial_index,
        )
        chosen, distance_m = _pick_nearest(feature, search_candidates)
        if chosen is not None and distance_m is not None and distance_m <= join_config.global_max_distance_m:
            score = max(0.30, _join_confidence_for_distance(distance_m, join_config.global_max_distance_m) - 0.20)
            return chosen, {
                "join_method": "nearest_global_coordinates",
                "join_confidence_score": round(score, 4),
                "join_confidence_tier": _join_tier_from_score_and_distance(
                    score=score,
                    distance_m=distance_m,
                    high_confidence_distance_m=join_config.high_confidence_distance_m,
                    medium_confidence_distance_m=join_config.medium_confidence_distance_m,
                    max_allowed_tier="low",
                ),
                "join_distance_m": round(distance_m, 2),
                "join_confidence_max_allowed_tier": "low",
                "diagnostic_candidate_pool_count": len(search_candidates),
                "match_tier": "fallback",
            }
    return None, {
        "join_method": "unmatched",
        "join_confidence_score": 0.0,
        "join_confidence_tier": "low",
        "join_distance_m": None,
        "join_confidence_max_allowed_tier": "low",
        "diagnostic_candidate_pool_count": (
            len(indexes["by_event"].get(feature.event_id, []))
            if feature.event_id and isinstance(indexes.get("by_event"), dict)
            else len(indexes.get("all", []))
        ),
        "match_tier": "none",
        "unmatched_reason": (
            "no_unused_outcome_match_within_constraints"
            if excluded_outcome_keys
            else "no_outcome_match_within_constraints"
        ),
    }


def _load_outcomes(path: Path) -> tuple[list[OutcomeRecord], dict[str, int]]:
    payload = _load_json(path)
    rows, stats = _extract_rows_with_filter_stats(payload)
    return ([_as_outcome_record(row) for row in rows], stats)


def _outcome_dedupe_key(row: OutcomeRecord) -> str:
    return _outcome_identity_key(row)


def _load_outcomes_from_paths(paths: list[Path]) -> tuple[list[OutcomeRecord], dict[str, int], dict[str, Any]]:
    deduped: dict[str, OutcomeRecord] = {}
    per_source_loaded: dict[str, int] = {}
    source_filter_stats: dict[str, dict[str, int]] = {}
    duplicate_outcome_rows_removed_count = 0
    for path in sorted({Path(path).expanduser() for path in paths}, key=lambda token: str(token)):
        p = Path(path).expanduser()
        rows, stats = _load_outcomes(p)
        per_source_loaded[str(p)] = len(rows)
        source_filter_stats[str(p)] = stats
        for row in rows:
            dedupe_key = _outcome_dedupe_key(row)
            if dedupe_key in deduped:
                duplicate_outcome_rows_removed_count += 1
            deduped[dedupe_key] = row
    merged = sorted(
        deduped.values(),
        key=lambda row: (row.event_id, row.event_year or 0, row.record_id, row.source_record_id),
    )
    total_raw_row_count = sum(int((stats or {}).get("raw_row_count") or 0) for stats in source_filter_stats.values())
    total_invalid_non_dict_row_count = sum(int((stats or {}).get("invalid_non_dict_row_count") or 0) for stats in source_filter_stats.values())
    total_invalid_records_container_count = sum(
        int((stats or {}).get("invalid_records_container_count") or 0) for stats in source_filter_stats.values()
    )
    diagnostics = {
        "source_filter_stats": dict(sorted(source_filter_stats.items())),
        "total_raw_row_count": total_raw_row_count,
        "total_valid_row_count_before_dedupe": sum(per_source_loaded.values()),
        "total_invalid_non_dict_row_count": total_invalid_non_dict_row_count,
        "total_invalid_records_container_count": total_invalid_records_container_count,
        "duplicate_outcome_rows_removed_count": duplicate_outcome_rows_removed_count,
        "total_outcome_rows_after_dedupe": len(merged),
        "outcome_prejoin_filter_reason_counts": {
            "invalid_non_dict_outcome_row": total_invalid_non_dict_row_count,
            "invalid_outcome_records_container": total_invalid_records_container_count,
            "duplicate_outcome_identity": duplicate_outcome_rows_removed_count,
        },
    }
    return merged, per_source_loaded, diagnostics


def _load_feature_records(paths: list[Path]) -> tuple[list[FeatureRecord], list[dict[str, Any]], dict[str, Any]]:
    records: list[FeatureRecord] = []
    missing_artifacts: list[dict[str, Any]] = []
    source_filter_stats: dict[str, dict[str, int]] = {}
    invalid_feature_records_container_count = 0
    for path in sorted({Path(path).expanduser() for path in paths}, key=lambda token: str(token)):
        p = Path(path).expanduser()
        if not p.exists():
            missing_artifacts.append({"feature_artifact_path": str(p), "reason": "missing_feature_artifact"})
            continue
        payload = _load_json(p)
        rows, stats = _extract_rows_with_filter_stats(payload)
        source_filter_stats[str(p)] = stats
        invalid_feature_records_container_count += int(stats.get("invalid_records_container_count") or 0)
        for row in rows:
            records.append(_as_feature_record(row, str(p)))
    records.sort(key=_stable_feature_sort_key)
    missing_artifacts = sorted(missing_artifacts, key=lambda item: (str(item.get("feature_artifact_path") or ""), str(item.get("reason") or "")))
    diagnostics = {
        "source_filter_stats": dict(sorted(source_filter_stats.items())),
        "total_raw_row_count": sum(int((stats or {}).get("raw_row_count") or 0) for stats in source_filter_stats.values()),
        "total_valid_row_count": len(records),
        "total_invalid_non_dict_row_count": sum(int((stats or {}).get("invalid_non_dict_row_count") or 0) for stats in source_filter_stats.values()),
        "total_invalid_records_container_count": invalid_feature_records_container_count,
        "missing_feature_artifact_count": len(missing_artifacts),
        "feature_prejoin_filter_reason_counts": {
            "missing_feature_artifact": len(missing_artifacts),
            "invalid_non_dict_feature_row": sum(int((stats or {}).get("invalid_non_dict_row_count") or 0) for stats in source_filter_stats.values()),
            "invalid_feature_records_container": invalid_feature_records_container_count,
        },
    }
    return records, missing_artifacts, diagnostics


def _resolve_latest_normalized_outcomes(root: Path) -> list[Path]:
    resolved_root = Path(root).expanduser()
    if not resolved_root.exists():
        return []
    run_dirs = sorted([path for path in resolved_root.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / "normalized_outcomes.json"
        if candidate.exists():
            return [candidate]
    return []


def _resolve_all_normalized_outcomes(root: Path) -> list[Path]:
    resolved_root = Path(root).expanduser()
    if not resolved_root.exists():
        return []
    out: list[Path] = []
    run_dirs = sorted([path for path in resolved_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    for run_dir in run_dirs:
        candidate = run_dir / "normalized_outcomes.json"
        if candidate.exists():
            out.append(candidate)
    return out


def _resolve_normalized_outcomes_from_run_ids(root: Path, run_ids: list[str]) -> list[Path]:
    resolved_root = Path(root).expanduser()
    resolved: list[Path] = []
    for run_id in run_ids:
        token = str(run_id or "").strip()
        if not token:
            continue
        candidate = resolved_root / token / "normalized_outcomes.json"
        if candidate.exists():
            resolved.append(candidate)
    return resolved


def _resolve_feature_artifacts_from_dirs(directories: list[Path], artifact_glob: str) -> list[Path]:
    resolved: list[Path] = []
    for directory in directories:
        root = Path(directory).expanduser()
        if not root.exists():
            continue
        for candidate in sorted(root.glob(str(artifact_glob))):
            if candidate.is_file():
                resolved.append(candidate)
    return resolved


def _resolve_feature_artifacts_from_root(root: Path, artifact_glob: str) -> list[Path]:
    resolved_root = Path(root).expanduser()
    if not resolved_root.exists():
        return []
    return [path for path in sorted(resolved_root.glob(str(artifact_glob))) if path.is_file()]


def _coalesce_float(*values: Any) -> float | None:
    for value in values:
        out = _safe_float(value)
        if out is not None:
            return out
    return None


def _extract_feature_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    return {
        "wildfire_risk_score": _coalesce_float(
            scores.get("wildfire_risk_score"),
            payload.get("wildfire_risk_score"),
            payload.get("overall_wildfire_risk"),
            (payload.get("assessment") or {}).get("wildfire_risk_score")
            if isinstance(payload.get("assessment"), dict)
            else None,
        ),
        "site_hazard_score": _coalesce_float(
            scores.get("site_hazard_score"),
            payload.get("site_hazard_score"),
            (payload.get("assessment") or {}).get("site_hazard_score")
            if isinstance(payload.get("assessment"), dict)
            else None,
        ),
        "home_ignition_vulnerability_score": _coalesce_float(
            scores.get("home_ignition_vulnerability_score"),
            payload.get("home_ignition_vulnerability_score"),
            (payload.get("assessment") or {}).get("home_ignition_vulnerability_score")
            if isinstance(payload.get("assessment"), dict)
            else None,
        ),
        "insurance_readiness_score": _coalesce_float(
            scores.get("insurance_readiness_score"),
            payload.get("insurance_readiness_score"),
            (payload.get("assessment") or {}).get("insurance_readiness_score")
            if isinstance(payload.get("assessment"), dict)
            else None,
        ),
    }


def _population_stddev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return math.sqrt(variance)


def _compute_feature_variation_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_values: dict[str, list[float]] = {}
    class_numeric_values: dict[str, dict[str, list[float]]] = {"positive": {}, "negative": {}}
    class_counts: dict[str, int] = {"positive": 0, "negative": 0, "unknown": 0}
    for row in rows:
        feature_snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
        outcome_payload = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        adverse_outcome_binary = outcome_payload.get("adverse_outcome_binary")
        if isinstance(adverse_outcome_binary, bool):
            class_bucket = "positive" if adverse_outcome_binary else "negative"
        elif adverse_outcome_binary in (0, 1):
            class_bucket = "positive" if int(adverse_outcome_binary) == 1 else "negative"
        else:
            class_bucket = None
        if class_bucket is None:
            class_counts["unknown"] += 1
        else:
            class_counts[class_bucket] += 1
        for bag_name in ("raw_feature_vector", "transformed_feature_vector"):
            bag = feature_snapshot.get(bag_name) if isinstance(feature_snapshot.get(bag_name), dict) else {}
            for key, value in bag.items():
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                feature_key = str(key)
                numeric_values.setdefault(feature_key, []).append(float(numeric))
                if class_bucket:
                    class_numeric_values[class_bucket].setdefault(feature_key, []).append(float(numeric))

    per_feature: list[dict[str, Any]] = []
    near_zero_variance_features: list[str] = []
    for feature in sorted(numeric_values.keys()):
        values = numeric_values[feature]
        stddev = _population_stddev(values)
        feature_row = {
            "feature": feature,
            "count": len(values),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "mean": round(sum(values) / float(len(values)), 6),
            "stddev": (round(float(stddev), 9) if stddev is not None else None),
        }
        per_feature.append(feature_row)
        if len(values) >= 2 and stddev is not None and float(stddev) <= 1e-9:
            near_zero_variance_features.append(feature)

    key_features = [
        "burn_probability",
        "wildfire_hazard",
        "slope",
        "fuel_model",
        "canopy_cover",
        "ring_0_5_ft_vegetation_density",
        "ring_5_30_ft_vegetation_density",
        "near_structure_connectivity_index",
        "ring_30_100_ft_vegetation_density",
        "structure_density",
        "distance_to_nearest_structure_ft",
        "clustering_index",
        "building_age_material_proxy_risk",
    ]
    key_feature_stddev: dict[str, float | None] = {}
    key_feature_near_zero_variance: list[str] = []
    for feature in key_features:
        values = numeric_values.get(feature) or []
        if len(values) < 2:
            key_feature_stddev[feature] = None
            continue
        stddev = _population_stddev(values)
        key_feature_stddev[feature] = round(float(stddev), 9) if stddev is not None else None
        if stddev is not None and float(stddev) <= 1e-9:
            key_feature_near_zero_variance.append(feature)

    class_key_features = [
        "near_structure_vegetation_0_5_pct",
        "near_structure_connectivity_index",
        "ring_0_5_ft_vegetation_density",
        "ring_5_30_ft_vegetation_density",
        "ring_0_5_ft_vegetation_density_proxy_blend",
        "ring_5_30_ft_vegetation_density_proxy_blend",
        "near_structure_connectivity_index_proxy_blend",
        "nearest_high_fuel_patch_distance_ft_proxy_blend",
        "structure_density",
        "distance_to_nearest_structure_ft",
        "clustering_index",
        "building_age_proxy_year",
        "building_age_material_proxy_risk",
        "slope",
        "slope_index",
        "fuel_model",
        "fuel_index",
        "wildland_distance_index",
        "burn_probability",
        "burn_probability_index",
    ]
    class_feature_stats: dict[str, Any] = {}
    class_separation_rows: list[dict[str, Any]] = []
    for feature in class_key_features:
        positive_values = class_numeric_values["positive"].get(feature) or []
        negative_values = class_numeric_values["negative"].get(feature) or []
        positive_mean = (sum(positive_values) / float(len(positive_values))) if positive_values else None
        negative_mean = (sum(negative_values) / float(len(negative_values))) if negative_values else None
        mean_delta = None
        if positive_mean is not None and negative_mean is not None:
            mean_delta = float(positive_mean) - float(negative_mean)
            class_separation_rows.append(
                {
                    "feature": feature,
                    "positive_mean": round(float(positive_mean), 6),
                    "negative_mean": round(float(negative_mean), 6),
                    "mean_delta_positive_minus_negative": round(float(mean_delta), 6),
                    "absolute_mean_delta": round(abs(float(mean_delta)), 6),
                }
            )
        class_feature_stats[feature] = {
            "positive_count": len(positive_values),
            "negative_count": len(negative_values),
            "positive_mean": (round(float(positive_mean), 6) if positive_mean is not None else None),
            "negative_mean": (round(float(negative_mean), 6) if negative_mean is not None else None),
            "positive_stddev": (
                round(float(_population_stddev(positive_values) or 0.0), 9) if len(positive_values) >= 2 else None
            ),
            "negative_stddev": (
                round(float(_population_stddev(negative_values) or 0.0), 9) if len(negative_values) >= 2 else None
            ),
            "mean_delta_positive_minus_negative": (round(float(mean_delta), 6) if mean_delta is not None else None),
        }
    class_separation_rows.sort(
        key=lambda item: (
            -float(item.get("absolute_mean_delta") or 0.0),
            str(item.get("feature") or ""),
        )
    )
    class_features_without_separation = sorted(
        [
            feature
            for feature, stats in class_feature_stats.items()
            if int(stats.get("positive_count") or 0) > 0
            and int(stats.get("negative_count") or 0) > 0
            and (_safe_float(stats.get("mean_delta_positive_minus_negative")) is None or abs(float(stats.get("mean_delta_positive_minus_negative") or 0.0)) <= 1e-6)
        ]
    )
    class_features_with_separation_count = sum(
        1
        for stats in class_feature_stats.values()
        if _safe_float(stats.get("mean_delta_positive_minus_negative")) is not None
        and abs(float(stats.get("mean_delta_positive_minus_negative") or 0.0)) > 1e-6
    )

    feature_stddev_by_name: dict[str, float | None] = {}
    feature_non_null_count: dict[str, int] = {}
    for feature_name, values in numeric_values.items():
        feature_non_null_count[feature_name] = int(len(values))
        stddev = _population_stddev(values)
        feature_stddev_by_name[feature_name] = float(stddev) if stddev is not None else None

    def _feature_group_summary(feature_names: tuple[str, ...]) -> dict[str, Any]:
        available_count = 0
        non_zero_variance_count = 0
        stddev_map: dict[str, float | None] = {}
        non_zero_variance_features: list[str] = []
        for name in feature_names:
            count = int(feature_non_null_count.get(name) or 0)
            stddev = feature_stddev_by_name.get(name)
            stddev_map[name] = (round(float(stddev), 9) if stddev is not None else None)
            if count > 0:
                available_count += 1
            if count >= 2 and stddev is not None and float(stddev) > 1e-9:
                non_zero_variance_count += 1
                non_zero_variance_features.append(name)
        return {
            "feature_names": list(feature_names),
            "available_feature_count": available_count,
            "non_zero_variance_feature_count": non_zero_variance_count,
            "non_zero_variance_features": non_zero_variance_features,
            "feature_stddev": stddev_map,
        }

    structure_feature_variation = _feature_group_summary(STRUCTURE_PROXY_FEATURE_KEYS)
    near_structure_vegetation_variation = _feature_group_summary(NEAR_STRUCTURE_VEGETATION_FEATURE_KEYS)

    return {
        "numeric_feature_count": len(numeric_values),
        "features_with_variation_count": len(
            [
                item
                for item in per_feature
                if _safe_float(item.get("stddev")) is not None and float(item.get("stddev") or 0.0) > 1e-9
            ]
        ),
        "near_zero_variance_feature_count": len(near_zero_variance_features),
        "near_zero_variance_features": near_zero_variance_features[:100],
        "key_feature_stddev": key_feature_stddev,
        "key_feature_near_zero_variance": key_feature_near_zero_variance,
        "class_counts": class_counts,
        "class_key_feature_stats": class_feature_stats,
        "class_key_feature_separation": class_separation_rows[:20],
        "class_features_with_separation_count": class_features_with_separation_count,
        "class_features_without_separation": class_features_without_separation,
        "structure_feature_variation": structure_feature_variation,
        "near_structure_vegetation_feature_variation": near_structure_vegetation_variation,
        "feature_stats": per_feature[:300],
    }


def _quantile_thresholds(values: list[float], *, lower_q: float = 1.0 / 3.0, upper_q: float = 2.0 / 3.0) -> tuple[float, float] | None:
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        single = float(sorted_values[0])
        return (single, single)
    low_idx = max(0, min(len(sorted_values) - 1, int(round(lower_q * (len(sorted_values) - 1)))))
    high_idx = max(0, min(len(sorted_values) - 1, int(round(upper_q * (len(sorted_values) - 1)))))
    low = float(sorted_values[low_idx])
    high = float(sorted_values[high_idx])
    if high < low:
        high = low
    return (low, high)


def _tercile_bucket(value: float | None, thresholds: tuple[float, float] | None) -> str:
    if value is None or thresholds is None:
        return "unknown"
    low, high = thresholds
    numeric = float(value)
    if numeric <= low:
        return "low"
    if numeric >= high:
        return "high"
    return "medium"


def _extract_diversity_proxies(row: dict[str, Any]) -> dict[str, float | str | None]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    event = row.get("event") if isinstance(row.get("event"), dict) else {}
    feature = row.get("feature") if isinstance(row.get("feature"), dict) else {}
    feature_snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
    raw_vector = feature_snapshot.get("raw_feature_vector") if isinstance(feature_snapshot.get("raw_feature_vector"), dict) else {}
    transformed_vector = (
        feature_snapshot.get("transformed_feature_vector")
        if isinstance(feature_snapshot.get("transformed_feature_vector"), dict)
        else {}
    )
    property_context = feature.get("property_level_context") if isinstance(feature.get("property_level_context"), dict) else {}

    hazard = _coalesce_float(
        scores.get("site_hazard_score"),
        transformed_vector.get("hazard_severity_index"),
        transformed_vector.get("burn_probability_index"),
        raw_vector.get("wildfire_hazard"),
        raw_vector.get("burn_probability"),
    )
    vegetation = _coalesce_float(
        transformed_vector.get("ring_0_5_ft_vegetation_density"),
        transformed_vector.get("ring_5_30_ft_vegetation_density"),
        transformed_vector.get("near_structure_vegetation_0_5_pct"),
        transformed_vector.get("canopy_index"),
        raw_vector.get("ring_0_5_ft_vegetation_density"),
        raw_vector.get("ring_5_30_ft_vegetation_density"),
        raw_vector.get("canopy_cover"),
    )
    terrain = _coalesce_float(
        transformed_vector.get("slope_index"),
        raw_vector.get("slope"),
    )
    region_key = (
        str(property_context.get("resolved_region_id") or "").strip()
        if isinstance(property_context, dict)
        else ""
    )
    if not region_key:
        region_key = str(event.get("event_id") or "unknown_event")
    return {
        "hazard_proxy": hazard,
        "vegetation_proxy": vegetation,
        "terrain_proxy": terrain,
        "region_proxy": region_key,
    }


def _compute_diversity_spread(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "reason": "no_rows",
        }
    hazard_values: list[float] = []
    vegetation_values: list[float] = []
    terrain_values: list[float] = []
    extracted: list[dict[str, Any]] = []
    region_counts: dict[str, int] = {}
    for row in rows:
        proxies = _extract_diversity_proxies(row)
        extracted.append(proxies)
        region = str(proxies.get("region_proxy") or "unknown")
        region_counts[region] = region_counts.get(region, 0) + 1
        hv = _safe_float(proxies.get("hazard_proxy"))
        vv = _safe_float(proxies.get("vegetation_proxy"))
        tv = _safe_float(proxies.get("terrain_proxy"))
        if hv is not None:
            hazard_values.append(float(hv))
        if vv is not None:
            vegetation_values.append(float(vv))
        if tv is not None:
            terrain_values.append(float(tv))

    hazard_thresholds = _quantile_thresholds(hazard_values)
    vegetation_thresholds = _quantile_thresholds(vegetation_values)
    terrain_thresholds = _quantile_thresholds(terrain_values)
    hazard_bin_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    vegetation_bin_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    terrain_bin_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    combo_counts: dict[str, int] = {}
    for proxies in extracted:
        h_bin = _tercile_bucket(_safe_float(proxies.get("hazard_proxy")), hazard_thresholds)
        v_bin = _tercile_bucket(_safe_float(proxies.get("vegetation_proxy")), vegetation_thresholds)
        t_bin = _tercile_bucket(_safe_float(proxies.get("terrain_proxy")), terrain_thresholds)
        hazard_bin_counts[h_bin] = hazard_bin_counts.get(h_bin, 0) + 1
        vegetation_bin_counts[v_bin] = vegetation_bin_counts.get(v_bin, 0) + 1
        terrain_bin_counts[t_bin] = terrain_bin_counts.get(t_bin, 0) + 1
        combo = f"h:{h_bin}|v:{v_bin}|t:{t_bin}"
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    region_total = sum(region_counts.values())
    max_region = max(region_counts.items(), key=lambda item: (item[1], item[0])) if region_counts else ("unknown", 0)
    max_region_share = (float(max_region[1]) / float(region_total)) if region_total > 0 else 0.0

    return {
        "available": True,
        "region_count": len(region_counts),
        "region_counts": dict(sorted(region_counts.items())),
        "max_region_key": str(max_region[0]),
        "max_region_share": round(max_region_share, 4),
        "hazard_proxy_thresholds": (
            [round(float(hazard_thresholds[0]), 4), round(float(hazard_thresholds[1]), 4)]
            if hazard_thresholds is not None
            else None
        ),
        "vegetation_proxy_thresholds": (
            [round(float(vegetation_thresholds[0]), 4), round(float(vegetation_thresholds[1]), 4)]
            if vegetation_thresholds is not None
            else None
        ),
        "terrain_proxy_thresholds": (
            [round(float(terrain_thresholds[0]), 4), round(float(terrain_thresholds[1]), 4)]
            if terrain_thresholds is not None
            else None
        ),
        "hazard_bin_counts": hazard_bin_counts,
        "vegetation_bin_counts": vegetation_bin_counts,
        "terrain_bin_counts": terrain_bin_counts,
        "non_empty_bins": {
            "hazard": sum(1 for key in ("low", "medium", "high") if int(hazard_bin_counts.get(key) or 0) > 0),
            "vegetation": sum(1 for key in ("low", "medium", "high") if int(vegetation_bin_counts.get(key) or 0) > 0),
            "terrain": sum(1 for key in ("low", "medium", "high") if int(terrain_bin_counts.get(key) or 0) > 0),
        },
        "strata_combo_count": len(combo_counts),
        "top_strata_combos": sorted(
            [
                {"strata": key, "count": value}
                for key, value in combo_counts.items()
            ],
            key=lambda item: (-int(item.get("count") or 0), str(item.get("strata") or "")),
        )[:20],
    }


def _build_scored_record_maps(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_event_record: dict[tuple[str, str], dict[str, Any]] = {}
    by_record: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id") or "").strip()
        record_id = str(row.get("record_id") or "").strip()
        source_record_id = str(row.get("source_record_id") or "").strip()
        if event_id and record_id:
            by_event_record[(event_id, record_id)] = row
        if record_id:
            by_record.setdefault(record_id, row)
        if source_record_id:
            by_record.setdefault(source_record_id, row)
    return by_event_record, by_record


def _backfill_missing_feature_scores(
    *,
    feature_rows: list[FeatureRecord],
    run_dir: Path,
    auto_score_missing: bool,
    auto_score_all: bool = False,
) -> dict[str, Any]:
    missing_rows = [
        row
        for row in feature_rows
        if _extract_feature_scores(row.payload).get("wildfire_risk_score") is None
    ]
    per_artifact_missing: dict[str, int] = {}
    for row in missing_rows:
        per_artifact_missing[row.artifact_path] = per_artifact_missing.get(row.artifact_path, 0) + 1
    diagnostics: dict[str, Any] = {
        "auto_score_missing_enabled": bool(auto_score_missing),
        "auto_score_all_enabled": bool(auto_score_all),
        "rescore_mode": ("all_records" if auto_score_all else "missing_scores_only"),
        "total_feature_rows": len(feature_rows),
        "missing_score_record_count_before": len(missing_rows),
        "missing_score_by_artifact_before": dict(sorted(per_artifact_missing.items())),
        "rescoring_attempted": False,
        "rescoring_artifact_count": 0,
        "rescoring_artifact_path": None,
        "rescoring_record_count": 0,
        "backfilled_record_count": 0,
        "remaining_missing_score_record_count": len(missing_rows),
        "warnings": [],
    }
    if not feature_rows:
        return diagnostics
    if not missing_rows and not auto_score_all:
        return diagnostics
    if not auto_score_missing and not auto_score_all:
        diagnostics["warnings"].append(
            "Feature artifacts are missing wildfire scores; auto-score backfill is disabled."
        )
        return diagnostics

    rows_to_patch = list(feature_rows if auto_score_all else missing_rows)
    artifact_paths = sorted({row.artifact_path for row in rows_to_patch})
    try:
        from backend.event_backtesting import run_event_backtest  # lazy import

        rescoring_dir = run_dir / "_auto_scored_event_backtest"
        rescoring_dir.mkdir(parents=True, exist_ok=True)
        diagnostics["rescoring_attempted"] = True
        diagnostics["rescoring_artifact_count"] = len(artifact_paths)
        artifact = run_event_backtest(
            dataset_paths=artifact_paths,
            output_dir=rescoring_dir,
            use_runtime_context_when_no_overrides=True,
        )
        diagnostics["rescoring_artifact_path"] = str(artifact.get("artifact_path") or "")
        scored_rows = artifact.get("records") if isinstance(artifact.get("records"), list) else []
        diagnostics["rescoring_record_count"] = len(scored_rows)
        diagnostics["rescoring_runtime_context_mode_when_overrides_missing"] = (
            artifact.get("runtime_context_mode_when_overrides_missing")
            if isinstance(artifact, dict)
            else None
        )
        by_event_record, by_record = _build_scored_record_maps(scored_rows)
        patched = 0
        for row in rows_to_patch:
            before_scores = _extract_feature_scores(row.payload)
            before_wildfire = before_scores.get("wildfire_risk_score")
            scored = by_event_record.get((row.event_id, row.record_id))
            if scored is None:
                scored = by_record.get(row.record_id) or by_record.get(row.source_record_id)
            if not isinstance(scored, dict):
                continue
            scored_scores = scored.get("scores") if isinstance(scored.get("scores"), dict) else {}
            target_scores = row.payload.setdefault("scores", {})
            if not isinstance(target_scores, dict):
                target_scores = {}
                row.payload["scores"] = target_scores
            for key in (
                "wildfire_risk_score",
                "site_hazard_score",
                "home_ignition_vulnerability_score",
                "insurance_readiness_score",
            ):
                if auto_score_all and scored_scores.get(key) is not None:
                    target_scores[key] = scored_scores.get(key)
                elif target_scores.get(key) is None and scored_scores.get(key) is not None:
                    target_scores[key] = scored_scores.get(key)
            confidence = scored.get("confidence") if isinstance(scored.get("confidence"), dict) else {}
            if (auto_score_all or row.payload.get("confidence_tier") is None) and confidence.get("confidence_tier") is not None:
                row.payload["confidence_tier"] = confidence.get("confidence_tier")
            if (auto_score_all or row.payload.get("confidence_score") is None) and confidence.get("confidence_score") is not None:
                row.payload["confidence_score"] = confidence.get("confidence_score")
            if (auto_score_all or row.payload.get("use_restriction") is None) and confidence.get("use_restriction") is not None:
                row.payload["use_restriction"] = confidence.get("use_restriction")
            if (auto_score_all or row.payload.get("evidence_quality_summary") is None) and isinstance(scored.get("evidence_quality_summary"), dict):
                row.payload["evidence_quality_summary"] = scored.get("evidence_quality_summary")
            if (auto_score_all or row.payload.get("coverage_summary") is None) and isinstance(scored.get("coverage_summary"), dict):
                row.payload["coverage_summary"] = scored.get("coverage_summary")
            if (auto_score_all or row.payload.get("factor_contribution_breakdown") is None) and isinstance(scored.get("factor_contribution_breakdown"), dict):
                row.payload["factor_contribution_breakdown"] = scored.get("factor_contribution_breakdown")
            if (auto_score_all or row.payload.get("raw_feature_vector") is None) and isinstance(scored.get("raw_feature_vector"), dict):
                row.payload["raw_feature_vector"] = scored.get("raw_feature_vector")
            if (auto_score_all or row.payload.get("transformed_feature_vector") is None) and isinstance(scored.get("transformed_feature_vector"), dict):
                row.payload["transformed_feature_vector"] = scored.get("transformed_feature_vector")
            if (auto_score_all or row.payload.get("compression_flags") is None) and isinstance(scored.get("compression_flags"), list):
                row.payload["compression_flags"] = scored.get("compression_flags")
            if (auto_score_all or row.payload.get("model_governance") is None) and isinstance(scored.get("model_governance"), dict):
                row.payload["model_governance"] = scored.get("model_governance")
            row.payload["score_backfill_source"] = (
                "event_backtest_auto_rescore_all"
                if auto_score_all
                else "event_backtest_auto_rescore_missing"
            )
            after_wildfire = _extract_feature_scores(row.payload).get("wildfire_risk_score")
            if after_wildfire is not None and (
                before_wildfire is None
                or not math.isclose(float(before_wildfire), float(after_wildfire), rel_tol=0.0, abs_tol=1e-9)
            ):
                patched += 1
        diagnostics["backfilled_record_count"] = patched
    except Exception as exc:
        diagnostics["warnings"].append(f"Auto-score backfill failed: {exc}")

    remaining_missing = [
        row
        for row in feature_rows
        if _extract_feature_scores(row.payload).get("wildfire_risk_score") is None
    ]
    diagnostics["remaining_missing_score_record_count"] = len(remaining_missing)
    return diagnostics


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
        "row_confidence_tier",
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
                "row_confidence_tier": row.get("evaluation", {}).get("row_confidence_tier"),
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
        f"- Candidate match attempts: `{quality.get('candidate_match_attempt_count')}`",
        f"- Matched before filters: `{quality.get('matched_candidate_count_before_filters')}`",
        f"- Outcome source files: `{quality.get('outcomes_loaded_by_source_path')}`",
        f"- Outcomes by event: `{quality.get('outcomes_by_event_counts')}`",
        f"- Feature rows loaded: `{quality.get('total_feature_rows_loaded')}`",
        f"- Feature rows by event: `{quality.get('feature_rows_by_event_counts')}`",
        f"- Joined rows: `{quality.get('total_joined_records')}`",
        f"- Joined rows before property-event dedupe: `{quality.get('total_rows_before_property_event_dedupe')}`",
        f"- Unique property_event_id count: `{quality.get('unique_property_event_id_count')}`",
        f"- Duplicate property_event rows removed: `{quality.get('duplicate_property_event_rows_removed_count')}`",
        f"- Duplication factor (before/unique): `{quality.get('duplication_factor')}`",
        f"- Filtered records: `{quality.get('filtered_records_count')}`",
        f"- Final dataset size: `{quality.get('final_dataset_size')}`",
        f"- Join rate: `{quality.get('join_rate')}`",
        f"- Match rate (%): `{quality.get('match_rate_percent')}`",
        "",
        "## Join Quality",
        f"- Join method counts: `{quality.get('join_method_counts')}`",
        f"- Join confidence tier counts: `{quality.get('join_confidence_tier_counts')}`",
        f"- Match tier counts: `{quality.get('match_tier_counts')}`",
        f"- Row confidence tier counts: `{quality.get('row_confidence_tier_counts')}`",
        f"- Fallback usage summary: `{quality.get('fallback_usage_summary')}`",
        f"- Join confidence score stats: `{quality.get('join_confidence_score_stats')}`",
        f"- Join confidence distance stats by tier: `{quality.get('join_confidence_tier_distance_stats')}`",
        f"- Retention fallback: `{quality.get('retention_fallback')}`",
        f"- Non-high confidence reason counts: `{quality.get('join_confidence_non_high_reason_counts')}`",
        f"- High-confidence threshold diagnostics: `{quality.get('high_confidence_threshold_diagnostics')}`",
        f"- Average join distance (m): `{quality.get('average_join_distance_m')}`",
        f"- Median join distance (m): `{quality.get('median_join_distance_m')}`",
        f"- Distance percentiles (m): `{quality.get('distance_percentiles_m')}`",
        f"- Match distance histogram (m): `{quality.get('match_distance_histogram_m')}`",
        f"- Distance outlier threshold (m): `{quality.get('distance_outlier_threshold_m')}`",
        f"- Distance outlier examples: `{quality.get('distance_outlier_examples')}`",
        f"- Low-confidence joins: `{quality.get('low_confidence_join_count')}`",
        f"- Duplicate matches prevented: `{quality.get('duplicate_outcome_match_prevented_count')}`",
        f"- Coordinate normalization summary: `{quality.get('coordinate_normalization_summary')}`",
        f"- Join tier examples: `{quality.get('join_confidence_tier_examples')}`",
        f"- Duplicate rows removed by join tier: `{quality.get('duplicate_rows_removed_by_join_confidence_tier')}`",
        f"- Duplicate rows removed by join method: `{quality.get('duplicate_rows_removed_by_join_method')}`",
        f"- Feature variation diagnostics: `{quality.get('feature_variation_diagnostics')}`",
        f"- Diversity spread diagnostics: `{quality.get('diversity_spread')}`",
        "",
        "## Coverage",
        f"- By event joined counts: `{quality.get('by_event_join_counts')}`",
        f"- By label joined counts: `{quality.get('by_label_join_counts')}`",
        f"- Unmatched feature rows by event: `{quality.get('unmatched_feature_rows_by_event_counts')}`",
        f"- Excluded rows: `{quality.get('excluded_row_count')}`",
        f"- Excluded reason counts: `{quality.get('excluded_reason_counts')}`",
        "",
        "## Leakage Guardrails",
        f"- Leakage warning count: `{len(quality.get('leakage_warnings') or [])}`",
        "",
        "## Score Availability",
        f"- Score backfill summary: `{quality.get('score_backfill')}`",
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


def _build_filter_summary_markdown(
    *,
    run_id: str,
    generated_at: str,
    filter_summary: dict[str, Any],
) -> str:
    lines = [
        "# Evaluation Dataset Filter Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        "- Purpose: account for all rows filtered during evaluation dataset construction.",
        "",
        "## Stage Counts",
        f"- Outcomes raw rows: `{filter_summary.get('outcomes_raw_rows')}`",
        f"- Outcomes post-preprocessing rows: `{filter_summary.get('outcomes_after_prejoin_filters')}`",
        f"- Feature raw rows: `{filter_summary.get('features_raw_rows')}`",
        f"- Feature candidate rows: `{filter_summary.get('feature_candidate_rows')}`",
        f"- Candidate match attempts: `{filter_summary.get('candidate_match_attempt_count')}`",
        f"- Matched before post-join filters: `{filter_summary.get('matched_candidate_count_before_filters')}`",
        f"- Joined rows: `{filter_summary.get('joined_rows')}`",
        f"- Joined rows before property-event dedupe: `{filter_summary.get('joined_rows_before_property_event_dedupe')}`",
        f"- Unique property_event_id count: `{filter_summary.get('unique_property_event_id_count')}`",
        f"- Duplicate property_event rows removed: `{filter_summary.get('duplicate_property_event_rows_removed_count')}`",
        f"- Duplication factor (before/unique): `{filter_summary.get('duplication_factor')}`",
        f"- Filtered rows: `{filter_summary.get('filtered_rows_total')}`",
        "",
        "## Filter Reasons",
        f"- Filter reason counts: `{filter_summary.get('filter_reason_counts')}`",
        f"- Filter reason percentages: `{filter_summary.get('filter_reason_percentages')}`",
        "",
        "## Soft Flags (rows retained)",
        f"- Soft-flag counts: `{filter_summary.get('soft_flag_counts')}`",
        f"- Retention fallback triggered: `{filter_summary.get('retention_fallback_triggered')}`",
        f"- Retention fallback used: `{filter_summary.get('retention_fallback_used')}`",
        f"- Retention target: `{filter_summary.get('retention_min_records_target')}`",
        "",
        "## Accounting",
        f"- Accounting check passed: `{filter_summary.get('no_silent_data_loss_guarantee')}`",
        f"- Accounting details: `{filter_summary.get('accounting')}`",
    ]
    return "\n".join(lines) + "\n"


def _build_dataset_quality_report(*, quality: dict[str, Any]) -> dict[str, Any]:
    variation = (
        quality.get("feature_variation_diagnostics")
        if isinstance(quality.get("feature_variation_diagnostics"), dict)
        else {}
    )
    fallback_summary = (
        quality.get("fallback_usage_summary")
        if isinstance(quality.get("fallback_usage_summary"), dict)
        else {}
    )
    return {
        "total_labeled_rows": int(quality.get("total_joined_records") or 0),
        "unique_property_event_id_count": int(quality.get("unique_property_event_id_count") or 0),
        "positive_label_count": int((quality.get("by_label_join_counts") or {}).get("destroyed", 0))
        + int((quality.get("by_label_join_counts") or {}).get("major_damage", 0)),
        "negative_label_count": int((quality.get("by_label_join_counts") or {}).get("no_damage", 0))
        + int((quality.get("by_label_join_counts") or {}).get("minor_damage", 0)),
        "unknown_label_count": int((quality.get("by_label_join_counts") or {}).get("unknown", 0)),
        "fallback_heavy_fraction": _safe_float(fallback_summary.get("fallback_heavy_fraction")) or 0.0,
        "feature_variation_summary": {
            "numeric_feature_count": int(variation.get("numeric_feature_count") or 0),
            "features_with_variation_count": int(variation.get("features_with_variation_count") or 0),
            "near_zero_variance_feature_count": int(variation.get("near_zero_variance_feature_count") or 0),
            "key_feature_near_zero_variance": (
                variation.get("key_feature_near_zero_variance")
                if isinstance(variation.get("key_feature_near_zero_variance"), list)
                else []
            ),
        },
        "structure_feature_variation": (
            variation.get("structure_feature_variation")
            if isinstance(variation.get("structure_feature_variation"), dict)
            else {}
        ),
        "near_structure_vegetation_feature_variation": (
            variation.get("near_structure_vegetation_feature_variation")
            if isinstance(variation.get("near_structure_vegetation_feature_variation"), dict)
            else {}
        ),
    }


def _build_dataset_quality_markdown(
    *,
    run_id: str,
    generated_at: str,
    dataset_quality: dict[str, Any],
) -> str:
    lines = [
        "# Evaluation Dataset Quality Report",
        "",
        "- Dataset-quality diagnostics for public-outcome validation inputs.",
        "- Public observed outcomes remain directional validation signals, not insurer claims truth.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Total labeled rows: `{dataset_quality.get('total_labeled_rows')}`",
        f"- Unique property_event_id rows: `{dataset_quality.get('unique_property_event_id_count')}`",
        f"- Positive labels: `{dataset_quality.get('positive_label_count')}`",
        f"- Negative labels: `{dataset_quality.get('negative_label_count')}`",
        f"- Unknown labels: `{dataset_quality.get('unknown_label_count')}`",
        f"- Fallback-heavy fraction: `{dataset_quality.get('fallback_heavy_fraction')}`",
        "",
        "## Feature Variation",
        f"- Summary: `{dataset_quality.get('feature_variation_summary')}`",
        f"- Structure feature variation: `{dataset_quality.get('structure_feature_variation')}`",
        f"- Near-structure vegetation variation: `{dataset_quality.get('near_structure_vegetation_feature_variation')}`",
    ]
    return "\n".join(lines) + "\n"


def _build_pipeline_audit_markdown(
    *,
    run_id: str,
    generated_at: str,
    quality: dict[str, Any],
    filter_summary: dict[str, Any],
    sample_joined_rows: list[dict[str, Any]],
    sample_filtered_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Pipeline Audit Report",
        "",
        "- Developer audit trace for evaluation dataset construction.",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        "",
        "## Ingestion",
        f"- Outcomes loaded: `{quality.get('total_outcomes_loaded')}`",
        f"- Outcome load diagnostics: `{quality.get('outcome_load_diagnostics')}`",
        f"- Feature rows loaded: `{quality.get('total_feature_rows_loaded')}`",
        f"- Feature load diagnostics: `{quality.get('feature_load_diagnostics')}`",
        "",
        "## Join Pipeline",
        f"- Candidate match attempts: `{quality.get('candidate_match_attempt_count')}`",
        f"- Matched before post-join filtering: `{quality.get('matched_candidate_count_before_filters')}`",
        f"- Joined rows before property-event dedupe: `{quality.get('total_rows_before_property_event_dedupe')}`",
        f"- Joined rows (final): `{quality.get('total_joined_records')}`",
        f"- Unique property_event_id count: `{quality.get('unique_property_event_id_count')}`",
        f"- Duplicate property_event rows removed: `{quality.get('duplicate_property_event_rows_removed_count')}`",
        f"- Duplication factor (before/unique): `{quality.get('duplication_factor')}`",
        f"- Match rate: `{quality.get('match_rate_percent')}%`",
        f"- Join method counts: `{quality.get('join_method_counts')}`",
        f"- Join confidence tier counts: `{quality.get('join_confidence_tier_counts')}`",
        f"- Distance histogram: `{quality.get('match_distance_histogram_m')}`",
        f"- High-confidence diagnostics: `{quality.get('high_confidence_threshold_diagnostics')}`",
        f"- Non-high reason counts: `{quality.get('join_confidence_non_high_reason_counts')}`",
        f"- Feature variation diagnostics: `{quality.get('feature_variation_diagnostics')}`",
        "",
        "## Filtering",
        f"- Filter summary: `{filter_summary}`",
        "",
        "## Joined Sample Rows",
    ]
    if sample_joined_rows:
        for row in sample_joined_rows:
            lines.append(
                "- "
                + str(
                    {
                        "property_event_id": row.get("property_event_id"),
                        "feature_lat": row.get("feature_lat"),
                        "feature_lon": row.get("feature_lon"),
                        "outcome_lat": row.get("outcome_lat"),
                        "outcome_lon": row.get("outcome_lon"),
                        "join_distance_m": row.get("join_distance_m"),
                        "join_confidence_score": row.get("join_confidence_score"),
                        "join_confidence_tier": row.get("join_confidence_tier"),
                        "join_method": row.get("join_method"),
                        "candidate_pool_count": row.get("candidate_pool_count"),
                        "outcome_label": row.get("outcome_label"),
                        "non_high_reason_codes": row.get("non_high_reason_codes"),
                    }
                )
            )
    else:
        lines.append("- No joined sample rows available.")
    lines.append("")
    lines.append("## Filtered Sample Rows")
    if sample_filtered_rows:
        for row in sample_filtered_rows:
            lines.append(f"- {row}")
    else:
        lines.append("- No filtered rows.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Audit output is for debugging data flow and join behavior, not predictive-accuracy claims.")
    return "\n".join(lines) + "\n"


def _build_join_quality_warnings(*, quality: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    tier_counts = (
        quality.get("join_confidence_tier_counts")
        if isinstance(quality.get("join_confidence_tier_counts"), dict)
        else {}
    )
    high_count = int(tier_counts.get("high") or 0)
    if high_count <= 0:
        warnings.append(
            "No high-confidence matches were found; review source alignment, thresholds, and coordinate quality."
        )

    avg_distance = _safe_float(quality.get("average_join_distance_m"))
    threshold_m = _safe_float(
        (
            (quality.get("warning_thresholds") or {}).get("high_average_distance_m")
            if isinstance(quality.get("warning_thresholds"), dict)
            else None
        )
    )
    if avg_distance is not None and threshold_m is not None and avg_distance > threshold_m:
        warnings.append(
            f"Average join distance is high ({round(float(avg_distance), 3)} m > {round(float(threshold_m), 3)} m)."
        )
    retention = quality.get("retention_fallback") if isinstance(quality.get("retention_fallback"), dict) else {}
    if bool(retention.get("triggered")):
        warnings.append(
            "Minimum-retention fallback mode was triggered; lower-confidence joins were included to prevent near-zero dataset collapse."
        )
    duplicate_removed = int(quality.get("duplicate_property_event_rows_removed_count") or 0)
    if duplicate_removed > 0:
        warnings.append(
            "Property-event dedupe removed duplicate joined rows; inspect duplication metrics for source artifact overlap."
        )
    feature_variation = (
        quality.get("feature_variation_diagnostics")
        if isinstance(quality.get("feature_variation_diagnostics"), dict)
        else {}
    )
    key_constant = (
        feature_variation.get("key_feature_near_zero_variance")
        if isinstance(feature_variation.get("key_feature_near_zero_variance"), list)
        else []
    )
    if key_constant:
        warnings.append(
            f"Key features with near-zero variance detected: {key_constant}. Verify per-row spatial sampling and data coverage."
        )
    structure_variation = (
        feature_variation.get("structure_feature_variation")
        if isinstance(feature_variation.get("structure_feature_variation"), dict)
        else {}
    )
    structure_non_zero = int(structure_variation.get("non_zero_variance_feature_count") or 0)
    if structure_non_zero <= 0:
        warnings.append(
            "Structure-level proxy features show zero non-zero-variance coverage; structure-level discrimination will remain weak."
        )
    near_structure_vegetation_variation = (
        feature_variation.get("near_structure_vegetation_feature_variation")
        if isinstance(feature_variation.get("near_structure_vegetation_feature_variation"), dict)
        else {}
    )
    near_veg_non_zero = int(near_structure_vegetation_variation.get("non_zero_variance_feature_count") or 0)
    if near_veg_non_zero <= 0:
        warnings.append(
            "Near-structure vegetation proxy features show zero non-zero-variance coverage; near-home sensitivity will remain limited."
        )
    class_counts = (
        feature_variation.get("class_counts")
        if isinstance(feature_variation.get("class_counts"), dict)
        else {}
    )
    positive_n = int(class_counts.get("positive") or 0)
    negative_n = int(class_counts.get("negative") or 0)
    if positive_n > 0 and negative_n > 0:
        class_features_with_separation = int(feature_variation.get("class_features_with_separation_count") or 0)
        if class_features_with_separation <= 0:
            warnings.append(
                "Outcome classes show no measurable separation across key features; predictive signal may remain weak."
            )
        class_without = (
            feature_variation.get("class_features_without_separation")
            if isinstance(feature_variation.get("class_features_without_separation"), list)
            else []
        )
        if class_without:
            warnings.append(
                f"Key features without positive/negative class separation: {class_without[:8]}."
            )
    elif positive_n == 0 or negative_n == 0:
        warnings.append(
            "Class-separation diagnostics are limited because one or both outcome classes are absent in joined rows."
        )
    diversity_spread = (
        quality.get("diversity_spread")
        if isinstance(quality.get("diversity_spread"), dict)
        else {}
    )
    if bool(diversity_spread.get("available")):
        max_region_share = _safe_float(diversity_spread.get("max_region_share"))
        if max_region_share is not None and float(max_region_share) > 0.75:
            warnings.append(
                f"Dataset appears region-clustered (max_region_share={round(float(max_region_share), 3)}); include more regions/events for better generalization checks."
            )
        non_empty_bins = (
            diversity_spread.get("non_empty_bins")
            if isinstance(diversity_spread.get("non_empty_bins"), dict)
            else {}
        )
        for axis in ("hazard", "vegetation", "terrain"):
            axis_bins = int(non_empty_bins.get(axis) or 0)
            if axis_bins < 2:
                warnings.append(
                    f"Diversity spread is weak for {axis} (non_empty_bins={axis_bins}); add properties spanning low/medium/high {axis} conditions."
                )
    else:
        warnings.append("Diversity spread diagnostics unavailable; verify joined rows include usable hazard/vegetation/terrain proxies.")
    return warnings


def _build_join_quality_report_markdown(
    *,
    run_id: str,
    generated_at: str,
    quality: dict[str, Any],
) -> str:
    tier_counts = (
        quality.get("join_confidence_tier_counts")
        if isinstance(quality.get("join_confidence_tier_counts"), dict)
        else {}
    )
    tier_examples = (
        quality.get("join_confidence_tier_examples")
        if isinstance(quality.get("join_confidence_tier_examples"), dict)
        else {}
    )
    warnings = quality.get("join_quality_warnings")
    warnings = warnings if isinstance(warnings, list) else []

    high_examples = tier_examples.get("high") if isinstance(tier_examples.get("high"), list) else []
    low_examples = tier_examples.get("low") if isinstance(tier_examples.get("low"), list) else []

    lines = [
        "# Join Quality Report",
        "",
        "Public-outcome match quality diagnostics for evaluation-dataset joins.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Total outcomes: `{quality.get('total_outcomes_loaded')}`",
        f"- Matched records: `{quality.get('total_joined_records')}`",
        f"- Matched records before property-event dedupe: `{quality.get('total_rows_before_property_event_dedupe')}`",
        f"- Unique property_event_id count: `{quality.get('unique_property_event_id_count')}`",
        f"- Duplicate property_event rows removed: `{quality.get('duplicate_property_event_rows_removed_count')}`",
        f"- Duplication factor (before/unique): `{quality.get('duplication_factor')}`",
        f"- Match rate: `{quality.get('join_rate')}` (`{quality.get('match_rate_percent')}%`)",
        f"- Confidence-tier counts: `{tier_counts}`",
        f"- Non-high confidence reasons: `{quality.get('join_confidence_non_high_reason_counts')}`",
        f"- High-confidence threshold diagnostics: `{quality.get('high_confidence_threshold_diagnostics')}`",
        f"- Feature variation diagnostics: `{quality.get('feature_variation_diagnostics')}`",
        "",
        "## Distance Analysis",
        f"- Min distance (m): `{quality.get('min_join_distance_m')}`",
        f"- Mean distance (m): `{quality.get('average_join_distance_m')}`",
        f"- Max distance (m): `{quality.get('max_join_distance_m')}`",
        f"- Median distance (m): `{quality.get('median_join_distance_m')}`",
        f"- Distance histogram (m): `{quality.get('match_distance_histogram_m')}`",
        "",
        "## Example Matches",
        f"- High-confidence examples: `{high_examples}`",
        f"- Low-confidence examples: `{low_examples}`",
        "",
        "## Warnings",
    ]
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Caveats",
            "- Match diagnostics describe join quality only; they do not establish predictive accuracy.",
            "- Public-outcome datasets are heterogeneous and incomplete.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_public_outcome_evaluation_dataset(
    *,
    outcomes_path: Path | None = None,
    outcomes_paths: list[Path] | None = None,
    feature_artifacts: list[Path],
    output_root: Path = Path("benchmark/public_outcomes/evaluation_dataset"),
    run_id: str | None = None,
    exact_match_distance_m: float = 3.0,
    near_match_distance_m: float = 30.0,
    max_distance_m: float = 120.0,
    global_max_distance_m: float | None = 1000.0,
    buffer_match_radius_m: float = 80.0,
    high_confidence_distance_m: float = DEFAULT_HIGH_CONFIDENCE_DISTANCE_M,
    medium_confidence_distance_m: float = 100.0,
    event_year_tolerance_years: int = 1,
    enable_global_nearest_fallback: bool = True,
    allow_duplicate_outcome_matches: bool = False,
    address_token_overlap_min: float = 0.75,
    min_retained_records: int = 0,
    auto_relax_for_min_retention: bool = False,
    auto_score_missing: bool = True,
    auto_score_all: bool = False,
    audit_mode: bool = False,
    audit_sample_rows: int = 10,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_token = str(run_id or _timestamp_id())
    generated_at = _deterministic_generated_at(run_id)
    run_dir = Path(output_root).expanduser() / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    configured_outcome_paths = sorted(
        {Path(path).expanduser() for path in (outcomes_paths or []) if str(path).strip()},
        key=lambda token: str(token),
    )
    if not configured_outcome_paths and outcomes_path is not None:
        configured_outcome_paths = [Path(outcomes_path).expanduser()]
    if not configured_outcome_paths:
        raise ValueError("At least one normalized outcomes path is required.")

    outcomes, outcomes_by_source, outcome_load_diagnostics = _load_outcomes_from_paths(configured_outcome_paths)
    if not outcomes:
        raise ValueError("No normalized outcome rows available.")
    outcomes_by_event_counts: dict[str, int] = {}
    outcomes_by_year_counts: dict[str, int] = {}
    for outcome in outcomes:
        event_key = str(outcome.event_id or "unknown_event")
        outcomes_by_event_counts[event_key] = outcomes_by_event_counts.get(event_key, 0) + 1
        year_key = str(outcome.event_year or "unknown")
        outcomes_by_year_counts[year_key] = outcomes_by_year_counts.get(year_key, 0) + 1
    feature_artifacts = sorted({Path(path).expanduser() for path in feature_artifacts}, key=lambda token: str(token))
    feature_rows, missing_feature_artifacts, feature_load_diagnostics = _load_feature_records(feature_artifacts)
    if not feature_rows:
        raise ValueError("No feature rows were loaded from provided feature artifacts.")
    feature_rows_by_event_counts: dict[str, int] = {}
    feature_rows_by_year_counts: dict[str, int] = {}
    for feature in feature_rows:
        event_key = str(feature.event_id or "unknown_event")
        feature_rows_by_event_counts[event_key] = feature_rows_by_event_counts.get(event_key, 0) + 1
        year_key = str(feature.event_year or "unknown")
        feature_rows_by_year_counts[year_key] = feature_rows_by_year_counts.get(year_key, 0) + 1
    score_backfill = _backfill_missing_feature_scores(
        feature_rows=feature_rows,
        run_dir=run_dir,
        auto_score_missing=bool(auto_score_missing),
        auto_score_all=bool(auto_score_all),
    )

    indexes = _build_indexes(outcomes)
    join_config = JoinConfig(
        exact_match_distance_m=max(0.0, float(exact_match_distance_m)),
        near_match_distance_m=max(0.0, float(near_match_distance_m)),
        max_distance_m=float(max_distance_m),
        global_max_distance_m=float(global_max_distance_m if global_max_distance_m is not None else max_distance_m),
        buffer_match_radius_m=max(0.0, float(buffer_match_radius_m)),
        high_confidence_distance_m=max(0.0, float(high_confidence_distance_m)),
        medium_confidence_distance_m=max(float(high_confidence_distance_m), float(medium_confidence_distance_m)),
        event_year_tolerance_years=max(0, int(event_year_tolerance_years)),
        enable_global_nearest_fallback=bool(enable_global_nearest_fallback),
        allow_duplicate_outcome_matches=bool(allow_duplicate_outcome_matches),
        address_token_overlap_min=max(0.35, min(1.0, float(address_token_overlap_min))),
    )
    feature_spatial_context_by_identity = _build_feature_spatial_context(feature_rows)
    def _run_join_pass(*, pass_name: str, pass_join_config: JoinConfig, retention_fallback_mode: bool) -> dict[str, Any]:
        joined_rows_pass: list[dict[str, Any]] = []
        excluded_rows_pass: list[dict[str, Any]] = list(missing_feature_artifacts)
        leakage_warnings_pass: list[str] = []
        join_method_counts_pass: dict[str, int] = {}
        join_tier_counts_pass: dict[str, int] = {}
        match_tier_counts_pass: dict[str, int] = {}
        join_tier_distance_values_pass: dict[str, list[float]] = {}
        join_tier_examples_pass: dict[str, list[dict[str, Any]]] = {}
        feature_coordinate_modes_pass: dict[str, int] = {}
        outcome_coordinate_modes_pass: dict[str, int] = {}
        duplicate_outcome_match_prevented_count_pass = 0
        candidate_match_attempt_count_pass = 0
        matched_candidate_count_before_filters_pass = 0
        soft_flag_counts_pass: dict[str, int] = {}
        non_high_reason_counts_pass: dict[str, int] = {}
        within_high_distance_threshold_count_pass = 0
        just_above_high_distance_threshold_count_pass = 0
        just_below_high_score_threshold_count_pass = 0
        used_outcome_keys_pass: set[str] = set()
        by_event_counts_pass: dict[str, int] = {}
        by_label_counts_pass: dict[str, int] = {}
        unmatched_feature_rows_by_event_counts_pass: dict[str, int] = {}
        low_confidence_join_count_pass = 0

        for feature in sorted(feature_rows, key=_stable_feature_sort_key):
            candidate_match_attempt_count_pass += 1
            feature_coord_mode = str(feature.payload.get("_coordinate_normalization_mode") or "unknown")
            feature_coordinate_modes_pass[feature_coord_mode] = feature_coordinate_modes_pass.get(feature_coord_mode, 0) + 1
            matched, join_meta = _choose_outcome(
                feature,
                indexes,
                join_config=pass_join_config,
                excluded_outcome_keys=(used_outcome_keys_pass if not pass_join_config.allow_duplicate_outcome_matches else None),
            )
            method = str(join_meta.get("join_method") or "unmatched")
            if matched is None:
                unmatched_event_key = str(feature.event_id or "unknown_event")
                unmatched_feature_rows_by_event_counts_pass[unmatched_event_key] = (
                    unmatched_feature_rows_by_event_counts_pass.get(unmatched_event_key, 0) + 1
                )
                excluded_rows_pass.append(
                    {
                        "feature_artifact_path": feature.artifact_path,
                        "record_id": feature.record_id,
                        "event_id": feature.event_id,
                        "reason": str(join_meta.get("unmatched_reason") or "no_outcome_match_within_constraints"),
                        "join_pass": pass_name,
                    }
                )
                continue
            matched_candidate_count_before_filters_pass += 1
            matched_key = _outcome_identity_key(matched)
            if not pass_join_config.allow_duplicate_outcome_matches:
                if matched_key in used_outcome_keys_pass:
                    duplicate_outcome_match_prevented_count_pass += 1
                    excluded_rows_pass.append(
                        {
                            "feature_artifact_path": feature.artifact_path,
                            "record_id": feature.record_id,
                            "event_id": feature.event_id,
                            "reason": "duplicate_outcome_match_prevented",
                            "outcome_identity_key": matched_key,
                            "join_pass": pass_name,
                        }
                    )
                    continue
                used_outcome_keys_pass.add(matched_key)
            outcome_coord_mode = str(matched.payload.get("_coordinate_normalization_mode") or "unknown")
            outcome_coordinate_modes_pass[outcome_coord_mode] = outcome_coordinate_modes_pass.get(outcome_coord_mode, 0) + 1

            join_method_counts_pass[method] = join_method_counts_pass.get(method, 0) + 1
            tier = str(join_meta.get("join_confidence_tier") or "low")
            join_tier_counts_pass[tier] = join_tier_counts_pass.get(tier, 0) + 1
            match_tier = str(join_meta.get("match_tier") or "unknown")
            match_tier_counts_pass[match_tier] = match_tier_counts_pass.get(match_tier, 0) + 1
            join_distance_value = _safe_float(join_meta.get("join_distance_m"))
            join_score_value = _safe_float(join_meta.get("join_confidence_score")) or 0.0
            max_allowed_tier = str(join_meta.get("join_confidence_max_allowed_tier") or _join_max_allowed_tier_for_method(method))
            join_confidence_debug = _build_join_confidence_debug(
                score=join_score_value,
                distance_m=join_distance_value,
                high_confidence_distance_m=pass_join_config.high_confidence_distance_m,
                medium_confidence_distance_m=pass_join_config.medium_confidence_distance_m,
                max_allowed_tier=max_allowed_tier,
                resolved_tier=tier,
            )
            join_meta["join_confidence_max_allowed_tier"] = max_allowed_tier
            join_meta["join_confidence_debug"] = join_confidence_debug
            join_meta["join_confidence_non_high_reason_codes"] = list(join_confidence_debug.get("non_high_reason_codes") or [])
            if bool(join_confidence_debug.get("near_high_distance_threshold")):
                just_above_high_distance_threshold_count_pass += 1
            if bool(join_confidence_debug.get("just_below_high_score_threshold")):
                just_below_high_score_threshold_count_pass += 1
            if join_distance_value is not None and float(join_distance_value) <= float(pass_join_config.high_confidence_distance_m):
                within_high_distance_threshold_count_pass += 1
            for reason in (join_confidence_debug.get("non_high_reason_codes") if isinstance(join_confidence_debug.get("non_high_reason_codes"), list) else []):
                token = str(reason)
                non_high_reason_counts_pass[token] = non_high_reason_counts_pass.get(token, 0) + 1
            if join_distance_value is not None:
                join_tier_distance_values_pass.setdefault(tier, []).append(float(join_distance_value))
            examples = join_tier_examples_pass.setdefault(tier, [])
            if len(examples) < 3:
                examples.append(
                    {
                        "feature_record_id": feature.record_id or feature.source_record_id,
                        "outcome_record_id": matched.record_id or matched.source_record_id,
                        "event_id": feature.event_id or matched.event_id,
                        "join_method": method,
                        "join_distance_m": (round(float(join_distance_value), 2) if join_distance_value is not None else None),
                        "address_text": feature.payload.get("address_text") or matched.payload.get("address_text"),
                    }
                )
            if tier == "low":
                low_confidence_join_count_pass += 1

            severity, binary = _derive_severity_and_binary(matched.payload)
            leak_flags = _detect_leakage_flags(feature.payload, feature.payload.get("event_date"))
            if leak_flags:
                leakage_warnings_pass.append(
                    f"{feature.event_id or 'unknown_event'}/{feature.record_id or 'unknown_record'}: {','.join(leak_flags)}"
                )

            caveat_flags: list[str] = []
            if not _event_year_consistent_with_tolerance(feature, matched, pass_join_config.event_year_tolerance_years):
                caveat_flags.append("event_year_mismatch")
            distance_m = _safe_float(join_meta.get("join_distance_m"))
            if distance_m is not None and distance_m > float(pass_join_config.max_distance_m) * 0.6:
                caveat_flags.append("high_join_distance")
            if tier == "low":
                caveat_flags.append("low_confidence_join")
            if leak_flags:
                caveat_flags.append("leakage_warning_present")
            if retention_fallback_mode:
                caveat_flags.append("retention_fallback_mode")

            extracted_scores = _extract_feature_scores(feature.payload)
            wildfire_risk = extracted_scores.get("wildfire_risk_score")
            site_hazard = extracted_scores.get("site_hazard_score")
            home_vuln = extracted_scores.get("home_ignition_vulnerability_score")
            readiness = extracted_scores.get("insurance_readiness_score")
            raw_feature_vector = (
                feature.payload.get("raw_feature_vector")
                if isinstance(feature.payload.get("raw_feature_vector"), dict)
                else {}
            )
            transformed_feature_vector = (
                feature.payload.get("transformed_feature_vector")
                if isinstance(feature.payload.get("transformed_feature_vector"), dict)
                else {}
            )
            enriched_raw_feature_vector, enriched_transformed_feature_vector, feature_observation_summary = (
                _enrich_feature_vectors_with_property_proxies(
                    feature=feature,
                    raw_feature_vector=raw_feature_vector,
                    transformed_feature_vector=transformed_feature_vector,
                    spatial_context=feature_spatial_context_by_identity.get(_feature_identity_key(feature)),
                )
            )

            confidence_payload = feature.payload.get("confidence") if isinstance(feature.payload.get("confidence"), dict) else {}
            evidence_summary = feature.payload.get("evidence_quality_summary") if isinstance(feature.payload.get("evidence_quality_summary"), dict) else {}
            coverage_summary = feature.payload.get("coverage_summary") if isinstance(feature.payload.get("coverage_summary"), dict) else {}
            missing_feature_fields = sorted(
                [
                    key
                    for key, value in {
                        "wildfire_risk_score": wildfire_risk,
                        "site_hazard_score": site_hazard,
                        "home_ignition_vulnerability_score": home_vuln,
                        "insurance_readiness_score": readiness,
                    }.items()
                    if value is None
                ]
            )
            evidence_tier = str((evidence_summary or {}).get("evidence_tier") or "").strip().lower()
            fallback_usage = _derive_fallback_usage(
                evidence_summary=evidence_summary,
                coverage_summary=coverage_summary,
                evidence_tier=evidence_tier,
            )
            fallback_heavy = bool(fallback_usage.get("fallback_heavy"))
            soft_flags: list[str] = []
            if tier == "low":
                soft_flags.append("low_confidence_join")
            if missing_feature_fields:
                soft_flags.append("missing_features")
            if fallback_heavy:
                soft_flags.append("fallback_heavy")
            if retention_fallback_mode:
                soft_flags.append("retention_fallback_mode")
            if int(feature_observation_summary.get("inferred_count") or 0) > 0:
                soft_flags.append("inferred_structure_or_vegetation_proxies")
            if int(feature_observation_summary.get("missing_count") or 0) > 0:
                soft_flags.append("missing_structure_or_vegetation_proxies")
            for flag in soft_flags:
                soft_flag_counts_pass[flag] = soft_flag_counts_pass.get(flag, 0) + 1
            if fallback_heavy:
                caveat_flags.append("fallback_heavy_evidence")
            elif float(fallback_usage.get("fallback_weight_fraction") or 0.0) >= FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD:
                caveat_flags.append("elevated_fallback_usage")
            if int(feature_observation_summary.get("inferred_count") or 0) > 0:
                caveat_flags.append("inferred_structure_or_vegetation_features")
            if int(feature_observation_summary.get("missing_count") or 0) > 0:
                caveat_flags.append("missing_structure_or_vegetation_features")
            row_confidence_tier = _derive_row_confidence_tier(
                join_confidence_tier=str(join_meta.get("join_confidence_tier") or "low"),
                model_confidence_tier=str(confidence_payload.get("confidence_tier") or feature.payload.get("confidence_tier") or "unknown"),
                evidence_quality_tier=str(evidence_summary.get("evidence_tier") or "unknown"),
            )

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
                    "coordinate_normalization_mode": feature.payload.get("_coordinate_normalization_mode"),
                    "parcel_identifier": feature.payload.get("parcel_identifier") or feature.payload.get("parcel_id"),
                    "property_level_context": (
                        feature.payload.get("property_level_context")
                        if isinstance(feature.payload.get("property_level_context"), dict)
                        else {}
                    ),
                },
                "outcome": {
                    "record_id": matched.payload.get("record_id"),
                    "source_record_id": matched.payload.get("source_record_id"),
                    "source_name": matched.payload.get("source_name"),
                    "source_path": matched.payload.get("source_path"),
                    "address_text": matched.payload.get("address_text"),
                    "latitude": matched.latitude,
                    "longitude": matched.longitude,
                    "coordinate_normalization_mode": matched.payload.get("_coordinate_normalization_mode"),
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
                    "raw_feature_vector": enriched_raw_feature_vector,
                    "transformed_feature_vector": enriched_transformed_feature_vector,
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
                    "max_distance_m": float(pass_join_config.max_distance_m),
                    "high_confidence_distance_m": float(pass_join_config.high_confidence_distance_m),
                    "medium_confidence_distance_m": float(pass_join_config.medium_confidence_distance_m),
                    "global_max_distance_m": float(pass_join_config.global_max_distance_m),
                    "event_year_tolerance_years": int(pass_join_config.event_year_tolerance_years),
                    "global_fallback_enabled": bool(pass_join_config.enable_global_nearest_fallback),
                    "event_year_consistent": _event_year_consistent_with_tolerance(
                        feature,
                        matched,
                        pass_join_config.event_year_tolerance_years,
                    ),
                    "join_pass": pass_name,
                    "retention_fallback_mode": bool(retention_fallback_mode),
                },
                "evaluation": {
                    "row_usable": True,
                    "row_confidence_tier": row_confidence_tier,
                    "soft_filter_flags": soft_flags,
                    "missing_feature_fields": missing_feature_fields,
                    "fallback_heavy": bool(fallback_heavy),
                    "fallback_usage": fallback_usage,
                    "feature_observation_summary": feature_observation_summary,
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
            joined_rows_pass.append(joined)

            event_key = str(joined["event"]["event_id"] or "unknown_event")
            by_event_counts_pass[event_key] = by_event_counts_pass.get(event_key, 0) + 1
            label_key = str(joined["outcome"]["damage_label"] or "unknown")
            by_label_counts_pass[label_key] = by_label_counts_pass.get(label_key, 0) + 1

        joined_rows_pass.sort(key=lambda row: str(row.get("property_event_id") or ""))
        return {
            "joined_rows": joined_rows_pass,
            "excluded_rows": excluded_rows_pass,
            "leakage_warnings": leakage_warnings_pass,
            "join_method_counts": join_method_counts_pass,
            "join_tier_counts": join_tier_counts_pass,
            "match_tier_counts": match_tier_counts_pass,
            "join_tier_distance_values": join_tier_distance_values_pass,
            "join_tier_examples": join_tier_examples_pass,
            "feature_coordinate_modes": feature_coordinate_modes_pass,
            "outcome_coordinate_modes": outcome_coordinate_modes_pass,
            "duplicate_outcome_match_prevented_count": duplicate_outcome_match_prevented_count_pass,
            "candidate_match_attempt_count": candidate_match_attempt_count_pass,
            "matched_candidate_count_before_filters": matched_candidate_count_before_filters_pass,
            "soft_flag_counts": soft_flag_counts_pass,
            "non_high_reason_counts": non_high_reason_counts_pass,
            "within_high_distance_threshold_count": within_high_distance_threshold_count_pass,
            "just_above_high_distance_threshold_count": just_above_high_distance_threshold_count_pass,
            "just_below_high_score_threshold_count": just_below_high_score_threshold_count_pass,
            "by_event_counts": by_event_counts_pass,
            "by_label_counts": by_label_counts_pass,
            "unmatched_feature_rows_by_event_counts": unmatched_feature_rows_by_event_counts_pass,
            "low_confidence_join_count": low_confidence_join_count_pass,
        }

    primary_pass = _run_join_pass(pass_name="primary", pass_join_config=join_config, retention_fallback_mode=False)
    selected_pass = primary_pass
    retention_target = max(0, int(min_retained_records))
    retention_fallback_summary: dict[str, Any] = {
        "enabled": bool(auto_relax_for_min_retention and retention_target > 0),
        "triggered": False,
        "used": False,
        "target_min_records": retention_target,
        "primary_joined_records": len(primary_pass["joined_rows"]),
        "fallback_joined_records": None,
        "active_pass": "primary",
        "reason": None,
        "relaxed_join_config": None,
    }
    if retention_fallback_summary["enabled"] and len(primary_pass["joined_rows"]) < retention_target:
        retention_fallback_summary["triggered"] = True
        retention_fallback_summary["reason"] = (
            f"Primary join produced {len(primary_pass['joined_rows'])} rows, below minimum retention target {retention_target}."
        )
        relaxed_join_config = JoinConfig(
            exact_match_distance_m=float(join_config.exact_match_distance_m),
            near_match_distance_m=max(float(join_config.near_match_distance_m), 75.0),
            max_distance_m=max(float(join_config.max_distance_m), 500.0),
            global_max_distance_m=max(float(join_config.global_max_distance_m), 5000.0),
            buffer_match_radius_m=max(float(join_config.buffer_match_radius_m), 250.0),
            high_confidence_distance_m=float(join_config.high_confidence_distance_m),
            medium_confidence_distance_m=max(float(join_config.medium_confidence_distance_m), 250.0),
            event_year_tolerance_years=max(int(join_config.event_year_tolerance_years), 2),
            enable_global_nearest_fallback=True,
            allow_duplicate_outcome_matches=True,
            address_token_overlap_min=min(float(join_config.address_token_overlap_min), 0.45),
        )
        fallback_pass = _run_join_pass(
            pass_name="retention_fallback_relaxed",
            pass_join_config=relaxed_join_config,
            retention_fallback_mode=True,
        )
        retention_fallback_summary["fallback_joined_records"] = len(fallback_pass["joined_rows"])
        retention_fallback_summary["relaxed_join_config"] = {
            "exact_match_distance_m": float(relaxed_join_config.exact_match_distance_m),
            "near_match_distance_m": float(relaxed_join_config.near_match_distance_m),
            "max_distance_m": float(relaxed_join_config.max_distance_m),
            "global_max_distance_m": float(relaxed_join_config.global_max_distance_m),
            "buffer_match_radius_m": float(relaxed_join_config.buffer_match_radius_m),
            "high_confidence_distance_m": float(relaxed_join_config.high_confidence_distance_m),
            "medium_confidence_distance_m": float(relaxed_join_config.medium_confidence_distance_m),
            "event_year_tolerance_years": int(relaxed_join_config.event_year_tolerance_years),
            "enable_global_nearest_fallback": bool(relaxed_join_config.enable_global_nearest_fallback),
            "allow_duplicate_outcome_matches": bool(relaxed_join_config.allow_duplicate_outcome_matches),
            "address_token_overlap_min": float(relaxed_join_config.address_token_overlap_min),
        }
        if len(fallback_pass["joined_rows"]) >= len(primary_pass["joined_rows"]):
            selected_pass = fallback_pass
            join_config = relaxed_join_config
            retention_fallback_summary["used"] = True
            retention_fallback_summary["active_pass"] = "retention_fallback_relaxed"
        else:
            retention_fallback_summary["active_pass"] = "primary"

    joined_rows_pre_property_event_dedupe = list(selected_pass["joined_rows"])
    joined_rows, dedupe_removed_rows, property_event_dedupe_stats = _dedupe_joined_rows_by_property_event_id(
        joined_rows_pre_property_event_dedupe
    )
    excluded_rows = list(selected_pass["excluded_rows"]) + dedupe_removed_rows
    leakage_warnings = list(selected_pass["leakage_warnings"])
    join_method_counts: dict[str, int] = {}
    join_tier_counts: dict[str, int] = {}
    match_tier_counts: dict[str, int] = {}
    join_tier_distance_values: dict[str, list[float]] = {}
    join_tier_examples: dict[str, list[dict[str, Any]]] = {}
    feature_coordinate_modes = dict(selected_pass["feature_coordinate_modes"])
    outcome_coordinate_modes = dict(selected_pass["outcome_coordinate_modes"])
    duplicate_outcome_match_prevented_count = int(selected_pass["duplicate_outcome_match_prevented_count"])
    candidate_match_attempt_count = int(selected_pass["candidate_match_attempt_count"])
    matched_candidate_count_before_filters = int(selected_pass["matched_candidate_count_before_filters"])
    soft_flag_counts: dict[str, int] = {}
    non_high_reason_counts: dict[str, int] = {}
    within_high_distance_threshold_count = 0
    just_above_high_distance_threshold_count = 0
    just_below_high_score_threshold_count = 0
    by_event_counts: dict[str, int] = {}
    by_label_counts: dict[str, int] = {}
    unmatched_feature_rows_by_event_counts = dict(selected_pass["unmatched_feature_rows_by_event_counts"])
    low_confidence_join_count = 0
    fallback_usage_classification_counts: dict[str, int] = {}
    fallback_usage_reason_counts: dict[str, int] = {}
    fallback_weight_values: list[float] = []
    rows_with_any_fallback_inputs = 0
    rows_with_elevated_fallback_weight = 0
    for row in joined_rows:
        join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
        method = str(join_meta.get("join_method") or "unknown")
        tier = str(join_meta.get("join_confidence_tier") or "low")
        match_tier = str(join_meta.get("match_tier") or "unknown")
        join_method_counts[method] = join_method_counts.get(method, 0) + 1
        join_tier_counts[tier] = join_tier_counts.get(tier, 0) + 1
        match_tier_counts[match_tier] = match_tier_counts.get(match_tier, 0) + 1
        if tier == "low":
            low_confidence_join_count += 1
        distance_value = _safe_float(join_meta.get("join_distance_m"))
        if distance_value is not None:
            join_tier_distance_values.setdefault(tier, []).append(float(distance_value))
            if float(distance_value) <= float(join_config.high_confidence_distance_m):
                within_high_distance_threshold_count += 1
        debug_payload = join_meta.get("join_confidence_debug") if isinstance(join_meta.get("join_confidence_debug"), dict) else {}
        if bool(debug_payload.get("near_high_distance_threshold")):
            just_above_high_distance_threshold_count += 1
        if bool(debug_payload.get("just_below_high_score_threshold")):
            just_below_high_score_threshold_count += 1
        non_high_codes = (
            debug_payload.get("non_high_reason_codes")
            if isinstance(debug_payload.get("non_high_reason_codes"), list)
            else []
        )
        for reason in non_high_codes:
            token = str(reason)
            non_high_reason_counts[token] = non_high_reason_counts.get(token, 0) + 1
        examples = join_tier_examples.setdefault(tier, [])
        if len(examples) < 3:
            examples.append(
                {
                    "feature_record_id": (row.get("feature") or {}).get("record_id"),
                    "outcome_record_id": (row.get("outcome") or {}).get("record_id"),
                    "event_id": (row.get("event") or {}).get("event_id"),
                    "join_method": method,
                    "join_distance_m": (round(float(distance_value), 2) if distance_value is not None else None),
                    "address_text": (row.get("feature") or {}).get("address_text") or (row.get("outcome") or {}).get("address_text"),
                }
            )
        for flag in ((row.get("evaluation") or {}).get("soft_filter_flags") or []):
            soft_flag_counts[str(flag)] = soft_flag_counts.get(str(flag), 0) + 1
        fallback_usage = ((row.get("evaluation") or {}).get("fallback_usage"))
        fallback_usage = fallback_usage if isinstance(fallback_usage, dict) else {}
        fallback_class = str(fallback_usage.get("classification") or "unknown")
        fallback_usage_classification_counts[fallback_class] = fallback_usage_classification_counts.get(fallback_class, 0) + 1
        fallback_weight = _safe_float(fallback_usage.get("fallback_weight_fraction"))
        if fallback_weight is not None:
            fallback_weight_values.append(float(fallback_weight))
            if float(fallback_weight) >= FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD:
                rows_with_elevated_fallback_weight += 1
        fallback_factor_count = int(_safe_float(fallback_usage.get("fallback_factor_count")) or 0)
        coverage_fallback_count = int(_safe_float(fallback_usage.get("coverage_fallback_count")) or 0)
        if fallback_factor_count > 0 or coverage_fallback_count > 0:
            rows_with_any_fallback_inputs += 1
        for reason in (fallback_usage.get("fallback_heavy_reasons") if isinstance(fallback_usage.get("fallback_heavy_reasons"), list) else []):
            token = str(reason)
            fallback_usage_reason_counts[token] = fallback_usage_reason_counts.get(token, 0) + 1
        event_key = str((row.get("event") or {}).get("event_id") or "unknown_event")
        by_event_counts[event_key] = by_event_counts.get(event_key, 0) + 1
        label_key = str((row.get("outcome") or {}).get("damage_label") or "unknown")
        by_label_counts[label_key] = by_label_counts.get(label_key, 0) + 1
    join_rate = round(len(joined_rows) / float(len(feature_rows)), 4) if feature_rows else 0.0
    join_distances = [
        float((row.get("join_metadata") or {}).get("join_distance_m"))
        for row in joined_rows
        if _safe_float((row.get("join_metadata") or {}).get("join_distance_m")) is not None
    ]
    mean_join_distance = (
        round(sum(join_distances) / float(len(join_distances)), 3)
        if join_distances
        else None
    )
    sorted_distances = sorted(join_distances)
    median_join_distance = (
        (
            sorted_distances[len(sorted_distances) // 2]
            if len(sorted_distances) % 2 == 1
            else (sorted_distances[(len(sorted_distances) // 2) - 1] + sorted_distances[len(sorted_distances) // 2]) / 2.0
        )
        if sorted_distances
        else None
    )
    distance_histogram_bins = [10.0, 20.0, 50.0, 100.0, 250.0]
    distance_histogram: dict[str, int] = {
        "0_10m": 0,
        "10_20m": 0,
        "20_50m": 0,
        "50_100m": 0,
        "100_250m": 0,
        "250m_plus": 0,
    }
    for distance in join_distances:
        d = float(distance)
        if d < distance_histogram_bins[0]:
            distance_histogram["0_10m"] += 1
        elif d < distance_histogram_bins[1]:
            distance_histogram["10_20m"] += 1
        elif d < distance_histogram_bins[2]:
            distance_histogram["20_50m"] += 1
        elif d < distance_histogram_bins[3]:
            distance_histogram["50_100m"] += 1
        elif d < distance_histogram_bins[4]:
            distance_histogram["100_250m"] += 1
        else:
            distance_histogram["250m_plus"] += 1
    p95_distance = None
    if sorted_distances:
        p95_idx = max(0, min(len(sorted_distances) - 1, int(math.ceil(0.95 * len(sorted_distances)) - 1)))
        p95_distance = round(float(sorted_distances[p95_idx]), 3)
    outlier_threshold = max(float(join_config.medium_confidence_distance_m), float(p95_distance or 0.0))
    distance_outliers: list[dict[str, Any]] = []
    for row in joined_rows:
        join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
        distance_m = _safe_float(join_meta.get("join_distance_m"))
        if distance_m is None or distance_m < outlier_threshold:
            continue
        distance_outliers.append(
            {
                "feature_record_id": (row.get("feature") or {}).get("record_id"),
                "event_id": (row.get("event") or {}).get("event_id"),
                "join_method": join_meta.get("join_method"),
                "join_confidence_tier": join_meta.get("join_confidence_tier"),
                "join_distance_m": round(float(distance_m), 3),
                "match_tier": join_meta.get("match_tier"),
            }
        )
    distance_outliers.sort(key=lambda item: float(item.get("join_distance_m") or 0.0), reverse=True)
    distance_outliers = distance_outliers[:20]
    confidence_scores = [
        _safe_float((row.get("join_metadata") or {}).get("join_confidence_score"))
        for row in joined_rows
    ]
    confidence_scores = [value for value in confidence_scores if value is not None]
    fallback_weight_values_sorted = sorted(fallback_weight_values)
    fallback_weight_median = None
    if fallback_weight_values_sorted:
        midpoint = len(fallback_weight_values_sorted) // 2
        if len(fallback_weight_values_sorted) % 2 == 1:
            fallback_weight_median = fallback_weight_values_sorted[midpoint]
        else:
            fallback_weight_median = (
                fallback_weight_values_sorted[midpoint - 1]
                + fallback_weight_values_sorted[midpoint]
            ) / 2.0
    row_confidence_tiers: dict[str, int] = {}
    for row in joined_rows:
        tier = str(((row.get("evaluation") or {}).get("row_confidence_tier")) or "unknown")
        row_confidence_tiers[tier] = row_confidence_tiers.get(tier, 0) + 1
    excluded_reason_counts: dict[str, int] = {}
    for row in excluded_rows:
        reason = str((row or {}).get("reason") or "unknown")
        excluded_reason_counts[reason] = excluded_reason_counts.get(reason, 0) + 1
    feature_variation_diagnostics = _compute_feature_variation_diagnostics(joined_rows)
    diversity_spread = _compute_diversity_spread(joined_rows)

    join_quality = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "total_outcomes_loaded": len(outcomes),
        "outcomes_loaded_by_source_path": dict(sorted(outcomes_by_source.items())),
        "outcomes_by_event_counts": dict(sorted(outcomes_by_event_counts.items())),
        "outcomes_by_year_counts": dict(sorted(outcomes_by_year_counts.items())),
        "total_feature_rows_loaded": len(feature_rows),
        "candidate_match_attempt_count": candidate_match_attempt_count,
        "matched_candidate_count_before_filters": matched_candidate_count_before_filters,
        "feature_rows_by_event_counts": dict(sorted(feature_rows_by_event_counts.items())),
        "feature_rows_by_year_counts": dict(sorted(feature_rows_by_year_counts.items())),
        "total_rows_before_property_event_dedupe": int(
            property_event_dedupe_stats.get("total_rows_before_property_event_dedupe") or 0
        ),
        "unique_property_event_id_count": int(property_event_dedupe_stats.get("unique_property_event_id_count") or 0),
        "duplicate_property_event_rows_removed_count": int(
            property_event_dedupe_stats.get("duplicate_property_event_rows_removed_count") or 0
        ),
        "property_event_ids_with_duplicates_count": int(
            property_event_dedupe_stats.get("property_event_ids_with_duplicates_count") or 0
        ),
        "duplication_factor": property_event_dedupe_stats.get("duplication_factor"),
        "duplicate_property_event_ids_examples": (
            property_event_dedupe_stats.get("duplicate_property_event_ids_examples")
            if isinstance(property_event_dedupe_stats.get("duplicate_property_event_ids_examples"), list)
            else []
        ),
        "duplicate_rows_removed_by_join_confidence_tier": (
            property_event_dedupe_stats.get("duplicate_rows_removed_by_join_confidence_tier")
            if isinstance(property_event_dedupe_stats.get("duplicate_rows_removed_by_join_confidence_tier"), dict)
            else {}
        ),
        "duplicate_rows_removed_by_join_method": (
            property_event_dedupe_stats.get("duplicate_rows_removed_by_join_method")
            if isinstance(property_event_dedupe_stats.get("duplicate_rows_removed_by_join_method"), dict)
            else {}
        ),
        "total_joined_records": len(joined_rows),
        "join_rate": join_rate,
        "match_rate_percent": round(join_rate * 100.0, 2),
        "min_join_distance_m": (round(float(min(sorted_distances)), 3) if sorted_distances else None),
        "average_join_distance_m": mean_join_distance,
        "median_join_distance_m": (round(float(median_join_distance), 3) if median_join_distance is not None else None),
        "max_join_distance_m": (round(float(max(sorted_distances)), 3) if sorted_distances else None),
        "distance_percentiles_m": {
            "p50": (round(float(median_join_distance), 3) if median_join_distance is not None else None),
            "p95": p95_distance,
        },
        "match_distance_histogram_m": distance_histogram,
        "distance_outlier_threshold_m": round(outlier_threshold, 3),
        "distance_outlier_examples": distance_outliers,
        "join_method_counts": dict(sorted(join_method_counts.items())),
        "join_confidence_tier_counts": dict(sorted(join_tier_counts.items())),
        "join_confidence_tier_distance_stats": {
            tier: {
                "count": len(values),
                "mean_distance_m": round(sum(values) / float(len(values)), 3),
                "min_distance_m": round(min(values), 3),
                "max_distance_m": round(max(values), 3),
            }
            for tier, values in sorted(join_tier_distance_values.items())
            if values
        },
        "join_confidence_tier_examples": dict(sorted(join_tier_examples.items())),
        "match_tier_counts": dict(sorted(match_tier_counts.items())),
        "row_confidence_tier_counts": dict(sorted(row_confidence_tiers.items())),
        "fallback_usage_summary": {
            "classification_counts": dict(sorted(fallback_usage_classification_counts.items())),
            "fallback_heavy_count": int(fallback_usage_classification_counts.get("fallback_heavy") or 0),
            "fallback_heavy_fraction": (
                round(float(fallback_usage_classification_counts.get("fallback_heavy") or 0) / float(len(joined_rows)), 4)
                if joined_rows
                else 0.0
            ),
            "rows_with_any_fallback_inputs": int(rows_with_any_fallback_inputs),
            "rows_with_elevated_fallback_weight": int(rows_with_elevated_fallback_weight),
            "mean_fallback_weight_fraction": (
                round(sum(fallback_weight_values) / float(len(fallback_weight_values)), 4)
                if fallback_weight_values
                else None
            ),
            "median_fallback_weight_fraction": (
                round(float(fallback_weight_median), 4)
                if fallback_weight_median is not None
                else None
            ),
            "fallback_heavy_reason_counts": dict(sorted(fallback_usage_reason_counts.items())),
            "thresholds": {
                "fallback_heavy_weight_threshold": FALLBACK_HEAVY_WEIGHT_THRESHOLD,
                "fallback_heavy_elevated_weight_threshold": FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD,
                "fallback_heavy_factor_ratio_threshold": FALLBACK_HEAVY_FACTOR_RATIO_THRESHOLD,
                "fallback_heavy_missing_ratio_threshold": FALLBACK_HEAVY_MISSING_RATIO_THRESHOLD,
                "fallback_heavy_coverage_fallback_count_threshold": FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD,
            },
        },
        "join_confidence_score_stats": {
            "mean": (round(sum(confidence_scores) / float(len(confidence_scores)), 4) if confidence_scores else None),
            "min": (round(min(confidence_scores), 4) if confidence_scores else None),
            "max": (round(max(confidence_scores), 4) if confidence_scores else None),
            "count": len(confidence_scores),
        },
        "low_confidence_join_count": low_confidence_join_count,
        "duplicate_outcome_match_prevented_count": duplicate_outcome_match_prevented_count,
        "allow_duplicate_outcome_matches": bool(join_config.allow_duplicate_outcome_matches),
        "coordinate_normalization_summary": {
            "feature_rows_by_mode": dict(sorted(feature_coordinate_modes.items())),
            "matched_outcomes_by_mode": dict(sorted(outcome_coordinate_modes.items())),
        },
        "feature_variation_diagnostics": feature_variation_diagnostics,
        "diversity_spread": diversity_spread,
        "by_event_join_counts": dict(sorted(by_event_counts.items())),
        "by_label_join_counts": dict(sorted(by_label_counts.items())),
        "unmatched_feature_rows_by_event_counts": dict(sorted(unmatched_feature_rows_by_event_counts.items())),
        "excluded_row_count": len(excluded_rows),
        "filtered_records_count": len(excluded_rows),
        "final_dataset_size": len(joined_rows),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "excluded_rows": excluded_rows[:200],
        "leakage_warnings": sorted(set(leakage_warnings)),
        "score_backfill": score_backfill,
        "outcome_load_diagnostics": outcome_load_diagnostics,
        "feature_load_diagnostics": feature_load_diagnostics,
        "soft_flag_counts": dict(sorted(soft_flag_counts.items())),
        "join_confidence_non_high_reason_counts": dict(sorted(non_high_reason_counts.items())),
        "high_confidence_threshold_diagnostics": {
            "high_confidence_distance_threshold_m": float(join_config.high_confidence_distance_m),
            "medium_confidence_distance_threshold_m": float(join_config.medium_confidence_distance_m),
            "high_confidence_score_threshold": float(HIGH_CONFIDENCE_SCORE_THRESHOLD),
            "within_high_distance_threshold_count": within_high_distance_threshold_count,
            "just_above_high_distance_threshold_count": just_above_high_distance_threshold_count,
            "just_below_high_score_threshold_count": just_below_high_score_threshold_count,
            "high_confidence_match_count": int(join_tier_counts.get("high") or 0),
            "moderate_confidence_match_count": int(join_tier_counts.get("moderate") or 0),
            "low_confidence_match_count": int(join_tier_counts.get("low") or 0),
        },
        "retention_fallback": retention_fallback_summary,
        "retention_fallback_triggered": bool(retention_fallback_summary.get("triggered")),
        "retention_fallback_used": bool(retention_fallback_summary.get("used")),
        "retention_min_records_target": int(retention_target),
        "warning_thresholds": {
            "high_average_distance_m": round(float(join_config.medium_confidence_distance_m), 3),
        },
    }
    filter_reason_counts: dict[str, int] = {}
    for source_counts in (
        (outcome_load_diagnostics.get("outcome_prejoin_filter_reason_counts") if isinstance(outcome_load_diagnostics, dict) else {}),
        (feature_load_diagnostics.get("feature_prejoin_filter_reason_counts") if isinstance(feature_load_diagnostics, dict) else {}),
        excluded_reason_counts,
    ):
        if not isinstance(source_counts, dict):
            continue
        for key, value in source_counts.items():
            filter_reason_counts[str(key)] = filter_reason_counts.get(str(key), 0) + int(value or 0)
    total_filtered_rows = int(sum(int(v or 0) for v in filter_reason_counts.values()))
    filter_reason_percentages = (
        {
            key: round((float(value) / float(total_filtered_rows)) * 100.0, 3)
            for key, value in sorted(filter_reason_counts.items())
            if total_filtered_rows > 0
        }
        if total_filtered_rows > 0
        else {}
    )
    post_join_filtered_count = int(sum(excluded_reason_counts.values()))
    accounting = {
        "candidate_match_attempt_count": candidate_match_attempt_count,
        "joined_rows": len(joined_rows),
        "post_join_filtered_rows": post_join_filtered_count,
        "joined_plus_post_join_filtered": len(joined_rows) + post_join_filtered_count,
        "matches_attempts_accounted": (len(joined_rows) + post_join_filtered_count) == candidate_match_attempt_count,
    }
    filter_summary = {
        "outcomes_raw_rows": int(outcome_load_diagnostics.get("total_raw_row_count") or 0),
        "outcomes_after_prejoin_filters": len(outcomes),
        "features_raw_rows": int(feature_load_diagnostics.get("total_raw_row_count") or 0),
        "feature_candidate_rows": len(feature_rows),
        "candidate_match_attempt_count": candidate_match_attempt_count,
        "matched_candidate_count_before_filters": matched_candidate_count_before_filters,
        "joined_rows": len(joined_rows),
        "joined_rows_before_property_event_dedupe": int(
            property_event_dedupe_stats.get("total_rows_before_property_event_dedupe") or 0
        ),
        "unique_property_event_id_count": int(property_event_dedupe_stats.get("unique_property_event_id_count") or 0),
        "duplicate_property_event_rows_removed_count": int(
            property_event_dedupe_stats.get("duplicate_property_event_rows_removed_count") or 0
        ),
        "duplication_factor": property_event_dedupe_stats.get("duplication_factor"),
        "filtered_rows_total": total_filtered_rows,
        "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
        "filter_reason_percentages": dict(sorted(filter_reason_percentages.items())),
        "soft_flag_counts": dict(sorted(soft_flag_counts.items())),
        "retention_fallback_triggered": bool(retention_fallback_summary.get("triggered")),
        "retention_fallback_used": bool(retention_fallback_summary.get("used")),
        "retention_min_records_target": int(retention_target),
        "accounting": accounting,
        "no_silent_data_loss_guarantee": bool(accounting.get("matches_attempts_accounted")),
    }
    join_quality["filter_summary"] = filter_summary
    join_quality["no_silent_data_loss_guarantee"] = bool(filter_summary.get("no_silent_data_loss_guarantee"))
    join_quality["join_quality_warnings"] = _build_join_quality_warnings(quality=join_quality)
    dataset_quality_report = _build_dataset_quality_report(quality=join_quality)
    join_quality["dataset_quality_report"] = dataset_quality_report
    join_confidence_debug_rows: list[dict[str, Any]] = []
    for row in joined_rows:
        join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
        debug = join_meta.get("join_confidence_debug") if isinstance(join_meta.get("join_confidence_debug"), dict) else {}
        join_confidence_debug_rows.append(
            {
                "property_event_id": row.get("property_event_id"),
                "event_id": (row.get("event") or {}).get("event_id"),
                "feature_record_id": (row.get("feature") or {}).get("record_id"),
                "outcome_record_id": (row.get("outcome") or {}).get("record_id"),
                "join_method": join_meta.get("join_method"),
                "join_confidence_score": join_meta.get("join_confidence_score"),
                "join_confidence_tier": join_meta.get("join_confidence_tier"),
                "join_distance_m": join_meta.get("join_distance_m"),
                "max_allowed_tier": join_meta.get("join_confidence_max_allowed_tier"),
                "non_high_reason_codes": (
                    debug.get("non_high_reason_codes")
                    if isinstance(debug.get("non_high_reason_codes"), list)
                    else []
                ),
                "near_high_distance_threshold": bool(debug.get("near_high_distance_threshold")),
                "just_below_high_score_threshold": bool(debug.get("just_below_high_score_threshold")),
            }
        )
    join_quality["join_confidence_debug_examples"] = join_confidence_debug_rows[:20]

    jsonl_path = run_dir / "evaluation_dataset.jsonl"
    _write_jsonl(jsonl_path, joined_rows)
    csv_path = run_dir / "evaluation_dataset.csv"
    _write_csv(csv_path, joined_rows)
    join_metrics_path = run_dir / "join_quality_metrics.json"
    join_metrics_path.write_text(json.dumps(join_quality, indent=2, sort_keys=True), encoding="utf-8")
    # Backward-compatible alias kept for existing consumers.
    join_report_path = run_dir / "join_quality_report.json"
    join_report_path.write_text(json.dumps(join_quality, indent=2, sort_keys=True), encoding="utf-8")
    join_report_md_path = run_dir / "join_quality_report.md"
    join_report_md_path.write_text(
        _build_join_quality_report_markdown(
            run_id=run_token,
            generated_at=generated_at,
            quality=join_quality,
        ),
        encoding="utf-8",
    )
    join_confidence_debug_path = run_dir / "join_confidence_debug.jsonl"
    _write_jsonl(join_confidence_debug_path, join_confidence_debug_rows)
    summary_path = run_dir / "summary.md"
    summary_path.write_text(
        _build_markdown_summary(run_id=run_token, generated_at=generated_at, quality=join_quality),
        encoding="utf-8",
    )
    filter_summary_path = run_dir / "filter_summary.json"
    filter_summary_path.write_text(json.dumps(filter_summary, indent=2, sort_keys=True), encoding="utf-8")
    filter_summary_md_path = run_dir / "filter_summary.md"
    filter_summary_md_path.write_text(
        _build_filter_summary_markdown(
            run_id=run_token,
            generated_at=generated_at,
            filter_summary=filter_summary,
        ),
        encoding="utf-8",
    )
    dataset_quality_path = run_dir / "dataset_quality_report.json"
    dataset_quality_path.write_text(json.dumps(dataset_quality_report, indent=2, sort_keys=True), encoding="utf-8")
    dataset_quality_md_path = run_dir / "dataset_quality_report.md"
    dataset_quality_md_path.write_text(
        _build_dataset_quality_markdown(
            run_id=run_token,
            generated_at=generated_at,
            dataset_quality=dataset_quality_report,
        ),
        encoding="utf-8",
    )

    audit_sample_size = max(5, min(10, int(audit_sample_rows)))
    sample_joined_rows: list[dict[str, Any]] = []
    for row in joined_rows[:audit_sample_size]:
        feature_payload = row.get("feature") if isinstance(row.get("feature"), dict) else {}
        outcome_payload = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        join_payload = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
        sample_joined_rows.append(
            {
                "property_event_id": row.get("property_event_id"),
                "feature_lat": feature_payload.get("latitude"),
                "feature_lon": feature_payload.get("longitude"),
                "outcome_lat": outcome_payload.get("latitude"),
                "outcome_lon": outcome_payload.get("longitude"),
                "join_distance_m": join_payload.get("join_distance_m"),
                "join_confidence_score": join_payload.get("join_confidence_score"),
                "join_confidence_tier": join_payload.get("join_confidence_tier"),
                "join_method": join_payload.get("join_method"),
                "candidate_pool_count": join_payload.get("diagnostic_candidate_pool_count"),
                "outcome_label": outcome_payload.get("damage_label"),
                "non_high_reason_codes": join_payload.get("join_confidence_non_high_reason_codes"),
            }
        )
    sample_filtered_rows: list[dict[str, Any]] = []
    for row in excluded_rows[:audit_sample_size]:
        if isinstance(row, dict):
            sample_filtered_rows.append(
                {
                    "feature_artifact_path": row.get("feature_artifact_path"),
                    "record_id": row.get("record_id"),
                    "event_id": row.get("event_id"),
                    "reason": row.get("reason"),
                    "join_pass": row.get("join_pass"),
                }
            )
    pipeline_audit_report_path: Path | None = None
    if bool(audit_mode):
        pipeline_audit_report_path = run_dir / "pipeline_audit_report.md"
        pipeline_audit_report_path.write_text(
            _build_pipeline_audit_markdown(
                run_id=run_token,
                generated_at=generated_at,
                quality=join_quality,
                filter_summary=filter_summary,
                sample_joined_rows=sample_joined_rows,
                sample_filtered_rows=sample_filtered_rows,
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_token,
        "generated_at": generated_at,
        "inputs": {
            "normalized_outcomes_paths": [str(path) for path in configured_outcome_paths],
            "feature_artifacts": [str(Path(path).expanduser()) for path in feature_artifacts],
            "join_config": {
                "max_distance_m": float(join_config.max_distance_m),
                "exact_match_distance_m": float(join_config.exact_match_distance_m),
                "near_match_distance_m": float(join_config.near_match_distance_m),
                "buffer_match_radius_m": float(join_config.buffer_match_radius_m),
                "high_confidence_distance_m": float(join_config.high_confidence_distance_m),
                "medium_confidence_distance_m": float(join_config.medium_confidence_distance_m),
                "global_max_distance_m": float(join_config.global_max_distance_m),
                "event_year_tolerance_years": int(join_config.event_year_tolerance_years),
                "enable_global_nearest_fallback": bool(join_config.enable_global_nearest_fallback),
                "allow_duplicate_outcome_matches": bool(join_config.allow_duplicate_outcome_matches),
                "address_token_overlap_min": float(join_config.address_token_overlap_min),
            },
            "min_retained_records": int(retention_target),
            "auto_relax_for_min_retention": bool(auto_relax_for_min_retention),
            "auto_score_missing": bool(auto_score_missing),
            "auto_score_all": bool(auto_score_all),
            "audit_mode": bool(audit_mode),
            "audit_sample_rows": int(audit_sample_size),
        },
        "artifacts": {
            "evaluation_dataset_jsonl": str(jsonl_path),
            "evaluation_dataset_csv": str(csv_path),
            "join_quality_metrics_json": str(join_metrics_path),
            "join_quality_report_markdown": str(join_report_md_path),
            "join_quality_report_json": str(join_report_path),
            "join_confidence_debug_jsonl": str(join_confidence_debug_path),
            "filter_summary_json": str(filter_summary_path),
            "filter_summary_markdown": str(filter_summary_md_path),
            "dataset_quality_report_json": str(dataset_quality_path),
            "dataset_quality_report_markdown": str(dataset_quality_md_path),
            "summary_markdown": str(summary_path),
            "pipeline_audit_report_markdown": (
                str(pipeline_audit_report_path) if pipeline_audit_report_path is not None else None
            ),
        },
        "summary": {
            "outcomes_loaded": len(outcomes),
            "feature_rows_loaded": len(feature_rows),
            "joined_records": len(joined_rows),
            "rows_before_property_event_dedupe": int(
                property_event_dedupe_stats.get("total_rows_before_property_event_dedupe") or 0
            ),
            "unique_property_event_id_count": int(property_event_dedupe_stats.get("unique_property_event_id_count") or 0),
            "duplicate_property_event_rows_removed_count": int(
                property_event_dedupe_stats.get("duplicate_property_event_rows_removed_count") or 0
            ),
            "duplication_factor": property_event_dedupe_stats.get("duplication_factor"),
            "join_rate": join_rate,
            "excluded_rows": len(excluded_rows),
            "low_confidence_join_count": low_confidence_join_count,
            "duplicate_outcome_match_prevented_count": duplicate_outcome_match_prevented_count,
            "leakage_warning_count": len(set(leakage_warnings)),
            "score_backfilled_records": int(score_backfill.get("backfilled_record_count") or 0),
            "remaining_missing_scores": int(score_backfill.get("remaining_missing_score_record_count") or 0),
            "no_silent_data_loss_guarantee": bool(filter_summary.get("no_silent_data_loss_guarantee")),
            "retention_fallback_triggered": bool(retention_fallback_summary.get("triggered")),
            "retention_fallback_used": bool(retention_fallback_summary.get("used")),
            "fallback_heavy_fraction": dataset_quality_report.get("fallback_heavy_fraction"),
            "structure_features_non_zero_variance_count": (
                ((dataset_quality_report.get("structure_feature_variation") or {}).get("non_zero_variance_feature_count"))
            ),
            "near_structure_vegetation_features_non_zero_variance_count": (
                ((dataset_quality_report.get("near_structure_vegetation_feature_variation") or {}).get("non_zero_variance_feature_count"))
            ),
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
        "join_quality_metrics_path": str(join_metrics_path),
        "join_quality_markdown_path": str(join_report_md_path),
        "join_quality_report_path": str(join_report_path),
        "join_confidence_debug_path": str(join_confidence_debug_path),
        "filter_summary_path": str(filter_summary_path),
        "dataset_quality_report_path": str(dataset_quality_path),
        "dataset_quality_markdown_path": str(dataset_quality_md_path),
        "pipeline_audit_report_path": (
            str(pipeline_audit_report_path) if pipeline_audit_report_path is not None else None
        ),
        "joined_record_count": len(joined_rows),
        "rows_before_property_event_dedupe": int(
            property_event_dedupe_stats.get("total_rows_before_property_event_dedupe") or 0
        ),
        "unique_property_event_id_count": int(property_event_dedupe_stats.get("unique_property_event_id_count") or 0),
        "duplicate_property_event_rows_removed_count": int(
            property_event_dedupe_stats.get("duplicate_property_event_rows_removed_count") or 0
        ),
        "duplication_factor": property_event_dedupe_stats.get("duplication_factor"),
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
        action="append",
        default=[],
        help="Path to normalized public outcomes JSON (records array expected). May be supplied multiple times.",
    )
    parser.add_argument(
        "--outcomes-root",
        default="",
        help=(
            "Optional root with normalized outcome runs (benchmark/public_outcomes/normalized). "
            "If --outcomes is omitted, normalized outcomes are resolved from this root."
        ),
    )
    parser.add_argument(
        "--outcomes-root-mode",
        choices=("all", "latest"),
        default="all",
        help=(
            "When --outcomes-root is provided without explicit --outcomes-run-id, choose whether to "
            "use all normalized_outcomes.json runs or only the latest run (default: all)."
        ),
    )
    parser.add_argument(
        "--outcomes-run-id",
        action="append",
        default=[],
        help=(
            "Optional run id under --outcomes-root to include. May be supplied multiple times. "
            "Ignored when explicit --outcomes paths are supplied."
        ),
    )
    parser.add_argument(
        "--feature-artifact",
        action="append",
        default=[],
        help="Event-backtest feature artifact JSON path. May be supplied multiple times.",
    )
    parser.add_argument(
        "--feature-artifact-dir",
        action="append",
        default=[],
        help=(
            "Optional directory to discover feature artifact JSON files. "
            "Use with --feature-artifact-glob."
        ),
    )
    parser.add_argument(
        "--feature-artifact-glob",
        default="*.json",
        help="Glob used for --feature-artifact-dir discovery (default: *.json).",
    )
    parser.add_argument(
        "--feature-artifact-search-root",
        default="benchmark/public_outcomes/evaluation_dataset",
        help=(
            "Root directory used by --rapid-max-coverage for recursive feature artifact discovery."
        ),
    )
    parser.add_argument(
        "--feature-artifact-search-glob",
        default="**/_auto_scored_event_backtest/*.json",
        help=(
            "Recursive glob under --feature-artifact-search-root used by --rapid-max-coverage."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/public_outcomes/evaluation_dataset",
        help="Root output directory for timestamped run bundles.",
    )
    parser.add_argument("--run-id", default="", help="Optional fixed run id for deterministic output naming.")
    parser.add_argument(
        "--exact-match-distance-m",
        type=float,
        default=3.0,
        help="Distance threshold for exact coordinate matches (meters).",
    )
    parser.add_argument(
        "--near-match-distance-m",
        type=float,
        default=30.0,
        help="Distance threshold for near coordinate matches (meters).",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=120.0,
        help="Extended nearest-neighbor join distance in meters for event-level geospatial joins.",
    )
    parser.add_argument(
        "--buffer-match-radius-m",
        type=float,
        default=80.0,
        help="Buffered event-level matching radius in meters used before extended nearest-neighbor fallback.",
    )
    parser.add_argument(
        "--high-confidence-distance-m",
        type=float,
        default=DEFAULT_HIGH_CONFIDENCE_DISTANCE_M,
        help="Distance threshold (meters) for high join-confidence distance tier.",
    )
    parser.add_argument(
        "--medium-confidence-distance-m",
        type=float,
        default=100.0,
        help="Distance threshold (meters) for moderate join-confidence distance tier.",
    )
    parser.add_argument(
        "--global-max-distance-m",
        type=float,
        default=1000.0,
        help="Maximum nearest-neighbor distance in meters for global fallback joins.",
    )
    parser.add_argument(
        "--event-year-tolerance-years",
        type=int,
        default=1,
        help="Allowed event year mismatch tolerance for event-name fallback joins.",
    )
    parser.add_argument(
        "--enable-global-nearest-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow/disallow low-confidence global nearest-neighbor fallback joins.",
    )
    parser.add_argument(
        "--allow-duplicate-outcome-matches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow/disallow one-to-many outcome reuse across feature rows (default: disallow).",
    )
    parser.add_argument(
        "--address-token-overlap-min",
        type=float,
        default=0.75,
        help="Minimum token-overlap ratio for approximate event+address joins.",
    )
    parser.add_argument(
        "--min-retained-records",
        type=int,
        default=20,
        help=(
            "Minimum target for joined records. If the primary join returns fewer rows and "
            "--auto-relax-for-min-retention is enabled, a relaxed fallback pass is attempted."
        ),
    )
    parser.add_argument(
        "--auto-relax-for-min-retention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic relaxed fallback matching when joined rows are below --min-retained-records.",
    )
    parser.add_argument(
        "--auto-score-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When feature artifacts lack wildfire score fields, run event-backtest scoring to backfill "
            "scores before join (enabled by default)."
        ),
    )
    parser.add_argument(
        "--auto-score-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rescore all feature rows from artifacts before joining outcomes. This is useful after "
            "model-weight changes to regenerate comparable validation datasets."
        ),
    )
    parser.add_argument(
        "--include-normalized-outcomes-as-feature-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include normalized outcomes JSON files as additional feature artifact inputs "
            "(useful for rapid coverage expansion)."
        ),
    )
    parser.add_argument(
        "--rapid-max-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rapidly maximize joined dataset coverage by aggregating all outcomes runs, "
            "discovering all auto-scored feature artifacts, and allowing duplicate outcome matches."
        ),
    )
    parser.add_argument(
        "--audit-mode",
        "--audit_mode",
        dest="audit_mode",
        nargs="?",
        const="true",
        default="false",
        help=(
            "Enable verbose pipeline audit diagnostics and emit pipeline_audit_report.md. "
            "Accepts true/false (example: --audit_mode true)."
        ),
    )
    parser.add_argument(
        "--audit-sample-rows",
        "--audit_sample_rows",
        dest="audit_sample_rows",
        type=int,
        default=10,
        help="Number of joined/filtered sample rows to show in audit output (clamped to 5-10).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory.")
    args = parser.parse_args()

    explicit_feature_artifacts = [Path(token).expanduser() for token in (args.feature_artifact or []) if str(token).strip()]
    discovered_feature_artifacts = _resolve_feature_artifacts_from_dirs(
        [Path(token).expanduser() for token in (args.feature_artifact_dir or []) if str(token).strip()],
        artifact_glob=str(args.feature_artifact_glob or "*.json"),
    )
    feature_artifacts = sorted({path for path in (explicit_feature_artifacts + discovered_feature_artifacts)})
    explicit_outcomes = [Path(token).expanduser() for token in (args.outcomes or []) if str(token).strip()]
    outcomes_root = Path(args.outcomes_root).expanduser() if str(args.outcomes_root or "").strip() else None
    if bool(args.rapid_max_coverage) and outcomes_root is None:
        outcomes_root = Path("benchmark/public_outcomes/normalized").expanduser()
    resolved_from_root: list[Path] = []
    if not explicit_outcomes and outcomes_root is not None:
        run_ids = [str(token) for token in (args.outcomes_run_id or []) if str(token).strip()]
        if run_ids:
            resolved_from_root = _resolve_normalized_outcomes_from_run_ids(outcomes_root, run_ids)
        else:
            root_mode = "all" if bool(args.rapid_max_coverage) else str(args.outcomes_root_mode)
            if root_mode == "latest":
                resolved_from_root = _resolve_latest_normalized_outcomes(outcomes_root)
            else:
                resolved_from_root = _resolve_all_normalized_outcomes(outcomes_root)
    outcomes_paths = sorted({path for path in (explicit_outcomes + resolved_from_root)})
    if not outcomes_paths:
        raise ValueError("At least one outcomes source is required (--outcomes or --outcomes-root).")
    if bool(args.rapid_max_coverage):
        discovered_recursive = _resolve_feature_artifacts_from_root(
            Path(args.feature_artifact_search_root).expanduser(),
            artifact_glob=str(args.feature_artifact_search_glob or "**/_auto_scored_event_backtest/*.json"),
        )
        feature_artifacts = sorted({*feature_artifacts, *discovered_recursive})
        sample_backtest = Path("benchmark/event_backtest_sample_v1.json")
        if sample_backtest.exists():
            feature_artifacts = sorted({*feature_artifacts, sample_backtest.expanduser()})
    if bool(args.include_normalized_outcomes_as_feature_artifacts):
        feature_artifacts = sorted({*feature_artifacts, *outcomes_paths})
    if not feature_artifacts:
        raise ValueError(
            "At least one feature artifact is required (--feature-artifact, --feature-artifact-dir, or --rapid-max-coverage)."
        )
    allow_duplicate_outcome_matches = bool(args.allow_duplicate_outcome_matches or args.rapid_max_coverage)
    audit_mode = _parse_bool_flag(args.audit_mode, default=False)
    audit_sample_rows = max(5, min(10, int(args.audit_sample_rows)))
    result = build_public_outcome_evaluation_dataset(
        outcomes_paths=outcomes_paths,
        feature_artifacts=feature_artifacts,
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        exact_match_distance_m=float(args.exact_match_distance_m),
        near_match_distance_m=float(args.near_match_distance_m),
        max_distance_m=float(args.max_distance_m),
        buffer_match_radius_m=float(args.buffer_match_radius_m),
        high_confidence_distance_m=float(args.high_confidence_distance_m),
        medium_confidence_distance_m=float(args.medium_confidence_distance_m),
        global_max_distance_m=float(args.global_max_distance_m),
        event_year_tolerance_years=int(args.event_year_tolerance_years),
        enable_global_nearest_fallback=bool(args.enable_global_nearest_fallback),
        allow_duplicate_outcome_matches=allow_duplicate_outcome_matches,
        address_token_overlap_min=float(args.address_token_overlap_min),
        min_retained_records=int(args.min_retained_records),
        auto_relax_for_min_retention=bool(args.auto_relax_for_min_retention),
        auto_score_missing=bool(args.auto_score_missing),
        auto_score_all=bool(args.auto_score_all),
        audit_mode=bool(audit_mode),
        audit_sample_rows=int(audit_sample_rows),
        overwrite=bool(args.overwrite),
    )
    join_report = json.loads(Path(result["join_quality_metrics_path"]).read_text(encoding="utf-8"))
    join_confidence_debug_rows: list[dict[str, Any]] = []
    join_confidence_debug_path = Path(result.get("join_confidence_debug_path") or "")
    if str(join_confidence_debug_path).strip() and join_confidence_debug_path.is_file():
        with join_confidence_debug_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    join_confidence_debug_rows.append(payload)
    print(
        f"[public-eval-ds] Loaded outcomes={join_report.get('total_outcomes_loaded')} "
        f"feature_rows={join_report.get('total_feature_rows_loaded')}"
    )
    print(
        f"[public-eval-ds] Stage counts: "
        f"candidate_attempts={join_report.get('candidate_match_attempt_count')} "
        f"matched_before_filters={join_report.get('matched_candidate_count_before_filters')} "
        f"joined_before_property_event_dedupe={join_report.get('total_rows_before_property_event_dedupe')} "
        f"filtered={join_report.get('filtered_records_count')} "
        f"final_dataset_size={join_report.get('final_dataset_size')}"
    )
    print(
        "[public-eval-ds] Property-event dedupe: "
        f"unique_property_event_id={join_report.get('unique_property_event_id_count')} "
        f"duplicate_rows_removed={join_report.get('duplicate_property_event_rows_removed_count')} "
        f"duplication_factor={join_report.get('duplication_factor')}"
    )
    filter_summary = join_report.get("filter_summary") if isinstance(join_report.get("filter_summary"), dict) else {}
    if filter_summary:
        print(
            "[public-eval-ds] Filter summary: "
            f"no_silent_data_loss_guarantee={filter_summary.get('no_silent_data_loss_guarantee')} "
            f"reason_counts={filter_summary.get('filter_reason_counts')} "
            f"reason_percentages={filter_summary.get('filter_reason_percentages')} "
            f"soft_flags={filter_summary.get('soft_flag_counts')}"
        )
        accounting = filter_summary.get("accounting") if isinstance(filter_summary.get("accounting"), dict) else {}
        if accounting:
            print(f"[public-eval-ds] Filter accounting: {accounting}")
    retention_fallback = join_report.get("retention_fallback") if isinstance(join_report.get("retention_fallback"), dict) else {}
    if retention_fallback:
        print(
            "[public-eval-ds] Minimum-retention fallback: "
            f"enabled={retention_fallback.get('enabled')} "
            f"triggered={retention_fallback.get('triggered')} "
            f"used={retention_fallback.get('used')} "
            f"target={retention_fallback.get('target_min_records')} "
            f"primary_joined={retention_fallback.get('primary_joined_records')} "
            f"fallback_joined={retention_fallback.get('fallback_joined_records')} "
            f"active_pass={retention_fallback.get('active_pass')}"
        )
        if retention_fallback.get("triggered"):
            print(
                "[public-eval-ds] WARNING: minimum-retention fallback mode triggered; lower-confidence matches "
                "were included and are explicitly flagged in `evaluation.soft_filter_flags`."
            )
    print(
        f"[public-eval-ds] Matched rows={join_report.get('total_joined_records')} "
        f"excluded={join_report.get('excluded_row_count')} "
        f"join_rate={join_report.get('join_rate')}"
    )
    print(
        "[public-eval-ds] Coverage: "
        f"by_event={join_report.get('by_event_join_counts')} "
        f"outcomes_by_event={join_report.get('outcomes_by_event_counts')} "
        f"feature_rows_by_event={join_report.get('feature_rows_by_event_counts')} "
        f"unmatched_by_event={join_report.get('unmatched_feature_rows_by_event_counts')} "
        f"by_label={join_report.get('by_label_join_counts')} "
        f"join_tiers={join_report.get('join_confidence_tier_counts')} "
        f"row_confidence_tiers={join_report.get('row_confidence_tier_counts')} "
        f"match_tiers={join_report.get('match_tier_counts')}"
    )
    print(
        "[public-eval-ds] Feature variation: "
        f"{join_report.get('feature_variation_diagnostics')}"
    )
    print(
        "[public-eval-ds] Dataset quality report: "
        f"{join_report.get('dataset_quality_report')}"
    )
    print(
        "[public-eval-ds] Diversity spread: "
        f"{join_report.get('diversity_spread')}"
    )
    print(
        "[public-eval-ds] Fallback usage: "
        f"{join_report.get('fallback_usage_summary')}"
    )
    print(
        "[public-eval-ds] Distance diagnostics: "
        f"avg_m={join_report.get('average_join_distance_m')} "
        f"median_m={join_report.get('median_join_distance_m')} "
        f"tier_stats={join_report.get('join_confidence_tier_distance_stats')} "
        f"histogram={join_report.get('match_distance_histogram_m')} "
        f"outlier_threshold_m={join_report.get('distance_outlier_threshold_m')}"
    )
    print(
        "[public-eval-ds] Coordinate normalization: "
        f"{join_report.get('coordinate_normalization_summary')}"
    )
    print(
        "[public-eval-ds] Duplicate prevention: "
        f"allow_duplicates={join_report.get('allow_duplicate_outcome_matches')} "
        f"prevented={join_report.get('duplicate_outcome_match_prevented_count')}"
    )
    print(
        "[public-eval-ds] Join examples by tier: "
        f"{join_report.get('join_confidence_tier_examples')}"
    )
    join_warnings = join_report.get("join_quality_warnings") if isinstance(join_report.get("join_quality_warnings"), list) else []
    if join_warnings:
        print(f"[public-eval-ds] Join-quality warnings: {join_warnings}")
    if join_confidence_debug_rows:
        print(
            "[public-eval-ds] Join-confidence classification diagnostics: "
            f"rows={len(join_confidence_debug_rows)} path={join_confidence_debug_path}"
        )
        max_print = 500
        for idx, row in enumerate(join_confidence_debug_rows):
            if idx >= max_print:
                remaining = len(join_confidence_debug_rows) - max_print
                print(
                    "[public-eval-ds] Join-confidence diagnostics truncated in stdout "
                    f"(remaining_rows={remaining}); full list available at {join_confidence_debug_path}"
                )
                break
            print(
                "[public-eval-ds][join-confidence] "
                f"property_event_id={row.get('property_event_id')} "
                f"method={row.get('join_method')} "
                f"distance_m={row.get('join_distance_m')} "
                f"score={row.get('join_confidence_score')} "
                f"tier={row.get('join_confidence_tier')} "
                f"non_high_reasons={row.get('non_high_reason_codes')}"
            )
    if bool(audit_mode):
        dataset_rows: list[dict[str, Any]] = []
        with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    dataset_rows.append(payload)
        print(
            "[public-eval-ds][audit] Ingestion counts: "
            f"outcomes={join_report.get('total_outcomes_loaded')} "
            f"feature_rows={join_report.get('total_feature_rows_loaded')}"
        )
        print(
            "[public-eval-ds][audit] Join candidates: "
            f"candidate_attempts={join_report.get('candidate_match_attempt_count')} "
            f"matched_before_filters={join_report.get('matched_candidate_count_before_filters')} "
            f"join_methods={join_report.get('join_method_counts')}"
        )
        print(
            "[public-eval-ds][audit] Match distances: "
            f"avg_m={join_report.get('average_join_distance_m')} "
            f"median_m={join_report.get('median_join_distance_m')} "
            f"histogram={join_report.get('match_distance_histogram_m')}"
        )
        print(
            "[public-eval-ds][audit] Filtering steps: "
            f"filter_summary={(join_report.get('filter_summary') if isinstance(join_report.get('filter_summary'), dict) else {})}"
        )
        print(
            "[public-eval-ds][audit] Final dataset: "
            f"size={join_report.get('final_dataset_size')} excluded={join_report.get('excluded_row_count')}"
        )
        for idx, row in enumerate(dataset_rows[:audit_sample_rows]):
            feature_payload = row.get("feature") if isinstance(row.get("feature"), dict) else {}
            outcome_payload = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
            join_payload = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
            print(
                "[public-eval-ds][audit][sample] "
                f"index={idx} property_event_id={row.get('property_event_id')} "
                f"feature_lat={feature_payload.get('latitude')} feature_lon={feature_payload.get('longitude')} "
                f"outcome_lat={outcome_payload.get('latitude')} outcome_lon={outcome_payload.get('longitude')} "
                f"distance_m={join_payload.get('join_distance_m')} "
                f"confidence={join_payload.get('join_confidence_tier')} "
                f"score={join_payload.get('join_confidence_score')} "
                f"outcome_label={outcome_payload.get('damage_label')}"
            )
        audit_path = result.get("pipeline_audit_report_path")
        if isinstance(audit_path, str) and audit_path.strip():
            print(f"[public-eval-ds][audit] Audit report written: {audit_path}")
    score_backfill = join_report.get("score_backfill") if isinstance(join_report.get("score_backfill"), dict) else {}
    if score_backfill:
        print(
            "[public-eval-ds] Score backfill: "
            f"mode={score_backfill.get('rescore_mode')} "
            f"before_missing={score_backfill.get('missing_score_record_count_before')} "
            f"backfilled={score_backfill.get('backfilled_record_count')} "
            f"remaining_missing={score_backfill.get('remaining_missing_score_record_count')}"
        )
        warnings = score_backfill.get("warnings") if isinstance(score_backfill.get("warnings"), list) else []
        if warnings:
            print(f"[public-eval-ds] Score-backfill warnings: {warnings}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
