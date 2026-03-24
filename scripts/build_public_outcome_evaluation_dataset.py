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

SCHEMA_VERSION = "1.3.0"
WGS84_COORD_DECIMALS = 6
LEAKAGE_TOKENS = (
    "outcome",
    "damage",
    "destroyed",
    "structure_loss_or_major_damage",
    "adverse_outcome",
    "label",
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
    high_confidence_distance_m: float = 20.0
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
        return rows
    return [row for row in rows if _outcome_identity_key(row) not in excluded_outcome_keys]


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
    within.sort(key=lambda item: item[1])
    chosen, distance_m = within[0]
    return chosen, distance_m, len(within)


def _pick_nearest(feature: FeatureRecord, candidates: list[OutcomeRecord]) -> tuple[OutcomeRecord | None, float | None]:
    if not candidates:
        return None, None
    best: OutcomeRecord | None = None
    best_distance: float | None = None
    for row in candidates:
        d = _haversine_m(feature.latitude, feature.longitude, row.latitude, row.longitude)
        if d is None:
            continue
        if best is None or (best_distance is not None and d < best_distance) or best_distance is None:
            best = row
            best_distance = d
    return best, best_distance


def _assess_join_tier(score: float) -> str:
    if score >= 0.9:
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
) -> str:
    by_score = _assess_join_tier(float(score))
    by_distance = _distance_join_tier(
        distance_m=distance_m,
        high_confidence_distance_m=high_confidence_distance_m,
        medium_confidence_distance_m=medium_confidence_distance_m,
    )
    # When distance is available, enforce explicit distance-based tiers.
    resolved_rank = _tier_rank(by_distance) if distance_m is not None else _tier_rank(by_score)
    capped_rank = min(resolved_rank, _tier_rank(max_allowed_tier))
    return _tier_name(capped_rank)


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
                ),
                "join_distance_m": distance_m,
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
        for row in candidates:
            overlap = _token_overlap_ratio(feature.address_norm, row.address_norm)
            if overlap > best_overlap:
                best_overlap = overlap
                best = row
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
                "match_tier": distance_tier if distance_tier != "outside" else "extended",
            }

    # 10) Approximate global address overlap fallback.
    if feature.address_norm:
        best: OutcomeRecord | None = None
        best_overlap = 0.0
        for row in _filter_unused_outcomes(indexes["all"], excluded_outcome_keys=excluded_outcome_keys):
            overlap = _token_overlap_ratio(feature.address_norm, row.address_norm)
            if overlap > best_overlap:
                best_overlap = overlap
                best = row
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
                "match_tier": "fallback",
            }
    return None, {
        "join_method": "unmatched",
        "join_confidence_score": 0.0,
        "join_confidence_tier": "low",
        "join_distance_m": None,
        "match_tier": "none",
        "unmatched_reason": (
            "no_unused_outcome_match_within_constraints"
            if excluded_outcome_keys
            else "no_outcome_match_within_constraints"
        ),
    }


def _load_outcomes(path: Path) -> list[OutcomeRecord]:
    payload = _load_json(path)
    rows = _iter_outcome_rows(payload)
    return [_as_outcome_record(row) for row in rows]


def _outcome_dedupe_key(row: OutcomeRecord) -> str:
    return _outcome_identity_key(row)


def _load_outcomes_from_paths(paths: list[Path]) -> tuple[list[OutcomeRecord], dict[str, int]]:
    deduped: dict[str, OutcomeRecord] = {}
    per_source_loaded: dict[str, int] = {}
    for path in paths:
        p = Path(path).expanduser()
        rows = _load_outcomes(p)
        per_source_loaded[str(p)] = len(rows)
        for row in rows:
            deduped[_outcome_dedupe_key(row)] = row
    merged = sorted(
        deduped.values(),
        key=lambda row: (row.event_id, row.event_year or 0, row.record_id, row.source_record_id),
    )
    return merged, per_source_loaded


def _load_feature_records(paths: list[Path]) -> tuple[list[FeatureRecord], list[dict[str, Any]]]:
    records: list[FeatureRecord] = []
    missing_artifacts: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            missing_artifacts.append({"feature_artifact_path": str(p), "reason": "missing_feature_artifact"})
            continue
        payload = _load_json(p)
        rows = _iter_feature_rows(payload)
        for row in rows:
            records.append(_as_feature_record(row, str(p)))
    return records, missing_artifacts


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
    if not missing_rows:
        return diagnostics
    if not auto_score_missing:
        diagnostics["warnings"].append(
            "Feature artifacts are missing wildfire scores; auto-score backfill is disabled."
        )
        return diagnostics

    artifact_paths = sorted(per_artifact_missing.keys())
    try:
        from backend.event_backtesting import run_event_backtest  # lazy import

        rescoring_dir = run_dir / "_auto_scored_event_backtest"
        rescoring_dir.mkdir(parents=True, exist_ok=True)
        diagnostics["rescoring_attempted"] = True
        diagnostics["rescoring_artifact_count"] = len(artifact_paths)
        artifact = run_event_backtest(
            dataset_paths=artifact_paths,
            output_dir=rescoring_dir,
        )
        diagnostics["rescoring_artifact_path"] = str(artifact.get("artifact_path") or "")
        scored_rows = artifact.get("records") if isinstance(artifact.get("records"), list) else []
        diagnostics["rescoring_record_count"] = len(scored_rows)
        by_event_record, by_record = _build_scored_record_maps(scored_rows)
        patched = 0
        for row in missing_rows:
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
                if target_scores.get(key) is None and scored_scores.get(key) is not None:
                    target_scores[key] = scored_scores.get(key)
            confidence = scored.get("confidence") if isinstance(scored.get("confidence"), dict) else {}
            if row.payload.get("confidence_tier") is None and confidence.get("confidence_tier") is not None:
                row.payload["confidence_tier"] = confidence.get("confidence_tier")
            if row.payload.get("confidence_score") is None and confidence.get("confidence_score") is not None:
                row.payload["confidence_score"] = confidence.get("confidence_score")
            if row.payload.get("use_restriction") is None and confidence.get("use_restriction") is not None:
                row.payload["use_restriction"] = confidence.get("use_restriction")
            if row.payload.get("evidence_quality_summary") is None and isinstance(scored.get("evidence_quality_summary"), dict):
                row.payload["evidence_quality_summary"] = scored.get("evidence_quality_summary")
            if row.payload.get("coverage_summary") is None and isinstance(scored.get("coverage_summary"), dict):
                row.payload["coverage_summary"] = scored.get("coverage_summary")
            if row.payload.get("factor_contribution_breakdown") is None and isinstance(scored.get("factor_contribution_breakdown"), dict):
                row.payload["factor_contribution_breakdown"] = scored.get("factor_contribution_breakdown")
            if row.payload.get("raw_feature_vector") is None and isinstance(scored.get("raw_feature_vector"), dict):
                row.payload["raw_feature_vector"] = scored.get("raw_feature_vector")
            if row.payload.get("transformed_feature_vector") is None and isinstance(scored.get("transformed_feature_vector"), dict):
                row.payload["transformed_feature_vector"] = scored.get("transformed_feature_vector")
            if row.payload.get("compression_flags") is None and isinstance(scored.get("compression_flags"), list):
                row.payload["compression_flags"] = scored.get("compression_flags")
            if row.payload.get("model_governance") is None and isinstance(scored.get("model_governance"), dict):
                row.payload["model_governance"] = scored.get("model_governance")
            row.payload["score_backfill_source"] = "event_backtest_auto_rescore"
            if _extract_feature_scores(row.payload).get("wildfire_risk_score") is not None:
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
        f"- Outcome source files: `{quality.get('outcomes_loaded_by_source_path')}`",
        f"- Feature rows loaded: `{quality.get('total_feature_rows_loaded')}`",
        f"- Joined rows: `{quality.get('total_joined_records')}`",
        f"- Join rate: `{quality.get('join_rate')}`",
        f"- Match rate (%): `{quality.get('match_rate_percent')}`",
        "",
        "## Join Quality",
        f"- Join method counts: `{quality.get('join_method_counts')}`",
        f"- Join confidence tier counts: `{quality.get('join_confidence_tier_counts')}`",
        f"- Match tier counts: `{quality.get('match_tier_counts')}`",
        f"- Row confidence tier counts: `{quality.get('row_confidence_tier_counts')}`",
        f"- Join confidence score stats: `{quality.get('join_confidence_score_stats')}`",
        f"- Join confidence distance stats by tier: `{quality.get('join_confidence_tier_distance_stats')}`",
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
        "",
        "## Coverage",
        f"- By event joined counts: `{quality.get('by_event_join_counts')}`",
        f"- By label joined counts: `{quality.get('by_label_join_counts')}`",
        f"- Excluded rows: `{quality.get('excluded_row_count')}`",
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
    high_confidence_distance_m: float = 20.0,
    medium_confidence_distance_m: float = 100.0,
    event_year_tolerance_years: int = 1,
    enable_global_nearest_fallback: bool = True,
    allow_duplicate_outcome_matches: bool = False,
    address_token_overlap_min: float = 0.75,
    auto_score_missing: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_token = str(run_id or _timestamp_id())
    generated_at = _deterministic_generated_at(run_id)
    run_dir = Path(output_root).expanduser() / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    configured_outcome_paths = [Path(path).expanduser() for path in (outcomes_paths or []) if str(path).strip()]
    if not configured_outcome_paths and outcomes_path is not None:
        configured_outcome_paths = [Path(outcomes_path).expanduser()]
    if not configured_outcome_paths:
        raise ValueError("At least one normalized outcomes path is required.")

    outcomes, outcomes_by_source = _load_outcomes_from_paths(configured_outcome_paths)
    if not outcomes:
        raise ValueError("No normalized outcome rows available.")
    feature_rows, missing_feature_artifacts = _load_feature_records(feature_artifacts)
    if not feature_rows:
        raise ValueError("No feature rows were loaded from provided feature artifacts.")
    score_backfill = _backfill_missing_feature_scores(
        feature_rows=feature_rows,
        run_dir=run_dir,
        auto_score_missing=bool(auto_score_missing),
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
    joined_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = list(missing_feature_artifacts)
    leakage_warnings: list[str] = []
    join_method_counts: dict[str, int] = {}
    join_tier_counts: dict[str, int] = {}
    match_tier_counts: dict[str, int] = {}
    join_tier_distance_values: dict[str, list[float]] = {}
    join_tier_examples: dict[str, list[dict[str, Any]]] = {}
    feature_coordinate_modes: dict[str, int] = {}
    outcome_coordinate_modes: dict[str, int] = {}
    duplicate_outcome_match_prevented_count = 0
    used_outcome_keys: set[str] = set()
    by_event_counts: dict[str, int] = {}
    by_label_counts: dict[str, int] = {}
    low_confidence_join_count = 0

    for feature in sorted(feature_rows, key=lambda row: (row.artifact_path, row.event_id, row.record_id)):
        feature_coord_mode = str(feature.payload.get("_coordinate_normalization_mode") or "unknown")
        feature_coordinate_modes[feature_coord_mode] = feature_coordinate_modes.get(feature_coord_mode, 0) + 1
        matched, join_meta = _choose_outcome(
            feature,
            indexes,
            join_config=join_config,
            excluded_outcome_keys=(used_outcome_keys if not join_config.allow_duplicate_outcome_matches else None),
        )
        method = str(join_meta.get("join_method") or "unmatched")
        if matched is None:
            excluded_rows.append(
                {
                    "feature_artifact_path": feature.artifact_path,
                    "record_id": feature.record_id,
                    "event_id": feature.event_id,
                    "reason": str(join_meta.get("unmatched_reason") or "no_outcome_match_within_constraints"),
                }
            )
            continue
        matched_key = _outcome_identity_key(matched)
        if not join_config.allow_duplicate_outcome_matches:
            if matched_key in used_outcome_keys:
                duplicate_outcome_match_prevented_count += 1
                excluded_rows.append(
                    {
                        "feature_artifact_path": feature.artifact_path,
                        "record_id": feature.record_id,
                        "event_id": feature.event_id,
                        "reason": "duplicate_outcome_match_prevented",
                        "outcome_identity_key": matched_key,
                    }
                )
                continue
            used_outcome_keys.add(matched_key)
        outcome_coord_mode = str(matched.payload.get("_coordinate_normalization_mode") or "unknown")
        outcome_coordinate_modes[outcome_coord_mode] = outcome_coordinate_modes.get(outcome_coord_mode, 0) + 1

        join_method_counts[method] = join_method_counts.get(method, 0) + 1
        tier = str(join_meta.get("join_confidence_tier") or "low")
        join_tier_counts[tier] = join_tier_counts.get(tier, 0) + 1
        match_tier = str(join_meta.get("match_tier") or "unknown")
        match_tier_counts[match_tier] = match_tier_counts.get(match_tier, 0) + 1
        join_distance_value = _safe_float(join_meta.get("join_distance_m"))
        if join_distance_value is not None:
            join_tier_distance_values.setdefault(tier, []).append(float(join_distance_value))
        examples = join_tier_examples.setdefault(tier, [])
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
            low_confidence_join_count += 1

        severity, binary = _derive_severity_and_binary(matched.payload)
        leak_flags = _detect_leakage_flags(feature.payload, feature.payload.get("event_date"))
        if leak_flags:
            leakage_warnings.append(
                f"{feature.event_id or 'unknown_event'}/{feature.record_id or 'unknown_record'}: {','.join(leak_flags)}"
            )

        caveat_flags: list[str] = []
        if not _event_year_consistent_with_tolerance(feature, matched, join_config.event_year_tolerance_years):
            caveat_flags.append("event_year_mismatch")
        distance_m = _safe_float(join_meta.get("join_distance_m"))
        if distance_m is not None and distance_m > float(join_config.max_distance_m) * 0.6:
            caveat_flags.append("high_join_distance")
        if tier == "low":
            caveat_flags.append("low_confidence_join")
        if leak_flags:
            caveat_flags.append("leakage_warning_present")

        extracted_scores = _extract_feature_scores(feature.payload)
        wildfire_risk = extracted_scores.get("wildfire_risk_score")
        site_hazard = extracted_scores.get("site_hazard_score")
        home_vuln = extracted_scores.get("home_ignition_vulnerability_score")
        readiness = extracted_scores.get("insurance_readiness_score")

        confidence_payload = feature.payload.get("confidence") if isinstance(feature.payload.get("confidence"), dict) else {}
        evidence_summary = feature.payload.get("evidence_quality_summary") if isinstance(feature.payload.get("evidence_quality_summary"), dict) else {}
        coverage_summary = feature.payload.get("coverage_summary") if isinstance(feature.payload.get("coverage_summary"), dict) else {}
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
                "raw_feature_vector": (
                    feature.payload.get("raw_feature_vector")
                    if isinstance(feature.payload.get("raw_feature_vector"), dict)
                    else {}
                ),
                "transformed_feature_vector": (
                    feature.payload.get("transformed_feature_vector")
                    if isinstance(feature.payload.get("transformed_feature_vector"), dict)
                    else {}
                ),
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
                "max_distance_m": float(join_config.max_distance_m),
                "high_confidence_distance_m": float(join_config.high_confidence_distance_m),
                "medium_confidence_distance_m": float(join_config.medium_confidence_distance_m),
                "global_max_distance_m": float(join_config.global_max_distance_m),
                "event_year_tolerance_years": int(join_config.event_year_tolerance_years),
                "global_fallback_enabled": bool(join_config.enable_global_nearest_fallback),
                "event_year_consistent": _event_year_consistent_with_tolerance(
                    feature,
                    matched,
                    join_config.event_year_tolerance_years,
                ),
            },
            "evaluation": {
                "row_usable": True,
                "row_confidence_tier": row_confidence_tier,
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
        joined_rows.append(joined)

        event_key = str(joined["event"]["event_id"] or "unknown_event")
        by_event_counts[event_key] = by_event_counts.get(event_key, 0) + 1
        label_key = str(joined["outcome"]["damage_label"] or "unknown")
        by_label_counts[label_key] = by_label_counts.get(label_key, 0) + 1

    joined_rows.sort(key=lambda row: str(row.get("property_event_id") or ""))
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
    row_confidence_tiers: dict[str, int] = {}
    for row in joined_rows:
        tier = str(((row.get("evaluation") or {}).get("row_confidence_tier")) or "unknown")
        row_confidence_tiers[tier] = row_confidence_tiers.get(tier, 0) + 1

    join_quality = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "total_outcomes_loaded": len(outcomes),
        "outcomes_loaded_by_source_path": dict(sorted(outcomes_by_source.items())),
        "total_feature_rows_loaded": len(feature_rows),
        "total_joined_records": len(joined_rows),
        "join_rate": join_rate,
        "match_rate_percent": round(join_rate * 100.0, 2),
        "average_join_distance_m": mean_join_distance,
        "median_join_distance_m": (round(float(median_join_distance), 3) if median_join_distance is not None else None),
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
        "by_event_join_counts": dict(sorted(by_event_counts.items())),
        "by_label_join_counts": dict(sorted(by_label_counts.items())),
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows[:200],
        "leakage_warnings": sorted(set(leakage_warnings)),
        "score_backfill": score_backfill,
    }

    jsonl_path = run_dir / "evaluation_dataset.jsonl"
    _write_jsonl(jsonl_path, joined_rows)
    csv_path = run_dir / "evaluation_dataset.csv"
    _write_csv(csv_path, joined_rows)
    join_report_path = run_dir / "join_quality_report.json"
    join_report_path.write_text(json.dumps(join_quality, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = run_dir / "summary.md"
    summary_path.write_text(
        _build_markdown_summary(run_id=run_token, generated_at=generated_at, quality=join_quality),
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
            "auto_score_missing": bool(auto_score_missing),
        },
        "artifacts": {
            "evaluation_dataset_jsonl": str(jsonl_path),
            "evaluation_dataset_csv": str(csv_path),
            "join_quality_report_json": str(join_report_path),
            "summary_markdown": str(summary_path),
        },
        "summary": {
            "outcomes_loaded": len(outcomes),
            "feature_rows_loaded": len(feature_rows),
            "joined_records": len(joined_rows),
            "join_rate": join_rate,
            "excluded_rows": len(excluded_rows),
            "low_confidence_join_count": low_confidence_join_count,
            "duplicate_outcome_match_prevented_count": duplicate_outcome_match_prevented_count,
            "leakage_warning_count": len(set(leakage_warnings)),
            "score_backfilled_records": int(score_backfill.get("backfilled_record_count") or 0),
            "remaining_missing_scores": int(score_backfill.get("remaining_missing_score_record_count") or 0),
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
        "join_quality_report_path": str(join_report_path),
        "joined_record_count": len(joined_rows),
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
        default=20.0,
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
        "--auto-score-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When feature artifacts lack wildfire score fields, run event-backtest scoring to backfill "
            "scores before join (enabled by default)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory.")
    args = parser.parse_args()

    explicit_feature_artifacts = [Path(token).expanduser() for token in (args.feature_artifact or []) if str(token).strip()]
    discovered_feature_artifacts = _resolve_feature_artifacts_from_dirs(
        [Path(token).expanduser() for token in (args.feature_artifact_dir or []) if str(token).strip()],
        artifact_glob=str(args.feature_artifact_glob or "*.json"),
    )
    feature_artifacts = sorted({path for path in (explicit_feature_artifacts + discovered_feature_artifacts)})
    if not feature_artifacts:
        raise ValueError("At least one feature artifact is required (--feature-artifact or --feature-artifact-dir).")
    explicit_outcomes = [Path(token).expanduser() for token in (args.outcomes or []) if str(token).strip()]
    outcomes_root = Path(args.outcomes_root).expanduser() if str(args.outcomes_root or "").strip() else None
    resolved_from_root: list[Path] = []
    if not explicit_outcomes and outcomes_root is not None:
        run_ids = [str(token) for token in (args.outcomes_run_id or []) if str(token).strip()]
        if run_ids:
            resolved_from_root = _resolve_normalized_outcomes_from_run_ids(outcomes_root, run_ids)
        else:
            if str(args.outcomes_root_mode) == "latest":
                resolved_from_root = _resolve_latest_normalized_outcomes(outcomes_root)
            else:
                resolved_from_root = _resolve_all_normalized_outcomes(outcomes_root)
    outcomes_paths = sorted({path for path in (explicit_outcomes + resolved_from_root)})
    if not outcomes_paths:
        raise ValueError("At least one outcomes source is required (--outcomes or --outcomes-root).")
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
        allow_duplicate_outcome_matches=bool(args.allow_duplicate_outcome_matches),
        address_token_overlap_min=float(args.address_token_overlap_min),
        auto_score_missing=bool(args.auto_score_missing),
        overwrite=bool(args.overwrite),
    )
    join_report = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    print(
        f"[public-eval-ds] Loaded outcomes={join_report.get('total_outcomes_loaded')} "
        f"feature_rows={join_report.get('total_feature_rows_loaded')}"
    )
    print(
        f"[public-eval-ds] Matched rows={join_report.get('total_joined_records')} "
        f"excluded={join_report.get('excluded_row_count')} "
        f"join_rate={join_report.get('join_rate')}"
    )
    print(
        "[public-eval-ds] Coverage: "
        f"by_event={join_report.get('by_event_join_counts')} "
        f"by_label={join_report.get('by_label_join_counts')} "
        f"join_tiers={join_report.get('join_confidence_tier_counts')} "
        f"row_confidence_tiers={join_report.get('row_confidence_tier_counts')} "
        f"match_tiers={join_report.get('match_tier_counts')}"
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
    score_backfill = join_report.get("score_backfill") if isinstance(join_report.get("score_backfill"), dict) else {}
    if score_backfill:
        print(
            "[public-eval-ds] Score backfill: "
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
