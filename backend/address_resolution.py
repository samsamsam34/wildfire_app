from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from backend.region_registry import list_prepared_regions, resolve_region_file


LOGGER = logging.getLogger("wildfire_app.address_resolution")
DEFAULT_LOCAL_ALIAS_PATH = Path("config") / "local_address_fallbacks.json"

_NORMALIZATION_REPLACEMENTS = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bcourt\b": "ct",
    r"\bplace\b": "pl",
    r"\bhighway\b": "hwy",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bwashington\b": "wa",
}

_DIRECT_ADDRESS_KEYS = (
    "address",
    "full_address",
    "formatted_address",
    "site_address",
    "situs_address",
    "prop_addr",
    "prop_address",
    "addr",
    "addr_full",
)
_HOUSE_KEYS = ("house_number", "housenumber", "number", "st_num", "addr_num")
_STREET_KEYS = ("street", "street_name", "road", "rd_name", "street_full", "prop_road", "road_name")
_CITY_KEYS = ("city", "town", "locality", "municipality", "prop_city")
_STATE_KEYS = ("state", "st", "province", "prop_st")
_ZIP_KEYS = ("zip", "zipcode", "postal", "postcode", "prop_zip")
_LAT_KEYS = ("latitude", "lat", "y", "lat_dd", "lat_wgs84")
_LON_KEYS = ("longitude", "lon", "lng", "x", "lon_dd", "lon_wgs84")

_AUTO_USABLE_TIERS = {"high", "medium"}
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0, "unknown": -1}


def normalize_address_for_matching(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    for pattern, replacement in _NORMALIZATION_REPLACEMENTS.items():
        normalized = re.sub(pattern, replacement, normalized)
    return " ".join(normalized.split())


def _zip5(value: str) -> str:
    match = re.search(r"\b(\d{5})(?:-\d{4})?\b", str(value or ""))
    return match.group(1) if match else ""


def _extract_address_components(address: str) -> dict[str, str]:
    submitted = str(address or "").strip()
    normalized = normalize_address_for_matching(submitted)
    parts = [p.strip() for p in submitted.split(",") if p.strip()]
    first_part = parts[0] if parts else submitted
    number_match = re.match(r"^\s*(\d+[a-zA-Z0-9-]*)\s+(.*)$", first_part)
    house_number = number_match.group(1).strip().lower() if number_match else ""
    street = number_match.group(2).strip() if number_match else first_part
    postal = _zip5(submitted)
    city = parts[1].strip().lower() if len(parts) >= 2 else ""
    tail = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) == 2 else "")
    tail_norm = normalize_address_for_matching(tail)
    state = ""
    for token in tail_norm.split():
        if token in {"wa", "or", "mt", "id", "ca", "nv", "co", "ut", "az", "nm"}:
            state = token
            break
    return {
        "submitted_address": submitted,
        "normalized_address": normalized,
        "house_number": house_number,
        "street": normalize_address_for_matching(street),
        "city": normalize_address_for_matching(city),
        "state": state,
        "postal": postal,
    }


def _flatten_coords(values: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if isinstance(values, (list, tuple)):
        if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values[:2]):
            out.append((float(values[0]), float(values[1])))
        else:
            for item in values:
                out.extend(_flatten_coords(item))
    return out


def _feature_lon_lat(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry")
    if isinstance(geometry, dict):
        coords = geometry.get("coordinates")
        if geometry.get("type") == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                return None
        all_coords = _flatten_coords(coords)
        if all_coords:
            lon_avg = sum(c[0] for c in all_coords) / float(len(all_coords))
            lat_avg = sum(c[1] for c in all_coords) / float(len(all_coords))
            return float(lon_avg), float(lat_avg)
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    for lat_key in _LAT_KEYS:
        for lon_key in _LON_KEYS:
            if lat_key not in props or lon_key not in props:
                continue
            try:
                lat = float(props[lat_key])
                lon = float(props[lon_key])
            except (TypeError, ValueError):
                continue
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return float(lon), float(lat)
    return None


def _first_non_empty(props: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in props and str(props[key]).strip():
            return str(props[key]).strip()
    return ""


def _candidate_address(feature: dict[str, Any]) -> str:
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    direct = _first_non_empty(props, _DIRECT_ADDRESS_KEYS)
    if direct:
        return direct
    house = _first_non_empty(props, _HOUSE_KEYS)
    street = _first_non_empty(props, _STREET_KEYS)
    city = _first_non_empty(props, _CITY_KEYS)
    state = _first_non_empty(props, _STATE_KEYS)
    postal = _first_non_empty(props, _ZIP_KEYS)
    chunks = [v for v in [f"{house} {street}".strip(), city, state, postal] if v]
    return ", ".join(chunks)


def _candidate_component(props: dict[str, Any], keys: tuple[str, ...]) -> str:
    return normalize_address_for_matching(_first_non_empty(props, keys))


def _street_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _match_score(
    *,
    input_components: dict[str, str],
    candidate_address: str,
    feature_props: dict[str, Any],
) -> float:
    input_norm = input_components["normalized_address"]
    candidate_norm = normalize_address_for_matching(candidate_address)
    if not candidate_norm:
        return 0.0

    score = difflib.SequenceMatcher(None, input_norm, candidate_norm).ratio()
    street_score = _street_similarity(
        input_components["street"],
        _candidate_component(feature_props, _STREET_KEYS) or candidate_norm,
    )
    score = (score * 0.7) + (street_score * 0.3)

    input_house = input_components["house_number"]
    candidate_house = _candidate_component(feature_props, _HOUSE_KEYS)
    if input_house:
        if candidate_house and candidate_house == input_house:
            score += 0.18
        elif candidate_house and candidate_house != input_house:
            score -= 0.25

    input_city = input_components["city"]
    candidate_city = _candidate_component(feature_props, _CITY_KEYS)
    if input_city and candidate_city:
        if input_city == candidate_city:
            score += 0.08
        else:
            score -= 0.08

    input_state = input_components["state"]
    candidate_state = _candidate_component(feature_props, _STATE_KEYS)
    if input_state and candidate_state and input_state != candidate_state:
        score -= 0.12
    if input_state and candidate_state and input_state == candidate_state:
        score += 0.03

    input_postal = input_components["postal"]
    candidate_postal = _first_non_empty(feature_props, _ZIP_KEYS)
    if input_postal and candidate_postal:
        if str(candidate_postal).startswith(input_postal):
            score += 0.05
        else:
            score -= 0.06

    return max(0.0, min(1.0, score))


def _load_geojson_features(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("features") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _load_location_resolution_source_config() -> list[dict[str, Any]]:
    config_path_raw = str(os.getenv("WF_LOCATION_RESOLUTION_SOURCE_CONFIG") or "").strip()
    config_path = Path(config_path_raw) if config_path_raw else Path("config") / "location_resolution_sources.json"
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("sources") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return []
    sources: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        path_value = str(row.get("path") or "").strip()
        if not path_value:
            continue
        path_obj = Path(path_value).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        sources.append(
            {
                "name": str(row.get("name") or path_obj.stem),
                "path": str(path_obj),
                "source_type": str(row.get("source_type") or "local_authoritative_dataset"),
                "priority": int(row.get("priority") or 100),
                "enabled": bool(row.get("enabled", True)),
                "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            }
        )
    sources.sort(key=lambda r: (int(r.get("priority") or 100), str(r.get("name") or "")))
    return [row for row in sources if row.get("enabled")]


def _env_source_paths() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = [
        ("WF_WA_STATEWIDE_PARCEL_PATH", "wa_state_current_parcels", "statewide_parcel_dataset"),
        ("WF_OKANOGAN_ADDRESS_POINTS_PATH", "okanogan_county_addressing", "county_address_dataset"),
    ]
    for env_key, name, source_type in mapping:
        raw = str(os.getenv(env_key) or "").strip()
        if not raw:
            continue
        path_obj = Path(raw).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        rows.append(
            {
                "name": name,
                "path": str(path_obj),
                "source_type": source_type,
                "priority": 50,
                "enabled": True,
                "metadata": {"env_var": env_key},
            }
        )
    return rows


def _load_alias_candidates(alias_path: Path) -> list[dict[str, Any]]:
    if not alias_path.exists():
        return []
    try:
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("addresses") if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    if not isinstance(records, list):
        return rows
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        try:
            lat = float(row.get("latitude"))
            lon = float(row.get("longitude"))
        except (TypeError, ValueError):
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        address = str(row.get("address") or "").strip()
        if not address:
            continue
        rows.append(
            {
                "type": "Feature",
                "properties": {
                    "address": address,
                    "city": row.get("city"),
                    "state": row.get("state"),
                    "postal": row.get("postal"),
                    "region_id": row.get("region_id"),
                    "source_name": row.get("source_name") or "local_alias",
                    "source_type": "local_alias",
                    "row_index": idx,
                },
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    return rows


def _distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    lat_mid = math.radians((a_lat + b_lat) / 2.0)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1.0, 111_320.0 * math.cos(lat_mid))
    return float(math.hypot((a_lat - b_lat) * meters_per_deg_lat, (a_lon - b_lon) * meters_per_deg_lon))


def validate_local_fallback_records(alias_path: str | Path | None = None) -> dict[str, Any]:
    chosen = Path(alias_path).expanduser() if alias_path else DEFAULT_LOCAL_ALIAS_PATH
    if not chosen.is_absolute():
        chosen = Path.cwd() / chosen

    records = _load_alias_candidates(chosen)
    warnings: list[str] = []
    errors: list[str] = []

    if not records:
        return {
            "alias_path": str(chosen),
            "record_count": 0,
            "warnings": ["No fallback alias records found."],
            "errors": [],
            "valid": True,
        }

    normalized_index: dict[str, list[tuple[float, float, int]]] = {}
    for feature in records:
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        address = str(props.get("address") or "").strip()
        components = _extract_address_components(address)
        row_index = int(props.get("row_index") or 0)
        coords = _feature_lon_lat(feature)
        if coords is None:
            errors.append(f"Row {row_index} is missing valid coordinates.")
            continue
        if not components["house_number"] or not components["street"]:
            warnings.append(
                f"Row {row_index} fallback '{address}' is street/locality-only and will not be auto-used for property coordinates."
            )

        normalized = components["normalized_address"]
        normalized_index.setdefault(normalized, []).append((coords[1], coords[0], row_index))

    for normalized, rows in normalized_index.items():
        if len(rows) <= 1:
            continue
        unique_coords: list[tuple[float, float]] = []
        for lat, lon, _ in rows:
            if (lat, lon) not in unique_coords:
                unique_coords.append((lat, lon))
        if len(unique_coords) <= 1:
            warnings.append(f"Duplicate fallback address '{normalized}' appears {len(rows)} times with same coordinates.")
            continue
        max_sep = 0.0
        for idx, (lat1, lon1) in enumerate(unique_coords):
            for lat2, lon2 in unique_coords[idx + 1 :]:
                max_sep = max(max_sep, _distance_m(lat1, lon1, lat2, lon2))
        if max_sep >= 30.0:
            errors.append(
                f"Fallback address '{normalized}' has conflicting coordinates separated by ~{max_sep:.1f} m."
            )
        else:
            warnings.append(
                f"Fallback address '{normalized}' has duplicate entries with minor coordinate deltas (~{max_sep:.1f} m)."
            )

    valid = not errors
    if warnings:
        LOGGER.warning("local_fallback_validation_warnings %s", json.dumps({"alias_path": str(chosen), "warnings": warnings[:10]}))
    if errors:
        LOGGER.error("local_fallback_validation_errors %s", json.dumps({"alias_path": str(chosen), "errors": errors[:10]}))

    return {
        "alias_path": str(chosen),
        "record_count": len(records),
        "warnings": warnings,
        "errors": errors,
        "valid": valid,
    }


def validate_address_point_source(path: str | Path, source_name: str) -> dict[str, Any]:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj

    features = _load_geojson_features(path_obj)
    warnings: list[str] = []
    errors: list[str] = []
    normalized_index: dict[str, list[tuple[float, float]]] = {}

    for idx, feature in enumerate(features):
        lon_lat = _feature_lon_lat(feature)
        if lon_lat is None:
            errors.append(f"{source_name}: feature[{idx}] missing valid coordinates.")
            continue
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        candidate_addr = _candidate_address(feature)
        components = _extract_address_components(candidate_addr)
        normalized = components["normalized_address"]
        if not components["house_number"] or not components["street"]:
            warnings.append(
                f"{source_name}: feature[{idx}] has incomplete address components (house/street missing)."
            )
            continue
        normalized_index.setdefault(normalized, []).append((float(lon_lat[1]), float(lon_lat[0])))

    for normalized, rows in normalized_index.items():
        if len(rows) <= 1:
            continue
        unique = sorted(set(rows))
        if len(unique) > 1:
            max_sep = 0.0
            for i, (lat1, lon1) in enumerate(unique):
                for lat2, lon2 in unique[i + 1 :]:
                    max_sep = max(max_sep, _distance_m(lat1, lon1, lat2, lon2))
            if max_sep >= 20.0:
                errors.append(
                    f"{source_name}: normalized address '{normalized}' has conflicting coordinates (~{max_sep:.1f} m apart)."
                )
            else:
                warnings.append(
                    f"{source_name}: normalized address '{normalized}' is duplicated with minor coordinate deltas (~{max_sep:.1f} m)."
                )
        else:
            warnings.append(f"{source_name}: normalized address '{normalized}' is duplicated.")

    return {
        "source_name": source_name,
        "path": str(path_obj),
        "record_count": len(features),
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _region_candidate_priority(address_components: dict[str, str], manifest: dict[str, Any]) -> int:
    normalized = address_components["normalized_address"]
    region_id = normalize_address_for_matching(str(manifest.get("region_id") or ""))
    display_name = normalize_address_for_matching(str(manifest.get("display_name") or ""))
    score = 0
    if region_id and any(token in normalized for token in region_id.split()):
        score += 1
    if display_name and any(token in normalized for token in display_name.split()):
        score += 1
    if address_components["state"] == "wa":
        score += 1
    return score


def _candidate_components(candidate_address: str, feature_props: dict[str, Any]) -> dict[str, str]:
    parsed = _extract_address_components(candidate_address)
    house = _candidate_component(feature_props, _HOUSE_KEYS) or parsed["house_number"]
    street = _candidate_component(feature_props, _STREET_KEYS) or parsed["street"]
    city = _candidate_component(feature_props, _CITY_KEYS) or parsed["city"]
    state = _candidate_component(feature_props, _STATE_KEYS) or parsed["state"]
    postal = _first_non_empty(feature_props, _ZIP_KEYS) or parsed["postal"]
    return {
        "normalized_address": parsed["normalized_address"],
        "house_number": normalize_address_for_matching(house),
        "street": normalize_address_for_matching(street),
        "city": normalize_address_for_matching(city),
        "state": normalize_address_for_matching(state),
        "postal": _zip5(postal),
    }


def _classify_candidate_match(
    *,
    input_components: dict[str, str],
    candidate_components: dict[str, str],
    candidate_score: float,
    min_score: float,
) -> dict[str, Any]:
    input_house = input_components["house_number"]
    candidate_house = candidate_components["house_number"]
    input_street = input_components["street"]
    candidate_street = candidate_components["street"]

    house_match = bool(input_house and candidate_house and input_house == candidate_house)
    house_mismatch = bool(input_house and candidate_house and input_house != candidate_house)
    street_ratio = _street_similarity(input_street, candidate_street)

    city_match = bool(input_components["city"] and candidate_components["city"] and input_components["city"] == candidate_components["city"])
    state_match = bool(input_components["state"] and candidate_components["state"] and input_components["state"] == candidate_components["state"])
    postal_match = bool(input_components["postal"] and candidate_components["postal"] and input_components["postal"] == candidate_components["postal"])
    regional_match = city_match or postal_match or state_match

    exact_match = bool(
        input_components["normalized_address"]
        and candidate_components["normalized_address"]
        and input_components["normalized_address"] == candidate_components["normalized_address"]
    )

    if exact_match:
        return {
            "match_type": "exact_normalized_address",
            "confidence_tier": "high",
            "auto_usable": True,
            "score_gate_passed": True,
            "street_similarity": round(street_ratio, 3),
        }

    if house_mismatch and street_ratio >= 0.86:
        return {
            "match_type": "house_number_mismatch",
            "confidence_tier": "low",
            "auto_usable": False,
            "score_gate_passed": False,
            "street_similarity": round(street_ratio, 3),
        }

    if house_match and street_ratio >= 0.86 and regional_match:
        score_gate = bool(candidate_score >= min_score)
        return {
            "match_type": "address_component_match",
            "confidence_tier": "medium" if score_gate else "low",
            "auto_usable": bool(score_gate),
            "score_gate_passed": score_gate,
            "street_similarity": round(street_ratio, 3),
        }

    if house_match and street_ratio >= 0.86:
        return {
            "match_type": "house_and_street_partial",
            "confidence_tier": "low",
            "auto_usable": False,
            "score_gate_passed": bool(candidate_score >= min_score),
            "street_similarity": round(street_ratio, 3),
        }

    if street_ratio >= 0.9:
        return {
            "match_type": "street_only_match",
            "confidence_tier": "low",
            "auto_usable": False,
            "score_gate_passed": False,
            "street_similarity": round(street_ratio, 3),
        }

    if regional_match:
        return {
            "match_type": "locality_or_postal_only",
            "confidence_tier": "low",
            "auto_usable": False,
            "score_gate_passed": False,
            "street_similarity": round(street_ratio, 3),
        }

    return {
        "match_type": "no_viable_match",
        "confidence_tier": "low",
        "auto_usable": False,
        "score_gate_passed": False,
        "street_similarity": round(street_ratio, 3),
    }


def _candidate_is_relevant(match_type: str, score: float) -> bool:
    match_type = str(match_type or "")
    if match_type in {"exact_normalized_address", "address_component_match", "house_number_mismatch"}:
        return True
    if match_type in {"street_only_match", "house_and_street_partial"}:
        return score >= 0.72
    if match_type == "locality_or_postal_only":
        return score >= 0.8
    return score >= 0.85


def _source_type_allowed(source_type: str, allowed_source_types: set[str] | None) -> bool:
    if not allowed_source_types:
        return True
    return str(source_type or "") in allowed_source_types


def _normalize_locality_preferences(values: list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for value in values:
        normalized = normalize_address_for_matching(str(value or ""))
        if normalized:
            out.add(normalized)
    return out


def resolve_local_address_candidate(
    *,
    address: str,
    regions_root: str,
    alias_path: str | None = None,
    include_authoritative_sources: bool = True,
    include_alias_sources: bool = True,
    min_auto_confidence_tier: str = "medium",
    allowed_source_types: set[str] | None = None,
    preferred_localities: list[str] | tuple[str, ...] | None = None,
    preferred_postal: str | None = None,
    required_state: str | None = None,
    top_candidate_limit: int = 3,
) -> dict[str, Any]:
    address_components = _extract_address_components(address)
    min_score_raw = str(os.getenv("WF_LOCAL_ADDRESS_MATCH_MIN_SCORE", "0.76")).strip()
    try:
        min_score = max(0.5, min(0.98, float(min_score_raw)))
    except ValueError:
        min_score = 0.76

    chosen_alias_path = Path(alias_path).expanduser() if alias_path else DEFAULT_LOCAL_ALIAS_PATH
    if not chosen_alias_path.is_absolute():
        chosen_alias_path = Path.cwd() / chosen_alias_path

    required_rank = _CONFIDENCE_RANK.get(str(min_auto_confidence_tier).strip().lower(), _CONFIDENCE_RANK["medium"])
    preferred_locality_norms = _normalize_locality_preferences(preferred_localities)
    preferred_postal_zip = _zip5(preferred_postal or "")
    required_state_norm = normalize_address_for_matching(required_state or "")
    top_candidate_limit = max(1, min(50, int(top_candidate_limit or 3)))
    candidates: list[dict[str, Any]] = []
    searched_sources: list[str] = []
    attempted_sources: list[str] = []
    source_validations: list[dict[str, Any]] = []

    validation_report: dict[str, Any] | None = None
    if include_alias_sources:
        attempted_sources.append("explicit_fallback_records")
        validation_report = validate_local_fallback_records(chosen_alias_path)
        alias_features = _load_alias_candidates(chosen_alias_path)
        if alias_features:
            searched_sources.append(str(chosen_alias_path))
            for feature in alias_features:
                lon_lat = _feature_lon_lat(feature)
                if lon_lat is None:
                    continue
                props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
                addr = _candidate_address(feature)
                score = _match_score(
                    input_components=address_components,
                    candidate_address=addr,
                    feature_props=props,
                )
                components = _candidate_components(addr, props)
                match_eval = _classify_candidate_match(
                    input_components=address_components,
                    candidate_components=components,
                    candidate_score=score,
                    min_score=min_score,
                )
                confidence = str(match_eval["confidence_tier"])
                if not _candidate_is_relevant(str(match_eval["match_type"]), float(score)):
                    continue
                source_type = "explicit_fallback_record"
                if not _source_type_allowed(source_type, allowed_source_types):
                    continue
                auto_usable = bool(match_eval["auto_usable"]) and _CONFIDENCE_RANK.get(confidence, -1) >= required_rank
                candidates.append(
                    {
                        "latitude": float(lon_lat[1]),
                        "longitude": float(lon_lat[0]),
                        "match_score": round(score, 4),
                        "ranking_score": round(score, 4),
                        "matched_address": addr,
                        "region_id": props.get("region_id"),
                        "source": "local_alias",
                        "source_type": source_type,
                        "source_path": str(chosen_alias_path),
                        "feature_properties": props,
                        "match_type": match_eval["match_type"],
                        "confidence_tier": confidence,
                        "auto_usable": auto_usable,
                        "street_similarity": match_eval["street_similarity"],
                        "candidate_components": components,
                    }
                )

    if include_authoritative_sources:
        attempted_sources.append("local_authoritative_datasets")
        manifests = list_prepared_regions(base_dir=regions_root)
        manifests.sort(
            key=lambda manifest: (
                -_region_candidate_priority(address_components, manifest),
                str(manifest.get("region_id") or ""),
            )
        )
        for manifest in manifests:
            region_id = str(manifest.get("region_id") or "")
            for layer_key in ("address_points", "parcel_address_points", "parcels"):
                local_path = resolve_region_file(manifest, layer_key, base_dir=regions_root)
                if not local_path:
                    continue
                path_obj = Path(local_path)
                if not path_obj.exists():
                    continue
                searched_sources.append(str(path_obj))
                if layer_key == "address_points":
                    source_type = "prepared_region_address_dataset"
                elif layer_key == "parcel_address_points":
                    source_type = "prepared_region_parcel_address_dataset"
                else:
                    source_type = "prepared_region_parcel_dataset"
                if not _source_type_allowed(source_type, allowed_source_types):
                    continue
                source_validations.append(validate_address_point_source(path_obj, f"{region_id}:{layer_key}"))
                for feature in _load_geojson_features(path_obj):
                    lon_lat = _feature_lon_lat(feature)
                    if lon_lat is None:
                        continue
                    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
                    addr = _candidate_address(feature)
                    score = _match_score(
                        input_components=address_components,
                        candidate_address=addr,
                        feature_props=props,
                    )
                    components = _candidate_components(addr, props)
                    match_eval = _classify_candidate_match(
                        input_components=address_components,
                        candidate_components=components,
                        candidate_score=score,
                        min_score=min_score,
                    )
                    confidence = str(match_eval["confidence_tier"])
                    if not _candidate_is_relevant(str(match_eval["match_type"]), float(score)):
                        continue
                    auto_usable = bool(match_eval["auto_usable"]) and _CONFIDENCE_RANK.get(confidence, -1) >= required_rank
                candidates.append(
                    {
                        "latitude": float(lon_lat[1]),
                        "longitude": float(lon_lat[0]),
                        "match_score": round(score, 4),
                        "ranking_score": round(score, 4),
                        "matched_address": addr,
                        "region_id": region_id or None,
                            "source": layer_key,
                            "source_type": source_type,
                            "source_path": str(path_obj),
                            "feature_properties": props,
                            "match_type": match_eval["match_type"],
                            "confidence_tier": confidence,
                            "auto_usable": auto_usable,
                            "street_similarity": match_eval["street_similarity"],
                            "candidate_components": components,
                        }
                    )

        configured_sources = _load_location_resolution_source_config()
        configured_sources.extend(_env_source_paths())
        for source_entry in configured_sources:
            path_obj = Path(str(source_entry.get("path") or "")).expanduser()
            if not path_obj.exists():
                continue
            source_name = str(source_entry.get("name") or path_obj.stem)
            source_type = str(source_entry.get("source_type") or "local_authoritative_dataset")
            searched_sources.append(str(path_obj))
            attempted_sources.append(source_name)
            source_validations.append(validate_address_point_source(path_obj, source_name))
            if not _source_type_allowed(source_type, allowed_source_types):
                continue
            for feature in _load_geojson_features(path_obj):
                lon_lat = _feature_lon_lat(feature)
                if lon_lat is None:
                    continue
                props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
                addr = _candidate_address(feature)
                score = _match_score(
                    input_components=address_components,
                    candidate_address=addr,
                    feature_props=props,
                )
                components = _candidate_components(addr, props)
                match_eval = _classify_candidate_match(
                    input_components=address_components,
                    candidate_components=components,
                    candidate_score=score,
                    min_score=min_score,
                )
                confidence = str(match_eval["confidence_tier"])
                if not _candidate_is_relevant(str(match_eval["match_type"]), float(score)):
                    continue
                auto_usable = bool(match_eval["auto_usable"]) and _CONFIDENCE_RANK.get(confidence, -1) >= required_rank
                candidates.append(
                    {
                        "latitude": float(lon_lat[1]),
                        "longitude": float(lon_lat[0]),
                        "match_score": round(score, 4),
                        "ranking_score": round(score, 4),
                        "matched_address": addr,
                        "region_id": props.get("region_id"),
                        "source": source_name,
                        "source_type": source_type,
                        "source_path": str(path_obj),
                        "feature_properties": props,
                        "match_type": match_eval["match_type"],
                        "confidence_tier": confidence,
                        "auto_usable": auto_usable,
                        "street_similarity": match_eval["street_similarity"],
                        "candidate_components": components,
                        "source_metadata": dict(source_entry.get("metadata") or {}),
                    }
                )

    def _source_priority(row: dict[str, Any]) -> int:
        source_type = str(row.get("source_type") or "")
        if source_type in {"county_address_dataset", "prepared_region_address_dataset", "prepared_region_parcel_address_dataset"}:
            return 0
        if source_type in {"statewide_parcel_dataset", "prepared_region_parcel_dataset"}:
            return 1
        if source_type == "explicit_fallback_record":
            return 2
        return 3

    for row in candidates:
        ranking_score = float(row.get("match_score") or 0.0)
        components = row.get("candidate_components") if isinstance(row.get("candidate_components"), dict) else {}
        candidate_city = normalize_address_for_matching(str((components or {}).get("city") or ""))
        candidate_state = normalize_address_for_matching(str((components or {}).get("state") or ""))
        candidate_postal = _zip5(str((components or {}).get("postal") or ""))

        if preferred_locality_norms and candidate_city:
            if candidate_city in preferred_locality_norms:
                ranking_score += 0.08
            else:
                ranking_score -= 0.03
        if preferred_postal_zip and candidate_postal:
            if candidate_postal == preferred_postal_zip:
                ranking_score += 0.06
            else:
                ranking_score -= 0.05
        if required_state_norm and candidate_state:
            if candidate_state == required_state_norm:
                ranking_score += 0.02
            else:
                ranking_score -= 0.12

        row["ranking_score"] = round(max(0.0, min(1.0, ranking_score)), 4)

    candidates.sort(
        key=lambda row: (
            -int(bool(row.get("auto_usable"))),
            -_CONFIDENCE_RANK.get(str(row.get("confidence_tier") or "unknown"), -1),
            -float(row.get("ranking_score") or row.get("match_score") or 0.0),
            _source_priority(row),
            str(row.get("region_id") or ""),
            str(row.get("matched_address") or ""),
        )
    )

    top = candidates[0] if candidates else None
    matched = bool(top and top.get("auto_usable"))
    confidence = str(top.get("confidence_tier")) if matched and top else None

    diagnostics: list[str] = []
    if validation_report:
        for warning in list(validation_report.get("warnings") or [])[:5]:
            diagnostics.append(warning)
        for error in list(validation_report.get("errors") or [])[:5]:
            diagnostics.append(error)
    for report in source_validations:
        for warning in list(report.get("warnings") or [])[:3]:
            diagnostics.append(warning)
        for error in list(report.get("errors") or [])[:3]:
            diagnostics.append(error)

    if not searched_sources:
        diagnostics.append("No local address-point datasets or fallback records were available for local resolution.")

    if candidates and not matched:
        best = candidates[0]
        diagnostics.append(
            "Local candidates were found but none passed confidence requirements "
            f"(top match_type={best.get('match_type')}, confidence={best.get('confidence_tier')}, "
            f"score={float(best.get('match_score') or 0.0):.2f})."
        )

    if matched and top:
        diagnostics.append(
            f"Resolved via local source {top.get('source')} with {top.get('match_type')} "
            f"(confidence={top.get('confidence_tier')}, score={float(top.get('match_score') or 0.0):.2f})."
        )

    if any((report.get("errors") or []) for report in source_validations):
        LOGGER.warning(
            "local_address_source_validation %s",
            json.dumps(
                {
                    "source_reports": [
                        {
                            "source_name": report.get("source_name"),
                            "record_count": report.get("record_count"),
                            "warning_count": len(report.get("warnings") or []),
                            "error_count": len(report.get("errors") or []),
                        }
                        for report in source_validations
                    ]
                },
                sort_keys=True,
            ),
        )

    return {
        "matched": matched,
        "confidence": confidence,
        "threshold": min_score,
        "input_address": address_components["submitted_address"],
        "normalized_address": address_components["normalized_address"],
        "candidate_count": len(candidates),
        "searched_sources": searched_sources[:10],
        "candidate_sources_attempted": attempted_sources,
        "candidates_found": len(candidates),
        "best_match": top if matched else None,
        "best_candidate": top,
        "top_candidates": candidates[:top_candidate_limit],
        "diagnostics": diagnostics,
        "match_method": str(top.get("match_type") or "none") if top else "none",
        "auto_usable": bool(top.get("auto_usable")) if top else False,
        "validation": validation_report,
        "source_validations": source_validations,
        "failure_reason": (
            None
            if matched
            else (
                str(top.get("match_type"))
                if top
                else "no_local_candidates"
            )
        ),
    }


def _iter_authoritative_source_paths(regions_root: str) -> list[tuple[str, Path, str]]:
    paths: list[tuple[str, Path, str]] = []
    manifests = list_prepared_regions(base_dir=regions_root)
    for manifest in manifests:
        region_id = str(manifest.get("region_id") or "")
        for layer_key in ("address_points", "parcel_address_points", "parcels"):
            local_path = resolve_region_file(manifest, layer_key, base_dir=regions_root)
            if not local_path:
                continue
            path_obj = Path(local_path).expanduser()
            if not path_obj.exists():
                continue
            if layer_key == "address_points":
                source_type = "prepared_region_address_dataset"
            elif layer_key == "parcel_address_points":
                source_type = "prepared_region_parcel_address_dataset"
            else:
                source_type = "prepared_region_parcel_dataset"
            paths.append((f"{region_id}:{layer_key}", path_obj, source_type))

    configured_sources = _load_location_resolution_source_config()
    configured_sources.extend(_env_source_paths())
    for source_entry in configured_sources:
        path_obj = Path(str(source_entry.get("path") or "")).expanduser()
        if not path_obj.exists():
            continue
        source_name = str(source_entry.get("name") or path_obj.stem)
        source_type = str(source_entry.get("source_type") or "local_authoritative_dataset")
        paths.append((source_name, path_obj, source_type))
    return paths


def infer_localities_for_zip(
    *,
    zip_code: str,
    regions_root: str,
    state_hint: str | None = "wa",
    max_localities: int = 6,
) -> dict[str, Any]:
    postal = _zip5(zip_code or "")
    state_norm = normalize_address_for_matching(state_hint or "")
    max_localities = max(1, min(20, int(max_localities or 6)))
    if not postal:
        return {
            "zip_code": "",
            "localities": [],
            "searched_sources": [],
            "diagnostics": ["ZIP code was missing or invalid; locality inference skipped."],
        }

    locality_counts: dict[str, int] = {}
    locality_labels: dict[str, str] = {}
    searched_sources: list[str] = []
    diagnostics: list[str] = []

    max_features_per_source_raw = str(os.getenv("WF_ZIP_LOCALITY_SCAN_MAX_FEATURES", "60000")).strip()
    try:
        max_features_per_source = max(5000, int(max_features_per_source_raw))
    except ValueError:
        max_features_per_source = 60000

    for source_name, path_obj, source_type in _iter_authoritative_source_paths(regions_root):
        features = _load_geojson_features(path_obj)
        if not features:
            continue
        searched_sources.append(f"{source_name}:{source_type}")
        scanned = 0
        matched = 0
        for feature in features:
            if scanned >= max_features_per_source:
                diagnostics.append(f"{source_name} scan capped at {max_features_per_source} records.")
                break
            scanned += 1
            props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            candidate_postal = _zip5(_first_non_empty(props, _ZIP_KEYS))
            if candidate_postal != postal:
                continue
            candidate_state = normalize_address_for_matching(_first_non_empty(props, _STATE_KEYS))
            if state_norm and candidate_state and candidate_state != state_norm:
                continue
            locality_raw = _first_non_empty(props, _CITY_KEYS)
            if not locality_raw:
                locality_raw = _extract_address_components(_candidate_address(feature)).get("city") or ""
            locality_norm = normalize_address_for_matching(locality_raw)
            if not locality_norm:
                continue
            matched += 1
            locality_counts[locality_norm] = locality_counts.get(locality_norm, 0) + 1
            locality_labels.setdefault(locality_norm, str(locality_raw).strip() or locality_norm.title())
        if matched > 0 and len(locality_counts) >= max_localities:
            break

    ranked = sorted(locality_counts.items(), key=lambda row: (-row[1], row[0]))
    localities = [locality_labels.get(norm, norm.title()) for norm, _count in ranked[:max_localities]]
    if not localities:
        diagnostics.append(f"No localities found for ZIP {postal} in authoritative sources.")

    return {
        "zip_code": postal,
        "localities": localities,
        "searched_sources": searched_sources[:20],
        "diagnostics": diagnostics[:20],
    }
