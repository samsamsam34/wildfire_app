from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Any

from backend.region_registry import list_prepared_regions, resolve_region_file


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


def normalize_address_for_matching(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    for pattern, replacement in _NORMALIZATION_REPLACEMENTS.items():
        normalized = re.sub(pattern, replacement, normalized)
    return " ".join(normalized.split())


def _extract_address_components(address: str) -> dict[str, str]:
    submitted = str(address or "").strip()
    normalized = normalize_address_for_matching(submitted)
    parts = [p.strip() for p in submitted.split(",") if p.strip()]
    first_part = parts[0] if parts else submitted
    number_match = re.match(r"^\s*(\d+[a-zA-Z0-9-]*)\s+(.*)$", first_part)
    house_number = number_match.group(1).strip().lower() if number_match else ""
    street = number_match.group(2).strip() if number_match else first_part
    zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", submitted)
    postal = zip_match.group(1) if zip_match else ""
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
    street_score = difflib.SequenceMatcher(
        None,
        input_components["street"],
        _candidate_component(feature_props, _STREET_KEYS) or candidate_norm,
    ).ratio()
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


def resolve_local_address_candidate(
    *,
    address: str,
    regions_root: str,
    alias_path: str | None = None,
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

    candidates: list[dict[str, Any]] = []
    searched_sources: list[str] = []

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
            candidates.append(
                {
                    "latitude": float(lon_lat[1]),
                    "longitude": float(lon_lat[0]),
                    "match_score": round(score, 4),
                    "matched_address": addr,
                    "region_id": props.get("region_id"),
                    "source": "local_alias",
                    "source_path": str(chosen_alias_path),
                    "feature_properties": props,
                }
            )

    manifests = list_prepared_regions(base_dir=regions_root)
    manifests.sort(
        key=lambda manifest: (
            -_region_candidate_priority(address_components, manifest),
            str(manifest.get("region_id") or ""),
        )
    )
    for manifest in manifests:
        region_id = str(manifest.get("region_id") or "")
        for layer_key in ("address_points", "parcel_address_points"):
            local_path = resolve_region_file(manifest, layer_key, base_dir=regions_root)
            if not local_path:
                continue
            path_obj = Path(local_path)
            if not path_obj.exists():
                continue
            searched_sources.append(str(path_obj))
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
                candidates.append(
                    {
                        "latitude": float(lon_lat[1]),
                        "longitude": float(lon_lat[0]),
                        "match_score": round(score, 4),
                        "matched_address": addr,
                        "region_id": region_id or None,
                        "source": layer_key,
                        "source_path": str(path_obj),
                        "feature_properties": props,
                    }
                )

    candidates.sort(
        key=lambda row: (
            -float(row.get("match_score") or 0.0),
            str(row.get("region_id") or ""),
            str(row.get("matched_address") or ""),
        )
    )
    top = candidates[0] if candidates else None
    matched = bool(top and float(top.get("match_score") or 0.0) >= min_score)
    confidence = "low"
    if matched:
        score = float(top.get("match_score") or 0.0)
        if score >= 0.9:
            confidence = "high"
        elif score >= 0.8:
            confidence = "medium"
    diagnostics: list[str] = []
    if not searched_sources:
        diagnostics.append("No local address-point datasets or alias files were configured for fallback matching.")
    if candidates and not matched:
        diagnostics.append(
            f"Local candidates found but top score {float(top.get('match_score') or 0.0):.2f} "
            f"was below threshold {min_score:.2f}."
        )
    if matched and top:
        diagnostics.append(
            f"Resolved via local fallback source {top.get('source')} (score={float(top.get('match_score') or 0.0):.2f})."
        )
    return {
        "matched": matched,
        "confidence": confidence if matched else None,
        "threshold": min_score,
        "input_address": address_components["submitted_address"],
        "normalized_address": address_components["normalized_address"],
        "candidate_count": len(candidates),
        "searched_sources": searched_sources[:10],
        "best_match": top if matched else None,
        "top_candidates": candidates[:3],
        "diagnostics": diagnostics,
    }
