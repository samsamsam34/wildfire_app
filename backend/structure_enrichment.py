from __future__ import annotations

from typing import Any


PROVENANCE_OBSERVED_PUBLIC_RECORD = "observed_public_record"
PROVENANCE_INFERRED_FROM_GEOMETRY = "inferred_from_geometry"
PROVENANCE_USER_PROVIDED = "user_provided"
PROVENANCE_MISSING = "missing"

PROVENANCE_STATUSES = {
    PROVENANCE_OBSERVED_PUBLIC_RECORD,
    PROVENANCE_INFERRED_FROM_GEOMETRY,
    PROVENANCE_USER_PROVIDED,
    PROVENANCE_MISSING,
}


PUBLIC_RECORD_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "year_built": (
        "year_built",
        "yr_built",
        "yearbuilt",
        "built_year",
        "construction_year",
    ),
    "building_area_sqft": (
        "gross_building_area",
        "gross_building_area_sqft",
        "building_area_sqft",
        "building_sqft",
        "living_area",
        "total_bldg_area",
        "bldg_area",
        "sqft",
    ),
    "land_use_class": (
        "land_use",
        "land_use_class",
        "occupancy",
        "occupancy_class",
        "use_code",
        "property_use",
    ),
    "roof_material_public_record": (
        "roof_material",
        "roof_type",
        "roof_cover",
        "roof_covering",
        "assessor_roof_material",
        "material_roof",
    ),
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_key(key: str) -> str:
    normalized = []
    for ch in str(key or "").strip().lower():
        if ch.isalnum():
            normalized.append(ch)
        else:
            normalized.append("_")
    return "_".join(part for part in "".join(normalized).split("_") if part)


def _normalize_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    normalized: dict[str, Any] = {}
    for raw_key, value in record.items():
        key = _normalize_key(str(raw_key))
        if not key:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if key not in normalized:
            normalized[key] = value
    return normalized


def _coerce_year(value: Any) -> int | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    year = int(round(numeric))
    if 1700 <= year <= 2100:
        return year
    return None


def _coerce_area_sqft(value: Any) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if numeric <= 0.0:
        return None
    return round(float(numeric), 1)


def _coerce_land_use(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_roof_material(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lookup_first(
    field_key: str,
    *,
    sources: list[tuple[str, dict[str, Any]]],
) -> tuple[Any | None, str | None]:
    aliases = PUBLIC_RECORD_FIELD_ALIASES.get(field_key) or ()
    for source_name, source_payload in sources:
        for alias in aliases:
            normalized_alias = _normalize_key(alias)
            if normalized_alias in source_payload:
                value = source_payload.get(normalized_alias)
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                return value, source_name
    return None, None


def extract_structure_public_record_fields(
    *,
    parcel_properties: dict[str, Any] | None = None,
    address_point_properties: dict[str, Any] | None = None,
    region_public_record_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: list[tuple[str, dict[str, Any]]] = []
    region_payload = _normalize_record(region_public_record_fields)
    parcel_payload = _normalize_record(parcel_properties)
    address_payload = _normalize_record(address_point_properties)
    if region_payload:
        sources.append(("region_public_record", region_payload))
    if parcel_payload:
        sources.append(("parcel_properties", parcel_payload))
    if address_payload:
        sources.append(("address_point_properties", address_payload))

    value_map: dict[str, Any] = {
        "year_built": None,
        "building_area_sqft": None,
        "land_use_class": None,
        "roof_material_public_record": None,
    }
    source_map: dict[str, str | None] = {
        "year_built": None,
        "building_area_sqft": None,
        "land_use_class": None,
        "roof_material_public_record": None,
    }

    for field_key in list(value_map.keys()):
        raw_value, source_name = _lookup_first(field_key, sources=sources)
        coerced: Any = None
        if field_key == "year_built":
            coerced = _coerce_year(raw_value)
        elif field_key == "building_area_sqft":
            coerced = _coerce_area_sqft(raw_value)
        elif field_key == "land_use_class":
            coerced = _coerce_land_use(raw_value)
        elif field_key == "roof_material_public_record":
            coerced = _coerce_roof_material(raw_value)
        value_map[field_key] = coerced
        source_map[field_key] = source_name if coerced is not None else None

    return {
        **value_map,
        "source_by_field": source_map,
    }


def _provenance_confidence(provenance: str) -> float:
    if provenance == PROVENANCE_OBSERVED_PUBLIC_RECORD:
        return 0.82
    if provenance == PROVENANCE_INFERRED_FROM_GEOMETRY:
        return 0.58
    if provenance == PROVENANCE_USER_PROVIDED:
        return 0.78
    return 0.0


def enrich_structure_attributes(
    *,
    base_structure_attributes: dict[str, Any] | None,
    public_record_fields: dict[str, Any] | None = None,
    user_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = dict(base_structure_attributes or {})
    records = dict(public_record_fields or {})
    users = dict(user_attributes or {})

    area_sqft = _coerce_area_sqft(((base.get("area") or {}).get("sqft")))
    age_proxy_year = _coerce_year(((base.get("estimated_age_proxy") or {}).get("proxy_year")))
    user_year_built = _coerce_year(users.get("construction_year"))
    user_roof = _coerce_roof_material(users.get("roof_type"))

    record_year = _coerce_year(records.get("year_built"))
    record_area_sqft = _coerce_area_sqft(records.get("building_area_sqft"))
    record_land_use = _coerce_land_use(records.get("land_use_class"))
    record_roof = _coerce_roof_material(records.get("roof_material_public_record"))

    year_built = None
    year_provenance = PROVENANCE_MISSING
    if user_year_built is not None:
        year_built = user_year_built
        year_provenance = PROVENANCE_USER_PROVIDED
    elif record_year is not None:
        year_built = record_year
        year_provenance = PROVENANCE_OBSERVED_PUBLIC_RECORD
    elif age_proxy_year is not None:
        year_built = age_proxy_year
        year_provenance = PROVENANCE_INFERRED_FROM_GEOMETRY

    building_area_sqft = None
    area_provenance = PROVENANCE_MISSING
    if record_area_sqft is not None:
        building_area_sqft = record_area_sqft
        area_provenance = PROVENANCE_OBSERVED_PUBLIC_RECORD
    elif area_sqft is not None:
        building_area_sqft = area_sqft
        area_provenance = PROVENANCE_INFERRED_FROM_GEOMETRY

    land_use_class = None
    land_use_provenance = PROVENANCE_MISSING
    if record_land_use is not None:
        land_use_class = record_land_use
        land_use_provenance = PROVENANCE_OBSERVED_PUBLIC_RECORD

    roof_material_public_record = None
    roof_provenance = PROVENANCE_MISSING
    if user_roof is not None:
        roof_material_public_record = user_roof
        roof_provenance = PROVENANCE_USER_PROVIDED
    elif record_roof is not None:
        roof_material_public_record = record_roof
        roof_provenance = PROVENANCE_OBSERVED_PUBLIC_RECORD

    base["year_built"] = year_built
    base["building_area_sqft"] = building_area_sqft
    base["land_use_class"] = land_use_class
    base["roof_material_public_record"] = roof_material_public_record
    base["attribute_provenance"] = {
        "year_built": year_provenance,
        "building_area_sqft": area_provenance,
        "land_use_class": land_use_provenance,
        "roof_material_public_record": roof_provenance,
    }
    base["attribute_confidence"] = {
        "year_built": _provenance_confidence(year_provenance),
        "building_area_sqft": _provenance_confidence(area_provenance),
        "land_use_class": _provenance_confidence(land_use_provenance),
        "roof_material_public_record": _provenance_confidence(roof_provenance),
    }
    source_by_field = records.get("source_by_field") if isinstance(records.get("source_by_field"), dict) else {}
    base["attribute_sources"] = {
        "year_built": source_by_field.get("year_built"),
        "building_area_sqft": source_by_field.get("building_area_sqft"),
        "land_use_class": source_by_field.get("land_use_class"),
        "roof_material_public_record": source_by_field.get("roof_material_public_record"),
    }
    return base
