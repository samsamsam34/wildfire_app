from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _normalize_fraction_or_percent(value: Any) -> float | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    if parsed <= 1.0:
        parsed *= 100.0
    return _clamp_0_100(parsed)


def _geocode_confidence_score(geocode_meta: dict[str, Any]) -> float:
    score = _normalize_fraction_or_percent(geocode_meta.get("confidence_score"))
    status = str(geocode_meta.get("geocode_status") or "").strip().lower()
    precision = str(geocode_meta.get("geocode_precision") or "").strip().lower()
    if score is None:
        score = {
            "rooftop": 92.0,
            "parcel_or_address_point": 88.0,
            "parcel": 85.0,
            "user_selected_point": 86.0,
            "interpolated": 68.0,
            "approximate": 48.0,
            "unknown": 55.0,
        }.get(precision, 55.0)
    if status in {"provider_error", "no_match", "rejected"}:
        score = min(score, 20.0)
    elif status in {"ambiguous_match", "low_confidence"}:
        score = min(score, 55.0)
    return round(_clamp_0_100(score), 1)


def _parcel_confidence_score(property_level_context: dict[str, Any]) -> tuple[float, str]:
    parcel_resolution = property_level_context.get("parcel_resolution")
    if isinstance(parcel_resolution, dict):
        confidence = _normalize_fraction_or_percent(parcel_resolution.get("confidence"))
        if confidence is not None:
            return round(confidence, 1), str(parcel_resolution.get("status") or "not_found").strip().lower()
    parcel_id = property_level_context.get("parcel_id")
    parcel_geometry = property_level_context.get("parcel_geometry")
    lookup_method = str(property_level_context.get("parcel_lookup_method") or "").strip().lower()
    if parcel_id or isinstance(parcel_geometry, dict):
        if lookup_method == "contains_point":
            return 92.0, "matched"
        if lookup_method == "multiple_candidates":
            return 62.0, "multiple_candidates"
        if lookup_method == "nearest_within_tolerance":
            lookup_distance = _to_float(property_level_context.get("parcel_lookup_distance_m")) or 0.0
            return round(_clamp_0_100(80.0 - (0.6 * lookup_distance)), 1), "matched"
        return 86.0, "matched"
    return 0.0, "not_found"


def _footprint_confidence_score(property_level_context: dict[str, Any]) -> tuple[float, str, str]:
    footprint_resolution = property_level_context.get("footprint_resolution")
    if isinstance(footprint_resolution, dict):
        raw = footprint_resolution.get("confidence_score")
        confidence = _normalize_fraction_or_percent(raw)
        status = str(footprint_resolution.get("match_status") or "none").strip().lower()
        method = str(footprint_resolution.get("match_method") or "").strip().lower()
        if confidence is not None:
            return round(confidence, 1), status or "none", method
    status = str(property_level_context.get("structure_match_status") or "none").strip().lower()
    method = str(property_level_context.get("structure_match_method") or "").strip().lower()
    confidence = _normalize_fraction_or_percent(property_level_context.get("structure_match_confidence"))
    if confidence is None:
        confidence = 90.0 if status == "matched" else (55.0 if status == "ambiguous" else 0.0)
    return round(_clamp_0_100(confidence), 1), status or "none", method


def _candidate_count(property_level_context: dict[str, Any]) -> int:
    try:
        if property_level_context.get("candidate_structure_count") is not None:
            return max(0, int(property_level_context.get("candidate_structure_count")))
    except (TypeError, ValueError):
        pass
    candidates = property_level_context.get("structure_match_candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return 0


def _overall_property_confidence(
    *,
    geocode_confidence: float,
    parcel_confidence: float,
    footprint_confidence: float,
    parcel_status: str,
    footprint_status: str,
    multiple_footprints_on_parcel: bool,
    footprint_outside_parcel: bool,
) -> float:
    components: list[tuple[float, float]] = [(0.35, geocode_confidence)]
    if parcel_status in {"matched", "multiple_candidates"}:
        components.append((0.30, parcel_confidence))
    if footprint_status in {"matched", "ambiguous"}:
        components.append((0.35, footprint_confidence))
    if not components:
        return 0.0
    weight_sum = sum(weight for weight, _ in components)
    blended = sum(weight * score for weight, score in components) / max(weight_sum, 1e-6)

    if parcel_status == "not_found" and footprint_status == "matched":
        blended *= 0.88
    elif footprint_status == "none" and parcel_status in {"matched", "multiple_candidates"}:
        blended *= 0.86
    if parcel_status == "multiple_candidates":
        blended -= 8.0
    if footprint_status == "ambiguous":
        blended -= 9.0
    if multiple_footprints_on_parcel:
        blended -= 6.0
    if footprint_outside_parcel:
        blended -= 14.0
    return round(_clamp_0_100(blended), 1)


def build_property_linkage_summary(
    *,
    geocode_meta: dict[str, Any],
    property_level_context: dict[str, Any],
) -> dict[str, Any]:
    geocode_confidence = _geocode_confidence_score(geocode_meta if isinstance(geocode_meta, dict) else {})
    plc = property_level_context if isinstance(property_level_context, dict) else {}
    parcel_confidence, parcel_status = _parcel_confidence_score(plc)
    footprint_confidence, footprint_status, footprint_method = _footprint_confidence_score(plc)

    structure_candidate_count = _candidate_count(plc)
    multiple_footprints_on_parcel = (
        parcel_status in {"matched", "multiple_candidates"}
        and footprint_status in {"matched", "ambiguous"}
        and structure_candidate_count > 1
    )

    # If parcel matched but structure required non-parcel-aware footprint selection,
    # treat as potential cross-dataset misalignment.
    footprint_outside_parcel = (
        parcel_status in {"matched", "multiple_candidates"}
        and footprint_status in {"matched", "ambiguous"}
        and footprint_method not in {"parcel_intersection", "point_in_footprint"}
    )
    if footprint_method == "parcel_intersection":
        footprint_outside_parcel = False

    overall_property_confidence = _overall_property_confidence(
        geocode_confidence=geocode_confidence,
        parcel_confidence=parcel_confidence,
        footprint_confidence=footprint_confidence,
        parcel_status=parcel_status,
        footprint_status=footprint_status,
        multiple_footprints_on_parcel=multiple_footprints_on_parcel,
        footprint_outside_parcel=footprint_outside_parcel,
    )

    selection_reason = []
    if footprint_method in {"parcel_intersection", "point_in_footprint", "nearest_building_fallback"}:
        selection_reason.append(f"Selected structure via {footprint_method}.")
    if structure_candidate_count > 1:
        selection_reason.append(
            "Structure candidate ranking prioritized proximity, size plausibility, and parcel centrality."
        )
    if parcel_status == "not_found" and footprint_status == "matched":
        selection_reason.append(
            "Parcel match is missing; structure linkage relies on geocode-to-footprint proximity."
        )
    if multiple_footprints_on_parcel:
        selection_reason.append(
            "Multiple structures were present on the parcel; confidence is reduced."
        )
    if footprint_outside_parcel:
        selection_reason.append(
            "Matched footprint does not align cleanly with parcel linkage; review geometry alignment."
        )

    return {
        "geocode_confidence": geocode_confidence,
        "parcel_confidence": round(_clamp_0_100(parcel_confidence), 1),
        "footprint_confidence": round(_clamp_0_100(footprint_confidence), 1),
        "overall_property_confidence": overall_property_confidence,
        "parcel_status": parcel_status,
        "footprint_status": footprint_status,
        "footprint_match_method": footprint_method or None,
        "multiple_footprints_on_parcel": bool(multiple_footprints_on_parcel),
        "footprint_outside_parcel": bool(footprint_outside_parcel),
        "structure_candidate_count": int(max(0, structure_candidate_count)),
        "selection_notes": selection_reason[:4],
    }

