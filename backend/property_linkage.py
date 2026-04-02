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


def _anchor_stage(geocode_meta: dict[str, Any], property_level_context: dict[str, Any]) -> tuple[str, float, str | None]:
    geocode_score = _geocode_confidence_score(geocode_meta)
    anchor_score = _normalize_fraction_or_percent(
        property_level_context.get("property_anchor_quality_score")
        if property_level_context.get("property_anchor_quality_score") is not None
        else property_level_context.get("anchor_quality_score")
    )
    anchor_confidence = round(_clamp_0_100(anchor_score if anchor_score is not None else geocode_score), 1)
    geocode_status = str(geocode_meta.get("geocode_status") or "").strip().lower()
    geocode_precision = str(geocode_meta.get("geocode_precision") or "").strip().lower()
    anchor_source = (
        str(property_level_context.get("property_anchor_source") or "").strip()
        or str(geocode_meta.get("geocode_provider") or geocode_meta.get("provider") or "").strip()
        or None
    )
    if geocode_status in {"provider_error", "no_match", "rejected"}:
        return "unresolved", round(min(anchor_confidence, 20.0), 1), anchor_source
    if geocode_status in {"ambiguous_match", "low_confidence"}:
        return "ambiguous", round(min(anchor_confidence, 55.0), 1), anchor_source
    if geocode_precision in {"interpolated", "approximate", "unknown"} or anchor_confidence < 60.0:
        return "approximate", round(min(anchor_confidence, 72.0), 1), anchor_source
    return "resolved", anchor_confidence, anchor_source


def _parcel_confidence_score(property_level_context: dict[str, Any]) -> tuple[float, str, str | None, int]:
    parcel_resolution = property_level_context.get("parcel_resolution")
    if isinstance(parcel_resolution, dict):
        confidence = _normalize_fraction_or_percent(parcel_resolution.get("confidence"))
        status = str(parcel_resolution.get("status") or "not_found").strip().lower() or "not_found"
        source = str(parcel_resolution.get("source") or "").strip() or None
        try:
            candidate_count = max(0, int(parcel_resolution.get("candidates_considered") or 0))
        except (TypeError, ValueError):
            candidate_count = 0
        if candidate_count == 0 and isinstance(parcel_resolution.get("candidate_summaries"), list):
            candidate_count = len(parcel_resolution.get("candidate_summaries") or [])
        if confidence is not None:
            return round(confidence, 1), status, source, candidate_count
    parcel_id = property_level_context.get("parcel_id")
    parcel_geometry = property_level_context.get("parcel_geometry")
    lookup_method = str(property_level_context.get("parcel_lookup_method") or "").strip().lower()
    source = (
        str(
            property_level_context.get("parcel_source")
            or property_level_context.get("parcel_source_name")
            or ""
        ).strip()
        or None
    )
    if parcel_id or isinstance(parcel_geometry, dict):
        if lookup_method == "contains_point":
            return 92.0, "matched", source, 1
        if lookup_method == "multiple_candidates":
            return 62.0, "multiple_candidates", source, 2
        if lookup_method == "nearest_within_tolerance":
            lookup_distance = _to_float(property_level_context.get("parcel_lookup_distance_m")) or 0.0
            return round(_clamp_0_100(80.0 - (0.6 * lookup_distance)), 1), "matched", source, 1
        return 86.0, "matched", source, 1
    if lookup_method in {"provider_unavailable", "lookup_unavailable"}:
        return 18.0, "provider_unavailable", source, 0
    return 0.0, "not_found", source, 0


def _footprint_confidence_score(property_level_context: dict[str, Any]) -> tuple[float, str, str, str | None, int]:
    footprint_resolution = property_level_context.get("footprint_resolution")
    if isinstance(footprint_resolution, dict):
        raw = footprint_resolution.get("confidence_score")
        confidence = _normalize_fraction_or_percent(raw)
        status = str(footprint_resolution.get("match_status") or "none").strip().lower()
        method = str(footprint_resolution.get("match_method") or "").strip().lower()
        source = str(footprint_resolution.get("selected_source") or "").strip() or None
        try:
            candidate_count = max(0, int(footprint_resolution.get("candidates_considered") or 0))
        except (TypeError, ValueError):
            candidate_count = 0
        if candidate_count == 0 and isinstance(footprint_resolution.get("candidate_summaries"), list):
            candidate_count = len(footprint_resolution.get("candidate_summaries") or [])
        if candidate_count == 0 and isinstance(property_level_context.get("structure_match_candidates"), list):
            candidate_count = len(property_level_context.get("structure_match_candidates") or [])
        if source is None:
            source = (
                str(
                    property_level_context.get("footprint_source_name")
                    or property_level_context.get("footprint_source")
                    or property_level_context.get("building_source")
                    or ""
                ).strip()
                or None
            )
        if confidence is not None:
            return round(confidence, 1), status or "none", method, source, candidate_count
    status = str(property_level_context.get("structure_match_status") or "none").strip().lower()
    method = str(property_level_context.get("structure_match_method") or "").strip().lower()
    confidence = _normalize_fraction_or_percent(property_level_context.get("structure_match_confidence"))
    source = (
        str(
            property_level_context.get("footprint_source_name")
            or property_level_context.get("footprint_source")
            or property_level_context.get("building_source")
            or ""
        ).strip()
        or None
    )
    candidate_count = _candidate_count(property_level_context)
    if confidence is None:
        confidence = 90.0 if status == "matched" else (55.0 if status == "ambiguous" else 0.0)
    return round(_clamp_0_100(confidence), 1), status or "none", method, source, candidate_count


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
    anchor_confidence: float,
    anchor_status: str,
    parcel_confidence: float,
    footprint_confidence: float,
    parcel_status: str,
    footprint_status: str,
    multiple_footprints_on_parcel: bool,
    footprint_outside_parcel: bool,
) -> float:
    components: list[tuple[float, float]] = [(0.34, anchor_confidence)]
    if parcel_status in {"matched", "multiple_candidates", "provider_unavailable"}:
        components.append((0.33, parcel_confidence))
    if footprint_status in {"matched", "ambiguous", "provider_unavailable"}:
        components.append((0.33, footprint_confidence))
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
        blended -= 14.0
        blended = min(blended, 62.0)
    if footprint_status in {"none", "error"}:
        blended = min(blended, 56.0)
    if anchor_status in {"ambiguous", "approximate"}:
        blended = min(blended, 74.0)
    if anchor_status == "unresolved":
        blended = min(blended, 45.0)
    if multiple_footprints_on_parcel:
        blended -= 6.0
        blended = min(blended, 76.0)
    if footprint_outside_parcel:
        blended -= 14.0
    return round(_clamp_0_100(blended), 1)


def build_property_linkage_summary(
    *,
    geocode_meta: dict[str, Any],
    property_level_context: dict[str, Any],
) -> dict[str, Any]:
    plc = property_level_context if isinstance(property_level_context, dict) else {}
    normalized_geocode_meta = geocode_meta if isinstance(geocode_meta, dict) else {}
    anchor_status, anchor_confidence, anchor_source = _anchor_stage(normalized_geocode_meta, plc)
    parcel_confidence, parcel_status, parcel_source, parcel_candidate_count = _parcel_confidence_score(plc)
    footprint_confidence, footprint_status, footprint_method, footprint_source, footprint_candidate_count = _footprint_confidence_score(plc)
    selected_structure_id = (
        str(
            plc.get("selected_structure_id")
            or plc.get("matched_structure_id")
            or ""
        ).strip()
        or None
    )

    structure_candidate_count = max(_candidate_count(plc), footprint_candidate_count)
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
    no_confident_structure_match = (
        footprint_status in {"ambiguous", "none", "error", "provider_unavailable"}
        or footprint_confidence < 60.0
        or (footprint_status == "matched" and selected_structure_id is None and structure_candidate_count > 1)
    )

    mismatch_flags: list[str] = []
    if anchor_status in {"ambiguous", "approximate", "unresolved"}:
        mismatch_flags.append("low_anchor_confidence")
    if parcel_status == "multiple_candidates":
        mismatch_flags.append("parcel_ambiguous")
    if parcel_status in {"not_found", "provider_unavailable"} and footprint_status == "matched":
        mismatch_flags.append("footprint_without_parcel_match")
    if footprint_outside_parcel:
        mismatch_flags.append("footprint_parcel_misalignment")
    if multiple_footprints_on_parcel:
        mismatch_flags.append("multiple_footprints_on_parcel")
    if footprint_status == "ambiguous":
        mismatch_flags.append("footprint_ambiguous")
    if no_confident_structure_match:
        mismatch_flags.append("no_confident_structure_match")
    geocode_to_anchor_distance = _to_float(plc.get("geocode_to_anchor_distance_m"))
    if geocode_to_anchor_distance is not None and geocode_to_anchor_distance >= 35.0:
        mismatch_flags.append("anchor_far_from_geocode")
    structure_match_distance = _to_float(plc.get("structure_match_distance_m"))
    if (
        footprint_status == "matched"
        and structure_match_distance is not None
        and structure_match_distance >= 30.0
    ):
        mismatch_flags.append("footprint_far_from_anchor")
    if footprint_status == "matched" and selected_structure_id is None:
        mismatch_flags.append("selected_structure_id_missing")
    mismatch_flags = list(dict.fromkeys(mismatch_flags))

    overall_property_confidence = _overall_property_confidence(
        anchor_confidence=anchor_confidence,
        anchor_status=anchor_status,
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
    if parcel_status == "matched" and str(plc.get("parcel_lookup_method") or "").strip().lower() == "contains_point":
        selection_reason.append("Parcel linkage preferred containing parcel geometry over nearest fallback.")
    elif parcel_status == "matched":
        selection_reason.append("Parcel linkage used nearest eligible parcel candidate.")
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
    if no_confident_structure_match:
        selection_reason.append(
            "No confident structure match was available; assessment retained conservative geometry assumptions."
        )

    return {
        "anchor_status": anchor_status,
        "anchor_confidence": anchor_confidence,
        "anchor_source": anchor_source,
        "parcel_status": parcel_status,
        "parcel_confidence": round(_clamp_0_100(parcel_confidence), 1),
        "parcel_source": parcel_source,
        "parcel_candidate_count": int(max(0, parcel_candidate_count)),
        "footprint_status": footprint_status,
        "footprint_confidence": round(_clamp_0_100(footprint_confidence), 1),
        "footprint_source": footprint_source,
        "footprint_candidate_count": int(max(0, footprint_candidate_count)),
        "selected_structure_id": selected_structure_id,
        "overall_property_confidence": overall_property_confidence,
        "mismatch_flags": mismatch_flags,

        # Legacy compatibility mirrors.
        "geocode_confidence": anchor_confidence,
        "parcel_confidence": round(_clamp_0_100(parcel_confidence), 1),
        "footprint_confidence": round(_clamp_0_100(footprint_confidence), 1),
        "footprint_match_method": footprint_method or None,
        "multiple_footprints_on_parcel": bool(multiple_footprints_on_parcel),
        "footprint_outside_parcel": bool(footprint_outside_parcel),
        "structure_candidate_count": int(max(0, structure_candidate_count)),
        "selection_notes": selection_reason[:4],
    }
