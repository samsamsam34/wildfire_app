from __future__ import annotations

from backend.models import PropertyAttributes


CANONICAL_ROOF_TYPES = {
    "wood",
    "untreated wood shake",
    "class a",
    "metal",
    "tile",
    "composite",
    "unknown",
}


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())
    return cleaned or None


def normalize_roof_type(value: str | None) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None

    if cleaned in {"unknown", "unsure", "n/a"}:
        return "unknown"

    if any(token in cleaned for token in ("class a", "classa", "fire rated", "fire-resistant")):
        return "class a"
    if any(token in cleaned for token in ("wood shake", "cedar shake", "shake roof", "shingle shake")):
        return "untreated wood shake"
    if cleaned == "wood":
        return "wood"
    if "metal" in cleaned:
        return "metal"
    if any(token in cleaned for token in ("tile", "slate", "clay")):
        return "tile"
    if any(token in cleaned for token in ("asphalt", "composition", "composite", "fiberglass", "architectural")):
        return "composite"
    return cleaned


def normalize_vent_type(value: str | None) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None

    if cleaned in {"unknown", "unsure", "n/a"}:
        return "unknown"
    if any(token in cleaned for token in ("ember", "covered vent", "covered vents", "screen", "screens")):
        return "ember-resistant"
    if any(token in cleaned for token in ("standard", "open", "unscreened")):
        return "standard"
    return cleaned


def normalize_siding_type(value: str | None) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    if any(token in cleaned for token in ("stucco", "masonry", "brick")):
        return "stucco/masonry"
    if "fiber cement" in cleaned:
        return "fiber cement"
    if "wood" in cleaned:
        return "wood"
    if any(token in cleaned for token in ("vinyl", "composite")):
        return "vinyl/composite"
    return cleaned


def normalize_property_attributes(attrs: PropertyAttributes) -> PropertyAttributes:
    return attrs.model_copy(
        update={
            "roof_type": normalize_roof_type(attrs.roof_type),
            "vent_type": normalize_vent_type(attrs.vent_type),
            "siding_type": normalize_siding_type(attrs.siding_type),
        }
    )


def normalized_attribute_changes(
    original: PropertyAttributes,
    normalized: PropertyAttributes,
) -> dict[str, dict[str, str]]:
    changes: dict[str, dict[str, str]] = {}
    for field in ("roof_type", "vent_type", "siding_type"):
        before = getattr(original, field, None)
        after = getattr(normalized, field, None)
        if before is None or after is None:
            continue
        if before != after:
            changes[field] = {"input": before, "normalized": after}
    return changes

