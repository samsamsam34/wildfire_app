from __future__ import annotations

from backend.models import PropertyAttributes
from backend.normalization import normalize_property_attributes, normalize_roof_type, normalize_vent_type


def test_roof_type_normalization_maps_ui_enums() -> None:
    assert normalize_roof_type("wood_shake") == "untreated wood shake"
    assert normalize_roof_type("cedar_shake") == "untreated wood shake"
    assert normalize_roof_type("class_a_asphalt_composition") == "class a"
    assert normalize_roof_type("asphalt_composition") == "composite"


def test_vent_type_normalization_maps_ui_enums() -> None:
    assert normalize_vent_type("ember_resistant_vents") == "ember-resistant"
    assert normalize_vent_type("covered_vents_screens") == "ember-resistant"
    assert normalize_vent_type("standard_vents") == "standard"


def test_unknown_normalization_value_falls_back_safely() -> None:
    assert normalize_roof_type("vegetated roof system") == "vegetated roof system"
    assert normalize_vent_type("prototype vent alpha") == "prototype vent alpha"


def test_normalize_property_attributes_updates_categorical_fields() -> None:
    attrs = PropertyAttributes(
        roof_type="class_a_asphalt_composition",
        vent_type="ember_resistant_vents",
        siding_type="stucco_masonry",
    )
    normalized = normalize_property_attributes(attrs)

    assert normalized.roof_type == "class a"
    assert normalized.vent_type == "ember-resistant"
    assert normalized.siding_type == "stucco/masonry"
