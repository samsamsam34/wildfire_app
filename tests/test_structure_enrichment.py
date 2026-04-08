from __future__ import annotations

from backend.structure_enrichment import (
    PROVENANCE_INFERRED_FROM_GEOMETRY,
    PROVENANCE_MISSING,
    PROVENANCE_OBSERVED_PUBLIC_RECORD,
    PROVENANCE_USER_PROVIDED,
    enrich_structure_attributes,
    extract_structure_public_record_fields,
)


def test_structure_enrichment_partial_public_record_availability() -> None:
    public_fields = extract_structure_public_record_fields(
        parcel_properties={
            "YEAR_BUILT": 1986,
            "occupancy": "single_family_residential",
        }
    )
    enriched = enrich_structure_attributes(
        base_structure_attributes={
            "area": {"sqft": 1920.0, "source": "building_footprint_geometry"},
            "estimated_age_proxy": {"proxy_year": 1975, "era_bucket": "1960_1979"},
        },
        public_record_fields=public_fields,
        user_attributes=None,
    )

    assert enriched["year_built"] == 1986
    assert enriched["building_area_sqft"] == 1920.0
    assert enriched["land_use_class"] == "single_family_residential"
    assert enriched["roof_material_public_record"] is None

    provenance = enriched["attribute_provenance"]
    assert provenance["year_built"] == PROVENANCE_OBSERVED_PUBLIC_RECORD
    assert provenance["building_area_sqft"] == PROVENANCE_INFERRED_FROM_GEOMETRY
    assert provenance["land_use_class"] == PROVENANCE_OBSERVED_PUBLIC_RECORD
    assert provenance["roof_material_public_record"] == PROVENANCE_MISSING

    confidence = enriched["attribute_confidence"]
    assert 0.0 < float(confidence["year_built"]) <= 1.0
    assert 0.0 < float(confidence["building_area_sqft"]) <= 1.0
    assert 0.0 < float(confidence["land_use_class"]) <= 1.0
    assert float(confidence["roof_material_public_record"]) == 0.0


def test_structure_enrichment_user_provided_overrides_public_record_fields() -> None:
    public_fields = extract_structure_public_record_fields(
        parcel_properties={
            "yearbuilt": "1974",
            "gross_building_area": "2480",
            "roof_material": "wood shake",
        }
    )
    enriched = enrich_structure_attributes(
        base_structure_attributes={
            "area": {"sqft": 2050.0, "source": "building_footprint_geometry"},
            "estimated_age_proxy": {"proxy_year": 1969, "era_bucket": "1960_1979"},
        },
        public_record_fields=public_fields,
        user_attributes={
            "construction_year": 2003,
            "roof_type": "class a",
        },
    )

    assert enriched["year_built"] == 2003
    assert enriched["roof_material_public_record"] == "class a"
    assert enriched["building_area_sqft"] == 2480.0

    provenance = enriched["attribute_provenance"]
    assert provenance["year_built"] == PROVENANCE_USER_PROVIDED
    assert provenance["roof_material_public_record"] == PROVENANCE_USER_PROVIDED
    assert provenance["building_area_sqft"] == PROVENANCE_OBSERVED_PUBLIC_RECORD
    assert provenance["land_use_class"] == PROVENANCE_MISSING


def test_structure_enrichment_handles_missing_records_with_explicit_missing_provenance() -> None:
    public_fields = extract_structure_public_record_fields(parcel_properties=None, address_point_properties=None)
    enriched = enrich_structure_attributes(
        base_structure_attributes={},
        public_record_fields=public_fields,
        user_attributes=None,
    )
    provenance = enriched["attribute_provenance"]
    assert provenance["year_built"] == PROVENANCE_MISSING
    assert provenance["building_area_sqft"] == PROVENANCE_MISSING
    assert provenance["land_use_class"] == PROVENANCE_MISSING
    assert provenance["roof_material_public_record"] == PROVENANCE_MISSING


def test_structure_enrichment_does_not_promote_geometry_age_proxy_by_default() -> None:
    enriched = enrich_structure_attributes(
        base_structure_attributes={
            "estimated_age_proxy": {"proxy_year": 1988, "era_bucket": "1980_1999"},
        },
        public_record_fields={},
        user_attributes=None,
    )
    assert enriched["year_built"] is None
    assert enriched["attribute_provenance"]["year_built"] == PROVENANCE_MISSING


def test_structure_enrichment_can_opt_in_geometry_age_proxy_for_year_built() -> None:
    enriched = enrich_structure_attributes(
        base_structure_attributes={
            "estimated_age_proxy": {"proxy_year": 1988, "era_bucket": "1980_1999"},
        },
        public_record_fields={},
        user_attributes=None,
        allow_geometry_age_proxy=True,
    )
    assert enriched["year_built"] == 1988
    assert enriched["attribute_provenance"]["year_built"] == PROVENANCE_INFERRED_FROM_GEOMETRY
