# Region Onboarding Readiness

Prepared-region onboarding now includes an explicit property-specific readiness artifact in:

- `data/regions/<region_id>/manifest.json`
- `catalog.property_specific_readiness`

This artifact is used to communicate whether a region has the minimum parcel/footprint/context inputs needed for property-specific homeowner assessments.

## Readiness fields

`catalog.property_specific_readiness` includes:

- `parcel_ready`: parcel polygon coverage is available.
- `footprint_ready`: building footprint coverage is available.
- `parcel_footprint_linkage_quality`: coarse linkage quality (`high|moderate|low|unavailable`).
- `naip_ready`: NAIP structure-feature artifact is available.
- `structure_enrichment_ready`: public-record structure enrichment is available (or expected via configured parcel/address enrichment layers).
- `overall_readiness`: `property_specific | address_level | limited_regional`.

Backward-compatible fields remain:

- `readiness`: `property_specific_ready | address_level_only | limited_regional_ready`
- `signals`
- `missing_supporting_layers`

## How to run

1. Build or update a region:

```bash
python scripts/prepare_region_from_catalog_or_sources.py \
  --region-id <region_id> \
  --display-name "<name>" \
  --bbox <min_lon> <min_lat> <max_lon> <max_lat> \
  --validate
```

2. Validate and print readiness:

```bash
python scripts/validate_prepared_region.py --region-id <region_id>
```

3. Inspect manifest output:

- `catalog.property_specific_readiness`
- `catalog.validation_summary`
- `catalog.missing_reason_by_layer`

## Interpretation

- `overall_readiness=property_specific`: region has parcel + footprint + NAIP-structure support and can support higher-specificity homeowner behavior (subject to per-property data quality).
- `overall_readiness=address_level`: usable for address-level assessments; property-level differentiation is limited.
- `overall_readiness=limited_regional`: treat output as broader regional context unless additional data is prepared.
