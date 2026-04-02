# Geometry Source Registry

`wildfire_app` now uses a configuration-driven geometry source registry for prepared regions.

Purpose:
- make parcel and footprint source precedence explicit
- keep source normalization rules and confidence weights versioned
- avoid ad hoc fallback ordering drift between regions

## Configuration

Default file:
- `config/geometry_source_registry.json`

Optional override:
- `WF_GEOMETRY_SOURCE_REGISTRY_PATH=/path/to/geometry_source_registry.json`

The registry supports:
- `source_order` by category (`parcel_sources`, `footprint_sources`)
- per-source definitions (`layer_keys`, fallback behavior, display metadata)
- schema normalization hints
- confidence weights by source
- region-specific overrides under `regions.<region_id>`

## Prepared-Region Manifest Output

Each prepared region manifest now includes:
- `geometry_source_manifest`
- `building_sources` (backward-compatible alias)
- `parcel_sources` (explicit parcel precedence list)

`geometry_source_manifest` contains:
- `region_id`
- `parcel_sources` (normalized entries with availability and selected layer key/file)
- `footprint_sources` (normalized entries with availability and selected layer key/file)
- `default_source_order`
- `schema_normalization_rules`
- `confidence_weights`
- `source_versions`
- `known_limitations`

## Notes

- This registry controls geometry-source precedence and diagnostics.
- It does **not** change wildfire scoring formulas.
- Nearest-parcel fallback remains available only with explicit confidence downgrade and cautionary diagnostics.
