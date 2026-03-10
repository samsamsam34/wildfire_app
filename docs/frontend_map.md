# Frontend Assessment Map

The frontend now includes a property-centric map panel that renders after a successful assessment.

## What is shown

- assessed property point
- subject building footprint (when available)
- defensible-space zones (`0-5 ft`, `5-30 ft`, `30-100 ft`, `100-300 ft`)
- nearby historic fire perimeters (when available)
- nearby structure footprints (optional context layer)

## API

Map data is provided by:

- `GET /report/{assessment_id}/map`

Response includes:

- map center (`latitude`, `longitude`)
- explicit geometry anchors (WGS84 GeoJSON):
  - `geocoded_address_point`
  - `property_anchor_point`
  - `matched_structure_centroid` (when a footprint match exists)
  - `matched_structure_footprint` (when a footprint match exists)
  - `display_point_source` (`matched_structure_centroid` for high-confidence matches, else `property_anchor_point`)
- geocode + structure-match diagnostics for map debugging:
  - `geocode_provider`, `geocoded_address`, `geocode_location_type`, `geocode_precision`
  - `property_anchor_source`, `property_anchor_precision`, `source_conflict_flag`
  - `structure_match_status`, `structure_match_method`, `structure_match_confidence`,
    `structure_match_distance_m`, `candidate_structure_count`
- layer definitions (display name, availability, default visibility, legend text)
- compact GeoJSON feature collections by layer key
- limitations/warnings when geometry or overlays are missing/partial

## Frontend behavior

- Leaflet renders a basemap and map layers.
- Layer toggles are generated from backend layer metadata.
- GeoJSON is rendered directly in `[longitude, latitude]` order until Leaflet consumes it.
- If structure geometry is unavailable, defensible-space rings fall back to point-proxy geometry.
- If overlays are missing, map still renders with property + available layers and shows limitation text.

## Notes

- Runtime still depends on prepared local region assets.
- No large GIS files are streamed directly to the browser.
- Overlays are clipped/filtered around the assessed property to keep payload size manageable.
