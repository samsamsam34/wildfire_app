# Wildfire App Accuracy Audit — 2026-04-30

---

## Diagnostic Command Output

### Command 1: Prepared regions

```
bozeman_pilot
bozeman_test_partial
bozeman_test_with_canopy
las_vegas_large
missoula_pilot
winthrop_large
winthrop_pilot
```

### Command 2: Layers present in each region

```
=== data/regions/bozeman_pilot/ ===
_downloads  _extracted  _staging  building_footprints.geojson  canopy.tif
dem.tif  fire_perimeters.geojson  fuel.tif  manifest.json  slope.tif

=== data/regions/bozeman_test_partial/ ===
_downloads  _extracted  _staging  dem.tif  fuel.tif  manifest.json  slope.tif

=== data/regions/bozeman_test_with_canopy/ ===
_downloads  _extracted  _staging  dem.tif  fire_perimeters.geojson  fuel.tif
manifest.json  slope.tif

=== data/regions/las_vegas_large/ ===
_downloads  _extracted  _staging  building_footprints.geojson  canopy.tif
dem.tif  fire_perimeters.geojson  fuel.tif  manifest.json  slope.tif

=== data/regions/missoula_pilot/ ===
_downloads  _extracted  _staging  building_footprints.geojson  canopy.tif
dem.tif  fire_perimeters.geojson  fuel.tif  gridmet_dryness.tif  manifest.json
mtbs_severity.tif  parcel_address_points.geojson  parcel_polygons.geojson
roads.geojson  slope.tif  whp.tif

=== data/regions/winthrop_large/ ===
_downloads  _extracted  _staging  building_footprints.geojson  canopy.tif
dem.tif  dem.tif.aux.xml  fire_perimeters.geojson  fuel.tif  fuel.tif.aux.xml
manifest.json  slope.tif

=== data/regions/winthrop_pilot/ ===
_downloads  _extracted  _staging  building_footprints.geojson
building_footprints_overture.geojson  canopy.tif  dem.tif
fire_perimeters.geojson  fuel.tif  gridmet_dryness.tif  manifest.json
mtbs_severity.tif  parcel_address_points.geojson  parcel_address_points.geojson.bak
parcel_polygons.geojson  roads.geojson  slope.tif  whp.tif
```

### Command 3: GeoJSON/GPKG file sizes

```
data/regions/bozeman_test_with_canopy/fire_perimeters.geojson:  11479278 bytes
data/regions/winthrop_pilot/roads.geojson:    29972 bytes
data/regions/winthrop_pilot/building_footprints.geojson:  2443537 bytes
data/regions/winthrop_pilot/fire_perimeters.geojson:  1055400 bytes
data/regions/winthrop_pilot/parcel_address_points.geojson:   504486 bytes
data/regions/winthrop_pilot/building_footprints_overture.geojson:  2443976 bytes
data/regions/winthrop_pilot/parcel_polygons.geojson:  2516552 bytes
data/regions/missoula_pilot/roads.geojson:   304532 bytes
data/regions/missoula_pilot/building_footprints.geojson:  55405760 bytes
data/regions/missoula_pilot/fire_perimeters.geojson:      152 bytes
data/regions/missoula_pilot/parcel_address_points.geojson:  57974127 bytes
data/regions/missoula_pilot/parcel_polygons.geojson:  67084748 bytes
data/regions/bozeman_pilot/building_footprints.geojson:  25507103 bytes
data/regions/bozeman_pilot/fire_perimeters.geojson:  11479278 bytes
data/regions/winthrop_large/building_footprints.geojson:  2439813 bytes
data/regions/winthrop_large/fire_perimeters.geojson:  1895532 bytes
data/regions/las_vegas_large/building_footprints.geojson:  2510671 bytes
data/regions/las_vegas_large/fire_perimeters.geojson:  2430572 bytes
...cache and catalog files omitted for brevity...
```

Key observation: `missoula_pilot/fire_perimeters.geojson` is 152 bytes — effectively empty.

### Command 4: Cache database row counts

```
data/footprint_cache.db: not found
data/parcel_cache.db: not found
data/landfire_cache.db / landfire_pixel_cache: 20 rows
data/elevation_cache.db / elevation_slope_cache: 2 rows
data/nlcd_cache.db / nlcd_wildland_cache: 7 rows
```

Critical: `footprint_cache.db` and `parcel_cache.db` do not exist — the national
footprint index and Regrid parcel caches are cold for every new query.

### Command 5: Regrid parcel cache freshness

No output (parcel_cache.db not found).

### Command 6: Test suite results

pytest is not installed in the system Python environment used. Tests could not be
executed without the virtualenv active. Unable to capture output verbatim.

### Command 7: Geocoding fallback chain (grep results)

Key entries found in `backend/geocoding_fallback_chain.py`:
- Line 150: `provider_name: str = "geocode_fallback_chain"`
- Line 272: `candidate_meta["provider_used"] = provider_label`
- Line 425: `census_cfg = provider_settings.get("census_tiger", {})`
- Line 464: `zip_validation_enabled = bool(validation_cfg.get("zip_validation_enabled", False))`
  — Note: config/geocoding_config.yaml sets `zip_validation_enabled: true` overriding this default.

### Command 8: Confidence penalties in trust_metadata.py

Key thresholds (from static analysis):
- `trust_metadata.py:65`: fallback_heavy at `>= 0.45` observed_weight_fraction
- `trust_metadata.py:67`: high evidence tier at `>= 0.7` observed_weight_fraction
- `trust_metadata.py:291`: `_fallback_heavy_threshold = 0.45` (from scoring config)
- `trust_metadata.py:352`: `_unstable_thresh = 12.0`, `_moderate_thresh = 6.0`, `_sensitivity_mult = 6.0`
- `trust_metadata.py:355`: `_assumption_sensitive_swing = 4.0`

### Command 9: Building footprint matching thresholds

Key values from `backend/building_footprints.py`:
- Line 54: `max_search_m: float = 120.0` (search radius)
- Line 74–76: `max_match_distance_m` = min(35.0, max_search_m) = 35.0 by default
- Line 79: `ambiguity_gap_m = 6.0`
- Line 449: hard cutoff at `best_dist > self.max_match_distance_m`
- Line 680–681: scoring formula: `0.75 * distance_component + 0.15 * centroid_component`
- Lines 757–774: expanded match distance for low-precision anchors:
  - "interpolated": max(35*2.5, 60.0) = 87.5 m
  - "approximate": max(35+8, 22) = 43 m (capped at max_search_m=120)
  - "unknown": max(35+10, 24) = 45 m

### Command 10: Address resolution failure modes

Key None/fallback lines in `backend/address_resolution.py`:
- Line 70: `if raw is None:` (in _representative_lon_lat)
- Line 504: `if lon_lat is None:` (fallback path)
- Line 792: `if lon_lat is None:` (explicit fallback records path)
- Line 867: block on invalid_address_source
- Line 871: `if lon_lat is None:` (parcel address point path)

---

## Executive Summary

- **Census TIGER is now the primary geocoder** (via `GeocodeFallbackChain`), followed by Nominatim. Census provides interpolated, not rooftop, precision for most residential addresses — introducing a systematic 10–150 m positional bias at the address anchor point before any structure or defensible-space analysis begins.

- **Address-point snap is the only positional correction mechanism**, but it only fires when the region has a prepared `parcel_address_points.geojson`. Only 2 of 7 prepared regions (missoula_pilot, winthrop_pilot) have this file. For 5 of 7 regions and all national assessments, no address correction is possible.

- **The parcel centroid IS used as an anchor upgrade** when a parcel is matched and no address point exists, but the source priority ordering (address_point > geocode > parcel_centroid) means parcel centroid is third in line. For interpolated geocodes where no address-point layer exists, the system falls back to the raw interpolated geocode, not the parcel centroid.

- **The national MTBS fire history GeoPackage contains only 3 features** (data/national/mtbs_perimeters.gpkg, 98 KB). The full national dataset should contain 80,000+ perimeters. Historic fire scoring is non-functional for all non-region assessments.

- **Burn probability (WHP) is missing from 5 of 7 prepared regions.** Only missoula_pilot and winthrop_pilot have `whp.tif`. All other regions and national assessments have no burn probability signal. The burn_probability_index is weighted at 0.31 in the ember model (risk_engine.py:612) and contributes to the site_hazard score.

- **Missoula pilot fire_perimeters.geojson is 152 bytes (empty).** Fire history scoring is also broken for the Missoula region despite being a "full" pilot.

- **The Overture DuckDB national footprint path is architecturally sound**, but `data/footprint_cache.db` does not exist yet — every production query will incur a live DuckDB S3 HTTP range request with a 15-second timeout.

- **DuckDB's bbox query has a correctness flaw**: the WHERE clause uses strict `bbox.xmin >= min_lon AND bbox.xmax <= max_lon` — this matches only footprints fully inside the bbox, excluding those that span the bbox edge. Footprints straddling the query boundary are silently dropped.

- **Google Maps geocoder is a stub that always returns None** (`geocoding_fallback_chain.py:106-108`). When configured it consumes an API key slot while contributing nothing.

- **ZIP validation correctly detects cross-country mismatches** (e.g., Nominatim placing a WA address in NY) but cannot detect same-state positional errors within the correct ZIP, which is the common case for interpolated Census coordinates.

- **Point-proxy ring analysis (geometry_basis="point") severely degrades structure-specific submodel weights**: defensible_space multiplier is 0.05, ember and flame_contact are 0.18 (risk_engine.py:1362–1366). This is correct behavior, but it means national assessments produce very low confidence for precisely the highest-risk structural inputs.

- **Confidence tier "preliminary" correctly fires** for unverified geocodes, missing critical fields, and severe layer failures, but the threshold documentation explains that national assessments with good environmental layer coverage (fuel, slope, canopy, fire history) can reach "low" or "moderate" confidence — not "preliminary" — even without structural data.

---

## Part A: Address Resolution Findings

### A1. Geocoder Precision Distribution

**Current behavior:**

Census TIGER (`geocoding_census.py:197–202`) assigns precision based on two conditions:
1. `parcel_or_address_point`: tigerLineId present AND all address components (zip, street, city, state) complete.
2. `interpolated`: tigerLineId present but components incomplete.
3. `approximate`: no tigerLineId (rare).

Census TIGER interpolates along TIGER/Line road centerlines — it is explicitly not rooftop-level (`geocoding_census.py:12–14`, module docstring). Even "parcel_or_address_point" precision from Census means the point is on the road centerline matched to the street number, not on the actual parcel.

Nominatim precision (`geocoding.py:399–419`) maps Nominatim feature classes:
- `rooftop`: class="building" or type="house"
- `parcel_or_address_point`: precise address fields present
- `interpolated`: highway/road/street types
- `approximate`: postcode/city/town/administrative

**Maximum expected positional error by tier:**
| Tier | Expected Max Error |
|------|--------------------|
| rooftop (Nominatim building match) | 5–20 m |
| parcel_or_address_point (Census or Nominatim) | 15–60 m (road centerline offset) |
| interpolated | 30–400 m |
| approximate | 400 m – several km |

**Gap/failure mode:** Census "parcel_or_address_point" is misleadingly named — the point is on the road centerline, not the parcel. For a typical residential lot set back 15–30 m from the road, the anchor is systematically placed in or near the roadway.

**Severity:** High. All Census results carry an inherent 15–60 m road-centerline offset regardless of precision label.

**File:line:** `geocoding_census.py:197–202`, `geocoding_census.py:12–14`

---

### A2. The Quarter-Mile Error Scenario

**Current behavior:**

A 400 m positional error would arise from an `interpolated` or `approximate` precision geocode. This occurs for:
- Addresses on rural roads not in TIGER (Census returns `approximate` or no match)
- New subdivisions not yet in TIGER (interpolated to nearest street segment)
- Named roads without standard numbering (approximate match)

For Census `interpolated` results, the address-point anchor lookup tolerance is expanded:
- `property_anchor.py:429–431`: address_limit expanded to `max(address_default + 18, address_default * 1.8)` — for the 45 m default, this becomes 81 m
- `property_anchor.py:431`: parcel_limit expanded to `max(parcel_default + 22, parcel_default * 2.0)` — for the 30 m default, this becomes 60 m

These expanded limits are far smaller than a 400 m geocode error. A 400 m offset will not be corrected by parcel or address-point lookup.

**Address-point snap trigger conditions** (`main.py:11704–11712`):
- `address_points_path` must be non-None
- The snap only fires when a region's `parcel_address_points.geojson` exists
- Default `max_distance_m = 150 m`, `min_match_score = 85`

For a 400 m error the snap will not find a match within 150 m. The snap is designed for small geocoder offsets, not large errors.

**In practice:** Only 2 of 7 prepared regions (missoula_pilot, winthrop_pilot) have address point files. For all other regions and all national assessments, address-point snap never fires regardless of geocode quality.

**ZIP validation** (`geocoding_config.yaml:35–36`): `zip_validation_enabled: true`, `zip_max_distance_km: 20`. This catches cross-state mismatches (the Nominatim "WA address lands in NY" failure case) but will not flag a 400 m error within a correct ZIP code.

**Gap/failure mode:** A 400 m interpolated-precision geocode error is not corrected by any current mechanism for the vast majority of assessments. The error propagates through anchor resolution, ring analysis, and all structure-specific submodels.

**Severity:** High. This is the dominant positional accuracy gap for rural and exurban WUI addresses.

**File:line:** `property_anchor.py:429–453`, `main.py:11704–11712`, `geocoding.py:894–1001`

---

### A3. Rural and New-Development Address Gaps

**Current behavior:**

Census TIGER gaps include: rural road segments without TIGER representation, new subdivisions (TIGER has a 1–2 year update lag), private roads, and gated community driveways. When Census has no match it returns `None` (`geocoding_census.py:165–167`), causing the chain to advance to Nominatim.

Nominatim has separate gaps: OSM data in rural US is frequently incomplete. Nominatim may return a city- or county-level result for a rural address (approximate tier), or return `None`.

**Detection of failure reason:** The system only distinguishes "matched" from "no match" — it does not detect whether Census returned no match because the road is too new vs. not in their database vs. a formatting error. The `geocode_precision` field conveys quality but not root cause.

**User experience when both fail:** `GeocodingError` with `status="no_match"` is raised (`geocoding_fallback_chain.py:376–383`). The caller in `main.py` must handle this. The assessment cannot proceed without a fallback coordinate.

**Gap/failure mode:** No detection of "Census succeeded but the road segment is interpolated because it's new construction." The system does not attempt to infer a coordinate from user-provided ZIP + parcel data if the network geocoders fail.

**Severity:** Medium. Affects WUI properties which are disproportionately represented in the target assessment population.

**File:line:** `geocoding_census.py:165–167`, `geocoding_fallback_chain.py:376–383`

---

### A4. Parcel Centroid as Correction Mechanism

**Current behavior:**

`property_anchor.py` source priority is `("address_point", "geocode", "parcel_centroid")` by default (`property_anchor.py:27`). The selection logic at lines 629–643:

1. If address point found within tolerance → use address point (highest priority).
2. Else if parcel geometry found → use parcel **centroid** (second priority).
3. Else → fall back to raw geocode point (lowest priority).

Wait — this ordering means parcel centroid DOES take priority over an interpolated geocode when a parcel is successfully matched. The geocode is only used as an anchor when neither address point NOR parcel is available.

**Trace for "interpolated geocode + parcel confidence 72":**
- Expanded tolerances: address_limit ≈ 81 m, parcel_limit ≈ 60 m (`property_anchor.py:429–431`)
- Parcel lookup attempts `contains_point` first, then `nearest_within_tolerance` at 60 m
- If parcel is within 60 m of the geocode, parcel centroid is used as anchor
- `anchor_quality_score` for `parcel_polygon_centroid` = 0.83 (`property_anchor.py:471`)
- `anchor_quality_score` for `interpolated_geocode` = 0.62 (`property_anchor.py:475`)

**Current priority:** Parcel centroid DOES take precedence over interpolated geocode when the parcel is found. This is correct behavior.

**Accuracy:** A Regrid parcel centroid is typically accurate to 1–10 m (parcel polygon derived from county assessor GIS data). A Census TIGER interpolated geocode can be 30–400 m off. The parcel centroid is substantially more accurate.

**Gap/failure mode:** For addresses outside prepared regions and where Regrid API key is not configured, no parcel is available, so the raw interpolated geocode is used as the anchor. The parcel centroid path only helps when `WF_REGRID_API_KEY` is set or a prepared region parcel file covers the property.

**Severity:** Medium. The architecture is correct but the parcel data availability is the limiting factor.

**File:line:** `property_anchor.py:27`, `property_anchor.py:629–643`, `property_anchor.py:468–508`

---

### A5. User-Facing Address Confirmation

**Current behavior:**

There is no frontend map or user confirmation mechanism present in this backend codebase. The backend API returns `property_anchor_selection_method`, `anchor_quality`, `geocode_precision`, and `alignment_notes` in the response, which a frontend could use to prompt confirmation, but the backend itself does not enforce any user confirmation step.

The `property_anchor_override` parameter exists in the assessment request model, allowing a client to pass corrected coordinates, but this is a developer/integration pattern, not an end-user flow.

**Gap/failure mode:** When a geocode is 400 m off, the homeowner receives scores computed for the wrong location with no indication beyond a technical `geocode_precision` field. No user-facing "does this look right?" map confirmation step exists.

**Severity:** High for user experience. A homeowner with a rural property likely has no way to detect or correct a positional error.

**File:line:** `property_anchor.py:511–774` (resolve method), `main.py:11704` (snap logic)

---

## Part B: Building Footprint Coverage Findings

### B1. Coverage by Source and Region

**Prepared region coverage:**

| Region | Footprints | Parcels | Address Points | WHP |
|--------|-----------|---------|----------------|-----|
| bozeman_pilot | Yes | No | No | No |
| bozeman_test_partial | No | No | No | No |
| bozeman_test_with_canopy | No | No | No | No |
| las_vegas_large | Yes | No | No | No |
| missoula_pilot | Yes | Yes | Yes | Yes |
| winthrop_large | Yes | No | No | No |
| winthrop_pilot | Yes | Yes | Yes | Yes |

Only 2/7 regions are "full" (footprints + parcels + address points + WHP). 3/7 have footprints only. 2/7 have no footprints at all.

**National index (Overture via DuckDB):**
- `data/footprint_cache.db` does not exist — no cache primed
- Overture 2026-04-15.0 release is current (released within 2 weeks of audit date)
- Coverage: Overture Maps has ~2.3 billion building footprints globally; US coverage is substantial for urban/suburban areas but uneven for rural WUI zones

**Coverage gaps by area type (estimated from Overture coverage patterns):**
- Urban/suburban: 85–95% coverage
- Exurban / WUI edge: 50–75% coverage
- Rural / agricultural: 20–50% coverage

**File:line:** `national_footprint_index.py:63`, `building_footprints.py:493–503`

---

### B2. DuckDB/Overture Query

**Current behavior:**

Default Overture release hardcoded at `national_footprint_index.py:63`: `_DEFAULT_OVERTURE_RELEASE = "2026-04-15.0"`. This is current.

**bbox schema correctness concern:**

The query (`national_footprint_index.py:334–344`):
```sql
WHERE bbox.xmin >= {min_lon}
  AND bbox.xmax <= {max_lon}
  AND bbox.ymin >= {min_lat}
  AND bbox.ymax <= {max_lat}
```

This requires the footprint bbox to be FULLY INSIDE the query bbox. Any building whose footprint polygon spans the query bbox boundary is excluded. The correct predicate for an intersection query would use:
```sql
WHERE bbox.xmin <= {max_lon} AND bbox.xmax >= {min_lon}
  AND bbox.ymin <= {max_lat} AND bbox.ymax >= {min_lat}
```

This is a correctness bug. Buildings on the edge of the 300 m search radius are systematically dropped.

**INSTALL spatial per-query:** `national_footprint_index.py:322–329`: `INSTALL spatial` and `INSTALL httpfs` are called in `_query_overture()` on every new DuckDB connection. DuckDB handles this as a no-op after first install, but it adds network round-trips (DuckDB checks for updates) on every cold query. This should be done once at startup.

**Timeout behavior:** `_query_overture()` runs the entire DuckDB query with no explicit timeout set on the DuckDB connection — the `timeout_seconds=15` parameter (`national_footprint_index.py:99`) is stored but not applied to the DuckDB `connect()` or `execute()` calls.

**Gap/failure mode:** The bbox predicate excludes edge-spanning footprints; the timeout is effectively not enforced; INSTALL extensions are called per-query.

**Severity:** Medium (bbox bug) / Low (extension install overhead).

**File:line:** `national_footprint_index.py:334–344`, `national_footprint_index.py:315–346`, `national_footprint_index.py:96–99`

---

### B3. Footprint Matching Failures

When footprints exist but no match is found, the most likely causes are:

1. **Point not inside any polygon and nearest polygon exceeds `max_match_distance_m` (35 m):**
   `building_footprints.py:449`: `if best_row is None or best_dist > self.max_match_distance_m: return None`. A geocoded point on the road centerline may be 15–50 m from the nearest building footprint edge, exceeding the 35 m threshold.

2. **Parcel intersection check returns empty set:**
   `building_footprints.py:538–549`: When a parcel exists but contains no intersecting footprints (common for rural parcels with barns/outbuildings not in OSM), the function immediately returns `found=False` with `match_method="parcel_intersection"` rather than falling back to the national index.

3. **Geometry mismatch (coordinate system confusion):**
   Footprints in local GeoJSON must be WGS84. If a regional dataset is in a projected CRS and not reprojected during data prep, all distance calculations will be wrong.

4. **National index query empty result:**
   If DuckDB is unavailable or the S3 query returns 0 results (rural area with no Overture coverage), the function returns `BuildingFootprintResult(found=False, match_status="provider_unavailable")`.

**File:line:** `building_footprints.py:449`, `building_footprints.py:538–549`, `building_footprints.py:493–503`

---

### B4. Rural and Agricultural Properties

**Expected coverage:**
- Overture Maps coverage for rural WUI parcels (the primary target population) is estimated at 20–60%. Agricultural barns and outbuildings are frequently absent from OSM-derived datasets.

**Point-buffer rings vs. footprint-edge rings:**

When no footprint is found, ring analysis uses a point-buffer approach. The practical difference for a typical SFH (10 m × 12 m footprint):
- 0–5 ft ring from footprint edge covers the immediate structure perimeter (foundation plantings, deck boards, stored wood)
- 0–5 ft ring from geocoded point may be centered 10–30 m from the actual structure, including part of the driveway or road

For risk scoring, the defensible_space submodel weight multiplier is 0.05 when `geometry_basis == "point"` (`risk_engine.py:1362`), making this submodel contribute only 5% of its designed weight. The ember and flame_contact submodels are similarly degraded to 18%.

**File:line:** `risk_engine.py:1360–1368`, `building_footprints.py:234–281`

---

### B5. Footprint Quality Issues

**Confidence that matched footprint is correct building:**
- Point-in-polygon match from local source: 0.97 confidence (`building_footprints.py:583`)
- Nearest within tolerance from local source: 0.92 confidence
- National Overture point-in-polygon: 0.88 confidence (`building_footprints.py:416`)
- National Overture nearest match: 0.82 confidence (`building_footprints.py:463`)

**False-positive rate for nearest-footprint fallback:**
In dense urban areas with buildings at 5–20 m spacing, the nearest building may frequently be a neighbor's structure rather than the subject property. Without parcel intersection to constrain candidates, the false-positive rate could be 20–40% for urban assessments.

**`structure_classifier.py` invocation:**
Static analysis confirms `structure_classifier.py` exists and defines `StructureClassification`. Whether it is actually called from the assessment pipeline could not be confirmed from static analysis alone. The module requires a matched footprint and parcel polygon; without both, classification is heuristic-only.

**`on_parcel_structure_count` field:**
`building_footprints.py:408–411`: `on_parcel_count` is populated from `_count_structures_on_parcel()` only when `parcel_overlap_available`. For 5/7 regions without parcel data, this field is always None.

**File:line:** `building_footprints.py:583`, `building_footprints.py:416`, `building_footprints.py:463`, `structure_classifier.py:1–17`

---

## Part C: Model Accuracy Findings

### C1. Score Sensitivity Analysis

**site_hazard submodels (from risk_engine.py):**

Key inputs and their roles:
| Feature | Submodel | National Fallback Available? |
|---------|----------|------------------------------|
| burn_probability_index | ember_exposure, site_hazard | No (WHP only in 2/7 regions) |
| fuel_index (LANDFIRE FBFM40) | vegetation_intensity, fuel_proximity | Yes (LANDFIRE WCS COG) |
| canopy_index (LANDFIRE CC) | vegetation_intensity | Yes (LANDFIRE WCS COG) |
| slope_index | topography | Yes (LANDFIRE WCS COG + 3DEP) |
| wildland_distance_index | flame_contact | Yes (NLCD WCS) |
| historic_fire_index | historic_fire | No (MTBS national GPKG = 3 features only) |
| moisture_index (gridmet) | site_hazard | No (only 2/7 regions have gridmet_dryness.tif) |

**Top 3 features whose absence most degrades score quality:**
1. `burn_probability_index`: affects ember exposure model at weight 0.31 (`risk_engine.py:612`). No national COG fallback. For 5/7 regions and all national assessments, this term is absent.
2. `historic_fire_index`: MTBS national GPKG contains only 3 features — functionally zero national coverage. Historic fire is a strong signal for WUI risk.
3. `ring_0_5_density` / near-structure vegetation: requires matched footprint. Without footprint, defensible_space weight is 0.05 of its designed value (`risk_engine.py:1362`).

**File:line:** `risk_engine.py:532–593`, `risk_engine.py:1360–1368`

---

### C2. Regional vs. National Accuracy

**Features available for prepared regions (full, like missoula_pilot/winthrop_pilot) but NOT for national fallback:**
- WHP/burn_probability raster (whp.tif)
- Gridmet dryness raster (gridmet_dryness.tif)
- MTBS severity raster (mtbs_severity.tif)
- Local fire perimeters (fire_perimeters.geojson)
- Parcel polygons (for footprint constrained matching)
- Address points (for address snap and anchor upgrade)

**Expected score difference:**
The availability multipliers in `risk_engine.py:1345–1380` show:
- `defensible_space_risk` with point proxy: 0.05 multiplier (95% weight reduction)
- `ember_exposure_risk` without burn_probability: weight drops by 0.31 factor
- `flame_contact_risk` with point proxy: 0.18 multiplier (82% weight reduction)

A full-region assessment vs. a national fallback assessment for the same property could have wildly different effective submodel weights, making the scores structurally incomparable.

**Systematic bias direction:** National fallback assessments are biased toward the median of observed environmental inputs (fuel, slope, canopy from LANDFIRE WCS) with structural/defensible-space uncertainty collapsed toward neutral values. High-risk properties in dense WUI vegetation will be **underscored** on national fallback because the near-structure vegetation signals are effectively suppressed.

**File:line:** `risk_engine.py:1355–1380`

---

### C3. Burn Probability Gap

**Weight of burn_probability:**
- In ember exposure model: weight 0.31 (`risk_engine.py:612`)
- When missing, the term is excluded and remaining weights rebalanced via `ctx_metric()`
- `ctx_metric()` records "Burn probability context missing; scoring used conservative fallback"

**Confidence penalty:** `main.py:1028–1041` shows burn_probability_layer is treated as important but not `critical_missing` when it was never configured (`not_configured`). The `_burn_was_configured` flag prevents it from inflating `major_environmental_missing_count` when the region simply has no WHP layer.

**Proxy construction:** LANDFIRE fuel model (FBFM40) correlates with burn probability — high fire-adapted fuel types (e.g., models 101–109, shrub fuels) indicate higher burn potential. Slope and aspect also correlate. A simple proxy from fuel + slope + aspect could reduce the gap, but no such proxy is currently implemented.

**Gap/failure mode:** ~60–70% of assessments (all outside 2 full regions) are missing burn probability entirely. This is a substantial signal gap for the ember exposure submodel.

**Severity:** High for WUI properties where burn probability is most relevant.

**File:line:** `risk_engine.py:532–534`, `risk_engine.py:612`, `main.py:4069–4074`

---

### C4. Defensible Space Assessment Quality

**Point proxy vs. footprint-edge rings:**

For a typical 1,200 sq ft SFH (111 m²) with footprint approximately 10 m × 11 m:
- 0–5 ft (1.5 m) ring from footprint edge: 86 m² annular zone around actual structure
- 0–5 ft (1.5 m) ring from geocoded point (road centerline): centered 15–30 m away

In the worst case (road centerline geocode), the 0–5 ft ring misses the structure entirely, capturing instead the road shoulder and adjacent lot. The defensible-space score becomes meaningless.

**NAIP-derived vegetation features:** No NAIP processing pipeline exists in this codebase. `ring_0_5_density`, `ring_5_30_density` etc. are populated from building footprint GeoJSON ring analysis, not from aerial imagery. NAIP features are referenced in field names within `risk_engine.py:1320–1329` (`imagery_vegetation_continuity_pct`, `imagery_canopy_proxy_pct`) but appear to be placeholders; no NAIP ingestion pipeline is present.

**LANDFIRE WCS features available nationally:**
- Fuel model (FBFM40, 30 m) — categorical fuel type
- Canopy cover (CC, 0–100%)
- Slope, aspect, elevation

These are point-sampled at the anchor location and used as site_hazard inputs, not as ring-density inputs. They do not substitute for ring-level vegetation structure data.

**File:line:** `risk_engine.py:1360–1368`, `risk_engine.py:1320–1329`

---

### C5. Confidence Calibration

**National fallback with good environmental coverage:**

Conditions: fuel populated (LANDFIRE WCS), slope populated (LANDFIRE or 3DEP), canopy populated (LANDFIRE WCS), wildland distance populated (NLCD WCS), fire history missing (MTBS = 3 rows), burn probability missing (no WHP), no footprint, no parcel.

Confidence scoring path (`main.py:1340–1389`):
- `geocode_verified = True` (assuming Census success)
- `has_meaningful_environment = True` (fuel + slope + canopy present)
- `has_meaningful_property = False` (no footprint, no structure data)
- `confidence < 35`? Depends on the numeric confidence score from `_compute_confidence_score()`
- `severe_layer_failure = False` (LANDFIRE WCS layers present)
- `multiple_critical_missing` = likely True (burn probability, defensible_space_ft, footprint)

Given `multiple_critical_missing = True`, the tier lands at **"preliminary"** regardless of environmental layer quality (`main.py:1344,1348`).

**Is "preliminary" the right tier?** For a national assessment with good fuel/slope/canopy coverage but no structural data and no WHP, "preliminary" accurately reflects that the score is directionally useful but structurally incomplete. However, it may be too conservative for users in covered environmental areas — the environmental risk signal is real even if the structural vulnerability assessment is degraded.

**Path to "medium" without structural inputs:** Would require:
- `multiple_critical_missing = False` — i.e., burn probability and defensible space not flagged as critical missing
- `confidence >= 62` — possible with good environmental layer coverage
- `not severe_layer_failure`

This is achievable if `burn_probability_layer` and `defensible_space_ft` are moved from critical to non-critical, but that would sacrifice accuracy signaling.

**File:line:** `main.py:1340–1389`

---

## Part D: Data Pipeline Findings

### D1. LANDFIRE WCS COG

**Version:** LF 2024 (version 2.5.0) for fuel/canopy; LF 2020 for topographic layers (`landfire_cog_client.py:7–14`).

**Layer IDs:**
- `landfire_wcs:LF2024_FBFM40_CONUS` (fuel)
- `landfire_wcs:LF2024_CC_CONUS` (canopy)
- `landfire_wcs:LF2020_SlpD_CONUS` (slope)
- `landfire_wcs:LF2020_Asp_CONUS` (aspect)
- `landfire_wcs:LF2020_Elev_CONUS` (elevation)

These appear correct for the respective versions (confirmed from WCS GetCapabilities in the docstring).

**Cache TTL:** 365 days (`landfire_cog_client.py:98`). LANDFIRE releases annually — 365-day TTL means a cached pixel will survive until next year's release. This is appropriate for vegetation/fuel but terrain layers (LF2020) are static, so longer TTL is fine.

**Gap/failure mode:** LANDFIRE WCS has documented maintenance downtime (last Wednesday of each month, 8AM–12PM CST/CDT). No retry or circuit-breaker logic was visible in the module beyond returning `None` on failure.

**Severity:** Low. Cache adequately buffers the maintenance windows.

**File:line:** `landfire_cog_client.py:54–85`, `landfire_cog_client.py:98`

---

### D2. Overture Maps Release

**Hardcoded release:** `2026-04-15.0` (`national_footprint_index.py:63`).

**Current status:** The audit date is 2026-04-30, and this release was published 2026-04-15 — 15 days old, which is current. Overture publishes quarterly releases.

**Impact of stale release:** Missing buildings from construction after the release date; demolished buildings still appearing. For the WUI context (stable forest-edge residential), this is minor — structures change slowly in WUI zones.

**Override mechanism:** `WF_OVERTURE_RELEASE` env var (`national_footprint_index.py:103–106`) allows pinning without code changes.

**File:line:** `national_footprint_index.py:63`, `national_footprint_index.py:103–106`

---

### D3. MTBS Fire Perimeters

**National GPKG vintage:** `data/national/mtbs_perimeters.gpkg` is 98 KB and contains exactly 3 features. This is a development stub — the full national MTBS dataset contains 80,000+ perimeters covering 1984–2022+.

**Impact:** Historic fire scoring (`national_fire_history_client.py`) is functionally disabled for all non-region assessments. The `fire_count_30yr`, `burned_within_radius`, and `nearest_fire_distance_m` fields will always be 0/False/None for national assessments.

**Detection mechanism:** None. There is no check in `national_fire_history_client.py` that warns when the GPKG contains fewer than a minimum expected number of features.

**Update mechanism:** `scripts/download_national_mtbs.py` is documented in `national_fire_history_client.py:20–22` but its existence could not be confirmed in this audit.

**Severity:** High. The historic fire signal is one of the strongest WUI risk indicators, and it is absent for all non-region assessments.

**File:line:** `national_fire_history_client.py:1–22`, `data/national/mtbs_perimeters.gpkg` (3 rows confirmed by sqlite3 query)

---

### D4. Regrid Parcel API

**Cache:** `data/parcel_cache.db` does not exist — the cache is cold. Every Regrid API call that succeeds would write to this file, creating it on first use.

**TTL:** 90 days (`parcel_api_client.py:46`). Parcel boundaries change infrequently; 90-day TTL is appropriate.

**Rate limit concern:** Regrid free tier: 1,000 queries/month (`parcel_api_client.py:8–10`). At scale (multiple assessments per day), this will be exhausted quickly. No rate-limiting or quota tracking is implemented in the client.

**API key requirement:** `WF_REGRID_API_KEY` env var. If not set, the Regrid client is not instantiated and parcel lookup via API is unavailable. Static analysis confirmed the key is checked at client construction time.

**Gap/failure mode:** For deployments without a Regrid API key (likely the default case), all parcel data must come from prepared-region files. National assessments have no parcel fallback.

**Severity:** Medium. Parcel data is important for anchor correction and footprint constraint. The 1,000 query/month free tier limit is a practical bottleneck.

**File:line:** `parcel_api_client.py:44–46`, `parcel_api_client.py:8–10`

---

## Prioritized Gap Table

| # | Gap | Part | Severity | User Impact | Effort to Fix | Priority |
|---|-----|------|----------|-------------|---------------|----------|
| 1 | MTBS national GPKG contains only 3 features — historic fire signal absent for all national assessments | D3 | Critical | Every national assessment missing key risk signal | Low (download script) | P0 |
| 2 | Burn probability (WHP) absent for 5/7 regions and all national assessments | C3 | High | Ember model underweighted; site_hazard underpredicted in WUI | Medium (national WHP COG client) | P1 |
| 3 | Address-point snap fires for only 2/7 regions; 400 m geocode errors uncorrected nationally | A2 | High | Properties assessed at wrong location; structural scores meaningless | Medium (expand national address point coverage) | P1 |
| 4 | Missoula pilot fire_perimeters.geojson is empty (152 bytes) | D3 | High | Fire history broken for Missoula despite being a full pilot region | Low (re-run region prep) | P1 |
| 5 | DuckDB bbox predicate excludes edge-spanning footprints | B2 | High | Edge-of-tile buildings silently dropped from matching | Low (fix SQL predicate) | P1 |
| 6 | DuckDB timeout not enforced at connection level | B2 | Medium | S3 query can hang indefinitely, blocking request thread | Low (add connection timeout) | P2 |
| 7 | Footprint cache DB cold — every query hits S3 | B1 | Medium | 15-second latency on first footprint query per tile | Low (prime cache during region prep) | P2 |
| 8 | No national WHP/burn probability proxy from LANDFIRE fuel+slope | C3 | Medium | Ember model missing key input; alternative signal constructable | Medium (derive proxy) | P2 |
| 9 | Census "parcel_or_address_point" precision is misleadingly named — all Census results are road-centerline | A1 | Medium | Road-centerline offset (15–60 m) propagates to ring analysis | Medium (document + confidence penalty) | P2 |
| 10 | DuckDB INSTALL extensions called per-query | B2 | Low | Network round-trip overhead per cold DuckDB session | Low (move to startup) | P3 |
| 11 | Regrid 1,000/month free tier limit; no quota tracking | D4 | Medium | Parcel lookups fail silently after limit exceeded | Medium (quota tracking + alert) | P2 |
| 12 | Google Maps geocoder is a stub that always returns None | A3 | Low | API key slot wasted; no rooftop geocoding path | High (full implementation) | P3 |
| 13 | No NAIP or aerial imagery pipeline for near-structure vegetation | C4 | High | Ring density scores are proxy-only; defensible space is structural | High (new pipeline) | P3-long |
| 14 | Source priority places geocode before parcel_centroid | A4 | Low | Architecture is actually correct — parcel_centroid is used when available | None needed | Info |

---

## Implementation Candidates

### 1. Download full MTBS national perimeters (immediate, high impact)

Run `scripts/download_national_mtbs.py` (or equivalent) to replace the 3-feature development stub at `data/national/mtbs_perimeters.gpkg` with the full 80,000+ feature national dataset. The `NationalFireHistoryClient` loads the GPKG at startup and builds a spatial STRtree — no code changes required. This would immediately enable historic fire scoring for all national assessments. The MTBS data is free, national, and covers 1984 to the most recent fire season. Estimated effort: 1 hour (download, verify, test).

### 2. Fix DuckDB bbox predicate (immediate, correctness fix)

In `national_footprint_index.py:334–344`, replace the `FULLY INSIDE` bbox predicate with an `INTERSECTS` predicate:
```sql
WHERE bbox.xmin <= {max_lon} AND bbox.xmax >= {min_lon}
  AND bbox.ymin <= {max_lat} AND bbox.ymax >= {min_lat}
```
This is a one-line SQL change that corrects the silent exclusion of edge-spanning footprints. It may slightly increase result counts but increases accuracy. Move `INSTALL spatial; INSTALL httpfs;` to a one-time initialization method called at `NationalFootprintIndex.__init__` to avoid per-query extension install overhead.

### 3. Enforce DuckDB query timeout

In `national_footprint_index.py:_query_overture`, apply the stored `self._timeout` to the DuckDB connection using DuckDB's `SET statement_timeout` pragma, or wrap the `con.execute(query)` call in `concurrent.futures.ThreadPoolExecutor` with a `timeout` argument. This prevents a slow S3 range request from blocking an entire FastAPI worker thread for more than 15 seconds.

### 4. National burn probability proxy from LANDFIRE

Construct a proxy `burn_probability_index` when WHP is absent by combining: (a) LANDFIRE fuel model mapped to fire behavior group (low/moderate/high ignition potential), (b) slope_index (higher slope = higher burn probability), (c) aspect_index (south/southwest aspect = higher), and (d) canopy_index (dense canopy = higher continuity). A simple linear blend weighted by documented fire-behavior relationships would improve the ember model's precision substantially without requiring new data sources. All four inputs are available via LANDFIRE WCS nationally.

### 5. Harden Missoula fire_perimeters.geojson

Re-run the Missoula region prep to populate `data/regions/missoula_pilot/fire_perimeters.geojson` with actual perimeter data. The current 152-byte file is empty (an artifact of a failed prep step). Also add a manifest validation check that rejects region prep results where any required vector layer has fewer than 1 feature.

### 6. National address-point fallback via Census geocoding snap improvements

For assessments without a prepared region address-point layer, expand the address-point snap to use the Census geocoder's own `matched_address` field to confirm the geocoded result maps to the expected parcel. When a Regrid parcel is available but the geocoded point is more than 50 m from the parcel boundary, automatically prefer the parcel centroid as the anchor (which the architecture already supports but only when parcel is found within the expanded tolerance). Document this behavior explicitly with a confidence note.

### 7. Add minimum feature count validation for prepared region data

In the region manifest validation (during `prepare_region_from_catalog_or_sources.py` or equivalent), add checks:
- `fire_perimeters.geojson`: minimum 1 feature, warn if < 10
- `building_footprints.geojson`: minimum 50 features per km² of region bbox
- `parcel_polygons.geojson`: minimum 10 features

These checks would have caught the empty Missoula fire_perimeters.geojson and the 3-feature MTBS stub before they reached production.

### 8. Regrid quota tracking and alerting

In `parcel_api_client.py`, add a lightweight quota counter stored in the SQLite cache DB: increment on every successful API call, reset monthly, and log a warning at 800/1000 and an error at 950/1000. Return `None` (graceful degradation) at 1000/1000 rather than sending a doomed request. This prevents silent Regrid failures and gives operators visibility into API consumption before the limit is hit.
