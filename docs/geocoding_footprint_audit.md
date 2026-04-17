# Geocoding, Parcel, and Building Footprint Audit

**Date:** 2026-04-16  
**Scope:** Full read-through of `geocoding.py`, `address_resolution.py`, `parcel_resolution.py`, `property_anchor.py`, `building_footprints.py`, config files, tests, and `main.py` integration points. No code was modified.

---

## 1. Executive Summary

**What works:**
- Nominatim-based geocoding with multi-variant query expansion, importance filtering, ambiguity delta, and house-number mismatch detection is more sophisticated than a naive single-query integration.
- Local-first address resolution falls back gracefully through a prioritized chain: manual overrides → prepared region address points → parcel centroids → network geocoder.
- Parcel resolution correctly implements point-containment first, then distance-proximity fallback, with confidence penalties at each step.
- Building footprint matching correctly prioritizes point-in-polygon over nearest-neighbor, and uses parcel intersection to constrain candidates when a parcel is available.
- The defensible space rings (0–5 ft, 5–30 ft, 30–100 ft, 100–300 ft) are anchored to the matched footprint edge — not the geocoded point or parcel centroid. This is architecturally correct.
- Neighbor count excludes the subject building via edge-to-edge distance (< 0.5 m), preventing self-counting.

**What's broken or missing:**
- **Single geocoding provider.** Only Nominatim is used in practice. There is a secondary geocoder slot but no real fallback chain (it requires a different `WF_GEOCODE_SECONDARY_SEARCH_URL` to be set; nothing fills it by default). Rural and low-density addresses are frequently missed or returned at interpolated/approximate precision by Nominatim.
- **No geocode-vs-parcel validation.** The system does not check whether the returned geocoded point actually falls inside the expected parcel, ZIP code, or county. A plausible-looking but wrong result silently propagates.
- **No parcel data for most of the US.** The parcel pipeline only covers hand-prepared regions (currently Missoula/Okanogan pilots). Any address outside a prepared region skips parcel matching entirely and falls back to a bounding-box approximation.
- **No footprint data for most of the US.** Same constraint: footprints are loaded from locally prepared GeoJSON files. Outside prepared regions there is nothing to load.
- **Accessory structure classification is absent.** All non-subject buildings in the ring zones are treated identically — a neighbor's house, an on-parcel ADU, and a garden shed are all counted the same way. There is no parcel-boundary-based classification of on-parcel vs. off-parcel structures.
- **Defensible space rings are not clipped to parcel boundaries by default.** Ring clipping only occurs if a parcel polygon was successfully matched; for the majority of assessments outside prepared regions this clipping does not happen, so rings bleed across property lines.
- **The residential-area bias (190 m² target) will misrank large structures.** Farm barns, ADUs, and any structure significantly larger than a typical single-family home are penalized in footprint matching.

**Biggest risks:**
1. Nominatim returns street-centerline coordinates for 30–40% of US residential addresses; without parcel snapping this becomes the assessment anchor, shifting ring analysis by 15–50 m.
2. Outside prepared regions, all three subsystems (parcel, footprint, rings) degrade to bounding-box approximations — the confidence system correctly penalizes this, but the user has no visibility into the degradation.
3. No national footprint or parcel coverage path exists today; expanding coverage requires per-region manual data preparation.

---

## 2. Geocoding Audit

### 2.1 Pipeline and Provider

**Entry point:** `POST /risk/assess` → `_resolve_location_for_route()` (`main.py` ~line 12680) → `Geocoder.geocode()` or `resolve_local_address_candidate()` (`address_resolution.py`).

**Geocoder class:** `backend/geocoding.py`, class `Geocoder` (lines 84–672).

- **Provider:** OpenStreetMap Nominatim (`nominatim.openstreetmap.org/search`). Configurable via `WF_GEOCODE_SEARCH_URL` and `WF_GEOCODE_PROVIDER_NAME` but no built-in alternative is wired up.
- **Secondary geocoder:** A second `Geocoder` instance exists (`main.py` lines 205–210) with its own `WF_GEOCODE_SECONDARY_*` env vars, but the integration code does not automatically fall back to it on failure — it is used only where explicitly called.

**Query expansion:** Up to 7 variants generated (base, unit-stripped, abbreviations expanded, first-part-only, state-context). This improves hit rate for formatting variations.

**Quality filters (all configurable via env vars):**
- `WF_GEOCODE_MIN_IMPORTANCE` (default 0.02 in prod): Rejects low-importance matches unless the address is structurally precise (house + road + locality + regional anchor).
- `WF_GEOCODE_AMBIGUITY_DELTA` (default 0.0, i.e., disabled): Would reject if the second-ranked candidate's importance is within delta of the top match and display names differ.
- House number mismatch: Rejects if submitted house number does not match the candidate's extracted number.

**Precision classification** (`geocoding.py` lines 399–419):
- `rooftop`: class="building" or type="house"
- `parcel_or_address_point`: precise address fields present
- `interpolated`: highway/road/street types
- `approximate`: postcode/city/town/administrative
- `unknown`: fallback

**Local fallback chain** (`address_resolution.py`, `resolve_local_address_candidate()`):
1. Manual override records (`config/local_address_fallbacks.json`)
2. Prepared region address-point layers
3. Parcel address-point layers
4. Environment-variable-configured custom sources (`WF_LOCATION_RESOLUTION_SOURCE_CONFIG`, `WF_WA_STATEWIDE_PARCEL_PATH`, `WF_OKANOGAN_ADDRESS_POINTS_PATH`)
5. Network Nominatim call

Address point snapping: `snap_geocode_to_address_points()` (`geocoding.py` lines 894–1001) can post-process a network geocode by snapping it to the nearest matching address point within 150 m with a fuzzy match score ≥ 85/100.

### 2.2 Failure Mode Analysis — Geocoding

| Scenario | Handled? | Detail |
|---|---|---|
| Geocodes to street centerline (common) | **Poorly** | No validation that returned point is inside parcel or on correct side of street. Precision is classified as `interpolated`, which triggers a downstream confidence penalty and widens the footprint match window — but the wrong anchor propagates. |
| Rural address with 100 m+ error | **Poorly** | Rural expansion factor (`WF_ADDRESS_POINT_RURAL_EXPANSION_FACTOR`) exists but defaults to 1.0 (disabled). Address point and parcel distance tolerances (45 m and 30 m defaults) will fail to match in large-lot rural areas. |
| Address outside any prepared region | **Not handled** | Local fallback chain finds nothing; network geocode is used as-is with no parcel or footprint matching downstream. |
| Ambiguous city match (two cities, same street name) | **Partially** | House number mismatch filter catches some cases; ambiguity delta is off by default. No geographic bounding (e.g., must be within input ZIP code). |
| Missing house number in input | **Partially** | Query variants attempt street-only matching; match type is downgraded to `street_only_match` with reduced confidence. |

---

## 3. Parcel Data Audit

### 3.1 Data Format and Storage

**Primary class:** `backend/parcel_resolution.py`, class `ParcelResolutionClient` (lines 66–539).  
**Anchor selection:** `backend/property_anchor.py`, class `PropertyAnchorResolver` (lines 101–770).

Parcels are stored as GeoJSON files (Polygon/MultiPolygon geometry). Paths come from:
- Prepared region manifests (per-region files under `data/`)
- `WF_WA_STATEWIDE_PARCEL_PATH` env var
- `config/geometry_source_registry.json` source definitions

No spatial database or spatial index (e.g., R-tree) is used at runtime. Each query scans all features in the loaded GeoJSON and performs Shapely containment/distance checks. For large county parcel layers this will be slow.

### 3.2 Matching Logic

**Step 1 — Point containment** (`parcel_resolution.py` lines 396–445): Filters features that contain the anchor point. Ties broken by source priority, smallest area, then centroid distance. Confidence: 96 − 5×source_rank.

**Step 2 — Distance proximity** (lines 447–506): If no containment found, matches nearest feature within `max_lookup_distance_m` (default 30 m). Confidence: 45 + 0.42×overlap_score − 4×source_rank, clamped [20, 90].

**Step 3 — Fallback** (lines 508–539): No match found. Returns a 25 m bounding-box approximation centered on the anchor point. Confidence 18.

**Ambiguity handling:**
- Multiple containing polygons: flags ambiguity if area_ratio ≤ 1.2 and centroid_gap ≤ 2 m. Confidence drops to 68.
- Multiple nearby polygons: flags ambiguity if distance_gap ≤ 2.5 m and overlap_gap ≤ 6. Confidence drops to 62.

**Attributes extracted:** Parcel polygon geometry, parcel ID (from `parcel_id`/`apn`/`APN`/`parcel_number` etc.), source name and vintage. No owner/tax/zoning attributes are extracted.

### 3.3 Failure Mode Analysis — Parcel

| Scenario | Handled? | Detail |
|---|---|---|
| No parcel data for region | **Not handled** | Falls back to 25 m bounding box. All downstream ring analysis uses the bounding box; footprint matching has no parcel to constrain candidates. |
| Geocoded point lands on street centerline (outside parcel boundary) | **Poorly** | Containment fails; distance proximity (30 m default) may or may not bridge the gap depending on street width and lot setback. Lot-line properties with small setbacks (< 5 m) often fail. |
| Corner lot or L-shaped parcel | **Handled** | Shapely containment/distance handles arbitrary polygon shapes correctly. |
| Stacked parcels (condos, air rights) | **Partially** | Multiple containment detected and flagged ambiguous, confidence 68. No vertical disambiguation. |
| Address straddles parcel boundary (added addition) | **Not handled** | Containment may fail; nearest-parcel match at 30 m could pick the wrong parcel. |
| No spatial index, large county dataset | **Not handled** | Full linear scan of all features per query. No in-memory R-tree built at load time. Performance degrades linearly with dataset size. |

---

## 4. Building Footprint Audit

### 4.1 Data Sources and Loading

**Primary class:** `backend/building_footprints.py`, class `BuildingFootprintClient` (lines 44–683).

Footprints are loaded from locally prepared GeoJSON files. Configured paths come from:
- `WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON` env var
- `WF_LAYER_FEMA_STRUCTURES_GEOJSON` env var
- Per-region manifest entries

Feature cache: `@lru_cache(maxsize=4)` caches the last 4 loaded files in memory. No spatial index built at load.

Source preference (from `config/geometry_source_registry.json`): building_footprints > overture > microsoft > fema_structures.

### 4.2 Matching Logic

**Phase 1 — Parcel intersection** (lines 270–291): If a parcel polygon was matched, filter footprints that intersect the parcel. If no intersecting footprints found and parcel is available, return no match. This is the correct behavior — it prevents cross-parcel contamination.

**Phase 2 — Point-in-polygon** (lines 293–371): Checks whether the anchor point is inside any footprint. Ranks by parcel intersection area, residential area score, source rank. Confidence: 0.97 (parcel-based) or 0.92 (point-based).

**Phase 3 — Nearest footprint fallback** (lines 373–607): If no containing footprint found, scores all candidates within `max_search_m` (120 m) using a weighted composite:
- 75% distance component (1 − dist/max_match_dist, clamped [0,1])
- 15% centroid component
- 10% residential area score (Gaussian centered at 190 m², σ = 320 m²)
- 12% parcel overlap bonus (if parcel available)
- 6% source rank bias

Effective max match distance varies by geocode precision:
- rooftop: 87.5 m
- interpolated/approximate: 43–45 m
- unknown: 41 m
- default: 35 m

**Ambiguity detection:**
- Parcel intersection mode: area_diff ≤ 15 m² AND centroid_diff ≤ 4 m → ambiguous
- Nearest mode: distance_gap ≤ 6 m AND score_diff < 0.08 AND area_score_diff < 0.18 → ambiguous
- Ambiguous results return no match (fallback to point proxy for rings).

**Ring anchoring:** Defensible space rings (`compute_structure_rings()`, lines 686–741) are computed as Shapely buffer differences around the **matched footprint edge** in EPSG:3857 — correct. If a parcel polygon is available, rings are clipped to the parcel boundary. If not, rings extend freely across property lines.

**`multiple_structures_on_parcel`** field (line 835): hardcoded to `"unknown"`. No classification logic implemented.

### 4.3 Failure Mode Analysis — Footprints

| Scenario | Handled? | Detail |
|---|---|---|
| No footprint data for region | **Not handled** | Returns `found=False`, `match_status="provider_unavailable"`. Rings fall back to buffers around the geocoded/anchor point. This is a reasonable degradation but means ring analysis loses accuracy. |
| Multi-structure parcel (house + ADU + garage + barn) | **Partially** | The footprint with the highest composite score is selected as the primary. But the 190 m² Gaussian bias may select an ADU or garage over a large farmhouse. Accessory structures are not labeled; they count identically to neighbor structures in ring metrics. |
| Adjacent parcels with close footprints | **Partially** | Parcel intersection filter prevents cross-parcel contamination when a parcel is matched. Without a parcel, the nearest-neighbor fallback can select a neighbor's house if it is closer than the subject building. |
| Corner lot / geocode near street | **Poorly** | If the geocoded point lands on the street side, the nearest footprint may be the house across the street. No street-edge exclusion logic exists. |
| Large footprint (barn, warehouse) | **Poorly** | Residential area score heavily penalizes structures > ~350 m² (score approaches 0). A 400 m² barn scores below a 190 m² neighbor's house in the nearest-fallback composite. |
| No footprint, no parcel | **Not handled** | Rings are anchored to the geocoded point as a buffer. No boundary constraints apply. |

---

## 5. Nearby Structure Differentiation Audit

### Current Mechanism

The differentiation between the primary structure and neighbors is implicit, not explicit:

1. **Primary structure selection:** `get_building_footprint()` returns a single "matched" footprint. This is implicitly the primary structure. No explicit label is applied; the selection is based on composite score (distance-biased).

2. **Neighbor exclusion:** `get_neighbor_structure_metrics()` (`building_footprints.py` lines 609–683) accepts the matched footprint as `subject_footprint`. Any footprint whose edge is within 0.5 m of the subject footprint edge is excluded from neighbor counts. All others within the search radius are counted.

3. **Accessory structure handling:** None. There is no logic to distinguish an on-parcel ADU from a neighboring house. Parcel clipping is not applied during neighbor counting — only ring geometry clipping is attempted.

4. **`multiple_structures_on_parcel`** is hardcoded to `"unknown"` (`building_footprints.py` line 835). This field exists in the schema but is never populated.

### Assessment

The current logic is fragile for multi-structure properties:
- A garage or ADU on the same parcel is counted identically to a neighbor's house.
- On large rural parcels with multiple outbuildings, the ring analysis will count on-parcel structures as "neighbors," which inflates the nearby structure count and may over-penalize the property.
- There is no parcel-boundary-based partitioning: structures within a ring zone are not tagged as on-parcel vs. off-parcel.

---

## 6. Implementation Plan

### 6.1 Geocoding Improvements

**Recommended strategy: hierarchical fallback chain**

1. **Census TIGER/Line Geocoder** (free, national, REST API) as primary for US addresses. More reliable than Nominatim for residential addresses, returns address-matched points (not street centerlines for most addresses), and is officially maintained. Documented output includes match score and match type (exact/non-exact).

2. **Nominatim** (current) as second-tier fallback. Keep existing logic and quality filters.

3. **Google Maps Geocoding API** (paid, ~$5/1,000 queries) as final-tier fallback for low-confidence results from tiers 1 and 2. Google returns rooftop-precision coordinates for the vast majority of US addresses and is the most accurate widely available geocoder.

**Confidence thresholding and validation:**

- After geocoding, validate the returned point against the expected ZIP code using a nationwide ZIP code centroid lookup (free, available as CSV). If the geocoded point is > 20 km from the expected ZIP centroid, flag for manual review or retry.
- If a parcel was matched, check whether the geocoded point falls within the matched parcel boundary. If not, use the parcel centroid as the assessment anchor and set `source_conflict_flag=True` and `anchor_quality=low`.
- Introduce a `WF_GEOCODE_REQUIRE_ZIP_VALIDATION=true` env var to make ZIP validation opt-in for production.

**Street centerline correction:**

- After a Nominatim `interpolated`-precision result, always attempt address-point snap (`snap_geocode_to_address_points()`) if address point data is available, and always attempt parcel containment check. If the snapped point falls inside a parcel and the original geocode does not, prefer the snapped point.

**Low-confidence geocode handling:**

- If geocode precision is `approximate` or `unknown` and no address-point snap is possible, surface a `geocode_warning` flag in the API response and optionally block automated assessment (configurable via `WF_REQUIRE_GEOCODE_CONFIDENCE=medium`).

### 6.2 Parcel Data Pipeline

**Recommended national parcel strategy (budget-conscious):**

The best realistic option for a solo developer is a combination of:

1. **Regrid free tier / open parcel data** (`openparcelmap.org`): Partial national coverage, available as GeoJSON/Shapefile. Free tier is limited but useful for piloting. Full coverage requires a subscription (~$300–600/year for API access).

2. **County/state open data portals** (free): Approximately 40–50% of US counties publish parcel GIS data. Quality and update frequency vary. Useful for building out regional coverage incrementally.

3. **Microsoft US Building Footprints** + **US address points from OpenAddresses** as a proxy when parcel polygons are unavailable: OpenAddresses provides free address points for most of the US; combined with a building footprint, this gives a reasonable anchor even without parcel boundaries.

4. **On-demand Regrid API** (paid per-query, ~$0.01–0.05/query): For a consumer product, querying Regrid's parcel API at assessment time avoids the need to pre-download and store county datasets. Realistic for moderate query volumes.

**Matching logic improvements:**

- Build an in-memory R-tree spatial index (using Shapely's STRtree or `rtree` package) over loaded parcel features at startup, rather than scanning all features per query. This is a ~10-line change with large performance impact for county-scale datasets.
- After containment match, verify that the matched parcel's address field (if present) fuzzy-matches the input address. If mismatch exceeds a threshold, log a conflict warning.
- For stacked/condo parcels: return all containing parcels, rank by address match, and flag if more than one is returned.

**Parcel attributes to extract per assessment:**
- `parcel_id` / APN
- Parcel polygon geometry (for ring clipping and footprint constraint)
- Parcel centroid (for fallback anchor)
- Parcel area (m²)
- Owner name (if present, for future insurance/ownership verification)
- Land use code / zoning (if present, for structure type inference)
- Address fields from parcel record (for geocode validation)

### 6.3 Building Footprint Pipeline

**Recommended data strategy:**

| Source | Coverage | Freshness | Attributes | Cost |
|---|---|---|---|---|
| Microsoft ML Building Footprints | National US (130M+ buildings) | 2023 snapshot, periodic updates | Footprint geometry only | Free |
| Overture Maps Buildings | National US, global | Quarterly updates | Height, class, source confidence | Free |
| OSM Buildings | Variable; urban > rural | Real-time | Rich attributes but inconsistent | Free |
| Google Open Buildings | US + global | Annual snapshot | Confidence score, footprint | Free |

**Recommendation:** Use **Overture Maps** as the primary source (better attributes and update cadence than Microsoft, better coverage than OSM) with **Microsoft ML Footprints** as a fallback for rural areas where Overture coverage is thin. Download national tiles and tile-index them for spatial lookup.

**Primary structure identification:**

When a parcel polygon is available:
1. Filter footprints to those intersecting the parcel.
2. Rank intersecting footprints by: parcel intersection area (highest first), then footprint area (largest first, to prefer main structure over outbuildings).
3. The largest parcel-intersecting footprint is the primary structure.
4. All other parcel-intersecting footprints are accessory structures.
5. Footprints outside the parcel but within ring zones are neighboring structures.

When no parcel is available (fallback heuristics):
1. Start from the geocoded/anchor point.
2. Select the footprint containing the point (if any).
3. If none, select the nearest footprint within 35 m.
4. Label all other footprints within ring zones as "unknown — may be on-parcel or neighboring."

**Remove or generalize the 190 m² residential bias.** Replace the fixed Gaussian with a configurable target area and standard deviation, or remove the area component from the matching score entirely when a parcel intersection is available (parcel intersection is a stronger signal than area heuristics).

**Ring anchoring:**

Rings should always be anchored to the **footprint edge** (current behavior). Ensure that:
- If no footprint is found, rings fall back to point-buffer mode and a `ring_anchor_mode=point_fallback` field is set in the response.
- If a parcel is matched but no footprint, rings are anchored to the parcel boundary (better than geocoded point for capturing defensible space).

### 6.4 Nearby Structure Differentiation

**Recommended algorithm:**

```
Given: parcel_polygon, subject_footprint, all_footprints_in_rings

1. on_parcel = [f for f in all_footprints_in_rings if f intersects parcel_polygon]
2. off_parcel = [f for f in all_footprints_in_rings if f not in on_parcel]

3. primary = subject_footprint  (already selected by footprint matching)
4. accessory = [f for f in on_parcel if f != primary]
   → sub-classify: detached_garage, ADU, shed, barn by area thresholds:
     - < 30 m²: shed/small accessory
     - 30–80 m²: garage / carport
     - > 80 m², residential class: ADU

5. neighbors = off_parcel (all structures outside parcel boundary but within ring zones)

6. Expose in assessment response:
   - primary_structure_footprint_m2
   - accessory_structure_count
   - accessory_structures: [{type, area_m2, distance_from_primary_ft}]
   - neighboring_structure_count_100ft
   - neighboring_structure_count_300ft
```

**Fallback (no parcel):**

When no parcel polygon is available, use a 0.5 m edge-distance threshold (current behavior) to exclude the subject footprint. All remaining structures are labeled "nearby" without on/off-parcel classification. Set `structure_classification=heuristic` in the response.

**Parcel boundary requirement:**

Accurate on-parcel vs. neighboring classification requires parcel data. Without it, all classification is heuristic. The current `multiple_structures_on_parcel="unknown"` field should be set to `"estimated"` when heuristics are used and `"parcel_derived"` when parcel-constrained classification is performed.

### 6.5 National Coverage Architecture

**Problem:** The current architecture requires per-region manual data preparation. This blocks coverage expansion.

**Proposed hybrid architecture:**

```
Assessment Request (any US address)
        │
        ├─ Geocoding: Census TIGER → Nominatim → Google (fallback chain)
        │
        ├─ Parcel: On-demand Regrid API call (or local tile if pre-downloaded)
        │
        ├─ Footprints: National tile index (Overture + Microsoft pre-tiled)
        │   → Cloud-optimized GeoTIFF / FlatGeobuf tiles indexed by lat/lon bbox
        │   → Tile fetched on demand from object storage (S3/GCS), cached locally
        │
        ├─ Risk layers (LANDFIRE, slope, historic fire):
        │   → LANDFIRE tiles are available as Cloud-Optimized GeoTIFFs
        │   → Range-request only the bounding box needed for the property
        │   → No pre-download required; responses cached per tile
        │
        └─ Scoring: existing pipeline (no changes needed)
```

**Key components for national coverage:**

| Component | Approach | Effort (dev-weeks) |
|---|---|---|
| Geocoding fallback chain (Census + Google) | Add Census TIGER geocoder module, wire secondary fallback | 1–2 |
| Parcel on-demand API (Regrid) | New `parcel_api_client.py`; call Regrid parcel-by-location endpoint | 1–2 |
| National footprint tile index | Download Overture + Microsoft national datasets, tile into FlatGeobuf by bbox, index in SQLite or DuckDB | 3–5 |
| LANDFIRE COG range-requests | Replace file-based raster reads with HTTP range requests to LANDFIRE COG endpoints | 2–3 |
| On-demand region assembly | Replace prepared-region gate with on-demand tile assembly per assessment | 2–4 |
| Caching layer (tile + parcel results) | Redis or SQLite-based TTL cache to avoid repeated API calls for same area | 1–2 |
| **Total** | | **10–18 dev-weeks** |

**Realistic phasing for a solo developer:**

- **Phase 1 (2–3 weeks):** Geocoding fallback chain + ZIP validation. Immediate accuracy improvement for all assessments.
- **Phase 2 (3–5 weeks):** Regrid parcel API integration. Unlocks parcel-constrained footprint and ring analysis for any US address.
- **Phase 3 (4–6 weeks):** National footprint tile index (Overture). Unlocks accurate footprint matching nationally.
- **Phase 4 (3–4 weeks):** LANDFIRE COG range-requests. Eliminates dependency on pre-prepared regional raster data.

### 6.6 Prioritized Execution Order

| Task | Impact | Effort | Priority |
|---|---|---|---|
| Geocoding fallback: add Census TIGER geocoder as primary | High — fixes street-centerline issue for majority of addresses | Low (1–2 weeks) | **P1** |
| ZIP-code validation of geocode result | High — catches wrong-city geocode silently passing through | Low (< 1 week) | **P1** |
| Parcel on-demand API (Regrid) | High — unlocks parcel-constrained footprint matching nationally | Medium (1–2 weeks) | **P2** |
| Build in-memory R-tree index over parcel/footprint datasets | Medium — performance; also required for county-scale datasets | Low (< 1 week) | **P2** |
| National footprint tile index (Overture + Microsoft) | High — unlocks accurate footprint matching nationally | High (3–5 weeks) | **P3** |
| On/off-parcel structure classification | Medium — accuracy of ring analysis and neighbor counts | Low (1 week, requires parcel data) | **P3** |
| Remove/generalize 190 m² residential footprint bias | Medium — accuracy for rural/large-structure properties | Low (< 1 week) | **P3** |
| LANDFIRE COG range-requests | High — eliminates pre-prep requirement for risk layers | Medium (2–3 weeks) | **P4** |
| On-demand region assembly (replaces region gate) | High — enables truly national coverage | High (2–4 weeks) | **P4** |
| Regrid API result caching layer | Low-medium — cost and latency reduction | Low (1 week) | **P5** |

---

## 7. Files Changed / New Files Needed

### New files to create

| File | Description |
|---|---|
| `backend/geocoding_census.py` | Census TIGER geocoder client; wraps the Census Geocoding Services API REST endpoint; returns `GeocodeResult` compatible with existing `Geocoder` output |
| `backend/geocoding_fallback_chain.py` | Orchestrates ordered fallback: Census → Nominatim → Google; exposes single `geocode_with_fallback(address)` entrypoint |
| `backend/parcel_api_client.py` | On-demand Regrid (or county open-data) parcel API client; fetches parcel polygon + attributes by lat/lon; caches results by bbox in SQLite |
| `backend/national_footprint_index.py` | FlatGeobuf tile index for Overture + Microsoft national footprints; spatial lookup by bbox using DuckDB or SQLite R-tree index |
| `backend/structure_classifier.py` | Classifies footprints on a matched parcel into primary / accessory / neighboring; exposes `classify_structures(parcel_polygon, all_footprints, subject_footprint)` |
| `scripts/build_national_footprint_tiles.py` | One-time script to download Overture/Microsoft national datasets and tile into FlatGeobuf files indexed by geohash or grid cell |
| `scripts/validate_geocoder_chain.py` | Benchmark script to run the fallback chain against a test address set and report precision tier distribution |
| `config/geocoding_config.yaml` | Configures geocoder chain parameters: provider order, confidence thresholds, ZIP validation toggle, fallback conditions |

### Files to modify

| File | Changes needed |
|---|---|
| `backend/geocoding.py` | Add Census TIGER geocoder as a provider option; add ZIP-code validation post-processing; expose `provider_chain` config |
| `backend/address_resolution.py` | Wire `geocoding_fallback_chain.py` as the network geocoder; add geocode-vs-parcel validation step after resolution |
| `backend/parcel_resolution.py` | Add STRtree spatial index built at load time; add Regrid API fallback when local data unavailable; add parcel address-field validation against input address |
| `backend/property_anchor.py` | Prefer parcel centroid over interpolated geocode when parcel is confidently matched (current priority may need adjustment for interpolated-precision geocodes that hit a valid parcel) |
| `backend/building_footprints.py` | Remove/generalize 190 m² residential bias; integrate `structure_classifier.py` to populate `multiple_structures_on_parcel` field; add `ring_anchor_mode` field to output |
| `backend/feature_enrichment.py` | Pass `ring_anchor_mode` and `structure_classification` fields through to scoring input metadata |
| `backend/homeowner_report.py` | Surface geocode warnings, parcel match status, and structure classification mode in user-facing report |
| `backend/main.py` | Wire `geocoding_fallback_chain.py` into `_resolve_location_for_route()`; add `geocode_warning` field to response if geocode confidence is low |
| `config/geometry_source_registry.json` | Add Overture and Microsoft national tile index as fallback sources for footprints and parcels |
| `tests/test_geocoding.py` | Add tests for Census TIGER geocoder, fallback chain behavior, and ZIP validation |
| `tests/test_building_footprint_matching.py` | Add tests for on/off-parcel classification, large-structure matching, and ring anchor mode |
| `tests/test_parcel_resolution.py` | Add tests for R-tree index correctness, API fallback behavior, and address validation |
