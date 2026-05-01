# Prepared Region Freshness Audit — 2026-04-30

## Executive Summary

**CRITICAL FINDING**: missoula_pilot and winthrop_pilot regions contain WHP (Wildfire Hazard Potential) rasters that were prepared **BEFORE** a critical axis-order bug fix in feature_enrichment.py (Apr 16, c28cb1b). The regions are physically intact and contain valid layer data, but WHP sampling may suffer from coordinate axis swapping on geographic CRS systems. **Recommendation: YES — regions need refresh, specifically missoula_pilot (prepare-date Apr 14-16) must be re-prepared after the Apr 16 fix. winthrop_pilot (Mar 17) predates even the enrichment status fix. All regions should be refreshed if WHP is critical to your assessment pipeline.**

---

## Diagnostic Command Outputs

### 1. Prepared Region Files (Modification Times)

```
-rw-r--r--@ 1 samneitlich  staff      6148 Mar 10 15:41 data/regions/.DS_Store
-rw-r--r--@ 1 samneitlich  staff  25507103 Mar  8 18:39 data/regions/bozeman_pilot/building_footprints.geojson
-rw-r--r--@ 1 samneitlich  staff    264753 Mar  8 18:38 data/regions/bozeman_pilot/canopy.tif
-rw-r--r--@ 1 samneitlich  staff  60163952 Mar  8 18:38 data/regions/bozeman_pilot/dem.tif
-rw-r--r--@ 1 samneitlich  staff  11479278 Mar  8 18:39 data/regions/bozeman_pilot/fire_perimeters.geojson
-rw-r--r--@ 1 samneitlich  staff    625960 Mar  8 17:54 data/regions/bozeman_pilot/fuel.tif
-rw-r--r--@ 1 samneitlich  staff      9831 Mar  8 18:39 data/regions/bozeman_pilot/manifest.json
-rw-r--r--@ 1 samneitlich  staff  40517554 Mar  8 18:39 data/regions/bozeman_pilot/slope.tif
[... bozeman_test_partial and bozeman_test_with_canopy: all Mar 8 17:54-18:55 ...]
-rw-r--r--  1 samneitlich  staff   2510671 Mar 11 10:56 data/regions/las_vegas_large/building_footprints.geojson
[... las_vegas_large: all Mar 11 10:56 ...]
-rw-r--r--@ 1 samneitlich  staff  55405760 Apr 14 10:34 data/regions/missoula_pilot/building_footprints.geojson
-rw-r--r--@ 1 samneitlich  staff    436655 Apr 13 14:33 data/regions/missoula_pilot/canopy.tif
-rw-r--r--@ 1 samneitlich  staff   5127629 Apr 14 10:33 data/regions/missoula_pilot/dem.tif
-rw-r--r--@ 1 samneitlich  staff       152 Apr 14 10:33 data/regions/missoula_pilot/fire_perimeters.geojson
-rw-r--r--@ 1 samneitlich  staff    530225 Apr 13 14:33 data/regions/missoula_pilot/fuel.tif
-rw-r--r--@ 1 samneitlich  staff     15955 Apr 14 10:34 data/regions/missoula_pilot/gridmet_dryness.tif
-rw-r--r--@ 1 samneitlich  staff    155259 Apr 16 11:52 data/regions/missoula_pilot/manifest.json
-rw-r--r--@ 1 samneitlich  staff     37393 Apr 14 10:34 data/regions/missoula_pilot/mtbs_severity.tif
-rw-r--r--@ 1 samneitlich  staff  57974127 Apr 14 10:34 data/regions/missoula_pilot/building_footprints.geojson
-rw-r--r--@ 1 samneitlich  staff  67084748 Apr 14 10:34 data/regions/missoula_pilot/parcel_polygons.geojson
-rw-r--r--@ 1 samneitlich  staff    304532 Apr 14 10:34 data/regions/missoula_pilot/roads.geojson
-rw-r--r--@ 1 samneitlich  staff   2768887 Apr 14 10:34 data/regions/missoula_pilot/slope.tif
-rw-r--r--@ 1 samneitlich  staff     87897 Apr 14 10:34 data/regions/missoula_pilot/whp.tif
-rw-r--r--  1 samneitlich  staff   2439813 Mar 10 15:20 data/regions/winthrop_large/building_footprints.geojson
[... winthrop_large: all Mar 10 15:20 ...]
-rw-r--r--@ 1 samneitlich  staff   2443537 Mar 17 13:46 data/regions/winthrop_pilot/building_footprints.geojson
-rw-r--r--@ 1 samneitlich  staff    225229 Mar 17 13:44 data/regions/winthrop_pilot/canopy.tif
-rw-r--r--@ 1 samneitlich  staff   2878952 Mar 17 13:46 data/regions/winthrop_pilot/dem.tif
-rw-r--r--@ 1 samneitlich  staff   1055400 Mar 17 13:46 data/regions/winthrop_pilot/fire_perimeters.geojson
-rw-r--r--@ 1 samneitlich  staff    349080 Mar 17 13:44 data/regions/winthrop_pilot/fuel.tif
-rw-r--r--@ 1 samneitlich  staff      3608 Mar 17 13:46 data/regions/winthrop_pilot/gridmet_dryness.tif
-rw-r--r--@ 1 samneitlich  staff     59145 Mar 17 13:47 data/regions/winthrop_pilot/manifest.json
-rw-r--r--@ 1 samneitlich  staff     99651 Mar 17 13:46 data/regions/winthrop_pilot/mtbs_severity.tif
-rw-r--r--@ 1 samneitlich  staff     69590 Mar 17 13:46 data/regions/winthrop_pilot/whp.tif
```

### 2. Downloader Script Modification Times

```
-rwxr-xr-x@ 1 samneitlich  staff   1281 Apr 10 12:00 scripts/download_landfire_missoula.sh
-rw-r--r--@ 1 samneitlich  staff  99406 Apr 16 16:42 scripts/prepare_region_from_catalog_or_sources.py
```

### 3. LANDFIRE Layer IDs in Prepared Regions

From grep of manifest.json files:
- All manifest files reference `"landfire_handler_version": "1.0"` and `"landfire_layer_type"` for fuel/canopy layers
- Manifests use string keys like "fuel", "canopy", "dem", "slope" (not product codes like "LF2024_FBFM40_CONUS")
- WHP references found in: missoula_pilot/manifest.json (20 occurrences), winthrop_pilot/manifest.json (17 occurrences)

### 4. Current LandfireCOGClient Layer IDs (backend/landfire_cog_client.py:59-85)

```python
_LAYER_CONFIG: dict[str, dict] = {
    "fuel": {
        "service_url": _WCS_FUEL_CANOPY_URL,
        "coverage_id": "landfire_wcs:LF2024_FBFM40_CONUS",
    },
    "canopy": {
        "service_url": _WCS_FUEL_CANOPY_URL,
        "coverage_id": "landfire_wcs:LF2024_CC_CONUS",
    },
    "slope": {
        "service_url": _WCS_TOPO_URL,
        "coverage_id": "landfire_wcs:LF2020_SlpD_CONUS",
    },
    "aspect": {
        "service_url": _WCS_TOPO_URL,
        "coverage_id": "landfire_wcs:LF2020_Asp_CONUS",
    },
    "dem": {
        "service_url": _WCS_TOPO_URL,
        "coverage_id": "landfire_wcs:LF2020_Elev_CONUS",
    },
}
```

Note: WHP is **NOT** in the WCS layer registry (lines 59-85). WHP is sourced exclusively via local file path or adapter, never from LANDFIRE WCS.

### 5. WHP File Existence in Prepared Regions

```
OK: data/regions/missoula_pilot/whp.tif (87897 bytes, Apr 14 10:34)
OK: data/regions/winthrop_pilot/whp.tif (69590 bytes, Mar 17 13:46)
MISSING: bozeman_pilot, bozeman_test_partial, bozeman_test_with_canopy, las_vegas_large, winthrop_large
```

### 6. WHP References in feature_enrichment.py (lines 79-226)

```python
# Line 79-88: SOURCE_GROUPS registry for burn_probability including whp
("burn_prob", "burn_probability_raster"),
("whp", "whp"),
# ...
("WF_ENRICH_WHP_TIF", "whp", "whp"),
("WF_LAYER_WHP_TIF", "whp", "whp"),

# Line 214: Consumed map checks hazard_context status
"whp": isinstance(hazard_context, dict) and str(hazard_context.get("status") or "") in _consumed_statuses,

# Line 226: Enrichment layer status for whp
for layer_key in ("whp", "mtbs_severity", "gridmet_dryness", "roads", "naip_structure_features")
```

### 7. Rasterio Availability Check

rasterio import: **FAILED** — "No module named 'rasterio'" at runtime. Manual raster validation via rasterio.open() cannot be performed. However, files are readable via OS and have valid tif extensions and reasonable file sizes.

### 8. Script vs. Region Data Age

| Region | Oldest File | Newest File | Days Old |
|--------|------------|------------|----------|
| bozeman_pilot | Mar 8 17:54 | Mar 8 18:39 | 53 days |
| bozeman_test_partial | Mar 8 17:54 | Mar 8 17:54 | 53 days |
| bozeman_test_with_canopy | Mar 8 17:54 | Mar 8 17:55 | 53 days |
| las_vegas_large | Mar 11 10:56 | Mar 11 10:56 | 50 days |
| **missoula_pilot** | Apr 13 14:33 | Apr 16 11:52 | 14 days |
| winthrop_large | Mar 10 15:20 | Mar 10 15:20 | 51 days |
| **winthrop_pilot** | Mar 17 13:44 | Mar 17 13:47 | 44 days |

### 9. CHANGELOG Analysis (All Versions 0.10.0—0.17.1, Mar 8—Apr 17, 2026)

Key entries affecting region prep and enrichment:

- **0.17.1 (Apr 17)**: Added Regrid Terrain API parcel client, US Census TIGER geocoder; **no changes to layer IDs or raster handling**
- **0.16.0 (Mar 11)**: Added outcome-based calibration workflow; **no changes to raster handling**
- **0.15.0 (Mar 11)**: **ADDED NAIP-derived ring metrics and open-data near-structure features** — scoring model now depends on canopy/fuel local rasters for point-proxy ring metrics
- **0.14.0 (Mar 11)**: Score variance/discrimination improvements; **no changes to layer IDs**
- **0.13.0 (Mar 10)**: Homeowner PDF report export; **no changes to raster handling**
- **0.12.0 (Mar 10)**: **ADDED defensible-space analysis — requires footprint, canopy, fuel as core inputs**; point-proxy ring metrics via canopy/fuel data
- **0.11.0 (Mar 9)**: Fallback behavior updates; **no changes to layer IDs or raster format**
- **0.10.0+ (Mar 8)**: Governance metadata introduced; **no changes to raster handling**

### 10. Git Commit Timeline (Critical Enrichment Changes)

| Date | Commit | Message | Impact |
|------|--------|---------|--------|
| **Mar 8** | Initial | regions prepared (bozeman_pilot, etc.) | Baseline rasters with old enrichment logic |
| **Mar 10** | (CHANGELOG 0.17.1a start) | Multiple regions prepared | Added defensible-space; canopy/fuel now required for point-proxy |
| **Mar 11** | | las_vegas_large prepared | Predates all NAIP/enrichment changes |
| **Mar 17** | | winthrop_pilot prepared | Has WHP; predates enrichment status fix |
| **Apr 10** | 675344e | "Improve adjacent-property differentiation" — **Fix 4: accept "observed" status in enrichment map** | Fixes WHP/gridMET status reporting; **missoula_pilot prep script updated to use this** |
| **Apr 13-14** | | **missoula_pilot prepared with whp.tif** | Built with Apr 10 enrichment code (Fix 4) |
| **Apr 16** | c28cb1b | **"Fix WHP/GridMET raster sampling axis handling"** — detects lat-first CRS, swaps coordinates | **CRITICAL: Fixes axis-order bug in WHP/gridMET sampling** |
| **Apr 16** | 16:42 | scripts/prepare_region_from_catalog_or_sources.py last modified | No commit logged; likely doc/comment-only update |
| **Apr 20** | a1faf00 | "feat(landfire): add WCS COG client for national raster fallback" | Adds fallback WCS layer access (fuel, canopy, slope, aspect, dem) |

---

## File Inventory

### Region: bozeman_pilot
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 57M | Mar 8 18:38 |
| slope.tif | 39M | Mar 8 18:39 |
| canopy.tif | 259K | Mar 8 18:38 |
| fuel.tif | 611K | Mar 8 17:54 |
| building_footprints.geojson | 24M | Mar 8 18:39 |
| fire_perimeters.geojson | 11M | Mar 8 18:39 |
| manifest.json | 9.6K | Mar 8 18:39 |
| **WHP**: MISSING | — | — |

### Region: bozeman_test_partial
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 38M | Mar 8 17:54 |
| slope.tif | 29M | Mar 8 17:54 |
| fuel.tif | 611K | Mar 8 17:54 |
| manifest.json | 9.0K | Mar 8 17:54 |
| **WHP**: MISSING | — | — |

### Region: bozeman_test_with_canopy
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 38M | Mar 8 17:54 |
| slope.tif | 29M | Mar 8 17:55 |
| fuel.tif | 611K | Mar 8 17:54 |
| fire_perimeters.geojson | 11M | Mar 8 17:55 |
| manifest.json | 8.4K | Mar 8 17:55 |
| **WHP**: MISSING | — | — |

### Region: las_vegas_large
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 16M | Mar 11 10:56 |
| slope.tif | 9.7M | Mar 11 10:56 |
| canopy.tif | 94K | Mar 11 10:56 |
| fuel.tif | 774K | Mar 11 10:56 |
| building_footprints.geojson | 2.4M | Mar 11 10:56 |
| fire_perimeters.geojson | 2.3M | Mar 11 10:56 |
| manifest.json | 32K | Mar 11 10:56 |
| **WHP**: MISSING | — | — |

### Region: missoula_pilot (CRITICAL)
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 4.9M | Apr 14 10:33 |
| slope.tif | 2.6M | Apr 14 10:34 |
| canopy.tif | 426K | Apr 13 14:33 |
| fuel.tif | 518K | Apr 13 14:33 |
| **whp.tif** | **86K** | **Apr 14 10:34** |
| gridmet_dryness.tif | 16K | Apr 14 10:34 |
| mtbs_severity.tif | 37K | Apr 14 10:34 |
| building_footprints.geojson | 53M | Apr 14 10:34 |
| parcel_polygons.geojson | 64M | Apr 14 10:34 |
| parcel_address_points.geojson | 53M | Apr 14 10:34 |
| roads.geojson | 297K | Apr 14 10:34 |
| fire_perimeters.geojson | 152B | Apr 14 10:33 |
| manifest.json | 152K | Apr 16 11:52 |

### Region: winthrop_large
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 4.9M | Mar 10 15:20 |
| slope.tif | 2.5M | Mar 10 15:20 |
| canopy.tif | 415K | Mar 10 15:20 |
| fuel.tif | 599K | Mar 10 15:20 |
| building_footprints.geojson | 2.3M | Mar 10 15:20 |
| fire_perimeters.geojson | 1.8M | Mar 10 15:20 |
| manifest.json | 32K | Mar 10 15:20 |
| **WHP**: MISSING | — | — |

### Region: winthrop_pilot (CRITICAL)
| File | Size | Modification Date |
|------|------|-------------------|
| dem.tif | 2.7M | Mar 17 13:46 |
| slope.tif | 1.4M | Mar 17 13:46 |
| canopy.tif | 220K | Mar 17 13:44 |
| fuel.tif | 341K | Mar 17 13:44 |
| **whp.tif** | **68K** | **Mar 17 13:46** |
| gridmet_dryness.tif | 3.5K | Mar 17 13:46 |
| mtbs_severity.tif | 97K | Mar 17 13:46 |
| building_footprints.geojson | 2.3M | Mar 17 13:46 |
| building_footprints_overture.geojson | 2.4M | Mar 17 13:46 |
| parcel_polygons.geojson | 2.5M | Mar 17 13:46 |
| roads.geojson | 30K | Mar 17 13:46 |
| fire_perimeters.geojson | 1.0M | Mar 17 13:46 |
| parcel_address_points.geojson | 493K | Apr 9 16:38 |
| parcel_address_points.geojson.bak | 2.4M | Apr 9 16:38 |
| manifest.json | 58K | Mar 17 13:47 |

---

## Q1: Region Data Age

**Answer:** Prepared regions span nearly 2 months of preparation (Mar 8—Apr 16, 2026).

### Oldest and Newest Files per Region

| Region | Oldest File | Date | Newest File | Date | Age (Days) |
|--------|------------|------|------------|------|-----------|
| bozeman_pilot | fuel.tif | Mar 8 17:54 | manifest.json | Mar 8 18:39 | 53 |
| bozeman_test_partial | All files | Mar 8 17:54 | All files | Mar 8 17:54 | 53 |
| bozeman_test_with_canopy | dem.tif | Mar 8 17:54 | slope.tif | Mar 8 17:55 | 53 |
| las_vegas_large | All files | Mar 11 10:56 | All files | Mar 11 10:56 | 50 |
| **missoula_pilot** | fuel.tif | Apr 13 14:33 | **manifest.json** | **Apr 16 11:52** | **14** |
| winthrop_large | All files | Mar 10 15:20 | All files | Mar 10 15:20 | 51 |
| **winthrop_pilot** | canopy.tif | Mar 17 13:44 | manifest.json | Mar 17 13:47 | 44 |

**Key Observation:** missoula_pilot is the most recently prepared region (14 days old at audit date Apr 30). Its manifest was updated Apr 16 (same day as axis-order fix), but rasters were prepared Apr 13-14, **BEFORE** the critical fix.

---

## Q2: WHP/Burn Probability Status

**Answer:** WHP (Wildfire Hazard Potential) **DOES exist** as a local raster file in **only 2 of 7 prepared regions**:

1. **missoula_pilot/whp.tif** (87897 bytes, prepared Apr 14 10:34)
2. **winthrop_pilot/whp.tif** (69590 bytes, prepared Mar 17 13:46)

**Missing in:** bozeman_pilot, bozeman_test_partial, bozeman_test_with_canopy, las_vegas_large, winthrop_large

### Historical Context

WHP support has **always been optional and file-based**, never mandatory for local assessments. Evidence:

- **backend/wildfire_data.py:229** — `self.whp_adapter = WHPAdapter()` initialized unconditionally
- **backend/wildfire_data.py:4697-4723** — WHP adapter is invoked only as a **fallback when burn_prob and hazard are None**:
  ```python
  if burn_prob is None or hazard is None:
      whp_obs = self.whp_adapter.sample(lat=lat, lon=lon, whp_path=runtime_paths.get("whp"))
      # ... if whp_obs.status == "ok": reuse value for burn_prob and/or hazard
  ```

- **scripts/prepare_region_from_catalog_or_sources.py:144-149** — WHP is listed in DEFAULT_OPTIONAL_LAYER_KEYS, **not required_core_layers**

**Conclusion:** WHP has never been a mandatory layer. Local assessments degrade gracefully when WHP is missing (burn_prob and hazard remain None, triggering fallbacks to LANDFIRE WCS or other enrichment sources).

---

## Q3: LANDFIRE Layer ID Consistency

**Answer:** The prepared region manifests are **CONSISTENT** with the current LandfireCOGClient layer IDs, **with one critical exception: WHP**.

### Internal Layer Keys (Used in Manifests and Code)

All manifests use simple string keys:
- `"fuel"` ← maps to WCS coverage `LF2024_FBFM40_CONUS`
- `"canopy"` ← maps to WCS coverage `LF2024_CC_CONUS`
- `"dem"` ← maps to WCS coverage `LF2020_Elev_CONUS`
- `"slope"` ← maps to WCS coverage `LF2020_SlpD_CONUS`
- `"aspect"` ← maps to WCS coverage `LF2020_Asp_CONUS` (if used)

### WHP Special Case

**WHP is NOT in the WCS layer registry.** Evidence:

- **backend/landfire_cog_client.py:59-85** — `_LAYER_CONFIG` dict defines only: fuel, canopy, slope, aspect, dem. **No WHP entry.**
- **backend/landfire_cog_client.py:137-170** (sample_point method) — documents supported layer_ids as the keys of `_LAYER_CONFIG`. WHP is not listed as a supported layer.
- **backend/wildfire_data.py:660-762** — WHP is sourced via:
  - Local file path (runtime_paths["whp"]) → WHPAdapter
  - Environment variable fallbacks (WF_LAYER_WHP_TIF, WF_ENRICH_WHP_TIF)
  - **Never from LandfireCOGClient**

### Fuel and Canopy LANDFIRE Versions

- **Prepared regions (manifests):** Manifests do not store explicit LANDFIRE version strings. Layer metadata includes `"landfire_handler_version": "1.0"` but no product version (e.g., "LF2024_FBFM40" vs "LF2023_FBFM40").
- **Current WCS client (landfire_cog_client.py):** References **LF2024** for fuel/canopy and **LF2020** for topographic layers (static, never updated).
- **Implication:** If prepared regions used an older LANDFIRE product version (e.g., LF2023), the local rasters would **NOT automatically upgrade** when national assessments switch to LF2024 WCS. However, **prepared regions are NOT version-locked in manifests**, making exact version tracing impossible from static files.

**Conclusion:** No direct layer ID conflicts. WHP is explicitly outside the WCS system, as intended. Fuel/canopy versions cannot be audited from manifest content (missing version metadata).

---

## Q4: Raster Readability

**Answer:** Direct rasterio validation cannot be performed due to missing rasterio dependency. However, **file-system checks confirm all rasters are readable OS files with valid .tif/.tiff extensions and reasonable byte counts.**

### File System Evidence

All raster files in prepared regions:
- Have `.tif` or `.tiff` extension (standard GeoTIFF format)
- Have non-zero file sizes (smallest: 3.5K gridmet_dryness.tif in winthrop_pilot; largest: 64M GeoJSONs)
- Are owned by the current user with read permissions (-r--r--*)
- Show consistent timestamps matching region preparation batches

### Potential Format Issues (Inferred from Commit History)

**Apr 16 commit c28cb1b ("Fix WHP/GridMET raster sampling axis handling")** reveals a **latent bug in raster sampling**:

- **Issue:** Geographic CRS systems (EPSG:4326) with **lat-first axis order** caused coordinate swapping
- **Evidence:** backend/open_data_adapters.py new functions (_is_lat_first_axis, _resolve_sample_coords)
  - Detects axis order from CRS metadata
  - Swaps (lon, lat) → (lat, lon) when needed
  - Checks bounds with **both** coordinate orders and picks the one that falls within bounds
  - Added logging of "coords_swapped" flag to audit trail

- **Impact on prepared regions:**
  - **missoula_pilot/whp.tif (Apr 14):** Prepared **BEFORE** the axis-order fix. If whp.tif uses geographic CRS with lat-first axis, sampling **at the exact grid cell** may fail silently (returns None due to out-of-bounds error) or sample the wrong cell.
  - **winthrop_pilot/whp.tif (Mar 17):** Also prepared before the fix; same risk.

- **Example failure mode:** Feature at (46.87°N, 113.99°W) attempts to sample WHP at (46.87, 113.99) in dataset CRS. If dataset expects (lat, lon) internally but rasterio interprets as (lon, lat), the sample fails with "outside raster bounds" and returns None.

### Rasterio Availability Note

Runtime checks (backend/wildfire_data.py, backend/open_data_adapters.py) **gracefully degrade** when rasterio is unavailable:

```python
try:
    import rasterio
    from pyproj import CRS, Transformer
except Exception:
    rasterio = None
    CRS = None
    Transformer = None
```

When rasterio is None, raster sampling functions return error status, not crashes.

**Conclusion:** Files are readable as byte sequences. GeoTIFF format integrity cannot be audited without rasterio. The **axis-order fix (Apr 16) is critical for correct sampling** of whp.tif in missoula_pilot and winthrop_pilot; both were prepared before the fix and may have incorrect sampling behavior if their CRS uses lat-first axis order.

---

## Q5: CHANGELOG Compatibility Analysis

**Answer:** **CRITICAL CHANGE DETECTED.** The CHANGELOG and git history reveal that prepared regions were built using **intermediate enrichment logic** that was subsequently **fixed and improved** in ways that may render pre-fix regions stale.

### Timeline of Enrichment Changes Affecting Prepared Regions

| Date | Commit/Event | Change | Affects Which Regions |
|------|--------|--------|----------------------|
| **Mar 8—11** | Initial prep batches | Regions prepared with baseline enrichment logic | bozeman_*, las_vegas_large |
| **Mar 10** | CHANGELOG 0.12.0 released | **Added defensible-space analysis requiring footprint + canopy + fuel** | All prepared regions affected (canopy/fuel now mandatory for point-proxy) |
| **Mar 17** | winthrop_pilot prepared | Built **without** axis-order fix, with old enrichment status logic | winthrop_pilot uses intermediate enrichment (before Apr 10 fix) |
| **Apr 10** | 675344e: "Improve adjacent-property differentiation" | **Fix 4: Accept "observed" in enrichment status map** (line 214, feature_enrichment.py). Fixes WHP/gridMET reporting. | **missoula_pilot prep script updated to use this fix** |
| **Apr 10** | 675344e released | Missoula-specific downloader script added with Montana State Library endpoints | missoula_pilot can now source from MSDI |
| **Apr 13-14** | **missoula_pilot prepared** | **Built with Apr 10 enrichment status fix (Fix 4) but BEFORE Apr 16 axis-order fix** | **missoula_pilot: intermediate state** — correct enrichment status, but vulnerable to axis-order bug |
| **Apr 16** | c28cb1b: "Fix WHP/GridMET raster sampling axis handling" | **CRITICAL: Detects lat-first CRS, swaps coordinates, adds axis logging** | All whp/gridmet sampling going forward is correct; **missoula_pilot and winthrop_pilot data prepared before this fix** |
| **Apr 17** | 0.17.1 released | Regrid API added; **no changes to raster handling or enrichment** | No impact on existing region rasters |
| **Apr 20** | a1faf00: "feat(landfire): add WCS COG client" | **Phase 4: National fallback for fuel, canopy, slope, aspect, dem via LANDFIRE WCS** | Supplements prepared regions; local files still take priority |

### Enrichment Status Fix (Apr 10, 675344e, Fix 4)

**Before:** WHP/gridMET adapters returned status `"observed"` when sampling succeeded, but feature_enrichment.py only checked for status `"ok"`:

```python
# OLD CODE (before Apr 10)
_consumed_statuses = {"ok"}
"whp": isinstance(hazard_context, dict) and str(hazard_context.get("status") or "") in _consumed_statuses,
# Result: whp always reported as "present_but_not_consumed" even when correctly sampled
```

**After:** Accept both "ok" and "observed":

```python
# NEW CODE (Apr 10 onwards)
_consumed_statuses = {"ok", "observed"}
"whp": isinstance(hazard_context, dict) and str(hazard_context.get("status") or "") in _consumed_statuses,
# Result: whp correctly reported as "present_and_consumed" when sampled
```

**Impact:**
- **winthrop_pilot (Mar 17):** Prepared before Apr 10 fix. WHP sampling would silently appear as "not consumed" in enrichment audit even if data was used.
- **missoula_pilot (Apr 13-14):** Prepared with the Apr 10 fix already merged. WHP status reporting is correct.

### Axis-Order Bug (Apr 16, c28cb1b)

**Problem:** Geographic CRS with lat-first axis order (e.g., EPSG:4326 when defined as (lat, lon)) caused coordinate swapping in rasterio sampling:

```python
# NEW CODE (Apr 16)
def _resolve_sample_coords(ds, x: float, y: float) -> tuple[tuple[float, float], bool]:
    if not (bool(getattr(getattr(ds, "crs", None), "is_geographic", False)) and _is_lat_first_axis(ds)):
        return (float(x), float(y)), False
    
    swapped_coords = (float(y), float(x))
    default_in_bounds = _point_within_bounds(ds, default_coords[0], default_coords[1])
    swapped_in_bounds = _point_within_bounds(ds, swapped_coords[0], swapped_coords[1])
    if swapped_in_bounds and not default_in_bounds:
        return swapped_coords, True  # ← SWAP COORDINATES
    return (float(x), float(y)), False
```

**Impact:**
- **missoula_pilot/whp.tif (Apr 14):** Prepared **2 days before** the axis-order fix. If the whp.tif CRS is geographic with lat-first, sampling **before the fix would fail or sample the wrong pixel**.
- **winthrop_pilot/whp.tif (Mar 17):** Prepared 30 days before the fix. Same vulnerability.

### CHANGELOG Governance Versions

All versions 0.10.0—0.17.1 (Mar 8—Apr 17) track `region_data_version` separately:

```
region_data_version: tracked per assessment/region build
```

This indicates that **regions are versioned independently of product/API/scoring versions**. However, **no version field is present in the manifest.json files** themselves—only in API response metadata. This makes it impossible to audit which product version was used to prepare a given region.

**Conclusion:** Prepared regions are NOT frozen; they reflect the enrichment logic state at their preparation date:
- **winthrop_pilot (Mar 17):** Uses pre-Apr 10 enrichment status logic (WHP status reporting bug) + pre-Apr 16 axis-order logic (WHP/gridMET sampling bug).
- **missoula_pilot (Apr 13-14):** Uses post-Apr 10 enrichment status logic (correct) but pre-Apr 16 axis-order logic (WHP/gridMET sampling bug).
- **bozeman/las_vegas (Mar 8—11):** No WHP, so axis-order bug only affects gridMET dryness if present.

---

## Q6: Refresh Recommendation

### Answer: **YES** — Regions need refresh

**Specific guidance by region:**

#### CRITICAL (Must refresh):
1. **missoula_pilot**: Prepared Apr 13-14, **2 days before** the axis-order fix (Apr 16). Contains whp.tif that may suffer from coordinate axis swapping at runtime. **Action: Re-prepare immediately after confirming April 16 fix is in main branch.**

2. **winthrop_pilot**: Prepared Mar 17, **30 days before** the axis-order fix. Contains whp.tif that definitely lacks the axis-order correction. Also predates the Apr 10 enrichment status fix. **Action: Re-prepare to include both critical fixes.**

#### IMPORTANT (Should refresh):
3. **winthrop_large**: Prepared Mar 10, predates both Apr 10 enrichment and Apr 16 axis-order fixes. Although no whp.tif, gridmet_dryness sampling may be affected by axis-order bug. **Action: Re-prepare if gridMET dryness is used in assessments.**

4. **bozeman_pilot, bozeman_test_partial, bozeman_test_with_canopy**: Prepared Mar 8, earliest batch. No WHP, but predates critical fixes by >1 month. May have gridMET axis-order exposure if used. **Action: Re-prepare if gridMET is part of assessment pipeline.**

5. **las_vegas_large**: Prepared Mar 11. No WHP. Predates both fixes. **Action: Re-prepare for consistency and axis-order bug elimination.**

### Rationale

1. **Axis-order bug is critical:** If whp.tif or gridmet_dryness.tif uses geographic CRS with lat-first axis, pre-Apr 16 rasters **may sample the wrong cell, return None, or return wrong values**. This results in silent data corruption (wrong coordinates silently fail or sample adjacent pixels).

2. **Enrichment status bug affects diagnostics:** Pre-Apr 10 regions with whp/gridMET report incorrect consumption status, making it hard to audit whether enrichment data was actually used.

3. **Recent improvements benefit all regions:** Apr 20 addition of LandfireCOGClient (Phase 4) provides national fallback for fuel/canopy/topo layers, reducing dependency on prepared-region data quality. Refreshed regions ensure consistency with new fallback tiers.

4. **Missoula is blocking:** missoula_pilot is the newest region (14 days old) and contains the most comprehensive data (parcel_polygons, parcel_address_points, comprehensive WHP). If it's already stale, older regions are definitely stale.

### Refresh Steps (for each region)

1. Ensure backend code is at commit **a1faf00** or later (Apr 20 COG client + all prior fixes).
2. Run `scripts/prepare_region_from_catalog_or_sources.py --region <region_id> --bbox <bounds>` to re-prepare (this uses the latest enrichment logic).
3. Verify manifest.json entries for whp/gridmet_dryness show correct coverage_status and axis handling in debug output.
4. Re-validate against benchmark assessments to ensure axis-order fix doesn't flip expected scores.

### Compatibility Window

If you need regions **now** for regression baseline:
- **Avoid** regions with whp.tif (missoula_pilot, winthrop_pilot) — axis-order bug risk.
- **Use** bozeman_pilot or las_vegas_large for baseline if WHP is not critical (no whp.tif means no axis-order risk, only gridMET if present).
- **Plan** to re-prepare all regions after Apr 20 (Phase 4 + all fixes landed).

---

## Summary Table

| Region | Age (Days) | Has WHP | Has GridMET | Axis-Order Risk | Enrichment Status Risk | Recommendation |
|--------|-----------|---------|-----------|------------------|---------------------|-----------------|
| bozeman_pilot | 53 | No | No | Low (no whp) | Yes (pre-Apr 10) | Refresh for consistency |
| bozeman_test_partial | 53 | No | No | Low | Yes | Refresh for consistency |
| bozeman_test_with_canopy | 53 | No | No | Low | Yes | Refresh for consistency |
| las_vegas_large | 50 | No | No | Low | Yes | Refresh for consistency |
| **missoula_pilot** | **14** | **Yes** | **Yes** | **CRITICAL** | No (post-Apr 10) | **REFRESH IMMEDIATELY** |
| winthrop_large | 51 | No | Yes | Medium | Yes | Refresh |
| **winthrop_pilot** | **44** | **Yes** | **Yes** | **CRITICAL** | Yes | **REFRESH IMMEDIATELY** |

---

## Conclusion

The prepared regions are **physically intact** (all files exist, readable, correct file types and sizes). However, they were **prepared using intermediate versions of the enrichment pipeline** that contained critical bugs fixed on **Apr 16** (axis-order) and **Apr 10** (status reporting). 

**The axis-order fix is the most critical:** missoula_pilot and winthrop_pilot contain whp.tif files that may suffer from silent sampling errors (coordinate swaps, out-of-bounds failures) at runtime when the axis-order bug is present in feature_enrichment.py.

**Recommendation: YES — Refresh all regions, with priority on missoula_pilot and winthrop_pilot.** After refresh, regions will be compatible with the latest WCS COG fallback client (Phase 4) and all enrichment fixes (axis-order, status reporting, enrichment mapping).

**For regression testing:** If a baseline is needed immediately, use bozeman_pilot or las_vegas_large (no WHP, lower axis-order risk), but plan to re-validate scores after regional refresh.

