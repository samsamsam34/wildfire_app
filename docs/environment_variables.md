# Environment Variables

This table lists environment variables referenced by `backend/*.py`.

| Variable | Default | Description |
|---|---|---|
| APP_ENV | (none) | Used by backend `geocoding.py`. |
| GDAL_HTTP_TIMEOUT | (none) | Used by backend `national_elevation_client.py`. |
| OPENAI_API_KEY | (none) | Used by backend `homeowner_report.py`. |
| PYTEST_CURRENT_TEST | (none) | Used by backend `geocoding.py`. |
| WF_ADDRESS_POINTS_SNAP_MAX_DISTANCE_M | "150" | Used by backend `main.py`. |
| WF_ADDRESS_POINTS_SNAP_MIN_SCORE | "85" | Used by backend `main.py`. |
| WF_ADDRESS_POINT_MIN_COMPLETE_RATIO | "0.1" | Used by backend `address_resolution.py`. |
| WF_ADDRESS_POINT_MIN_POINT_RATIO | "0.85" | Used by backend `address_resolution.py`. |
| WF_ADDRESS_POINT_SOURCE_NAME | (none) | Used by backend `property_anchor.py`. |
| WF_ALLOWED_ORIGINS | "http://localhost:4173,http://localhost:8000" | Comma-separated CORS allowlist origins. |
| WF_ALLOW_LEGACY_LAYER_FALLBACK | "true" | Used by backend `wildfire_data.py`. |
| WF_AUTO_REGION_PREP_TILE_DEG | "0.25" | Used by backend `main.py`. |
| WF_BUILDING_FOOTPRINT_DATE | (none) | Used by backend `main.py`. |
| WF_BUILDING_FOOTPRINT_VERSION | (none) | Used by backend `main.py`. |
| WF_BUILDING_SOURCE_PRIORITY | "building_footprints_overture,building_footprints_microsoft,building_footprints,fema_structures" | Comma-separated building footprint source priority used by `wildfire_data._resolve_building_source_paths`; unknown tokens are ignored and available sources are appended as fallback. |
| WF_DEBUG_ERRORS | "false" | When true, include exception details in 500 responses. |
| WF_DEBUG_MODE | (none) | Used by backend `geocoding.py`. |
| WF_ELEVATION_CACHE_DB | "data/elevation_cache.db" | Used by backend `wildfire_data.py`. |
| WF_ELEVATION_COG_ENABLED | "true" | Used by backend `wildfire_data.py`. |
| WF_ENRICH_MICROSOFT_BUILDINGS_PATH | "" | Used by backend `wildfire_data.py`. |
| WF_ENV | (none) | Used by backend `geocoding.py`. |
| WF_FEATURE_BUNDLE_CACHE_DIR | (none) | Used by backend `feature_bundle_cache.py`. |
| WF_FEATURE_BUNDLE_CACHE_TTL_SEC | str(6 * 3600 | Used by backend `feature_bundle_cache.py`. |
| WF_FOOTPRINT_CACHE_DB | "data/footprint_cache.db" | Used by backend `wildfire_data.py`. |
| WF_FOOTPRINT_SOURCE_NAME | "" | Used by backend `wildfire_data.py`. |
| WF_FOOTPRINT_SOURCE_VINTAGE | "" | Used by backend `wildfire_data.py`. |
| WF_GEOCODE_ALLOW_PRECISE_LOW_IMPORTANCE | "true" | Used by backend `geocoding.py`. |
| WF_GEOCODE_AMBIGUITY_DELTA | "0.0" | Used by backend `geocoding.py`. |
| WF_GEOCODE_BACKOFF_QUERY_LIMIT | "6" | Used by backend `main.py`. |
| WF_GEOCODE_MAX_CANDIDATES | "5" | Used by backend `geocoding.py`. |
| WF_GEOCODE_MAX_QUERY_VARIANTS | "7" | Used by backend `geocoding.py`. |
| WF_GEOCODE_MIN_IMPORTANCE | default_min_importance | Used by backend `geocoding.py`. |
| WF_GEOCODE_PROVIDER_NAME | (none) | Used by backend `geocoding.py`. |
| WF_GEOCODE_SEARCH_URL | (none) | Used by backend `geocoding.py`. |
| WF_GEOCODE_SECONDARY_PROVIDER_NAME | "Secondary Geocoder" | Used by backend `main.py`. |
| WF_GEOCODE_SECONDARY_SEARCH_URL | "" | Used by backend `main.py`. |
| WF_GEOCODE_SECONDARY_USER_AGENT | _nominatim_geocoder.user_agent | Used by backend `main.py`. |
| WF_GEOCODE_TIMEOUT_SECONDS | "8" | Used by backend `geocoding.py`. |
| WF_GEOCODING_CONFIG_PATH | str(_DEFAULT_CONFIG_PATH | Used by backend `geocoding_fallback_chain.py`. |
| WF_GEOMETRY_SOURCE_REGISTRY_PATH | "" | Used by backend `geometry_source_registry.py`. |
| WF_GOOGLE_GEOCODE_API_KEY | (none) | Used by backend `geocoding_fallback_chain.py`. |
| WF_HOMEOWNER_EXPLANATION_MODEL | (none) | Used by backend `homeowner_report.py`. |
| WF_LANDFIRE_CACHE_DB | "data/landfire_cache.db" | Used by backend `wildfire_data.py`. |
| WF_LANDFIRE_COG_ENABLED | "true" | Used by backend `wildfire_data.py`. |
| WF_LAYER_ADDRESS_POINTS_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_ASPECT_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON | "" | Used by backend `building_footprints.py`. |
| WF_LAYER_BUILDING_FOOTPRINTS_MICROSOFT_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_BUILDING_FOOTPRINTS_OVERTURE_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_BURN_PROB_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_BURN_PROB_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_BURN_PROB_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_CANOPY_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_CANOPY_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_CANOPY_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_DEM_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_DEM_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_DEM_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_FEMA_STRUCTURES_GEOJSON | "" | Used by backend `building_footprints.py`. |
| WF_LAYER_FIRE_PERIMETERS_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_FIRE_PERIMETERS_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_FIRE_PERIMETERS_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_FUEL_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_FUEL_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_FUEL_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_GRIDMET_DRYNESS_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_HAZARD_SEVERITY_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_HAZARD_SEVERITY_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_HAZARD_SEVERITY_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_MOISTURE_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_MTBS_SEVERITY_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_NAIP_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_NAIP_IMAGERY_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_NAIP_STRUCTURE_FEATURES_JSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_NAIP_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_OSM_ROADS_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_PARCELS_EXTRA_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_PARCELS_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_PARCEL_ADDRESS_POINTS_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_PARCEL_POLYGONS_GEOJSON | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_SLOPE_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_SLOPE_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_SLOPE_VERSION | (none) | Used by backend `main.py`. |
| WF_LAYER_WHP_DATE | (none) | Used by backend `main.py`. |
| WF_LAYER_WHP_TIF | "" | Used by backend `wildfire_data.py`. |
| WF_LAYER_WHP_VERSION | (none) | Used by backend `main.py`. |
| WF_LOCAL_ADDRESS_FALLBACK_PATH | "" | Used by backend `main.py`. |
| WF_LOCAL_ADDRESS_MATCH_MIN_SCORE | "0.76" | Used by backend `address_resolution.py`. |
| WF_LOCATION_RESOLUTION_SOURCE_CONFIG | (none) | Used by backend `address_resolution.py`. |
| WF_MTBS_GPKG_PATH | "data/national/mtbs_perimeters.gpkg" | Used by backend `wildfire_data.py`. |
| WF_NAIP_FEATURE_MATCH_MAX_DISTANCE_M | "45" | Used by backend `wildfire_data.py`. |
| WF_NLCD_CACHE_DB | "data/nlcd_cache.db" | Used by backend `wildfire_data.py`. |
| WF_NLCD_COG_ENABLED | "true" | Used by backend `wildfire_data.py`. |
| WF_NO_GROUND_TRUTH_EVAL_DIR | (none) | Used by backend `no_ground_truth_paths.py`. |
| WF_OVERTURE_BUILDINGS_VERSION | "" | Used by backend `wildfire_data.py`. |
| WF_OVERTURE_RELEASE | _DEFAULT_OVERTURE_RELEASE | Used by backend `national_footprint_index.py`. |
| WF_PARCEL_CACHE_DB | "data/parcel_cache.db" | Used by backend `wildfire_data.py`. |
| WF_PARCEL_SOURCE_PRIORITY | "county_gis,open_parcel,prepared_region" | Comma-separated parcel source class priority used by `ParcelResolutionClient._resolve_source_priority`; valid classes are `county_gis`, `open_parcel`, and `prepared_region`. |
| WF_PARCEL_SOURCE_NAME | (none) | Used by backend `property_anchor.py`. |
| WF_POINT_SELECTION_MAX_SNAP_DISTANCE_M | str(getattr(footprint_client, "max_match_distance_m", 25.0) or 25.0) | Max distance (meters) for snapping a user-selected point to a footprint in `wildfire_data.py`; larger matches are rejected for point mode. |
| WF_POINT_SELECTION_MIN_SNAP_CONFIDENCE | "0.62" | Minimum confidence threshold for snapping a user-selected point to a footprint in `wildfire_data.py`; lower-confidence matches are rejected for point mode. |
| WF_POINT_SELECTION_USE_PARCEL_CONTEXT | "true" | Used by backend `wildfire_data.py`. |
| WF_PROPERTY_ANCHOR_SOURCE_PRIORITY | ",".join(DEFAULT_PRIORITY | Used by backend `property_anchor.py`. |
| WF_PUBLIC_CALIBRATION_ARTIFACT | "" | Used by backend `calibration.py`. |
| WF_PUBLIC_OUTCOME_CALIBRATION_DIR | (none) | Used by backend `public_outcome_governance.py`. |
| WF_PUBLIC_OUTCOME_VALIDATION_DIR | (none) | Used by backend `public_outcome_governance.py`. |
| WF_RATE_LIMIT_ASSESS | 10/minute | Rate limit for /risk/assess and /risk/reassess per IP. |
| WF_RATE_LIMIT_ASSESS_DAILY | 100/day | Daily rate limit for /risk/assess and /risk/reassess per IP. |
| WF_RATE_LIMIT_BYPASS_KEYS | (none) | Comma-separated API keys exempt from rate limiting. |
| WF_RATE_LIMIT_SIMULATE | 20/minute | Rate limit for /risk/simulate per IP. |
| WF_REDIS_URL | "memory://" | Optional limiter storage backend URI; defaults to in-memory storage. |
| WF_REGION_DATA_DIR | str(wildfire_data.region_data_dir | Used by backend `main.py`. |
| WF_REGION_PREP_SOURCE_CONFIG | (none) | Used by backend `main.py`. |
| WF_REGRID_API_KEY | "" | Used by backend `wildfire_data.py`. |
| WF_SELECTABLE_STRUCTURE_MAX_CANDIDATES | "80" | Used by backend `assessment_map.py`. |
| WF_STRUCTURE_ALIGNMENT_WARN_DISTANCE_M | "20.0" | Used by backend `wildfire_data.py`. |
| WF_STRUCTURE_DISPLAY_MIN_CONFIDENCE | "0.8" | Used by backend `wildfire_data.py`. |
| WF_TRUST_REFERENCE_ARTIFACT_DIR | (none) | Used by backend `trust_metadata.py`. |
| WF_USE_PREPARED_REGIONS | "true" | Used by backend `wildfire_data.py`. |
| WF_WHP_PROXY_ENABLED | "true" | Enables/disables initialization of proxy-based WHP enrichment in `wildfire_data.py`; when disabled, `_whp_client` is left unset. |
| WF_ZIP_LOCALITY_SCAN_MAX_FEATURES | "60000" | Used by backend `address_resolution.py`. |
| WILDFIRE_API_KEYS | "" | Comma-separated API keys accepted by require_api_key dependency. |
| WILDFIRE_READINESS_BONUSES_JSON | (none) | Used by backend `scoring_config.py`. |
| WILDFIRE_READINESS_PENALTIES_JSON | (none) | Used by backend `scoring_config.py`. |
| WILDFIRE_SCORING_PARAMETERS_PATH | str(DEFAULT_SCORING_PARAMETERS_PATH | Used by backend `scoring_config.py`. |
| WILDFIRE_SUBMODEL_WEIGHTS_JSON | (none) | Used by backend `scoring_config.py`. |
| WILDFIRE_VEGETATION_INDEX_PARAMS_JSON | (none) | Used by backend `scoring_config.py`. |
