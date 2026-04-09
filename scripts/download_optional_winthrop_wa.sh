#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REGION_ID="${WF_REGION_ID:-winthrop_pilot}"
DISPLAY_NAME="${WF_DISPLAY_NAME:-Winthrop Pilot}"
MIN_LON="${WF_BBOX_MIN_LON:--120.36}"
MIN_LAT="${WF_BBOX_MIN_LAT:-48.30}"
MAX_LON="${WF_BBOX_MAX_LON:--119.98}"
MAX_LAT="${WF_BBOX_MAX_LAT:-48.67}"
DOWNLOAD_TIMEOUT="${WF_DOWNLOAD_TIMEOUT:-120}"
DOWNLOAD_RETRIES="${WF_DOWNLOAD_RETRIES:-3}"
VALIDATE_AFTER_PREP="${WF_VALIDATE_AFTER_PREP:-1}"
ALLOW_PARTIAL_FILL="${WF_ALLOW_PARTIAL_COVERAGE_FILL:-0}"

# USFS geoplatform replacements (apps.fs.usda.gov endpoints now commonly return 403).
export WF_DEFAULT_WHP_ENDPOINT="${WF_DEFAULT_WHP_ENDPOINT:-https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WildfireHazardPotentialClassified/ImageServer}"
export WF_DEFAULT_MTBS_SEVERITY_ENDPOINT="${WF_DEFAULT_MTBS_SEVERITY_ENDPOINT:-https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_MTBS_CONUS/ImageServer}"

# gridMET fm1000 annual snapshot (override this URL if you want a different year/product).
export WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL="${WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL:-https://www.northwestknowledge.net/metdata/data/fm1000_2026.nc}"

# WA-specific optional sources for parcel context.
export WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT="${WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT:-https://services.arcgis.com/jsIt88o09Q0r1j8h/ArcGIS/rest/services/Current_Parcels/FeatureServer/0}"
# Do not default parcel_address_points to parcel polygons; use a real local address-point dataset when available.
DEFAULT_OKANOGAN_ADDRESS_POINTS_PATH="${ROOT_DIR}/data/address_points/okanogan/okanogan_address_points.geojson"
if [[ -z "${WF_DEFAULT_PARCEL_ADDRESS_POINTS_PATH:-}" && -f "$DEFAULT_OKANOGAN_ADDRESS_POINTS_PATH" ]]; then
  export WF_DEFAULT_PARCEL_ADDRESS_POINTS_PATH="$DEFAULT_OKANOGAN_ADDRESS_POINTS_PATH"
fi
export WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT="${WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT:-}"

# If no Overture source is configured, use the USA structures service as a practical optional fallback.
export WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT="${WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT:-https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0}"

cmd=(
  "$PYTHON_BIN"
  scripts/prepare_region_from_catalog_or_sources.py
  --region-id "$REGION_ID"
  --display-name "$DISPLAY_NAME"
  --bbox "$MIN_LON" "$MIN_LAT" "$MAX_LON" "$MAX_LAT"
  --prefer-bbox-downloads
  --allow-full-download-fallback
  --overwrite
  --download-timeout "$DOWNLOAD_TIMEOUT"
  --download-retries "$DOWNLOAD_RETRIES"
)

case "$ALLOW_PARTIAL_FILL" in
  1|true|TRUE|yes|YES)
    cmd+=(--allow-partial-coverage-fill)
    ;;
esac

case "$VALIDATE_AFTER_PREP" in
  1|true|TRUE|yes|YES)
    cmd+=(--validate)
    ;;
esac

cd "$ROOT_DIR"
echo "Preparing ${REGION_ID} with WA optional sources for bbox [$MIN_LON, $MIN_LAT, $MAX_LON, $MAX_LAT]..."
echo "WHP endpoint: $WF_DEFAULT_WHP_ENDPOINT"
echo "MTBS endpoint: $WF_DEFAULT_MTBS_SEVERITY_ENDPOINT"
echo "gridMET URL: $WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL"
echo "Parcel endpoint: $WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT"
echo "Parcel address points endpoint: ${WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT:-<unset>}"
echo "Parcel address points path: ${WF_DEFAULT_PARCEL_ADDRESS_POINTS_PATH:-<unset>}"
echo "Overture fallback endpoint: $WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT"
exec "${cmd[@]}"
