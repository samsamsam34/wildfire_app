#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REGION_ID="${WF_REGION_ID:-missoula_pilot}"
DISPLAY_NAME="${WF_DISPLAY_NAME:-Missoula Pilot}"
MIN_LON="${WF_BBOX_MIN_LON:--114.30}"
MIN_LAT="${WF_BBOX_MIN_LAT:-46.70}"
MAX_LON="${WF_BBOX_MAX_LON:--113.70}"
MAX_LAT="${WF_BBOX_MAX_LAT:-47.10}"
DOWNLOAD_TIMEOUT="${WF_DOWNLOAD_TIMEOUT:-90}"
DOWNLOAD_RETRIES="${WF_DOWNLOAD_RETRIES:-3}"
VALIDATE_AFTER_PREP="${WF_VALIDATE_AFTER_PREP:-0}"

cmd=(
  "$PYTHON_BIN"
  scripts/prepare_region_from_catalog_or_sources.py
  --region-id "$REGION_ID"
  --display-name "$DISPLAY_NAME"
  --bbox "$MIN_LON" "$MIN_LAT" "$MAX_LON" "$MAX_LAT"
  --prefer-bbox-downloads
  --allow-full-download-fallback
  --allow-partial-coverage-fill
  --overwrite
  --download-timeout "$DOWNLOAD_TIMEOUT"
  --download-retries "$DOWNLOAD_RETRIES"
)

case "$VALIDATE_AFTER_PREP" in
  1|true|TRUE|yes|YES)
    cmd+=(--validate)
    ;;
esac

cd "$ROOT_DIR"
echo "Running Missoula region prep for bbox [$MIN_LON, $MIN_LAT, $MAX_LON, $MAX_LAT]..."
echo "Region: $REGION_ID  Validate: $VALIDATE_AFTER_PREP"
exec "${cmd[@]}"
