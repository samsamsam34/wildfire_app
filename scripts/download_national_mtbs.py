"""
One-time download script: MTBS national fire perimeters → GeoPackage.

Downloads the Monitoring Trends in Burn Severity (MTBS) national burned area
boundaries shapefile from USGS ScienceBase, extracts it, reprojects to
EPSG:4326, and saves as a spatial-indexed GeoPackage at:
    data/national/mtbs_perimeters.gpkg

Usage:
    python scripts/download_national_mtbs.py [--force]

    --force   re-download even if the GPKG is < 365 days old

Runtime:
    One-time admin operation. NOT called at assessment time.
    File size: ~374 MB download, ~100-200 MB GeoPackage.
    Memory during conversion: ~1-2 GB (full national dataset in memory).

Data source:
    USGS ScienceBase item 5e7229b8e4b01d509268afba (MTBS Burned Area Boundaries)
    Version 12.0, April 2025. Updated annually.
    URL: https://www.sciencebase.gov/catalog/item/5e7229b8e4b01d509268afba

Fields retained in output:
    Fire_ID, Fire_Name, Year, StartMonth, BurnBndAc,
    low_severity_pct, mod_severity_pct, high_severity_pct, geometry
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Output paths
OUTPUT_DIR = Path("data/national")
OUTPUT_GPKG = OUTPUT_DIR / "mtbs_perimeters.gpkg"
RAW_ZIP = OUTPUT_DIR / "mtbs_perimeters_raw.zip"

# USGS ScienceBase direct download URL for MTBS perimeter shapefile (v12.0, April 2025)
MTBS_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    "5e7229b8e4b01d509268afba"
    "?f=__disk__a5%2F0c%2F74%2Fa50c74cf6fbe3184470ed0dda1c9090750489c14"
)

# GPKG max age before re-download is triggered
MAX_AGE_DAYS = 365

# Minimum feature count for a valid national MTBS GeoPackage.
# The full dataset is ~80,000+ perimeters; anything below this threshold
# indicates a partial download, dev stub, or corrupt file.
MIN_EXPECTED_FEATURES = 25_000

# Output fields written to the GeoPackage.  The client in
# backend/national_fire_history_client.py reads these exact field names.
OUTPUT_FIELDS = [
    "Fire_ID", "Fire_Name", "Year", "StartMonth",
    "BurnBndAc", "low_severity_pct", "mod_severity_pct", "high_severity_pct",
]

# Field name mapping for MTBS v12 (April 2025) schema.
# Prior releases used different column names; this handles both.
#
# v12 schema: Event_ID, Incid_Name, Ig_Date (Timestamp), BurnBndAc,
#             Low_T / Mod_T / High_T (area in hectares by severity class)
# Legacy schema: Fire_ID, Fire_Name, Year, StartMonth, low_severity_pct, ...
#
# The mapping below normalises v12 names to the canonical output names.
# Fields absent in a given release are added as None after the rename.
_V12_RENAME: dict[str, str] = {
    "Event_ID":   "Fire_ID",
    "Incid_Name": "Fire_Name",
    # Ig_Date is handled specially below (extract year + month)
    # BurnBndAc is the same in both schema versions
    # FIRE_ID / FIRE_NAME / YEAR cover legacy all-caps variants
    "FIRE_ID":    "Fire_ID",
    "FIRE_NAME":  "Fire_Name",
    "YEAR":       "Year",
    "STARTMONTH": "StartMonth",
    "BURNBNDAC":  "BurnBndAc",
}


def _gpkg_age_days() -> float | None:
    """Return age of OUTPUT_GPKG in days, or None if it does not exist."""
    if not OUTPUT_GPKG.exists():
        return None
    mtime = OUTPUT_GPKG.stat().st_mtime
    age = (time.time() - mtime) / 86400.0
    return age


def _gpkg_feature_count(path: Path) -> int | None:
    """Return the feature count of a GeoPackage, or None if unreadable."""
    try:
        import geopandas as gpd
        return len(gpd.read_file(str(path), layer="fire_perimeters"))
    except Exception:
        try:
            import geopandas as gpd
            return len(gpd.read_file(str(path)))
        except Exception:
            return None


def _should_skip_download(force: bool) -> bool:
    """Return True only if the GPKG exists, is recent, AND has enough features.

    The previous check only tested file age — a 96 KB stub from a partial write
    or dev placeholder would pass the age check and silently block re-download.
    This version also validates feature count so truncated files are not skipped.
    """
    if force:
        return False
    if not OUTPUT_GPKG.exists():
        return False
    age_days = _gpkg_age_days() or 0.0
    if age_days > MAX_AGE_DAYS:
        return False
    # File is recent — validate feature count before accepting it.
    count = _gpkg_feature_count(OUTPUT_GPKG)
    if count is None:
        print(f"Existing GPKG is unreadable — re-downloading.")
        return False
    if count < MIN_EXPECTED_FEATURES:
        print(
            f"Existing GPKG has only {count:,} features "
            f"(expected >= {MIN_EXPECTED_FEATURES:,}) — re-downloading."
        )
        return False
    return True


def _validate_gpkg(output_path: Path) -> None:
    """Raise ValueError if the written GeoPackage looks truncated or corrupt.

    Called immediately after the GeoPackage is written.  If validation fails,
    the partial file should be deleted by the caller before exiting.
    """
    count = _gpkg_feature_count(output_path)
    if count is None:
        raise ValueError(
            f"MTBS GeoPackage validation failed: could not read {output_path}. "
            "The file may be corrupt or the write was interrupted."
        )
    if count < MIN_EXPECTED_FEATURES:
        raise ValueError(
            f"MTBS GeoPackage validation failed: {count:,} features written, "
            f"expected >= {MIN_EXPECTED_FEATURES:,}. "
            "The download may be incomplete or the source URL may have changed."
        )
    print(f"Validation passed: {count:,} features written.")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download url to dest with a simple progress indicator."""
    import urllib.request

    print(f"Downloading MTBS perimeters from ScienceBase...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * downloaded / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct:.1f}% ({mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
        print()  # newline after progress
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def _convert_to_gpkg(zip_path: Path, output_path: Path) -> dict:
    """Extract shapefile from zip and convert to spatial-indexed GeoPackage.

    Handles both the MTBS v12 (2025) schema and legacy schema variants.
    The v12 schema renames several fields and replaces severity percentages
    with per-class area totals (Low_T / Mod_T / High_T in hectares).
    """
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        raise RuntimeError("geopandas is required. Install with: pip install geopandas")

    print(f"Extracting and converting shapefile to GeoPackage...")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find shapefile(s) — use the largest (the national perimeter dataset)
        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise RuntimeError(f"No .shp file found in {zip_path}")
        shp_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        shp_path = shp_files[0]
        print(f"  Reading: {shp_path.name}")

        gdf = gpd.read_file(str(shp_path))
        print(f"  Read {len(gdf):,} features, CRS: {gdf.crs}")
        print(f"  Source columns: {[c for c in gdf.columns if c != 'geometry']}")

        # --- Step 1: rename columns to canonical output names ---
        rename_map = {c: _V12_RENAME[c] for c in gdf.columns if c in _V12_RENAME}
        if rename_map:
            gdf = gdf.rename(columns=rename_map)

        # --- Step 2: extract Year and StartMonth from Ig_Date (v12 schema) ---
        # v12 uses Ig_Date (a Timestamp column) instead of separate Year/StartMonth.
        if "Year" not in gdf.columns and "Ig_Date" in gdf.columns:
            print("  Extracting Year and StartMonth from Ig_Date (v12 schema)")
            gdf["Year"] = gdf["Ig_Date"].apply(
                lambda d: int(d.year) if pd.notna(d) and hasattr(d, "year") else None
            )
            gdf["StartMonth"] = gdf["Ig_Date"].apply(
                lambda d: int(d.month) if pd.notna(d) and hasattr(d, "month") else None
            )
        elif "Year" in gdf.columns:
            # Legacy schema: ensure Year is stored as int, not float
            gdf["Year"] = gdf["Year"].apply(
                lambda v: int(v) if v is not None and not (
                    isinstance(v, float) and v != v  # NaN check
                ) else None
            )

        # --- Step 3: compute severity percentages from v12 Low_T/Mod_T/High_T ---
        # v12 stores area (hectares) per severity class; convert to percentages.
        # Legacy schema already has low_severity_pct / mod_severity_pct / high_severity_pct.
        for out_field, src_field in [
            ("low_severity_pct", "Low_T"),
            ("mod_severity_pct", "Mod_T"),
            ("high_severity_pct", "High_T"),
        ]:
            if out_field not in gdf.columns and src_field in gdf.columns:
                # Compute as fraction of (Low_T + Mod_T + High_T) total
                total_cols = [c for c in ("Low_T", "Mod_T", "High_T") if c in gdf.columns]
                total = gdf[total_cols].clip(lower=0).sum(axis=1)
                gdf[out_field] = gdf[src_field].clip(lower=0).div(total.replace(0, float("nan"))) * 100.0
                gdf[out_field] = gdf[out_field].round(2)

        # --- Step 4: reproject to EPSG:4326 ---
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            print(f"  Reprojecting from {gdf.crs} → EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        # --- Step 5: select output fields; add missing ones as None ---
        keep = [f for f in OUTPUT_FIELDS if f in gdf.columns]
        gdf = gdf[keep + ["geometry"]].copy()
        for field in OUTPUT_FIELDS:
            if field not in gdf.columns:
                gdf[field] = None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Writing GeoPackage: {output_path}")
        gdf.to_file(str(output_path), driver="GPKG", layer="fire_perimeters")

        # --- Summary: safe year-range extraction (handles NaN) ---
        valid_years = gdf["Year"].dropna() if "Year" in gdf.columns else pd.Series([], dtype=float)
        year_min = int(valid_years.min()) if len(valid_years) > 0 else None
        year_max = int(valid_years.max()) if len(valid_years) > 0 else None
        size_mb = output_path.stat().st_size / (1024 * 1024)

        return {
            "feature_count": len(gdf),
            "year_min": year_min,
            "year_max": year_max,
            "size_mb": round(size_mb, 1),
            "crs": "EPSG:4326",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MTBS national fire perimeters.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if GPKG exists")
    parser.add_argument("--convert-only", action="store_true",
                        help="Skip download — convert the existing raw zip to GPKG")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotency check — validates feature count, not just file age.
    if not args.convert_only and _should_skip_download(args.force):
        age = _gpkg_age_days() or 0.0
        size_mb = OUTPUT_GPKG.stat().st_size / (1024 * 1024)
        count = _gpkg_feature_count(OUTPUT_GPKG)
        print(f"GPKG exists, is {age:.1f} days old, and has {count:,} features. Skipping download.")
        print(f"  Path: {OUTPUT_GPKG}  Size: {size_mb:.1f} MB")
        print("Use --force to re-download.")
        return

    t0 = time.time()

    if args.convert_only:
        if not RAW_ZIP.exists():
            print(f"ERROR: --convert-only requested but {RAW_ZIP} does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"Skipping download — converting existing zip: {RAW_ZIP}")
    else:
        # Download (~374 MB)
        print(f"Downloading MTBS national fire perimeters (~374 MB zip)...")
        _download_with_progress(MTBS_URL, RAW_ZIP)
        print(f"Download complete ({(time.time() - t0):.0f}s). Converting to GeoPackage...")

    # Convert
    result = _convert_to_gpkg(RAW_ZIP, OUTPUT_GPKG)

    # Validate before declaring success
    print("Validating output...")
    try:
        _validate_gpkg(OUTPUT_GPKG)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        # Delete the partial/invalid file so future runs are not blocked.
        try:
            OUTPUT_GPKG.unlink()
            print(f"Deleted invalid output file: {OUTPUT_GPKG}", file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)

    elapsed = time.time() - t0

    # Cleanup zip
    try:
        RAW_ZIP.unlink()
        print(f"  Cleaned up raw zip.")
    except Exception:
        pass

    print()
    print("=" * 60)
    print("MTBS GeoPackage ready:")
    print(f"  Path:          {OUTPUT_GPKG}")
    print(f"  Features:      {result['feature_count']:,}")
    print(f"  Year range:    {result['year_min']} – {result['year_max']}")
    print(f"  Size:          {result['size_mb']} MB")
    print(f"  CRS:           {result['crs']}")
    print(f"  Elapsed:       {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
