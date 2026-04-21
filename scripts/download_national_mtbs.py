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

# Fields to retain from the MTBS shapefile
KEEP_FIELDS = {
    "Fire_ID", "Fire_Name", "Year", "StartMonth",
    "BurnBndAc", "low_severity_pct", "mod_severity_pct", "high_severity_pct",
}

# Alternate field name mappings (MTBS has inconsistent casing across releases)
FIELD_ALIASES: dict[str, str] = {
    "FIRE_ID": "Fire_ID",
    "FIRE_NAME": "Fire_Name",
    "YEAR": "Year",
    "STARTMONTH": "StartMonth",
    "BURNBNDAC": "BurnBndAc",
    "Ig_Date": "StartMonth",  # some releases use ignition date instead
}


def _gpkg_age_days() -> float | None:
    """Return age of OUTPUT_GPKG in days, or None if it does not exist."""
    if not OUTPUT_GPKG.exists():
        return None
    mtime = OUTPUT_GPKG.stat().st_mtime
    age = (time.time() - mtime) / 86400.0
    return age


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
    """Extract shapefile from zip and convert to spatial-indexed GeoPackage."""
    try:
        import geopandas as gpd
    except ImportError:
        raise RuntimeError("geopandas is required. Install with: pip install geopandas")

    print(f"Extracting and converting shapefile to GeoPackage...")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find shapefile(s)
        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise RuntimeError(f"No .shp file found in {zip_path}")

        # Use the largest shapefile (the national perimeter dataset)
        shp_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        shp_path = shp_files[0]
        print(f"  Reading: {shp_path.name}")

        gdf = gpd.read_file(str(shp_path))
        print(f"  Read {len(gdf):,} features, CRS: {gdf.crs}")

        # Normalize field names (handle varying MTBS capitalization)
        rename_map = {}
        for col in gdf.columns:
            if col in FIELD_ALIASES:
                rename_map[col] = FIELD_ALIASES[col]
        if rename_map:
            gdf = gdf.rename(columns=rename_map)

        # Reproject to EPSG:4326 if needed
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            print(f"  Reprojecting from {gdf.crs} → EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")

        # Keep only required fields + geometry; add missing severity fields as None
        keep = [c for c in KEEP_FIELDS if c in gdf.columns]
        missing = KEEP_FIELDS - set(gdf.columns) - {"geometry"}
        gdf = gdf[keep + ["geometry"]].copy()
        for field in missing:
            gdf[field] = None

        # Ensure Year is integer where possible
        if "Year" in gdf.columns:
            gdf["Year"] = gdf["Year"].apply(
                lambda v: int(v) if v is not None and str(v).strip().isdigit() else None
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Writing GeoPackage: {output_path}")
        gdf.to_file(str(output_path), driver="GPKG", layer="fire_perimeters")

        year_min = gdf["Year"].min() if "Year" in gdf.columns else None
        year_max = gdf["Year"].max() if "Year" in gdf.columns else None
        size_mb = output_path.stat().st_size / (1024 * 1024)

        return {
            "feature_count": len(gdf),
            "year_min": int(year_min) if year_min is not None else None,
            "year_max": int(year_max) if year_max is not None else None,
            "size_mb": round(size_mb, 1),
            "crs": "EPSG:4326",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MTBS national fire perimeters.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if GPKG exists")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotency check
    age = _gpkg_age_days()
    if age is not None and age < MAX_AGE_DAYS and not args.force:
        size_mb = OUTPUT_GPKG.stat().st_size / (1024 * 1024)
        print(f"GPKG already exists and is {age:.1f} days old (<{MAX_AGE_DAYS} day TTL). Skipping download.")
        print(f"  Path: {OUTPUT_GPKG}  Size: {size_mb:.1f} MB")
        print("Use --force to re-download.")
        return

    # Download
    _download_with_progress(MTBS_URL, RAW_ZIP)

    # Convert
    result = _convert_to_gpkg(RAW_ZIP, OUTPUT_GPKG)

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
    print("=" * 60)


if __name__ == "__main__":
    main()
