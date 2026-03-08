from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.data_prep.sources import default_cache_root, stage_landfire_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage LANDFIRE source assets in cache for offline/admin region preparation.\n"
            "This command is portable: local laptop today, VM/batch later with path/env overrides."
        )
    )
    parser.add_argument("--fuel-url", required=True, help="LANDFIRE fuel source URL (archive or raster).")
    parser.add_argument("--canopy-url", default=None, help="Optional LANDFIRE canopy source URL.")
    parser.add_argument("--cache-root", default=None, help="Cache root directory (default: WILDFIRE_APP_CACHE_ROOT or data/cache).")
    parser.add_argument("--download-timeout", type=float, default=45.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--force-redownload", action="store_true", help="Force archive re-download even if cached.")
    parser.add_argument("--force-reextract", action="store_true", help="Force LANDFIRE raster re-extraction from archive.")
    parser.add_argument("--fuel-checksum", default=None, help="Optional fuel checksum (sha256:<hex>).")
    parser.add_argument("--canopy-checksum", default=None, help="Optional canopy checksum (sha256:<hex>).")
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser() if args.cache_root else default_cache_root()
    result = stage_landfire_assets(
        fuel_url=args.fuel_url,
        canopy_url=args.canopy_url,
        cache_root=cache_root,
        timeout_seconds=args.download_timeout,
        retries=max(0, int(args.download_retries)),
        retry_backoff_seconds=max(0.0, float(args.retry_backoff_seconds)),
        force_redownload=bool(args.force_redownload),
        force_reextract=bool(args.force_reextract),
        checksums={
            "fuel": args.fuel_checksum,
            "canopy": args.canopy_checksum,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
