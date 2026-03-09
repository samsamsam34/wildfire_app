#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_region_from_catalog_or_sources import _parse_bbox, prepare_region_from_catalog_or_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan catalog-driven region build for a new area without writing outputs."
    )
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--bbox", nargs="+", required=True)
    parser.add_argument("--catalog-root", default=None)
    parser.add_argument("--regions-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--source-config", default=None)
    parser.add_argument("--skip-optional-layers", action="store_true")
    parser.add_argument("--allow-partial-coverage-fill", action="store_true")
    parser.add_argument("--prefer-bbox-downloads", action="store_true")
    parser.add_argument("--allow-full-download-fallback", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    plan = prepare_region_from_catalog_or_sources(
        region_id=args.region_id,
        display_name=args.display_name or args.region_id.replace("_", " ").title(),
        bounds=_parse_bbox(args.bbox),
        catalog_root=Path(args.catalog_root).expanduser() if args.catalog_root else None,
        regions_root=Path(args.regions_root).expanduser() if args.regions_root else None,
        cache_root=Path(args.cache_root).expanduser() if args.cache_root else None,
        source_config_path=args.source_config,
        skip_optional_layers=bool(args.skip_optional_layers),
        allow_partial_coverage_fill=bool(args.allow_partial_coverage_fill),
        prefer_bbox_downloads=bool(args.prefer_bbox_downloads),
        allow_full_download_fallback=bool(args.allow_full_download_fallback),
        plan_only=True,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
