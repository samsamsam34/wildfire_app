from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.data_prep.catalog import ingest_catalog_vector
from backend.data_prep.prepare_region import parse_bbox


def _parse_bbox(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return parse_bbox(values[0])
    if len(values) == 4:
        return parse_bbox(",".join(values))
    raise ValueError("--bbox expects one comma string or four numbers")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest one canonical vector layer into data/catalog for reuse across many region builds."
    )
    parser.add_argument("--layer", required=True, help="Layer name (fire_perimeters, building_footprints, roads, etc.)")
    parser.add_argument("--source-path", default=None, help="Local source vector path")
    parser.add_argument("--source-url", default=None, help="Source URL for vector download")
    parser.add_argument("--source-endpoint", default=None, help="Provider endpoint (for bbox-first query)")
    parser.add_argument(
        "--provider-type",
        default="file_download",
        help="Provider type (arcgis_feature_service, vector_service, file_download, local_file)",
    )
    parser.add_argument("--bbox", nargs="+", default=None, help="Optional bbox for catalog chunk ingest")
    parser.add_argument("--catalog-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--prefer-bbox-downloads", action="store_true")
    parser.add_argument("--allow-full-download-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metadata = ingest_catalog_vector(
        layer_name=args.layer,
        source_path=args.source_path,
        source_url=args.source_url,
        source_endpoint=args.source_endpoint,
        provider_type=args.provider_type,
        bounds=_parse_bbox(args.bbox),
        catalog_root=Path(args.catalog_root).expanduser() if args.catalog_root else None,
        cache_root=Path(args.cache_root).expanduser() if args.cache_root else None,
        prefer_bbox_downloads=bool(args.prefer_bbox_downloads),
        allow_full_download_fallback=bool(args.allow_full_download_fallback),
        timeout_seconds=float(args.download_timeout),
        retries=max(0, int(args.download_retries)),
        backoff_seconds=max(0.0, float(args.retry_backoff_seconds)),
        force=bool(args.force),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

