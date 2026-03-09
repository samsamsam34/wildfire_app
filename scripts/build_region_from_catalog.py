from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.data_prep.catalog import build_region_from_catalog
from backend.data_prep.prepare_region import parse_bbox


def _parse_bbox(values: list[str]) -> dict[str, float]:
    if len(values) == 1:
        return parse_bbox(values[0])
    if len(values) == 4:
        return parse_bbox(",".join(values))
    raise ValueError("--bbox expects one comma string or four numbers")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prepared region assets from canonical data/catalog layers (no raw-source acquisition during build)."
    )
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--bbox", nargs="+", required=True)
    parser.add_argument("--catalog-root", default=None)
    parser.add_argument("--regions-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-core-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-optional-layers", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--target-resolution", type=float, default=None)
    parser.add_argument("--compression", default="DEFLATE")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-expected-cells", type=int, default=None)
    args = parser.parse_args()

    manifest = build_region_from_catalog(
        region_id=args.region_id,
        display_name=args.display_name or args.region_id.replace("_", " ").title(),
        bounds=_parse_bbox(args.bbox),
        catalog_root=Path(args.catalog_root).expanduser() if args.catalog_root else None,
        regions_root=Path(args.regions_root).expanduser() if args.regions_root else None,
        overwrite=bool(args.overwrite),
        require_core_layers=bool(args.require_core_layers),
        skip_optional_layers=bool(args.skip_optional_layers),
        allow_partial=bool(args.allow_partial),
        target_resolution=args.target_resolution,
        validate=bool(args.validate),
        raster_compression=args.compression,
        tile_size=max(16, int(args.tile_size)),
        max_expected_cells=args.max_expected_cells,
    )
    print(
        json.dumps(
            {
                "region_id": manifest.get("region_id"),
                "final_status": manifest.get("final_status"),
                "preparation_status": manifest.get("preparation_status"),
                "prepared_layers": manifest.get("prepared_layers", []),
                "failed_layers": manifest.get("failed_layers", []),
                "required_blockers": manifest.get("required_blockers", []),
                "catalog": manifest.get("catalog", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

