from __future__ import annotations

import argparse
import json

from backend.data_prep.validate_region import validate_prepared_region


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a prepared region for runtime compatibility.\n"
            "This is an offline/admin validation pass; runtime scoring does not download GIS data."
        )
    )
    parser.add_argument("--region-id", required=True, help="Prepared region id (folder name under data/regions).")
    parser.add_argument("--out-dir", dest="region_data_dir", default=None, help="Optional region data root override.")
    parser.add_argument("--sample-lat", type=float, default=None, help="Optional sample latitude for smoke test.")
    parser.add_argument("--sample-lon", type=float, default=None, help="Optional sample longitude for smoke test.")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write validation_run_at/validation_status/runtime_compatibility_status back into manifest.json.",
    )
    args = parser.parse_args()

    result = validate_prepared_region(
        region_id=args.region_id,
        base_dir=args.region_data_dir,
        sample_lat=args.sample_lat,
        sample_lon=args.sample_lon,
        update_manifest=args.update_manifest,
    )

    print(
        json.dumps(
            {
                "region_id": result.get("region_id"),
                "manifest_path": result.get("manifest_path"),
                "validation_status": result.get("validation_status"),
                "runtime_compatibility_status": result.get("runtime_compatibility_status"),
                "ready_for_runtime": "yes" if result.get("ready_for_runtime") else "no",
                "scoring_readiness": result.get("scoring_readiness"),
                "prepared_for": {
                    "full_scoring": result.get("scoring_readiness") == "full_scoring",
                    "partial_scoring_only": result.get("scoring_readiness") == "partial_scoring_only",
                    "insufficient_data_behavior_only": result.get("scoring_readiness")
                    == "insufficient_data_behavior_only",
                },
                "footprint_ring_support": result.get("footprint_ring_support"),
                "blockers": result.get("blockers", []),
                "warnings": result.get("warnings", []),
                "sample_test": result.get("sample_test"),
                "next_action": (
                    "Ready for runtime scoring."
                    if result.get("ready_for_runtime")
                    else "Fix blockers and rerun validation before runtime use."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if not result.get("ready_for_runtime"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
