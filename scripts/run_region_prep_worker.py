from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import AssessmentStore
from scripts.prepare_region_from_catalog_or_sources import prepare_region_from_catalog_or_sources


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _job_prepare_kwargs(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    request = job.get("request") or {}
    bbox = request.get("bbox") or job.get("requested_bbox") or {}
    region_id = str(request.get("region_id") or job.get("region_id") or "")
    display_name = str(request.get("display_name") or job.get("display_name") or region_id)

    source_config_path = args.source_config if args.source_config is not None else request.get("source_config_path")
    target_resolution = args.target_resolution if args.target_resolution is not None else _float(
        request.get("target_resolution"),
        None,
    )

    return {
        "region_id": region_id,
        "display_name": display_name,
        "bounds": {
            "min_lon": float(bbox.get("min_lon")),
            "min_lat": float(bbox.get("min_lat")),
            "max_lon": float(bbox.get("max_lon")),
            "max_lat": float(bbox.get("max_lat")),
        },
        "catalog_root": Path(args.catalog_root).expanduser() if args.catalog_root else None,
        "regions_root": Path(args.regions_root).expanduser() if args.regions_root else None,
        "cache_root": Path(args.cache_root).expanduser() if args.cache_root else None,
        "source_config_path": str(source_config_path) if source_config_path else None,
        "require_core_layers": _bool(request.get("require_core_layers"), True),
        "skip_optional_layers": _bool(request.get("skip_optional_layers"), False),
        "validate": _bool(request.get("validate"), True),
        "overwrite": _bool(request.get("overwrite"), False),
        "allow_partial_coverage_fill": _bool(request.get("allow_partial_coverage_fill"), True),
        "prefer_bbox_downloads": _bool(request.get("prefer_bbox_downloads"), True),
        "allow_full_download_fallback": _bool(request.get("allow_full_download_fallback"), True),
        "target_resolution": target_resolution,
        "timeout_seconds": float(args.download_timeout),
        "retries": int(args.download_retries),
        "backoff_seconds": float(args.retry_backoff_seconds),
    }


def run_worker(args: argparse.Namespace) -> int:
    store = AssessmentStore(args.db_path)
    processed = 0

    while True:
        if args.max_jobs is not None and processed >= args.max_jobs:
            break

        job = store.claim_next_region_prep_job()
        if not job:
            if args.once:
                break
            time.sleep(max(0.1, float(args.poll_interval)))
            continue

        job_id = str(job.get("job_id"))
        print(f"[region-prep-worker] claimed job {job_id} for region {job.get('region_id')}")
        try:
            kwargs = _job_prepare_kwargs(job, args)
            result = prepare_region_from_catalog_or_sources(**kwargs)
            manifest_path = str(result.get("manifest_path") or "")
            store.update_region_prep_job(
                job_id,
                status="completed",
                manifest_path=manifest_path or None,
                result=result,
            )
            print(
                f"[region-prep-worker] completed job {job_id} status=completed "
                f"manifest={manifest_path or 'n/a'}"
            )
        except Exception as exc:
            store.update_region_prep_job(
                job_id,
                status="failed",
                error_message=str(exc),
            )
            print(f"[region-prep-worker] failed job {job_id}: {exc}")
        processed += 1

    print(f"[region-prep-worker] exiting after processing {processed} job(s)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline worker that processes queued region prep jobs and builds prepared regions "
            "using catalog/source orchestration."
        )
    )
    parser.add_argument("--db-path", default="wildfire_app.db")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Process until queue is empty, then exit.")
    parser.add_argument("--max-jobs", type=int, default=None)

    parser.add_argument("--catalog-root", default=None)
    parser.add_argument("--regions-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--source-config", default=None)
    parser.add_argument("--target-resolution", type=float, default=None)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.5)
    args = parser.parse_args()

    exit_code = run_worker(args)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
