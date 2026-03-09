from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_prep.catalog import (
    LAYER_TYPES,
    build_region_from_catalog,
    default_cache_root,
    default_catalog_root,
    ingest_catalog_raster,
    ingest_catalog_vector,
)
from backend.data_prep.prepare_region import parse_bbox
from backend.data_prep.validate_region import validate_prepared_region
from scripts.catalog_coverage import (
    build_catalog_coverage_plan,
    optional_layers,
    required_core_layers,
)

DEFAULT_SOURCE_REGISTRY_CANDIDATES = (
    Path("config") / "source_registry.json",
    Path("config") / "source_config.example.json",
)


def _parse_bbox(values: Sequence[str]) -> dict[str, float]:
    if len(values) == 1:
        return parse_bbox(values[0])
    if len(values) == 4:
        return parse_bbox(",".join(values))
    raise ValueError("--bbox expects one comma string or four numbers")


def _discover_default_source_config_path() -> Path | None:
    env_path = os.getenv("WF_SOURCE_CONFIG_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
    for candidate in DEFAULT_SOURCE_REGISTRY_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(v) for v in value]
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}") and len(value) > 3:
            key = value[2:-1].strip()
            return os.getenv(key, "")
        return value
    return value


def _normalize_layer_config(candidate: Any) -> dict[str, Any]:
    cfg = candidate if isinstance(candidate, dict) else {}
    cleaned = _resolve_env_refs(cfg)
    if not isinstance(cleaned, dict):
        return {}
    normalized: dict[str, Any] = {k: v for k, v in cleaned.items() if v is not None}
    for key in ("source_path", "local_path", "source_url", "full_download_url", "source_endpoint"):
        val = normalized.get(key)
        if isinstance(val, str):
            normalized[key] = val.strip()
    return normalized


def _load_source_config(path_or_none: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_path: Path | None = None
    if path_or_none:
        selected_path = Path(path_or_none).expanduser()
        if not selected_path.exists():
            raise ValueError(f"Source config not found: {selected_path}")
    else:
        selected_path = _discover_default_source_config_path()

    if not selected_path:
        return {}, {"source_config_path": None, "default_source_registry_used": False}

    with open(selected_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Source config must be a JSON object.")
    if isinstance(payload.get("layers"), dict):
        raw_layers = payload["layers"]
    else:
        raw_layers = payload
    if not isinstance(raw_layers, dict):
        raw_layers = {}
    layers = {str(k): _normalize_layer_config(v) for k, v in raw_layers.items()}
    return layers, {
        "source_config_path": str(selected_path),
        "default_source_registry_used": (not path_or_none),
    }


def _layer_config(layer_key: str, source_config: dict[str, Any]) -> dict[str, Any]:
    candidate = source_config.get(layer_key)
    return _normalize_layer_config(candidate)


def _layer_is_configured(layer_cfg: dict[str, Any]) -> bool:
    if not isinstance(layer_cfg, dict) or not layer_cfg:
        return False
    for key in ("source_path", "local_path", "source_url", "full_download_url", "source_endpoint"):
        val = layer_cfg.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def _provider_default(layer_type: str) -> str:
    if layer_type == "raster":
        return "arcgis_image_service"
    return "arcgis_feature_service"


def _ingest_layer_for_bbox(
    *,
    layer_key: str,
    bounds: dict[str, float],
    layer_cfg: dict[str, Any],
    catalog_root: Path,
    cache_root: Path,
    prefer_bbox_downloads: bool,
    allow_full_download_fallback: bool,
    target_resolution: float | None,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    force: bool,
) -> dict[str, Any]:
    layer_type = LAYER_TYPES.get(layer_key, "vector")
    provider_type = str(layer_cfg.get("provider_type") or _provider_default(layer_type))
    source_path = layer_cfg.get("source_path") or layer_cfg.get("local_path")
    source_url = layer_cfg.get("source_url") or layer_cfg.get("full_download_url")
    source_endpoint = layer_cfg.get("source_endpoint")
    per_layer_resolution = layer_cfg.get("target_resolution", target_resolution)

    if layer_type == "raster":
        metadata = ingest_catalog_raster(
            layer_name=layer_key,
            source_path=str(source_path) if source_path else None,
            source_url=str(source_url) if source_url else None,
            source_endpoint=str(source_endpoint) if source_endpoint else None,
            provider_type=provider_type,
            bounds=bounds,
            catalog_root=catalog_root,
            cache_root=cache_root,
            prefer_bbox_downloads=prefer_bbox_downloads,
            allow_full_download_fallback=allow_full_download_fallback,
            target_resolution=float(per_layer_resolution) if per_layer_resolution is not None else None,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
            force=force,
        )
    else:
        metadata = ingest_catalog_vector(
            layer_name=layer_key,
            source_path=str(source_path) if source_path else None,
            source_url=str(source_url) if source_url else None,
            source_endpoint=str(source_endpoint) if source_endpoint else None,
            provider_type=provider_type,
            bounds=bounds,
            catalog_root=catalog_root,
            cache_root=cache_root,
            prefer_bbox_downloads=prefer_bbox_downloads,
            allow_full_download_fallback=allow_full_download_fallback,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
            force=force,
        )
    return metadata


def _plan_acquisition_steps(
    *,
    coverage_plan: dict[str, Any],
    required_layers: Sequence[str],
    optional_layer_keys: Sequence[str],
    source_config: dict[str, Any],
    allow_partial_coverage_fill: bool,
) -> tuple[list[dict[str, Any]], bool, dict[str, dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    buildable_with_config = True
    diagnostics: dict[str, dict[str, Any]] = {}
    layers = coverage_plan.get("layers", {})

    for layer_key in list(required_layers) + list(optional_layer_keys):
        layer_state = layers.get(layer_key, {})
        status = str(layer_state.get("coverage_status") or "none")
        needs_fill = status == "none" or (status == "partial" and allow_partial_coverage_fill)
        cfg = _layer_config(layer_key, source_config)
        has_config = _layer_is_configured(cfg)
        layer_type = LAYER_TYPES.get(layer_key, "vector")
        diag = {
            "layer_key": layer_key,
            "layer_type": layer_type,
            "required": layer_key in required_layers,
            "coverage_status": status,
            "config_available": has_config,
            "config_status": (
                "configured"
                if has_config
                else ("missing_source_details" if bool(cfg) else "not_configured")
            ),
            "planned_action": "use_existing_catalog",
            "blocking_reason": None,
        }
        if needs_fill:
            diag["planned_action"] = "acquire_and_ingest"
            steps.append(
                {
                    "layer_key": layer_key,
                    "current_coverage_status": status,
                    "action": "acquire_and_ingest",
                    "config_available": has_config,
                }
            )
            if layer_key in required_layers and not has_config:
                buildable_with_config = False
                diag["blocking_reason"] = (
                    "Required layer missing source configuration (source_path/source_url/source_endpoint)."
                )
        else:
            steps.append(
                {
                    "layer_key": layer_key,
                    "current_coverage_status": status,
                    "action": "use_existing_catalog",
                    "config_available": has_config,
                }
            )
        diagnostics[layer_key] = diag
    return steps, buildable_with_config, diagnostics


def _annotate_manifest_with_orchestration(
    *,
    manifest: dict[str, Any],
    bounds: dict[str, float],
    coverage_before: dict[str, Any],
    coverage_after: dict[str, Any],
    acquired_layers: list[dict[str, Any]],
    failed_acquisitions: list[dict[str, Any]],
    optional_omissions: list[str],
    source_config_meta: dict[str, Any],
    required_layer_diagnostics: dict[str, Any],
    optional_layer_diagnostics: dict[str, Any],
) -> None:
    acquired_by_layer = {str(item["layer_key"]): item for item in acquired_layers if isinstance(item, dict)}
    manifest.setdefault("catalog", {})
    manifest["catalog"]["built_from_catalog"] = True
    manifest["catalog"]["requested_bbox"] = {
        "min_lon": float(bounds["min_lon"]),
        "min_lat": float(bounds["min_lat"]),
        "max_lon": float(bounds["max_lon"]),
        "max_lat": float(bounds["max_lat"]),
    }
    manifest["catalog"]["coverage_before"] = coverage_before
    manifest["catalog"]["coverage_after"] = coverage_after
    manifest["catalog"]["acquired_layers"] = acquired_layers
    manifest["catalog"]["failed_acquisitions"] = failed_acquisitions
    manifest["catalog"]["optional_omissions"] = optional_omissions
    manifest["catalog"]["source_registry"] = source_config_meta
    manifest["catalog"]["required_layer_diagnostics"] = required_layer_diagnostics
    manifest["catalog"]["optional_layer_diagnostics"] = optional_layer_diagnostics

    layer_states = coverage_after.get("layers", {})
    for layer_key, layer_meta in (manifest.get("layers") or {}).items():
        state = layer_states.get(layer_key, {})
        layer_meta["catalog_coverage_status"] = state.get("coverage_status", "unknown")
        if layer_key == "slope":
            layer_meta["catalog_source_origin"] = "derived"
            layer_meta["catalog_fetched_this_run"] = False
            continue
        acquired = acquired_by_layer.get(layer_key)
        if acquired:
            method = str(acquired.get("acquisition_method") or "")
            if method in {"bbox_export", "cached_bbox_export"}:
                origin = "newly_ingested_bbox_coverage"
            elif "full_download" in method:
                origin = "full_download_fallback"
            else:
                origin = "newly_ingested_source"
            layer_meta["catalog_source_origin"] = origin
            layer_meta["catalog_fetched_this_run"] = True
            layer_meta["catalog_acquisition_method"] = method
        else:
            layer_meta["catalog_source_origin"] = "existing_catalog_coverage"
            layer_meta["catalog_fetched_this_run"] = False


def prepare_region_from_catalog_or_sources(
    *,
    region_id: str,
    display_name: str,
    bounds: dict[str, float],
    catalog_root: Path | None = None,
    regions_root: Path | None = None,
    cache_root: Path | None = None,
    source_config: dict[str, Any] | None = None,
    source_config_path: str | None = None,
    require_core_layers: bool = True,
    skip_optional_layers: bool = False,
    validate: bool = False,
    overwrite: bool = False,
    allow_partial_coverage_fill: bool = False,
    prefer_bbox_downloads: bool = True,
    allow_full_download_fallback: bool = True,
    plan_only: bool = False,
    target_resolution: float | None = None,
    timeout_seconds: float = 60.0,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    cat_root = Path(catalog_root or default_catalog_root()).expanduser()
    reg_root = Path(regions_root or (Path("data") / "regions")).expanduser()
    cache = Path(cache_root or default_cache_root()).expanduser()
    if source_config is not None:
        cfg = source_config
        cfg_meta = {
            "source_config_path": None,
            "default_source_registry_used": False,
            "inline_source_config_used": True,
        }
    else:
        cfg, cfg_meta = _load_source_config(source_config_path)
        cfg_meta["inline_source_config_used"] = False
    required = required_core_layers()
    optional = [] if skip_optional_layers else optional_layers()

    coverage_before = build_catalog_coverage_plan(
        bounds=bounds,
        required_layers=required,
        optional_layer_keys=optional,
        catalog_root=cat_root,
    )
    planned_steps, buildable_with_config, layer_plan_diagnostics = _plan_acquisition_steps(
        coverage_plan=coverage_before,
        required_layers=required,
        optional_layer_keys=optional,
        source_config=cfg,
        allow_partial_coverage_fill=allow_partial_coverage_fill,
    )
    required_layer_diagnostics = {
        key: value for key, value in layer_plan_diagnostics.items() if key in required
    }
    optional_layer_diagnostics = {
        key: value for key, value in layer_plan_diagnostics.items() if key in optional
    }
    required_layers_covered = sorted(
        [layer for layer in required if coverage_before["layers"].get(layer, {}).get("coverage_status") == "full"]
    )
    required_layers_missing = sorted(
        [layer for layer in required if coverage_before["layers"].get(layer, {}).get("coverage_status") == "none"]
    )
    required_layers_partial = sorted(
        [layer for layer in required if coverage_before["layers"].get(layer, {}).get("coverage_status") == "partial"]
    )
    optional_layers_missing = sorted(
        [layer for layer in optional if coverage_before["layers"].get(layer, {}).get("coverage_status") == "none"]
    )
    optional_layers_partial = sorted(
        [layer for layer in optional if coverage_before["layers"].get(layer, {}).get("coverage_status") == "partial"]
    )
    layer_use_existing_catalog = sorted(
        [step["layer_key"] for step in planned_steps if step.get("action") == "use_existing_catalog"]
    )
    layer_remote_acquisition_planned = sorted(
        [step["layer_key"] for step in planned_steps if step.get("action") == "acquire_and_ingest"]
    )
    required_blockers = sorted(
        [
            layer
            for layer, diag in required_layer_diagnostics.items()
            if diag.get("planned_action") == "acquire_and_ingest" and not diag.get("config_available")
        ]
    )

    if plan_only:
        return {
            "mode": "plan_only",
            "region_id": region_id,
            "display_name": display_name,
            "requested_bbox": bounds,
            "source_registry": cfg_meta,
            "coverage_plan": coverage_before,
            "acquisition_steps": planned_steps,
            "operator_summary": {
                "required_layers_covered": required_layers_covered,
                "required_layers_partial": required_layers_partial,
                "required_layers_missing": required_layers_missing,
                "optional_layers_partial": optional_layers_partial,
                "optional_layers_missing": optional_layers_missing,
                "layers_using_existing_catalog": layer_use_existing_catalog,
                "layers_requiring_acquisition": layer_remote_acquisition_planned,
            },
            "required_layer_diagnostics": required_layer_diagnostics,
            "optional_layer_diagnostics": optional_layer_diagnostics,
            "buildable_estimate": {
                "buildable_from_existing_catalog": bool(coverage_before["summary"]["buildable_from_catalog"]),
                "buildable_with_current_config": bool(buildable_with_config),
                "required_blockers": required_blockers,
            },
            "recommended_actions": (
                coverage_before.get("summary", {}).get("recommended_actions", [])
                + [
                    (
                        f"Provide source configuration for required layer '{layer}' in "
                        f"{cfg_meta.get('source_config_path') or 'a source config file'}."
                    )
                    for layer in required_blockers
                ]
            ),
        }

    acquired_layers: list[dict[str, Any]] = []
    failed_acquisitions: list[dict[str, Any]] = []
    optional_omissions: list[str] = []
    required_failures: list[dict[str, Any]] = []

    for step in planned_steps:
        if step.get("action") != "acquire_and_ingest":
            continue
        layer_key = str(step["layer_key"])
        layer_cfg = _layer_config(layer_key, cfg)
        if not _layer_is_configured(layer_cfg):
            message = (
                f"Required layer '{layer_key}' has no usable source configuration "
                "(expected source_path/source_url/source_endpoint)."
            )
            if layer_key in required and require_core_layers:
                item = {
                    "layer_key": layer_key,
                    "failure_type": "no_source_config",
                    "error": message,
                }
                failed_acquisitions.append(item)
                required_failures.append(item)
            else:
                optional_omissions.append(layer_key)
            continue
        try:
            metadata = _ingest_layer_for_bbox(
                layer_key=layer_key,
                bounds=bounds,
                layer_cfg=layer_cfg,
                catalog_root=cat_root,
                cache_root=cache,
                prefer_bbox_downloads=prefer_bbox_downloads,
                allow_full_download_fallback=allow_full_download_fallback,
                target_resolution=target_resolution,
                timeout_seconds=timeout_seconds,
                retries=retries,
                backoff_seconds=backoff_seconds,
                force=overwrite,
            )
            acquired_layers.append(
                {
                    "layer_key": layer_key,
                    "catalog_path": metadata.get("catalog_path"),
                    "provider_type": metadata.get("provider_type"),
                    "acquisition_method": metadata.get("acquisition_method"),
                    "source_url": metadata.get("source_url"),
                    "source_endpoint": metadata.get("source_endpoint"),
                    "bbox_used": metadata.get("bbox_used"),
                    "cache_hit": bool(metadata.get("cache_hit", False)),
                }
            )
        except Exception as exc:
            item = {
                "layer_key": layer_key,
                "failure_type": "acquisition_or_ingest_failed",
                "error": str(exc),
            }
            if layer_key in required and require_core_layers:
                failed_acquisitions.append(item)
                required_failures.append(item)
            else:
                optional_omissions.append(layer_key)

    coverage_after = build_catalog_coverage_plan(
        bounds=bounds,
        required_layers=required,
        optional_layer_keys=optional,
        catalog_root=cat_root,
    )
    required_missing_after = list(coverage_after["summary"]["required_missing"])
    if failed_acquisitions:
        required_missing_after.extend(
            [str(item.get("layer_key")) for item in failed_acquisitions if str(item.get("layer_key")) in required]
        )
    required_missing_after = sorted(set(required_missing_after))

    if require_core_layers and required_missing_after:
        uncovered = [
            {
                "layer_key": layer,
                "failure_type": "coverage_incomplete_after_ingest",
                "error": "Layer is still missing or partial after acquisition.",
            }
            for layer in required_missing_after
        ]
        required_failures.extend(uncovered)
        no_cfg = sorted(
            {
                str(item.get("layer_key"))
                for item in required_failures
                if str(item.get("failure_type")) == "no_source_config"
            }
        )
        fetch_failed = sorted(
            {
                str(item.get("layer_key"))
                for item in required_failures
                if str(item.get("failure_type")) == "acquisition_or_ingest_failed"
            }
        )
        incomplete = sorted(
            {
                str(item.get("layer_key"))
                for item in required_failures
                if str(item.get("failure_type")) == "coverage_incomplete_after_ingest"
            }
        )
        guidance: list[str] = []
        if no_cfg:
            guidance.append(
                "Missing source config for: "
                + ", ".join(no_cfg)
                + f". Add these in {cfg_meta.get('source_config_path') or 'the source registry'} "
                "(source_path/source_url/source_endpoint)."
            )
        if fetch_failed:
            guidance.append("Acquisition/ingest failed for: " + ", ".join(fetch_failed) + ". Check provider endpoint/auth/network.")
        if incomplete:
            guidance.append("Coverage still incomplete for: " + ", ".join(incomplete) + ". Retry with larger bbox or alternate source.")
        raise ValueError(
            "Cannot build region due to required core layer blockers: "
            + ", ".join(required_missing_after)
            + ". "
            + " ".join(guidance)
        )

    manifest = build_region_from_catalog(
        region_id=region_id,
        display_name=display_name,
        bounds=bounds,
        catalog_root=cat_root,
        regions_root=reg_root,
        overwrite=overwrite,
        require_core_layers=require_core_layers,
        skip_optional_layers=skip_optional_layers,
        allow_partial=not require_core_layers,
        validate=False,
        target_resolution=target_resolution,
    )

    _annotate_manifest_with_orchestration(
        manifest=manifest,
        bounds=bounds,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        acquired_layers=acquired_layers,
        failed_acquisitions=failed_acquisitions,
        optional_omissions=sorted(set(optional_omissions)),
        source_config_meta=cfg_meta,
        required_layer_diagnostics={
            "required_layers": required,
            "required_failures": required_failures,
            "required_missing_after": required_missing_after,
        },
        optional_layer_diagnostics={
            "optional_layers": optional,
            "optional_omissions": sorted(set(optional_omissions)),
        },
    )

    validation_result = None
    if validate:
        validation_result = validate_prepared_region(
            region_id=region_id,
            base_dir=str(reg_root),
            update_manifest=True,
        )
        manifest.setdefault("catalog", {})
        orchestration = manifest["catalog"].setdefault("orchestration_validation", {})
        orchestration["result"] = validation_result

    manifest_path = reg_root / region_id / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    return {
        "mode": "executed",
        "region_id": region_id,
        "display_name": display_name,
        "requested_bbox": bounds,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "source_registry": cfg_meta,
        "operator_summary": {
            "required_layers_covered": required_layers_covered,
            "required_layers_partial": required_layers_partial,
            "required_layers_missing": required_layers_missing,
            "optional_layers_partial": optional_layers_partial,
            "optional_layers_missing": optional_layers_missing,
            "layers_using_existing_catalog": layer_use_existing_catalog,
            "layers_requiring_acquisition": layer_remote_acquisition_planned,
        },
        "required_layer_diagnostics": required_layer_diagnostics,
        "optional_layer_diagnostics": optional_layer_diagnostics,
        "acquired_layers": acquired_layers,
        "failed_acquisitions": failed_acquisitions,
        "required_failures": required_failures,
        "optional_omissions": sorted(set(optional_omissions)),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "validation": validation_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare any region from canonical catalog coverage, automatically acquiring and ingesting "
            "missing bbox coverage from configured sources when needed."
        )
    )
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--bbox", nargs="+", required=True)
    parser.add_argument("--catalog-root", default=None)
    parser.add_argument("--regions-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--source-config", default=None)
    parser.add_argument("--require-core-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-optional-layers", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial-coverage-fill", action="store_true")
    parser.add_argument("--prefer-bbox-downloads", action="store_true")
    parser.add_argument("--allow-full-download-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--target-resolution", type=float, default=None)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.5)
    args = parser.parse_args()

    result = prepare_region_from_catalog_or_sources(
        region_id=args.region_id,
        display_name=args.display_name or args.region_id.replace("_", " ").title(),
        bounds=_parse_bbox(args.bbox),
        catalog_root=Path(args.catalog_root).expanduser() if args.catalog_root else None,
        regions_root=Path(args.regions_root).expanduser() if args.regions_root else None,
        cache_root=Path(args.cache_root).expanduser() if args.cache_root else None,
        source_config_path=args.source_config,
        require_core_layers=bool(args.require_core_layers),
        skip_optional_layers=bool(args.skip_optional_layers),
        validate=bool(args.validate),
        overwrite=bool(args.overwrite),
        allow_partial_coverage_fill=bool(args.allow_partial_coverage_fill),
        prefer_bbox_downloads=bool(args.prefer_bbox_downloads),
        allow_full_download_fallback=bool(args.allow_full_download_fallback),
        plan_only=bool(args.plan_only),
        target_resolution=args.target_resolution,
        timeout_seconds=float(args.download_timeout),
        retries=max(0, int(args.download_retries)),
        backoff_seconds=max(0.0, float(args.retry_backoff_seconds)),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
