from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.parse
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
from backend.region_registry import load_region_manifest
from scripts.catalog_coverage import (
    build_catalog_coverage_plan,
    optional_layers,
    required_layer_policy,
    required_core_layers,
)

DEFAULT_SOURCE_REGISTRY_CANDIDATES = (
    Path("config") / "source_registry.json",
    Path("config") / "source_config.example.json",
)
SUPPORTED_PROVIDER_TYPES = {
    "arcgis_image_service",
    "arcgis_feature_service",
    "file_download",
    "vector_service",
    "overture_buildings",
    "local_file",
}

PREFERRED_FEATURE_SERVICE_ENDPOINT_PREFIXES: dict[str, tuple[str, ...]] = {
    "building_footprints": (
        "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0",
    ),
    "fire_perimeters": (
        "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0",
    ),
}

LAYER_SOURCE_HINTS: dict[str, dict[str, Any]] = {
    "whp": {
        "layer_type": "raster",
        "env_vars": ["WF_DEFAULT_WHP_ENDPOINT", "WF_DEFAULT_WHP_FULL_URL"],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "mtbs_severity": {
        "layer_type": "raster",
        "env_vars": ["WF_DEFAULT_MTBS_SEVERITY_ENDPOINT", "WF_DEFAULT_MTBS_SEVERITY_FULL_URL"],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "gridmet_dryness": {
        "layer_type": "raster",
        "env_vars": ["WF_DEFAULT_GRIDMET_DRYNESS_ENDPOINT", "WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL"],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "naip_imagery": {
        "layer_type": "raster",
        "env_vars": ["WF_DEFAULT_NAIP_ENDPOINT", "WF_DEFAULT_NAIP_FULL_URL", "WF_DEFAULT_NAIP_PATH"],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "roads": {
        "layer_type": "vector",
        "env_vars": ["WF_DEFAULT_ROADS_ENDPOINT", "WF_DEFAULT_ROADS_FULL_URL"],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "building_footprints_overture": {
        "layer_type": "vector",
        "env_vars": [
            "WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT",
            "WF_DEFAULT_OVERTURE_BUILDINGS_URL",
            "WF_DEFAULT_OVERTURE_BUILDINGS_PATH",
        ],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "building_footprints_microsoft": {
        "layer_type": "vector",
        "env_vars": [
            "WF_DEFAULT_MICROSOFT_BUILDINGS_ENDPOINT",
            "WF_DEFAULT_MICROSOFT_BUILDINGS_URL",
            "WF_DEFAULT_MICROSOFT_BUILDINGS_PATH",
        ],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "fema_structures": {
        "layer_type": "vector",
        "env_vars": [
            "WF_DEFAULT_FEMA_STRUCTURES_ENDPOINT",
            "WF_DEFAULT_FEMA_STRUCTURES_URL",
            "WF_DEFAULT_FEMA_STRUCTURES_PATH",
        ],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "parcel_polygons": {
        "layer_type": "vector",
        "env_vars": [
            "WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT",
            "WF_DEFAULT_PARCEL_POLYGONS_URL",
            "WF_DEFAULT_PARCEL_POLYGONS_PATH",
        ],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "parcel_polygons_override": {
        "layer_type": "vector",
        "env_vars": [
            "WF_DEFAULT_PARCEL_POLYGONS_OVERRIDE_ENDPOINT",
            "WF_DEFAULT_PARCEL_POLYGONS_OVERRIDE_URL",
            "WF_DEFAULT_PARCEL_POLYGONS_OVERRIDE_PATH",
        ],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
    "parcel_address_points": {
        "layer_type": "vector",
        "env_vars": [
            "WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT",
            "WF_DEFAULT_PARCEL_ADDRESS_POINTS_URL",
            "WF_DEFAULT_PARCEL_ADDRESS_POINTS_PATH",
        ],
        "registry_keys": ["source_endpoint", "source_url", "source_path"],
    },
}

DEFAULT_OPTIONAL_LAYER_KEYS: tuple[str, ...] = (
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "roads",
)
ENRICHMENT_LAYER_KEYS: tuple[str, ...] = (
    "building_footprints_overture",
    "building_footprints_microsoft",
    "fema_structures",
    "parcel_polygons",
    "parcel_polygons_override",
    "parcel_address_points",
    "naip_imagery",
)
ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


class RegionPrepExecutionError(ValueError):
    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def _parse_bbox(values: Sequence[str]) -> dict[str, float]:
    if len(values) == 1:
        return parse_bbox(values[0])
    if len(values) == 4:
        return parse_bbox(",".join(values))
    raise ValueError("--bbox expects one comma string or four numbers")


def _coerce_bounds(bounds: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bounds, dict):
        return None
    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    if not all(k in bounds for k in keys):
        return None
    try:
        return (
            float(bounds["min_lon"]),
            float(bounds["min_lat"]),
            float(bounds["max_lon"]),
            float(bounds["max_lat"]),
        )
    except (TypeError, ValueError):
        return None


def _contains_bbox(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _inspect_existing_prepared_region(
    *,
    region_id: str,
    bounds: dict[str, float],
    regions_root: Path,
) -> dict[str, Any]:
    manifest = load_region_manifest(region_id, base_dir=str(regions_root))
    requested = _coerce_bounds(bounds)
    if manifest is None:
        return {
            "status": "not_found",
            "region_id": region_id,
            "manifest_path": str(regions_root / region_id / "manifest.json"),
            "message": "No prepared region manifest exists for this region_id.",
            "covers_requested_bbox": False,
        }

    manifest_bounds = _coerce_bounds(manifest.get("bounds"))
    if manifest_bounds is None or requested is None:
        return {
            "status": "invalid_manifest",
            "region_id": region_id,
            "manifest_path": manifest.get("_manifest_path"),
            "message": "Prepared region manifest exists but bounds are missing/invalid.",
            "covers_requested_bbox": False,
        }

    covers = _contains_bbox(manifest_bounds, requested)
    if covers:
        return {
            "status": "covered",
            "region_id": region_id,
            "manifest_path": manifest.get("_manifest_path"),
            "message": "Prepared region manifest already covers the requested bbox.",
            "covers_requested_bbox": True,
            "manifest_bounds": {
                "min_lon": manifest_bounds[0],
                "min_lat": manifest_bounds[1],
                "max_lon": manifest_bounds[2],
                "max_lat": manifest_bounds[3],
            },
        }
    return {
        "status": "present_outside_bbox",
        "region_id": region_id,
        "manifest_path": manifest.get("_manifest_path"),
        "message": "Prepared region exists but does not fully cover the requested bbox.",
        "covers_requested_bbox": False,
        "manifest_bounds": {
            "min_lon": manifest_bounds[0],
            "min_lat": manifest_bounds[1],
            "max_lon": manifest_bounds[2],
            "max_lat": manifest_bounds[3],
        },
    }


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


def _resolve_env_var_ref(value: str) -> str:
    if not (value.startswith("${") and value.endswith("}") and len(value) > 3):
        return value
    token = value[2:-1].strip()
    if not token:
        return ""
    if ":-" in token:
        key, default_value = token.split(":-", 1)
        key = key.strip()
        env_value = os.getenv(key, "")
        return env_value if env_value else default_value
    return os.getenv(token, "")


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(v) for v in value]
    if isinstance(value, str):
        return _resolve_env_var_ref(value)
    return value


def _env_reference_detail(raw_value: Any) -> dict[str, Any] | None:
    if not isinstance(raw_value, str):
        return None
    match = ENV_REF_PATTERN.match(raw_value.strip())
    if not match:
        return None
    env_key = match.group(1).strip()
    default_value = match.group(2) if match.group(2) is not None else None
    env_value = os.getenv(env_key, "")
    used_default = bool(default_value) and not bool(env_value)
    missing_without_default = default_value is None and not bool(env_value)
    return {
        "env_var": env_key,
        "default_value": default_value,
        "env_value_present": bool(env_value),
        "used_default": used_default,
        "missing_without_default": missing_without_default,
    }


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
        return {}, {
            "source_config_path": None,
            "default_source_registry_used": False,
            "source_config_candidates": [str(p) for p in DEFAULT_SOURCE_REGISTRY_CANDIDATES],
            "raw_layer_entries": {},
        }

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
    raw_entries = {
        str(k): (copy.deepcopy(v) if isinstance(v, dict) else {})
        for k, v in raw_layers.items()
    }
    layers = {str(k): _normalize_layer_config(v) for k, v in raw_layers.items()}
    return layers, {
        "source_config_path": str(selected_path),
        "default_source_registry_used": (not path_or_none),
        "source_config_candidates": [str(p) for p in DEFAULT_SOURCE_REGISTRY_CANDIDATES],
        "raw_layer_entries": raw_entries,
    }


def _layer_config(layer_key: str, source_config: dict[str, Any]) -> dict[str, Any]:
    candidate = source_config.get(layer_key)
    return _normalize_layer_config(candidate)


def _sanitize_url(url: str | None) -> str:
    text = str(url or "").strip().strip("'\"")
    while text and text[-1] in {":", ";", ","}:
        text = text[:-1].rstrip()
    return text


def _is_probably_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(_sanitize_url(value))
    return parsed.scheme in {"http", "https", "file"} and bool(parsed.netloc or parsed.path)


def _stale_endpoint_warning(layer_key: str, source_endpoint: str) -> str | None:
    expected_prefixes = PREFERRED_FEATURE_SERVICE_ENDPOINT_PREFIXES.get(layer_key, ())
    endpoint = _sanitize_url(source_endpoint).lower()
    if not expected_prefixes or not endpoint:
        return None
    if any(endpoint.startswith(prefix.lower()) for prefix in expected_prefixes):
        return None
    return "Configured endpoint may be stale; compare with current default source registry service root."


def _layer_config_guidance(layer_key: str) -> str | None:
    hint = LAYER_SOURCE_HINTS.get(layer_key)
    if not hint:
        return None
    env_vars = ", ".join(hint.get("env_vars", []))
    registry_keys = ", ".join(hint.get("registry_keys", []))
    return (
        f"Set one of [{registry_keys}] for '{layer_key}' in the source registry or via env overrides "
        f"({env_vars})."
    )


def _validate_layer_source_config(*, layer_key: str, layer_cfg: dict[str, Any]) -> dict[str, Any]:
    guidance = _layer_config_guidance(layer_key)
    if not isinstance(layer_cfg, dict) or not layer_cfg:
        message = "No source-registry entry exists for this layer."
        if guidance:
            message = f"{message} {guidance}"
        return {
            "config_present": False,
            "config_valid": False,
            "provider_type": _provider_default(LAYER_TYPES.get(layer_key, "vector")),
            "config_status": "missing_layer_entry",
            "missing_required_fields": ["provider_type", "source details"],
            "invalid_fields": [],
            "actionable_error": message,
            "advisory_warning": None,
        }

    layer_type = LAYER_TYPES.get(layer_key, "vector")
    default_provider = _provider_default(layer_type)
    provider_type = str(layer_cfg.get("provider_type") or default_provider).strip().lower()
    source_path = str(layer_cfg.get("source_path") or "").strip()
    local_path = str(layer_cfg.get("local_path") or "").strip()
    source_url = str(layer_cfg.get("source_url") or "").strip()
    full_download_url = str(layer_cfg.get("full_download_url") or "").strip()
    source_endpoint = str(layer_cfg.get("source_endpoint") or "").strip()
    endpoint_warning = _stale_endpoint_warning(layer_key, source_endpoint)

    has_local_path = bool(source_path or local_path)
    has_url = bool(source_url or full_download_url)
    has_endpoint = bool(source_endpoint)

    invalid_fields: list[str] = []
    if source_url and not _is_probably_url(source_url):
        invalid_fields.append("source_url")
    if full_download_url and not _is_probably_url(full_download_url):
        invalid_fields.append("full_download_url")
    if source_endpoint and not _is_probably_url(source_endpoint):
        invalid_fields.append("source_endpoint")
    if has_local_path:
        candidate = Path(source_path or local_path).expanduser()
        if not candidate.exists():
            invalid_fields.append("source_path")

    if provider_type not in SUPPORTED_PROVIDER_TYPES:
        message = (
            f"Unsupported provider_type '{provider_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDER_TYPES))}."
        )
        if guidance:
            message = f"{message} {guidance}"
        return {
            "config_present": True,
            "config_valid": False,
            "provider_type": provider_type or default_provider,
            "config_status": "unsupported_provider_type",
            "missing_required_fields": [],
            "invalid_fields": ["provider_type"],
            "actionable_error": message,
            "advisory_warning": endpoint_warning,
        }

    missing_required_fields: list[str] = []
    if provider_type in {"arcgis_image_service", "arcgis_feature_service"}:
        if not (has_local_path or has_endpoint or has_url):
            missing_required_fields.append("source_endpoint|source_url|source_path")
    elif provider_type in {"file_download", "vector_service", "overture_buildings"}:
        if not (has_local_path or has_url):
            missing_required_fields.append("source_url|full_download_url|source_path")
    elif provider_type == "local_file":
        if not has_local_path:
            missing_required_fields.append("source_path|local_path")

    if invalid_fields:
        message = (
            "Source entry contains invalid or unreadable fields: "
            + ", ".join(sorted(set(invalid_fields)))
        )
        if guidance:
            message = f"{message}. {guidance}"
        return {
            "config_present": True,
            "config_valid": False,
            "provider_type": provider_type,
            "config_status": "invalid_source_details",
            "missing_required_fields": missing_required_fields,
            "invalid_fields": sorted(set(invalid_fields)),
            "actionable_error": message,
            "advisory_warning": endpoint_warning,
        }

    if missing_required_fields:
        message = "Missing required source details: " + ", ".join(missing_required_fields)
        if guidance:
            message = f"{message}. {guidance}"
        return {
            "config_present": True,
            "config_valid": False,
            "provider_type": provider_type,
            "config_status": "missing_source_details",
            "missing_required_fields": missing_required_fields,
            "invalid_fields": [],
            "actionable_error": message,
            "advisory_warning": endpoint_warning,
        }

    return {
        "config_present": True,
        "config_valid": True,
        "provider_type": provider_type,
        "config_status": "configured",
        "missing_required_fields": [],
        "invalid_fields": [],
        "actionable_error": None,
        "advisory_warning": endpoint_warning,
    }


def _provider_default(layer_type: str) -> str:
    if layer_type == "raster":
        return "arcgis_image_service"
    return "arcgis_feature_service"


def _layer_source_entry_summary(
    *,
    layer_key: str,
    layer_cfg: dict[str, Any],
    source_config_meta: dict[str, Any],
) -> dict[str, Any]:
    raw_entries = source_config_meta.get("raw_layer_entries") or {}
    raw_entry = raw_entries.get(layer_key) if isinstance(raw_entries, dict) else None
    raw_entry = raw_entry if isinstance(raw_entry, dict) else {}
    entry_fields = (
        "provider_type",
        "source_path",
        "local_path",
        "source_url",
        "full_download_url",
        "source_endpoint",
    )
    resolved_entry = {
        field: layer_cfg.get(field)
        for field in entry_fields
        if str(layer_cfg.get(field) or "").strip()
    }
    raw_entry_used = {
        field: raw_entry.get(field)
        for field in entry_fields
        if str(raw_entry.get(field) or "").strip()
    }

    env_refs: list[dict[str, Any]] = []
    for field in ("source_path", "local_path", "source_url", "full_download_url", "source_endpoint"):
        detail = _env_reference_detail(raw_entry.get(field))
        if detail:
            detail["field"] = field
            env_refs.append(detail)

    defaulted_env_vars = sorted({str(item["env_var"]) for item in env_refs if item.get("used_default")})
    missing_env_vars = sorted(
        {str(item["env_var"]) for item in env_refs if item.get("missing_without_default")}
    )
    env_overrides_present = sorted(
        {str(item["env_var"]) for item in env_refs if item.get("env_value_present")}
    )
    return {
        "source_registry_entry_used": raw_entry_used,
        "source_registry_entry_resolved": resolved_entry,
        "source_registry_defaults_used": bool(defaulted_env_vars),
        "source_registry_defaulted_env_vars": defaulted_env_vars,
        "source_registry_missing_env_vars": missing_env_vars,
        "source_registry_env_overrides_present": env_overrides_present,
        "source_registry_required_env_vars": list(LAYER_SOURCE_HINTS.get(layer_key, {}).get("env_vars", [])),
        "source_registry_used_default_registry": bool(source_config_meta.get("default_source_registry_used")),
    }


def _status_followup(
    *,
    layer_key: str,
    status_classification: str,
    source_config_meta: dict[str, Any],
) -> tuple[str, bool]:
    source_path = str(source_config_meta.get("source_config_path") or "config/source_registry.json")
    if status_classification == "not_configured":
        return (
            f"Configure source details for '{layer_key}' in {source_path}, then rerun prep.",
            False,
        )
    if status_classification == "optional_and_skipped":
        return (
            f"Rerun prep without --skip-optional-layers to ingest '{layer_key}'.",
            True,
        )
    if status_classification in {"configured_but_fetch_failed", "configured_but_outside_coverage", "configured_but_empty_result"}:
        return (
            f"Verify source coverage/access for '{layer_key}' and rerun prep.",
            True,
        )
    if status_classification in {"present_from_existing_catalog", "present_after_acquisition"}:
        return ("No action required.", True)
    return ("Inspect layer diagnostics and rerun prep.", True)


def _classify_layer_state(
    *,
    layer_diag: dict[str, Any],
    coverage_status_after: str | None,
    acquired_layers: set[str],
    optional_skipped: bool = False,
) -> str:
    if optional_skipped:
        return "optional_and_skipped"

    coverage_status = str(coverage_status_after or layer_diag.get("coverage_status") or "none")
    if coverage_status in {"full", "partial"}:
        if str(layer_diag.get("layer_key") or "") in acquired_layers:
            return "present_after_acquisition"
        return "present_from_existing_catalog"

    if not bool(layer_diag.get("config_valid")):
        return "not_configured"

    failure_reason = str(layer_diag.get("failure_reason") or "")
    if failure_reason in {"empty_result"}:
        return "configured_but_empty_result"
    if failure_reason in {"outside_extent", "coverage_recording_or_recheck_failure"}:
        return "configured_but_outside_coverage"
    if failure_reason:
        return "configured_but_fetch_failed"

    return "configured_but_outside_coverage"


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

    def _run_ingest(bounds_override: dict[str, float] | None) -> dict[str, Any]:
        if layer_type == "raster":
            return ingest_catalog_raster(
                layer_name=layer_key,
                source_path=str(source_path) if source_path else None,
                source_url=str(source_url) if source_url else None,
                source_endpoint=str(source_endpoint) if source_endpoint else None,
                provider_type=provider_type,
                bounds=bounds_override,
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
        return ingest_catalog_vector(
            layer_name=layer_key,
            source_path=str(source_path) if source_path else None,
            source_url=str(source_url) if source_url else None,
            source_endpoint=str(source_endpoint) if source_endpoint else None,
            provider_type=provider_type,
            bounds=bounds_override,
            catalog_root=catalog_root,
            cache_root=cache_root,
            prefer_bbox_downloads=prefer_bbox_downloads,
            allow_full_download_fallback=allow_full_download_fallback,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
            force=force,
        )

    try:
        metadata = _run_ingest(bounds)
    except Exception as exc:
        error_text = str(exc).lower()
        if (
            provider_type == "local_file"
            and allow_full_download_fallback
            and bounds is not None
            and any(
                token in error_text
                for token in (
                    "does not intersect aoi",
                    "intersection is empty",
                    "clip window is empty",
                    "produced no intersecting features",
                    "produced no features",
                    "outside extent",
                )
            )
        ):
            metadata = _run_ingest(None)
            warnings = metadata.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            warnings.append(
                "local_file_full_source_fallback_used: source did not intersect requested bbox; ingested full source for operator diagnostics."
            )
            metadata["warnings"] = warnings
        else:
            raise
    return metadata


def _plan_acquisition_steps(
    *,
    coverage_plan: dict[str, Any],
    required_layers: Sequence[str],
    optional_layer_keys: Sequence[str],
    source_config: dict[str, Any],
    source_config_meta: dict[str, Any],
    allow_partial_coverage_fill: bool,
    skip_optional_layers: bool,
) -> tuple[list[dict[str, Any]], bool, dict[str, dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    buildable_with_config = True
    diagnostics: dict[str, dict[str, Any]] = {}
    layers = coverage_plan.get("layers", {})

    for layer_key in list(required_layers) + list(optional_layer_keys):
        is_required = layer_key in required_layers
        is_optional = layer_key in optional_layer_keys
        layer_state = layers.get(layer_key, {})
        status = str(layer_state.get("coverage_status") or "none")
        needs_fill = status == "none" or (status == "partial" and allow_partial_coverage_fill)
        cfg = _layer_config(layer_key, source_config)
        validation = _validate_layer_source_config(layer_key=layer_key, layer_cfg=cfg)
        has_config = bool(validation.get("config_valid"))
        layer_type = LAYER_TYPES.get(layer_key, "vector")
        source_entry = _layer_source_entry_summary(
            layer_key=layer_key,
            layer_cfg=cfg,
            source_config_meta=source_config_meta,
        )
        diag = {
            "layer_key": layer_key,
            "layer_type": layer_type,
            "required": is_required,
            "coverage_status": status,
            "config_present": bool(validation.get("config_present")),
            "config_available": has_config,
            "config_valid": has_config,
            "config_status": str(validation.get("config_status") or "missing_source_details"),
            "provider_type": validation.get("provider_type"),
            "missing_required_fields": list(validation.get("missing_required_fields") or []),
            "invalid_fields": list(validation.get("invalid_fields") or []),
            "actionable_error": validation.get("actionable_error"),
            "advisory_warning": validation.get("advisory_warning"),
            "planned_action": "use_existing_catalog",
            "blocking_reason": None,
            "status_classification": "present_from_existing_catalog",
            "failure_reason": None,
            "source_config_stage_required": False,
            **source_entry,
        }
        if is_optional and skip_optional_layers:
            diag["planned_action"] = "optional_and_skipped"
            diag["status_classification"] = "optional_and_skipped"
            step = {
                "layer_key": layer_key,
                "current_coverage_status": status,
                "action": "optional_and_skipped",
                "config_available": has_config,
                "config_status": diag["config_status"],
                "provider_type": diag["provider_type"],
            }
            steps.append(step)
        elif needs_fill:
            diag["planned_action"] = "acquire_and_ingest"
            diag["status_classification"] = (
                "configured_but_outside_coverage" if has_config else "not_configured"
            )
            if not has_config:
                diag["source_config_stage_required"] = True
            steps.append(
                {
                    "layer_key": layer_key,
                    "current_coverage_status": status,
                    "action": "acquire_and_ingest",
                    "config_available": has_config,
                    "config_status": diag["config_status"],
                    "provider_type": diag["provider_type"],
                }
            )
            if is_required and not has_config:
                buildable_with_config = False
                diag["blocking_reason"] = str(
                    validation.get("actionable_error")
                    or "Required layer is missing source configuration."
                )
                diag["source_config_stage_required"] = True
        else:
            steps.append(
                {
                    "layer_key": layer_key,
                    "current_coverage_status": status,
                    "action": "use_existing_catalog",
                    "config_available": has_config,
                }
            )
        next_step, rerun_ok = _status_followup(
            layer_key=layer_key,
            status_classification=str(diag["status_classification"]),
            source_config_meta=source_config_meta,
        )
        diag["operator_next_step"] = next_step
        diag["rerun_prep_only_sufficient"] = rerun_ok
        diagnostics[layer_key] = diag
    return steps, buildable_with_config, diagnostics


def _build_compact_summary(
    *,
    mode: str,
    region_id: str,
    prepared_region_status: dict[str, Any],
    required_layers: Sequence[str],
    optional_layer_keys: Sequence[str],
    coverage_summary: dict[str, Any],
    acquired_layers: Sequence[dict[str, Any]],
    failed_acquisitions: Sequence[dict[str, Any]],
    optional_omissions: Sequence[str],
    validation_result: dict[str, Any] | None,
    final_status: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "region_id": region_id,
        "final_status": final_status,
        "prepared_region_status": prepared_region_status.get("status"),
        "required_core_layers": list(required_layers),
        "optional_layers": list(optional_layer_keys),
        "required_missing_after": list(coverage_summary.get("required_missing", [])),
        "required_partial_after": list(coverage_summary.get("required_partial", [])),
        "optional_missing_after": list(coverage_summary.get("optional_missing", [])),
        "acquired_layer_count": len(list(acquired_layers)),
        "failed_acquisition_count": len(list(failed_acquisitions)),
        "optional_omission_count": len(list(optional_omissions)),
        "validation_status": (validation_result or {}).get("validation_status"),
        "ready_for_runtime": (validation_result or {}).get("ready_for_runtime"),
        "property_specific_readiness": (validation_result or {}).get("property_specific_readiness", {}).get("readiness"),
    }


def _format_bbox_for_cli(bounds: dict[str, float]) -> str:
    return (
        f"{float(bounds['min_lon']):.6f} "
        f"{float(bounds['min_lat']):.6f} "
        f"{float(bounds['max_lon']):.6f} "
        f"{float(bounds['max_lat']):.6f}"
    )


def _build_operator_next_steps(
    *,
    region_id: str,
    display_name: str,
    bounds: dict[str, float],
    source_config_meta: dict[str, Any],
    required_layer_diagnostics: dict[str, dict[str, Any]],
    optional_layer_diagnostics: dict[str, dict[str, Any]],
    skip_optional_layers: bool,
    mode: str,
) -> dict[str, Any]:
    all_diags = {**required_layer_diagnostics, **optional_layer_diagnostics}
    not_configured_layers = sorted(
        [layer for layer, diag in all_diags.items() if str(diag.get("status_classification")) == "not_configured"]
    )
    rerun_only_fixable_layers = sorted(
        [
            layer
            for layer, diag in all_diags.items()
            if str(diag.get("status_classification"))
            in {"configured_but_fetch_failed", "configured_but_outside_coverage", "configured_but_empty_result", "optional_and_skipped"}
            and bool(diag.get("rerun_prep_only_sufficient"))
        ]
    )
    required_blocked_layers = sorted(
        [
            layer
            for layer, diag in required_layer_diagnostics.items()
            if str(diag.get("status_classification")) == "not_configured"
        ]
    )
    env_overrides_needed: dict[str, list[str]] = {}
    registry_keys_needed: dict[str, list[str]] = {}
    for layer in not_configured_layers:
        hint = LAYER_SOURCE_HINTS.get(layer, {})
        env_vars = [str(v) for v in (hint.get("env_vars") or []) if str(v).strip()]
        registry_keys = [str(v) for v in (hint.get("registry_keys") or []) if str(v).strip()]
        if env_vars:
            env_overrides_needed[layer] = env_vars
        if registry_keys:
            registry_keys_needed[layer] = registry_keys

    source_path = str(source_config_meta.get("source_config_path") or "config/source_registry.json")
    base_cmd = (
        "python3 scripts/prepare_region_from_catalog_or_sources.py "
        f"--region-id {region_id} "
        f"--display-name \"{display_name}\" "
        f"--bbox {_format_bbox_for_cli(bounds)} "
        "--prefer-bbox-downloads "
        "--allow-full-download-fallback "
        "--allow-partial-coverage-fill"
    )
    if skip_optional_layers:
        base_cmd = f"{base_cmd} --skip-optional-layers"
    rerun_validate_cmd = f"{base_cmd} --validate"
    rerun_plan_cmd = f"{base_cmd} --plan-only"

    return {
        "mode": mode,
        "core_build_status": ("blocked_config" if required_blocked_layers else "ready_or_retriable"),
        "optional_enrichment_status": (
            "skipped_by_flag"
            if skip_optional_layers
            else ("config_blocked_or_partial" if not_configured_layers else "configured_or_present")
        ),
        "required_config_blockers": required_blocked_layers,
        "all_config_blockers": not_configured_layers,
        "rerun_only_fixable_layers": rerun_only_fixable_layers,
        "source_config_path": source_path,
        "required_source_env_overrides": env_overrides_needed,
        "required_source_registry_keys": registry_keys_needed,
        "next_commands": {
            "plan": rerun_plan_cmd,
            "execute_and_validate": rerun_validate_cmd,
        },
        "operator_checklist": [
            f"Configure missing source details in {source_path} for: {', '.join(not_configured_layers)}."
            if not_configured_layers
            else "No source-config blockers detected.",
            f"Rerun prep with validation: {rerun_validate_cmd}",
            "Use plan-only first if you changed source configuration.",
        ],
    }


def _build_cli_error_payload(
    *,
    exc: Exception,
    region_id: str,
    display_name: str,
    requested_bbox: dict[str, float],
    mode: str,
) -> dict[str, Any]:
    error_type = type(exc).__name__
    failure_stage = "unknown"
    issue_type = "execution_error"
    actionable_message = (
        "Check source config, provider connectivity, and required-layer coverage diagnostics."
    )

    if isinstance(exc, NameError):
        issue_type = "internal_code_error"
        failure_stage = "internal_layer_definition_reference"
        actionable_message = (
            "Internal layer-definition reference failed. Verify shared layer constants/imports "
            "before rerunning."
        )
    elif "failure_stage=coverage_incomplete_after_ingest" in str(exc):
        failure_stage = "coverage_incomplete_after_ingest"
        issue_type = "required_layer_coverage_blocker"
    elif "Source config not found" in str(exc):
        failure_stage = "source_config_load"
        issue_type = "source_config_missing"
        actionable_message = "Provide a valid --source-config path or configure the default source registry."
    elif "Required layer" in str(exc) and "source configuration" in str(exc):
        failure_stage = "required_source_configuration"
        issue_type = "required_layer_config_missing"
        actionable_message = (
            "Add source_path/source_url/source_endpoint for each missing required core layer."
        )

    payload = {
        "mode": mode,
        "final_status": "failed",
        "region_id": region_id,
        "display_name": display_name,
        "requested_bbox": requested_bbox,
        "failure_stage": failure_stage,
        "issue_type": issue_type,
        "error_type": error_type,
        "error": str(exc),
        "actionable_message": actionable_message,
    }
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        payload.update(details)
    if isinstance(exc, NameError):
        payload["missing_constant"] = str(exc).replace("name ", "").replace(" is not defined", "").strip("'")
    return payload


def _classify_execution_failure(error: str) -> tuple[str, str]:
    msg = error.lower()
    if "invalid_request_url=" in msg or "unknown url type" in msg or "invalid url" in msg:
        return "invalid_request_url", "Request URL is invalid. Check source_endpoint/source_url configuration."
    if "source configuration" in msg or "missing source details" in msg:
        return "config_validation_failed", "Layer source configuration is invalid or incomplete."
    if "http error 404" in msg or "http_status=404" in msg:
        return "endpoint_not_found", "Configured provider endpoint was not found (HTTP 404)."
    if "http error 400" in msg or "http_status=400" in msg:
        if "f=geojson" in msg:
            return "unsupported_output_format", "Provider rejected GeoJSON output format; use JSON/esri-json fallback."
        return "invalid_query", "Provider rejected request parameters (HTTP 400)."
    if "provider_http_error" in msg:
        return "provider_http_error", "Provider request failed. Check endpoint and query parameters."
    if "provider_payload_error" in msg:
        return "provider_payload_error", "Provider returned an error payload."
    if "esri_json_parse_error" in msg:
        return "esri_json_parse_error", "Provider JSON response could not be parsed into valid features."
    if "empty_result" in msg or "empty feature set" in msg:
        return "empty_result", "Provider query returned no features for this bbox."
    if "output_write_failure" in msg:
        return "output_write_failure", "Fetched features could not be written to local vector output."
    if "catalog registration failed" in msg:
        return "catalog_registration_failure", "Catalog index/registration update failed after ingest."
    if "bounds metadata failure" in msg:
        return "bounds_metadata_failure", "Catalog bounds metadata could not be computed or persisted."
    if "failed download after retries" in msg or "urlopen error" in msg or "http error" in msg:
        return "remote_provider_error", "Provider request failed. Check endpoint connectivity/auth and retry."
    if "html/error payload" in msg or "json/error payload" in msg or "json/error content" in msg:
        return "invalid_provider_payload", "Provider returned an error payload instead of layer data."
    if "no download url resolved" in msg:
        return "request_construction_failure", "Source configuration could not resolve a valid request URL."
    if "provider error" in msg and "vector source" in msg:
        return "provider_query_error", "Provider returned a vector query error payload."
    if "raster has no crs" in msg:
        return "invalid_raster_crs", "Raster payload is missing CRS metadata."
    if "does not intersect bbox" in msg:
        return "outside_extent", "Fetched data does not intersect the requested bbox."
    if "required for catalog raster operations" in msg or "required for catalog vector operations" in msg:
        return "runtime_dependency_missing", "Missing local geospatial runtime dependency during ingest."
    return "ingest_failure", "Catalog ingest failed. Check payload validity and catalog write path."


def _extract_provider_error_context(error: str) -> dict[str, Any]:
    status_code: int | None = None
    request_url: str | None = None
    response_content_type: str | None = None
    provider_error_snippet: str | None = None

    m_http = re.search(r"HTTP Error\s+(\d+)", error, flags=re.IGNORECASE)
    if m_http:
        status_code = int(m_http.group(1))
    else:
        m_status = re.search(r"http_status=(\d+)", error, flags=re.IGNORECASE)
        if m_status:
            status_code = int(m_status.group(1))

    # First prefer explicit query URLs when present.
    url_matches = re.findall(r"https?://[^\s)]+", error)
    if url_matches:
        cleaned_urls = [_sanitize_url(u.strip(".,;'\"[]")) for u in url_matches]
        query_urls = [u for u in cleaned_urls if _is_probably_url(u) and "/query?" in u.lower()]
        if query_urls:
            request_url = query_urls[-1]
        else:
            generic_urls = [u for u in cleaned_urls if _is_probably_url(u)]
            if generic_urls:
                request_url = generic_urls[-1]
    if request_url is None:
        # Fallback to explicit invalid_request_url marker.
        m_invalid = re.search(r"invalid_request_url=([^\s;)]+)", error, flags=re.IGNORECASE)
        if m_invalid:
            candidate = _sanitize_url(m_invalid.group(1).strip())
            if _is_probably_url(candidate):
                request_url = candidate

    m_ct = re.search(r"content_type=([^,)]+)", error, flags=re.IGNORECASE)
    if m_ct:
        response_content_type = m_ct.group(1).strip()

    m_body = re.search(r"body_snippet=([^)]+)\)", error, flags=re.IGNORECASE)
    if m_body:
        provider_error_snippet = m_body.group(1).strip()
    elif ":" in error:
        # Keep short and readable for top-level diagnostics.
        provider_error_snippet = error.split(":")[-1].strip()[:240]

    return {
        "response_status_code": status_code,
        "request_url": request_url,
        "response_content_type": response_content_type,
        "provider_error_snippet": provider_error_snippet,
    }


def _layer_presence(
    *,
    layer_keys: Sequence[str],
    coverage_after: dict[str, Any],
) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    layers = coverage_after.get("layers", {}) if isinstance(coverage_after, dict) else {}
    for layer_key in layer_keys:
        status = str((layers.get(layer_key) or {}).get("coverage_status") or "none")
        if status in {"full", "partial"}:
            present.append(layer_key)
        else:
            missing.append(layer_key)
    return sorted(set(present)), sorted(set(missing))


def _property_specific_readiness(
    *,
    required_layers_present: Sequence[str],
    optional_layers_present: Sequence[str],
    enrichment_layers_present: Sequence[str],
    layer_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required_set = set(required_layers_present)
    optional_set = set(optional_layers_present)
    enrichment_set = set(enrichment_layers_present)

    configured_anchor_layers = [
        layer
        for layer in ("parcel_polygons", "parcel_address_points", "building_footprints_overture")
        if bool((layer_diagnostics.get(layer) or {}).get("config_valid"))
    ]
    configured_anchor_available = bool(set(configured_anchor_layers).intersection(enrichment_set))
    anchor_signal = configured_anchor_available or not configured_anchor_layers
    hazard_signal = bool({"whp", "gridmet_dryness", "mtbs_severity"}.intersection(optional_set))
    roads_signal = "roads" in optional_set
    building_signal = "building_footprints" in required_set
    vegetation_signal = (
        ("naip_imagery" in enrichment_set)
        or ("naip_structure_features" in enrichment_set)
        or ("canopy" in required_set)
    )

    if building_signal and roads_signal and hazard_signal and vegetation_signal and anchor_signal:
        readiness = "property_specific_ready"
    elif building_signal and (roads_signal or anchor_signal):
        readiness = "address_level_only"
    else:
        readiness = "limited_regional_ready"

    missing_supporting_layers: list[str] = []
    if not roads_signal:
        missing_supporting_layers.append("roads")
    if not hazard_signal:
        missing_supporting_layers.append("whp|gridmet_dryness|mtbs_severity")
    if not vegetation_signal:
        missing_supporting_layers.append("naip_imagery|naip_structure_features|canopy")
    if configured_anchor_layers and not configured_anchor_available:
        missing_supporting_layers.extend(configured_anchor_layers)

    return {
        "readiness": readiness,
        "signals": {
            "building_footprints": building_signal,
            "roads": roads_signal,
            "hazard_context": hazard_signal,
            "near_structure_vegetation": vegetation_signal,
            "configured_anchor_context": anchor_signal,
        },
        "configured_anchor_layers": configured_anchor_layers,
        "missing_supporting_layers": sorted(set(missing_supporting_layers)),
    }


def _source_used_by_layer(
    *,
    layer_diagnostics: dict[str, dict[str, Any]],
    acquired_layers: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    acquired_map = {
        str(item.get("layer_key")): item
        for item in acquired_layers
        if isinstance(item, dict) and item.get("layer_key")
    }
    summary: dict[str, dict[str, Any]] = {}
    for layer_key, diag in layer_diagnostics.items():
        acquired = acquired_map.get(layer_key, {})
        resolved_entry = dict(diag.get("source_registry_entry_resolved") or {})
        summary[layer_key] = {
            "provider_type": diag.get("provider_type"),
            "catalog_source_origin": (
                "present_after_acquisition"
                if layer_key in acquired_map
                else "present_from_existing_catalog"
            ),
            "source_registry_entry": dict(diag.get("source_registry_entry_used") or {}),
            "source_registry_entry_resolved": resolved_entry,
            "source_endpoint": acquired.get("source_endpoint") or resolved_entry.get("source_endpoint"),
            "source_url": acquired.get("source_url") or resolved_entry.get("source_url") or resolved_entry.get("full_download_url"),
            "source_path": resolved_entry.get("source_path") or resolved_entry.get("local_path"),
            "acquisition_method": acquired.get("acquisition_method"),
            "status_classification": diag.get("status_classification"),
        }
    return summary


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
    required_layers: Sequence[str],
    optional_layers: Sequence[str],
    enrichment_layers: Sequence[str],
    layer_diagnostics: dict[str, dict[str, Any]],
    validation_result: dict[str, Any] | None,
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
    manifest["catalog"]["layer_diagnostics"] = layer_diagnostics
    manifest["catalog"]["required_layer_diagnostics"] = {
        layer: layer_diagnostics.get(layer)
        for layer in required_layers
        if layer in layer_diagnostics
    }
    manifest["catalog"]["optional_layer_diagnostics"] = {
        layer: layer_diagnostics.get(layer)
        for layer in set(optional_layers).union(set(enrichment_layers))
        if layer in layer_diagnostics
    }

    required_present, required_missing = _layer_presence(
        layer_keys=required_layers,
        coverage_after=coverage_after,
    )
    optional_present, optional_missing = _layer_presence(
        layer_keys=optional_layers,
        coverage_after=coverage_after,
    )
    enrichment_present, enrichment_missing = _layer_presence(
        layer_keys=enrichment_layers,
        coverage_after=coverage_after,
    )
    missing_reason_by_layer = {
        layer: str((layer_diagnostics.get(layer) or {}).get("status_classification") or "missing")
        for layer in sorted(set(required_missing + optional_missing + enrichment_missing))
    }
    source_used_by_layer = _source_used_by_layer(
        layer_diagnostics=layer_diagnostics,
        acquired_layers=acquired_layers,
    )
    readiness = _property_specific_readiness(
        required_layers_present=required_present,
        optional_layers_present=optional_present,
        enrichment_layers_present=enrichment_present,
        layer_diagnostics=layer_diagnostics,
    )

    manifest["catalog"]["required_layers_present"] = required_present
    manifest["catalog"]["required_layers_missing"] = required_missing
    manifest["catalog"]["optional_layers_present"] = optional_present
    manifest["catalog"]["optional_layers_missing"] = optional_missing
    manifest["catalog"]["enrichment_layers_present"] = enrichment_present
    manifest["catalog"]["enrichment_layers_missing"] = enrichment_missing
    manifest["catalog"]["missing_reason_by_layer"] = missing_reason_by_layer
    manifest["catalog"]["source_used_by_layer"] = source_used_by_layer
    manifest["catalog"]["property_specific_readiness"] = readiness
    manifest["catalog"]["validation_summary"] = (
        {
            "ready_for_runtime": validation_result.get("ready_for_runtime"),
            "validation_status": validation_result.get("validation_status"),
            "runtime_compatibility_status": validation_result.get("runtime_compatibility_status"),
            "scoring_readiness": validation_result.get("scoring_readiness"),
            "property_specific_readiness": validation_result.get("property_specific_readiness"),
        }
        if isinstance(validation_result, dict)
        else {"validation_status": "not_requested"}
    )

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
    layer_policy = required_layer_policy()
    required = list(layer_policy.get("required_core_layers") or required_core_layers())
    derived_core = list(layer_policy.get("derived_core_layers") or ["slope"])
    optional = list(layer_policy.get("optional_layers") or optional_layers())
    for enrichment_layer in ENRICHMENT_LAYER_KEYS:
        if enrichment_layer not in optional:
            optional.append(enrichment_layer)
    optional = list(dict.fromkeys(optional))
    standard_optional = [layer for layer in optional if layer in DEFAULT_OPTIONAL_LAYER_KEYS]
    enrichment_optional = [layer for layer in optional if layer in ENRICHMENT_LAYER_KEYS]
    prepared_region_status = _inspect_existing_prepared_region(
        region_id=region_id,
        bounds=bounds,
        regions_root=reg_root,
    )
    stage_status: dict[str, dict[str, Any]] = {
        "prepared_region_check": {
            "status": "ok" if prepared_region_status.get("status") in {"covered", "present_outside_bbox"} else "missing",
            "details": prepared_region_status.get("message"),
        },
        "coverage_plan": {"status": "ok", "details": None},
        "acquisition": {"status": "not_started", "details": None},
        "region_build": {"status": "not_started", "details": None},
        "validation": {"status": "not_requested", "details": None},
    }

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
        source_config_meta=cfg_meta,
        allow_partial_coverage_fill=allow_partial_coverage_fill,
        skip_optional_layers=skip_optional_layers,
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
            if diag.get("planned_action") == "acquire_and_ingest" and not diag.get("config_valid")
        ]
    )
    optional_config_warnings = sorted(
        {
            str(diag.get("actionable_error"))
            for layer, diag in optional_layer_diagnostics.items()
            if str(diag.get("planned_action")) == "acquire_and_ingest"
            and not bool(diag.get("config_valid"))
            and diag.get("actionable_error")
        }
    )
    layer_status_by_layer = {
        layer: str(diag.get("status_classification") or "unknown")
        for layer, diag in layer_plan_diagnostics.items()
    }
    operator_next_steps_plan = _build_operator_next_steps(
        region_id=region_id,
        display_name=display_name,
        bounds=bounds,
        source_config_meta=cfg_meta,
        required_layer_diagnostics=required_layer_diagnostics,
        optional_layer_diagnostics=optional_layer_diagnostics,
        skip_optional_layers=bool(skip_optional_layers),
        mode="plan_only",
    )

    if plan_only:
        recommended_actions = list(coverage_before.get("summary", {}).get("recommended_actions", []))
        if prepared_region_status.get("status") == "covered":
            recommended_actions.append("Region is already prepared and covers this bbox; execution would reuse existing files unless --overwrite is set.")
        if prepared_region_status.get("status") == "present_outside_bbox":
            recommended_actions.append("Existing region manifest does not cover the requested bbox; execution will rebuild with expanded/new coverage.")
        recommended_actions.extend(
            [
                f"Required layer '{layer}': {required_layer_diagnostics.get(layer, {}).get('actionable_error')}"
                for layer in required_blockers
            ]
        )
        recommended_actions.extend([f"Optional layer config: {msg}" for msg in optional_config_warnings])
        recommended_actions.extend(
            [
                f"{layer}: {diag.get('operator_next_step')}"
                for layer, diag in sorted(optional_layer_diagnostics.items())
                if str(diag.get("status_classification")) in {"not_configured", "configured_but_outside_coverage", "optional_and_skipped"}
            ]
        )
        compact_summary = _build_compact_summary(
            mode="plan_only",
            region_id=region_id,
            prepared_region_status=prepared_region_status,
            required_layers=required,
            optional_layer_keys=optional,
            coverage_summary=coverage_before.get("summary", {}),
            acquired_layers=[],
            failed_acquisitions=[],
            optional_omissions=[],
            validation_result=None,
            final_status="dry_run_ready" if not required_blockers else "dry_run_partial",
        )
        return {
            "mode": "plan_only",
            "region_id": region_id,
            "display_name": display_name,
            "requested_bbox": bounds,
            "required_layer_policy": {
                "required_core_layers": required,
                "derived_core_layers": derived_core,
                "optional_layers": optional,
            },
            "layer_status_by_layer": layer_status_by_layer,
            "prepared_region_status": prepared_region_status,
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
            "enrichment_summary": {
                "optional_layers": standard_optional,
                "enrichment_layers": enrichment_optional,
                "skip_optional_layers": bool(skip_optional_layers),
            },
            "required_layer_diagnostics": required_layer_diagnostics,
            "optional_layer_diagnostics": optional_layer_diagnostics,
            "buildable_estimate": {
                "buildable_from_existing_catalog": bool(coverage_before["summary"]["buildable_from_catalog"]),
                "buildable_with_current_config": bool(buildable_with_config),
                "required_blockers": required_blockers,
                "region_already_prepared": prepared_region_status.get("status") == "covered",
            },
            "optional_config_warnings": optional_config_warnings,
            "operator_next_steps": operator_next_steps_plan,
            "stage_status": stage_status,
            "recommended_actions": recommended_actions,
            "compact_summary": compact_summary,
        }

    if prepared_region_status.get("status") == "covered" and not overwrite:
        stage_status["acquisition"] = {"status": "skipped", "details": "Region already prepared for requested bbox."}
        stage_status["region_build"] = {"status": "skipped", "details": "Region already prepared; skipping rebuild."}
        validation_result = None
        if validate:
            validation_result = validate_prepared_region(
                region_id=region_id,
                base_dir=str(reg_root),
                update_manifest=True,
            )
            stage_status["validation"] = {
                "status": "ok" if validation_result.get("validation_status") == "passed" else "failed",
                "details": validation_result.get("blockers", []),
            }
        manifest_path = Path(str(prepared_region_status.get("manifest_path") or reg_root / region_id / "manifest.json"))
        manifest = load_region_manifest(region_id, base_dir=str(reg_root)) or {"region_id": region_id}
        compact_summary = _build_compact_summary(
            mode="executed",
            region_id=region_id,
            prepared_region_status=prepared_region_status,
            required_layers=required,
            optional_layer_keys=optional,
            coverage_summary=coverage_before.get("summary", {}),
            acquired_layers=[],
            failed_acquisitions=[],
            optional_omissions=[],
            validation_result=validation_result,
            final_status="already_prepared",
        )
        return {
            "mode": "executed",
            "final_status": "already_prepared",
            "region_id": region_id,
            "display_name": display_name,
            "requested_bbox": bounds,
            "required_layer_policy": {
                "required_core_layers": required,
                "derived_core_layers": derived_core,
                "optional_layers": optional,
            },
            "layer_status_by_layer": layer_status_by_layer,
            "prepared_region_status": prepared_region_status,
            "source_registry": cfg_meta,
            "coverage_before": coverage_before,
            "coverage_after": coverage_before,
            "operator_summary": {
                "required_layers_covered": required_layers_covered,
                "required_layers_partial": required_layers_partial,
                "required_layers_missing": required_layers_missing,
                "optional_layers_partial": optional_layers_partial,
                "optional_layers_missing": optional_layers_missing,
                "layers_using_existing_catalog": layer_use_existing_catalog,
                "layers_requiring_acquisition": [],
            },
            "enrichment_summary": {
                "optional_layers": standard_optional,
                "enrichment_layers": enrichment_optional,
                "skip_optional_layers": bool(skip_optional_layers),
            },
            "required_layer_diagnostics": required_layer_diagnostics,
            "optional_layer_diagnostics": optional_layer_diagnostics,
            "optional_config_warnings": optional_config_warnings,
            "operator_next_steps": _build_operator_next_steps(
                region_id=region_id,
                display_name=display_name,
                bounds=bounds,
                source_config_meta=cfg_meta,
                required_layer_diagnostics=required_layer_diagnostics,
                optional_layer_diagnostics=optional_layer_diagnostics,
                skip_optional_layers=bool(skip_optional_layers),
                mode="executed",
            ),
            "per_layer_execution_diagnostics": {},
            "acquired_layers": [],
            "failed_acquisitions": [],
            "required_failures": [],
            "optional_omissions": [],
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "validation": validation_result,
            "stage_status": stage_status,
            "compact_summary": compact_summary,
        }

    acquired_layers: list[dict[str, Any]] = []
    failed_acquisitions: list[dict[str, Any]] = []
    optional_omissions: list[str] = []
    required_failures: list[dict[str, Any]] = []
    per_layer_execution_diagnostics: dict[str, dict[str, Any]] = {}
    stage_status["acquisition"] = {"status": "running", "details": "Evaluating missing catalog coverage and ingesting required layers."}

    for step in planned_steps:
        layer_key = str(step["layer_key"])
        layer_diag = layer_plan_diagnostics.get(layer_key, {})
        if step.get("action") == "optional_and_skipped":
            layer_diag["status_classification"] = "optional_and_skipped"
            layer_diag["failure_reason"] = "optional_skipped_by_flag"
            next_step, rerun_ok = _status_followup(
                layer_key=layer_key,
                status_classification="optional_and_skipped",
                source_config_meta=cfg_meta,
            )
            layer_diag["operator_next_step"] = next_step
            layer_diag["rerun_prep_only_sufficient"] = rerun_ok
            continue
        if step.get("action") != "acquire_and_ingest":
            layer_diag["status_classification"] = "present_from_existing_catalog"
            next_step, rerun_ok = _status_followup(
                layer_key=layer_key,
                status_classification="present_from_existing_catalog",
                source_config_meta=cfg_meta,
            )
            layer_diag["operator_next_step"] = next_step
            layer_diag["rerun_prep_only_sufficient"] = rerun_ok
            continue
        layer_cfg = _layer_config(layer_key, cfg)
        provider_type = str(layer_cfg.get("provider_type") or _provider_default(LAYER_TYPES.get(layer_key, "vector")))
        source_endpoint = layer_cfg.get("source_endpoint")
        source_url = layer_cfg.get("source_url") or layer_cfg.get("full_download_url")
        request_mode = (
            "bbox_export_or_query"
            if bool(step.get("config_status") == "configured" and step.get("provider_type") in {"arcgis_image_service", "arcgis_feature_service"})
            else "full_download_or_local"
        )
        layer_execution = {
            "layer_key": layer_key,
            "provider_type": provider_type,
            "source_endpoint": source_endpoint,
            "source_url": source_url,
            "request_mode": request_mode,
            "bbox_used": dict(bounds),
            "request_url": None,
            "diagnostic_summary": None,
            "fetch_attempted": False,
            "fetch_succeeded": False,
            "response_status_code": None,
            "response_content_type": None,
            "provider_error_snippet": None,
            "advisory_warning": None,
            "output_temp_path": None,
            "catalog_ingest_attempted": False,
            "catalog_ingest_succeeded": False,
            "catalog_entry_path": None,
            "catalog_entry_id": None,
            "coverage_status_after_ingest": "none",
            "failure_reason": None,
            "actionable_error": None,
        }
        per_layer_execution_diagnostics[layer_key] = layer_execution
        validation = _validate_layer_source_config(layer_key=layer_key, layer_cfg=layer_cfg)
        endpoint_warning = validation.get("advisory_warning")
        layer_execution["advisory_warning"] = endpoint_warning
        if endpoint_warning:
            layer_execution["diagnostic_summary"] = str(endpoint_warning)
        if not bool(validation.get("config_valid")):
            message = (
                f"Required layer '{layer_key}' has invalid source configuration: "
                f"{validation.get('actionable_error') or 'missing source details'}"
            )
            reason_code, actionable = _classify_execution_failure(message)
            layer_diag["failure_reason"] = reason_code
            layer_diag["status_classification"] = "not_configured"
            layer_execution["failure_reason"] = reason_code
            layer_execution["actionable_error"] = actionable
            layer_execution["advisory_warning"] = endpoint_warning
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
            layer_execution["catalog_ingest_attempted"] = True
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
            ingest_diag = metadata.get("ingest_diagnostics") if isinstance(metadata, dict) else {}
            if not isinstance(ingest_diag, dict):
                ingest_diag = {}
            layer_execution["fetch_attempted"] = bool(ingest_diag.get("fetch_attempted", False))
            layer_execution["fetch_succeeded"] = bool(ingest_diag.get("fetch_succeeded", True))
            layer_execution["request_url"] = ingest_diag.get("request_url") or metadata.get("source_url")
            layer_execution["response_status_code"] = ingest_diag.get("http_status")
            layer_execution["response_content_type"] = ingest_diag.get("response_content_type")
            layer_execution["output_temp_path"] = ingest_diag.get("temp_input_path")
            layer_execution["catalog_ingest_succeeded"] = bool(ingest_diag.get("catalog_ingest_succeeded", True))
            layer_execution["catalog_entry_path"] = metadata.get("catalog_path")
            layer_execution["catalog_entry_id"] = metadata.get("item_id")
            if isinstance(metadata.get("warnings"), list) and metadata.get("warnings"):
                layer_execution["diagnostic_summary"] = "; ".join(str(w) for w in metadata.get("warnings", [])[:3])
            if metadata.get("acquisition_method") == "bbox_export_json_fallback" and any(
                "geojson_unsupported_fallback_to_json_succeeded" == str(w)
                for w in metadata.get("warnings", [])
            ):
                layer_execution["diagnostic_summary"] = (
                    "GeoJSON query rejected (HTTP 400); JSON fallback succeeded as expected."
                )
            if endpoint_warning:
                if layer_execution.get("diagnostic_summary"):
                    layer_execution["diagnostic_summary"] = (
                        f"{layer_execution['diagnostic_summary']}; {endpoint_warning}"
                    )
                else:
                    layer_execution["diagnostic_summary"] = str(endpoint_warning)
            layer_diag["failure_reason"] = None
            layer_diag["status_classification"] = "present_after_acquisition"
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
            error_text = str(exc)
            reason_code, actionable = _classify_execution_failure(error_text)
            context = _extract_provider_error_context(error_text)
            layer_diag["failure_reason"] = reason_code
            layer_diag["status_classification"] = _classify_layer_state(
                layer_diag={**layer_diag, "failure_reason": reason_code},
                coverage_status_after="none",
                acquired_layers=set(),
            )
            layer_execution["failure_reason"] = reason_code
            layer_execution["actionable_error"] = actionable
            layer_execution["catalog_ingest_succeeded"] = False
            layer_execution["fetch_succeeded"] = False
            # If ingest was attempted for a remote layer, count fetch as attempted even on provider/query failures.
            layer_execution["fetch_attempted"] = bool(source_endpoint or source_url)
            layer_execution["request_url"] = context.get("request_url")
            layer_execution["response_status_code"] = context.get("response_status_code")
            layer_execution["response_content_type"] = context.get("response_content_type")
            layer_execution["provider_error_snippet"] = context.get("provider_error_snippet")
            layer_execution["diagnostic_summary"] = error_text[:400]
            if endpoint_warning:
                layer_execution["diagnostic_summary"] = (
                    f"{layer_execution['diagnostic_summary']} | {endpoint_warning}"
                )
                layer_execution["actionable_error"] = (
                    f"{actionable} {endpoint_warning}"
                ).strip()
            item = {
                "layer_key": layer_key,
                "failure_type": "acquisition_or_ingest_failed",
                "error": error_text,
                "failure_reason": reason_code,
                "actionable_error": actionable,
                "request_url": context.get("request_url"),
                "response_status_code": context.get("response_status_code"),
                "response_content_type": context.get("response_content_type"),
                "provider_error_snippet": context.get("provider_error_snippet"),
                "provider_type": provider_type,
                "source_endpoint": source_endpoint,
                "source_url": source_url,
            }
            if layer_key in required and require_core_layers:
                failed_acquisitions.append(item)
                required_failures.append(item)
            else:
                optional_omissions.append(layer_key)

    acquisition_status = "ok"
    if failed_acquisitions:
        acquisition_status = "failed"
    elif optional_omissions:
        acquisition_status = "partial"
    stage_status["acquisition"] = {
        "status": acquisition_status,
        "details": {
            "acquired_layers": [item.get("layer_key") for item in acquired_layers],
            "optional_omissions": sorted(set(optional_omissions)),
            "failed_acquisitions": [item.get("layer_key") for item in failed_acquisitions],
        },
    }

    coverage_after = build_catalog_coverage_plan(
        bounds=bounds,
        required_layers=required,
        optional_layer_keys=optional,
        catalog_root=cat_root,
    )
    acquired_layer_keys = {str(item.get("layer_key")) for item in acquired_layers if isinstance(item, dict)}
    for layer_key, layer_diag in layer_plan_diagnostics.items():
        coverage_status_after = str(
            coverage_after.get("layers", {}).get(layer_key, {}).get("coverage_status")
            or layer_diag.get("coverage_status")
            or "none"
        )
        optional_skipped = str(layer_diag.get("planned_action")) == "optional_and_skipped"
        status_classification = _classify_layer_state(
            layer_diag=layer_diag,
            coverage_status_after=coverage_status_after,
            acquired_layers=acquired_layer_keys,
            optional_skipped=optional_skipped,
        )
        layer_diag["coverage_status_after_ingest"] = coverage_status_after
        layer_diag["status_classification"] = status_classification
        next_step, rerun_ok = _status_followup(
            layer_key=layer_key,
            status_classification=status_classification,
            source_config_meta=cfg_meta,
        )
        layer_diag["operator_next_step"] = next_step
        layer_diag["rerun_prep_only_sufficient"] = rerun_ok

    for layer_key, layer_diag in per_layer_execution_diagnostics.items():
        layer_diag["coverage_status_after_ingest"] = str(
            coverage_after.get("layers", {}).get(layer_key, {}).get("coverage_status") or "none"
        )
    required_missing_after = list(coverage_after["summary"]["required_missing"])
    if failed_acquisitions:
        required_missing_after.extend(
            [str(item.get("layer_key")) for item in failed_acquisitions if str(item.get("layer_key")) in required]
        )
    required_missing_after = sorted(set(required_missing_after))

    build_require_core_layers = bool(require_core_layers)
    build_allow_partial = not bool(require_core_layers)
    forced_partial_due_to_local_outside_coverage = False

    if require_core_layers and required_missing_after:
        local_outside_coverage_only = True
        for layer in required_missing_after:
            diag = per_layer_execution_diagnostics.get(layer)
            if not isinstance(diag, dict):
                local_outside_coverage_only = False
                break
            if not bool(diag.get("catalog_ingest_succeeded")):
                local_outside_coverage_only = False
                break
            if str(diag.get("provider_type") or "").strip().lower() != "local_file":
                local_outside_coverage_only = False
                break
            if str(diag.get("failure_reason") or "").strip():
                local_outside_coverage_only = False
                break

        if local_outside_coverage_only:
            forced_partial_due_to_local_outside_coverage = True
            build_require_core_layers = False
            build_allow_partial = True
            stage_status["region_build"] = {
                "status": "running",
                "details": (
                    "Required layers were ingested from local_file sources but catalog bounds do not intersect "
                    "the requested bbox; proceeding with partial region build for operator diagnostics."
                ),
            }
        else:
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
            for layer in required_missing_after:
                diag = per_layer_execution_diagnostics.get(layer)
                if isinstance(diag, dict) and not diag.get("failure_reason"):
                    diag["failure_reason"] = "coverage_recording_or_recheck_failure"
                    diag["actionable_error"] = (
                        "Layer ingest completed but catalog coverage still reports missing/partial. "
                        "Check bounds metadata/index update for this layer."
                    )
                layer_plan_diag = layer_plan_diagnostics.get(layer)
                if isinstance(layer_plan_diag, dict) and not layer_plan_diag.get("failure_reason"):
                    layer_plan_diag["failure_reason"] = "coverage_recording_or_recheck_failure"
                    layer_plan_diag["status_classification"] = "configured_but_outside_coverage"
                    next_step, rerun_ok = _status_followup(
                        layer_key=layer,
                        status_classification="configured_but_outside_coverage",
                        source_config_meta=cfg_meta,
                    )
                    layer_plan_diag["operator_next_step"] = next_step
                    layer_plan_diag["rerun_prep_only_sufficient"] = rerun_ok
            stage_failures = {
                "acquisition": sorted(
                    {
                        layer
                        for layer, diag in per_layer_execution_diagnostics.items()
                        if diag.get("failure_reason")
                        in {
                            "remote_provider_error",
                            "provider_http_error",
                            "provider_payload_error",
                            "endpoint_not_found",
                            "unsupported_output_format",
                            "invalid_query",
                            "invalid_request_url",
                            "esri_json_parse_error",
                            "empty_result",
                            "invalid_provider_payload",
                            "provider_query_error",
                            "request_construction_failure",
                        }
                    }
                ),
                "ingest": sorted(
                    {
                        layer
                        for layer, diag in per_layer_execution_diagnostics.items()
                        if diag.get("failure_reason")
                        in {
                            "invalid_raster_crs",
                            "outside_extent",
                            "runtime_dependency_missing",
                            "output_write_failure",
                            "catalog_registration_failure",
                            "bounds_metadata_failure",
                            "ingest_failure",
                        }
                    }
                ),
                "coverage_recheck": sorted(set(required_missing_after)),
            }
            stage_status["region_build"] = {
                "status": "failed",
                "details": {
                    "failure_stage": "coverage_incomplete_after_ingest",
                    "required_missing_after": required_missing_after,
                },
            }
            raise RegionPrepExecutionError(
                "Cannot build region due to required core layer blockers (failure_stage=coverage_incomplete_after_ingest): "
                + ", ".join(required_missing_after)
                + ". "
                + " ".join(guidance),
                details={
                    "failed_required_layers": sorted(set(required_missing_after)),
                    "per_layer_execution_diagnostics": per_layer_execution_diagnostics,
                    "stage_failures": stage_failures,
                    "failed_acquisitions": failed_acquisitions,
                },
            )

    stage_status["region_build"] = {"status": "running", "details": "Building region from catalog coverage."}
    manifest = build_region_from_catalog(
        region_id=region_id,
        display_name=display_name,
        bounds=bounds,
        catalog_root=cat_root,
        regions_root=reg_root,
        overwrite=overwrite,
        require_core_layers=build_require_core_layers,
        skip_optional_layers=skip_optional_layers,
        allow_partial=build_allow_partial,
        validate=False,
        target_resolution=target_resolution,
    )
    stage_status["region_build"] = {"status": "ok", "details": "Prepared region artifacts built successfully."}

    _annotate_manifest_with_orchestration(
        manifest=manifest,
        bounds=bounds,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        acquired_layers=acquired_layers,
        failed_acquisitions=failed_acquisitions,
        optional_omissions=sorted(set(optional_omissions)),
        source_config_meta=cfg_meta,
        required_layers=required,
        optional_layers=standard_optional,
        enrichment_layers=enrichment_optional,
        layer_diagnostics=layer_plan_diagnostics,
        validation_result=None,
    )

    validation_result = None
    if validate:
        stage_status["validation"] = {"status": "running", "details": "Running prepared-region validation checks."}
        validation_result = validate_prepared_region(
            region_id=region_id,
            base_dir=str(reg_root),
            update_manifest=True,
        )
        stage_status["validation"] = {
            "status": "ok" if validation_result.get("validation_status") == "passed" else "failed",
            "details": validation_result.get("blockers", []),
        }
        manifest.setdefault("catalog", {})
        orchestration = manifest["catalog"].setdefault("orchestration_validation", {})
        orchestration["result"] = validation_result
        manifest["catalog"]["validation_summary"] = {
            "ready_for_runtime": validation_result.get("ready_for_runtime"),
            "validation_status": validation_result.get("validation_status"),
            "runtime_compatibility_status": validation_result.get("runtime_compatibility_status"),
            "scoring_readiness": validation_result.get("scoring_readiness"),
            "property_specific_readiness": validation_result.get("property_specific_readiness"),
        }

    manifest_path = reg_root / region_id / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    final_status = "success"
    if optional_omissions:
        final_status = "partial"
    if forced_partial_due_to_local_outside_coverage:
        final_status = "partial"
    if validation_result and validation_result.get("validation_status") == "failed":
        final_status = "partial"
    layer_status_by_layer = {
        layer: str(diag.get("status_classification") or "unknown")
        for layer, diag in layer_plan_diagnostics.items()
    }

    compact_summary = _build_compact_summary(
        mode="executed",
        region_id=region_id,
        prepared_region_status=prepared_region_status,
        required_layers=required,
        optional_layer_keys=optional,
        coverage_summary=coverage_after.get("summary", {}),
        acquired_layers=acquired_layers,
        failed_acquisitions=failed_acquisitions,
        optional_omissions=optional_omissions,
        validation_result=validation_result,
        final_status=final_status,
    )

    return {
        "mode": "executed",
        "final_status": final_status,
        "region_id": region_id,
        "display_name": display_name,
        "requested_bbox": bounds,
        "required_layer_policy": {
            "required_core_layers": required,
            "derived_core_layers": derived_core,
            "optional_layers": optional,
        },
        "layer_status_by_layer": layer_status_by_layer,
        "prepared_region_status": prepared_region_status,
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
        "enrichment_summary": {
            "optional_layers": standard_optional,
            "enrichment_layers": enrichment_optional,
            "skip_optional_layers": bool(skip_optional_layers),
        },
        "required_layer_diagnostics": required_layer_diagnostics,
        "optional_layer_diagnostics": optional_layer_diagnostics,
        "optional_config_warnings": optional_config_warnings,
        "operator_next_steps": _build_operator_next_steps(
            region_id=region_id,
            display_name=display_name,
            bounds=bounds,
            source_config_meta=cfg_meta,
            required_layer_diagnostics=required_layer_diagnostics,
            optional_layer_diagnostics=optional_layer_diagnostics,
            skip_optional_layers=bool(skip_optional_layers),
            mode="executed",
        ),
        "per_layer_execution_diagnostics": per_layer_execution_diagnostics,
        "acquired_layers": acquired_layers,
        "failed_acquisitions": failed_acquisitions,
        "required_failures": required_failures,
        "optional_omissions": sorted(set(optional_omissions)),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "validation": validation_result,
        "stage_status": stage_status,
        "compact_summary": compact_summary,
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

    requested_bbox = _parse_bbox(args.bbox)
    try:
        result = prepare_region_from_catalog_or_sources(
            region_id=args.region_id,
            display_name=args.display_name or args.region_id.replace("_", " ").title(),
            bounds=requested_bbox,
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
    except Exception as exc:
        error_payload = _build_cli_error_payload(
            exc=exc,
            region_id=args.region_id,
            display_name=args.display_name or args.region_id.replace("_", " ").title(),
            requested_bbox=requested_bbox,
            mode=("executed" if not args.plan_only else "plan_only"),
        )
        print(json.dumps(error_payload, indent=2, sort_keys=True))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
