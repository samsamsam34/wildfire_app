from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.data_prep.fetchers.common import (
    BoundingBox,
    ParcelFetchResult,
    clip_and_filter_polygon_features,
    normalize_bounds,
    normalize_parcel_feature,
    write_parcel_geojson,
)
from backend.data_prep.fetchers.regrid_fetcher import RegridParcelFetcher
from backend.data_prep.sources.acquisition import ArcGISFeatureServiceProvider

LOGGER = logging.getLogger(__name__)

DEFAULT_STATE_GIS_REGISTRY_PATH = Path("config") / "state_gis_registry.json"

SUPPORTED_PARCEL_SOURCES = (
    "auto",
    "regrid",
    "overture",
    "state_gis",
    "county_search",
)


def load_state_gis_registry(path: str | Path | None = None) -> dict[str, Any]:
    chosen = Path(path) if path is not None else DEFAULT_STATE_GIS_REGISTRY_PATH
    chosen = chosen.expanduser()
    if not chosen.exists():
        return {"version": 1, "states": {}}
    try:
        payload = json.loads(chosen.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "states": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "states": {}}
    return payload


def infer_state_for_bounds(bounds: BoundingBox) -> str | None:
    """Approximate state inference from region bbox centroid for key wildfire states."""
    normalized = normalize_bounds(bounds)
    lat = (normalized["min_lat"] + normalized["max_lat"]) / 2.0
    lon = (normalized["min_lon"] + normalized["max_lon"]) / 2.0
    # Approximate bounding boxes for requested initial states.
    state_boxes = {
        "MT": (-116.1, 44.2, -104.0, 49.2),
        "WA": (-124.9, 45.4, -116.7, 49.2),
        "OR": (-124.9, 41.8, -116.4, 46.4),
        "CA": (-124.6, 32.2, -114.0, 42.2),
        "CO": (-109.2, 36.8, -101.9, 41.2),
    }
    for state, (min_lon, min_lat, max_lon, max_lat) in state_boxes.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return state
    return None


def _normalize_existing_geojson(
    *,
    geojson_path: Path,
    bounds: BoundingBox,
    source: str,
) -> list[dict[str, Any]]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else []
    if not isinstance(features, list):
        features = []
    clipped = clip_and_filter_polygon_features(features=[f for f in features if isinstance(f, dict)], bounds=bounds)
    return [
        normalize_parcel_feature(
            geometry=dict(feature.get("geometry") or {}),
            properties=dict(feature.get("properties") or {}),
            source=source,
        )
        for feature in clipped
    ]


def _fetch_arcgis_parcels(
    *,
    endpoint: str,
    bounds: BoundingBox,
    region_dir: Path,
    source_id: str,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> ParcelFetchResult:
    try:
        provider = ArcGISFeatureServiceProvider(
            endpoint=str(endpoint).strip(),
            supports_geojson_direct=False,
            preferred_response_format="json",
            require_return_geometry=True,
        )
        acquisition = provider.fetch_bbox(
            layer_key="parcel_polygons",
            bounds=normalize_bounds(bounds),
            cache_root=region_dir / ".cache" / "parcel_sources" / source_id,
            target_resolution=None,
            timeout_seconds=float(timeout_seconds),
            retries=int(retries),
            backoff_seconds=float(backoff_seconds),
        )
    except Exception as exc:
        return ParcelFetchResult(
            source_id=source_id,
            success=False,
            message=f"ArcGIS fetch failed: {exc}",
        )
    if not acquisition.local_path:
        return ParcelFetchResult(
            source_id=source_id,
            success=False,
            message="ArcGIS fetch returned no local_path.",
        )
    geojson_path = Path(acquisition.local_path)
    if not geojson_path.exists():
        return ParcelFetchResult(
            source_id=source_id,
            success=False,
            message=f"ArcGIS output missing: {geojson_path}",
        )
    try:
        normalized = _normalize_existing_geojson(
            geojson_path=geojson_path,
            bounds=normalize_bounds(bounds),
            source=source_id,
        )
    except Exception as exc:
        return ParcelFetchResult(
            source_id=source_id,
            success=False,
            message=f"ArcGIS output normalization failed: {exc}",
        )
    if not normalized:
        return ParcelFetchResult(
            source_id=source_id,
            success=False,
            message="ArcGIS endpoint returned no polygon features in bbox.",
        )
    output_path = write_parcel_geojson(
        features=normalized,
        region_dir=region_dir,
        source_id=source_id,
        bounds=normalize_bounds(bounds),
    )
    return ParcelFetchResult(
        source_id=source_id,
        success=True,
        output_path=str(output_path),
        record_count=len(normalized),
        diagnostics={
            "endpoint": endpoint,
            "acquisition_method": acquisition.acquisition_method,
            "bytes_downloaded": acquisition.bytes_downloaded,
        },
        warnings=list(acquisition.warnings or []),
    )


def _county_search_terms(region_id: str, display_name: str | None) -> list[str]:
    tokens = []
    region_token = str(region_id or "").strip().replace("_", " ")
    if region_token:
        tokens.append(region_token)
    if display_name:
        tokens.append(str(display_name).strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        cleaned = " ".join(token.split())
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            ordered.append(cleaned)
    return ordered


def _search_county_arcgis_endpoints(
    *,
    region_id: str,
    display_name: str | None,
    timeout_seconds: float,
) -> list[str]:
    endpoints: list[str] = []
    for term in _county_search_terms(region_id, display_name):
        query = f"{term} parcels FeatureServer"
        url = (
            "https://www.arcgis.com/sharing/rest/search?"
            + urllib.parse.urlencode({"q": query, "num": "25", "f": "json"})
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wildfire-app/1.0"})
            with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue
        results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            continue
        for row in results:
            if not isinstance(row, dict):
                continue
            service_url = str(row.get("url") or "").strip()
            if not service_url:
                continue
            lower = service_url.lower()
            if "featureserver" in lower:
                if lower.endswith("/featureserver"):
                    service_url = service_url.rstrip("/") + "/0"
                if service_url not in endpoints:
                    endpoints.append(service_url)
            elif "mapserver" in lower:
                if lower.endswith("/mapserver"):
                    service_url = service_url.rstrip("/") + "/0"
                if service_url not in endpoints:
                    endpoints.append(service_url)
    return endpoints


def _update_region_manifest_after_parcel_fetch(
    *,
    region_dir: Path,
    region_id: str,
    bounds: BoundingBox,
    result: ParcelFetchResult,
    attempts: list[dict[str, Any]],
) -> Path:
    manifest_path = region_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}

    files = manifest.get("files")
    if not isinstance(files, dict):
        files = {}
    files["parcel_polygons"] = "parcel_polygons.geojson"
    manifest["files"] = files
    manifest["region_id"] = str(manifest.get("region_id") or region_id)
    manifest.setdefault("bounds", normalize_bounds(bounds))
    manifest["parcel_polygons_availability"] = {
        "available": bool(result.success),
        "layer_key": "parcel_polygons",
        "path": "parcel_polygons.geojson" if result.success else None,
        "confidence_weight": 0.92 if result.success else 0.28,
        "source": result.source_id if result.success else None,
        "record_count": int(result.record_count or 0),
        "attempts": attempts,
        "warnings": list(result.warnings or []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def fetch_parcels_for_region(
    *,
    region_id: str,
    bounds: BoundingBox,
    region_dir: Path,
    parcel_source: str = "auto",
    state_code: str | None = None,
    display_name: str | None = None,
    state_registry_path: str | Path | None = None,
    timeout_seconds: float = 60.0,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    source_mode = str(parcel_source or "auto").strip().lower()
    if source_mode not in SUPPORTED_PARCEL_SOURCES:
        raise ValueError(f"Unsupported parcel_source='{parcel_source}'. Supported: {', '.join(SUPPORTED_PARCEL_SOURCES)}")

    normalized_bounds = normalize_bounds(bounds)
    inferred_state = (str(state_code or "").strip().upper() or infer_state_for_bounds(normalized_bounds))
    state_registry = load_state_gis_registry(state_registry_path)
    attempts: list[dict[str, Any]] = []
    region_dir = Path(region_dir).expanduser()
    region_dir.mkdir(parents=True, exist_ok=True)

    def _record(result: ParcelFetchResult) -> None:
        attempts.append(
            {
                "source": result.source_id,
                "success": bool(result.success),
                "record_count": int(result.record_count or 0),
                "message": result.message,
                "warnings": list(result.warnings or []),
            }
        )

    def _run_regrid() -> ParcelFetchResult:
        fetcher = RegridParcelFetcher(timeout_seconds=timeout_seconds)
        return fetcher.fetch(bounds=normalized_bounds, region_dir=region_dir)

    def _run_overture() -> ParcelFetchResult:
        if not inferred_state:
            return ParcelFetchResult(
                source_id="overture",
                success=False,
                message="State code is unknown; cannot resolve state-specific Overture parquet source.",
            )
        # Lazy import avoids loading optional pyarrow dependency unless needed.
        from backend.data_prep.fetchers.overture_fetcher import OvertureParcelFetcher

        fetcher = OvertureParcelFetcher(timeout_seconds=timeout_seconds)
        return fetcher.fetch(bounds=normalized_bounds, region_dir=region_dir, state_code=inferred_state)

    def _run_state_gis() -> ParcelFetchResult:
        if not inferred_state:
            return ParcelFetchResult(
                source_id="state_gis",
                success=False,
                message="State code is unknown; cannot query state GIS registry.",
            )
        states = state_registry.get("states") if isinstance(state_registry, dict) else {}
        state_entry = states.get(inferred_state) if isinstance(states, dict) else None
        endpoints = state_entry.get("parcel_endpoints") if isinstance(state_entry, dict) else []
        if not isinstance(endpoints, list) or not endpoints:
            return ParcelFetchResult(
                source_id="state_gis",
                success=False,
                message=f"No parcel endpoints registered for state={inferred_state}.",
            )
        last_failure: ParcelFetchResult | None = None
        for endpoint in endpoints:
            endpoint_text = str(endpoint or "").strip()
            if not endpoint_text:
                continue
            result = _fetch_arcgis_parcels(
                endpoint=endpoint_text,
                bounds=normalized_bounds,
                region_dir=region_dir,
                source_id=f"state_gis:{inferred_state}",
                timeout_seconds=timeout_seconds,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
            if result.success:
                return result
            last_failure = result
        return last_failure or ParcelFetchResult(
            source_id="state_gis",
            success=False,
            message=f"State GIS endpoints failed for state={inferred_state}.",
        )

    def _run_county_search() -> ParcelFetchResult:
        endpoints = _search_county_arcgis_endpoints(
            region_id=region_id,
            display_name=display_name,
            timeout_seconds=timeout_seconds,
        )
        if not endpoints:
            return ParcelFetchResult(
                source_id="county_search",
                success=False,
                message="No county parcel FeatureServer candidates found via ArcGIS search.",
            )
        last_failure: ParcelFetchResult | None = None
        for endpoint in endpoints:
            result = _fetch_arcgis_parcels(
                endpoint=endpoint,
                bounds=normalized_bounds,
                region_dir=region_dir,
                source_id="county_search",
                timeout_seconds=timeout_seconds,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
            if result.success:
                return result
            last_failure = result
        return last_failure or ParcelFetchResult(
            source_id="county_search",
            success=False,
            message="County search candidates failed.",
        )

    chain = {
        "regrid": _run_regrid,
        "overture": _run_overture,
        "state_gis": _run_state_gis,
        "county_search": _run_county_search,
    }
    order = ["regrid", "overture", "state_gis", "county_search"] if source_mode == "auto" else [source_mode]

    winner: ParcelFetchResult | None = None
    for source_id in order:
        result = chain[source_id]()
        _record(result)
        LOGGER.info(
            "parcel_fetch_attempt source=%s success=%s records=%s message=%s",
            source_id,
            result.success,
            result.record_count,
            result.message,
        )
        if result.success:
            winner = result
            break

    if winner is None:
        winner = ParcelFetchResult(
            source_id="none",
            success=False,
            message="All parcel fetch sources failed.",
        )

    manifest_path = _update_region_manifest_after_parcel_fetch(
        region_dir=region_dir,
        region_id=region_id,
        bounds=normalized_bounds,
        result=winner,
        attempts=attempts,
    )
    return {
        "region_id": region_id,
        "state_code": inferred_state,
        "parcel_source_requested": source_mode,
        "parcel_source_used": winner.source_id if winner.success else None,
        "success": bool(winner.success),
        "record_count": int(winner.record_count or 0),
        "output_path": winner.output_path,
        "message": winner.message,
        "warnings": list(winner.warnings or []),
        "attempts": attempts,
        "manifest_path": str(manifest_path),
        "result": asdict(winner),
    }
