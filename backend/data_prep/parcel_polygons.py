from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.data_prep.sources.acquisition import ArcGISFeatureServiceProvider
from backend.region_registry import get_region_data_dir

try:
    from shapely.geometry import box, mapping, shape
except Exception:  # pragma: no cover - optional dependency
    box = None
    mapping = None
    shape = None


MISSOULA_PILOT_BBOX = {
    "min_lat": 46.80,
    "max_lat": 46.95,
    "min_lon": -114.20,
    "max_lon": -113.85,
}

MISSOULA_COUNTY_PARCEL_ENDPOINT = (
    "https://gis.missoulacounty.us/arcgis/rest/services/Base/Cadastral/MapServer/23"
)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _normalize_bounds(bounds: dict[str, float] | None) -> dict[str, float]:
    chosen = dict(bounds or MISSOULA_PILOT_BBOX)
    return {
        "min_lon": float(chosen["min_lon"]),
        "min_lat": float(chosen["min_lat"]),
        "max_lon": float(chosen["max_lon"]),
        "max_lat": float(chosen["max_lat"]),
    }


def _bbox_intersects_bounds(
    *,
    bbox: tuple[float, float, float, float],
    bounds: dict[str, float],
) -> bool:
    minx, miny, maxx, maxy = bbox
    return not (
        maxx < bounds["min_lon"]
        or minx > bounds["max_lon"]
        or maxy < bounds["min_lat"]
        or miny > bounds["max_lat"]
    )


def _coords_bounds(coords: Any) -> tuple[float, float, float, float] | None:
    flat: list[tuple[float, float]] = []

    def _walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                flat.append((float(value[0]), float(value[1])))
                return
            for part in value:
                _walk(part)

    _walk(coords)
    if not flat:
        return None
    minx = min(pt[0] for pt in flat)
    maxx = max(pt[0] for pt in flat)
    miny = min(pt[1] for pt in flat)
    maxy = max(pt[1] for pt in flat)
    return (minx, miny, maxx, maxy)


def _clip_polygon_features(
    *,
    source_payload: dict[str, Any],
    bounds: dict[str, float],
) -> list[dict[str, Any]]:
    features = source_payload.get("features") if isinstance(source_payload, dict) else []
    if not isinstance(features, list):
        return []

    clipped: list[dict[str, Any]] = []
    aoi = None
    if box is not None:
        aoi = box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])

    for row in features:
        if not isinstance(row, dict):
            continue
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            continue
        geometry_type = str(geometry.get("type") or "")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue

        props = dict(row.get("properties") or {})

        # Preferred clip path when shapely is available.
        if aoi is not None and shape is not None and mapping is not None:
            try:
                geom = shape(geometry)
            except Exception:
                continue
            if geom.is_empty or not geom.intersects(aoi):
                continue
            clipped_geom = geom.intersection(aoi)
            if clipped_geom.is_empty:
                continue
            if str(getattr(clipped_geom, "geom_type", "")) not in {"Polygon", "MultiPolygon"}:
                continue
            clipped.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(clipped_geom),
                }
            )
            continue

        # Fallback: bbox-only intersection check.
        bbox = _coords_bounds(geometry.get("coordinates"))
        if bbox is None:
            continue
        if not _bbox_intersects_bounds(bbox=bbox, bounds=bounds):
            continue
        clipped.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": geometry,
            }
        )
    return clipped


def _write_geojson(
    *,
    features: list[dict[str, Any]],
    bounds: dict[str, float],
    output_path: Path,
    source_endpoint: str,
) -> dict[str, Any]:
    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source_endpoint": source_endpoint,
            "clipped_to_bbox": bounds,
            "record_count": len(features),
            "generated_at": _now_iso(),
        },
        "features": features,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload["metadata"]


def _update_region_manifest_parcel_polygons(
    *,
    region_dir: Path,
    region_id: str,
    bounds: dict[str, float],
    endpoint: str,
    output_filename: str,
    record_count: int,
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

    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    files = dict(files)
    files["parcel_polygons"] = output_filename
    manifest["files"] = files
    manifest.setdefault("region_id", region_id)
    manifest.setdefault("display_name", region_id.replace("_", " ").title())
    manifest.setdefault("bounds", bounds)
    manifest["parcel_polygons_availability"] = {
        "available": True,
        "layer_key": "parcel_polygons",
        "path": output_filename,
        "record_count": int(record_count),
        "source_endpoint": endpoint,
        "confidence_weight": 0.92,
        "updated_at": _now_iso(),
    }
    manifest.setdefault("notes", [])
    notes = list(manifest.get("notes") or [])
    note = "Parcel polygons are available and support parcel-first structure matching."
    if note not in notes:
        notes.append(note)
    manifest["notes"] = notes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def download_and_clip_missoula_parcel_polygons(
    *,
    region_id: str = "missoula_pilot",
    regions_root: str | Path | None = None,
    endpoint: str | None = None,
    bounds: dict[str, float] | None = None,
    timeout_seconds: float = 45.0,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    """Download Missoula parcel polygons and write region-clipped GeoJSON.

    Uses ArcGIS JSON query pagination (resultOffset/resultRecordCount) through
    ``ArcGISFeatureServiceProvider`` to reliably fetch all features.
    """
    chosen_endpoint = str(
        endpoint
        or os.getenv("WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT")
        or MISSOULA_COUNTY_PARCEL_ENDPOINT
    ).strip()
    if not chosen_endpoint:
        raise ValueError(
            "Missing parcel endpoint. Set WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT or pass endpoint explicitly."
        )

    chosen_bounds = _normalize_bounds(bounds)
    region_root = Path(regions_root) if regions_root is not None else get_region_data_dir()
    region_root = Path(region_root).expanduser()
    region_dir = region_root / str(region_id).strip()
    region_dir.mkdir(parents=True, exist_ok=True)

    provider = ArcGISFeatureServiceProvider(
        endpoint=chosen_endpoint,
        supports_geojson_direct=False,
        preferred_response_format="json",
        require_return_geometry=True,
    )
    acquisition = provider.fetch_bbox(
        layer_key="parcel_polygons",
        bounds=chosen_bounds,
        cache_root=region_dir / ".cache" / "parcel_polygons",
        target_resolution=None,
        timeout_seconds=float(timeout_seconds),
        retries=int(retries),
        backoff_seconds=float(backoff_seconds),
    )
    if not acquisition.local_path:
        raise ValueError("Parcel polygon download produced no local output path.")
    downloaded_path = Path(acquisition.local_path)
    if not downloaded_path.exists():
        raise ValueError(f"Parcel polygon download output is missing: {downloaded_path}")

    source_payload = json.loads(downloaded_path.read_text(encoding="utf-8"))
    clipped_features = _clip_polygon_features(
        source_payload=source_payload if isinstance(source_payload, dict) else {},
        bounds=chosen_bounds,
    )
    if not clipped_features:
        raise ValueError("Parcel polygon download succeeded, but clipping yielded no intersecting parcel polygons.")

    output_filename = "parcel_polygons.geojson"
    output_path = region_dir / output_filename
    metadata = _write_geojson(
        features=clipped_features,
        bounds=chosen_bounds,
        output_path=output_path,
        source_endpoint=chosen_endpoint,
    )
    manifest_path = _update_region_manifest_parcel_polygons(
        region_dir=region_dir,
        region_id=str(region_id).strip(),
        bounds=chosen_bounds,
        endpoint=chosen_endpoint,
        output_filename=output_filename,
        record_count=int(metadata.get("record_count") or 0),
    )
    return {
        "region_id": str(region_id).strip(),
        "endpoint": chosen_endpoint,
        "bounds": chosen_bounds,
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "record_count": int(metadata.get("record_count") or 0),
        "acquisition_method": acquisition.acquisition_method,
        "bytes_downloaded": acquisition.bytes_downloaded,
        "warnings": list(acquisition.warnings or []),
    }
