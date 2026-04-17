from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.data_prep.sources.acquisition import ArcGISFeatureServiceProvider
from backend.region_registry import get_region_data_dir


MISSOULA_PILOT_BBOX = {
    "min_lat": 46.80,
    "max_lat": 46.95,
    "min_lon": -114.20,
    "max_lon": -113.85,
}


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


def _feature_point_within_bounds(feature: dict[str, Any], bounds: dict[str, float]) -> bool:
    geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
    if str(geometry.get("type") or "") != "Point":
        return False
    coords = geometry.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return False
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except (TypeError, ValueError):
        return False
    return (
        bounds["min_lon"] <= lon <= bounds["max_lon"]
        and bounds["min_lat"] <= lat <= bounds["max_lat"]
    )


def _write_geojson(
    *,
    source_payload: dict[str, Any],
    bounds: dict[str, float],
    output_path: Path,
    source_endpoint: str,
) -> dict[str, Any]:
    features = source_payload.get("features") if isinstance(source_payload, dict) else []
    if not isinstance(features, list):
        features = []
    clipped = [row for row in features if isinstance(row, dict) and _feature_point_within_bounds(row, bounds)]
    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source_endpoint": source_endpoint,
            "clipped_to_bbox": bounds,
            "record_count": len(clipped),
            "generated_at": _now_iso(),
        },
        "features": clipped,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload["metadata"]


def _update_region_manifest_address_points(
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
    files["address_points"] = output_filename
    manifest["files"] = files
    manifest.setdefault("region_id", region_id)
    manifest.setdefault("display_name", region_id.replace("_", " ").title())
    manifest.setdefault("bounds", bounds)
    manifest["address_points_availability"] = {
        "available": True,
        "layer_key": "address_points",
        "path": output_filename,
        "record_count": int(record_count),
        "source_endpoint": endpoint,
        "updated_at": _now_iso(),
    }
    manifest.setdefault("notes", [])
    notes = list(manifest.get("notes") or [])
    note = "Address points are available and can be used for geocode snap refinement."
    if note not in notes:
        notes.append(note)
    manifest["notes"] = notes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def download_and_clip_missoula_address_points(
    *,
    region_id: str = "missoula_pilot",
    regions_root: str | Path | None = None,
    endpoint: str | None = None,
    bounds: dict[str, float] | None = None,
    timeout_seconds: float = 45.0,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    """Download Missoula County E911 address points and write clipped GeoJSON.

    Uses ArcGIS pagination via resultOffset/resultRecordCount by forcing the
    provider into JSON-query mode.
    """
    chosen_endpoint = str(
        endpoint
        or os.getenv("WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT")
        or ""
    ).strip()
    if not chosen_endpoint:
        raise ValueError(
            "Missing address-point endpoint. Set WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT "
            "or pass endpoint explicitly."
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
        layer_key="address_points",
        bounds=chosen_bounds,
        cache_root=region_dir / ".cache" / "address_points",
        target_resolution=None,
        timeout_seconds=float(timeout_seconds),
        retries=int(retries),
        backoff_seconds=float(backoff_seconds),
    )
    if not acquisition.local_path:
        raise ValueError("Address-point download produced no local output path.")
    downloaded_path = Path(acquisition.local_path)
    if not downloaded_path.exists():
        raise ValueError(f"Address-point download output is missing: {downloaded_path}")

    source_payload = json.loads(downloaded_path.read_text(encoding="utf-8"))
    output_filename = "address_points.geojson"
    output_path = region_dir / output_filename
    metadata = _write_geojson(
        source_payload=source_payload if isinstance(source_payload, dict) else {},
        bounds=chosen_bounds,
        output_path=output_path,
        source_endpoint=chosen_endpoint,
    )
    manifest_path = _update_region_manifest_address_points(
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
