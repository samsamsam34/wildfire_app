from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_REGION_DATA_DIR = Path("data") / "regions"
REQUIRED_REGION_FILES = (
    "dem",
    "slope",
    "fuel",
    "canopy",
    "fire_perimeters",
    "building_footprints",
)


def get_region_data_dir(base_dir: str | None = None) -> Path:
    if base_dir:
        return Path(base_dir).expanduser()
    env_dir = os.getenv("WF_REGION_DATA_DIR", "")
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_REGION_DATA_DIR


def _coerce_bounds(bounds: Any) -> tuple[float, float, float, float] | None:
    if isinstance(bounds, dict):
        keys = ["min_lon", "min_lat", "max_lon", "max_lat"]
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
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            min_lon, min_lat, max_lon, max_lat = [float(v) for v in bounds]
            return min_lon, min_lat, max_lon, max_lat
        except (TypeError, ValueError):
            return None
    return None


def _manifest_path(region_id: str, base_dir: str | None = None) -> Path:
    root = get_region_data_dir(base_dir)
    return root / region_id / "manifest.json"


def load_region_manifest(region_id: str, base_dir: str | None = None) -> dict[str, Any] | None:
    manifest_path = _manifest_path(region_id, base_dir=base_dir)
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        return None
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_region_dir"] = str(manifest_path.parent)
    return manifest


def list_prepared_regions(base_dir: str | None = None) -> list[dict[str, Any]]:
    root = get_region_data_dir(base_dir)
    if not root.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        manifest = load_region_manifest(child.name, base_dir=base_dir)
        if manifest:
            manifests.append(manifest)
    return manifests


def region_contains_point(manifest: dict[str, Any], lat: float, lon: float) -> bool:
    bounds = _coerce_bounds(manifest.get("bounds"))
    if bounds is None:
        return False
    min_lon, min_lat, max_lon, max_lat = bounds
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def validate_region_files(manifest: dict[str, Any], base_dir: str | None = None) -> tuple[bool, list[str]]:
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        return False, ["Manifest missing files mapping."]

    region_dir = Path(str(manifest.get("_region_dir") or ""))
    if not region_dir:
        region_id = str(manifest.get("region_id") or "")
        if region_id:
            region_dir = get_region_data_dir(base_dir) / region_id
    missing: list[str] = []
    for key in REQUIRED_REGION_FILES:
        rel = files.get(key)
        if not rel:
            missing.append(f"{key} missing from manifest files")
            continue
        file_path = Path(rel)
        if not file_path.is_absolute():
            file_path = region_dir / file_path
        if not file_path.exists():
            missing.append(f"{key} file not found: {file_path}")
    return len(missing) == 0, missing


def resolve_region_file(
    manifest: dict[str, Any],
    layer_key: str,
    base_dir: str | None = None,
) -> str | None:
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        return None
    rel = files.get(layer_key)
    if not rel:
        return None

    file_path = Path(str(rel))
    if not file_path.is_absolute():
        region_dir = Path(str(manifest.get("_region_dir") or ""))
        if not region_dir:
            region_id = str(manifest.get("region_id") or "")
            if not region_id:
                return None
            region_dir = get_region_data_dir(base_dir) / region_id
        file_path = region_dir / file_path
    return str(file_path)


def find_region_for_point(lat: float, lon: float, base_dir: str | None = None) -> dict[str, Any] | None:
    for manifest in list_prepared_regions(base_dir=base_dir):
        if region_contains_point(manifest, lat=lat, lon=lon):
            return manifest
    return None
