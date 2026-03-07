from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.region_registry import DEFAULT_REGION_DATA_DIR, REQUIRED_REGION_FILES


STANDARD_LAYER_FILENAMES = {
    "dem": "dem.tif",
    "slope": "slope.tif",
    "fuel": "fuel.tif",
    "canopy": "canopy.tif",
    "fire_perimeters": "fire_perimeters.geojson",
    "building_footprints": "building_footprints.geojson",
    "burn_probability": "burn_probability.tif",
    "wildfire_hazard": "wildfire_hazard.tif",
    "moisture": "moisture.tif",
    "aspect": "aspect.tif",
}


def _stage_file(src: Path, dest: Path, *, copy_files: bool) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy_files:
        shutil.copy2(src, dest)
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def _default_layer_metadata(path: Path) -> dict[str, Any]:
    loaded_at = datetime.now(tz=timezone.utc).isoformat()
    return {
        "source_name": path.name,
        "source_type": "local_file",
        "dataset_version": None,
        "freshness_timestamp": None,
        "loaded_at": loaded_at,
    }


def prepare_region_layers(
    *,
    region_id: str,
    display_name: str,
    bounds: dict[str, float],
    layer_sources: dict[str, str],
    region_data_dir: str | Path | None = None,
    crs: str = "EPSG:4326",
    copy_files: bool = False,
    source_metadata: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not region_id.strip():
        raise ValueError("region_id is required")
    if not display_name.strip():
        raise ValueError("display_name is required")
    for key in ["min_lon", "min_lat", "max_lon", "max_lat"]:
        if key not in bounds:
            raise ValueError(f"bounds must include {key}")

    for required in REQUIRED_REGION_FILES:
        if not layer_sources.get(required):
            raise ValueError(f"Missing required layer source: {required}")

    root = Path(region_data_dir or os.getenv("WF_REGION_DATA_DIR") or DEFAULT_REGION_DATA_DIR).expanduser()
    region_dir = root / region_id
    if region_dir.exists() and not force:
        raise ValueError(f"Region directory already exists: {region_dir} (use force=True to overwrite)")
    region_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    layers_meta: dict[str, dict[str, Any]] = {}
    metadata = source_metadata or {}
    for layer_key, source in layer_sources.items():
        if layer_key not in STANDARD_LAYER_FILENAMES:
            continue
        src_path = Path(source).expanduser()
        if not src_path.exists():
            raise ValueError(f"Layer source not found for {layer_key}: {src_path}")
        dest_name = STANDARD_LAYER_FILENAMES[layer_key]
        dest_path = region_dir / dest_name
        _stage_file(src_path, dest_path, copy_files=copy_files)
        files[layer_key] = dest_name
        layer_meta = dict(_default_layer_metadata(src_path))
        layer_meta.update(metadata.get(layer_key, {}))
        layers_meta[layer_key] = layer_meta

    manifest = {
        "region_id": region_id,
        "display_name": display_name,
        "prepared_at": datetime.now(tz=timezone.utc).isoformat(),
        "preparation_status": "prepared",
        "crs": crs,
        "bounds": {
            "min_lon": float(bounds["min_lon"]),
            "min_lat": float(bounds["min_lat"]),
            "max_lon": float(bounds["max_lon"]),
            "max_lat": float(bounds["max_lat"]),
        },
        "files": files,
        "layers": layers_meta,
    }

    with open(region_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest
