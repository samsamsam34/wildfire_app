from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.building_footprints import BuildingFootprintClient, RING_KEYS, compute_structure_rings
from backend.feature_bundle_cache import FeatureBundleCache
from backend.feature_enrichment import (
    apply_enrichment_source_fallbacks,
    build_feature_bundle_summary,
)
from backend.naip_features import (
    load_naip_feature_artifact,
    percentile_from_quantiles,
    resolve_naip_feature_path,
    structure_feature_key,
)
from backend.layer_diagnostics import (
    initialize_layer_audit,
    summarize_layer_audit,
    update_layer_audit,
)
from backend.property_anchor import PropertyAnchorResolver
from backend.open_data_adapters import (
    GridMETAdapter,
    MTBSAdapter,
    OSMRoadAdapter,
    WHPAdapter,
)
from backend.data_prep.region_lookup import find_region_for_point as lookup_region_for_point
from backend.region_registry import (
    resolve_region_file,
    validate_region_files,
)

try:
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import LineString, Point, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - graceful runtime fallback when geo deps are missing
    np = None
    rasterio = None
    Transformer = None
    LineString = None
    Point = None
    mapping = None
    shape = None
    shapely_transform = None


@dataclass
class WildfireContext:
    environmental_index: Optional[float]
    slope_index: Optional[float]
    aspect_index: Optional[float]
    fuel_index: Optional[float]
    moisture_index: Optional[float]
    canopy_index: Optional[float]
    wildland_distance_index: Optional[float]
    historic_fire_index: Optional[float]
    burn_probability_index: Optional[float]
    hazard_severity_index: Optional[float]
    access_exposure_index: Optional[float] = None
    burn_probability: Optional[float] = None
    wildfire_hazard: Optional[float] = None
    slope: Optional[float] = None
    fuel_model: Optional[float] = None
    canopy_cover: Optional[float] = None
    historic_fire_distance: Optional[float] = None
    wildland_distance: Optional[float] = None
    environmental_layer_status: Dict[str, str] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    structure_ring_metrics: Dict[str, Dict[str, float | None]] = field(default_factory=dict)
    property_level_context: Dict[str, Any] = field(default_factory=dict)
    region_context: Dict[str, Any] = field(default_factory=dict)
    hazard_context: Dict[str, Any] = field(default_factory=dict)
    moisture_context: Dict[str, Any] = field(default_factory=dict)
    historical_fire_context: Dict[str, Any] = field(default_factory=dict)
    access_context: Dict[str, Any] = field(default_factory=dict)
    layer_coverage_audit: List[Dict[str, Any]] = field(default_factory=list)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)


def compute_environmental_data_completeness(context: WildfireContext) -> float:
    status = context.environmental_layer_status or {}
    if not status:
        return 0.0

    tracked_layers = [
        "burn_probability",
        "hazard",
        "slope",
        "fuel",
        "canopy",
        "fire_history",
    ]
    available = sum(1 for layer in tracked_layers if status.get(layer) == "ok")
    return round((available / float(len(tracked_layers))) * 100.0, 1)


class WildfireDataClient:
    """Layer-backed wildfire context provider.

    Expected dataset environment variables (GeoTIFF/GeoJSON):
    - WF_LAYER_BURN_PROB_TIF
    - WF_LAYER_HAZARD_SEVERITY_TIF
    - WF_LAYER_SLOPE_TIF (optional when DEM is provided)
    - WF_LAYER_ASPECT_TIF (optional when DEM is provided)
    - WF_LAYER_DEM_TIF
    - WF_LAYER_FUEL_TIF
    - WF_LAYER_CANOPY_TIF
    - WF_LAYER_MOISTURE_TIF (optional; higher values should represent drier fuels)
    - WF_LAYER_FIRE_PERIMETERS_GEOJSON
    """

    def __init__(self) -> None:
        self.base_paths = {
            "burn_prob": os.getenv("WF_LAYER_BURN_PROB_TIF", ""),
            "hazard": os.getenv("WF_LAYER_HAZARD_SEVERITY_TIF", ""),
            "whp": os.getenv("WF_LAYER_WHP_TIF", ""),
            "slope": os.getenv("WF_LAYER_SLOPE_TIF", ""),
            "aspect": os.getenv("WF_LAYER_ASPECT_TIF", ""),
            "dem": os.getenv("WF_LAYER_DEM_TIF", ""),
            "fuel": os.getenv("WF_LAYER_FUEL_TIF", ""),
            "canopy": os.getenv("WF_LAYER_CANOPY_TIF", ""),
            "moisture": os.getenv("WF_LAYER_MOISTURE_TIF", ""),
            "gridmet_dryness": os.getenv("WF_LAYER_GRIDMET_DRYNESS_TIF", ""),
            "perimeters": os.getenv("WF_LAYER_FIRE_PERIMETERS_GEOJSON", ""),
            "mtbs_severity": os.getenv("WF_LAYER_MTBS_SEVERITY_TIF", ""),
            "footprints_overture": os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_OVERTURE_GEOJSON", ""),
            "footprints_microsoft": (
                os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_MICROSOFT_GEOJSON", "")
                or os.getenv("WF_ENRICH_MICROSOFT_BUILDINGS_PATH", "")
            ),
            "footprints": os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON", ""),
            "fema_structures": os.getenv("WF_LAYER_FEMA_STRUCTURES_GEOJSON", ""),
            "address_points": (
                os.getenv("WF_LAYER_ADDRESS_POINTS_GEOJSON", "")
                or os.getenv("WF_LAYER_PARCEL_ADDRESS_POINTS_GEOJSON", "")
            ),
            "parcels": (
                os.getenv("WF_LAYER_PARCELS_GEOJSON", "")
                or os.getenv("WF_LAYER_PARCEL_POLYGONS_GEOJSON", "")
            ),
            "roads": os.getenv("WF_LAYER_OSM_ROADS_GEOJSON", ""),
            "naip_imagery": os.getenv("WF_LAYER_NAIP_IMAGERY_TIF", ""),
            "naip_structure_features": os.getenv("WF_LAYER_NAIP_STRUCTURE_FEATURES_JSON", ""),
        }
        # Active runtime paths for the most recent collect_context() call.
        self.paths = dict(self.base_paths)
        self.region_data_dir = os.getenv("WF_REGION_DATA_DIR", str(Path("data") / "regions"))
        self.use_prepared_regions = os.getenv("WF_USE_PREPARED_REGIONS", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.allow_legacy_layer_fallback = os.getenv("WF_ALLOW_LEGACY_LAYER_FALLBACK", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.footprints = BuildingFootprintClient()
        self.whp_adapter = WHPAdapter()
        self.gridmet_adapter = GridMETAdapter()
        self.mtbs_adapter = MTBSAdapter()
        self.osm_adapter = OSMRoadAdapter()
        self.feature_bundle_cache = FeatureBundleCache()

    @staticmethod
    def _to_index(value: float, src_min: float, src_max: float) -> float:
        if src_max <= src_min:
            return 50.0
        v = max(src_min, min(src_max, value))
        return round(100.0 * (v - src_min) / (src_max - src_min), 1)

    @staticmethod
    def _local_percentile_rank(value: float | None, samples: Iterable[float]) -> float | None:
        if value is None:
            return None
        vals = [float(v) for v in samples if v is not None]
        if not vals:
            return None
        less_or_equal = sum(1 for v in vals if v <= float(value))
        return round((less_or_equal / float(len(vals))) * 100.0, 1)

    @staticmethod
    def _blend_indices(terms: list[tuple[float, float | None]]) -> float | None:
        available = [(w, float(v)) for w, v in terms if v is not None and w > 0.0]
        if not available:
            return None
        denom = sum(w for w, _ in available)
        if denom <= 0.0:
            return None
        return round(sum(w * v for w, v in available) / denom, 1)

    @staticmethod
    def _meters_to_lat_deg(meters: float) -> float:
        return meters / 111_320.0

    @staticmethod
    def _meters_to_lon_deg(meters: float, lat: float) -> float:
        denom = 111_320.0 * max(0.1, math.cos(math.radians(lat)))
        return meters / denom

    @staticmethod
    def _file_exists(path: str) -> bool:
        return bool(path) and Path(path).exists()

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_centroid_feature_key(key: str) -> tuple[float, float] | None:
        token = str(key or "").strip()
        if not token.startswith("centroid:"):
            return None
        payload = token.split(":", 1)[1]
        parts = payload.split(",")
        if len(parts) != 2:
            return None
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon

    @staticmethod
    def _haversine_distance_m(
        lat_a: float,
        lon_a: float,
        lat_b: float,
        lon_b: float,
    ) -> float:
        lat1 = math.radians(float(lat_a))
        lon1 = math.radians(float(lon_a))
        lat2 = math.radians(float(lat_b))
        lon2 = math.radians(float(lon_b))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2.0) ** 2)
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
        return float(6371000.0 * c)

    def _resolve_naip_feature_artifact_path(
        self,
        *,
        runtime_paths: dict[str, str],
        region_context: dict[str, Any],
    ) -> str | None:
        manifest_path = str(region_context.get("manifest_path") or "").strip() or None
        runtime_path = str(runtime_paths.get("naip_structure_features") or "").strip() or None
        return resolve_naip_feature_path(
            region_manifest_path=manifest_path,
            runtime_path=runtime_path,
        )

    def _apply_naip_feature_enrichment(
        self,
        *,
        ring_context: dict[str, Any],
        runtime_paths: dict[str, str],
        region_context: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        assumptions: list[str] = []
        sources: list[str] = []

        artifact_path = self._resolve_naip_feature_artifact_path(
            runtime_paths=runtime_paths,
            region_context=region_context,
        )
        if not artifact_path:
            return ring_context, assumptions, sources

        payload = load_naip_feature_artifact(artifact_path)
        features_by_key = payload.get("features_by_key")
        keys_by_structure_id = payload.get("keys_by_structure_id")
        quantiles = payload.get("quantiles")
        if not isinstance(features_by_key, dict) or not features_by_key:
            assumptions.append("NAIP structure feature artifact exists but contains no usable feature rows.")
            return ring_context, assumptions, sources
        keys_by_structure_id = keys_by_structure_id if isinstance(keys_by_structure_id, dict) else {}
        quantiles = quantiles if isinstance(quantiles, dict) else {}

        matched_structure_id = str(ring_context.get("matched_structure_id") or "").strip() or None
        centroid = ring_context.get("footprint_centroid")
        centroid_lat = self._coerce_float((centroid or {}).get("latitude")) if isinstance(centroid, dict) else None
        centroid_lon = self._coerce_float((centroid or {}).get("longitude")) if isinstance(centroid, dict) else None

        selected_key: str | None = None
        selected_feature: dict[str, Any] | None = None
        match_method: str | None = None
        match_distance_m: float | None = None

        if matched_structure_id and matched_structure_id in keys_by_structure_id:
            selected_key = str(keys_by_structure_id.get(matched_structure_id))
            raw = features_by_key.get(selected_key)
            if isinstance(raw, dict):
                selected_feature = raw
                match_method = "structure_id"
        if selected_feature is None and matched_structure_id:
            direct_key = structure_feature_key(
                structure_id=matched_structure_id,
                centroid_lat=None,
                centroid_lon=None,
            )
            if direct_key and isinstance(features_by_key.get(direct_key), dict):
                selected_key = direct_key
                selected_feature = features_by_key.get(direct_key)
                match_method = "structure_id"
        if selected_feature is None:
            centroid_key = structure_feature_key(
                structure_id=None,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
            )
            if centroid_key and isinstance(features_by_key.get(centroid_key), dict):
                selected_key = centroid_key
                selected_feature = features_by_key.get(centroid_key)
                match_method = "centroid"

        if selected_feature is None and centroid_lat is not None and centroid_lon is not None:
            try:
                nearest_centroid_max_distance_m = float(
                    str(os.getenv("WF_NAIP_FEATURE_MATCH_MAX_DISTANCE_M", "45")).strip()
                )
            except ValueError:
                nearest_centroid_max_distance_m = 45.0
            nearest_centroid_max_distance_m = max(5.0, min(500.0, nearest_centroid_max_distance_m))

            best_key: str | None = None
            best_row: dict[str, Any] | None = None
            best_distance_m: float | None = None
            for key, row in features_by_key.items():
                if not isinstance(row, dict):
                    continue
                parsed = self._parse_centroid_feature_key(str(key))
                if parsed is None:
                    continue
                distance_m = self._haversine_distance_m(
                    float(centroid_lat),
                    float(centroid_lon),
                    float(parsed[0]),
                    float(parsed[1]),
                )
                if best_distance_m is None or distance_m < best_distance_m:
                    best_distance_m = distance_m
                    best_key = str(key)
                    best_row = row
            if (
                best_row is not None
                and best_key
                and best_distance_m is not None
                and best_distance_m <= nearest_centroid_max_distance_m
            ):
                selected_key = best_key
                selected_feature = best_row
                match_method = "nearest_centroid"
                match_distance_m = round(float(best_distance_m), 2)
                assumptions.append(
                    f"NAIP feature row matched by nearest centroid within {match_distance_m:.1f} m."
                )

        if not isinstance(selected_feature, dict):
            return ring_context, assumptions, sources

        ring_metrics = ring_context.get("ring_metrics")
        if not isinstance(ring_metrics, dict):
            ring_metrics = {}

        selected_ring_metrics = selected_feature.get("ring_metrics")
        selected_ring_metrics = selected_ring_metrics if isinstance(selected_ring_metrics, dict) else {}
        if not selected_ring_metrics:
            assumptions.append("NAIP feature row found but no ring metrics were available for blending.")
            return ring_context, assumptions, sources

        for ring_key in RING_KEYS:
            existing = ring_metrics.get(ring_key)
            existing = dict(existing) if isinstance(existing, dict) else {}
            imagery = selected_ring_metrics.get(ring_key)
            if not isinstance(imagery, dict):
                continue

            imagery_density = self._coerce_float(imagery.get("vegetation_density_proxy"))
            existing_density = self._coerce_float(existing.get("vegetation_density"))
            if imagery_density is not None and existing_density is not None:
                blended_density = round((0.60 * existing_density) + (0.40 * imagery_density), 1)
            elif imagery_density is not None:
                blended_density = round(imagery_density, 1)
            else:
                blended_density = existing_density

            if blended_density is not None:
                existing["vegetation_density"] = blended_density
            existing["imagery_vegetation_cover_pct"] = self._coerce_float(imagery.get("vegetation_cover_pct"))
            existing["imagery_canopy_proxy_pct"] = self._coerce_float(imagery.get("canopy_proxy_pct"))
            existing["imagery_high_fuel_proxy_pct"] = self._coerce_float(imagery.get("high_fuel_proxy_pct"))
            existing["imagery_impervious_low_fuel_pct"] = self._coerce_float(imagery.get("impervious_low_fuel_pct"))
            existing["imagery_vegetation_continuity_pct"] = self._coerce_float(imagery.get("vegetation_continuity_pct"))
            existing["imagery_ndvi_or_exg_mean"] = self._coerce_float(imagery.get("ndvi_or_exg_mean"))
            local_pct = self._coerce_float(imagery.get("vegetation_cover_pct_local_percentile"))
            if local_pct is None:
                local_pct = percentile_from_quantiles(
                    existing.get("imagery_vegetation_cover_pct"),
                    quantiles.get(f"{ring_key}.vegetation_cover_pct") if isinstance(quantiles, dict) else None,
                )
            if local_pct is not None:
                existing["imagery_vegetation_cover_local_percentile"] = round(local_pct, 1)
            existing["basis"] = "footprint_naip_blended"

            ring_metrics[ring_key] = existing
            ring_metrics[ring_key.replace("ring_", "zone_")] = dict(existing)

        ring_context["ring_metrics"] = ring_metrics
        ring_context["naip_feature_artifact_path"] = artifact_path
        ring_context["naip_feature_match_key"] = selected_key
        ring_context["naip_feature_match_method"] = match_method
        ring_context["naip_feature_match_distance_m"] = match_distance_m
        ring_context["naip_feature_source"] = "prepared_region_naip"
        ring_context["imagery_ring_metrics"] = selected_ring_metrics
        ring_context["near_structure_vegetation_0_5_pct"] = self._coerce_float(
            selected_feature.get("near_structure_vegetation_0_5_pct")
        )
        if ring_context.get("near_structure_vegetation_0_5_pct") is None:
            ring_context["near_structure_vegetation_0_5_pct"] = self._coerce_float(
                ((ring_metrics.get("ring_0_5_ft") or {}).get("vegetation_density"))
            )
        ring_context["near_structure_vegetation_5_30_pct"] = self._coerce_float(
            selected_feature.get("near_structure_vegetation_5_30_pct")
        )
        if ring_context.get("near_structure_vegetation_5_30_pct") is None:
            ring_context["near_structure_vegetation_5_30_pct"] = self._coerce_float(
                ((ring_metrics.get("ring_5_30_ft") or {}).get("vegetation_density"))
            )
        ring_context["canopy_adjacency_proxy_pct"] = self._coerce_float(
            selected_feature.get("canopy_adjacency_proxy_pct")
        )
        ring_context["vegetation_continuity_proxy_pct"] = self._coerce_float(
            selected_feature.get("vegetation_continuity_proxy_pct")
        )
        ring_context["nearest_high_fuel_patch_distance_ft"] = self._coerce_float(
            selected_feature.get("nearest_high_fuel_patch_distance_ft")
        )
        ring_context["imagery_local_percentiles"] = {
            key: value
            for key, value in {
                "near_structure_vegetation_0_5_pct": percentile_from_quantiles(
                    self._coerce_float(selected_feature.get("near_structure_vegetation_0_5_pct")),
                    quantiles.get("near_structure_vegetation_0_5_pct") if isinstance(quantiles, dict) else None,
                ),
                "canopy_adjacency_proxy_pct": percentile_from_quantiles(
                    self._coerce_float(selected_feature.get("canopy_adjacency_proxy_pct")),
                    quantiles.get("canopy_adjacency_proxy_pct") if isinstance(quantiles, dict) else None,
                ),
                "vegetation_continuity_proxy_pct": percentile_from_quantiles(
                    self._coerce_float(selected_feature.get("vegetation_continuity_proxy_pct")),
                    quantiles.get("vegetation_continuity_proxy_pct") if isinstance(quantiles, dict) else None,
                ),
            }.items()
            if value is not None
        }

        assumptions.append(
            "Near-structure vegetation metrics were enriched with precomputed NAIP imagery features."
        )
        sources.append("NAIP imagery-derived ring features")
        ring_context["near_structure_features"] = self._build_near_structure_feature_block(
            ring_context=ring_context
        )
        return ring_context, assumptions, sources

    def _legacy_layer_configured(self, configured_paths: dict[str, str]) -> bool:
        configured = [
            configured_paths.get("dem"),
            configured_paths.get("slope"),
            configured_paths.get("fuel"),
            configured_paths.get("canopy"),
            configured_paths.get("perimeters"),
            configured_paths.get("footprints_overture"),
            configured_paths.get("footprints_microsoft"),
            configured_paths.get("footprints"),
            configured_paths.get("address_points"),
            configured_paths.get("parcels"),
            configured_paths.get("whp"),
            configured_paths.get("gridmet_dryness"),
            configured_paths.get("roads"),
            configured_paths.get("naip_structure_features"),
            configured_paths.get("naip_imagery"),
        ]
        return any(self._file_exists(path or "") for path in configured)

    @staticmethod
    def _extract_region_catalog_runtime_summary(region_manifest: dict[str, Any] | None) -> dict[str, Any]:
        catalog = (region_manifest or {}).get("catalog")
        if not isinstance(catalog, dict):
            return {
                "property_specific_readiness": "limited_regional_ready",
                "validation_summary": {},
                "required_layers_missing": [],
                "optional_layers_missing": [],
                "enrichment_layers_missing": [],
                "missing_reason_by_layer": {},
            }
        readiness = catalog.get("property_specific_readiness")
        readiness_value = "limited_regional_ready"
        if isinstance(readiness, dict):
            candidate = str(readiness.get("readiness") or "").strip()
            if candidate:
                readiness_value = candidate
        return {
            "property_specific_readiness": readiness_value,
            "validation_summary": (
                dict(catalog.get("validation_summary"))
                if isinstance(catalog.get("validation_summary"), dict)
                else {}
            ),
            "required_layers_missing": [
                str(v) for v in (catalog.get("required_layers_missing") or []) if str(v).strip()
            ],
            "optional_layers_missing": [
                str(v) for v in (catalog.get("optional_layers_missing") or []) if str(v).strip()
            ],
            "enrichment_layers_missing": [
                str(v) for v in (catalog.get("enrichment_layers_missing") or []) if str(v).strip()
            ],
            "missing_reason_by_layer": (
                dict(catalog.get("missing_reason_by_layer"))
                if isinstance(catalog.get("missing_reason_by_layer"), dict)
                else {}
            ),
        }

    def _resolve_runtime_layer_paths(
        self, lat: float, lon: float
    ) -> tuple[dict[str, str], dict[str, Any], list[str], list[str]]:
        assumptions: list[str] = []
        sources: list[str] = []
        configured_paths = dict(self.base_paths)
        if self.paths and self.paths != self.base_paths:
            configured_paths.update(self.paths)
        runtime_paths = dict(configured_paths)
        region_context: dict[str, Any] = {
            "region_status": "legacy_fallback",
            "region_id": None,
            "region_display_name": None,
            "manifest_path": None,
            "building_sources": [],
            "property_specific_readiness": "limited_regional_ready",
            "validation_summary": {},
            "required_layers_missing": [],
            "optional_layers_missing": [],
            "enrichment_layers_missing": [],
            "missing_reason_by_layer": {},
        }

        region_manifest: dict[str, Any] | None = None
        if self.use_prepared_regions:
            lookup = lookup_region_for_point(lat=lat, lon=lon, regions_root=self.region_data_dir)
            if lookup.get("covered"):
                manifest = lookup.get("manifest")
                if isinstance(manifest, dict):
                    region_manifest = manifest
            else:
                diagnostics = list(lookup.get("diagnostics") or [])
                assumptions.extend(diagnostics[:3])

        if region_manifest:
            valid, missing = validate_region_files(region_manifest, base_dir=self.region_data_dir)
            if valid:
                catalog_runtime_summary = self._extract_region_catalog_runtime_summary(region_manifest)
                region_context = {
                    "region_status": "prepared",
                    "region_id": region_manifest.get("region_id"),
                    "region_display_name": region_manifest.get("display_name"),
                    "manifest_path": region_manifest.get("_manifest_path"),
                    "building_sources": [],
                    "property_specific_readiness": catalog_runtime_summary["property_specific_readiness"],
                    "validation_summary": catalog_runtime_summary["validation_summary"],
                    "required_layers_missing": catalog_runtime_summary["required_layers_missing"],
                    "optional_layers_missing": catalog_runtime_summary["optional_layers_missing"],
                    "enrichment_layers_missing": catalog_runtime_summary["enrichment_layers_missing"],
                    "missing_reason_by_layer": catalog_runtime_summary["missing_reason_by_layer"],
                }
                sources.append(f"Prepared region: {region_manifest.get('region_id')}")
                layer_key_map = {
                    "burn_prob": ("burn_probability", "burn_prob"),
                    "hazard": ("wildfire_hazard", "hazard"),
                    "whp": ("whp", "wildfire_hazard_potential"),
                    "slope": ("slope",),
                    "aspect": ("aspect",),
                    "dem": ("dem",),
                    "fuel": ("fuel",),
                    "canopy": ("canopy",),
                    "moisture": ("moisture",),
                    "gridmet_dryness": ("gridmet_dryness", "gridmet"),
                    "perimeters": ("fire_perimeters", "perimeters"),
                    "mtbs_severity": ("mtbs_severity", "burn_severity"),
                    "footprints_overture": ("building_footprints_overture", "overture_buildings"),
                    "footprints_microsoft": ("building_footprints_microsoft", "microsoft_buildings"),
                    "footprints": ("building_footprints", "footprints"),
                    "fema_structures": ("fema_structures",),
                    "address_points": ("address_points", "parcel_address_points"),
                    "parcels": ("parcel_polygons", "parcels"),
                    "roads": ("roads", "osm_roads", "road_network"),
                    "naip_imagery": ("naip_imagery", "naip_rgb", "naip"),
                    "naip_structure_features": ("naip_structure_features",),
                }
                for runtime_key, manifest_keys in layer_key_map.items():
                    resolved: str | None = None
                    for manifest_key in manifest_keys:
                        resolved = resolve_region_file(region_manifest, manifest_key, base_dir=self.region_data_dir)
                        if resolved:
                            break
                    if resolved:
                        runtime_paths[runtime_key] = resolved
                configured_building_sources = region_manifest.get("building_sources")
                if isinstance(configured_building_sources, list):
                    region_context["building_sources"] = [str(v) for v in configured_building_sources if str(v).strip()]
                else:
                    region_context["building_sources"] = []
                return runtime_paths, region_context, assumptions, sources

            region_context = {
                "region_status": "invalid_manifest",
                "region_id": region_manifest.get("region_id"),
                "region_display_name": region_manifest.get("display_name"),
                "manifest_path": region_manifest.get("_manifest_path"),
                "building_sources": [],
                "property_specific_readiness": "limited_regional_ready",
                "validation_summary": {},
                "required_layers_missing": [],
                "optional_layers_missing": [],
                "enrichment_layers_missing": [],
                "missing_reason_by_layer": {},
            }
            assumptions.append("Prepared region manifest is missing required files for this area.")
            assumptions.extend([f"Region file validation: {reason}" for reason in missing[:5]])

        if self.allow_legacy_layer_fallback and self._legacy_layer_configured(configured_paths):
            if not region_manifest:
                assumptions.append("Prepared region not found for location; using legacy direct layer paths.")
                region_context = {
                    "region_status": "legacy_fallback",
                    "region_id": None,
                    "region_display_name": None,
                    "manifest_path": None,
                    "building_sources": [],
                    "property_specific_readiness": "limited_regional_ready",
                    "validation_summary": {},
                    "required_layers_missing": [],
                    "optional_layers_missing": [],
                    "enrichment_layers_missing": [],
                    "missing_reason_by_layer": {},
                }
            sources.append("Legacy direct layer paths")
            return runtime_paths, region_context, assumptions, sources

        assumptions.append(
            "Region not prepared for this location; initialize prepared regional layers before high-confidence scoring."
        )
        return (
            {k: "" for k in runtime_paths.keys()},
            {
                "region_status": "region_not_prepared",
                "region_id": None,
                "region_display_name": None,
                "manifest_path": None,
                "building_sources": [],
                "property_specific_readiness": "limited_regional_ready",
                "validation_summary": {},
                "required_layers_missing": [],
                "optional_layers_missing": [],
                "enrichment_layers_missing": [],
                "missing_reason_by_layer": {},
            },
            assumptions,
            sources,
        )

    @staticmethod
    def _resolve_building_source_paths(
        runtime_paths: dict[str, str],
        region_context: dict[str, Any],
    ) -> list[str]:
        env_priority = str(
            os.getenv(
                "WF_BUILDING_SOURCE_PRIORITY",
                "building_footprints_overture,building_footprints_microsoft,building_footprints,fema_structures",
            )
        ).strip()
        tokens = [token.strip().lower() for token in env_priority.split(",") if token.strip()]

        configured = region_context.get("building_sources")
        if isinstance(configured, list) and configured:
            normalized = [str(v).strip().lower() for v in configured if str(v).strip()]
            if normalized:
                tokens = normalized

        runtime_key_map = {
            "building_footprints_overture": "footprints_overture",
            "overture_buildings": "footprints_overture",
            "building_footprints_microsoft": "footprints_microsoft",
            "microsoft_buildings": "footprints_microsoft",
            "building_footprints": "footprints",
            "existing_building_dataset": "footprints",
            "osm_buildings": "fema_structures",
            "fema_structures": "fema_structures",
        }

        ordered: list[str] = []
        for token in tokens:
            runtime_key = runtime_key_map.get(token, token)
            candidate = str(runtime_paths.get(runtime_key) or "").strip()
            if candidate and candidate not in ordered:
                ordered.append(candidate)

        # Safety fallback if priority did not resolve.
        for runtime_key in ("footprints_overture", "footprints_microsoft", "footprints", "fema_structures"):
            candidate = str(runtime_paths.get(runtime_key) or "").strip()
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _resolve_parcel_source_paths(
        self,
        runtime_paths: dict[str, str],
    ) -> list[str]:
        ordered: list[str] = []

        def _add(candidate: str | None) -> None:
            normalized = self._normalize_source_path(candidate)
            if normalized and normalized not in ordered:
                ordered.append(normalized)

        _add(runtime_paths.get("parcels"))

        extra_env_tokens = [
            token.strip()
            for token in str(os.getenv("WF_LAYER_PARCELS_EXTRA_GEOJSON", "")).split(",")
            if token.strip()
        ]
        for token in extra_env_tokens:
            _add(token)

        for env_key in (
            "WF_DEFAULT_COUNTY_PARCEL_PATH",
            "WF_DEFAULT_OPEN_PARCEL_PATH",
            "WF_DEFAULT_PARCEL_POLYGONS_PATH",
        ):
            _add(os.getenv(env_key, ""))

        if not ordered:
            repo_root = Path(__file__).resolve().parents[1]
            catalog_dir = repo_root / "data" / "catalog" / "vectors" / "parcel_polygons"
            regions_dir = repo_root / "data" / "regions"
            if catalog_dir.exists():
                for path in sorted(catalog_dir.glob("*.geojson")):
                    _add(str(path))
            if regions_dir.exists():
                for path in sorted(regions_dir.glob("*/parcel_polygons.geojson")):
                    _add(str(path))

        return ordered

    @lru_cache(maxsize=16)
    def _open_raster(self, path: str):
        return rasterio.open(path)

    @lru_cache(maxsize=8)
    def _load_perimeters(self, path: str) -> List[Any]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", []) if isinstance(payload, dict) else []
        geoms = []
        for feat in features:
            geom = feat.get("geometry") if isinstance(feat, dict) else None
            if geom:
                geoms.append(shape(geom))
        return geoms

    def _to_dataset_crs(self, ds, lon: float, lat: float) -> tuple[float, float]:
        if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return lon, lat
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return transformer.transform(lon, lat)

    def _to_geom_crs(self, geom: Any, ds) -> Any:
        if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return geom
        if not (Transformer and shapely_transform):
            return geom
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return shapely_transform(transformer.transform, geom)

    def _polygon_raster_values(self, path: str, polygon: Any) -> list[float]:
        if not (rasterio and np is not None and mapping and self._file_exists(path)):
            return []

        ds = self._open_raster(path)
        try:
            geom_ds = self._to_geom_crs(polygon, ds)
            window = rasterio.features.geometry_window(ds, [mapping(geom_ds)])
            arr = ds.read(1, window=window, masked=True)
            inside_mask = rasterio.features.geometry_mask(
                [mapping(geom_ds)],
                transform=ds.window_transform(window),
                out_shape=arr.shape,
                invert=True,
                all_touched=True,
            )
            vals = arr[inside_mask]
            if hasattr(vals, "compressed"):
                vals = vals.compressed()
            return [float(v) for v in vals.tolist()]
        except Exception:
            return []

    def _summarize_ring_canopy(self, ring_geometry: Any, canopy_path: str) -> dict[str, float] | None:
        canopy_vals = self._polygon_raster_values(canopy_path, ring_geometry)
        if not canopy_vals:
            return None

        canopy_mean = float(sum(canopy_vals) / len(canopy_vals))
        canopy_max = float(max(canopy_vals))
        coverage_pct = float(sum(1 for v in canopy_vals if v >= 20.0) / len(canopy_vals) * 100.0)
        vegetation_density = float(self._to_index(canopy_mean, 0.0, 100.0))

        return {
            "canopy_mean": round(canopy_mean, 1),
            "canopy_max": round(canopy_max, 1),
            "coverage_pct": round(coverage_pct, 1),
            "vegetation_density": round(vegetation_density, 1),
        }

    def _summarize_ring_fuel_presence(self, ring_geometry: Any, fuel_path: str) -> float | None:
        fuel_vals = self._polygon_raster_values(fuel_path, ring_geometry)
        return self._fuel_presence_proxy_from_values(fuel_vals)

    def _fuel_presence_proxy_from_values(self, fuel_vals: list[float]) -> float | None:
        if not fuel_vals:
            return None
        wildland_cells = 0
        for raw in fuel_vals:
            code = int(round(raw))
            if (1 <= code <= 13) or (101 <= code <= 149):
                wildland_cells += 1

        return round((wildland_cells / len(fuel_vals)) * 100.0, 1)

    @staticmethod
    def _fuel_code_to_combustibility_index(raw: float | None) -> float | None:
        if raw is None:
            return None
        try:
            code = int(round(float(raw)))
        except (TypeError, ValueError):
            return None
        if 1 <= code <= 3:
            return 20.0 + (code - 1) * 4.0
        if 4 <= code <= 9:
            return 34.0 + (code - 4) * 4.2
        if 10 <= code <= 13:
            return 58.0 + (code - 10) * 6.0
        if 101 <= code <= 109:
            return 68.0 + (code - 101) * 2.0
        if 121 <= code <= 124:
            return 82.0 + (code - 121) * 3.0
        if 141 <= code <= 149:
            return 62.0 + (code - 141) * 2.5
        return None

    def _sample_combined_vegetation_index(
        self,
        *,
        canopy_path: str,
        fuel_path: str,
        lat: float,
        lon: float,
    ) -> float | None:
        canopy_raw = self._sample_raster_point(canopy_path, lat, lon) if self._file_exists(canopy_path) else None
        canopy_idx = self._coerce_float(self._to_index(canopy_raw, 0.0, 100.0)) if canopy_raw is not None else None
        if canopy_idx is not None:
            canopy_idx = max(0.0, min(100.0, float(canopy_idx)))
        fuel_raw = self._sample_raster_point(fuel_path, lat, lon) if self._file_exists(fuel_path) else None
        fuel_idx = self._fuel_code_to_combustibility_index(fuel_raw)

        if canopy_idx is not None and fuel_idx is not None:
            return round(max(0.0, min(100.0, (0.68 * canopy_idx) + (0.32 * fuel_idx))), 1)
        if canopy_idx is not None:
            return round(canopy_idx, 1)
        if fuel_idx is not None:
            return round(max(0.0, min(100.0, fuel_idx)), 1)
        return None

    def _offset_point_by_bearing(
        self,
        *,
        origin_lat: float,
        origin_lon: float,
        distance_m: float,
        bearing_deg: float,
    ) -> tuple[float, float]:
        theta = math.radians(float(bearing_deg))
        d_lat = self._meters_to_lat_deg(float(distance_m) * math.sin(theta))
        d_lon = self._meters_to_lon_deg(float(distance_m) * math.cos(theta), origin_lat)
        return float(origin_lat + d_lat), float(origin_lon + d_lon)

    @staticmethod
    def _extract_intersection_points(geom: Any) -> list[Any]:
        if geom is None:
            return []
        geom_type = str(getattr(geom, "geom_type", ""))
        if geom_type == "Point":
            return [geom]
        if geom_type == "MultiPoint":
            return [pt for pt in list(getattr(geom, "geoms", []) or []) if str(getattr(pt, "geom_type", "")) == "Point"]
        if geom_type in {"GeometryCollection", "MultiLineString", "LineString"}:
            points: list[Any] = []
            for part in list(getattr(geom, "geoms", []) or [geom]):
                part_type = str(getattr(part, "geom_type", ""))
                if part_type == "Point":
                    points.append(part)
                elif part_type == "LineString":
                    coords = list(getattr(part, "coords", []) or [])
                    if coords:
                        for coord in (coords[0], coords[-1]):
                            try:
                                if Point is not None:
                                    points.append(Point(float(coord[0]), float(coord[1])))
                            except Exception:
                                continue
            return points
        return []

    def _estimate_footprint_edge_distance_m(
        self,
        *,
        footprint: Any | None,
        origin_lat: float,
        origin_lon: float,
        bearing_deg: float,
        max_search_m: float = 250.0,
    ) -> float:
        if footprint is None or LineString is None or Transformer is None or shapely_transform is None:
            return 0.0
        try:
            to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
            centroid_m = shapely_transform(to_3857, Point(origin_lon, origin_lat))
            footprint_m = shapely_transform(to_3857, footprint)
            theta = math.radians(float(bearing_deg))
            target_x = float(centroid_m.x + (math.cos(theta) * max_search_m))
            target_y = float(centroid_m.y + (math.sin(theta) * max_search_m))
            ray = LineString([(float(centroid_m.x), float(centroid_m.y)), (target_x, target_y)])
            intersections = footprint_m.boundary.intersection(ray)
            points = self._extract_intersection_points(intersections)
            distances = []
            for pt in points:
                try:
                    distance = float(max(0.0, pt.distance(centroid_m)))
                except Exception:
                    continue
                if distance > 0.05:
                    distances.append(distance)
            if not distances:
                return 0.0
            return float(min(distances))
        except Exception:
            return 0.0

    def _compute_structure_aware_vegetation_features(
        self,
        *,
        origin_lat: float,
        origin_lon: float,
        canopy_path: str,
        fuel_path: str,
        slope_path: str = "",
        footprint: Any | None,
    ) -> dict[str, Any]:
        directions: list[tuple[str, float]] = [
            ("east", 0.0),
            ("north", 90.0),
            ("west", 180.0),
            ("south", 270.0),
        ]
        has_footprint = footprint is not None
        sector_samples: dict[str, dict[str, float | None]] = {}
        near_0_5_values: list[float] = []
        near_5_30_values: list[float] = []
        continuous_candidates_ft: list[float] = []
        threshold_index = 42.0
        continuity_steps_ft = [2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]

        for direction_name, bearing_deg in directions:
            edge_offset_m = self._estimate_footprint_edge_distance_m(
                footprint=footprint,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                bearing_deg=bearing_deg,
            )
            close_lat, close_lon = self._offset_point_by_bearing(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                distance_m=edge_offset_m + (3.0 * 0.3048),
                bearing_deg=bearing_deg,
            )
            zone1_lat, zone1_lon = self._offset_point_by_bearing(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                distance_m=edge_offset_m + (18.0 * 0.3048),
                bearing_deg=bearing_deg,
            )
            close_index = self._sample_combined_vegetation_index(
                canopy_path=canopy_path,
                fuel_path=fuel_path,
                lat=close_lat,
                lon=close_lon,
            )
            zone1_index = self._sample_combined_vegetation_index(
                canopy_path=canopy_path,
                fuel_path=fuel_path,
                lat=zone1_lat,
                lon=zone1_lon,
            )
            slope_deg = self._sample_raster_point(slope_path, zone1_lat, zone1_lon) if slope_path else None
            slope_index = (
                round(float(self._to_index(float(slope_deg), 0.0, 60.0)), 1)
                if slope_deg is not None
                else None
            )
            uphill_fuel_concentration = None
            if zone1_index is not None and slope_index is not None:
                uphill_fuel_concentration = round(
                    max(0.0, min(100.0, (0.70 * float(zone1_index)) + (0.30 * float(slope_index)))),
                    1,
                )
            if close_index is not None:
                near_0_5_values.append(float(close_index))
            if zone1_index is not None:
                near_5_30_values.append(float(zone1_index))
            risk_terms: list[tuple[float, float]] = []
            if close_index is not None:
                risk_terms.append((0.45, float(close_index)))
            if zone1_index is not None:
                risk_terms.append((0.30, float(zone1_index)))
            if slope_index is not None:
                risk_terms.append((0.10, float(slope_index)))
            if uphill_fuel_concentration is not None:
                risk_terms.append((0.15, float(uphill_fuel_concentration)))
            sector_risk_score = None
            if risk_terms:
                numerator = sum(weight * value for weight, value in risk_terms)
                denominator = sum(weight for weight, _ in risk_terms) or 1.0
                sector_risk_score = round(max(0.0, min(100.0, numerator / denominator)), 1)
            sector_samples[direction_name] = {
                "edge_offset_m": round(float(edge_offset_m), 2),
                "veg_0_5_index": round(float(close_index), 1) if close_index is not None else None,
                "veg_5_30_index": round(float(zone1_index), 1) if zone1_index is not None else None,
                "slope_deg": round(float(slope_deg), 1) if slope_deg is not None else None,
                "slope_index": slope_index,
                "uphill_fuel_concentration": uphill_fuel_concentration,
                "sector_risk_score": sector_risk_score,
            }

            streak = 0
            first_hit_ft: float | None = None
            for distance_ft in continuity_steps_ft:
                sample_lat, sample_lon = self._offset_point_by_bearing(
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    distance_m=edge_offset_m + (distance_ft * 0.3048),
                    bearing_deg=bearing_deg,
                )
                sample_index = self._sample_combined_vegetation_index(
                    canopy_path=canopy_path,
                    fuel_path=fuel_path,
                    lat=sample_lat,
                    lon=sample_lon,
                )
                if sample_index is not None and float(sample_index) >= threshold_index:
                    streak += 1
                    if first_hit_ft is None:
                        first_hit_ft = float(distance_ft)
                    if streak >= 2 and first_hit_ft is not None:
                        continuous_candidates_ft.append(first_hit_ft)
                        break
                else:
                    streak = 0
                    first_hit_ft = None

        near_0_5_pct = (
            round(sum(near_0_5_values) / len(near_0_5_values), 1)
            if near_0_5_values
            else None
        )
        near_5_30_pct = (
            round(sum(near_5_30_values) / len(near_5_30_values), 1)
            if near_5_30_values
            else None
        )
        directional_concentration = None
        if near_0_5_values:
            directional_concentration = round(
                max(0.0, min(100.0, max(near_0_5_values) - (sum(near_0_5_values) / len(near_0_5_values)))),
                1,
            )
        canopy_fuel_asymmetry = None
        if near_5_30_values:
            canopy_fuel_asymmetry = round(
                max(0.0, min(100.0, max(near_5_30_values) - min(near_5_30_values))),
                1,
            )
        nearest_continuous_distance_ft = (
            round(min(continuous_candidates_ft), 1)
            if continuous_candidates_ft
            else None
        )
        sector_scores = {
            str(direction): float(payload.get("sector_risk_score"))
            for direction, payload in sector_samples.items()
            if isinstance(payload, dict) and payload.get("sector_risk_score") is not None
        }
        max_risk_direction = None
        if sector_scores:
            max_risk_direction = max(
                sorted(sector_scores.keys()),
                key=lambda direction: float(sector_scores.get(direction) or 0.0),
            )

        return {
            "near_structure_vegetation_0_5_pct": near_0_5_pct,
            "near_structure_vegetation_5_30_pct": near_5_30_pct,
            "vegetation_edge_directional_concentration_pct": directional_concentration,
            "canopy_dense_fuel_asymmetry_pct": canopy_fuel_asymmetry,
            "nearest_continuous_vegetation_distance_ft": nearest_continuous_distance_ft,
            "vegetation_directional_sectors": sector_samples,
            "vegetation_directional_precision": (
                "footprint_boundary"
                if has_footprint
                else "point_proxy"
            ),
            "vegetation_directional_precision_score": (
                0.9
                if has_footprint
                else 0.45
            ),
            "vegetation_directional_basis": (
                "structure_boundary_relative"
                if has_footprint
                else "point_proxy_relative"
            ),
            "directional_risk": {
                "max_risk_direction": max_risk_direction,
                "sector_scores": sector_scores,
                "basis": (
                    "footprint_boundary"
                    if has_footprint
                    else "point_proxy"
                ),
                "precision_flag": (
                    "footprint_relative"
                    if has_footprint
                    else "fallback_point_proxy"
                ),
            },
        }

    def _build_near_structure_feature_block(
        self,
        *,
        ring_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = ring_context if isinstance(ring_context, dict) else {}
        ring_metrics = context.get("ring_metrics") if isinstance(context.get("ring_metrics"), dict) else {}
        ring_0_5 = ring_metrics.get("ring_0_5_ft") or ring_metrics.get("zone_0_5_ft") or {}
        ring_5_30 = ring_metrics.get("ring_5_30_ft") or ring_metrics.get("zone_5_30_ft") or {}

        veg_density_0_5 = self._coerce_float(context.get("near_structure_vegetation_0_5_pct"))
        if veg_density_0_5 is None:
            veg_density_0_5 = self._coerce_float((ring_0_5 or {}).get("vegetation_density"))
        veg_density_5_30 = self._coerce_float(context.get("near_structure_vegetation_5_30_pct"))
        if veg_density_5_30 is None:
            veg_density_5_30 = self._coerce_float((ring_5_30 or {}).get("vegetation_density"))

        canopy_overlap = self._coerce_float(context.get("canopy_adjacency_proxy_pct"))
        if canopy_overlap is None:
            canopy_overlap = self._coerce_float((ring_0_5 or {}).get("imagery_canopy_proxy_pct"))
        if canopy_overlap is None:
            canopy_overlap = self._coerce_float((ring_0_5 or {}).get("coverage_pct"))

        hardscape_ratio = self._coerce_float((ring_0_5 or {}).get("imagery_impervious_low_fuel_pct"))
        if hardscape_ratio is None and veg_density_0_5 is not None:
            hardscape_ratio = round(max(0.0, min(100.0, 100.0 - veg_density_0_5)), 1)
        elif hardscape_ratio is None and veg_density_5_30 is not None:
            hardscape_ratio = round(max(0.0, min(100.0, 100.0 - veg_density_5_30)), 1)

        geometry_type = str(
            ring_metrics.get("geometry_type")
            or ("footprint" if bool(context.get("footprint_used")) else "point")
        ).strip().lower()
        if geometry_type not in {"footprint", "point"}:
            geometry_type = "footprint" if bool(context.get("footprint_used")) else "point"
        precision_flag = str(
            ring_metrics.get("precision_flag")
            or ("footprint_relative" if geometry_type == "footprint" else "fallback_point_proxy")
        ).strip().lower()
        if not precision_flag:
            precision_flag = "footprint_relative" if geometry_type == "footprint" else "fallback_point_proxy"

        imagery_available = bool(context.get("naip_feature_source") == "prepared_region_naip")
        if imagery_available and geometry_type == "footprint":
            confidence_flag = "high"
        elif imagery_available:
            confidence_flag = "moderate"
        else:
            confidence_flag = "low"

        notes: list[str] = []
        if imagery_available:
            notes.append("Near-structure features are derived from NAIP imagery enrichment.")
        else:
            notes.append("Near-structure features are approximated from fallback canopy/fuel layers.")
        if precision_flag == "fallback_point_proxy":
            notes.append("Point-based proxy geometry reduces near-structure feature precision.")

        return {
            "veg_density_0_5": veg_density_0_5,
            "veg_density_5_30": veg_density_5_30,
            "canopy_overlap": canopy_overlap,
            "hardscape_ratio": hardscape_ratio,
            "geometry_type": geometry_type,
            "precision_flag": precision_flag,
            "imagery_available": imagery_available,
            "confidence_flag": confidence_flag,
            "source": "naip_imagery" if imagery_available else "fallback_layers",
            "notes": notes,
        }

    def _compute_structure_relative_slope(
        self,
        *,
        slope_path: str,
        origin_lat: float,
        origin_lon: float,
        vegetation_directional_sectors: dict[str, Any] | None,
        geometry_precision_flag: str,
    ) -> dict[str, Any]:
        local_slope = self._sample_raster_point(slope_path, origin_lat, origin_lon) if slope_path else None
        local_slope = round(float(local_slope), 2) if local_slope is not None else None

        sector_slopes: list[float] = []
        sectors = vegetation_directional_sectors if isinstance(vegetation_directional_sectors, dict) else {}
        for payload in sectors.values():
            if not isinstance(payload, dict):
                continue
            slope_deg = self._coerce_float(payload.get("slope_deg"))
            if slope_deg is not None:
                sector_slopes.append(float(slope_deg))

        slope_samples_30 = (
            self._sample_circle(
                slope_path,
                origin_lat,
                origin_lon,
                radius_m=30.0 * 0.3048,
                step_m=10.0,
            )
            if slope_path
            else []
        )
        if slope_samples_30:
            slope_within_30_ft = round(sum(slope_samples_30) / len(slope_samples_30), 2)
        elif sector_slopes:
            slope_within_30_ft = round(sum(sector_slopes) / len(sector_slopes), 2)
        else:
            slope_within_30_ft = local_slope

        slope_reference = local_slope if local_slope is not None else slope_within_30_ft
        uphill_gradient_deg = None
        downhill_gradient_deg = None
        if slope_reference is not None and sector_slopes:
            uphill_gradient_deg = round(max(0.0, max(sector_slopes) - float(slope_reference)), 2)
            downhill_gradient_deg = round(max(0.0, float(slope_reference) - min(sector_slopes)), 2)

        uphill_exposure = (
            round(max(0.0, min(100.0, (float(uphill_gradient_deg) / 25.0) * 100.0)), 1)
            if uphill_gradient_deg is not None
            else None
        )
        downhill_buffer = (
            round(max(0.0, min(100.0, (float(downhill_gradient_deg) / 25.0) * 100.0)), 1)
            if downhill_gradient_deg is not None
            else None
        )
        confidence_flag = "high" if geometry_precision_flag == "footprint_relative" else "moderate"
        if slope_reference is None:
            confidence_flag = "low"

        return {
            "local_slope": local_slope,
            "slope_within_30_ft": slope_within_30_ft,
            "uphill_gradient_deg": uphill_gradient_deg,
            "downhill_gradient_deg": downhill_gradient_deg,
            "uphill_exposure": uphill_exposure,
            "downhill_buffer": downhill_buffer,
            "precision_flag": geometry_precision_flag,
            "confidence_flag": confidence_flag,
            "source": "slope_raster" if slope_path else "unavailable",
        }

    def _sample_annulus(
        self,
        path: str,
        lat: float,
        lon: float,
        *,
        inner_radius_m: float,
        outer_radius_m: float,
        radial_step_m: float = 12.0,
        angular_step_m: float = 12.0,
    ) -> list[float]:
        if not (rasterio and self._file_exists(path)):
            return []
        if outer_radius_m <= 0 or outer_radius_m <= inner_radius_m:
            return []

        values: list[float] = []
        radial = max(inner_radius_m + radial_step_m, inner_radius_m)
        while radial <= outer_radius_m + 1e-6:
            points = max(12, int(2 * math.pi * radial / max(1.0, angular_step_m)))
            for i in range(points):
                theta = 2.0 * math.pi * i / points
                d_lat = self._meters_to_lat_deg(radial * math.sin(theta))
                d_lon = self._meters_to_lon_deg(radial * math.cos(theta), lat)
                sample = self._sample_raster_point(path, lat + d_lat, lon + d_lon)
                if sample is not None:
                    values.append(sample)
            radial += radial_step_m
        return values

    @staticmethod
    def _annulus_area_sqft(inner_ft: float, outer_ft: float) -> float:
        if outer_ft <= inner_ft or outer_ft <= 0.0:
            return 0.0
        return max(0.0, math.pi * ((outer_ft ** 2) - (inner_ft ** 2)))

    @staticmethod
    def _geometry_area_sqft(geom: Any | None) -> float | None:
        if geom is None or getattr(geom, "is_empty", True):
            return None
        if not (Transformer and shapely_transform):
            return None
        try:
            to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
            geom_m = shapely_transform(to_3857, geom)
            area_m2 = float(max(0.0, geom_m.area))
        except Exception:
            return None
        sqft = area_m2 * 10.7639
        return round(max(0.0, sqft), 1)

    @staticmethod
    def _geometry_perimeter_ft(geom: Any | None) -> float | None:
        if geom is None or getattr(geom, "is_empty", True):
            return None
        if not (Transformer and shapely_transform):
            return None
        try:
            to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
            geom_m = shapely_transform(to_3857, geom)
            perimeter_m = float(max(0.0, geom_m.length))
        except Exception:
            return None
        return round(max(0.0, perimeter_m / 0.3048), 1)

    @staticmethod
    def _shape_complexity_proxy(
        *,
        area_sqft: float | None,
        perimeter_ft: float | None,
    ) -> float | None:
        if area_sqft is None or perimeter_ft is None:
            return None
        area_m2 = float(area_sqft) / 10.7639
        perimeter_m = float(perimeter_ft) * 0.3048
        if area_m2 <= 0.0 or perimeter_m <= 0.0:
            return None
        # Compactness ratio: 1.0 is circle-like, higher means more irregular/complex.
        ratio = (perimeter_m * perimeter_m) / max(1e-6, 4.0 * math.pi * area_m2)
        complexity = max(0.0, min(100.0, (ratio - 1.0) * 120.0))
        return round(complexity, 1)

    @staticmethod
    def _density_context_tier(density_index: float | None) -> str:
        if density_index is None:
            return "unknown"
        if density_index >= 70.0:
            return "dense"
        if density_index >= 40.0:
            return "moderate"
        return "sparse"

    @staticmethod
    def _age_proxy_era(proxy_year: float | None) -> str | None:
        if proxy_year is None:
            return None
        if proxy_year < 1960.0:
            return "pre_1960"
        if proxy_year < 1980.0:
            return "1960_1979"
        if proxy_year < 2000.0:
            return "1980_1999"
        if proxy_year < 2015.0:
            return "2000_2014"
        return "2015_plus"

    def _build_structure_attributes(
        self,
        *,
        footprint: Any | None,
        neighbor_metrics: dict[str, Any] | None,
        proxy_year: float | None,
    ) -> dict[str, Any]:
        area_sqft = self._geometry_area_sqft(footprint)
        perimeter_ft = self._geometry_perimeter_ft(footprint)
        shape_complexity = self._shape_complexity_proxy(
            area_sqft=area_sqft,
            perimeter_ft=perimeter_ft,
        )
        nearby_100 = self._coerce_float((neighbor_metrics or {}).get("nearby_structure_count_100_ft"))
        nearby_300 = self._coerce_float((neighbor_metrics or {}).get("nearby_structure_count_300_ft"))
        nearest_ft = self._coerce_float((neighbor_metrics or {}).get("nearest_structure_distance_ft"))
        density_index: float | None = None
        if nearby_100 is not None or nearby_300 is not None or nearest_ft is not None:
            c100 = max(0.0, nearby_100 or 0.0)
            c300 = max(0.0, nearby_300 or 0.0)
            density_component = min(
                100.0,
                ((min(c100, 8.0) / 8.0) * 70.0) + ((min(c300, 24.0) / 24.0) * 30.0),
            )
            proximity_component = None
            if nearest_ft is not None:
                proximity_component = max(0.0, min(100.0, 100.0 - ((nearest_ft / 300.0) * 100.0)))
            if proximity_component is None:
                density_index = round(density_component, 1)
            else:
                density_index = round((0.7 * density_component) + (0.3 * proximity_component), 1)
        inferred_age_proxy = (
            {
                "proxy_year": round(float(proxy_year), 1),
                "era_bucket": self._age_proxy_era(float(proxy_year)),
            }
            if proxy_year is not None
            else None
        )
        provenance = {
            "area": "observed" if area_sqft is not None else "unavailable",
            "density_context": "inferred" if density_index is not None else "unavailable",
            "estimated_age_proxy": "inferred" if inferred_age_proxy is not None else "unavailable",
            "shape_complexity": "inferred" if shape_complexity is not None else "unavailable",
        }
        return {
            "area": {
                "sqft": area_sqft,
                "source": "building_footprint_geometry" if area_sqft is not None else None,
            },
            "density_context": {
                "index": density_index,
                "tier": self._density_context_tier(density_index),
                "nearby_structure_count_100_ft": nearby_100,
                "nearby_structure_count_300_ft": nearby_300,
                "nearest_structure_distance_ft": nearest_ft,
                "source": "neighbor_structure_footprints" if density_index is not None else None,
            },
            "estimated_age_proxy": inferred_age_proxy,
            "shape_complexity": {
                "index": shape_complexity,
                "source": "building_footprint_shape_proxy" if shape_complexity is not None else None,
            },
            "provenance": provenance,
        }

    @staticmethod
    def _estimated_overlap_area_sqft(
        *,
        ring_area_sqft: float | None,
        coverage_pct: float | None,
    ) -> float | None:
        if ring_area_sqft is None or coverage_pct is None:
            return None
        overlap = ring_area_sqft * (max(0.0, min(100.0, float(coverage_pct))) / 100.0)
        return round(max(0.0, overlap), 1)

    def _build_point_proxy_ring_metrics(
        self,
        *,
        lat: float,
        lon: float,
        canopy_path: str,
        fuel_path: str,
    ) -> dict[str, dict[str, float | None]]:
        zone_bounds_ft = {
            "ring_0_5_ft": (0.0, 5.0),
            "ring_5_30_ft": (5.0, 30.0),
            "ring_30_100_ft": (30.0, 100.0),
            "ring_100_300_ft": (100.0, 300.0),
        }
        zone_aliases = {
            "ring_0_5_ft": "zone_0_5_ft",
            "ring_5_30_ft": "zone_5_30_ft",
            "ring_30_100_ft": "zone_30_100_ft",
            "ring_100_300_ft": "zone_100_300_ft",
        }
        metrics: dict[str, dict[str, float | None]] = {}
        for ring_key, (inner_ft, outer_ft) in zone_bounds_ft.items():
            inner_m = inner_ft * 0.3048
            outer_m = outer_ft * 0.3048
            canopy_vals = self._sample_annulus(
                canopy_path,
                lat,
                lon,
                inner_radius_m=inner_m,
                outer_radius_m=outer_m,
            )
            fuel_vals = self._sample_annulus(
                fuel_path,
                lat,
                lon,
                inner_radius_m=inner_m,
                outer_radius_m=outer_m,
            )
            canopy_stats: dict[str, float] | None = None
            if canopy_vals:
                canopy_mean = float(sum(canopy_vals) / len(canopy_vals))
                canopy_max = float(max(canopy_vals))
                coverage_pct = float(sum(1 for v in canopy_vals if v >= 20.0) / len(canopy_vals) * 100.0)
                canopy_stats = {
                    "canopy_mean": round(canopy_mean, 1),
                    "canopy_max": round(canopy_max, 1),
                    "coverage_pct": round(coverage_pct, 1),
                    "vegetation_density": round(float(self._to_index(canopy_mean, 0.0, 100.0)), 1),
                }

            fuel_presence = self._fuel_presence_proxy_from_values(fuel_vals)
            if canopy_stats is None and fuel_presence is None:
                zone = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": None,
                    "coverage_pct": None,
                    "fuel_presence_proxy": None,
                    "basis": "point_proxy",
                }
            elif canopy_stats is None:
                zone = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": fuel_presence,
                    "coverage_pct": fuel_presence,
                    "fuel_presence_proxy": fuel_presence,
                    "basis": "point_proxy",
                }
            else:
                vegetation_density = canopy_stats["vegetation_density"]
                if fuel_presence is not None:
                    vegetation_density = round((vegetation_density + fuel_presence) / 2.0, 1)
                ring_area_sqft = round(self._annulus_area_sqft(inner_ft, outer_ft), 1)
                zone = {
                    "canopy_mean": canopy_stats["canopy_mean"],
                    "canopy_max": canopy_stats["canopy_max"],
                    "vegetation_density": vegetation_density,
                    "coverage_pct": canopy_stats["coverage_pct"],
                    "fuel_presence_proxy": fuel_presence,
                    "ring_area_sqft": ring_area_sqft,
                    "vegetated_overlap_area_sqft": self._estimated_overlap_area_sqft(
                        ring_area_sqft=ring_area_sqft,
                        coverage_pct=canopy_stats["coverage_pct"],
                    ),
                    "basis": "point_proxy",
                }
            if "ring_area_sqft" not in zone:
                ring_area_sqft = round(self._annulus_area_sqft(inner_ft, outer_ft), 1)
                zone["ring_area_sqft"] = ring_area_sqft
                zone["vegetated_overlap_area_sqft"] = self._estimated_overlap_area_sqft(
                    ring_area_sqft=ring_area_sqft,
                    coverage_pct=self._coerce_float(zone.get("coverage_pct")),
                )
            metrics[ring_key] = zone
            metrics[zone_aliases[ring_key]] = dict(zone)
        metrics["_meta"] = {
            "geometry_type": "point",
            "precision_flag": "fallback_point_proxy",
            "ring_generation_mode": "point_annulus_fallback",
            "ring_definition_ft": {
                "ring_0_5_ft": [0.0, 5.0],
                "ring_5_30_ft": [5.0, 30.0],
                "ring_30_100_ft": [30.0, 100.0],
            },
        }
        metrics["geometry_type"] = "point"
        metrics["precision_flag"] = "fallback_point_proxy"
        return metrics

    @staticmethod
    def _extract_structure_id_from_payload(raw: dict[str, Any] | None) -> str | None:
        if not isinstance(raw, dict):
            return None
        for container_key in ("properties",):
            props = raw.get(container_key)
            if not isinstance(props, dict):
                continue
            for key in ("structure_id", "building_id", "id", "objectid", "OBJECTID", "globalid", "GlobalID"):
                value = props.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        for key in ("structure_id", "building_id", "id"):
            value = raw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _normalize_source_path(path: str | None) -> str:
        if not path:
            return ""
        try:
            return str(Path(str(path)).expanduser().resolve())
        except Exception:
            return str(path)

    @staticmethod
    def _infer_footprint_source_label(source_path: str | None) -> str | None:
        raw = str(source_path or "").strip()
        if not raw:
            return None
        lower = raw.lower()
        if "microsoft" in lower:
            return "microsoft_building_footprints"
        if "overture" in lower or "osm" in lower:
            return "openstreetmap_buildings"
        if "fema" in lower and "structure" in lower:
            return "openstreetmap_buildings"
        return "regional_dataset"

    @staticmethod
    def _coerce_selected_structure_geometry(
        raw_geometry: dict[str, Any] | None,
    ) -> tuple[Any | None, str | None]:
        if not isinstance(raw_geometry, dict):
            return None, "User-selected structure geometry payload is missing or invalid."
        if not shape:
            return None, "Shapely is unavailable; cannot parse user-selected structure geometry."
        geometry_obj = raw_geometry.get("geometry") if raw_geometry.get("type") == "Feature" else raw_geometry
        if not isinstance(geometry_obj, dict):
            return None, "User-selected structure payload is missing a GeoJSON geometry object."
        try:
            geom = shape(geometry_obj)
        except Exception as exc:
            return None, f"User-selected structure geometry could not be parsed: {exc}"
        if geom.is_empty:
            return None, "User-selected structure geometry is empty."
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            return None, "User-selected structure must be a polygon footprint."
        return geom, None

    @staticmethod
    def _coerce_user_selected_point(
        raw_point: dict[str, Any] | None,
    ) -> tuple[tuple[float, float] | None, str | None]:
        if not isinstance(raw_point, dict):
            return None, "User-selected point payload is missing or invalid."
        try:
            lat = float(raw_point.get("latitude"))
            lon = float(raw_point.get("longitude"))
        except (TypeError, ValueError):
            return None, "User-selected point must include numeric latitude and longitude."
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None, "User-selected point coordinates are out of bounds."
        return (lat, lon), None

    def _compute_structure_ring_metrics(
        self,
        lat: float,
        lon: float,
        *,
        canopy_path: str,
        fuel_path: str,
        slope_path: str = "",
        footprint_priority_paths: list[str] | None = None,
        footprint_source_labels: dict[str, str] | None = None,
        footprint_path: str | None = None,
        fallback_footprint_path: str | None = None,
        parcel_polygon: Any | None = None,
        use_parcel_association_for_point_mode: bool = False,
        property_anchor_point: dict[str, Any] | None = None,
        anchor_precision: str | None = None,
        structure_geometry_source: str | None = None,
        selection_mode: str | None = None,
        user_selected_point: dict[str, Any] | None = None,
        selected_structure_id: str | None = None,
        selected_structure_geometry: dict[str, Any] | None = None,
        geocoded_lat: float | None = None,
        geocoded_lon: float | None = None,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        assumptions: list[str] = []
        sources: list[str] = []
        footprint_client = self.footprints
        source_paths: list[str] = []
        if isinstance(footprint_priority_paths, list):
            source_paths = [str(p).strip() for p in footprint_priority_paths if str(p).strip()]
        if not source_paths:
            source_paths = [p for p in [footprint_path, fallback_footprint_path] if p]
        if source_paths:
            footprint_client = BuildingFootprintClient(path=source_paths[0], extra_paths=source_paths[1:])
        source_label_map: dict[str, str] = {}
        if isinstance(footprint_source_labels, dict):
            for path_key, label in footprint_source_labels.items():
                normalized_path = self._normalize_source_path(path_key)
                normalized_label = str(label).strip().lower()
                if normalized_path and normalized_label and normalized_path not in source_label_map:
                    source_label_map[normalized_path] = normalized_label

        def _selected_source_label(source_path: str | None) -> str | None:
            normalized = self._normalize_source_path(source_path)
            if normalized and normalized in source_label_map:
                return source_label_map[normalized]
            return self._infer_footprint_source_label(source_path)

        source_labels_considered: list[str] = []
        for source_path in source_paths:
            label = _selected_source_label(source_path)
            if label and label not in source_labels_considered:
                source_labels_considered.append(label)

        normalized_geometry_source = str(structure_geometry_source or "auto_detected").strip().lower()
        if normalized_geometry_source not in {"auto_detected", "user_selected", "user_modified"}:
            normalized_geometry_source = "auto_detected"
        normalized_selection_mode = str(selection_mode or "polygon").strip().lower()
        if normalized_selection_mode not in {"polygon", "point"}:
            normalized_selection_mode = "polygon"
        user_selected_point_coords: tuple[float, float] | None = None
        if normalized_selection_mode == "point":
            point_payload = property_anchor_point if isinstance(property_anchor_point, dict) else user_selected_point
            user_selected_point_coords, user_point_error = self._coerce_user_selected_point(point_payload)
            if user_selected_point_coords is None:
                assumptions.append(user_point_error or "User-selected point was invalid.")
                assumptions.append("Falling back to polygon/auto structure matching.")
                normalized_selection_mode = "polygon"

        query_lat = float(lat)
        query_lon = float(lon)
        if normalized_selection_mode == "point" and user_selected_point_coords is not None:
            query_lat, query_lon = user_selected_point_coords
            assumptions.append("Using user-selected map point for structure lookup and ring analysis.")

        geocode_fallback_lat = float(geocoded_lat) if geocoded_lat is not None else float(lat)
        geocode_fallback_lon = float(geocoded_lon) if geocoded_lon is not None else float(lon)

        def _infer_point_from_parcel(
            polygon: Any | None,
            *,
            default_lat: float,
            default_lon: float,
        ) -> tuple[float, float]:
            if polygon is None:
                return default_lat, default_lon
            try:
                if bool(getattr(polygon, "is_empty", True)):
                    return default_lat, default_lon
                rep = polygon.representative_point()
                if rep is not None and getattr(rep, "is_empty", True) is False:
                    return float(rep.y), float(rep.x)
                centroid = polygon.centroid
                if centroid is not None and getattr(centroid, "is_empty", True) is False:
                    return float(centroid.y), float(centroid.x)
            except Exception:
                return default_lat, default_lon
            return default_lat, default_lon

        def _derive_age_material_proxy(
            neighbor_metrics_payload: dict[str, Any] | None,
        ) -> tuple[float | None, float | None]:
            if not isinstance(neighbor_metrics_payload, dict):
                return None, None
            nearby_100 = self._coerce_float(neighbor_metrics_payload.get("nearby_structure_count_100_ft"))
            nearby_300 = self._coerce_float(neighbor_metrics_payload.get("nearby_structure_count_300_ft"))
            nearest_ft = self._coerce_float(neighbor_metrics_payload.get("nearest_structure_distance_ft"))
            if nearby_100 is None and nearby_300 is None and nearest_ft is None:
                return None, None
            c100 = max(0.0, nearby_100 or 0.0)
            c300 = max(0.0, nearby_300 or 0.0)
            density_component = min(
                100.0,
                ((min(c100, 8.0) / 8.0) * 70.0) + ((min(c300, 24.0) / 24.0) * 30.0),
            )
            proximity_component = None
            if nearest_ft is not None:
                proximity_component = max(0.0, min(100.0, 100.0 - ((nearest_ft / 300.0) * 100.0)))
            risk_terms: list[tuple[float, float]] = [(0.68, density_component)]
            if proximity_component is not None:
                risk_terms.append((0.32, proximity_component))
            risk_num = sum(weight * value for weight, value in risk_terms)
            risk_den = sum(weight for weight, _ in risk_terms)
            proxy_risk = max(0.0, min(100.0, risk_num / max(risk_den, 1e-6)))
            # Coarse "era" proxy used only when explicit construction-year data is missing.
            proxy_year = max(1945.0, min(2018.0, 2018.0 - ((proxy_risk / 100.0) * 58.0)))
            return round(proxy_year, 1), round(proxy_risk, 1)

        selected_geom: Any | None = None
        if normalized_geometry_source in {"user_selected", "user_modified"}:
            selected_geom, geometry_error = self._coerce_selected_structure_geometry(selected_structure_geometry)
            if selected_geom is None:
                assumptions.append(geometry_error or "User-selected structure geometry was invalid.")
                assumptions.append("Falling back to automatic building footprint detection.")

        if selected_geom is not None:
            centroid = selected_geom.centroid
            selected_id = selected_structure_id or self._extract_structure_id_from_payload(selected_structure_geometry)
            assumptions.append("Structure footprint was confirmed by the user on the map.")
            result = BuildingFootprintResult(
                found=True,
                footprint=selected_geom,
                centroid=(float(centroid.y), float(centroid.x)),
                source="user_selected_structure",
                confidence=1.0,
                match_status="matched",
                match_method=normalized_geometry_source,
                matched_structure_id=selected_id,
                match_distance_m=0.0,
                candidate_count=1,
                candidate_summaries=[],
                assumptions=[],
            )
            sources.append("User-selected structure footprint")
        else:
            try:
                try:
                    result = footprint_client.get_building_footprint(
                        query_lat,
                        query_lon,
                        parcel_polygon=(
                            parcel_polygon
                            if (normalized_selection_mode != "point" or use_parcel_association_for_point_mode)
                            else None
                        ),
                        anchor_precision=anchor_precision,
                    )
                except TypeError:
                    # Backward-compatible path for tests/mocks that still expose (lat, lon) signature.
                    result = footprint_client.get_building_footprint(query_lat, query_lon)
            except Exception as exc:  # pragma: no cover - defensive guard for malformed sources
                assumptions.append(f"Building footprint lookup failed: {exc}")
                assumptions.append("Building footprint analysis unavailable; using point-based vegetation context.")
                structure_attributes = self._build_structure_attributes(
                    footprint=None,
                    neighbor_metrics=None,
                    proxy_year=None,
                )
                return {
                    "footprint_used": False,
                    "footprint_found": False,
                    "footprint_status": "error",
                    "footprint_source": None,
                    "geometry_basis": "point",
                    "footprint_confidence": 0.0,
                    "structure_match_status": "error",
                    "structure_match_method": None,
                    "structure_selection_method": "footprint_lookup_error",
                    "matched_structure_id": None,
                    "structure_match_confidence": 0.0,
                    "structure_match_distance_m": None,
                    "candidate_structure_count": 0,
                    "structure_match_candidates": [],
                    "structure_geometry_source": "auto_detected",
                    "selection_mode": normalized_selection_mode,
                    "property_anchor_point": (
                        {"latitude": query_lat, "longitude": query_lon}
                        if normalized_selection_mode == "point"
                        else {"latitude": lat, "longitude": lon}
                    ),
                    "user_selected_point": (
                        {"latitude": query_lat, "longitude": query_lon}
                        if normalized_selection_mode == "point"
                        else None
                    ),
                    "selected_structure_id": selected_structure_id,
                    "selected_structure_geometry": selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None,
                    "final_structure_geometry_source": (
                        "user_selected_point_unsnapped"
                        if normalized_selection_mode == "point"
                        else "auto_detected"
                    ),
                    "structure_geometry_confidence": 0.35 if normalized_selection_mode == "point" else 0.0,
                    "snapped_structure_distance_m": None,
                    "user_selected_point_in_footprint": False,
                    "display_point_source": "property_anchor_point",
                    "matched_structure_centroid": None,
                    "matched_structure_footprint": None,
                    "fallback_mode": "point_based",
                    "geometry_source": "raw_geocode_point",
                    "geometry_confidence": 0.0,
                    "ring_generation_mode": "point_annulus_fallback",
                    "footprint_resolution": {
                        "selected_source": None,
                        "confidence_score": 0.0,
                        "candidates_considered": 0,
                        "fallback_used": True,
                        "match_status": "error",
                        "match_method": None,
                        "match_distance_m": None,
                        "sources_considered": source_labels_considered,
                    },
                    "ring_metrics": None,
                    "nearest_vegetation_distance_ft": None,
                    "near_structure_vegetation_0_5_pct": None,
                    "near_structure_vegetation_5_30_pct": None,
                    "vegetation_edge_directional_concentration_pct": None,
                    "canopy_dense_fuel_asymmetry_pct": None,
                    "nearest_continuous_vegetation_distance_ft": None,
                    "vegetation_directional_sectors": {},
                    "vegetation_directional_precision": "point_proxy",
                    "vegetation_directional_precision_score": 0.2,
                    "vegetation_directional_basis": "point_proxy_relative",
                    "directional_risk": {},
                    "structure_relative_slope": {},
                    "neighboring_structure_metrics": None,
                    "structure_attributes": structure_attributes,
                }, assumptions, sources

        assumptions.extend(result.assumptions)
        match_status = str(
            getattr(
                result,
                "match_status",
                "matched" if bool(getattr(result, "found", False) and getattr(result, "footprint", None) is not None) else "none",
            )
        )
        match_method = getattr(result, "match_method", None)
        match_confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        match_distance = getattr(result, "match_distance_m", None)
        candidate_count = int(getattr(result, "candidate_count", 0) or 0)
        candidate_summaries = list(getattr(result, "candidate_summaries", []) or [])
        force_unsnapped_point = False
        if normalized_selection_mode == "point" and result.found and result.footprint is not None:
            try:
                point_snap_min_confidence = float(
                    str(
                        os.getenv(
                            "WF_POINT_SELECTION_MIN_SNAP_CONFIDENCE",
                            "0.62",
                        )
                    ).strip()
                )
            except ValueError:
                point_snap_min_confidence = 0.62
            point_snap_min_confidence = max(0.0, min(1.0, point_snap_min_confidence))

            try:
                point_snap_max_distance_m = float(
                    str(
                        os.getenv(
                            "WF_POINT_SELECTION_MAX_SNAP_DISTANCE_M",
                            str(getattr(footprint_client, "max_match_distance_m", 25.0) or 25.0),
                        )
                    ).strip()
                )
            except ValueError:
                point_snap_max_distance_m = float(getattr(footprint_client, "max_match_distance_m", 25.0) or 25.0)
            point_snap_max_distance_m = max(1.0, point_snap_max_distance_m)

            inside_selected_point = bool(match_distance is not None and float(match_distance) <= 0.5)
            weak_confidence = match_confidence < point_snap_min_confidence
            too_far = match_distance is None or float(match_distance) > point_snap_max_distance_m
            if not inside_selected_point and (weak_confidence or too_far):
                force_unsnapped_point = True
                match_status = "none"
                assumptions.append(
                    "Nearest footprint match from selected point was low confidence; using selected point as anchor instead of forcing a structure snap."
                )

        if not result.found or result.footprint is None or force_unsnapped_point:
            if force_unsnapped_point:
                assumptions.append("Using point-based vegetation context because structure snap confidence was weak.")
            else:
                assumptions.append("Building footprint analysis unavailable; using point-based vegetation context.")
            fallback_query_lat = float(query_lat)
            fallback_query_lon = float(query_lon)
            fallback_geometry_source = "raw_geocode_point"
            fallback_selection_method = "raw_geocode_point_fallback"
            geometry_basis = "point"
            structure_geometry_confidence = 0.25 if normalized_selection_mode == "point" else 0.2
            if parcel_polygon is not None:
                fallback_query_lat, fallback_query_lon = _infer_point_from_parcel(
                    parcel_polygon,
                    default_lat=fallback_query_lat,
                    default_lon=fallback_query_lon,
                )
                fallback_geometry_source = "parcel_geometry_inferred_home_location"
                fallback_selection_method = "parcel_inferred_home_location"
                geometry_basis = "parcel"
                structure_geometry_confidence = 0.55
                assumptions.append(
                    "Building footprint unavailable; using parcel geometry to infer a home location for ring sampling."
                )
            elif normalized_selection_mode == "point" and user_selected_point_coords is not None:
                fallback_geometry_source = "user_selected_map_point_unsnapped"
                fallback_selection_method = (
                    "point_unsnapped_low_confidence_or_distance"
                    if force_unsnapped_point
                    else "point_unsnapped_no_match"
                )
                structure_geometry_confidence = 0.42
                assumptions.append(
                    "User-selected point could not be snapped to a trusted structure footprint; using point-annulus fallback."
                )
            else:
                fallback_query_lat = geocode_fallback_lat
                fallback_query_lon = geocode_fallback_lon
                fallback_geometry_source = "raw_geocode_point"
                fallback_selection_method = "raw_geocode_point_fallback"
            point_proxy_metrics = self._build_point_proxy_ring_metrics(
                lat=fallback_query_lat,
                lon=fallback_query_lon,
                canopy_path=canopy_path,
                fuel_path=fuel_path,
            )
            nearest_vegetation_distance_ft: float | None = None
            for ring_key, approx_ft in [
                ("ring_0_5_ft", 3.0),
                ("ring_5_30_ft", 18.0),
                ("ring_30_100_ft", 65.0),
                ("ring_100_300_ft", 180.0),
            ]:
                density = (point_proxy_metrics.get(ring_key) or {}).get("vegetation_density")
                if density is not None and float(density) >= 40.0:
                    nearest_vegetation_distance_ft = approx_ft
                    break
            if any(
                (point_proxy_metrics.get(zone) or {}).get("vegetation_density") is not None
                for zone in ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft")
            ):
                assumptions.append("Structure ring metrics were approximated from point-based annulus sampling.")
                sources.append("Point-proxy ring vegetation summaries")
            structure_veg_features = self._compute_structure_aware_vegetation_features(
                origin_lat=fallback_query_lat,
                origin_lon=fallback_query_lon,
                canopy_path=canopy_path,
                fuel_path=fuel_path,
                slope_path=slope_path,
                footprint=None,
            )
            structure_relative_slope = self._compute_structure_relative_slope(
                slope_path=slope_path,
                origin_lat=fallback_query_lat,
                origin_lon=fallback_query_lon,
                vegetation_directional_sectors=structure_veg_features.get("vegetation_directional_sectors"),
                geometry_precision_flag="fallback_point_proxy",
            )
            neighbor_metrics = footprint_client.get_neighbor_structure_metrics(
                lat=fallback_query_lat,
                lon=fallback_query_lon,
                subject_footprint=None,
                source_path=result.source,
                radius_m=300.0 * 0.3048,
            )
            proxy_year, proxy_risk = _derive_age_material_proxy(neighbor_metrics)
            structure_attributes = self._build_structure_attributes(
                footprint=None,
                neighbor_metrics=neighbor_metrics,
                proxy_year=proxy_year,
            )
            status = "not_found"
            assumptions_blob = " ".join(result.assumptions).lower()
            if "not configured" in assumptions_blob or "missing" in assumptions_blob:
                status = "provider_unavailable"
            elif match_status == "ambiguous":
                status = "ambiguous"
            elif match_status == "none":
                status = "not_found"
            final_geometry_source = (
                "user_selected_point_unsnapped"
                if fallback_geometry_source == "user_selected_map_point_unsnapped"
                else (
                    "parcel_inferred_home_location"
                    if fallback_geometry_source == "parcel_geometry_inferred_home_location"
                    else "raw_geocode_point"
                )
            )
            structure_selection_method = "no_footprint_found"
            if status == "provider_unavailable":
                structure_selection_method = "footprint_provider_unavailable"
            elif match_status == "ambiguous":
                structure_selection_method = "ambiguous_candidates_fallback"
            if fallback_selection_method:
                structure_selection_method = fallback_selection_method
            footprint_resolution = {
                "selected_source": None,
                "confidence_score": max(0.0, min(1.0, float(match_confidence or 0.0))),
                "candidates_considered": int(max(candidate_count, len(candidate_summaries))),
                "fallback_used": True,
                "match_status": str(match_status or "none"),
                "match_method": str(match_method) if match_method else None,
                "match_distance_m": (
                    float(match_distance)
                    if match_distance is not None
                    else None
                ),
                "sources_considered": source_labels_considered,
            }
            return {
                "footprint_used": False,
                "footprint_found": False,
                "footprint_status": status,
                "footprint_source": result.source,
                "geometry_basis": geometry_basis,
                "footprint_confidence": result.confidence,
                "structure_match_status": match_status or "none",
                "structure_match_method": match_method,
                "structure_selection_method": structure_selection_method,
                "matched_structure_id": getattr(result, "matched_structure_id", None),
                "structure_match_confidence": match_confidence,
                "structure_match_distance_m": match_distance,
                "candidate_structure_count": candidate_count,
                "structure_match_candidates": candidate_summaries,
                "structure_geometry_source": "auto_detected",
                "selection_mode": normalized_selection_mode,
                "user_selected_point": (
                    {"latitude": query_lat, "longitude": query_lon}
                    if normalized_selection_mode == "point"
                    else None
                ),
                "selected_structure_id": selected_structure_id,
                "selected_structure_geometry": selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None,
                "final_structure_geometry_source": final_geometry_source,
                "structure_geometry_confidence": structure_geometry_confidence,
                "snapped_structure_distance_m": None,
                "user_selected_point_in_footprint": False,
                "display_point_source": "property_anchor_point",
                "fallback_mode": "point_based",
                "geometry_source": fallback_geometry_source,
                "geometry_confidence": structure_geometry_confidence,
                "ring_generation_mode": "point_annulus_fallback",
                "footprint_resolution": footprint_resolution,
                "ring_metrics": point_proxy_metrics if point_proxy_metrics else None,
                "nearest_vegetation_distance_ft": nearest_vegetation_distance_ft,
                "near_structure_vegetation_0_5_pct": structure_veg_features.get("near_structure_vegetation_0_5_pct"),
                "near_structure_vegetation_5_30_pct": structure_veg_features.get("near_structure_vegetation_5_30_pct"),
                "vegetation_edge_directional_concentration_pct": structure_veg_features.get(
                    "vegetation_edge_directional_concentration_pct"
                ),
                "canopy_dense_fuel_asymmetry_pct": structure_veg_features.get(
                    "canopy_dense_fuel_asymmetry_pct"
                ),
                "nearest_continuous_vegetation_distance_ft": structure_veg_features.get(
                    "nearest_continuous_vegetation_distance_ft"
                ),
                "vegetation_directional_sectors": structure_veg_features.get("vegetation_directional_sectors"),
                "vegetation_directional_precision": structure_veg_features.get("vegetation_directional_precision"),
                "vegetation_directional_precision_score": structure_veg_features.get(
                    "vegetation_directional_precision_score"
                ),
                "vegetation_directional_basis": structure_veg_features.get("vegetation_directional_basis"),
                "directional_risk": structure_veg_features.get("directional_risk") or {},
                "structure_relative_slope": structure_relative_slope,
                "neighboring_structure_metrics": neighbor_metrics,
                "building_age_proxy_year": proxy_year,
                "building_age_material_proxy_risk": proxy_risk,
                "structure_attributes": structure_attributes,
            }, assumptions, sources

        sources.append("Building footprint source")
        rings, ring_assumptions = compute_structure_rings(result.footprint)
        assumptions.extend(ring_assumptions)

        ring_metrics: dict[str, dict[str, float | None]] = {}
        zone_aliases = {
            "ring_0_5_ft": "zone_0_5_ft",
            "ring_5_30_ft": "zone_5_30_ft",
            "ring_30_100_ft": "zone_30_100_ft",
            "ring_100_300_ft": "zone_100_300_ft",
        }
        for ring_key in ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "ring_100_300_ft"):
            ring_geom = rings.get(ring_key)
            if ring_geom is None:
                metrics = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": None,
                    "coverage_pct": None,
                    "fuel_presence_proxy": None,
                }
                ring_metrics[ring_key] = metrics
                ring_metrics[zone_aliases[ring_key]] = dict(metrics)
                continue

            canopy_stats = self._summarize_ring_canopy(ring_geom, canopy_path=canopy_path)
            fuel_presence = self._summarize_ring_fuel_presence(ring_geom, fuel_path=fuel_path)

            if canopy_stats is None:
                metrics = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": fuel_presence,
                    "coverage_pct": fuel_presence,
                    "fuel_presence_proxy": fuel_presence,
                }
            else:
                vegetation_density = canopy_stats["vegetation_density"]
                if fuel_presence is not None:
                    vegetation_density = round((vegetation_density + fuel_presence) / 2.0, 1)
                ring_area_sqft = self._geometry_area_sqft(ring_geom)
                metrics = {
                    "canopy_mean": canopy_stats["canopy_mean"],
                    "canopy_max": canopy_stats["canopy_max"],
                    "vegetation_density": vegetation_density,
                    "coverage_pct": canopy_stats["coverage_pct"],
                    "fuel_presence_proxy": fuel_presence,
                    "ring_area_sqft": ring_area_sqft,
                    "vegetated_overlap_area_sqft": self._estimated_overlap_area_sqft(
                        ring_area_sqft=ring_area_sqft,
                        coverage_pct=canopy_stats["coverage_pct"],
                    ),
                }
            if "ring_area_sqft" not in metrics:
                ring_area_sqft = self._geometry_area_sqft(ring_geom)
                metrics["ring_area_sqft"] = ring_area_sqft
                metrics["vegetated_overlap_area_sqft"] = self._estimated_overlap_area_sqft(
                    ring_area_sqft=ring_area_sqft,
                    coverage_pct=self._coerce_float(metrics.get("coverage_pct")),
                )
            ring_metrics[ring_key] = metrics
            ring_metrics[zone_aliases[ring_key]] = dict(metrics)

        nearest_vegetation_distance_ft: float | None = None
        for ring_key, approx_ft in [
            ("ring_0_5_ft", 3.0),
            ("ring_5_30_ft", 18.0),
            ("ring_30_100_ft", 65.0),
            ("ring_100_300_ft", 180.0),
        ]:
            density = (ring_metrics.get(ring_key) or {}).get("vegetation_density")
            if density is not None and float(density) >= 40.0:
                nearest_vegetation_distance_ft = approx_ft
                break
        structure_veg_features = self._compute_structure_aware_vegetation_features(
            origin_lat=query_lat,
            origin_lon=query_lon,
            canopy_path=canopy_path,
            fuel_path=fuel_path,
            slope_path=slope_path,
            footprint=result.footprint,
        )
        slope_origin_lat = float(result.centroid[0]) if result.centroid else float(query_lat)
        slope_origin_lon = float(result.centroid[1]) if result.centroid else float(query_lon)
        structure_relative_slope = self._compute_structure_relative_slope(
            slope_path=slope_path,
            origin_lat=slope_origin_lat,
            origin_lon=slope_origin_lon,
            vegetation_directional_sectors=structure_veg_features.get("vegetation_directional_sectors"),
            geometry_precision_flag="footprint_relative",
        )

        neighbor_metrics = footprint_client.get_neighbor_structure_metrics(
            lat=query_lat,
            lon=query_lon,
            subject_footprint=result.footprint,
            source_path=result.source,
            radius_m=300.0 * 0.3048,
        )
        proxy_year, proxy_risk = _derive_age_material_proxy(neighbor_metrics)
        structure_attributes = self._build_structure_attributes(
            footprint=result.footprint,
            neighbor_metrics=neighbor_metrics,
            proxy_year=proxy_year,
        )

        if ring_metrics:
            sources.append("Structure ring vegetation summaries")
        directional_sectors = structure_veg_features.get("vegetation_directional_sectors")
        if isinstance(directional_sectors, dict) and directional_sectors:
            normalized_directional = {
                str(k): self._coerce_float(v)
                for k, v in directional_sectors.items()
                if str(k).strip()
            }
        else:
            normalized_directional = {}
        ring_metrics["_meta"] = {
            "geometry_type": "footprint",
            "precision_flag": "footprint_relative",
            "ring_generation_mode": "footprint_aware_rings",
            "ring_definition_ft": {
                "ring_0_5_ft": [0.0, 5.0],
                "ring_5_30_ft": [5.0, 30.0],
                "ring_30_100_ft": [30.0, 100.0],
            },
            "directional_segments": normalized_directional,
        }
        ring_metrics["geometry_type"] = "footprint"
        ring_metrics["precision_flag"] = "footprint_relative"

        display_confidence_floor_raw = str(os.getenv("WF_STRUCTURE_DISPLAY_MIN_CONFIDENCE", "0.8")).strip()
        try:
            display_confidence_floor = max(0.0, min(1.0, float(display_confidence_floor_raw)))
        except ValueError:
            display_confidence_floor = 0.8

        structure_match_confidence = match_confidence
        if normalized_geometry_source in {"user_selected", "user_modified"} and selected_geom is not None:
            final_structure_geometry_source = "user_selected_polygon"
            structure_geometry_confidence = 1.0
            snapped_structure_distance_m = 0.0
            structure_selection_method = "user_selected_polygon"
        elif normalized_selection_mode == "point":
            final_structure_geometry_source = "user_selected_point_snapped"
            structure_geometry_confidence = max(0.5, structure_match_confidence)
            snapped_structure_distance_m = (
                float(match_distance) if match_distance is not None else 0.0
            )
            structure_selection_method = (
                "point_inside_footprint_snap"
                if (match_distance is not None and float(match_distance) <= 0.5)
                else (
                    "point_parcel_intersection_snap"
                    if str(match_method or "") == "parcel_intersection"
                    else "point_nearest_footprint_snap"
                )
            )
        else:
            final_structure_geometry_source = "auto_detected"
            structure_geometry_confidence = structure_match_confidence
            snapped_structure_distance_m = None
            structure_selection_method = (
                "parcel_intersection"
                if str(match_method or "") == "parcel_intersection"
                else ("point_in_footprint" if str(match_method or "") == "point_in_footprint" else "nearest_building_fallback")
            )
        geometry_source = "trusted_building_footprint"
        if final_structure_geometry_source == "user_selected_point_snapped":
            geometry_source = "user_selected_map_point_snapped_structure"
        geometry_confidence = max(
            float(structure_geometry_confidence or 0.0),
            float(structure_match_confidence or 0.0),
        )
        display_point_source = (
            "matched_structure_centroid"
            if (
                match_status == "matched"
                and ring_metrics
                and (
                    final_structure_geometry_source == "user_selected_point_snapped"
                    or structure_match_confidence >= display_confidence_floor
                )
            )
            else "property_anchor_point"
        )

        selected_source_label = _selected_source_label(result.source)
        if selected_source_label is None and source_labels_considered:
            selected_source_label = source_labels_considered[0]
        footprint_resolution = {
            "selected_source": selected_source_label,
            "confidence_score": max(0.0, min(1.0, float(structure_match_confidence or 0.0))),
            "candidates_considered": int(max(candidate_count, len(candidate_summaries), 1)),
            "fallback_used": False,
            "match_status": str(match_status or "matched"),
            "match_method": str(match_method) if match_method else None,
            "match_distance_m": (float(match_distance) if match_distance is not None else None),
            "sources_considered": source_labels_considered,
        }

        return {
            "footprint_used": bool(ring_metrics),
            "footprint_found": result.found,
            "footprint_status": "used" if ring_metrics else "error",
            "footprint_source": result.source,
            "geometry_basis": "footprint",
            "footprint_confidence": result.confidence,
            "structure_match_status": match_status or ("matched" if result.found else "none"),
            "structure_match_method": match_method,
            "structure_selection_method": structure_selection_method,
            "matched_structure_id": getattr(result, "matched_structure_id", None),
            "structure_match_confidence": structure_match_confidence,
            "structure_match_distance_m": match_distance,
            "candidate_structure_count": candidate_count,
            "structure_match_candidates": candidate_summaries,
            "structure_geometry_source": (
                normalized_geometry_source if normalized_geometry_source in {"user_selected", "user_modified"} and selected_geom is not None else "auto_detected"
            ),
            "selection_mode": normalized_selection_mode,
            "user_selected_point": (
                {"latitude": query_lat, "longitude": query_lon}
                if normalized_selection_mode == "point"
                else None
            ),
            "selected_structure_id": (
                selected_structure_id
                or getattr(result, "matched_structure_id", None)
                or self._extract_structure_id_from_payload(selected_structure_geometry)
            ),
            "selected_structure_geometry": (
                selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None
            ),
            "final_structure_geometry_source": final_structure_geometry_source,
            "structure_geometry_confidence": structure_geometry_confidence,
            "snapped_structure_distance_m": snapped_structure_distance_m,
            "user_selected_point_in_footprint": (
                bool(normalized_selection_mode == "point" and match_distance is not None and float(match_distance) <= 0.5)
            ),
            "display_point_source": display_point_source,
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "geometry_source": geometry_source,
            "geometry_confidence": geometry_confidence,
            "ring_generation_mode": "footprint_aware_rings",
            "footprint_resolution": footprint_resolution,
            "ring_metrics": ring_metrics,
            "nearest_vegetation_distance_ft": nearest_vegetation_distance_ft,
            "near_structure_vegetation_0_5_pct": structure_veg_features.get("near_structure_vegetation_0_5_pct"),
            "near_structure_vegetation_5_30_pct": structure_veg_features.get("near_structure_vegetation_5_30_pct"),
            "vegetation_edge_directional_concentration_pct": structure_veg_features.get(
                "vegetation_edge_directional_concentration_pct"
            ),
            "canopy_dense_fuel_asymmetry_pct": structure_veg_features.get(
                "canopy_dense_fuel_asymmetry_pct"
            ),
            "nearest_continuous_vegetation_distance_ft": structure_veg_features.get(
                "nearest_continuous_vegetation_distance_ft"
            ),
            "vegetation_directional_sectors": structure_veg_features.get("vegetation_directional_sectors"),
            "vegetation_directional_precision": structure_veg_features.get("vegetation_directional_precision"),
            "vegetation_directional_precision_score": structure_veg_features.get(
                "vegetation_directional_precision_score"
            ),
            "vegetation_directional_basis": structure_veg_features.get("vegetation_directional_basis"),
            "directional_risk": structure_veg_features.get("directional_risk") or {},
            "structure_relative_slope": structure_relative_slope,
            "neighboring_structure_metrics": neighbor_metrics,
            "building_age_proxy_year": proxy_year,
            "building_age_material_proxy_risk": proxy_risk,
            "structure_attributes": structure_attributes,
            "footprint_centroid": {
                "latitude": result.centroid[0] if result.centroid else None,
                "longitude": result.centroid[1] if result.centroid else None,
            },
        }, assumptions, sources

    def _sample_raster_point_raw(self, path: str, lat: float, lon: float) -> float | None:
        if not (rasterio and self._file_exists(path)):
            return None
        ds = self._open_raster(path)
        x, y = self._to_dataset_crs(ds, lon, lat)
        sample = next(ds.sample([(x, y)]))[0]
        nodata = ds.nodata
        if nodata is not None and float(sample) == float(nodata):
            return None
        return float(sample)

    def _sample_raster_point(self, path: str, lat: float, lon: float) -> float | None:
        try:
            return self._sample_raster_point_raw(path, lat, lon)
        except Exception:
            return None

    def _sample_raster_nearby(
        self,
        path: str,
        lat: float,
        lon: float,
        *,
        max_radius_m: float = 120.0,
        step_m: float = 30.0,
    ) -> tuple[float | None, float | None]:
        if not (rasterio and self._file_exists(path)):
            return None, None
        radius_steps = max(1, int(max_radius_m / max(step_m, 1.0)))
        for ring in range(1, radius_steps + 1):
            radius_m = ring * step_m
            points = max(8, int(round((2.0 * math.pi * radius_m) / max(step_m, 1.0))))
            for idx in range(points):
                theta = 2.0 * math.pi * idx / float(points)
                d_lat = self._meters_to_lat_deg(radius_m * math.sin(theta))
                d_lon = self._meters_to_lon_deg(radius_m * math.cos(theta), lat)
                sample = self._sample_raster_point(path, lat + d_lat, lon + d_lon)
                if sample is not None:
                    return float(sample), float(radius_m)
        return None, None

    def _sample_layer_value_detailed(self, path: str, lat: float, lon: float) -> tuple[float | None, str, str | None]:
        if not path:
            return None, "not_configured", "Layer path is not configured."
        if not self._file_exists(path):
            return None, "missing_file", "Configured layer file is missing."
        if not rasterio:
            return None, "sampling_failed", "Raster dependencies unavailable for sampling."
        try:
            ds = self._open_raster(path)
            x, y = self._to_dataset_crs(ds, lon, lat)
            bounds = ds.bounds
            if x < bounds.left or x > bounds.right or y < bounds.bottom or y > bounds.top:
                nearby_value, nearby_distance_m = self._sample_raster_nearby(path, lat, lon)
                if nearby_value is not None:
                    return (
                        float(nearby_value),
                        "ok_nearby",
                        f"Property point was outside layer extent; nearest valid sample within {nearby_distance_m:.1f} m used.",
                    )
                return None, "outside_extent", "Property point is outside layer extent."
            sample = next(ds.sample([(x, y)]))[0]
            nodata = ds.nodata
            if nodata is not None and float(sample) == float(nodata):
                nearby_value, nearby_distance_m = self._sample_raster_nearby(path, lat, lon)
                if nearby_value is not None:
                    return (
                        float(nearby_value),
                        "ok_nearby",
                        f"Property point sampled nodata; nearest valid sample within {nearby_distance_m:.1f} m used.",
                    )
                return None, "outside_extent", "Layer sampled nodata at property location."
            if np is not None and hasattr(np, "isnan") and np.isnan(sample):
                nearby_value, nearby_distance_m = self._sample_raster_nearby(path, lat, lon)
                if nearby_value is not None:
                    return (
                        float(nearby_value),
                        "ok_nearby",
                        f"Property point sampled nodata; nearest valid sample within {nearby_distance_m:.1f} m used.",
                    )
                return None, "outside_extent", "Layer sampled nodata at property location."
            return float(sample), "ok", None
        except Exception as exc:
            return None, "sampling_failed", str(exc)

    def _sample_layer_value(self, path: str, lat: float, lon: float) -> Tuple[float | None, str]:
        value, status, _reason = self._sample_layer_value_detailed(path, lat, lon)
        if status in {"ok", "ok_nearby"} and value is not None:
            return float(value), "ok"
        if status == "sampling_failed":
            return None, "error"
        return None, "missing"

    def _sample_circle(self, path: str, lat: float, lon: float, radius_m: float, step_m: float = 30.0) -> List[float]:
        if not (rasterio and self._file_exists(path)):
            return []

        values: List[float] = []
        rings = max(1, int(radius_m / step_m))
        for ring in range(1, rings + 1):
            r = ring * step_m
            points = max(8, int(2 * math.pi * r / step_m))
            for i in range(points):
                theta = 2.0 * math.pi * i / points
                d_lat = self._meters_to_lat_deg(r * math.sin(theta))
                d_lon = self._meters_to_lon_deg(r * math.cos(theta), lat)
                sample = self._sample_raster_point(path, lat + d_lat, lon + d_lon)
                if sample is not None:
                    values.append(sample)
        return values

    def _derive_slope_aspect_from_dem(self, dem_path: str, lat: float, lon: float, cell_m: float = 30.0) -> tuple[float | None, float | None]:
        center = self._sample_raster_point(dem_path, lat, lon)
        if center is None:
            return None, None

        d_lat = self._meters_to_lat_deg(cell_m)
        d_lon = self._meters_to_lon_deg(cell_m, lat)

        n = self._sample_raster_point(dem_path, lat + d_lat, lon)
        s = self._sample_raster_point(dem_path, lat - d_lat, lon)
        e = self._sample_raster_point(dem_path, lat, lon + d_lon)
        w = self._sample_raster_point(dem_path, lat, lon - d_lon)

        if None in {n, s, e, w}:
            return None, None

        dzdx = (e - w) / (2 * cell_m)
        dzdy = (n - s) / (2 * cell_m)
        slope_rad = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy))
        slope_deg = math.degrees(slope_rad)

        aspect_rad = math.atan2(dzdx, -dzdy)
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0

        return slope_deg, aspect_deg

    def _fuel_combustibility_index(self, fuel_values: Iterable[float]) -> float:
        vals = [v for v in fuel_values if v is not None]
        if not vals:
            return 50.0

        weighted = []
        for raw in vals:
            code = int(round(raw))
            if 1 <= code <= 3:
                weighted.append(20.0 + (code - 1) * 4.0)  # 20-28
            elif 4 <= code <= 9:
                weighted.append(34.0 + (code - 4) * 4.2)  # 34-55
            elif 10 <= code <= 13:
                weighted.append(58.0 + (code - 10) * 6.0)  # 58-76
            elif 101 <= code <= 109:
                weighted.append(68.0 + (code - 101) * 2.0)  # 68-84
            elif 121 <= code <= 124:
                weighted.append(82.0 + (code - 121) * 3.0)  # 82-91
            elif 141 <= code <= 149:
                weighted.append(62.0 + (code - 141) * 2.5)  # 62-82
            else:
                weighted.append(50.0)
        return round(sum(weighted) / len(weighted), 1)

    def _wildland_distance_metrics(
        self,
        lat: float,
        lon: float,
        fuel_path: str,
        canopy_path: str,
    ) -> Tuple[float | None, float | None]:
        if not (rasterio and (self._file_exists(fuel_path) or self._file_exists(canopy_path))):
            return None, None

        for radius in (30, 60, 120, 250, 500, 1000, 2000):
            samples = max(8, int(2 * math.pi * radius / 30))
            for i in range(samples):
                theta = 2.0 * math.pi * i / samples
                d_lat = self._meters_to_lat_deg(radius * math.sin(theta))
                d_lon = self._meters_to_lon_deg(radius * math.cos(theta), lat)
                p_lat, p_lon = lat + d_lat, lon + d_lon
                fuel = self._sample_raster_point(fuel_path, p_lat, p_lon) if self._file_exists(fuel_path) else None
                canopy = self._sample_raster_point(canopy_path, p_lat, p_lon) if self._file_exists(canopy_path) else None

                is_wildland = False
                if fuel is not None:
                    fuel_code = int(round(fuel))
                    is_wildland = (1 <= fuel_code <= 13) or (101 <= fuel_code <= 149)
                if canopy is not None and canopy >= 20:
                    is_wildland = True

                if is_wildland:
                    index = round(max(0.0, min(100.0, 100.0 - (radius / 2000.0) * 100.0)), 1)
                    return float(radius), index

        # Layer exists and no contiguous wildland found within search radius.
        return 2000.0, 0.0

    def _historical_fire_metrics(self, lat: float, lon: float, perimeter_path: str) -> Tuple[float | None, float | None, str]:
        if not (shape and Point and self._file_exists(perimeter_path)):
            return None, None, "missing"

        try:
            geoms = self._load_perimeters(perimeter_path)
        except Exception:
            return None, None, "error"

        if not geoms:
            return None, 0.0, "ok"

        pt = Point(lon, lat)
        near_1km = 0
        near_5km = 0
        nearest_km: float | None = None

        lat_1 = self._meters_to_lat_deg(1000)
        lon_1 = self._meters_to_lon_deg(1000, lat)
        lat_5 = self._meters_to_lat_deg(5000)
        lon_5 = self._meters_to_lon_deg(5000, lat)

        for g in geoms:
            minx, miny, maxx, maxy = g.bounds
            if not (maxx < lon - lon_5 or minx > lon + lon_5 or maxy < lat - lat_5 or miny > lat + lat_5):
                near_5km += 1
            if not (maxx < lon - lon_1 or minx > lon + lon_1 or maxy < lat - lat_1 or miny > lat + lat_1):
                near_1km += 1
            if g.contains(pt):
                near_1km = max(near_1km, 2)
                near_5km = max(near_5km, 3)

            try:
                # Shapely distance here is in degrees for EPSG:4326; convert approximately to km.
                km = float(pt.distance(g)) * 111.32
                nearest_km = km if nearest_km is None else min(nearest_km, km)
            except Exception:
                pass

        score = near_1km * 22.0 + near_5km * 6.0
        return nearest_km, round(max(0.0, min(100.0, score)), 1), "ok"

    def collect_context(
        self,
        lat: float,
        lon: float,
        *,
        geocode_precision: str | None = None,
        structure_geometry_source: str | None = None,
        selection_mode: str | None = None,
        property_anchor_point: dict[str, Any] | None = None,
        user_selected_point: dict[str, Any] | None = None,
        selected_structure_id: str | None = None,
        selected_structure_geometry: dict[str, Any] | None = None,
    ) -> WildfireContext:
        assumptions: List[str] = []
        sources: List[str] = []
        normalized_geocode_precision = str(geocode_precision or "unknown").strip().lower() or "unknown"
        normalized_structure_geometry_source = str(structure_geometry_source or "auto_detected").strip().lower()
        if normalized_structure_geometry_source not in {"auto_detected", "user_selected", "user_modified"}:
            normalized_structure_geometry_source = "auto_detected"
        normalized_selection_mode = str(selection_mode or "polygon").strip().lower()
        if normalized_selection_mode not in {"polygon", "point"}:
            normalized_selection_mode = "polygon"
        user_selected_point_coords: tuple[float, float] | None = None
        if normalized_selection_mode == "point":
            point_payload = property_anchor_point if isinstance(property_anchor_point, dict) else user_selected_point
            user_selected_point_coords, point_error = self._coerce_user_selected_point(point_payload)
            if user_selected_point_coords is None:
                assumptions.append(point_error or "User-selected point was invalid.")
                normalized_selection_mode = "polygon"
        runtime_paths, region_context, runtime_assumptions, runtime_sources = self._resolve_runtime_layer_paths(lat, lon)
        assumptions.extend(runtime_assumptions)
        sources.extend(runtime_sources)
        runtime_paths, enrichment_source_status, enrichment_notes = apply_enrichment_source_fallbacks(runtime_paths)
        assumptions.extend(enrichment_notes[:8])
        for group_key, status_row in enrichment_source_status.items():
            if str(status_row.get("status") or "") != "observed":
                continue
            source_name = str(status_row.get("source") or "").strip()
            if not source_name:
                continue
            sources.append(f"{group_key} source: {source_name}")
        self.paths = dict(runtime_paths)
        cache_key = self.feature_bundle_cache.build_key(
            lat=lat,
            lon=lon,
            runtime_paths=runtime_paths,
            region_context=region_context,
            extras={
                "geocode_precision": normalized_geocode_precision,
                "structure_geometry_source": normalized_structure_geometry_source,
                "selection_mode": normalized_selection_mode,
                "user_selected_point": (
                    {
                        "latitude": round(float(user_selected_point_coords[0]), 7),
                        "longitude": round(float(user_selected_point_coords[1]), 7),
                    }
                    if user_selected_point_coords is not None
                    else None
                ),
                "selected_structure_id": str(selected_structure_id or ""),
                "selected_structure_geometry_hash": (
                    hashlib.sha256(
                        json.dumps(selected_structure_geometry, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                    if isinstance(selected_structure_geometry, dict)
                    else None
                ),
            },
        )
        cached_bundle = self.feature_bundle_cache.load(cache_key)
        if isinstance(cached_bundle, dict):
            cached_context = cached_bundle.get("wildfire_context")
            if isinstance(cached_context, dict):
                try:
                    restored = WildfireContext(**cached_context)
                    if not isinstance(restored.property_level_context, dict):
                        restored.property_level_context = {}
                    restored.property_level_context["feature_bundle_cache_hit"] = True
                    restored.property_level_context["feature_bundle_id"] = cache_key
                    if "Feature bundle cache" not in restored.data_sources:
                        restored.data_sources = list(restored.data_sources) + ["Feature bundle cache"]
                    if "Loaded precomputed feature bundle from cache." not in restored.assumptions:
                        restored.assumptions = list(restored.assumptions) + [
                            "Loaded precomputed feature bundle from cache.",
                        ]
                    return restored
                except Exception:
                    pass
        layer_audit = initialize_layer_audit(runtime_paths, region_context)
        environmental_layer_status: dict[str, str] = {
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "missing",
            "fuel": "missing",
            "canopy": "missing",
            "fire_history": "missing",
        }
        property_level_context: dict[str, Any] = {
            "footprint_used": False,
            "footprint_found": False,
            "footprint_status": "not_found",
            "geometry_basis": "point",
            "structure_match_status": "none",
            "structure_match_method": None,
            "structure_selection_method": "no_footprint_found",
            "structure_match_confidence": 0.0,
            "structure_match_distance_m": None,
            "candidate_structure_count": 0,
            "structure_match_candidates": [],
            "footprint_source": None,
            "display_point_source": "property_anchor_point",
            "fallback_mode": "point_based",
            "geometry_source": "raw_geocode_point",
            "geometry_confidence": 0.0,
            "ring_generation_mode": "point_annulus_fallback",
            "ring_metrics": None,
            "near_structure_vegetation_0_5_pct": None,
            "near_structure_vegetation_5_30_pct": None,
            "vegetation_edge_directional_concentration_pct": None,
            "canopy_dense_fuel_asymmetry_pct": None,
            "nearest_continuous_vegetation_distance_ft": None,
            "vegetation_directional_sectors": {},
            "vegetation_directional_precision": "point_proxy",
            "vegetation_directional_precision_score": 0.0,
            "vegetation_directional_basis": "point_proxy_relative",
            "directional_risk": {},
            "structure_relative_slope": {},
            "structure_attributes": {
                "area": {"sqft": None, "source": None},
                "density_context": {
                    "index": None,
                    "tier": "unknown",
                    "nearby_structure_count_100_ft": None,
                    "nearby_structure_count_300_ft": None,
                    "nearest_structure_distance_ft": None,
                    "source": None,
                },
                "estimated_age_proxy": None,
                "shape_complexity": {"index": None, "source": None},
                "provenance": {
                    "area": "unavailable",
                    "density_context": "unavailable",
                    "estimated_age_proxy": "unavailable",
                    "shape_complexity": "unavailable",
                },
            },
            "hazard_context": {},
            "moisture_context": {},
            "historical_fire_context": {},
            "access_context": {},
            "feature_sampling": {},
            "feature_bundle_id": cache_key,
            "feature_bundle_cache_hit": False,
            "property_anchor_point": {"latitude": float(lat), "longitude": float(lon)},
            "property_anchor_source": "geocoded_address_point",
            "property_anchor_precision": "unknown",
            "geocoded_address_point": {"latitude": float(lat), "longitude": float(lon)},
            "assessed_property_display_point": {"latitude": float(lat), "longitude": float(lon)},
            "structure_geometry_source": normalized_structure_geometry_source,
            "selection_mode": normalized_selection_mode,
            "user_selected_point": (
                {"latitude": user_selected_point_coords[0], "longitude": user_selected_point_coords[1]}
                if user_selected_point_coords is not None
                else None
            ),
            "selected_structure_id": selected_structure_id,
            "selected_structure_geometry": selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None,
            "final_structure_geometry_source": "auto_detected",
            "structure_geometry_confidence": 0.0,
            "snapped_structure_distance_m": None,
            "region_status": region_context.get("region_status"),
            "region_id": region_context.get("region_id"),
            "region_display_name": region_context.get("region_display_name"),
            "region_manifest_path": region_context.get("manifest_path"),
            "region_property_specific_readiness": region_context.get("property_specific_readiness"),
            "region_validation_summary": dict(region_context.get("validation_summary") or {}),
            "region_required_layers_missing": list(region_context.get("required_layers_missing") or []),
            "region_optional_layers_missing": list(region_context.get("optional_layers_missing") or []),
            "region_enrichment_layers_missing": list(region_context.get("enrichment_layers_missing") or []),
            "region_missing_reason_by_layer": dict(region_context.get("missing_reason_by_layer") or {}),
            "anchor_quality": "low",
            "anchor_quality_score": 0.0,
        }
        structure_ring_metrics: dict[str, dict[str, float | None]] = {}

        def _status_for_env(sample_status: str) -> str:
            if sample_status in {"ok", "ok_nearby"}:
                return "ok"
            if sample_status == "sampling_failed":
                return "error"
            return "missing"

        if not (rasterio and np is not None and Transformer is not None):
            assumptions.append("Geospatial stack unavailable; install rasterio/numpy/pyproj/shapely.")
            for lk in ["dem", "slope", "fuel", "canopy", "whp", "mtbs_severity", "gridmet_dryness", "roads"]:
                update_layer_audit(
                    layer_audit,
                    lk,
                    sample_attempted=True,
                    sample_succeeded=False,
                    coverage_status="sampling_failed",
                    failure_reason="Geospatial stack unavailable.",
                )
            layer_audit_rows = [layer_audit[k] for k in sorted(layer_audit.keys())]
            coverage_summary = summarize_layer_audit(layer_audit_rows)
            return WildfireContext(
                environmental_index=None,
                slope_index=None,
                aspect_index=None,
                fuel_index=None,
                moisture_index=None,
                canopy_index=None,
                wildland_distance_index=None,
                historic_fire_index=None,
                burn_probability_index=None,
                hazard_severity_index=None,
                access_exposure_index=None,
                burn_probability=None,
                wildfire_hazard=None,
                slope=None,
                fuel_model=None,
                canopy_cover=None,
                historic_fire_distance=None,
                wildland_distance=None,
                environmental_layer_status=environmental_layer_status,
                data_sources=sources,
                assumptions=assumptions,
                structure_ring_metrics=structure_ring_metrics,
                property_level_context=property_level_context,
                region_context=region_context,
                hazard_context={},
                moisture_context={},
                historical_fire_context={},
                access_context={},
                layer_coverage_audit=layer_audit_rows,
                coverage_summary=coverage_summary,
            )

        anchor_resolver = PropertyAnchorResolver(
            address_points_path=runtime_paths.get("address_points"),
            parcels_path=runtime_paths.get("parcels"),
            parcels_paths=self._resolve_parcel_source_paths(runtime_paths),
        )
        explicit_anchor_override: tuple[float, float] | None = None
        explicit_anchor_source: str | None = None
        explicit_anchor_precision: str | None = None
        if isinstance(property_anchor_point, dict):
            explicit_anchor_override, _override_error = self._coerce_user_selected_point(property_anchor_point)
            if explicit_anchor_override is not None:
                explicit_anchor_source = "user_selected_point"
                explicit_anchor_precision = "user_selected_point"
        if explicit_anchor_override is None and normalized_selection_mode == "point" and user_selected_point_coords is not None:
            explicit_anchor_override = user_selected_point_coords
            explicit_anchor_source = "user_selected_point"
            explicit_anchor_precision = "user_selected_point"
        anchor = anchor_resolver.resolve(
            geocoded_lat=float(lat),
            geocoded_lon=float(lon),
            geocode_provider=None,
            geocode_precision=normalized_geocode_precision,
            geocoded_address=None,
            property_anchor_override=explicit_anchor_override,
            property_anchor_override_source=explicit_anchor_source,
            property_anchor_override_precision=explicit_anchor_precision,
        )
        property_level_context.update(anchor.to_context())
        assumptions.extend(anchor.diagnostics[:2])
        assumptions.extend(anchor.alignment_notes[:2])
        if anchor.address_point_source_name:
            sources.append(f"Address points: {anchor.address_point_source_name}")
        if anchor.parcel_source_name:
            sources.append(f"Parcels: {anchor.parcel_source_name}")

        update_layer_audit(
            layer_audit,
            "address_points",
            sample_attempted=True,
            sample_succeeded=anchor.parcel_address_point_geojson is not None,
            coverage_status=(
                "observed"
                if anchor.parcel_address_point_geojson is not None
                else ("not_configured" if not runtime_paths.get("address_points") else "fallback_used")
            ),
            raw_value_preview={
                "source_name": anchor.address_point_source_name,
                "source_vintage": anchor.address_point_source_vintage,
            }
            if anchor.parcel_address_point_geojson is not None
            else None,
            failure_reason=(
                None
                if anchor.parcel_address_point_geojson is not None
                else "No nearby address-point candidate was selected."
            ),
        )
        update_layer_audit(
            layer_audit,
            "parcels",
            sample_attempted=True,
            sample_succeeded=anchor.parcel_geometry_geojson is not None,
            coverage_status=(
                "observed"
                if anchor.parcel_geometry_geojson is not None
                else ("not_configured" if not runtime_paths.get("parcels") else "fallback_used")
            ),
            raw_value_preview={"parcel_id": anchor.parcel_id} if anchor.parcel_geometry_geojson is not None else None,
            failure_reason=(
                None if anchor.parcel_geometry_geojson is not None else "No containing parcel polygon found for anchor."
            ),
        )

        building_source_paths = self._resolve_building_source_paths(runtime_paths, region_context)
        footprint_source_labels: dict[str, str] = {}
        for runtime_key, label in (
            ("footprints_microsoft", "microsoft_building_footprints"),
            ("footprints_overture", "openstreetmap_buildings"),
            ("fema_structures", "openstreetmap_buildings"),
            ("footprints", "regional_dataset"),
        ):
            candidate_path = str(runtime_paths.get(runtime_key) or "").strip()
            if not candidate_path:
                continue
            normalized_path = self._normalize_source_path(candidate_path)
            if normalized_path and normalized_path not in footprint_source_labels:
                footprint_source_labels[normalized_path] = label
        structure_query_lat = float(anchor.anchor_latitude)
        structure_query_lon = float(anchor.anchor_longitude)
        parcel_for_matching = anchor.parcel_polygon
        point_mode_use_parcel_context = str(
            os.getenv("WF_POINT_SELECTION_USE_PARCEL_CONTEXT", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if normalized_selection_mode == "point" and user_selected_point_coords is not None:
            structure_query_lat = float(user_selected_point_coords[0])
            structure_query_lon = float(user_selected_point_coords[1])
            assumptions.append(
                "Point-based structure selection mode enabled; structure lookup is centered on the user-selected map location."
            )
            if point_mode_use_parcel_context and anchor.parcel_polygon is not None:
                assumptions.append(
                    "Point-based structure selection is using parcel context to improve building matching reliability."
                )
            else:
                parcel_for_matching = None
                assumptions.append(
                    "Point-based structure selection is running without parcel context; weak or distant matches will remain unsnapped."
                )

        ring_context, ring_assumptions, ring_sources = self._compute_structure_ring_metrics(
            structure_query_lat,
            structure_query_lon,
            canopy_path=runtime_paths.get("canopy", ""),
            fuel_path=runtime_paths.get("fuel", ""),
            slope_path=runtime_paths.get("slope", ""),
            footprint_priority_paths=building_source_paths,
            footprint_source_labels=footprint_source_labels,
            footprint_path=runtime_paths.get("footprints"),
            fallback_footprint_path=runtime_paths.get("fema_structures"),
            parcel_polygon=parcel_for_matching,
            use_parcel_association_for_point_mode=point_mode_use_parcel_context,
            property_anchor_point=(
                {"latitude": explicit_anchor_override[0], "longitude": explicit_anchor_override[1]}
                if explicit_anchor_override is not None
                else property_anchor_point
            ),
            anchor_precision=(
                "user_selected_point"
                if (normalized_selection_mode == "point" and user_selected_point_coords is not None)
                else (anchor.anchor_precision or normalized_geocode_precision)
            ),
            structure_geometry_source=normalized_structure_geometry_source,
            selection_mode=normalized_selection_mode,
            user_selected_point=(
                {"latitude": user_selected_point_coords[0], "longitude": user_selected_point_coords[1]}
                if user_selected_point_coords is not None
                else None
            ),
            selected_structure_id=selected_structure_id,
            selected_structure_geometry=selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None,
            geocoded_lat=float(lat),
            geocoded_lon=float(lon),
        )
        ring_context, naip_assumptions, naip_sources = self._apply_naip_feature_enrichment(
            ring_context=ring_context,
            runtime_paths=runtime_paths,
            region_context=region_context,
        )
        ring_context["near_structure_features"] = self._build_near_structure_feature_block(
            ring_context=ring_context
        )
        ring_assumptions.extend(naip_assumptions)
        ring_sources.extend(naip_sources)

        ring_metrics_after_enrichment = ring_context.get("ring_metrics")
        if isinstance(ring_metrics_after_enrichment, dict):
            nearest_veg_ft = None
            for ring_key, approx_ft in [
                ("ring_0_5_ft", 3.0),
                ("ring_5_30_ft", 18.0),
                ("ring_30_100_ft", 65.0),
                ("ring_100_300_ft", 180.0),
            ]:
                density = self._coerce_float((ring_metrics_after_enrichment.get(ring_key) or {}).get("vegetation_density"))
                if density is not None and density >= 40.0:
                    nearest_veg_ft = approx_ft
                    break
            if nearest_veg_ft is not None:
                ring_context["nearest_vegetation_distance_ft"] = nearest_veg_ft

        property_level_context = ring_context
        property_level_context.update(
            {
                **anchor.to_context(),
                "region_status": region_context.get("region_status"),
                "region_id": region_context.get("region_id"),
                "region_display_name": region_context.get("region_display_name"),
                "region_manifest_path": region_context.get("manifest_path"),
                "region_property_specific_readiness": region_context.get("property_specific_readiness"),
                "region_validation_summary": dict(region_context.get("validation_summary") or {}),
                "region_required_layers_missing": list(region_context.get("required_layers_missing") or []),
                "region_optional_layers_missing": list(region_context.get("optional_layers_missing") or []),
                "region_enrichment_layers_missing": list(region_context.get("enrichment_layers_missing") or []),
                "region_missing_reason_by_layer": dict(region_context.get("missing_reason_by_layer") or {}),
                "building_sources": list(region_context.get("building_sources") or []),
                "structure_geometry_source": str(
                    ring_context.get("structure_geometry_source")
                    or normalized_structure_geometry_source
                    or "auto_detected"
                ),
                "selection_mode": str(ring_context.get("selection_mode") or normalized_selection_mode or "polygon"),
                "user_selected_point": (
                    ring_context.get("user_selected_point")
                    if isinstance(ring_context.get("user_selected_point"), dict)
                    else (
                        {"latitude": user_selected_point_coords[0], "longitude": user_selected_point_coords[1]}
                        if user_selected_point_coords is not None
                        else None
                    )
                ),
                "selected_structure_id": ring_context.get("selected_structure_id") or selected_structure_id,
                "selected_structure_geometry": (
                    ring_context.get("selected_structure_geometry")
                    if isinstance(ring_context.get("selected_structure_geometry"), dict)
                    else (selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None)
                ),
                "final_structure_geometry_source": (
                    str(ring_context.get("final_structure_geometry_source") or "auto_detected")
                ),
                "structure_geometry_confidence": (
                    float(ring_context.get("structure_geometry_confidence"))
                    if ring_context.get("structure_geometry_confidence") is not None
                    else 0.0
                ),
                "snapped_structure_distance_m": (
                    float(ring_context.get("snapped_structure_distance_m"))
                    if ring_context.get("snapped_structure_distance_m") is not None
                    else None
                ),
                "geometry_basis": str(ring_context.get("geometry_basis") or ("footprint" if ring_context.get("footprint_used") else "point")),
                "geometry_source": str(
                    ring_context.get("geometry_source")
                    or (
                        "trusted_building_footprint"
                        if ring_context.get("footprint_used")
                        else "raw_geocode_point"
                    )
                ),
                "geometry_confidence": (
                    float(ring_context.get("geometry_confidence"))
                    if ring_context.get("geometry_confidence") is not None
                    else (
                        float(ring_context.get("structure_geometry_confidence"))
                        if ring_context.get("structure_geometry_confidence") is not None
                        else 0.0
                    )
                ),
                "ring_generation_mode": str(
                    ring_context.get("ring_generation_mode")
                    or ("footprint_aware_rings" if ring_context.get("footprint_used") else "point_annulus_fallback")
                ),
                "footprint_source": ring_context.get("footprint_source"),
                "structure_selection_method": (
                    str(ring_context.get("structure_selection_method") or ring_context.get("structure_match_method") or "no_footprint_found")
                ),
            }
        )
        if normalized_selection_mode == "point" and user_selected_point_coords is not None:
            property_level_context["property_anchor_point"] = {
                "latitude": float(user_selected_point_coords[0]),
                "longitude": float(user_selected_point_coords[1]),
            }
            property_level_context["property_anchor_source"] = "user_selected_point"
            property_level_context["property_anchor_precision"] = "user_selected_point"
            property_level_context["display_point_source"] = (
                "matched_structure_centroid"
                if str(property_level_context.get("final_structure_geometry_source") or "") == "user_selected_point_snapped"
                else "property_anchor_point"
            )
            assumptions.append(
                "User-selected map point is being used as the property anchor for this assessment."
            )
        display_point_source = str(property_level_context.get("display_point_source") or "property_anchor_point")
        if (
            display_point_source == "matched_structure_centroid"
            and isinstance(property_level_context.get("footprint_centroid"), dict)
            and property_level_context["footprint_centroid"].get("latitude") is not None
            and property_level_context["footprint_centroid"].get("longitude") is not None
        ):
            property_level_context["assessed_property_display_point"] = {
                "latitude": float(property_level_context["footprint_centroid"]["latitude"]),
                "longitude": float(property_level_context["footprint_centroid"]["longitude"]),
            }
        else:
            property_level_context["display_point_source"] = "property_anchor_point"
            property_level_context["assessed_property_display_point"] = dict(
                property_level_context.get("property_anchor_point") or {
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )

        geocode_to_anchor_distance_m = anchor.geocode_to_anchor_distance_m
        if normalized_selection_mode == "point" and user_selected_point_coords is not None:
            try:
                lat1 = math.radians(float(lat))
                lon1 = math.radians(float(lon))
                lat2 = math.radians(float(user_selected_point_coords[0]))
                lon2 = math.radians(float(user_selected_point_coords[1]))
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = (
                    math.sin(dlat / 2.0) ** 2
                    + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2.0) ** 2)
                )
                c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
                geocode_to_anchor_distance_m = round(6371000.0 * c, 2)
            except Exception:
                geocode_to_anchor_distance_m = anchor.geocode_to_anchor_distance_m
        property_level_context["geocode_to_anchor_distance_m"] = geocode_to_anchor_distance_m
        property_level_context["source_conflict_flag"] = bool(anchor.source_conflict_flag)
        property_level_context["alignment_notes"] = list(anchor.alignment_notes)
        property_level_context["parcel_source"] = (
            property_level_context.get("parcel_source_name")
            or property_level_context.get("parcel_source")
        )
        property_level_context["anchor_quality"] = str(property_level_context.get("property_anchor_quality") or "low")
        property_level_context["anchor_quality_score"] = (
            float(property_level_context.get("property_anchor_quality_score"))
            if property_level_context.get("property_anchor_quality_score") is not None
            else 0.0
        )
        selected_source_name = str(
            (ring_context.get("footprint_resolution") or {}).get("selected_source") or ""
        ).strip()
        if not selected_source_name:
            selected_source_name = str(
                os.getenv("WF_FOOTPRINT_SOURCE_NAME", "") or Path(str(ring_context.get("footprint_source") or "")).stem
            ).strip()
        property_level_context["footprint_source_name"] = selected_source_name or None
        property_level_context["footprint_source_vintage"] = str(
            os.getenv("WF_FOOTPRINT_SOURCE_VINTAGE", "")
        ) or None
        property_level_context["building_source"] = property_level_context.get("footprint_source_name")
        selected_footprint_source = str(ring_context.get("footprint_source") or "").lower()
        if "overture" in selected_footprint_source:
            property_level_context["building_source_version"] = (
                str(os.getenv("WF_OVERTURE_BUILDINGS_VERSION", "")).strip()
                or property_level_context.get("footprint_source_vintage")
            )
        else:
            property_level_context["building_source_version"] = property_level_context.get("footprint_source_vintage")
        property_level_context["building_source_confidence"] = (
            float(property_level_context.get("structure_match_confidence"))
            if property_level_context.get("structure_match_confidence") is not None
            else None
        )
        try:
            alignment_warn_distance_m = float(
                str(os.getenv("WF_STRUCTURE_ALIGNMENT_WARN_DISTANCE_M", "20.0")).strip() or "20.0"
            )
        except ValueError:
            alignment_warn_distance_m = 20.0
        property_level_context["structure_alignment_error_flag"] = bool(
            property_level_context.get("structure_match_status") == "matched"
            and (property_level_context.get("structure_match_distance_m") or 0.0) > alignment_warn_distance_m
        )
        if property_level_context["structure_alignment_error_flag"]:
            property_level_context.setdefault("alignment_notes", [])
            property_level_context["alignment_notes"].append(
                "Matched structure is relatively far from the address anchor; verify building alignment."
            )
        footprint_status = str(ring_context.get("footprint_status") or "not_found")
        update_layer_audit(
            layer_audit,
            "building_footprints_overture",
            sample_attempted=True,
            sample_succeeded=bool(ring_context.get("footprint_used"))
            and str(ring_context.get("footprint_source") or "").lower().find("overture") >= 0,
            coverage_status=(
                "observed"
                if (
                    bool(ring_context.get("footprint_used"))
                    and str(ring_context.get("footprint_source") or "").lower().find("overture") >= 0
                )
                else (
                    "not_configured"
                    if not runtime_paths.get("footprints_overture")
                    else "fallback_used"
                )
            ),
            failure_reason=(
                None
                if (
                    bool(ring_context.get("footprint_used"))
                    and str(ring_context.get("footprint_source") or "").lower().find("overture") >= 0
                )
                else "Overture footprint source unavailable or not selected."
            ),
        )
        update_layer_audit(
            layer_audit,
            "building_footprints_microsoft",
            sample_attempted=True,
            sample_succeeded=bool(ring_context.get("footprint_used"))
            and str(ring_context.get("footprint_source") or "").lower().find("microsoft") >= 0,
            coverage_status=(
                "observed"
                if (
                    bool(ring_context.get("footprint_used"))
                    and str(ring_context.get("footprint_source") or "").lower().find("microsoft") >= 0
                )
                else (
                    "not_configured"
                    if not runtime_paths.get("footprints_microsoft")
                    else "fallback_used"
                )
            ),
            failure_reason=(
                None
                if (
                    bool(ring_context.get("footprint_used"))
                    and str(ring_context.get("footprint_source") or "").lower().find("microsoft") >= 0
                )
                else "Microsoft footprint source unavailable or not selected."
            ),
        )
        update_layer_audit(
            layer_audit,
            "building_footprints",
            sample_attempted=True,
            sample_succeeded=bool(ring_context.get("footprint_used")),
            coverage_status=(
                "observed"
                if ring_context.get("footprint_used")
                else ("not_configured" if footprint_status == "provider_unavailable" else "fallback_used")
            ),
            failure_reason=None if ring_context.get("footprint_used") else f"footprint_status={footprint_status}",
        )
        naip_artifact_path = str(property_level_context.get("naip_feature_artifact_path") or "").strip()
        naip_feature_used = bool(property_level_context.get("naip_feature_source") == "prepared_region_naip")
        update_layer_audit(
            layer_audit,
            "naip_structure_features",
            sample_attempted=True,
            sample_succeeded=naip_feature_used,
            coverage_status=(
                "observed"
                if naip_feature_used
                else (
                    "not_configured"
                    if not naip_artifact_path
                    else "fallback_used"
                )
            ),
            raw_value_preview={"artifact_path": naip_artifact_path} if naip_feature_used else None,
            failure_reason=(
                None
                if naip_feature_used
                else "NAIP structure-feature artifact unavailable for matched structure."
            ),
        )
        naip_imagery_path = str(runtime_paths.get("naip_imagery") or "").strip()
        update_layer_audit(
            layer_audit,
            "naip_imagery",
            sample_attempted=bool(naip_imagery_path),
            sample_succeeded=bool(naip_imagery_path and self._file_exists(naip_imagery_path)),
            coverage_status=(
                "observed"
                if (naip_imagery_path and self._file_exists(naip_imagery_path))
                else ("not_configured" if not naip_imagery_path else "missing_file")
            ),
            raw_value_preview={"path": naip_imagery_path} if naip_imagery_path else None,
            failure_reason=(
                None
                if (naip_imagery_path and self._file_exists(naip_imagery_path))
                else ("NAIP imagery path not configured." if not naip_imagery_path else "Configured NAIP imagery file is missing.")
            ),
        )
        neighbor_metrics = ring_context.get("neighboring_structure_metrics")
        neighbor_metrics_observed = bool(
            isinstance(neighbor_metrics, dict)
            and any(
                self._coerce_float((neighbor_metrics or {}).get(field)) is not None
                for field in (
                    "nearby_structure_count_100_ft",
                    "nearby_structure_count_300_ft",
                    "nearest_structure_distance_ft",
                )
            )
        )
        update_layer_audit(
            layer_audit,
            "neighbor_structures",
            sample_attempted=True,
            sample_succeeded=neighbor_metrics_observed,
            coverage_status=(
                "observed"
                if neighbor_metrics_observed
                else "fallback_used"
            ),
            raw_value_preview=neighbor_metrics if isinstance(neighbor_metrics, dict) else None,
            failure_reason=None if isinstance(neighbor_metrics, dict) else "Neighbor structure metrics unavailable.",
        )
        structure_ring_metrics = ring_context.get("ring_metrics", {}) or {}
        assumptions.extend(ring_assumptions)
        sources.extend(ring_sources)

        burn_prob, burn_status_detail, burn_reason = self._sample_layer_value_detailed(runtime_paths["burn_prob"], lat, lon)
        hazard, hazard_status_detail, hazard_reason = self._sample_layer_value_detailed(runtime_paths["hazard"], lat, lon)
        update_layer_audit(
            layer_audit,
            "dem",
            sample_attempted=True,
            sample_succeeded=self._file_exists(runtime_paths.get("dem", "")),
            coverage_status="observed" if self._file_exists(runtime_paths.get("dem", "")) else "missing_file",
        )

        hazard_context: dict[str, Any] = {
            "source": None,
            "status": "missing",
            "raw_value": None,
            "hazard_class": None,
        }
        if burn_prob is None or hazard is None:
            whp_obs = self.whp_adapter.sample(lat=lat, lon=lon, whp_path=runtime_paths.get("whp"))
            update_layer_audit(
                layer_audit,
                "whp",
                sample_attempted=True,
                sample_succeeded=whp_obs.status == "ok",
                coverage_status=(
                    "observed"
                    if whp_obs.status == "ok"
                    else ("missing_file" if runtime_paths.get("whp") else "not_configured")
                ),
                raw_value_preview=whp_obs.raw_value,
                failure_reason=None if whp_obs.status == "ok" else "; ".join(whp_obs.notes[:2]),
            )
            if whp_obs.status == "ok":
                if burn_prob is None:
                    burn_prob = whp_obs.raw_value
                    burn_status_detail = "ok"
                if hazard is None:
                    hazard = whp_obs.raw_value
                    hazard_status_detail = "ok"
                hazard_context = {
                    "source": whp_obs.source_dataset,
                    "status": "observed",
                    "raw_value": whp_obs.raw_value,
                    "hazard_class": whp_obs.hazard_class,
                    "burn_probability_index": whp_obs.burn_probability_index,
                    "hazard_severity_index": whp_obs.hazard_severity_index,
                    "notes": whp_obs.notes,
                }
                sources.append("USFS WHP raster")
            else:
                assumptions.extend(whp_obs.notes)

        environmental_layer_status["burn_probability"] = _status_for_env(burn_status_detail)
        update_layer_audit(
            layer_audit,
            "whp",
            note=burn_reason if burn_reason else None,
        )
        update_layer_audit(
            layer_audit,
            "whp",
            note=hazard_reason if hazard_reason else None,
        )
        if burn_prob is None:
            assumptions.append("Burn probability layer unavailable at property location.")
            burn_probability_index = None
        elif hazard_context.get("burn_probability_index") is not None:
            burn_probability_index = float(hazard_context["burn_probability_index"])
        else:
            burn_probability_index = self._to_index(burn_prob, 0.0, 1.0 if burn_prob <= 1.0 else 100.0)
            sources.append("Burn probability raster")

        environmental_layer_status["hazard"] = _status_for_env(hazard_status_detail)
        if hazard is None:
            assumptions.append("Wildfire hazard severity layer unavailable at property location.")
            hazard_severity_index = None
        elif hazard_context.get("hazard_severity_index") is not None:
            hazard_severity_index = float(hazard_context["hazard_severity_index"])
        else:
            hazard_severity_index = self._to_index(hazard, 0.0, 5.0 if hazard <= 5.0 else 100.0)
            sources.append("Wildfire hazard severity raster")

        slope, slope_status_detail, slope_reason = self._sample_layer_value_detailed(runtime_paths["slope"], lat, lon)
        aspect, aspect_status_detail, _aspect_reason = self._sample_layer_value_detailed(runtime_paths["aspect"], lat, lon)
        if slope is None or aspect is None:
            dem_path = runtime_paths["dem"]
            if self._file_exists(dem_path):
                derived_slope, derived_aspect = self._derive_slope_aspect_from_dem(dem_path, lat, lon)
                if slope is None and derived_slope is not None:
                    slope = derived_slope
                    slope_status_detail = "ok"
                    update_layer_audit(
                        layer_audit,
                        "slope",
                        sample_attempted=True,
                        sample_succeeded=True,
                        coverage_status="observed",
                        raw_value_preview=round(float(derived_slope), 2),
                        note="Slope derived from DEM fallback.",
                    )
                if aspect is None and derived_aspect is not None:
                    aspect = derived_aspect
                    aspect_status_detail = "ok"
                if derived_slope is not None and derived_aspect is not None:
                    sources.append("DEM-derived slope/aspect")
                else:
                    assumptions.append("Could not derive slope/aspect from DEM at property location.")
            else:
                assumptions.append("Slope/aspect rasters missing and DEM not configured.")

        if slope_status_detail not in {"ok", "ok_nearby"}:
            update_layer_audit(
                layer_audit,
                "slope",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "outside_extent"
                    if slope_status_detail == "outside_extent"
                    else ("missing_file" if slope_status_detail == "missing_file" else "not_configured")
                    if slope_status_detail in {"missing_file", "not_configured"}
                    else "sampling_failed"
                ),
                failure_reason=slope_reason,
            )
        elif layer_audit["slope"]["coverage_status"] != "observed":
            update_layer_audit(
                layer_audit,
                "slope",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status=("partial" if slope_status_detail == "ok_nearby" else "observed"),
                raw_value_preview=round(float(slope), 2) if slope is not None else None,
                note=slope_reason if slope_status_detail == "ok_nearby" else None,
            )
        environmental_layer_status["slope"] = _status_for_env(slope_status_detail)
        slope_index = None if slope is None else self._to_index(slope, 0.0, 45.0)
        slope_feature_details: dict[str, Any] = {
            "raw_point_value": slope,
            "index": slope_index,
            "scope": "property_specific" if slope is not None else "fallback",
        }
        if slope is not None and "DEM-derived slope/aspect" not in sources:
            sources.append("Slope raster")

        if aspect is None:
            aspect_index = None
        else:
            a = float(aspect) % 360.0
            # Continuous aspect-exposure transform centered on southwest-facing terrain
            # to avoid coarse 3-bin compression in topography-driven spread signal.
            sw_peak = 225.0
            angular_diff = abs(((a - sw_peak + 180.0) % 360.0) - 180.0)  # 0..180
            aspect_index = round(40.0 + (35.0 * (1.0 - (angular_diff / 180.0))), 1)
            if "DEM-derived slope/aspect" not in sources:
                sources.append("Aspect raster")
        aspect_feature_details: dict[str, Any] = {
            "raw_point_value": aspect,
            "index": aspect_index,
            "scope": "property_specific" if aspect is not None else "fallback",
        }

        fuel_path = runtime_paths["fuel"]
        fuel_center, fuel_status_detail, fuel_reason = self._sample_layer_value_detailed(fuel_path, lat, lon)
        fuel_samples = self._sample_circle(fuel_path, lat, lon, radius_m=100.0) if self._file_exists(fuel_path) else []
        fuel_feature_details: dict[str, Any] = {
            "raw_point_value": fuel_center,
            "sample_count": len(fuel_samples),
            "sampling_radius_m": 100.0,
        }
        if fuel_samples:
            fuel_neighborhood_index = self._fuel_combustibility_index(fuel_samples)
            fuel_center_index = self._fuel_combustibility_index([fuel_center]) if fuel_center is not None else None
            fuel_local_percentile = self._local_percentile_rank(fuel_center, fuel_samples)
            fuel_index = self._blend_indices(
                [
                    (0.55, fuel_neighborhood_index),
                    (0.30, fuel_center_index),
                    (0.15, fuel_local_percentile),
                ]
            )
            fuel_model = round(sum(fuel_samples) / len(fuel_samples), 2)
            environmental_layer_status["fuel"] = "ok"
            sources.append("Fuel model raster")
            fuel_feature_details.update(
                {
                    "neighborhood_index": fuel_neighborhood_index,
                    "point_index": fuel_center_index,
                    "local_percentile": fuel_local_percentile,
                    "blended_index": fuel_index,
                    "scope": "neighborhood_level",
                }
            )
            update_layer_audit(
                layer_audit,
                "fuel",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(fuel_model, 2),
            )
        else:
            fuel_index = None
            fuel_model = None
            environmental_layer_status["fuel"] = _status_for_env(fuel_status_detail)
            fuel_feature_details.update(
                {
                    "neighborhood_index": None,
                    "point_index": None,
                    "local_percentile": None,
                    "blended_index": None,
                    "scope": "fallback",
                }
            )
            update_layer_audit(
                layer_audit,
                "fuel",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "partial"
                    if fuel_status_detail == "ok_nearby"
                    else
                    "outside_extent"
                    if fuel_status_detail == "outside_extent"
                    else ("missing_file" if fuel_status_detail == "missing_file" else "not_configured")
                    if fuel_status_detail in {"missing_file", "not_configured"}
                    else "sampling_failed"
                ),
                raw_value_preview=fuel_center,
                failure_reason=fuel_reason,
            )
            assumptions.append("Fuel model unavailable within 100m neighborhood.")

        canopy_path = runtime_paths["canopy"]
        canopy_center, canopy_status_detail, canopy_reason = self._sample_layer_value_detailed(canopy_path, lat, lon)
        canopy_samples = self._sample_circle(canopy_path, lat, lon, radius_m=100.0) if self._file_exists(canopy_path) else []
        canopy_feature_details: dict[str, Any] = {
            "raw_point_value": canopy_center,
            "sample_count": len(canopy_samples),
            "sampling_radius_m": 100.0,
        }
        if canopy_samples:
            canopy_mean = sum(canopy_samples) / len(canopy_samples)
            canopy_neighborhood_index = self._to_index(canopy_mean, 0.0, 100.0)
            canopy_center_index = self._to_index(float(canopy_center), 0.0, 100.0) if canopy_center is not None else None
            canopy_local_percentile = self._local_percentile_rank(canopy_center, canopy_samples)
            canopy_index = self._blend_indices(
                [
                    (0.50, canopy_neighborhood_index),
                    (0.35, canopy_center_index),
                    (0.15, canopy_local_percentile),
                ]
            )
            canopy_cover = round(canopy_mean, 2)
            environmental_layer_status["canopy"] = "ok"
            sources.append("Canopy density raster")
            canopy_feature_details.update(
                {
                    "neighborhood_index": canopy_neighborhood_index,
                    "point_index": canopy_center_index,
                    "local_percentile": canopy_local_percentile,
                    "blended_index": canopy_index,
                    "scope": "neighborhood_level",
                }
            )
            update_layer_audit(
                layer_audit,
                "canopy",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(canopy_cover, 2),
            )
        else:
            canopy_index = None
            canopy_cover = None
            environmental_layer_status["canopy"] = _status_for_env(canopy_status_detail)
            canopy_feature_details.update(
                {
                    "neighborhood_index": None,
                    "point_index": None,
                    "local_percentile": None,
                    "blended_index": None,
                    "scope": "fallback",
                }
            )
            update_layer_audit(
                layer_audit,
                "canopy",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "partial"
                    if canopy_status_detail == "ok_nearby"
                    else
                    "outside_extent"
                    if canopy_status_detail == "outside_extent"
                    else ("missing_file" if canopy_status_detail == "missing_file" else "not_configured")
                    if canopy_status_detail in {"missing_file", "not_configured"}
                    else "sampling_failed"
                ),
                raw_value_preview=canopy_center,
                failure_reason=canopy_reason,
            )
            assumptions.append("Canopy density unavailable within 100m neighborhood.")

        moisture_context: dict[str, Any] = {
            "source": None,
            "status": "missing",
            "raw_value": None,
            "dryness_index": None,
        }
        moisture, _moisture_status = self._sample_layer_value(runtime_paths["moisture"], lat, lon)
        if moisture is None:
            gridmet_obs = self.gridmet_adapter.sample_dryness(
                lat=lat,
                lon=lon,
                dryness_raster_path=runtime_paths.get("gridmet_dryness"),
            )
            update_layer_audit(
                layer_audit,
                "gridmet_dryness",
                sample_attempted=True,
                sample_succeeded=gridmet_obs.status == "ok",
                coverage_status=(
                    "observed"
                    if gridmet_obs.status == "ok"
                    else ("missing_file" if runtime_paths.get("gridmet_dryness") else "not_configured")
                ),
                raw_value_preview=gridmet_obs.raw_value,
                failure_reason=None if gridmet_obs.status == "ok" else "; ".join(gridmet_obs.notes[:2]),
            )
            if gridmet_obs.status == "ok":
                moisture = gridmet_obs.raw_value
                moisture_index = gridmet_obs.dryness_index
                moisture_context = {
                    "source": gridmet_obs.source_dataset,
                    "status": "observed",
                    "raw_value": gridmet_obs.raw_value,
                    "dryness_index": gridmet_obs.dryness_index,
                    "rolling_window_days": gridmet_obs.rolling_window_days,
                    "notes": gridmet_obs.notes,
                }
                sources.append("gridMET dryness proxy")
            else:
                moisture_index = None
                assumptions.extend(gridmet_obs.notes)
                assumptions.append("Moisture/fuel dryness context unavailable from configured sources.")
        else:
            moisture_index = self._to_index(moisture, 0.0, 100.0)
            moisture_context = {
                "source": "Moisture/fuel dryness raster",
                "status": "observed",
                "raw_value": moisture,
                "dryness_index": moisture_index,
            }
            sources.append("Moisture/fuel dryness raster")
            update_layer_audit(
                layer_audit,
                "gridmet_dryness",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(float(moisture), 2),
                note="Primary moisture layer sampled successfully.",
            )

        wildland_distance, wildland_distance_index = self._wildland_distance_metrics(lat, lon, fuel_path, canopy_path)
        if wildland_distance is not None:
            sources.append("Distance to wildland vegetation (derived)")
        else:
            assumptions.append("Fuel/canopy rasters missing; wildland distance unavailable.")

        mtbs_summary = self.mtbs_adapter.summarize(
            lat=lat,
            lon=lon,
            perimeter_path=runtime_paths.get("perimeters"),
            burn_severity_path=runtime_paths.get("mtbs_severity"),
        )
        historical_fire_context: dict[str, Any] = {
            "source": mtbs_summary.source_dataset,
            "status": mtbs_summary.status,
            "nearest_perimeter_km": mtbs_summary.nearest_perimeter_km,
            "intersects_prior_burn": mtbs_summary.intersects_prior_burn,
            "nearby_high_severity": mtbs_summary.nearby_high_severity,
            "notes": mtbs_summary.notes,
        }
        if mtbs_summary.status == "ok":
            historic_fire_distance = mtbs_summary.nearest_perimeter_km
            historic_fire_index = mtbs_summary.fire_history_index
            fire_history_status = "ok"
            sources.append("MTBS historical fire context")
            update_layer_audit(
                layer_audit,
                "fire_perimeters",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=historic_fire_distance,
            )
            update_layer_audit(
                layer_audit,
                "mtbs_severity",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=mtbs_summary.nearby_high_severity,
            )
        else:
            historic_fire_distance, historic_fire_index, fire_history_status = self._historical_fire_metrics(
                lat, lon, runtime_paths["perimeters"]
            )
            if fire_history_status == "ok":
                sources.append("Historical fire perimeter recurrence")
                update_layer_audit(
                    layer_audit,
                    "fire_perimeters",
                    sample_attempted=True,
                    sample_succeeded=True,
                    coverage_status="observed",
                    raw_value_preview=historic_fire_distance,
                )
            else:
                assumptions.extend(mtbs_summary.notes)
                assumptions.append("Historical perimeter layer unavailable; recurrence unavailable.")
                update_layer_audit(
                    layer_audit,
                    "fire_perimeters",
                    sample_attempted=True,
                    sample_succeeded=False,
                    coverage_status=(
                        "missing_file" if runtime_paths.get("perimeters") else "not_configured"
                    ),
                    failure_reason="Historical perimeter sampling unavailable.",
                )
            update_layer_audit(
                layer_audit,
                "mtbs_severity",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "fallback_used" if runtime_paths.get("mtbs_severity") else "not_configured"
                ),
                failure_reason="MTBS severity context unavailable; perimeter fallback used.",
            )
        environmental_layer_status["fire_history"] = fire_history_status

        access_summary = self.osm_adapter.summarize(
            lat=lat,
            lon=lon,
            roads_path=runtime_paths.get("roads"),
        )
        access_context: dict[str, Any] = {
            "source": access_summary.source_dataset,
            "status": access_summary.status,
            "distance_to_nearest_road_m": access_summary.distance_to_nearest_road_m,
            "road_segments_within_300m": access_summary.road_segments_within_300m,
            "intersections_within_300m": access_summary.intersections_within_300m,
            "dead_end_indicator": access_summary.dead_end_indicator,
            "notes": access_summary.notes,
        }
        access_exposure_index = access_summary.access_exposure_index
        if access_summary.status == "ok" and access_exposure_index is not None:
            sources.append("OSM road-network access metrics")
            update_layer_audit(
                layer_audit,
                "roads",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=access_exposure_index,
            )
        else:
            assumptions.extend(access_summary.notes)
            update_layer_audit(
                layer_audit,
                "roads",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "missing_file"
                    if runtime_paths.get("roads") and not self._file_exists(runtime_paths.get("roads", ""))
                    else ("not_configured" if not runtime_paths.get("roads") else "sampling_failed")
                ),
                failure_reason="; ".join(access_summary.notes[:2]) if access_summary.notes else "Road sampling unavailable.",
            )

        property_level_context.update(
            {
                "hazard_context": hazard_context,
                "moisture_context": moisture_context,
                "historical_fire_context": historical_fire_context,
                "access_context": access_context,
                "enrichment_source_status": enrichment_source_status,
                "feature_sampling": {
                    "burn_probability": {
                        "raw_point_value": burn_prob,
                        "index": burn_probability_index,
                        "scope": "region_level" if burn_prob is not None else "fallback",
                    },
                    "hazard_severity": {
                        "raw_point_value": hazard,
                        "index": hazard_severity_index,
                        "scope": "region_level" if hazard is not None else "fallback",
                    },
                    "slope": slope_feature_details,
                    "aspect": aspect_feature_details,
                    "fuel_model": fuel_feature_details,
                    "canopy_cover": canopy_feature_details,
                    "moisture_dryness": {
                        "raw_point_value": moisture,
                        "index": moisture_index,
                        "scope": "region_level" if moisture is not None else "fallback",
                    },
                    "wildland_distance": {
                        "raw_point_value": wildland_distance,
                        "index": wildland_distance_index,
                        "scope": "neighborhood_level" if wildland_distance is not None else "fallback",
                    },
                    "historic_fire": {
                        "raw_point_value": historic_fire_distance,
                        "index": historic_fire_index,
                        "scope": "region_level" if historic_fire_index is not None else "fallback",
                    },
                    "near_structure_vegetation_0_5_pct": {
                        "raw_point_value": property_level_context.get("near_structure_vegetation_0_5_pct"),
                        "index": property_level_context.get("near_structure_vegetation_0_5_pct"),
                        "local_percentile": (
                            (property_level_context.get("imagery_local_percentiles") or {}).get("near_structure_vegetation_0_5_pct")
                            if isinstance(property_level_context.get("imagery_local_percentiles"), dict)
                            else None
                        ),
                        "scope": (
                            "property_specific"
                            if property_level_context.get("near_structure_vegetation_0_5_pct") is not None
                            else "fallback"
                        ),
                    },
                    "canopy_adjacency_proxy_pct": {
                        "raw_point_value": property_level_context.get("canopy_adjacency_proxy_pct"),
                        "index": property_level_context.get("canopy_adjacency_proxy_pct"),
                        "local_percentile": (
                            (property_level_context.get("imagery_local_percentiles") or {}).get("canopy_adjacency_proxy_pct")
                            if isinstance(property_level_context.get("imagery_local_percentiles"), dict)
                            else None
                        ),
                        "scope": (
                            "property_specific"
                            if property_level_context.get("canopy_adjacency_proxy_pct") is not None
                            else "fallback"
                        ),
                    },
                    "vegetation_continuity_proxy_pct": {
                        "raw_point_value": property_level_context.get("vegetation_continuity_proxy_pct"),
                        "index": property_level_context.get("vegetation_continuity_proxy_pct"),
                        "local_percentile": (
                            (property_level_context.get("imagery_local_percentiles") or {}).get("vegetation_continuity_proxy_pct")
                            if isinstance(property_level_context.get("imagery_local_percentiles"), dict)
                            else None
                        ),
                        "scope": (
                            "property_specific"
                            if property_level_context.get("vegetation_continuity_proxy_pct") is not None
                            else "fallback"
                        ),
                    },
                    "nearest_high_fuel_patch_distance_ft": {
                        "raw_point_value": property_level_context.get("nearest_high_fuel_patch_distance_ft"),
                        "index": (
                            round(
                                max(
                                    0.0,
                                    min(
                                        100.0,
                                        100.0
                                        - (
                                            float(property_level_context.get("nearest_high_fuel_patch_distance_ft"))
                                            / 300.0
                                        )
                                        * 100.0,
                                    ),
                                ),
                                1,
                            )
                            if property_level_context.get("nearest_high_fuel_patch_distance_ft") is not None
                            else None
                        ),
                        "scope": (
                            "property_specific"
                            if property_level_context.get("nearest_high_fuel_patch_distance_ft") is not None
                            else "fallback"
                        ),
                    },
                },
            }
        )
        layer_audit_rows = [layer_audit[k] for k in sorted(layer_audit.keys())]
        feature_bundle_summary = build_feature_bundle_summary(
            lat=lat,
            lon=lon,
            region_context=region_context,
            property_level_context=property_level_context,
            source_status=enrichment_source_status,
            runtime_paths=runtime_paths,
            environmental_layer_status=environmental_layer_status,
            layer_coverage_audit=layer_audit_rows,
        )
        property_level_context["feature_bundle_summary"] = feature_bundle_summary
        property_level_context["feature_bundle_data_sources"] = dict(feature_bundle_summary.get("data_sources") or {})
        property_level_context["feature_bundle_coverage_flags"] = dict(feature_bundle_summary.get("coverage_flags") or {})
        coverage_summary = summarize_layer_audit(layer_audit_rows)
        property_level_context.update(
            {
                "layer_coverage_audit": layer_audit_rows,
                "coverage_summary": coverage_summary,
                "runtime_paths": {k: v for k, v in runtime_paths.items() if v},
            }
        )

        weighted_terms = [
            (0.20, burn_probability_index),
            (0.16, hazard_severity_index),
            (0.12, slope_index),
            (0.06, aspect_index),
            (0.12, fuel_index),
            (0.10, moisture_index),
            (0.10, canopy_index),
            (0.08, wildland_distance_index),
            (0.06, historic_fire_index),
        ]
        available_terms = [(w, v) for (w, v) in weighted_terms if v is not None]
        environmental = None
        if available_terms:
            numerator = sum(w * float(v) for w, v in available_terms)
            denominator = sum(w for w, _ in available_terms)
            environmental = round(numerator / denominator, 1)

        context_result = WildfireContext(
            environmental_index=environmental,
            slope_index=None if slope_index is None else round(slope_index, 1),
            aspect_index=None if aspect_index is None else round(aspect_index, 1),
            fuel_index=None if fuel_index is None else round(fuel_index, 1),
            moisture_index=None if moisture_index is None else round(moisture_index, 1),
            canopy_index=None if canopy_index is None else round(canopy_index, 1),
            wildland_distance_index=None if wildland_distance_index is None else round(wildland_distance_index, 1),
            historic_fire_index=None if historic_fire_index is None else round(historic_fire_index, 1),
            burn_probability_index=None if burn_probability_index is None else round(burn_probability_index, 1),
            hazard_severity_index=None if hazard_severity_index is None else round(hazard_severity_index, 1),
            access_exposure_index=None if access_exposure_index is None else round(access_exposure_index, 1),
            burn_probability=burn_prob,
            wildfire_hazard=hazard,
            slope=slope,
            fuel_model=fuel_model,
            canopy_cover=canopy_cover,
            historic_fire_distance=None if historic_fire_distance is None else round(historic_fire_distance, 2),
            wildland_distance=wildland_distance,
            environmental_layer_status=environmental_layer_status,
            data_sources=sorted(set(sources)),
            assumptions=assumptions,
            structure_ring_metrics=structure_ring_metrics,
            property_level_context=property_level_context,
            region_context=region_context,
            hazard_context=hazard_context,
            moisture_context=moisture_context,
            historical_fire_context=historical_fire_context,
            access_context=access_context,
            layer_coverage_audit=layer_audit_rows,
            coverage_summary=coverage_summary,
        )
        cache_payload = {
            "wildfire_context": asdict(context_result),
        }
        self.feature_bundle_cache.save(cache_key, cache_payload)
        return context_result
