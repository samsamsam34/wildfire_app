from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import urllib.request
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

try:
    from shapely import wkb as shapely_wkb
    from shapely.geometry import box, mapping, shape
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional dependency
    shapely_wkb = None
    box = None
    mapping = None
    shape = None
    unary_union = None


class OvertureParcelFetcher:
    """Fetch approximate parcel-like boundaries from Overture layers."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _geo_ready() -> bool:
        return bool(shapely_wkb and box and mapping and shape)

    def _download_to_tempfile(self, url: str) -> Path:
        parsed = urllib.parse.urlparse(url)
        suffix = Path(parsed.path).suffix or ".parquet"
        fd, temp_path = tempfile.mkstemp(prefix="overture_parcels_", suffix=suffix)
        os.close(fd)
        target = Path(temp_path)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wildfire-app/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response, open(target, "wb") as out:
                out.write(response.read())
        except Exception:
            if target.exists():
                target.unlink(missing_ok=True)
            raise
        return target

    @staticmethod
    def _resolve_source(
        *,
        explicit_path_env: str,
        explicit_url_env: str,
        template_env: str,
        state_code: str,
        release: str,
    ) -> str | None:
        local_path = str(os.getenv(explicit_path_env, "")).strip()
        if local_path:
            return local_path
        explicit_url = str(os.getenv(explicit_url_env, "")).strip()
        if explicit_url:
            return explicit_url
        template = str(os.getenv(template_env, "")).strip()
        if template:
            return template.format(state=state_code.lower(), STATE=state_code.upper(), release=release)
        return None

    @staticmethod
    def _is_parcel_like(properties: dict[str, Any]) -> bool:
        tokens = " ".join(
            str(properties.get(key) or "")
            for key in ("type", "subtype", "class", "category", "kind", "land_use")
        ).lower()
        if not tokens:
            return True
        parcel_tokens = (
            "parcel",
            "lot",
            "residential",
            "property",
            "land",
            "tract",
            "cadastral",
            "division",
        )
        return any(token in tokens for token in parcel_tokens)

    def _county_mask_from_tiger(self, bounds: BoundingBox):
        tiger_path = str(os.getenv("WF_TIGER_COUNTY_PATH", "")).strip()
        if not tiger_path or not self._geo_ready():
            return None
        path = Path(tiger_path).expanduser()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        features = payload.get("features") if isinstance(payload, dict) else []
        if not isinstance(features, list):
            return None
        aoi = box(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])
        polygons = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            geom = feature.get("geometry")
            if not isinstance(geom, dict):
                continue
            try:
                shp = shape(geom)
            except Exception:
                continue
            if shp.is_empty or not shp.intersects(aoi):
                continue
            polygons.append(shp.intersection(aoi))
        if not polygons:
            return None
        if unary_union is None:
            return polygons[0]
        return unary_union(polygons)

    def _read_parquet_features(
        self,
        *,
        source_ref: str,
        source_label: str,
        bounds: BoundingBox,
        county_mask: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        try:
            import pyarrow.parquet as pq
        except Exception as exc:  # pragma: no cover - optional dependency
            return [], [f"pyarrow parquet reader unavailable: {exc}"]

        cleanup_path: Path | None = None
        path = source_ref
        if source_ref.lower().startswith(("http://", "https://")):
            cleanup_path = self._download_to_tempfile(source_ref)
            path = str(cleanup_path)

        try:
            table = pq.read_table(path)
        except Exception as exc:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
            return [], [f"Failed reading Overture parquet {source_label}: {exc}"]

        try:
            columns = table.column_names
            geometry_col = "geometry" if "geometry" in columns else None
            if geometry_col is None and "geom" in columns:
                geometry_col = "geom"
            if geometry_col is None:
                return [], [f"Overture parquet {source_label} has no geometry column."]

            rows = table.to_pylist()
            raw_features: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_geom = row.get(geometry_col)
                if raw_geom is None:
                    continue
                geom = None
                if self._geo_ready():
                    try:
                        if isinstance(raw_geom, (bytes, bytearray)):
                            geom = shapely_wkb.loads(bytes(raw_geom))
                        elif isinstance(raw_geom, memoryview):
                            geom = shapely_wkb.loads(raw_geom.tobytes())
                    except Exception:
                        geom = None
                if geom is None or bool(getattr(geom, "is_empty", True)):
                    continue
                if str(getattr(geom, "geom_type", "")) not in {"Polygon", "MultiPolygon"}:
                    continue
                if county_mask is not None:
                    try:
                        if not geom.intersects(county_mask):
                            continue
                        geom = geom.intersection(county_mask)
                    except Exception:
                        continue
                properties = {
                    key: value
                    for key, value in row.items()
                    if key != geometry_col
                }
                if not self._is_parcel_like(properties):
                    continue
                raw_features.append(
                    {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": mapping(geom),
                    }
                )
            clipped = clip_and_filter_polygon_features(features=raw_features, bounds=bounds)
            normalized = [
                normalize_parcel_feature(
                    geometry=dict(feature.get("geometry") or {}),
                    properties=dict(feature.get("properties") or {}),
                    source=f"overture_{source_label}",
                )
                for feature in clipped
            ]
            return normalized, warnings
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

    def fetch(
        self,
        *,
        bounds: BoundingBox,
        region_dir: Path,
        state_code: str,
    ) -> ParcelFetchResult:
        if not self._geo_ready():
            return ParcelFetchResult(
                source_id="overture",
                success=False,
                message="Shapely/WKB support is required for Overture parquet parsing.",
            )

        normalized_bounds = normalize_bounds(bounds)
        release = str(os.getenv("WF_OVERTURE_RELEASE", "2024-10-23.0")).strip()
        division_source = self._resolve_source(
            explicit_path_env="WF_OVERTURE_DIVISION_PARQUET_PATH",
            explicit_url_env="WF_OVERTURE_DIVISION_PARQUET_URL",
            template_env="WF_OVERTURE_DIVISION_PARQUET_URL_TEMPLATE",
            state_code=state_code,
            release=release,
        )
        land_use_source = self._resolve_source(
            explicit_path_env="WF_OVERTURE_LAND_USE_PARQUET_PATH",
            explicit_url_env="WF_OVERTURE_LAND_USE_PARQUET_URL",
            template_env="WF_OVERTURE_LAND_USE_PARQUET_URL_TEMPLATE",
            state_code=state_code,
            release=release,
        )
        if not division_source and not land_use_source:
            return ParcelFetchResult(
                source_id="overture",
                success=False,
                message=(
                    "No Overture parquet source configured. Set WF_OVERTURE_*_PARQUET_PATH/URL"
                    " or *_URL_TEMPLATE."
                ),
            )

        county_mask = self._county_mask_from_tiger(normalized_bounds)
        warnings: list[str] = []
        features: list[dict[str, Any]] = []

        if division_source:
            division_features, division_warnings = self._read_parquet_features(
                source_ref=division_source,
                source_label="division",
                bounds=normalized_bounds,
                county_mask=county_mask,
            )
            features.extend(division_features)
            warnings.extend(division_warnings)
        if land_use_source:
            land_use_features, land_use_warnings = self._read_parquet_features(
                source_ref=land_use_source,
                source_label="land_use",
                bounds=normalized_bounds,
                county_mask=county_mask,
            )
            features.extend(land_use_features)
            warnings.extend(land_use_warnings)

        if not features:
            return ParcelFetchResult(
                source_id="overture",
                success=False,
                message="Overture fetch completed but yielded no parcel-like polygons.",
                warnings=warnings,
            )

        output_path = write_parcel_geojson(
            features=features,
            region_dir=region_dir,
            source_id="overture",
            bounds=normalized_bounds,
        )
        return ParcelFetchResult(
            source_id="overture",
            success=True,
            output_path=str(output_path),
            record_count=len(features),
            warnings=warnings,
            diagnostics={"state_code": state_code.upper(), "release": release},
        )
