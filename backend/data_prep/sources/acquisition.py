from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


BoundingBox = dict[str, float]


@dataclass
class SourceProviderCapabilities:
    supports_bbox_export: bool
    supports_full_download: bool
    supports_resume: bool = False
    preferred_output_format: str = "bin"


@dataclass
class AcquisitionResult:
    layer_key: str
    provider_type: str
    acquisition_method: str
    source_endpoint: str | None
    source_url: str | None
    local_path: str | None
    bbox_used: str | None
    output_resolution: float | None
    cache_hit: bool
    warnings: list[str]


class SourceProvider(Protocol):
    capabilities: SourceProviderCapabilities

    def fetch_bbox(self, *, layer_key: str, bounds: BoundingBox, cache_root: Path, target_resolution: float | None,
                   timeout_seconds: float, retries: int, backoff_seconds: float) -> AcquisitionResult:
        ...

    def fetch_full(self, *, layer_key: str, source_url: str) -> AcquisitionResult:
        ...

    def clip_local(self, *, layer_key: str, local_path: str) -> AcquisitionResult:
        ...


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _stable_bbox(bounds: BoundingBox) -> str:
    return (
        f"{bounds['min_lon']:.6f},{bounds['min_lat']:.6f},"
        f"{bounds['max_lon']:.6f},{bounds['max_lat']:.6f}"
    )


def _looks_like_html(path: Path) -> bool:
    try:
        data = path.read_bytes()[:2048].lower()
    except Exception:
        return False
    return b"<html" in data or b"<!doctype html" in data or b"<body" in data


def _download_with_retry(
    *,
    url: str,
    out_path: Path,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, open(out_path, "wb") as out:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if out_path.stat().st_size <= 0:
                raise ValueError(f"download returned empty file for {url}")
            if _looks_like_html(out_path):
                raise ValueError(f"download returned HTML/error content for {url}")
            return
        except Exception as exc:  # pragma: no cover - runtime network behavior
            last_exc = exc
            if attempt >= max(0, retries):
                break
            time.sleep(max(0.0, backoff_seconds) * (2**attempt))
    raise ValueError(f"Failed download after retries for {url}: {last_exc}") from last_exc


def _deg_to_meters_lon(deg: float, lat: float) -> float:
    import math

    return abs(deg) * (111_320.0 * max(0.1, math.cos(math.radians(lat))))


def _deg_to_meters_lat(deg: float) -> float:
    return abs(deg) * 110_574.0


class ArcGISImageServiceProvider:
    capabilities = SourceProviderCapabilities(
        supports_bbox_export=True,
        supports_full_download=True,
        supports_resume=False,
        preferred_output_format="tiff",
    )

    def __init__(self, endpoint: str, full_download_url: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.full_download_url = full_download_url

    def _export_endpoint(self) -> str:
        if self.endpoint.endswith("/exportImage"):
            return self.endpoint
        return f"{self.endpoint}/exportImage"

    def _build_export_url(
        self,
        *,
        bounds: BoundingBox,
        target_resolution: float | None,
        out_sr: int = 4326,
        bbox_sr: int = 4326,
    ) -> str:
        bbox = _stable_bbox(bounds)
        lon_span = bounds["max_lon"] - bounds["min_lon"]
        lat_span = bounds["max_lat"] - bounds["min_lat"]
        lat_mid = (bounds["min_lat"] + bounds["max_lat"]) / 2.0
        resolution_m = max(1.0, float(target_resolution or 30.0))
        width = int(max(64, min(8192, _deg_to_meters_lon(lon_span, lat_mid) / resolution_m)))
        height = int(max(64, min(8192, _deg_to_meters_lat(lat_span) / resolution_m)))
        params = {
            "bbox": bbox,
            "bboxSR": str(bbox_sr),
            "imageSR": str(out_sr),
            "size": f"{width},{height}",
            "format": "tiff",
            "f": "image",
        }
        return f"{self._export_endpoint()}?{urllib.parse.urlencode(params)}"

    def fetch_bbox(
        self,
        *,
        layer_key: str,
        bounds: BoundingBox,
        cache_root: Path,
        target_resolution: float | None,
        timeout_seconds: float,
        retries: int,
        backoff_seconds: float,
    ) -> AcquisitionResult:
        bbox = _stable_bbox(bounds)
        cache_dir = cache_root / "bbox_exports"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key_payload = {
            "provider": "arcgis_image_service",
            "endpoint": self.endpoint,
            "layer_key": layer_key,
            "bbox": bbox,
            "target_resolution": target_resolution,
            "format": "tiff",
            "version": "1",
        }
        digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        out_path = cache_dir / f"{digest}_{layer_key}.tif"
        if out_path.exists() and out_path.stat().st_size > 0 and not _looks_like_html(out_path):
            return AcquisitionResult(
                layer_key=layer_key,
                provider_type="arcgis_image_service",
                acquisition_method="cached_bbox_export",
                source_endpoint=self.endpoint,
                source_url=self._build_export_url(bounds=bounds, target_resolution=target_resolution),
                local_path=str(out_path),
                bbox_used=bbox,
                output_resolution=target_resolution,
                cache_hit=True,
                warnings=[],
            )

        url = self._build_export_url(bounds=bounds, target_resolution=target_resolution)
        _download_with_retry(
            url=url,
            out_path=out_path,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_image_service",
            acquisition_method="bbox_export",
            source_endpoint=self.endpoint,
            source_url=url,
            local_path=str(out_path),
            bbox_used=bbox,
            output_resolution=target_resolution,
            cache_hit=False,
            warnings=[],
        )

    def fetch_full(self, *, layer_key: str, source_url: str) -> AcquisitionResult:
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_image_service",
            acquisition_method="full_download_clip",
            source_endpoint=self.endpoint,
            source_url=source_url,
            local_path=None,
            bbox_used=None,
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )

    def clip_local(self, *, layer_key: str, local_path: str) -> AcquisitionResult:
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_image_service",
            acquisition_method="local_existing",
            source_endpoint=self.endpoint,
            source_url=None,
            local_path=local_path,
            bbox_used=None,
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )


class ArcGISFeatureServiceProvider:
    capabilities = SourceProviderCapabilities(
        supports_bbox_export=True,
        supports_full_download=True,
        supports_resume=False,
        preferred_output_format="geojson",
    )

    def __init__(self, endpoint: str, full_download_url: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.full_download_url = full_download_url

    def _query_endpoint(self) -> str:
        if self.endpoint.endswith("/query"):
            return self.endpoint
        return f"{self.endpoint}/query"

    def _build_query_url(self, *, bounds: BoundingBox) -> str:
        bbox = _stable_bbox(bounds)
        params = {
            "where": "1=1",
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
        }
        return f"{self._query_endpoint()}?{urllib.parse.urlencode(params)}"

    def fetch_bbox(
        self,
        *,
        layer_key: str,
        bounds: BoundingBox,
        cache_root: Path,
        target_resolution: float | None,
        timeout_seconds: float,
        retries: int,
        backoff_seconds: float,
    ) -> AcquisitionResult:
        bbox = _stable_bbox(bounds)
        cache_dir = cache_root / "bbox_exports"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key_payload = {
            "provider": "arcgis_feature_service",
            "endpoint": self.endpoint,
            "layer_key": layer_key,
            "bbox": bbox,
            "version": "1",
        }
        digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        out_path = cache_dir / f"{digest}_{layer_key}.geojson"
        if out_path.exists() and out_path.stat().st_size > 0 and not _looks_like_html(out_path):
            return AcquisitionResult(
                layer_key=layer_key,
                provider_type="arcgis_feature_service",
                acquisition_method="cached_bbox_export",
                source_endpoint=self.endpoint,
                source_url=self._build_query_url(bounds=bounds),
                local_path=str(out_path),
                bbox_used=bbox,
                output_resolution=target_resolution,
                cache_hit=True,
                warnings=[],
            )
        url = self._build_query_url(bounds=bounds)
        _download_with_retry(
            url=url,
            out_path=out_path,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_feature_service",
            acquisition_method="bbox_export",
            source_endpoint=self.endpoint,
            source_url=url,
            local_path=str(out_path),
            bbox_used=bbox,
            output_resolution=target_resolution,
            cache_hit=False,
            warnings=[],
        )

    def fetch_full(self, *, layer_key: str, source_url: str) -> AcquisitionResult:
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_feature_service",
            acquisition_method="full_download_clip",
            source_endpoint=self.endpoint,
            source_url=source_url,
            local_path=None,
            bbox_used=None,
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )

    def clip_local(self, *, layer_key: str, local_path: str) -> AcquisitionResult:
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_feature_service",
            acquisition_method="local_existing",
            source_endpoint=self.endpoint,
            source_url=None,
            local_path=local_path,
            bbox_used=None,
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )


def _boolish(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def acquire_layer_from_config(
    *,
    layer_key: str,
    layer_type: str,
    layer_config: dict[str, Any],
    bounds: BoundingBox,
    cache_root: Path,
    prefer_bbox_downloads: bool,
    allow_full_download_fallback: bool,
    target_resolution: float | None,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> AcquisitionResult | None:
    provider_type = str(layer_config.get("provider_type") or "").strip().lower()
    if not provider_type:
        return None

    source_endpoint = str(layer_config.get("source_endpoint") or layer_config.get("source_url") or "").strip()
    full_download_url = str(layer_config.get("full_download_url") or layer_config.get("source_url") or "").strip() or None
    local_path = str(layer_config.get("local_path") or "").strip() or None
    supports_bbox_export = _boolish(layer_config.get("supports_bbox_export"), provider_type.startswith("arcgis_"))

    if local_path:
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type=provider_type,
            acquisition_method="local_existing",
            source_endpoint=source_endpoint or None,
            source_url=None,
            local_path=local_path,
            bbox_used=None,
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )

    if provider_type == "arcgis_image_service":
        provider = ArcGISImageServiceProvider(endpoint=source_endpoint, full_download_url=full_download_url)
    elif provider_type == "arcgis_feature_service":
        provider = ArcGISFeatureServiceProvider(endpoint=source_endpoint, full_download_url=full_download_url)
    elif provider_type in {"file_download", "vector_service"}:
        if not full_download_url:
            return None
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type=provider_type,
            acquisition_method="full_download_clip",
            source_endpoint=source_endpoint or None,
            source_url=full_download_url,
            local_path=None,
            bbox_used=None,
            output_resolution=target_resolution,
            cache_hit=False,
            warnings=[],
        )
    elif provider_type == "local_file":
        if not full_download_url and not local_path:
            return None
        return AcquisitionResult(
            layer_key=layer_key,
            provider_type=provider_type,
            acquisition_method="local_existing",
            source_endpoint=source_endpoint or None,
            source_url=None,
            local_path=full_download_url or local_path,
            bbox_used=None,
            output_resolution=None,
            cache_hit=False,
            warnings=[],
        )
    else:
        return None

    if prefer_bbox_downloads and supports_bbox_export:
        try:
            return provider.fetch_bbox(
                layer_key=layer_key,
                bounds=bounds,
                cache_root=cache_root,
                target_resolution=target_resolution,
                timeout_seconds=timeout_seconds,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
        except Exception as exc:
            if allow_full_download_fallback and full_download_url:
                return AcquisitionResult(
                    layer_key=layer_key,
                    provider_type=provider_type,
                    acquisition_method="full_download_clip",
                    source_endpoint=source_endpoint or None,
                    source_url=full_download_url,
                    local_path=None,
                    bbox_used=_stable_bbox(bounds),
                    output_resolution=target_resolution,
                    cache_hit=False,
                    warnings=[f"bbox export failed; fallback to full download: {exc}"],
                )
            raise

    if full_download_url:
        return provider.fetch_full(layer_key=layer_key, source_url=full_download_url)
    return None


def default_source_config() -> dict[str, Any]:
    return {
        "version": 1,
        "generated_at": _now(),
        "layers": {},
    }

