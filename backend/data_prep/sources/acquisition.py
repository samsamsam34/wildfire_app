from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
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
    http_status: int | None = None
    response_content_type: str | None = None
    bytes_downloaded: int | None = None


class ArcGISFeatureQueryError(ValueError):
    def __init__(
        self,
        *,
        endpoint: str,
        attempted_formats: list[str],
        attempted_urls: list[str],
        errors: list[str],
    ) -> None:
        self.endpoint = endpoint
        self.attempted_formats = attempted_formats
        self.attempted_urls = attempted_urls
        self.errors = errors
        summary = "; ".join(errors[-2:]) if errors else "no provider error details"
        super().__init__(
            "provider_http_error: ArcGIS feature bbox query failed "
            f"(endpoint={endpoint}, attempted_formats={attempted_formats}, "
            f"attempted_urls={attempted_urls}, last_error={summary})"
        )


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


def _is_probably_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _sanitize_url(url: str | None) -> str:
    text = str(url or "").strip().strip("'\"")
    while text and text[-1] in {":", ";", ","}:
        # Keep scheme-only sentinel intact (e.g. "https:") for validation to reject.
        if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9+.-]*:", text):
            break
        text = text[:-1].rstrip()
    return text


def _validate_request_url(url: str) -> None:
    cleaned = _sanitize_url(url)
    if not _is_probably_http_url(cleaned):
        raise ValueError(f"invalid_request_url={cleaned}")


def _looks_like_html(path: Path) -> bool:
    try:
        data = path.read_bytes()[:2048].lower()
    except Exception:
        return False
    return b"<html" in data or b"<!doctype html" in data or b"<body" in data


def _extract_json_error(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if not raw:
        return "empty payload"
    stripped = raw.lstrip()
    if not stripped.startswith(b"{"):
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("error"), dict):
        err = payload["error"]
        message = str(err.get("message") or "unknown provider error")
        details = err.get("details") or []
        if isinstance(details, list) and details:
            message = f"{message}: {'; '.join(str(d) for d in details)}"
        return message
    if payload.get("type") != "FeatureCollection":
        return "unexpected JSON payload"
    return None


def _download_with_retry(
    *,
    url: str,
    out_path: Path,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    url = _sanitize_url(url)
    _validate_request_url(url)
    last_exc: Exception | None = None
    last_status: int | None = None
    last_content_type: str | None = None
    last_bytes_downloaded: int | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            bytes_downloaded = 0
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, open(out_path, "wb") as out:
                status = getattr(response, "status", None)
                content_type = response.headers.get("Content-Type") if getattr(response, "headers", None) else None
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    bytes_downloaded += len(chunk)
                    out.write(chunk)
            if out_path.stat().st_size <= 0:
                raise ValueError(f"download returned empty file for {url}")
            if _looks_like_html(out_path):
                raise ValueError(f"download returned HTML/error content for {url}")
            json_error = _extract_json_error(out_path)
            if json_error:
                raise ValueError(f"download returned JSON/error content for {url}: {json_error}")
            return {
                "http_status": int(status) if status is not None else None,
                "response_content_type": content_type,
                "bytes_downloaded": bytes_downloaded,
            }
        except Exception as exc:  # pragma: no cover - runtime network behavior
            last_exc = exc
            body_snippet = None
            if isinstance(exc, urllib.error.HTTPError):
                last_status = int(exc.code)
                last_content_type = exc.headers.get("Content-Type") if exc.headers else None
                try:
                    body = exc.read(512)
                    if body:
                        body_snippet = re.sub(r"\s+", " ", body.decode("utf-8", errors="replace")).strip()
                except Exception:
                    body_snippet = None
            last_bytes_downloaded = out_path.stat().st_size if out_path.exists() else None
            if attempt >= max(0, retries):
                break
            time.sleep(max(0.0, backoff_seconds) * (2**attempt))
    body_suffix = f", body_snippet={body_snippet}" if body_snippet else ""
    raise ValueError(
        f"Failed download after retries for {url}: {last_exc} "
        f"(http_status={last_status}, content_type={last_content_type}, bytes_downloaded={last_bytes_downloaded}{body_suffix})"
    ) from last_exc


def _download_json_with_retry(
    *,
    url: str,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    url = _sanitize_url(url)
    _validate_request_url(url)
    last_exc: Exception | None = None
    last_status: int | None = None
    last_content_type: str | None = None
    last_body_snippet: str | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                status = getattr(response, "status", None)
                content_type = response.headers.get("Content-Type") if getattr(response, "headers", None) else None
                raw = response.read()
            if not raw:
                raise ValueError("download returned empty json payload")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("download returned non-object json payload")
            if isinstance(payload.get("error"), dict):
                err = payload["error"]
                message = str(err.get("message") or "unknown provider error")
                details = err.get("details") or []
                if isinstance(details, list) and details:
                    message = f"{message}: {'; '.join(str(d) for d in details)}"
                raise ValueError(f"provider_error={message}")
            return payload, {
                "http_status": int(status) if status is not None else None,
                "response_content_type": content_type,
                "bytes_downloaded": len(raw),
            }
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, urllib.error.HTTPError):
                last_status = int(exc.code)
                last_content_type = exc.headers.get("Content-Type") if exc.headers else None
                try:
                    body = exc.read(512)
                    if body:
                        last_body_snippet = re.sub(r"\s+", " ", body.decode("utf-8", errors="replace")).strip()
                except Exception:
                    last_body_snippet = None
            if attempt >= max(0, retries):
                break
            time.sleep(max(0.0, backoff_seconds) * (2**attempt))
    body_suffix = f", body_snippet={last_body_snippet}" if last_body_snippet else ""
    raise ValueError(
        f"Failed json request after retries for {url}: {last_exc} "
        f"(http_status={last_status}, content_type={last_content_type}{body_suffix})"
    ) from last_exc


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
        self.endpoint = _sanitize_url(endpoint).rstrip("/")
        self.full_download_url = _sanitize_url(full_download_url) or None

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
        download_meta = _download_with_retry(
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
            http_status=download_meta.get("http_status"),
            response_content_type=download_meta.get("response_content_type"),
            bytes_downloaded=download_meta.get("bytes_downloaded"),
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

    def __init__(
        self,
        endpoint: str,
        full_download_url: str | None = None,
        *,
        supports_geojson_direct: bool = True,
        require_return_geometry: bool = True,
        preferred_response_format: str | None = None,
    ):
        self.endpoint = _sanitize_url(endpoint).rstrip("/")
        self.full_download_url = _sanitize_url(full_download_url) or None
        self.supports_geojson_direct = supports_geojson_direct
        self.require_return_geometry = require_return_geometry
        self.preferred_response_format = (preferred_response_format or "").strip().lower()

    def _query_endpoint(self) -> str:
        if self.endpoint.endswith("/query"):
            return self.endpoint
        return f"{self.endpoint}/query"

    def _build_query_url(
        self,
        *,
        bounds: BoundingBox,
        response_format: str = "geojson",
        return_geometry: bool = True,
        result_offset: int | None = None,
        result_record_count: int | None = None,
    ) -> str:
        bbox = _stable_bbox(bounds)
        params = {
            "where": "1=1",
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "returnGeometry": "true" if return_geometry else "false",
            "f": response_format,
        }
        if result_offset is not None:
            params["resultOffset"] = str(max(0, result_offset))
        if result_record_count is not None:
            params["resultRecordCount"] = str(max(1, result_record_count))
        return f"{self._query_endpoint()}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def _esri_geometry_to_geojson(geom: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(geom, dict):
            return None
        if "x" in geom and "y" in geom:
            return {"type": "Point", "coordinates": [geom["x"], geom["y"]]}
        if isinstance(geom.get("points"), list):
            return {"type": "MultiPoint", "coordinates": geom["points"]}
        if isinstance(geom.get("paths"), list):
            paths = geom["paths"]
            if len(paths) == 1:
                return {"type": "LineString", "coordinates": paths[0]}
            return {"type": "MultiLineString", "coordinates": paths}
        if isinstance(geom.get("rings"), list):
            return {"type": "Polygon", "coordinates": geom["rings"]}
        return None

    def _write_geojson_from_esri_json(
        self,
        *,
        payload: dict[str, Any],
        out_path: Path,
    ) -> int:
        if isinstance(payload.get("error"), dict):
            err = payload["error"]
            message = str(err.get("message") or "unknown provider error")
            details = err.get("details") or []
            if isinstance(details, list) and details:
                message = f"{message}: {'; '.join(str(d) for d in details)}"
            raise ValueError(f"provider_payload_error={message}")
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError("esri_json_parse_error: provider json response missing features array")
        out_features: list[dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            properties = feature.get("attributes") if isinstance(feature.get("attributes"), dict) else {}
            geometry = self._esri_geometry_to_geojson(feature.get("geometry"))
            if geometry is None:
                continue
            out_features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometry,
                }
            )
        if not out_features:
            raise ValueError("empty_result: provider json response returned empty feature set")
        try:
            out_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": out_features}),
                encoding="utf-8",
            )
        except Exception as exc:
            raise ValueError(f"output_write_failure: unable to write converted GeoJSON: {exc}") from exc
        return len(out_features)

    @staticmethod
    def _query_page_size() -> int:
        raw_value = str(os.getenv("WF_ARCGIS_FEATURE_QUERY_PAGE_SIZE", "2000")).strip()
        try:
            return max(250, min(5000, int(raw_value)))
        except ValueError:
            return 2000

    @staticmethod
    def _query_max_pages() -> int:
        raw_value = str(os.getenv("WF_ARCGIS_FEATURE_QUERY_MAX_PAGES", "100")).strip()
        try:
            return max(1, min(1000, int(raw_value)))
        except ValueError:
            return 100

    def _download_paginated_esri_json(
        self,
        *,
        bounds: BoundingBox,
        timeout_seconds: float,
        retries: int,
        backoff_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
        page_size = self._query_page_size()
        max_pages = self._query_max_pages()
        all_features: list[dict[str, Any]] = []
        total_bytes = 0
        request_count = 0
        offset = 0
        request_url = self._build_query_url(
            bounds=bounds,
            response_format="json",
            return_geometry=self.require_return_geometry,
            result_offset=0,
            result_record_count=page_size,
        )
        response_content_type: str | None = None
        http_status: int | None = None
        saw_transfer_limit = False
        # Detected after the first page: the server's actual max-record-count cap,
        # which may be smaller than our requested page_size.  Using the actual
        # returned count as the threshold lets the loop terminate correctly
        # regardless of which cap (1 000 / 2 000 / 10 000) the server enforces.
        effective_batch_size: int | None = None

        for _ in range(max_pages):
            page_url = self._build_query_url(
                bounds=bounds,
                response_format="json",
                return_geometry=self.require_return_geometry,
                result_offset=offset,
                result_record_count=page_size,
            )
            payload, page_meta = _download_json_with_retry(
                url=page_url,
                timeout_seconds=timeout_seconds,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
            request_count += 1
            request_url = page_url
            total_bytes += int(page_meta.get("bytes_downloaded") or 0)
            response_content_type = page_meta.get("response_content_type")
            http_status = page_meta.get("http_status")
            features = payload.get("features")
            if not isinstance(features, list):
                raise ValueError("esri_json_parse_error: provider json response missing features array")
            all_features.extend([feature for feature in features if isinstance(feature, dict)])
            current_count = len(features)
            exceeded_transfer_limit = bool(payload.get("exceededTransferLimit"))
            saw_transfer_limit = saw_transfer_limit or exceeded_transfer_limit

            # On the first page with results, record how many features the
            # server actually returned.  Some servers cap at a maxRecordCount
            # that is smaller than our requested page_size without setting
            # exceededTransferLimit, so we cannot rely on the flag alone.
            # Using the actual returned count as the reference lets the loop
            # terminate correctly regardless of the server's cap value.
            if effective_batch_size is None and current_count > 0:
                effective_batch_size = current_count

            if current_count <= 0:
                break
            # Stop when fewer features than the effective batch size were
            # returned — this is the reliable cross-server termination signal.
            if current_count < (effective_batch_size or page_size):
                break
            offset += current_count
        else:
            raise ValueError(
                "provider_pagination_limit_reached: ArcGIS feature pagination exceeded "
                f"{max_pages} pages (page_size={page_size})"
            )

        if not all_features:
            raise ValueError("empty_result: provider json response returned empty feature set")

        warnings: list[str] = []
        if request_count > 1:
            warnings.append(f"json_pagination_requests={request_count}")
        if saw_transfer_limit:
            warnings.append("json_exceeded_transfer_limit_detected")

        combined_payload = {"features": all_features}
        return combined_payload, {
            "http_status": int(http_status) if http_status is not None else None,
            "response_content_type": response_content_type,
            "bytes_downloaded": total_bytes,
        }, warnings, request_url

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
                source_url=self._build_query_url(bounds=bounds, response_format="geojson"),
                local_path=str(out_path),
                bbox_used=bbox,
                output_resolution=target_resolution,
                cache_hit=True,
                warnings=[],
            )
        warnings: list[str] = []
        format_order: list[str] = []
        if self.preferred_response_format in {"geojson", "json"}:
            format_order.append(self.preferred_response_format)
        if self.supports_geojson_direct and "geojson" not in format_order:
            format_order.append("geojson")
        if "json" not in format_order:
            format_order.append("json")

        request_url: str | None = None
        acquisition_method = "bbox_export"
        meta: dict[str, Any] | None = None
        last_exc: Exception | None = None
        attempted_urls: list[str] = []
        attempt_errors: list[str] = []

        for fmt in format_order:
            try:
                if fmt == "geojson":
                    geojson_url = self._build_query_url(
                        bounds=bounds,
                        response_format="geojson",
                        return_geometry=self.require_return_geometry,
                    )
                    attempted_urls.append(geojson_url)
                    download_meta = _download_with_retry(
                        url=geojson_url,
                        out_path=out_path,
                        timeout_seconds=timeout_seconds,
                        retries=retries,
                        backoff_seconds=backoff_seconds,
                    )
                    request_url = geojson_url
                    acquisition_method = "bbox_export"
                    meta = download_meta
                    break

                json_url = self._build_query_url(
                    bounds=bounds,
                    response_format="json",
                    return_geometry=self.require_return_geometry,
                    result_offset=0,
                    result_record_count=self._query_page_size(),
                )
                attempted_urls.append(json_url)
                payload, json_meta, pagination_warnings, request_url = self._download_paginated_esri_json(
                    bounds=bounds,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
                self._write_geojson_from_esri_json(payload=payload, out_path=out_path)
                acquisition_method = "bbox_export_json_fallback"
                meta = json_meta
                warnings.extend(pagination_warnings)
                if any(
                    "geojson_query_failed" in str(w).lower() and "http error 400" in str(w).lower()
                    for w in warnings
                ):
                    warnings.append("geojson_unsupported_fallback_to_json_succeeded")
                break
            except Exception as exc:
                last_exc = exc
                warnings.append(f"{fmt}_query_failed={exc}")
                attempt_errors.append(f"{fmt}_query_failed={exc}")
                continue

        if meta is None or request_url is None:
            raise ArcGISFeatureQueryError(
                endpoint=self.endpoint,
                attempted_formats=format_order,
                attempted_urls=attempted_urls,
                errors=attempt_errors or ([str(last_exc)] if last_exc else []),
            ) from last_exc

        return AcquisitionResult(
            layer_key=layer_key,
            provider_type="arcgis_feature_service",
            acquisition_method=acquisition_method,
            source_endpoint=self.endpoint,
            source_url=request_url,
            local_path=str(out_path),
            bbox_used=bbox,
            output_resolution=target_resolution,
            cache_hit=False,
            warnings=warnings,
            http_status=meta.get("http_status"),
            response_content_type=meta.get("response_content_type"),
            bytes_downloaded=meta.get("bytes_downloaded"),
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

    source_endpoint = _sanitize_url(layer_config.get("source_endpoint") or layer_config.get("source_url") or "")
    full_download_url = _sanitize_url(layer_config.get("full_download_url") or layer_config.get("source_url") or "") or None
    local_path = str(layer_config.get("local_path") or "").strip() or None
    supports_bbox_export = _boolish(layer_config.get("supports_bbox_export"), provider_type.startswith("arcgis_"))
    supports_geojson_direct = _boolish(layer_config.get("supports_geojson_direct"), True)
    require_return_geometry = _boolish(layer_config.get("require_return_geometry"), True)
    preferred_response_format = str(layer_config.get("query_format") or "").strip().lower() or None

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
        provider = ArcGISFeatureServiceProvider(
            endpoint=source_endpoint,
            full_download_url=full_download_url,
            supports_geojson_direct=supports_geojson_direct,
            require_return_geometry=require_return_geometry,
            preferred_response_format=preferred_response_format,
        )
    elif provider_type in {"file_download", "vector_service", "overture_buildings"}:
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
