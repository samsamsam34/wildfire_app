from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from shutil import copyfileobj
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LANDFIRE_HANDLER_VERSION = "1.0"
LANDFIRE_RASTER_SUFFIXES = (".tif", ".tiff", ".img")


@dataclass
class LandfireArchiveResolution:
    is_landfire_archive: bool
    raster_path: Path
    layer_key: str
    source_fingerprint: str
    archive_path: Path | None = None
    archive_size_bytes: int | None = None
    archive_cache_path: str | None = None
    index_path: str | None = None
    extracted_raster_path: str | None = None
    extraction_performed: bool = False
    subset_cache_path: str | None = None
    subset_reused: bool = False
    processing_notes: list[str] | None = None


def default_cache_root() -> Path:
    env = os.getenv("WILDFIRE_APP_CACHE_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("data") / "cache"


def default_data_root() -> Path:
    env = os.getenv("WILDFIRE_APP_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("data")


def default_tmp_root() -> Path:
    env = os.getenv("WILDFIRE_APP_TMP_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("tmp")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _stable_bounds_key(bounds: dict[str, float]) -> str:
    return (
        f"{bounds['min_lon']:.6f},{bounds['min_lat']:.6f},"
        f"{bounds['max_lon']:.6f},{bounds['max_lat']:.6f}"
    )


def _fingerprint_for_path(path: Path) -> str:
    stat = path.stat()
    token = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cache_root(cache_dir: Path) -> Path:
    return cache_dir / "landfire"


def _archives_root(cache_dir: Path) -> Path:
    return _cache_root(cache_dir) / "archives"


def _staging_root(cache_dir: Path) -> Path:
    return _cache_root(cache_dir) / "staging"


def _index_path(cache_dir: Path, fingerprint: str) -> Path:
    return _cache_root(cache_dir) / "index" / f"{fingerprint}.json"


def _extracted_root(cache_dir: Path, fingerprint: str) -> Path:
    return _cache_root(cache_dir) / "extracted" / fingerprint


def subset_cache_path(
    *,
    cache_dir: Path,
    source_fingerprint: str,
    layer_key: str,
    bounds: dict[str, float],
) -> Path:
    token = f"{source_fingerprint}|{layer_key}|{_stable_bounds_key(bounds)}|{LANDFIRE_HANDLER_VERSION}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return _cache_root(cache_dir) / "subsets" / f"{digest}_{layer_key}.tif"


def cache_path_for_url(url: str, cache_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower() or ".bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return _archives_root(cache_dir) / f"{digest}{suffix}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, checksum: str | None) -> str:
    if not checksum:
        return "not_provided"
    expected = checksum.strip().lower()
    if expected.startswith("sha256:"):
        expected = expected.split(":", 1)[1]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}: expected {expected}, got {actual}")
    return "verified"


def download_with_resume(
    *,
    url: str,
    cache_dir: Path,
    timeout_seconds: float = 45.0,
    retries: int = 2,
    retry_backoff_seconds: float = 1.5,
    force_redownload: bool = False,
    progress_log: list[str] | None = None,
) -> dict[str, Any]:
    archives_dir = _archives_root(cache_dir)
    archives_dir.mkdir(parents=True, exist_ok=True)
    target = cache_path_for_url(url, cache_dir=cache_dir)
    part_path = target.with_suffix(target.suffix + ".part")

    if force_redownload:
        target.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    if target.exists() and target.stat().st_size > 0:
        if progress_log is not None:
            progress_log.append(f"LANDFIRE: using cached archive {target}")
        return {
            "archive_path": str(target),
            "cache_hit_download": True,
            "bytes_downloaded": 0,
            "retry_count_used": 0,
            "download_status": "cache_hit",
            "resumed": False,
            "downloaded_at": _now(),
        }

    bytes_downloaded = 0
    resumed = False
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        req = urllib.request.Request(url)
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")
            if progress_log is not None:
                progress_log.append(f"LANDFIRE: resuming download at byte {existing_size}")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", 200) or 200)
                append_mode = existing_size > 0 and status == 206
                if existing_size > 0 and not append_mode:
                    part_path.unlink(missing_ok=True)
                    existing_size = 0
                mode = "ab" if append_mode else "wb"
                resumed = append_mode
                with open(part_path, mode) as out:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        bytes_downloaded += len(chunk)
            part_path.replace(target)
            if progress_log is not None:
                progress_log.append(f"LANDFIRE: download complete ({bytes_downloaded} bytes transferred)")
            return {
                "archive_path": str(target),
                "cache_hit_download": False,
                "bytes_downloaded": int(bytes_downloaded),
                "retry_count_used": attempt,
                "download_status": "ok",
                "resumed": resumed,
                "downloaded_at": _now(),
            }
        except Exception as exc:  # pragma: no cover - runtime network behavior
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = retry_backoff_seconds * (2**attempt)
            if progress_log is not None:
                progress_log.append(f"LANDFIRE: transient download failure, retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise ValueError(f"LANDFIRE download failed for {url} after {retries + 1} attempts: {last_exc}") from last_exc


def _candidate_score(layer_key: str, filename: str) -> int:
    name = filename.lower()
    score = 0
    if layer_key == "fuel":
        for token, weight in (
            ("fuel", 200),
            ("fbfm", 180),
            ("fbfm", 180),
            ("f40", 140),
            ("surface", 80),
        ):
            if token in name:
                score += weight
    elif layer_key == "canopy":
        for token, weight in (
            ("canopy", 220),
            ("cover", 150),
            ("crown", 120),
            ("cc", 60),
        ):
            if token in name:
                score += weight
    ext = Path(name).suffix.lower()
    if ext in {".tif", ".tiff"}:
        score += 30
    elif ext == ".img":
        score += 10
    return score


def _scan_archive_index(archive_path: Path, cache_dir: Path, progress_log: list[str] | None = None) -> dict[str, Any]:
    fingerprint = _fingerprint_for_path(archive_path)
    idx_path = _index_path(cache_dir, fingerprint)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    if idx_path.exists():
        if progress_log is not None:
            progress_log.append(f"LANDFIRE: reusing cached archive index {idx_path}")
        with open(idx_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload

    if progress_log is not None:
        progress_log.append(f"LANDFIRE: scanning archive contents {archive_path.name}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

    rasters = [
        {
            "name": info.filename,
            "size": int(info.file_size),
            "compressed_size": int(info.compress_size),
        }
        for info in infos
        if Path(info.filename).suffix.lower() in LANDFIRE_RASTER_SUFFIXES
    ]
    if not rasters:
        raise ValueError(
            f"LANDFIRE archive has no raster candidates: {archive_path}. "
            "Provide a smaller regional/local raster source."
        )

    selected: dict[str, dict[str, Any] | None] = {}
    for layer_key in ("fuel", "canopy"):
        scored: list[tuple[int, str, int]] = []
        for entry in rasters:
            score = _candidate_score(layer_key, entry["name"])
            if score > 0:
                scored.append((score, entry["name"], int(entry["size"])))
        if not scored:
            selected[layer_key] = None
            continue
        scored.sort(key=lambda x: (-x[0], x[1]))
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise ValueError(
                f"LANDFIRE {layer_key} raster selection is ambiguous in archive {archive_path.name}. "
                "Provide a pre-clipped local source for deterministic processing."
            )
        selected[layer_key] = {"name": scored[0][1], "score": scored[0][0], "size": scored[0][2]}

    payload = {
        "handler_version": LANDFIRE_HANDLER_VERSION,
        "archive_path": str(archive_path),
        "archive_size_bytes": int(archive_path.stat().st_size),
        "archive_fingerprint": fingerprint,
        "scanned_at": _now(),
        "rasters": rasters,
        "selected": selected,
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def _warn_large_archive(archive_path: Path, warnings: list[str] | None) -> None:
    threshold = int(os.getenv("WF_LANDFIRE_LARGE_ARCHIVE_WARN_BYTES", str(2 * 1024 * 1024 * 1024)))
    size_bytes = int(archive_path.stat().st_size)
    if size_bytes > threshold and warnings is not None:
        warnings.append(
            f"LANDFIRE archive is large ({size_bytes} bytes). Consider a smaller regional/local source for faster prep."
        )


def _check_disk_space(target_dir: Path, required_bytes: int) -> None:
    usage = shutil.disk_usage(target_dir)
    if usage.free < required_bytes:
        raise ValueError(
            "Insufficient disk space for LANDFIRE extraction. "
            f"Required ~{required_bytes} bytes, available {usage.free} bytes."
        )


def resolve_landfire_raster(
    *,
    layer_key: str,
    source_path: Path,
    cache_dir: Path,
    bounds: dict[str, float],
    progress_log: list[str] | None = None,
    warnings: list[str] | None = None,
) -> LandfireArchiveResolution:
    suffix = source_path.suffix.lower()
    fingerprint = _fingerprint_for_path(source_path)
    subset_path = subset_cache_path(
        cache_dir=cache_dir,
        source_fingerprint=fingerprint,
        layer_key=layer_key,
        bounds=bounds,
    )

    if suffix != ".zip":
        return LandfireArchiveResolution(
            is_landfire_archive=False,
            raster_path=source_path,
            layer_key=layer_key,
            source_fingerprint=fingerprint,
            archive_cache_path=str(source_path),
            extracted_raster_path=str(source_path),
            subset_cache_path=str(subset_path),
            subset_reused=subset_path.exists(),
            processing_notes=["LANDFIRE handler bypassed archive processing for non-zip source."],
        )

    _warn_large_archive(source_path, warnings)
    index_payload = _scan_archive_index(source_path, cache_dir=cache_dir, progress_log=progress_log)
    selected = (index_payload.get("selected") or {}).get(layer_key)
    if not selected:
        raise ValueError(
            f"LANDFIRE archive does not contain a deterministic {layer_key} raster candidate. "
            "Provide a smaller local/regional raster source."
        )

    selected_name = str(selected["name"])
    extracted_root = _extracted_root(cache_dir, index_payload["archive_fingerprint"])
    extracted_path = extracted_root / Path(selected_name).name

    if extracted_path.exists() and extracted_path.stat().st_size > 0:
        extraction_performed = False
        if progress_log is not None:
            progress_log.append(f"LANDFIRE: reusing extracted {layer_key} raster {extracted_path.name}")
    else:
        if progress_log is not None:
            progress_log.append(f"LANDFIRE: extracting {layer_key} raster {Path(selected_name).name}")
        extracted_root.mkdir(parents=True, exist_ok=True)
        estimate = int(selected.get("size") or 0)
        if estimate > 0:
            _check_disk_space(extracted_root, int(estimate * 2.0))
        with zipfile.ZipFile(source_path, "r") as zf:
            with zf.open(selected_name, "r") as src, open(extracted_path, "wb") as dst:
                copyfileobj(src, dst)
        extraction_performed = True

    if progress_log is not None and subset_path.exists():
        progress_log.append(f"LANDFIRE: reusing cached clipped subset for {layer_key}")

    return LandfireArchiveResolution(
        is_landfire_archive=True,
        raster_path=extracted_path,
        layer_key=layer_key,
        source_fingerprint=index_payload["archive_fingerprint"],
        archive_path=source_path,
        archive_size_bytes=int(index_payload.get("archive_size_bytes") or source_path.stat().st_size),
        archive_cache_path=str(source_path),
        index_path=str(_index_path(cache_dir, index_payload["archive_fingerprint"])),
        extracted_raster_path=str(extracted_path),
        extraction_performed=extraction_performed,
        subset_cache_path=str(subset_path),
        subset_reused=subset_path.exists(),
        processing_notes=["LANDFIRE archive handler used selective extraction and subset caching."],
    )


def latest_staged_metadata_path(cache_dir: Path) -> Path:
    return _staging_root(cache_dir) / "latest.json"


def load_latest_staged_assets(cache_root: Path) -> dict[str, Any] | None:
    path = latest_staged_metadata_path(cache_root)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    return payload


def stage_landfire_assets(
    *,
    fuel_url: str,
    canopy_url: str | None = None,
    cache_root: Path | None = None,
    timeout_seconds: float = 45.0,
    retries: int = 2,
    retry_backoff_seconds: float = 1.5,
    force_redownload: bool = False,
    force_reextract: bool = False,
    checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    cache_dir = (cache_root or default_cache_root()).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = _staging_root(cache_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    progress: list[str] = []
    warnings: list[str] = []
    layer_urls = {"fuel": fuel_url, "canopy": canopy_url}
    layer_meta: dict[str, Any] = {}
    bounds_stub = {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0}

    for layer_key, url in layer_urls.items():
        if not url:
            continue
        progress.append(f"LANDFIRE: processing {layer_key} source")
        dl = download_with_resume(
            url=url,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            force_redownload=force_redownload,
            progress_log=progress,
        )
        archive_path = Path(dl["archive_path"])
        checksum_status = verify_checksum(archive_path, (checksums or {}).get(layer_key))

        if force_reextract and archive_path.suffix.lower() == ".zip":
            fingerprint = _fingerprint_for_path(archive_path)
            extracted_root = _extracted_root(cache_dir, fingerprint)
            shutil.rmtree(extracted_root, ignore_errors=True)

        resolution = resolve_landfire_raster(
            layer_key=layer_key,
            source_path=archive_path,
            cache_dir=cache_dir,
            bounds=bounds_stub,
            progress_log=progress,
            warnings=warnings,
        )
        layer_meta[layer_key] = {
            "source_url": url,
            "source_archive_path": str(archive_path),
            "extracted_path": resolution.extracted_raster_path,
            "cache_hit_download": bool(dl["cache_hit_download"]),
            "cache_hit_extract": not bool(resolution.extraction_performed),
            "bytes_downloaded": int(dl["bytes_downloaded"]),
            "retry_count_used": int(dl["retry_count_used"]),
            "checksum_status": checksum_status,
            "archive_index_path": resolution.index_path,
            "archive_size_bytes": resolution.archive_size_bytes,
            "handler_version": LANDFIRE_HANDLER_VERSION,
            "updated_at": _now(),
        }

    result = {
        "status": "ok",
        "cache_root": str(cache_dir),
        "layers": layer_meta,
        "warnings": sorted(set(warnings)),
        "progress_log": progress,
        "created_at": _now(),
        "updated_at": _now(),
    }
    latest_path = latest_staged_metadata_path(cache_dir)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    result["metadata_path"] = str(latest_path)
    return result
