from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


class FeatureBundleCache:
    """Lightweight file-backed cache for property feature bundles."""

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        enabled: bool | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
            or os.getenv("WF_FEATURE_BUNDLE_CACHE_DIR")
            or (Path("data") / "cache" / "feature_bundles")
        )
        self.enabled = _env_bool("WF_FEATURE_BUNDLE_CACHE_ENABLED", True) if enabled is None else bool(enabled)
        self.read_enabled = _env_bool("WF_FEATURE_BUNDLE_CACHE_READ", True)
        self.write_enabled = _env_bool("WF_FEATURE_BUNDLE_CACHE_WRITE", True)
        ttl_raw = os.getenv("WF_FEATURE_BUNDLE_CACHE_TTL_SEC", str(6 * 3600))
        if ttl_seconds is None:
            try:
                ttl_seconds = int(str(ttl_raw).strip())
            except ValueError:
                ttl_seconds = 6 * 3600
        self.ttl_seconds = max(0, int(ttl_seconds))
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_runtime_paths(runtime_paths: dict[str, str]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for key, value in sorted(runtime_paths.items()):
            path = str(value or "").strip()
            if not path:
                normalized[key] = {"path": None, "exists": False, "mtime_ns": None}
                continue
            p = Path(path)
            exists = p.exists()
            mtime_ns: int | None = None
            if exists:
                try:
                    mtime_ns = p.stat().st_mtime_ns
                except OSError:
                    mtime_ns = None
            normalized[key] = {"path": path, "exists": exists, "mtime_ns": mtime_ns}
        return normalized

    @classmethod
    def build_key(
        cls,
        *,
        lat: float,
        lon: float,
        runtime_paths: dict[str, str],
        region_context: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "runtime_paths": cls._normalize_runtime_paths(runtime_paths),
            "region_context": {
                "region_status": (region_context or {}).get("region_status"),
                "region_id": (region_context or {}).get("region_id"),
                "manifest_path": (region_context or {}).get("manifest_path"),
            },
            "extras": extras or {},
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return digest

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        if not (self.enabled and self.read_enabled):
            return None
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        cached_at = payload.get("cached_at_epoch_s")
        if self.ttl_seconds > 0:
            try:
                age = time.time() - float(cached_at)
            except (TypeError, ValueError):
                age = float(self.ttl_seconds + 1)
            if age > float(self.ttl_seconds):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                return None
        data = payload.get("payload")
        return data if isinstance(data, dict) else None

    def save(self, key: str, payload: dict[str, Any]) -> None:
        if not (self.enabled and self.write_enabled):
            return
        path = self._path_for_key(key)
        tmp_path = path.with_suffix(".tmp")
        envelope = {
            "cached_at_epoch_s": time.time(),
            "payload": payload,
        }
        try:
            tmp_path.write_text(
                json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
