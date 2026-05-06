"""
Startup data-integrity checks for the wildfire risk scoring model.

Run once at application boot via run_data_integrity_checks(). Consolidates all
data-quality guards added across previous sessions:
  1. National MTBS GeoPackage feature count
  2. WHP raster availability matrix across prepared regions
  3. Per-region fire_perimeters.geojson empty-file guard
  4. Calibration artifact quality gate
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("wildfire_app.startup_checks")

_MTBS_PATH_DEFAULT = "data/national/mtbs_perimeters.gpkg"
_MTBS_CRITICAL_MIN = 100
_MTBS_WARNING_MIN = 10_000
_FIRE_PERIMETERS_MIN_BYTES = 500


def _check_mtbs(mtbs_path: str) -> tuple[int, int]:
    """
    Return (critical_count, warning_count) for the MTBS GeoPackage.
    Logs CRITICAL if feature count < 100, WARNING if < 10,000.
    """
    path = Path(mtbs_path)
    if not path.exists():
        LOGGER.critical(
            "startup_checks: MTBS GeoPackage not found at %s — "
            "historic_fire_index will be None for all assessments. "
            "Run scripts/download_national_mtbs.py to download.",
            mtbs_path,
        )
        return 1, 0

    try:
        import fiona  # type: ignore

        with fiona.open(mtbs_path, layer="fire_perimeters") as src:
            count = len(src)
    except ImportError:
        size_mb = path.stat().st_size / 1_048_576.0
        if size_mb < 10:
            LOGGER.warning(
                "startup_checks: fiona unavailable; MTBS file is %.1fMB — may be truncated (expected ~500MB).",
                size_mb,
            )
            return 0, 1
        LOGGER.warning(
            "startup_checks: fiona unavailable; cannot count MTBS features. File size %.1fMB looks plausible.",
            size_mb,
        )
        return 0, 0
    except Exception as exc:
        LOGGER.critical(
            "startup_checks: MTBS GeoPackage could not be opened at %s — error: %s",
            mtbs_path,
            exc,
        )
        return 1, 0

    if count < _MTBS_CRITICAL_MIN:
        LOGGER.critical(
            "startup_checks: MTBS GeoPackage has only %d features (critical minimum is %d) — "
            "dataset is effectively a stub. historic_fire_index will be None for all assessments. "
            "Run scripts/download_national_mtbs.py to download the full dataset.",
            count,
            _MTBS_CRITICAL_MIN,
        )
        return 1, 0
    if count < _MTBS_WARNING_MIN:
        LOGGER.warning(
            "startup_checks: MTBS GeoPackage has only %d features (expected >= %d) — "
            "dataset may be a partial download. Fire history scoring will be incomplete.",
            count,
            _MTBS_WARNING_MIN,
        )
        return 0, 1

    LOGGER.info("startup_checks: MTBS GeoPackage OK — %d features at %s.", count, mtbs_path)
    return 0, 0


def _check_whp_matrix(region_data_dir: str) -> tuple[int, int]:
    """
    Log INFO summary of WHP raster availability across all prepared regions.
    Returns (critical_count, warning_count) — WHP absence is a warning, not critical.
    """
    try:
        from backend.region_registry import list_prepared_regions, resolve_region_file  # noqa: PLC0415
    except ImportError:
        LOGGER.warning("startup_checks: could not import region_registry — WHP matrix skipped.")
        return 0, 1

    try:
        regions = list_prepared_regions(base_dir=region_data_dir)
    except Exception as exc:
        LOGGER.warning("startup_checks: list_prepared_regions failed — %s", exc)
        return 0, 1

    if not regions:
        LOGGER.info("startup_checks: no prepared regions found — WHP matrix skipped.")
        return 0, 0

    whp_keys = ("burn_prob", "whp", "wildfire_hazard_potential", "burn_probability")
    have_whp: list[str] = []
    missing_whp: list[str] = []

    for manifest in regions:
        region_id = str(manifest.get("region_id") or manifest.get("display_name") or "unknown")
        found = any(
            resolve_region_file(manifest, key, base_dir=region_data_dir)
            for key in whp_keys
        )
        (have_whp if found else missing_whp).append(region_id)

    total = len(regions)
    n_have = len(have_whp)
    if missing_whp:
        LOGGER.info(
            "startup_checks: WHP availability — %d of %d regions have burn_probability data (%s). "
            "ember_exposure_risk burn_probability sub-weight will be suppressed for: %s.",
            n_have,
            total,
            ", ".join(have_whp) if have_whp else "none",
            ", ".join(missing_whp),
        )
        return 0, len(missing_whp)
    LOGGER.info(
        "startup_checks: WHP availability — all %d regions have burn_probability data.", total
    )
    return 0, 0


def _check_fire_perimeters(region_data_dir: str) -> tuple[int, int]:
    """
    Check each prepared region's fire_perimeters.geojson for empty/stub condition.
    Returns (critical_count, warning_count).
    """
    try:
        from backend.region_registry import list_prepared_regions, resolve_region_file  # noqa: PLC0415
    except ImportError:
        LOGGER.warning("startup_checks: could not import region_registry — fire_perimeters check skipped.")
        return 0, 1

    try:
        regions = list_prepared_regions(base_dir=region_data_dir)
    except Exception as exc:
        LOGGER.warning("startup_checks: list_prepared_regions failed — %s", exc)
        return 0, 1

    critical = 0
    for manifest in regions:
        region_id = str(manifest.get("region_id") or manifest.get("display_name") or "unknown")
        perimeter_path = resolve_region_file(manifest, "fire_perimeters", base_dir=region_data_dir)
        if not perimeter_path:
            perimeter_path = resolve_region_file(manifest, "perimeters", base_dir=region_data_dir)
        if not perimeter_path:
            continue
        path_obj = Path(perimeter_path)
        if not path_obj.exists():
            continue
        try:
            size = path_obj.stat().st_size
        except OSError:
            continue
        if size < _FIRE_PERIMETERS_MIN_BYTES:
            LOGGER.critical(
                "startup_checks: region %r — fire_perimeters.geojson at %s is only %d bytes. "
                "Treating as empty/missing; historic fire scoring disabled for this region. "
                "Re-run data prep to regenerate.",
                region_id,
                perimeter_path,
                size,
            )
            critical += 1
            continue
        # Quick feature-count check without loading the full file.
        try:
            import json  # noqa: PLC0415
            with open(perimeter_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            features = payload.get("features", []) if isinstance(payload, dict) else []
            if not features:
                LOGGER.critical(
                    "startup_checks: region %r — fire_perimeters.geojson at %s has 0 features. "
                    "Historic fire scoring disabled for this region. Re-run data prep.",
                    region_id,
                    perimeter_path,
                )
                critical += 1
            else:
                LOGGER.info(
                    "startup_checks: region %r — fire_perimeters.geojson OK (%d features).",
                    region_id,
                    len(features),
                )
        except Exception as exc:
            LOGGER.warning(
                "startup_checks: region %r — could not parse fire_perimeters.geojson: %s",
                region_id,
                exc,
            )

    return critical, 0


def _check_calibration() -> tuple[int, int]:
    """
    Run calibration artifact quality gate. Returns (critical_count, warning_count).
    """
    try:
        from backend.calibration import (  # noqa: PLC0415
            CALIBRATION_ARTIFACT_SAFE,
            _DEFAULT_ARTIFACT_PATH,
            _check_quality_thresholds,
            _extract_quality_metrics,
            _load_json,
        )
    except ImportError as exc:
        LOGGER.warning("startup_checks: could not import calibration module — %s", exc)
        return 0, 1

    artifact = _load_json(_DEFAULT_ARTIFACT_PATH)
    if not artifact:
        LOGGER.info(
            "startup_checks: calibration artifact not present at %s — calibration disabled (OK for production).",
            _DEFAULT_ARTIFACT_PATH,
        )
        return 0, 0

    metrics = _extract_quality_metrics(artifact)
    failures = _check_quality_thresholds(metrics)
    env_path = os.getenv("WF_PUBLIC_CALIBRATION_ARTIFACT", "").strip()

    if failures:
        LOGGER.critical(
            "startup_checks: calibration artifact fails quality thresholds — CALIBRATION_ARTIFACT_SAFE=False. "
            "Failures: %s",
            "; ".join(failures),
        )
        if env_path:
            LOGGER.critical(
                "startup_checks: WF_PUBLIC_CALIBRATION_ARTIFACT is set but artifact fails quality gate — "
                "calibration will not be applied. Unset the env var to suppress this message.",
            )
        return 1, 0

    LOGGER.info(
        "startup_checks: calibration artifact passed all quality thresholds — CALIBRATION_ARTIFACT_SAFE=True."
    )
    return 0, 0


def run_data_integrity_checks(
    mtbs_path: str | None = None,
    region_data_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run all data-integrity checks and log a final summary.

    Parameters
    ----------
    mtbs_path : path to the national MTBS GeoPackage (defaults to env/default location)
    region_data_dir : base directory for prepared regions (defaults to env/default)

    Returns
    -------
    dict with keys: critical_count, warning_count, operational_status
    """
    _mtbs_path = mtbs_path or os.getenv("WF_MTBS_GPKG_PATH", _MTBS_PATH_DEFAULT)
    _region_dir = region_data_dir or os.getenv("WF_REGION_DATA_DIR", str(Path("data") / "regions"))

    LOGGER.info("startup_checks: running data integrity checks...")

    total_critical = 0
    total_warnings = 0

    c, w = _check_mtbs(_mtbs_path)
    total_critical += c
    total_warnings += w

    c, w = _check_whp_matrix(_region_dir)
    total_critical += c
    total_warnings += w

    c, w = _check_fire_perimeters(_region_dir)
    total_critical += c
    total_warnings += w

    c, w = _check_calibration()
    total_critical += c
    total_warnings += w

    if total_critical > 0:
        operational_status = "NO"
    elif total_warnings > 0:
        operational_status = "DEGRADED"
    else:
        operational_status = "YES"

    LOGGER.info(
        "startup_checks: Data integrity check complete. "
        "Critical issues: %d, Warnings: %d. Model operational: %s",
        total_critical,
        total_warnings,
        operational_status,
    )

    return {
        "critical_count": total_critical,
        "warning_count": total_warnings,
        "operational_status": operational_status,
    }
