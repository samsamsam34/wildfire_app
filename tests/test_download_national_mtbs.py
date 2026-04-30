"""
Tests for scripts/download_national_mtbs.py validation and idempotency logic.

No real network access or file downloads are performed — all GeoPackage reads
are mocked.  Tests skip if geopandas is unavailable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _gpd():
    try:
        import geopandas as gpd
        return gpd
    except ImportError:
        pytest.skip("geopandas not available")


# Import the helpers under test — the script is importable as a module.
def _import_script():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "download_national_mtbs",
        Path("scripts/download_national_mtbs.py"),
    )
    mod = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
    spec.loader.exec_module(mod)
    return mod


def _load_script():
    """Import the download script module, skipping if import fails."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_national_mtbs",
            Path("scripts/download_national_mtbs.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception as e:
        pytest.skip(f"Could not import download script: {e}")


# ---------------------------------------------------------------------------
# Test 1: _validate_gpkg passes for 80,000 features
# ---------------------------------------------------------------------------

def test_validate_gpkg_passes_for_full_dataset(tmp_path: Path) -> None:
    """_validate_gpkg raises no exception when the GeoPackage has 80,000 features."""
    _gpd()
    script = _load_script()

    mock_gdf = MagicMock()
    mock_gdf.__len__ = MagicMock(return_value=80_000)

    with patch("geopandas.read_file", return_value=mock_gdf):
        # Should not raise
        script._validate_gpkg(tmp_path / "mock.gpkg")


# ---------------------------------------------------------------------------
# Test 2: _validate_gpkg fails for 3 features (original truncated file)
# ---------------------------------------------------------------------------

def test_validate_gpkg_raises_for_truncated_file(tmp_path: Path) -> None:
    """_validate_gpkg raises ValueError when only 3 features are present."""
    _gpd()
    script = _load_script()

    mock_gdf = MagicMock()
    mock_gdf.__len__ = MagicMock(return_value=3)

    with patch("geopandas.read_file", return_value=mock_gdf):
        with pytest.raises(ValueError) as exc_info:
            script._validate_gpkg(tmp_path / "mock.gpkg")

    msg = str(exc_info.value)
    assert "3" in msg, f"Error message should include the feature count: {msg}"
    # Message should reference the minimum threshold
    assert "25,000" in msg or "25000" in msg, (
        f"Error message should reference the minimum expected count: {msg}"
    )


# ---------------------------------------------------------------------------
# Test 3: _should_skip_download returns False for truncated existing file
# ---------------------------------------------------------------------------

def test_idempotency_rejects_truncated_existing_file(tmp_path: Path) -> None:
    """A recent file with only 3 features must NOT be skipped — re-download required."""
    _gpd()
    script = _load_script()

    # Create a real but tiny file so os.path.exists() and os.path.getmtime() work.
    stub = tmp_path / "mtbs_perimeters.gpkg"
    stub.write_bytes(b"x" * 1000)  # 1 KB stub

    mock_gdf = MagicMock()
    mock_gdf.__len__ = MagicMock(return_value=3)

    original_output = script.OUTPUT_GPKG
    try:
        script.OUTPUT_GPKG = stub
        with patch("geopandas.read_file", return_value=mock_gdf):
            result = script._should_skip_download(force=False)
    finally:
        script.OUTPUT_GPKG = original_output

    assert result is False, (
        "_should_skip_download must return False for a recent file with only 3 features"
    )


# ---------------------------------------------------------------------------
# Test 4: _should_skip_download returns True for valid existing file
# ---------------------------------------------------------------------------

def test_idempotency_accepts_valid_existing_file(tmp_path: Path) -> None:
    """A recent file with 80,000 features should be skipped (no re-download needed)."""
    _gpd()
    script = _load_script()

    stub = tmp_path / "mtbs_perimeters.gpkg"
    stub.write_bytes(b"x" * (150 * 1024 * 1024))  # 150 MB stub (realistic size)

    mock_gdf = MagicMock()
    mock_gdf.__len__ = MagicMock(return_value=80_000)

    original_output = script.OUTPUT_GPKG
    try:
        script.OUTPUT_GPKG = stub
        with patch("geopandas.read_file", return_value=mock_gdf):
            result = script._should_skip_download(force=False)
    finally:
        script.OUTPUT_GPKG = original_output

    assert result is True, (
        "_should_skip_download must return True for a recent, valid file"
    )


# ---------------------------------------------------------------------------
# Test 5: _should_skip_download returns False when force=True
# ---------------------------------------------------------------------------

def test_idempotency_force_flag_always_redownloads(tmp_path: Path) -> None:
    """--force must bypass the skip check regardless of file state."""
    _gpd()
    script = _load_script()

    stub = tmp_path / "mtbs_perimeters.gpkg"
    stub.write_bytes(b"x" * (150 * 1024 * 1024))

    mock_gdf = MagicMock()
    mock_gdf.__len__ = MagicMock(return_value=80_000)

    original_output = script.OUTPUT_GPKG
    try:
        script.OUTPUT_GPKG = stub
        with patch("geopandas.read_file", return_value=mock_gdf):
            result = script._should_skip_download(force=True)
    finally:
        script.OUTPUT_GPKG = original_output

    assert result is False, (
        "_should_skip_download must return False when force=True"
    )


# ---------------------------------------------------------------------------
# Test 6: _should_skip_download returns False for stale file (> 365 days)
# ---------------------------------------------------------------------------

def test_idempotency_rejects_stale_file(tmp_path: Path) -> None:
    """A file older than 365 days must be re-downloaded regardless of feature count."""
    _gpd()
    script = _load_script()

    stub = tmp_path / "mtbs_perimeters.gpkg"
    stub.write_bytes(b"x" * (150 * 1024 * 1024))
    # Set mtime to 400 days ago
    stale_mtime = time.time() - (400 * 86400)
    os.utime(str(stub), (stale_mtime, stale_mtime))

    mock_gdf = MagicMock()
    mock_gdf.__len__ = MagicMock(return_value=80_000)

    original_output = script.OUTPUT_GPKG
    try:
        script.OUTPUT_GPKG = stub
        with patch("geopandas.read_file", return_value=mock_gdf):
            result = script._should_skip_download(force=False)
    finally:
        script.OUTPUT_GPKG = original_output

    assert result is False, (
        "_should_skip_download must return False for a file older than 365 days"
    )


# ---------------------------------------------------------------------------
# Test 7: _should_skip_download returns False when file does not exist
# ---------------------------------------------------------------------------

def test_idempotency_returns_false_when_file_missing(tmp_path: Path) -> None:
    """No existing file → must not skip download."""
    _gpd()
    script = _load_script()

    original_output = script.OUTPUT_GPKG
    try:
        script.OUTPUT_GPKG = tmp_path / "nonexistent.gpkg"
        result = script._should_skip_download(force=False)
    finally:
        script.OUTPUT_GPKG = original_output

    assert result is False
