from __future__ import annotations

import json

from backend.naip_features import (
    NAIP_FEATURES_FILENAME,
    build_quantiles,
    percentile_from_quantiles,
    resolve_naip_feature_path,
    structure_feature_key,
)


def test_structure_feature_key_prefers_structure_id():
    key = structure_feature_key(structure_id="abc123", centroid_lat=46.1, centroid_lon=-120.2)
    assert key == "structure_id:abc123"


def test_structure_feature_key_centroid_fallback():
    key = structure_feature_key(structure_id=None, centroid_lat=46.12345678, centroid_lon=-120.98765432)
    assert key == "centroid:46.123457,-120.987654"


def test_quantile_lookup_interpolation():
    quantiles = build_quantiles([10, 20, 30, 40, 50], percentiles=[10, 50, 90])
    percentile = percentile_from_quantiles(35.0, quantiles)
    assert quantiles
    assert percentile is not None
    assert 50.0 <= percentile <= 90.0


def test_resolve_naip_feature_path_prefers_runtime_path(tmp_path):
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"region_id": "x"}), encoding="utf-8")

    resolved = resolve_naip_feature_path(
        region_manifest_path=str(manifest_path),
        runtime_path=str(runtime_path),
    )
    assert resolved == str(runtime_path)


def test_resolve_naip_feature_path_manifest_relative(tmp_path):
    region_dir = tmp_path / "region_a"
    region_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = region_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"region_id": "region_a"}), encoding="utf-8")
    artifact_path = region_dir / NAIP_FEATURES_FILENAME
    artifact_path.write_text(json.dumps({"feature_count": 1}), encoding="utf-8")

    resolved = resolve_naip_feature_path(
        region_manifest_path=str(manifest_path),
        runtime_path=None,
    )
    assert resolved == str(artifact_path)

