from __future__ import annotations

from typing import Any, Mapping

# Product/runtime semantic versions.
PRODUCT_VERSION = "0.11.0"
API_VERSION = "1.0.0"

# Scoring and governance dimensions.
SCORING_MODEL_VERSION = "1.6.0"
DEFAULT_RULESET_VERSION = "1.0.0"
RULESET_LOGIC_VERSION = "1.1.0"
FACTOR_SCHEMA_VERSION = "1.1.0"
BENCHMARK_PACK_VERSION = "1.0.0"
CALIBRATION_VERSION = "0.1.0"

# Default dataset bundle marker when no explicit region data version is available.
DATA_BUNDLE_VERSION = "unversioned"

# Backward compatibility aliases.
MODEL_VERSION = SCORING_MODEL_VERSION
LEGACY_MODEL_VERSION = "1.0.0"

GOVERNANCE_KEYS = (
    "product_version",
    "api_version",
    "scoring_model_version",
    "ruleset_version",
    "rules_logic_version",
    "factor_schema_version",
    "benchmark_pack_version",
    "calibration_version",
    "region_data_version",
    "data_bundle_version",
)

CRITICAL_COMPARABILITY_KEYS = (
    "scoring_model_version",
    "ruleset_version",
    "rules_logic_version",
    "factor_schema_version",
)

REVIEW_COMPARABILITY_KEYS = (
    "calibration_version",
    "region_data_version",
    "data_bundle_version",
)

RELEASE_NOTE_REQUIRED_SECTIONS = (
    "Version changes",
    "Reason",
    "Expected effect on outputs",
    "Migration/interpretation notes",
    "Historical comparison validity",
)


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_model_governance(
    *,
    ruleset_version: str = DEFAULT_RULESET_VERSION,
    region_data_version: str | None = None,
    benchmark_pack_version: str | None = None,
    data_bundle_version: str | None = None,
    product_version: str | None = None,
    api_version: str | None = None,
    scoring_model_version: str | None = None,
    rules_logic_version: str | None = None,
    factor_schema_version: str | None = None,
    calibration_version: str | None = None,
) -> dict[str, str | None]:
    return {
        "product_version": _clean_optional(product_version) or PRODUCT_VERSION,
        "api_version": _clean_optional(api_version) or API_VERSION,
        "scoring_model_version": _clean_optional(scoring_model_version) or SCORING_MODEL_VERSION,
        "ruleset_version": _clean_optional(ruleset_version) or DEFAULT_RULESET_VERSION,
        "rules_logic_version": _clean_optional(rules_logic_version) or RULESET_LOGIC_VERSION,
        "factor_schema_version": _clean_optional(factor_schema_version) or FACTOR_SCHEMA_VERSION,
        "benchmark_pack_version": _clean_optional(benchmark_pack_version) or BENCHMARK_PACK_VERSION,
        "calibration_version": _clean_optional(calibration_version) or CALIBRATION_VERSION,
        "region_data_version": _clean_optional(region_data_version),
        "data_bundle_version": _clean_optional(data_bundle_version) or DATA_BUNDLE_VERSION,
    }


def normalize_model_governance(payload: Mapping[str, Any] | None) -> dict[str, str | None]:
    raw = dict(payload or {})
    return build_model_governance(
        ruleset_version=str(raw.get("ruleset_version") or DEFAULT_RULESET_VERSION),
        region_data_version=_clean_optional(raw.get("region_data_version")),
        benchmark_pack_version=_clean_optional(raw.get("benchmark_pack_version")),
        data_bundle_version=_clean_optional(raw.get("data_bundle_version")),
        product_version=_clean_optional(raw.get("product_version")),
        api_version=_clean_optional(raw.get("api_version")),
        scoring_model_version=_clean_optional(raw.get("scoring_model_version")),
        rules_logic_version=_clean_optional(raw.get("rules_logic_version")),
        factor_schema_version=_clean_optional(raw.get("factor_schema_version")),
        calibration_version=_clean_optional(raw.get("calibration_version")),
    )


def compare_model_governance(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left_g = normalize_model_governance(left)
    right_g = normalize_model_governance(right)
    differences: dict[str, dict[str, str | None]] = {}
    for key in GOVERNANCE_KEYS:
        if left_g.get(key) != right_g.get(key):
            differences[key] = {"left": left_g.get(key), "right": right_g.get(key)}

    critical_differences = [k for k in CRITICAL_COMPARABILITY_KEYS if k in differences]
    review_differences = [k for k in REVIEW_COMPARABILITY_KEYS if k in differences]

    warnings: list[str] = []
    if "benchmark_pack_version" in differences:
        warnings.append("Benchmark expectations changed between these runs.")
    if "api_version" in differences:
        warnings.append("API contract version differs; schema comparison may require care.")
    if review_differences:
        warnings.append(
            "Data/calibration dimensions differ; compare directionally and review regional/evidence context."
        )
    if critical_differences:
        warnings.append(
            "Scoring/rules schema dimensions differ; direct score comparison is not recommended."
        )

    if critical_differences:
        label = "not_directly_comparable"
    elif review_differences:
        label = "comparable_with_review"
    else:
        label = "directly_comparable"

    return {
        "left": left_g,
        "right": right_g,
        "differences": differences,
        "critical_differences": critical_differences,
        "review_differences": review_differences,
        "directly_comparable": label == "directly_comparable",
        "comparison_label": label,
        "warnings": warnings,
    }


def release_note_template(version: str, release_date: str) -> str:
    return (
        f"## [{version}] - {release_date}\n"
        "### Version changes\n"
        "- `product_version`: `<x.y.z>` (`major|minor|patch`)\n"
        "- `api_version`: `<x.y.z>`\n"
        "- `scoring_model_version`: `<x.y.z>`\n"
        "- `ruleset_version`: `<x.y.z>` (or per-ruleset note)\n"
        "- `rules_logic_version`: `<x.y.z>`\n"
        "- `factor_schema_version`: `<x.y.z>`\n"
        "- `benchmark_pack_version`: `<x.y.z>`\n"
        "- `calibration_version`: `<x.y.z>`\n"
        "- `region_data_version`: `<id|snapshot>`\n"
        "- `data_bundle_version`: `<id|snapshot>`\n\n"
        "### Reason\n"
        "- Why this release exists and what changed.\n\n"
        "### Expected effect on outputs\n"
        "- What users should expect to change (or not change) in API/scoring outputs.\n\n"
        "### Migration/interpretation notes\n"
        "- Any client migration notes or interpretation guidance.\n\n"
        "### Historical comparison validity\n"
        "- `directly_comparable` | `comparable_with_review` | `not_directly_comparable`\n"
    )
