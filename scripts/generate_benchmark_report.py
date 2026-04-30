#!/usr/bin/env python3
"""Generate a brief benchmark report with PNG figures from benchmark outputs.

Reads:
- reports/benchmark_results.csv
- reports/benchmark_diagnostics.md

Writes:
- reports/risk_vs_confidence.png
- reports/risk_by_scenario.png
- reports/confidence_by_scenario.png
- reports/optional_input_effects.png (if paired data exists)
- reports/benchmark_summary.md
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_CSV = REPORTS_DIR / "benchmark_results.csv"
DIAGNOSTICS_MD = REPORTS_DIR / "benchmark_diagnostics.md"

RISK_VS_CONFIDENCE_PNG = REPORTS_DIR / "risk_vs_confidence.png"
RISK_BY_SCENARIO_PNG = REPORTS_DIR / "risk_by_scenario.png"
CONF_BY_SCENARIO_PNG = REPORTS_DIR / "confidence_by_scenario.png"
OPTIONAL_EFFECTS_PNG = REPORTS_DIR / "optional_input_effects.png"
SUMMARY_MD = REPORTS_DIR / "benchmark_summary.md"

WIDTH = 1200
HEIGHT = 760
MARGIN_LEFT = 90
MARGIN_RIGHT = 40
MARGIN_TOP = 70
MARGIN_BOTTOM = 90
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM


def _as_float(value: Any) -> float | None:
    try:
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_flags(value: str) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(";") if part.strip()}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "id": str(row.get("id") or "").strip(),
                    "address": str(row.get("address") or "").strip(),
                    "scenario_group": str(row.get("scenario_group") or "").strip() or "unknown",
                    "risk_score": _as_float(row.get("risk_score")),
                    "confidence_score": _as_float(row.get("confidence_score")),
                    "flags": _parse_flags(str(row.get("missing_data_flags") or "")),
                }
            )
    return rows


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _new_canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN_LEFT, 22), title, fill="black", font=_font(24))
    draw.rectangle(
        [MARGIN_LEFT, MARGIN_TOP, WIDTH - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM],
        outline=(180, 180, 180),
        width=1,
    )
    return img, draw


def _map_plot_point(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[int, int]:
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    nx = (x - x_min) / (x_max - x_min)
    ny = (y - y_min) / (y_max - y_min)
    px = int(MARGIN_LEFT + nx * PLOT_W)
    py = int(HEIGHT - MARGIN_BOTTOM - ny * PLOT_H)
    return px, py


def _draw_grid_and_axes(
    draw: ImageDraw.ImageDraw,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    tick_font = _font(13)
    label_font = _font(16)

    for i in range(6):
        fx = i / 5.0
        x = int(MARGIN_LEFT + fx * PLOT_W)
        y0 = MARGIN_TOP
        y1 = HEIGHT - MARGIN_BOTTOM
        draw.line([(x, y0), (x, y1)], fill=(238, 238, 238), width=1)
        xv = x_min + fx * (x_max - x_min)
        draw.text((x - 14, y1 + 8), f"{xv:.0f}", fill=(80, 80, 80), font=tick_font)

    for i in range(6):
        fy = i / 5.0
        y = int(HEIGHT - MARGIN_BOTTOM - fy * PLOT_H)
        draw.line([(MARGIN_LEFT, y), (WIDTH - MARGIN_RIGHT, y)], fill=(238, 238, 238), width=1)
        yv = y_min + fy * (y_max - y_min)
        draw.text((MARGIN_LEFT - 44, y - 8), f"{yv:.0f}", fill=(80, 80, 80), font=tick_font)

    draw.text((MARGIN_LEFT + PLOT_W // 2 - 85, HEIGHT - MARGIN_BOTTOM + 42), x_label, fill="black", font=label_font)
    draw.text((18, MARGIN_TOP - 2), y_label, fill="black", font=label_font)


def _scenario_palette(groups: list[str]) -> dict[str, tuple[int, int, int]]:
    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    uniq = sorted(set(groups))
    return {name: colors[i % len(colors)] for i, name in enumerate(uniq)}


def _draw_legend(draw: ImageDraw.ImageDraw, palette: dict[str, tuple[int, int, int]]) -> None:
    x = WIDTH - MARGIN_RIGHT - 260
    y = MARGIN_TOP + 12
    box_h = 25 * len(palette) + 60
    draw.rectangle([x - 10, y - 10, x + 250, y + box_h], outline=(210, 210, 210), fill=(255, 255, 255))
    draw.text((x, y), "Scenario", fill="black", font=_font(14))
    yy = y + 22
    for key in sorted(palette.keys()):
        color = palette[key]
        draw.ellipse([x, yy + 3, x + 10, yy + 13], fill=color)
        draw.text((x + 16, yy), key, fill=(40, 40, 40), font=_font(12))
        yy += 18
    draw.text((x, yy + 5), "Markers", fill="black", font=_font(14))
    yy += 24
    draw.ellipse([x, yy + 3, x + 10, yy + 13], outline="black", width=2)
    draw.text((x + 16, yy), "fallback_heavy", fill=(40, 40, 40), font=_font(12))
    yy += 18
    draw.line([(x, yy + 3), (x + 10, yy + 13)], fill="black", width=2)
    draw.line([(x + 10, yy + 3), (x, yy + 13)], fill="black", width=2)
    draw.text((x + 16, yy), "missing_geometry", fill=(40, 40, 40), font=_font(12))


def _plot_risk_vs_confidence(rows: list[dict[str, Any]]) -> None:
    img, draw = _new_canvas("Risk vs Confidence by Scenario Group")
    _draw_grid_and_axes(
        draw,
        x_label="Confidence Score",
        y_label="Wildfire Risk Score",
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
    )
    palette = _scenario_palette([row["scenario_group"] for row in rows])

    for row in rows:
        risk = row["risk_score"]
        conf = row["confidence_score"]
        if risk is None or conf is None:
            continue
        px, py = _map_plot_point(conf, risk, 0.0, 100.0, 0.0, 100.0)
        c = palette[row["scenario_group"]]
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=c)
        if "fallback_heavy" in row["flags"]:
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline="black", width=2)
        if "missing_geometry" in row["flags"]:
            draw.line([(px - 6, py - 6), (px + 6, py + 6)], fill="black", width=2)
            draw.line([(px + 6, py - 6), (px - 6, py + 6)], fill="black", width=2)

    _draw_legend(draw, palette)
    img.save(RISK_VS_CONFIDENCE_PNG)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _draw_box(
    draw: ImageDraw.ImageDraw,
    values: list[float],
    x_center: int,
    width: int,
    y_min: float,
    y_max: float,
    color: tuple[int, int, int],
) -> None:
    if not values:
        return
    vals = sorted(values)
    q1 = _percentile(vals, 0.25)
    q2 = _percentile(vals, 0.50)
    q3 = _percentile(vals, 0.75)
    low = vals[0]
    high = vals[-1]

    def ypx(v: float) -> int:
        return _map_plot_point(0.0, v, 0.0, 1.0, y_min, y_max)[1]

    y_q1 = ypx(q1)
    y_q2 = ypx(q2)
    y_q3 = ypx(q3)
    y_low = ypx(low)
    y_high = ypx(high)

    draw.rectangle([x_center - width // 2, y_q3, x_center + width // 2, y_q1], outline=color, width=2)
    draw.line([(x_center - width // 2, y_q2), (x_center + width // 2, y_q2)], fill=color, width=2)
    draw.line([(x_center, y_q3), (x_center, y_high)], fill=color, width=2)
    draw.line([(x_center, y_q1), (x_center, y_low)], fill=color, width=2)
    draw.line([(x_center - width // 4, y_high), (x_center + width // 4, y_high)], fill=color, width=2)
    draw.line([(x_center - width // 4, y_low), (x_center + width // 4, y_low)], fill=color, width=2)


def _plot_box_by_scenario(
    rows: list[dict[str, Any]],
    output: Path,
    title: str,
    key: str,
    y_label: str,
) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)):
            grouped[row["scenario_group"]].append(float(val))

    labels = sorted(grouped.keys())
    vals = [v for arr in grouped.values() for v in arr]
    y_min = min(vals) if vals else 0.0
    y_max = max(vals) if vals else 1.0
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0
    y_min = min(0.0, y_min)
    if key == "confidence_score":
        y_max = max(100.0, y_max)
    else:
        y_max = max(y_max * 1.1, y_max + 1.0)

    img, draw = _new_canvas(title)
    _draw_grid_and_axes(
        draw,
        x_label="Scenario Group",
        y_label=y_label,
        x_min=0.0,
        x_max=float(max(1, len(labels))),
        y_min=y_min,
        y_max=y_max,
    )
    palette = _scenario_palette(labels)
    n = max(1, len(labels))
    for i, label in enumerate(labels):
        x_center = int(MARGIN_LEFT + ((i + 0.5) / n) * PLOT_W)
        box_w = max(24, int(PLOT_W / (n * 3)))
        _draw_box(draw, grouped[label], x_center, box_w, y_min, y_max, palette[label])
        short = label if len(label) <= 16 else (label[:14] + "..")
        draw.text((x_center - 28, HEIGHT - MARGIN_BOTTOM + 24), short, fill=(60, 60, 60), font=_font(12))

    _draw_legend(draw, palette)
    img.save(output)


def _paired_deltas(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    by_address: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["scenario_group"] != "mitigation_improved":
            continue
        if row["risk_score"] is None:
            continue
        by_address[row["address"]].append(float(row["risk_score"]))
    deltas: list[tuple[str, float]] = []
    for address, vals in by_address.items():
        if len(vals) < 2:
            continue
        deltas.append((address, max(vals) - min(vals)))
    return deltas


def _plot_optional_input_effects(rows: list[dict[str, Any]]) -> bool:
    deltas = _paired_deltas(rows)
    if not deltas:
        return False

    img, draw = _new_canvas("Optional Input Effects (Paired Scenario Risk Delta)")
    max_delta = max(v for _, v in deltas)
    y_max = max(1.0, max_delta * 1.15)
    _draw_grid_and_axes(
        draw,
        x_label="Paired Scenario",
        y_label="Risk Score Delta",
        x_min=0.0,
        x_max=float(max(1, len(deltas))),
        y_min=0.0,
        y_max=y_max,
    )

    n = len(deltas)
    for i, (address, delta) in enumerate(deltas):
        x0 = int(MARGIN_LEFT + ((i + 0.12) / n) * PLOT_W)
        x1 = int(MARGIN_LEFT + ((i + 0.88) / n) * PLOT_W)
        _px, yv = _map_plot_point(0.0, delta, 0.0, 1.0, 0.0, y_max)
        y_base = HEIGHT - MARGIN_BOTTOM
        draw.rectangle([x0, yv, x1, y_base], fill=(76, 120, 168))
        draw.text((x0, yv - 16), f"{delta:.2f}", fill=(40, 40, 40), font=_font(12))
        short = address if len(address) <= 18 else (address[:16] + "..")
        draw.text((x0, y_base + 24), short, fill=(60, 60, 60), font=_font(11))

    img.save(OPTIONAL_EFFECTS_PNG)
    return True


def _read_diagnostics(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risks = [float(r["risk_score"]) for r in rows if r["risk_score"] is not None]
    confs = [float(r["confidence_score"]) for r in rows if r["confidence_score"] is not None]
    low_low = [
        r
        for r in rows
        if r["risk_score"] is not None
        and r["confidence_score"] is not None
        and r["risk_score"] <= 5.0
        and r["confidence_score"] <= 10.0
    ]
    fallback_heavy = [r for r in rows if "fallback_heavy" in r["flags"]]
    missing_geometry = [r for r in rows if "missing_geometry" in r["flags"]]
    scenario_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        scenario_counts[r["scenario_group"]] += 1
    return {
        "total": len(rows),
        "risk_mean": mean(risks) if risks else 0.0,
        "risk_median": median(risks) if risks else 0.0,
        "conf_mean": mean(confs) if confs else 0.0,
        "conf_median": median(confs) if confs else 0.0,
        "low_low_count": len(low_low),
        "fallback_heavy_count": len(fallback_heavy),
        "missing_geometry_count": len(missing_geometry),
        "scenario_counts": dict(sorted(scenario_counts.items())),
    }


def _write_summary(rows: list[dict[str, Any]], has_optional_effects: bool) -> None:
    stats = _stats(rows)
    diagnostics_text = _read_diagnostics(DIAGNOSTICS_MD)
    suspicious_note = (
        "Suspiciously low high_regional_hazard rows were flagged in diagnostics."
        if "Suspiciously low risk in `high_regional_hazard` rows" in diagnostics_text
        else "No high_regional_hazard anomaly line found in diagnostics."
    )
    scenario_counts_text = ", ".join([f"{k} ({v})" for k, v in stats["scenario_counts"].items()])

    lines: list[str] = [
        "# Benchmark Summary",
        "",
        "## Short Summary",
        f"- Loaded **{stats['total']}** benchmark assessments from `reports/benchmark_results.csv`.",
        f"- Mean wildfire risk score: **{stats['risk_mean']:.2f}** (median **{stats['risk_median']:.2f}**).",
        f"- Mean confidence score: **{stats['conf_mean']:.2f}** (median **{stats['conf_median']:.2f}**).",
        f"- Rows with low risk + low confidence (risk <= 5 and confidence <= 10): **{stats['low_low_count']}**.",
        f"- Rows flagged `fallback_heavy`: **{stats['fallback_heavy_count']}**.",
        f"- Rows flagged `missing_geometry`: **{stats['missing_geometry_count']}**.",
        f"- {suspicious_note}",
        f"- Scenario groups observed: {scenario_counts_text}.",
        "",
        "## Key Findings",
        "- A large share of low risk outputs occur with very low confidence, so risk values should not be interpreted alone.",
        "- Fallback-heavy runs are common and align with weaker confidence and sparse geometry evidence.",
        "- Scenario consistency is uneven: high regional hazard-labeled rows include very low risk values in this output.",
        "",
        "## Figures",
        "![Risk vs Confidence](./risk_vs_confidence.png)",
        "",
        "![Risk by Scenario](./risk_by_scenario.png)",
        "",
        "![Confidence by Scenario](./confidence_by_scenario.png)",
        "",
    ]
    if has_optional_effects:
        lines += [
            "![Optional Input Effects](./optional_input_effects.png)",
            "",
        ]
    else:
        lines += [
            "_Optional input effects figure omitted: not enough paired rows with valid risk values._",
            "",
        ]
    lines += [
        "## Key Takeaways for Model Reliability",
        "- Confidence and data-availability context should be read together with risk score.",
        "- Fallback-heavy or missing-geometry evidence can reduce comparability across properties.",
        "- Scenario intent labels need mapped scoring inputs to validate those scenarios reliably.",
        "",
        "## Next Steps (Non-invasive)",
        "- Add reporting guardrails that prominently flag low-confidence + fallback-heavy results.",
        "- Improve benchmark fixtures so scenario hints map to accepted scoring attributes.",
        "- Prioritize data-availability improvements (geometry coverage, missing layer fill) to reduce fallback-heavy runs.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing required input: {RESULTS_CSV}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(RESULTS_CSV)

    _plot_risk_vs_confidence(rows)
    _plot_box_by_scenario(
        rows,
        output=RISK_BY_SCENARIO_PNG,
        title="Risk Score Distribution by Scenario Group",
        key="risk_score",
        y_label="Wildfire Risk Score",
    )
    _plot_box_by_scenario(
        rows,
        output=CONF_BY_SCENARIO_PNG,
        title="Confidence Score Distribution by Scenario Group",
        key="confidence_score",
        y_label="Confidence Score",
    )
    has_optional_effects = _plot_optional_input_effects(rows)
    _write_summary(rows, has_optional_effects=has_optional_effects)

    print(f"Wrote {RISK_VS_CONFIDENCE_PNG}")
    print(f"Wrote {RISK_BY_SCENARIO_PNG}")
    print(f"Wrote {CONF_BY_SCENARIO_PNG}")
    if has_optional_effects:
        print(f"Wrote {OPTIONAL_EFFECTS_PNG}")
    print(f"Wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
