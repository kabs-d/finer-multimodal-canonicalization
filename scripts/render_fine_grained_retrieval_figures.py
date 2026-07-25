#!/usr/bin/env python3
"""Render compact SVG figures for fine-grained retrieval results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "artifacts" / "results" / "fine_grained_retrieval"
FIGURE_ROOT = PROJECT_ROOT / "docs" / "frozen_encoder" / "figures"

RUNS = {
    "OpenAI->LAION": "cub_openai_vitb32_to_laion_vitb32_linear_fine_grained_retrieval",
    "OpenAI->FLAVA": "cub_openai_vitl14_to_flava_linear_fine_grained_retrieval",
}

CONDITIONS = [
    ("Native target", "native_target", "#67c5d0"),
    ("Oxford Q", "oxford_aligned_source", "#f2aa45"),
    ("CUB-train Q", "cub_train_aligned_source", "#9bc75f"),
    ("Unaligned", "unaligned_source", "#b7b7b7"),
]


@dataclass(frozen=True)
class Bar:
    group: str
    label: str
    value: float
    color: str


@dataclass(frozen=True)
class Panel:
    title: str
    bars: list[Bar]
    ymax: float
    ymin: float = -1.0


def load(run_id: str) -> dict:
    return json.loads((RESULT_ROOT / run_id / "summary.json").read_text())


def metric(data: dict, condition: str, k: int, metric_name: str) -> float:
    return 100.0 * float(data["aggregate"][condition][f"k{k}"][metric_name])


def legend() -> str:
    pieces = []
    x = 18
    for label, _, color in CONDITIONS:
        pieces.append(f'<rect x="{x}" y="18" width="11" height="11" fill="{color}"/>')
        pieces.append(
            f'<text x="{x + 16}" y="28" font-size="11" font-family="Times New Roman, serif">{label}</text>'
        )
        x += 16 + 7 * len(label) + 28
    return "\n".join(pieces)


def render_panel(panel: Panel, x0: int, y0: int, width: int, height: int) -> str:
    left, right, top, bottom = 44, 12, 34, 58
    plot_x = x0 + left
    plot_y = y0 + top
    plot_w = width - left - right
    plot_h = height - top - bottom
    groups = []
    for bar in panel.bars:
        if bar.group not in groups:
            groups.append(bar.group)
    labels = []
    for bar in panel.bars:
        if bar.label not in labels:
            labels.append(bar.label)
    group_w = plot_w / len(groups)
    bar_w = min(15, group_w / (len(labels) + 1.3))
    span = panel.ymax - panel.ymin
    zero_y = plot_y + plot_h - plot_h * (0 - panel.ymin) / span
    out = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 15}" text-anchor="middle" '
        f'font-size="13" font-weight="600" font-family="Times New Roman, serif">{panel.title}</text>'
    ]
    for tick in range(0, 6):
        value = panel.ymin + span * tick / 5
        y = plot_y + plot_h - plot_h * (value - panel.ymin) / span
        out.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" '
            'stroke="#999" stroke-width="0.6" stroke-dasharray="2 2"/>'
        )
        out.append(
            f'<text x="{plot_x - 7}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="9" font-family="Times New Roman, serif">{value:.1f}</text>'
        )
    out.append(
        f'<line x1="{plot_x}" y1="{zero_y:.1f}" x2="{plot_x + plot_w}" y2="{zero_y:.1f}" '
        'stroke="#333" stroke-width="0.9"/>'
    )
    out.append(
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" '
        'stroke="#444" stroke-width="0.8"/>'
    )
    for group_index, group in enumerate(groups):
        group_bars = [bar for bar in panel.bars if bar.group == group]
        center = plot_x + group_w * (group_index + 0.5)
        total_w = len(group_bars) * bar_w + (len(group_bars) - 1) * 3
        start = center - total_w / 2
        for bar_index, bar in enumerate(group_bars):
            x = start + bar_index * (bar_w + 3)
            y = plot_y + plot_h - plot_h * (bar.value - panel.ymin) / span
            top_y = min(y, zero_y)
            h = abs(zero_y - y)
            out.append(
                f'<rect x="{x:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{bar.color}"/>'
            )
            out.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{top_y - 3:.1f}" text-anchor="middle" '
                f'font-size="8" font-family="Times New Roman, serif">{bar.value:.1f}</text>'
            )
        out.append(
            f'<text x="{center:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" '
            f'font-size="10" font-family="Times New Roman, serif">{group}</text>'
        )
    out.append(
        f'<text x="{x0 + 12}" y="{plot_y + plot_h / 2:.1f}" transform="rotate(-90 {x0 + 12} {plot_y + plot_h / 2:.1f})" '
        'text-anchor="middle" font-size="10" font-family="Times New Roman, serif">points over random</text>'
    )
    return "\n".join(out)


def render_figure(path: Path, panels: list[Panel], caption: str) -> None:
    panel_w, panel_h = 330, 250
    width = panel_w * len(panels)
    height = 48 + panel_h + 34
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        legend(),
    ]
    for index, panel in enumerate(panels):
        body.append(render_panel(panel, index * panel_w, 48, panel_w, panel_h))
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="11" font-family="Times New Roman, serif">{caption}</text>'
    )
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def panels_for(metric_name: str, title_suffix: str, ymax: float) -> list[Panel]:
    panels = []
    for k in [1, 5, 10]:
        bars = []
        for group, run_id in RUNS.items():
            data = load(run_id)
            random_value = metric(data, "random_same_species", k, metric_name)
            for label, condition, color in CONDITIONS:
                value = metric(data, condition, k, metric_name) - random_value
                bars.append(Bar(group, label, value, color))
        panels.append(Panel(f"{title_suffix}@{k}", bars, ymax=ymax))
    return panels


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    render_figure(
        FIGURE_ROOT / "fine_retrieval_attribute_overlap.svg",
        panels_for("same_species_attribute_overlap", "Attribute overlap", 3.5),
        "Same-species retrieval gain over random; higher means neighbors share more query attributes.",
    )
    render_figure(
        FIGURE_ROOT / "fine_retrieval_rare_recall.svg",
        panels_for("rare_attribute_recall", "Rare recall", 4.0),
        "Rare-attribute recall gain over random; rare attributes are train-prevalence bottom quartile.",
    )


if __name__ == "__main__":
    main()
