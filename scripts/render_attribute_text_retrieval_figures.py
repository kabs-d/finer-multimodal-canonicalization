#!/usr/bin/env python3
"""Render Phase II global attribute-text retrieval figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "artifacts" / "results" / "attribute_text_retrieval"
FIGURE_ROOT = PROJECT_ROOT / "docs" / "phase2" / "figures"

RUNS = {
    "OpenAI B/32 → LAION B/32": "cub_openai_vitb32_to_laion_vitb32_linear_attribute_text_global_retrieval",
    "OpenAI L/14 → FLAVA": "cub_openai_vitl14_to_flava_linear_attribute_text_global_retrieval",
}
CONDITIONS = [
    ("Native source", "native_source", "#4c78a8"),
    ("Native target", "native_target", "#67c5d0"),
    ("Oxford Q", "oxford_aligned_source_to_target", "#f2aa45"),
    ("CUB-train Q", "cub_train_aligned_source_to_target", "#9bc75f"),
    ("Unaligned", "unaligned_source_to_target", "#b7b7b7"),
    ("Random", "__random__", "#ffffff"),
]


@dataclass(frozen=True)
class Bar:
    group: str
    label: str
    value: float
    color: str
    outline: bool = False


@dataclass(frozen=True)
class Panel:
    title: str
    bars: list[Bar]
    ylabel: str
    ymin: float = 0.0
    ymax: float = 70.0
    chance_line: float | None = None


def load(run_id: str) -> dict:
    return json.loads((RESULT_ROOT / run_id / "summary.json").read_text())


def metric(data: dict, subset: str, condition: str, name: str) -> float:
    if condition == "__random__":
        if name == "ranking_accuracy_macro":
            return 100.0 * data["aggregate"][subset]["native_source"]["random_ranking_accuracy"]
        return 100.0 * data["aggregate"][subset]["native_source"]["random_precision_at_k"]
    return 100.0 * data["aggregate"][subset][condition][name]


def legend() -> str:
    pieces = []
    x = 18
    for label, _, color in CONDITIONS:
        outline = ' stroke="#444" stroke-width="0.8"' if label == "Random" else ""
        pieces.append(f'<rect x="{x}" y="17" width="11" height="11" fill="{color}"{outline}/>')
        pieces.append(
            f'<text x="{x + 16}" y="27" font-size="11" font-family="Times New Roman, serif">{label}</text>'
        )
        x += 16 + 6.6 * len(label) + 20
    return "\n".join(pieces)


def render_panel(panel: Panel, x0: int, y0: int, width: int, height: int) -> str:
    left, right, top, bottom = 50, 14, 34, 62
    plot_x = x0 + left
    plot_y = y0 + top
    plot_w = width - left - right
    plot_h = height - top - bottom
    groups = []
    for bar in panel.bars:
        if bar.group not in groups:
            groups.append(bar.group)
    group_w = plot_w / len(groups)
    max_bars = max(sum(bar.group == group for bar in panel.bars) for group in groups)
    bar_w = min(14, group_w / (max_bars + 1.5))
    span = panel.ymax - panel.ymin
    out = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 15}" text-anchor="middle" '
        f'font-size="13" font-weight="600" font-family="Times New Roman, serif">{panel.title}</text>'
    ]
    for tick in range(6):
        value = panel.ymin + span * tick / 5
        y = plot_y + plot_h - plot_h * (value - panel.ymin) / span
        out.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" '
            'stroke="#999" stroke-width="0.6" stroke-dasharray="2 2"/>'
        )
        out.append(
            f'<text x="{plot_x - 7}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="9" font-family="Times New Roman, serif">{value:.0f}</text>'
        )
    if panel.chance_line is not None:
        y = plot_y + plot_h - plot_h * (panel.chance_line - panel.ymin) / span
        out.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" '
            'stroke="#333" stroke-width="1.0"/>'
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
            h = plot_y + plot_h - y
            outline = ' stroke="#444" stroke-width="0.8"' if bar.outline else ""
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{bar.color}"{outline}/>'
            )
            out.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
                f'font-size="7.5" font-family="Times New Roman, serif">{bar.value:.1f}</text>'
            )
        out.append(
            f'<text x="{center:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" '
            f'font-size="10" font-family="Times New Roman, serif">{group}</text>'
        )
    out.append(
        f'<text x="{x0 + 13}" y="{plot_y + plot_h / 2:.1f}" '
        f'transform="rotate(-90 {x0 + 13} {plot_y + plot_h / 2:.1f})" '
        f'text-anchor="middle" font-size="10" font-family="Times New Roman, serif">{panel.ylabel}</text>'
    )
    return "\n".join(out)


def render(path: Path, panels: list[Panel], caption: str) -> None:
    panel_w, panel_h = 440, 268
    width = panel_w * len(panels)
    height = 48 + panel_h + 35
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        legend(),
    ]
    for index, panel in enumerate(panels):
        body.append(render_panel(panel, panel_w * index, 48, panel_w, panel_h))
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="11" font-family="Times New Roman, serif">{caption}</text>'
    )
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def panel_for(subset: str, metric_name: str, title: str, ylabel: str, ymax: float, chance: float | None = None) -> Panel:
    bars = []
    for group, run_id in RUNS.items():
        data = load(run_id)
        for label, condition, color in CONDITIONS:
            bars.append(
                Bar(
                    group,
                    label,
                    metric(data, subset, condition, metric_name),
                    color,
                    outline=(condition == "__random__"),
                )
            )
    return Panel(title, bars, ylabel=ylabel, ymax=ymax, chance_line=chance)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    render(
        FIGURE_ROOT / "global_attribute_p10.svg",
        [
            panel_for("clip_readable", "precision_at_10_macro", "CLIP-readable subset", "P@10 (%)", 50),
            panel_for("all_312", "precision_at_10_macro", "All 312 CUB attributes", "P@10 (%)", 40),
        ],
        "Attribute-only text retrieval over all CUB test images with visible labels; random is the attribute base rate.",
    )
    render(
        FIGURE_ROOT / "global_attribute_ranking.svg",
        [
            panel_for("clip_readable", "ranking_accuracy_macro", "CLIP-readable subset", "positive > negative (%)", 70, 50),
            panel_for("all_312", "ranking_accuracy_macro", "All 312 CUB attributes", "positive > negative (%)", 70, 50),
        ],
        "Ranking accuracy compares every visible-positive image against every visible-negative image for each attribute.",
    )
    (FIGURE_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "global_attribute_p10": "global_attribute_p10.svg",
                "global_attribute_ranking": "global_attribute_ranking.svg",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
