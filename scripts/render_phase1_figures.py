#!/usr/bin/env python3
"""Render compact paper-style SVG figures for the frozen-encoder README."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "artifacts" / "results" / "frozen_decoder"
FIGURE_ROOT = PROJECT_ROOT / "docs" / "frozen_encoder" / "figures"

PAIRS = {
    "LAION": {
        "label": "OpenAI B/32 -> LAION B/32",
        "short": "OpenAI->LAION",
        "linear": "cub_openai_vitb32_to_laion_vitb32_linear",
        "cubq": "cub_openai_vitb32_to_laion_vitb32_cub_train_q_linear",
        "mlp": "cub_openai_vitb32_to_laion_vitb32_mlp_h512",
    },
    "FLAVA": {
        "label": "OpenAI L/14 -> FLAVA",
        "short": "OpenAI->FLAVA",
        "linear": "cub_openai_vitl14_to_flava_linear",
        "cubq": "cub_openai_vitl14_to_flava_cub_train_q_linear",
        "mlp": "cub_openai_vitl14_to_flava_mlp_h512",
    },
}

COLORS = {
    "Before Q": "#e779b8",
    "Native source": "#e779b8",
    "Native target": "#67c5d0",
    "Oxford Q": "#f2aa45",
    "CUB-train Q": "#9bc75f",
    "Aligned source": "#f2aa45",
    "Unaligned source": "#b7b7b7",
    "Species only": "#8f8f8f",
    "Linear": "#f2aa45",
    "MLP": "#9bc75f",
    "Two-layer MLP": "#9bc75f",
    "Linear native": "#67c5d0",
    "Linear aligned": "#f2aa45",
    "Two-layer MLP native": "#9bc75f",
    "Two-layer MLP aligned": "#5c9e56",
    "Native MLP": "#67c5d0",
    "Unaligned MLP": "#b7b7b7",
    "Aligned MLP": "#5c9e56",
    "CUB-aligned MLP": "#9bc75f",
    "Oxford-Q aligned": "#5c9e56",
    "CUB-Q aligned": "#9bc75f",
}


@dataclass(frozen=True)
class Bar:
    group: str
    label: str
    value: float
    color: str


@dataclass(frozen=True)
class Panel:
    title: str
    ylabel: str
    bars: list[Bar]
    ymax: float
    percent: bool = False
    chance: float | None = None


def load_json(run_id: str, name: str) -> dict:
    return json.loads((RESULT_ROOT / run_id / name).read_text(encoding="utf-8"))


def metric_mean(run_id: str, condition: str, metric: str) -> float:
    data = load_json(run_id, "attribute_interpretability.json")
    return data["aggregate"][condition]["per_bird_recovery"][metric]["mean"]


def within_species(run_id: str, condition: str) -> float:
    data = load_json(run_id, "attribute_interpretability.json")
    return data["aggregate"][condition]["within_species"]["macro_pair_accuracy"]["mean"]


def species_only(metric: str) -> float:
    data = load_json(PAIRS["LAION"]["linear"], "attribute_interpretability.json")
    if metric == "within_species":
        return data["species_only"]["within_species"]["macro_pair_accuracy"]
    return data["species_only"]["per_bird_recovery"][metric]


def fmt(value: float, percent: bool) -> str:
    if percent:
        return f"{100.0 * value:.1f}"
    if value < 1.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def legend(labels: Iterable[str]) -> str:
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    x = 20
    pieces = []
    for label in seen:
        color = COLORS[label]
        pieces.append(f'<rect x="{x}" y="18" width="11" height="11" fill="{color}"/>')
        pieces.append(
            f'<text x="{x + 16}" y="28" font-size="11" font-family="Times New Roman, serif">{label}</text>'
        )
        x += 16 + 7 * len(label) + 28
    return "\n".join(pieces)


def render_panel(panel: Panel, x0: int, y0: int, width: int, height: int) -> str:
    left, right, top, bottom = 42, 12, 34, 52
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
    bars_by_group = {
        group: [bar for bar in panel.bars if bar.group == group] for group in groups
    }
    group_w = plot_w / len(groups)
    bar_w = min(18, group_w / (len(labels) + 1.4))
    out = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 15}" text-anchor="middle" '
        f'font-size="13" font-weight="600" font-family="Times New Roman, serif">{panel.title}</text>'
    ]
    for tick in range(0, 6):
        value = panel.ymax * tick / 5
        y = plot_y + plot_h - plot_h * value / panel.ymax
        out.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" '
            'stroke="#999" stroke-width="0.6" stroke-dasharray="2 2"/>'
        )
        out.append(
            f'<text x="{plot_x - 7}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="9" font-family="Times New Roman, serif">{fmt(value, panel.percent)}</text>'
        )
    if panel.chance is not None:
        y = plot_y + plot_h - plot_h * panel.chance / panel.ymax
        out.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" '
            'stroke="#333" stroke-width="0.8" stroke-dasharray="4 3"/>'
        )
    out.append(
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" '
        'stroke="#444" stroke-width="0.8"/>'
    )
    out.append(
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" '
        'stroke="#444" stroke-width="0.8"/>'
    )
    for g_index, group in enumerate(groups):
        bars = bars_by_group[group]
        center = plot_x + group_w * (g_index + 0.5)
        total_w = bar_w * len(bars) + 3 * (len(bars) - 1)
        start = center - total_w / 2
        for b_index, bar in enumerate(bars):
            h = plot_h * max(0.0, bar.value) / panel.ymax
            x = start + b_index * (bar_w + 3)
            y = plot_y + plot_h - h
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{bar.color}"/>'
            )
            out.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
                f'font-size="8" font-family="Times New Roman, serif">{fmt(bar.value, panel.percent)}</text>'
            )
        out.append(
            f'<text x="{center:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" '
            f'font-size="10" font-family="Times New Roman, serif">{group}</text>'
        )
    out.append(
        f'<text x="{x0 + 12}" y="{plot_y + plot_h / 2:.1f}" transform="rotate(-90 {x0 + 12} {plot_y + plot_h / 2:.1f})" '
        f'text-anchor="middle" font-size="10" font-family="Times New Roman, serif">{panel.ylabel}</text>'
    )
    return "\n".join(out)


def render_figure(
    path: Path,
    panels: list[Panel],
    caption: str,
    *,
    columns: int = 2,
    panel_width: int = 330,
) -> None:
    panel_w, panel_h = panel_width, 250
    rows = (len(panels) + columns - 1) // columns
    width = columns * panel_w
    height = 48 + rows * panel_h + 34
    labels = [bar.label for panel in panels for bar in panel.bars]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        legend(labels),
    ]
    for index, panel in enumerate(panels):
        col = index % columns
        row = index // columns
        body.append(render_panel(panel, col * panel_w, 48 + row * panel_h, panel_w, panel_h))
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="11" font-family="Times New Roman, serif">{caption}</text>'
    )
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def geometry_panels() -> list[Panel]:
    image_bars, text_bars = [], []
    for pair in PAIRS.values():
        ox = load_json(pair["linear"], "alignment_metrics.json")["Cosine"]
        cq = load_json(pair["cubq"], "alignment_metrics.json")["Cosine"]
        group = pair["short"]
        image_bars.extend(
            [
                Bar(group, "Before Q", ox["Image_before"], COLORS["Before Q"]),
                Bar(group, "Oxford Q", ox["Image_after"], COLORS["Oxford Q"]),
                Bar(group, "CUB-train Q", cq["Image_after"], COLORS["CUB-train Q"]),
            ]
        )
        text_bars.extend(
            [
                Bar(group, "Before Q", ox["Text_before"], COLORS["Before Q"]),
                Bar(group, "Oxford Q", ox["Text_after"], COLORS["Oxford Q"]),
                Bar(group, "CUB-train Q", cq["Text_after"], COLORS["CUB-train Q"]),
            ]
        )
    return [
        Panel("(a) Image-image cosine", "paired cosine", image_bars, 1.0),
        Panel("(b) Text-text cosine", "paired cosine", text_bars, 1.0),
    ]


def class_panels() -> list[Panel]:
    retrieval_bars, zeroshot_bars = [], []
    for pair in PAIRS.values():
        ox = load_json(pair["linear"], "alignment_metrics.json")
        cq = load_json(pair["cubq"], "alignment_metrics.json")
        group = pair["short"]
        retrieval_bars.extend(
            [
                Bar(group, "Before Q", ox["ImageImage"]["Baseline"], COLORS["Before Q"]),
                Bar(group, "Oxford Q", ox["ImageImage"]["Procrustes"], COLORS["Oxford Q"]),
                Bar(group, "CUB-train Q", cq["ImageImage"]["Procrustes"], COLORS["CUB-train Q"]),
            ]
        )
        zeroshot_bars.extend(
            [
                Bar(group, "Native source", ox["ImageText"]["A_to_A"], COLORS["Native source"]),
                Bar(group, "Native target", ox["ImageText"]["B_to_B"], COLORS["Native target"]),
                Bar(group, "Oxford Q", ox["ImageText"]["Aligned_imgA_to_aligned_textA"], COLORS["Oxford Q"]),
                Bar(group, "CUB-train Q", cq["ImageText"]["Aligned_imgA_to_aligned_textA"], COLORS["CUB-train Q"]),
            ]
        )
    return [
        Panel("(a) Image-image retrieval", "top-1 accuracy", retrieval_bars, 1.0, percent=True),
        Panel("(b) Joint zero-shot species", "top-1 accuracy", zeroshot_bars, 0.75, percent=True),
    ]


def readout_count_panels() -> list[Panel]:
    specs = [
        ("(a) Correctly recovered", "mean_correctly_recovered", 24.0),
        ("(b) Missed", "mean_missed", 20.0),
        ("(c) Hallucinated", "mean_hallucinated", 72.0),
    ]
    panels = []
    for title, metric, ymax in specs:
        bars = []
        for pair in PAIRS.values():
            group = pair["short"]
            bars.extend(
                [
                    Bar(group, "Native target", metric_mean(pair["linear"], "native_target", metric), COLORS["Native target"]),
                    Bar(group, "Oxford Q", metric_mean(pair["linear"], "aligned_source", metric), COLORS["Oxford Q"]),
                    Bar(group, "CUB-train Q", metric_mean(pair["cubq"], "aligned_source", metric), COLORS["CUB-train Q"]),
                    Bar(group, "Unaligned source", metric_mean(pair["linear"], "unaligned_source", metric), COLORS["Unaligned source"]),
                ]
            )
        panels.append(Panel(title, "attributes / bird", bars, ymax))
    return panels


def within_species_panel() -> list[Panel]:
    bars = []
    for pair in PAIRS.values():
        group = pair["short"]
        bars.extend(
            [
                Bar(group, "Native target", within_species(pair["linear"], "native_target"), COLORS["Native target"]),
                Bar(group, "Oxford Q", within_species(pair["linear"], "aligned_source"), COLORS["Oxford Q"]),
                Bar(group, "CUB-train Q", within_species(pair["cubq"], "aligned_source"), COLORS["CUB-train Q"]),
                Bar(group, "Unaligned source", within_species(pair["linear"], "unaligned_source"), COLORS["Unaligned source"]),
                Bar(group, "Species only", species_only("within_species"), COLORS["Species only"]),
            ]
        )
    return [Panel("Species-controlled attribute ranking", "pair accuracy", bars, 0.7, percent=True, chance=0.5)]


def mlp_panels() -> list[Panel]:
    recovered, ranking = [], []
    for pair in PAIRS.values():
        group = pair["short"]
        recovered.extend(
            [
                Bar(group, "Linear", metric_mean(pair["linear"], "aligned_source", "mean_correctly_recovered"), COLORS["Linear"]),
                Bar(group, "MLP", metric_mean(pair["mlp"], "aligned_source", "mean_correctly_recovered"), COLORS["MLP"]),
            ]
        )
        ranking.extend(
            [
                Bar(group, "Linear", within_species(pair["linear"], "aligned_source"), COLORS["Linear"]),
                Bar(group, "MLP", within_species(pair["mlp"], "aligned_source"), COLORS["MLP"]),
            ]
        )
    return [
        Panel("(a) Recovered attributes", "attributes / bird", recovered, 22.0),
        Panel("(b) Within-species ranking", "pair accuracy", ranking, 0.7, percent=True, chance=0.5),
    ]


def bidirectional_mlp_panel(pair: str, title: str) -> list[Panel]:
    """One compact native/unaligned/aligned MLP figure for a model pair."""
    deep = load_probe("deep_mlp_probe", pair)
    bars = [
        Bar("Source → target", "Native MLP", deep["source_native_percent_mean"] / 100, COLORS["Native MLP"]),
        Bar("Source → target", "Unaligned MLP", deep["source_decoder_on_unaligned_target_percent_mean"] / 100, COLORS["Unaligned MLP"]),
        Bar("Source → target", "Aligned MLP", deep["source_decoder_on_aligned_target_percent_mean"] / 100, COLORS["Aligned MLP"]),
        Bar("Source → target", "CUB-aligned MLP", deep["source_decoder_on_cub_aligned_target_percent_mean"] / 100, COLORS["CUB-aligned MLP"]),
        Bar("Target → source", "Native MLP", deep["target_native_percent_mean"] / 100, COLORS["Native MLP"]),
        Bar("Target → source", "Unaligned MLP", deep["target_decoder_on_unaligned_source_percent_mean"] / 100, COLORS["Unaligned MLP"]),
        Bar("Target → source", "Aligned MLP", deep["target_decoder_on_aligned_source_percent_mean"] / 100, COLORS["Aligned MLP"]),
        Bar("Target → source", "CUB-aligned MLP", deep["target_decoder_on_cub_aligned_source_percent_mean"] / 100, COLORS["CUB-aligned MLP"]),
    ]
    return [Panel(title, "attributes recovered (%)", bars, 0.8, percent=True)]


def unidirectional_mlp_panel() -> list[Panel]:
    """Target-space MLP applied to native, unaligned, and aligned sources."""
    specs = [
        ("OpenAI → LAION", "cub_openai_vitb32_to_laion_vitb32_linear"),
        ("OpenAI → FLAVA", "cub_openai_vitl14_to_flava_linear"),
    ]
    bars = []
    for group, pair in specs:
        deep = load_probe("deep_mlp_probe", pair)
        bars.extend(
            [
                Bar(group, "Native target", deep["target_native_percent_mean"] / 100, COLORS["Native MLP"]),
                Bar(group, "Unaligned source", deep["target_decoder_on_unaligned_source_percent_mean"] / 100, COLORS["Unaligned MLP"]),
                Bar(group, "Oxford-Q aligned", deep["target_decoder_on_aligned_source_percent_mean"] / 100, COLORS["Aligned MLP"]),
                Bar(group, "CUB-Q aligned", deep["target_decoder_on_cub_aligned_source_percent_mean"] / 100, COLORS["CUB-aligned MLP"]),
            ]
        )
    return [Panel("Target-space MLP transfer", "attributes recovered (%)", bars, 0.8, percent=True)]


def load_probe(probe: str, pair: str) -> dict:
    path = PROJECT_ROOT / "artifacts" / "results" / probe / pair / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figures = [
        ("geometry_alignment.svg", geometry_panels(), "Q moves cross-model image/text pairs from near-zero cosine to strong agreement.", 2, 330),
        ("class_level_transfer.svg", class_panels(), "In-domain Q mainly improves image retrieval; zero-shot transfer is compared with native baselines.", 2, 330),
        ("readout_counts.svg", readout_count_panels(), "CUB-train Q recovers more true attributes, with a hallucination tradeoff.", 3, 330),
        ("within_species_ranking.svg", within_species_panel(), "Above-chance same-species ranking shows fine-grained signal beyond species identity.", 1, 330),
        ("mlp_capacity.svg", mlp_panels(), "The MLP recovers more attributes but does not improve species-controlled transfer.", 2, 330),
        ("decoder_transfer_laion.svg", bidirectional_mlp_panel("cub_openai_vitb32_to_laion_vitb32_linear", "OpenAI B/32 → LAION B/32"), "Each group compares the two-layer MLP on native, unaligned, Oxford-aligned, and CUB-aligned embeddings.", 1, 660),
        ("decoder_transfer_flava.svg", bidirectional_mlp_panel("cub_openai_vitl14_to_flava_linear", "OpenAI L/14 → FLAVA"), "Each group compares the two-layer MLP on native, unaligned, Oxford-aligned, and CUB-aligned embeddings.", 1, 660),
        ("unidirectional_mlp_transfer.svg", unidirectional_mlp_panel(), "A target-space MLP evaluated on native and source embeddings before and after alignment.", 1, 660),
    ]
    manifest = {"figures": []}
    for name, panels, caption, columns, panel_width in figures:
        render_figure(
            FIGURE_ROOT / name,
            panels,
            caption,
            columns=columns,
            panel_width=panel_width,
        )
        manifest["figures"].append(
            {
                "file": name,
                "panels": [panel.title for panel in panels],
                "source_root": str(RESULT_ROOT.relative_to(PROJECT_ROOT)),
            }
        )
    (FIGURE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
