"""Audit answerable CUB species-attribute prompts for text-to-image retrieval."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .datasets import CUB2002011, validate_cub


DEFAULT_THRESHOLDS = (1, 2, 3, 5)
DEFAULT_MANIFEST_MIN_POSITIVE = 3
DEFAULT_MANIFEST_MIN_NEGATIVE = 3


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean_token(value: str) -> str:
    value = value.replace("_-_", " ")
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = value.replace("(", "").replace(")", "")
    return " ".join(value.split())


def clean_attribute_phrase(raw_attribute_name: str) -> str:
    """Convert a raw CUB attribute string into a simple text-prompt phrase."""
    if "::" not in raw_attribute_name:
        return _clean_token(raw_attribute_name)
    category, value = raw_attribute_name.split("::", 1)
    category = category.removeprefix("has_")
    clean_value = _clean_token(value)

    if category == "bill_shape":
        return f"{clean_value} bill"
    if category == "bill_length":
        if value == "about_the_same_as_head":
            return "bill about the same length as head"
        if value == "longer_than_head":
            return "bill longer than head"
        if value == "shorter_than_head":
            return "bill shorter than head"
        return f"bill {clean_value}"
    if category == "tail_shape":
        return clean_value if "tail" in clean_value else f"{clean_value} tail"
    if category == "wing_shape":
        return clean_value if "wing" in clean_value else f"{clean_value} wing"
    if category == "shape":
        return f"{clean_value} body shape"
    if category == "size":
        return f"{clean_value.split()[0]} bird"

    body_part = category
    if body_part.endswith("_color"):
        body_part = body_part[: -len("_color")]
    elif body_part.endswith("_pattern"):
        body_part = body_part[: -len("_pattern")]
    body_part = _clean_token(body_part)
    if body_part:
        return f"{clean_value} {body_part}"
    return clean_value


def prompt_for(species_name: str, attribute_phrase: str) -> str:
    return f"a photo of a {species_name} with {attribute_phrase}."


def split_arrays(dataset: CUB2002011) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_indices = np.asarray([record.image_id - 1 for record in dataset.records])
    labels = np.asarray([record.class_index for record in dataset.records], dtype=np.int64)
    attributes = dataset.attributes[image_indices].numpy().astype(bool)
    mask = dataset.attribute_mask[image_indices].numpy().astype(bool)
    return labels, attributes, mask


def train_prevalence_and_rare(
    train_attributes: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    visible = train_mask.astype(bool).sum(axis=0)
    positives = (train_attributes.astype(bool) & train_mask.astype(bool)).sum(axis=0)
    prevalence = np.divide(
        positives,
        np.maximum(visible, 1),
        out=np.zeros_like(visible, dtype=np.float64),
        where=visible > 0,
    )
    cutoff = float(np.quantile(prevalence[visible > 0], 0.25, method="linear"))
    rare = (visible > 0) & (prevalence <= cutoff)
    return prevalence, rare, cutoff


def threshold_flags(
    positive_count: int,
    negative_count: int,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
) -> dict[str, bool]:
    return {
        f"valid_ge{threshold}_pos_ge{threshold}_neg": (
            positive_count >= threshold and negative_count >= threshold
        )
        for threshold in thresholds
    }


def build_audit_rows(
    class_names: list[str],
    attribute_names: list[str],
    test_labels: np.ndarray,
    test_attributes: np.ndarray,
    test_mask: np.ndarray,
    train_prevalence: np.ndarray,
    rare_attributes: np.ndarray,
    *,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    rows: list[dict] = []
    attribute_phrases = [clean_attribute_phrase(name) for name in attribute_names]
    thresholds = tuple(sorted(int(threshold) for threshold in thresholds))
    for species_index, species_name in enumerate(class_names):
        species_mask = test_labels == species_index
        for attribute_index, raw_attribute_name in enumerate(attribute_names):
            visible = species_mask & test_mask[:, attribute_index]
            positives = visible & test_attributes[:, attribute_index]
            positive_count = int(positives.sum())
            visible_count = int(visible.sum())
            negative_count = int(visible_count - positive_count)
            row = {
                "species_index": species_index,
                "species_id": species_index + 1,
                "species_name": species_name,
                "attribute_index": attribute_index,
                "attribute_id": attribute_index + 1,
                "raw_attribute_name": raw_attribute_name,
                "attribute_phrase": attribute_phrases[attribute_index],
                "prompt": prompt_for(species_name, attribute_phrases[attribute_index]),
                "test_visible": visible_count,
                "test_positive": positive_count,
                "test_negative": negative_count,
                "train_positive_prevalence": float(train_prevalence[attribute_index]),
                "rare_bottom_quartile": bool(rare_attributes[attribute_index]),
            }
            row.update(threshold_flags(positive_count, negative_count, thresholds))
            rows.append(row)
    return rows


def summarize_attributes(rows: list[dict], attribute_count: int) -> list[dict]:
    summary = []
    for attribute_index in range(attribute_count):
        group = [row for row in rows if row["attribute_index"] == attribute_index]
        first = group[0]
        summary.append(
            {
                "attribute_index": attribute_index,
                "attribute_id": attribute_index + 1,
                "raw_attribute_name": first["raw_attribute_name"],
                "attribute_phrase": first["attribute_phrase"],
                "train_positive_prevalence": first["train_positive_prevalence"],
                "rare_bottom_quartile": first["rare_bottom_quartile"],
                "species_valid_ge1": sum(row["valid_ge1_pos_ge1_neg"] for row in group),
                "species_valid_ge2": sum(row["valid_ge2_pos_ge2_neg"] for row in group),
                "species_valid_ge3": sum(row["valid_ge3_pos_ge3_neg"] for row in group),
                "species_valid_ge5": sum(row["valid_ge5_pos_ge5_neg"] for row in group),
                "test_visible_total": sum(row["test_visible"] for row in group),
                "test_positive_total": sum(row["test_positive"] for row in group),
                "test_negative_total": sum(row["test_negative"] for row in group),
            }
        )
    return summary


def summarize_species(rows: list[dict], class_names: list[str]) -> list[dict]:
    summary = []
    for species_index, species_name in enumerate(class_names):
        group = [row for row in rows if row["species_index"] == species_index]
        summary.append(
            {
                "species_index": species_index,
                "species_id": species_index + 1,
                "species_name": species_name,
                "valid_ge1_prompts": sum(row["valid_ge1_pos_ge1_neg"] for row in group),
                "valid_ge2_prompts": sum(row["valid_ge2_pos_ge2_neg"] for row in group),
                "valid_ge3_prompts": sum(row["valid_ge3_pos_ge3_neg"] for row in group),
                "valid_ge5_prompts": sum(row["valid_ge5_pos_ge5_neg"] for row in group),
                "visible_cells": sum(row["test_visible"] for row in group),
                "positive_cells": sum(row["test_positive"] for row in group),
                "negative_cells": sum(row["test_negative"] for row in group),
            }
        )
    return summary


def aggregate_summary(
    rows: list[dict],
    attribute_summary: list[dict],
    species_summary: list[dict],
    *,
    train_examples: int,
    test_examples: int,
    rare_cutoff: float,
    manifest_min_positive: int,
    manifest_min_negative: int,
) -> dict:
    total = len(rows)
    valid_ge1 = sum(row["valid_ge1_pos_ge1_neg"] for row in rows)
    valid_ge2 = sum(row["valid_ge2_pos_ge2_neg"] for row in rows)
    valid_ge3 = sum(row["valid_ge3_pos_ge3_neg"] for row in rows)
    valid_ge5 = sum(row["valid_ge5_pos_ge5_neg"] for row in rows)
    prompt_manifest_count = sum(
        row["test_positive"] >= manifest_min_positive
        and row["test_negative"] >= manifest_min_negative
        for row in rows
    )
    sparse_attributes = sorted(
        attribute_summary,
        key=lambda row: (row["species_valid_ge3"], row["test_positive_total"]),
    )[:10]
    sparse_species = sorted(
        species_summary,
        key=lambda row: (row["valid_ge3_prompts"], row["positive_cells"]),
    )[:10]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CUB_200_2011",
        "train_examples": train_examples,
        "test_examples": test_examples,
        "species": len(species_summary),
        "attributes": len(attribute_summary),
        "possible_species_attribute_prompts": total,
        "valid_groups": {
            "ge1_positive_ge1_negative": valid_ge1,
            "ge2_positive_ge2_negative": valid_ge2,
            "ge3_positive_ge3_negative": valid_ge3,
            "ge5_positive_ge5_negative": valid_ge5,
        },
        "valid_group_fraction": {
            "ge1_positive_ge1_negative": valid_ge1 / total,
            "ge2_positive_ge2_negative": valid_ge2 / total,
            "ge3_positive_ge3_negative": valid_ge3 / total,
            "ge5_positive_ge5_negative": valid_ge5 / total,
        },
        "recommended_prompt_filter": {
            "min_test_positive": manifest_min_positive,
            "min_test_negative": manifest_min_negative,
            "prompt_count": prompt_manifest_count,
        },
        "rare_attribute_policy": "bottom quartile by official CUB train visible-positive prevalence",
        "rare_prevalence_cutoff": rare_cutoff,
        "rare_attributes": sum(row["rare_bottom_quartile"] for row in attribute_summary),
        "lowest_coverage_attributes_by_ge3": sparse_attributes,
        "lowest_coverage_species_by_ge3": sparse_species,
    }


def run_attribute_prompt_audit(
    data_root: Path | str,
    output_root: Path | str,
    *,
    manifest_min_positive: int = DEFAULT_MANIFEST_MIN_POSITIVE,
    manifest_min_negative: int = DEFAULT_MANIFEST_MIN_NEGATIVE,
    force: bool = False,
) -> Path:
    data_root = Path(data_root)
    output_dir = Path(output_root) / "audit"
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        return output_dir

    validate_cub(data_root)
    train = CUB2002011(data_root, "train")
    test = CUB2002011(data_root, "test")
    train_labels, train_attributes, train_mask = split_arrays(train)
    test_labels, test_attributes, test_mask = split_arrays(test)
    del train_labels

    prevalence, rare, rare_cutoff = train_prevalence_and_rare(
        train_attributes,
        train_mask,
    )
    rows = build_audit_rows(
        test.class_names,
        test.attribute_names,
        test_labels,
        test_attributes,
        test_mask,
        prevalence,
        rare,
    )
    attribute_summary = summarize_attributes(rows, len(test.attribute_names))
    species_summary = summarize_species(rows, test.class_names)
    prompt_manifest = [
        row
        for row in rows
        if row["test_positive"] >= manifest_min_positive
        and row["test_negative"] >= manifest_min_negative
    ]
    summary = aggregate_summary(
        rows,
        attribute_summary,
        species_summary,
        train_examples=len(train),
        test_examples=len(test),
        rare_cutoff=rare_cutoff,
        manifest_min_positive=manifest_min_positive,
        manifest_min_negative=manifest_min_negative,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "species_attribute_audit.csv", rows)
    _write_csv(output_dir / "attribute_summary.csv", attribute_summary)
    _write_csv(output_dir / "species_summary.csv", species_summary)
    _write_csv(output_dir / "prompt_manifest.csv", prompt_manifest)
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir
