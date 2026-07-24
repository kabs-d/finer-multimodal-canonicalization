"""Metrics used in the paper baseline."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def _binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with tied scores handled as one threshold."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("average precision requires at least one positive")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_positive = np.cumsum(sorted_labels)
    cumulative_total = np.arange(1, labels.size + 1)
    threshold_ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
        labels.size - 1,
    ]
    true_positive = cumulative_positive[threshold_ends]
    predicted_positive = cumulative_total[threshold_ends]
    positive_in_group = np.diff(np.r_[0, true_positive])
    precision = true_positive / predicted_positive
    return float(np.sum((positive_in_group / positives) * precision))


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via average ranks, including exact tie handling."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires positive and negative observations")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def l2_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / (values.norm(dim=-1, keepdim=True) + eps)


@torch.inference_mode()
def paired_cosine(source: torch.Tensor, target: torch.Tensor) -> float:
    if source.shape != target.shape:
        raise ValueError("paired cosine requires equal shapes")
    return (
        (l2_normalize(source) * l2_normalize(target))
        .sum(dim=-1)
        .mean()
        .item()
    )


@torch.inference_mode()
def class_retrieval_top1(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    query_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> float:
    similarities = l2_normalize(queries) @ l2_normalize(candidates).T
    nearest = similarities.argmax(dim=-1)
    predictions = candidate_labels.to(nearest.device)[nearest]
    return (
        predictions == query_labels.to(predictions.device)
    ).float().mean().item()


@torch.inference_mode()
def zero_shot_top1(
    images: torch.Tensor,
    class_text: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    similarities = l2_normalize(images) @ l2_normalize(class_text).T
    predictions = similarities.argmax(dim=-1)
    return (predictions == labels.to(predictions.device)).float().mean().item()


def aggregate_nested(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute population mean/std for nested numeric dictionaries."""
    if not records:
        raise ValueError("at least one record is required")
    means: dict[str, Any] = {}
    standard_deviations: dict[str, Any] = {}
    for key, first_value in records[0].items():
        if isinstance(first_value, Mapping):
            nested_mean, nested_std = aggregate_nested([record[key] for record in records])
            means[key] = nested_mean
            standard_deviations[key] = nested_std
        else:
            values = np.asarray([record[key] for record in records], dtype=np.float64)
            means[key] = float(values.mean())
            standard_deviations[key] = float(values.std())
    return means, standard_deviations


def multilabel_metrics(
    labels: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int | None]]]:
    """Masked AP/AUROC metrics for binary attributes."""
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != mask.shape or labels.shape != probabilities.shape:
        raise ValueError("labels, mask, and probabilities must have equal shapes")
    if eligible is None:
        eligible = np.ones(labels.shape[1], dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (labels.shape[1],):
        raise ValueError("eligible must contain one flag per attribute")

    per_attribute: list[dict[str, float | int | None]] = []
    average_precisions: list[float] = []
    aurocs: list[float] = []
    for attribute_index in range(labels.shape[1]):
        observed = mask[:, attribute_index]
        observed_labels = labels[observed, attribute_index]
        observed_probabilities = probabilities[observed, attribute_index]
        positives = int(observed_labels.sum())
        negatives = int(observed_labels.size - positives)
        ap: float | None = None
        auroc: float | None = None
        is_defined = positives > 0 and negatives > 0
        if eligible[attribute_index] and is_defined:
            ap = _binary_average_precision(
                observed_labels, observed_probabilities
            )
            auroc = _binary_auroc(observed_labels, observed_probabilities)
            average_precisions.append(ap)
            aurocs.append(auroc)
        per_attribute.append(
            {
                "attribute_index": attribute_index,
                "eligible": bool(eligible[attribute_index] and is_defined),
                "observed": int(observed_labels.size),
                "positives": positives,
                "negatives": negatives,
                "average_precision": ap,
                "auroc": auroc,
            }
        )

    flattened = mask & eligible[None, :]
    flat_labels = labels[flattened]
    flat_probabilities = probabilities[flattened]
    if not average_precisions or flat_labels.size == 0:
        raise ValueError("no eligible attributes with defined labels")
    micro_ap = _binary_average_precision(flat_labels, flat_probabilities)
    return (
        {
            "eligible_attributes": len(average_precisions),
            "macro_map": float(np.mean(average_precisions)),
            "micro_map": micro_ap,
            "macro_auroc": float(np.mean(aurocs)),
        },
        per_attribute,
    )
