"""Fine-grained CUB retrieval under canonical alignment."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .alignment import OrthogonalAlignment
from .decoder_experiment import _sha256, read_decoder_config
from .metrics import l2_normalize


DEFAULT_K_VALUES = (1, 5, 10)
DEFAULT_RANDOM_SEED = 2026


def _required_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


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


def _load_embedding_cache(embedding_root: Path, run_id: str) -> tuple[dict, dict, Path, Path]:
    run_dir = embedding_root / run_id
    train_path = run_dir / "cub_train.pt"
    test_path = run_dir / "cub_test.pt"
    _required_file(train_path, "CUB train embedding cache")
    _required_file(test_path, "CUB test embedding cache")
    return (
        torch.load(train_path, map_location="cpu"),
        torch.load(test_path, map_location="cpu"),
        train_path,
        test_path,
    )


def _load_alignment(path: Path, config: dict) -> dict:
    _required_file(path, "alignment artifact")
    payload = torch.load(path, map_location="cpu")
    if payload["source_model"] != config["source_model"]:
        raise ValueError(f"alignment source model does not match config: {path}")
    if payload["target_model"] != config["target_model"]:
        raise ValueError(f"alignment target model does not match config: {path}")
    return payload


def _aligned_source(
    source: torch.Tensor,
    rotation: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
) -> torch.Tensor:
    alignment = OrthogonalAlignment(
        rotation.float(),
        source_mean.float(),
        target_mean.float(),
    )
    return l2_normalize(alignment.transform(source.float(), centered=True))


def evaluable_attributes(
    train_attributes: np.ndarray,
    train_mask: np.ndarray,
    test_attributes: np.ndarray,
    test_mask: np.ndarray,
) -> np.ndarray:
    """Attributes with enough visible labels to support train-defined rarity and test metrics."""
    train_labels = train_attributes.astype(bool)
    test_labels = test_attributes.astype(bool)
    train_valid = train_mask.astype(bool)
    test_valid = test_mask.astype(bool)
    train_positive = (train_labels & train_valid).sum(axis=0)
    train_negative = ((~train_labels) & train_valid).sum(axis=0)
    test_positive = (test_labels & test_valid).sum(axis=0)
    test_negative = ((~test_labels) & test_valid).sum(axis=0)
    return (
        (train_positive > 0)
        & (train_negative > 0)
        & (test_positive > 0)
        & (test_negative > 0)
    )


def rare_attributes_bottom_quartile(
    train_attributes: np.ndarray,
    train_mask: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select rare attributes by bottom-quartile visible-positive prevalence."""
    train_labels = train_attributes.astype(bool)
    train_valid = train_mask.astype(bool)
    visible = train_valid.sum(axis=0)
    positives = (train_labels & train_valid).sum(axis=0)
    prevalence = np.divide(
        positives,
        np.maximum(visible, 1),
        out=np.zeros_like(visible, dtype=np.float64),
        where=visible > 0,
    )
    eligible_prevalence = prevalence[eligible]
    if eligible_prevalence.size == 0:
        raise ValueError("no eligible attributes available for rare-attribute selection")
    cutoff = float(np.quantile(eligible_prevalence, 0.25, method="linear"))
    rare = eligible & (prevalence <= cutoff)
    if not rare.any():
        rare[np.flatnonzero(eligible)[np.argmin(eligible_prevalence)]] = True
    return rare, prevalence


def same_species_candidate_indices(
    labels: np.ndarray,
    query_index: int,
) -> np.ndarray:
    labels = np.asarray(labels)
    matches = np.flatnonzero(labels == labels[query_index])
    return matches[matches != query_index]


def _attribute_overlap_for_candidates(
    query_index: int,
    candidate_indices: np.ndarray,
    attributes: np.ndarray,
    mask: np.ndarray,
    eligible: np.ndarray,
) -> float | None:
    query_positive = (
        (attributes[query_index].astype(bool))
        & (mask[query_index].astype(bool))
        & eligible
    )
    denominator = int(query_positive.sum())
    if denominator == 0 or candidate_indices.size == 0:
        return None
    candidate_positive = (
        attributes[candidate_indices].astype(bool)
        & mask[candidate_indices].astype(bool)
    )
    shared = candidate_positive[:, query_positive].sum(axis=1) / denominator
    return float(shared.mean())


def _rare_recall_for_candidates(
    query_index: int,
    candidate_indices: np.ndarray,
    attributes: np.ndarray,
    mask: np.ndarray,
    rare: np.ndarray,
) -> float | None:
    query_rare_positive = (
        attributes[query_index].astype(bool)
        & mask[query_index].astype(bool)
        & rare
    )
    denominator = int(query_rare_positive.sum())
    if denominator == 0 or candidate_indices.size == 0:
        return None
    candidate_positive = (
        attributes[candidate_indices].astype(bool)
        & mask[candidate_indices].astype(bool)
    )
    recovered = candidate_positive[:, query_rare_positive].any(axis=0)
    return float(recovered.mean())


def _topk_indices(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    labels: np.ndarray,
    k_values: Iterable[int],
) -> dict[int, list[np.ndarray]]:
    max_k = max(k_values)
    query_embeddings = l2_normalize(query_embeddings.float())
    candidate_embeddings = l2_normalize(candidate_embeddings.float())
    output: dict[int, list[np.ndarray]] = {int(k): [] for k in k_values}
    for query_index in range(query_embeddings.shape[0]):
        candidates = same_species_candidate_indices(labels, query_index)
        if candidates.size == 0:
            for k in k_values:
                output[int(k)].append(candidates)
            continue
        similarities = (
            query_embeddings[query_index : query_index + 1]
            @ candidate_embeddings[candidates].T
        ).squeeze(0)
        take = min(max_k, candidates.size)
        order = torch.topk(similarities, k=take).indices.cpu().numpy()
        ranked = candidates[order]
        for k in k_values:
            output[int(k)].append(ranked[: min(int(k), ranked.size)])
    return output


def _random_indices(
    labels: np.ndarray,
    k_values: Iterable[int],
    *,
    seed: int,
) -> dict[int, list[np.ndarray]]:
    max_k = max(k_values)
    rng = np.random.default_rng(seed)
    output: dict[int, list[np.ndarray]] = {int(k): [] for k in k_values}
    for query_index in range(labels.shape[0]):
        candidates = same_species_candidate_indices(labels, query_index)
        if candidates.size > 0:
            shuffled = candidates.copy()
            rng.shuffle(shuffled)
            ranked = shuffled[: min(max_k, shuffled.size)]
        else:
            ranked = candidates
        for k in k_values:
            output[int(k)].append(ranked[: min(int(k), ranked.size)])
    return output


def evaluate_retrieval_rankings(
    rankings: dict[str, dict[int, list[np.ndarray]]],
    attributes: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    rare: np.ndarray,
) -> tuple[dict, list[dict], list[dict]]:
    """Compute per-query, aggregate, and per-attribute retrieval metrics."""
    per_query: list[dict] = []
    per_attribute: list[dict] = []
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    attributes_bool = attributes.astype(bool)
    mask_bool = mask.astype(bool)

    for condition, by_k in rankings.items():
        summary[condition] = {}
        for k, ranked_lists in sorted(by_k.items()):
            overlaps: list[float] = []
            rare_recalls: list[float] = []
            for query_index, candidate_indices in enumerate(ranked_lists):
                overlap = _attribute_overlap_for_candidates(
                    query_index,
                    candidate_indices,
                    attributes,
                    mask,
                    eligible,
                )
                rare_recall = _rare_recall_for_candidates(
                    query_index,
                    candidate_indices,
                    attributes,
                    mask,
                    rare,
                )
                if overlap is not None:
                    overlaps.append(overlap)
                if rare_recall is not None:
                    rare_recalls.append(rare_recall)
                per_query.append(
                    {
                        "condition": condition,
                        "k": int(k),
                        "query_index": int(query_index),
                        "image_id": None,
                        "species_label": int(labels[query_index]),
                        "candidate_count": int(candidate_indices.size),
                        "attribute_overlap": overlap,
                        "rare_attribute_recall": rare_recall,
                    }
                )

            key = f"k{k}"
            summary[condition][key] = {
                "same_species_attribute_overlap": float(np.mean(overlaps))
                if overlaps
                else None,
                "same_species_attribute_overlap_queries": int(len(overlaps)),
                "rare_attribute_recall": float(np.mean(rare_recalls))
                if rare_recalls
                else None,
                "rare_attribute_recall_queries": int(len(rare_recalls)),
            }

            for attribute_index in np.flatnonzero(rare):
                query_mask = attributes_bool[:, attribute_index] & mask_bool[:, attribute_index]
                values = []
                for query_index in np.flatnonzero(query_mask):
                    candidate_indices = ranked_lists[int(query_index)]
                    if candidate_indices.size == 0:
                        continue
                    recovered = bool(
                        (
                            attributes_bool[candidate_indices, attribute_index]
                            & mask_bool[candidate_indices, attribute_index]
                        ).any()
                    )
                    values.append(float(recovered))
                per_attribute.append(
                    {
                        "condition": condition,
                        "k": int(k),
                        "attribute_index": int(attribute_index),
                        "rare": True,
                        "query_positives": int(query_mask.sum()),
                        "recall_at_k": float(np.mean(values)) if values else None,
                    }
                )
    return summary, per_query, per_attribute


def _attach_image_ids(per_query: list[dict], image_ids: np.ndarray) -> None:
    for row in per_query:
        row["image_id"] = int(image_ids[row["query_index"]])


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


def run_fine_grained_retrieval(
    config_path: Path | str,
    embedding_root: Path | str,
    output_root: Path | str,
    alignment_root: Path | str,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    force: bool = False,
) -> Path:
    config_path = Path(config_path)
    embedding_root = Path(embedding_root)
    output_root = Path(output_root)
    alignment_root = Path(alignment_root)
    config = read_decoder_config(config_path)
    base_run_id = config["run_id"]
    run_id = f"{base_run_id}_fine_grained_retrieval"
    result_dir = output_root / run_id
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not force:
        return result_dir

    k_values = tuple(sorted({int(k) for k in k_values}))
    if not k_values or min(k_values) < 1:
        raise ValueError("k values must be positive integers")

    train_features, test_features, train_path, test_path = _load_embedding_cache(
        embedding_root,
        base_run_id,
    )
    source_test = test_features["source"].float()
    target_test = test_features["target"].float()
    source_train = train_features["source"].float()
    target_train = train_features["target"].float()

    oxford_path = alignment_root / _oxford_alignment_name(config)
    cub_train_q_run_id = base_run_id.replace("_linear", "_cub_train_q_linear")
    cub_train_q_path = alignment_root / f"{cub_train_q_run_id}.pt"
    oxford = _load_alignment(oxford_path, config)
    cub_train_q = _load_alignment(cub_train_q_path, config)

    oxford_aligned = _aligned_source(
        source_test,
        oxford["rotation"],
        source_train.mean(dim=0, keepdim=True),
        target_train.mean(dim=0, keepdim=True),
    )
    cub_train_aligned = _aligned_source(
        source_test,
        cub_train_q["rotation"],
        cub_train_q["source_mean"],
        cub_train_q["target_mean"],
    )

    labels = test_features["labels"].cpu().numpy().astype(np.int64)
    attributes = test_features["attributes"].cpu().numpy().astype(np.int64)
    mask = test_features["attribute_mask"].cpu().numpy().astype(bool)
    train_attributes = train_features["attributes"].cpu().numpy().astype(np.int64)
    train_mask = train_features["attribute_mask"].cpu().numpy().astype(bool)
    eligible = evaluable_attributes(train_attributes, train_mask, attributes, mask)
    rare, prevalence = rare_attributes_bottom_quartile(
        train_attributes,
        train_mask,
        eligible,
    )

    rankings = {
        "native_target": _topk_indices(target_test, target_test, labels, k_values),
        "oxford_aligned_source": _topk_indices(
            oxford_aligned,
            target_test,
            labels,
            k_values,
        ),
        "cub_train_aligned_source": _topk_indices(
            cub_train_aligned,
            target_test,
            labels,
            k_values,
        ),
        "unaligned_source": _topk_indices(source_test, target_test, labels, k_values),
        "random_same_species": _random_indices(
            labels,
            k_values,
            seed=random_seed,
        ),
    }
    summary, per_query, per_attribute = evaluate_retrieval_rankings(
        rankings,
        attributes,
        mask,
        labels,
        eligible,
        rare,
    )
    _attach_image_ids(
        per_query,
        test_features["image_ids"].cpu().numpy().astype(np.int64),
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "base_run_id": base_run_id,
        "config_path": str(config_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_pair": {
            "source_model": config["source_model"],
            "target_model": config["target_model"],
        },
        "k_values": list(k_values),
        "conditions": list(rankings.keys()),
        "candidate_policy": (
            "Same official CUB test species as query, excluding the query image."
        ),
        "metric_policy": {
            "same_species_attribute_overlap": (
                "For each query and top-k set, average over retrieved candidates "
                "the fraction of query visible-positive evaluable attributes also "
                "visible-positive in that candidate."
            ),
            "rare_attribute_recall": (
                "For each query with visible-positive rare attributes, fraction "
                "of those rare positives recovered by at least one top-k candidate."
            ),
        },
        "rare_attribute_policy": "Bottom quartile by official CUB train visible-positive prevalence among evaluable attributes.",
        "eligible_attributes": int(eligible.sum()),
        "rare_attributes": int(rare.sum()),
        "rare_prevalence_cutoff": float(prevalence[rare].max()),
        "test_queries": int(labels.shape[0]),
        "random_seed": int(random_seed),
    }
    output = {
        **metadata,
        "aggregate": summary,
        "rare_attribute_indices": np.flatnonzero(rare).astype(int).tolist(),
    }
    summary_path.write_text(
        json.dumps(_jsonable(output), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(result_dir / "per_query.csv", per_query)
    _write_csv(result_dir / "per_attribute.csv", per_attribute)
    manifest = {
        **metadata,
        "artifacts": {
            "summary": "summary.json",
            "per_query": "per_query.csv",
            "per_attribute": "per_attribute.csv",
        },
        "inputs": {
            "train_embeddings": str(train_path.resolve()),
            "train_embeddings_sha256": _sha256(train_path),
            "test_embeddings": str(test_path.resolve()),
            "test_embeddings_sha256": _sha256(test_path),
            "oxford_alignment": str(oxford_path.resolve()),
            "oxford_alignment_sha256": _sha256(oxford_path),
            "cub_train_q_alignment": str(cub_train_q_path.resolve()),
            "cub_train_q_alignment_sha256": _sha256(cub_train_q_path),
        },
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    return result_dir


def _oxford_alignment_name(config: dict) -> str:
    source = config["source_model"]
    target = config["target_model"]
    source_key = f"{source['pretrained']}_{source['name']}".lower().replace("-", "")
    target_kind = target["kind"]
    if target_kind == "open_clip":
        target_key = f"{target['pretrained']}_{target['name']}".lower().replace("-", "")
    else:
        target_key = target["name"].lower()
    if source_key == "openai_vitb32" and target_key == "laion400m_e31_vitb32":
        return "oxford_openai_vitb32_to_laion_vitb32.pt"
    if source_key == "openai_vitl14" and target_key == "flava":
        return "oxford_openai_vitl14_to_flava.pt"
    raise ValueError(
        "do not know Oxford alignment artifact name for "
        f"{config['source_model']} -> {config['target_model']}"
    )
