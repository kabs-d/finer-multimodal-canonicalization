"""Frozen CUB attribute-decoder transfer with a fixed Oxford rotation."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .alignment import OrthogonalAlignment, fit_orthogonal_alignment
from .datasets import CUB2002011, validate_cub
from .decoders import build_decoder
from .metrics import (
    class_retrieval_top1,
    l2_normalize,
    multilabel_metrics,
    paired_cosine,
    zero_shot_top1,
)
from .models import load_encoder


def read_decoder_config(path: Path | str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run_id",
        "dataset",
        "source_model",
        "target_model",
        "feature_extraction",
        "decoder",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"decoder config is missing keys: {sorted(missing)}")
    if config["dataset"]["name"] != "cub":
        raise ValueError("frozen decoder experiments require dataset.name='cub'")
    alignment = config.get("alignment", {})
    if alignment.get("rotation_fit_dataset") != "oxford":
        raise ValueError("the frozen decoder phase requires an Oxford-fitted rotation")
    if alignment.get("refit_rotation_on_cub") is not False:
        raise ValueError("refitting the rotation on CUB is outside this phase")
    if alignment.get("recompute_means_on_cub_train") is not True:
        raise ValueError("CUB train-set recentering must remain enabled")
    if alignment.get("raw_rotation_ablation") is not False:
        raise ValueError("the raw-Q ablation is intentionally disabled")
    feature_config = config["feature_extraction"]
    if feature_config.get("encoders_frozen") is not True:
        raise ValueError("both encoders must remain frozen")
    decoder_config = config["decoder"]
    if decoder_config.get("architecture") not in {"linear", "mlp"}:
        raise ValueError("decoder architecture must be 'linear' or 'mlp'")
    if decoder_config.get("architecture") == "mlp":
        if decoder_config.get("hidden_dim") != 512:
            raise ValueError("the prespecified MLP requires hidden_dim=512")
        if decoder_config.get("activation") != "gelu":
            raise ValueError("the prespecified MLP requires activation='gelu'")
        dropout = decoder_config.get("dropout")
        if dropout is None or not 0 <= dropout < 1:
            raise ValueError("MLP dropout must be in [0, 1)")
    if decoder_config.get("outputs") != config["dataset"].get("attributes"):
        raise ValueError("decoder outputs must equal the dataset attribute count")
    if not decoder_config.get("seeds"):
        raise ValueError("at least one decoder seed is required")
    return config


def _encoder_audit(encoder) -> dict:
    parameters = list(encoder.model.parameters())
    return {
        "training_mode": bool(encoder.model.training),
        "parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
    }


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _matrix_diagnostics(values: torch.Tensor) -> dict[str, float | int]:
    values = values.double()
    singular_values = torch.linalg.svdvals(values)
    largest = singular_values.max()
    tolerance = max(values.shape) * torch.finfo(values.dtype).eps * largest
    nonzero = singular_values[singular_values > tolerance]
    stable_rank = (
        torch.linalg.matrix_norm(values, ord="fro").square()
        / largest.square().clamp_min(torch.finfo(values.dtype).tiny)
    )
    condition = (
        (nonzero.max() / nonzero.min()).item()
        if nonzero.numel()
        else float("inf")
    )
    return {
        "rows": values.shape[0],
        "columns": values.shape[1],
        "numerical_rank": int(nonzero.numel()),
        "stable_rank": float(stable_rank.item()),
        "condition_number_nonzero": float(condition),
        "largest_singular_value": float(largest.item()),
        "smallest_retained_singular_value": (
            float(nonzero.min().item()) if nonzero.numel() else 0.0
        ),
    }


def materialize_oxford_alignment(
    upstream_embedding_prefix: Path | str,
    output_path: Path | str,
    *,
    source_model: dict,
    target_model: dict,
) -> Path:
    """Serialize the exact Oxford-fitted rotation used by a decoder run."""
    prefix = Path(upstream_embedding_prefix)
    train_path = Path(f"{prefix}_tr_img.pt")
    if not train_path.is_file():
        raise FileNotFoundError(f"missing Oxford image embeddings: {train_path}")
    data = torch.load(train_path, map_location="cpu")
    source = data["i1"].float()
    target = data["i2"].float()
    alignment = fit_orthogonal_alignment(source, target)
    source_centered = source - alignment.source_mean
    target_centered = target - alignment.target_mean
    cross_covariance = source_centered.T @ target_centered
    payload = {
        "schema_version": 1,
        "fit_dataset": "oxford",
        "fit_examples": int(source.shape[0]),
        "dimension": int(source.shape[1]),
        "rotation": alignment.rotation.cpu(),
        "source_mean": alignment.source_mean.cpu(),
        "target_mean": alignment.target_mean.cpu(),
        "source_model": source_model,
        "target_model": target_model,
        "orthogonality_frobenius_error": alignment.orthogonality_error,
        "source_embedding_diagnostics": _matrix_diagnostics(source_centered),
        "target_embedding_diagnostics": _matrix_diagnostics(target_centered),
        "cross_covariance_diagnostics": _matrix_diagnostics(cross_covariance),
        "source_cache": str(train_path.resolve()),
        "source_cache_sha256": _sha256(train_path),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    metadata = {key: value for key, value in payload.items() if not torch.is_tensor(value)}
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _load_or_compute(path: Path, compute: Callable[[], dict]) -> dict:
    if path.is_file():
        return torch.load(path, map_location="cpu")
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(result, temporary)
    temporary.replace(path)
    return result


@torch.inference_mode()
def _extract_features(
    dataset: CUB2002011,
    source_encoder,
    target_encoder,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list,
    )
    source_values, target_values = [], []
    labels, image_ids, attributes, masks = [], [], [], []
    for batch in tqdm(loader, desc=f"CUB {len(dataset)} image embeddings"):
        source_images = torch.stack(
            [source_encoder.preprocess(record["image"]) for record in batch]
        ).to(device)
        target_images = torch.stack(
            [target_encoder.preprocess(record["image"]) for record in batch]
        ).to(device)
        source_values.append(source_encoder.encode_image(source_images).cpu())
        target_values.append(target_encoder.encode_image(target_images).cpu())
        labels.append(
            torch.tensor([record["label"] for record in batch], dtype=torch.long)
        )
        image_ids.append(
            torch.tensor([record["image_id"] for record in batch], dtype=torch.long)
        )
        attributes.append(
            torch.stack([record["attributes"] for record in batch]).float()
        )
        masks.append(
            torch.stack([record["attribute_mask"] for record in batch]).bool()
        )
    return {
        "source": torch.cat(source_values),
        "target": torch.cat(target_values),
        "labels": torch.cat(labels),
        "image_ids": torch.cat(image_ids),
        "attributes": torch.cat(attributes),
        "attribute_mask": torch.cat(masks),
    }


def _class_text_embeddings(
    encoder,
    class_names: list[str],
    prompt: str | None,
) -> torch.Tensor:
    texts = (
        [prompt.format(name) for name in class_names]
        if prompt is not None
        else class_names
    )
    return l2_normalize(encoder.encode_text(texts)).cpu()


def _paper_alignment_metrics(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    aligned_images: torch.Tensor,
    labels: torch.Tensor,
    source_classes: torch.Tensor,
    target_classes: torch.Tensor,
    aligned_classes: torch.Tensor,
) -> dict:
    class_labels = torch.arange(source_classes.shape[0], device=labels.device)
    return {
        "ImageImage": {
            "Baseline": class_retrieval_top1(
                source_images, target_images, labels, labels
            ),
            "Procrustes": class_retrieval_top1(
                aligned_images, target_images, labels, labels
            ),
        },
        "TextText": {
            "Baseline": class_retrieval_top1(
                source_classes, target_classes, class_labels, class_labels
            ),
            "Procrustes": class_retrieval_top1(
                aligned_classes, target_classes, class_labels, class_labels
            ),
        },
        "ImageText": {
            "A_to_A": zero_shot_top1(source_images, source_classes, labels),
            "B_to_B": zero_shot_top1(target_images, target_classes, labels),
            "Aligned_imgA_to_textB": zero_shot_top1(
                aligned_images, target_classes, labels
            ),
            "Aligned_imgA_to_aligned_textA": zero_shot_top1(
                aligned_images, aligned_classes, labels
            ),
            "ImgB_to_aligned_textA": zero_shot_top1(
                target_images, aligned_classes, labels
            ),
        },
        "Cosine": {
            "Image_before": paired_cosine(source_images, target_images),
            "Image_after": paired_cosine(aligned_images, target_images),
            "Text_before": paired_cosine(source_classes, target_classes),
            "Text_after": paired_cosine(aligned_classes, target_classes),
        },
    }


def _species_stratified_split(
    labels: torch.Tensor,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    train_indices, validation_indices = [], []
    for class_index in sorted(np.unique(labels_np)):
        indices = np.flatnonzero(labels_np == class_index)
        rng.shuffle(indices)
        validation_count = max(1, int(round(indices.size * validation_fraction)))
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    return np.asarray(sorted(train_indices)), np.asarray(sorted(validation_indices))


def _attribute_eligibility(
    attributes: torch.Tensor,
    mask: torch.Tensor,
    train_indices: np.ndarray,
    test_attributes: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    min_train_positive: int,
    min_train_negative: int,
    min_test_positive: int,
    min_test_negative: int,
) -> np.ndarray:
    train_labels = attributes[train_indices].bool()
    train_valid = mask[train_indices]
    train_positive = (train_labels & train_valid).sum(dim=0)
    train_negative = ((~train_labels) & train_valid).sum(dim=0)
    test_labels = test_attributes.bool()
    test_positive = (test_labels & test_mask).sum(dim=0)
    test_negative = ((~test_labels) & test_mask).sum(dim=0)
    eligible = (
        (train_positive >= min_train_positive)
        & (train_negative >= min_train_negative)
        & (test_positive >= min_test_positive)
        & (test_negative >= min_test_negative)
    )
    return eligible.numpy()


def _positive_weights(
    attributes: torch.Tensor,
    mask: torch.Tensor,
    indices: np.ndarray,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    labels = attributes[indices].bool()
    valid = mask[indices]
    positives = (labels & valid).sum(dim=0).float()
    negatives = ((~labels) & valid).sum(dim=0).float()
    return (negatives / positives.clamp_min(1)).clamp(minimum, maximum)


@torch.inference_mode()
def _predict(model: nn.Module, features: torch.Tensor, device: torch.device) -> np.ndarray:
    logits = model(features.to(device))
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def _train_linear_decoder(
    features: torch.Tensor,
    attributes: torch.Tensor,
    attribute_mask: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    eligible: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pos_weight_min: float,
    pos_weight_max: float,
) -> tuple[nn.Module, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_decoder(
        {"architecture": "linear"},
        features.shape[1],
        attributes.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positive_weight = _positive_weights(
        attributes,
        attribute_mask,
        train_indices,
        pos_weight_min,
        pos_weight_max,
    ).to(device)
    rng = np.random.default_rng(seed)
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        total_loss, total_observed = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_labels = attributes[indices].to(device)
            batch_mask = attribute_mask[indices].to(device)
            logits = model(batch_features)
            losses = functional.binary_cross_entropy_with_logits(
                logits,
                batch_labels,
                pos_weight=positive_weight,
                reduction="none",
            )
            observed = batch_mask.sum().clamp_min(1)
            loss = (losses * batch_mask).sum() / observed
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float((losses.detach() * batch_mask).sum().item())
            total_observed += int(batch_mask.sum().item())

        model.eval()
        validation_probabilities = _predict(
            model, features[validation_indices], device
        )
        validation_metrics, _ = multilabel_metrics(
            attributes[validation_indices].numpy(),
            attribute_mask[validation_indices].numpy(),
            validation_probabilities,
            eligible=eligible,
        )
        score = float(validation_metrics["macro_map"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_observed, 1),
                "validation_macro_map": score,
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("linear decoder did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_map": best_score,
        "epochs_run": len(history),
        "history": history,
    }


def _aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    conditions = seed_results[0]["conditions"].keys()
    metric_names = ["macro_map", "micro_map", "macro_auroc"]
    output = {}
    for condition in conditions:
        output[condition] = {}
        for metric in metric_names:
            values = np.asarray(
                [
                    result["conditions"][condition][metric]
                    for result in seed_results
                ],
                dtype=np.float64,
            )
            output[condition][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
    native = np.asarray(
        [result["conditions"]["native_target"]["macro_map"] for result in seed_results]
    )
    aligned = np.asarray(
        [result["conditions"]["aligned_source"]["macro_map"] for result in seed_results]
    )
    output["transfer"] = {
        "macro_map_gap_aligned_minus_native": {
            "mean": float((aligned - native).mean()),
            "std": float((aligned - native).std()),
        },
        "macro_map_retention": {
            "mean": float((aligned / np.clip(native, 1e-12, None)).mean()),
            "std": float((aligned / np.clip(native, 1e-12, None)).std()),
        },
    }
    return output


def _bootstrap_gap(
    labels: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    native_probabilities: np.ndarray,
    aligned_probabilities: np.ndarray,
    eligible: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    class_ids = np.asarray(sorted(np.unique(species)))
    class_to_column = {
        class_index: column for column, class_index in enumerate(class_ids)
    }
    species_columns = np.asarray(
        [class_to_column[class_index] for class_index in species], dtype=np.int64
    )
    draw_counts = rng.multinomial(
        class_ids.size,
        np.full(class_ids.size, 1.0 / class_ids.size),
        size=replicates,
    )
    attribute_gaps = np.full(
        (replicates, labels.shape[1]), np.nan, dtype=np.float64
    )
    eligible_indices = np.flatnonzero(eligible)
    for attribute_index in tqdm(
        eligible_indices, desc="species-stratified bootstrap attributes"
    ):
        observed = mask[:, attribute_index]
        observed_labels = labels[observed, attribute_index]
        observed_species = species_columns[observed]
        native_ap = _cluster_bootstrap_ap(
            observed_labels,
            native_probabilities[observed, attribute_index],
            observed_species,
            draw_counts,
        )
        aligned_ap = _cluster_bootstrap_ap(
            observed_labels,
            aligned_probabilities[observed, attribute_index],
            observed_species,
            draw_counts,
        )
        attribute_gaps[:, attribute_index] = aligned_ap - native_ap
    defined_counts = np.sum(~np.isnan(attribute_gaps), axis=1)
    if np.any(defined_counts == 0):
        raise RuntimeError("bootstrap produced a replicate without defined attributes")
    gap_values = np.nanmean(attribute_gaps, axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "sampling_unit": "species",
        "defined_attributes_per_replicate_min": int(defined_counts.min()),
        "defined_attributes_per_replicate_max": int(defined_counts.max()),
        "mean_gap": float(gap_values.mean()),
        "ci95_lower": float(np.percentile(gap_values, 2.5)),
        "ci95_upper": float(np.percentile(gap_values, 97.5)),
    }


def _cluster_bootstrap_ap(
    labels: np.ndarray,
    scores: np.ndarray,
    species_columns: np.ndarray,
    draw_counts: np.ndarray,
    *,
    chunk_size: int = 100,
) -> np.ndarray:
    """AP for cluster-bootstrap replicates without materializing duplicate rows."""
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order].astype(np.float64)
    ordered_scores = scores[order]
    ordered_species = species_columns[order]
    threshold_ends = np.r_[
        np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]),
        labels.size - 1,
    ]
    output = np.full(draw_counts.shape[0], np.nan, dtype=np.float64)
    for start in range(0, draw_counts.shape[0], chunk_size):
        end = min(start + chunk_size, draw_counts.shape[0])
        weights = draw_counts[start:end, ordered_species]
        cumulative_total = np.cumsum(weights, axis=1)
        cumulative_positive = np.cumsum(
            weights * ordered_labels[None, :], axis=1
        )
        predicted_positive = cumulative_total[:, threshold_ends]
        true_positive = cumulative_positive[:, threshold_ends]
        positive_in_group = np.diff(
            np.concatenate(
                [
                    np.zeros((end - start, 1), dtype=np.float64),
                    true_positive,
                ],
                axis=1,
            ),
            axis=1,
        )
        total_positive = true_positive[:, -1]
        total_negative = cumulative_total[:, -1] - total_positive
        denominator = total_positive[:, None] * predicted_positive
        contributions = np.divide(
            positive_in_group * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
        valid = (total_positive > 0) & (total_negative > 0)
        chunk_values = contributions.sum(axis=1)
        output[start:end][valid] = chunk_values[valid]
    return output


def _write_per_attribute_csv(
    path: Path,
    attribute_names: list[str],
    eligible: np.ndarray,
    per_seed_attributes: list[dict[str, list[dict]]],
) -> None:
    conditions = [
        "native_target",
        "aligned_source",
        "unaligned_source",
        "native_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "attribute_index",
            "attribute_name",
            "attribute_family",
            "eligible",
            "observed_test",
            "positives_test",
            "negatives_test",
            *[f"{condition}_ap_mean" for condition in conditions],
            "aligned_minus_native_target_ap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for attribute_index, name in enumerate(attribute_names):
            first = per_seed_attributes[0]["native_target"][attribute_index]
            row = {
                "attribute_index": attribute_index,
                "attribute_name": name,
                "attribute_family": name.split("::", 1)[0],
                "eligible": bool(eligible[attribute_index]),
                "observed_test": first["observed"],
                "positives_test": first["positives"],
                "negatives_test": first["negatives"],
            }
            for condition in conditions:
                values = [
                    seed[condition][attribute_index]["average_precision"]
                    for seed in per_seed_attributes
                    if seed[condition][attribute_index]["average_precision"] is not None
                ]
                row[f"{condition}_ap_mean"] = (
                    float(np.mean(values)) if values else ""
                )
            native = row["native_target_ap_mean"]
            aligned = row["aligned_source_ap_mean"]
            row["aligned_minus_native_target_ap"] = (
                float(aligned - native)
                if isinstance(native, float) and isinstance(aligned, float)
                else ""
            )
            writer.writerow(row)


def run_frozen_decoder(
    config_path: Path | str,
    alignment_path: Path | str,
    data_root: Path | str,
    embedding_root: Path | str,
    prediction_root: Path | str,
    output_root: Path | str,
    model_cache_root: Path | str,
    *,
    device_name: str = "cuda",
    force: bool = False,
) -> Path:
    config = read_decoder_config(config_path)
    dataset_audit = validate_cub(data_root)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(output_root) / config["run_id"]
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        print(f"complete decoder result already exists: {summary_path}")
        return output_dir

    alignment_payload = torch.load(alignment_path, map_location="cpu")
    if alignment_payload["source_model"] != config["source_model"]:
        raise ValueError("alignment source model does not match decoder config")
    if alignment_payload["target_model"] != config["target_model"]:
        raise ValueError("alignment target model does not match decoder config")

    train_dataset = CUB2002011(data_root, "train")
    test_dataset = CUB2002011(data_root, "test")
    source_encoder = load_encoder(
        config["source_model"], device, Path(model_cache_root)
    )
    target_encoder = load_encoder(
        config["target_model"], device, Path(model_cache_root)
    )
    encoder_audit_before = {
        "source": _encoder_audit(source_encoder),
        "target": _encoder_audit(target_encoder),
    }
    if any(
        record["training_mode"] or record["trainable_parameters"]
        for record in encoder_audit_before.values()
    ):
        raise RuntimeError("encoder freeze audit failed before feature extraction")
    feature_config = config["feature_extraction"]
    cache_dir = Path(embedding_root) / config["run_id"]
    train_features = _load_or_compute(
        cache_dir / "cub_train.pt",
        lambda: _extract_features(
            train_dataset,
            source_encoder,
            target_encoder,
            device,
            batch_size=feature_config["batch_size"],
            num_workers=feature_config["num_workers"],
        ),
    )
    test_features = _load_or_compute(
        cache_dir / "cub_test.pt",
        lambda: _extract_features(
            test_dataset,
            source_encoder,
            target_encoder,
            device,
            batch_size=feature_config["batch_size"],
            num_workers=feature_config["num_workers"],
        ),
    )
    expected_dimension = alignment_payload["dimension"]
    if train_features["source"].shape[1] != expected_dimension:
        raise RuntimeError("CUB source features do not match Oxford Q dimension")
    if train_features["target"].shape[1] != expected_dimension:
        raise RuntimeError("CUB target features do not match Oxford Q dimension")

    source_mean = train_features["source"].mean(dim=0, keepdim=True).to(device)
    target_mean = train_features["target"].mean(dim=0, keepdim=True).to(device)
    fixed_alignment = OrthogonalAlignment(
        alignment_payload["rotation"].to(device),
        source_mean,
        target_mean,
    )
    source_test = l2_normalize(test_features["source"].to(device))
    target_test = l2_normalize(test_features["target"].to(device))
    aligned_test = l2_normalize(
        fixed_alignment.transform(source_test, centered=True)
    )
    labels_test = test_features["labels"].to(device)

    prompt = config["dataset"]["prompt"]
    source_raw_text = _class_text_embeddings(
        source_encoder, train_dataset.class_names, None
    )
    target_raw_text = _class_text_embeddings(
        target_encoder, train_dataset.class_names, None
    )
    class_counts = torch.bincount(
        train_features["labels"], minlength=len(train_dataset.class_names)
    ).float()
    text_source_mean = (
        source_raw_text * class_counts[:, None]
    ).sum(dim=0, keepdim=True) / class_counts.sum()
    text_target_mean = (
        target_raw_text * class_counts[:, None]
    ).sum(dim=0, keepdim=True) / class_counts.sum()
    source_classes = _class_text_embeddings(
        source_encoder, train_dataset.class_names, prompt
    ).to(device)
    target_classes = _class_text_embeddings(
        target_encoder, train_dataset.class_names, prompt
    ).to(device)
    aligned_classes = l2_normalize(
        (source_classes - text_source_mean.to(device))
        @ fixed_alignment.rotation
        + text_target_mean.to(device)
    )
    alignment_metrics = _paper_alignment_metrics(
        source_test,
        target_test,
        aligned_test,
        labels_test,
        source_classes,
        target_classes,
        aligned_classes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment_metrics.json").write_text(
        json.dumps(alignment_metrics, indent=2) + "\n", encoding="utf-8"
    )

    decoder_config = config["decoder"]
    train_indices, validation_indices = _species_stratified_split(
        train_features["labels"],
        validation_fraction=decoder_config["validation_fraction"],
        seed=decoder_config["split_seed"],
    )
    eligible = _attribute_eligibility(
        train_features["attributes"],
        train_features["attribute_mask"],
        train_indices,
        test_features["attributes"],
        test_features["attribute_mask"],
        min_train_positive=decoder_config["min_train_positive"],
        min_train_negative=decoder_config["min_train_negative"],
        min_test_positive=decoder_config["min_test_positive"],
        min_test_negative=decoder_config["min_test_negative"],
    )
    if not eligible.any():
        raise RuntimeError("no CUB attributes meet the configured support thresholds")

    source_train = l2_normalize(train_features["source"])
    target_train = l2_normalize(train_features["target"])
    source_test_cpu = source_test.cpu()
    target_test_cpu = target_test.cpu()
    aligned_test_cpu = aligned_test.cpu()
    test_labels_np = test_features["attributes"].numpy().astype(np.int64)
    test_mask_np = test_features["attribute_mask"].numpy()
    seed_results = []
    per_seed_attributes = []
    native_probabilities_all = []
    aligned_probabilities_all = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(prediction_root) / config["run_id"]
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for seed in decoder_config["seeds"]:
        target_decoder, target_training = _train_linear_decoder(
            target_train,
            train_features["attributes"],
            train_features["attribute_mask"],
            train_indices,
            validation_indices,
            eligible,
            seed=seed,
            device=device,
            learning_rate=decoder_config["learning_rate"],
            weight_decay=decoder_config["weight_decay"],
            batch_size=decoder_config["batch_size"],
            max_epochs=decoder_config["max_epochs"],
            patience=decoder_config["patience"],
            pos_weight_min=decoder_config["pos_weight_min"],
            pos_weight_max=decoder_config["pos_weight_max"],
        )
        source_decoder, source_training = _train_linear_decoder(
            source_train,
            train_features["attributes"],
            train_features["attribute_mask"],
            train_indices,
            validation_indices,
            eligible,
            seed=seed,
            device=device,
            learning_rate=decoder_config["learning_rate"],
            weight_decay=decoder_config["weight_decay"],
            batch_size=decoder_config["batch_size"],
            max_epochs=decoder_config["max_epochs"],
            patience=decoder_config["patience"],
            pos_weight_min=decoder_config["pos_weight_min"],
            pos_weight_max=decoder_config["pos_weight_max"],
        )
        probabilities = {
            "native_target": _predict(target_decoder, target_test_cpu, device),
            "aligned_source": _predict(target_decoder, aligned_test_cpu, device),
            "unaligned_source": _predict(target_decoder, source_test_cpu, device),
            "native_source": _predict(source_decoder, source_test_cpu, device),
        }
        condition_metrics = {}
        condition_attributes = {}
        for condition, values in probabilities.items():
            metrics, per_attribute = multilabel_metrics(
                test_labels_np,
                test_mask_np,
                values,
                eligible=eligible,
            )
            condition_metrics[condition] = metrics
            condition_attributes[condition] = per_attribute
        seed_result = {
            "seed": seed,
            "target_decoder_training": target_training,
            "source_decoder_training": source_training,
            "conditions": condition_metrics,
        }
        seed_results.append(seed_result)
        per_seed_attributes.append(condition_attributes)
        native_probabilities_all.append(probabilities["native_target"])
        aligned_probabilities_all.append(probabilities["aligned_source"])
        torch.save(
            {
                "target_decoder": {
                    key: value.detach().cpu()
                    for key, value in target_decoder.state_dict().items()
                },
                "source_decoder": {
                    key: value.detach().cpu()
                    for key, value in source_decoder.state_dict().items()
                },
                "seed": seed,
                "eligible_attributes": torch.from_numpy(eligible),
            },
            checkpoint_dir / f"seed_{seed}.pt",
        )
        np.savez_compressed(
            prediction_dir / f"seed_{seed}.npz",
            image_ids=test_features["image_ids"].numpy(),
            labels=test_labels_np,
            mask=test_mask_np,
            species=test_features["labels"].numpy(),
            **probabilities,
        )

    native_mean_probabilities = np.mean(native_probabilities_all, axis=0)
    aligned_mean_probabilities = np.mean(aligned_probabilities_all, axis=0)
    bootstrap = _bootstrap_gap(
        test_labels_np,
        test_mask_np,
        test_features["labels"].numpy(),
        native_mean_probabilities,
        aligned_mean_probabilities,
        eligible,
        replicates=decoder_config["bootstrap_replicates"],
        seed=decoder_config["bootstrap_seed"],
    )
    aggregate = _aggregate_seed_metrics(seed_results)
    aggregate["transfer"]["paired_species_bootstrap"] = bootstrap
    summary = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "alignment_source": "fixed Oxford rotation with CUB training means",
        "alignment_metrics": alignment_metrics,
        "eligible_attributes": int(eligible.sum()),
        "decoder_split": {
            "training_examples": int(train_indices.size),
            "validation_examples": int(validation_indices.size),
            "official_test_examples": len(test_dataset),
            "split_seed": decoder_config["split_seed"],
        },
        "aggregate": aggregate,
        "seeds": seed_results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_per_attribute_csv(
        output_dir / "per_attribute.csv",
        train_dataset.attribute_names,
        eligible,
        per_seed_attributes,
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "alignment_path": str(Path(alignment_path).resolve()),
        "alignment_sha256": _sha256(Path(alignment_path)),
        "data_root": str(Path(data_root).resolve()),
        "embedding_root": str(Path(embedding_root).resolve()),
        "prediction_root": str(Path(prediction_root).resolve()),
        "model_cache_root": str(Path(model_cache_root).resolve()),
        "dataset_audit": dataset_audit,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "encoder_training": False,
        "encoder_audit_before": encoder_audit_before,
        "encoder_audit_after": {
            "source": _encoder_audit(source_encoder),
            "target": _encoder_audit(target_encoder),
        },
        "rotation_refit_on_cub": False,
        "raw_rotation_ablation": False,
        "cub_recentering": True,
        "cub_train_feature_diagnostics": {
            "source": _matrix_diagnostics(
                train_features["source"] - train_features["source"].mean(0)
            ),
            "target": _matrix_diagnostics(
                train_features["target"] - train_features["target"].mean(0)
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2))
    return output_dir
