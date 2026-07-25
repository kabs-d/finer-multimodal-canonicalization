"""CUB-train-fitted Q control using cached frozen features and decoders."""

import json
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .alignment import OrthogonalAlignment, fit_orthogonal_alignment
from .attribute_analysis import (
    _aggregate_seed_results as _aggregate_attribute_seed_results,
    _species_prevalence_scores,
    per_bird_recovery,
    select_f1_thresholds,
    within_species_ranking_accuracy,
)
from .datasets import CUB2002011, validate_cub
from .decoder_experiment import (
    _aggregate_seed_metrics,
    _attribute_eligibility,
    _bootstrap_gap,
    _class_text_embeddings,
    _matrix_diagnostics,
    _paper_alignment_metrics,
    _sha256,
    _species_stratified_split,
    _write_per_attribute_csv,
    read_decoder_config,
)
from .decoders import build_decoder, decoder_probabilities
from .metrics import l2_normalize, multilabel_metrics
from .models import load_encoder


def _required_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def _save_cub_train_alignment(
    config: dict,
    train_features: dict,
    test_features: dict,
    output_path: Path,
) -> dict:
    train_image_ids = train_features["image_ids"].long()
    test_image_ids = test_features["image_ids"].long()
    overlap = set(train_image_ids.tolist()) & set(test_image_ids.tolist())
    if overlap:
        raise RuntimeError(
            "CUB train/test leakage detected while fitting Q: "
            f"{len(overlap)} overlapping image IDs"
        )

    source = train_features["source"].float()
    target = train_features["target"].float()
    alignment = fit_orthogonal_alignment(source, target)
    source_centered = source - alignment.source_mean
    target_centered = target - alignment.target_mean
    cross_covariance = source_centered.T @ target_centered
    payload = {
        "schema_version": 1,
        "fit_dataset": "cub",
        "fit_split": "official_train",
        "fit_examples": int(source.shape[0]),
        "heldout_split": "official_test",
        "heldout_examples": int(test_features["source"].shape[0]),
        "train_test_image_id_overlap": 0,
        "dimension": int(source.shape[1]),
        "rotation": alignment.rotation.cpu(),
        "source_mean": alignment.source_mean.cpu(),
        "target_mean": alignment.target_mean.cpu(),
        "source_model": config["source_model"],
        "target_model": config["target_model"],
        "orthogonality_frobenius_error": alignment.orthogonality_error,
        "source_embedding_diagnostics": _matrix_diagnostics(source_centered),
        "target_embedding_diagnostics": _matrix_diagnostics(target_centered),
        "cross_covariance_diagnostics": _matrix_diagnostics(cross_covariance),
        "fit_image_ids": train_image_ids.cpu(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    metadata = {
        key: value for key, value in payload.items() if not torch.is_tensor(value)
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _attribute_interpretability(
    config: dict,
    train_features: dict,
    test_features: dict,
    result_dir: Path,
    checkpoint_dir: Path,
    prediction_root: Path,
    eligible: np.ndarray,
    validation_indices: np.ndarray,
    fit_indices: np.ndarray,
) -> dict:
    decoder_config = config["decoder"]
    run_id = config["run_id"]
    train_labels = train_features["attributes"].numpy().astype(np.int64)
    train_mask = train_features["attribute_mask"].numpy().astype(bool)
    train_species = train_features["labels"].numpy().astype(np.int64)
    test_labels = test_features["attributes"].numpy().astype(np.int64)
    test_mask = test_features["attribute_mask"].numpy().astype(bool)
    test_species = test_features["labels"].numpy().astype(np.int64)
    target_train = l2_normalize(train_features["target"]).float()

    seed_results = []
    for seed in decoder_config["seeds"]:
        checkpoint = torch.load(
            checkpoint_dir / f"seed_{seed}.pt",
            map_location="cpu",
        )
        model = build_decoder(
            config["decoder"],
            target_train.shape[1],
            train_features["attributes"].shape[1],
        )
        model.load_state_dict(checkpoint["target_decoder"])
        validation_scores = decoder_probabilities(
            model,
            target_train[validation_indices],
            torch.device("cpu"),
        ).numpy()
        thresholds = select_f1_thresholds(
            train_labels[validation_indices],
            train_mask[validation_indices],
            validation_scores,
            eligible,
        )
        with np.load(prediction_root / run_id / f"seed_{seed}.npz") as predictions:
            conditions = {}
            for condition in [
                "native_target",
                "aligned_source",
                "unaligned_source",
            ]:
                scores = predictions[condition]
                conditions[condition] = {
                    "per_bird_recovery": per_bird_recovery(
                        test_labels,
                        test_mask,
                        scores,
                        thresholds,
                        eligible,
                    ),
                    "within_species": within_species_ranking_accuracy(
                        test_labels,
                        test_mask,
                        scores,
                        test_species,
                        eligible,
                    ),
                }
        seed_results.append({"seed": seed, "conditions": conditions})

    validation_species_scores = _species_prevalence_scores(
        train_labels,
        train_mask,
        train_species,
        fit_indices,
        train_species[validation_indices],
    )
    species_thresholds = select_f1_thresholds(
        train_labels[validation_indices],
        train_mask[validation_indices],
        validation_species_scores,
        eligible,
    )
    test_species_scores = _species_prevalence_scores(
        train_labels,
        train_mask,
        train_species,
        fit_indices,
        test_species,
    )
    species_only = {
        "uses_ground_truth_species": True,
        "per_bird_recovery": per_bird_recovery(
            test_labels,
            test_mask,
            test_species_scores,
            species_thresholds,
            eligible,
        ),
        "within_species": within_species_ranking_accuracy(
            test_labels,
            test_mask,
            test_species_scores,
            test_species,
            eligible,
        ),
    }
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "feature_run_id": config["feature_extraction"]["reuse_run_id"],
        "alignment_source": "CUB official train image Procrustes rotation",
        "eligible_attributes": int(eligible.sum()),
        "test_birds": int(test_labels.shape[0]),
        "threshold_protocol": (
            "Per-attribute F1 thresholds selected on target-native validation "
            "scores and frozen for native, aligned, and unaligned test inputs."
        ),
        "per_bird_count_policy": (
            "Counts use visible eligible labels only and exclude true negatives."
        ),
        "within_species_policy": (
            "Equal-weight macro average over attribute-species groups containing "
            "at least one visible positive and negative; ties score 0.5."
        ),
        "aggregate": _aggregate_attribute_seed_results(seed_results),
        "species_only": species_only,
        "seeds": seed_results,
    }
    destination = result_dir / "attribute_interpretability.json"
    destination.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return output


def run_cub_train_q_control(
    config_path: Path | str,
    data_root: Path | str,
    embedding_root: Path | str,
    prediction_root: Path | str,
    output_root: Path | str,
    alignment_root: Path | str,
    model_cache_root: Path | str,
    *,
    device_name: str = "cuda",
    force: bool = False,
) -> Path:
    config = read_decoder_config(config_path)
    if config["alignment"].get("rotation_fit_dataset") != "cub_train":
        raise ValueError("CUB-train-Q control requires rotation_fit_dataset='cub_train'")
    if config["decoder"]["architecture"] != "linear":
        raise ValueError("CUB-train-Q control reuses the linear decoder only")
    base_run_id = config["feature_extraction"].get("reuse_run_id")
    if not base_run_id:
        raise ValueError("feature_extraction.reuse_run_id is required")

    dataset_audit = validate_cub(data_root)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(output_root) / config["run_id"]
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        print(f"complete CUB-train-Q result already exists: {summary_path}")
        return output_dir

    feature_dir = Path(embedding_root) / base_run_id
    train_path = feature_dir / "cub_train.pt"
    test_path = feature_dir / "cub_test.pt"
    _required_file(train_path, "cached CUB train features")
    _required_file(test_path, "cached CUB test features")
    train_features = torch.load(train_path, map_location="cpu")
    test_features = torch.load(test_path, map_location="cpu")

    base_result_dir = Path(output_root) / base_run_id
    base_checkpoint_dir = base_result_dir / "checkpoints"
    for seed in config["decoder"]["seeds"]:
        _required_file(
            base_checkpoint_dir / f"seed_{seed}.pt",
            f"base linear decoder checkpoint for seed {seed}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = Path(alignment_root) / f"{config['run_id']}.pt"
    alignment_payload = _save_cub_train_alignment(
        config,
        train_features,
        test_features,
        alignment_path,
    )
    alignment = OrthogonalAlignment(
        alignment_payload["rotation"].to(device),
        alignment_payload["source_mean"].to(device),
        alignment_payload["target_mean"].to(device),
    )

    source_test = l2_normalize(test_features["source"].to(device))
    target_test = l2_normalize(test_features["target"].to(device))
    aligned_test = l2_normalize(alignment.transform(source_test, centered=True))
    labels_test = test_features["labels"].to(device)

    train_dataset = CUB2002011(data_root, "train")
    source_encoder = load_encoder(
        config["source_model"],
        device,
        Path(model_cache_root),
    )
    target_encoder = load_encoder(
        config["target_model"],
        device,
        Path(model_cache_root),
    )
    prompt = config["dataset"]["prompt"]
    source_raw_text = _class_text_embeddings(
        source_encoder,
        train_dataset.class_names,
        None,
    )
    target_raw_text = _class_text_embeddings(
        target_encoder,
        train_dataset.class_names,
        None,
    )
    class_counts = torch.bincount(
        train_features["labels"],
        minlength=len(train_dataset.class_names),
    ).float()
    text_source_mean = (
        source_raw_text * class_counts[:, None]
    ).sum(dim=0, keepdim=True) / class_counts.sum()
    text_target_mean = (
        target_raw_text * class_counts[:, None]
    ).sum(dim=0, keepdim=True) / class_counts.sum()
    source_classes = _class_text_embeddings(
        source_encoder,
        train_dataset.class_names,
        prompt,
    ).to(device)
    target_classes = _class_text_embeddings(
        target_encoder,
        train_dataset.class_names,
        prompt,
    ).to(device)
    aligned_classes = l2_normalize(
        (source_classes - text_source_mean.to(device))
        @ alignment.rotation
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
    (output_dir / "alignment_metrics.json").write_text(
        json.dumps(alignment_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    del source_encoder, target_encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    decoder_config = config["decoder"]
    fit_indices, validation_indices = _species_stratified_split(
        train_features["labels"],
        validation_fraction=decoder_config["validation_fraction"],
        seed=decoder_config["split_seed"],
    )
    eligible = _attribute_eligibility(
        train_features["attributes"],
        train_features["attribute_mask"],
        fit_indices,
        test_features["attributes"],
        test_features["attribute_mask"],
        min_train_positive=decoder_config["min_train_positive"],
        min_train_negative=decoder_config["min_train_negative"],
        min_test_positive=decoder_config["min_test_positive"],
        min_test_negative=decoder_config["min_test_negative"],
    )
    if not eligible.any():
        raise RuntimeError("no CUB attributes meet the configured support thresholds")

    source_test_cpu = source_test.cpu()
    target_test_cpu = target_test.cpu()
    aligned_test_cpu = aligned_test.cpu()
    test_labels_np = test_features["attributes"].numpy().astype(np.int64)
    test_mask_np = test_features["attribute_mask"].numpy()
    prediction_dir = Path(prediction_root) / config["run_id"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    per_seed_attributes = []
    native_probabilities_all = []
    aligned_probabilities_all = []

    for seed in decoder_config["seeds"]:
        checkpoint = torch.load(
            base_checkpoint_dir / f"seed_{seed}.pt",
            map_location="cpu",
        )
        target_decoder = build_decoder(
            config["decoder"],
            target_test_cpu.shape[1],
            train_features["attributes"].shape[1],
        ).to(device)
        source_decoder = build_decoder(
            {"architecture": "linear"},
            source_test_cpu.shape[1],
            train_features["attributes"].shape[1],
        ).to(device)
        target_decoder.load_state_dict(checkpoint["target_decoder"])
        source_decoder.load_state_dict(checkpoint["source_decoder"])
        probabilities = {
            "native_target": decoder_probabilities(
                target_decoder,
                target_test_cpu,
                device,
            ).numpy(),
            "aligned_source": decoder_probabilities(
                target_decoder,
                aligned_test_cpu,
                device,
            ).numpy(),
            "unaligned_source": decoder_probabilities(
                target_decoder,
                source_test_cpu,
                device,
            ).numpy(),
            "native_source": decoder_probabilities(
                source_decoder,
                source_test_cpu,
                device,
            ).numpy(),
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
        seed_results.append(
            {
                "seed": seed,
                "reused_checkpoint": str(
                    (base_checkpoint_dir / f"seed_{seed}.pt").resolve()
                ),
                "conditions": condition_metrics,
            }
        )
        per_seed_attributes.append(condition_attributes)
        native_probabilities_all.append(probabilities["native_target"])
        aligned_probabilities_all.append(probabilities["aligned_source"])
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
        "base_run_id": base_run_id,
        "alignment_source": "CUB official train image Procrustes rotation",
        "alignment_metrics": alignment_metrics,
        "eligible_attributes": int(eligible.sum()),
        "decoder_split": {
            "training_examples": int(fit_indices.size),
            "validation_examples": int(validation_indices.size),
            "official_test_examples": int(test_features["source"].shape[0]),
            "split_seed": decoder_config["split_seed"],
        },
        "aggregate": aggregate,
        "seeds": seed_results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_per_attribute_csv(
        output_dir / "per_attribute.csv",
        train_dataset.attribute_names,
        eligible,
        per_seed_attributes,
    )
    _attribute_interpretability(
        config,
        train_features,
        test_features,
        output_dir,
        base_checkpoint_dir,
        Path(prediction_root),
        eligible,
        validation_indices,
        fit_indices,
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "alignment_path": str(alignment_path.resolve()),
        "alignment_sha256": _sha256(alignment_path),
        "data_root": str(Path(data_root).resolve()),
        "embedding_root": str(Path(embedding_root).resolve()),
        "feature_run_id": base_run_id,
        "prediction_root": str(Path(prediction_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "base_checkpoint_dir": str(base_checkpoint_dir.resolve()),
        "model_cache_root": str(Path(model_cache_root).resolve()),
        "dataset_audit": dataset_audit,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "encoder_image_inference": False,
        "decoder_training": False,
        "rotation_refit_on_cub": True,
        "raw_rotation_ablation": False,
        "cub_recentering": True,
        "cub_train_q_fit": {
            "examples": int(train_features["source"].shape[0]),
            "fit_split": "official_train",
            "heldout_split": "official_test",
            "train_test_image_id_overlap": 0,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], indent=2))
    return output_dir
