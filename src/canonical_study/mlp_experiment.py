"""Target-only MLP decoder training from existing frozen feature caches."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .alignment import OrthogonalAlignment
from .decoder_experiment import (
    _attribute_eligibility,
    _positive_weights,
    _species_stratified_split,
    read_decoder_config,
)
from .decoders import build_decoder, decoder_probabilities
from .metrics import l2_normalize, multilabel_metrics


def _train_target_decoder(
    features: torch.Tensor,
    attributes: torch.Tensor,
    attribute_mask: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    eligible: np.ndarray,
    decoder_config: dict,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_decoder(
        decoder_config,
        features.shape[1],
        attributes.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=decoder_config["learning_rate"],
        weight_decay=decoder_config["weight_decay"],
    )
    positive_weight = _positive_weights(
        attributes,
        attribute_mask,
        train_indices,
        decoder_config["pos_weight_min"],
        decoder_config["pos_weight_max"],
    ).to(device)
    rng = np.random.default_rng(seed)
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, decoder_config["max_epochs"] + 1):
        model.train()
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        total_loss, total_observed = 0.0, 0
        for start in range(0, len(shuffled), decoder_config["batch_size"]):
            indices = shuffled[
                start : start + decoder_config["batch_size"]
            ]
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

        validation_probabilities = decoder_probabilities(
            model, features[validation_indices], device
        ).numpy()
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
        if epochs_without_improvement >= decoder_config["patience"]:
            break
    if best_state is None:
        raise RuntimeError("MLP decoder did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_map": best_score,
        "epochs_run": len(history),
        "history": history,
    }


def run_cached_mlp_decoder(
    config_path: Path | str,
    alignment_path: Path | str,
    embedding_root: Path | str,
    prediction_root: Path | str,
    output_root: Path | str,
    *,
    device_name: str = "cuda",
    force: bool = False,
) -> Path:
    config = read_decoder_config(config_path)
    decoder_config = config["decoder"]
    if decoder_config["architecture"] != "mlp":
        raise ValueError("the cached MLP runner requires architecture='mlp'")
    feature_run_id = config["feature_extraction"].get("reuse_run_id")
    if not feature_run_id:
        raise ValueError("MLP config must declare feature_extraction.reuse_run_id")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(output_root) / config["run_id"]
    completion_path = output_dir / "mlp_summary.json"
    if completion_path.is_file() and not force:
        print(f"complete MLP result already exists: {completion_path}")
        return output_dir

    feature_dir = Path(embedding_root) / feature_run_id
    train_path = feature_dir / "cub_train.pt"
    test_path = feature_dir / "cub_test.pt"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"required frozen feature caches are missing under {feature_dir}"
        )
    train_features = torch.load(train_path, map_location="cpu")
    test_features = torch.load(test_path, map_location="cpu")
    alignment_payload = torch.load(alignment_path, map_location="cpu")
    if alignment_payload["source_model"] != config["source_model"]:
        raise ValueError("alignment source model does not match MLP config")
    if alignment_payload["target_model"] != config["target_model"]:
        raise ValueError("alignment target model does not match MLP config")
    dimension = int(alignment_payload["dimension"])
    if (
        train_features["source"].shape[1] != dimension
        or train_features["target"].shape[1] != dimension
    ):
        raise RuntimeError("cached features do not match Oxford Q dimension")

    source_mean = train_features["source"].mean(0, keepdim=True).to(device)
    target_mean = train_features["target"].mean(0, keepdim=True).to(device)
    alignment = OrthogonalAlignment(
        alignment_payload["rotation"].to(device),
        source_mean,
        target_mean,
    )
    source_test = l2_normalize(test_features["source"].to(device))
    target_test = l2_normalize(test_features["target"].to(device))
    aligned_test = l2_normalize(alignment.transform(source_test)).cpu()
    source_test = source_test.cpu()
    target_test = target_test.cpu()
    target_train = l2_normalize(train_features["target"])

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
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(prediction_root) / config["run_id"]
    prediction_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in decoder_config["seeds"]:
        model, training = _train_target_decoder(
            target_train,
            train_features["attributes"],
            train_features["attribute_mask"],
            train_indices,
            validation_indices,
            eligible,
            decoder_config,
            seed=seed,
            device=device,
        )
        probabilities = {
            "native_target": decoder_probabilities(
                model, target_test, device
            ).numpy().astype(np.float32),
            "aligned_source": decoder_probabilities(
                model, aligned_test, device
            ).numpy().astype(np.float32),
            "unaligned_source": decoder_probabilities(
                model, source_test, device
            ).numpy().astype(np.float32),
        }
        torch.save(
            {
                "target_decoder": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "decoder_config": decoder_config,
                "input_dim": int(target_train.shape[1]),
                "output_dim": int(train_features["attributes"].shape[1]),
                "seed": seed,
                "eligible_attributes": torch.from_numpy(eligible),
            },
            checkpoint_dir / f"seed_{seed}.pt",
        )
        np.savez_compressed(
            prediction_dir / f"seed_{seed}.npz",
            image_ids=test_features["image_ids"].numpy(),
            labels=test_features["attributes"].numpy().astype(np.int64),
            mask=test_features["attribute_mask"].numpy(),
            species=test_features["labels"].numpy(),
            **probabilities,
        )
        seed_results.append(training)
        print(
            f"seed {seed}: best epoch {training['best_epoch']}, "
            f"validation macro mAP {training['best_validation_macro_map']:.6f}"
        )

    summary = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "decoder": decoder_config,
        "feature_run_id": feature_run_id,
        "eligible_attributes": int(eligible.sum()),
        "training_examples": int(train_indices.size),
        "validation_examples": int(validation_indices.size),
        "test_examples": int(test_features["labels"].shape[0]),
        "seeds": seed_results,
    }
    completion_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "alignment_path": str(Path(alignment_path).resolve()),
        "feature_cache": {
            "run_id": feature_run_id,
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
        },
        "device": str(device),
        "encoder_inference_repeated": False,
        "encoder_training": False,
        "rotation_refit_on_cub": False,
        "cub_recentering": True,
        "trained_on_aligned_source": False,
        "selected_on_aligned_source": False,
        "paper_classification_repeated": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir
