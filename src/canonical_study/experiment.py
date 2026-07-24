"""End-to-end frozen-model canonicalization baseline."""

import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .alignment import apply_shared_rotation, fit_orthogonal_alignment
from .datasets import OxfordPets, validate_oxford
from .metrics import (
    aggregate_nested,
    class_retrieval_top1,
    l2_normalize,
    paired_cosine,
    zero_shot_top1,
)
from .models import load_encoder


def read_config(path: Path | str) -> dict:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run_id",
        "dataset",
        "source_model",
        "target_model",
        "seeds",
        "batch_size",
        "num_workers",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    if config["dataset"]["name"] != "oxford":
        raise ValueError("this reproduction currently supports Oxford-IIIT Pet only")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_or_compute(path: Path, compute: Callable[[], dict]) -> dict:
    if path.is_file():
        return torch.load(path, map_location="cpu")
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".partial")
    torch.save(result, temporary_path)
    temporary_path.replace(path)
    return result


def _load_upstream_embeddings(prefix: Path) -> tuple[dict, dict, dict]:
    """Read author-code caches and translate their compact field names."""
    paths = {
        "fit_images": Path(f"{prefix}_tr_img.pt"),
        "evaluation_images": Path(f"{prefix}_te_img.pt"),
        "fit_text": Path(f"{prefix}_tr_txt.pt"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing upstream embedding caches: {missing}")
    upstream_fit_images = torch.load(paths["fit_images"], map_location="cpu")
    upstream_evaluation_images = torch.load(
        paths["evaluation_images"], map_location="cpu"
    )
    upstream_fit_text = torch.load(paths["fit_text"], map_location="cpu")
    fit_images = {
        "source": upstream_fit_images["i1"],
        "target": upstream_fit_images["i2"],
        "labels": upstream_fit_images["l"],
    }
    evaluation_images = {
        "source": upstream_evaluation_images["i1"],
        "target": upstream_evaluation_images["i2"],
        "labels": upstream_evaluation_images["l"],
    }
    fit_text = {
        "source": upstream_fit_text["t1"],
        "target": upstream_fit_text["t2"],
    }
    return fit_images, evaluation_images, fit_text


@torch.inference_mode()
def _image_embeddings(loader, source, target, device: torch.device) -> dict:
    source_values, target_values, labels = [], [], []
    for batch in tqdm(loader, desc="image embeddings"):
        source_images = torch.stack(
            [source.preprocess(record["image"]) for record in batch]
        ).to(device)
        target_images = torch.stack(
            [target.preprocess(record["image"]) for record in batch]
        ).to(device)
        source_values.append(source.encode_image(source_images).cpu())
        target_values.append(target.encode_image(target_images).cpu())
        labels.append(
            torch.tensor([record["label"] for record in batch], dtype=torch.long)
        )
    return {
        "source": torch.cat(source_values),
        "target": torch.cat(target_values),
        "labels": torch.cat(labels),
    }


@torch.inference_mode()
def _text_embeddings(loader, source, target) -> dict:
    source_values, target_values = [], []
    for batch in tqdm(loader, desc="instance text embeddings"):
        texts = [record["text"] for record in batch]
        source_values.append(source.encode_text(texts).cpu())
        target_values.append(target.encode_text(texts).cpu())
    return {
        "source": torch.cat(source_values),
        "target": torch.cat(target_values),
    }


@torch.inference_mode()
def _class_embeddings(
    class_names: list[str],
    prompt: str,
    encoder,
) -> torch.Tensor:
    texts = [prompt.format(name.replace("_", " ")) for name in class_names]
    return l2_normalize(encoder.encode_text(texts)).cpu()


def _metrics(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    source_classes: torch.Tensor,
    target_classes: torch.Tensor,
    labels: torch.Tensor,
    aligned_images: torch.Tensor,
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


def run_baseline(
    config_path: Path | str,
    data_root: Path | str,
    embedding_root: Path | str,
    output_root: Path | str,
    model_cache_root: Path | str,
    device_name: str = "cuda",
    force: bool = False,
    upstream_embedding_prefix: Path | str | None = None,
) -> Path:
    config = read_config(config_path)
    validate_oxford(data_root)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_output = Path(output_root) / config["run_id"]
    centered_path = run_output / "centered" / "metrics.json"
    uncentered_path = run_output / "uncentered" / "metrics.json"
    if centered_path.is_file() and uncentered_path.is_file() and not force:
        print(f"complete results already exist under {run_output}")
        return run_output

    fit_dataset = OxfordPets(data_root, config["dataset"]["fit_split"])
    evaluation_dataset = OxfordPets(
        data_root, config["dataset"]["evaluation_split"]
    )
    collate = lambda records: records
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate,
    )

    cache_root = Path(model_cache_root)
    source = load_encoder(config["source_model"], device, cache_root)
    target = load_encoder(config["target_model"], device, cache_root)
    embedding_dir = Path(embedding_root) / config["run_id"]

    if upstream_embedding_prefix is not None:
        fit_images, evaluation_images, fit_text = _load_upstream_embeddings(
            Path(upstream_embedding_prefix)
        )
    else:
        fit_images = _load_or_compute(
            embedding_dir / "fit_images.pt",
            lambda: _image_embeddings(fit_loader, source, target, device),
        )
        evaluation_images = _load_or_compute(
            embedding_dir / "evaluation_images.pt",
            lambda: _image_embeddings(evaluation_loader, source, target, device),
        )
        fit_text = _load_or_compute(
            embedding_dir / "fit_instance_text.pt",
            lambda: _text_embeddings(fit_loader, source, target),
        )
    source_classes = _class_embeddings(
        fit_dataset.class_names, config["dataset"]["prompt"], source
    ).to(device)
    target_classes = _class_embeddings(
        fit_dataset.class_names, config["dataset"]["prompt"], target
    ).to(device)

    fit_source = fit_images["source"].to(device)
    fit_target = fit_images["target"].to(device)
    source_images = evaluation_images["source"].to(device)
    target_images = evaluation_images["target"].to(device)
    labels = evaluation_images["labels"].to(device)
    fit_source_text = fit_text["source"].to(device)
    fit_target_text = fit_text["target"].to(device)

    alignment = fit_orthogonal_alignment(fit_source, fit_target)
    text_source_mean = fit_source_text.mean(dim=0, keepdim=True)
    text_target_mean = fit_target_text.mean(dim=0, keepdim=True)

    centered_images = alignment.transform(source_images, centered=True)
    centered_classes = apply_shared_rotation(
        source_classes,
        alignment.rotation,
        text_source_mean,
        text_target_mean,
        centered=True,
    )
    uncentered_images = alignment.transform(source_images, centered=False)
    uncentered_classes = apply_shared_rotation(
        source_classes,
        alignment.rotation,
        text_source_mean,
        text_target_mean,
        centered=False,
    )
    centered_metrics = _metrics(
        source_images,
        target_images,
        source_classes,
        target_classes,
        labels,
        centered_images,
        centered_classes,
    )
    uncentered_metrics = _metrics(
        source_images,
        target_images,
        source_classes,
        target_classes,
        labels,
        uncentered_images,
        uncentered_classes,
    )

    def repeated(metrics: dict) -> dict:
        seed_records = {
            f"seed_{seed}": metrics for seed in config["seeds"]
        }
        mean, std = aggregate_nested(list(seed_records.values()))
        return {**seed_records, "mean": mean, "std": std}

    for path, metrics in (
        (centered_path, centered_metrics),
        (uncentered_path, uncentered_metrics),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(repeated(metrics), indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "embedding_root": str(Path(embedding_root).resolve()),
        "upstream_embedding_prefix": (
            str(Path(upstream_embedding_prefix).resolve())
            if upstream_embedding_prefix is not None
            else None
        ),
        "model_cache_root": str(cache_root.resolve()),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "orthogonality_frobenius_error": alignment.orthogonality_error,
        "fit_examples": len(fit_dataset),
        "evaluation_examples": len(evaluation_dataset),
        "classes": fit_dataset.num_classes,
        "seed_note": (
            "The released baseline has no stochastic anchor sampling or shuffling; "
            "the requested seeds therefore repeat the same deterministic evaluation."
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (run_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(centered_metrics, indent=2))
    return run_output
