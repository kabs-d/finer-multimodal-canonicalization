"""Global attribute-text CUB retrieval under canonical alignment."""

from __future__ import annotations

import csv
import gc
import importlib.machinery
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .attribute_prompt_audit import clean_attribute_phrase
from .alignment import OrthogonalAlignment
from .decoder_experiment import _sha256, read_decoder_config
from .fine_grained_retrieval import _load_alignment, _load_embedding_cache, _oxford_alignment_name
from .metrics import l2_normalize
from .models import load_encoder


DEFAULT_K_VALUES = (1, 5, 10)
CLIP_READABLE_ATTRIBUTE_INDICES = [
    1, 2, 3, 7,
    9, 10, 14, 15, 20, 21, 22,
    54, 55, 56, 57,
    59, 63, 64, 69, 70,
    73, 74, 75, 76, 77, 78,
    96, 97, 100, 101, 102, 103, 104,
    106, 110, 111, 115, 116, 117, 118,
    121, 125, 126, 130, 131, 132, 133,
    149, 150, 151,
    198, 202, 203, 207, 208, 209, 210,
    212, 213, 214, 215, 216,
    217, 218, 220,
    236, 237, 238, 239,
    240, 241, 242, 243,
    244, 245, 246, 247,
    278, 279, 283, 284, 288, 289, 290, 291,
    294, 298, 299, 304, 305, 306,
    308, 309, 310, 311,
]


def attribute_only_prompt(attribute_phrase: str) -> str:
    return f"a photo of a bird with {attribute_phrase}."


def _install_torchvision_text_only_shim() -> None:
    try:
        import torchvision  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    if "torchvision" in sys.modules:
        return

    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    functional = types.ModuleType("torchvision.transforms.functional")
    ops = types.ModuleType("torchvision.ops")
    misc = types.ModuleType("torchvision.ops.misc")
    for module in [torchvision, transforms, functional, ops, misc]:
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, loader=None)

    class FrozenBatchNorm2d(nn.BatchNorm2d):
        def __init__(self, num_features: int, eps: float = 1e-5):
            super().__init__(num_features, eps=eps, affine=True, track_running_stats=True)
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    class _UnavailableTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, image):
            raise RuntimeError("torchvision is unavailable; this runner only encodes text")

    class Compose(list):
        def __call__(self, image):
            for transform in self:
                image = transform(image)
            return image

    class InterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    misc.FrozenBatchNorm2d = FrozenBatchNorm2d
    transforms.Compose = Compose
    transforms.Normalize = _UnavailableTransform
    transforms.RandomResizedCrop = _UnavailableTransform
    transforms.ToTensor = _UnavailableTransform
    transforms.Resize = _UnavailableTransform
    transforms.CenterCrop = _UnavailableTransform
    transforms.ColorJitter = _UnavailableTransform
    transforms.Grayscale = _UnavailableTransform
    transforms.InterpolationMode = InterpolationMode
    transforms.functional = functional
    ops.misc = misc
    torchvision.transforms = transforms
    torchvision.ops = ops
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = functional
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.ops.misc"] = misc


@torch.inference_mode()
def _encode_open_clip_texts(
    spec: dict,
    prompts: list[str],
    *,
    model_cache_root: Path,
    device_name: str,
    batch_size: int,
) -> torch.Tensor:
    _install_torchvision_text_only_shim()
    import open_clip

    standard_cache = Path.home() / ".open_clip"
    open_clip_cache = Path(
        os.environ.get(
            "CANONICAL_STUDY_OPENCLIP_CACHE",
            standard_cache if standard_cache.is_dir() else model_cache_root / "open_clip",
        )
    )
    open_clip_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(open_clip_cache))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(open_clip_cache))
    device = torch.device(device_name)
    model = open_clip.create_model(
        spec["name"],
        pretrained=spec["pretrained"],
        cache_dir=str(open_clip_cache),
    ).to(device).eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(spec["name"])
    batches = []
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[start : start + batch_size]).to(device)
        batches.append(model.encode_text(tokens).detach().cpu())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return l2_normalize(torch.cat(batches, dim=0).float())


@torch.inference_mode()
def _encode_texts(
    spec: dict,
    prompts: list[str],
    *,
    model_cache_root: Path,
    device_name: str,
    batch_size: int,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("text batch size must be positive")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for text encoding but is unavailable")
    if spec["kind"] == "open_clip":
        return _encode_open_clip_texts(
            spec,
            prompts,
            model_cache_root=model_cache_root,
            device_name=device_name,
            batch_size=batch_size,
        )
    device = torch.device(device_name)
    encoder = load_encoder(spec, device, model_cache_root)
    batches = []
    for start in range(0, len(prompts), batch_size):
        batches.append(encoder.encode_text(prompts[start : start + batch_size]).detach().cpu())
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return l2_normalize(torch.cat(batches, dim=0).float())


def aligned_source_text_embeddings(
    source_text: torch.Tensor,
    target_text: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    alignment = OrthogonalAlignment(
        rotation.float(),
        source_text.float().mean(dim=0, keepdim=True),
        target_text.float().mean(dim=0, keepdim=True),
    )
    return l2_normalize(alignment.transform(source_text.float(), centered=True))


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _attribute_rows_from_audit(audit_root: Path, attribute_count: int) -> list[dict]:
    summary_path = audit_root / "attribute_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"missing attribute audit summary: {summary_path}; run audit-cub-attribute-prompts first"
        )
    by_index = {}
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_index[int(row["attribute_index"])] = row
    rows = []
    for index in range(attribute_count):
        if index not in by_index:
            raise ValueError(f"attribute audit is missing index {index}")
        audit_row = by_index[index]
        raw = audit_row["raw_attribute_name"]
        phrase = audit_row.get("attribute_phrase") or clean_attribute_phrase(str(raw))
        rows.append(
            {
                "attribute_index": index,
                "attribute_id": index + 1,
                "raw_attribute_name": str(raw),
                "attribute_phrase": phrase,
                "attribute_prompt": attribute_only_prompt(phrase),
                "clip_readable_subset": index in set(CLIP_READABLE_ATTRIBUTE_INDICES),
            }
        )
    return rows


def _ranking_accuracy(scores: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> tuple[float | None, int]:
    positive_scores = scores[positives]
    negative_scores = scores[negatives]
    pair_count = int(positive_scores.size * negative_scores.size)
    if pair_count == 0:
        return None, 0
    diff = positive_scores[:, None] - negative_scores[None, :]
    wins = float((diff > 0).sum())
    ties = float((diff == 0).sum())
    return (wins + 0.5 * ties) / pair_count, pair_count


def _precision_at_k(scores: np.ndarray, positives_visible: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="mergesort")[: min(int(k), scores.size)]
    return float(positives_visible[order].mean())


def evaluate_global_attribute_retrieval(
    condition_text_embeddings: dict[str, torch.Tensor],
    condition_image_embeddings: dict[str, torch.Tensor],
    attributes: np.ndarray,
    mask: np.ndarray,
    attribute_rows: list[dict],
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> tuple[dict, list[dict]]:
    k_values = tuple(sorted({int(k) for k in k_values}))
    if set(condition_text_embeddings) != set(condition_image_embeddings):
        raise ValueError("text and image conditions must match")
    attributes = attributes.astype(bool)
    mask = mask.astype(bool)
    rows: list[dict] = []
    for condition in condition_text_embeddings:
        text = l2_normalize(condition_text_embeddings[condition].float())
        images = l2_normalize(condition_image_embeddings[condition].float())
        similarities = (text @ images.T).cpu().numpy()
        for attribute in attribute_rows:
            index = int(attribute["attribute_index"])
            visible = mask[:, index]
            positives = visible & attributes[:, index]
            negatives = visible & ~attributes[:, index]
            ranking_accuracy, pair_count = _ranking_accuracy(
                similarities[index],
                positives,
                negatives,
            )
            if ranking_accuracy is None:
                continue
            scores_visible = similarities[index][visible]
            positives_visible = attributes[:, index][visible]
            row = {
                **attribute,
                "condition": condition,
                "visible": int(visible.sum()),
                "positive": int(positives.sum()),
                "negative": int(negatives.sum()),
                "positive_prevalence": float(positives.sum() / max(visible.sum(), 1)),
                "pair_count": pair_count,
                "ranking_accuracy": ranking_accuracy,
            }
            for k in k_values:
                row[f"precision_at_{k}"] = _precision_at_k(
                    scores_visible,
                    positives_visible,
                    k,
                )
            rows.append(row)
    return _aggregate(rows, k_values=k_values), rows


def _aggregate(rows: list[dict], *, k_values: Iterable[int]) -> dict:
    k_values = tuple(sorted({int(k) for k in k_values}))
    output = {}
    for subset_name, predicate in [
        ("all_312", lambda row: True),
        ("clip_readable", lambda row: bool(row["clip_readable_subset"])),
    ]:
        output[subset_name] = {}
        subset_rows = [row for row in rows if predicate(row)]
        for condition in sorted({row["condition"] for row in subset_rows}):
            group = [row for row in subset_rows if row["condition"] == condition]
            metrics = {
                "attributes": len(group),
                "random_precision_at_k": float(np.mean([row["positive_prevalence"] for row in group])),
                "random_ranking_accuracy": 0.5,
                "ranking_accuracy_macro": float(np.mean([row["ranking_accuracy"] for row in group])),
                "ranking_gain_over_random": float(np.mean([row["ranking_accuracy"] for row in group]) - 0.5),
            }
            for k in k_values:
                value = float(np.mean([row[f"precision_at_{k}"] for row in group]))
                random_value = metrics["random_precision_at_k"]
                metrics[f"precision_at_{k}_macro"] = value
                metrics[f"precision_at_{k}_gain_over_random"] = value - random_value
            output[subset_name][condition] = metrics
    return output


def run_attribute_text_global_retrieval(
    config_path: Path | str,
    embedding_root: Path | str,
    output_root: Path | str,
    alignment_root: Path | str,
    model_cache_root: Path | str,
    *,
    audit_root: Path | str | None = None,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    text_batch_size: int = 64,
    device_name: str = "cuda",
    force: bool = False,
) -> Path:
    config_path = Path(config_path)
    embedding_root = Path(embedding_root)
    output_root = Path(output_root)
    alignment_root = Path(alignment_root)
    model_cache_root = Path(model_cache_root)
    audit_root = Path(audit_root) if audit_root is not None else output_root / "audit"
    config = read_decoder_config(config_path)
    base_run_id = config["run_id"]
    run_id = f"{base_run_id}_attribute_text_global_retrieval"
    result_dir = output_root / run_id
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not force:
        return result_dir

    train_features, test_features, train_path, test_path = _load_embedding_cache(
        embedding_root,
        base_run_id,
    )
    source_images = l2_normalize(test_features["source"].float())
    target_images = l2_normalize(test_features["target"].float())
    attributes = test_features["attributes"].cpu().numpy().astype(bool)
    mask = test_features["attribute_mask"].cpu().numpy().astype(bool)
    attribute_rows = _attribute_rows_from_audit(
        audit_root,
        int(test_features["attributes"].shape[1]),
    )
    prompts = [row["attribute_prompt"] for row in attribute_rows]

    source_text = _encode_texts(
        config["source_model"],
        prompts,
        model_cache_root=model_cache_root,
        device_name=device_name,
        batch_size=text_batch_size,
    )
    target_text = _encode_texts(
        config["target_model"],
        prompts,
        model_cache_root=model_cache_root,
        device_name=device_name,
        batch_size=text_batch_size,
    )
    oxford_path = alignment_root / _oxford_alignment_name(config)
    cub_train_q_path = alignment_root / f"{base_run_id.replace('_linear', '_cub_train_q_linear')}.pt"
    oxford = _load_alignment(oxford_path, config)
    cub_train_q = _load_alignment(cub_train_q_path, config)
    oxford_aligned_text = aligned_source_text_embeddings(source_text, target_text, oxford["rotation"])
    cub_train_aligned_text = aligned_source_text_embeddings(source_text, target_text, cub_train_q["rotation"])

    condition_text = {
        "native_source": source_text,
        "native_target": target_text,
        "unaligned_source_to_target": source_text,
        "oxford_aligned_source_to_target": oxford_aligned_text,
        "cub_train_aligned_source_to_target": cub_train_aligned_text,
    }
    condition_images = {
        "native_source": source_images,
        "native_target": target_images,
        "unaligned_source_to_target": target_images,
        "oxford_aligned_source_to_target": target_images,
        "cub_train_aligned_source_to_target": target_images,
    }
    aggregate, per_attribute = evaluate_global_attribute_retrieval(
        condition_text,
        condition_images,
        attributes,
        mask,
        attribute_rows,
        k_values=k_values,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "base_run_id": base_run_id,
        "config_path": str(config_path.resolve()),
        "candidate_policy": "all official CUB test images where the queried attribute is visible",
        "prompt_policy": "attribute-only prompts: 'a photo of a bird with {attribute_phrase}.'",
        "clip_readable_attribute_indices": CLIP_READABLE_ATTRIBUTE_INDICES,
        "k_values": list(k_values),
        "conditions": list(condition_text),
    }
    (summary_path).write_text(
        json.dumps(_jsonable({**metadata, "aggregate": aggregate}), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(result_dir / "per_attribute.csv", per_attribute)
    manifest = {
        **metadata,
        "artifacts": {"summary": "summary.json", "per_attribute": "per_attribute.csv"},
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
