"""Frozen encoder adapters for OpenCLIP and FLAVA."""

import os
from pathlib import Path
from typing import Protocol

import torch

from .metrics import l2_normalize


class EncoderPairMember(Protocol):
    preprocess: object

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, texts: list[str]) -> torch.Tensor: ...


class OpenClipEncoder:
    def __init__(self, spec: dict, device: torch.device, cache_root: Path):
        import open_clip

        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_root))
        os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root))
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["name"],
            pretrained=spec["pretrained"],
            cache_dir=str(cache_root),
        )
        self.model = model.to(device).eval().requires_grad_(False)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(spec["name"])
        self.device = device

    @torch.inference_mode()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.model.encode_image(images.to(self.device)))

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return l2_normalize(self.model.encode_text(tokens))


class FlavaEncoder:
    def __init__(self, spec: dict, device: torch.device, cache_root: Path):
        from transformers import AutoProcessor, FlavaModel

        model_id = spec.get("model_id", "facebook/flava-full")
        revision = spec.get("revision", "main")
        self.model = FlavaModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_root),
        ).to(device).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_root),
        )
        self.preprocess = self._preprocess
        self.device = device

    def _preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

    @torch.inference_mode()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model.get_image_features(
            pixel_values=images.to(self.device)
        )[:, 0, :]
        return l2_normalize(output)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        accepted = {"input_ids", "attention_mask", "token_type_ids"}
        tokens = {
            key: value.to(self.device)
            for key, value in tokens.items()
            if key in accepted
        }
        output = self.model.get_text_features(**tokens)
        if output.ndim == 3:
            output = output[:, 0, :]
        return l2_normalize(output)


def load_encoder(spec: dict, device: torch.device, cache_root: Path):
    kind = spec["kind"]
    if kind == "open_clip":
        standard_cache = Path.home() / ".open_clip"
        open_clip_cache = Path(
            os.environ.get(
                "CANONICAL_STUDY_OPENCLIP_CACHE",
                standard_cache if standard_cache.is_dir() else cache_root / "open_clip",
            )
        )
        return OpenClipEncoder(spec, device, open_clip_cache)
    if kind == "flava":
        return FlavaEncoder(spec, device, cache_root / "huggingface")
    raise ValueError(f"unsupported encoder kind: {kind}")
