"""Linear decoder utilities shared by training and post-hoc analysis."""

from collections.abc import Mapping

import torch
from torch import nn


def build_decoder(
    decoder_config: Mapping,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    architecture = decoder_config["architecture"]
    if architecture == "linear":
        return nn.Linear(input_dim, output_dim)
    raise ValueError(f"unsupported decoder architecture: {architecture}")


@torch.inference_mode()
def decoder_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    return torch.sigmoid(model(features.to(device))).cpu()
