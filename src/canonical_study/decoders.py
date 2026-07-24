"""Decoder architectures shared by training and post-hoc analysis."""

from collections.abc import Mapping

import torch
from torch import nn


class AttributeMLP(nn.Module):
    """A prespecified one-hidden-layer CUB attribute decoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_decoder(
    decoder_config: Mapping,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    architecture = decoder_config["architecture"]
    if architecture == "linear":
        return nn.Linear(input_dim, output_dim)
    if architecture == "mlp":
        if decoder_config.get("activation") != "gelu":
            raise ValueError("the prespecified MLP requires activation='gelu'")
        return AttributeMLP(
            input_dim,
            output_dim,
            hidden_dim=int(decoder_config["hidden_dim"]),
            dropout=float(decoder_config["dropout"]),
        )
    raise ValueError(f"unsupported decoder architecture: {architecture}")


@torch.inference_mode()
def decoder_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    return torch.sigmoid(model(features.to(device))).cpu()
