"""Orthogonal Procrustes alignment for row-vector representations."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OrthogonalAlignment:
    """An affine alignment ``(x - source_mean) @ rotation + target_mean``."""

    rotation: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor

    def transform(self, values: torch.Tensor, *, centered: bool = True) -> torch.Tensor:
        if centered:
            return (values - self.source_mean) @ self.rotation + self.target_mean
        return values @ self.rotation

    @property
    def orthogonality_error(self) -> float:
        identity = torch.eye(
            self.rotation.shape[0],
            dtype=self.rotation.dtype,
            device=self.rotation.device,
        )
        return torch.linalg.matrix_norm(
            self.rotation.T @ self.rotation - identity, ord="fro"
        ).item()


def fit_orthogonal_alignment(
    source: torch.Tensor,
    target: torch.Tensor,
) -> OrthogonalAlignment:
    """Fit the least-squares orthogonal map from paired source to target rows."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must both be rank-2 tensors")
    if source.shape != target.shape:
        raise ValueError(
            f"paired matrices must have the same shape, got {source.shape} and {target.shape}"
        )
    if source.shape[0] < 2:
        raise ValueError("at least two paired observations are required")

    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    cross_covariance = (source - source_mean).T @ (target - target_mean)
    left, _, right_h = torch.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right_h
    return OrthogonalAlignment(rotation, source_mean, target_mean)


def apply_shared_rotation(
    values: torch.Tensor,
    rotation: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    *,
    centered: bool,
) -> torch.Tensor:
    """Apply an image-fitted rotation with means supplied for any modality."""
    if centered:
        return (values - source_mean) @ rotation + target_mean
    return values @ rotation

