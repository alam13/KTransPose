"""Differentiable TIH loss and Kabsch utilities for KTransPose.

The proposed TIH objective follows the original formulation:

    L_TIH = lambda_GPA * L_GPA + lambda_LAA * L_LAA

where
    L_GPA = Kabsch-aligned RMSD between predicted and target absolute poses,
    L_LAA = mean squared error between their intraligand pairwise-distance
            matrices.

Everything is implemented in PyTorch. No NumPy conversion or tensor detach is
used inside the loss, so both terms contribute gradients during training.
"""
from __future__ import annotations

import torch
from torch import nn

EPS = 1e-12


def _validate_pose(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"pred and target must be rank-2 tensors [N, 3], got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.shape != target.shape or pred.size(-1) != 3:
        raise ValueError(
            f"pred and target must have the same [N, 3] shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.size(0) == 0:
        raise ValueError("A pose must contain at least one atom")


def raw_rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSD without rigid alignment, in the input coordinate unit."""
    _validate_pose(pred, target)
    squared = torch.sum((pred.float() - target.float()) ** 2)
    return torch.sqrt(squared / pred.size(0) + EPS)


def kabsch_transform(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-vector rotation and translation aligning pred to target.

    The aligned pose is computed as::

        aligned = pred @ rotation.T + translation

    The reflection correction is differentiable almost everywhere and avoids
    introducing an improper rotation.
    """
    _validate_pose(pred, target)
    pred_f = pred.float()
    target_f = target.float()

    pred_centroid = pred_f.mean(dim=0, keepdim=True)
    target_centroid = target_f.mean(dim=0, keepdim=True)

    # With fewer than three atoms, rotation is underdetermined. Translation-only
    # alignment is the most stable and explicit fallback.
    if pred.size(0) < 3:
        rotation = torch.eye(3, dtype=pred_f.dtype, device=pred_f.device)
        translation = target_centroid - pred_centroid
        return rotation, translation

    pred_centered = pred_f - pred_centroid
    target_centered = target_f - target_centroid
    covariance = pred_centered.transpose(0, 1) @ target_centered

    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    vu_t = vh.transpose(0, 1) @ u.transpose(0, 1)

    correction = torch.eye(3, dtype=pred_f.dtype, device=pred_f.device)
    correction[-1, -1] = torch.where(
        torch.det(vu_t) < 0,
        pred_f.new_tensor(-1.0),
        pred_f.new_tensor(1.0),
    )

    rotation = vh.transpose(0, 1) @ correction @ u.transpose(0, 1)
    translation = target_centroid - pred_centroid @ rotation.transpose(0, 1)
    return rotation, translation


def kabsch_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rigidly align pred to target using differentiable Kabsch."""
    rotation, translation = kabsch_transform(pred, target)
    return pred.float() @ rotation.transpose(0, 1) + translation


def gpa_rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Global pose accuracy term: Kabsch-aligned RMSD."""
    aligned = kabsch_align(pred, target)
    return raw_rmsd(aligned, target.float())


def pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {tuple(coords.shape)}")
    return torch.cdist(coords.float(), coords.float(), p=2)


def laa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Local atomic alignment term: pairwise-distance matrix MSE."""
    _validate_pose(pred, target)
    pred_dist = pairwise_distance_matrix(pred)
    target_dist = pairwise_distance_matrix(target)
    return torch.mean((pred_dist - target_dist) ** 2)


def laa_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise-distance RMSE, useful as an interpretable diagnostic."""
    return torch.sqrt(laa_loss(pred, target) + EPS)


class TIHLoss(nn.Module):
    """Differentiable GPA + LAA objective."""

    def __init__(self, lambda_gpa: float = 0.5, lambda_laa: float = 0.5):
        super().__init__()
        if lambda_gpa < 0 or lambda_laa < 0:
            raise ValueError("TIH weights must be non-negative")
        if lambda_gpa + lambda_laa <= 0:
            raise ValueError("At least one TIH weight must be positive")
        self.lambda_gpa = float(lambda_gpa)
        self.lambda_laa = float(lambda_laa)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gpa_value = gpa_rmsd(pred, target)
        laa_value = laa_rmse(pred, target)

        return (
            self.lambda_gpa * gpa_value
            + self.lambda_laa * laa_value
        )

    # def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    #     raw_value = raw_rmsd(pred, target)
    #     gpa_value = gpa_rmsd(pred, target)
    #     laa_value = laa_loss(pred, target)
    #     return {
    #         "raw_rmsd": raw_value,
    #         "gpa_rmsd": gpa_value,
    #         "laa": laa_value,
    #         "laa_rmse": torch.sqrt(laa_value + EPS),
    #         "tih": self.lambda_gpa * gpa_value + self.lambda_laa * laa_value,
    #     }
    def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_value = raw_rmsd(pred, target)
        gpa_value = gpa_rmsd(pred, target)

        laa_mse_value = laa_loss(pred, target)
        laa_rmse_value = torch.sqrt(laa_mse_value + EPS)

        tih_value = (
            self.lambda_gpa * gpa_value
            + self.lambda_laa * laa_rmse_value
        )

        return {
            "raw_rmsd": raw_value,
            "gpa_rmsd": gpa_value,
            "laa": laa_mse_value, 
            "laa_rmse": laa_rmse_value,
            "tih": tih_value,
        }
