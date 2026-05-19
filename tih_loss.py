"""Transformation-invariant hybrid (TIH) loss utilities for KTransPose.

This module formalizes the loss requested by reviewers and can be imported in
training code without changing model architecture.
"""

from __future__ import annotations

import torch
from torch import nn


def _center(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered coordinates and centroid.

    Args:
        coords: Tensor of shape [N, 3].
    """
    centroid = coords.mean(dim=0, keepdim=True)
    return coords - centroid, centroid


def kabsch_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align ``pred`` to ``target`` with a differentiable Kabsch transform.

    Args:
        pred: Tensor [N, 3] predicted coordinates.
        target: Tensor [N, 3] ground-truth coordinates.

    Returns:
        Aligned predicted coordinates [N, 3].
    """
    pred_c, pred_mu = _center(pred)
    target_c, target_mu = _center(target)

    cov = pred_c.transpose(0, 1) @ target_c
    u, _, vh = torch.linalg.svd(cov)

    # Reflection handling
    d = torch.eye(3, device=pred.device, dtype=pred.dtype)
    d[-1, -1] = torch.sign(torch.det(vh.transpose(0, 1) @ u.transpose(0, 1)))

    r = vh.transpose(0, 1) @ d @ u.transpose(0, 1)
    t = target_mu.squeeze(0) - (r @ pred_mu.squeeze(0))
    return (pred @ r.transpose(0, 1)) + t


def pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance matrix for coordinates [N, 3]."""
    return torch.cdist(coords, coords, p=2)


class TIHLoss(nn.Module):
    """Transformation-Invariant Hybrid (TIH) loss.

    TIH = lambda_gpa * GPA + lambda_laa * LAA

    GPA = sqrt( mean_i || Kabsch(pred)_i - target_i ||^2 )
    LAA = mean_{i,j} ( ||pred_i - pred_j|| - ||target_i - target_j|| )^2
    """

    def __init__(self, lambda_gpa: float = 0.5, lambda_laa: float | None = None):
        super().__init__()
        if lambda_laa is None:
            lambda_laa = 1.0 - lambda_gpa
        if lambda_gpa < 0 or lambda_laa < 0:
            raise ValueError("TIH weights must be non-negative.")
        self.lambda_gpa = lambda_gpa
        self.lambda_laa = lambda_laa

    def gpa(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        aligned = kabsch_align(pred, target)
        return torch.sqrt(torch.mean((aligned - target) ** 2) + 1e-12)

    def laa(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d_pred = pairwise_distance_matrix(pred)
        d_true = pairwise_distance_matrix(target)
        return torch.mean((d_pred - d_true) ** 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gpa = self.gpa(pred, target)
        laa = self.laa(pred, target)
        return self.lambda_gpa * gpa + self.lambda_laa * laa
