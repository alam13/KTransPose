"""Reusable iterative refinement helpers for KTransPose experiments.

This module implements iterative pose refinement for KTransPose-style
coordinate-displacement models.

At each refinement step:
1. the model predicts a displacement for flexible ligand atoms,
2. the current pose is updated using that displacement,
3. pose-dependent graph edges are optionally rebuilt.

The default setting uses shared weights across refinement steps because the
main training/evaluation script reuses the same trained model for iterative
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass
class IterationConfig:
    """Configuration for iterative pose refinement.

    Args:
        steps: Number of refinement steps. Must be >= 1.
        recompute_edges: If True, rebuild edges touching flexible atoms after
            every pose update. If False, coordinates and residual targets are
            still updated, but edge_index/dist are kept unchanged.
        share_weights: If True, reuse one model at every refinement step. If
            False, ``models`` passed to IterativeRefiner.run must be a sequence
            with one model per step.
        edge_threshold: Distance threshold in Angstrom for rebuilding edges.
        space: Coordinate scale used by the dataset. If coordinates are stored
            normalized by 100, keep this as 100.0.
    """

    steps: int = 3
    recompute_edges: bool = True
    share_weights: bool = True
    edge_threshold: float = 6.0
    space: float = 100.0


def _as_bool_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert flexible_idx-style input to a boolean mask on the target device."""
    mask = mask.to(device)

    if mask.dtype == torch.bool:
        return mask

    return mask.bool()


def build_pose_edges(
    coords: torch.Tensor,
    flexible_idx: torch.Tensor,
    existing_edge_index: torch.Tensor | None = None,
    existing_edge_attr: torch.Tensor | None = None,
    edge_threshold: float = 6.0,
    space: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild graph edges touching flexible atoms after a pose update.

    Rigid-rigid edges are copied from the existing graph because their
    distances do not change. Edges where at least one endpoint is flexible are
    recomputed using the updated coordinates.
    """
    if coords.dim() != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {tuple(coords.shape)}")

    device = coords.device
    dtype = coords.dtype
    flexible_idx = _as_bool_mask(flexible_idx, device)

    if flexible_idx.numel() != coords.size(0):
        raise ValueError(
            f"flexible_idx length must match number of nodes: "
            f"{flexible_idx.numel()} vs {coords.size(0)}"
        )

    keep_edges: list[torch.Tensor] = []
    keep_attr: list[torch.Tensor] = []

    if existing_edge_index is not None and existing_edge_attr is not None:
        existing_edge_index = existing_edge_index.to(device=device, dtype=torch.long)
        existing_edge_attr = existing_edge_attr.to(device=device, dtype=dtype)

        if existing_edge_index.dim() != 2 or existing_edge_index.size(0) != 2:
            raise ValueError(
                f"existing_edge_index must have shape [2, E], got {tuple(existing_edge_index.shape)}"
            )

        if existing_edge_attr.dim() != 2 or existing_edge_attr.size(0) != existing_edge_index.size(1):
            raise ValueError(
                "existing_edge_attr must have shape [E, edge_dim] matching existing_edge_index"
            )

        rigid_edge_mask = (
            ~flexible_idx[existing_edge_index[0]]
            & ~flexible_idx[existing_edge_index[1]]
        )

        if rigid_edge_mask.any():
            keep_edges.append(existing_edge_index[:, rigid_edge_mask])
            keep_attr.append(existing_edge_attr[rigid_edge_mask])

    threshold = float(edge_threshold) / float(space)
    n = coords.size(0)

    src: list[int] = []
    dst: list[int] = []
    attrs: list[torch.Tensor] = []

    zero = coords.new_zeros(())

    for i in range(n):
        for j in range(i + 1, n):
            i_flex = bool(flexible_idx[i].item())
            j_flex = bool(flexible_idx[j].item())

            if not (i_flex or j_flex):
                continue

            dist = torch.linalg.norm(coords[i] - coords[j])

            if dist <= threshold:
                if i_flex and j_flex:
                    attr = torch.stack([zero, zero, dist])
                else:
                    attr = torch.stack([zero, dist, zero])

                src.extend([i, j])
                dst.extend([j, i])
                attrs.extend([attr, attr])

    if src:
        new_edges = torch.tensor([src, dst], dtype=torch.long, device=device)
        new_attr = torch.stack(attrs).to(dtype=dtype)
        keep_edges.append(new_edges)
        keep_attr.append(new_attr)

    if not keep_edges:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=dtype, device=device),
        )

    return torch.cat(keep_edges, dim=1), torch.cat(keep_attr, dim=0)


def apply_flexible_displacement(
    data,
    pred: torch.Tensor,
    edge_threshold: float = 6.0,
    space: float = 100.0,
    recompute_edges: bool = True,
):
    """Return a cloned data object after applying one flexible-atom displacement.

    The coordinates and residual target ``y`` are always updated. If
    ``recompute_edges`` is False, the existing edge_index and dist are kept.
    """
    if not hasattr(data, "flexible_idx"):
        raise AttributeError("data must contain flexible_idx for flexible-pose refinement")

    updated = data.clone()

    mask = _as_bool_mask(updated.flexible_idx, updated.x.device)
    num_flexible = int(mask.sum().item())

    if pred.dim() != 2 or pred.size(-1) != 3:
        raise ValueError(f"pred must have shape [N_flexible, 3], got {tuple(pred.shape)}")

    if pred.size(0) != num_flexible:
        raise ValueError(
            f"pred has {pred.size(0)} rows, but data has {num_flexible} flexible atoms"
        )

    pred_x = pred.to(device=updated.x.device, dtype=updated.x.dtype)
    pred_y = pred.to(device=updated.y.device, dtype=updated.y.dtype)

    updated.x = updated.x.clone()
    updated.y = updated.y.clone()

    updated.x[mask, -3:] = updated.x[mask, -3:] + pred_x
    updated.y = updated.y - pred_y

    if recompute_edges:
        updated.edge_index, updated.dist = build_pose_edges(
            updated.x[:, -3:],
            mask,
            existing_edge_index=data.edge_index,
            existing_edge_attr=data.dist,
            edge_threshold=edge_threshold,
            space=space,
        )
    else:
        updated.edge_index = data.edge_index.clone()
        updated.dist = data.dist.clone()

    return updated


class IterativeRefiner:
    """Run K-step pose refinement using one model or a sequence of models.

    If ``share_weights=True``, the same model is reused at every step.

    If ``share_weights=False``, ``models`` must be a sequence containing one
    model per refinement step.

    The method returns:
        total_pred: Sum of all step displacements, shape [N_flexible, 3].
        step_outputs: List of per-step displacements.
        final_pose: Data object after applying all predicted displacements.
    """

    def __init__(self, config: IterationConfig):
        if config.steps < 1:
            raise ValueError("IterationConfig.steps must be at least 1.")
        self.config = config

    def _model_for_step(self, models, step: int):
        if self.config.share_weights:
            return models

        if not isinstance(models, Sequence) or len(models) <= step:
            raise ValueError(
                "When share_weights=False, provide a sequence with one model per refinement step."
            )

        return models[step]

    def _default_predict(self, model, pose):
        pred = model(pose.x.float(), pose.edge_index, pose.dist.float())

        if hasattr(pose, "flexible_idx") and pred.size(0) == pose.x.size(0):
            pred = pred[_as_bool_mask(pose.flexible_idx, pred.device)]

        return pred

    def run(
        self,
        data,
        models,
        predict_fn: Callable | None = None,
        edge_builder: Callable | None = None,
    ):
        """Run iterative refinement.

        Args:
            data: Input PyG Data object.
            models: One model if share_weights=True, otherwise a sequence.
            predict_fn: Optional custom prediction function with signature
                ``predict_fn(model, pose, step) -> pred``.
            edge_builder: Optional custom update function with signature
                ``edge_builder(pose, pred) -> updated_pose``.

        Returns:
            total_pred: Sum of all predicted flexible-atom displacements.
            step_outputs: List of per-step predictions.
            final_pose: Updated pose after all refinement steps.
        """
        pose = data
        step_outputs: list[torch.Tensor] = []
        total_pred: torch.Tensor | None = None

        for step in range(self.config.steps):
            model = self._model_for_step(models, step)

            if predict_fn is None:
                pred = self._default_predict(model, pose)
            else:
                pred = predict_fn(model, pose, step)

            step_outputs.append(pred)
            total_pred = pred if total_pred is None else total_pred + pred

            if edge_builder is not None:
                pose = edge_builder(pose, pred)
            else:
                pose = apply_flexible_displacement(
                    pose,
                    pred,
                    edge_threshold=self.config.edge_threshold,
                    space=self.config.space,
                    recompute_edges=self.config.recompute_edges,
                )

        assert total_pred is not None
        return total_pred, step_outputs, pose
