"""Reusable iterative refinement helpers for KTransPose experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class IterationConfig:
    steps: int = 3
    recompute_edges: bool = True
    share_weights: bool = False


class IterativeRefiner:
    """Run K-step pose refinement over one model or a model list.

    - If ``share_weights`` is True, reuses the same model each step.
    - If False, expects ``models`` to contain one model per step.
    """

    def __init__(self, config: IterationConfig):
        self.config = config

    def run(
        self,
        data,
        models,
        edge_builder: Callable | None = None,
    ):
        pose = data
        outputs = []

        for k in range(self.config.steps):
            model = models if self.config.share_weights else models[k]
            pred = model(pose.x.float(), pose.edge_index, pose.dist.float())
            outputs.append(pred)

            if hasattr(pose, "flexible_idx"):
                pose.x[pose.flexible_idx.bool(), -3:] = pred
            else:
                pose.x[:, -3:] = pred

            if self.config.recompute_edges:
                if edge_builder is None:
                    raise ValueError("edge_builder must be provided when recompute_edges=True")
                pose.edge_index, pose.dist = edge_builder(pose.x[:, -3:])

        return outputs, pose
