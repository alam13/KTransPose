"""Train MedusaGraph baseline or stage-wise TIH KTransPose.

Protocol implemented here
-------------------------
1. MedusaGraph baseline:
   - train with ordinary MSE only;
   - no Kabsch operation is used by its training or validation loss;
   - Kabsch-aligned test RMSD is computed later by ``test3.py``.

2. Proposed method:
   - train with differentiable TIH = GPA + LAA;
   - GPA uses differentiable PyTorch Kabsch;
   - Iter1 uses the original poses;
   - Iter2 inputs are updated once by frozen Iter1;
   - Iter3 inputs are updated once by frozen Iter1 and once by frozen Iter2;
   - every newly trained stage performs one model forward pass per batch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import PDBBindCoor
from iterative_refinement import build_pose_edges
from model import Net_coor, Net_coor_cent, Net_coor_res, loss_fn_cos
from tih_loss import TIHLoss, gpa_rmsd, laa_loss, laa_rmse, raw_rmsd

SPACE = 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["Net_coor", "Net_coor_res", "Net_coor_cent"], default="Net_coor")
    p.add_argument("--loss", choices=["MSELoss", "TIH", "L1Loss", "CosineEmbeddingLoss", "CosineAngle"], default="TIH")
    p.add_argument("--loss_reduction", default="mean")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epoch", type=int, default=150)
    p.add_argument("--start_epoch", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--data_path", required=True)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--edge_dim", type=int, default=3)
    p.add_argument("--n_graph_layer", type=int, default=4)
    p.add_argument("--d_graph_layer", type=int, default=256)
    p.add_argument("--n_FC_layer", type=int, default=0)
    p.add_argument("--d_FC_layer", type=int, default=512)
    p.add_argument("--dropout_rate", type=float, default=0.3)
    p.add_argument("--weight_bias", type=float, default=1.0)
    p.add_argument("--last", default="log")
    p.add_argument("--residue", action="store_true")
    p.add_argument("--flexible", action="store_true")
    p.add_argument("--class_dir", action="store_true")
    p.add_argument("--step_len", type=float, default=0.03)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--output_dim", type=int, default=3)
    p.add_argument("--atomwise", type=int, default=0)
    p.add_argument("--hinge", type=float, default=0.0)
    p.add_argument("--edge", action="store_true")
    p.add_argument("--KD", default="No")
    p.add_argument("--KD_soft", default="exp")
    p.add_argument("--th", type=float, default=3.0)
    p.add_argument("--plt_dir", default="best_model_plt")
    p.add_argument("--tot_seed", type=int, default=8)

    p.add_argument("--iterative", type=int, choices=[1, 2, 3], default=1,
                   help="1=original input; 2=one prior-stage input update; 3=two prior-stage input updates")
    p.add_argument("--input_refine_checkpoints", nargs="*", default=[],
                   help="ordered prior-stage checkpoints, each applied once before training")
    p.add_argument("--edge_threshold", type=float, default=6.0)

    p.add_argument("--tih_lambda_gpa", type=float, default=0.5)
    p.add_argument("--tih_lambda_laa", type=float, default=0.5)

    p.add_argument("--selection_metric", choices=["val_loss", "avg_raw_rmsd_per_complex_ang", "avg_tih_per_complex"], default="val_loss")
    p.add_argument("--pre_model", default="None")
    p.add_argument("--model_dir", default="runs/models")
    p.add_argument("--artifact_dir", default="runs/artifacts")
    p.add_argument("--output", default="none")
    p.add_argument("--pose_limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save_predictions_npz", action="store_true")
    return p.parse_args()


args = parse_args()
expected_updates = args.iterative - 1
if len(args.input_refine_checkpoints) != expected_updates:
    raise ValueError(
        f"Iterative-{args.iterative} requires exactly {expected_updates} prior-stage "
        f"checkpoint(s), received {len(args.input_refine_checkpoints)}"
    )
if args.loss == "MSELoss" and args.iterative != 1:
    raise ValueError("The MedusaGraph MSE baseline is defined as the one-stage baseline (iterative=1)")
if args.class_dir:
    raise ValueError("class_dir is not supported by this reviewer protocol")

print(args, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def best_model_path(model_dir: str) -> str:
    path = Path(model_dir)
    if path.suffix.lower() in {".pt", ".pth"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path / "best_model.pt")


def load_state(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    loaded = torch.load(str(path), map_location=device)
    if isinstance(loaded, torch.nn.Module):
        return loaded.state_dict()
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        return loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(loaded)}")
    return loaded


def get_pdb_id(data) -> str:
    pdb = data.pdb
    if isinstance(pdb, (list, tuple)):
        pdb = pdb[0]
    if isinstance(pdb, np.ndarray):
        pdb = pdb.tolist()
    if isinstance(pdb, bytes):
        pdb = pdb.decode("utf-8")
    return str(pdb)


def flexible_mask(data) -> torch.Tensor:
    return data.flexible_idx.bool()


def input_pose(data) -> torch.Tensor:
    return data.x[flexible_mask(data), -3:].float()


def target_displacement(data) -> torch.Tensor:
    return data.y.float()


def target_pose(data) -> torch.Tensor:
    return input_pose(data) + target_displacement(data)


def build_model(num_features: int, device: torch.device):
    if args.model_type == "Net_coor":
        return Net_coor(num_features, args).to(device)
    if args.model_type == "Net_coor_res":
        return Net_coor_res(num_features, args).to(device)
    return Net_coor_cent(num_features, args).to(device)


def model_predict(model, data) -> torch.Tensor:
    mask = flexible_mask(data)
    if args.model_type == "Net_coor_cent":
        pred = model(data.x.float(), data.edge_index, data.dist.float(), data.batch, mask)
        if pred.dim() == 2 and pred.size(0) > 1:
            pred = pred[data.batch[mask]]
        else:
            pred = pred.reshape(1, -1).repeat(target_displacement(data).size(0), 1)
    else:
        pred = model(data.x.float(), data.edge_index, data.dist.float())
        if pred.size(0) == data.x.size(0):
            pred = pred[mask]
    pred = pred.float()
    if pred.shape != target_displacement(data).shape:
        raise ValueError(
            f"Prediction shape {tuple(pred.shape)} does not match target displacement "
            f"shape {tuple(target_displacement(data).shape)}"
        )
    return pred


def graph_segments(data) -> list[torch.Tensor]:
    mask = flexible_mask(data)
    if not hasattr(data, "batch"):
        return [torch.arange(int(mask.sum()), device=data.x.device)]
    flex_batch = data.batch[mask]
    return [(flex_batch == graph_id).nonzero(as_tuple=True)[0] for graph_id in flex_batch.unique(sorted=True)]


def loss_for_batch(pred: torch.Tensor, data, criterion) -> torch.Tensor:
    if args.loss == "TIH":
        pred_abs = input_pose(data) + pred
        true_abs = target_pose(data)
        losses = [criterion(pred_abs[idx], true_abs[idx]) for idx in graph_segments(data) if idx.numel()]
        return torch.stack(losses).mean()
    if args.loss == "CosineEmbeddingLoss":
        return criterion(pred, target_displacement(data), pred.new_ones(pred.size(0)))
    return criterion(pred, target_displacement(data))


def update_pose_graph(data, pred: torch.Tensor):
    """Apply one frozen-stage displacement and rebuild the pose graph."""
    updated = data.clone()
    mask = flexible_mask(updated)
    applied = pred.detach().to(updated.x.device, updated.x.dtype)
    updated.x = updated.x.clone()
    updated.y = updated.y.clone()
    updated.x[mask, -3:] = updated.x[mask, -3:] + applied
    updated.y = updated.y - applied.to(updated.y.device, updated.y.dtype)
    edge_index, edge_attr = build_pose_edges(
        updated.x[:, -3:],
        mask,
        existing_edge_index=data.edge_index,
        existing_edge_attr=data.dist,
        edge_threshold=args.edge_threshold,
        space=SPACE,
    )
    updated.edge_index = edge_index.to(updated.x.device)
    updated.dist = edge_attr.to(updated.x.device)
    return updated


set_seed(args.seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

base_train_dataset = PDBBindCoor(root=args.data_path, split="train")
base_val_dataset = PDBBindCoor(root=args.data_path, split="test")
num_features = int(base_train_dataset.num_features)
model = build_model(num_features, device)

if args.pre_model != "None":
    model.load_state_dict(load_state(args.pre_model, device))

if args.loss == "TIH":
    criterion = TIHLoss(args.tih_lambda_gpa, args.tih_lambda_laa)
elif args.loss == "MSELoss":
    criterion = torch.nn.MSELoss(reduction=args.loss_reduction)
elif args.loss == "L1Loss":
    criterion = torch.nn.L1Loss(reduction=args.loss_reduction)
elif args.loss == "CosineEmbeddingLoss":
    criterion = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
else:
    criterion = loss_fn_cos(device, reduction=args.loss_reduction)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def materialize(dataset) -> list:
    return [dataset[i].clone() for i in range(len(dataset))]


@torch.no_grad()
def refine_dataset_once(dataset: list, checkpoint: str, label: str) -> list:
    model.load_state_dict(load_state(checkpoint, device))
    model.eval()
    refined = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in tqdm(loader, desc=label):
        data = data.to(device)
        pred = model_predict(model, data)
        refined.append(update_pose_graph(data, pred).cpu())
    return refined


train_dataset = materialize(base_train_dataset)
val_dataset = materialize(base_val_dataset)
for update_idx, checkpoint in enumerate(args.input_refine_checkpoints, start=1):
    train_dataset = refine_dataset_once(train_dataset, checkpoint, f"Refine train stage {update_idx}")
    val_dataset = refine_dataset_once(val_dataset, checkpoint, f"Refine validation stage {update_idx}")

# Refinement temporarily loaded prior-stage weights. Restore the requested stage
# initialization (or fresh initialization when pre_model=None).
if args.pre_model != "None":
    model.load_state_dict(load_state(args.pre_model, device))
else:
    model = build_model(num_features, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"train={len(train_dataset)} validation={len(val_dataset)}", flush=True)

artifact_dir = Path(args.artifact_dir)
artifact_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = best_model_path(args.model_dir)
manifest_file = artifact_dir / "run_manifest.json"
best_metrics_file = artifact_dir / "best_epoch_metrics.json"
best_pose_file = artifact_dir / "best_epoch_per_pose.csv"
best_complex_file = artifact_dir / "best_epoch_per_complex.csv"
epoch_file = artifact_dir / "epoch_metrics.csv"

protocol_name = "medusagraph_mse" if args.loss == "MSELoss" else "proposed_tih"
write_json(
    manifest_file,
    {
        "created_at": datetime.now().isoformat(),
        "protocol_name": protocol_name,
        "model_type": args.model_type,
        "training_objective": args.loss,
        "command_args": vars(args),
        "dataset": {
            "data_path": args.data_path,
            "train_size": len(train_dataset),
            "validation_size": len(val_dataset),
            "num_features": num_features,
            "coordinate_scale_angstrom": SPACE,
        },
        "tih_definition": None if args.loss != "TIH" else {
            "formula": "lambda_gpa * Kabsch_RMSD + lambda_laa * pairwise_distance_MSE",
            "lambda_gpa": args.tih_lambda_gpa,
            "lambda_laa": args.tih_lambda_laa,
            "kabsch_implementation": "differentiable PyTorch SVD",
        },
        "baseline_kabsch_policy": (
            "No Kabsch in baseline training/validation loss; Kabsch is applied only by test3.py"
            if args.loss == "MSELoss" else None
        ),
        "stagewise_refinement": {
            "stage": args.iterative,
            "prior_checkpoints": args.input_refine_checkpoints,
            "one_application_per_prior_checkpoint": True,
            "one_forward_pass_per_training_batch": True,
            "rebuild_graph_after_each_prior_update": True,
        },
    },
)


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_epoch(epoch: int) -> float:
    model.train()
    total = 0.0
    count = 0
    start = time()
    for data in tqdm(train_loader, desc=f"Train epoch {epoch}"):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model_predict(model, data)
        loss = loss_for_batch(pred, data, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    print(f"Epoch {epoch} training took {time() - start:.1f}s", flush=True)
    return total / max(count, 1)


@torch.no_grad()
def validate(epoch: int):
    """Validate without Kabsch for MSE baseline; with TIH diagnostics for TIH."""
    model.eval()
    rows = []
    total_loss = 0.0
    for pose_index, data in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch}")):
        data = data.to(device)
        pred = model_predict(model, data)
        loss = loss_for_batch(pred, data, criterion)
        total_loss += float(loss.cpu())
        pred_abs = input_pose(data) + pred
        true_abs = target_pose(data)
        raw_value = raw_rmsd(pred_abs, true_abs)

        row = {
            "epoch": epoch,
            "pose_index": pose_index,
            "pdb": get_pdb_id(data),
            "training_objective": args.loss,
            "loss": float(loss.cpu()),
            "raw_rmsd_ang": float(raw_value.cpu()) * SPACE,
        }
        if args.loss == "TIH":
            row.update({
                "gpa_rmsd_ang": float(gpa_rmsd(pred_abs, true_abs).cpu()) * SPACE,
                "laa": float(laa_loss(pred_abs, true_abs).cpu()),
                "laa_rmse_ang": float(laa_rmse(pred_abs, true_abs).cpu()) * SPACE,
                "tih": float(criterion(pred_abs, true_abs).cpu()),
            })
        rows.append(row)
        if args.pose_limit > 0 and len(rows) >= args.pose_limit:
            break

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["pdb"], []).append(row)

    per_complex = []
    for pdb, group_rows in grouped.items():
        item = {
            "epoch": epoch,
            "pdb": pdb,
            "num_poses": len(group_rows),
            "avg_raw_rmsd_ang": float(np.mean([r["raw_rmsd_ang"] for r in group_rows])),
            "avg_loss": float(np.mean([r["loss"] for r in group_rows])),
        }
        if args.loss == "TIH":
            item.update({
                "avg_gpa_rmsd_ang": float(np.mean([r["gpa_rmsd_ang"] for r in group_rows])),
                "avg_laa": float(np.mean([r["laa"] for r in group_rows])),
                "avg_tih": float(np.mean([r["tih"] for r in group_rows])),
            })
        per_complex.append(item)

    metrics = {
        "epoch": epoch,
        "val_loss": total_loss / max(len(rows), 1),
        "avg_raw_rmsd_per_complex_ang": float(np.mean([r["avg_raw_rmsd_ang"] for r in per_complex])),
        "num_poses": len(rows),
        "num_complexes": len(per_complex),
    }
    if args.loss == "TIH":
        metrics.update({
            "avg_gpa_rmsd_per_complex_ang": float(np.mean([r["avg_gpa_rmsd_ang"] for r in per_complex])),
            "avg_laa_per_complex": float(np.mean([r["avg_laa"] for r in per_complex])),
            "avg_tih_per_complex": float(np.mean([r["avg_tih"] for r in per_complex])),
        })
    return metrics, per_complex, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


best_value = float("inf")
best_epoch = 0
for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
    train_loss = train_epoch(epoch)
    metrics, per_complex, per_pose = validate(epoch)
    if args.selection_metric == "avg_tih_per_complex" and args.loss != "TIH":
        raise ValueError("avg_tih_per_complex can only select a TIH-trained model")
    selected = metrics[args.selection_metric]

    epoch_row = {"epoch": epoch, "train_loss": train_loss, **metrics}
    append_csv(epoch_file, epoch_row)

    message = (
        f"Epoch: {epoch} Train Loss: {train_loss:.8f} Validation Loss: {metrics['val_loss']:.8f} "
        f"Raw RMSD: {metrics['avg_raw_rmsd_per_complex_ang']:.6f} "
    )
    if args.loss == "TIH":
        message += (
            f"GPA/Kabsch RMSD: {metrics['avg_gpa_rmsd_per_complex_ang']:.6f} "
            f"LAA: {metrics['avg_laa_per_complex']:.8f} TIH: {metrics['avg_tih_per_complex']:.8f} "
        )
    message += f"Selected {args.selection_metric}: {selected:.8f}"
    print(message, flush=True)
    if args.output != "none":
        ensure_parent(args.output)
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    if selected < best_value:
        previous = None if math.isinf(best_value) else best_value
        best_value = selected
        best_epoch = epoch
        torch.save(model.state_dict(), checkpoint_file)
        write_rows(best_pose_file, per_pose)
        write_rows(best_complex_file, per_complex)
        write_json(
            best_metrics_file,
            {
                "best_epoch": best_epoch,
                "best_model_path": checkpoint_file,
                "selection_metric": args.selection_metric,
                "best_selected_value": best_value,
                "previous_best_selected_value": previous,
                "training_objective": args.loss,
                **metrics,
            },
        )
        print(f"Saved best model at epoch {epoch}: {selected:.8f}", flush=True)

print(f"Best epoch={best_epoch}, {args.selection_metric}={best_value}", flush=True)
