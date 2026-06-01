import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn import metrics

import os
import argparse
import math
import json
import numpy as np
from time import time
from tqdm import tqdm
from datetime import datetime

from dataset import PDBBindCoor
from model import loss_fn_cos
import plot


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", help="which model we use", type=str, default="Net_coor_res")
parser.add_argument("--loss", help="which loss function we use", type=str, default="L1Loss")
parser.add_argument("--loss_reduction", help="reduction approach for loss function", type=str, default="mean")
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=1000)
parser.add_argument("--start_epoch", help="epoch", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=64)
parser.add_argument("--atomwise", help="if we train the model atomwisely", type=int, default=0)
parser.add_argument("--gpu_id", help="id of gpu", type=int, default=3)
parser.add_argument("--data_path", help="train keys", type=str, default="/gpfs/group/mtk2/cyberstar/hzj5142/GNN/GNN/DGNN/data/pdbbind/pdbbind_rmsd_srand200/")
parser.add_argument("--heads", help="number of heads for multi-attention", type=int, default=1)
parser.add_argument("--edge_dim", help="dimension of edge feature", type=int, default=3)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=1)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=256)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=0)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=512)
parser.add_argument("--output", help="train result", type=str, default="none")
parser.add_argument("--model_dir", help="save best model", type=str, default="best_model")
parser.add_argument("--pre_model", help="pre trained model", type=str, default="None")
parser.add_argument("--th", help="threshold for positive pose", type=float, default=3.00)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.3)
parser.add_argument("--weight_bias", help="weight bias", type=float, default=1.0)
parser.add_argument("--last", help="activation of last layer", type=str, default="log")
parser.add_argument("--KD", help="if we apply knowledge distillation (Yes / No)", type=str, default="No")
parser.add_argument("--KD_soft", help="function convert rmsd to softlabel", type=str, default="exp")
parser.add_argument("--edge", help="if we use edge attr", type=bool, default=False)
parser.add_argument("--plt_dir", help="path to the plot figure", type=str, default="best_model_plt")
parser.add_argument("--flexible", help="if we only calculate flexible nodes", default=False, action="store_true")
parser.add_argument("--residue", help="if we apply residue connection to CONV layers", default=False, action="store_true")
parser.add_argument("--iterative", help="if we iteratively calculate the pose", type=int, default=0)
parser.add_argument("--pose_limit", help="maximum poses to be evaluated", type=int, default=0)
parser.add_argument("--step_len", help="length of the moving vector", type=float, default=0.03)
parser.add_argument("--class_dir", help="classify the direction on each axis", default=False, action="store_true")
parser.add_argument("--hinge", help="rate of hinge loss", type=float, default=0)
parser.add_argument("--tot_seed", help="num of seeds in the dataset", type=int, default=8)

parser.add_argument("--best_metrics_file", type=str, default="runs/reviewer_artifacts/best_epoch_metrics.json")
parser.add_argument("--best_per_complex_file", type=str, default="runs/reviewer_artifacts/best_epoch_per_complex.jsonl")
parser.add_argument("--checkpoint_events_file", type=str, default="runs/reviewer_artifacts/checkpoint_events.jsonl")

args = parser.parse_args()
print(args)

if args.atomwise:
    args.batch_size = 1


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path, payload):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path, payload):
    ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


path = args.data_path

train_dataset = PDBBindCoor(root=path, split="train")
test_dataset = PDBBindCoor(root=path, split="test")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

train_loader_size = len(train_loader.dataset)
test_dataset_size = len(test_dataset)
test_loader_size = len(test_loader.dataset)

print("total 0 subdatasets")
print(f"train_loader_size: {train_loader_size}")
print(f"test_dataset_size: {test_dataset_size}, test_loader_size: {test_loader_size}")

weight = 4.100135326385498 + args.weight_bias
print(f"weight: 1:{weight}")


def _dir_2_coor(out, length):
    out = out.exp()
    x = out[:, 4:8].sum(1) - out[:, :4].sum(1)
    y = out[:, [2, 3, 6, 7]].sum(1) - out[:, [0, 1, 4, 5]].sum(1)
    z = out[:, [1, 3, 5, 7]].sum(1) - out[:, [0, 2, 4, 6]].sum(1)
    ans = torch.stack([x, y, z], 1)
    return ans * length


gpu_id = str(args.gpu_id)
device_str = "cuda:" + gpu_id if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print("cuda" if torch.cuda.is_available() else "cpu")

from model import Net_coor, Net_coor_res, Net_coor_dir, Net_coor_len, Net_coor_cent

if args.model_type == "Net_coor_res":
    model = Net_coor_res(train_dataset.num_features, args).to(device)
elif args.model_type == "Net_coor":
    model = Net_coor(train_dataset.num_features, args).to(device)
elif args.model_type == "Net_coor_dir":
    model = Net_coor_dir(train_dataset.num_features, args).to(device)
elif args.model_type == "Net_coor_len":
    model = Net_coor_len(train_dataset.num_features, args).to(device)
elif args.model_type == "Net_coor_cent":
    model = Net_coor_cent(train_dataset.num_features, args).to(device)
else:
    raise ValueError(f"Unknown model_type: {args.model_type}")

if args.pre_model != "None":
    model = torch.load(args.pre_model, map_location=device_str).to(device)

if args.loss == "L1Loss":
    loss_op = torch.nn.L1Loss(reduction=args.loss_reduction)
elif args.loss == "MSELoss":
    loss_op = torch.nn.MSELoss(reduction=args.loss_reduction)
elif args.loss == "CosineEmbeddingLoss":
    loss_op = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
    cos_target = torch.tensor([1]).to(device)
elif args.loss == "CosineAngle":
    loss_op = loss_fn_cos(device, reduction=args.loss_reduction)
    loss_op2 = torch.nn.CosineEmbeddingLoss(reduction=args.loss_reduction)
    cos_target = torch.tensor([1]).to(device)
else:
    raise ValueError(f"Unknown loss: {args.loss}")

if args.class_dir:
    loss_op = torch.nn.CrossEntropyLoss()
    assert args.model_type == "Net_coor_dir"

hinge = torch.tensor([args.hinge]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def bond_dist(data, pred, fix_idx):
    x = data.edge_index[0, fix_idx]
    y = data.edge_index[1, fix_idx]
    node_x = data.x[x, -3:] + pred[x]
    node_y = data.x[y, -3:] + pred[y]
    dist = torch.nn.MSELoss(reduction="none")(node_x, node_y)
    return dist.sum(-1).sqrt()


def train():
    model.train()

    total_loss = 0
    tot = 0
    t = time()
    pbar = tqdm(total=train_loader_size)
    pbar.set_description("Training poses...")

    for data in train_loader:
        with torch.cuda.amp.autocast():
            data = data.to(device)

            if args.atomwise:
                flexible_len = data.flexible_len.cpu().item()
                all_atom_idx = torch.randperm(flexible_len)
                avg_loss = 0.0

                for idx in range(args.atomwise):
                    st = (idx * flexible_len) // args.atomwise
                    ed = ((idx + 1) * flexible_len) // args.atomwise
                    atom_idx = all_atom_idx[st:ed]

                    optimizer.zero_grad()
                    pred = model(data.x, data.edge_index, data.dist)[atom_idx]
                    loss = loss_op(pred, data.y[atom_idx])
                    avg_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                total_loss += avg_loss / args.atomwise
                tot += 1
                pbar.update(1)
                continue

            optimizer.zero_grad()

            if args.flexible:
                if args.model_type != "Net_coor_cent":
                    pred = model(data.x, data.edge_index, data.dist)[data.flexible_idx.bool()]

                if args.class_dir:
                    y = data.y[data.flexible_idx.bool()].gt(0).long()
                    y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                    loss = loss_op(pred, y)
                elif args.loss == "CosineEmbeddingLoss":
                    loss = loss_op(pred, data.y[data.flexible_idx.bool()], cos_target)
                elif args.loss == "CosineAngle":
                    loss = loss_op(pred, data.y[data.flexible_idx.bool()])
                elif args.model_type == "Net_coor_len":
                    length = data.y[data.flexible_idx.bool()].square().sum(1).sqrt().reshape(pred.size()[0], 1)
                    loss = loss_op(pred, length)
                elif args.model_type == "Net_coor_cent":
                    pred = model(data.x, data.edge_index, data.dist, data.batch, data.flexible_idx.bool())
                    y = global_mean_pool(data.y[data.flexible_idx.bool()], data.batch[data.flexible_idx.bool()])
                    loss = loss_op(pred, y)
                elif args.hinge != 0:
                    fix_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                    bond_diff = bond_dist(data, pred, fix_idx) - bond_dist(data, data.y, fix_idx)
                    l = fix_idx.size()[0]
                    loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(
                        bond_diff,
                        torch.LongTensor([-1] * l).to(device),
                    )
                    loss = loss_op(pred, data.y) + loss1 * hinge
                else:
                    loss = loss_op(pred, data.y)
            else:
                pred = model(data.x, data.edge_index, data.dist)
                loss = loss_op(pred, data.y)

            if args.loss in ["CosineEmbeddingLoss", "CosineAngle"]:
                total_loss += loss.item() / pred.size()[0] * args.batch_size
            else:
                total_loss += loss.item() * args.batch_size

        loss.backward()
        optimizer.step()
        tot += 1
        pbar.update(1)

    pbar.close()
    print(f"trained {tot} batches, take {time() - t}s")
    return total_loss / train_loader_size


@torch.no_grad()
def test(loader, epoch):
    model.eval()
    t = time()

    total_loss = 0
    total_rmsd_in = 0.0
    all_rmsds = []
    all_rmsds_in = []

    pose_idx = 0
    total_rmsds = [0.0 for _ in range(args.iterative)]
    avg_rmsd = 0.0
    tn = 0
    fpl = [0 for _ in range(8)]

    diff_complex = 0
    rmsd_per_pdb = []
    rmsd_per_pdb_in = []
    num_pose_per_pdb = []
    pdb = ""
    pdbs = []

    pbar = tqdm(total=test_loader_size)
    pbar.set_description("Testing poses...")

    for data in loader:
        pbar.update(1)

        num_atoms = data.x.size()[0]
        num_flexible_atoms = data.x[data.flexible_idx.bool()].size()[0]

        if data.pdb != pdb:
            diff_complex += 1
            rmsd_per_pdb.append(0.0)
            rmsd_per_pdb_in.append(0.0)
            num_pose_per_pdb.append(0)
            pdb = data.pdb
            pdbs.append(pdb[0])

        if args.flexible:
            if args.model_type != "Net_coor_cent":
                out = model(
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.dist.to(device),
                )[data.flexible_idx.bool()]

            if args.class_dir:
                y = data.y[data.flexible_idx.bool()].gt(0).long().to(device)
                y = y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]
                loss = loss_op(out, y)

                for i in range(8):
                    fpl[i] += y.eq(i).sum().cpu().item()
                tn += y.size()[0]

                out = _dir_2_coor(out, args.step_len)
            elif args.loss == "CosineEmbeddingLoss":
                loss = loss_op(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)
            elif args.loss == "CosineAngle":
                loss = (1 - loss_op2(out, data.y.to(device)[data.flexible_idx.bool()], cos_target)).acos().sum()
            elif args.model_type == "Net_coor_len":
                length = data.y.to(device)[data.flexible_idx.bool()].square().sum(1).sqrt().reshape(out.size()[0], 1)
                loss = loss_op(out, length)
                out = data.y.to(device)[data.flexible_idx.bool()]
            elif args.model_type == "Net_coor_cent":
                pred = model(
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.dist.to(device),
                    data.batch.to(device),
                    data.flexible_idx.bool().to(device),
                ).cpu()
                y = global_mean_pool(data.y[data.flexible_idx.bool()], data.batch[data.flexible_idx.bool()])
                loss = loss_op(pred, y)
                out = pred.repeat(num_flexible_atoms, 1).to(device)
            elif args.hinge != 0:
                fix_idx = (data.dist[:, 0] != 0).nonzero(as_tuple=True)[0]
                data_device = data.to(device)
                bond_diff = bond_dist(data_device, out, fix_idx) - bond_dist(data_device, data_device.y, fix_idx)
                loss1 = torch.nn.HingeEmbeddingLoss(margin=0.001)(
                    bond_diff,
                    torch.LongTensor([-1 for _ in fix_idx]).to(device),
                )
                loss = loss_op(out, data.y.to(device)) + loss1 * args.hinge
            else:
                loss = loss_op(out, data.y.to(device))

            rmsd = math.sqrt(
                F.mse_loss(data.y.to(device), out, reduction="sum").cpu().item()
                / num_flexible_atoms
            )

        else:
            out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))
            loss = loss_op(out, data.y.to(device))

            rmsd_sum = F.mse_loss(
                data.y[data.flexible_idx.bool()],
                out.cpu()[data.flexible_idx.bool()],
                reduction="sum",
            ).item()
            rmsd = math.sqrt(rmsd_sum / num_flexible_atoms)

        all_rmsds.append(rmsd)
        rmsd_per_pdb[-1] += rmsd
        num_pose_per_pdb[-1] += 1

        if epoch <= 1:
            if args.flexible:
                rmsd_in = math.sqrt(torch.sum(torch.square(data.y)).item() / num_flexible_atoms)
            else:
                rmsd_sum_in = F.mse_loss(data.y, data.x[:, -3:], reduction="sum").item()
                rmsd_in = math.sqrt(rmsd_sum_in / num_atoms)

            total_rmsd_in += rmsd_in
            all_rmsds_in.append(rmsd_in)
            rmsd_per_pdb_in[-1] += rmsd_in

        if args.loss in ["CosineEmbeddingLoss", "CosineAngle"]:
            total_loss += loss.item() / num_flexible_atoms
        else:
            total_loss += loss.item()

        pose_idx += 1
        if args.pose_limit > 0 and pose_idx >= args.pose_limit:
            break

    pbar.close()
    elapsed = time() - t

    print(f"Spend {elapsed}s")
    if args.class_dir:
        print(f"class_dir histogram x: {fpl}, all: {tn}")
    else:
        print("class_dir disabled: histogram counters are not used.")

    if pose_idx > 0 and len(total_rmsds) > 0:
        print([i / pose_idx for i in total_rmsds])
    if pose_idx > 0:
        print(avg_rmsd / pose_idx)

    print(f"diff_complex {diff_complex}")
    assert diff_complex % args.tot_seed == 0
    diff_complex = diff_complex // args.tot_seed
    print(f"diff_complex {diff_complex}")

    for ii in range(1, args.tot_seed):
        for jj in range(diff_complex):
            idx = ii * diff_complex + jj
            num_pose_per_pdb[jj] += num_pose_per_pdb[idx]
            rmsd_per_pdb[jj] += rmsd_per_pdb[idx]
            rmsd_per_pdb_in[jj] += rmsd_per_pdb_in[idx]

    avg_rmsd_per_pdb = sum(
        [r / max(d, 1) for r, d in zip(rmsd_per_pdb[:diff_complex], num_pose_per_pdb[:diff_complex])]
    ) / diff_complex

    if epoch <= 1:
        avg_rmsd_per_pdb_in = sum(
            [r / max(d, 1) for r, d in zip(rmsd_per_pdb_in[:diff_complex], num_pose_per_pdb[:diff_complex])]
        ) / diff_complex
    else:
        avg_rmsd_per_pdb_in = None

    rmsd_arr = np.array(all_rmsds, dtype=np.float32) if len(all_rmsds) else np.array([0.0], dtype=np.float32)
    rmsd_in_arr = np.array(all_rmsds_in, dtype=np.float32) if len(all_rmsds_in) else np.array([], dtype=np.float32)

    metrics_payload = {
        "test_loss": float(total_loss / max(pose_idx, 1)),
        "avg_rmsd_per_complex": float(avg_rmsd_per_pdb*100),
        "avg_input_rmsd_per_complex": float(avg_rmsd_per_pdb_in*100) if avg_rmsd_per_pdb_in is not None else None,
        "rmsd_mean": float(rmsd_arr.mean()),
        "rmsd_median": float(np.median(rmsd_arr)),
        "rmsd_std": float(rmsd_arr.std()),
        "rmsd_min": float(rmsd_arr.min()),
        "rmsd_q25": float(np.percentile(rmsd_arr*100, 25)),
        "rmsd_q75": float(np.percentile(rmsd_arr*100, 75)),
        "rmsd_p90": float(np.percentile(rmsd_arr*100, 90)),
        "rmsd_max": float(rmsd_arr.max()),
        "success_rate_2": float((rmsd_arr*100 <= 2.0).mean()),
        "success_rate_3": float((rmsd_arr*100 <= 3.0).mean()),
        "success_rate_5": float((rmsd_arr*100 <= 5.0).mean()),
        "num_poses": int(pose_idx),
        "num_complexes": int(diff_complex),
        "avg_poses_per_complex": float(pose_idx / max(diff_complex, 1)),
        "elapsed_sec": float(elapsed),
        "poses_per_sec": float(pose_idx / max(elapsed, 1e-8)),
    }

    if len(rmsd_in_arr):
        metrics_payload.update({
            "input_rmsd_mean": float(rmsd_in_arr.mean()),
            "input_rmsd_median": float(np.median(rmsd_in_arr)),
            "input_rmsd_std": float(rmsd_in_arr.std()),
            "input_rmsd_min": float(rmsd_in_arr.min()),
            "input_rmsd_q25": float(np.percentile(rmsd_in_arr*100, 25)),
            "input_rmsd_q75": float(np.percentile(rmsd_in_arr*100, 75)),
            "input_rmsd_p90": float(np.percentile(rmsd_in_arr*100, 90)),
            "input_rmsd_max": float(rmsd_in_arr.max()),
        })

    per_complex_metrics = []
    for jj in range(diff_complex):
        denom = max(num_pose_per_pdb[jj], 1)
        per_complex_metrics.append({
            "pdb": str(pdbs[jj]) if jj < len(pdbs) else str(jj),
            "num_poses": int(num_pose_per_pdb[jj]),
            "avg_rmsd": float(rmsd_per_pdb[jj] / denom),
            "avg_input_rmsd": float(rmsd_per_pdb_in[jj] / denom) if epoch <= 1 else None,
        })

    return metrics_payload, per_complex_metrics


if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.plt_dir):
    os.makedirs(args.plt_dir)

min_rmsd = 10.0
best_epoch = 0

for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
    train_loss = train()
    print(f"Train Loss: {train_loss}")

    eval_metrics, per_complex_metrics = test(test_loader, epoch)

    test_loss = eval_metrics["test_loss"]
    rmsd = eval_metrics["avg_rmsd_per_complex"]
    rmsd_in = eval_metrics["avg_input_rmsd_per_complex"]

    print(f"Epoch: {epoch} Test Loss: {test_loss}  Avg RMSD: {rmsd}")

    if epoch <= 1:
        print(f"Avg RMSD of inputs: {rmsd_in}")
        if args.output != "none":
            with open(args.output, "a") as f:
                f.write(f"Avg RMSD of inputs: {rmsd_in}\n")

    if args.output != "none":
        with open(args.output, "a") as f:
            f.write(f"Epoch: {epoch} Test Loss: {test_loss}  Avg RMSD: {rmsd}\n")

    if epoch > 3 and min_rmsd > rmsd:
        saved_model_dir = os.path.join(args.model_dir, f"model_{epoch}.pt")
        torch.save(model.state_dict(), saved_model_dir)
        os.system(f"chmod 777 {saved_model_dir}")

        old_best_rmsd = min_rmsd
        min_rmsd = rmsd
        best_epoch = epoch

        print(f"save model at epoch {epoch}, rmsd of {rmsd} !!!!!!!!")

        if args.output != "none":
            with open(args.output, "a") as f:
                f.write(f"save model at epoch {epoch}, rmsd of {rmsd} !!!!!!!!\n")

        best_payload = {
            "best_epoch": int(best_epoch),
            "best_model_path": saved_model_dir,
            "previous_best_rmsd": float(old_best_rmsd*100),
            "best_rmsd": float(min_rmsd*100),
            "train_loss": float(train_loss*100),
            "test_loss": float(test_loss*100),
            "selection_metric": "avg_rmsd_per_complex",
            "created_at": datetime.now().isoformat(),
            **eval_metrics,
        }

        write_json(args.best_metrics_file, best_payload)

        ensure_parent(args.best_per_complex_file)
        with open(args.best_per_complex_file, "w") as pf:
            for row in per_complex_metrics:
                pf.write(json.dumps({"epoch": int(epoch), **row}) + "\n")

        checkpoint_event = {
            "epoch": int(epoch),
            "saved_model_path": saved_model_dir,
            "rmsd": float(rmsd)*100,
            "previous_best_rmsd": float(old_best_rmsd)*100,
            "created_at": datetime.now().isoformat(),
        }

        append_jsonl(args.checkpoint_events_file, checkpoint_event)

        print(f"Updated best epoch metrics: {args.best_metrics_file}")
        print(f"Updated best epoch per-complex table: {args.best_per_complex_file}")

    print("")

if args.output != "none":
    os.system(f"chmod 777 {args.output}")

print(f"\nBest model at epoch {best_epoch}, rmsd is {min_rmsd*100}")
print(f"Best metrics saved at: {args.best_metrics_file}")
print(f"Best per-complex metrics saved at: {args.best_per_complex_file}")
print(f"Checkpoint history saved at: {args.checkpoint_events_file}")
