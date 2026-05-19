import argparse, csv, json, os, re, statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"Epoch:\s*(\d+)\s*Train Loss:\s*([0-9eE+\-.]+)\s*Validation Loss:\s*([0-9eE+\-.]+)\s*Avg RMSD:\s*([0-9eE+\-.]+)")


def parse_log(path):
    rows=[]
    with open(path) as f:
        for line in f:
            m=EPOCH_RE.search(line)
            if m:
                rows.append({"epoch":int(m.group(1)),"train_loss":float(m.group(2)),"val_loss":float(m.group(3)),"avg_rmsd":float(m.group(4))})
    return rows


def save_csv(rows, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=["epoch","train_loss","val_loss","avg_rmsd"])
        w.writeheader();w.writerows(rows)


def plot_curve(rows, out_png):
    x=[r['epoch'] for r in rows]; y=[r['avg_rmsd'] for r in rows]
    plt.figure(figsize=(7,4)); plt.plot(x,y,marker='o'); plt.xlabel('Epoch'); plt.ylabel('Avg RMSD (Å)'); plt.title('Validation RMSD by Epoch'); plt.grid(True,alpha=0.3); plt.tight_layout();
    Path(out_png).parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_png,dpi=200)


def summary(rows):
    vals=[r['avg_rmsd'] for r in rows]
    b=min(vals); be=rows[vals.index(b)]['epoch']
    return {"best_rmsd":b,"best_epoch":be,"mean_rmsd":statistics.mean(vals),"std_rmsd":statistics.pstdev(vals)}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--log',required=True)
    ap.add_argument('--out_dir',required=True)
    args=ap.parse_args()
    rows=parse_log(args.log)
    if not rows:
        raise SystemExit('No epoch rows parsed; check training output format.')
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    save_csv(rows, out/'metrics_by_epoch.csv')
    plot_curve(rows, out/'fig_rmsd_curve.png')
    with open(out/'summary.json','w') as f: json.dump(summary(rows),f,indent=2)
    print(f'Saved artifacts to {out}')

