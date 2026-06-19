# KTransPose

KTransPose is a graph neural network workflow for protein-ligand pose refinement. The current experimental focus is practical raw-coordinate RMSD: the model must improve the ligand pose without using the test-set ground-truth pose as an inference-time alignment target.

## Current Protocol

The repository contains two training/evaluation modes:

1. **MSE baseline**
   - Trains `Net_coor` with direct coordinate/displacement MSE.
   - Used as the raw-coordinate anchor checkpoint.
   - Select checkpoints by `avg_raw_rmsd_per_complex_ang`.

2. **TIH refinement**
   - Trains with TIH loss, combining Kabsch/GPA shape alignment and local
     pairwise-distance alignment.
   - Kabsch/GPA is used as a training/diagnostic signal.
   - Raw RMSD remains the practical metric for checkpoint selection.

Important: Kabsch-aligned RMSD can show whether ligand shape is being learned, but it should not be treated as the deployable result when the target pose is not available at inference time.

## Environment

The tested environment is Python 3.10 with CUDA-enabled PyTorch.

Create or activate your environment, then install:
pip install -r requirements.txt

The local experiments used:

Python 3.10.20
PyTorch 2.7.0+cu128
PyTorch Geometric 2.7.0
CUDA 12.8

## Dataset Preparation

Generate the processed coordinate dataset:

python convert_data_to_disk.py ^
  --cv=0 ^
  --input_list=data/pdb_list_ ^
  --output_file=pdbbind_rmsd_srand_coor2 ^
  --thread_num=1 ^
  --use_new_data ^
  --bond_th=6 ^
  --pocket_th=12 ^
  --groundtruth_dir=data/pdbbind/ ^
  --pdbbind_dir=data/medusadock_output ^
  --label_list_file=KtransPose_tmp ^
  --dataset=coor2 ^
  --pdb_version=2016


Move train/test folders under `raw`:


mkdir KtransPose_tmp\pdbbind_rmsd_srand_coor2\raw
move KtransPose_tmp\pdbbind_rmsd_srand_coor2\test KtransPose_tmp\pdbbind_rmsd_srand_coor2\raw\
move KtransPose_tmp\pdbbind_rmsd_srand_coor2\train KtransPose_tmp\pdbbind_rmsd_srand_coor2\raw\

## Train Raw MSE Anchor

This checkpoint is the practical raw-coordinate baseline.

python train_coor.py ^
  --data_path KtransPose_tmp\pdbbind_rmsd_srand_coor2 ^
  --gpu_id 0 ^
  --epoch 50 ^
  --start_epoch 1 ^
  --batch_size 16 ^
  --model_type Net_coor ^
  --loss MSELoss ^
  --flexible ^
  --residue ^
  --n_graph_layer 4 ^
  --d_graph_layer 256 ^
  --selection_metric avg_raw_rmsd_per_complex_ang ^
  --lr 0.0001 ^
  --output runs\raw_anchor_baseline.log ^
  --model_dir runs\raw_anchor_baseline_models ^
  --artifact_dir runs\artifacts\raw_anchor_baseline

The expected best checkpoint path is:


runs\raw_anchor_baseline_models\best_model.pt

## Train TIH Iter1 Refinement

Use the raw anchor as initialization and continue with TIH while still selecting by raw RMSD.
python train_coor.py ^
  --data_path KtransPose_tmp\pdbbind_rmsd_srand_coor2 ^
  --gpu_id 0 ^
  --epoch 50 ^
  --start_epoch 1 ^
  --batch_size 16 ^
  --model_type Net_coor ^
  --loss TIH ^
  --tih_lambda_gpa 0.5 ^
  --tih_lambda_laa 0.5 ^
  --flexible ^
  --residue ^
  --n_graph_layer 4 ^
  --d_graph_layer 256 ^
  --selection_metric avg_raw_rmsd_per_complex_ang ^
  --lr 0.00005 ^
  --pre_model runs\raw_anchor_baseline_models\best_model.pt ^
  --output runs\tih_iter1.log ^
  --model_dir runs\tih_iter1_models ^
  --artifact_dir runs\artifacts\tih_iter1

## Optional Iterative Refinement

Iter2 and Iter3 are stage-wise refinements. Each prior checkpoint is applied once to update the input pose before the next stage is trained.

Iter2 example:
python train_coor.py ^
  --data_path KtransPose_tmp\pdbbind_rmsd_srand_coor2 ^
  --gpu_id 0 ^
  --epoch 50 ^
  --batch_size 16 ^
  --model_type Net_coor ^
  --loss TIH ^
  --flexible ^
  --residue ^
  --n_graph_layer 4 ^
  --d_graph_layer 256 ^
  --iterative 2 ^
  --input_refine_checkpoints runs\tih_iter1_models\best_model.pt ^
  --selection_metric avg_raw_rmsd_per_complex_ang ^
  --lr 0.00005 ^
  --output runs\tih_iter2.log ^
  --model_dir runs\tih_iter2_models ^
  --artifact_dir runs\artifacts\tih_iter2

## Evaluation

Create a JSON run configuration, for example `test_runs_example.json`:
{
  "runs": [
    {
      "name": "raw_anchor",
      "training_objective": "MSELoss",
      "checkpoint_chain": ["runs/raw_anchor_baseline_models/best_model.pt"],
      "model_type": "Net_coor"
    },
    {
      "name": "tih_iter1",
      "training_objective": "TIH",
      "checkpoint_chain": ["runs/tih_iter1_models/best_model.pt"],
      "model_type": "Net_coor"
    }
  ]
}

Run evaluation:
python test.py ^
  --data_path KtransPose_tmp\pdbbind_rmsd_srand_coor2 ^
  --split test2 ^
  --runs_config test_runs_example.json ^
  --out_dir runs\evaluation_test2 ^
  --gpu_id 0

The key output fields are:
raw_rmsd_ang
gpa_rmsd_ang
kabsch_rmsd_ang
avg_raw_rmsd_per_complex_ang
avg_gpa_rmsd_per_complex_ang

Use `avg_raw_rmsd_per_complex_ang` as the practical metric. Use Kabsch/GPA-aligned RMSD as a shape-quality diagnostic.

## Practical Interpretation

If GPA/Kabsch RMSD is low but raw RMSD is high, the model is learning ligand shape but not global placement. In that case, prioritize changes that improve global translation/placement before adding more rankers or reporting oracle selection numbers.

Oracle-best candidate RMSD is useful only as a diagnostic ceiling. It should notbe reported as the final deployable result because it requires ground-truth RMSD
to select the besse
