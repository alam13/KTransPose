# KTransPose

# To train the model by your self, you need to first generated the dataset:
python convert_data_to_disk.py --cv=0 --input_list=data/pdb_list_ --output_file=pdbbind_rmsd_srand_coor2 --thread_num=1 --use_new_data --bond_th=6 --pocket_th=12 --groundtruth_dir=data/pdbbind/ --pdbbind_dir=data/medusadock_output --label_list_file=KTransPose_tmp --dataset=coor2 --pdb_version=2016

mkdir KtransPose_tmp/pdbbind_rmsd_srand_coor2/raw

mv KtransPose_tmp/pdbbind_rmsd_srand_coor2/test/ KtransPose_tmp/pdbbind_rmsd_srand_coor2/raw/

mv KtransPose_tmp/pdbbind_rmsd_srand_coor2/train/ KtransPose_tmp/pdbbind_rmsd_srand_coor2/raw/

# Then train the protein-ligand pose prediction model with:
python train_coor.py --gpu_id=0 --n_graph_layer=4 --d_graph_layer=256 --start_epoch=1 --epoch=350 --flexible --model_dir=KtransPose_tmp/models_4_256_atom_hinge0 --data_path=KtransPose_tmp/pdbbind_rmsd_srand_coor2 --heads=1 --batch_size=1 --model_type=Net_coor --residue --edge_dim=3 --loss_reduction=mean --output=KtransPose_tmp/output_4_256_atom_hinge0 --hinge=0 --tot_seed=1


