#!/bin/bash
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=5




dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
# python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-1      --exp_name  $exp_name
wgt_sem_loss=4e-2
val_interval=100000
train_iterations=100000
layer_dim=256
network_type='mlp'
label_size=1024

exp_name=logs_357/0316_mlp_$layer_dim

python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_size-1       --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  15   --wgt_sem_loss   $wgt_sem_loss   --val_interval  $val_interval   --layer_dim  $layer_dim  --network_type  $network_type   --label_size   $label_size 











