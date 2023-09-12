#!/bin/bash
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=4


exp_name='logs_357/0324_seperate_1'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
val_interval=100000
train_iterations=100000
label_size=1024
stop_semantic_grad=True
use_pano_lift=False
network_type=gpnerf
python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_size-1      --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  15     --val_interval  $val_interval   --label_size   $label_size  --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift  --network_type   $network_type











