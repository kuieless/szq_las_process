#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7

dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='residence' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"


dataset_path=/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels
config_file=configs/residence.yaml


batch_size=10240
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=gpnerf     #  gpnerf   sdf
dataset_type=memory

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=False


depth_dji_loss=False
wgt_depth_mse_loss=0

exp_name=logs_dji/0915_residence_density


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --geo_init_method=idr   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0   \
    --log2_hashmap_size=19
