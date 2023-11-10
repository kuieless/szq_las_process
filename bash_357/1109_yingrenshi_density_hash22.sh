#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4


dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
config_file=configs/yingrenshi.yaml


batch_size=10240
train_iterations=200000
val_interval=50000
ckpt_interval=50000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

use_scaling=False
sampling_mesh_guidance=False


enable_semantic=False


depth_dji_loss=False
wgt_depth_mse_loss=0

exp_name=logs_yingrenshi/1109_yingrenshi_density_hash22


log2_hashmap_size=22
desired_resolution=8192

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --geo_init_method=idr   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0  \
    --log2_hashmap_size=$log2_hashmap_size   --desired_resolution=$desired_resolution
