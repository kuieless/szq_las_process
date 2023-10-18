#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5


dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
config_file=configs/yingrenshi.yaml


batch_size=10240
train_iterations=200000
val_interval=50000
ckpt_interval=50000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=False


depth_dji_loss=False
wgt_depth_mse_loss=0

exp_name=logs_dji/0927_1_yingrenshi_density


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --geo_init_method=idr   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0
