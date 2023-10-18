#!/bin/bash
# 为了测试大batch会不会有孔洞

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=4

exp_name=logs_dji/0830_train_residence_subset_sdf_as_gpnerf_cos

dataset_path=/data/yuqi/Datasets/MegaNeRF/residence_subset
config_file=configs/residence.yaml


batch_size=10240
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=sdf     #  gpnerf   sdf
dataset_type=memory

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=False

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True
