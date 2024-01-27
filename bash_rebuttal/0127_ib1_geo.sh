#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=4


dataset_path=/data/yuqi/Datasets/InstanceBuilding/3D/scene1/output
config_file=configs/instancebuilding1.yaml


batch_size=10240
train_iterations=200000
val_interval=20000
ckpt_interval=20000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=False


depth_dji_loss=False
wgt_depth_mse_loss=0

exp_name=logs_rebuttal/0127_ib1_geo

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --geo_init_method=idr   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0 
    # \
    # --check_depth=True