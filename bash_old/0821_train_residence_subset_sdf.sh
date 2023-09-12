#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=7

exp_name=logs_dji/0821_train_residence_subset_sdf

dataset_path=/data/yuqi/Datasets/MegaNeRF/residence_subset
config_file=configs/residence.yaml


batch_size=20480
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=sdf     #  gpnerf   sdf
dataset_type=memory

enable_semantic=False
use_scaling=False

python gp_nerf/train.py  --wandb_id  None   --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  \
    --train_iterations   $train_iterations     --val_interval  $val_interval    --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --dataset_type $dataset_type     --use_scaling  $use_scaling
