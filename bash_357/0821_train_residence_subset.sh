#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=6

exp_name=logs_dji/0821_train_residence_subset

dataset_path=/data/yuqi/Datasets/MegaNeRF/residence_subset
config_file=configs/residence.yaml


batch_size=65536
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type='gpnerf'     #  gpnerf   sdf
dataset_type=memory



python gp_nerf/train.py  --exp_name  $exp_name  --config_file  $config_file   --dataset_path  $dataset_path   \
    --train_iterations   $train_iterations  --val_interval  $val_interval   --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --network_type    $network_type     --dataset_type $dataset_type   \







