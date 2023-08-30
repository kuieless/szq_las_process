#!/bin/bash
# 为了测试sdf在dji数据上的效果，之前用density的方法都不错，还有做mesh_guidance的对比实验

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5

exp_name=logs_dji/0830_train_dji_sdf

dataset_path=/data/yuqi/Datasets/DJI/subset
config_file=dji/cuhksz_ray_xml.yaml


batch_size=10240
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=sdf     #  gpnerf   sdf
dataset_type=memory_depth_dji

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=False

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True
