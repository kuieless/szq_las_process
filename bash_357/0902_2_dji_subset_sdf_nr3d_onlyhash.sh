#!/bin/bash
#  在子集上测试长方体hash的效果，默认使用sampling_mesh_guidance
# 1. 直接测试长方体 hash + 坐标 + plane的效果
# 2. 只测试长方体hash的效果
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6

exp_name=logs_dji/0902_2_dji_subset_sdf_nr3d_onlyhash_nablas

dataset_path=/data/yuqi/Datasets/DJI/subset
config_file=dji/cuhksz_ray_xml.yaml


batch_size=10240   # 前三个实验忘记改batch_size（2048）了，应该变为10240
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=sdf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=True

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True   \
    --use_plane=False    --sdf_include_input=False   --nr3d_nablas=True