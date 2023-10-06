#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1


dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
config_file=configs/yingrenshi.yaml


batch_size=40960
train_iterations=200000
val_interval=50000
ckpt_interval=50000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

use_scaling=False
sampling_mesh_guidance=True


enable_semantic=True

freeze_geo=True
label_name=merge
separate_semantic=True
ckpt_path=logs_dji/1003_yingrenshi_density_depth_hash22/0/models/200000.pt

# depth_dji_loss=True
# wgt_depth_mse_loss=1

exp_name=logs_dji/1006_yingrenshi_density_depth_hash22_semantic_gt_separate

log2_hashmap_size=22
desired_resolution=8192

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --log2_hashmap_size=$log2_hashmap_size   --desired_resolution=$desired_resolution  \
    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path  --wgt_sem_loss=1 \
    --separate_semantic=$separate_semantic   --label_name=$label_name
