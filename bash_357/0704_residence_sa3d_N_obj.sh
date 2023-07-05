#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6

exp_name='logs_357/0704_residence_sa3d_N_obj'

batch_size=1
train_iterations=100
val_interval=10
ckpt_interval=200
sample_ray_num=8192

enable_semantic=True
dataset_path=/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G_geo_residence/0/models/200000.pt
# ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0626_residence_sa3d_N_obj/26/models/200.pt
dataset_type='mega_sa3d'
num_semantic_classes=1
use_mask_type='densegrid_mlp'  # densegrid_mlp
sa3d_whole_image=True

config_file=configs/residence.yaml


python gp_nerf/train.py  --use_mask_type  $use_mask_type --sa3d_whole_image  $sa3d_whole_image  \
        --config_file   $config_file      \
        --num_semantic_classes   $num_semantic_classes  \
        --ckpt_path   $ckpt_path   --freeze_geo   True     --sample_ray_num  $sample_ray_num  \
        --exp_name  $exp_name  --enable_semantic  $enable_semantic   --dataset_type   $dataset_type   \
        --dataset_path  $dataset_path     --batch_size  $batch_size   \
        --train_iterations   $train_iterations      --val_interval  $val_interval    --ckpt_interval   $ckpt_interval









