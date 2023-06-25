#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6

exp_name='logs_357/0623_horns_grid_X'

batch_size=1
train_iterations=500
val_interval=500
ckpt_interval=500
sample_ray_num=8192

enable_semantic=True
dataset_path=/data/yuqi/Datasets/NeRF/nerf_llff_data/horns 
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0618_geo_horns_bound1/0/models/20000.pt
dataset_type='llff_sa3d'
num_semantic_classes=1
aabb_bound=1
use_densegrid_mask=True
sa3d_whole_image=False

python gp_nerf/train.py  --use_densegrid_mask  $use_densegrid_mask --sa3d_whole_image  $sa3d_whole_image  \
        --num_semantic_classes   $num_semantic_classes  \
        --ckpt_path   $ckpt_path   --freeze_geo   True     --aabb_bound   $aabb_bound  --sample_ray_num  $sample_ray_num  \
        --exp_name  $exp_name  --use_scaling  False  --enable_semantic  $enable_semantic   --dataset_type   $dataset_type   \
        --dataset_path  $dataset_path     --batch_size  $batch_size   \
        --train_iterations   $train_iterations      --val_interval  $val_interval    --ckpt_interval   $ckpt_interval









