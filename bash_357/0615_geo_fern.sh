#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6

exp_name='logs_357/0619_geo_fern'

batch_size=1
train_iterations=100000
val_interval=2000
ckpt_interval=20000
sample_ray_num=10240

enable_semantic=False
dataset_path=/data/yuqi/Datasets/NeRF/nerf_llff_data/fern 


python gp_nerf/train.py  --aabb_bound   2  --sample_ray_num  $sample_ray_num  --exp_name  $exp_name  --use_scaling  False  --enable_semantic  $enable_semantic   --dataset_type   llff   --dataset_path  $dataset_path    --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --batch_size  $batch_size   









