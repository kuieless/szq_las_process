#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7

exp_name='logs_357/0615_geo_fern_semantic'

batch_size=1
train_iterations=20000
val_interval=5000
ckpt_interval=20000
sample_ray_num=4096

enable_semantic=True

python gp_nerf/train.py  --ckpt_path   logs_357/0615_geo_fern/18/models/20000.pt    --freeze_geo   True     --aabb_bound   2  --sample_ray_num  $sample_ray_num  --exp_name  $exp_name  --use_scaling  False  --enable_semantic  $enable_semantic   --dataset_type   llff   --dataset_path  /data/yuqi/Datasets/NeRF/nerf_llff_data/fern     --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --batch_size  $batch_size   









