#!/bin/bash
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=5


exp_name='logs_357/0422_sem_grad_stop_merge_clip'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
val_interval=100000
train_iterations=50000
ckpt_interval=50000

clip_grad_max=10

label_name='merge'
label_type='m2f_custom'
stop_semantic_grad=True
use_pano_lift=False
wandb_id=gpnerf_semantic  #gpnerf_semantic   None
wandb_run_name=$dataset2-0422_sem_grad_stop_merge_clip
python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-merge      --exp_name  $exp_name  --train_iterations   $train_iterations    --ckpt_interval  $ckpt_interval  --desired_chunks  20     --val_interval  $val_interval   --label_name   $label_name   --label_type   $label_type   --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift  --wandb_id  $wandb_id   --wandb_run_name  $wandb_run_name  --clip_grad_max  $clip_grad_max











