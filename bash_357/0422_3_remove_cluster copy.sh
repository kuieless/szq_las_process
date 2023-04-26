#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6


exp_name='logs_357/0422_3_remove_cluster'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
val_interval=50000
train_iterations=100000
ckpt_interval=50000 
clip_grad_max=0

label_name='m2f_new'
label_type='m2f_custom'
stop_semantic_grad=True
use_pano_lift=False
wandb_id=None  #gpnerf_semantic   None
wandb_run_name=$dataset2-0422_3_remove_cluster


python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-m2f-new     --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  20     --val_interval  $val_interval    --ckpt_interval   $ckpt_interval  --label_name   $label_name   --label_type   $label_type   --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift  --wandb_id  $wandb_id   --wandb_run_name  $wandb_run_name    











