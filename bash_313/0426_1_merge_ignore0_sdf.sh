#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6


exp_name='logs_313/0426_1_merge_ignore0_sdf'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
label_type='m2f_custom'
stop_semantic_grad=True
use_pano_lift=False

batch_size=5210
train_iterations=100000
val_interval=50000
ckpt_interval=50000

ignore_index=0
label_name='merge'
wandb_id=gpnerf_semantic  #gpnerf_semantic   None
wandb_run_name=$exp_name
network_type='sdf'


python gp_nerf/train.py  --network_type   $network_type   --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name     --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  20     --val_interval  $val_interval    --ckpt_interval   $ckpt_interval  --label_name   $label_name   --label_type   $label_type   --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift  --wandb_id  $wandb_id   --wandb_run_name  $wandb_run_name    --batch_size  $batch_size   --ignore_index  $ignore_index 











