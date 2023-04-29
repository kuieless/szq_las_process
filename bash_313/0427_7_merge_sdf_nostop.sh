#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6


exp_name='logs_313/0427_7_merge_sdf_nostop'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
stop_semantic_grad=False
use_pano_lift=False

batch_size=5210
train_iterations=100000
val_interval=50000
ckpt_interval=50000
wandb_id=gpnerf_semantic  #gpnerf_semantic   None
wandb_run_name=$exp_name


network_type='sdf'     #  gpnerf   sdf
ignore_index=-1
label_name='merge'      # merge    m2f



python gp_nerf/train.py  --wandb_id  $wandb_id   --exp_name  $exp_name  --wandb_run_name  $wandb_run_name  --label_name   $label_name --ignore_index  $ignore_index  --network_type    $network_type   --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4       --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift     --batch_size  $batch_size   











